"""DeepImpact with a bidirectional-MNTP Llama backbone.

Drop-in sibling of `DeepImpact` (models/original.py): same interface
(process_query / process_document / process_query_and_document /
get_query_document_token_mask / load / forward / compute_term_impacts /
get_impact_scores*), same training loop, same collate functions — only the
backbone and the word→token alignment differ.

Alignment lessons ported from the sae_splade Phase-0 work:
- The model consumes RAW, CASED text (never casefolded, never pre-split into
  words): Llama's byte-level BPE encodes a word differently with and without
  its leading space, so `is_split_into_words=True` or feeding lowercased text
  puts the backbone out of distribution.
- Words are aligned to tokens via the fast tokenizer's `offset_mapping`,
  which requires tokenizers >= 0.20.1 for Llama-3 (earlier versions return
  degenerate offsets; enforced below).
- Term KEYS (used for query matching and the inverted index) are normalized
  the same way BERT-uncased normalizes: lowercase + accent stripping. This
  keeps the BERT and Llama arms' term vocabularies aligned.
"""

import os
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.utils.checkpoint import ModelCheckpoint

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_MNTP_ADAPTER = "soyuj/llama3.2-1b-bidirectional-mntp-msmarco"

# words = alnum runs or single punctuation marks, per BertPreTokenizer semantics
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def _normalize_key(word: str) -> str:
    """BERT-uncased-equivalent normalization for term keys only."""
    word = word.casefold()
    word = unicodedata.normalize("NFD", word)
    return "".join(c for c in word if unicodedata.category(c) != "Mn")


@dataclass
class LlamaEncoding:
    """Minimal stand-in for tokenizers.Encoding exposing the three fields the
    Trainer reads (.ids, .attention_mask, .type_ids)."""

    ids: List[int]
    attention_mask: List[int]
    type_ids: List[int]


class _TokenizerShim:
    """Exposes the two tokenizers.Tokenizer methods train.py calls
    (enable_truncation / enable_padding) and records their settings on the
    owning model class."""

    def __init__(self, owner: type) -> None:
        self._owner = owner

    def enable_truncation(self, max_length: int, **_) -> None:
        self._owner.max_length = max_length

    def enable_padding(self, length: Optional[int] = None, **_) -> None:
        self._owner.pad_to_length = length


class DeepImpactLlama(nn.Module):
    max_length = 300
    pad_to_length: Optional[int] = None
    punctuation = _PUNCTUATION

    base_model_name_or_path = DEFAULT_BASE_MODEL
    mntp_adapter_path = DEFAULT_MNTP_ADAPTER
    # 0 => full fine-tune (default, mirroring the BERT arm); >0 => LoRA rank
    lora_r = 0
    lora_alpha = 32
    lora_dropout = 0.05

    _hf_tokenizer = None  # populated lazily by _tokenizer()

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.llama = backbone
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------ #
    # tokenizer plumbing
    # ------------------------------------------------------------------ #

    @classmethod
    def _tokenizer(cls):
        if cls._hf_tokenizer is None:
            import tokenizers as _tk
            from packaging.version import Version
            from transformers import AutoTokenizer

            if Version(_tk.__version__) < Version("0.20.1"):
                raise RuntimeError(
                    f"tokenizers {_tk.__version__} returns broken Llama-3 "
                    "offset_mapping; install tokenizers>=0.20.1"
                )
            token = os.environ.get("HF_TOKEN")
            cls._hf_tokenizer = AutoTokenizer.from_pretrained(
                cls.base_model_name_or_path,
                token=token,
                use_fast=True,
            )
            # Llama ships without a pad token; pad with EOS (attention_mask
            # zeroes it out, so the choice is inert for scoring)
            if cls._hf_tokenizer.pad_token is None:
                cls._hf_tokenizer.pad_token = cls._hf_tokenizer.eos_token
        return cls._hf_tokenizer

    # `model_cls.tokenizer.enable_truncation(...)` compatibility with train.py
    class _TokenizerDescriptor:
        def __get__(self, _obj, objtype=None):
            return _TokenizerShim(objtype)

    tokenizer = _TokenizerDescriptor()

    # ------------------------------------------------------------------ #
    # text processing (interface parity with DeepImpact)
    # ------------------------------------------------------------------ #

    @classmethod
    def process_query(cls, query: str) -> Set[str]:
        return {
            _normalize_key(m.group(0))
            for m in _WORD_RE.finditer(query)
            if m.group(0) not in cls.punctuation
        }

    @classmethod
    def process_document(cls, document: str) -> Tuple[LlamaEncoding, Dict[str, int]]:
        """Encode RAW document text; map each unique normalized term to the
        index of the first token of its first occurrence (via offsets)."""
        tok = cls._tokenizer()
        enc = tok(
            document,
            truncation=True,
            max_length=cls.max_length,
            padding="max_length" if cls.pad_to_length else False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]

        # first-token index for each character-start, walked in lockstep
        term_to_token_index: Dict[str, int] = {}
        token_i = 0
        n_tokens = len(offsets)
        for m in _WORD_RE.finditer(document):
            word = m.group(0)
            if word in cls.punctuation:
                continue
            key = _normalize_key(word)
            if key in term_to_token_index:
                continue
            start = m.start()
            while token_i < n_tokens and (
                offsets[token_i][1] <= start or offsets[token_i][0] == offsets[token_i][1]
            ):
                token_i += 1
            if token_i >= n_tokens:
                break  # truncated: remaining words have no tokens
            if offsets[token_i][0] <= start < offsets[token_i][1]:
                term_to_token_index[key] = token_i

        encoding = LlamaEncoding(
            ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            type_ids=[0] * len(enc["input_ids"]),
        )
        return encoding, term_to_token_index

    @classmethod
    def process_query_and_document(
        cls, query: str, document: str, max_length: Optional[int] = None
    ) -> Tuple[LlamaEncoding, torch.Tensor]:
        query_terms = cls.process_query(query)
        encoded, term_to_token_index = cls.process_document(document)
        return encoded, cls.get_query_document_token_mask(
            query_terms, term_to_token_index, max_length
        )

    @classmethod
    def get_query_document_token_mask(
        cls,
        query_terms: Set[str],
        term_to_token_index: Dict[str, int],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        if max_length is None:
            max_length = cls.max_length
        mask = np.zeros(max_length, dtype=bool)
        token_indices = [v for k, v in term_to_token_index.items() if k in query_terms]
        mask[token_indices] = True
        return torch.from_numpy(mask)

    # ------------------------------------------------------------------ #
    # model
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,  # ignored; interface parity
    ) -> torch.Tensor:
        out = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        return self.impact_score_encoder(out.last_hidden_state)

    @classmethod
    def load(cls, checkpoint_path: Optional[Union[str, Path]] = None) -> "DeepImpactLlama":
        from peft import LoraConfig, PeftModel, get_peft_model

        from .bidirectional_llama import LlamaBiModel

        token = os.environ.get("HF_TOKEN")
        auth = {"token": token} if token else {}

        backbone = LlamaBiModel.from_pretrained(
            cls.base_model_name_or_path,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **auth,
        )
        # MNTP adaptation is a frozen part of the backbone: merge it first,
        # then (optionally) attach ONE fresh retrieval LoRA.
        backbone = PeftModel.from_pretrained(backbone, cls.mntp_adapter_path, **auth)
        backbone = backbone.merge_and_unload()

        if cls.lora_r > 0:
            lora = LoraConfig(
                r=cls.lora_r,
                lora_alpha=cls.lora_alpha,
                lora_dropout=cls.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            backbone = get_peft_model(backbone, lora)

        # non-reentrant checkpointing composes with DDP(find_unused_parameters=True)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()

        model = cls(backbone, hidden_size=backbone.config.hidden_size)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
        cls._tokenizer()  # fail fast on tokenizer problems
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------ #
    # inference/indexing parity
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_term_impacts(
        documents_term_to_token_index_map: List[Dict[str, int]],
        outputs: torch.Tensor,
    ) -> List[List[Tuple[str, float]]]:
        impact_scores = outputs.squeeze(-1).float().cpu().numpy()
        term_impacts = []
        for i, term_to_token_index_map in enumerate(documents_term_to_token_index_map):
            term_impacts.append(
                [
                    (term, impact_scores[i][token_index])
                    for term, token_index in term_to_token_index_map.items()
                ]
            )
        return term_impacts

    def get_impact_scores(self, document: str) -> List[Tuple[str, float]]:
        return self.get_impact_scores_batch([document])[0]

    def get_impact_scores_batch(self, documents: List[str]) -> List[List[Tuple[str, float]]]:
        cls = type(self)
        old_pad = cls.pad_to_length
        cls.pad_to_length = cls.max_length  # uniform lengths for batching
        try:
            encoded_docs, term_maps = [], []
            for doc in documents:
                enc, term_map = self.process_document(doc)
                encoded_docs.append(enc)
                term_maps.append(term_map)
        finally:
            cls.pad_to_length = old_pad

        input_ids = torch.tensor([e.ids for e in encoded_docs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(
            [e.attention_mask for e in encoded_docs], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            outputs = self(input_ids, attention_mask)

        return self.compute_term_impacts(term_maps, outputs)
