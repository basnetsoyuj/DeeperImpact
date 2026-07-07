from __future__ import annotations

import torch
from peft import PeftModel
from torch import nn
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
)

from .bidir_utils import is_transformers_attn_greater_or_equal_4_43_1


class ModifiedLlamaAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class ModifiedLlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


LLAMA_ATTENTION_CLASSES = {
    "eager": ModifiedLlamaAttention,
    "flash_attention_2": ModifiedLlamaFlashAttention2,
    "sdpa": ModifiedLlamaSdpaAttention,
}


class ModifiedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class LlamaBiModel(LlamaModel):
    """Standalone bidirectional Llama model used by the MNTP adapter."""

    _no_split_modules = ["ModifiedLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        if not is_transformers_attn_greater_or_equal_4_43_1():
            raise ValueError(
                "LlamaBiModel expects transformers >= 4.43.1 because it tracks "
                "the modern Llama attention implementation."
            )

        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                ModifiedLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # Zero upper triangle means tokens can attend bidirectionally. Padding is
        # still applied below by writing `min_dtype` into masked positions.
        causal_mask = torch.zeros(
            (sequence_length, target_length),
            dtype=dtype,
            device=device,
        )
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
            -1,
            1,
        )
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0],
            1,
            -1,
            -1,
        )

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :,
                    None,
                    None,
                    :,
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[
                    ...,
                    :mask_length,
                ].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask,
                min_dtype,
            )

        return causal_mask


class LlamaBiForMNTP(LlamaForCausalLM):
    """Minimal causal-LM wrapper kept for compatibility with MNTP checkpoints."""

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaBiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    def save_peft_model(self, path):
        self.save_pretrained(path)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
