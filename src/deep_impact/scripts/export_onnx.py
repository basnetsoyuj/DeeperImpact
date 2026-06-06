"""Export a trained DeepImpact checkpoint to a single-file ONNX model.

Inputs : input_ids, attention_mask, token_type_ids  (int64,   [batch, seq])
Output : impact_scores                               (float32, [batch, seq]);
         the impact of a term is the score at its first subword token, as in
         DeepImpact.compute_term_impacts.

    python -m src.deep_impact.scripts.export_onnx \
        --model_checkpoint_path soyuj/deeper-impact --output_path onnx/model.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.deep_impact.models import DeepImpact

# Two documents of different lengths, so the parity batch exercises padding and
# attention masking (not just a single full-length sequence).
SAMPLE_DOCUMENTS = [
    'The presence of communication amid scientific minds was equally important '
    'to the success of the Manhattan Project as scientific intellect was. The '
    'only cloud hanging over the impressive achievement of the atomic '
    'researchers and engineers is what their success truly meant; hundreds of '
    'thousands of innocent lives obliterated.',
    'Photosynthesis converts sunlight into chemical energy stored in glucose.',
]


class DeepImpactOnnxWrapper(nn.Module):
    def __init__(self, model: DeepImpact):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Squeeze [batch, seq, 1] -> [batch, seq] so consumers never reshape.
        return self.model(input_ids, attention_mask, token_type_ids).squeeze(-1)


def export(model: DeepImpact, output_path: Path, opset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DeepImpactOnnxWrapper(model).eval()

    batch, seq = 1, 32
    dummy_inputs = (
        torch.zeros((batch, seq), dtype=torch.long),
        torch.ones((batch, seq), dtype=torch.long),
        torch.zeros((batch, seq), dtype=torch.long),
    )

    print(f'Exporting to {output_path} (opset {opset})')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['impact_scores'],
            dynamic_axes={
                name: {0: 'batch', 1: 'sequence'}
                for name in ('input_ids', 'attention_mask', 'token_type_ids', 'impact_scores')
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    onnx.checker.check_model(str(output_path))
    print(f'Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB); onnx.checker passed')


def _encode_padded_batch(model: DeepImpact, documents):
    """Tokenize each document and right-pad them into one rectangular int64 batch."""
    encodings = [model.process_document(doc) for doc in documents]
    max_len = max(len(encoded.ids) for encoded, _ in encodings)
    pad_id = model.config.pad_token_id

    input_ids, attention_mask, token_type_ids = [], [], []
    for encoded, _ in encodings:
        pad = max_len - len(encoded.ids)
        input_ids.append(encoded.ids + [pad_id] * pad)
        attention_mask.append(encoded.attention_mask + [0] * pad)
        token_type_ids.append(encoded.type_ids + [0] * pad)

    arrays = (
        np.array(input_ids, dtype=np.int64),
        np.array(attention_mask, dtype=np.int64),
        np.array(token_type_ids, dtype=np.int64),
    )
    return encodings, arrays


def verify_parity(model: DeepImpact, output_path: Path, tolerance: float) -> None:
    encodings, (input_ids, attention_mask, token_type_ids) = _encode_padded_batch(model, SAMPLE_DOCUMENTS)

    # Drive both backends from the identical padded batch so the only variable
    # is PyTorch vs ONNX Runtime.
    with torch.no_grad():
        torch_scores = model(
            torch.from_numpy(input_ids),
            torch.from_numpy(attention_mask),
            torch.from_numpy(token_type_ids),
        ).squeeze(-1).cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    onnx_scores = session.run(
        None,
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        },
    )[0]

    # Compare only at each term's first-subword index, the positions consumers read.
    diffs = [
        abs(float(torch_scores[i][j]) - float(onnx_scores[i][j]))
        for i, (_, term_to_token_index) in enumerate(encodings)
        for j in term_to_token_index.values()
    ]
    if not diffs:
        raise SystemExit('Parity check produced no terms to compare')

    max_abs_diff = max(diffs)
    print(f'Parity vs PyTorch over {len(SAMPLE_DOCUMENTS)} documents, {len(diffs)} terms: '
          f'max |diff| = {max_abs_diff:.3g} (tolerance {tolerance:.3g})')
    if max_abs_diff > tolerance:
        raise SystemExit('Parity check FAILED')
    print('Parity check passed')


def run(model_checkpoint_path: str, output_path: Path, opset: int, tolerance: float,
        skip_verify: bool) -> None:
    print(f'Loading DeepImpact checkpoint: {model_checkpoint_path}')
    model = DeepImpact.from_pretrained(model_checkpoint_path).eval()
    export(model, output_path, opset)
    if not skip_verify:
        verify_parity(model, output_path, tolerance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export a DeepImpact checkpoint to ONNX and verify parity with PyTorch.')
    parser.add_argument('--model_checkpoint_path', type=str, default='soyuj/deeper-impact',
                        help='HuggingFace Hub id or local path of the trained DeepImpact checkpoint.')
    parser.add_argument('--output_path', type=Path, default=Path('onnx/model.onnx'),
                        help='Destination path for the exported ONNX model.')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version.')
    parser.add_argument('--tolerance', type=float, default=1e-4,
                        help='Maximum allowed |PyTorch - ONNX Runtime| impact-score difference.')
    parser.add_argument('--skip_verify', action='store_true', help='Skip the post-export parity check.')

    run(**vars(parser.parse_args()))
