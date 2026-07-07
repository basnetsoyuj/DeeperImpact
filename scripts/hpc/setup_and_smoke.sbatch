#!/bin/bash
#SBATCH --account=torch_pr_73_tandon_advanced
#SBATCH --job-name=di-setup-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# One-shot environment finalization + GPU smoke test for the llama-mntp-port
# branch. The two repro training jobs are submitted with
# --dependency=afterok:<this job>, so they only start if this passes.

set -euo pipefail

RESEARCH_ROOT="${SAE_IR_RESEARCH_ROOT:-/scratch/sjb8193/sae_ir_research}"
REPO_DIR="${REPO_DIR:-$RESEARCH_ROOT/DeeperImpact}"
cd "$REPO_DIR"
mkdir -p logs

export TMPDIR=/scratch/sjb8193/tmp
export HF_HOME=/scratch/sjb8193/.cache/huggingface
export HF_TOKEN="$(cat /scratch/sjb8193/.huggingface/token)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$HF_HOME"

# pin torch to the version proven with transformers 4.46.3 on this cluster
.venv/bin/pip install -q --no-cache-dir "torch==2.5.1"
.venv/bin/python - << 'PY'
import torch, transformers, tokenizers, peft
print("env:", torch.__version__, transformers.__version__, tokenizers.__version__, peft.__version__)
assert torch.__version__.startswith("2.5.1"), torch.__version__
PY

.venv/bin/python - << 'PY'
import torch
from src.deep_impact.models import DeepImpact, DeepImpactLlama

# ---------- 1. alignment (no GPU needed) ----------
DeepImpactLlama.tokenizer.enable_truncation(max_length=300)
DeepImpactLlama.tokenizer.enable_padding(length=300)

doc = "The Sony PS-LX300USB térm test. The Sony again, with ethanol-based fuel."
enc, term_map = DeepImpactLlama.process_document(doc)
assert len(enc.ids) == 300 and len(enc.attention_mask) == 300
assert "sony" in term_map and "term" in term_map and "ethanol" in term_map, term_map
tok = DeepImpactLlama._tokenizer()
raw = tok(doc, truncation=True, max_length=300, return_offsets_mapping=True)
for key, tix in term_map.items():
    s, e = raw["offset_mapping"][tix]
    covered = doc[s:e]
    assert key[0] in covered.casefold(), (key, covered)
# first occurrence: "sony" appears twice; its token must COVER the first one
# (Llama BPE offsets include the leading space, so use containment, not ==)
s, e = raw["offset_mapping"][term_map["sony"]]
assert s <= doc.index("Sony") < e, (s, e, doc.index("Sony"))
assert e <= doc.index("Sony again") + len("Sony"), "matched the second occurrence"

q_terms = DeepImpactLlama.process_query("Sony PS-LX300USB ethanol?")
assert "sony" in q_terms and "ethanol" in q_terms and "?" not in q_terms
mask = DeepImpactLlama.get_query_document_token_mask(q_terms, term_map, 300)
assert mask.sum().item() >= 2
print("alignment: PASS —", dict(list(term_map.items())[:6]))

# ---------- 2. Llama model load + forward/backward (GPU) ----------
model = DeepImpactLlama.load().cuda()
n_docs = 56
encs, masks = [], []
q = "what is ethanol fuel"
for i in range(n_docs):
    e, m = DeepImpactLlama.process_query_and_document(q, f"Ethanol is a fuel made by fermenting crops. Variant {i}.", 300)
    encs.append(e); masks.append(m)
input_ids = torch.tensor([e.ids for e in encs], dtype=torch.long).cuda()
attn = torch.tensor([e.attention_mask for e in encs], dtype=torch.long).cuda()
mask_t = torch.stack(masks).unsqueeze(-1).cuda()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    scores = model(input_ids, attn)
    group = (mask_t * scores).sum(dim=1).squeeze(-1).view(1, -1)
    teacher = torch.randn_like(group)
    loss = torch.nn.KLDivLoss(reduction="none")(
        torch.log_softmax(group, dim=1), torch.softmax(teacher, dim=1)
    ).sum(dim=1).mean()
loss.backward()
g = model.impact_score_encoder[0].weight.grad
bg = next(p.grad for p in model.llama.parameters() if p.requires_grad and p.grad is not None)
assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
assert torch.isfinite(bg).all()
assert torch.isfinite(loss) and scores.shape == (n_docs, 300, 1)
print(f"llama forward/backward: PASS — loss={loss.item():.4f}, group={tuple(group.shape)}")
del model, scores, group, loss
torch.cuda.empty_cache()

# ---------- 3. BERT path unchanged sanity (fp16, original flow) ----------
bert = DeepImpact.load().cuda()
DeepImpact.tokenizer.enable_truncation(max_length=300, strategy='longest_first')
DeepImpact.tokenizer.enable_padding(length=300)
enc, m = DeepImpact.process_query_and_document(q, "Ethanol is a fuel produced by fermenting crops.", 300)
ii = torch.tensor([enc.ids], dtype=torch.long).cuda()
am = torch.tensor([enc.attention_mask], dtype=torch.long).cuda()
ti = torch.tensor([enc.type_ids], dtype=torch.long).cuda()
with torch.cuda.amp.autocast():
    out = bert(ii, am, ti)
s = (m.unsqueeze(-1).cuda() * out[0]).sum()
assert torch.isfinite(s)
print(f"bert path: PASS — score={s.item():.4f}")
print("SMOKE: ALL PASS")
PY
echo "setup+smoke complete"
