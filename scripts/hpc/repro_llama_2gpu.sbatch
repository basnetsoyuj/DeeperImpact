#!/bin/bash
#SBATCH --account=torch_pr_73_tandon_advanced
#SBATCH --job-name=di-orig-llama
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00

# Llama-3.2-1B (bidirectional + MNTP merged) through the ORIGINAL DeeperImpact
# training loop — identical recipe to the BERT repro (2 GPUs x batch_size 1 x
# 55-doc groups, KL distil, shuffled positive-anchored grouping), differing
# only in: backbone, bf16 autocast instead of fp16 (fp16 overflows Llama),
# full fine-tune by default (--llama_lora_r 0), gradient checkpointing on.
# NanoBEIR in-training eval disabled (--eval_every 0): too slow through a 1B
# backbone; index+eval happens offline from checkpoints instead.
# Resumable: auto-loads DeepImpactLlama_latest.pt from checkpoint_dir.
# Requires HF_TOKEN with access to meta-llama/Llama-3.2-1B.
# mkdir -p logs before sbatch.

set -euo pipefail

RESEARCH_ROOT="${SAE_IR_RESEARCH_ROOT:-/scratch/sjb8193/sae_ir_research}"
REPO_DIR="${REPO_DIR:-$RESEARCH_ROOT/DeeperImpact}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
DATA="$RESEARCH_ROOT/sae_splade/data/msmarco"

SCORES="${SCORES:-$DATA/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz}"
QUERIES="${QUERIES:-$DATA/train_queries/queries/raw.tsv}"
COLLECTION="${COLLECTION:-$RESEARCH_ROOT/data_files/collection_llama.tsv}"
QRELS_TSV="${QRELS_TSV:-$RESEARCH_ROOT/data_files/qrels.train.tsv}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints/repro_llama_mntp}"

mkdir -p "$REPO_DIR/logs" "$CKPT_DIR"
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc_per_node=2 -m src.deep_impact.train \
  --dataset_path "$SCORES" \
  --queries_path "$QUERIES" \
  --collection_path "$COLLECTION" \
  --qrels_path "$QRELS_TSV" \
  --checkpoint_dir "$CKPT_DIR" \
  --backbone llama_mntp \
  --distil_kl \
  --shuffle_positive \
  --batch_size 1 \
  --max_length 300 \
  --lr 3e-6 \
  --save_every 10000 \
  --eval_every 0
