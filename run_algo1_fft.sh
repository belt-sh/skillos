#!/usr/bin/env bash
# SkillOS Algorithm 1 — FULL fine-tune, ZeRO-3 + vLLM colocate, beta=0.001.
# v8 recipe minus LoRA. ZeRO-3 is the EMPIRICALLY-working path on this stack
# (ZeRO-2+vLLM HUNG 2026-06-21); see config header.
#
# Usage:
#   ./run_algo1_fft.sh                 # full run (max_steps from config)
#   SKILLOS_PHASE_BUDGET_S=240 ./run_algo1_fft.sh   # smoke: cut rollouts fast
#   ./run_algo1_fft.sh <checkpoint>    # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
export WANDB_RUN_ID="${WANDB_RUN_ID:-algo1fft}"
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=25
export SKILLOS_EXECUTOR_TIMEOUT_S=900
# Overridable so a smoke can cut rollouts fast and reach the backward/optimizer
# step quickly (the part that differs from LoRA for OOM + step-time).
export SKILLOS_PHASE_BUDGET_S="${SKILLOS_PHASE_BUDGET_S:-3600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_algo1_fft_${ts}.log"
mkdir -p logs
echo "Launching SkillOS Algorithm 1 FFT (ZeRO-3 + beta=0.001) → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}  PHASE_BUDGET_S=$SKILLOS_PHASE_BUDGET_S"

accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1_fft.yaml \
  > "$LOG" 2>&1
