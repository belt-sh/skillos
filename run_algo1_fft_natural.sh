#!/usr/bin/env bash
# SkillOS Algorithm 1 FFT — NATURAL type distribution (DIVERGENCES #0).
# Identical to run_algo1_fft.sh except: natural-frequency group types instead of
# uniform round-robin, distinct output_dir + wandb run. Tests whether the
# uniform-vs-natural training type distribution drives the bimodal trajectory.
#
# Usage:
#   ./run_algo1_fft_natural.sh                 # full run (max_steps from config)
#   ./run_algo1_fft_natural.sh <checkpoint>    # resume

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
export WANDB_RUN_ID="${WANDB_RUN_ID:-algo1fftnatural}"
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=25
export SKILLOS_EXECUTOR_TIMEOUT_S=900
export SKILLOS_PHASE_BUDGET_S="${SKILLOS_PHASE_BUDGET_S:-3600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_algo1_fft_natural_${ts}.log"
mkdir -p logs
echo "Launching SkillOS Algorithm 1 FFT NATURAL (ZeRO-3 + beta=0.001) → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}  PHASE_BUDGET_S=$SKILLOS_PHASE_BUDGET_S"

accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1_fft_natural.yaml \
  > "$LOG" 2>&1
