#!/usr/bin/env bash
# SkillOS 8×H100 ALFWorld FULL fine-tuning launcher (FSDP full-shard).
# Mirrors run.sh's env contract but uses FSDP (configs/accelerate_fsdp.yaml)
# + the FFT training config (configs/alfworld_8xh100_fft.yaml) instead of
# DDP + LoRA. DDP would OOM on 8B optimizer states.
#
# Usage:
#   ./run_fft.sh                                       # fresh FFT run
#   ./run_fft.sh ./output/alfworld-8xh100-v3-fft/checkpoint-N   # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
if [ -n "$CHECKPOINT" ]; then
  export WANDB_RUN_ID=fftv3a1          # FFT v3 run; resume continues this curve
  export WANDB_RESUME=allow
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=16
export SKILLOS_PARALLEL_JUDGES=16
export SKILLOS_EXECUTOR_MAX_STEPS=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT_MS=3600000

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_fft_${ts}.log"
echo "Launching SkillOS FFT training → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  accelerate=configs/accelerate_fsdp.yaml (FSDP FULL_SHARD)"
echo "  config=configs/alfworld_8xh100_fft.yaml (use_lora=false, lr 1e-6, max_steps 60)"

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  -m skillos.train --config configs/alfworld_8xh100_fft.yaml \
  > "$LOG" 2>&1
