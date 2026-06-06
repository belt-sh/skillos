#!/usr/bin/env bash
# SkillOS Algorithm 1 v7 — LoRA r=32 + KL ANCHOR (beta=0.001, paper Table 4).
# Identical to run_algo1.sh except points at configs/alfworld_8xh100_algo1_v7_lora_kl.yaml.
#
# Why this branch: see header of configs/alfworld_8xh100_algo1_v7_lora_kl.yaml.
# To go back to FFT (v6): `./run_algo1_v6_kl.sh`.
#
# Usage:
#   ./run_algo1_v7_lora_kl.sh                                                        # fresh run
#   ./run_algo1_v7_lora_kl.sh ./output/alfworld-8xh100-algo1-v7-lora-kl/checkpoint-N # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
export WANDB_RUN_ID=algo1v7lorakl
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=25
export SKILLOS_EXECUTOR_TIMEOUT_S=900
export SKILLOS_PHASE_BUDGET_S=1500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT_MS=3600000
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_algo1_v7_lora_kl_${ts}.log"
mkdir -p logs
echo "Launching SkillOS Algorithm 1 v7 (LoRA r=32 + beta=0.001) → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  config=configs/alfworld_8xh100_algo1_v7_lora_kl.yaml"

accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1_v7_lora_kl.yaml \
  > "$LOG" 2>&1
