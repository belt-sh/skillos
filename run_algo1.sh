#!/usr/bin/env bash
# SkillOS 8×H100 ALFWorld FULL fine-tuning launcher — PAPER ALGORITHM 1.
# Same stack as run_pathb.sh (DeepSpeed ZeRO-2 + vLLM colocate) but uses the
# Algorithm 1 multi-position curator env (scripts.train_algo1) where each
# rollout walks |G|=10 related ALFWorld tasks via the curate_and_advance
# mega-tool. Curator retrained from scratch — no resume from pathbv4.
#
# Usage:
#   ./run_algo1.sh                                                  # fresh run
#   ./run_algo1.sh ./output/alfworld-8xh100-algo1/checkpoint-N      # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
export WANDB_RUN_ID=algo1v1
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

# Algorithm 1: each rollout does 10 executor episodes (serial within a rollout,
# parallel across rollouts via TRL's tool loop iteration phase). Per-step:
# 32 instances × 8 generations × 10 positions = 2560 executor calls. infsh is
# elastic so widen the pool; the slow part is the concurrent infsh act()s.
export SKILLOS_PARALLEL_ROLLOUTS=64
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=30
# Per-position phase wall budget. With 10 positions per rollout the cumulative
# wall can dwarf Path B's single-rollout phase; keep this < the 1800s NCCL
# collective watchdog so a stalled position doesn't kill the run.
export SKILLOS_PHASE_BUDGET_S=1500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT_MS=3600000
# Position-level rank skew (one rank's position-3 takes longer than another's)
# can trigger the torch NCCL heartbeat false-abort during the post-position
# collective. Disable it; the real watchdog is NCCL_TIMEOUT_MS.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_algo1_${ts}.log"
mkdir -p logs
echo "Launching SkillOS Algorithm 1 training → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  config=configs/alfworld_8xh100_algo1.yaml (Algorithm 1, |G|=10, mega-tool)"

accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1.yaml \
  > "$LOG" 2>&1
