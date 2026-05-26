#!/usr/bin/env bash
# SkillOS 8×H100 ALFWorld FULL fine-tuning launcher — PATH B (transfer-probe r_task).
# Same stack as run_fft.sh (DeepSpeed ZeRO-2 + vLLM colocate) but uses the Path B
# training config, which fixes the within-group r_task signal (see the config
# header and DIVERGENCES #6).
#
# Usage:
#   ./run_pathb.sh                                              # fresh run
#   ./run_pathb.sh ./output/alfworld-8xh100-v4-pathb/checkpoint-N   # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
export WANDB_RUN_ID=pathbv4
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

# Higher parallelism: Path B runs 1 seed + num_probe_tasks executor episodes per
# rollout (~3x the old executor load). infsh is elastic, so widen the pool to
# keep step time tractable (the textworld env.step is under a global lock but is
# sub-ms; the slow part is the concurrent infsh act() calls).
export SKILLOS_PARALLEL_ROLLOUTS=40
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT_MS=3600000
# Transfer-probe + randomized seed tasks make ranks finish their seed/probe
# executor phases at different times (different task difficulty). A fast rank
# then waits at the post-seed generation collective for a slow rank. The torch
# NCCL *heartbeat* monitor (default 480s) is a watchdog-of-the-watchdog that
# false-aborts on this benign wait — well before our 1200s per-rollout sentinel
# or the 1h NCCL_TIMEOUT_MS collective timeout that actually guard real hangs.
# Disable that monitor and raise its timeout above the rollout tolerance.
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_pathb_${ts}.log"
echo "Launching SkillOS Path B training → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  config=configs/alfworld_8xh100_pathb.yaml (transfer-probe r_task, num_probe_tasks=2)"

accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  -m skillos.train --config configs/alfworld_8xh100_pathb.yaml \
  > "$LOG" 2>&1
