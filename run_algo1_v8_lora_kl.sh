#!/usr/bin/env bash
# SkillOS Algorithm 1 v8 — LoRA r=32 + KL anchor + group-collapse fixes.
# See docs/postmortem-2026-06-10-algo1-group-collapse.md and the header of
# configs/alfworld_8xh100_algo1_v8_lora_kl.yaml.
#
# Usage:
#   ./run_algo1_v8_lora_kl.sh                                                        # fresh run
#   ./run_algo1_v8_lora_kl.sh ./output/alfworld-8xh100-algo1-v8-lora-kl/checkpoint-N # resume

set -e
cd "$(dirname "$0")"
CHECKPOINT="${1:-}"

source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Belt-and-braces for cross-rank determinism: task-sequence seeding no longer
# uses builtin hash(), but pin the salt anyway so any future hash() use
# cannot diverge across ranks (postmortem bug 2).
export PYTHONHASHSEED=0

export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
export WANDB_RUN_ID=algo1v8lorakl
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=25
export SKILLOS_EXECUTOR_TIMEOUT_S=900
# Per-rollout wall-clock deadline (NOW ENFORCED, see Algo1CuratorEnv): once a
# rollout runs past this, remaining positions are cut (executor skipped, masked
# from r_task). 60 min lets normal rollouts finish all 10 positions while
# capping rank skew to ~75 min (<< the 4h NCCL collective timeout) so a slow
# composite-verb group can't strand the other ranks and SIGABRT the run.
export SKILLOS_PHASE_BUDGET_S=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NB: the real NCCL collective timeout is the dist.init_process_group pre-init
# in scripts/train_algo1.py (SKILLOS_NCCL_TIMEOUT_S, default 4h). NCCL_TIMEOUT_MS
# is a no-op under accelerate and was dropped from this launcher.
export TORCH_NCCL_ENABLE_MONITORING=0

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_algo1_v8_lora_kl_${ts}.log"
mkdir -p logs
echo "Launching SkillOS Algorithm 1 v8 (LoRA r=32 + beta=0.001 + postmortem fixes) → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  config=configs/alfworld_8xh100_algo1_v8_lora_kl.yaml"

accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1_v8_lora_kl.yaml \
  > "$LOG" 2>&1
