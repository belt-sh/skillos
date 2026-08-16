#!/usr/bin/env bash
# Measure the two fixes BEFORE spending 3 days of 8xH100 on them.
#
# WHAT WE NEED TO KNOW, and could not answer from the old runs because nothing
# printed it:
#   1. How many informed positions actually get measured per rollout. The audit
#      said median 1 of 9. The fix neutralises the fully-unmeasured rollouts, but
#      if the median is still 1 the reward is estimated from one episode and the
#      real problem is the deadline budget, not the masking.
#   2. Whether the reformat retry actually recovers unparseable actions, and what
#      coercion drops to.
#
# Reads the per-step `[algo1] reward health:` line the fixed reward_func emits.
# 3 steps is enough: the health line prints every step.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
export WANDB_MODE=disabled

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXECUTOR_TIMEOUT_S=900
export SKILLOS_EXECUTOR_RETRY_PARSE=1
# The variable under test. Old runs used 3600 and cut 61-79% of positions.
export SKILLOS_PHASE_BUDGET_S="${SKILLOS_PHASE_BUDGET_S:-3600}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0

CFG="${SMOKE_CONFIG:-configs/alfworld_smoke_reward_health.yaml}"
LOG="logs/smoke_reward_health_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[$(date -u)] smoke: PHASE_BUDGET_S=$SKILLOS_PHASE_BUDGET_S config=$CFG"
echo "  log: $LOG"
accelerate launch --config_file configs/accelerate_zero3.yaml \
  -m scripts.train_algo1 --config "$CFG" > "$LOG" 2>&1
rc=$?
echo "[$(date -u)] smoke exited rc=$rc"
echo "=== reward health lines ==="
grep "reward health" "$LOG" || echo "  NONE — the fix did not run; investigate"
echo "=== unmeasured rollouts ==="
grep -c "NO measured informed position" "$LOG" || true
echo "=== reformat ==="
grep -c "executor-reformat" output/infsh_tasks.jsonl 2>/dev/null || echo "  no reformat calls logged"
