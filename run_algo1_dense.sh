#!/usr/bin/env bash
# FULL-FIDELITY ALFWorld Algorithm 1, TRL + ZeRO-3, full fine-tune.
#
# The point of this run: measure r_task from all 9 informed positions per
# rollout, as the paper specifies. Every previous run measured 2.3 of 9 because
# the phase budget cut the rest (DIVERGENCES #16), and that was the single
# largest fidelity gap in the reproduction.
#
# TIMEOUTS SERVE FIDELITY, NOT THE REVERSE. Previously the phase budget was
# sized to keep 8 ranks inside the NCCL collective timeout, and positions were
# sacrificed to it. Now the budget is sized for the work (9 remote ALFWorld
# episodes, measured at ~3.9h) and the collective timeout is set to twice that,
# so rank skew can never reach it.
#
#   PHASE_BUDGET_S = 18000 (5h)   ~3.9h measured need + margin
#   NCCL_TIMEOUT_S = 36000 (10h)  2x the budget; skew is bounded by the budget
#
# Cost: ~4h/step, ~10 days for 60 steps. Checkpoints every 5 steps, so a crash
# loses at most 5 steps and scripts/dense_supervisor.sh resumes automatically.
#
# Usage:
#   ./run_algo1_dense.sh                # fresh
#   ./run_algo1_dense.sh <checkpoint>   # resume
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
export WANDB_RUN_ID="${WANDB_RUN_ID:-densefft}"
export WANDB_RESUME=allow
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

export SKILLOS_PARALLEL_ROLLOUTS=256
export SKILLOS_PARALLEL_JUDGES=24
export SKILLOS_EXECUTOR_MAX_STEPS=30      # paper avg 21.1; 30 leaves headroom
export SKILLOS_EXECUTOR_TIMEOUT_S=900     # per episode; never binding in the smoke
export SKILLOS_EXECUTOR_RETRY_PARSE=1     # reformat before coercing (0.15% coercion)

# The two knobs this run exists to change.
export SKILLOS_PHASE_BUDGET_S="${SKILLOS_PHASE_BUDGET_S:-18000}"
# Wall-clock for running positions the curator left unplayed (DIVERGENCES #18).
# The paper's loop runs every position; ours lets the curator stop, so we finish
# the protocol ourselves before scoring it. Positions abandoned to this budget
# are marked infrastructure losses and leave the r_task denominator, so the
# budget can never charge the curator for our impatience.
export SKILLOS_COMPLETION_BUDGET_S="${SKILLOS_COMPLETION_BUDGET_S:-5400}"
export SKILLOS_NCCL_TIMEOUT_S="${SKILLOS_NCCL_TIMEOUT_S:-36000}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0     # supervisor handles stalls, not SIGABRT

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/algo1_dense_${ts}.log"
mkdir -p logs
echo "Launching FULL-FIDELITY ALFWorld Algorithm 1 -> $LOG"
echo "  PHASE_BUDGET_S=$SKILLOS_PHASE_BUDGET_S  NCCL_TIMEOUT_S=$SKILLOS_NCCL_TIMEOUT_S"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  watch: grep 'reward health' $LOG   # median measured positions must be ~9"

# Fail here, in seconds, rather than ten minutes into a 4-day run. Any public
# method on the env becomes a curator tool (grpo_trainer.py:501-504); on
# 2026-08-18 a helper added as public crashed rank 0 during schema generation.
.venv/bin/python tests/test_env_tool_surface.py || {
  echo "ABORT: environment tool surface is wrong (see test output above)" >&2
  exit 1
}

accelerate launch \
  --config_file configs/accelerate_zero3.yaml \
  -m scripts.train_algo1 --config configs/alfworld_dense_fft.yaml \
  > "$LOG" 2>&1
