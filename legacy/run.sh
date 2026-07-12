#!/usr/bin/env bash
# SkillOS 8×H100 ALFWorld training launcher.
#
# Captures the full env contract for a clean run: PyTorch allocator,
# retry budgets, observability, NCCL margin, and wandb-run-resume so
# that resumed checkpoints continue the same wandb curve.
#
# Usage:
#   ./run.sh                                              # fresh run
#   ./run.sh ./output/alfworld-8xh100/checkpoint-14       # resume from N
#
# Logs land in logs/alfworld_8xh100_<timestamp>.log so successive runs
# don't clobber each other's logs (useful for post-mortem diff).

set -e
cd "$(dirname "$0")"

CHECKPOINT="${1:-}"

# Python env
source .venv/bin/activate

# ALFWorld dataset (heuristic envs need this even though we use infsh executor)
export ALFWORLD_DATA="$HOME/.cache/alfworld"

# Force HF to use the local cache without contacting the hub. The base
# Qwen3-8B weights are fully cached; without this, 8 ranks calling
# from_pretrained() concurrently race on hub verification and one can
# spuriously raise "does not appear to have a file named model-...safetensors"
# even though all shards are present. Offline mode eliminates that race.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Wandb: project + entity. When resuming, also rejoin the existing run so the
# reward curve doesn't fragment into a new run each time. WANDB_RUN_ID is the
# id of the 8xH100 LoRA run we want to attach to; bump this for a new fresh
# run. WANDB_RESUME=allow tolerates either case (new or existing).
export WANDB_PROJECT=skillos
export WANDB_ENTITY=okaris
if [ -n "$CHECKPOINT" ]; then
  export WANDB_RUN_ID=ws8l5g2t   # v2 run (alfworld-8xh100-lora-v2); resume continues this curve
  export WANDB_RESUME=allow
fi

# SkillOS env-var knobs (defaults baked in code; overridden here when we want
# something different from the default).
export SKILLOS_PARALLEL_ROLLOUTS=16              # threadpool for concurrent rollouts
export SKILLOS_PARALLEL_JUDGES=16                # threadpool for concurrent judges
export SKILLOS_EXECUTOR_MAX_STEPS=30             # paper avg trajectory ~21 steps; default was 10 (clipped)
# Per-future timeouts and the judge retry budget have their own defaults +
# rationale in curator_env.py and rewards/judge.py; override them via the
# SKILLOS_*_TIMEOUT_S / SKILLOS_JUDGE_* env vars here if a run needs it.

# PyTorch CUDA allocator: switch to expandable segments to avoid the
# fragmentation OOM hit at step 15 on the prior run. The LoRA-on-DDP path
# leaves ~5–10 GB headroom per H100; without expandable segments, that
# headroom fragments into chunks too small to satisfy the few-GB backward-
# pass tensors and crashes even though enough memory is "reserved."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL collective watchdog: 1 hr. Belt-and-suspenders past the env-level
# per-future timeouts so a slow opt-step (curator gen on long prompts)
# doesn't trip the multi-rank watchdog.
export NCCL_TIMEOUT_MS=3600000

# Pipe resume path through to train.py via env var (config takes precedence
# if set; this is just the launcher convenience path).
if [ -n "$CHECKPOINT" ]; then
  export SKILLOS_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
fi

ts="$(date +%Y%m%d_%H%M%S)"
LOG="logs/alfworld_8xh100_${ts}.log"
echo "Launching SkillOS training → $LOG"
echo "  CHECKPOINT=${CHECKPOINT:-<fresh>}"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  NCCL_TIMEOUT_MS=$NCCL_TIMEOUT_MS"
echo "  SKILLOS_EXECUTOR_MAX_STEPS=$SKILLOS_EXECUTOR_MAX_STEPS"
[ -n "$CHECKPOINT" ] && echo "  WANDB_RESUME=$WANDB_RESUME WANDB_RUN_ID=$WANDB_RUN_ID"

accelerate launch \
  --num_processes 8 \
  --multi_gpu \
  --mixed_precision bf16 \
  -m skillos.train --config configs/alfworld_8xh100.yaml \
  > "$LOG" 2>&1
