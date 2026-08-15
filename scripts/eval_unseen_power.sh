#!/usr/bin/env bash
# valid_unseen power experiment: 3 arms, fresh split, matched epoch.
#
# WHY. On valid_seen the trained curator beats the Gemini-2.5-Pro curator by
# +5.7pp but at p=0.31 — directionally the paper's claim, statistically nothing.
# 140 games is not enough power for a 5pp effect. valid_unseen adds a second,
# genuinely held-out 140 games. Pooling the two splits roughly doubles n.
#
# It is also a cleaner protocol than valid_seen alone: the arm being tested
# (reasoning-curator checkpoint-50) was SELECTED as best-of-5 on valid_seen, so
# valid_seen cannot honestly confirm it. valid_unseen is a true test set for it.
#
# All three arms run in the same week against each other. Per
# canonical-baseline-was-drift-outlier, never pair across measurement epochs.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=output/reeval/unseen-power
mkdir -p "$OUT" logs/reeval

# Patient retry: eval must never coerce or abandon for want of a retry.
export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=6
export SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2
export SKILLOS_EXEC_BACKOFF_BASE_S=20
export SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE="${SKILLOS_EVAL_MAX_ERROR_RATE:-0.02}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# valid_unseen holds exactly 134 games. Asking for 140 would wrap the env and
# double-count six of them, which silently breaks the pairing.
GAMES="${UNSEEN_GAMES:-134}"
BATCH="${UNSEEN_BATCH:-15}"
GPU="${UNSEEN_GPU:-4}"          # 0-3 are waves B/C, 6-7 are the reasoning sweep
R2A=output/reasoning-8xh100-algo1-fft/checkpoint-50

launch () {   # $1 = name, $2.. = args
  local name=$1; shift
  if [ -s "$OUT/$name.jsonl" ]; then echo "[$(date -u)] SKIP $name"; return 0; fi
  echo "[$(date -u)] START $name"
  .venv/bin/python -u -m scripts.eval_streaming_curation \
    --num-games "$GAMES" --batch-size "$BATCH" --split valid_unseen \
    --out "$OUT/$name.jsonl" "$@" \
    > "logs/reeval/unseen_$name.log" 2>&1 &
}

# Arm A: no notes. No curator, no GPU.
launch no_memory --mode no_memory

# Arm B: Gemini 2.5 Pro as curator. Remote, no GPU.
launch gemini_curator --mode closed_loop \
  --curator-backend remote --curator-app google/gemini-2-5-pro \
  --curator-temperature 0

# Arm C: the trained 8B curator (best-of-5 on valid_seen: reasoning ckpt50).
export CUDA_VISIBLE_DEVICES="$GPU"
launch r2a_ckpt50 --mode closed_loop \
  --curator-checkpoint "$R2A" --curator-device cuda \
  --curator-temperature 0 --curator-max-new-tokens 1536
unset CUDA_VISIBLE_DEVICES

wait
echo "[$(date -u)] unseen power experiment done"
grep -H "=== done\|data integrity" logs/reeval/unseen_*.log

.venv/bin/python -m scripts.compare_eval_arms \
  --arm "no_memory=$OUT/no_memory.jsonl" \
  --arm "gemini_curator=$OUT/gemini_curator.jsonl" \
  --arm "r2a_ckpt50=$OUT/r2a_ckpt50.jsonl" \
  | tee "$OUT/comparison.txt"
