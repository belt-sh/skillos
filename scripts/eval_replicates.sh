#!/usr/bin/env bash
# Raise statistical power by replicating rollouts per game.
#
# ALFWorld has only 274 valid games, so paired n cannot be increased by adding
# games. MDE at 80% power on 140 games is 13.0pp, which is the size of the
# effect this literature reports; see docs/paper/05b_power.md. The only route
# left is repeated rollouts per game, analysed as per-game success RATES with a
# paired test, which suppresses measurement variance without changing the game
# population.
#
# Three arms x two additional rollouts on valid_unseen, giving three rollouts
# per game per arm. Executor temperature stays at the paper's 0.6, so replicates
# genuinely resample the executor's policy rather than repeating one trace.
set -uo pipefail
cd "$(dirname "$0")/.."
export SKILLOS_EXECUTOR_MAX_STEPS=30 SKILLOS_EXEC_MAX_RESUBS=6 SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2 SKILLOS_EXEC_BACKOFF_BASE_S=20 SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE=0.02 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=output/reeval/unseen-replicates
REAS=output/reasoning-8xh100-algo1-fft
mkdir -p "$OUT" logs/reeval
GPU="${REPL_GPU:-7}"

for rep in 2 3; do
  if [ ! -s "$OUT/no_memory_r$rep.jsonl" ]; then
    .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode no_memory --num-games 134 --batch-size 15 --split valid_unseen \
      --out "$OUT/no_memory_r$rep.jsonl" > "logs/reeval/repl_nomem_r$rep.log" 2>&1 &
  fi
  if [ ! -s "$OUT/gemini_r$rep.jsonl" ]; then
    .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-backend remote --curator-app google/gemini-2-5-pro \
      --curator-temperature 0 \
      --num-games 134 --batch-size 15 --split valid_unseen \
      --out "$OUT/gemini_r$rep.jsonl" > "logs/reeval/repl_gemini_r$rep.log" 2>&1 &
  fi
  if [ ! -s "$OUT/r2a_ckpt50_r$rep.jsonl" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-checkpoint "$REAS/checkpoint-50" --curator-device cuda \
      --curator-temperature 0 --curator-max-new-tokens 1536 \
      --num-games 134 --batch-size 15 --split valid_unseen \
      --out "$OUT/r2a_ckpt50_r$rep.jsonl" > "logs/reeval/repl_r2a_r$rep.log" 2>&1 &
  fi
  wait
  echo "[$(date -u)] replicate $rep done"
done
grep -H "data integrity" logs/reeval/repl_*.log
