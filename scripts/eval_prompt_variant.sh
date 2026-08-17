#!/usr/bin/env bash
# Does dropping the <think> instruction help, hurt, or do nothing?
#
# The paper's Figure 9 prompt tells the executor to reason inside <think></think>.
# Measured 2026-08-17: on a native-reasoning endpoint the model never complies —
# reasoning goes to the provider's separate channel and the response is just the
# action. So the instruction may be dead weight, or it may be actively causing
# unparseable output (the coercion source, and a candidate for part of the 8pp
# absolute baseline gap).
#
# "The model ignores it anyway" is a hypothesis, so measure it: two no-memory arms
# on the same 140 games, same week, differing only in that one sentence. No GPU
# needed (no curator), so this runs alongside training; inference.sh autoscales.
set -uo pipefail
cd "$(dirname "$0")/.."
export SKILLOS_EXECUTOR_MAX_STEPS=30 SKILLOS_EXEC_MAX_RESUBS=6 SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2 SKILLOS_EXEC_BACKOFF_BASE_S=20 SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE=0.02
export SKILLOS_EXECUTOR_RETRY_PARSE=1
OUT=output/reeval/prompt-variant
mkdir -p "$OUT" logs/reeval

for mode in paper native; do
  [ -s "$OUT/nomem_$mode.jsonl" ] && { echo "SKIP $mode"; continue; }
  SKILLOS_EXECUTOR_PROMPT="$mode" .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode no_memory --num-games 140 --batch-size 15 --split valid_seen \
    --out "$OUT/nomem_$mode.jsonl" > "logs/reeval/prompt_$mode.log" 2>&1 &
done
wait
echo "[$(date -u)] prompt variant arms done"
grep -H "TOTAL:\|data integrity\|reformat retries" logs/reeval/prompt_paper.log logs/reeval/prompt_native.log
.venv/bin/python -m scripts.paper_stats --family "prompt-variant" \
  --base "$OUT/nomem_paper.jsonl" --arm "native_thinking=$OUT/nomem_native.jsonl"
