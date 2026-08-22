#!/usr/bin/env bash
# Eval sweep for dense10 using LOCAL vLLM executor (no inference.sh credits needed).
# GPU 7: vLLM executor server (already running on port 8002)
# GPUs 0-6: curator checkpoints (one per arm, 7 arms per wave)
set -u
cd "$(dirname "$0")/.."

WEIGHTS=output/alfworld-dense-fft-paperloop/eval_weights
EVAL=output/eval-dense10
ALL_CKPTS="5 10 15 20 25 30 35 40 45 50 55 60"

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL"

arm_ok () {
  local f="$1"
  [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 140 ]
}

run_wave () {
  local gpu=0 ck pids=()
  for ck in "$@"; do
    echo "[$(date -u)] starting ckpt$ck on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-checkpoint "$WEIGHTS/step-$ck" \
      --num-games 140 --batch-size 5 --split valid_seen \
      --executor vllm \
      --executor-temperature 0.6 --executor-top-p 0.95 --executor-top-k 20 \
      --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
      --out "$EVAL/ckpt$ck.jsonl" > "/tmp/dense10_ckpt$ck.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu+1))
  done
  echo "[$(date -u)] waiting for ${#pids[@]} arms..."
  local p
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  echo "[$(date -u)] wave done"
}

# ── Phase 1: contemporaneous no_memory baseline ──
if ! arm_ok "$EVAL/no_memory.jsonl"; then
  echo "[$(date -u)] running contemporaneous no_memory baseline (no curator, no GPU needed)"
  .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode no_memory \
    --num-games 140 --batch-size 5 --split valid_seen \
    --executor vllm \
    --executor-temperature 0.6 --executor-top-p 0.95 --executor-top-k 20 \
    --out "$EVAL/no_memory.jsonl" > "/tmp/dense10_no_memory.log" 2>&1
  if arm_ok "$EVAL/no_memory.jsonl"; then
    echo "[$(date -u)] baseline complete"
  else
    echo "[$(date -u)] baseline FAILED — check /tmp/dense10_no_memory.log"
    exit 1
  fi
fi

# ── Phase 2: checkpoint arms in 7-arm waves (GPUs 0-6) ──
while true; do
  REMAINING=()
  for ck in $ALL_CKPTS; do arm_ok "$EVAL/ckpt$ck.jsonl" || REMAINING+=("$ck"); done
  if [ "${#REMAINING[@]}" -eq 0 ]; then break; fi
  echo "[$(date -u)] remaining arms: ${REMAINING[*]}"

  set -- "${REMAINING[@]}"
  while [ "$#" -gt 0 ]; do
    wave=("$1"); shift
    for _ in 1 2 3; do [ "$#" -gt 0 ] && { wave+=("$1"); shift; }; done
    echo "[$(date -u)] WAVE: ${wave[*]}"
    run_wave "${wave[@]}"
  done
done

echo "[$(date -u)] ALL ARMS COMPLETE — running comparator"
CMP_ARGS=(--arm "no_memory=$EVAL/no_memory.jsonl")
for ck in $ALL_CKPTS; do CMP_ARGS+=(--arm "ckpt$ck=$EVAL/ckpt$ck.jsonl"); done
.venv/bin/python -m scripts.compare_eval_arms "${CMP_ARGS[@]}" | tee "$EVAL/comparison_canonical.txt"
echo "[$(date -u)] SWEEP COMPLETE"
