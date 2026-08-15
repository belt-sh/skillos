#!/usr/bin/env bash
# Re-run the reasoning evals under the fixed harness.
#
# scripts/eval_reasoning.py used to score an upstream API failure as a WRONG
# ANSWER (`correct: False` plus an `error` field). Contamination was small, 0 to
# 1.6% per arm, and could have been corrected arithmetically from the recorded
# error fields, but a re-run is the honest fix: an arm should be measured, not
# reconstructed.
#
# GPQA HANDLING. These outputs contain GPQA-Diamond problem text, model
# responses and gold answers. Dataset access is conditional on none of that
# reaching a git-tracked or web-visible file. So:
#   - outputs go to output/reeval/reasoning/, which is inside gitignored output/
#   - the reasoning dirs stay OUT of EVAL_DIRS in hf_publish_artifacts.sh
#   - that script now also greps staged CONTENT for gpqa markers, not just names
# Only aggregate accuracies may be quoted anywhere public.
set -uo pipefail
cd "$(dirname "$0")/.."

CONC="${REEVAL_R_CONCURRENCY:-2}"     # arms in flight; each pins one GPU
PAR="${REEVAL_R_PARALLEL:-8}"         # problems in flight per arm (was 16)
GPU_BASE="${REEVAL_R_GPU_BASE:-4}"    # waves B/C are using GPUs 0-3

export SKILLOS_EXEC_MAX_RESUBS=6
export SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2
export SKILLOS_EXEC_BACKOFF_BASE_S=20
export SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE="${SKILLOS_EVAL_MAX_ERROR_RATE:-0.02}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REAS=output/reasoning-8xh100-algo1-fft
OUT=output/reeval/reasoning
mkdir -p "$OUT" logs/reeval

run_arm () {   # $1 = name, $2 = gpu, $3.. = extra args
  local name=$1 gpu=$2; shift 2
  if [ -s "$OUT/$name.jsonl" ]; then echo "[$(date -u)] SKIP $name"; return 0; fi
  echo "[$(date -u)] START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u -m scripts.eval_reasoning \
    --dataset all --parallel "$PAR" --out "$OUT/$name.jsonl" "$@" \
    > "logs/reeval/reasoning_$name.log" 2>&1 &
}

# Baselines first (no curator, no GPU needed, but keep the arg shape uniform).
run_arm nomem 0 --mode no_memory
wait

i=0
for ck in 5 10 15 20 25 30 35 40 45 50 55 60; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 30; done
  run_arm "ckpt$ck" "$((GPU_BASE + i % 4))" \
    --mode closed_loop --curator-checkpoint "$REAS/checkpoint-$ck"
  i=$((i+1))
  sleep 10
done
wait

echo "[$(date -u)] reasoning re-eval done"
grep -H "TOTAL:\|data integrity" logs/reeval/reasoning_*.log
