#!/usr/bin/env bash
# Re-run ALFWorld eval arms under the fixed harness.
#
# Why this exists: the old harness answered an executor API failure by playing
# admissible[0] and scoring the episode as a normal task failure. Four r2a arms
# ran 52-65% invented actions and eval-v8 ckpt60 ran 12%, so their numbers are
# artifacts. The harness now abandons the episode, excludes it from the rate,
# counts action coercions, and aborts an arm that loses more than
# SKILLOS_EVAL_MAX_ERROR_RATE of its episodes.
#
# Two deliberate choices here:
#   1. Output goes to output/reeval/... — the historical files are evidence and
#      are never overwritten.
#   2. Request rate is LOWER and patience is HIGHER than the original sweeps.
#      The old settings (max_resubs=2, poll 150s) were tuned to fail fast so a
#      stuck training rollout could not storm inference.sh. Eval has no NCCL
#      lockstep, so failing fast buys nothing and costs data.
set -uo pipefail
cd "$(dirname "$0")/.."

WAVE="${1:-a}"
CONC="${REEVAL_CONCURRENCY:-3}"          # arms in flight (was 8 in old sweeps)
BATCH="${REEVAL_BATCH:-10}"             # games per wave (was 20)
GAMES="${REEVAL_GAMES:-140}"

# Patient, low-rate executor retry. Backoff absorbs a 429 instead of escalating.
export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=6
export SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2
export SKILLOS_EXEC_BACKOFF_BASE_S=20
export SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE="${SKILLOS_EVAL_MAX_ERROR_RATE:-0.02}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

FFT1=output/alfworld-8xh100-algo1-fft
FFT2=output/alfworld-8xh100-algo1-fft-seed2
FFT3=output/alfworld-8xh100-algo1-fft-seed3
REAS=output/reasoning-8xh100-algo1-fft
VERL=output/verl-merged-hf-real

mkdir -p logs/reeval

# queue entries: name|mode|curator_ckpt|executor_app|out
QUEUE=()

add () { QUEUE+=("$1|$2|$3|$4|$5"); }

case "$WAVE" in
a)
  # Baselines first: every paired test is measured against these, so they have
  # to exist under the fixed harness before any arm means anything.
  add nomem_8b   no_memory   -  openrouter/qwen3-8b  output/reeval/baseline/no_memory_8b.jsonl
  add nomem_32b  no_memory   -  openrouter/qwen3-32b output/reeval/baseline/no_memory_32b.jsonl
  # The four voided reasoning-transfer arms (52-65% invented actions).
  for ck in 45 50 55 60; do
    add "r2a_ckpt$ck" closed_loop "$REAS/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/reasoning-to-alfworld/ckpt$ck.jsonl"
  done
  # eval-fft arms with nonzero contamination (0.03-0.60%).
  for ck in 25 30 35 40 60; do
    add "fft1_ckpt$ck" closed_loop "$FFT1/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/fft/ckpt$ck.jsonl"
  done
  ;;
b)
  # Full clean sweeps, for consistency with the re-run baselines.
  for ck in 5 10 15 20 30 35 45 50 55; do
    add "r2a_ckpt$ck" closed_loop "$REAS/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/reasoning-to-alfworld/ckpt$ck.jsonl"
  done
  for ck in 5 10 15 20 45 50 55; do
    add "fft1_ckpt$ck" closed_loop "$FFT1/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/fft/ckpt$ck.jsonl"
  done
  for ck in 5 10 15 20 25 30 35 40 45 50 55 60; do
    add "fft2_ckpt$ck" closed_loop "$FFT2/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/fft-seed2/ckpt$ck.jsonl"
    add "fft3_ckpt$ck" closed_loop "$FFT3/checkpoint-$ck" openrouter/qwen3-8b \
        "output/reeval/fft-seed3/ckpt$ck.jsonl"
  done
  ;;
c)
  # 32B-executor transfer + verl. Same curators, bigger executor.
  for ck in 5 10 15 20 25 30 35 40 45 50 55 60; do
    add "t32_s2_ckpt$ck" closed_loop "$FFT2/checkpoint-$ck" openrouter/qwen3-32b \
        "output/reeval/transfer-32b-seed2/ckpt$ck.jsonl"
    add "t32_s3_ckpt$ck" closed_loop "$FFT3/checkpoint-$ck" openrouter/qwen3-32b \
        "output/reeval/transfer-32b-seed3/ckpt$ck.jsonl"
  done
  for ck in 5 10 15 20 25 30 35 40 45 50 55 60; do
    add "verl_ckpt$ck" closed_loop "$VERL/step_$ck" openrouter/qwen3-8b \
        "output/reeval/verl-gigpo-real/ckpt$ck.jsonl"
  done
  ;;
*) echo "usage: $0 {a|b|c}"; exit 2 ;;
esac

echo "[$(date -u)] wave=$WAVE arms=${#QUEUE[@]} concurrency=$CONC batch=$BATCH"

launch () {
  local spec="$1"
  IFS='|' read -r name mode ckpt app out <<< "$spec"
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then
    echo "[$(date -u)] SKIP $name (exists: $out)"
    return 0
  fi
  local args=(--mode "$mode" --num-games "$GAMES" --batch-size "$BATCH"
              --split valid_seen --executor-app "$app" --out "$out")
  if [ "$mode" = closed_loop ]; then
    args+=(--curator-checkpoint "$ckpt" --curator-device cuda
           --curator-temperature 0 --curator-max-new-tokens 1536)
  fi
  echo "[$(date -u)] START $name -> $out"
  CUDA_VISIBLE_DEVICES="$((RUNNING % 8))" \
    .venv/bin/python -u -m scripts.eval_streaming_curation "${args[@]}" \
    > "logs/reeval/$name.log" 2>&1 &
}

RUNNING=0
pids=()
names=()
for spec in "${QUEUE[@]}"; do
  # Block until a slot frees.
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 20; done
  launch "$spec"
  pids+=($!)
  names+=("${spec%%|*}")
  RUNNING=$((RUNNING+1))
  sleep 5   # stagger submissions so a wave start is not a request spike
done

echo "[$(date -u)] all ${#pids[@]} arms dispatched; waiting"
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    rc=$?
    echo "[$(date -u)] FAILED ${names[$i]} (rc=$rc)"
    [ "$rc" = "3" ] && echo "   ^ aborted by the data-integrity gate, not a crash"
    fail=$((fail+1))
  fi
done
echo "[$(date -u)] wave=$WAVE done; failures=$fail"
