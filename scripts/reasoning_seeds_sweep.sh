#!/usr/bin/env bash
# Eval sweep for reasoning-curator seeds 1/2/3 on ALFWorld valid_unseen.
# Contemporaneous no_memory baseline, then key checkpoints from each seed.
# No gate — API confirmed working.
set -u
cd "$(dirname "$0")/.."

EVAL=output/eval-reasoning-seeds
SPLIT=valid_unseen
NGAMES=134

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL"

arm_ok () {
  local f="$1"
  [ -f "$f" ] && [ "$(wc -l < "$f")" -eq "$NGAMES" ]
}

run_wave () {
  local pids=() gpu=0
  shift  # first arg is label, rest are "name:ckptpath" pairs
  for spec in "$@"; do
    local name="${spec%%:*}"
    local ckpt="${spec#*:}"
    echo "[$(date -u)] starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-checkpoint "$ckpt" \
      --num-games "$NGAMES" --batch-size 20 --split "$SPLIT" \
      --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
      --out "$EVAL/${name}.jsonl" > "/tmp/rsweep_${name}.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu+1))
  done
  echo "[$(date -u)] waiting for ${#pids[@]} arms..."
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  echo "[$(date -u)] wave done"
}

# ── Phase 1: contemporaneous no_memory baseline on valid_unseen ──
if ! arm_ok "$EVAL/no_memory.jsonl"; then
  echo "[$(date -u)] running contemporaneous no_memory baseline ($SPLIT, $NGAMES games)"
  .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode no_memory \
    --num-games "$NGAMES" --batch-size 20 --split "$SPLIT" \
    --out "$EVAL/no_memory.jsonl" > "/tmp/rsweep_no_memory.log" 2>&1
  if arm_ok "$EVAL/no_memory.jsonl"; then
    echo "[$(date -u)] baseline complete"
  else
    echo "[$(date -u)] baseline FAILED — check /tmp/rsweep_no_memory.log"
    exit 1
  fi
fi

# ── Phase 2: key checkpoints from all 3 seeds ──
# Seed-1 (original): ckpt45,50,55,60 (the neighbourhood around the positive)
# Seed-2 and Seed-3: same checkpoints for comparability
SEED1=output/reasoning-8xh100-algo1-fft
SEED2=output/reasoning-8xh100-algo1-fft-seed2
SEED3=output/reasoning-8xh100-algo1-fft-seed3

# Wave A: seed-1 ckpts (4 arms) + seed-2 ckpts (4 arms) = 8 GPUs
WAVE_A=()
for ck in 45 50 55 60; do
  arm_ok "$EVAL/s1_ckpt${ck}.jsonl" || WAVE_A+=("s1_ckpt${ck}:${SEED1}/checkpoint-${ck}")
done
for ck in 45 50 55 60; do
  arm_ok "$EVAL/s2_ckpt${ck}.jsonl" || WAVE_A+=("s2_ckpt${ck}:${SEED2}/checkpoint-${ck}")
done

if [ "${#WAVE_A[@]}" -gt 0 ]; then
  echo "[$(date -u)] WAVE A: ${WAVE_A[*]}"
  run_wave "A" "${WAVE_A[@]}"
fi

# Wave B: seed-3 ckpts (4 arms)
WAVE_B=()
for ck in 45 50 55 60; do
  arm_ok "$EVAL/s3_ckpt${ck}.jsonl" || WAVE_B+=("s3_ckpt${ck}:${SEED3}/checkpoint-${ck}")
done

if [ "${#WAVE_B[@]}" -gt 0 ]; then
  echo "[$(date -u)] WAVE B: ${WAVE_B[*]}"
  run_wave "B" "${WAVE_B[@]}"
fi

echo "[$(date -u)] ALL ARMS COMPLETE — running comparator"
CMP_ARGS=(--arm "no_memory=$EVAL/no_memory.jsonl")
for s in s1 s2 s3; do
  for ck in 45 50 55 60; do
    [ -f "$EVAL/${s}_ckpt${ck}.jsonl" ] && CMP_ARGS+=(--arm "${s}_ckpt${ck}=$EVAL/${s}_ckpt${ck}.jsonl")
  done
done
.venv/bin/python -m scripts.compare_eval_arms "${CMP_ARGS[@]}" | tee "$EVAL/comparison_canonical.txt"
echo "[$(date -u)] SWEEP COMPLETE"
