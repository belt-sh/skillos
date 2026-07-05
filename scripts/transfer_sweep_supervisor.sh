#!/usr/bin/env bash
# Cross-executor transfer sweep: best 8B-trained curators driving the 32B
# executor (paper's generalization claim). 4 arms, one wave, self-healing
# against provider thin-pool storms (same pattern as natural_sweep_supervisor).
set -u
cd "$(dirname "$0")/.."

EVAL=output/eval-transfer-32b
APP=openrouter/qwen3-32b
STORM_THRESHOLD=50
mkdir -p "$EVAL"

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# name|checkpoint(empty=no_memory)|gpu
ARMS="no_memory||-
fft_s1_ckpt20|output/alfworld-8xh100-algo1-fft/checkpoint-20|0
fft_s2_ckpt35|output/alfworld-8xh100-algo1-fft-seed2/checkpoint-35|1
v8lora_ckpt30|output/alfworld-8xh100-algo1-v8-lora-kl/checkpoint-30|2"

probe_round () {
  timeout 600 .venv/bin/python -c "
from concurrent.futures import ThreadPoolExecutor
from inferencesh import inference
from skillos.utils.infsh_auth import resolve_infsh_api_key
client = inference(api_key=resolve_infsh_api_key())
def one(i):
    try:
        r = client.tasks.run({'app':'$APP','infra':'cloud','variant':'default','input':{'text':f'say {i}','max_tokens':8,'temperature':0.6}})
        return 'ok' if r and r.get('output') else 'bad'
    except Exception:
        return 'bad'
if one(0) != 'ok':
    print('FAIL-single'); raise SystemExit
with ThreadPoolExecutor(10) as p:
    if any(x!='ok' for x in p.map(one, range(10))):
        print('FAIL-10'); raise SystemExit
with ThreadPoolExecutor(40) as p:
    bad = sum(1 for x in p.map(one, range(40)) if x!='ok')
print('PASS' if bad==0 else f'FAIL-40({bad})')
" 2>/dev/null | tail -1
}

gate () {
  local streak=0
  while true; do
    R=$(probe_round)
    echo "[$(date -u)] gate: $R (streak=$streak)"
    if [ "$R" = "PASS" ]; then
      streak=$((streak+1))
      [ "$streak" -ge 2 ] && { echo "[$(date -u)] gate OPEN"; return 0; }
    else
      streak=0
    fi
    sleep 300
  done
}

arm_ok () {
  [ -f "$EVAL/$1.jsonl" ] && [ "$(wc -l < "$EVAL/$1.jsonl")" -eq 140 ]
}

storm_count () {
  local tot=0 n f
  for f in /tmp/transfer_*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c "providers have been ignored" "$f") || n=0
    tot=$((tot+n))
  done
  echo "$tot"
}

while true; do
  REMAINING=()
  while IFS='|' read -r name ckpt gpu; do
    arm_ok "$name" || REMAINING+=("$name|$ckpt|$gpu")
  done <<< "$ARMS"
  [ "${#REMAINING[@]}" -eq 0 ] && break
  echo "[$(date -u)] remaining arms: ${REMAINING[*]%%|*}"

  gate
  rm -f /tmp/transfer_*.log

  pids=()
  for spec in "${REMAINING[@]}"; do
    IFS='|' read -r name ckpt gpu <<< "$spec"
    if [ -z "$ckpt" ]; then
      CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.eval_streaming_curation \
        --mode no_memory --executor infsh --executor-app "$APP" \
        --num-games 140 --batch-size 20 --split valid_seen \
        --out "$EVAL/$name.jsonl" > "/tmp/transfer_$name.log" 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
        --mode closed_loop --curator-checkpoint "$ckpt" \
        --executor infsh --executor-app "$APP" \
        --num-games 140 --batch-size 20 --split valid_seen \
        --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
        --out "$EVAL/$name.jsonl" > "/tmp/transfer_$name.log" 2>&1 &
    fi
    pids+=($!)
  done

  stormed=0
  while true; do
    alive=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && break
    if [ "$(storm_count)" -gt "$STORM_THRESHOLD" ]; then
      echo "[$(date -u)] STORM ($(storm_count)) — killing wave"
      kill -9 "${pids[@]}" 2>/dev/null; sleep 5
      pkill -9 -f "eval_streaming[_]curation" 2>/dev/null
      while IFS='|' read -r name ckpt gpu; do
        arm_ok "$name" || rm -f "$EVAL/$name.jsonl"
      done <<< "$ARMS"
      stormed=1; break
    fi
    sleep 120
  done
  [ "$stormed" -eq 1 ] && { echo "[$(date -u)] back to gate"; sleep 300; }
done

echo "[$(date -u)] ALL ARMS COMPLETE — comparator (ref = 32B no_memory)"
.venv/bin/python -m scripts.compare_eval_arms \
  --arm "no_memory=$EVAL/no_memory.jsonl" \
  --arm "fft_s1_ckpt20=$EVAL/fft_s1_ckpt20.jsonl" \
  --arm "fft_s2_ckpt35=$EVAL/fft_s2_ckpt35.jsonl" \
  --arm "v8lora_ckpt30=$EVAL/v8lora_ckpt30.jsonl" \
  | tee "$EVAL/comparison.txt"
echo "[$(date -u)] SWEEP COMPLETE"
