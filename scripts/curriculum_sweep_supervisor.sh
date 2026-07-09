#!/usr/bin/env bash
# Self-healing supervisor for the natural-distribution checkpoint sweep during
# the 2026-07-02 OpenRouter "All providers ignored" instability.
# State machine: GATE -> LAUNCH (4-arm waves) -> MONITOR -> on storm: KILL +
# WIPE poisoned partials + back to GATE (completed arms are kept) -> COMPARATOR.
set -u
cd "$(dirname "$0")/.."

OUT=output/alfworld-8xh100-algo1-fft-curriculum
EVAL=output/eval-fft-curriculum
BASE=output/eval-pathbv4/no_memory.jsonl
ALL_CKPTS="5 10 15 20 25 30 35 40 45"
STORM_THRESHOLD=50

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL"
ln -sf ../eval-pathbv4/no_memory.jsonl "$EVAL/no_memory.jsonl"

probe_round () {  # 1 single + 10-burst + 40-burst, all must pass
  timeout 600 .venv/bin/python -c "
from concurrent.futures import ThreadPoolExecutor
from inferencesh import inference
from skillos.utils.infsh_auth import resolve_infsh_api_key
client = inference(api_key=resolve_infsh_api_key())
def one(i):
    try:
        r = client.tasks.run({'app':'openrouter/qwen3-8b','infra':'cloud','variant':'default','input':{'text':f'say {i}','max_tokens':8,'temperature':0.6}})
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
      [ "$streak" -ge 2 ] && { echo "[$(date -u)] gate OPEN (2 consecutive passes)"; return 0; }
      sleep 300
    else
      streak=0
      sleep 300
    fi
  done
}

arm_ok () {  # completed = jsonl has 140 lines
  [ -f "$EVAL/ckpt$1.jsonl" ] && [ "$(wc -l < "$EVAL/ckpt$1.jsonl")" -eq 140 ]
}

storm_count () {
  local tot=0 n f
  for f in /tmp/curr_ckpt*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c "providers have been ignored" "$f") || n=0
    tot=$((tot+n))
  done
  echo "$tot"
}

run_wave () {  # up to 4 ckpts, GPU-pinned; returns 1 if storm-aborted
  local gpu=0 ck pids=()
  for ck in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-checkpoint "$OUT/checkpoint-$ck" \
      --num-games 140 --batch-size 20 --split valid_seen \
      --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
      --out "$EVAL/ckpt$ck.jsonl" > "/tmp/curr_ckpt$ck.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu+1))
  done
  while true; do
    local alive=0 p
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && return 0
    if [ "$(storm_count)" -gt "$STORM_THRESHOLD" ]; then
      echo "[$(date -u)] STORM ($(storm_count) errors) — killing wave"
      kill -9 "${pids[@]}" 2>/dev/null
      sleep 5
      pkill -9 -f "eval_streaming[_]curation" 2>/dev/null
      for ck in "$@"; do arm_ok "$ck" || rm -f "$EVAL/ckpt$ck.jsonl"; done
      return 1
    fi
    sleep 120
  done
}

while true; do
  # figure out what's still missing
  REMAINING=()
  for ck in $ALL_CKPTS; do arm_ok "$ck" || REMAINING+=("$ck"); done
  if [ "${#REMAINING[@]}" -eq 0 ]; then break; fi
  echo "[$(date -u)] remaining arms: ${REMAINING[*]}"

  gate
  rm -f /tmp/curr_ckpt*.log   # reset storm counter for the new attempt

  # run remaining in 4-arm waves; on storm, restart the whole loop (re-gate)
  set -- "${REMAINING[@]}"
  stormed=0
  while [ "$#" -gt 0 ]; do
    wave=("$1"); shift
    for _ in 1 2 3; do [ "$#" -gt 0 ] && { wave+=("$1"); shift; }; done
    echo "[$(date -u)] WAVE: ${wave[*]}"
    if ! run_wave "${wave[@]}"; then stormed=1; break; fi
    echo "[$(date -u)] wave done: ${wave[*]}"
  done
  [ "$stormed" -eq 1 ] && { echo "[$(date -u)] back to gate"; sleep 300; }
done

echo "[$(date -u)] ALL ARMS COMPLETE — running comparator"
CMP_ARGS=(--arm "no_memory=$BASE")
for ck in $ALL_CKPTS; do CMP_ARGS+=(--arm "ckpt$ck=$EVAL/ckpt$ck.jsonl"); done
.venv/bin/python -m scripts.compare_eval_arms "${CMP_ARGS[@]}" | tee "$EVAL/comparison_canonical.txt"
echo "[$(date -u)] SWEEP COMPLETE"
