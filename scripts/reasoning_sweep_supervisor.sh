#!/usr/bin/env bash
# Reasoning-curator checkpoint sweep: closed-loop eval of all 12 reasoning
# checkpoints on AIME24+25 + GPQA-Diamond (258 problems each).
# 8-arm waves (one GPU per arm for CuratorInference), storm-resilient.
set -u
cd "$(dirname "$0")/.."

TRAIN_OUT=output/reasoning-8xh100-algo1-fft
EVAL=output/eval-reasoning-sweep
ALL_CKPTS="5 10 15 20 25 30 35 40 45 50 55 60"
STORM_THRESHOLD=80
N_PROBLEMS=258  # 60 AIME + 198 GPQA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL" logs

probe_round () {
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
    echo "[$(date -u)] gate: $R (streak=$streak)" >> logs/reasoning_sweep.log
    if [ "$R" = "PASS" ]; then
      streak=$((streak+1))
      [ "$streak" -ge 2 ] && { echo "[$(date -u)] gate OPEN" >> logs/reasoning_sweep.log; return 0; }
      sleep 300
    else
      streak=0
      sleep 300
    fi
  done
}

arm_ok () {
  [ -f "$EVAL/ckpt$1.jsonl" ] && [ "$(wc -l < "$EVAL/ckpt$1.jsonl")" -ge "$N_PROBLEMS" ]
}

storm_count () {
  local tot=0 n f
  for f in logs/reasoning_sweep_ckpt*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c "providers have been ignored\|429\|rate.limit" "$f" 2>/dev/null) || n=0
    tot=$((tot+n))
  done
  echo "$tot"
}

run_wave () {
  local gpu=0 ck pids=()
  for ck in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_reasoning \
      --mode closed_loop \
      --curator-checkpoint "$TRAIN_OUT/checkpoint-$ck" \
      --dataset all \
      --executor-app openrouter/qwen3-8b \
      --max-tokens 8192 --temperature 0.6 --top-p 0.95 --reasoning medium \
      --parallel 16 \
      --out "$EVAL/ckpt$ck.jsonl" > "logs/reasoning_sweep_ckpt$ck.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu+1))
  done
  echo "[$(date -u)] wave PIDs: ${pids[*]}" >> logs/reasoning_sweep.log

  while true; do
    local alive=0 p
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && return 0
    if [ "$(storm_count)" -gt "$STORM_THRESHOLD" ]; then
      echo "[$(date -u)] STORM ($(storm_count) errors) — killing wave" >> logs/reasoning_sweep.log
      kill -9 "${pids[@]}" 2>/dev/null
      sleep 5
      pkill -9 -f "eval_reasoning" 2>/dev/null
      for ck in "$@"; do arm_ok "$ck" || rm -f "$EVAL/ckpt$ck.jsonl"; done
      return 1
    fi
    sleep 120
  done
}

echo "[$(date -u)] reasoning sweep supervisor started" >> logs/reasoning_sweep.log

while true; do
  REMAINING=()
  for ck in $ALL_CKPTS; do arm_ok "$ck" || REMAINING+=("$ck"); done
  if [ "${#REMAINING[@]}" -eq 0 ]; then break; fi
  echo "[$(date -u)] remaining arms: ${REMAINING[*]}" >> logs/reasoning_sweep.log

  gate
  rm -f logs/reasoning_sweep_ckpt*.log

  set -- "${REMAINING[@]}"
  stormed=0
  while [ "$#" -gt 0 ]; do
    wave=("$1"); shift
    for _ in 1 2 3 4 5 6 7; do [ "$#" -gt 0 ] && { wave+=("$1"); shift; }; done
    echo "[$(date -u)] WAVE: ${wave[*]}" >> logs/reasoning_sweep.log
    if ! run_wave "${wave[@]}"; then stormed=1; break; fi
    echo "[$(date -u)] wave done: ${wave[*]}" >> logs/reasoning_sweep.log
  done
  [ "$stormed" -eq 1 ] && { echo "[$(date -u)] back to gate" >> logs/reasoning_sweep.log; sleep 300; }
done

echo "[$(date -u)] ALL ARMS COMPLETE" >> logs/reasoning_sweep.log

# Summary
.venv/bin/python -c "
import json, sys, os
eval_dir = '$EVAL'
base_aime = [json.loads(l) for l in open('output/eval-reasoning/nomem_aime.jsonl')]
base_gpqa = [json.loads(l) for l in open('output/eval-reasoning/nomem_gpqa.jsonl')]
base_all = base_aime + base_gpqa
base_ok = sum(r['correct'] for r in base_all)
base_n = len(base_all)
print(f'baseline: {base_ok}/{base_n} = {100*base_ok/base_n:.1f}%')
print()
for ck in [5,10,15,20,25,30,35,40,45,50,55,60]:
    fp = os.path.join(eval_dir, f'ckpt{ck}.jsonl')
    if not os.path.exists(fp): print(f'ckpt{ck}: MISSING'); continue
    rows = [json.loads(l) for l in open(fp)]
    ok = sum(r['correct'] for r in rows); n = len(rows)
    aime = [r for r in rows if r['kind']=='aime']; gpqa = [r for r in rows if r['kind'] in ('gpqa','gpqa_ft')]
    aime_ok = sum(r['correct'] for r in aime); gpqa_ok = sum(r['correct'] for r in gpqa)
    delta = 100*ok/n - 100*base_ok/base_n if n else 0
    print(f'ckpt{ck:02d}: {ok}/{n} = {100*ok/n:.1f}% (delta={delta:+.1f}pp) | AIME {aime_ok}/{len(aime)} = {100*aime_ok/len(aime):.1f}% | GPQA {gpqa_ok}/{len(gpqa)} = {100*gpqa_ok/len(gpqa):.1f}%')
" 2>&1 | tee "$EVAL/summary.txt"

echo "[$(date -u)] SWEEP COMPLETE" >> logs/reasoning_sweep.log
