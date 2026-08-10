#!/usr/bin/env bash
# Held-out checkpoint sweep for the verl-agent GiGPO run (60 steps, 2026-07-28).
# Modeled on seed3_sweep_supervisor.sh (proven pattern): API gate -> GPU-pinned
# waves -> storm detect/wipe/re-gate -> McNemar comparator vs no_memory.
#
# verl-specific: verl saves FSDP shards, so each arm needs the MERGED HF copy
# at /mnt/nvme/output/verl-merged-hf/step_N (see verl_merge_all.sh). Arms whose
# merge isn't ready yet are skipped and retried on the next outer pass.
#
# Defaults now target the REAL-ALFWorld run (real episodes + ground-truth
# success + BM25 + judged r_cnt), finished 2026-08-09 at 60/60 steps. The
# earlier pass measured the SIMPLIFIED scaffold and its outputs are kept at
# verl-merged-hf / output/eval-verl-gigpo.
#
# Both dirs MUST be fresh per run. arm_ok() treats any existing 140-line
# ckpt$N.jsonl as a finished arm, so pointing EVAL at a populated dir skips
# every arm and re-runs the comparator on the OLD numbers under a new name.
set -u
cd "$(dirname "$0")/.."

MERGED=${MERGED:-/mnt/nvme/output/verl-merged-hf-real}
EVAL=${EVAL:-output/eval-verl-gigpo-real}
BASE=${BASE:-output/eval-pathbv4/no_memory.jsonl}   # reusable 140-game baseline
ALL_CKPTS="5 10 15 20 25 30 35 40 45 50 55 60"
STORM_THRESHOLD=100
LOG=${LOG:-logs/verl_sweep_real.log}   # fresh log; the old run's ends in VERL_SWEEP_COMPLETE

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$EVAL" logs
ln -sf ../eval-pathbv4/no_memory.jsonl "$EVAL/no_memory.jsonl"
echo $$ > logs/verl_sweep.pid

log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" >> "$LOG"; }

probe_round () {  # 1 single + 10-burst + 40-burst; all must pass
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
  local streak=0 R
  while true; do
    R=$(probe_round)
    log "gate: ${R:-EMPTY} (streak=$streak)"
    if [ "${R:-}" = "PASS" ]; then
      streak=$((streak+1))
      [ "$streak" -ge 2 ] && { log "gate OPEN"; return 0; }
      sleep 120
    else
      streak=0; sleep 300
    fi
  done
}

arm_ok    () { [ -f "$EVAL/ckpt$1.jsonl" ] && [ "$(wc -l < "$EVAL/ckpt$1.jsonl")" -eq 140 ]; }
merge_ok  () { [ -f "$MERGED/step_$1/config.json" ] && ls "$MERGED/step_$1"/model-*.safetensors >/dev/null 2>&1; }

storm_count () {
  local tot=0 n f
  for f in logs/verl_sweep_ckpt*.log; do
    [ -f "$f" ] || continue
    n=$(grep -c "providers have been ignored\|All providers ignored" "$f" 2>/dev/null || true)
    n=${n:-0}; n=${n//[^0-9]/}
    tot=$((tot + ${n:-0}))
  done
  echo "$tot"
}

run_wave () {  # GPU-pinned arms, one per GPU; returns 1 if storm-aborted
  local gpu=0 ck pids=()
  for ck in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-checkpoint "$MERGED/step_$ck" \
      --num-games 140 --batch-size 20 --split valid_seen \
      --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
      --out "$EVAL/ckpt$ck.jsonl" > "logs/verl_sweep_ckpt$ck.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu+1))
  done
  while true; do
    local alive=0 p
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" -eq 0 ] && return 0
    if [ "$(storm_count)" -gt "$STORM_THRESHOLD" ]; then
      log "STORM ($(storm_count) provider errors) — killing wave, wiping partials"
      kill -9 "${pids[@]}" 2>/dev/null
      sleep 5
      pgrep -f 'eval_streaming[_]curation' | xargs -r kill -9 2>/dev/null
      for ck in "$@"; do arm_ok "$ck" || rm -f "$EVAL/ckpt$ck.jsonl"; done
      return 1
    fi
    sleep 120
  done
}

log "=================================================================="
log "verl GiGPO sweep start (12 ckpts, 140 held-out games each, valid_seen)"

while true; do
  REMAINING=(); WAITING=()
  for ck in $ALL_CKPTS; do
    arm_ok "$ck" && continue
    if merge_ok "$ck"; then REMAINING+=("$ck"); else WAITING+=("$ck"); fi
  done

  if [ "${#REMAINING[@]}" -eq 0 ] && [ "${#WAITING[@]}" -eq 0 ]; then break; fi

  if [ "${#REMAINING[@]}" -eq 0 ]; then
    log "all runnable arms done; waiting on merges for: ${WAITING[*]}"
    sleep 120; continue
  fi

  log "runnable: ${REMAINING[*]}${WAITING:+  (awaiting merge: ${WAITING[*]})}"
  gate
  rm -f logs/verl_sweep_ckpt*.log   # reset storm counter for this attempt

  set -- "${REMAINING[@]}"
  stormed=0
  while [ "$#" -gt 0 ]; do
    wave=("$1"); shift
    for _ in 1 2 3 4 5 6 7; do [ "$#" -gt 0 ] && { wave+=("$1"); shift; }; done
    log "WAVE: ${wave[*]}"
    if ! run_wave "${wave[@]}"; then stormed=1; break; fi
    log "wave done: ${wave[*]}"
  done
  [ "$stormed" -eq 1 ] && { log "storm -> back to gate"; sleep 300; }
done

log "ALL ARMS COMPLETE — running McNemar comparator"
CMP_ARGS=(--arm "no_memory=$BASE")
for ck in $ALL_CKPTS; do CMP_ARGS+=(--arm "ckpt$ck=$EVAL/ckpt$ck.jsonl"); done
.venv/bin/python -m scripts.compare_eval_arms "${CMP_ARGS[@]}" \
  | tee "$EVAL/comparison_canonical.txt" >> "$LOG"
log "VERL_SWEEP_COMPLETE"
