#!/usr/bin/env bash
# Self-healing supervisor for the verl-agent GiGPO SkillOS run.
#
# State machine per attempt:
#   PREFLIGHT (tailscaled RSS guard + DNS gate) -> LAUNCH -> WAIT
#   -> on exit: if ckpt advanced => resume; if stalled N times => FATAL stop.
#
# Root causes this guards against, learned the hard way:
#   * tailscaled memory leak ballooning to ~1.8TB, which trips Ray's node-OOM
#     killer and takes the trainer down (2026-07-23 crash at step 19).
#   * transient DNS failure for api.inference.sh (same leak breaks resolved).
#
# Durable logs only — never /tmp (tmp-cleaner has eaten a watcher log before).
set -u

VERL=/home/ubuntu/verl-skillos
CKPT=/mnt/nvme/output/verl-skillos-checkpoints
LOGDIR=/home/ubuntu/skillos/logs
SUP_LOG="$LOGDIR/verl_supervisor.log"
TRAIN_LOG="$LOGDIR/verl_skillos_gigpo_alfworld.log"
TOTAL_STEPS=60
MAX_STALLED_ATTEMPTS=3          # consecutive attempts with zero ckpt progress
TAILSCALED_RSS_LIMIT_KB=$((100 * 1024 * 1024))   # 100 GB

mkdir -p "$LOGDIR"
PIDFILE="$LOGDIR/verl_supervisor.pid"
echo $$ > "$PIDFILE"
# Stop this supervisor with:  kill -9 $(cat logs/verl_supervisor.pid)
# Do NOT use `pgrep -f verl_supervisor` — that pattern also matches watchers
# tailing verl_supervisor.log, and killing those loses your tripwires.

log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" >> "$SUP_LOG"; }

latest_ckpt () {
  local f="$CKPT/latest_checkpointed_iteration.txt"
  if [ -f "$f" ]; then tr -dc '0-9' < "$f"; else echo 0; fi
}

# --- guard: tailscaled leak -------------------------------------------------
guard_tailscaled () {
  local pid rss
  pid=$(pgrep -x tailscaled | head -1) || return 0
  [ -z "$pid" ] && return 0
  rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
  [ -z "$rss" ] && return 0
  if [ "$rss" -gt "$TAILSCALED_RSS_LIMIT_KB" ]; then
    log "GUARD tailscaled rss=$((rss/1048576))GB > limit — restarting it"
    sudo systemctl restart tailscaled >/dev/null 2>&1
    sleep 20
    log "GUARD tailscaled restarted; node RAM used=$(free -g | awk '/^Mem:/{print $3}')GB"
  fi
}

# --- guard: DNS + infsh reachability ---------------------------------------
gate_network () {
  local tries=0
  while [ "$tries" -lt 60 ]; do
    if getent hosts api.inference.sh >/dev/null 2>&1; then return 0; fi
    tries=$((tries+1))
    log "GATE dns for api.inference.sh failed (try $tries) — sleeping 60s"
    sleep 60
  done
  log "GATE dns never recovered after 60 tries"
  return 1
}

# --- clean stale partial checkpoint ---------------------------------------
# A crash mid-save leaves an undersized global_step_N. verl's resume would
# choke on it, so drop any step dir that is not the recorded latest.
clean_partial_ckpts () {
  local latest part n
  latest=$(latest_ckpt)
  for part in "$CKPT"/global_step_*; do
    [ -d "$part" ] || continue
    n=$(basename "$part" | tr -dc '0-9')
    if [ "$n" -gt "$latest" ]; then
      log "CLEAN dropping partial checkpoint $part (> latest=$latest)"
      rm -rf "$part"
    fi
  done
}

kill_stragglers () {
  pgrep -f 'main[_]ppo' | xargs -r kill -9 2>/dev/null
  pgrep -f 'r[a]y::' | xargs -r kill -9 2>/dev/null
  pgrep -f 'SkillOSGroup[W]orker' | xargs -r kill -9 2>/dev/null
  sleep 5
}

log "=================================================================="
log "supervisor start (target ${TOTAL_STEPS} steps, resume from $(latest_ckpt))"

stalled=0
attempt=0

while true; do
  before=$(latest_ckpt)

  if [ "$before" -ge "$TOTAL_STEPS" ]; then
    log "DONE ${TOTAL_STEPS}/${TOTAL_STEPS} reached — supervisor exiting"
    echo "VERL_RUN_COMPLETE step=$before" >> "$SUP_LOG"
    break
  fi

  attempt=$((attempt+1))
  kill_stragglers
  clean_partial_ckpts
  guard_tailscaled
  gate_network || { log "FATAL network gate failed"; echo "VERL_RUN_FATAL network" >> "$SUP_LOG"; break; }

  log "LAUNCH attempt=$attempt resume_from=$before ram_used=$(free -g | awk '/^Mem:/{print $3}')GB"
  ( cd "$VERL" && source .venv/bin/activate && bash examples/gigpo_trainer/run_skillos.sh ) \
      > "$TRAIN_LOG" 2>&1
  rc=$?

  after=$(latest_ckpt)
  log "EXIT rc=$rc ckpt ${before} -> ${after}"

  # classify the failure so the log says something actionable
  if grep -q "OutOfMemoryError\|low on memory" "$TRAIN_LOG" 2>/dev/null; then
    log "CAUSE node OOM (check tailscaled / other host processes)"
  elif grep -q "CUDA out of memory" "$TRAIN_LOG" 2>/dev/null; then
    log "CAUSE CUDA OOM (reduce micro batch)"
  elif grep -q "name resolution\|NameResolutionError" "$TRAIN_LOG" 2>/dev/null; then
    log "CAUSE DNS failure to api.inference.sh"
  fi

  if [ "$after" -ge "$TOTAL_STEPS" ]; then
    log "DONE reached ${after}/${TOTAL_STEPS}"
    echo "VERL_RUN_COMPLETE step=$after" >> "$SUP_LOG"
    break
  fi

  if [ "$after" -gt "$before" ]; then
    stalled=0
    log "PROGRESS advanced to ${after} — resuming"
  else
    stalled=$((stalled+1))
    log "STALLED no checkpoint progress (${stalled}/${MAX_STALLED_ATTEMPTS})"
    if [ "$stalled" -ge "$MAX_STALLED_ATTEMPTS" ]; then
      log "FATAL ${MAX_STALLED_ATTEMPTS} consecutive attempts with no progress — giving up"
      echo "VERL_RUN_FATAL stalled_at=${after}" >> "$SUP_LOG"
      break
    fi
  fi

  sleep 60
done

log "supervisor exit"
