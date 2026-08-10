#!/usr/bin/env bash
# One-shot compact status for the verl-agent GiGPO SkillOS run.
# Safe to run any time; read-only. Used by the hourly heartbeat monitor and
# by hand:  bash scripts/verl_status.sh
set -u

# The real-ALFWORLD run (2026-07-30 onward) writes to *-real-checkpoints; the
# old scaffold run used *-checkpoints. Pointing at the stale dir reported
# ckpt=0/60 for two days while global_step_5/10 existed under the new one.
CKPT=/mnt/nvme/output/verl-skillos-real-checkpoints
LOGDIR=/home/ubuntu/skillos/logs
TRAIN_LOG="$LOGDIR/verl_skillos_gigpo_alfworld.log"
SUP_LOG="$LOGDIR/verl_supervisor.log"
PIDFILE="$LOGDIR/verl_supervisor.pid"
TOTAL=60
# verl's per-step metrics are block-buffered through Ray, so they reach the
# CONSOLE log late or not at all — but wandb's own output.log gets them. Read
# metrics from there. Liveness comes from the infsh task log (a line per
# executor call), because at ~3.7h/step console silence is normal, not a hang.
WB_LOG=$(ls -t /home/ubuntu/verl-skillos/wandb/run-*/files/output.log 2>/dev/null | head -1)
TASKLOG=/home/ubuntu/verl-skillos/output/infsh_tasks.jsonl
METRIC_SRC="${WB_LOG:-$TRAIN_LOG}"

# --- authoritative progress = last saved checkpoint ------------------------
ck=0
[ -f "$CKPT/latest_checkpointed_iteration.txt" ] && \
  ck=$(tr -dc '0-9' < "$CKPT/latest_checkpointed_iteration.txt")
ck=${ck:-0}

# --- in-flight step + pace from the tqdm bar -------------------------------
bar=$(grep -oE '[0-9]+/'"$TOTAL"' \[[^]]*\]' "$TRAIN_LOG" 2>/dev/null | tail -1)
step=$(echo "$bar" | grep -oE '^[0-9]+' || true)
sit=$(echo "$bar" | grep -oE '[0-9]+\.[0-9]+s/it' | tr -d 's/it' || true)
step=${step:-0}

# --- ETA from pace ---------------------------------------------------------
# `s/it` from tqdm is a CUMULATIVE AVERAGE over the whole run, not the current
# rate. One 10.9h throttled step early on still inflates it days later, so this
# ETA reads pessimistic. Keep it (it's the conservative bound) but also compute
# eta_recent from the last 5 per-step deltas of the elapsed clock.
eta="unknown"
if [ -n "${sit:-}" ] && [ "$step" -gt 0 ] && [ "$step" -lt "$TOTAL" ]; then
  rem=$(( TOTAL - step ))
  secs=$(printf '%.0f' "$(echo "$sit $rem" | awk '{print $1*$2}')")
  eta="$(( secs / 3600 ))h$(( (secs % 3600) / 60 ))m"
fi

# percent complete against the 60-step schedule
pct=$(awk -v s="$step" -v t="$TOTAL" 'BEGIN{printf "%.1f%%", 100*s/t}')

# recent pace: diff the elapsed [HH:MM:SS<...] field across the last 6 bars
eta_recent="unknown"; pace_recent="?"
if [ "$step" -gt 1 ] && [ "$step" -lt "$TOTAL" ]; then
  read -r pace_recent eta_recent <<<"$(
    grep -oE '[0-9]+/'"$TOTAL"' \[[0-9]+:[0-9]{2}:[0-9]{2}<' "$TRAIN_LOG" 2>/dev/null \
      | grep -oE '[0-9]+:[0-9]{2}:[0-9]{2}' | tail -6 \
      | awk -F: '{print $1*3600+$2*60+$3}' \
      | awk -v rem="$(( TOTAL - step ))" '
          NR>1 {d=$1-p; if (d>0) {s+=d; c++}} {p=$1}
          END{ if(c){ m=s/c; printf "%.0fs/it %dh%dm", m, int(m*rem/3600), int((m*rem)%3600/60) }
               else printf "? unknown" }'
  )"
fi

# --- is it learning? mean success_rate, early vs recent --------------------
mapfile -t sr < <(grep -oE 'episode/success_rate:[0-9.]+' "$METRIC_SRC" 2>/dev/null \
                   | grep -oE '[0-9.]+$')
n=${#sr[@]}
sr_recent="n/a"; sr_early="n/a"
# NOTE: `${arr[@]: -5}` on an array shorter than 5 uses an out-of-range
# negative offset, expands to nothing, and printf then emits ONE BLANK LINE.
# Without the NF guard awk scores that blank as a 0.0 sample and reports a
# fake 0.000 — which once looked exactly like a training collapse. Clamp the
# window to the array length AND ignore blank lines.
if [ "$n" -ge 1 ]; then
  k=5; [ "$n" -lt 5 ] && k="$n"
  sr_recent=$(printf '%s\n' "${sr[@]: -$k}" | awk 'NF{s+=$1;c++} END{if(c)printf "%.3f",s/c; else printf "n/a"}')
  sr_early=$(printf  '%s\n' "${sr[@]:0:$k}" | awk 'NF{s+=$1;c++} END{if(c)printf "%.3f",s/c; else printf "n/a"}')
fi
# metrics lag: verl's console metrics are block-buffered through Ray, so these
# trail the tqdm step counter. wandb is authoritative for the learning curve.
sr_n="$n"

# --- health ----------------------------------------------------------------
alive_sup="DOWN"; alive_train="DOWN"
[ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && alive_sup="up"
pgrep -f 'main[_]ppo' >/dev/null 2>&1 && alive_train="up"

ram=$(free -g | awk '/^Mem:/{print $3}')
ts_pid=$(pgrep -x tailscaled | head -1)
ts_rss="n/a"
if [ -n "${ts_pid:-}" ]; then
  ts_rss=$(awk '/^VmRSS:/{printf "%.1fGB", $2/1048576}' "/proc/$ts_pid/status" 2>/dev/null)
fi
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
       | awk '{s+=$1;c++} END{if(c)printf "%d%%",s/c; else print "n/a"}')
logage=$(( $(date +%s) - $(stat -c %Y "$TRAIN_LOG" 2>/dev/null || date +%s) ))
# real liveness: seconds since the last executor call was logged
callage=$(( $(date +%s) - $(stat -c %Y "$TASKLOG" 2>/dev/null || date +%s) ))
calls=$(wc -l < "$TASKLOG" 2>/dev/null || echo 0)
attempts=$(grep -c '^\[.*LAUNCH' "$SUP_LOG" 2>/dev/null || echo 0)
# --- reward-signal audit -------------------------------------------------
# r_task must be a real share of the composite. It was ~10% when 9 intermediate
# positions each emitted lambda_f*fc_valid (total ~10 vs composite max 2.15),
# which trained "emit valid tool calls" instead of "write transferable skills".
# Also track distinct rounds seen: if that stops growing, task draws are frozen
# (success froze at exactly 0.203 for 3 steps that way).
rp=$(grep -oE 'REWARD_PARTS .*' "$TRAIN_LOG" 2>/dev/null | tail -200)
rtask_share="n/a"; rounds="n/a"
if [ -n "$rp" ]; then
  rtask_share=$(printf '%s\n' "$rp" | grep -oE 'r_task=[0-9.]+ .*total=[0-9.]+' \
    | sed -E 's/r_task=([0-9.]+).*total=([0-9.]+)/\1 \2/' \
    | awk 'NF==2 && $2>0 {s+=$1/$2; c++} END{if(c) printf "%.0f%%", 100*s/c; else printf "n/a"}')
  rounds=$(printf '%s\n' "$rp" | grep -oE 'round=[0-9]+' | sort -u | wc -l | tr -d ' ')
fi
disk=$(df -h /mnt/nvme | awk 'NR==2{print $4}')

printf 'VERL_HB pct=%s eta_recent=%s pace_recent=%s ckpt=%s/%s step=%s eta=%s pace=%ss/it sup=%s train=%s gpu=%s calls=%s callage=%ss ram=%sGB ts=%s attempts=%s free=%s sr_recent=%s sr_early=%s sr_n=%s rtask_share=%s rounds=%s\n' \
  "$pct" "$eta_recent" "$pace_recent" \
  "$ck" "$TOTAL" "$step" "$eta" "${sit:-?}" "$alive_sup" "$alive_train" \
  "$gpu" "${calls:-0}" "${callage:-0}" "$ram" "$ts_rss" "$attempts" "$disk" \
  "$sr_recent" "$sr_early" "$sr_n" "$rtask_share" "$rounds"
