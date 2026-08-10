#!/usr/bin/env bash
# One-shot host/reward health check for the verl run. Prints a line ONLY when
# something is wrong; silent when healthy. Driven by a Monitor loop.
#
# State (last API_ERROR count) is kept in a file, not a shell var, so the
# caller can be a plain `while` loop with no state of its own.
#
# NOTE on counting: `grep -c` prints "0" AND exits 1 when there are no matches.
# So `n=$(grep -c X f || echo 0)` yields "0\n0" and breaks $(( )) arithmetic —
# that bug killed the previous version of this monitor. Use `|| true` and
# normalize instead.
set -u

L=/home/ubuntu/skillos/logs/verl_skillos_gigpo_alfworld.log
STATE=/home/ubuntu/skillos/logs/.health_api_err_count

count_matches () {  # safe: always prints a single integer
  local pat="$1" file="$2" n
  n=$(grep -c "$pat" "$file" 2>/dev/null || true)
  n=${n:-0}
  n=${n//[^0-9]/}
  echo "${n:-0}"
}

# --- DNS: the failure that silently cost 4 training steps -----------------
if ! getent hosts api.inference.sh >/dev/null 2>&1; then
  echo "DNS_DOWN api.inference.sh will not resolve — executor calls failing NOW"
fi

# --- workers pinned to a stale resolver ----------------------------------
# glibc caches resolv.conf per process. If resolv.conf is NEWER than the
# running trainer, live workers may be querying a resolver that no longer
# answers (exactly what happened 2026-07-28: 361 API_ERROR, 4 dead steps).
wpid=$(pgrep -f 'main[_]ppo' | head -1 || true)
if [ -n "${wpid:-}" ] && [ -e /etc/resolv.conf ]; then
  w_start=$(stat -c %Y "/proc/$wpid" 2>/dev/null || echo 0)
  rc_mtime=$(stat -c %Y /etc/resolv.conf 2>/dev/null || echo 0)
  if [ "$rc_mtime" -gt "$w_start" ]; then
    echo "STALE_RESOLVER resolv.conf changed after trainer started — workers may hold a dead resolver; restart needed"
  fi
fi

if [ -f "$L" ]; then
  # --- API error RATE (not cumulative — cumulative is how 361 went unseen)
  cur=$(count_matches 'API_ERROR' "$L")
  prev=0
  [ -f "$STATE" ] && prev=$(tr -dc '0-9' < "$STATE" 2>/dev/null || echo 0)
  prev=${prev:-0}
  # log is truncated on each supervisor relaunch; a drop means reset
  if [ "$cur" -lt "$prev" ]; then prev=0; fi
  if [ $(( cur - prev )) -gt 50 ]; then
    echo "API_ERROR_STORM +$(( cur - prev )) new in last interval (total ${cur}) — reward signal at risk"
  fi
  echo "$cur" > "$STATE"

  # --- systemic-failure signature: consecutive EXACT zeros ---------------
  z=$(grep -oE 'episode/success_rate:[0-9.]+' "$L" 2>/dev/null | tail -3 \
        | grep -c ':0\.000$' 2>/dev/null || true)
  z=${z:-0}; z=${z//[^0-9]/}
  if [ "${z:-0}" -ge 3 ]; then
    echo "SUCCESS_RATE_ZERO last 3 steps all exactly 0.000 — systemic failure, not noise"
  fi

  # --- hang detection ----------------------------------------------------
  # Liveness must come from the infsh TASK log, not the console log. With real
  # ALFWorld episodes a step takes ~3.7h, so tqdm writes to the console once
  # per step and worker error lines are sporadic — an hour of console silence
  # is NORMAL and produced a false "possible hang" alert on 2026-07-29.
  # The task log gets a line per executor call (~74/min), so it is the real
  # heartbeat. Fall back to the console log only if the task log is absent.
  TASKLOG=/home/ubuntu/verl-skillos/output/infsh_tasks.jsonl
  if [ -n "${wpid:-}" ]; then
    probe="$TASKLOG"; label="executor calls"
    [ -f "$probe" ] || { probe="$L"; label="console log"; }
    age=$(( $(date +%s) - $(stat -c %Y "$probe" 2>/dev/null || date +%s) ))
    if [ "$age" -gt 1800 ]; then
      echo "VERL_STALLED no $label for $((age/60))min while trainer alive — likely a genuine hang"
    fi
  fi
fi

# --- memory --------------------------------------------------------------
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "${avail:-9999}" -lt 400 ]; then
  echo "RAM_PRESSURE only ${avail}GB available of 2015GB"
fi
exit 0
