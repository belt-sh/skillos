#!/usr/bin/env bash
# Keep the full-fidelity run alive for ~10 days without a human watching it.
#
# Why a supervisor rather than the NCCL watchdog: with TORCH_NCCL_ENABLE_MONITORING=0
# a genuine hang does not abort, it stalls, silently burning GPU hours. And with a
# 10h collective timeout, waiting for NCCL to notice is far too slow. So we watch
# the only thing that actually proves progress -- the log growing -- and restart
# from the last checkpoint when it stops.
#
# STALL_S must exceed one legitimate step (~4h) with margin, or we will kill a
# healthy run mid-step. 6h.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=output/alfworld-dense-fft-seed2r
STALL_S="${DENSE_STALL_S:-21600}"     # 6h with no log growth = stalled
POLL_S="${DENSE_POLL_S:-600}"
MAX_RESTARTS="${DENSE_MAX_RESTARTS:-12}"

latest_ckpt () {
  ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1
}

for attempt in $(seq 1 "$MAX_RESTARTS"); do
  ck=$(latest_ckpt)
  if [ -n "$ck" ] && [ "$ck" -ge 60 ]; then
    echo "[$(date -u)] checkpoint-60 exists — run complete"; break
  fi
  resume=""
  [ -n "$ck" ] && resume="$OUT/checkpoint-$ck"
  echo "[$(date -u)] attempt $attempt/$MAX_RESTARTS, resume=${resume:-<fresh>}"

  ./run_algo1_dense.sh $resume &
  train_pid=$!
  sleep 120
  LOG=$(ls -t logs/algo1_dense_*.log 2>/dev/null | head -1)
  echo "[$(date -u)] watching $LOG (pid $train_pid, stall threshold ${STALL_S}s)"

  while kill -0 "$train_pid" 2>/dev/null; do
    sleep "$POLL_S"
    if [ -f "$LOG" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
      if [ "$age" -gt "$STALL_S" ]; then
        echo "[$(date -u)] STALLED: no log growth for ${age}s — killing to resume"
        pkill -f "train_algo1 --config configs/alfworld_dense_fft.yaml" || true
        sleep 20
        pkill -9 -f "train_algo1 --config configs/alfworld_dense_fft.yaml" || true
        break
      fi
    fi
  done
  wait "$train_pid" 2>/dev/null || true

  # Report fidelity every attempt: this run exists to make this number ~9.
  if [ -f "$LOG" ]; then
    echo "[$(date -u)] fidelity so far:"
    grep "reward health" "$LOG" | tail -3 || echo "  no health lines yet"
  fi
  sleep 60
done

echo "[$(date -u)] === dense run supervisor done ==="
ls -d "$OUT"/checkpoint-* 2>/dev/null | tail -3
