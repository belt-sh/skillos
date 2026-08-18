#!/usr/bin/env bash
# Preserve the 16-bit weights of EVERY training step for the checkpoint sweep.
#
# WHY THIS EXISTS
# ---------------
# A ZeRO-3 checkpoint here is 107 GB: 92 GB of sharded fp32 optimizer state in
# global_step*/, plus the 16 GB consolidated model.safetensors that
# zero3_save_16bit_model produces. With save_steps=1 and save_total_limit=12,
# HF Trainer rotates whole checkpoint dirs, so by step 60 only steps 49-60 would
# still exist.
#
# That would have destroyed the run's purpose. The question this run answers is
# the SHAPE of held-out accuracy across training (monotone as the paper reports,
# or the bimodal oscillation we keep measuring). A rolling window of the last 12
# steps cannot show a shape. And nothing would have errored — the run would have
# reported success at step 60 with the early checkpoints quietly gone. Same class
# as every other bug in this project: the failure does not announce itself.
#
# Eval only ever loads model.safetensors + config + tokenizer. The 92 GB of Adam
# state is resume-only, and resume never needs to reach further back than the
# last completed step. So archive the cheap part for all 60 steps (~960 GB of
# 20 TB free on /mnt/nvme) and let Trainer rotate the expensive part.
#
# Deliberately an EXTERNAL daemon, not a config change: raising save_total_limit
# to 60 would hold 6.4 TB of optimizer state we will never read, and any config
# change costs a restart, which costs the ~1.9h step in flight.
set -uo pipefail

OUT="${1:-output/alfworld-dense-fft-seed1}"
DEST="$OUT/eval_weights"
POLL="${ARCHIVE_POLL_S:-120}"
FINAL_STEP="${ARCHIVE_FINAL_STEP:-60}"

# config/tokenizer are small; model.safetensors is the 16 GB payload.
FILES=(model.safetensors config.json generation_config.json
       tokenizer.json tokenizer_config.json chat_template.jinja
       trainer_state.json)

mkdir -p "$DEST"
echo "[$(date -u)] archiver watching $OUT -> $DEST (poll ${POLL}s, until step $FINAL_STEP)"

# A checkpoint dir appears before it is fully written. Trust it only when
# trainer_state.json parses AND its global_step matches the directory number —
# that file is written last, after the weights.
complete() {
  local ck="$1" n="$2"
  [ -s "$ck/model.safetensors" ] || return 1
  python3 - "$ck/trainer_state.json" "$n" <<'PY' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1]) as f:
        s = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if int(s.get("global_step", -1)) == int(sys.argv[2]) else 1)
PY
}

while true; do
  for ck in $(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    n="${ck##*checkpoint-}"
    [ -f "$DEST/step-$n/.complete" ] && continue
    complete "$ck" "$n" || continue

    tmp="$DEST/.step-$n.partial"
    rm -rf "$tmp"; mkdir -p "$tmp"
    ok=1
    for f in "${FILES[@]}"; do
      [ -e "$ck/$f" ] || continue                 # optional files may be absent
      cp -f "$ck/$f" "$tmp/$f" || { ok=0; break; }
    done
    # Refuse to publish weights we cannot verify byte-for-byte.
    if [ "$ok" = 1 ] && [ -s "$tmp/model.safetensors" ] \
       && [ "$(stat -c%s "$ck/model.safetensors")" = "$(stat -c%s "$tmp/model.safetensors")" ]; then
      touch "$tmp/.complete"
      rm -rf "$DEST/step-$n"
      mv "$tmp" "$DEST/step-$n"
      echo "[$(date -u)] archived step $n ($(du -sh "$DEST/step-$n" | cut -f1))"
    else
      echo "[$(date -u)] step $n copy incomplete or size mismatch — will retry" >&2
      rm -rf "$tmp"
    fi
  done

  if [ -f "$DEST/step-$FINAL_STEP/.complete" ]; then
    echo "[$(date -u)] step $FINAL_STEP archived — run complete, archiver exiting"
    ls -1d "$DEST"/step-* | wc -l | xargs echo "  steps preserved:"
    break
  fi
  sleep "$POLL"
done
