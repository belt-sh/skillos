#!/usr/bin/env bash
# Merge every verl FSDP checkpoint to HF format so the eval harness can load them.
set -u
# Paths are env-overridable. Defaults point at the REAL-ALFWorld run (finished
# 2026-08-09, 60/60 steps). The old simplified-scaffold run lives at
# verl-skillos-checkpoints -> verl-merged-hf and is kept for reference: do NOT
# reuse that OUT dir, or the "already merged, skip" branch silently passes the
# old models through and the sweep re-measures the old scaffold.
CK=${CK:-/mnt/nvme/output/verl-skillos-real-checkpoints}
OUT=${OUT:-/mnt/nvme/output/verl-merged-hf-real}
LOG=${LOG:-/home/ubuntu/skillos/logs/verl_merge_all_real.log}
mkdir -p "$OUT"
cd /home/ubuntu/verl-skillos && source .venv/bin/activate
for n in 5 10 15 20 25 30 35 40 45 50 55 60; do
  t="$OUT/step_$n"
  if [ -f "$t/config.json" ] && ls "$t"/model-*.safetensors >/dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] step_$n already merged, skip" >> "$LOG"; continue
  fi
  echo "[$(date -u +%H:%M:%S)] merging step_$n" >> "$LOG"
  python3 scripts/model_merger.py merge --backend fsdp \
    --local_dir "$CK/global_step_$n/actor" --target_dir "$t" >> "$LOG" 2>&1 \
    && echo "[$(date -u +%H:%M:%S)] step_$n OK $(du -sh "$t" | cut -f1)" >> "$LOG" \
    || echo "[$(date -u +%H:%M:%S)] step_$n FAILED" >> "$LOG"
done
echo "MERGE_ALL_DONE $(ls -d $OUT/step_* 2>/dev/null | wc -l)/12" >> "$LOG"
