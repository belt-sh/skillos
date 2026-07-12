#!/usr/bin/env bash
# Natural-distribution FFT checkpoint sweep: 12 arms (ckpt 5..60) in two
# GPU-pinned waves, then paired-by-gamefile McNemar vs the CANONICAL fixed
# baseline (eval-pathbv4/no_memory.jsonl, 33.6%) — never a reconstructed one.
# Mirrors the seed-1/seed-2 sweep settings for direct comparability.
set -u

OUT=output/alfworld-8xh100-algo1-fft-natural
EVAL=output/eval-fft-natural
BASE=output/eval-pathbv4/no_memory.jsonl
mkdir -p "$EVAL"
ln -sf ../eval-pathbv4/no_memory.jsonl "$EVAL/no_memory.jsonl"

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm () {  # $1=ckpt  $2=gpu
  local CK=$1 GPU=$2
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop --curator-checkpoint "$OUT/checkpoint-$CK" \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "$EVAL/ckpt$CK.jsonl" > "/tmp/natural_ckpt$CK.log" 2>&1 &
}

echo "[$(date -u)] WAVE 1: ckpt 5 10 15 20 25 30 35 40 on GPU 0-7"
i=0; for CK in 5 10 15 20 25 30 35 40; do run_arm "$CK" "$i"; i=$((i+1)); done
wait
echo "[$(date -u)] WAVE 1 done"

echo "[$(date -u)] WAVE 2: ckpt 45 50 55 60 on GPU 0-3"
i=0; for CK in 45 50 55 60; do run_arm "$CK" "$i"; i=$((i+1)); done
wait
echo "[$(date -u)] WAVE 2 done"

echo "[$(date -u)] COMPARATOR"
.venv/bin/python -m scripts.compare_eval_arms \
  --arm "no_memory=$BASE" \
  --arm "ckpt5=$EVAL/ckpt5.jsonl"   --arm "ckpt10=$EVAL/ckpt10.jsonl" \
  --arm "ckpt15=$EVAL/ckpt15.jsonl" --arm "ckpt20=$EVAL/ckpt20.jsonl" \
  --arm "ckpt25=$EVAL/ckpt25.jsonl" --arm "ckpt30=$EVAL/ckpt30.jsonl" \
  --arm "ckpt35=$EVAL/ckpt35.jsonl" --arm "ckpt40=$EVAL/ckpt40.jsonl" \
  --arm "ckpt45=$EVAL/ckpt45.jsonl" --arm "ckpt50=$EVAL/ckpt50.jsonl" \
  --arm "ckpt55=$EVAL/ckpt55.jsonl" --arm "ckpt60=$EVAL/ckpt60.jsonl" \
  | tee "$EVAL/comparison_canonical.txt"
echo "[$(date -u)] SWEEP COMPLETE"
