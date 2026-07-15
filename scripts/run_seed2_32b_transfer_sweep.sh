#!/usr/bin/env bash
# Seed-2 32B transfer sweep — mirrors the seed-3 sweep to check whether the
# 8B/32B decorrelation replicates on a second seed.
# 12 arms (ckpt 5..60) driving qwen3-32b executor, paired McNemar vs the
# canonical 49.3% 32B no-memory baseline (reused from eval-transfer-32b/).
set -u
cd "$(dirname "$0")/.."

EVAL=output/eval-transfer-32b-seed2
mkdir -p "$EVAL"
ln -sf ../eval-transfer-32b/no_memory.jsonl "$EVAL/no_memory.jsonl"

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -u)] WAVE 1: ckpt 5 10 15 20 25 30 35 40 on GPU 0-7"
gpu=0
declare -a W1=()
for CK in 5 10 15 20 25 30 35 40; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop \
    --curator-checkpoint "output/alfworld-8xh100-algo1-fft-seed2/checkpoint-$CK" \
    --executor infsh --executor-app openrouter/qwen3-32b \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "$EVAL/ckpt$CK.jsonl" \
    > "logs/eval_transfer_32b_seed2_ckpt${CK}.log" 2>&1 &
  W1+=($!); gpu=$((gpu+1))
done
wait
echo "[$(date -u)] WAVE 1 done"

echo "[$(date -u)] WAVE 2: ckpt 45 50 55 60 on GPU 0-3"
gpu=0
for CK in 45 50 55 60; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop \
    --curator-checkpoint "output/alfworld-8xh100-algo1-fft-seed2/checkpoint-$CK" \
    --executor infsh --executor-app openrouter/qwen3-32b \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "$EVAL/ckpt$CK.jsonl" \
    > "logs/eval_transfer_32b_seed2_ckpt${CK}.log" 2>&1 &
  gpu=$((gpu+1))
done
wait
echo "[$(date -u)] WAVE 2 done"

.venv/bin/python -m scripts.compare_eval_arms \
  --arm no_memory=$EVAL/no_memory.jsonl \
  $(for CK in 5 10 15 20 25 30 35 40 45 50 55 60; do echo "--arm ckpt$CK=$EVAL/ckpt$CK.jsonl"; done) \
  | tee "$EVAL/comparison.txt"
echo "[$(date -u)] SWEEP COMPLETE"
