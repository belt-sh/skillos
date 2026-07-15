#!/usr/bin/env bash
# Post-seed-3 experiment queue. Runs three experiments serially/parallel where
# GPU math works, all against known-healthy inference.sh executors.
#
# Wave 1 (parallel, 8 GPUs):
#   - Cross-domain reasoning closed_loop × 3 curators   (GPU 0-2)
#   - 32B transfer sweep on seed-3, 5 arms              (GPU 3-7)
# Wave 2 (parallel, 4 GPUs used):
#   - 32B transfer sweep on seed-3, 4 remaining arms    (GPU 0-3)
# Then: baseline variance study (n=5, no local GPUs needed)
#
# Everything logs to logs/, JSONLs to output/. Watchers must poll durable
# output artifacts, not /tmp.
set -u
cd "$(dirname "$0")/.."

export SKILLOS_EXECUTOR_MAX_STEPS=30
export SKILLOS_EXEC_MAX_RESUBS=2
export SKILLOS_EXEC_POLL_MAX_S=150
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Wave 1 arms ----------------------------------------------------------

# 1a. Cross-domain reasoning closed_loop for the three best curators.
#     Serial-per-arm inside, but different arms run in parallel on different GPUs.
mkdir -p output/eval-reasoning-transfer
declare -A CURATORS=(
  [v8lora]=output/alfworld-8xh100-algo1-v8-lora-kl/checkpoint-30
  [fftS2]=output/alfworld-8xh100-algo1-fft-seed2/checkpoint-35
  [fftS3]=output/alfworld-8xh100-algo1-fft-seed3/checkpoint-55
)
gpu=0
declare -a WAVE1_PIDS=()
for name in v8lora fftS2 fftS3; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_reasoning \
    --mode closed_loop --curator-checkpoint "${CURATORS[$name]}" \
    --dataset aime --parallel 1 \
    --out "output/eval-reasoning-transfer/${name}.jsonl" \
    > "logs/eval_reasoning_transfer_${name}.log" 2>&1 &
  WAVE1_PIDS+=($!)
  echo "[$(date -u)] LAUNCH reasoning-cl name=$name gpu=$gpu pid=$!"
  gpu=$((gpu+1))
done

# 1b. 32B transfer sweep on seed-3 ckpts — wave 1: 5 arms on GPU 3-7.
mkdir -p output/eval-transfer-32b-seed3
ln -sf ../eval-transfer-32b/no_memory.jsonl output/eval-transfer-32b-seed3/no_memory.jsonl
declare -a SEED3_W1=(15 25 35 45 55)
for CK in "${SEED3_W1[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop \
    --curator-checkpoint "output/alfworld-8xh100-algo1-fft-seed3/checkpoint-$CK" \
    --executor infsh --executor-app openrouter/qwen3-32b \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "output/eval-transfer-32b-seed3/ckpt$CK.jsonl" \
    > "logs/eval_transfer_32b_seed3_ckpt${CK}.log" 2>&1 &
  WAVE1_PIDS+=($!)
  echo "[$(date -u)] LAUNCH transfer32b seed3 ckpt=$CK gpu=$gpu pid=$!"
  gpu=$((gpu+1))
done

echo "[$(date -u)] wave 1 running — PIDs: ${WAVE1_PIDS[*]}"
wait
echo "[$(date -u)] WAVE 1 COMPLETE"

# ---- Wave 2 --------------------------------------------------------------

# 32B transfer sweep on seed-3 remaining ckpts.
declare -a SEED3_W2=(5 10 20 30 40 50 60)  # remaining, 7 arms
# NB: seed-3 has ckpts 5..60 step 5 = 12 total; wave 1 covered 5 (15/25/35/45/55).
# Wave 2 has 7 arms; run 4 at a time so no single wave saturates the 32B pool.
mkdir -p logs
gpu=0
declare -a WAVE2A_PIDS=()
for CK in 5 10 20 30; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop \
    --curator-checkpoint "output/alfworld-8xh100-algo1-fft-seed3/checkpoint-$CK" \
    --executor infsh --executor-app openrouter/qwen3-32b \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "output/eval-transfer-32b-seed3/ckpt$CK.jsonl" \
    > "logs/eval_transfer_32b_seed3_ckpt${CK}.log" 2>&1 &
  WAVE2A_PIDS+=($!); gpu=$((gpu+1))
done
wait
echo "[$(date -u)] WAVE 2A COMPLETE"
gpu=0
declare -a WAVE2B_PIDS=()
for CK in 40 50 60; do
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop \
    --curator-checkpoint "output/alfworld-8xh100-algo1-fft-seed3/checkpoint-$CK" \
    --executor infsh --executor-app openrouter/qwen3-32b \
    --num-games 140 --batch-size 20 --split valid_seen \
    --curator-device cuda --curator-temperature 0 --curator-max-new-tokens 1536 \
    --out "output/eval-transfer-32b-seed3/ckpt$CK.jsonl" \
    > "logs/eval_transfer_32b_seed3_ckpt${CK}.log" 2>&1 &
  WAVE2B_PIDS+=($!); gpu=$((gpu+1))
done
wait
echo "[$(date -u)] WAVE 2 COMPLETE"

# ---- Final: comparators for both experiments -----------------------------

# Reasoning transfer summary (n=90 total: aime24+25 no_memory result reused).
.venv/bin/python - <<'PY' | tee output/eval-reasoning-transfer/summary.txt
import json, glob
paper_nomem = {"AIME24":76.0,"AIME25":71.1}
for f in sorted(glob.glob('output/eval-reasoning-transfer/*.jsonl')):
    rows=[json.loads(l) for l in open(f)]
    by={}
    for r in rows:
        ds=r['id'].split('-')[0]; by.setdefault(ds,[]).append(r['correct'])
    line=f.split('/')[-1]+": "
    for ds in sorted(by):
        ok=sum(by[ds]); n=len(by[ds]); line += f"{ds} {ok}/{n}={100*ok/n:.1f}%  "
    ok=sum(r['correct'] for r in rows); n=len(rows)
    print(line + f"OVERALL {ok}/{n}={100*ok/n:.1f}%")
PY

# 32B transfer sweep comparator vs canonical 49.3% no_memory baseline
.venv/bin/python -m scripts.compare_eval_arms \
  --arm no_memory=output/eval-transfer-32b-seed3/no_memory.jsonl \
  $(for CK in 5 10 15 20 25 30 35 40 45 50 55 60; do
      echo "--arm ckpt$CK=output/eval-transfer-32b-seed3/ckpt$CK.jsonl"
    done) \
  | tee output/eval-transfer-32b-seed3/comparison.txt

echo "[$(date -u)] QUEUE COMPLETE"
