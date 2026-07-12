#!/usr/bin/env bash
# Full 140-game no-memory eval with the sweep's best config (D):
# thinking ON, temp 0.6, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5.
# 8 shards (offsets 0..126), one vLLM server each (ports 8001-8008).
set -u
cd /home/ubuntu/skillos
PY=/home/ubuntu/skillos/.venv/bin/python
CKPT=/tmp/baseline-eval
mkdir -p output
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  port=$((8001 + i))
  off=$((i * 18))
  nohup "$PY" -m scripts.eval_alfworld_parallel \
    --checkpoint "$CKPT" --split valid_seen --num-games 18 --batch-size 18 \
    --max-steps 30 --executor vllm --model Qwen/Qwen3-8B \
    --base-url http://localhost:$port/v1 --game-offset $off \
    --enable-thinking --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 --max-tokens 8192 \
    --out output/eval-D-shard$i.jsonl \
    > output/eval-D-shard$i.log 2>&1 &
  PIDS+=($!)
  echo "launched D-shard$i pid=$! port=$port off=$off"
done
echo "PIDS=${PIDS[*]}"
