#!/usr/bin/env bash
# 2x2 Qwen3-8B config sweep for the no-memory ALFWorld baseline.
# (thinking vs non-thinking) x (official sampling vs +presence_penalty anti-loop).
# Each config runs the SAME 70 games (offsets 0 + 35) on its own 2 vLLM servers
# so results are directly comparable. Reference A (no top_k) = prior 140-game run.
set -u
cd /home/ubuntu/skillos
PY=/home/ubuntu/skillos/.venv/bin/python
CKPT=/tmp/baseline-eval
COMMON="--checkpoint $CKPT --split valid_seen --num-games 35 --batch-size 35 \
  --max-steps 30 --executor vllm --model Qwen/Qwen3-8B --max-tokens 8192 \
  --top-k 20 --min-p 0"
mkdir -p output

launch () { # name port offset thinkflag extra
  local name=$1 port=$2 off=$3 think=$4; shift 4
  nohup "$PY" -m scripts.eval_alfworld_parallel $COMMON \
    --base-url http://localhost:$port/v1 --game-offset $off $think "$@" \
    --out output/sweep-$name.jsonl \
    > output/sweep-$name.log 2>&1 &
  echo "launched $name pid=$! port=$port off=$off think='$think' extra='$*'"
}

# B: thinking + official (temp .6 / top_p .95)
launch B-sh0 8001 0  --enable-thinking --temperature 0.6 --top-p 0.95
launch B-sh1 8002 35 --enable-thinking --temperature 0.6 --top-p 0.95
# C: non-thinking + official (temp .7 / top_p .8)
launch C-sh0 8003 0  --no-thinking --temperature 0.7 --top-p 0.8
launch C-sh1 8004 35 --no-thinking --temperature 0.7 --top-p 0.8
# D: thinking + anti-loop (presence_penalty 1.5)
launch D-sh0 8005 0  --enable-thinking --temperature 0.6 --top-p 0.95 --presence-penalty 1.5
launch D-sh1 8006 35 --enable-thinking --temperature 0.6 --top-p 0.95 --presence-penalty 1.5
# E: non-thinking + anti-loop
launch E-sh0 8007 0  --no-thinking --temperature 0.7 --top-p 0.8 --presence-penalty 1.5
launch E-sh1 8008 35 --no-thinking --temperature 0.7 --top-p 0.8 --presence-penalty 1.5

echo "all 8 launched"
