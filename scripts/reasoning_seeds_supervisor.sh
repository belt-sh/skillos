#!/usr/bin/env bash
# Train reasoning-curator seeds 2 and 3, once the GPUs are free.
#
# WHY. The only positive result in this project is a reasoning-trained curator
# lifting held-out ALFWorld by +9.0pp. It rests on ONE training run and ONE
# checkpoint, which is exactly the practice the rest of the paper criticises.
# Two more seeds is the minimum that makes the claim reportable; if the effect
# is a single-run accident, these will show it.
#
# Seeds 123 and 456, matching the ALFWorld FFT seed-2/seed-3 convention.
# Everything else is identical to the seed-42 run.
#
# SEQUENCING. Training needs all 8 GPUs, so this waits for every evaluation
# process to exit, not just wave C: wave C itself, the reasoning re-eval sweep,
# the valid_unseen content controls, and the queued replicate arms. Then it runs
# the two seeds back to back, ~3 days each.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

POLL="${SEEDS_POLL_S:-300}"

# Any of these holding a GPU means training cannot start.
BUSY_PATTERNS='eval_streaming_curation|eval_reasoning|reeval_all\.sh|reeval_reasoning\.sh|eval_content_controls\.sh|eval_replicates\.sh|train_reasoning|train_algo1'

wait_for_free_gpus () {
  local waited=0
  while true; do
    if ! pgrep -af "$BUSY_PATTERNS" | grep -qv "reasoning_seeds_supervisor"; then
      # Belt and braces: also require the cards themselves to be near-idle, in
      # case something is holding memory that pgrep does not match.
      local used
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
             awk '{s+=$1} END {print s+0}')
      if [ "$used" -lt 8000 ]; then
        echo "[$(date -u)] GPUs free (${used} MiB in use) after ${waited}s"
        return 0
      fi
      echo "[$(date -u)] no eval processes but ${used} MiB still allocated; waiting"
    fi
    sleep "$POLL"; waited=$((waited + POLL))
  done
}

run_seed () {   # $1 = seed number (2 or 3)
  local n=$1
  local out="output/reasoning-8xh100-algo1-fft-seed$n"
  local cfg="configs/reasoning_8xh100_algo1_fft_seed$n.yaml"

  if [ -d "$out/checkpoint-60" ]; then
    echo "[$(date -u)] seed$n already complete, skipping"; return 0
  fi

  # Resume from the newest checkpoint if a prior attempt died partway.
  local resume=""
  local latest
  latest=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  [ -n "$latest" ] && resume="$latest" && echo "[$(date -u)] resuming seed$n from $latest"

  echo "[$(date -u)] === starting reasoning seed$n ($cfg) ==="
  source .venv/bin/activate
  export ALFWORLD_DATA="$HOME/.cache/alfworld"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
  export WANDB_PROJECT=skillos WANDB_ENTITY=okaris
  export WANDB_RUN_ID="reasoningfftseed$n" WANDB_RESUME=allow
  [ -n "$resume" ] && export SKILLOS_RESUME_FROM_CHECKPOINT="$resume"

  export SKILLOS_PARALLEL_ROLLOUTS=256
  export SKILLOS_PARALLEL_JUDGES=24
  export SKILLOS_EXECUTOR_MAX_STEPS=25
  export SKILLOS_EXECUTOR_TIMEOUT_S=900
  export SKILLOS_PHASE_BUDGET_S="${SKILLOS_PHASE_BUDGET_S:-3600}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ENABLE_MONITORING=0

  local log="logs/reasoning_seed${n}_$(date +%Y%m%d_%H%M%S).log"
  accelerate launch --config_file configs/accelerate_zero3.yaml \
    -m scripts.train_reasoning --config "$cfg" > "$log" 2>&1
  local rc=$?
  echo "[$(date -u)] seed$n exited rc=$rc, log=$log"
  unset SKILLOS_RESUME_FROM_CHECKPOINT
  return $rc
}

echo "[$(date -u)] reasoning-seeds supervisor armed; waiting for GPUs"
wait_for_free_gpus

for n in 2 3; do
  # Up to three attempts per seed: these runs have historically died to NCCL
  # timeouts and OOM, and a resume from the last checkpoint is cheap.
  for attempt in 1 2 3; do
    if run_seed "$n"; then break; fi
    echo "[$(date -u)] seed$n attempt $attempt failed; re-checking GPUs before retry"
    sleep 120
    wait_for_free_gpus
  done
done

echo "[$(date -u)] === both reasoning seeds done ==="
ls -d output/reasoning-8xh100-algo1-fft-seed{2,3}/checkpoint-* 2>/dev/null | tail -4
echo "NEXT: sweep both seeds on valid_unseen against a contemporaneous baseline"
echo "      (paper section 5.3 is blocked on this)"
