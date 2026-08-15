#!/usr/bin/env bash
# Two controls the curation literature generally omits, both on the held-out
# valid_unseen split against the same contemporaneous baseline.
#
# SHUFFLED: the trained curator's own repo, but retrieval returns a random 5
# skills instead of the BM25 top 5. Same curator, same repo growth, same number
# of skills in the prompt, same token budget. Only relevance is destroyed. If
# the +9.0pp lift survives this, the executor is being helped by extra markdown
# rather than by skills about its task, and the result means much less.
#
# ORACLE: eight hand-written skills, no curator at all. Bounds what any curator
# could be worth on this executor. If human notes also fail, the nulls are a
# property of the executor, not of GRPO.
set -uo pipefail
cd "$(dirname "$0")/.."
export SKILLOS_EXECUTOR_MAX_STEPS=30 SKILLOS_EXEC_MAX_RESUBS=6 SKILLOS_EXEC_POLL_MAX_S=300
export SKILLOS_EXEC_MAX_STREAM_RECONNECTS=2 SKILLOS_EXEC_BACKOFF_BASE_S=20 SKILLOS_EXEC_BACKOFF_CAP_S=180
export SKILLOS_EVAL_MAX_ERROR_RATE=0.02 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=output/reeval/unseen-power
REAS=output/reasoning-8xh100-algo1-fft

# Oracle needs no GPU and no curator.
if [ ! -s "$OUT/oracle_handwritten.jsonl" ]; then
  .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode no_memory --static-repo assets/oracle_skills \
    --num-games 134 --batch-size 15 --split valid_unseen \
    --out "$OUT/oracle_handwritten.jsonl" \
    > logs/reeval/unseen_oracle.log 2>&1 &
fi

# Shuffled retrieval, same curator checkpoint as the positive result.
if [ ! -s "$OUT/r2a_ckpt50_shuffled.jsonl" ]; then
  SKILLOS_RETRIEVAL_MODE=shuffled CUDA_VISIBLE_DEVICES=6 \
  .venv/bin/python -u -m scripts.eval_streaming_curation \
    --mode closed_loop --curator-checkpoint "$REAS/checkpoint-50" --curator-device cuda \
    --curator-temperature 0 --curator-max-new-tokens 1536 \
    --num-games 134 --batch-size 15 --split valid_unseen \
    --out "$OUT/r2a_ckpt50_shuffled.jsonl" \
    > logs/reeval/unseen_shuffled.log 2>&1 &
fi
wait
echo "[$(date -u)] content controls done"
grep -H "data integrity" logs/reeval/unseen_oracle.log logs/reeval/unseen_shuffled.log
