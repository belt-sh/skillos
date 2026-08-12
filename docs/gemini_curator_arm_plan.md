# Plan: Gemini-2.5-Pro-as-curator arm (tests the paper's headline economic claim)

Status: **in progress, started 2026-08-12.** Written before a context compaction so
the work can be resumed without re-deriving anything.

## Why

The paper's headline is that an RL-trained 8B curator **beats Gemini-2.5-Pro used
directly as the curator**, "especially for weaker executors". That is the claim
that makes SkillOS economically interesting, and it is the one arm the
reproduction never ran. `docs/x_article_draft.md` currently repeats it as the
paper's claim with no test behind it.

Three outcomes, all publishable:

- Gemini below our trained curator → first independent confirmation of the part
  everyone quotes.
- Gemini above → the central economic claim does not hold. Bigger finding than
  anything currently in the article.
- Both inside the ±3pp noise band → the curation task is not discriminative at
  n=140, which reframes the paper.

## What to build

`scripts/eval_streaming_curation.py` only supports a **local** curator
(`CuratorInference`, line ~180: `AutoModelForCausalLM` + `device_map=cuda`).
Add a sibling `RemoteCuratorInference` with the same `.curate(repo, traj_result)`
signature, so the eval loop at line ~406 (`curator.curate(...)`) is unchanged.

Fairness requirement: **identical prompts and tool palette.** Reuse, do not
re-author:

- `TOOLS_SCHEMA` (defined in `eval_streaming_curation.py`, ~line 57)
- `CURATOR_SYSTEM`, `CURATOR_INPUT_TEMPLATE` from `skillos.curator.prompts`
- `format_trajectory` from `skillos.curator.prompts`
- `apply_curation_ops`, `CurationOp` from `skillos.curator.model`
- top-5 BM25 retrieval via `repo.retrieve(task, top_k=5)`

Only the generation call differs.

### The app

`google/gemini-2-5-pro@7bzy1nx4` supports real function calling, so no
prompt-format hack is needed.

- input: `text`, `system_prompt`, `tools`, `temperature`, `max_tokens`,
  `reasoning_effort`
- output: `response`, `reasoning`, **`tool_calls`**, `usage`

Map `TOOLS_SCHEMA` into the `tools` field and read `tool_calls` back. Fall back to
parsing `response` for `<tool_call>` JSON blocks if `tool_calls` comes back empty,
and count how often that happens (it is a fairness caveat worth reporting).

New CLI flags to add:

    --curator-backend {local,remote}    default local (preserves every existing call site)
    --curator-app google/gemini-2-5-pro
    --curator-reasoning-effort medium

Auth: `skillos.utils.infsh_auth.resolve_infsh_api_key` (same helper the executor
and judge already use). Never put the key in env or a tracked file.

## How to run

Baselines already exist. Do NOT re-run them:

- 8B no-memory: `output/eval-pathbv4/no_memory.jsonl` = 47/140 = 33.6%
- 32B no-memory: `output/eval-transfer-32b/no_memory.jsonl` = 69/140 = 49.3%

Smoke test first (5 games, confirms function calling + real cost per call):

    export SKILLOS_EXECUTOR_MAX_STEPS=30 SKILLOS_EXEC_MAX_RESUBS=2 SKILLOS_EXEC_POLL_MAX_S=150
    .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-backend remote --curator-app google/gemini-2-5-pro \
      --num-games 5 --batch-size 5 --split valid_seen \
      --out output/eval-gemini-curator/smoke.jsonl

Arm 1, 8B executor (~3.5h, ~$8):

    .venv/bin/python -u -m scripts.eval_streaming_curation \
      --mode closed_loop --curator-backend remote --curator-app google/gemini-2-5-pro \
      --num-games 140 --batch-size 20 --split valid_seen \
      --curator-temperature 0 \
      --out output/eval-gemini-curator/gemini_8b.jsonl

Arm 2, 32B executor (~3.5h, ~$10). Same, plus:

      --executor-app openrouter/qwen3-32b \
      --out output/eval-gemini-curator/gemini_32b.jsonl

Compare (this is the deliverable):

    .venv/bin/python -m scripts.compare_eval_arms \
      --arm no_memory=output/eval-pathbv4/no_memory.jsonl \
      --arm gemini_curator=output/eval-gemini-curator/gemini_8b.jsonl \
      --arm trained_verl_ckpt30=output/eval-verl-gigpo-real/ckpt30.jsonl \
      --arm trained_trl_seed2_ckpt35=output/eval-fft-seed2/ckpt35.jsonl \
      | tee output/eval-gemini-curator/comparison_8b.txt

    .venv/bin/python -m scripts.compare_eval_arms \
      --arm no_memory=output/eval-transfer-32b/no_memory.jsonl \
      --arm gemini_curator=output/eval-gemini-curator/gemini_32b.jsonl \
      --arm trained_seed3_ckpt5=output/eval-transfer-32b-seed3/ckpt5.jsonl \
      | tee output/eval-gemini-curator/comparison_32b.txt

## Measured facts behind the estimates

- One 140-game arm took **3 to 3.5h** in the real sweep (`logs/verl_sweep_real.log`:
  8 arms in parallel 20:40 → 00:04).
- **The executor dominates wall time, not the curator.** Per 20-game wave:
  `exec=1870s` vs `curator=91s`. So a remote curator barely changes the clock,
  and with both remote the arm needs **no GPU at all** — several arms can run
  concurrently.
- Executor cost is **$0.0002 per call, measured** via `belt task cost` on three
  real sweep task ids. ~3,200 calls per arm ⇒ **$0.73**.
- Gemini curator: 140 calls/arm, ~8k input (matches the 6.5k curator prompt
  measured in training) and an unknown number of *thinking* tokens at $10/M out.
  Estimated **$7/arm**, uncertain by ~2×. The smoke test pins this down.
- Total for both arms: **~$18, call it $25 with retries.**

## Reporting

On completion, update: `docs/repro_report.md` (new finding + table),
`docs/x_article_draft.md` (the TL;DR currently states the Gemini claim untested),
`README.md`, `JOURNAL.md` (dated entry), and `DIVERGENCES.md` if the Gemini arm
required any deviation. Publish the rollouts to the existing
`inference-sh/skillos-alfworld-eval-arms` dataset via
`scripts/hf_publish_artifacts.sh` (add `eval-gemini-curator` to `EVAL_DIRS`).

Multiplicity note: this adds arms to a family that already has 50. Report the
Gemini comparison as a **pre-registered single hypothesis** (the paper's own
claim), not as another sweep arm, and say so explicitly.
