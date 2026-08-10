# Datasets

No dataset files are committed to this repo. This file explains how to obtain
each one, and which of them carry redistribution conditions.

`data/` is otherwise empty by design — see the root `.gitignore`.

## ALFWorld (agentic benchmark)

Installed as a package dependency; the game files download on first use.

```bash
uv pip install -e ".[alfworld]"
alfworld-download           # ~2GB into $ALFWORLD_DATA (default ~/.cache/alfworld)
```

Splits used here: `train` for GRPO rollouts, `valid_seen` for every held-out
number in the report (140 games, fixed and paired by gamefile across all arms).
No redistribution restriction — MIT licensed.

## DeepMath-103K (reasoning training data)

Public on the Hub, no gating:

```bash
# pulled automatically by skillos/reasoning/train_data.py on first run
python -c "from datasets import load_dataset; load_dataset('zwhe99/DeepMath-103K')"
```

## AIME24 / AIME25 (reasoning eval)

Public on the Hub, no gating. Loaded by `skillos/reasoning/datasets.py`.
30 problems each.

## GPQA-Diamond (reasoning eval) — GATED, with conditions

`Idavidrein/gpqa` is a **gated** dataset. To reproduce the GPQA column you must
request access yourself; we cannot redistribute it, in whole or in part.

```bash
hf auth login            # or: export HF_TOKEN=...
# then request access at https://huggingface.co/datasets/Idavidrein/gpqa
```

**The access condition matters and it binds anyone reproducing this work.** The
dataset is deliberately kept out of web-crawlable text so it does not leak into
future pretraining corpora. Access is granted on the condition that you do not
publish problem text, answer options, correct answers, or model responses to
those problems in any public-visible artifact — that includes git history, pull
requests, issues, docs, blog posts, and screenshots.

Practical consequences for this repo, all of which we follow:

- **Only aggregate accuracies** appear in `README.md`, `JOURNAL.md`,
  `DIVERGENCES.md`, and `docs/repro_report.md` (e.g. "118/198 = 59.6%"). Never a
  per-problem breakdown.
- **Per-problem eval outputs stay local.** They are written under `output/`,
  which is gitignored, and `.gitignore` additionally hard-blocks any path
  matching `*gpqa*` as a second line of defence against an `git add -f` slip.
- **Published eval artifacts exclude GPQA.** When the 140-game paired ALFWorld
  JSONLs are released so the McNemar tests can be recomputed, no GPQA rollout is
  included. The GPQA number is reproducible only by someone who has been granted
  their own access.
- `skillos/reasoning/prompts.py` contains prompt *templates* with a `{problem}`
  placeholder — no problem content is embedded in source.

If you are reviewing this repo for a leak, the check is:

```bash
git grep -il gpqa            # should return only code, config and aggregate-number docs
git log -p -- '*gpqa*'       # should return nothing
```

## WebShop

Not implemented. The paper's third benchmark; we skipped it deliberately (see
`DIVERGENCES.md` #8). Nothing to download.
