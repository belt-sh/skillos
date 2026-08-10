#!/usr/bin/env bash
# Publish the reproduction's artifacts to the Hugging Face Hub.
#
#   bash scripts/hf_publish_artifacts.sh            # stage + upload everything
#   STAGE_ONLY=1 bash scripts/hf_publish_artifacts.sh   # build staging dirs, don't upload
#   ONLY=eval bash scripts/hf_publish_artifacts.sh      # one target: verl|trl|eval
#
# Idempotent and resumable: staging uses hardlinks (same filesystem, no extra
# disk), and `hf upload` skips blobs already present on the Hub, so re-running
# after an interruption continues rather than restarting.
#
# Requires `hf auth login` first (check with `hf auth whoami`).
#
# WHAT IS PUBLISHED, AND WHY THIS SUBSET
# --------------------------------------
# * verl/GiGPO: ALL 12 checkpoints. This is the run docs/repro_report.md leads
#   with, and its central claim is the SHAPE of the checkpoint curve — which is
#   only independently re-testable if every arm is available.
# * TRL FFT: peak + final for each of the 3 seeds, plus seed-3 step 5 (the best
#   32B-transfer curator in the project). NOT all 36; the omitted arms are
#   listed in the model card so a subset can't be mistaken for a full sweep.
# * Eval JSONLs: every ALFWorld arm, so every McNemar test in the report can be
#   recomputed without a GPU. GPQA is excluded entirely (gated dataset — see
#   data/README.md); these files are ALFWorld rollouts only.
#
# NOT published: the v8-LoRA adapters (deleted from the box before release; its
# eval JSONLs survive so its numbers remain checkable, but the weights are gone)
# and the reasoning curator checkpoints (large, and the same-domain result is a
# null — say so rather than shipping 200G of it).
set -euo pipefail
cd "$(dirname "$0")/.."

ORG=${ORG:-inference-sh}
REPO_VERL=${REPO_VERL:-$ORG/skillos-curator-qwen3-8b-verl-gigpo}
REPO_TRL=${REPO_TRL:-$ORG/skillos-curator-qwen3-8b-trl-fft}
REPO_EVAL=${REPO_EVAL:-$ORG/skillos-alfworld-eval-arms}

VERL_MERGED=${VERL_MERGED:-/mnt/nvme/output/verl-merged-hf-real}
STAGE=${STAGE:-/mnt/nvme/output/hf-staging}
LOG=${LOG:-logs/hf_publish.log}
ONLY=${ONLY:-all}

mkdir -p "$STAGE" logs
log () { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG"; }

# Files that make a standalone loadable HF model. Everything else in a TRL
# checkpoint dir is DeepSpeed/trainer state (rng_state_*, scheduler.pt,
# trainer_state.json, training_args.bin, global_step*/, zero_to_fp32.py) and is
# ~90G of the 107G — deliberately excluded.
MODEL_FILES=(config.json generation_config.json model.safetensors
             tokenizer.json tokenizer_config.json chat_template.jinja
             added_tokens.json special_tokens_map.json vocab.json merges.txt)

# TRL arms: "<label>|<checkpoint dir>"
TRL_ARMS=(
  "fft-seed1-step20|/mnt/nvme/output/alfworld-8xh100-algo1-fft/checkpoint-20"
  "fft-seed1-step60|/mnt/nvme/output/alfworld-8xh100-algo1-fft/checkpoint-60"
  "fft-seed2-step35|/mnt/nvme/output/alfworld-8xh100-algo1-fft-seed2/checkpoint-35"
  "fft-seed2-step60|/mnt/nvme/output/alfworld-8xh100-algo1-fft-seed2/checkpoint-60"
  "fft-seed3-step5|/mnt/nvme/output/alfworld-8xh100-algo1-fft-seed3/checkpoint-5"
  "fft-seed3-step55|/mnt/nvme/output/alfworld-8xh100-algo1-fft-seed3/checkpoint-55"
  "fft-seed3-step60|/mnt/nvme/output/alfworld-8xh100-algo1-fft-seed3/checkpoint-60"
)

# ALFWorld eval dirs to publish. Reasoning/GPQA dirs are intentionally absent.
EVAL_DIRS=(eval-pathbv4 eval-v8 eval-fft eval-fft-seed2 eval-fft-seed3
           eval-fft-natural eval-fft-curriculum eval-verl-gigpo-real
           eval-transfer-32b eval-transfer-32b-seed2 eval-transfer-32b-seed3
           eval-reasoning-to-alfworld eval-decode)

# ----------------------------------------------------------------- staging ---

stage_model_dir () {  # $1 = source ckpt dir, $2 = dest dir
  local src=$1 dst=$2 f
  mkdir -p "$dst"
  for f in "${MODEL_FILES[@]}"; do
    [ -f "$src/$f" ] || continue
    [ -e "$dst/$f" ] && continue
    ln "$src/$f" "$dst/$f" 2>/dev/null || cp "$src/$f" "$dst/$f"
  done
  [ -f "$dst/config.json" ] && [ -f "$dst/model.safetensors" ]
}

stage_verl () {
  log "staging verl: 12 checkpoints -> $STAGE/verl"
  local n
  for n in 5 10 15 20 25 30 35 40 45 50 55 60; do
    local src="$VERL_MERGED/step_$n" dst="$STAGE/verl/step_$n"
    if [ ! -f "$src/config.json" ]; then log "  MISSING $src — skipping"; continue; fi
    mkdir -p "$dst"
    # verl merges to sharded safetensors; take everything that isn't a lockfile
    local f
    for f in "$src"/*; do
      local b; b=$(basename "$f")
      case "$b" in *.lock|*.tmp) continue;; esac
      [ -e "$dst/$b" ] || ln "$f" "$dst/$b" 2>/dev/null || cp "$f" "$dst/$b"
    done
  done
  log "  staged: $(du -sh "$STAGE/verl" | cut -f1) in $(ls -d "$STAGE"/verl/step_* | wc -l) dirs"
}

stage_trl () {
  log "staging TRL: ${#TRL_ARMS[@]} arms -> $STAGE/trl"
  local arm label src
  for arm in "${TRL_ARMS[@]}"; do
    label=${arm%%|*}; src=${arm##*|}
    if [ ! -f "$src/config.json" ]; then log "  MISSING $src — skipping $label"; continue; fi
    if stage_model_dir "$src" "$STAGE/trl/$label"; then
      log "  $label OK ($(du -sh "$STAGE/trl/$label" | cut -f1))"
    else
      log "  $label INCOMPLETE — check $src"
    fi
  done
  log "  staged: $(du -sh "$STAGE/trl" | cut -f1)"
}

stage_eval () {
  log "staging eval arms -> $STAGE/eval"
  local d
  mkdir -p "$STAGE/eval"
  for d in "${EVAL_DIRS[@]}"; do
    [ -d "output/$d" ] || { log "  MISSING output/$d"; continue; }
    mkdir -p "$STAGE/eval/$d"
    # -L to resolve the no_memory.jsonl symlinks into real files
    find "output/$d" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.txt' \) \
      ! -iname '*gpqa*' -exec cp -L {} "$STAGE/eval/$d/" \;
    find "output/$d" -maxdepth 1 -type l -name '*.jsonl' ! -iname '*gpqa*' \
      -exec cp -L {} "$STAGE/eval/$d/" \; 2>/dev/null || true
  done
  # hard guarantee: nothing GPQA-derived leaves this machine
  if find "$STAGE/eval" -iname '*gpqa*' | grep -q .; then
    log "ABORT: GPQA-matching file found in eval staging"; exit 1
  fi
  log "  staged: $(du -sh "$STAGE/eval" | cut -f1), $(find "$STAGE/eval" -name '*.jsonl' | wc -l) jsonl"
}

# -------------------------------------------------------------- model cards ---

CAVEAT=$(cat <<'MD'
## Read this before using a checkpoint

Three findings from the sweep that produced these weights should change how you
pick one:

1. **Do not use the final checkpoint.** In all five of our training runs, step 60
   lands back at the no-memory baseline. Held-out lift peaks mid-run.
2. **The peak checkpoint moves with the seed** (step 20 / 30 / 35 / 55 across
   runs), so there is no transferable "best step". Sweep, don't guess.
3. **Sweep on your target executor, not your training executor.** Per-checkpoint
   lift on the 8B executor that generated the training data barely predicts lift
   on a 32B executor (pooled Pearson r = -0.20 over 24 checkpoint pairs; -0.68
   within one seed). The best 32B-transfer curator we found is a *step 5*
   checkpoint.

And the statistical caveat, stated plainly: across five sweeps we tested **50
checkpoint arms** against one baseline. Family-wide Bonferroni sets the bar at
p < 0.001 and our best arm anywhere is p = 0.0026, so **no same-executor
ALFWorld lift in this project survives multiple-comparison correction.** Treat
every per-arm number below as a selection statistic, not an effect size.

## Absolute numbers are not comparable to the paper

Our ALFWorld no-memory baseline is 33.6%, versus the paper's 47.9% on nominally
the same executor. We could not close that gap (prompt wording, retrieval,
seeds, serving precision and decode parameters are all ruled out; the remaining
suspect is the ReAct/atomic-verb interaction). Compare *paired lifts* against
our baseline, not absolute success rates against the paper's.
MD
)

FOOTER=$(cat <<'MD'
## Provenance

- Code, full report and every divergence from the paper:
  <https://github.com/belt-sh/skillos>
- Findings write-up with figures: `docs/repro_report.md`
- Paired eval rollouts, so every number here can be recomputed without a GPU:
  [`inference-sh/skillos-alfworld-eval-arms`](https://huggingface.co/datasets/inference-sh/skillos-alfworld-eval-arms)
- Paper: [SkillOS: Learning Skill Curation for Self-Evolving Agents](https://arxiv.org/abs/2605.06614) (Ouyang et al., 2026)

This is an independent reproduction. It is not affiliated with or endorsed by
the paper's authors.

License: Apache 2.0, inherited from [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
MD
)

write_card_verl () {
  cat > "$STAGE/verl/README.md" <<MD
---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags: [skillos, grpo, verl, gigpo, alfworld, agent-memory, skill-curation, reproduction]
pipeline_tag: text-generation
---

# SkillOS curator — Qwen3-8B, verl-agent/GiGPO (all 12 checkpoints)

A *curator* model from an independent reproduction of
[SkillOS](https://arxiv.org/abs/2605.06614). The curator's only job is to write,
revise and delete markdown skill files after each rollout; a **frozen** executor
retrieves the top-k relevant skills before every new task. Trained with GRPO
against the paper's composite reward
\`r = r_task + λ_f·r_fc + λ_u·r_cnt + λ_c·r_comp\` (λ_f=1.0, λ_u=0.1, λ_c=0.05).

This repo holds **every checkpoint** of the run, in \`step_N/\` subfolders, so the
shape of the training curve is independently re-testable — that shape is the
main finding, and a single checkpoint cannot show it.

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "$REPO_VERL"
tok = AutoTokenizer.from_pretrained(name, subfolder="step_30")
model = AutoModelForCausalLM.from_pretrained(name, subfolder="step_30", dtype="bfloat16")
\`\`\`

## Held-out results

140 ALFWorld \`valid_seen\` games, paired by gamefile, McNemar against a fixed
no-memory baseline of 33.6% (47/140). Qwen3-8B executor.

| subfolder | success rate | Δ vs no-memory | p (uncorrected) |
|---|---|---|---|
| \`step_5\`  | 37.1% | +3.6pp | 0.50 |
| \`step_10\` | 39.3% | +5.7pp | 0.20 |
| \`step_15\` | 32.1% | −1.4pp | 0.86 |
| \`step_20\` | 39.3% | +5.7pp | 0.15 |
| \`step_25\` | 34.3% | +0.7pp | 1.00 |
| **\`step_30\`** | **40.7%** | **+7.1pp** | **0.099** |
| \`step_35\` | 38.6% | +5.0pp | 0.26 |
| \`step_40\` | 39.3% | +5.7pp | 0.20 |
| \`step_45\` | 37.1% | +3.6pp | 0.50 |
| \`step_50\` | 33.6% | +0.0pp | 1.00 |
| \`step_55\` | 40.0% | +6.4pp | 0.15 |
| \`step_60\` | 34.3% | +0.7pp | 1.00 |

Best arm is \`step_30\`. Note that the curve crosses its own baseline four times
and that \`step_60\` is indistinguishable from no memory.

$CAVEAT

## Training setup

| | |
|---|---|
| Framework | verl-agent / GiGPO + FSDP |
| Hardware | 8×H100 (curator) + remote inference for executor and judge |
| Environment | real ALFWorld (\`AlfredTWEnv\`), ground-truth \`score > 0\` success |
| Executor | Qwen3-8B, frozen, temp 0.6 / top_p 0.95 / top_k 20, ReAct to 30 steps |
| Judge (\`r_cnt\`) | Qwen3-32B |
| Retrieval | BM25 (paper §3.2) |
| Schedule | 60 GRPO steps, \`|G|\`=10 positions, group size 8, 640 episodes/step |
| Wall clock | 10.2 days (~15,986 remote executor calls per step) |

Training diagnostics: within GRPO groups \`r_task\` supplied 78.9% of the reward
variance reaching the gradient, and task reward moved +0.035 (95% CI ±0.034)
across the run while policy entropy collapsed 0.139 → 0.035. The reward
machinery was working; the learning signal was simply weak.

$FOOTER
MD
  log "wrote verl model card"
}

write_card_trl () {
  cat > "$STAGE/trl/README.md" <<MD
---
license: apache-2.0
base_model: Qwen/Qwen3-8B
tags: [skillos, grpo, trl, alfworld, agent-memory, skill-curation, reproduction]
pipeline_tag: text-generation
---

# SkillOS curator — Qwen3-8B, TRL full fine-tune (selected arms, 3 seeds)

*Curator* models from an independent reproduction of
[SkillOS](https://arxiv.org/abs/2605.06614), trained with TRL's GRPOTrainer
(DeepSpeed ZeRO-3 + vLLM colocate) rather than the paper's verl. Companion to
[\`$REPO_VERL\`](https://huggingface.co/$REPO_VERL), which holds a full 12-checkpoint
verl/GiGPO sweep.

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "$REPO_TRL"
tok = AutoTokenizer.from_pretrained(name, subfolder="fft-seed3-step5")
model = AutoModelForCausalLM.from_pretrained(name, subfolder="fft-seed3-step5", dtype="bfloat16")
\`\`\`

## What is here, and what is not

Three seeds × 60 GRPO steps were trained with checkpoints every 5 steps, i.e. 36
arms. **This repo publishes 7 of them:** each seed's peak and final checkpoint,
plus seed-3 step 5. The other 29 arms exist and were evaluated — their numbers
are in \`docs/repro_report.md\` Appendix A and their per-game rollouts are in the
[eval dataset](https://huggingface.co/datasets/inference-sh/skillos-alfworld-eval-arms) —
but the weights are not uploaded. **Do not read this subset as a full sweep.**

| subfolder | 8B Δ vs no-memory | p | 32B Δ | note |
|---|---|---|---|---|
| \`fft-seed1-step20\` | **+10.7pp** | 0.032 | −2.1pp | seed-1 peak on 8B |
| \`fft-seed1-step60\` | +5.7pp | 0.18 | — | seed-1 final |
| \`fft-seed2-step35\` | **+13.6pp** | 0.0026 | −4.3pp | best 8B arm in the project |
| \`fft-seed2-step60\` | +4.3pp | 0.33 | — | seed-2 final |
| **\`fft-seed3-step5\`** | −2.1pp | 0.68 | **+13.6pp** | **best 32B arm — 62.9% absolute** |
| \`fft-seed3-step55\` | **+11.4pp** | 0.011 | +2.1pp | seed-3 peak on 8B |
| \`fft-seed3-step60\` | +3.6pp | 0.47 | — | seed-3 final |

8B: 140 ALFWorld \`valid_seen\` games vs a 33.6% no-memory baseline.
32B: same games, Qwen3-32B executor, vs a 49.3% no-memory baseline.

Look at rows 3 and 5 together. \`fft-seed2-step35\` is the strongest curator we
trained as measured on the 8B executor, and it makes a 32B executor *worse*.
\`fft-seed3-step5\` — five GRPO steps of training — is the weakest-looking arm on
8B and the strongest on 32B, at 62.9% absolute (the paper reports 61.2% for its
32B configuration). If you only have budget to evaluate a few checkpoints,
evaluate them on the executor you intend to ship.

$CAVEAT

## Training setup

| | |
|---|---|
| Framework | TRL 1.4 GRPOTrainer + accelerate + DeepSpeed ZeRO-3 + vLLM colocate |
| Hardware | 8×H100 (curator) + remote inference for executor and judge |
| Environment | real ALFWorld (\`AlfredTWEnv\`) |
| Executor | Qwen3-8B, frozen, temp 0.6 / top_p 0.95 / top_k 20 |
| Judge (\`r_cnt\`) | Qwen3-32B |
| Schedule | 60 GRPO steps, \`|G|\`=10 positions, group size 8, β=0.001 |
| Seeds | 42 (seed-1), 123 (seed-2), 456 (seed-3) |
| Wall clock | ~2.9 days per run |

TRL is not 1:1 with verl (advantage normalisation, sampling semantics, buffer
handling), and this is documented as a deviation rather than hidden — see
\`DIVERGENCES.md\` #14. The reproduction ran both frameworks precisely so that
difference could be tested; the checkpoint-curve shape reproduces in both.

$FOOTER
MD
  log "wrote TRL model card"
}

write_card_eval () {
  cat > "$STAGE/eval/README.md" <<MD
---
license: apache-2.0
task_categories: [other]
tags: [alfworld, agent-evaluation, skillos, reproduction, mcnemar]
configs:
  - config_name: default
    data_files: "eval-*/*.jsonl"
---

# SkillOS reproduction — ALFWorld paired eval rollouts

Per-game outcomes for **every ALFWorld evaluation arm** in an independent
reproduction of [SkillOS](https://arxiv.org/abs/2605.06614). This is the
evidence behind every number in the report: with these files you can recompute
each significance test yourself, on a laptop, without a GPU or an API key.

Each \`eval-*/\` directory is one sweep; each \`ckptN.jsonl\` is one arm; each line
is one held-out ALFWorld game. Arms within a sweep are **paired by gamefile**,
which is what makes McNemar the right test — the same 140 \`valid_seen\` games in
the same order for every arm.

\`\`\`bash
git clone https://github.com/belt-sh/skillos && cd skillos
python -m scripts.compare_eval_arms \\
  --arm no_memory=eval-pathbv4/no_memory.jsonl \\
  --arm ckpt30=eval-verl-gigpo-real/ckpt30.jsonl
# -> 40.7% vs 33.6%, +7.1pp, McNemar p=0.0987
\`\`\`

## Directories

| directory | what it is | arms |
|---|---|---|
| \`eval-pathbv4\` | the canonical 33.6% no-memory baseline (\`no_memory.jsonl\`) reused by every 8B sweep | 7 |
| \`eval-verl-gigpo-real\` | verl/GiGPO run, every-5 sweep | 12 |
| \`eval-fft\`, \`eval-fft-seed2\`, \`eval-fft-seed3\` | TRL full fine-tune, 3 seeds | 8–12 each |
| \`eval-v8\` | TRL LoRA r=32 run (weights no longer exist; these rollouts are all that remains) | 6 |
| \`eval-fft-natural\`, \`eval-fft-curriculum\` | the paper's two grouping ablations — both null | 12 / 9 |
| \`eval-transfer-32b*\` | same games, Qwen3-32B executor, vs a 49.3% no-memory baseline | 3–12 each |
| \`eval-reasoning-to-alfworld\` | cross-domain: a DeepMath-trained curator evaluated on ALFWorld | 12 |
| \`eval-decode\` | executor decode-parameter sweep (null) | 4 |

## Two things worth knowing

**Pool the arms before you believe any of them.** Across the five 8B sweeps here
there are 50 arms tested against one baseline. Family-wide Bonferroni sets the
bar at p < 0.001; the best arm is p = 0.0026. No same-executor lift survives
correction. The per-arm p-values in these files are uncorrected.

**\`eval-reasoning-to-alfworld\` is the strongest signal in the set, and it is
negative.** The paper reports +13.3pp on ALFWorld from a reasoning-trained
curator; these rollouts give −14 to −18pp (p ≤ 0.0005) at every checkpoint past
step 40.

## GPQA is not here

The reproduction also evaluated GPQA-Diamond, a **gated** dataset whose access
condition forbids publishing problem text, options, answers or model responses.
No GPQA rollout is included in this repo, and none ever will be — only aggregate
accuracies appear in the report. Reproducing that column requires your own
access. See \`data/README.md\` in the code repo.

AIME rollouts are also excluded here; this dataset is ALFWorld only.

$FOOTER
MD
  log "wrote eval dataset card"
}

# ------------------------------------------------------------------ upload ---

upload () {  # $1 = repo id, $2 = local dir, $3 = repo type
  log "uploading $2 -> $1 ($3, $(du -sh "$2" | cut -f1))"
  hf repos create "$1" --repo-type "$3" --exist-ok >>"$LOG" 2>&1 || true
  if hf upload "$1" "$2" --repo-type "$3" \
       --commit-message "SkillOS reproduction artifacts" >>"$LOG" 2>&1; then
    log "  DONE $1"
  else
    log "  FAILED $1 — see $LOG (re-run to resume; uploads are incremental)"
    return 1
  fi
}

# -------------------------------------------------------------------- main ---

log "=================================================================="
log "hf_publish_artifacts start (ONLY=$ONLY, org=$ORG)"
hf auth whoami >>"$LOG" 2>&1 || { log "NOT AUTHENTICATED — run: hf auth login"; exit 1; }

rc=0
if [ "$ONLY" = all ] || [ "$ONLY" = eval ]; then
  stage_eval; write_card_eval
  [ -n "${STAGE_ONLY:-}" ] || upload "$REPO_EVAL" "$STAGE/eval" dataset || rc=1
fi
if [ "$ONLY" = all ] || [ "$ONLY" = trl ]; then
  stage_trl; write_card_trl
  [ -n "${STAGE_ONLY:-}" ] || upload "$REPO_TRL" "$STAGE/trl" model || rc=1
fi
if [ "$ONLY" = all ] || [ "$ONLY" = verl ]; then
  stage_verl; write_card_verl
  [ -n "${STAGE_ONLY:-}" ] || upload "$REPO_VERL" "$STAGE/verl" model || rc=1
fi

log "hf_publish_artifacts finished rc=$rc"
exit $rc
