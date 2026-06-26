# Divergences from the SkillOS paper

Living document. Every time the implementation deviates from
[arXiv:2605.06614v1](https://arxiv.org/abs/2605.06614), record it here with:

- **Paper:** what the paper does
- **Ours:** what we do
- **Why:** reason (hardware, library incompatibility, deliberate ablation, …)
- **Impact:** expected effect on results vs paper (best guess, revise as we learn)
- **Status:** `forced` (no way around) / `temporary` (will fix) / `ablation` (deliberate)

Order matters: things at the top affect results most.

> **Current setup (2026-06-26):** the result-producing runs are **full
> Algorithm 1** on 8×H100 — `algo1v8` (LoRA r=32 + KL) and `algo1fft` (full
> fine-tune), `configs/alfworld_8xh100_algo1_*.yaml`, vLLM colocate, frozen
> Qwen3-8B executor + Qwen3-32B judge on inference.sh. **Algorithm 1 SUPERSEDED
> "Path B"**: it walks the real evolving |G|=10 task sequence (`curate_and_advance`)
> and computes the paper's `r_task = mean success over positions 2..|G|`. Entries
> #6/#7/#11 below describe the abandoned Path B and are annotated as superseded.
>
> **The biggest live, result-relevant divergence is now task GROUPING (#0).**
> The paper's own ablation (Table 3) says grouping is the single most important
> design choice; we diverge from it in two un-tested ways (uniform type
> distribution + no curriculum). Framework is **TRL, not verl** (#14) — a
> confound present in every run.

---

## 0. Task grouping: uniform type distribution + no curriculum (UNTESTED — likely material)

This is the highest-priority open divergence because the paper's **Table 3
ablation names grouping the single most impactful design choice** (removing it is
the largest drop, 61.2 → 57.3). We diverge in three ways:

- **Type distribution.**
  - **Paper (§3.2.1 / B.2):** partition the real training set `D` into same-type
    groups, so the number of groups of each type follows ALFWorld's **natural type
    frequencies** (Pick-heavy: the valid split is ~25% Pick, ~9% Look, ~11% Heat).
  - **Ours:** `train_algo1.py` assigns group types **round-robin** (`i % 6`) →
    **uniform** ~1/6 each (~59 of 355 groups per type). We over-sample the rare
    types (Look, Heat) and under-sample Pick.
- **Within-group ordering / curriculum.**
  - **Paper (Stage 2, Table 5):** seed a task, append related successors via a
    soft-Jaccard dependency gate with an **easy→hard curriculum** (p↑=0.80,
    difficulty deltas).
  - **Ours:** `sample_group_seeds` does `rng.sample(same-type pool)` — random,
    no curriculum, no dependency structure. (Whether the curriculum applies to
    ALFWorld at all is genuinely ambiguous in the paper — it skips the *annotation*
    for ALFWorld but doesn't clearly say it skips the *ordering*.)
- **Grouping key.** Paper groups by soft-Jaccard over latent attributes; we group
  by the coarser ALFWorld **task type** (the paper does say the 6 types are used
  as the ALFWorld partition, so this part is defensible).
- **Impact (hypothesized):** a flat, unstructured type distribution is exactly the
  "narrow/odd probe distribution" we have been blaming for the **bimodal training
  trajectory** (peak → collapse → recovery) and the residual lift gap. It also
  creates a **train(uniform)/eval(natural) mismatch** the paper doesn't have — the
  held-out eval *is* the natural distribution.
- **Status:** `untested` — the cheapest high-value experiment available. First
  test: switch type assignment to natural ALFWorld frequencies (a few lines), keep
  everything else, and see whether the trajectory stabilizes.

## 1. ~~Single GPU~~ 8×H100, and ~~LoRA~~ full fine-tune (RESOLVED)

> **Update 2026-05-26:** migrated to 8×H100 + full fine-tune; this is no longer a
> divergence on the active path. LoRA remains available as a single-GPU fallback.
> Original entry kept below for history.

### Original entry — Single GPU, not 16× H100 — *and* LoRA, not full fine-tune

This is the biggest deviation. Worth splitting into two parts.

### 1a. Hardware
- **Paper:** 16× H100, distributed training
- **Ours:** 1× RTX 6000 Pro Blackwell (96 GB)
- **Why:** hardware available
- **Status:** `forced`

### 1b. LoRA r=32 (instead of paper's full fine-tune)
- **Paper:** full fine-tune of Qwen3-8B, all 8B parameters update
- **Ours:** LoRA r=32 on all-linear, ~87M trainable parameters (1.05% of model)
- **Why:** full FT on a single 96 GB GPU at paper effective batch (32 × 8
  rollouts) requires roughly 180 GB (bf16 + Adam fp32 + ref model + KV cache
  for 16 in-flight rollouts at max_completion=4096). Even with 8-bit Adam and
  bf16 master weights it's ~100-120 GB, still over the budget.
- **Impact:** This is the biggest blocker to claiming a paper-faithful result.
  The paper's headline number (RL-trained 8B beats Gemini-2.5-Pro at skill
  curation) is from updating all 8B params. LoRA caps how much behavior the
  curator can acquire — particularly important because the paper observes
  emergent meta-skills late in training that almost certainly require deeper
  parameter changes than r=32 can express. We can train and watch the loss
  decline, but we cannot claim "matched the paper" with LoRA alone.
- **Status:** `forced` until either (a) we add 8-bit Adam + CPU optimizer
  offload to squeeze full FT onto 96 GB at the cost of ~2× wall time, or
  (b) we move to 8× H100 where full FT fits trivially.

> **Update 2026-06-24 — FFT control run settles the LoRA question.** We ran
> `algo1fft` (`configs/alfworld_8xh100_algo1_fft.yaml`): **v8 EXACTLY minus
> LoRA** — full fine-tune, ZeRO-3 + vLLM colocate, `beta=0.001`, `lr=1e-6`, all
> other paper hyperparams identical. ZeRO-3 (not the older-stack "validated"
> ZeRO-2, which **hung** here) shards params + ref model so the KL term fits; 60
> steps, 0 OOM. The held-out McNemar sweep is **still bimodal** (peak ckpt20
> +10.7pp p=0.032 → collapse to baseline parity ckpt30 → recovery), the same
> oscillating shape as v8 LoRA (which peaked at ckpt30). **Conclusion: the
> bimodality is NOT a LoRA artifact** — it's framework/algorithm-level
> (small-batch GRPO on a narrow probe distribution, TRL path). FFT's best arm
> (+10.7pp) actually beats LoRA's best (+9.3pp) and lands within ~2.6pp of the
> paper's +13.3pp. So LoRA was a faithful-enough stand-in for the trajectory
> *shape*; it did not cause the deviation from the paper's monotone curve. See
> `JOURNAL.md` 2026-06-24 for the full table.

### Mitigations on the table (single-GPU path)
- **CPU-offload optimizer (DeepSpeed ZeRO-2 with offload_optimizer):** brings
  full-FT peak from ~180 GB to ~70-80 GB. Slows training ~1.5-2×.
- **8-bit Adam (bitsandbytes):** halves optimizer memory. Often combined with
  CPU offload for further savings.
- **`beta: 0` (drop the KL reference model):** frees 16 GB but is itself a
  divergence (see audit checklist — paper uses 0.001).
- **Smaller `max_completion_length`:** would free 15-20 GB of KV cache but
  truncates curator skill content, breaking r_fc and r_cnt.

### Migration path to 8× H100
- LoRA adapter (~350 MB) and checkpoints (`output/<run>/checkpoint-N/`) are
  portable. scp to the new box, drop under the same `output_dir`, set
  `resume_from_checkpoint` in the config.
- Use `configs/alfworld_multi_gpu.yaml` with `accelerate launch -m skillos.train`.
- On 8× H100, full FT fits trivially (640 GB aggregate), vLLM compat issues
  may also resolve, and per-step wall time drops ~8×.

## 2. ~~vLLM disabled~~ vLLM enabled (RESOLVED)

> **Update 2026-05-26:** vLLM is now **enabled** on the 8×H100 path
> (`use_vllm: true`, `vllm_mode: colocate`, `vllm_gpu_memory_utilization: 0.4`)
> with the torch 2.11 / transformers 5.9 / trl 1.4 / vllm 0.21 stack, which
> resolves the compatibility window described below. Original entry kept for
> history.

### Original entry — vLLM disabled

- **Paper:** uses vLLM (colocate or server mode) for fast rollout generation
- **Ours:** native HuggingFace `model.generate()` — vLLM 0.11/0.12 collide with
  TRL 1.4 + transformers 5.2 (`tokenizer.all_special_tokens_extended` removed
  in transformers 5, still called by vLLM 0.11; vLLM 0.12+ pins
  transformers<5; TRL 1.4 `environment_factory` requires transformers≥5.2)
- **Why:** unsolved upstream library compatibility window
- **Impact:** ~5–10× slower rollout generation. Doesn't change correctness or
  loss trajectory, only wall time. Big.
- **Status:** `resolved` — vLLM colocate active on the 8×H100 stack

## 3. Executor + judge run on inference.sh (remote API), not local GPUs

- **Paper:** all three models (curator/executor/judge) on the same H100 cluster
- **Ours:** executor and judge call `openrouter/qwen3-8b` and
  `openrouter/qwen3-32b` on inference.sh via the inferencesh Python SDK
- **Why:** frees 100% of local VRAM for the curator (the only model that's
  actually being trained)
- **Impact:** correctness unchanged — both endpoints serve the same exact
  weights. Latency higher (~5–13s per call vs ~1s local), cost ~$5/epoch.
- **Status:** `ablation` (different deployment, same models)

## 4. `max_prompt_length` not set in `GRPOConfig`

- **Paper:** Table 4 lists `max_prompt_length=16384`
- **Ours:** unset (TRL `GRPOConfig` does not expose this parameter; uses the
  tokenizer's model_max_length, which for Qwen3-8B is 32768 natively)
- **Why:** TRL API doesn't accept the field. The paper's 16384 was a
  truncation cap; our effective cap (32768) is *larger* than theirs, so no
  prompt that fit theirs would be truncated by ours.
- **Impact:** none — our cap is strictly more permissive
- **Status:** `forced` (API), benign

## 5. ALFWorld step cap

- **Paper:** "up to 30 steps, but executor only sees last 3 at a time"
- **Ours:** `SKILLOS_EXECUTOR_MAX_STEPS=30` on the active 8×H100 path
  (`run_pathb.sh`), matching the paper. Default in code is still 10; the env var
  raises it. (The old 10-step cap biased `r_task` low on heat/cool/pick2.)
- **Why:** infsh is elastic, so the extra executor steps are affordable now.
- **Impact:** matches paper's step budget; removes the low-`r_task` bias.
- **Status:** `resolved` on the 8×H100 path (env override to 30)

## 6. Grouped task streams — Path B → SUPERSEDED by Algorithm 1

> **Update 2026-06-26 — Algorithm 1 implements the real grouped task streams.**
> The active runs walk the evolving |G|=10 same-type sequence via
> `Algo1CuratorEnv.curate_and_advance`; the repo evolves position-to-position and
> `r_task` is the paper's mean over positions 2..|G|. Path B (the single-step
> probe stand-in) is no longer used. What REMAINS divergent is *how the groups are
> built* — see **#0** (uniform distribution, no curriculum). The entry below is
> kept for history.

- **Paper §3.2.1:** Tasks grouped via soft-Jaccard over attribute annotations
  (topic, skills, concepts, strategies, pitfalls); within a group, repo
  evolves across tasks 1..N and `r_task` averaged over tasks 2..N
- **Ours (was):** `data/grouping.py` placeholder; flat dataset; one shared repo
  across all rollouts — effectively a single group the size of the dataset.
  This is the root cause of the flat held-out lift (constant within-group
  `r_task`); see `JOURNAL.md`.
- **Ours (now):** **Path B** (#11) provides a pragmatic stand-in for the paper's
  within-group transfer signal — same-type probe tasks per rollout instead of an
  evolving 1..N sequence. The soft-Jaccard attribute grouping itself is still
  not implemented (we group by ALFWorld task *type*, which is coarser).
- **Status:** `temporary` — Path B restores the *signal*; full §3.2.1 attribute
  grouping + evolving sequence is the escalation if Path B falls short.

## 7. ~~Skill ops mutate shared repo~~ per-rollout ephemeral repo (RESOLVED)

> **Update 2026-05-26:** fixed by Path B — each `CuratorEnv` rollout now owns an
> ephemeral `SkillRepo` (`self._repo`), so parallel rollouts no longer race on a
> shared repo. Original entry kept for history.

### Original entry — Skill ops mutate shared repo immediately

- **Paper §3.1 (pseudocode):** ops applied to repo sequentially across rollouts
  within a task. Each rollout's reward is computed on the post-op state.
- **Ours:** tool methods on `CuratorEnv` mutate `_shared_skill_repo` directly
  the moment they're called by the curator. Multiple parallel rollouts race on
  the shared repo.
- **Why:** simpler implementation; rollouts are mostly empty-output in early
  training so the race is benign
- **Impact:** correctness drift when curator starts emitting ops in volume.
- **Status:** `resolved` — per-rollout ephemeral repo (Path B)

## 8. Reasoning + WebShop benchmarks not built

- **Paper:** trains and evaluates on three domains — ALFWorld, DeepMath-103K
  + GPQA-Diamond (reasoning), WebShop. Reports Tables 1-3 across all of them.
- **Ours:** only ALFWorld is wired. Reasoning + WebShop have:
  - prompt templates in `curator/prompts.py` (REASONING_EXECUTOR,
    REASONING_CORRECTNESS_JUDGE)
  - placeholder `data/grouping.py::group_reasoning_tasks`
  - declared extras in `pyproject.toml` (`sympy` for reasoning,
    `stable-baselines3` for webshop)
  - **no environment wrappers, no training configs, no dataset loaders**
- **Why:** scope — reproducing ALFWorld first as v0.1. Reasoning + WebShop
  scheduled for v0.2 in the README roadmap.
- **Impact:** Cannot reproduce the paper's cross-domain transfer claim
  ("+13.3% on ALFWorld from reasoning training") or the three-benchmark
  averages. Per-domain ALFWorld numbers are still comparable.
- **Status:** `temporary` (v0.2 work)

## 9. `max_completion_length: 8192` (LoRA pilot) instead of 4096

- **Paper:** `max_completion = 4096`
- **Ours (LoRA pilot only):** 8192 — doubled after the first pilot's step-1 stats
  showed `completions/clipped_ratio: 0.78`, i.e. 78% of curator completions hit
  the 4096-token cap. Most curator outputs (skill markdown + reasoning) need
  more headroom.
- **Why:** clipping at 78% truncates the very thing the reward function scores
  (the skill content). Reward signal is partial-credit on truncated text → noisy
  gradient. Doubling the cap lets the curator finish its writes.
- **Impact:** more activation memory per opt step (longest sequences scale O(N)
  with flash attn), slower wall time per step. Within 96 GB budget at LoRA r=32.
  Other configs (`alfworld_paper.yaml`, `loss_check.yaml`, `debug.yaml`) still
  use paper's 4096 — only the LoRA pilot diverges.
- **Status:** `ablation` (pragmatic, revisit when scaling)

---

## 10. Save strategy

- **Paper:** doesn't specify checkpointing cadence
- **Ours:** `save_strategy: steps`, `save_steps: 25`, `save_total_limit: 3`
  (override per config). Plus emergency `save_pretrained()` fallback inside
  `train.py` if `trainer.save_model()` crashes (e.g. wandb model-card bug).
- **Status:** `ablation` (engineering safety net, no learning impact)

---

## 11. Path B: transfer-probe `r_task` — SUPERSEDED (now paper-aligned)

> **Update 2026-06-26 — RESOLVED, no longer a divergence.** The result-producing
> Algorithm 1 runs compute `r_task = mean executor success over positions 2..|G|`
> over the real evolving sequence (`skillos/algo1/env.py`), exactly the paper's
> formula — NOT the 2-probe Path B stand-in described below. The only residual
> grouping divergence is **#0** (how groups are constructed), not how `r_task` is
> computed. Entry kept for history.

- **Paper §3.2.1 / Algorithm 1:** a training instance is a *sequence* of |G|
  related tasks; the repo evolves across tasks 1..N and
  `r_task = (1/(|G|−1))·Σ_{i=2}^{N} 1(ξ_i)` (success averaged over tasks 2..N).
- **Ours (Path B):** single curation step per rollout against an ephemeral repo,
  then `r_task = mean frozen-executor success over num_probe_tasks (=2)
  freshly-sampled SAME-TYPE probe tasks`. Probe games are shared within a GRPO
  group via deterministic `env.seed()`, so the within-group comparison isolates
  curation quality.
- **Why:** TRL's tool loop runs serially across the batch; a faithful evolving
  1..N sequence per rollout multiplies wall time past usability. Path B keeps
  the part Table 3 says matters most (within-group transfer variance) and drops
  the expensive part (the long sequence). It is the **direct fix** for the flat
  held-out lift — the old design's `r_task` was constant within a group and
  cancelled out of the GRPO advantage. Full narrative in `JOURNAL.md`.
- **Impact:** restores a non-degenerate task-success gradient (`frac_reward_zero_std=0`
  every step, confirmed). Coarser than the paper: we group by ALFWorld *type*,
  not soft-Jaccard attributes, and use 2 probes rather than an N-task average.
  Held-out lift TBD.
- **Status:** `ablation` — pragmatic; escalate to full Algorithm 1 if it falls short

## 12. Eval must build memory via streaming curation (not load a saved repo)

- **Consequence of #11:** training repos are now ephemeral, so there is no
  persisted training repo to evaluate. Eval has to construct memory by running
  the trained curator over a curation stream (the streaming closed-loop the
  paper actually uses), then measure executor success with that memory.
- **Note:** use the **same infsh executor** for eval as for training. The old
  41%-SR eval ran the executor via vLLM with `presence_penalty=1.5`, which the
  `InfshExecutor` does *not* send — a train/eval mismatch to avoid.
- **Status:** `temporary` — new eval harness pending the run finishing.

---

## 13. No-memory executor baseline ~14pp below the paper (UNRESOLVED — not a bug)

Our frozen-executor no-memory ALFWorld baseline is **~34%** (140 valid_seen),
vs the paper's **47.9%** — same Qwen3-8B, same 30-step cap, same Fig 9 prompt.
Audited the entire decode/serving/harness hypothesis tree (full ledger in
`JOURNAL.md`, 2026-06-21). **Everything checkable is ruled out:** prompt text
(Fig 9 verbatim, from `docs/skillos_paper.pdf`), precision (local bf16 ≈ remote),
decode (reasoning on, 8192 budget, parse clean), retrieval (BM25 top-5), seeds,
averaging, success detection (`scores>0 ⟺ info['won']`), env/ReAct harness
(diffed vs GiGPO `verl-agent` — only cosmetic admissible/history formatting
differs), and K=20 batching (no degradation signature).

Mechanism (real but not prompt-fixable): the executor ignores ALFWorld's atomic
`heat/cool/clean X with Y` actions and role-plays the physical procedure,
timing out on composite verbs (Clean/Cool/Heat/Pick2). A one-line grammar hint
did not move it at n=140 (+2.1pp, p=0.68).

> **Update 2026-06-25 — decode RULED OUT, gap is 8B-specific.** A 4-config
> executor-decode sweep (temp/top_p/top_k/reasoning, n=140 each) left the baseline
> within noise of itself (all p>0.5); matching GiGPO's `top_p 1.0 / top_k off` is
> *nominally worse* for Qwen3 (it fights the Qwen3 model card, whose thinking-mode
> values 0.6/0.95/20 are already our default). Meanwhile the **32B executor
> reproduces the paper's 54.5%** (we got 53.6%). Conclusion: harness is sound, the
> 8B zero-shot baseline genuinely doesn't reach 47.9% — a reproducibility finding /
> question for the authors. The lift is unaffected (measured vs this same baseline).

**Important:** this is a baseline (no-memory) divergence and does **not** invalidate
the curator result — the **+9.3pp ckpt30 lift is measured against this same
baseline** and is real. The residual is either the canonical zero-shot Qwen3-8B
genuinely scoring ~34% (paper's baseline optimistic) or an undocumented detail
in the paper's GiGPO-deferred executor harness. Open question for the authors.
NOTE: the knowledge entry `infsh/skillos-validated-executor-config` claims the
baseline reproduces at 42.9% — that is a **mis-record of the ckpt30 with-memory
number**; the correct prior note is `infsh/reasoning-fix-insufficient-baseline-gap`.

## 14. RL framework: TRL, not verl (confound in every run)

- **Paper:** trains with **verl** (+ FSDP) on 16×H100.
- **Ours:** **TRL 1.4** GRPOTrainer + DeepSpeed (ZeRO-3) + vLLM colocate on 8×H100.
- **Why:** the codebase is built on TRL; a verl port is a large effort.
- **Impact:** a genuine framework confound separate from LoRA-vs-FFT and the
  grouping divergence (#0). Different GRPO loss plumbing, advantage normalization,
  data sampler, and optimizer/sharding stack. This is a leading candidate (with
  #0 and small effective batch) for the **bimodal trajectory** that the paper does
  not report. Don't oversell repro fidelity — TRL ≠ verl is present in every run.
- **Status:** `forced` (framework choice), flagged as a known confound.

## Open audit items (verify next time we touch them)

- [x] Reward weights `λ_f=1.0, λ_u=0.1, λ_c=0.05` — confirmed exact match in
  `skillos/rewards/composite.py` (defaults `lambda_f=1.0, lambda_u=0.1, lambda_c=0.05`)
- [x] Curator's tool palette — paper exposes exactly `new_skill_insert`,
  `skill_update`, `skill_delete`. Bug found 2026-05-20: `compute_reward` was
  public, so TRL's `inspect.getmembers` auto-registered it as a 4th tool. Fixed
  by renaming → `_compute_reward`. The bug inflated `tools/failure_frequency`
  to 0.48 because curator was calling the no-op reward method as a tool.
- [x] Curator system prompt — bug found 2026-05-20: `CURATOR_SYSTEM` from
  Appendix A.1 was defined in `prompts.py` but never used. The dataset only
  shipped a terse user message, so the curator's system prompt was reduced to
  TRL's auto-generated "you have these tools" boilerplate. Without the paper's
  Action Guidelines ("If trajectory is correct, extract reusable knowledge…
  If incorrect, identify the failure point and extract skills that can help
  fix the issue"), Qwen3-8B refused to call any tool and emitted ~85 tokens
  of natural-language "no skill needed" instead. Fixed by including
  `{"role": "system", "content": CURATOR_SYSTEM}` in `build_dataset`.
- [x] Task description extraction — bug found 2026-05-20: ALFWorld puts the
  actual task on the line `Your task is to: …` near the *end* of the initial
  observation, not the start. We were taking `observation.split("\n")[0]`,
  which is the welcome banner. Fixed in `_extract_task_description` to scan
  lines for the "your task is" prefix.
- [ ] Top-K skill retrieval = 5 — confirmed in `curator_env.py` (`top_k=5`)
- [ ] `temperature=1.0` for curator generation — confirmed in configs
- [ ] `learning_rate=1e-6` — confirmed in configs
- [x] `beta` (GRPO KL coefficient) — **RESOLVED**: the active Algorithm 1 runs
  (`algo1v8`, `algo1fft`) use `beta: 0.001` (paper value); ZeRO-3 shards the ref
  model so it fits. The old `beta: 0.0` was a single-GPU/Path B memory mitigation,
  now obsolete.
- [ ] `steps_per_generation` defaults — TRL default = `gradient_accumulation_steps`;
  paper doesn't specify but this is the standard GRPO convention
