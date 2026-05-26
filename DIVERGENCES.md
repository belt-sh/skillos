# Divergences from the SkillOS paper

Living document. Every time the implementation deviates from
[arXiv:2605.06614v1](https://arxiv.org/abs/2605.06614), record it here with:

- **Paper:** what the paper does
- **Ours:** what we do
- **Why:** reason (hardware, library incompatibility, deliberate ablation, …)
- **Impact:** expected effect on results vs paper (best guess, revise as we learn)
- **Status:** `forced` (no way around) / `temporary` (will fix) / `ablation` (deliberate)

Order matters: things at the top affect results most.

> **Current setup (2026-05-26):** we have migrated off the single-GPU/LoRA path.
> The active run (`configs/alfworld_8xh100_pathb.yaml`, `run_pathb.sh`) is a
> **full fine-tune of Qwen3-8B on 8×H100 with vLLM colocate**, frozen Qwen3-8B
> executor + Qwen3-32B judge on inference.sh. Entries below are annotated where
> this supersedes the original single-GPU framing. The biggest *new* divergence
> is the **Path B transfer-probe `r_task`** (entry #11) — read `JOURNAL.md` for
> the why.

---

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

## 6. Grouped task streams — replaced by Path B transfer-probe (see #11)

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

## 11. Path B: transfer-probe `r_task` instead of paper's task-sequence average

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
- [ ] `beta` (GRPO KL coefficient) — paper uses 0.001; v3 set 0.001; **Path B
  config (`alfworld_8xh100_pathb.yaml`) sets `beta: 0.0`** (no KL term, drops the
  reference model). Confirm whether this is intended or should match the paper.
- [ ] `steps_per_generation` defaults — TRL default = `gradient_accumulation_steps`;
  paper doesn't specify but this is the standard GRPO convention
