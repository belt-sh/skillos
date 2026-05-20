# Divergences from the SkillOS paper

Living document. Every time the implementation deviates from
[arXiv:2605.06614v1](https://arxiv.org/abs/2605.06614), record it here with:

- **Paper:** what the paper does
- **Ours:** what we do
- **Why:** reason (hardware, library incompatibility, deliberate ablation, …)
- **Impact:** expected effect on results vs paper (best guess, revise as we learn)
- **Status:** `forced` (no way around) / `temporary` (will fix) / `ablation` (deliberate)

Order matters: things at the top affect results most.

---

## 1. Single GPU, not 16× H100

- **Paper:** 16× H100, distributed training, full fine-tune of Qwen3-8B
- **Ours:** 1× RTX 6000 Pro Blackwell (96 GB), LoRA fine-tune (r=32, all-linear)
- **Why:** hardware available
- **Impact:** Same effective batch (32 via gradient accumulation), but LoRA caps
  the trainable parameter count to ~87 M / ~8 B (1.05%). May limit how much the
  curator can change behavior vs full FT. Paper does not report LoRA ablations,
  so the gap is unquantified.
- **Status:** `forced`

## 2. vLLM disabled

- **Paper:** uses vLLM (colocate or server mode) for fast rollout generation
- **Ours:** native HuggingFace `model.generate()` — vLLM 0.11/0.12 collide with
  TRL 1.4 + transformers 5.2 (`tokenizer.all_special_tokens_extended` removed
  in transformers 5, still called by vLLM 0.11; vLLM 0.12+ pins
  transformers<5; TRL 1.4 `environment_factory` requires transformers≥5.2)
- **Why:** unsolved upstream library compatibility window
- **Impact:** ~5–10× slower rollout generation. Doesn't change correctness or
  loss trajectory, only wall time. Big.
- **Status:** `temporary` — revisit when upstreams align

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
- **Ours:** `SKILLOS_EXECUTOR_MAX_STEPS=10` by default (env var override)
  — 30 steps × num_generations × num_episodes is prohibitively slow on remote
  executor API; 10 steps still hits success on most ALFWorld task types
- **Why:** wall-time and cost
- **Impact:** Caps task success on harder games (heat/cool/pick2 often need
  15-20 steps). Our `r_task` will be biased low vs paper. Easy to re-tune.
- **Status:** `ablation` (revisit when wall time allows)

## 6. Grouped task streams not yet wired into training

- **Paper §3.2.1:** Tasks grouped via soft-Jaccard over attribute annotations
  (topic, skills, concepts, strategies, pitfalls); within a group, repo
  evolves across tasks 1..N and `r_task` averaged over tasks 2..N
- **Ours:** `data/grouping.py` exists but is a placeholder; training dataset is
  flat. The shared `_shared_skill_repo` persists across all rollouts without
  group boundaries — equivalent to one giant "group" the size of the dataset.
- **Why:** not implemented yet; was on the post-pipeline-validation TODO
- **Impact:** Significant — paper credits much of the curator's learning to
  the grouped structure (updates dominate inserts in late training because the
  curator sees its own skills get reused in adjacent tasks). Without grouping,
  the learning signal is sparser and the "update > insert" emergence may not
  show up.
- **Status:** `temporary` — implement once basic loss-decline is shown

## 7. Skill ops mutate shared repo immediately, not per-rollout-atomically

- **Paper §3.1 (pseudocode):** ops applied to repo sequentially across rollouts
  within a task. Each rollout's reward is computed on the post-op state.
- **Ours:** tool methods on `CuratorEnv` mutate `_shared_skill_repo` directly
  the moment they're called by the curator. Multiple parallel rollouts race on
  the shared repo.
- **Why:** simpler implementation; rollouts are mostly empty-output in early
  training so the race is benign
- **Impact:** correctness drift when curator starts emitting ops in volume. We
  expect noisier gradients but no failure mode (`SkillRepo` is internally
  thread-safe; ops just don't compose cleanly).
- **Status:** `temporary` — implement per-rollout local op buffers when it
  starts mattering

## 8. Save strategy

- **Paper:** doesn't specify checkpointing cadence
- **Ours:** `save_strategy: steps`, `save_steps: 25`, `save_total_limit: 3`
  (override per config). Plus emergency `save_pretrained()` fallback inside
  `train.py` if `trainer.save_model()` crashes (e.g. wandb model-card bug).
- **Status:** `ablation` (engineering safety net, no learning impact)

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
- [ ] `beta=0.001` (GRPO KL coefficient) — added in v3; was previously
  defaulting to TRL's `beta=0` (no KL term) which is a silent divergence we
  fixed
- [ ] `steps_per_generation` defaults — TRL default = `gradient_accumulation_steps`;
  paper doesn't specify but this is the standard GRPO convention
