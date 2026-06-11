# Postmortem: Algorithm 1 group-collapse bugs (v5/v6/v7)

**Date:** 2026-06-10
**Affected runs:** algo1 v5 (FFT, no KL), v6 (FFT + KL, OOM'd), v7 (LoRA r=32 + KL, completed 60 steps)
**Status of artifacts:** v5/v7 checkpoints are trained on a degenerate task distribution; their
gradient updates do not reflect the paper's Algorithm 1 protocol. ~8 days of 8×H100 time affected.

## Summary

The Algorithm 1 environment's slot→group bookkeeping silently collapsed: every rollout in every
training step mapped to `group_id = 0`, so each rank trained on **one fixed 10-task sequence of
type "pick"** for the entire run instead of streaming the 3553-episode training split. A second,
independent bug (`hash()` salting) made that one sequence **different on each rank**, so the 8
rollouts of a single GRPO advantage group walked two different task sets — within-group reward
differences were partly task-difficulty noise. Two further majors: the content-quality judge was
never wired into Algorithm 1 (r_cnt ≡ 0), and TRL's default `loss_type="dapo"` plus HF's default
linear-LR-decay were silently in effect (undeclared deviations from the paper's GRPO setup).

## Bug 1 (critical): `group_id` always 0 — one cached task sequence per run

- TRL's `environment_factory` creates `generation_batch_size` env instances **once** at trainer
  init and reuses them via `reset()` for every generation cycle
  (`trl/trainer/grpo_trainer.py:497`). With per-device batch 2 × steps_per_generation 2, that is
  4 envs per rank, so `self._slot` ∈ {0..3} forever.
- `Algo1CuratorEnv._group_id = slot // num_generations = slot // 8 = 0` always
  (`skillos/algo1/env.py:120-121`).
- `_group_sequences` is a module-level cache keyed by gid and never cleared (env.py:44), so
  gid 0's seeds were sampled at step 1 and reused for all 60 steps.
- `_group_types` was never populated by anything, so the task type always fell back to `"pick"`
  (env.py:319).
- Net effect: `num_episodes: 3553` in the config was decorative. The curator saw the same ~10
  pick-tasks every step for 93 hours. The reward plateau at ~1.3 is consistent with memorizing
  curation for a fixed small task set.
- The code's docstrings assumed TRL creates fresh envs per generation batch ("TRL feeds rows
  through sequentially"); that assumption is false for the experimental `environment_factory`
  API and was validated only by a single-process smoke test, where slots 0–31 do enumerate.

## Bug 2 (critical): `hash(task_type)` is process-salted — groups span ranks with different tasks

- `sample_group_seeds` seeded its RNG with `hash(task_type)` (`skillos/algo1/data.py:29`).
  Python salts string hashes per process; `PYTHONHASHSEED` was set nowhere. Each of the 8
  accelerate ranks sampled a *different* "group 0" sequence.
- One GRPO group (8 generations of one prompt) is sharded 4+4 across two ranks, so within a
  single advantage group, half the rollouts played task set A and half task set B. GRPO
  advantage then compares rewards across non-comparable conditions — the same failure *class*
  as the original flat-lift bug (r_task constant in group), in a noisier form.
- Perverse interaction: this bug *generated* within-group reward variance, which satisfied the
  `frac_reward_zero_std == 0` tripwire installed after the flat-lift incident. The guardrail
  for the previous bug masked this one.

## Bug 3 (major): judge never ran in Algorithm 1 — r_cnt ≡ 0

- `scripts/train_algo1.py` passed `judge_submit=None` to `configure_algo1` with a comment
  claiming "judge wiring is currently via classic._submit_judge"; no such wiring existed.
  The env guards every judge call with `if _judge_submit is not None` — so the paper's
  λ_u·r_cnt reward term was silently absent and the configured Qwen3-32B judge was never called.
- Related latent issue: `_parse_judge_score` returned 0.0 for *unparseable* judge output,
  conflating parse failure with a genuine VALID=false verdict, and the 0.0 would be memoized
  permanently by the sha256 cache.

## Bug 4 (major): undeclared loss/schedule deviations

- TRL 1.4.0 defaults `loss_type="dapo"` (token-level normalization over the global batch);
  never overridden — the runs did not use the paper's GRPO loss aggregation.
- HF Trainer's default `lr_scheduler_type="linear"` (decay to 0, no warmup) was active: v7's lr
  fell from 1e-5 to ~1.7e-7 by step 60. Undeclared deviation if the paper used constant lr.

## Bug 5 (minor): reward-shape issues

- Early-quit mis-scoring: if a rollout ended after the priming call, `_finalize_reward` scored
  r_task from position 0 — the empty-repo position that is explicitly supposed to be excluded.
- `r_comp` compared repo size against `_input_tokens` set once from the ~50-word session-start
  prompt and never accumulated, so r_comp ≈ 0 for any non-trivial repo and rewarded near-empty
  repos (bounded by λ_c=0.05).
- `algo1_use_mega_tool: true` in the configs is read nowhere.

## Why this was missed

1. **All observable training metrics looked healthy.** Reward rose then plateaued, KL anchored,
   grad norms sane, nonzero within-group reward std (produced by Bug 2 itself).
2. **Wrong mental model of an experimental TRL API**, validated by a single-process smoke test
   where the slot arithmetic happens to work.
3. **Both criticals are silent by construction** — no crashes, no warnings. Nothing logged which
   tasks each rollout actually played; one log line of `(gid, task_type, seeds)` per rollout
   would have exposed "gid=0, pick, same seeds" on day one.
4. **v5's +5.0pp held-out eval** read as pipeline validation and lowered suspicion.

## Fixes (v8)

| # | Fix | Where |
|---|-----|-------|
| 1 | Group identity driven by **dataset columns**: one row per group with `group_id`/`task_type`; TRL passes the row to `reset(**kwargs)` and its sampler repeats the same row for all 8 generations, so group identity survives env reuse and rank sharding. Env fails loudly if the columns are missing. | `scripts/train_algo1.py`, `skillos/algo1/env.py` |
| 2 | Stable seeding: `hash(task_type)` → md5-based stable hash; `PYTHONHASHSEED=0` pinned in the launcher as belt-and-braces. | `skillos/algo1/data.py`, `run_algo1_v8_lora_kl.sh` |
| 3 | Judge wired: `judge_submit=curator_env._submit_judge`; `_parse_judge_score` raises on unparseable output (future fails → score dropped, not cached as 0.0). | `scripts/train_algo1.py`, `skillos/rewards/judge.py` |
| 4 | `loss_type="grpo"`, `lr_scheduler_type="constant"` set explicitly. | `scripts/train_algo1.py`, v8 config |
| 5 | Early-quit → r_task = 0.0 (no informed positions = no task reward); `_input_tokens` accumulated across positions. | `skillos/algo1/env.py` |

## Guardrails so this class cannot recur

- **Per-rollout task observability:** every reset logs `slot, group_id, task_type, seeds`.
  A degenerate distribution is visible in the first step's log.
- **Fail-loud contract:** env refuses to reset without dataset-provided group identity.
- **Determinism:** no use of `hash()` for cross-process-stable values; `PYTHONHASHSEED` pinned.
- **Smoke at production topology:** 8-rank smoke (heuristic backends, small G) must show
  distinct group_ids across the batch and identical seeds within a group before any multi-day
  launch.

## Eval-pipeline findings from the same review (separate from training bugs)

- LoRA adapter loading in `eval_streaming_curation.py` is **correct** (verified empirically —
  transformers auto-detects `adapter_config.json` and attaches trained weights).
- `chat_template_kwargs={"enable_thinking": False}` is not a real `apply_chat_template`
  parameter — the curator was evaluated in *thinking* mode while trained without. Uniform
  across arms and identical to the pathbv4 protocol; fixed for v8 evals (changes comparability
  to pathbv4 numbers).
- Parallel eval arms must pass distinct `--curator-device cuda:N`; output files now refuse to
  clobber without `--overwrite`.

## What carries over unchanged

Async mega-tool rollout architecture (33h→2h per step), NCCL pre-init timeout, LoRA+KL
implicit-reference memory solution (no 16 GB ref copy), vLLM colocate adapter sync, OOM
root-cause analysis, eval McNemar/pairing protocol, and the operational LoRA recipe.
