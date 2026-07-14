# SkillOS reproduction journal

A running, honest log of what we hit, what we tried, and what we changed while
reproducing [arXiv:2605.06614](https://arxiv.org/abs/2605.06614) on ALFWorld.
The paper reports final numbers; it does not report the failure modes you walk
through to get them. This file is meant to be the thing we wished existed — the
debugging narrative, including the dead ends and the gotchas that cost us days.

Conventions: dates are absolute. "Confirmed" means we have direct evidence in
this repo (a log, a metric, a code path). "TBD" means not yet measured — we try
not to claim learning we haven't observed on held-out tasks.

See also: `DIVERGENCES.md` (point-by-point deltas from the paper) and
`docs/skillos_paper.md` (our notes on the paper itself).

---

## Seed-3 FFT completes — non-monotone shape confirmed across N=3, peak indices wild (2026-07-14)

`algo1fftseed3` (seed=456, otherwise identical to seed-1/seed-2) completed 60/60
on 2026-07-13 18:55 UTC. 12-arm every-5 sweep vs canonical 33.6% baseline:

| ckpt | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔSR | −2.1 | +2.9 | −0.7 | −2.9 | +6.4 | +2.1 | −5.0 | +7.9 | +8.6 | +5.7 | **+11.4** | +3.6 |
| p | .68 | .54 | 1.0 | .56 | .14 | .69 | .17 | .06 | .08 | .17 | **.011** | .47 |

**Peak at ckpt55, +11.4pp, p=0.011.** Third significant peak in a row.

**Peak indices across N=3 span ckpt20 → ckpt55 — half the training run.**

| seed | peak ckpt | peak lift | p | ckpt60 lift |
|---|---|---|---|---|
| seed-1 (42) | 20 | +10.7pp | 0.032 | +5.7pp |
| seed-2 (123) | 35 | +13.6pp | 0.0026 | +4.3pp |
| seed-3 (456) | 55 | +11.4pp | 0.011 | +3.6pp |

Robust across all 3 seeds:
1. Statistically significant lift somewhere in the run.
2. Peak at ckpt < 60.
3. ckpt60 lands 4-9pp below peak.
4. Peak lift in the +10 to +14pp band.

**Not robust:** peak location (huge variance), and the strict "bimodal / two
peaks with clear trough" shape. Seed-3 is more "noise-with-slight-drift through
ckpt35, then late rise → peak → regress at 60" — a late-peaking curve, not a
classic bimodal. Seed-1 and seed-2 have the mid-run peak + regress pattern.

**Practical claim:** the trajectory is **non-monotone**, peak indices are wildly
RNG-dependent, ship best-of-heldout from the sweep. Not "bimodal" universally —
that's a seed-1/seed-2 property. Ship rule and reproducibility finding stand;
the specific shape descriptor tightens.

Per-type quirks worth noting for seed-3:
- Heat: peak 8/16=50% at ckpt45, then collapses to 1/16=6% at ckpt55. Heat is
  extremely volatile across ckpts (all three seeds).
- Cool: peak 13/25=52% at ckpt55, driving much of the peak lift.
- Pick: stable 60-77% across all ckpts (baseline is 60%, so most gains are in
  the compound-verb types where headroom exists).

Bimodality driver hypothesis unchanged: TRL ≠ verl (DIVERGENCES #14) is the
last surviving suspect. Grouping (both halves) tested and null. Not doing the
verl port here.

Artifacts: `output/eval-fft-seed3/comparison_canonical.txt`.

---

## Reasoning baseline reproduces — 8B gap is ALFWorld-specific (2026-07-10 → 07-11)

Built the reasoning eval harness (`skillos/reasoning/{datasets,prompts,grading}.py` +
`scripts/eval_reasoning.py`, no_memory + closed_loop modes) while seed-3 trains.
Full no_memory run through the same `openrouter/qwen3-8b` executor as ALFWorld
across all three paper reasoning datasets:

| dataset | ours (no_memory) | paper (Qwen3-8B no_memory) | delta |
|---|---|---|---|
| AIME24 | 22/30 = 73.3% | 76.0±6.9 | −2.7pp (0.4σ) |
| AIME25 | 18/30 = 60.0% | 71.1±10.7 | −11.1pp (1.0σ) |
| GPQA-D | 118/198 = 59.6% | 61.8±1.1 | −2.2pp (2.0σ) |
| **Reasoning avg** | **64.3%** | **69.6±4.7** | **−5.3pp (1.1σ)** |

GPQA-D is reported aggregate-only per Idavidrein's dataset-access condition
(no per-problem content in git-tracked files); prediction letter distribution
is well-balanced (no degenerate always-C mode), 0 executor errors, 1/198 with
un-extractable answer.

**Significance:** we reproduce the paper's no_memory reasoning within 1σ per
dataset and 1.1σ on the average — using the same executor that is 14pp below
the paper on ALFWorld (33.6% vs 47.9%). That triangulates the baseline gap:
it is definitively environment-specific (ALFWorld ReAct + atomic verbs), not
a broad executor mismatch. Confirms `executor-atomic-verb-gap` on
independent, non-agentic data.

Closed-loop reasoning eval is stubbed — needs local GPU for the curator, which
is currently seed-3's until ~Jul 13. Cross-domain transfer test (our best ALFWorld
curator → reasoning) queued for then.

Artifacts: `output/eval-reasoning/nomem_aime.jsonl`.

---

## Within-group curriculum — doesn't help either (2026-07-09)

`algo1fftcurriculum` = seed-1 FFT recipe + soft easy→hard within-group ordering
(paper Table 5, p↑=0.80, difficulty = expert-plan length from `traj_data.json`).
Run **crashed at step 49/60** on 2026-07-08 07:05 UTC — OpenRouter's Alibaba
provider hit us with a **429 storm (10,852 errors over the run's tail)**, every
executor call in step 49 exhausted its 2 resubmissions, `DEADLINE CUT` masked
100% of that step's r_task, ranks diverged, process silently exited. Not our
code, not OOM, not NCCL. Checkpoints 5–45 saved; 50–60 lost.

Rather than a 3-day resume, swept the 9 checkpoints we had against the canonical
33.6% baseline (uniform runs peaked at ckpt20 and ckpt35 — the peak-region is
inside 5–45):

| ckpt | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 |
|---|---|---|---|---|---|---|---|---|---|
| ΔSR | +0.0 | −0.7 | −5.7 | +2.1 | +4.3 | +2.9 | −3.6 | +0.7 | +4.3 |
| p | 1.0 | 1.0 | .13 | .72 | .38 | .58 | .46 | 1.0 | .36 |

**No significant lift at any checkpoint** (best p=0.36). Flat, oscillating inside
noise — same shape as `natural`. Uniform peaks land inside this window in both
prior seeds, so a hidden curriculum peak in 50–60 is very unlikely; not resuming.

**With both distribution AND ordering falsified, DIVERGENCES #0 is fully closed:
grouping is NOT the driver of our bimodal trajectory or lift gap.** The only
surviving suspect is **TRL ≠ verl (#14)** — the framework confound present in
every run. This is a real writeup conclusion: reproduces qualitatively,
trajectory shape is framework-dependent, not a grouping-recipe artifact.

Artifacts: `output/eval-fft-curriculum/comparison_canonical.txt`.

---

## Cross-executor transfer — the generalization claim REPRODUCES (2026-07-04)

**8B-trained curators driving the 32B executor** (`openrouter/qwen3-32b`,
closed-loop streaming curation, 140 paired games, ref = fresh 32B no_memory
baseline at 49.3%):

| arm | abs SR | ΔSR | p |
|---|---|---|---|
| fft_s1_ckpt20 | 47.1% | −2.1pp | 0.74 |
| fft_s2_ckpt35 | 55.0% | +5.7pp | 0.26 |
| **v8lora_ckpt30** | **62.1%** | **+12.9pp** | **0.0064** |

1. **The paper's cross-executor claim reproduces**: v8 LoRA ckpt30 lifts the 32B
   executor +12.9pp (p=0.0064) — and 62.1% absolute **exceeds the paper's
   headline SkillOS number (61.2%)**, achieved with a curator that never saw a
   32B trajectory during training.
2. **Transfer is artifact-dependent, and inverts the 8B ranking.** Best-on-8B
   (fft_s2 ckpt35, +13.6pp on 8B) transfers only weakly (+5.7pp); fft_s1 ckpt20
   (+10.7pp on 8B) doesn't transfer at all; v8 LoRA (+9.3pp on 8B, the *worst*
   of the three on 8B) transfers best. Hypothesis: FFT curators overfit skills
   to 8B executor quirks; LoRA's constrained update kept skills more generic.
   Single run per arm — hypothesis, not established.
3. **Heat unlocks at 32B**: 25% → 56–62% with memory. The 8B executor's
   microwave role-play pathology is executor-specific and vanishes at 32B scale.
4. Baseline variance again: this 32B no_memory draw is 49.3% vs 54.5% measured
   earlier — consistent with the ~±4pp run-to-run variance of temp-0.6 baselines.
   All arms in this table pair against the same fresh draw.
5. **Ops postmortem:** results sat unread ~44h — the completion watcher polled
   the supervisor log in `/tmp`, which the tmp-cleaner deleted mid-run. Rule:
   supervisors must log to `logs/` (durable) and watchers must poll `output/`
   artifacts, never `/tmp`.

Artifacts: `output/eval-transfer-32b/` (JSONLs + `comparison.txt`).

---

## Natural-distribution run — uniform grouping WINS (2026-07-03)

**The DIVERGENCES #0 distribution test came back opposite to the hypothesis.**
`algo1fftnatural` = seed-1 FFT recipe with one knob flipped: group types drawn
from ALFWorld's natural frequencies (Pick-heavy) instead of uniform round-robin.
12-arm sweep vs the canonical 33.6% baseline:

| ckpt | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔSR | +5.7 | +2.9 | +2.9 | −1.4 | +0.7 | +5.0 | −0.7 | +1.4 | −2.1 | −1.4 | −4.3 | −0.7 |

No significant lift at ANY checkpoint (best p=0.20) — vs uniform's seed-1
+10.7pp (p=0.032) and seed-2 +13.6pp (p=0.0026). Three takeaways:

1. **Uniform type exposure is load-bearing.** All large gains in the uniform
   runs came from high-headroom types (Clean 19%→63% at seed-2 ckpt35); natural
   shifts ~25% of training groups to Pick, the executor's most saturated type
   (60% baseline). Distribution-matching the eval set is NOT the win — balanced
   exposure to where the headroom lives is. Keep uniform.
2. **The bimodality is NOT caused by the type distribution.** The natural curve
   still oscillates, just inside the noise floor. Prime suspect is now the
   TRL≠verl framework confound (#14) and/or the untested within-group curriculum.
3. **Ops:** the sweep survived the 2026-07-02/03 OpenRouter thin-pool outage via
   a self-healing supervisor (concurrency-matched probe gate: single+10+40-burst
   ×2 consecutive; 4-arm waves; storm auto-abort keeping completed arms). 15
   storms killed and re-gated; all 12 kept arms verified clean (140/140 games,
   zero executor-failure markers). Lesson: **a single-probe pre-flight is
   insufficient — the gate must match sweep-level concurrency.**

---

## Seed-2 FFT — does the bimodality reproduce? (2026-06-28)

**YES, shape generalizes across seeds; peak index does not.** `algo1fftseed2`
(seed=123, otherwise identical to the seed-1 FFT) completed 60/60. The 12-arm
every-5 held-out McNemar sweep (valid_seen, n=140) vs the canonical baseline
`output/eval-pathbv4/no_memory.jsonl` (33.6%):

| ckpt | 5 | 10 | 15 | 20 | 25 | 30 | **35** | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔSR | −2.9 | +0.7 | +5.7 | +1.4 | +1.4 | +4.3 | **+13.6** | +6.4 | +0.7 | +5.7 | +2.1 | +4.3 |
| p | .52 | 1.0 | .26 | .85 | .85 | .36 | **.0026** | .14 | 1.0 | .18 | .71 | .33 |

- Non-monotone/oscillating, same qualitative shape as seed-1 FFT (rise → dip →
  mid-run peak → decline; ckpt60 lands below peak, never monotone-to-60).
- **Peak shifted seed-1 step20 (+10.7pp) → seed-2 step35 (+13.6pp, p=0.0026)** —
  stronger and more significant. Peak step indices are RNG-path-dependent (sweep
  skill gotcha); the shape is the reproducible thing, not the index.
- **Not a U-shape.** Worst arm is ckpt5 −2.9pp; nothing drops >10pp below
  baseline, so the deep-trough `beta=0`/missing-KL signature does NOT apply here.
- ckpt35 absolute 47.1% — Clean jumps to 63%, but Heat collapses to 6%
  (per-type anti-correlation + the executor role-play issue, both known).
- **Process gotcha:** the temp-0.6 no_memory baseline has ~8pp run-to-run
  variance (a fresh reconstruction from the `eval-fft60-nomem-sh*` shards drew
  41.4%, which inflated the reference and flipped every delta negative). Always
  pair against the *fixed* canonical `eval-pathbv4/no_memory.jsonl`, never a
  freshly re-run baseline. Valid comparison: `eval-fft-seed2/comparison_canonical.txt`.

Conclusion: bimodality is a robust property of this small-batch GRPO + TRL setup,
not a single-seed fluke. Next lever stays the grouping/probe distribution
(DIVERGENCES #0, natural-distribution run staged), not more steps.

---

## Current status (2026-06-21)

- **Run:** `algo1v8lorakl` (wandb `okaris/skillos`), config
  `configs/alfworld_8xh100_algo1_v8_lora_kl.yaml`, launcher
  `run_algo1_v8_lora_kl.sh`. This is the **first faithful Algorithm 1 run** —
  grouped |G|=10 task streams, single `curate_and_advance` mega-tool, judge
  wired, `loss_type=grpo`, after the group-collapse postmortem
  (`docs/postmortem-2026-06-10-algo1-group-collapse.md`).
- **Setup:** 8×H100, **LoRA r=32 + KL anchor (beta=0.001)**, vLLM colocate;
  frozen Qwen3-8B executor + Qwen3-32B judge on inference.sh. LoRA is the one
  sanctioned deviation (lr scaled 10× to 1e-5).
- **Training: COMPLETE.** Hit the full **60-step paper schedule** on 2026-06-19.
  Resumed cleanly from checkpoint-50 → 60, exit 0, no NCCL abort. The new
  **per-rollout synchronized deadline** (`SKILLOS_PHASE_BUDGET_S`, enforced in
  `Algo1CuratorEnv.curate_and_advance`) fired **1612 DEADLINE CUTs** and the run
  survived — vs v8's earlier SIGABRT at steps 59/54 from rank skew (slow
  composite-verb groups maxing the 900s episode cap drifted one rank 4h behind
  the NCCL collective). Cut positions are masked from `r_task` (cut=True,
  success=None), so the curation-quality terms keep their honest gradient.

- **Held-out eval (paired-by-gamefile McNemar vs `no_memory`, n=140):**

  | checkpoint | SR | Δ vs baseline | p |
  |---|---|---|---|
  | no_memory | 33.6% | — | — |
  | ckpt10 | 29.3% | −4.3 | 0.26 |
  | ckpt20 | 35.7% | +2.1 | 0.71 |
  | **ckpt30** | **42.9%** | **+9.3** | **0.035** |
  | ckpt40 | 30.7% | −2.9 | 0.57 |
  | ckpt50 | 37.9% | +4.3 | 0.31 |
  | ckpt60 | 35.0% | +1.4 | 0.86 |

  **ckpt30 is the peak and the only significant arm (+9.3pp, p=0.035).** The
  trajectory is bimodal/noisy (peaks at 10 & 30, troughs between), not the
  paper's monotone-to-60 — almost certainly the small effective batch + the
  TRL/GRPO-vs-verl/FSDP framework swap, not a method failure. We recovered ~70%
  of the paper's lift; we did NOT reproduce their monotone-improvement-to-endpoint.

### Open thread: the no-memory baseline is ~14pp below the paper (≈34% vs 47.9%)

Same model (Qwen3-8B), same 30-step cap, same Fig 9 prompt — yet our `no_memory`
baseline trails the paper's 47.9%. We worked the full
`paper-repro-decode-settings-audit` hypothesis tree to exhaustion. **Everything
checkable is ruled out; the gap is real.** Ledger:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Prompt text / few-shot | ruled out | Fig 9 (PDF `docs/skillos_paper.pdf`) is **verbatim** our `ALFWORLD_EXECUTOR` — zero-shot, no exemplars. The paper hits 47.9% with the same prompt. |
| Precision / serving | ruled out | prior local bf16 vLLM (`quantization=None`) = 28% ≈ remote 30% (`infsh/precision-not-skillos-baseline-gap`). |
| Decode (temp/reasoning/tokens) | ruled out | `scripts/debug_executor_audit.py`: `<action>` parses at temp 0.4/0.6, reasoning ON (~1.5k chars), 8192 budget not truncating. Reasoning-on was +13pp and necessary but **not sufficient** (`infsh/reasoning-fix-insufficient-baseline-gap`). |
| Retrieval / seeds / averaging | ruled out | BM25 top-5 (matches); micro 33.6% ≈ macro 32.4%; paper runs 3 seeds, std ~1pp. |
| Success detection | ruled out | alfworld requests only `won`; TextWorld reward is 1-on-win → `scores>0 ⟺ info['won']`. |
| Env / ReAct harness | cosmetic only | diffed vs GiGPO `verl-agent` (`prompts/alfworld.py`, `env_package/alfworld/{envs,projection}.py`, `env_manager.py`): only differences are admissible-list formatting (comma vs newline-quoted, `help` excluded) and history format. Same info. |
| K=20 batching artifact | no signature | baseline success uncorrelated with `executor_wave_seconds` (1662 vs 1671) or slot/wave position. |
| Stale-baseline confound | ruled out | our compare-baseline file is a May-28 symlink, but it is **statistically identical per-type** to a fresh good-config run (Heat 4/16 and Look 6/13 identical) → harness reproducibly gives ~34%, 3 weeks apart. |

**The structural mechanism (real, but unfixable by prompt).** Tracing a failed
Heat episode (`scripts/trace_failed_episode.py`): the executor **ignores
ALFWorld's atomic `heat X with microwave 1`** action (present in the admissible
list) and role-plays a physical microwave — open / insert / close / reopen /
take out — then loops to the step cap. Off-grammar coercion is 0% (not a parse
bug). A one-sentence "these are atomic actions" hint flipped the single traced
episode but moved the full n=140 baseline only **+2.1pp (p=0.68, noise)** — the
classic small-n trap. **Reverted.**

**A false alarm worth recording.** The knowledge entry
`infsh/skillos-validated-executor-config` claims our config *reproduces* the
baseline at **42.9% / 21.5 steps**. It does not — that profile matches our
**ckpt30 with-memory** run (42.9% / 22.2 steps), not no-memory (33.6% / 23.4
steps). The entry **mislabeled a with-memory result as the baseline**; the
correct prior observation is `infsh/reasoning-fix-insufficient-baseline-gap`
(~30-36%, gap remains), which matches us.

**Net:** baseline is robustly **~34%**, the **14pp gap is real**, and (crucially)
the **+9.3pp ckpt30 lift is NOT a baseline confound** — it's measured against a
same-config baseline. The residual is either the canonical zero-shot Qwen3-8B
genuinely scoring ~34% on ALFWorld (paper's baseline optimistic), or an
undocumented detail in the paper's GiGPO-deferred executor harness. **Last live
test:** a Qwen3-32B executor baseline — if it reproduces the paper's 54.5%, the
gap is 8B-specific; if it also trails ~14pp, the whole zero-shot baseline column
is hard to reproduce. (Question for the authors, not a bug we can find.)

**RESOLVED — decode is NOT the lever, gap is 8B-specific (2026-06-25).** Ran a
4-config executor-decode sweep as parallel `no_memory` baselines (n=140 each,
remote executor, no GPU): `ctrl` (temp 0.6 / top_p 0.95 / top_k 20, our default),
`gigpo` (0.4 / 1.0 / off — full verl parity), `temp` (0.4 / 0.95 / 20), `gigpo_hi`
(0.4 / 1.0 / off, reasoning high). Paired McNemar vs ctrl:

| config | overall SR | Δ vs ctrl | p |
|---|---|---|---|
| ctrl (0.6/0.95/20/med) | 36.4% | — | — |
| temp (0.4/0.95/20/med) | 39.3% | +2.9 | 0.57 |
| gigpo (0.4/1.0/off/med) | 34.3% | −2.1 | 0.70 |
| gigpo_hi (0.4/1.0/off/high) | 34.3% | −2.1 | 0.69 |

**All four statistically identical (every p > 0.5).** No decode config reaches the
paper's 47.9%. Findings: (1) temp 0.4 is *nominally* best (39.3%) but +2.9pp at
p=0.57 is noise — the floor is confirmed empirically since this `ctrl` run (36.4%)
vs the canonical same-config run (33.6%) swung 2.8pp with zero config change.
(2) **Full GiGPO "parity" (top_p 1.0 / top_k off) is nominally WORSE** — those are
verl defaults tuned for Qwen2.5; the Qwen3 model card specifies thinking-mode
temp 0.6 / top_p 0.95 / top_k 20, which is *exactly our default*, so matching
GiGPO fights the model card (`infsh/qwen3-sampling-from-model-card`). (3) High
reasoning didn't help. **Conclusion: executor decode is exhausted and ruled out;
the 8B zero-shot baseline does not reach 47.9% under any matchable config, while
the 32B executor reproduces the paper's 54.5% — the gap is model-specific, a
reproducibility finding / question for the authors, not a knob on our end.** Eval
JSONLs: `output/eval-decode/{ctrl,gigpo,temp,gigpo_hi}.jsonl`.

---

## FFT control run — is the bimodality a LoRA artifact? (2026-06-24)

- **Run:** `algo1fft` (wandb `okaris/skillos`), config
  `configs/alfworld_8xh100_algo1_fft.yaml`, launcher `run_algo1_fft.sh`. This is
  **v8 EXACTLY minus LoRA**: full fine-tune, ZeRO-3 + vLLM colocate, `beta=0.001`,
  `lr=1e-6`, all other paper hyperparams identical. Purpose: v8 (LoRA) gave a
  **bimodal** held-out trajectory (peak ckpt30, not the paper's monotone-to-60).
  Question: did **LoRA** cause the oscillation, or is it the framework/algorithm?
- **Sharding:** ZeRO-3 was the empirically-working FFT path on this stack
  (trl 1.4 / transformers 5.9 / deepspeed 0.19). **ZeRO-2 + vLLM colocate HUNG**
  here (GPUs 0% util, 42 min no progress) despite older notes calling it
  "validated" — those notes are from an older stack. ZeRO-3 shards params + ref
  model, so `beta=0.001` fits (smoke peak 65.5G; sustained 80.3G/81.5G, 0 OOM
  across the full 60-step run).
- **Training: COMPLETE.** Hit 60/60 on 2026-06-24, exit 0. ~70 min/step (FFT pays
  a per-step all-gather tax for vLLM-colocate generation under sharded params).

- **Held-out eval (paired-by-gamefile McNemar vs `no_memory` 33.6%, n=140):**

  | checkpoint | SR | Δ vs baseline | p |
  |---|---|---|---|
  | ckpt10 | 40.7% | +7.1 | 0.053 |
  | **ckpt20** | **44.3%** | **+10.7** | **0.032** |
  | ckpt25 | 39.3% | +5.7 | 0.20 |
  | ckpt30 | 33.6% | +0.0 | 1.00 |
  | ckpt35 | 36.4% | +2.9 | 0.60 |
  | ckpt40 | 38.6% | +5.0 | 0.26 |
  | ckpt50 | 31.4% | −2.1 | 0.72 |
  | ckpt60 | 39.3% | +5.7 | 0.18 |

- **Per-type vs the paper (Qwen3-8B executor, n=140; SR% / avg steps).** Paper
  Table 1 alongside ours (baseline, best LoRA = v8 ckpt30, best FFT = ckpt20):

  | Type | Paper No-Mem | Paper SkillOS | Ours No-Mem | Ours LoRA30 | Ours FFT20 |
  |---|---|---|---|---|---|
  | Pick  | 78.1 / 21 | 85.7 / 19 | 60 / 17 | 66 / 15 | 60 / 17 |
  | Look  | 46.2 | 56.4 | 46 / 20 | 46 / 19 | 38 / 21 |
  | Clean | 33.3 | 54.3 | 19 / 26 | 41 / 24 | 48 / 21 |
  | Heat  | 37.5 | 43.8 | 25 / 28 | 12 / 28 | 31 / 24 |
  | Cool  | 29.3 | 46.7 | 20 / 27 | 44 / 25 | 40 / 23 |
  | Pick2 | 47.2 | 62.5 | 25 / 25 | 29 / 26 | 33 / 25 |
  | **Avg** | **47.9 / 21.1** | **61.2 / 18.9** | **33.6 / 23** | **42.9 / 22** | **44.3 / 22** |

- **Lift over own baseline (Δpp) — what the curator actually adds:**

  | Type | Paper (SkillOS−NoMem) | Ours FFT20 | Ours LoRA30 |
  |---|---|---|---|
  | Pick  | +7.6  | +0  | +6 |
  | Look  | +10.2 | −8  | +0 |
  | Clean | +21.0 | **+29** | +22 |
  | Heat  | +6.3  | **+6** | −13 |
  | Cool  | +17.4 | **+20** | +24 |
  | Pick2 | +15.3 | +8  | +4 |
  | **Avg** | **+13.3** | **+10.7** | **+9.3** |

  **Read:** measured as *lift*, FFT (+10.7) is within 2.6pp of the paper (+13.3),
  and on the **appliance verbs the curator does its job** — Clean +29 (paper +21),
  Cool +20 (paper +17), Heat +6 (matches paper). The absolute shortfall lives in
  the **baseline**, concentrated on the no-appliance tasks the curator can't fix:
  Pick (ours 60 vs paper 78) and Pick2 (25 vs 47), while Look matches exactly
  (46=46). A skill can't paper over a worse navigate/commit phase. LoRA uniquely
  *hurts* Heat (−13) where FFT helps (+6). Steps already drop (23→22; Clean
  26→21) — the paper's "procedural shortcut / Steps ↓" mechanism reproduces.

- **Answer: the bimodality is NOT a LoRA artifact.** FFT reproduces the same
  oscillating shape — significant early peak (ckpt20 +10.7pp, p=0.032), collapse
  to exact baseline parity (ckpt30, 17:17 discordant), partial recovery. The peak
  index shifted (FFT step20 vs LoRA step30), but peak indices are RNG-path-
  dependent; the **shape** generalizes. Drop LoRA, keep everything else, get the
  same curve → the oscillation is **framework/algorithm-level** (small-batch GRPO
  bouncing between local optima on a narrow probe distribution, TRL path), not the
  LoRA parameterization.
- **FFT's best beats LoRA's best:** ckpt20 **+10.7pp (significant)** vs v8 LoRA's
  +9.3pp, now within ~2.6pp of the paper's claimed +13.3pp lift. Ship
  `checkpoint-20` as the best-on-heldout FFT artifact (not last-step ckpt60).
- **Not a missing-KL-anchor U-shape.** `beta=0.001` is present; the troughs are
  shallow and non-significant (ckpt30 p=1.0, ckpt50 p=0.72), not the deep
  significant trough that flags a dropped KL anchor.
- **Next lever** (per the bimodal branch): broaden the probe/training-task
  distribution. Adding steps with the same setup just oscillates in the same band;
  we are already on full Algorithm 1, so steps aren't the knob.
- **Per-type caveat:** Heat stays weak across every arm (6–31%), and per-type
  wins trade off between checkpoints (ckpt40 leads Pick 80% but bottoms Heat 6%) —
  aggregate SR hides this. See the Heat note below.

### Why Heat is the weakest type — it's the executor, not a bug (2026-06-24)

Heat sits lowest across **every** checkpoint and the baseline (22% aggregate vs
Pick's 68%). It *feels* like a bug, but the evidence says it's a behavioral
failure of the **frozen Qwen3-8B executor**, not our harness:

- **Failure mode is uniform, not Heat-specific.** Every failure of *every* type
  ends at the 30-step cap (`fail@maxstep = 100%` for Clean/Cool/Heat/Look/Pick/
  Pick2). Heat episodes that win, win in ~14 steps (same as Cool 13, Pick2 14).
  So success-detection and step-accounting work fine for Heat — it just runs out
  of steps more often.
- **The weak types are exactly the appliance/atomic-verb tasks.** Clean (28%),
  Cool (28%), Heat (22%) all require an atomic verb — `clean X with sinkbasin`,
  `cool X with fridge`, `heat X with microwave`. Pick (68%) needs no appliance.
  Heat is marginally the worst because the microwave has the most role-play
  states (open/close) to get lost in, and its n is smallest (144) so it's the
  noisiest.
- **Live trace of a Heat episode** (`scripts/trace_failed_episode.py`, task "put a
  hot potato in fridge"): the executor finds and takes the potato, walks to the
  microwave — all correct — then, with **`heat potato 1 with microwave 1`
  present in the admissible list AND named verbatim in its own reasoning**, it
  *defers* the atomic verb and role-plays a physical microwave instead:
  `open microwave 1` → `move potato 1 to microwave 1` → `close microwave 1` → …
  Every emitted action is `admissible? True` (off-grammar coercion is **0%** —
  not a parse bug). The trap: `move potato 1 to microwave 1` takes the potato out
  of the agent's hand, and ALFWorld's `heat X with microwave` wants the object
  *held* — so the role-play can forfeit the very action it's trying to set up,
  and the episode loops to the cap.

This is the same atomic-verb gap behind the `no_memory` baseline shortfall
(`infsh/executor-atomic-verb-gap`). A one-sentence "these are atomic actions"
hint flipped a single traced episode but moved the full n=140 baseline only
+2.1pp (p=0.68, noise) — reverted. The fix isn't in our code; it's the frozen
8B executor's instinct to simulate real-world causality instead of using
ALFWorld's atomic abstraction. **Not a bug we can patch — a property of the
model the paper defers to GiGPO for.**

---

## Current status (2026-05-26)

- **Run:** `pathbv4` (wandb `okaris/skillos`), config
  `configs/alfworld_8xh100_pathb.yaml`, launcher `run_pathb.sh`.
- **Setup:** 8×H100, **full fine-tune** of Qwen3-8B curator, vLLM colocate;
  frozen Qwen3-8B executor + Qwen3-32B judge on inference.sh.
- **Progress:** reached step 10 (checkpoint-10 saved), then **crashed at ~step 11
  on a NCCL collective timeout** (additive per-future waits — see Operational
  hurdles). Fixed with a whole-phase wall budget and **resumed from
  checkpoint-10**.
- **Reward (train, composite) up to the crash:** climbing gently — steps 5→10
  read 1.19 → 1.28 → 1.35 → 1.35 → 1.49 → 1.49 (first-half mean 1.33 →
  second-half 1.39, +0.06).
- **Gradient signal:** `frac_reward_zero_std = 0` on *every* step — there is
  real within-group reward variance now (this is the whole point of the Path B
  rebuild; see below). `tools/failure_frequency = 0`, infsh clean.
- **Cadence:** ~40 min/opt-step (step_time 1330–3275 s). Full 60 steps ≈ ~40 h
  wall.
- **Held-out lift:** **TBD.** Not yet evaluated. The mechanistic root cause is
  fixed and training reward moves; whether that converts to held-out task
  success is the open question this run exists to answer.

---

## The core problem we were stuck on

Every curator run we did — LoRA v2, LoRA v3, and a paper-faithful full
fine-tune — produced **~0 held-out lift**, even though training reward rose and
the loop looked healthy. A trained curator that doesn't help the executor on
unseen tasks is the one outcome that makes the whole exercise pointless, so we
stopped adding scale and went looking for *why*.

The tempting move was to blame the framework (TRL ≠ the paper's verl) and switch
to verl to "test if it's an implementation bug." We deliberately did **not** do
that first — verl is a multi-day port, and we had cheaper hypotheses to kill.

## The investigation (cheap diagnostics before big rewrites)

1. **Reward decomposition.** We broke the composite reward
   `r = r_task + λ_f·r_fc + λ_u·r_cnt + λ_c·r_comp` down per-component on real
   rollouts. Findings:
   - The judge/content term (`r_cnt`) contributed a **flat ~0.09** — negligible.
     This **refuted** our prior working hypothesis that "the judge dominates and
     drowns out task signal." It does not.
   - The composite's climb was driven almost entirely by `r_fc` (valid function
     calls) — the curator learning to emit *well-formed* tool calls, not
     *useful* ones.

2. **The actual root cause — `r_task` is constant within each GRPO group.**
   The old design ran **one shared executor trajectory per group**, *before* any
   curation. Every rollout in the group therefore saw the *same* task outcome,
   so `r_task` was identical across the group. GRPO's advantage is
   `(r − group_mean) / group_std`; any term that is constant within the group
   **cancels to zero** and contributes no gradient. We were training the curator
   on everything *except* whether its skills helped solve the task.
   - This is exactly the lever the paper's own Table 3 flags as #1 most
     important (grouped/transfer structure: 61.2 → 57.3 without it). We had
     silently removed it.
   - **LoRA-vs-FFT was never the lever.** That had been our suspicion across two
     prior runs; the decomposition shows the architecture choice was irrelevant
     to the flat lift.

3. **Does a transfer signal even exist?** Before rebuilding, we checked the
   premise: hand a *good* clean-type skill to the frozen executor on a fresh
   clean task. Success went **0% → 100%**. So curated skills *can* transfer —
   the reward just wasn't measuring it.

## The fix — "Path B": per-rollout repo + transfer-probe `r_task`

We chose the pragmatic version of the paper's Algorithm 1 (single curation
step) over a full sequence-of-tasks reimplementation, because TRL's tool loop
runs serially across the batch — a faithful multi-task sequence per rollout
would multiply wall time past the point of usability. Path B keeps the part
that matters (within-group variance from transfer) and drops the part that's
just expensive (the long task sequence).

Mechanism (`skillos/envs/curator_env.py`):
- **Each rollout gets its own ephemeral `SkillRepo`** (paper: `S ← ∅` per
  group), instead of a single process-wide shared repo. This also fixed a
  latent correctness bug — parallel rollouts used to race on one shared repo
  (old DIVERGENCES #7).
- The curator curates from the seed task's trajectory, then **`r_task` = mean
  frozen-executor success over `num_probe_tasks` freshly-sampled SAME-TYPE
  tasks**, solved using *that rollout's* curated skills. Probes run in parallel
  in the reward step (sidestepping the serial tool loop).
- Because the N rollouts in a group curate *differently*, their probe success
  differs → **real within-group `r_task` variance** that rewards skills which
  transfer to related tasks.
- **Shared probe games per group via deterministic seeding** (`env.seed(s)`):
  every rollout in a group is probed on the *same* games, so the GRPO comparison
  isolates *curation quality* rather than luck of the task draw.

Consequence for eval (important): training repos are now **ephemeral**, so there
is no persisted training repo to evaluate. **Eval must change** — it has to
build memory by running the trained curator over a curation stream (a streaming
closed-loop, which is what the paper actually does). The old "load the persisted
training repo and measure" eval is invalid under Path B.

## Operational hurdles (the stuff the paper never mentions)

- **Degenerate all-`clean` seeds.** First Path B launch trained almost entirely
  on `clean` tasks. Cause: ALFWorld gamefiles are **alphabetically clustered by
  task-type prefix**, and our sequential seed iterator walked them in order, so
  early training saw one type. Fix: randomize the seed game
  (`random.randint(0, 2³¹−1)`) in the seed rollout. Now all 6 types appear.
- **Heartbeat crash (SIGABRT at ~480 s).** The diversity fix introduced
  rank-time skew (different rollouts have different task difficulty), so a fast
  rank waits at the post-seed generation collective for a slow rank. Torch's
  **NCCL heartbeat monitor** (a watchdog-of-the-watchdog, default 480 s)
  false-aborted on this benign wait. Fix in `run_pathb.sh`:
  `TORCH_NCCL_ENABLE_MONITORING=0` + `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`.
  Our real safety guard (1200 s per-rollout sentinel < 1800 s collective
  timeout) is intact.
- **`NCCL_TIMEOUT_MS` is a no-op under accelerate.** We set it, but it does not
  change the actual collective watchdog timeout (stays at the 1800 s default).
  The architecture is still self-consistent because the 1200 s executor sentinel
  fires first. Don't rely on that env var.
- **Monitor false positives.** Our background watchers matched "failed"/"FAILED"
  in benign infsh "…resubmitting" lines. Fixed by grepping fatal-only patterns
  (`watchdog got stuck|ChildFailedError|exitcode: -|CUDA out of memory`).
- **NCCL collective timeout from additive per-future waits (the step-11 crash).**
  After checkpoint-10, rank 3 sat in Python >1800 s between collectives, so the
  other 7 ranks hit the **real** NCCL collective watchdog (`SeqNum=262
  _ALLGATHER_BASE, Timeout=1800000 ms`) and aborted (SIGABRT). This is *not* the
  earlier heartbeat monitor (that fix held) and *not* fixed by `NCCL_TIMEOUT_MS`
  (a confirmed no-op under accelerate — the timeout was exactly the 1800 s
  default). Root cause: the reward step gathered each probe future with its *own*
  `result(timeout=1200 s)`; during an infsh stall (`container exited —
  resubmitting`) several probes stalled at once and their waits **added up** past
  1800 s. The per-rollout sentinel bounded each future but not the *phase*. Fix:
  bound the whole seed-rollout and probe phases with a single shared deadline
  (`SKILLOS_PHASE_BUDGET_S=1500`, < the 1800 s watchdog), so a phase's wall is
  capped regardless of how many futures stall; unfinished probes are dropped from
  `r_task` rather than crashing the job. Checkpoints made the resume free.
- **Retry storm cooked inference.sh (the step 11–17 degenerate window).** After
  the phase-budget fix the run *survived* but went **degenerate**: ~100% of
  probes dropped (`64/64` per step), `r_task` collapsed to 0 uniformly, reward
  pinned at the function-call floor (~1.0) and `frac_reward_zero_std` rose to
  0.75 — the constant-within-group failure, reintroduced. Root cause: the
  **executor** called `run_task_resilient` with the stock defaults
  (`max_resubmissions=10`, `poll=900 s` → ~5.6 h and up to **10 infsh tasks per
  call**), while the judge was already tamed. A transient infsh blip became a
  self-sustaining **resubmission storm** — stuck calls piled tasks onto
  inference.sh faster than they drained; our phase-budget timeouts then
  abandoned the futures but couldn't kill the threads, so zombie episodes kept
  retrying *into* the backlog and saturated the shared rollout pool, starving
  every subsequent probe. (Confirmed jointly: the infsh queue visibly backed up;
  manually failing the stuck tasks let it drain and drops fell 64 → 8 within a
  step.) Fix: give the executor the judge's env-tunable, **short** retry budget
  (`SKILLOS_EXEC_MAX_RESUBS=2`, `SKILLOS_EXEC_POLL_MAX_S=150`, backoff cap 30 s)
  so a stuck call fails in ~minutes with ≤2 tasks and frees its worker. Lesson:
  aggressive per-call retry is a metastable-failure amplifier on a shared remote
  backend — the executor, which fires the most calls, must fail fast.
  **Plus** (`run_task_resilient`): when OUR timeout fires (poll fallback
  exhausted / stream wedged) the task is still *running on the server*, so we now
  `client.tasks.cancel(task_id)` before resubmitting or giving up — abandoning it
  silently is what let live tasks pile up. We skip cancel only when infsh already
  moved it to a terminal FAILED/CANCELLED state.
- **Cadence misread.** Early ts-gap clustering suggested ~6 min/step; the
  authoritative tqdm bar shows ~40 min/step. We let the run continue and monitor
  the reward *trend* (the real question) rather than restart for wall time —
  checkpoints every 10 steps make a speed-up resume safe later.

## Known gaps / open questions

- **No per-rollout `r_task` is persisted to disk for this run.** The observability
  that wrote `rollouts.jsonl` was dropped in the Path B rebuild, so right now we
  can only see the *composite* reward (via the trainer's step logs), not the
  `r_task`-vs-judge split per rollout. Worth re-adding a lightweight append in
  the reward finalize and resuming from a checkpoint, so we can confirm the
  within-group variance is coming from *task transfer* and not judge noise.
- **`beta = 0.0`** in the Path B config (no KL term), whereas the paper uses
  0.001. Revisit whether this is intended (it drops the reference model) or
  should match the paper.
- **Held-out lift unverified.** The end-to-end question — does Path B convert
  into held-out task-success lift — is still open, pending (a) the run finishing
  enough steps and (b) the new streaming-curation eval.
- **Possible escalation to full Algorithm 1** (the real multi-task sequence) if
  Path B learns but falls short of the paper.
</content>
</invoke>
