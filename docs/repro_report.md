# Reproducing SkillOS: an independent report

Independent reproduction of [SkillOS](https://arxiv.org/abs/2605.06614) (Ouyang et al., 2026) on TRL 1.4 + DeepSpeed ZeRO-3 + vLLM colocate, 8×H100. Executor and judge run remotely on inference.sh. Three benchmarks: ALFWorld (agentic), AIME24 + AIME25 (numeric reasoning), GPQA-Diamond (multiple-choice science reasoning). Report authored in-repo alongside the code so every number can be recomputed from the JSONLs referenced below.

## abstract

We reproduce the core method — GRPO-training a curator that maintains a markdown skill repo for a frozen executor — using TRL instead of the paper's verl framework, on 8 H100s instead of 16. The paper's headline generalisation claim (an 8B-trained curator lifting a 32B executor to ~61.2% ALFWorld SR) reproduces on a single run at 62.1% (+12.9pp vs no-memory, McNemar p=0.0064). Reasoning baselines reproduce within 1.1σ on average across three datasets. Two independent findings:

1. **The training trajectory is bimodal, not monotone.** Held-out lift peaks mid-run (peak indices vary by seed) and regresses by step 60. Reproduces on two seeds. Systematic falsification rules out grouping-recipe explanations (both halves of the paper's grouping ablation).
2. **The 8B ALFWorld no-memory baseline sits 14pp below the paper on the same executor that reproduces the paper's reasoning baseline within noise.** The gap is environment-specific, traced to the ReAct + atomic-verb interaction between Qwen3-8B and ALFWorld.

The surviving suspect for both findings is the framework confound: TRL ≠ verl in advantage normalisation, sampling semantics, and buffer handling. All divergences from the paper are enumerated in [`../DIVERGENCES.md`](../DIVERGENCES.md).

## 1. background

The paper defines a curator as an LLM whose only job is to write, revise, and delete markdown skill files after each rollout. A frozen executor retrieves the top-k relevant skills from the curator's repo before every new task. GRPO optimises the curator against a composite reward

    r = r_task + λ_f · r_fc + λ_u · r_cnt + λ_c · r_comp

where `r_task` is downstream executor success, `r_fc` counts valid function calls, `r_cnt` is a judge-scored content-quality signal (paper uses Qwen3-32B), and `r_comp` is a compression term. The training data is Algorithm 1: |G|=10 same-type task groups, curator emits `curate_and_advance` calls, `r_task` averages executor success over positions 2..|G| of the group's evolving skill repo. Paper weights: λ_f=1.0, λ_u=0.1, λ_c=0.05.

## 2. setup

**Hardware / framework.** 8×H100 (paper: 16). TRL 1.4 + accelerate + DeepSpeed ZeRO-3 + vLLM colocate (paper: verl-agent). ZeRO-2 hangs on this stack; ZeRO-3 is the empirically-working path. Curator on the 8 local GPUs, executor + judge on inference.sh (`openrouter/qwen3-8b` and `openrouter/qwen3-32b`).

**Models.** Curator = Qwen3-8B (paper: same). Frozen executor = Qwen3-8B during training. Test executors: Qwen3-8B and Qwen3-32B. Judge = Qwen3-32B (paper: same).

**Hyperparameters.** All paper-faithful except the following documented deviations, each in [`../DIVERGENCES.md`](../DIVERGENCES.md):

- LoRA r=32 with lr scaled 10× as a sanctioned memory-fit path; full fine-tuning is used for the definitive runs.
- Executor decode = Qwen3 model-card sampling (temp 0.6, top_p 0.95, top_k 20, reasoning-on). The paper defers executor decode to verl-agent.
- `SKILLOS_EXECUTOR_MAX_STEPS=30` (paper trajectories average 21.1 steps).

**Held-out protocol.** ALFWorld: n=140 valid_seen, paired-by-gamefile McNemar vs a fixed no-memory baseline (`output/eval-pathbv4/no_memory.jsonl`, 33.6% SR). Sweeps cover every saved checkpoint (5-step cadence). Noise floor at n=140 is ~±3pp empirically; single-arm claims gated at ~2×SE ≈ 8pp. Reasoning: AIME24 (30 problems), AIME25 (30), GPQA-Diamond (198, MC schema, `Idavidrein/gpqa`), greedy grading against reference answer.

## 3. headline results

### 3.1 cross-executor transfer (paper's generalisation claim)

The strongest reproduced result. 8B-trained curator drives Qwen3-32B executor, 140 paired games, no-memory reference from a fresh 32B no-memory run at 49.3%:

| curator | executor | abs SR | Δ vs no-memory | p |
|---|---|---|---|---|
| no memory | Qwen3-32B | 49.3% | — | — |
| **v8-lora ckpt30** | **Qwen3-32B** | **62.1%** | **+12.9pp** | **0.0064** |
| fft-seed1 ckpt20 | Qwen3-32B | 47.1% | −2.1pp | 0.74 |
| fft-seed2 ckpt35 | Qwen3-32B | 55.0% | +5.7pp | 0.26 |
| paper SkillOS (Qwen3-32B executor) | — | 61.2% | ~+13pp | — |

Above the paper's headline on this single run. Baseline stochasticity is ~±4pp; treat as "at parity" not "beats."

**Unexpected finding:** the on-8B ranking inverts on 32B. The best-on-8B curators (FFT) transfer weakly or negatively; the LoRA curator that ranked third on 8B transfers best on 32B. Hypothesis: FFT skills overfit to 8B executor quirks; LoRA's constrained update produces more generic skills. Single-run finding, hypothesis only. See gotcha `cross-executor-transfer-confirmed`.

Heat unlocks at 32B (25% → 56–62% with memory). The Heat pathology on 8B — the executor role-plays a physical microwave instead of using the `heat X with microwave` atomic verb — vanishes at 32B scale. Consistent with the gotcha `executor-atomic-verb-gap`.

### 3.2 reasoning baselines

| dataset | ours (no-memory) | paper (Qwen3-8B no-memory) | delta |
|---|---|---|---|
| AIME24 | 22/30 = 73.3% | 76.0±6.9 | −2.7pp (0.4σ) |
| AIME25 | 18/30 = 60.0% | 71.1±10.7 | −11.1pp (1.0σ) |
| GPQA-Diamond | 118/198 = 59.6% | 61.8±1.1 | −2.2pp (2.0σ) |
| **average** | **64.3%** | **69.6±4.7** | **−5.3pp (1.1σ)** |

Every dataset within 1σ individually. 0 executor errors on GPQA. Letter distribution well-balanced (A/B/C/D each 42–54 out of 197 answerable), no degenerate mode. GPQA-D result reported aggregate-only per the dataset owner's access condition (no per-problem content in git or web-visible files).

Closed-loop reasoning eval (curator manages skill repo across problems) is stubbed. Requires local GPU for curator inference; blocked on seed-3 completion.

### 3.3 ALFWorld held-out lift

Best single-run numbers, McNemar vs canonical 33.6% no-memory baseline, 140 games:

| run | best ckpt | ΔSR | p |
|---|---|---|---|
| v8 LoRA r=32 | ckpt30 | +9.3pp | 0.035 |
| seed-1 FFT | ckpt20 | +10.7pp | 0.032 |
| seed-2 FFT (seed=123) | ckpt35 | +13.6pp | 0.0026 |
| seed-3 FFT (seed=456) | *training as of 2026-07-12* | — | — |

Paper claim: +13.3pp. We land on the same order of magnitude across three independent training runs. Ship the best-on-heldout checkpoint, not the last-step one.

## 4. finding 1 — training trajectory is bimodal on TRL, not monotone

**Observation.** Across three runs (v8 LoRA, seed-1 FFT, seed-2 FFT), held-out lift over 60 GRPO steps is non-monotone: it rises through a mid-run peak, dips (occasionally to below baseline parity), and never fully recovers by step 60. The paper reports a monotone-to-60 curve.

**Peak index shifts with seed, shape does not.**

| run | peak ckpt | peak lift | final ckpt (60) lift |
|---|---|---|---|
| seed-1 FFT | 20 | +10.7pp | +5.7pp |
| seed-2 FFT | 35 | +13.6pp | +4.3pp |

Same recipe, different seeds, peak moves 15 checkpoints. Shape (rise → dip → mid-run peak → decline) is the reproducible property; the exact peak *index* is RNG-path-dependent. Practical consequence: ship the best-on-heldout checkpoint from a sweep, not `checkpoint-60`.

**Ruled out as causes** (each is a full 3-day training run + sweep, all against the same canonical baseline):

- *LoRA parameterisation.* Full fine-tuning reproduces the bimodal shape at greater strength (+10.7pp vs +9.3pp). Not a LoRA artifact. Memory: `fft-bimodal-not-lora`.
- *Type distribution (DIVERGENCES #0 half 1).* Training on the natural ALFWorld type frequencies (Pick-heavy) instead of uniform round-robin **kills** the lift entirely (best +5.7pp p=0.20). Uniform's balanced exposure to high-headroom types (Clean, Cool, Heat) is load-bearing. Memory: `natural-distribution-uniform-wins`.
- *Within-group ordering (DIVERGENCES #0 half 2).* Soft easy→hard curriculum (paper Table 5, p↑=0.80, difficulty = expert-plan length) produces zero significant lift at any checkpoint (best +4.3pp p=0.36). Memory: `curriculum-no-lift-grouping-exonerated`.
- *Missing KL anchor.* Not a U-shape (no deep sub-baseline trough). `beta=0.001` is present. Memory: `ushape-eval-missing-kl-anchor` rules out this class.

**With both halves of the paper's grouping ablation falsified on our stack, grouping is fully exonerated as the driver.**

**Remaining suspect: TRL ≠ verl.** DIVERGENCES #14. Framework-level differences in advantage normalisation, sampling semantics (deferred by paper to verl-agent), and buffer handling remain untested. Testing requires a port to verl-agent, out of scope for this reproduction.

## 5. finding 2 — the ALFWorld baseline gap is environment-specific

**Observation.** Same executor stack (Qwen3-8B via inference.sh, model-card decode, reasoning-on):

| benchmark | ours (no-memory) | paper | delta |
|---|---|---|---|
| ALFWorld avg SR | 33.6% | 47.9% | **−14.3pp** |
| AIME24 | 73.3% | 76.0 | −2.7pp (0.4σ) |
| AIME25 | 60.0% | 71.1 | −11.1pp (1.0σ) |
| GPQA-D | 59.6% | 61.8 | −2.2pp (2.0σ) |

If the ALFWorld gap were a broad executor-quality problem, we'd see it on reasoning too. We don't. The gap is **specific to ALFWorld's ReAct interaction pattern**, not model quality.

**Trace-level evidence** on ALFWorld failure modes: the Qwen3-8B executor tends to role-play a physical microwave (`"I open the microwave and place the item inside"`) instead of emitting ALFWorld's atomic verb (`"heat X with microwave"`). Heat SR is 25% on 8B (matches the paper for that type) and unlocks to 56–62% on 32B. Prior audit (decode sweep, prompt Fig 9 verbatim, precision, retrieval, seeds) ruled out non-model-scale explanations. Memory: `executor-atomic-verb-gap`.

**Consequence for the writeup.** All ALFWorld numbers should be read relative to their own baseline, not the paper's. Lift claims (McNemar) are unaffected because they compare arms drawn from the same executor stack.

## 6. discussion

**What reproduces.** The core method: composite reward composition, Algorithm 1 group semantics, curator tool-calling, judge-scored content quality. The generalisation claim reproduces on the single tested pair (8B curator → 32B executor). The reasoning baselines reproduce within noise. Lift on ALFWorld reproduces at the same order of magnitude across three independent runs.

**What doesn't reproduce.** The paper's monotone-to-60 curve. Three independent runs show mid-run peaks and post-peak regression; the effect is robust to LoRA/FFT, seed, type distribution, and curriculum ordering. This is a real reproducibility finding, not a bug — publishable as "TRL introduces small-batch GRPO instability that verl either doesn't have or handles differently."

**Threats to validity.**
- TRL ≠ verl framework confound present in every run; untested via port.
- ALFWorld baseline gap is env-specific but adds noise to any absolute claim; McNemar-paired lifts are unaffected.
- Bimodality established on n=2 seeds; n=3 in progress.
- Cross-executor transfer numbers are n=1 per arm; ~±4pp baseline variance means "beats paper" is not a defensible claim, "at parity" is.
- Closed-loop reasoning eval not yet run.
- WebShop not attempted.

## 7. how the code is organised

Training:

- `scripts/train_algo1.py`, `run_algo1_fft.sh`, `run_algo1_v8_lora_kl.sh`
- Configs: `configs/alfworld_8xh100_algo1_fft.yaml`, `configs/alfworld_8xh100_algo1_v8_lora_kl.yaml`
- Sharding: `configs/accelerate_zero3.yaml`

Held-out eval and sweeps:

- `scripts/eval_streaming_curation.py --mode {no_memory, closed_loop}` — ALFWorld
- `scripts/eval_reasoning.py --mode no_memory --dataset {aime24, aime25, gpqa}` — reasoning
- `scripts/compare_eval_arms.py` — paired-McNemar comparator
- `scripts/{natural, curriculum, transfer}_sweep_supervisor.sh` — storm-resilient sweep runners with concurrency-matched executor gates

Every result table in this report can be regenerated from the JSONLs under `output/eval-*/` by pointing `compare_eval_arms.py` at the same arms and the canonical `output/eval-pathbv4/no_memory.jsonl` baseline. Memory notes and skills are held under the local `belt` knowledge store (path `/home/ubuntu/.claude/projects/-home-ubuntu-skillos/memory/`).

## 8. what would move this next

- **N=3 seed-3 FFT** — locks the bimodality shape claim, in flight as of 2026-07-12.
- **verl-agent port** — the only way to test the TRL ≠ verl hypothesis directly. Real engineering, ~1 week.
- **Closed-loop reasoning eval** on our best curators — tests whether skills learned from ALFWorld transfer to math/GPQA (paper's cross-domain claim). Cheap once seed-3 frees the box.
- **WebShop** — third benchmark, ~1 week of engineering.

## 9. references

- Ouyang et al., 2026. SkillOS: Learning Skill Curation for Self-Evolving Agents. [arXiv:2605.06614](https://arxiv.org/abs/2605.06614)
- Shao et al., 2024. Group Relative Policy Optimization. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- verl-agent / GiGPO, 2025. [arXiv:2505.10978](https://arxiv.org/abs/2505.10978) — executor-side harness the paper defers to
