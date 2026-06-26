# Reproducing SkillOS on ALFWorld: an independent report

**Status: DRAFT skeleton (2026-06-25).** Section bodies marked _[TODO]_ pull from
`JOURNAL.md` / `DIVERGENCES.md`; numbers in tables are final unless marked _prelim_.

Target framing: a **reproducibility report** (TMLR repro track / MLRC / workshop),
NOT a "beats-SOTA" paper. Scope is **ALFWorld only** (one of the paper's three
benchmarks) with a different RL framework (**TRL**, vs the paper's **verl**).
Reference: SkillOS, arXiv:2605.06614 (Ouyang et al.). Executor harness deferred by
the paper to GiGPO / verl-agent (arXiv:2505.10978).

---

## Abstract _[TODO]_

One paragraph: we independently reproduce the SkillOS curator-training method on
ALFWorld using TRL instead of verl. We confirm a significant held-out skill-lift,
but report two reproducibility findings: (1) the training trajectory is **bimodal,
not monotone**, robust to LoRA-vs-FFT; (2) the zero-shot **8B baseline does not
reproduce** (47.9% paper vs ~34% ours) while **32B does** (54.5% vs 53.6%).

## 1. Background & claim under test _[TODO]_
- SkillOS = GRPO-trained "curator" maintaining a SkillRepo for a frozen executor.
- Algorithm 1: |G|=10 grouped task streams, composite reward
  `r = r_task + λ_f r_fc + λ_u r_cnt + λ_c r_comp`.
- Headline claim being tested: trained 8B curator gives **+13.3pp** ALFWorld SR
  over no-memory (47.9 → 61.2), with **fewer steps** (procedural shortcuts).

## 2. Setup & deviations from paper
- Hardware: 8×H100 (paper: 16×H100). Framework: **TRL 1.4** + DeepSpeed ZeRO-3 +
  vLLM colocate (paper: verl). Executor + judge on inference.sh.
- Curator: Qwen3-8B. Frozen executor: Qwen3-8B (also 32B control). Judge: Qwen3-32B.
- **Sanctioned deviations** (see `DIVERGENCES.md`): (a) LoRA r=32 with lr×10
  as a memory-fit path, later removed for a full-FT control; (b) executor decode =
  Qwen3 model-card sampling (temp 0.6/top_p 0.95/top_k 20), reasoning on.
- Full hyperparameter table _[TODO: lift from configs + Table 4]_.

## 3. Methods / protocol
- Held-out eval: **paired-by-gamefile McNemar**, n=140, valid_seen, K=20 waves.
- Per-type SR (Pick/Look/Clean/Heat/Cool/Pick2) + avg steps, vs paper Table 1.
- Noise floor: empirically ~±3pp at n=140 (same config, two runs: 33.6 vs 36.4) →
  gate per-config claims at ~2×SE ≈ 8pp.

## 4. Results

### 4.1 Held-out lift (confirmed)
- LoRA (v8): peak **ckpt30 +9.3pp (p=0.035)**; bimodal trajectory.
- FFT: peak **ckpt20 +10.7pp (p=0.032)**; same bimodal shape. Best arm beats LoRA
  and lands within ~2.6pp of paper's +13.3pp.
- _[TODO: seed-2 FFT (`algo1fftseed2`, seed 123) — does the bimodal shape
  reproduce with peaks at different indices? Launched 2026-06-25.]_

### 4.2 Finding 1 — the trajectory is bimodal, not monotone
- LoRA peak step 30, FFT peak step 20, both with a collapse-to-parity trough and
  recovery. Reproduced across the LoRA→FFT change → framework/small-batch GRPO,
  not a parameterization artifact. _[strengthen with seed-2]_

### 4.3 Finding 2 — the 8B baseline does not reproduce; 32B does
- 8B no-memory: ~34% (ours) vs 47.9% (paper). Audited to exhaustion: prompt
  (Fig 9 verbatim), precision (local bf16 = remote), retrieval, seeds, averaging,
  and **decode** (4-config sweep, all p>0.5) — all ruled out.
- 32B no-memory: 53.6% (ours) vs 54.5% (paper) — reproduces. → gap is
  **model-specific**, not a harness bug.

### 4.4 Mechanism — skills substitute for executor capability
- Lift concentrates on appliance/atomic-verb tasks (Clean +29, Cool +20, Heat +6),
  matching/beating the paper's per-type lift; the curator injects the atomic-verb
  knowledge the zero-shot executor lacks. Trace evidence: the frozen executor
  role-plays a physical microwave instead of the atomic `heat X with microwave`.
- Steps drop (23→22), reproducing the paper's "procedural shortcut" mechanism.

### 4.5 Per-type results table _[TODO: drop the JOURNAL 2026-06-24 tables]_

## 5. Discussion
- What reproduces: the **method** (curator training → transferable held-out lift)
  and its **mechanism** (knowledge injection).
- What doesn't: the paper's **clean monotone curve** and **absolute 8B baseline**.
- Threats to validity: TRL≠verl framework confound; n=140 power (~8pp gate);
  single benchmark; Qwen2.5→Qwen3 executor (can't match GiGPO's 512-token cap
  without crippling Qwen3's native reasoning).

## 6. Will it translate to real-world long-horizon skill curation? _[TODO]_
- Methodology is domain-agnostic and sound (per-rollout ephemeral repo,
  transfer-probe r_task, composite reward, GRPO over curation sequences).
- The clean lift rode on ALFWorld's narrowness (nameable missing verbs); expect a
  noisier signal, harder reward engineering, and worse small-batch instability in
  open-ended domains. Next real test: port the training loop to a checkable but
  open-skill-space domain (coding/tool-use), not more ALFWorld.

## 7. Reproducibility checklist / artifacts
- Configs: `configs/alfworld_8xh100_algo1_*.yaml`. Launchers: `run_algo1_*.sh`.
- Eval: `scripts/eval_streaming_curation.py`, `scripts/compare_eval_arms.py`.
- Result JSONLs: `output/eval-{v8,fft,decode,algo1}/`.
- Narrative log: `JOURNAL.md`. Deltas: `DIVERGENCES.md`.

## Open gates before submission
1. **Multi-seed** the bimodality claim (seed-2 running; ideally a 3rd).
2. WebShop (2nd benchmark) **or** scope title to ALFWorld explicitly.
3. Cross-executor transfer (8B-trained curator → 32B executor) — paper's
   generalization claim, not yet reproduced.
