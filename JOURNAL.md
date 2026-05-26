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
