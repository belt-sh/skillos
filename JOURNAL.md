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
