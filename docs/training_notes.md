# SkillOS training — engineering notes (issues hit & fixed)

The things that make a paper-Algorithm-1 GRPO run actually complete on **8×H100
with TRL 1.4 / transformers 5.9 / deepspeed 0.19 / vLLM 0.21**, none of which the
paper (which used 16×H100 + verl) mentions. Symptom → root cause → fix. The
*code* fixes live in git; this is the operational knowledge that doesn't.

Stack note: the executor (Qwen3-8B) and content-quality judge (Qwen3-32B) run
**remotely on inference.sh**; only the curator trains on the local box.

---

## 1. Distributed / sharding (the big one)

**ZeRO-2 + vLLM colocate HANGS on this stack; use ZeRO-3.**
- Symptom: GPUs pinned at 0% util, 42 min with no progress, first vLLM generation
  never emits a token. No crash, no error — a silent wedge.
- Root cause: a bad interaction between ZeRO-2 and vLLM-colocate weight-gather on
  this exact library set. Older internal notes call ZeRO-2+vLLM "validated" —
  those are from an **older stack**; do not trust them here.
- Fix: **ZeRO-3 + vLLM colocate** (`configs/accelerate_zero3.yaml`). A 2026-06-21
  smoke ran clean (generation progressed, step closed). This is the empirically
  working FFT path.
- Bonus: ZeRO-3 shards **params + the KL reference model** (~2 GB/rank vs ZeRO-2's
  replicated ~16 GB), so the paper's `beta=0.001` KL term fits without OOM. ZeRO-2
  + KL had previously OOM'd (params 16G + ref 16G + vLLM KV). FSDP
  FULL_SHARD/SHARD_GRAD_OP wedged generation entirely.
- Cost: ZeRO-3 re-gathers params for the vLLM-colocate generation each step — a
  per-step all-gather tax. Full FFT runs ~70 min/step → a 60-step run is ~3 days
  (matches the paper's "~3 days on 16 H100").

**Validate a sharding change with ≥2 real steps, not just step 1.**
Step 1 fitting comfortably doesn't prove the run is healthy — the wedge above only
shows once generation actually has to produce tokens. Always smoke a fresh sharding
config to the point where rollouts provably progress (look for `rollout slot=` /
`executor` activity in the log), then watch the reward/KL trend over a few steps.

---

## 2. NCCL timeouts (two distinct failures, often confused)

**(a) Heartbeat monitor false-abort (~480 s).** Task-difficulty skew across ranks
means a fast rank waits at a post-generation collective for a slow rank; Torch's
NCCL **heartbeat monitor** (watchdog-of-the-watchdog, default 480 s) false-aborts
on that benign wait. Fix: `TORCH_NCCL_ENABLE_MONITORING=0` +
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`.

**(b) Real collective watchdog (1800 s).** A single rank sitting in Python >1800 s
between collectives trips the actual `_ALLGATHER_BASE` watchdog → SIGABRT on all
ranks. This is **not** the heartbeat monitor and **not** fixable with
`NCCL_TIMEOUT_MS` (a confirmed **no-op under accelerate** — the timeout stays at
the 1800 s default).
- Root cause we hit: the reward step gathered each probe future with its *own*
  `result(timeout=1200 s)`; during an inference.sh stall several probes stalled at
  once and their waits **added up** past 1800 s. Per-future bounds don't bound the
  *phase*.
- Fix: a single shared **phase deadline** `SKILLOS_PHASE_BUDGET_S=1500` (< 1800 s
  watchdog) around the whole seed-rollout and probe phases. Unfinished probes are
  dropped from `r_task` (masked, not failed) rather than crashing the run.
  Checkpoints every 5–10 steps make the resume free.

---

## 3. Remote-backend resilience (inference.sh)

**Aggressive per-call retry is a metastable-failure amplifier.** With stock
`run_task_resilient` defaults (`max_resubmissions=10`, `poll=900 s`), a transient
infsh blip became a self-sustaining **resubmission storm**: stuck calls piled tasks
on faster than they drained, phase-budget timeouts abandoned the futures but
couldn't kill the threads, and zombie episodes kept retrying into the backlog —
saturating the shared rollout pool and starving every subsequent probe
(`64/64` probes dropped, `r_task` collapsed to 0, reward pinned at the
function-call floor).
- Fix: the **executor fires the most calls, so it must fail fast** —
  `SKILLOS_EXEC_MAX_RESUBS=2`, `SKILLOS_EXEC_POLL_MAX_S=150`, backoff cap 30 s.
- Plus: when our timeout fires, the task is **still running on the server** —
  `client.tasks.cancel(task_id)` before resubmitting/giving up, else live tasks
  pile up. Skip cancel only if infsh already moved it to terminal FAILED/CANCELLED.
- inference.sh is **elastic** — it autoscales. Slowdowns are storms of our own
  making or cold starts, not a fixed-capacity contention ceiling. Run evals
  concurrently freely.

---

## 4. Data diversity

**Randomize the seed game, or you train on one task type.** ALFWorld gamefiles are
**alphabetically clustered by task-type prefix**; a sequential seed iterator walks
them in order, so early training saw almost entirely `clean` tasks. Fix: draw the
seed game with `random.randint(0, 2³¹−1)`. All 6 types then appear. (The canonical
6-type classifier now lives in `skillos/envs/task_types.py`.)

---

## 5. Reward-wiring gotchas (silent zeros)

These don't crash — they silently zero a reward term and you only notice in a flat
trajectory. From the 2026-06-10 group-collapse postmortem:
- **`judge_submit=None` silently zeroes `λ_u·r_cnt`** (the content-quality term).
  Wire it explicitly (`train_algo1.py` does).
- **Constant-within-group reward cancels in the GRPO advantage.** If every rollout
  in a `|G|`-group gets the same `r_task` (e.g. all dropped, or a degenerate
  `gid=0` / hash-salt bug), advantages are ~0 and nothing learns despite a
  "healthy"-looking reward. Watch `frac_reward_zero_std` — it must stay ~0.
- A public `compute_reward` method gets auto-registered by TRL's
  `inspect.getmembers` as a phantom 4th tool → rename internal helpers with `_`.

---

## 6. Memory / OOM

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` guards against
  fragmentation OOM on the long run.
- `vllm_gpu_memory_utilization: 0.3` is the smoke-proven colocate fraction under
  ZeRO-3 (FFT sustained ~80.3/81.5 GB, 0 OOM across 60 steps). If a resume OOMs,
  drop to 0.25.
- Per-GPU sitting flat at ~80/81.5 GB with <1 GB free is **stable, not a leak** —
  it held flat for days with 0 OOM events. Don't panic at the ceiling.

---

## 7. Operational hygiene

- **Cadence:** trust the tqdm bar (~70 min/step FFT, ~40 min/step LoRA), not early
  timestamp-gap clustering (which misread ~6 min/step). Don't restart for wall
  time; checkpoints make a later speed-up resume safe.
- **Monitors:** grep **fatal-only** patterns
  (`watchdog got stuck|ChildFailedError|exitcode: -|CUDA out of memory`). Benign
  infsh `…resubmitting` lines contain "failed"/"FAILED" and will false-trigger a
  naive monitor.
- **Executor `max_steps`:** the default 10 caps trajectories well below the paper's
  ~21-step average — set `SKILLOS_EXECUTOR_MAX_STEPS≥25`.

---

## Reference: the env vars that matter

| Var | Value | Why |
|---|---|---|
| `SKILLOS_PHASE_BUDGET_S` | 1500 | whole-phase deadline < 1800 s NCCL watchdog |
| `TORCH_NCCL_ENABLE_MONITORING` | 0 | disable the 480 s heartbeat false-abort |
| `SKILLOS_EXEC_MAX_RESUBS` | 2 | executor fails fast — no resubmission storm |
| `SKILLOS_EXEC_POLL_MAX_S` | 150 | "" |
| `SKILLOS_EXECUTOR_MAX_STEPS` | 25–30 | match the paper's trajectory length |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | fragmentation OOM guard |
| `NCCL_TIMEOUT_MS` | — | **no-op under accelerate; don't rely on it** |

Sharding config: `configs/accelerate_zero3.yaml`. Launchers: `run_algo1_fft.sh`,
`run_algo1_v8_lora_kl.sh`. Group-collapse detail:
`docs/postmortem-2026-06-10-algo1-group-collapse.md`.
