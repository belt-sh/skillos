"""Curator training environment -- the environment for the model we actually train.

The SkillOS training loop (paper §3.1):
1. Frozen executor solves one ALFWorld task per (per_device) prompt — ONCE
2. That trajectory is the *input* for num_generations curator samples (group)
3. Each curator sample emits insert/update/delete ops via tool calls
4. Reward = composite over the trajectory + ops; GRPO uses group-relative advantage

Concurrency:
- TRL drives env.reset() sequentially across the generation_batch_size envs.
  We pre-dispatch one executor rollout *per group* (not per slot) in a thread
  pool; envs in the same group share the trajectory future.
- alfworld + textworld + tatsu hold module-global parser state and aren't
  thread-safe, so reset()/step() are serialized behind a single global lock.
  Executor network I/O runs outside the lock so rollouts overlap on the slow
  part.
- Judge calls are dispatched async the moment a valid skill content op is
  emitted, with a sha256 content cache to dedupe identical curator outputs.
  compute_reward() awaits whatever's still pending.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import defaultdict, deque

from skillos.curator.prompts import CURATOR_INPUT_TEMPLATE
from skillos.envs.config import make_alfworld_env
from skillos.executor.executor import Executor, HeuristicExecutor, create_executor
from skillos.rewards.judge import Judge, HeuristicJudge, create_judge
from skillos.skills.repo import SkillRepo


# ---------------------------------------------------------------------------
# Module-level shared state. TRL creates fresh env instances per generation,
# so the skill repo / executor / judge / batch dispatcher live here.
# ---------------------------------------------------------------------------

_shared_skill_repo = SkillRepo()
_executor: Executor = HeuristicExecutor()
_judge: Judge = HeuristicJudge()

# Knobs (overridable via env vars or configure()).
_max_steps = int(os.environ.get("SKILLOS_EXECUTOR_MAX_STEPS", "10"))
_max_parallel_rollouts = int(os.environ.get("SKILLOS_PARALLEL_ROLLOUTS", "16"))
_max_parallel_judges = int(os.environ.get("SKILLOS_PARALLEL_JUDGES", "16"))
# Per-future deadlines. Must each be strictly less than the NCCL collective
# watchdog (default 1800 s) so a single stuck infsh call can't out-wait it
# and trip the multi-rank watchdog.
_executor_timeout_s = float(os.environ.get("SKILLOS_EXECUTOR_TIMEOUT_S", "1200"))
_judge_timeout_s = float(os.environ.get("SKILLOS_JUDGE_TIMEOUT_S", "600"))
_num_generations = 1  # set via configure()

# Path B (transfer-probe r_task): after the curator curates from the seed task's
# trajectory into its own per-rollout repo, we measure r_task as the frozen
# executor's success on `_num_probe_tasks` freshly-sampled tasks of the SAME
# ALFWorld type, using that rollout's curated skills. This makes r_task vary
# across the N samples of a group (they curate differently) — the within-group
# signal GRPO needs — and rewards skills that *transfer* to related tasks rather
# than ones that merely please the judge or re-solve the exact seed task.
_num_probe_tasks = int(os.environ.get("SKILLOS_NUM_PROBE_TASKS", "3"))
_probe_type_sample_cap = int(os.environ.get("SKILLOS_PROBE_TYPE_CAP", "12"))

# Concurrency primitives.
_batch_lock = threading.Lock()
_alfworld_global_lock = threading.Lock()
_judge_cache_lock = threading.Lock()

# Pools — created lazily on first use.
_rollout_pool: concurrent.futures.ThreadPoolExecutor | None = None
_judge_pool: concurrent.futures.ThreadPoolExecutor | None = None

# alfworld envs are per-group (one game source per task, shared across the
# group's N curator samples) — sized to num_groups, not generation_batch_size.
_alfworld_envs: list = []

# Active batch state: trajectory future per group_id, plus per-group "done"
# counter so we can release a group's future once all N slots consume it.
_group_trajectories: dict[int, concurrent.futures.Future] = {}
_group_done_counts: dict[int, int] = {}

# Judge cache (sha256(content) -> score). Identical curator outputs reuse.
_judge_cache: dict[str, float] = {}

# Instances, in creation order. Slot index used to derive group_id.
_instances: list = []

# Per-group seed task type (group_id -> one of the 6 ALFWorld types). Set once
# the group's seed rollout resolves; probes sample tasks of this same type.
_group_types: dict[int, str] = {}

# Borrowable pool of alfworld envs for probe episodes. Capped at the rollout
# parallelism so we never build more textworld games than can run at once;
# envs are reused (reset() advances to a fresh game) across steps.
_probe_env_lock = threading.Lock()
_probe_env_free: list = []
_probe_env_total = 0

# Seed -> task-type index, built once. Game selection is deterministic in the
# env seed, so we pre-scan which seeds land on which of the 6 ALFWorld types
# (read from the gamefile path). Probes then seed directly to a same-type game
# — no slow rejection sampling, and reliable for rare types.
_type_seeds: dict[str, list[int]] = {}
_type_seed_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Observability state (per-rollout logging, append-only artifacts, heartbeat)
# ---------------------------------------------------------------------------

_obs_lock = threading.Lock()
_rollouts_completed = 0          # all-time rollouts that completed _compute_reward
_step_rollouts_completed = 0     # rollouts since last step boundary (reset on step landing)
_step_rollouts_expected = 0      # batch_size * num_generations
_step_reward_sum = 0.0
_step_reward_count = 0
_rollouts_inflight = 0           # rollouts past reset() that haven't hit _compute_reward yet
_oldest_inflight_started_at: float | None = None

_log_every_n_rollouts = int(os.environ.get("SKILLOS_LOG_EVERY", "8"))
_heartbeat_interval_s = float(os.environ.get("SKILLOS_HEARTBEAT_S", "60"))

_artifacts_dir = os.environ.get("SKILLOS_ARTIFACTS_DIR", "output/skillos-live")
_rollouts_jsonl_path = os.path.join(_artifacts_dir, "rollouts.jsonl")
_skills_live_dir = os.path.join(_artifacts_dir, "skills")

_heartbeat_started = False
_pilot_start_time = time.time()


# ---------------------------------------------------------------------------
# Public configuration entrypoint
# ---------------------------------------------------------------------------

def configure(executor_config: dict | None = None,
              judge_config: dict | None = None,
              num_generations: int | None = None,
              num_probe_tasks: int | None = None) -> None:
    """Configure executor / judge / GRPO group size. Call before training."""
    global _executor, _judge, _num_generations, _num_probe_tasks
    if executor_config:
        _executor = create_executor(executor_config)
    if judge_config:
        _judge = create_judge(judge_config)
    if num_generations is not None:
        _num_generations = max(1, int(num_generations))
    if num_probe_tasks is not None:
        _num_probe_tasks = max(1, int(num_probe_tasks))


def reset_shared_state() -> None:
    """Reset shared state at the start of a new task group (paper §3.2.1)."""
    global _shared_skill_repo
    _shared_skill_repo = SkillRepo()


# ---------------------------------------------------------------------------
# Checkpoint state — skill repo + judge cache + rollouts snapshot.
# Shared by the TRL save/load callbacks (train.py) and the resume test
# (scripts/debug_checkpoint_resume.py) so neither hand-mirrors the other.
# ---------------------------------------------------------------------------

def save_curator_state(ckpt_root: str, rollouts_src: str | None = None) -> None:
    """Persist the skill repo, judge cache, and (optionally) a rollouts
    snapshot under ckpt_root. The skill repo is the eval artifact; the judge
    cache lets a resumed run skip already-paid judge calls; rollouts are
    snapshotted only for post-crash analytics (the live append-only file
    remains the source of truth)."""
    os.makedirs(ckpt_root, exist_ok=True)
    _shared_skill_repo.save(f"{ckpt_root}/skills")
    if rollouts_src and os.path.exists(rollouts_src):
        try:
            shutil.copyfile(rollouts_src, f"{ckpt_root}/rollouts.jsonl")
        except Exception as e:
            print(f"[ckpt] rollouts snapshot failed: {e}", file=sys.stderr)
    try:
        with _judge_cache_lock:
            cache_snap = dict(_judge_cache)
        with open(f"{ckpt_root}/judge_cache.json", "w") as f:
            json.dump(cache_snap, f)
    except Exception as e:
        print(f"[ckpt] judge cache snapshot failed: {e}", file=sys.stderr)


def load_curator_state(ckpt_root: str) -> int:
    """Restore the skill repo + judge cache from ckpt_root into the shared
    module state. Returns the number of skills loaded. Rollouts are not
    copied back — the live append-only file keeps growing across runs."""
    skills_dir = os.path.join(ckpt_root, "skills")
    n_skills = 0
    if os.path.isdir(skills_dir):
        loaded = SkillRepo.load(skills_dir)
        _shared_skill_repo.replace_skills(loaded.skills)
        n_skills = len(loaded.skills)
        print(f"[resume] loaded {n_skills} skills from {skills_dir}")
    cache_path = os.path.join(ckpt_root, "judge_cache.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            with _judge_cache_lock:
                _judge_cache.update(cache)
            print(f"[resume] restored judge cache: {len(cache)} entries")
        except Exception as e:
            print(f"[resume] judge cache restore failed: {e}", file=sys.stderr)
    return n_skills


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------

def _ensure_artifacts_dir() -> None:
    os.makedirs(_artifacts_dir, exist_ok=True)
    os.makedirs(_skills_live_dir, exist_ok=True)


def _append_rollout_record(record: dict) -> None:
    """Append a single rollout outcome to rollouts.jsonl (crash-safe)."""
    try:
        _ensure_artifacts_dir()
        with open(_rollouts_jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
    except Exception as e:
        print(f"[obs] rollout-jsonl write failed: {e}", file=sys.stderr)


def _dump_skills_live() -> None:
    """Persist current skill repo to disk *now* (not just on TRL save_steps).

    Crash-safe in the sense that whatever skills the LLM has paid to generate
    so far survive a mid-step crash.
    """
    try:
        _ensure_artifacts_dir()
        _shared_skill_repo.save(_skills_live_dir)
    except Exception as e:
        print(f"[obs] live-skill-dump failed: {e}", file=sys.stderr)


def _wandb_log_safe(payload: dict) -> None:
    """Log to wandb if a run is active; never raise."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(payload, commit=False)
    except Exception:
        pass


def _start_heartbeat_once() -> None:
    """Spin up the heartbeat daemon thread on first call (idempotent)."""
    global _heartbeat_started
    with _obs_lock:
        if _heartbeat_started:
            return
        _heartbeat_started = True

    def _heartbeat_loop() -> None:
        import torch
        while True:
            time.sleep(_heartbeat_interval_s)
            try:
                with _obs_lock:
                    completed = _rollouts_completed
                    step_done = _step_rollouts_completed
                    step_expected = _step_rollouts_expected
                    inflight = _rollouts_inflight
                    oldest_age = (
                        time.time() - _oldest_inflight_started_at
                        if _oldest_inflight_started_at else 0.0
                    )
                    mean_reward = (
                        _step_reward_sum / _step_reward_count
                        if _step_reward_count else 0.0
                    )
                uptime_s = int(time.time() - _pilot_start_time)
                gpu_str = ""
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1024**3
                    gpu_str = f" gpu={mem_gb:.1f}GB"
                print(
                    f"[heartbeat] uptime={uptime_s//60}m "
                    f"step_rollouts={step_done}/{step_expected} "
                    f"total_rollouts={completed} "
                    f"in_flight={inflight} "
                    f"oldest_inflight={oldest_age:.0f}s "
                    f"mean_reward={mean_reward:.3f}{gpu_str}",
                    flush=True,
                )
                _wandb_log_safe({
                    "live/rollouts_in_step": step_done,
                    "live/total_rollouts": completed,
                    "live/in_flight": inflight,
                    "live/oldest_inflight_s": oldest_age,
                    "live/step_mean_reward": mean_reward,
                })
            except Exception as e:
                print(f"[heartbeat] crashed once: {e}", file=sys.stderr)

    t = threading.Thread(target=_heartbeat_loop, daemon=True, name="skillos-heartbeat")
    t.start()


def _on_rollout_start(slot: int) -> None:
    """Marker that a slot's reset() has finished — rollout is now in-flight."""
    global _rollouts_inflight, _oldest_inflight_started_at
    with _obs_lock:
        _rollouts_inflight += 1
        if _oldest_inflight_started_at is None or _rollouts_inflight == 1:
            _oldest_inflight_started_at = time.time()


def _on_rollout_complete(slot: int, reward: float, executor_result: dict | None,
                         ops_applied: list[dict], r_task: float | None = None,
                         num_skills: int | None = None) -> None:
    """Record + log a completed rollout's outcome."""
    global _rollouts_completed, _step_rollouts_completed
    global _step_reward_sum, _step_reward_count, _rollouts_inflight
    global _oldest_inflight_started_at
    with _obs_lock:
        _rollouts_completed += 1
        _step_rollouts_completed += 1
        _step_reward_sum += reward
        _step_reward_count += 1
        _rollouts_inflight = max(0, _rollouts_inflight - 1)
        if _rollouts_inflight == 0:
            _oldest_inflight_started_at = None
        step_done = _step_rollouts_completed
        step_expected = _step_rollouts_expected
        completed = _rollouts_completed
        mean_reward = _step_reward_sum / max(_step_reward_count, 1)

    _append_rollout_record({
        "ts": time.time(),
        "slot": slot,
        "reward": reward,
        # r_task is the transfer-probe success rate (the held-out-relevant
        # signal); success mirrors it for back-compat with old analytics.
        "r_task": r_task,
        "num_skills": num_skills,
        "success": (r_task is not None and r_task > 0)
                   if r_task is not None
                   else bool(executor_result and executor_result.get("success")),
        "steps": (executor_result or {}).get("steps"),
        "ops": [
            {"name": o["name"], "valid": o["valid"]} for o in ops_applied
        ],
        "task_description": (executor_result or {}).get("task_description"),
    })

    if step_done % _log_every_n_rollouts == 0 or step_done == step_expected:
        print(
            f"[rollout] step_progress={step_done}/{step_expected} "
            f"total={completed} reward={reward:.3f} "
            f"step_mean={mean_reward:.3f}",
            flush=True,
        )
        _wandb_log_safe({
            "rollout/reward_latest": reward,
            "rollout/step_mean_reward": mean_reward,
            "rollout/step_progress": step_done,
        })


def _reset_step_counters(expected_total: int) -> None:
    """Called by train.py when a new opt step starts."""
    global _step_rollouts_completed, _step_rollouts_expected
    global _step_reward_sum, _step_reward_count
    with _obs_lock:
        _step_rollouts_completed = 0
        _step_rollouts_expected = expected_total
        _step_reward_sum = 0.0
        _step_reward_count = 0


def set_step_expected_rollouts(n: int) -> None:
    """Public hook for train.py to declare batch_size × num_generations."""
    _reset_step_counters(n)
    _start_heartbeat_once()


# ---------------------------------------------------------------------------
# Pool / env accessors
# ---------------------------------------------------------------------------

def _get_rollout_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _rollout_pool
    if _rollout_pool is None:
        _rollout_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_max_parallel_rollouts,
            thread_name_prefix="skillos-rollout",
        )
    return _rollout_pool


def _get_judge_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _judge_pool
    if _judge_pool is None:
        _judge_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_max_parallel_judges,
            thread_name_prefix="skillos-judge",
        )
    return _judge_pool


def _get_alfworld_env_for_group(group_id: int):
    """Get (or lazily build) the alfworld env dedicated to this group.

    Each group corresponds to one task; the env's reset() picks the next
    game in that group's history. With num_groups groups, we only ever need
    num_groups alfworld envs in memory.
    """
    with _alfworld_global_lock:
        while len(_alfworld_envs) <= group_id:
            _alfworld_envs.append(make_alfworld_env())
    return _alfworld_envs[group_id]


# ---------------------------------------------------------------------------
# Judge — async dispatch + sha256 memoization
# ---------------------------------------------------------------------------

def _score_with_cache(content: str) -> float:
    """Compute judge score with sha256 memoization.

    Errors are NOT swallowed — silent 0-scoring pollutes the reward signal
    and corrupts training. The underlying client (run_task_resilient) is
    responsible for retrying through transient inference.sh flakiness.
    Only a true upstream outage should reach this layer and re-raise,
    which crashes the step. Resume from the last checkpoint.
    """
    key = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with _judge_cache_lock:
        if key in _judge_cache:
            return _judge_cache[key]
    score = _judge.score(content)
    with _judge_cache_lock:
        _judge_cache[key] = score
    return score


def _submit_judge(content: str) -> concurrent.futures.Future:
    """Fire-and-forget judge call. Returns a Future the env can await later."""
    return _get_judge_pool().submit(_score_with_cache, content)


# ---------------------------------------------------------------------------
# Executor rollout — one per group
# ---------------------------------------------------------------------------

def _trajectory_result(trajectory=None, success=False, steps=0,
                       task_description="", skills_text="", gamefile="") -> dict:
    """The canonical executor-rollout result shape, shared by a real rollout
    and by the sentinel-failure path so the two can't drift."""
    return {
        "trajectory": trajectory or [],
        "success": success,
        "steps": steps,
        "task_description": task_description,
        "skills_text": skills_text,
        "gamefile": gamefile,
    }


def _probe_seed_base(seed_gamefile: str) -> int:
    """Deterministic per-group probe seed derived from the group's seed-task
    gamefile. Stable across processes/ranks (md5, not the salted builtin hash)
    so every rollout in a GRPO group — wherever it runs — probes the SAME games.
    Shared probe games make the GRPO advantage reflect curation quality rather
    than which rollout drew easier probe tasks (paper: one task set per group)."""
    h = hashlib.md5((seed_gamefile or "seed").encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _task_type(description: str) -> str:
    """Classify an ALFWorld task description into one of the 6 task types.

    The partition (paper §3.2.1) is over these types; probes sample tasks of
    the same type as the group's seed. Order matters: 'pick2' (two) before the
    transform verbs, which precede the generic pick/look fallbacks.
    """
    d = (description or "").lower()
    if "two" in d:
        return "pick2"
    if "clean" in d:
        return "clean"
    if "hot" in d or "heat" in d:
        return "heat"
    if "cool" in d:
        return "cool"
    if ("look" in d or "examine" in d) and "lamp" in d:
        return "look"
    return "pick"


def _borrow_probe_env():
    """Borrow a reusable alfworld env for a probe episode, or build one (up to
    the rollout-parallelism cap). Blocks briefly only if all are in use."""
    global _probe_env_total
    while True:
        with _probe_env_lock:
            if _probe_env_free:
                return _probe_env_free.pop()
            if _probe_env_total < _max_parallel_rollouts:
                _probe_env_total += 1
                build = True
            else:
                build = False
        if build:
            return make_alfworld_env()
        time.sleep(0.05)


def _return_probe_env(env) -> None:
    with _probe_env_lock:
        _probe_env_free.append(env)


def _run_executor_on_env(env, repo: SkillRepo, max_steps: int,
                         want_type: str | None = None,
                         seed: int | None = None) -> dict:
    """Reset `env` (re-rolling up to a cap until the task type matches
    `want_type`) and run the frozen executor through it using `repo`'s
    retrieved skills. Returns the canonical trajectory result.

    If `seed` is given, the env's game selection is made deterministic before
    the first reset, so two rollouts that pass the same (seed, want_type) land
    on the SAME game even on different pooled env instances.

    Network executor calls happen outside the alfworld lock so episodes on
    different envs overlap; only the textworld step/reset (not thread-safe)
    is serialized.
    """
    gamefile = ""

    def _reset_once():
        nonlocal gamefile
        with _alfworld_global_lock:
            obs, infos = env.reset()
        gf = infos.get("extra.gamefile", [""])
        gamefile = gf[0] if gf else ""
        return obs[0], infos.get("admissible_commands", [[]])[0]

    if seed is not None:
        with _alfworld_global_lock:
            env.seed(seed)
    observation, admissible_actions = _reset_once()
    task_description = _extract_task_description(observation)
    tries = 0
    while (want_type is not None and _task_type(task_description) != want_type
           and tries < _probe_type_sample_cap):
        tries += 1
        observation, admissible_actions = _reset_once()
        task_description = _extract_task_description(observation)

    # Skill retrieval reads only — safe outside the lock.
    retrieved = repo.retrieve(task_description, top_k=5)
    skills_text = repo.format_skills(retrieved)

    trajectory: list = []
    recent_history: deque[str] = deque(maxlen=3)
    success = False
    done = False
    step = 0

    while not done and step < max_steps:
        # Network I/O — concurrent across episodes.
        action = _executor.act(
            task_description=task_description,
            observation=observation,
            admissible_actions=admissible_actions,
            step_count=step,
            action_history="\n".join(recent_history),
            retrieved_skills=skills_text,
        )
        # alfworld + textworld + tatsu — serial.
        with _alfworld_global_lock:
            obs_n, scores, dones, infos = env.step([action])
        observation = obs_n[0]
        admissible_actions = infos.get("admissible_commands", [[]])[0]
        done = dones[0]
        score = scores[0]
        step += 1

        trajectory.append({"step": step, "action": action, "observation": observation})
        recent_history.append(f"ACTION: {action}\nOBSERVATION: {observation}")

        if done:
            success = score > 0

    return _trajectory_result(
        trajectory=trajectory,
        success=success,
        steps=step,
        task_description=task_description,
        skills_text=skills_text,
        gamefile=gamefile,
    )


def _run_seed_rollout(group_id: int, max_steps: int) -> dict:
    """Run the group's seed task (paper task 1) with an EMPTY repo, and record
    the group's task type so probes can sample same-type tasks.

    Runs in a worker thread; pre-dispatched per group so seeds overlap.
    """
    env = _get_alfworld_env_for_group(group_id)
    # Randomize the seed game. ALFWorld gamefiles are alphabetically clustered
    # by task type, and a fresh per-group env's iterator advances only one game
    # per step — so without this every early step trains on the same type
    # (observed: all-"clean" for the first steps). A random seed spreads seed
    # tasks across all 6 types each step (env.seed is deterministic but we want
    # variety here, so we draw a fresh seed per group per step).
    seed = random.randint(0, 2**31 - 1)
    result = _run_executor_on_env(env, SkillRepo(), max_steps, want_type=None, seed=seed)
    with _batch_lock:
        _group_types[group_id] = _task_type(result["task_description"])
    return result


def _run_probe(repo: SkillRepo, want_type: str | None, max_steps: int,
               seed: int | None = None) -> dict:
    """Run one transfer probe: a same-type task (pinned by `seed` so all
    rollouts in a group share it) solved by the frozen executor using `repo`'s
    curated skills. Borrows a pooled env."""
    env = _borrow_probe_env()
    try:
        return _run_executor_on_env(env, repo, max_steps, want_type=want_type, seed=seed)
    finally:
        _return_probe_env(env)


def _classify_gamefile(gf: str) -> str:
    """Task type from the ALFWorld gamefile path (reliable, no env run)."""
    g = (gf or "").lower()
    if "pick_two" in g:
        return "pick2"
    if "pick_clean" in g:
        return "clean"
    if "pick_heat" in g:
        return "heat"
    if "pick_cool" in g:
        return "cool"
    if "look_at" in g:
        return "look"
    return "pick"


def _build_type_seed_index(n: int = 400) -> None:
    """Scan `n` seeds once, bucketing each into its task type. Cheap (local
    resets, no executor) and one-time."""
    global _type_seeds
    with _type_seed_lock:
        if _type_seeds:
            return
        env = _borrow_probe_env()
        idx: dict[str, list[int]] = defaultdict(list)
        try:
            for s in range(n):
                with _alfworld_global_lock:
                    env.seed(s)
                    _, infos = env.reset()
                gf = (infos.get("extra.gamefile") or [""])[0]
                idx[_classify_gamefile(gf)].append(s)
        finally:
            _return_probe_env(env)
        _type_seeds = dict(idx)
        print("[curator_env] type-seed index: "
              + " ".join(f"{k}={len(v)}" for k, v in sorted(_type_seeds.items())),
              flush=True)


def _probe_seed_for(want_type: str, base: int, j: int) -> int:
    """Deterministic seed that lands a probe on a `want_type` game. `base` keys
    off the group's seed task so a group's rollouts share games; `j` indexes the
    probe slot so the k probes are distinct."""
    if not _type_seeds:
        _build_type_seed_index()
    seeds = _type_seeds.get(want_type) or _type_seeds.get("pick") or [base + j]
    return seeds[(base + j) % len(seeds)]


def compute_rewards_batched(environments: list) -> list[float]:
    """Reward entry for TRL's reward_func. Runs transfer probes for ALL rollouts
    concurrently (the slow, network-bound part), then composes each reward.

    r_task for a rollout = mean frozen-executor success over `_num_probe_tasks`
    freshly-sampled same-type tasks, solved using that rollout's curated repo.
    Because the N samples of a group curate different skills, their r_task
    differs — restoring the within-group GRPO signal that the old shared-
    trajectory r_task (constant within a group) could never provide.
    """
    pool = _get_rollout_pool()
    # Phase 1: fan out every rollout's probe episodes so they overlap. Each
    # group's probe games are fixed by its seed task (via _probe_seed_for), so
    # every rollout in the group is scored on the SAME same-type task set —
    # differences in r_task come only from curation quality, not probe-task luck.
    probe_futs = []
    for env in environments:
        base = _probe_seed_base(getattr(env, "_seed_gamefile", ""))
        seeds = [_probe_seed_for(env._group_type, base, j) for j in range(_num_probe_tasks)]
        probe_futs.append([
            pool.submit(_run_probe, env._repo, None, _max_steps, s) for s in seeds
        ])
    # Phase 2: gather successes and compose each reward.
    rewards = []
    for env, futs in zip(environments, probe_futs):
        successes = []
        for f in futs:
            try:
                res = f.result(timeout=_executor_timeout_s)
                successes.append(1.0 if res["success"] else 0.0)
            except Exception as e:
                # A stuck/failed probe degrades that sample's signal rather
                # than crashing the multi-rank step (mirrors the seed path).
                print(
                    f"[curator_env] probe failed ({type(e).__name__}: {e}); "
                    f"dropping it from r_task",
                    file=sys.stderr, flush=True,
                )
        r_task = sum(successes) / len(successes) if successes else 0.0
        env._finalize_reward(r_task)
        rewards.append(env.reward)
    return rewards


def _extract_task_description(observation: str) -> str:
    """Pull the real ALFWorld task from the initial observation.

    The observation is shaped like:
        -= Welcome to TextWorld, ALFRED! =-

        You are in the middle of a room. … (room description)

        Your task is to: put a cool bread in countertop.

    The task description is the line that starts with "Your task is to:". The
    naive `observation.split("\\n")[0]` we used previously returned the welcome
    banner, which left the curator with no goal to reason about and caused it
    to refuse all curation ops.
    """
    if not observation:
        return "Unknown task"
    for line in observation.splitlines():
        s = line.strip()
        if s.lower().startswith("your task is"):
            return s
    return observation.splitlines()[0].strip()


def _format_trajectory(result: dict) -> str:
    parts = []
    for step in result["trajectory"]:
        parts.append(f"Step {step['step']}: ACTION: {step['action']}")
        parts.append(f"        OBSERVATION: {step['observation']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CuratorEnv
# ---------------------------------------------------------------------------

class CuratorEnv:
    """Environment for training the skill curator with TRL GRPOTrainer.

    TRL auto-discovers the tool methods (new_skill_insert, skill_update,
    skill_delete) and exposes them to the model being trained.

    Slots are grouped by `_num_generations`: slots in the same group share
    one frozen-executor trajectory (paper §3.1) and only differ in the
    curator's sampled response.
    """

    def __init__(self):
        self.reward = 0.0
        self.done = False
        self._executor_result: dict | None = None
        self._ops_applied: list[dict] = []
        self._judge_futures: list[concurrent.futures.Future] = []
        self._input_tokens = 0
        # Per-rollout ephemeral repo (paper: SkillRepo starts empty per group).
        # The curator's ops mutate THIS, not a shared global, so each of the N
        # samples in a group builds a different repo -> different transfer-probe
        # success -> within-group r_task variance for GRPO.
        self._repo = SkillRepo()
        self._group_type = "pick"
        self._seed_gamefile = ""
        _instances.append(self)
        self._slot = len(_instances) - 1

    # ---- TRL hook -----------------------------------------------------

    def reset(self, **kwargs) -> str:
        """Run the group's seed task (empty repo) and return its trajectory for
        the curator to curate from. r_task is measured later, in the batched
        reward step, via transfer probes on this rollout's curated repo."""
        self.reward = 0.0
        self.done = False
        self._ops_applied = []
        self._judge_futures = []
        self._repo = SkillRepo()

        future = self._get_or_submit_group_future()
        try:
            result = future.result(timeout=_executor_timeout_s)
        except concurrent.futures.TimeoutError:
            # Don't out-wait the NCCL collective watchdog. Replace the group's
            # future with a resolved sentinel so other slots in this group
            # short-circuit instead of each waiting their own full timeout.
            print(
                f"[curator_env] executor rollout timed out after "
                f"{_executor_timeout_s:.0f}s for group {self._group_id} "
                f"(slot {self._slot}); using sentinel-failure trajectory",
                file=sys.stderr,
                flush=True,
            )
            result = self._sentinel_trajectory("executor-timeout")
        except Exception as e:
            # Executor rollout raised (e.g. resilient client exhausted its
            # retry budget during an inference.sh outage). Degrade to a
            # sentinel-failure trajectory rather than crashing the 8-rank run.
            print(
                f"[curator_env] executor rollout failed for group "
                f"{self._group_id} (slot {self._slot}): {type(e).__name__}: {e}; "
                f"using sentinel-failure trajectory (likely infsh outage)",
                file=sys.stderr,
                flush=True,
            )
            result = self._sentinel_trajectory("executor-error")
        self._executor_result = result
        self._seed_gamefile = result.get("gamefile", "")
        with _batch_lock:
            self._group_type = _group_types.get(self._group_id, _task_type(result["task_description"]))
        _on_rollout_start(self._slot)

        trajectory_text = _format_trajectory(result)
        result_text = "Success" if result["success"] else "Failure"
        curator_input = CURATOR_INPUT_TEMPLATE.format(
            task_description=result["task_description"],
            past_skills=result.get("skills_text", ""),
            agent_trajectory=trajectory_text,
            result=result_text,
        )
        self._input_tokens = len(curator_input.split())

        # If all N slots in this group have read the future, release it.
        self._maybe_release_group()

        return curator_input

    # ---- Tool methods (TRL discovers these) ---------------------------

    def new_skill_insert(self, skill_name: str, content: str) -> str:
        """Create a new skill in the skill repo.

        Args:
            skill_name: The name of the new skill to create.
            content: The markdown content for the new skill, including YAML frontmatter.

        Returns:
            Confirmation message or error.
        """
        success = self._repo.insert(skill_name, content)
        self._ops_applied.append({
            "name": "new_skill_insert",
            "arguments": {"skill_name": skill_name, "content": content},
            "valid": success,
        })
        if success and content:
            # Fire judge async right now; we'll await in compute_reward.
            self._judge_futures.append(_submit_judge(content))
        if success:
            return f"Skill '{skill_name}' created. Repo has {len(self._repo)} skills."
        return f"Failed to create '{skill_name}'. Already exists or invalid format."

    def skill_update(self, skill_name: str, new_name: str = "", new_content: str = "") -> str:
        """Update an existing skill in the skill repo.

        Args:
            skill_name: The name of the skill to update. Must exactly match an existing skill.
            new_name: Optional new name for the skill.
            new_content: Optional new content to replace the entire skill content.

        Returns:
            Confirmation message or error.
        """
        success = self._repo.update(
            skill_name,
            new_name=new_name if new_name else None,
            new_content=new_content if new_content else None,
        )
        self._ops_applied.append({
            "name": "skill_update",
            "arguments": {"skill_name": skill_name, "new_content": new_content},
            "valid": success,
        })
        if success and new_content:
            self._judge_futures.append(_submit_judge(new_content))
        if success:
            return f"Skill '{new_name or skill_name}' updated."
        return f"Failed to update '{skill_name}'. Does not exist."

    def skill_delete(self, skill_name: str) -> str:
        """Delete an existing skill from the skill repo.

        Args:
            skill_name: The name of the skill to delete.

        Returns:
            Confirmation message or error.
        """
        success = self._repo.delete(skill_name)
        self._ops_applied.append({
            "name": "skill_delete",
            "arguments": {"skill_name": skill_name},
            "valid": success,
        })
        if success:
            return f"Skill '{skill_name}' deleted. Repo has {len(self._repo)} skills."
        return f"Failed to delete '{skill_name}'. Does not exist."

    # ---- Reward -------------------------------------------------------

    # Leading underscore so TRL's `inspect.getmembers` filter does NOT expose
    # this as a callable tool to the curator (TRL only excludes `reset` and
    # underscore-prefixed methods).
    def _finalize_reward(self, r_task: float) -> float:
        """Compose the reward once r_task (transfer-probe success) is known.

        r = r_task + r_fc + λ_u·r_cnt + λ_c·r_comp, on this rollout's own repo.
        Judge futures fired during tool-call time are awaited here; most are
        already resolved by the time we get here.
        """
        from skillos.rewards.composite import composite_reward, reward_compression, reward_function_call

        r_fc = reward_function_call(self._ops_applied)

        r_cnt = 0.0
        if self._judge_futures:
            # Per-future timeout: one stalled judge call must not block the
            # rank's reward gather (which would trip the NCCL watchdog on
            # peer ranks). Dropped scores are preferable to a multi-hour
            # job loss; the underlying resilient client still retries inside
            # its own budget — we only refuse to wait past _judge_timeout_s.
            #
            # We ALSO drop scores when the resilient client has exhausted its
            # retry budget and raises (e.g. an inference.sh outage: "no cloud
            # workers available" / "worker disconnected"). A transient infra
            # outage must degrade the reward signal for that rollout, not crash
            # an 8-rank multi-hour run. This mirrors the timeout path.
            scores = []
            for f in self._judge_futures:
                try:
                    scores.append(f.result(timeout=_judge_timeout_s))
                except concurrent.futures.TimeoutError:
                    print(
                        f"[curator_env] judge call timed out after "
                        f"{_judge_timeout_s:.0f}s; dropping its score",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[curator_env] judge call failed ({type(e).__name__}: {e}); "
                        f"dropping its score (likely infsh outage)",
                        file=sys.stderr,
                        flush=True,
                    )
            r_cnt = sum(scores) / len(scores) if scores else 0.0

        r_comp = reward_compression(self._repo.total_tokens(), self._input_tokens)

        self.reward = composite_reward(r_task, r_fc, r_cnt, r_comp)
        _on_rollout_complete(
            self._slot, self.reward, self._executor_result, self._ops_applied,
            r_task=r_task, num_skills=len(self._repo),
        )
        return self.reward

    # ---- Group dispatch internals -------------------------------------

    def _sentinel_trajectory(self, task_tag: str) -> dict:
        """Build a failed-trajectory sentinel and install it as this group's
        resolved future, so the other slots in the group short-circuit instead
        of each re-waiting (timeout) or re-raising (error) on the dead future.
        """
        result = _trajectory_result(task_description=task_tag)
        with _batch_lock:
            resolved: concurrent.futures.Future = concurrent.futures.Future()
            resolved.set_result(result)
            _group_trajectories[self._group_id] = resolved
        return result

    @property
    def _group_id(self) -> int:
        return self._slot // _num_generations

    def _get_or_submit_group_future(self) -> concurrent.futures.Future:
        """Return this group's trajectory future, submitting *all* groups' futures
        on the first reset() of a new batch.

        TRL drives env.reset() sequentially across slots, so submitting only the
        *current* group's future would re-serialize the trajectory rollouts. We
        instead eagerly submit one future per group the first time any reset()
        finds the batch empty — all groups overlap on the executor API.
        """
        with _batch_lock:
            if not _group_trajectories:
                pool = _get_rollout_pool()
                # How many groups in this batch? Round up just in case.
                num_groups = (len(_instances) + _num_generations - 1) // _num_generations
                for gid in range(num_groups):
                    _group_trajectories[gid] = pool.submit(
                        _run_seed_rollout, gid, _max_steps,
                    )
            return _group_trajectories[self._group_id]

    def _maybe_release_group(self) -> None:
        """Once all N slots in this group consumed the future, drop it."""
        gid = self._group_id
        with _batch_lock:
            _group_done_counts[gid] = _group_done_counts.get(gid, 0) + 1
            if _group_done_counts[gid] >= _num_generations:
                _group_trajectories.pop(gid, None)
                _group_done_counts.pop(gid, None)
