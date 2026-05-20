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
import os
import threading
from collections import deque

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
_num_generations = 1  # set via configure()

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


# ---------------------------------------------------------------------------
# Public configuration entrypoint
# ---------------------------------------------------------------------------

def configure(executor_config: dict | None = None,
              judge_config: dict | None = None,
              num_generations: int | None = None) -> None:
    """Configure executor / judge / GRPO group size. Call before training."""
    global _executor, _judge, _num_generations
    if executor_config:
        _executor = create_executor(executor_config)
    if judge_config:
        _judge = create_judge(judge_config)
    if num_generations is not None:
        _num_generations = max(1, int(num_generations))


def reset_shared_state() -> None:
    """Reset shared state at the start of a new task group (paper §3.2.1)."""
    global _shared_skill_repo
    _shared_skill_repo = SkillRepo()


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

def _run_one_rollout(group_id: int, max_steps: int) -> dict:
    """Reset group_id's alfworld env and run the frozen executor through it.

    Runs in a worker thread. Executor calls (network) happen outside the
    alfworld lock so groups overlap on the slow part.
    """
    env = _get_alfworld_env_for_group(group_id)
    with _alfworld_global_lock:
        obs, infos = env.reset()
        observation = obs[0]
        admissible_actions = infos.get("admissible_commands", [[]])[0]
        task_description = _extract_task_description(observation)

    # Skill retrieval reads only — safe outside the lock.
    retrieved = _shared_skill_repo.retrieve(task_description, top_k=5)
    skills_text = _shared_skill_repo.format_skills(retrieved)

    trajectory: list = []
    recent_history: deque[str] = deque(maxlen=3)
    success = False
    done = False
    step = 0

    while not done and step < max_steps:
        # Network I/O — concurrent across groups.
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

    return {
        "trajectory": trajectory,
        "success": success,
        "steps": step,
        "task_description": task_description,
        "skills_text": skills_text,
    }


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
        _instances.append(self)
        self._slot = len(_instances) - 1

    # ---- TRL hook -----------------------------------------------------

    def reset(self, **kwargs) -> str:
        """Run frozen executor (once per group), return trajectory for curator."""
        self.reward = 0.0
        self.done = False
        self._ops_applied = []
        self._judge_futures = []

        future = self._get_or_submit_group_future()
        result = future.result()
        self._executor_result = result

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
        success = _shared_skill_repo.insert(skill_name, content)
        self._ops_applied.append({
            "name": "new_skill_insert",
            "arguments": {"skill_name": skill_name, "content": content},
            "valid": success,
        })
        if success and content:
            # Fire judge async right now; we'll await in compute_reward.
            self._judge_futures.append(_submit_judge(content))
        if success:
            return f"Skill '{skill_name}' created. Repo has {len(_shared_skill_repo)} skills."
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
        success = _shared_skill_repo.update(
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
        success = _shared_skill_repo.delete(skill_name)
        self._ops_applied.append({
            "name": "skill_delete",
            "arguments": {"skill_name": skill_name},
            "valid": success,
        })
        if success:
            return f"Skill '{skill_name}' deleted. Repo has {len(_shared_skill_repo)} skills."
        return f"Failed to delete '{skill_name}'. Does not exist."

    # ---- Reward -------------------------------------------------------

    # Leading underscore so TRL's `inspect.getmembers` filter does NOT expose
    # this as a callable tool to the curator (TRL only excludes `reset` and
    # underscore-prefixed methods).
    def _compute_reward(self) -> float:
        """Compute composite reward: r_task + r_fc + r_cnt + r_comp.

        Judge futures fired during tool-call time are awaited here; most are
        already resolved by the time we get here.
        """
        from skillos.rewards.composite import composite_reward, reward_compression, reward_function_call

        r_task = 1.0 if self._executor_result and self._executor_result["success"] else 0.0
        r_fc = reward_function_call(self._ops_applied)

        r_cnt = 0.0
        if self._judge_futures:
            # Let exceptions propagate — bad judge data is worse than a
            # crash + checkpoint resume. The resilient client should have
            # already retried up to ~100 min before giving up.
            scores = [f.result() for f in self._judge_futures]
            r_cnt = sum(scores) / len(scores) if scores else 0.0

        r_comp = reward_compression(_shared_skill_repo.total_tokens(), self._input_tokens)

        self.reward = composite_reward(r_task, r_fc, r_cnt, r_comp)
        return self.reward

    # ---- Group dispatch internals -------------------------------------

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
                        _run_one_rollout, gid, _max_steps,
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
