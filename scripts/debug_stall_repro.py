"""Minimal repro to isolate whether multi-rank NCCL timeouts originate from
unbounded `Future.result()` calls in `skillos.envs.curator_env` or from
something deeper in the env machinery.

Two scenarios run back-to-back on a single process:

  A) Heuristic executor + heuristic judge. Instant returns. Baseline:
     a healthy step should clear `in_flight` back to 0 within seconds.
  B) Same, but one judge call is replaced with a 90s sleep. Mirrors what
     a single hung infsh call would do during real training.

Expected outcomes:
  * A finishes 5 steps in < 30s; final in_flight == 0.
  * B *also* finishes the step (because we deliberately wait), but the
    step takes >= 90s — proof that *one stuck judge future* blocks the
    entire rank's reward gather. In real training that's the trigger
    for NCCL's 30-min watchdog.

Run:
  source .venv/bin/activate && python scripts/debug_stall_repro.py
"""
from __future__ import annotations

import os
import time

# Force ALFWorld off the disk path entirely — we don't need a real game env
# to exercise the rollout-pool / judge-pool / future plumbing. The stub
# executor never calls env.step(); we patch _run_one_rollout to skip alfworld.
os.environ.setdefault("ALFWORLD_DATA", "/tmp/__skip__")

import skillos.envs.curator_env as ce
from skillos.executor.executor import HeuristicExecutor
from skillos.rewards.judge import HeuristicJudge, Judge


# ----- stubs -----------------------------------------------------------------

def _stub_run_one_rollout(group_id: int, max_steps: int) -> dict:
    # Skip real alfworld; pretend the executor finished a trivial 3-step traj.
    return {
        "trajectory": [
            {"step": 1, "action": "look", "observation": "stub-obs-1"},
            {"step": 2, "action": "go north", "observation": "stub-obs-2"},
            {"step": 3, "action": "take key", "observation": "stub-obs-3"},
        ],
        "success": True,
        "steps": 3,
        "task_description": f"stub task g{group_id}",
        "skills_text": "",
    }


class SlowOnceJudge(Judge):
    """Heuristic judge, but the Nth call sleeps `delay` seconds."""

    def __init__(self, delay: float, slow_call_index: int):
        self._inner = HeuristicJudge()
        self._delay = delay
        self._slow_at = slow_call_index
        self._calls = 0

    def score(self, content: str) -> float:
        idx = self._calls
        self._calls += 1
        if idx == self._slow_at:
            print(f"   [SlowOnceJudge] call {idx} sleeping {self._delay}s")
            time.sleep(self._delay)
        return self._inner.score(content)


# ----- harness ---------------------------------------------------------------

def _drive_one_step(envs, step_idx: int) -> float:
    """Mimic TRL: reset all envs sequentially, fire a skill op per env,
    then call _compute_reward on each. Return wall time."""
    # Each "step" is a fresh batch — clear the group dispatcher state.
    ce._group_trajectories.clear()
    ce._group_done_counts.clear()

    t0 = time.time()
    for env in envs:
        env.reset()
    for env in envs:
        # Insert a unique skill so the judge cache doesn't dedupe to 0 calls.
        name = f"stub_skill_step{step_idx}_slot{env._slot}"
        body = (
            "---\n"
            f"name: {name}\n"
            "description: stub for repro\n"
            "---\n"
            "# Stub\n"
            "Some body text with enough words to clear the heuristic floor "
            "and exercise the judge call path end-to-end."
        )
        env.new_skill_insert(name, body)
    for env in envs:
        env._compute_reward()
    dt = time.time() - t0
    print(
        f"  step {step_idx}: dt={dt:5.1f}s  in_flight={ce._rollouts_inflight}  "
        f"oldest={(time.time()-ce._oldest_inflight_started_at) if ce._oldest_inflight_started_at else 0:.1f}s"
    )
    return dt


def _setup(num_envs: int, num_generations: int) -> list:
    from skillos.skills.repo import SkillRepo
    ce._executor = HeuristicExecutor()
    ce._num_generations = num_generations
    # Patch the rollout body so we don't touch ALFWorld.
    ce._run_one_rollout = _stub_run_one_rollout
    # Reset instance list, dispatcher, *and* skill repo so re-runs start clean
    # (otherwise scenario B's inserts collide with A's and the judge never fires).
    ce._instances.clear()
    ce._group_trajectories.clear()
    ce._group_done_counts.clear()
    ce._judge_cache.clear()
    ce._shared_skill_repo = SkillRepo()
    ce._rollouts_inflight = 0
    ce._rollouts_completed = 0
    ce._oldest_inflight_started_at = None
    ce.set_step_expected_rollouts(num_envs)
    return [ce.CuratorEnv() for _ in range(num_envs)]


def scenario_a_baseline():
    print("\n=== Scenario A: heuristic executor + heuristic judge (BASELINE) ===")
    ce._judge = HeuristicJudge()
    envs = _setup(num_envs=32, num_generations=8)
    total = 0.0
    for s in range(5):
        total += _drive_one_step(envs, s)
    print(f"  total wall time for 5 steps: {total:.1f}s")
    print(f"  final in_flight: {ce._rollouts_inflight} (expect 0)")
    return total


def scenario_b_one_stuck_judge():
    print("\n=== Scenario B: same, but judge call #0 sleeps 90s ===")
    ce._judge = SlowOnceJudge(delay=90.0, slow_call_index=0)
    envs = _setup(num_envs=32, num_generations=8)
    total = 0.0
    for s in range(2):
        total += _drive_one_step(envs, s)
    print(f"  total wall time for 2 steps: {total:.1f}s")
    print(f"  final in_flight: {ce._rollouts_inflight}")
    return total


if __name__ == "__main__":
    a = scenario_a_baseline()
    b = scenario_b_one_stuck_judge()
    print("\n--- Verdict ---")
    print(f"  A (no stall): {a:5.1f}s for 5 steps")
    print(f"  B (1 judge slept 90s): {b:5.1f}s for 2 steps")
    if b > 60 and a < 30:
        print("  >>> One unbounded judge future stalls the *whole* step.")
        print("  >>> Fix: add a timeout to `f.result()` in CuratorEnv._compute_reward")
        print("      and treat timeout as zero-content (drop the score), not as a crash.")
