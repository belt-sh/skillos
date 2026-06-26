"""Algorithm 1 multi-position CuratorEnv (paper §3.1-3.2).

Single mega-tool per position: `curate_and_advance(operations)` is the only
tool the curator may call. Each call:
  1. applies its `operations` (list of {op, skill_name, content?, new_*?}) to
     this rollout's per-slot SkillRepo S
  2. advances position k -> k+1
  3. if k+1 < |G|: runs the frozen executor on the next task ξ_{k+1} with
     skills retrieved from the updated S, and returns the trajectory text as
     the tool response (this is the next position's curator input)
  4. if k+1 == |G|: terminates the rollout and returns a short final summary

r_task is computed at finalize as mean executor success over positions
2..|G| (position 1's executor ran on empty S so it can't reflect curator
contribution).

Group coordination: 8 slots in the same GRPO group share the same task
sequence (same `_group_id`). Within a group, slots run independent rollouts
(their own S, their own executor calls). Across groups, sequences differ.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time
from typing import Any

from skillos.curator.prompts import CURATOR_INPUT_TEMPLATE, format_trajectory
from skillos.skills.repo import SkillRepo

# --- Module-level state (mirrors envs.curator_env structure) ----------------

_instances: list["Algo1CuratorEnv"] = []
_num_generations: int = 8     # paper N=8; overridden by configure()
_group_size: int = 10         # paper |G|=10; overridden by configure()
_batch_lock = threading.Lock()

# Group-level task sequence registry: {group_id: [seed_int, ...] of length G}.
# All N slots in a group look up the same sequence so they walk identical
# ξ_1..ξ_G but build independent repos.
_group_sequences: dict[int, list[int]] = {}
_group_types: dict[int, str] = {}

# Judge client handle; populated by configure(). The executor is reused from
# the classic env via _run_probe, so algo1 holds no executor handle of its own.
_judge_submit = None  # callable: content_str -> Future[float]

# Phase budget across all of a step's position-1 executor calls; shared
# deadline so per-slot timeouts don't stack past the NCCL watchdog.
_phase_budget_s: float = float(os.environ.get("SKILLOS_PHASE_BUDGET_S", "1500"))
_executor_timeout_s: float = float(os.environ.get("SKILLOS_EXECUTOR_TIMEOUT_S", "900"))
_judge_timeout_s: float = float(os.environ.get("SKILLOS_JUDGE_TIMEOUT_S", "60"))


def configure(*, judge_submit, num_generations: int, group_size: int) -> None:
    """Wire the env to a configured judge and pin the GRPO/G sizes."""
    global _judge_submit, _num_generations, _group_size
    _judge_submit = judge_submit
    _num_generations = num_generations
    _group_size = group_size


# --- Helpers ---------------------------------------------------------------

def _build_curator_input(task_description: str, past_skills: str,
                         trajectory_text: str, success: bool) -> str:
    return CURATOR_INPUT_TEMPLATE.format(
        task_description=task_description,
        past_skills=past_skills,
        agent_trajectory=trajectory_text,
        result="Success" if success else "Failure",
    )


# --- Env -------------------------------------------------------------------

class Algo1CuratorEnv:
    """One rollout = one walk through |G| positions.

    TRL discovers the single tool method `curate_and_advance`. Reset returns
    position 0's curator input. Each tool call advances the position and
    returns the next position's input (or rollout-complete).
    """

    def __init__(self):
        self._slot = len(_instances)
        _instances.append(self)

        self._position: int = 0
        self._repo = SkillRepo()
        self._executor_results: list[dict] = []
        self._ops_applied: list[dict] = []           # all ops across all positions
        self._curator_outputs_meta: list[dict] = []  # per-position: ops_count, valid_count
        self._judge_futures: list[concurrent.futures.Future] = []
        self._input_tokens: int = 0

        # Per-rollout group-shared task sequence (populated on each reset from
        # the dataset row's group_id/task_type columns).
        self._task_seeds: list[int] = []
        self._task_descriptions: list[str] = []
        self._group_type: str = "pick"
        # Group identity MUST come from the dataset row (set in reset). TRL
        # creates env instances once and reuses them, so slot arithmetic
        # (slot // num_generations) degenerates to a constant — see
        # docs/postmortem-2026-06-10-algo1-group-collapse.md, bug 1.
        # NB: plain attribute, NOT a raising property — TRL's tool discovery
        # walks inspect.getmembers(env), which evaluates properties pre-reset.
        self._group_id: int | None = None

        self.reward = 0.0
        self.done = False

    # ---- TRL hooks ----------------------------------------------------

    def reset(self, **kwargs: Any) -> str:
        """Initialize a new rollout. Returns an instructional prompt for the
        first curator turn — the curator should emit `curate_and_advance`
        with an empty operations list to fetch the first task's trajectory.

        IMPORTANT: reset does NOT run the executor. TRL calls reset()
        serially per rank in a Python for-loop (16 rollouts × 12.5 min =
        ~200 min wasted as serial executor wall). Instead, position-0's
        executor runs in the FIRST `curate_and_advance` call, which TRL
        dispatches via `asyncio.gather` (concurrent across the rank's 16
        rollouts). This collapses per-step wall from ~33 h to ~2 h.
        """
        self._position = 0
        self._repo = SkillRepo()
        self._executor_results = []
        self._ops_applied = []
        self._curator_outputs_meta = []
        self._judge_futures = []
        self.reward = 0.0
        self.done = False

        # Synchronized rollout deadline. All rollouts in a step reset within a
        # quick serial loop, so per-rollout deadlines are ~aligned. Once a
        # rollout passes its deadline, remaining positions are CUT (executor
        # skipped, position masked from r_task — NOT scored as a failure), so
        # the rollout finishes fast and every rank reaches the post-_generate
        # NCCL gather within ~one in-flight episode of each other. Bounds
        # inter-rank skew to << the collective timeout, preventing the 4h-hang
        # SIGABRT that killed v8 at steps 59 and 54 (skew from slow composite
        # -verb groups whose every position maxed out the 900s episode cap).
        self._deadline = time.time() + _phase_budget_s

        group_id = kwargs.get("group_id")
        task_type = kwargs.get("task_type")
        if group_id is None or task_type is None:
            raise RuntimeError(
                "Algo1CuratorEnv.reset requires 'group_id' and 'task_type' "
                "dataset columns (build_dataset in scripts/train_algo1.py). "
                "TRL passes the full dataset row as reset kwargs; without "
                "these, group identity silently collapses — see "
                "docs/postmortem-2026-06-10-algo1-group-collapse.md."
            )
        self._group_id = int(group_id)
        self._group_type = str(task_type)

        # Sample the group's task sequence (shared across N slots in group).
        self._ensure_group_sequence()

        curator_input = (
            "## Session Start\n"
            f"You have {_group_size} ALFWorld tasks to curate skills for. "
            "Call `curate_and_advance` with `operations=[]` to receive the "
            "first task's executor trajectory. After each tool response, "
            "emit one `curate_and_advance` call with curation operations "
            "(insert/update/delete) based on what helped or hurt the "
            "executor at that task."
        )
        self._input_tokens = len(curator_input.split())
        return curator_input

    async def curate_and_advance(self, operations: list[dict]) -> str:
        """Apply curation ops for this position, then advance to the next task.

        Async so TRL's `_tool_call_loop` dispatches one call per rollout via
        `asyncio.gather` instead of a serial Python for-loop. With sync tools
        TRL would process this rank's 16 rollouts one-at-a-time, multiplying
        the per-iteration wall by 16× (~33 h/opt step). With async + threaded
        executor calls below, all rollouts on a rank fan out concurrently and
        the iteration wall collapses to max(per-rollout-time).

        Args:
            operations: List of curation operations to apply this position.
                Each item is an object with fields: `op` (one of "insert",
                "update", "delete"), `skill_name` (string), and optional
                fields `content` (insert), `new_name`/`new_content` (update).
                Pass an empty list to advance without changing the repo.

        Returns:
            The next position's curator input as a string, or a
            rollout-complete marker once all G positions have been played.
        """
        # State machine: self._position counts executor episodes already
        # COMPLETED on this rollout (0..|G|). Each call to this method
        # finishes one more episode. Reset has run zero episodes; the first
        # tool call runs position 0 (so the model can see task 0's
        # trajectory); the last (Gth) call runs position G-1.

        # 1) Apply this position's curation ops to S. Empty/no-op on the
        # very first call (model hasn't seen any task yet).
        valid_count = 0
        for op in operations or []:
            applied_valid = self._apply_op(op)
            valid_count += int(applied_valid)
        self._curator_outputs_meta.append({
            "position": self._position,
            "ops": len(operations or []),
            "valid": valid_count,
        })

        # 2) Check if all G episodes have completed.
        if self._position >= _group_size:
            self.done = True
            return (
                f"[rollout complete] {self._position}/{_group_size} tasks "
                f"played. final repo size = {len(self._repo)} skills."
            )

        # 2b) Synchronized deadline: if this rollout has run past its budget,
        # CUT the remaining positions — skip the executor entirely (no 900s
        # wait) and record a masked result (cut=True, success=None) so r_task
        # excludes it rather than counting a false failure. Keeps ranks in
        # lockstep at the NCCL gather. The curator still emits ops for cut
        # positions, so r_fc / r_cnt (curation-quality terms) keep their
        # honest gradient — only the executor-outcome term is masked.
        import time as _time
        if _time.time() > self._deadline:
            print(f"[algo1] DEADLINE CUT slot={self._slot} group={self._group_id} "
                  f"position={self._position} — skipping executor (masked from r_task)",
                  file=sys.stderr, flush=True)
            self._executor_results.append({
                "task_description": f"<cut-position-{self._position}>",
                "trajectory": [], "success": None, "cut": True,
                "steps": 0, "gamefile": "", "skills_text": "",
            })
            self._position += 1
            return (
                f"[deadline reached] position {self._position}/{_group_size} cut — "
                "emit your final curation operations; no more executor feedback."
            )

        # 3) Run executor at the next position. Two reasons for the
        # asyncio wrapping:
        #   (a) `asyncio.to_thread` makes this coroutine yield during the
        #       blocking infsh I/O so TRL's gather can interleave all 16
        #       rollouts on a rank concurrently.
        #   (b) `asyncio.wait_for(..., timeout=_executor_timeout_s)` caps
        #       per-rollout episode wall so a single slow game can't make
        #       this rank lag far enough behind peers to trip the 30-min
        #       NCCL collective watchdog at the post-_generate gather. On
        #       timeout we return a sentinel trajectory and proceed —
        #       cumulative inter-rank skew stays bounded across the G
        #       positions of the rollout.
        import asyncio
        position = self._position
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_executor_at, position),
                timeout=_executor_timeout_s,
            )
        except asyncio.TimeoutError:
            print(f"[algo1] executor timeout at slot={self._slot} group="
                  f"{self._group_id} position={position} (budget="
                  f"{_executor_timeout_s:.0f}s) — using sentinel trajectory",
                  file=sys.stderr, flush=True)
            result = {
                "task_description": f"<timeout-position-{position}>",
                "trajectory": [],
                "success": False,
                "steps": 0,
                "gamefile": "",
                "skills_text": "",
            }
        self._executor_results.append(result)
        self._position += 1

        curator_input = _build_curator_input(
            task_description=result["task_description"],
            past_skills=result.get("skills_text", ""),
            trajectory_text=format_trajectory(result.get("trajectory", [])),
            success=bool(result.get("success")),
        )
        # Accumulate |χ| across positions so r_comp = 1 - |S|/|χ| compares the
        # repo against the full rollout input, not just the ~50-word session
        # prompt (which made r_comp reward near-empty repos).
        self._input_tokens += len(curator_input.split())
        return curator_input

    # ---- Op application -----------------------------------------------

    def _apply_op(self, op: dict) -> bool:
        """Apply one curation op to self._repo, record for r_fc, fire judge.
        Returns True if the op was valid + executed."""
        if not isinstance(op, dict):
            self._ops_applied.append({"name": "invalid", "valid": False})
            return False
        kind = (op.get("op") or "").lower()
        name = op.get("skill_name", "")
        if kind == "insert":
            content = op.get("content", "")
            ok = self._repo.insert(name, content) if (name and content) else False
            self._ops_applied.append({
                "name": "new_skill_insert",
                "arguments": {"skill_name": name, "content": content},
                "valid": ok,
            })
            if ok and content and _judge_submit is not None:
                self._judge_futures.append(_judge_submit(content))
            return ok
        if kind == "update":
            new_name = op.get("new_name") or None
            new_content = op.get("new_content") or None
            ok = self._repo.update(name, new_name=new_name, new_content=new_content) if name else False
            self._ops_applied.append({
                "name": "skill_update",
                "arguments": {"skill_name": name, "new_content": new_content or ""},
                "valid": ok,
            })
            if ok and new_content and _judge_submit is not None:
                self._judge_futures.append(_judge_submit(new_content))
            return ok
        if kind == "delete":
            ok = self._repo.delete(name) if name else False
            self._ops_applied.append({
                "name": "skill_delete",
                "arguments": {"skill_name": name},
                "valid": ok,
            })
            return ok
        self._ops_applied.append({"name": f"unknown:{kind}", "valid": False})
        return False

    # ---- Group sequence sampling --------------------------------------

    def _ensure_group_sequence(self) -> None:
        """Populate `self._task_seeds` for the |G| tasks this group walks
        through. Cached per group_id so the N slots in a GRPO group share an
        identical sequence."""
        # Reuse the existing same-type seed index from envs.curator_env so we
        # don't duplicate the one-time 400-seed ALFWorld bucketing scan.
        # NB: access via the module (not `from ... import _type_seeds`)
        # because `_build_type_seed_index` REASSIGNS the module attribute and
        # a `from`-import would snapshot the old empty dict.
        from skillos.envs import curator_env as _classic
        if not _classic._type_seeds:
            _classic._build_type_seed_index()

        from skillos.algo1.data import sample_group_seeds

        class _SeedIndexAdapter:
            def seeds_for_type(self, t: str) -> list[int]:
                return _classic._type_seeds.get(t) or _classic._type_seeds.get("pick") or []

        gid = self._group_id
        ttype = self._group_type
        with _batch_lock:
            if gid in _group_sequences:
                self._task_seeds = list(_group_sequences[gid])
            else:
                self._task_seeds = sample_group_seeds(
                    group_id=gid, task_type=ttype, group_size=_group_size,
                    seed_index=_SeedIndexAdapter())
                _group_sequences[gid] = list(self._task_seeds)
                _group_types[gid] = ttype
        self._task_descriptions = [""] * _group_size
        # Per-rollout task observability (postmortem guardrail): a degenerate
        # task distribution must be visible in the first step's log.
        print(f"[algo1] rollout slot={self._slot} gid={gid} type={ttype} "
              f"seeds={self._task_seeds}", flush=True)

    # ---- Executor invocation ------------------------------------------

    def _run_executor_at(self, position: int) -> dict:
        """Run frozen executor on this group's task at `position`. Reuses
        _run_probe from envs.curator_env, which handles env borrow/return,
        retrieval from self._repo, and the per-step executor loop."""
        from skillos.envs.curator_env import _run_probe, _max_steps
        seed = self._task_seeds[position]
        try:
            result = _run_probe(self._repo, None, _max_steps, seed)
        except Exception as e:
            print(f"[algo1] executor failed slot={self._slot} group={self._group_id} "
                  f"position={position}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            result = {
                "task_description": f"<executor-error-position-{position}>",
                "trajectory": [], "success": False, "steps": 0,
                "gamefile": "", "skills_text": "",
            }
        self._task_descriptions[position] = result.get("task_description", "")
        return result

    # ---- Reward -------------------------------------------------------

    def _finalize_reward(self) -> float:
        """Compose r = r_task + r_fc + λ_u·r_cnt + λ_c·r_comp.

        r_task = mean executor success over positions 2..|G|. The seed
        position (index 0) is excluded — it ran with empty S and credits no
        curator action."""
        from skillos.rewards.composite import (
            composite_reward, reward_compression, reward_function_call,
        )

        # r_task — mean over positions 2..|G| (1-indexed in the paper). We
        # use 1..G-1 here (0-indexed). Position 0 ran on empty S, so a rollout
        # that quit before any informed position earns NO task reward — falling
        # back to position 0's success would pay the curator for the
        # executor's empty-repo baseline.
        # Exclude position 0 (empty repo) AND deadline-cut positions: a cut
        # never ran the executor, so counting it as a failure would bias the
        # curator against slow task types. Masking = average over positions
        # that actually ran.
        tail = [r for r in self._executor_results[1:] if not r.get("cut")]
        if not tail:
            r_task = 0.0
        else:
            successes = [float(r.get("success") or 0.0) for r in tail]
            r_task = sum(successes) / len(successes)

        r_fc = reward_function_call(self._ops_applied)

        r_cnt = 0.0
        if self._judge_futures:
            scores = []
            for f in self._judge_futures:
                try:
                    scores.append(f.result(timeout=_judge_timeout_s))
                except concurrent.futures.TimeoutError:
                    print(f"[algo1] judge timeout after {_judge_timeout_s:.0f}s; dropping score",
                          file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[algo1] judge failed ({type(e).__name__}: {e}); dropping score",
                          file=sys.stderr, flush=True)
            r_cnt = sum(scores) / len(scores) if scores else 0.0

        r_comp = reward_compression(self._repo.total_tokens(), max(self._input_tokens, 1))

        self.reward = composite_reward(r_task, r_fc, r_cnt, r_comp)
        return self.reward
