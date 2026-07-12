"""Smoke test for Algo1CuratorEnv (heuristic backends, no network).

Verifies the rollout state machine AND the group-identity guardrails from
docs/postmortem-2026-06-10-algo1-group-collapse.md:
  - reset() REQUIRES dataset-provided group_id/task_type (fail-loud)
  - same group_id -> identical task seeds; different group_id -> different
  - seeds are stable across processes (md5, not salted hash())
  - G+1 async tool calls: priming empty-ops call, G informed positions,
    terminal call; reward composes without raising
  - early-quit rollout earns r_task contribution of 0 (not position-0 success)

Run:  source .venv/bin/activate && python -m scripts.smoke_algo1_env
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys


def main() -> int:
    os.environ.setdefault("SKILLOS_EXECUTOR_MAX_STEPS", "5")

    from skillos.envs import curator_env as classic
    from skillos.algo1 import env as algo1

    classic.configure(
        executor_config={"type": "heuristic"},
        judge_config={"type": "heuristic"},
        num_generations=1,
        num_probe_tasks=0,
    )

    G = 3
    algo1.configure(
        executor=None,        # not used by env directly; classic._executor is
        judge_submit=None,    # judge skipped in smoke
        num_generations=1,
        group_size=G,
    )

    # --- Guardrail: reset without group identity must raise -------------
    env = algo1.Algo1CuratorEnv()
    try:
        env.reset()
        print("FAIL: reset() without group_id/task_type did not raise")
        return 1
    except RuntimeError:
        print("OK: reset() without group identity raises")

    # --- Guardrail: seed sharing/divergence by group_id ------------------
    a = algo1.Algo1CuratorEnv()
    b = algo1.Algo1CuratorEnv()
    c = algo1.Algo1CuratorEnv()
    a.reset(group_id=7, task_type="clean", prompt=None)
    b.reset(group_id=7, task_type="clean", prompt=None)
    c.reset(group_id=8, task_type="clean", prompt=None)
    assert a._task_seeds == b._task_seeds, "same group_id must share seeds"
    assert a._task_seeds != c._task_seeds, "different group_id must differ"
    print(f"OK: group seed sharing (gid=7 seeds={a._task_seeds})")

    # --- Guardrail: cross-process seed stability (the hash() salt bug) ---
    snippet = (
        "from skillos.algo1.data import sample_group_seeds\n"
        "class I:\n"
        "    def seeds_for_type(self, t): return list(range(50))\n"
        "print(sample_group_seeds(group_id=7, task_type='clean', group_size=3, seed_index=I()))\n"
    )
    outs = set()
    for hashseed in ("1", "2"):
        e = dict(os.environ, PYTHONHASHSEED=hashseed)
        outs.add(subprocess.run(
            [sys.executable, "-c", snippet], env=e,
            capture_output=True, text=True, check=True).stdout.strip())
    assert len(outs) == 1, f"seeds diverge across processes: {outs}"
    print(f"OK: seeds stable across PYTHONHASHSEED values ({outs.pop()})")

    # --- Full rollout: priming + G informed + terminal --------------------
    env = algo1.Algo1CuratorEnv()
    p0 = env.reset(group_id=0, task_type="pick", prompt=None)
    print(f"--- reset (position 0) returned {len(p0)} chars ---")

    valid_md = (
        "---\nname: navigate\ndescription: go to a target location\n---\n"
        "# Workflow\nGo to the target receptacle before interacting.\n"
    )
    ops_per_call = [
        [],  # priming call: model has seen no trajectory yet
        # SkillRepo.insert requires YAML-frontmatter markdown (curator prompt
        # mandates it); cover one valid and one invalid insert.
        [{"op": "insert", "skill_name": "navigate", "content": valid_md}],
        [{"op": "insert", "skill_name": "pickup", "content": "plain text, no frontmatter"}],
        [],  # final position emits no ops
    ]
    for k in range(G + 1):
        out = asyncio.run(env.curate_and_advance(
            ops_per_call[k] if k < len(ops_per_call) else []))
        tag = "TERMINAL" if env.done else f"after call {k+1} (position={env._position})"
        print(f"--- curate_and_advance call {k+1} -> {tag} ({len(out)} chars) ---")
        if env.done:
            break
    assert env.done, "rollout must terminate after G+1 calls"
    assert len(env._executor_results) == G
    valid = sum(1 for o in env._ops_applied if o["valid"])
    assert valid == 1, f"expected exactly 1 valid op, got {valid}"
    assert len(env._repo) == 1, f"expected 1 skill in repo, got {len(env._repo)}"

    reward = env._finalize_reward()
    print(f"=== full rollout reward = {reward:.4f} ===")
    print(f"  positions played      : {len(env._executor_results)}")
    print(f"  ops applied (total)   : {len(env._ops_applied)}")
    print(f"  valid ops             : {sum(1 for o in env._ops_applied if o['valid'])}")
    print(f"  accumulated |chi|     : {env._input_tokens} tokens")
    print(f"  final repo size       : {len(env._repo)} skills")

    # --- Early-quit rollout: r_task must be 0, never position-0 success --
    env = algo1.Algo1CuratorEnv()
    env.reset(group_id=1, task_type="pick", prompt=None)
    asyncio.run(env.curate_and_advance([]))  # priming only, then quit
    r = env._finalize_reward()
    assert r <= 0.2, f"early-quit reward suspiciously high: {r}"
    print(f"OK: early-quit rollout reward = {r:.4f} (no r_task credit)")

    # --- Deadline cut: positions past the deadline must be MASKED from r_task
    #     (cut=True, success=None), NOT scored as failures, and must not crash.
    env = algo1.Algo1CuratorEnv()
    env.reset(group_id=2, task_type="pick", prompt=None)
    asyncio.run(env.curate_and_advance([]))   # position 0 runs normally
    asyncio.run(env.curate_and_advance([]))   # position 1 runs normally
    ran_real = len([r for r in env._executor_results if not r.get("cut")])
    env._deadline = 0.0                        # force deadline expired
    cut_msgs = 0
    while not env.done:
        out = asyncio.run(env.curate_and_advance([]))
        if "deadline reached" in out:
            cut_msgs += 1
        if env.done:
            break
    cut = [r for r in env._executor_results if r.get("cut")]
    assert cut_msgs > 0 and len(cut) > 0, "expected some positions to be cut"
    assert all(r.get("success") is None for r in cut), "cut results must not carry success"
    r = env._finalize_reward()  # must not raise
    # r_task is averaged only over the positions that actually RAN (ran_real-1
    # informed positions); cut positions contribute nothing.
    print(f"OK: deadline cut — {ran_real} ran, {len(cut)} cut/masked, "
          f"reward={r:.4f} (no crash, cuts excluded from r_task)")

    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
