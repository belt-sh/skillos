"""Phase A smoke test for Algo1CuratorEnv.

Configures the classic env primitives (executor pool, judge stub) with
heuristic backends, then drives a multi-position rollout through the new
Algorithm 1 env. Verifies:
  - reset() returns a curator input for position 0
  - curate_and_advance applies ops, advances position, returns next input
  - rollout terminates after |G| positions
  - _finalize_reward composes a numeric reward without raising

Run:  source .venv/bin/activate && python -m scripts.smoke_algo1_env
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Tiny G so the test runs in seconds with the heuristic executor.
    os.environ.setdefault("SKILLOS_EXECUTOR_MAX_STEPS", "5")

    from skillos.envs import curator_env as classic
    from skillos.algo1 import env as algo1

    # 1) Wire the classic primitives that algo1's env reuses (executor pool,
    # ALFWorld env factory, judge stub). Heuristic backends — no network.
    classic.configure(
        executor_config={"type": "heuristic"},
        judge_config={"type": "heuristic"},
        num_generations=1,
        num_probe_tasks=0,
    )

    # 2) Pin Algorithm 1 hyperparams for the smoke.
    G = 3
    algo1.configure(
        executor=None,        # not used by env directly; classic._executor is
        judge_submit=None,    # judge skipped in smoke
        num_generations=1,
        group_size=G,
    )

    # 3) Drive one full rollout.
    env = algo1.Algo1CuratorEnv()
    p0 = env.reset()
    print(f"--- reset (position 0) returned {len(p0)} chars ---")
    print(p0[:400] + ("...[truncated]" if len(p0) > 400 else ""))
    print()

    ops_per_position = [
        [{"op": "insert", "skill_name": "navigate", "content": "go to target location"}],
        [{"op": "insert", "skill_name": "pickup", "content": "take object from location"}],
        [],  # third position emits no ops — env should still advance/terminate
    ]
    for k in range(G):
        out = env.curate_and_advance(ops_per_position[k] if k < len(ops_per_position) else [])
        tag = "TERMINAL" if env.done else f"position {k+1}"
        print(f"--- curate_and_advance call {k+1} -> {tag} ({len(out)} chars) ---")
        print(out[:400] + ("...[truncated]" if len(out) > 400 else ""))
        print()
        if env.done:
            break

    reward = env._finalize_reward()
    print(f"=== final reward = {reward:.4f} ===")
    print(f"  positions played      : {len(env._executor_results)}")
    print(f"  ops applied (total)   : {len(env._ops_applied)}")
    print(f"  valid ops             : {sum(1 for o in env._ops_applied if o['valid'])}")
    print(f"  per-position meta     : {env._curator_outputs_meta}")
    print(f"  final repo size       : {len(env._repo)} skills")
    return 0


if __name__ == "__main__":
    sys.exit(main())
