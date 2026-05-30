"""SkillOS paper Algorithm 1 — multi-position CuratorEnv for TRL GRPO.

Each rollout iterates through |G|=10 related ALFWorld tasks. At position k:
executor solves ξ_k with retrieved skills from current S, curator emits one
`curate_and_advance(operations)` call, S ← apply(ops, S), advance to k+1.

Single mega-tool per position (instead of paper's three separate tools)
because TRL's `_tool_call_loop` counts iterations not positions — one
iteration <-> one position keeps the rollout structure clean. Curator is
retrained from scratch on this schema.

8 rollouts per GRPO group sample independent curator outputs but share the
same ξ_1..ξ_G sequence (paper §3.2, "data grouping size 10"). r_task =
mean success over positions 2..|G|: position 1 sees empty S, so the
gradient credits skills that help LATER positions — that's where
transferable/general curations are rewarded.
"""

from skillos.algo1.env import Algo1CuratorEnv, configure  # noqa: F401
