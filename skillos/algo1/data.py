"""Group-of-G task sampling for paper Algorithm 1.

A "group" = a sequence of G ALFWorld tasks ξ_1..ξ_G that is shared by all N
rollouts in a GRPO group. Paper §3.1 says "10 related tasks per group" but
doesn't fully spell out "related." We interpret this as **same task type**
(Pick, Clean, Heat, Cool, Look, Pick2) — the curator must produce skills
that transfer within a task type. Mixed-type variants can be wired later if
the same-type sampler turns out to be too narrow.

Deterministic per (group_id, task_type): the same group_id always yields
the same G task seeds so all N slots in a GRPO group walk an identical
ξ_1..ξ_G sequence. Seeds come from the existing AlfworldSeedIndex used in
[[skillos-transfer-probe-fix]].
"""

from __future__ import annotations

import random
from typing import List


def sample_group_seeds(
    group_id: int,
    task_type: str,
    group_size: int,
    seed_index,
) -> List[int]:
    """Return G seeds of the requested task_type, deterministic per group_id."""
    rng = random.Random(group_id * 1_000_003 + hash(task_type))
    pool = list(seed_index.seeds_for_type(task_type))
    if not pool:
        raise ValueError(f"no seeds for task_type={task_type!r}")
    # With replacement is fine if the pool is smaller than G; for ALFWorld
    # train split this only matters for rare types (Heat=16 seeds < 10? -
    # train pool is larger than the eval slice).
    if len(pool) >= group_size:
        return rng.sample(pool, group_size)
    out = list(pool)
    while len(out) < group_size:
        out.append(rng.choice(pool))
    return out
