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

import hashlib
import json
import os
import random
from typing import Callable, List, Optional

# Paper Table 5: within-group easy->hard curriculum applies the ascending step
# with probability p_up = 0.80 (else a random remaining task), so groups are
# mostly-sorted rather than strictly sorted.
CURRICULUM_P_UP = 0.80

_difficulty_cache: dict[str, int] = {}


def gamefile_difficulty(gamefile: str) -> int:
    """Expert-plan length (number of high-level PDDL subgoals) from the
    traj_data.json next to the gamefile — our difficulty proxy for the paper's
    easy->hard curriculum. Unknown/missing -> a middling constant so bad paths
    neither sink nor float."""
    if gamefile in _difficulty_cache:
        return _difficulty_cache[gamefile]
    d = 5
    try:
        with open(os.path.join(os.path.dirname(gamefile), "traj_data.json")) as f:
            d = len(json.load(f)["plan"]["high_pddl"])
    except Exception:
        pass
    _difficulty_cache[gamefile] = d
    return d


def order_easy_to_hard(seeds: List[int], difficulty: Callable[[int], int],
                       rng: random.Random) -> List[int]:
    """Soft easy->hard ordering: at each position take the easiest remaining
    task with p=CURRICULUM_P_UP, else a random remaining one. Deterministic
    given `rng` (which is keyed per group), so all N rollouts of a GRPO group
    still share an identical sequence."""
    remaining = sorted(seeds, key=difficulty)
    out: List[int] = []
    while remaining:
        i = 0 if rng.random() < CURRICULUM_P_UP else rng.randrange(len(remaining))
        out.append(remaining.pop(i))
    return out


def _stable_hash(s: str) -> int:
    # NOT builtin hash(): Python salts str hashes per process, so each of the
    # 8 accelerate ranks would derive a different RNG seed and group members
    # sharded across ranks would walk different task sequences (postmortem
    # 2026-06-10, bug 2).
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def sample_group_seeds(
    group_id: int,
    task_type: str,
    group_size: int,
    seed_index,
    curriculum: bool = False,
    difficulty: Optional[Callable[[int], int]] = None,
) -> List[int]:
    """Return G seeds of the requested task_type, deterministic per group_id.

    With `curriculum=True` (paper Table 5), the sampled seeds are re-ordered
    soft easy->hard using `difficulty(seed)`. curriculum=False is byte-identical
    to the pre-curriculum behavior.
    """
    rng = random.Random(group_id * 1_000_003 + _stable_hash(task_type))
    pool = list(seed_index.seeds_for_type(task_type))
    if not pool:
        raise ValueError(f"no seeds for task_type={task_type!r}")
    # With replacement is fine if the pool is smaller than G; for ALFWorld
    # train split this only matters for rare types (Heat=16 seeds < 10? -
    # train pool is larger than the eval slice).
    if len(pool) >= group_size:
        out = rng.sample(pool, group_size)
    else:
        out = list(pool)
        while len(out) < group_size:
            out.append(rng.choice(pool))
    if curriculum:
        if difficulty is None:
            raise ValueError("curriculum=True requires a difficulty function")
        out = order_easy_to_hard(out, difficulty, rng)
    return out
