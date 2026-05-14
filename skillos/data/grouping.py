"""Grouped task stream construction (Section 3.2.1, Appendix B.2).

For ALFWorld: uses the 6 built-in task types as natural groups.
For reasoning: uses attribute annotation + soft-Jaccard similarity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TaskGroup:
    """A group of related tasks for training."""
    tasks: list[dict]  # each dict has at minimum {"id": str, "description": str}
    group_type: str  # e.g. "pick", "clean", "heat" for ALFWorld


def group_alfworld_tasks(
    tasks: list[dict],
    group_size: int = 8,
    seed: int = 42,
) -> list[TaskGroup]:
    """Group ALFWorld tasks by task type (6 types: pick, look, clean, heat, cool, pick2).

    ALFWorld has natural task type annotations — the paper uses these directly
    instead of the attribute annotation pipeline (Appendix B.2).
    """
    rng = random.Random(seed)

    # Bucket by task type
    buckets: dict[str, list[dict]] = {}
    for task in tasks:
        task_type = task.get("type", "unknown")
        buckets.setdefault(task_type, []).append(task)

    groups = []
    for task_type, bucket in buckets.items():
        rng.shuffle(bucket)
        # Create groups of group_size from each bucket
        for i in range(0, len(bucket), group_size):
            chunk = bucket[i : i + group_size]
            if len(chunk) >= 2:  # need at least 2 for r_task (skip first)
                groups.append(TaskGroup(tasks=chunk, group_type=task_type))

    rng.shuffle(groups)
    return groups


def group_reasoning_tasks(
    tasks: list[dict],
    group_size: int = 8,
    seed: int = 42,
) -> list[TaskGroup]:
    """Group reasoning tasks by difficulty and topic similarity.

    Simplified version — full implementation needs:
    1. Attribute annotation via LLM (topics, skills, concepts, strategies, pitfalls)
    2. Soft-Jaccard similarity with sentence embeddings
    3. Dependency gate filtering
    4. Curriculum ordering by difficulty

    This implementation groups by difficulty bands as a starting point.
    """
    rng = random.Random(seed)

    # Sort by difficulty if available
    sorted_tasks = sorted(tasks, key=lambda t: t.get("difficulty", 0))

    groups = []
    for i in range(0, len(sorted_tasks), group_size):
        chunk = sorted_tasks[i : i + group_size]
        if len(chunk) >= 2:
            groups.append(TaskGroup(tasks=chunk, group_type="reasoning"))

    rng.shuffle(groups)
    return groups
