"""Training-data loader for the reasoning benchmark.

Paper §4.1 uses DeepMath-103K, sampling ~33K and grouping into ~20K instances.
Groups are same-topic. Nine topic buckets in the source at the 2nd level of the
`topic` hierarchy. Deterministic per group_id, mirrors
`skillos/envs/curator_env.py::_build_type_seed_index` shape so the algo1 env
can consume it without changes.
"""

from __future__ import annotations

import hashlib
import random
import threading
from functools import lru_cache


DEEPMATH_TOPICS = (
    "Algebra",
    "Calculus",
    "Precalculus",
    "Geometry",
    "Applied Mathematics",
    "Number Theory",
    "Discrete Mathematics",
    "Differential Equations",
    "Other",
)

_dataset = None
_topic_index: dict[str, list[int]] = {}
_lock = threading.Lock()


def _topic_bucket(raw_topic: str) -> str:
    parts = [p.strip() for p in raw_topic.split("->")]
    return parts[1] if len(parts) > 1 else parts[0]


def _load_dataset():
    global _dataset
    if _dataset is None:
        from datasets import load_dataset
        _dataset = load_dataset("zwhe99/DeepMath-103K", split="train")
    return _dataset


def build_topic_index() -> None:
    """One-time scan: bucket every problem's index by its 2nd-level topic."""
    global _topic_index
    with _lock:
        if _topic_index:
            return
        ds = _load_dataset()
        idx: dict[str, list[int]] = {t: [] for t in DEEPMATH_TOPICS}
        for i, row in enumerate(ds):
            b = _topic_bucket(row["topic"])
            idx.setdefault(b, []).append(i)
        _topic_index = idx
        counts = " ".join(f"{k}={len(v)}" for k, v in sorted(idx.items()))
        print(f"[reasoning.train_data] DeepMath-103K topic index: {counts}",
              flush=True)


def seeds_for_topic(topic: str) -> list[int]:
    if not _topic_index:
        build_topic_index()
    return _topic_index.get(topic, [])


@lru_cache(maxsize=8192)
def get_problem(idx: int) -> dict:
    """Fetch one row by dataset index. LRU-cached because the same seeds are
    drawn many times across GRPO rollouts of the same group."""
    ds = _load_dataset()
    r = ds[idx]
    return {
        "question": r["question"],
        "final_answer": str(r["final_answer"]).strip(),
        "topic": _topic_bucket(r["topic"]),
        "difficulty": float(r.get("difficulty") or 0.0),
    }


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")


def sample_group_seeds(group_id: int, topic: str, group_size: int) -> list[int]:
    """Return G dataset indices of the given topic, deterministic per group_id.
    Curriculum ordering not applied here — paper's easy→hard curriculum was
    falsified on ALFWorld (see gotcha `curriculum-no-lift-grouping-exonerated`).
    """
    pool = seeds_for_topic(topic)
    if not pool:
        raise ValueError(f"no DeepMath problems for topic={topic!r}")
    rng = random.Random(group_id * 1_000_003 + _stable_hash(topic))
    if len(pool) >= group_size:
        return rng.sample(pool, group_size)
    out = list(pool)
    while len(out) < group_size:
        out.append(rng.choice(pool))
    return out
