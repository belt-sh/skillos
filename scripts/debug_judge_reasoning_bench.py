"""Benchmark Qwen3-32B judge with reasoning ON vs OFF on a realistic skill.

Runs N=3 calls in each mode, reports per-call wall time, mean latency, and
the parsed VALID verdict (should be identical across modes if reasoning is
just preamble the parser ignores).
"""
from __future__ import annotations

import time

from skillos.rewards.judge import InfshJudge


# Realistic skill markdown the curator might emit during ALFWorld training.
# Size is in the typical range (~250 words) so latency numbers reflect real
# training-time judge calls, not a degenerate tiny prompt.
SKILL_BODY = """\
---
name: place_cooled_lettuce_in_diningtable
description: Cool lettuce in the fridge, then place it on the dining table.
---

# Place Cooled Lettuce in Diningtable

## When to use
- Task asks to "put a cool X in Y" for a perishable item that can be cooled.
- Lettuce (or similar produce) is present in the room and a fridge is reachable.

## Workflow
1. Locate the lettuce by examining countertops, shelves, and dining tables.
2. `take lettuce <id>` from its current container/surface.
3. Navigate to the fridge: `go to fridge 1`.
4. If the fridge is closed, `open fridge 1`.
5. `cool lettuce <id> with fridge 1`. The simulator will confirm the cool action.
6. `close fridge 1` (some scoring rubrics require this).
7. Navigate to the target surface: `go to diningtable 1`.
8. `put lettuce <id> in/on diningtable 1`.

## When NOT to use
- Task involves heating or warming the item — use a different skill.
- The target is a microwave or oven rather than a dining table.
- The item is already known to be cool (skip the cool step).
"""


def _time_calls(judge: InfshJudge, label: str, n: int = 3) -> None:
    latencies = []
    verdicts = []
    for i in range(n):
        t0 = time.time()
        score = judge.score(SKILL_BODY)
        dt = time.time() - t0
        latencies.append(dt)
        verdicts.append(score)
        print(f"  [{label}] call {i+1}: {dt:5.2f}s  VALID={int(score)}")
    mean = sum(latencies) / len(latencies)
    print(f"  [{label}] mean over {n}: {mean:5.2f}s  verdicts={verdicts}")
    return mean


if __name__ == "__main__":
    print("Benchmarking openrouter/qwen3-32b judge on a realistic ALFWorld skill")
    print(f"  prompt length: ~{len(SKILL_BODY.split())} words\n")

    print("--- reasoning OFF (reasoning_effort='none') ---")
    judge_off = InfshJudge(
        app="openrouter/qwen3-32b",
        temperature=0.0,
        max_tokens=256,
        context_size=8192,
        reasoning_effort="none",
    )
    off_mean = _time_calls(judge_off, "off", n=3)

    print("\n--- reasoning ON (reasoning_effort=None → app default) ---")
    judge_on = InfshJudge(
        app="openrouter/qwen3-32b",
        temperature=0.0,
        max_tokens=256,
        context_size=8192,
        reasoning_effort=None,
    )
    on_mean = _time_calls(judge_on, "on", n=3)

    print(f"\n--- summary ---")
    print(f"  OFF mean: {off_mean:5.2f}s")
    print(f"  ON  mean: {on_mean:5.2f}s")
    if off_mean < on_mean:
        speedup = on_mean / off_mean
        print(f"  OFF is {speedup:.2f}× faster ({on_mean - off_mean:+.2f}s saved per call)")
    else:
        print(f"  OFF was not faster — surprising; check that the app respected reasoning_effort='none'.")
