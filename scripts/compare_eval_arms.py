"""Paired comparison of two or more streaming-curation eval arms.

Reads N JSONLs produced by scripts.eval_streaming_curation (one per arm),
inner-joins them on `gamefile` so the comparison is paired-per-gamefile, then:

- Reports per-arm SR (overall + per task type), restricted to games present in
  ALL arms (so the comparison is fair).
- Runs McNemar's test on every ordered pair of arms (A vs B):
  given the 2x2 contingency of paired outcomes (A pass / B pass, A pass / B fail,
  A fail / B pass, A fail / B fail), tests whether the discordant cells differ
  more than chance — i.e., whether B is meaningfully different from A.

McNemar uses scipy.stats if available; otherwise a binomial fallback on the
discordant pairs (the same null distribution, just spelled out).

Usage:
  python -m scripts.compare_eval_arms \\
      --arm "no_memory=output/eval-pathbv4/no_memory.jsonl" \\
      --arm "ckpt60=output/eval-pathbv4/ckpt60.jsonl" \\
      --arm "ckpt10=output/eval-pathbv4/ckpt10.jsonl"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_arm(path: str) -> dict[str, dict]:
    """Return {gamefile: record} for one arm. Later records win if duplicated."""
    by_gf: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            gf = r.get("gamefile") or ""
            if gf:
                by_gf[gf] = r
    return by_gf


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided binomial p-value on the discordant pairs (b, c).
    b = #games arm A passed but arm B failed; c = #games A failed but B passed.
    Under H0 the splits are 50/50 → discordant counts ~ Binomial(n, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    except Exception:
        # Manual two-sided exact binomial: 2 * P(X <= min(b,c) | n, 0.5),
        # capped at 1.0.
        k = min(b, c)
        from math import comb
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2.0 * tail)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True,
                   help="Repeatable: 'name=path/to.jsonl'. Order matters — first arm "
                        "is the reference for pairwise McNemar.")
    args = p.parse_args()

    arms: list[tuple[str, dict[str, dict]]] = []
    for spec in args.arm:
        if "=" not in spec:
            sys.exit(f"--arm must be name=path (got: {spec})")
        name, path = spec.split("=", 1)
        arms.append((name.strip(), load_arm(path.strip())))

    # Inner-join on gamefile so only fully-paired games count.
    common = set(arms[0][1].keys())
    for _, by in arms[1:]:
        common &= set(by.keys())
    n = len(common)
    if n == 0:
        sys.exit("ERROR: no shared gamefiles across the supplied arms.")

    print(f"=== {len(arms)} arms, {n} paired games ===\n")

    # Per-arm overall + per-type SR on the common set.
    print(f"{'arm':<14} {'overall':>10}    per-type SR (success/total)")
    for name, by in arms:
        ok = sum(int(by[gf].get("success", False)) for gf in common)
        per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for gf in common:
            r = by[gf]
            per_type[r.get("task_type", "?")][0] += int(r.get("success", False))
            per_type[r.get("task_type", "?")][1] += 1
        sr = ok / n
        type_str = "  ".join(
            f"{t}:{s}/{tot}={s/max(tot,1):.0%}"
            for t, (s, tot) in sorted(per_type.items())
        )
        print(f"{name:<14} {ok:>4}/{n} ={sr:>5.1%}    {type_str}")

    # Pairwise McNemar on the first arm vs each subsequent arm.
    ref_name, ref_by = arms[0]
    if len(arms) >= 2:
        print(f"\n=== McNemar vs reference arm '{ref_name}' ===")
        print(f"{'arm':<14} {'B-only':>7} {'A-only':>7} {'both':>6} {'neither':>8} {'delta_SR':>9} {'p (2-sided)':>12}")
        for name, by in arms[1:]:
            both_ok = a_only = b_only = neither = 0
            for gf in common:
                a = bool(ref_by[gf].get("success"))
                b = bool(by[gf].get("success"))
                if a and b: both_ok += 1
                elif a and not b: a_only += 1
                elif b and not a: b_only += 1
                else: neither += 1
            delta = (both_ok + b_only) / n - (both_ok + a_only) / n
            pval = mcnemar_p(a_only, b_only)
            print(f"{name:<14} {b_only:>7} {a_only:>7} {both_ok:>6} {neither:>8} "
                  f"{delta:>+9.1%} {pval:>12.4f}")
        print("  B-only = arm gained pass that reference missed (positive lift)")
        print("  A-only = reference passed but arm missed (negative lift)")
        print("  p < 0.05 means the discordant split is unlikely under 'no difference'.")


if __name__ == "__main__":
    main()
