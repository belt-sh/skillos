"""Paper-grade statistics for paired eval arms.

`scripts.compare_eval_arms` reports a delta and a McNemar p-value. That is
enough to decide whether to keep running an experiment and not enough to put in
a paper. This module adds the three things a reviewer will ask for:

1. **Effect size with a confidence interval.** A paired bootstrap over game
   files, which respects the pairing and makes no normality assumption. A null
   result is only interpretable next to the interval it excludes: "+1.9pp,
   95% CI [-6.4, +10.1]" says something, "p=0.84" does not.

2. **Multiplicity correction.** Holm-Bonferroni (family-wise) and
   Benjamini-Hochberg (false discovery rate) over an explicitly declared family.
   Sweeps in this project run 12 arms against one baseline; an uncorrected 0.05
   over 12 arms has a ~46% chance of producing at least one "significant" arm
   from noise alone.

3. **Post-hoc power.** For each comparison, the smallest true effect this arm
   had an 80% chance of detecting, given its own observed discordance rate. This
   is what makes a null publishable: not "we found nothing" but "we would have
   caught 8pp and we did not".

Usage:
    python -m scripts.paper_stats --family "unseen" \
        --base output/reeval/unseen-power/no_memory.jsonl \
        --arm "gemini=output/reeval/unseen-power/gemini_curator.jsonl" \
        --arm "trained=output/reeval/unseen-power/r2a_ckpt50.jsonl"

Also importable: `paired_bootstrap`, `mcnemar_exact`, `holm`, `bh`, `mde`.
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb, sqrt
from pathlib import Path

N_BOOT = 10000
BOOT_SEED = 20260815  # fixed so every table in the paper is reproducible


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_arm(path: str | Path) -> dict[str, bool]:
    """gamefile -> success, excluding rows the harness marked as errored.

    Errored rows are episodes abandoned to an upstream API failure. They carry
    `success: None` and must never be counted as failures; see the data
    integrity incident in the paper's Appendix B.
    """
    out: dict[str, bool] = {}
    for line in open(path):
        rec = json.loads(line)
        if rec.get("errored") or rec.get("success") is None:
            continue
        out[rec["gamefile"]] = bool(rec["success"])
    return out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0=0.5, summing all outcomes at most as
    likely as the observed one. Avoids the chi-square approximation, which is
    wrong at the discordant counts we actually see (often under 25)."""
    if n == 0:
        return 1.0
    p_obs = comb(n, k) * 0.5 ** n
    return min(1.0, sum(comb(n, i) * 0.5 ** n
                        for i in range(n + 1)
                        if comb(n, i) * 0.5 ** n <= p_obs * (1 + 1e-9)))


def mcnemar_exact(base: dict[str, bool], arm: dict[str, bool]) -> dict:
    keys = sorted(set(base) & set(arm))
    b_only = sum(1 for k in keys if arm[k] and not base[k])
    a_only = sum(1 for k in keys if base[k] and not arm[k])
    n = len(keys)
    return {
        "n": n,
        "b_only": b_only,
        "a_only": a_only,
        "discordant": b_only + a_only,
        "base_sr": sum(base[k] for k in keys) / n * 100 if n else 0.0,
        "arm_sr": sum(arm[k] for k in keys) / n * 100 if n else 0.0,
        "delta_pp": (b_only - a_only) / n * 100 if n else 0.0,
        "p": _binom_two_sided(b_only, b_only + a_only),
    }


def paired_bootstrap(base: dict[str, bool], arm: dict[str, bool],
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile CI on the paired delta, resampling game files with
    replacement. Game files are the unit of independence: the two arms saw the
    same game, so the pair travels together through the resample."""
    keys = sorted(set(base) & set(arm))
    if not keys:
        return (float("nan"), float("nan"))
    diffs = [int(arm[k]) - int(base[k]) for k in keys]
    n = len(diffs)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        deltas.append(s / n * 100)
    deltas.sort()
    return (deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot) - 1])


def mde(n_discordant: int, n: int, power: float = 0.80) -> float:
    """Minimum detectable effect, in percentage points, at alpha=0.05 two-sided
    and the given power, for a McNemar test with this observed discordance.

    Derivation: with d discordant pairs out of n, an effect of delta pp means an
    expected imbalance of delta*n/100 among those d. Normal approximation to the
    binomial gives the standard requirement

        |imbalance| >= (z_a/2 + z_b) * sqrt(d) / 2 * 2

    which we express back in percentage points of n. Approximate by design; it
    is reported to one decimal place and used only to say "we would have caught
    an effect of about this size".
    """
    if n == 0 or n_discordant == 0:
        return float("nan")
    z_a, z_b = 1.959964, 0.8416212  # two-sided 0.05, power 0.80
    return (z_a + z_b) * sqrt(n_discordant) / n * 100


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #

def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values. Controls family-wise error rate.
    Uniformly more powerful than plain Bonferroni and just as valid."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def bh(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (q-values). Controls false discovery
    rate, which is the more appropriate target for an exploratory sweep."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        running = min(running, pvals[i] * m / k)
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
# stratified pooling
# --------------------------------------------------------------------------- #

def pooled(strata: list[tuple[dict[str, bool], dict[str, bool]]]) -> dict:
    """Pool paired comparisons across strata (for us: the valid_seen and
    valid_unseen splits) by summing discordant cells.

    WARNING, and it belongs in the caption wherever this is used: pooling is
    only honest when no stratum was used to select the arm being tested. If a
    checkpoint was chosen as best-of-k on one split, that split's contribution
    is upward-biased and the pooled p-value is optimistic. Report the clean
    stratum alone as the primary result and the pool as secondary.
    """
    b = a = n = 0
    for base, arm in strata:
        r = mcnemar_exact(base, arm)
        b += r["b_only"]; a += r["a_only"]; n += r["n"]
    return {"n": n, "b_only": b, "a_only": a, "discordant": b + a,
            "delta_pp": (b - a) / n * 100 if n else 0.0,
            "p": _binom_two_sided(b, b + a)}


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="reference arm JSONL")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--family", default="unnamed",
                    help="Name of the multiple-comparison family these arms belong to. "
                         "Corrections are applied within it.")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    base = load_arm(args.base)
    arms = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms.append((name, load_arm(path)))

    rows = []
    for name, arm in arms:
        r = mcnemar_exact(base, arm)
        lo, hi = paired_bootstrap(base, arm, n_boot=args.n_boot)
        r.update(name=name, ci_lo=lo, ci_hi=hi,
                 mde=mde(r["discordant"], r["n"]))
        rows.append(r)

    ps = [r["p"] for r in rows]
    for r, h, q in zip(rows, holm(ps), bh(ps)):
        r["p_holm"], r["q_bh"] = h, q

    print(f"\n=== family '{args.family}': {len(rows)} arms vs {Path(args.base).name} ===")
    print(f"reference SR = {rows[0]['base_sr']:.1f}%  (n={rows[0]['n']} paired games)")
    print(f"bootstrap: {args.n_boot} resamples over game files, seed {BOOT_SEED}\n")
    hdr = (f"{'arm':<24}{'SR':>7}{'delta':>8}  {'95% CI':>16}"
           f"{'p':>9}{'p_holm':>9}{'q_BH':>8}{'MDE80':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<24}{r['arm_sr']:>6.1f}%{r['delta_pp']:>+7.1f}  "
              f"[{r['ci_lo']:>+6.1f},{r['ci_hi']:>+6.1f}]"
              f"{r['p']:>9.4f}{r['p_holm']:>9.4f}{r['q_bh']:>8.4f}{r['mde']:>7.1f}")
    print("\nMDE80 = smallest true effect this arm had 80% power to detect, in pp.")
    print("A null with MDE80 below the effect of interest is informative;")
    print("a null with MDE80 above it is merely underpowered.")


if __name__ == "__main__":
    main()
