"""Figures for the reproduction paper.

Deliberately separate from `make_report_figures.py` (repo report, dense) and
`make_article_figures.py` (editorial, one point per figure). Paper figures are
vector PDF, greyscale-safe, no chartjunk, and every one of them is regenerated
from the released JSONLs so it cannot drift from Appendix C.

Rule enforced here: a figure raises rather than plotting a partial sweep. A
half-finished family drawn as a line looks like a finished one.

    python -m scripts.make_paper_figures            # all available figures
    python -m scripts.make_paper_figures fig1       # one figure
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paper_stats import load_arm, mcnemar_exact, mde, paired_bootstrap

FIGDIR = Path("docs/paper/figures")
R = Path("output/reeval")
MIN_COMPLETE = 130

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

INK = "#1a1a1a"
MID = "#767676"
LIGHT = "#c8c8c8"


def require(path: Path) -> dict[str, bool]:
    p = Path(path)
    if not (p.exists() and p.stat().st_size):
        raise FileNotFoundError(f"figure needs {p}, not measured yet")
    arm = load_arm(p)
    if len(arm) < MIN_COMPLETE:
        raise ValueError(f"{p} has only {len(arm)} games, still running")
    return arm


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {FIGDIR / name}.pdf")


# --------------------------------------------------------------------------- #
# fig1: the control moved further than the effect
# --------------------------------------------------------------------------- #

def fig1() -> None:
    runs = [
        ("May 2026\n(used as canonical)", "output/eval-pathbv4/no_memory.jsonl", True),
        ("Aug rep 1", R / "baseline/no_memory_8b.jsonl", False),
        ("Aug rep 2", R / "baseline-replicates/no_memory_8b_run3.jsonl", False),
        ("Aug rep 3", R / "baseline-replicates/no_memory_8b_run4.jsonl", False),
        ("Aug rep 4", R / "baseline-replicates/no_memory_8b_run5.jsonl", False),
    ]
    vals, labels, stale = [], [], []
    for lab, path, is_stale in runs:
        arm = require(path)
        vals.append(sum(arm.values()) / len(arm) * 100)
        labels.append(lab)
        stale.append(is_stale)

    aug = [v for v, s in zip(vals, stale) if not s]
    mean = sum(aug) / len(aug)

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    xs = range(len(vals))
    ax.bar(xs, vals, width=0.6,
           color=[MID if s else INK for s in stale],
           edgecolor="none")
    ax.axhspan(min(aug), max(aug), color=LIGHT, alpha=0.45, zorder=0)
    ax.axhline(mean, color=INK, lw=0.8, ls=":", zorder=1)

    for x, v in zip(xs, vals):
        ax.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    ax.annotate("", xy=(0, vals[0]), xytext=(0, mean),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK))
    ax.text(0.28, (vals[0] + mean) / 2, f"{mean - vals[0]:.1f} pp",
            fontsize=8, va="center")

    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("no-memory success rate (%)")
    ax.set_ylim(0, max(vals) * 1.22)
    save(fig, "fig1_control_drift")


# --------------------------------------------------------------------------- #
# fig2: the sweep before and after re-pairing
# --------------------------------------------------------------------------- #

def fig2() -> None:
    ckpts = list(range(5, 65, 5))
    arms = {c: R / f"fft-seed2/ckpt{c}.jsonl" for c in ckpts}
    old_base = require("output/eval-pathbv4/no_memory.jsonl")
    new_base = require(R / "baseline/no_memory_8b.jsonl")

    # Both series use the SAME (re-run) arm files and differ only in which
    # control they subtract. That isolates the reference change. It is not the
    # same as "originally reported minus now", which also folds in the harness
    # fix; the caption says so.
    old_d, old_p, new_d, new_p, los, his, xs = [], [], [], [], [], [], []
    for c in ckpts:
        arm = require(arms[c])
        xs.append(c)
        ro = mcnemar_exact(old_base, arm)
        old_d.append(ro["delta_pp"]); old_p.append(ro["p"])
        r = mcnemar_exact(new_base, arm)
        new_d.append(r["delta_pp"]); new_p.append(r["p"])
        lo, hi = paired_bootstrap(new_base, arm)
        los.append(r["delta_pp"] - lo); his.append(hi - r["delta_pp"])

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(xs, old_d, "o--", color=MID, ms=4, lw=1.0,
            label="vs stale control (as previously reported)")
    ax.errorbar(xs, new_d, yerr=[los, his], fmt="o-", color=INK, ms=4, lw=1.3,
                capsize=2.5, elinewidth=0.8,
                label="vs same-week control, 95% CI")

    i = old_d.index(max(old_d))
    ax.annotate(f"{old_d[i]:+.1f} pp, p={old_p[i]:.3f}", xy=(xs[i], old_d[i]),
                xytext=(xs[i] - 24, old_d[i] + 1.5), fontsize=7.5, color=MID,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=MID))
    ax.annotate(f"{new_d[i]:+.1f} pp, p={new_p[i]:.2f}", xy=(xs[i], new_d[i]),
                xytext=(xs[i] - 21, new_d[i] - 4.5), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))

    ax.set_xlabel("curator training step")
    ax.set_ylabel("change in success rate (pp)")
    ax.set_ylim(min(l - e for l, e in zip(new_d, los)) - 3,
                max(old_d) + 9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=1)
    save(fig, "fig2_repairing_the_sweep")


# --------------------------------------------------------------------------- #
# fig3: power curve of the standard protocol
# --------------------------------------------------------------------------- #

def fig3() -> None:
    from math import sqrt
    z = 1.959964 + 0.8416212
    disc = 0.30
    ns = list(range(60, 3001, 10))
    mdes = [z * sqrt(disc * n) / n * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(ns, mdes, color=INK, lw=1.4)
    ax.axvline(140, color=MID, lw=0.9, ls="--")
    ax.axhline(13.0, color=MID, lw=0.9, ls="--")
    ax.plot([140], [13.0], "o", color=INK, ms=5)
    ax.annotate("standard protocol\n140 games, MDE = 13.0 pp",
                xy=(140, 13.0), xytext=(330, 15.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=INK))
    ax.axhspan(0, 5, color=LIGHT, alpha=0.5, zorder=0)
    ax.text(1500, 2.2, "effect sizes plausibly at stake\nfor agent memory",
            fontsize=7.5, color=MID)
    ax.annotate("", xy=(942, 5.0), xytext=(942, 0),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MID, ls=":"))
    ax.text(960, 6.2, "942 games needed\nto resolve 5 pp", fontsize=7.5, color=MID)

    ax.set_xlabel("paired games")
    ax.set_ylabel("minimum detectable effect at 80% power (pp)")
    ax.set_ylim(0, 22)
    ax.set_xlim(60, 3000)
    save(fig, "fig3_power")


# --------------------------------------------------------------------------- #
# fig4: held-out comparison with intervals
# --------------------------------------------------------------------------- #

def fig4() -> None:
    base = require(R / "unseen-power/no_memory.jsonl")
    wanted = [
        ("Gemini 2.5 Pro curator\n$0.0168 / call", R / "unseen-power/gemini_curator.jsonl"),
        ("trained 8B curator\n$0.0002 / call", R / "unseen-power/r2a_ckpt50.jsonl"),
        ("same curator,\nshuffled retrieval", R / "unseen-power/r2a_ckpt50_shuffled.jsonl"),
        ("hand-written skills\n(no curator)", R / "unseen-power/oracle_handwritten.jsonl"),
    ]
    labels, deltas, los, his, mdes = [], [], [], [], []
    for lab, path in wanted:
        try:
            arm = require(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  fig4: skipping {lab.splitlines()[0]} ({e})")
            continue
        r = mcnemar_exact(base, arm)
        lo, hi = paired_bootstrap(base, arm)
        labels.append(lab); deltas.append(r["delta_pp"])
        los.append(r["delta_pp"] - lo); his.append(hi - r["delta_pp"])
        mdes.append(mde(r["discordant"], r["n"]))
    if not labels:
        raise FileNotFoundError("fig4 needs at least one measured arm")

    fig, ax = plt.subplots(figsize=(5.4, 0.75 * len(labels) + 1.5))
    ys = range(len(labels))
    ax.axvline(0, color=INK, lw=0.9)
    ax.errorbar(deltas, ys, xerr=[los, his], fmt="o", color=INK, ms=5,
                capsize=3, elinewidth=0.9)
    for y, d, m in zip(ys, deltas, mdes):
        ax.plot([-m, m], [y - 0.28, y - 0.28], color=LIGHT, lw=2.5,
                solid_capstyle="butt", zorder=0)
        ax.text(d, y + 0.22, f"{d:+.1f} pp", fontsize=8, ha="center")

    ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("change in success rate vs no memory (pp)")
    save(fig, "fig4_heldout_comparison")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}


def main() -> None:
    want = sys.argv[1:] or list(FIGS)
    for name in want:
        print(f"{name}:")
        try:
            FIGS[name]()
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP {name}: {e}")


if __name__ == "__main__":
    main()
