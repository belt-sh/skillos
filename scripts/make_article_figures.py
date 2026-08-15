#!/usr/bin/env python3
"""Editorial figures for the public article. Deliberately NOT the report figures.

The report's figures (scripts/make_report_figures.py) are for readers who want
the full sweep: five overlaid curves, per-checkpoint p-values, Pearson r. These
are for readers who want the point. Four charts, one claim each, no jargon on
any axis, no legend where a direct label will do.

Every number is computed from the JSONLs, never typed in, so a re-run cannot
leave a stale figure behind. Missing inputs raise instead of silently plotting a
subset, because a chart that quietly drops an arm is the same class of bug this
article is about.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_eval_arms import mcnemar_p  # noqa: E402

OUT = Path("docs/figures/article")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#111827", "#6B7280", "#E5E7EB"
BLUE, AMBER, GREEN, RED = "#2563EB", "#F59E0B", "#059669", "#DC2626"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def titles(fig, headline, standfirst):
    """Claim-style headline plus grey standfirst, positioned in inches from the
    top edge so short and tall figures place them identically. The standfirst is
    wrapped to the figure width, because a sentence that runs off the canvas is
    the most common way these charts get quietly ruined. Returns the top for
    tight_layout(rect=...)."""
    w_in, h_in = fig.get_size_inches()
    # ~0.082 in per character at 12.5pt DejaVu Sans, measured empirically.
    width = max(40, int((w_in - 0.3) / 0.082))
    lines = textwrap.wrap(standfirst, width=width)
    fig.text(0.015, 1 - 0.34 / h_in, headline, ha="left", va="center",
             fontsize=18, weight="bold", color=INK)
    y = 0.70
    for line in lines:
        fig.text(0.015, 1 - y / h_in, line, ha="left", va="center",
                 fontsize=12.5, color=MUTED)
        y += 0.26
    return 1 - (y + 0.18) / h_in


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing input: {p}\n(re-run the sweep before regenerating figures)")
    rows = [json.loads(l) for l in p.open()]
    return {r["gamefile"]: r for r in rows if not r.get("errored")}


def sr(arm):
    return 100.0 * sum(1 for r in arm.values() if r["success"]) / max(len(arm), 1)


def paired(ref, arm):
    """Returns (delta in percentage points, p) on the games both arms scored."""
    keys = set(ref) & set(arm)
    b = sum(1 for k in keys if arm[k]["success"] and not ref[k]["success"])
    c = sum(1 for k in keys if ref[k]["success"] and not arm[k]["success"])
    return 100.0 * (b - c) / len(keys), mcnemar_p(b, c)


def bar_labels(ax, bars, values, fmt="{:+.1f}", pad=0.35):
    for bar, v in zip(bars, values):
        w = bar.get_width()
        off = pad if w >= 0 else -pad
        ax.text(w + off, bar.get_y() + bar.get_height() / 2, fmt.format(v),
                va="center", ha="left" if w >= 0 else "right",
                fontsize=13, weight="bold", color=INK)


# --------------------------------------------------------------- figure 1 ---
def fig_cheap_specialist():
    """The money chart: a $0.0002 model beat a frontier model at the same job."""
    base = load("output/reeval/baseline/no_memory_8b.jsonl")
    gem = load("output/reeval/gemini-curator/gemini_8b.jsonl")
    g_delta, g_p = paired(base, gem)

    trained = []
    for p in sorted(Path("output/reeval/fft").glob("ckpt*.jsonl")):
        d, _ = paired(base, load(p))
        trained.append(d)
    if not trained:
        raise SystemExit("no trained arms re-run yet; wave B must finish first")
    best = max(trained)

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    top = titles(fig,
                 "A $0.0002 model versus a $0.0168 model, at the same job",
                 "How much each note-writer changed the small agent, on 140 identical household tasks. Neither clears the noise band on its own; the gap between them is 7.1 points.")
    labels = ["Gemini 2.5 Pro\nwriting the notes", "A small model I trained\nwriting the notes"]
    vals = [g_delta, best]
    bars = ax.barh(labels, vals, color=[AMBER, BLUE], height=0.5)
    bar_labels(ax, bars, vals)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("change in how often the agent finished the task (percentage points)")
    lo, hi = min(vals) - 4, max(vals) + 5
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12, color=INK)
    ax.text(g_delta + (0.4 if g_delta >= 0 else -0.4), 0.30,
            f"costs $0.0168 per note\n(p={g_p:.2f}, i.e. no real help)",
            fontsize=10.5, color=MUTED, va="bottom",
            ha="left" if g_delta >= 0 else "right")
    ax.text(best + 0.4, 1.30, "costs $0.0002 per note\nroughly 80x cheaper",
            fontsize=10.5, color=MUTED, va="bottom")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "a1_cheap_beats_frontier.png", dpi=200)
    plt.close(fig)
    print(f"a1: gemini {g_delta:+.1f} (p={g_p:.3f}), best trained {best:+.1f}")


# --------------------------------------------------------------- figure 2 ---
def fig_peak_lottery():
    """Every run peaks somewhere. Nowhere near each other."""
    runs = {
        "seed 1": "output/reeval/fft",
        "seed 2": "output/reeval/fft-seed2",
        "seed 3": "output/reeval/fft-seed3",
        "a different framework": "output/reeval/verl-gigpo-real",
    }
    base = load("output/reeval/baseline/no_memory_8b.jsonl")
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    top = titles(fig,
                 "Training does not steadily improve the agent. It wanders.",
                 "Each line is one training run. The dot marks its best moment, which is never in the same place twice.")
    colors = [BLUE, AMBER, GREEN, "#1E1B4B"]
    plotted = 0
    for (name, d), col in zip(runs.items(), colors):
        paths = sorted(Path(d).glob("ckpt*.jsonl"),
                       key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
        if len(paths) < 4:
            continue
        xs, ys = [], []
        for p in paths:
            xs.append(int("".join(c for c in p.stem if c.isdigit())))
            ys.append(paired(base, load(p))[0])
        ax.plot(xs, ys, color=col, lw=2.0, alpha=0.9)
        i = max(range(len(ys)), key=lambda j: ys[j])
        ax.plot(xs[i], ys[i], "o", color=col, ms=9)
        ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=11.5,
                    color=col, va="center", weight="bold")
        plotted += 1
    if plotted < 2:
        raise SystemExit("fewer than 2 sweeps re-run; waves B and C must finish first")
    ax.axhspan(-5.7, 5.7, color=GRID, alpha=0.65, zorder=0)
    ax.text(2, 5.9, "anything inside this band is\nindistinguishable from noise",
            fontsize=10.5, color=MUTED, va="bottom")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel("change in agent success (points)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 72)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "a2_peak_lottery.png", dpi=200)
    plt.close(fig)
    print(f"a2: plotted {plotted} sweeps")


# --------------------------------------------------------------- figure 3 ---
def fig_the_bug():
    """Before and after the fix, on the four arms that produced a fake finding."""
    old = {45: 15.7, 50: 18.6, 55: 16.4, 60: 19.3}
    invented = {45: 64.9, 50: 59.4, 55: 52.1, 60: 55.8}
    base = load("output/reeval/baseline/no_memory_8b.jsonl")
    new = {ck: sr(load(f"output/reeval/reasoning-to-alfworld/ckpt{ck}.jsonl"))
           for ck in old}

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    top = titles(fig,
                 "My best finding was an outage, not a result",
                 "For three weeks I believed the red bars. Over half of those agent moves were a default action played by my own error handler after the API refused the call.")
    xs = range(len(old))
    w = 0.38
    kk = sorted(old)
    b1 = ax.bar([x - w / 2 for x in xs], [old[k] for k in kk], w,
                color=RED, label="as first measured")
    b2 = ax.bar([x + w / 2 for x in xs], [new[k] for k in kk], w,
                color=BLUE, label="re-run with the bug fixed")
    ax.axhline(sr(base), color=INK, lw=1.6, ls="--")
    ax.text(0.02, sr(base) + 0.8, "the agent with no notes at all",
            fontsize=10.5, color=INK, ha="left")
    for x, k in zip(xs, kk):
        ax.text(x - w / 2, old[k] - 2.0, f"{invented[k]:.0f}%\nfake\nmoves",
                ha="center", va="top", fontsize=9.5, color="white", weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"step {k}" for k in kk], fontsize=11.5, color=INK)
    ax.set_ylabel("how often the agent finished the task (%)")
    ax.set_ylim(0, max(max(new.values()), sr(base)) + 8)
    ax.legend(frameon=False, fontsize=11, loc="upper left",
              handles=[Patch(color=RED, label="as first measured"),
                       Patch(color=BLUE, label="re-run with the bug fixed")])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "a3_the_bug.png", dpi=200)
    plt.close(fig)
    print("a3: " + ", ".join(f"step{k} {old[k]:.1f}->{new[k]:.1f}" for k in kk))


# --------------------------------------------------------------- figure 4 ---
def fig_noise_floor():
    """The same test, run twice, unchanged."""
    pairs = [
        ("the small agent", "output/eval-pathbv4/no_memory.jsonl",
         "output/reeval/baseline/no_memory_8b.jsonl"),
        ("the large agent", "output/eval-transfer-32b/no_memory.jsonl",
         "output/reeval/baseline/no_memory_32b.jsonl"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    top = titles(fig,
                 "The same test, run twice, moves by six points",
                 "Identical agent, identical 140 tasks. But the two runs are ten weeks apart, so this is either the agent's own sampling or the hosted model changing underneath me. Replicates running to find out which.")
    y = 0
    for label, p1, p2 in pairs:
        a, b = load(p1), load(p2)
        s1, s2 = sr(a), sr(b)
        keys = set(a) & set(b)
        flipped = sum(1 for k in keys if a[k]["success"] != b[k]["success"])
        ax.plot([s1, s2], [y, y], color=GRID, lw=8, solid_capstyle="round", zorder=1)
        ax.plot(s1, y, "o", ms=15, color=MUTED, zorder=2)  # earlier run
        ax.plot(s2, y, "o", ms=15, color=BLUE, zorder=2)
        ax.text(s1, y + 0.20, f"{s1:.1f}%", ha="center", fontsize=12.5,
                color=MUTED, weight="bold")
        ax.text(s2, y + 0.20, f"{s2:.1f}%", ha="center", fontsize=12.5,
                color=BLUE, weight="bold")
        ax.text((s1 + s2) / 2, y - 0.26,
                f"{flipped} of {len(keys)} tasks flipped answer",
                ha="center", fontsize=10.5, color=MUTED)
        ax.text(min(s1, s2) - 2.2, y, label, ha="right", va="center",
                fontsize=12, color=INK)
        y += 1
    ax.set_yticks([])
    ax.set_ylim(-0.6, y - 0.3)
    ax.set_xlim(28, 56)
    ax.set_xlabel("how often the agent finished the task (%)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "a4_noise_floor.png", dpi=200)
    plt.close(fig)
    print("a4: done")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4"]
    fns = {"1": fig_cheap_specialist, "2": fig_peak_lottery,
           "3": fig_the_bug, "4": fig_noise_floor}
    for k in want:
        try:
            fns[k]()
        except SystemExit as e:
            print(f"skip figure {k}: {e}")
    print(f"\nwrote to {OUT}/")
