#!/usr/bin/env python3
"""Regenerate every figure in docs/repro_report.md from the eval artifacts.

Reads the McNemar comparator output (`comparison*.txt`) and the wandb
`output.log` for the verl run, so the figures cannot drift from the numbers:
if a sweep is re-run, re-run this and the plots follow.

    .venv/bin/python scripts/make_report_figures.py

Writes PNGs to docs/figures/. No GPQA per-problem content is read or written
(only aggregate accuracies, per the dataset access condition).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parsing ---

MCNEMAR_ROW = re.compile(
    r"^ckpt(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+([+-][\d.]+)%\s+([\d.]+)\s*$"
)
OVERALL_ROW = re.compile(r"^(\S+)\s+(\d+)/(\d+)\s*=\s*([\d.]+)%")


def parse_comparison(path: Path) -> dict:
    """-> {'delta': {ckpt: pp}, 'p': {ckpt: p}, 'sr': {arm: fraction}}"""
    delta, pval, sr = {}, {}, {}
    for line in path.read_text().splitlines():
        m = MCNEMAR_ROW.match(line.strip())
        if m:
            ck = int(m.group(1))
            delta[ck] = float(m.group(2))
            pval[ck] = float(m.group(3))
            continue
        m = OVERALL_ROW.match(line.strip())
        if m:
            sr[m.group(1)] = float(m.group(4))
    return {"delta": delta, "p": pval, "sr": sr}


def parse_wandb_log(path: Path, keys: list[str]) -> dict[str, dict[int, float]]:
    """Pull `step:N - k:v - k:v ...` lines out of a wandb console log."""
    out = {k: {} for k in keys}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("step:"):
            continue
        step = int(line.split(" ", 1)[0].split(":")[1])
        for field in line.split(" - "):
            if ":" not in field:
                continue
            k, _, v = field.rpartition(":")
            if k in out:
                try:
                    out[k][step] = float(v)
                except ValueError:
                    pass
    return out


# ------------------------------------------------------------------- data ---

# Labels are written for a reader who has never seen this repo: framework first,
# then what distinguishes the run. Internal codenames (v8, pathbv4) stay out.
SWEEPS = {
    "TRL  ·  LoRA": "output/eval-v8/comparison_canonical.txt",
    "TRL  ·  seed 1": "output/eval-fft/comparison_canonical.txt",
    "TRL  ·  seed 2": "output/eval-fft-seed2/comparison_canonical.txt",
    "TRL  ·  seed 3": "output/eval-fft-seed3/comparison_canonical.txt",
    "verl / GiGPO": "output/eval-verl-gigpo-real/comparison_canonical.txt",
}
ABLATIONS = {
    "natural task mix (unbalanced)": "output/eval-fft-natural/comparison_canonical.txt",
    "easy-to-hard ordering": "output/eval-fft-curriculum/comparison_canonical.txt",
}
TRANSFER_32B = {
    "seed 2": "output/eval-transfer-32b-seed2/comparison.txt",
    "seed 3": "output/eval-transfer-32b-seed3/comparison.txt",
}
REASONING_TRANSFER = "output/eval-reasoning-to-alfworld/comparison.txt"
VERL_WANDB = "/home/ubuntu/verl-skillos/wandb/run-20260730_081941-7rm65scp/files/output.log"
VERL_TRAIN_LOG = "logs/verl_skillos_gigpo_alfworld.log"

# paper weights (Ouyang et al. §3.2): r = r_task + lf*r_fc + lu*r_cnt + lc*r_comp
LAMBDA = {"r_task": 1.0, "r_fc": 1.0, "r_cnt": 0.1, "r_comp": 0.05}

# What each reward term actually measures, for readers who have not read the paper.
PLAIN = {
    "r_task": "did the agent\nsucceed?",
    "r_fc": "were the edits\nwell-formed?",
    "r_cnt": "judge score",
    "r_comp": "compression",
}

REWARD_PARTS = re.compile(
    r"REWARD_PARTS group=(\d+) round=(\d+) r_task=([\d.]+) r_fc=([\d.]+) "
    r"r_cnt=([\d.]+) r_comp=([\d.]+) total=([\d.]+)"
)


def parse_reward_parts(path: Path) -> list[dict]:
    """Per-rollout composite-reward decomposition emitted by the verl env."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = REWARD_PARTS.search(line)
        if m:
            rows.append({
                "group": m.group(1), "round": int(m.group(2)),
                "r_task": float(m.group(3)), "r_fc": float(m.group(4)),
                "r_cnt": float(m.group(5)), "r_comp": float(m.group(6)),
                "total": float(m.group(7)),
            })
    return rows

COLORS = ["#2563EB", "#F59E0B", "#059669", "#DC2626", "#1E1B4B"]

# --- house style -------------------------------------------------------------
# Editorial look for the public write-up: white ground, no chartjunk, one accent
# family shared with the repo banner (indigo/violet), warm amber and green for
# the other series, red reserved for negatives. Data is untouched; this only
# changes typography, spines, grid and DPI.
INK = "#111827"
MUTED = "#6B7280"
GRID = "#D1D5DB"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.edgecolor": "#9CA3AF",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 11,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.55,
    "lines.solid_capstyle": "round",
})


def style_axes(ax, ygrid=True, xgrid=False):
    """Light y-only grid behind the data, ticks trimmed."""
    ax.grid(axis="y", ls=":", zorder=0) if ygrid else None
    ax.grid(axis="x", ls=":", zorder=0) if xgrid else None
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def titles(fig, headline, standfirst=None):
    """Left-aligned headline + grey standfirst, positioned in INCHES.

    Fraction-based y positions collide on short figures and float away on tall
    ones, so measure from the top edge in inches and convert. Returns the `top`
    to pass to tight_layout(rect=...) so the axes never ride up into the text.
    """
    h = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.30 / h, headline, ha="left", va="center",
             fontsize=17, weight="bold", color=INK)
    if standfirst:
        fig.text(0.012, 1 - 0.64 / h, standfirst, ha="left", va="center",
                 fontsize=12, color=MUTED)
        return 1 - 0.98 / h
    return 1 - 0.55 / h


def _sig_marker(p: float) -> str:
    if p < 0.01:
        return "o"
    if p < 0.05:
        return "s"
    return ""


# ------------------------------------------------------------- figure 1 -----


def fig_reward_composition():
    """The reward machinery is healthy, yet r_task barely moves.

    Two shares are worth distinguishing. The composite reward's *level* is
    dominated by r_fc (a near-saturated function-call-validity term), which
    looks alarming, but GRPO centres advantages within each group, so only
    *within-group variance* reaches the gradient, and there r_task dominates.
    Decomposition: Var(total) = sum_i Cov(w_i x_i, total), pooled over groups.
    """
    rows = parse_reward_parts(ROOT / VERL_TRAIN_LOG)
    parts = ["r_task", "r_fc", "r_cnt", "r_comp"]

    level = {k: LAMBDA[k] * float(np.mean([r[k] for r in rows])) for k in parts}
    level_tot = sum(level.values())

    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    cov = {k: 0.0 for k in parts}
    vtot = 0.0
    n_g = 0
    for rs in groups.values():
        if len(rs) < 2:
            continue
        n_g += 1
        tot = np.array([r["total"] for r in rs])
        vtot += float(tot.var())
        for k in parts:
            x = LAMBDA[k] * np.array([r[k] for r in rs])
            cov[k] += float(((x - x.mean()) * (tot - tot.mean())).mean())

    colors = {"r_task": "#d62728", "r_fc": "#7f7f7f",
              "r_cnt": "#9467bd", "r_comp": "#8c564b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: reward level vs within-group variance, side by side
    series = [("what the reward\nis made of", {k: 100 * level[k] / level_tot for k in parts}),
              ("what actually\nteaches the model",
               {k: 100 * cov[k] / vtot for k in parts})]
    for y, (label, shares) in enumerate(series):
        left = 0.0
        for k in parts:
            f = shares[k]
            ax1.barh([y], [f], left=left, color=colors[k], edgecolor="w",
                     height=0.55)
            if f > 12:
                ax1.text(left + f / 2, y, f"{PLAIN[k]}\n{f:.0f}%", ha="center",
                         va="center", fontsize=10.5, color="w", weight="bold",
                         linespacing=1.35)
            left += f
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([s[0] for s in series], fontsize=8.5)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_xlabel("share of the training signal  (%)")
    ax1.text(0, -0.42, "The two thin slices are a judge's quality score and a "
             "compression bonus.", transform=ax1.transAxes, fontsize=10,
             color=MUTED, va="top")
    ax1.set_title("the reward looks like one thing, teaches another",
                  fontsize=12.5, loc="left")
    ax1.tick_params(length=0)
    for sp in ("bottom", "left"): ax1.spines[sp].set_visible(True)

    # -- right: r_task over the run, with a 95% CI on the early→late change
    by_round = {}
    for r in rows:
        by_round.setdefault(r["round"], []).append(r["r_task"])
    xs = sorted(by_round)
    ys = [float(np.mean(by_round[x])) for x in xs]
    ax2.plot(xs, ys, color=colors["r_task"], lw=1.0, alpha=0.35)
    roll = np.convolve(ys, np.ones(5) / 5, mode="valid")
    ax2.plot(xs[4:], roll, color=colors["r_task"], lw=3.0,
             label="agent success rate, 5-round average")
    early = np.array([v for x in xs if x <= 10 for v in by_round[x]])
    late = np.array([v for x in xs if x >= 51 for v in by_round[x]])
    for span, arr, lbl in (((1, 10), early, "first 10 rounds"),
                           ((51, 60), late, "last 10 rounds")):
        ax2.hlines(arr.mean(), *span, color=INK, lw=2.8)
        ax2.annotate(f"{lbl}\n{arr.mean():.3f}", (np.mean(span), arr.mean()),
                     textcoords="offset points", xytext=(0, -34), ha="center",
                     fontsize=8.5, color=INK)
    d = late.mean() - early.mean()
    se = float(np.sqrt(early.var(ddof=1) / len(early) + late.var(ddof=1) / len(late)))
    ax2.set_title(f"60 rounds of training buys {d:+.3f} \u00b1 {1.96 * se:.3f}",
                  fontsize=12.5, loc="left")
    ax2.set_xlabel("training round")
    ax2.set_ylabel("share of tasks the agent solved")
    ax2.set_ylim(0.15, 0.55)
    ax2.legend(loc="upper left")
    style_axes(ax2)

    top = titles(fig,
      "The reward was not broken. The learning was just weak.",
      "Most of the reward is a formatting term that never varies, so it cancels "
      "out. What teaches the model is task success, and it barely moves.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "fig1_reward_composition.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 2 -----


def fig_sweeps():
    """The headline: five independent runs, all non-monotone, peaks unaligned."""
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for (label, rel), color in zip(SWEEPS.items(), COLORS):
        d = parse_comparison(ROOT / rel)
        cks = sorted(d["delta"])
        ys = [d["delta"][c] for c in cks]
        lw = 3.0 if "verl" in label else 1.9
        ax.plot(cks, ys, "-", color=color, lw=lw, label=label, alpha=0.95,
                marker="o", ms=4.5, mfc="white", mew=1.4, zorder=3)
        # ring the significant arms
        for c in cks:
            mk = _sig_marker(d["p"][c])
            if mk:
                ax.plot([c], [d["delta"][c]], mk, mfc="none", mec=color,
                        ms=13, mew=2)
        peak = max(cks, key=lambda c: d["delta"][c])
        ax.annotate(f"{d['delta'][peak]:+.1f}", (peak, d["delta"][peak]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=color, weight="bold")

    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.axhspan(-3, 3, color="#E5E7EB", alpha=0.75, zorder=0)
    # label the noise band off to the right of the last checkpoint, clear of data
    ax.text(63.2, -1.5, "too small\nto call", va="center", fontsize=10,
            color=MUTED, style="italic")
    ax.set_xlabel("training step  (checkpoint saved every 5 steps)")
    ax.set_ylabel("change in agent success rate\nvs. no memory  (percentage points)")
    top = titles(fig,
      "More training does not steadily make the agent better",
      "Each point is 140 unseen household tasks, scored task by task against the "
      "same memory-free baseline. Every run peaks somewhere. None agree where.")
    ax.set_xticks(range(5, 65, 5))
    ax.set_xlim(2, 70)
    # legend below the axes: an in-axes legend hides the v8-LoRA ckpt10 point
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5)
    style_axes(ax)
    fig.text(0.988, 0.015,
             "circled: p<0.01    boxed: p<0.05    None of these survive "
             "correcting for the 50 checkpoints tested.",
             ha="right", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, top))
    fig.savefig(OUT / "fig2_checkpoint_sweeps.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 2 -----


def fig_decorrelation():
    """Best-on-8B is not best-on-32B: per-checkpoint 8B vs 32B lift."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    pairs = [("seed 2", "output/eval-fft-seed2/comparison_canonical.txt"),
             ("seed 3", "output/eval-fft-seed3/comparison_canonical.txt")]
    allx, ally = [], []
    for ax, (label, rel8) in zip(axes, pairs):
        d8 = parse_comparison(ROOT / rel8)["delta"]
        d32 = parse_comparison(ROOT / TRANSFER_32B[label])["delta"]
        cks = sorted(set(d8) & set(d32))
        xs = [d8[c] for c in cks]
        ys = [d32[c] for c in cks]
        allx += xs
        ally += ys
        sc = ax.scatter(xs, ys, c=cks, cmap="viridis", s=90, zorder=3,
                        edgecolor="k", lw=0.5)
        for c, x, y in zip(cks, xs, ys):
            ax.annotate(str(c), (x, y), textcoords="offset points",
                        xytext=(7, -3), fontsize=7.5)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        lim = max(abs(v) for v in xs + ys) + 3
        ax.plot([-lim, lim], [-lim, lim], ls="--", color="grey", lw=1,
                label="if the two agreed exactly")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{label}   ·   Pearson r = {r:+.2f}", fontsize=11, loc="left")
        ax.set_xlabel("helped the 8B agent  (percentage points)")
        ax.set_ylabel("helped the 32B agent\n(percentage points)")
        style_axes(ax, ygrid=True, xgrid=True)
        ax.legend(loc="lower right")
        fig.colorbar(sc, ax=ax, label="training step", pad=0.02)

    r_all = np.corrcoef(allx, ally)[0, 1]
    top = titles(fig,
      "A curator that helps a small agent may not help a big one",
      f"How much a checkpoint helps the 8B agent barely predicts how much it "
      f"helps the 32B agent. Correlation {r_all:+.2f} across 24 pairs.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "fig3_8b_32b_decorrelation.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 3 -----


def fig_verl_training():
    """verl/GiGPO training dynamics: entropy collapse, and the step-40 outage."""
    keys = ["episode/reward/mean", "episode/success_rate", "actor/entropy_loss",
            "actor/grad_norm", "critic/advantages/max", "critic/advantages/min"]
    m = parse_wandb_log(Path(VERL_WANDB), keys)
    steps = sorted(m["episode/reward/mean"])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    panels = [
        ("episode/reward/mean", "composite reward (mean)", "#1f77b4"),
        ("episode/success_rate", "train success rate", "#2ca02c"),
        ("actor/entropy_loss", "policy entropy", "#9467bd"),
        ("actor/grad_norm", "grad norm", "#d62728"),
    ]
    for ax, (key, title, color) in zip(axes.flat, panels):
        xs = sorted(m[key])
        ys = [m[key][s] for s in xs]
        ax.plot(xs, ys, color=color, lw=1.1, alpha=0.45)
        # 5-step rolling mean to show trend through the step noise
        if len(ys) >= 5:
            roll = np.convolve(ys, np.ones(5) / 5, mode="valid")
            ax.plot(xs[4:], roll, color=color, lw=2.3, label="5-step mean")
            ax.legend(fontsize=7.5, loc="best")
        ax.axvline(40, color=MUTED, ls="--", lw=1)
        ax.set_title(title, fontsize=11, loc="left")
        style_axes(ax)
    axes[1][0].set_xlabel("training step")
    axes[1][1].set_xlabel("training step")
    axes[0][0].annotate("step 40: executor outage",
                        xy=(40, 0.06), xycoords=("data", "axes fraction"),
                        xytext=(37, 0.06), textcoords=("data", "axes fraction"),
                        fontsize=7.5, color="grey", ha="right")
    ent = m["actor/entropy_loss"]
    e0 = np.mean([ent[s] for s in sorted(ent)[:5]])
    e1 = np.mean([ent[s] for s in sorted(ent)[-5:]])
    sr = m["episode/success_rate"]
    s0 = np.mean([sr[s] for s in sorted(sr)[:10]])
    s1 = np.mean([sr[s] for s in sorted(sr)[-10:]])
    top = titles(fig,
      "Reward rises, the agent does not improve, the policy narrows",
      f"verl/GiGPO on real ALFWorld · train success {s0:.3f} → {s1:.3f} "
      f"(flat) · policy entropy {e0:.3f} → {e1:.3f}")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "fig4_verl_training_dynamics.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 4 -----


def fig_reasoning_transfer():
    """Cross-domain transfer: paper claims +13.3pp, we measure a cliff."""
    d = parse_comparison(ROOT / REASONING_TRANSFER)
    cks = sorted(d["delta"])
    ys = [d["delta"][c] for c in cks]
    ps = [d["p"][c] for c in cks]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    colors = ["#059669" if y > 0 else "#DC2626" for y in ys]
    bars = ax.bar([str(c) for c in cks], ys, color=colors, alpha=0.85,
                  edgecolor="k", lw=0.5)
    for b, y, p in zip(bars, ys, ps):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.annotate(f"{y:+.1f}{star}",
                    (b.get_x() + b.get_width() / 2, y),
                    textcoords="offset points",
                    xytext=(0, 4 if y > 0 else -13),
                    ha="center", fontsize=8)
    ax.axhline(0, color=INK, lw=1.2)
    ax.axhline(13.3, color="#2563EB", ls="--", lw=1.8,
               label="what the paper reports: +13.3 points")
    ax.axhspan(-3, 3, color="#E5E7EB", alpha=0.75, zorder=0)
    ax.set_ylim(min(ys) - 5, 16)  # headroom so the -17.9 label isn't clipped
    ax.set_xlabel("training step of the maths-trained curator")
    ax.set_ylabel("change in agent success rate\non household tasks  (percentage points)")
    top = titles(fig,
      "Skills learned on maths actively hurt a household-task agent",
      "The paper reports a 13.3 point gain in this direction. Past step 40 I "
      "measure a 14 to 18 point loss.")
    ax.legend(loc="lower left")
    style_axes(ax)
    fig.text(0.988, 0.015, "* p<0.05   ** p<0.01   *** p<0.001", ha="right",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, top))
    fig.savefig(OUT / "fig5_reasoning_transfer_cliff.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 5 -----


def fig_ablations():
    """Both halves of the paper's grouping ablation are null on our stack."""
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ref = parse_comparison(ROOT / SWEEPS["TRL  ·  seed 2"])["delta"]
    cks = sorted(ref)
    ax.plot(cks, [ref[c] for c in cks], "-", color="#059669", lw=3.2,
            marker="o", ms=5, mfc="white", mew=1.5,
            label="balanced task mix (as in the paper)")
    for (label, rel), color in zip(ABLATIONS.items(), ["#F59E0B", "#7C3AED"]):
        d = parse_comparison(ROOT / rel)["delta"]
        cks = sorted(d)
        ax.plot(cks, [d[c] for c in cks], "-", color=color, lw=1.9,
                marker="o", ms=4.5, mfc="white", mew=1.4, label=label)
    ax.axhline(0, color=INK, lw=1.2)
    ax.axhspan(-3, 3, color="#E5E7EB", alpha=0.75, zorder=0)
    ax.set_xlabel("training step  (checkpoint saved every 5 steps)")
    ax.set_ylabel("change in agent success rate\nvs. no memory  (percentage points)")
    top = titles(fig,
      "Grouping matters, but it is the task mix, not the ordering",
      "Both alternatives to a balanced task mix produce no measurable gain")
    ax.set_xticks(range(5, 65, 5))
    ax.legend(loc="upper left")
    style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "fig6_grouping_ablations.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_reward_composition()
    fig_sweeps()
    fig_decorrelation()
    fig_verl_training()
    fig_reasoning_transfer()
    fig_ablations()
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size // 1024}KB")
