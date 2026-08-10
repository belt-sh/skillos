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

SWEEPS = {
    "v8 LoRA r=32 (TRL)": "output/eval-v8/comparison_canonical.txt",
    "FFT seed-1 (TRL)": "output/eval-fft/comparison_canonical.txt",
    "FFT seed-2 (TRL)": "output/eval-fft-seed2/comparison_canonical.txt",
    "FFT seed-3 (TRL)": "output/eval-fft-seed3/comparison_canonical.txt",
    "GiGPO (verl, real env)": "output/eval-verl-gigpo-real/comparison_canonical.txt",
}
ABLATIONS = {
    "natural type frequencies": "output/eval-fft-natural/comparison_canonical.txt",
    "easy→hard curriculum": "output/eval-fft-curriculum/comparison_canonical.txt",
}
TRANSFER_32B = {
    "FFT seed-2": "output/eval-transfer-32b-seed2/comparison.txt",
    "FFT seed-3": "output/eval-transfer-32b-seed3/comparison.txt",
}
REASONING_TRANSFER = "output/eval-reasoning-to-alfworld/comparison.txt"
VERL_WANDB = "/home/ubuntu/verl-skillos/wandb/run-20260730_081941-7rm65scp/files/output.log"
VERL_TRAIN_LOG = "logs/verl_skillos_gigpo_alfworld.log"

# paper weights (Ouyang et al. §3.2): r = r_task + lf*r_fc + lu*r_cnt + lc*r_comp
LAMBDA = {"r_task": 1.0, "r_fc": 1.0, "r_cnt": 0.1, "r_comp": 0.05}

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

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#000000"]


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
    looks alarming — but GRPO centres advantages within each group, so only
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: reward level vs within-group variance, side by side
    series = [("reward\nlevel", {k: 100 * level[k] / level_tot for k in parts}),
              ("within-group\nvariance\n(what the GRPO\nadvantage sees)",
               {k: 100 * cov[k] / vtot for k in parts})]
    for y, (label, shares) in enumerate(series):
        left = 0.0
        for k in parts:
            f = shares[k]
            ax1.barh([y], [f], left=left, color=colors[k], edgecolor="w",
                     height=0.55)
            if f > 5:
                ax1.text(left + f / 2, y, f"{k}\n{f:.0f}%", ha="center",
                         va="center", fontsize=9, color="w", weight="bold")
            left += f
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels([s[0] for s in series], fontsize=8.5)
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_xlabel("share of composite reward (%)")
    ax1.set_title("r_fc dominates the reward level, but r_task\n"
                  "dominates the signal that reaches the gradient",
                  fontsize=10)

    # -- right: r_task over the run, with a 95% CI on the early→late change
    by_round = {}
    for r in rows:
        by_round.setdefault(r["round"], []).append(r["r_task"])
    xs = sorted(by_round)
    ys = [float(np.mean(by_round[x])) for x in xs]
    ax2.plot(xs, ys, color=colors["r_task"], lw=1.0, alpha=0.35)
    roll = np.convolve(ys, np.ones(5) / 5, mode="valid")
    ax2.plot(xs[4:], roll, color=colors["r_task"], lw=2.6, label="r_task, 5-round mean")
    early = np.array([v for x in xs if x <= 10 for v in by_round[x]])
    late = np.array([v for x in xs if x >= 51 for v in by_round[x]])
    for span, arr, lbl in (((1, 10), early, "rounds 1–10"),
                           ((51, 60), late, "rounds 51–60")):
        ax2.hlines(arr.mean(), *span, color="k", lw=2.4)
        ax2.annotate(f"{lbl}\n{arr.mean():.3f}", (np.mean(span), arr.mean()),
                     textcoords="offset points", xytext=(0, -30), ha="center",
                     fontsize=8)
    d = late.mean() - early.mean()
    se = float(np.sqrt(early.var(ddof=1) / len(early) + late.var(ddof=1) / len(late)))
    ax2.set_title(f"downstream task reward over 60 rounds:\n"
                  f"{d:+.3f} (95% CI ±{1.96 * se:.3f}) — barely distinguishable "
                  f"from flat", fontsize=10)
    ax2.set_xlabel("Algorithm-1 round (≈ GRPO step)")
    ax2.set_ylabel("r_task (executor success over positions 2..|G|)")
    ax2.set_ylim(0.15, 0.55)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(alpha=0.25, ls=":")

    fig.suptitle("The reward machinery works as designed — and the curator still "
                 "barely improves the executor", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_reward_composition.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 2 -----


def fig_sweeps():
    """The headline: five independent runs, all non-monotone, peaks unaligned."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, rel), color in zip(SWEEPS.items(), COLORS):
        d = parse_comparison(ROOT / rel)
        cks = sorted(d["delta"])
        ys = [d["delta"][c] for c in cks]
        lw = 2.4 if "verl" in label else 1.6
        ax.plot(cks, ys, "-", color=color, lw=lw, label=label, alpha=0.9,
                marker=".", ms=8)
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

    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-3, 3, color="grey", alpha=0.13, zorder=0)
    # label the noise band off to the right of the last checkpoint, clear of data
    ax.text(62.5, -1.4, "n=140\nnoise band", va="center", fontsize=7.5,
            color="grey")
    ax.set_xlabel("curator checkpoint (GRPO step)")
    ax.set_ylabel("Δ success rate vs no-memory (pp)")
    ax.set_title("ALFWorld held-out lift is non-monotone in five independent runs\n"
                 "140 paired games, McNemar vs a fixed 33.6% no-memory baseline",
                 fontsize=11)
    ax.set_xticks(range(5, 65, 5))
    ax.set_xlim(2, 70)
    # legend below the axes: an in-axes legend hides the v8-LoRA ckpt10 point
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=5, frameon=False)
    ax.grid(alpha=0.25, ls=":")
    fig.text(0.985, 0.015,
             "○ p<0.01   □ p<0.05  (uncorrected; nothing survives "
             "family-wide Bonferroni)",
             ha="right", fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig2_checkpoint_sweeps.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 2 -----


def fig_decorrelation():
    """Best-on-8B is not best-on-32B: per-checkpoint 8B vs 32B lift."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    pairs = [("FFT seed-2", "output/eval-fft-seed2/comparison_canonical.txt"),
             ("FFT seed-3", "output/eval-fft-seed3/comparison_canonical.txt")]
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
                label="perfect agreement")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{label}   Pearson r = {r:+.2f}", fontsize=10)
        ax.set_xlabel("Δ on Qwen3-8B executor (pp)")
        ax.set_ylabel("Δ on Qwen3-32B executor (pp)")
        ax.grid(alpha=0.25, ls=":")
        ax.legend(fontsize=7.5, loc="lower right")
        fig.colorbar(sc, ax=ax, label="checkpoint", pad=0.02)

    r_all = np.corrcoef(allx, ally)[0, 1]
    fig.suptitle("Curator quality does not transfer across executor scale "
                 f"(pooled r = {r_all:+.2f}, n=24)", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_8b_32b_decorrelation.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 3 -----


def fig_verl_training():
    """verl/GiGPO training dynamics: entropy collapse, and the step-40 outage."""
    keys = ["episode/reward/mean", "episode/success_rate", "actor/entropy_loss",
            "actor/grad_norm", "critic/advantages/max", "critic/advantages/min"]
    m = parse_wandb_log(Path(VERL_WANDB), keys)
    steps = sorted(m["episode/reward/mean"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
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
        ax.axvline(40, color="grey", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, ls=":")
    axes[1][0].set_xlabel("GRPO step")
    axes[1][1].set_xlabel("GRPO step")
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
    fig.suptitle("verl/GiGPO real-ALFWorld run: composite reward rises, but train "
                 f"success is flat ({s0:.3f} → {s1:.3f}) and entropy collapses "
                 f"({e0:.3f} → {e1:.3f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_verl_training_dynamics.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 4 -----


def fig_reasoning_transfer():
    """Cross-domain transfer: paper claims +13.3pp, we measure a cliff."""
    d = parse_comparison(ROOT / REASONING_TRANSFER)
    cks = sorted(d["delta"])
    ys = [d["delta"][c] for c in cks]
    ps = [d["p"][c] for c in cks]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = ["#2ca02c" if y > 0 else "#d62728" for y in ys]
    bars = ax.bar([str(c) for c in cks], ys, color=colors, alpha=0.85,
                  edgecolor="k", lw=0.5)
    for b, y, p in zip(bars, ys, ps):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.annotate(f"{y:+.1f}{star}",
                    (b.get_x() + b.get_width() / 2, y),
                    textcoords="offset points",
                    xytext=(0, 4 if y > 0 else -13),
                    ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(13.3, color="#1f77b4", ls="--", lw=1.6,
               label="paper's claim: +13.3pp")
    ax.axhspan(-3, 3, color="grey", alpha=0.13, zorder=0)
    ax.set_ylim(min(ys) - 5, 16)  # headroom so the -17.9 label isn't clipped
    ax.set_xlabel("reasoning-curator checkpoint (GRPO step, DeepMath-103K)")
    ax.set_ylabel("Δ ALFWorld success rate vs no-memory (pp)")
    ax.set_title("Cross-domain transfer reverses sign: a reasoning-trained curator\n"
                 "actively harms an embodied executor after step 40", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, ls=":", axis="y")
    fig.text(0.985, 0.02, "* p<0.05   ** p<0.01   *** p<0.001", ha="right",
             fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig5_reasoning_transfer_cliff.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------- figure 5 -----


def fig_ablations():
    """Both halves of the paper's grouping ablation are null on our stack."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ref = parse_comparison(ROOT / SWEEPS["FFT seed-2 (TRL)"])["delta"]
    cks = sorted(ref)
    ax.plot(cks, [ref[c] for c in cks], "-", color="#2ca02c", lw=2.2,
            marker=".", ms=9, label="uniform round-robin (as in paper)")
    for (label, rel), color in zip(ABLATIONS.items(), ["#ff7f0e", "#9467bd"]):
        d = parse_comparison(ROOT / rel)["delta"]
        cks = sorted(d)
        ax.plot(cks, [d[c] for c in cks], "-", color=color, lw=1.7,
                marker=".", ms=8, label=label)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-3, 3, color="grey", alpha=0.13, zorder=0)
    ax.set_xlabel("curator checkpoint (GRPO step)")
    ax.set_ylabel("Δ success rate vs no-memory (pp)")
    ax.set_title("Task-grouping ablations: both alternatives to uniform "
                 "round-robin are null", fontsize=11)
    ax.set_xticks(range(5, 65, 5))
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
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
