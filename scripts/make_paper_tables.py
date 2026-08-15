"""Generate the paper's appendix tables directly from eval JSONLs.

Hand-maintained result tables drift from the data they describe. This project
already retracted two findings; a third caused by a stale number copied into a
table would be avoidable and embarrassing. So every table in the appendix is
generated here, from the JSONLs, at build time.

Emits `docs/paper/10_appendix_tables.md`. Run before `scripts/build_paper.sh`.

Each family declares its reference arm explicitly, and the reference must come
from the same measurement epoch as its arms; the epoch is stated per family so a
reader can check. Multiplicity corrections are applied within a family, never
across families. Arms whose JSONL is absent or incomplete are listed as pending
rather than silently dropped, so a half-finished sweep cannot masquerade as a
complete one.
"""

from __future__ import annotations

from pathlib import Path

from scripts.paper_stats import (bh, holm, load_arm, mcnemar_exact, mde,
                                 paired_bootstrap)

OUT = Path("docs/paper/10_appendix_tables.md")
R = Path("output/reeval")
CKPTS = list(range(5, 65, 5))
# Arms below this row count are treated as still running, not as results.
# The smallest complete arm in this project is valid_unseen at 134 games.
MIN_COMPLETE = 130

# (title, epoch note, reference arm or None, [(display name, path)])
FAMILIES = [
    ("Baseline replicates: is the control stable?",
     "August 2026, fixed harness. Absolute rates only; no reference arm.",
     None,
     [("canonical (May 2026, retired)", Path("output/eval-pathbv4/no_memory.jsonl")),
      ("replicate 1", R / "baseline/no_memory_8b.jsonl"),
      ("replicate 2", R / "baseline-replicates/no_memory_8b_run3.jsonl"),
      ("replicate 3", R / "baseline-replicates/no_memory_8b_run4.jsonl"),
      ("replicate 4", R / "baseline-replicates/no_memory_8b_run5.jsonl")]),

    ("ALFWorld-trained curator, full fine-tune seed-2, 8B executor",
     "August 2026, contemporaneous control",
     R / "baseline/no_memory_8b.jsonl",
     [(f"ckpt{c}", R / f"fft-seed2/ckpt{c}.jsonl") for c in CKPTS]),

    ("ALFWorld-trained curator, full fine-tune seed-1, 8B executor",
     "August 2026, contemporaneous control",
     R / "baseline/no_memory_8b.jsonl",
     [(f"ckpt{c}", R / f"fft/ckpt{c}.jsonl") for c in CKPTS]),

    ("Reasoning-trained curator on ALFWorld, valid-seen split",
     "August 2026, contemporaneous control",
     R / "baseline/no_memory_8b.jsonl",
     [(f"ckpt{c}", R / f"reasoning-to-alfworld/ckpt{c}.jsonl") for c in CKPTS]),

    ("Held-out valid-unseen split: curator comparison and content controls",
     "August 2026, contemporaneous control. 134 games.",
     R / "unseen-power/no_memory.jsonl",
     [("Gemini 2.5 Pro curator", R / "unseen-power/gemini_curator.jsonl"),
      ("trained curator ckpt45", R / "unseen-power/r2a_ckpt45.jsonl"),
      ("trained curator ckpt50", R / "unseen-power/r2a_ckpt50.jsonl"),
      ("trained curator ckpt55", R / "unseen-power/r2a_ckpt55.jsonl"),
      ("trained curator ckpt60", R / "unseen-power/r2a_ckpt60.jsonl"),
      ("ckpt50, shuffled retrieval", R / "unseen-power/r2a_ckpt50_shuffled.jsonl"),
      ("hand-written skills (oracle)", R / "unseen-power/oracle_handwritten.jsonl")]),

    ("32B executor transfer, full fine-tune seed-2 curator",
     "August 2026, contemporaneous control",
     R / "baseline/no_memory_32b.jsonl",
     [(f"ckpt{c}", R / f"transfer-32b-seed2/ckpt{c}.jsonl") for c in CKPTS]),

    ("verl/GiGPO curator, 8B executor",
     "August 2026, contemporaneous control",
     R / "baseline/no_memory_8b.jsonl",
     [(f"ckpt{c}", R / f"verl-gigpo-real/ckpt{c}.jsonl") for c in CKPTS]),
]


def render(title: str, epoch: str, ref: Path | None,
           arms: list[tuple[str, Path]]) -> str:
    lines = [f"## {title}", "", f"*{epoch}*", ""]

    # An arm that is still being written has a short JSONL. Including it produces
    # a table row with an absurd n and a meaningless CI, which is exactly the
    # kind of silent half-truth this script exists to prevent. Require at least
    # MIN_COMPLETE rows before an arm is eligible.
    def complete(p: Path) -> bool:
        p = Path(p)
        if not (p.exists() and p.stat().st_size):
            return False
        with open(p) as fh:
            return sum(1 for _ in fh) >= MIN_COMPLETE

    present = [(n, p) for n, p in arms if complete(p)]
    missing = [n for n, p in arms if not complete(p)]

    if not present:
        lines += ["Not yet measured.", ""]
        return "\n".join(lines)

    if ref is None:
        lines += ["| arm | success rate | games |", "|---|---|---|"]
        for name, path in present:
            a = load_arm(path)
            n_ok = sum(a.values())
            lines.append(f"| {name} | {n_ok / len(a) * 100:.1f}% | {len(a)} |")
        rates = []
        for name, path in present:
            if "retired" in name:
                continue
            a = load_arm(path)
            rates.append(sum(a.values()) / len(a) * 100)
        if len(rates) > 1:
            lines += ["", f"Same-epoch mean {sum(rates)/len(rates):.1f}%, "
                          f"spread {max(rates)-min(rates):.1f}pp over {len(rates)} "
                          f"replicates."]
        lines.append("")
        if missing:
            lines += [f"Pending: {', '.join(missing)}.", ""]
        return "\n".join(lines)

    if not (Path(ref).exists() and Path(ref).stat().st_size):
        lines += [f"Reference arm `{ref}` not yet measured; family withheld.", ""]
        return "\n".join(lines)

    base = load_arm(ref)
    rows = []
    for name, path in present:
        arm = load_arm(path)
        r = mcnemar_exact(base, arm)
        lo, hi = paired_bootstrap(base, arm)
        r.update(name=name, ci_lo=lo, ci_hi=hi, mde=mde(r["discordant"], r["n"]))
        rows.append(r)
    ps = [r["p"] for r in rows]
    for r, h, q in zip(rows, holm(ps), bh(ps)):
        r["p_holm"], r["q_bh"] = h, q

    lines += [f"Reference: {rows[0]['base_sr']:.1f}% "
              f"(`{Path(ref).name}`, n={rows[0]['n']} paired games). "
              f"MDE80 = smallest effect detectable at 80% power.", "",
              "| arm | SR | delta | 95% CI | p | p (Holm) | q (BH) | MDE80 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        star = " *" if r["p_holm"] < 0.05 else ""
        lines.append(
            f"| {r['name']}{star} | {r['arm_sr']:.1f}% | {r['delta_pp']:+.1f} | "
            f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {r['p']:.4f} | "
            f"{r['p_holm']:.3f} | {r['q_bh']:.3f} | {r['mde']:.1f} |")
    lines += ["", "\\* survives Holm correction within this family.", ""]
    if missing:
        lines += [f"Pending, not yet measured: {', '.join(missing)}.", ""]
    return "\n".join(lines)


def main() -> None:
    out = ["# Appendix C. Full result tables", "",
           "Generated by `scripts/make_paper_tables.py` from the released eval "
           "JSONLs. Do not edit by hand; regenerate. Every arm is 140 "
           "`valid_seen` games unless stated, paired by ALFWorld game file, "
           "tested with an exact McNemar. Confidence intervals are percentile "
           "intervals from 10,000 paired bootstrap resamples over game files, "
           "seed 20260815.", ""]
    for title, epoch, ref, arms in FAMILIES:
        out.append(render(title, epoch, ref, arms))
    OUT.write_text("\n".join(out))
    n_tables = sum(1 for f in FAMILIES)
    print(f"wrote {OUT} ({n_tables} families)")


if __name__ == "__main__":
    main()
