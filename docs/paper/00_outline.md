# Paper outline and claim ledger

Working title:

> **A Reproduction of SkillOS Under Contemporaneous Controls: The Curator Lift
> Does Not Survive a Same-Epoch Baseline**

Alternative, less combative:

> **Independent Reproduction of SkillOS: Seven Training Runs, Two Frameworks,
> and What Happens When You Re-Measure the Baseline**

Venue: arXiv (cs.LG / cs.AI), reproduction-study format. Not a refutation. The
honest frame is "we implemented it faithfully, we could not obtain the effect,
here is everything we ran and every way we tried to obtain it."

---

## Section map

| # | File | Status |
|---|---|---|
| 1 | `01_abstract.md` | drafted, numbers pending |
| 2 | `02_introduction.md` | drafted |
| 3 | `03_background_related.md` | drafted |
| 4 | `04_methodology.md` | drafted |
| 5 | `05_results.md` | skeleton, blocked on re-measurement |
| 6 | `06_threats.md` | drafted |
| 7 | `07_discussion.md` | drafted |
| A | `08_appendix_conduct.md` | drafted |
| B | `09_appendix_incidents.md` | to write |
| C | `10_appendix_tables.md` | auto-generate from JSONLs |

---

## Claim ledger

Every claim the paper will make, with the evidence and its current status.
Nothing enters the paper at status **pending** or **retracted**.

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | A same-epoch no-memory baseline is 39.8% ± 2.1pp; the 33.6% figure used for ten weeks is outside that spread | 4 replicates, 140 games | **solid** |
| C2 | With a same-epoch baseline, no ALFWorld checkpoint shows a significant lift on the training executor | seed-1 (7 arms), seed-2 (12 arms) re-run | **solid** |
| C3 | The trajectory is non-monotone with a seed-dependent peak index | 3 TRL seeds + verl, peaks at ckpt20/35/55/30 | **solid** (shape claim only) |
| C4 | Two independent RL frameworks produce the same shape, so it is not a framework artifact | TRL+ZeRO3 vs verl-agent/GiGPO+FSDP | **solid**, re-pairing in flight |
| C5 | A frontier curator (Gemini 2.5 Pro) is not better than no notes | 140 games, -1.4pp, p=0.86 | **solid** |
| C6 | The trained 8B curator is not worse than the frontier curator at 84x lower cost | +5.7pp, p=0.31, `valid_unseen` in flight | **pending power** |
| C7 | Curation raises the executor's unparseable-action rate 2.1% -> 7.0-9.1% | parse telemetry, all arms | **solid** |
| C8 | Curator quality does not transfer across executor scale | 24 checkpoint pairs, pooled r = -0.20 | **pending re-pairing (wave C)** |
| C9 | Cross-domain (reasoning -> ALFWorld) transfer is null, not negative | 4 arms re-run post-fix | **solid**, was retracted-negative |
| C10 | Six candidate causes of the null are falsified | LoRA/FFT, framework, task distribution, curriculum, decode, prompt | **solid** |
| C11 | Our absolute baseline sits 8pp below the paper's, unexplained (was 14pp against the stale control) | 6 ruled-out causes | **solid** as a limitation |
| C12 | Within-group reward variance is task-dominated (79%), so the optimiser chases the right signal | 850 verl rollouts, 80 groups | **solid** |

Retired claims, to appear only in the retraction appendix:

- "Held-out lift is real at some checkpoint in every run (+7.1 to +13.6pp)."
  Void: paired against the drifted baseline.
- "Cross-domain transfer reproduces with the opposite sign (-14 to -18pp)."
  Void: measured during an API outage the harness scored as task failure.

---

## The three things that make this publishable rather than a blog post

1. **N.** Seven 60-step training runs, two RL frameworks, three seeds, two
   executor scales, roughly 100 evaluation arms of 140 paired games each.
2. **Falsification discipline.** Each candidate explanation for the null got its
   own full training run, not an argument.
3. **A transferable methodological result.** C1 is not about SkillOS. Anyone
   benchmarking against a hosted model API can manufacture a publishable effect
   by reusing a control measured a month earlier. We have the measurement that
   shows the size of that trap in practice.

## Gates before submission

- [ ] Every number in the paper measured in one epoch under the fixed harness.
- [ ] `valid_unseen` split completed, C6 resolved either way.
- [ ] Wave C completed, C4 and C8 re-paired.
- [ ] Authors of the original paper contacted with the setup and the null, and
      given time to respond. Their response summarised in the paper.
- [ ] Every figure regenerable from released JSONLs by a single script.
- [ ] GPQA: aggregate accuracies only, no problem text or model responses in the
      paper, appendix, or released artifacts.
