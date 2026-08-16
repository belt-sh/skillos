# 5. Results

All numbers in this section are paired by ALFWorld game file against a control
measured in the same week under the same harness build. Full tables with
bootstrap intervals, Holm-corrected p-values, BH q-values, and per-arm minimum
detectable effects are in Appendix C, generated directly from the released
evaluation records.

Markers: **[PENDING]** flags a number awaiting a run that is still in flight at
the time of writing. Nothing marked PENDING may appear in the abstract or in a
figure.

## 5.1 The control was not stable, and everything followed from that

![The control moved further than the effect under study. Same 140 ALFWorld games and the same harness build; the shaded band is the same-week replicate range.](figures/fig1_control_drift.pdf){width=90%}


Our no-memory Qwen3-8B baseline on 140 `valid_seen` games was measured once, in
May 2026, at 33.6%. Four replicates in August 2026, same games, same harness:

| run | success rate |
|---|---|
| May 2026 (used as canonical for ten weeks) | 33.6% |
| August replicate 1 | 39.3% |
| August replicate 2 | 39.3% |
| August replicate 3 | 39.3% |
| August replicate 4 | 41.4% |

Same-week mean 39.8%, spread 2.1pp. Genuine run-to-run variance at temperature
0.6 is therefore about 2pp, and the May figure lies outside it. The 32B control
moved similarly, from 49.3% to 43.6%, in the opposite direction, which rules out
a simple monotone improvement in the served model as the sole explanation.

We cannot attribute the shift to a single cause. Both the hosted endpoint and our
harness changed between the two measurements, and the harness change (Appendix
B.1) is itself sufficient to move a baseline. The relevant point is not the
mechanism but the magnitude: **a control that is not re-measured alongside its
arms can move further than the effect being studied.**

Every lift previously reported from this project was computed as
`arm - 33.6%`. Section 5.2 shows what happens when that is corrected.

## 5.2 With a contemporaneous control, the ALFWorld-trained curator shows no lift

![The same twelve seed-2 checkpoints against two reference measurements. Both series use identical arm measurements and differ only in which control is subtracted.](figures/fig2_repairing_the_sweep.pdf){width=90%}


Full 12-checkpoint sweep, full fine-tune seed-2, the run that had previously
produced our strongest result:

| | previously reported (vs 33.6%) | re-paired (vs 39.3%) |
|---|---|---|
| best arm | ckpt35, **+13.6pp**, p=0.0026 | ckpt35, +3.6pp, p=0.47 |
| arms above control | 11 of 12 | 4 of 12 |
| survives Holm within family | no | no |

Seed-1 behaves the same way: 5 of 7 re-paired arms are negative and the best is
+5.7pp. The oscillating checkpoint trajectory we had documented across three
seeds and two frameworks, and had spent four full training runs trying to
explain, was substantially an artifact of the shared stale reference. What remains
is a flat, noisy sweep with no trend.

The 95% bootstrap intervals on these arms span roughly ±8pp (Appendix C), and the
per-arm MDE at 80% power is 9 to 12pp. **We have not shown that the curator does
nothing.** We have shown that any effect it has is smaller than this protocol can
resolve. Section 5.x develops this.

## 5.3 Cross-domain transfer is the only positive result

A curator trained on mathematics (DeepMath-103K), evaluated on ALFWorld. The
checkpoint was selected as best-of-four on `valid_seen`, so `valid_unseen` is a
genuine held-out test for it.

| arm | valid_seen (140) | valid_unseen (134) |
|---|---|---|
| Gemini 2.5 Pro curator | -1.4pp, p=0.86 | +6.0pp, p=0.28 |
| reasoning curator ckpt50 | +4.3pp, p=0.31 | **+9.0pp, p=0.073** |

Pooled over both splits the trained curator gives +6.6pp, p=0.030, but we report
this as secondary: half the pool is the split the checkpoint was selected on, so
the pooled p-value is optimistic. The primary number is the held-out +9.0pp at
p=0.073, against an MDE of 12.9pp, which is suggestive and below our own
detection threshold.

The direction is notable independent of significance. Curators trained on
ALFWorld do not help an ALFWorld executor; a curator trained on mathematics
does. This is also the arm family that, measured during an API outage, produced
a large *negative* cross-domain effect that we retracted (Appendix B.1).

**Neighbouring checkpoints on the held-out split: [PENDING]** (ckpt45, 55, 60).
If they are also positive the effect is real; if only ckpt50 is, it is checkpoint
selection and we will say so.

**Two additional reasoning-curator training seeds: [PENDING].** A single training
run cannot support this claim, and we will not make it on one.

## 5.4 A frontier curator is not better than no curator

![Held-out split, 134 games, contemporaneous control. Bars are 95% paired bootstrap intervals; grey rules mark each arm's minimum detectable effect.](figures/fig4_heldout_comparison.pdf){width=90%}


Gemini 2.5 Pro, prompted with the paper's curator prompt and native function
calling, temperature 0, writing skills for the same frozen 8B executor:

| | valid_seen | valid_unseen | measured cost per curator call |
|---|---|---|---|
| Gemini 2.5 Pro | -1.4pp, p=0.86 | +6.0pp, p=0.28 | $0.0168 |
| trained 8B curator | +4.3pp, p=0.31 | +9.0pp, p=0.073 | $0.0002 |

The trained curator is ahead on both splits and pooled (+4.4pp, p=0.17), at 1/84
the cost per call, which is the direction the original paper claims. We cannot
call it significant. What we can say without qualification is that **the frontier
model does not beat writing no notes at all**, on either split, and that spending
84 times more per call bought nothing measurable here.

## 5.5 Retrieved skills make this executor produce more invalid actions

The executor's action parser falls back to the first admissible command when the
model's output cannot be parsed. This applies identically to all arms so it
cannot bias a paired test, but we had never measured it. Instrumented, on the
same 134 held-out games:

| arm | actions coerced |
|---|---|
| no memory | 2.1% |
| trained curator | 7.0% |
| Gemini 2.5 Pro curator | 9.1% |

Adding retrieved skills to the prompt triples to quadruples the rate at which
this 8B executor emits something the environment cannot accept. This is a direct
mechanism for memory failing to help, or hurting, a small model: the notes
compete with the output format for the model's limited instruction-following
capacity. It is also a candidate contributor to our absolute baseline gap, though
not the whole of it, since the no-memory rate is only 2.1%.

## 5.6 Content controls: is it the skills, or just more text?

**[PENDING]** Two controls on the held-out split, both running at the time of
writing:

- **Shuffled retrieval.** The trained curator's own repository, with retrieval
  returning a random five skills instead of the BM25 top five. Same curator, same
  repository, same prompt length, relevance destroyed. If +9.0pp survives, the
  executor is helped by extra markdown rather than by relevant skills.
- **Author-written oracle skills.** Eight skills written from the published
  ALFWorld action grammar and task-type definitions, with no curator and no
  access to eval data. Written by the LLM agent conducting the study, not by a
  human: an easier condition than the curator's, since the curator must infer
  the same content from rollouts. Bounds what a curator could be worth here.

We consider the shuffled control decisive for how much the Section 5.3 result is
worth, and we commit to reporting it either way.

## 5.7 Six candidate causes, each with a training run or full sweep behind it

Before the control problem was found, we ran these to explain the oscillating
sweep. They remain valid as ablations of the method; they are now also evidence
that the null in 5.2 is not caused by any of them.

| candidate | test | outcome |
|---|---|---|
| LoRA parameterisation | full fine-tune, 3 seeds | same behaviour, not a LoRA artifact |
| RL framework | verl-agent/GiGPO port, full sweep | same behaviour **[PENDING re-pairing]** |
| Task-type distribution | natural frequencies vs uniform | natural is worse, best +5.7pp p=0.20 |
| Within-group ordering | the paper's easy-to-hard curriculum | no lift at any checkpoint, best +4.3pp p=0.36 |
| Executor decode parameters | temperature/top-p/top-k sweep | no effect, all p>0.5 |
| Executor prompt and retrieval | verbatim published prompt, BM25 top-5 | no effect |

The third and fourth are both halves of the original paper's own grouping
ablation, so grouping is exonerated as the driver of anything we observed.

## 5.8 Cross-executor scale

**[PENDING re-pairing.]** Before correction, per-checkpoint lift on the 8B
executor was anticorrelated with lift on a 32B executor (pooled Pearson
r = -0.20 over 24 pairs, r = -0.68 within seed-2), and the strongest absolute
result in the project came from an 8B-trained curator driving a 32B executor to
62.9%. Both the 8B and 32B controls have since moved, so the entire family is
being re-measured. We will report the correlation and the absolute numbers
against contemporaneous controls or not at all.

## 5.9 Reward machinery

Decomposing 850 logged rollouts from the verl run: the composite reward's *level*
is dominated by the function-call-validity term (69%), but GRPO centres
advantages within groups, so only within-group variance reaches the gradient, and
there the task term supplies 79%. All 80 logged groups had non-zero task-reward
variance, so the group-collapse failure mode that invalidated our own earlier
runs is absent.

Given a healthy task-dominated gradient, training task reward rose from 0.331 to
0.366 over 60 steps: +0.035, 95% CI ±0.034. Train-time success was flat (0.170 to
0.167). Policy entropy collapsed from 0.139 to 0.035 while gradient norm rose
from 1.40 to 2.40, with the change accelerating around step 48.

The optimiser was chasing the right signal and barely moved it. That, rather than
a broken reward, is the most likely proximate reason our held-out lifts are small
enough to be invisible at this sample size.
