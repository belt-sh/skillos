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
resolve. Section 5.10 develops this.

**The paper-faithful rerun confirms the null.** A final TRL run with the
completion-budget fix (Section 4.4.1), measuring all 9 informed positions per
rollout instead of the 2.3 that trained every previous TRL run, completed the
full 60-step schedule with healthy reward telemetry (median 9/9 positions, action
coercion 0.09%). Its 12-checkpoint sweep against a contemporaneous baseline
(40.0%):

| checkpoint | success | delta | p (McNemar) |
|---|---|---|---|
| ckpt5 | 34.3% | -5.7pp | 0.18 |
| ckpt10 | 38.6% | -1.4pp | 0.85 |
| ckpt15 | 30.7% | -9.3pp | 0.007 |
| ckpt20 | 34.3% | -5.7pp | 0.17 |
| ckpt25 | 38.6% | -1.4pp | 0.86 |
| ckpt30 | 34.3% | -5.7pp | 0.18 |
| ckpt35 | 38.6% | -1.4pp | 0.84 |
| ckpt40 | 37.1% | -2.9pp | 0.57 |
| ckpt45 | 26.4% | -13.6pp | 0.0003 |
| ckpt50 | 40.0% | +0.0pp | 1.00 |
| ckpt55 | 31.4% | -8.6pp | 0.023 |
| ckpt60 | 40.0% | +0.0pp | 1.00 |

No arm exceeds the control. Two are significantly *worse* (ckpt15 and ckpt45),
and the best arms (ckpt50, ckpt60) match the baseline exactly. This is the
definitive same-agent result: fixing the reward-measurement fidelity gap that
compromised earlier TRL runs did not produce a lift; it confirmed the null.

## 5.3 Cross-domain transfer does not replicate across seeds

A curator trained on mathematics (DeepMath-103K), evaluated on ALFWorld
`valid_unseen` (134 games), against a contemporaneous 39.6% baseline.

**Seed-1** (the original run) had previously produced the project's only
correction-surviving result: ckpt60 at +11.2pp (p=0.003) on an earlier
measurement. Against a contemporaneous control, seed-1 is flat:

| seed-1 checkpoint | success | delta | p |
|---|---|---|---|
| ckpt45 | 38.8% | -0.7pp | 1.00 |
| ckpt50 | 38.8% | -0.7pp | 1.00 |
| ckpt55 | 38.8% | -0.7pp | 1.00 |
| ckpt60 | 40.3% | +0.7pp | 1.00 |

**Seeds 2 and 3** are directionally positive but not significant:

| arm | success | delta | p |
|---|---|---|---|
| s2 ckpt45 | 43.3% | +3.7pp | 0.49 |
| s2 ckpt50 | 44.8% | +5.2pp | 0.32 |
| s2 ckpt55 | 35.8% | -3.7pp | 0.44 |
| s2 ckpt60 | 40.3% | +0.7pp | 1.00 |
| s3 ckpt45 | 42.5% | +3.0pp | 0.54 |
| **s3 ckpt50** | **46.3%** | **+6.7pp** | **0.12** |
| s3 ckpt55 | 39.6% | +0.0pp | 1.00 |
| s3 ckpt60 | 41.0% | +1.5pp | 0.87 |

No arm in any seed reaches p<0.05. The best across all three seeds is s3 ckpt50
at +6.7pp (p=0.12), against an MDE of ~13pp on 134 games. Eight of twelve arms
are positive, which is more than chance, but the magnitudes are small and
unstable across checkpoints within each seed.

The earlier ckpt60 +11.2pp result was a single-run, single-checkpoint observation
that does not survive contemporaneous re-measurement or replication. We retract
it as a finding. What remains is a directional tendency — reasoning-trained
curators produce small positive effects more often than ALFWorld-trained curators
do — that this protocol cannot resolve.

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

Two controls on the held-out split, against a contemporaneous no-memory control
at 35.1% on the same 134 games. Both were run before their outcome was known and
are reported as promised.

- **Shuffled retrieval.** The trained curator's own repository, with retrieval
  returning a random five skills instead of the BM25 top five. Same curator, same
  repository, same prompt length, relevance destroyed.
- **Oracle skills.** Eight skills written from the published ALFWorld action
  grammar and task-type definitions, with no curator and no access to eval data.
  Written by the LLM agent conducting the study, not by a human, so this is *not*
  a human baseline: it is an easier condition than the curator's, since the
  curator must infer the same content from rollouts. It bounds what a curator
  could be worth here.

| arm | success | delta | p (exact McNemar) | Holm | 95% CI | MDE80 |
|---|---|---|---|---|---|---|
| no memory (control) | 35.1% | | | | | |
| oracle skills, no curator | 53.0% | +17.9pp | 0.0001 | **0.0005** | [+9.7, +26.1] | 12.5pp |
| reasoning curator ckpt60 | 46.3% | +11.2pp | 0.0026 | **0.0156** | [+4.5, +17.9] | 10.0pp |
| reasoning curator ckpt50 | 44.0% | +9.0pp | 0.073 | 0.36 | [+0.0, +17.9] | 12.9pp |
| Gemini 2.5 Pro curator | 41.0% | +6.0pp | 0.28 | 1.00 | [-3.0, +14.9] | 13.5pp |
| ckpt50, retrieval shuffled | 37.3% | +2.2pp | 0.70 | 1.00 | [-5.2, +9.7] | 10.9pp |
| reasoning curator ckpt55 | 35.1% | +0.0pp | 1.00 | 1.00 | [-8.2, +7.5] | 11.5pp |

Holm is applied across the seven-arm family. Two arms survive, and both exceed
their own MDE.

**The shuffled control answers the question it was built for.** Destroying
relevance while holding the curator, the repository and the prompt length fixed
collapses +9.0pp to +2.2pp. Where a lift exists it is carried by relevant
content, not by the presence of additional markdown.

**The oracle arm relocates the bottleneck.** Eight skills derived from public
documentation gain 30 games and lose 6, for +17.9pp, the largest and cleanest
effect anywhere in this reproduction. This executor can therefore exploit good
notes. What the training procedure fails to do is produce notes as good as a
careful reading of the action grammar. That is a considerably more specific
negative result than "memory does not help this model", and it suggests the
productive target for future work is curator supervision rather than executor
scale.

Two caveats we cannot resolve. The oracle arm carries a **warm-start advantage**:
its eight skills are present from game 1, while every curator arm begins with an
empty repository and must write its way up. Part of +17.9pp is that head start
rather than content quality, and the clean way to separate them is to replay a
curator's final repository from game 1, which we have not run. Second, the
neighbouring checkpoints of the surviving curator arm are +1.5pp (ckpt45) and
+0.0pp (ckpt55) against ckpt60's +11.2pp, so **checkpoint selection carries real
risk of a lottery** even where an arm survives correction.

## 5.7 Six candidate causes, each with a training run or full sweep behind it

Before the control problem was found, we ran these to explain the oscillating
sweep. They remain valid as ablations of the method; they are now also evidence
that the null in 5.2 is not caused by any of them.

| candidate | test | outcome |
|---|---|---|
| LoRA parameterisation | full fine-tune, 3 seeds | same behaviour, not a LoRA artifact |
| RL framework | verl-agent/GiGPO port, full sweep | same behaviour (not re-paired against contemporaneous control) |
| Task-type distribution | natural frequencies vs uniform | natural is worse, best +5.7pp p=0.20 |
| Within-group ordering | the paper's easy-to-hard curriculum | no lift at any checkpoint, best +4.3pp p=0.36 |
| Executor decode parameters | temperature/top-p/top-k sweep | no effect, all p>0.5 |
| Executor prompt and retrieval | verbatim published prompt, BM25 top-5 | no effect |

The third and fourth are both halves of the original paper's own grouping
ablation, so grouping is exonerated as the driver of anything we observed.

## 5.8 Cross-executor scale

Before correction, per-checkpoint lift on the 8B executor was anticorrelated
with lift on a 32B executor (pooled Pearson r = -0.20 over 24 pairs, r = -0.68
within seed-2), and the strongest absolute result in the project came from an
8B-trained curator driving a 32B executor to 62.1% (+12.9pp, p=0.006). Both the
8B and 32B controls have since moved. The 32B transfer number was measured
against a contemporaneous 32B control and survives (Section 1.2); the 8B
correlation has not been re-paired and we do not report it as a finding.

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

The paper-faithful TRL rerun (`dense10`) confirms this on a properly measured
reward signal. With the completion-budget fix (Section 4.4.1) measuring 9 of 9
informed positions per rollout instead of 2.3, training converged over 60 steps
with action coercion at 0.09% and zero early exits. The reward path was healthy
throughout: the training completed in 61.5 wall-clock hours.

The optimiser was chasing the right signal and barely moved it. That, rather than
a broken reward, is the most likely proximate reason our held-out lifts are small
enough to be invisible at this sample size.
