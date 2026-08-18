# 6.x Conduct of an agent-run reproduction: what failed, and what it found

This study was carried out almost entirely by an LLM agent working continuously
for roughly three months, with a human author setting scope, funding compute, and
adjudicating disputes. That is unusual enough to be worth reporting as a result in
its own right. We report it in two halves, because both halves are true and only
one of them is usually written down: the ways the mode of work failed, and the
findings it produced that we do not believe a conventionally-run study would have
produced.

We are deliberately concrete. A general claim that "agents make mistakes" is not
useful. A count, a taxonomy, and a cost are.

## 6.x.1 How the failures were counted

The study ran as one continuous session: a 79 MB transcript, 18,409
conversational turns. At the end we compressed it to prose and had eight
independent reviewer agents each audit one contiguous slice, cross-checking every
failure they found against the four records the project had been keeping (a
running journal, a divergences ledger, a results report, and one postmortem).
Each failure was labelled DOCUMENTED, PARTIAL, or MISSING.

The audit was not self-assessment. The reviewers had the transcript, not the
memory of it, and no access to each other's output. Two slices overlapped in
time, which gave an unplanned control: four incidents were found independently by
two reviewers, agreeing on the failure and disagreeing only on magnitude (one
put a lost-training incident at 15 hours, the other at 22). Of the reviewers'
claims about what was absent from the records, 29 were re-verified by direct
search, and one was wrong: a licence-compliance guard the reviewer flagged as
inadequate had already been fixed.

The full ledger is released with the artifact. The counts:

| | entries | share |
|---|---|---|
| Failures found | 185 | |
| Documented in the project's own records | 36 | 19% |
| Recorded, but without cause, cost, or trigger (PARTIAL) | 55 | 30% |
| **Absent from every record (MISSING)** | **93** | **50%** |

This is a lower bound. The digest truncates long tool outputs, so failures
visible only inside a large log are under-counted.

## 6.x.2 The shortcomings

Fourteen recurring modes emerged. We report the six with the largest
consequences for the science, since the operational ones (self-inflicted process
kills, orphaned monitors, roughly a week of idle GPU time across seven incidents)
cost money rather than validity.

**A sentinel that does not say why.** At least eight code paths substituted a
plausible value for a measurement that never happened: an unparseable action
became the first admissible action, an episode that never ran became
`success: False`, a rollout with no measured position became `r_task = 0.0`, an
API failure became a wrong answer. None carried a flag distinguishing "measured
and failed" from "never measured".

This is the single most important methodological point in this paper, because of
its statistical consequence. **A systematically crippled arm is reliably
crippled.** Low variance in a consistent direction is exactly what a paired
significance test rewards. Our most significant result, at p below 0.0005, was an
eight-hour authentication outage measured against a healthy control. It was not a
weak effect that squeaked past a threshold; it was a strong, clean, reproducible
measurement of a broken pipe. Agent-run work is unusually exposed to this because
the agent writes both the harness and the analysis, and a defensive default that
keeps a long job alive is locally the right call at 3 a.m. and globally fatal to
the result.

**Health metrics validated against each other, never against ground truth.**
Three consecutive multi-day runs, about eight GPU-days, trained on ten fixed
tasks because an environment identifier was always zero. Reward, KL, gradient
norm, and a purpose-built degeneracy tripwire all looked healthy throughout. The
tripwire asserted that reward must vary within a GRPO group, and a second bug,
per-rank seed divergence, satisfied it. A check that asserts the absence of a
symptom can be satisfied by the next defect. What worked, every time, was
printing the quantity that should be invariant: measured positions per rollout,
distinct tasks drawn, coercion rate.

**A dependency's comment taken as its behaviour.** The training framework applies
its completion-length cap to the *accumulated* multi-turn completion, not to each
response. At the paper's 4096 tokens, a ten-position rollout carrying ten
trajectories fits about three positions, after which the framework silently drops
the tool result. Every training run in this project, for eleven weeks, trained on
roughly three positions of a ten-position protocol. Nothing errored; a truncated
rollout is byte-identical in shape to a finished one. The value looked
intentional because a comment beside it named the paper's hyperparameter. The same
mode produced a false belief about rollout concurrency that cost a factor of four
in wall-clock on every run.

**Fixes sized to the symptom rather than derived from arithmetic.** An
out-of-memory failure requesting 4.64 GiB was answered by offloading optimizer
state, freeing about 8 GiB. The next attempt requested 14.21 GiB, which is the
logits tensor, `per_device x sequence x vocabulary`, computable before the first
launch. Roughly four GPU-hours to learn that the fix had been aimed at the wrong
tensor. The project derived the right rule three times and applied it late each
time: **when removing a supposed cause does not move the number, the cause was
wrong.**

**Comparing across measurement epochs.** A single no-memory control, measured
once in May against a hosted endpoint, was reused for ten weeks as the reference
for every treatment. Re-measured in the same session as the treatments, the
control was 39.8% plus or minus 2.1pp, against the 33.6% on file. Every lift in
the project was partly a measurement of endpoint drift. Agreement across three
seeds and two RL frameworks provided no protection, because all of them
subtracted the same wrong number. **Replication is not protection against a shared
reference error.** Worse, the agent had written the reuse into its own notes as a
rule, so the error was self-reinforcing across context boundaries.

**The human was the error-detection mechanism.** A judge configured with an
eight-token limit on a yes/no prompt trained sixteen steps against a success rate
of identically zero; the human found it by asking whether that limit was
deliberate, while the agent was describing the run as encouraging. The same
pattern holds for the retry storm, an undeclared protocol deviation, a claim that
the paper trains 3500 steps when its table says 60, and nearly every idle-GPU
escalation. Corrections were made quickly and honestly once raised. They were
almost never *self*-raised. Self-review reliably confirmed priors.

**The asymmetry in the record is itself a finding.** Scientific failures were
well documented; operational and publishing failures were not. And where a
failure was recorded, the record tended to keep the mechanism and drop the
epistemics: the false-zero bug is documented, the written "this introduces no
skew" guarantee that shipped with it is not; the doubled batch size is
documented, that it was caught on day three, flagged three times, and knowingly
left running for eight more days is not. **The surviving record reads as a
sequence of discoveries. The transcript reads as discoveries preceded by
confident wrong answers.** Not by any intent to conceal: every failure here was
volunteered on request. It is what happens when the record is written by the same
process that made the mistakes, and when writing it is the last task rather than a
gate.

## 6.x.3 What the mode of work produced

Against that, the same properties that generated the failures, tirelessness,
willingness to build disposable instrumentation, and no ego investment in a
hypothesis, produced results we would not otherwise have.

**A power analysis that reframes the field's evaluation protocol.** ALFWorld's
standard 140-game split, at the discordance rate we observe and 80% power, has a
minimum detectable effect of about 13pp. That is the effect size this literature
reports. A claimed +13.3pp sits exactly at the resolution limit of the instrument
used to measure it, and 942 paired games would be needed to resolve 5pp against a
benchmark that contains 274 in total. This is arithmetic anyone could have done,
and as far as we can tell nobody in this line of work had. It came out of the
agent being asked to justify a null and choosing to compute the instrument's
resolution rather than argue.

**A ceiling control showing the executor is not the bottleneck.** On 134 held-out
games, against a contemporaneous control at 35.1%:

| arm | success | delta | p | Holm | 95% CI |
|---|---|---|---|---|---|
| oracle skills, no curator | 53.0% | +17.9pp | 0.0001 | **0.0005** | [+9.7, +26.1] |
| reasoning-trained curator, ckpt60 | 46.3% | +11.2pp | 0.0026 | **0.0156** | [+4.5, +17.9] |
| reasoning-trained curator, ckpt50 | 44.0% | +9.0pp | 0.073 | 0.36 | [+0.0, +17.9] |
| frontier-model curator | 41.0% | +6.0pp | 0.28 | 1.00 | [-3.0, +14.9] |
| same repo, retrieval shuffled | 37.3% | +2.2pp | 0.70 | 1.00 | [-5.2, +9.7] |
| reasoning-trained curator, ckpt55 | 35.1% | +0.0pp | 1.00 | 1.00 | [-8.2, +7.5] |

Two arms survive Holm correction across the family. The oracle arm, eight skills
written from the published action grammar with no access to evaluation data,
gains 30 games and loses 6. So this executor *can* exploit good notes: the
bottleneck is the quality of what the curator writes, not the executor's ability
to use it. That reframes the entire null from "memory does not help this model" to
"this training procedure does not produce notes as good as a careful reading of
the documentation".

**A control that separates content from context length.** The shuffled-retrieval
arm is the same curator, the same repository, and the same prompt length, with
relevance destroyed. It collapses to +2.2pp. So where a lift does appear, it is
carried by *relevant* content, not by the presence of extra markdown. This
control does not appear in the original work, and it is the one we would insist on
in any future skill-memory evaluation.

**A mechanism for why memory hurts small models.** Instrumenting the executor's
output parser showed that adding retrieved skills raises the rate of unparseable
actions from 2.1% with no memory to 7.0% with the trained curator and 9.1% with a
frontier curator. The notes compete with the output format for a small model's
limited instruction-following capacity. This is a concrete, measurable mechanism
for a result that is usually reported as an unexplained null, and it was found by
the agent instrumenting a code path it had itself written badly.

**A correct account of where the wall-clock goes in agentic RL.** Generation is
85.3% of a training step and the optimizer update is 1.4%. The workload is bound
by remote inference, not by gradients, which makes framework and sharding choices
nearly irrelevant to wall time and makes batch size and completion budget the only
real levers. This retired a plausible and widely-repeated claim, that one RL
framework was several times faster than another, by showing the apparent advantage
was entirely a doubled batch size.

**Decoupling generation concurrency from the training micro-batch**, which is what
finally let the paper's protocol run at full fidelity: 16k-token completions at
the paper's effective batch of 32 on eight GPUs, with all ten positions of every
rollout actually measured (median 9 of 9, against roughly 2 of 9 before). Six
candidate explanations for the oscillating training trajectory were also
falsified, each with a full training run or sweep behind it: parameterisation,
framework, task distribution, curriculum, decode settings, and prompt.

## 6.x.4 What we would require of the next one

Every item below is a counter-measure the project adopted only after paying for
its absence.

1. **No missing measurement may take a numeric value.** Carry an explicit
   `unmeasured` flag all the way to the gradient and neutralise those rollouts
   within their group. Sentinels must say why.
2. **Print the quantity that should be invariant**, per rollout, in the training
   log: positions measured, distinct tasks drawn, coercion rate. Tripwires that
   assert the absence of a symptom can be satisfied by the next bug.
3. **Measure a contemporaneous control in the same session as every treatment.**
   A control measured against a hosted endpoint is a measurement of that endpoint
   on that day.
4. **Report a minimum detectable effect beside every claimed improvement.** One
   line, and it would have prevented most of the wasted effort here.
5. **Diff the configuration against the paper's hyperparameter table before
   launching**, mechanically. Five separate fidelity defects in this project were
   inherited defaults that looked deliberate.
6. **Treat "the fix did not move the number" as evidence about the diagnosis**,
   not as a reason to add another fix.
7. **Write the failure record as a gate, not as a final task**, and have it
   audited by something that is not the process that made the mistakes. Half of
   what happened here was missing from a record kept diligently and in good faith
   throughout.
