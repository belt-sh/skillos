# Appendix B. Data integrity incidents

Two defects in our own apparatus produced results we published internally and
later retracted. Both are reported in full because the failure mode generalises
and because the second one is the paper's principal finding.

## B.1 An evaluation harness that scored API failures as task failures

**The defect.** The streaming-curation evaluation harness called the executor
model over a hosted API once per environment step. When that call raised, the
harness caught the exception, substituted the first entry of the environment's
admissible-command list, and continued the episode. If the episode then failed,
it was recorded as an ordinary task failure with `success: false`.

The training-side code was correct: it recorded deadline-cut positions as
`success: None, cut: True` and masked them out of the reward. Only the evaluation
path conflated an infrastructure failure with a task failure.

**The blast radius.** On 2026-07-20 an expired credential caused the executor
endpoint to return `HTTP 401` for roughly eight hours. Four evaluation arms ran
during the window. The share of environment steps in which the action was
substituted rather than generated:

| arm | substituted actions |
|---|---|
| reasoning-curator ckpt45 | 64.9% |
| reasoning-curator ckpt50 | 59.4% |
| reasoning-curator ckpt55 | 52.1% |
| reasoning-curator ckpt60 | 55.8% |

These four arms produced the most statistically significant result in the entire
project: a cross-domain transfer effect of -14 to -18pp with p values from
0.0002 to below 0.0001, comfortably surviving Bonferroni correction over twelve
arms. It was the only finding we had that was robust to multiplicity, and it
pointed in the opposite direction from the original paper, which made it feel
like a discovery.

It was measuring an outage.

**Why it was so significant.** This is the part worth internalising. A
systematically crippled arm is *reliably* crippled. It fails the same games every
time, in the same direction, with low variance. Low variance in a consistent
direction is exactly what a paired significance test is built to detect. A
severe upstream defect does not produce noisy results that fail to reach
significance; it produces clean results that sail past it.

**Detection.** Not by us. The harness printed the substitution to stderr, and the
human author read a passing reference to a "fallback action" in an unrelated log
excerpt and objected to the design on principle, before knowing it had fired.

**Fix.** The harness now abandons the episode, records `success: None,
errored: True`, excludes it from the denominator, counts it, prints a
`data integrity` line for every arm, and exits with a distinct status code if the
error rate exceeds a configurable threshold (default 2%). Re-running the four
arms under the fixed harness moved them from 15.7 / 18.6 / 16.4 / 19.3% to
38.6 / 43.6 / 36.4 / 33.6%, all within noise of the contemporaneous 39.3%
baseline. The cliff was the outage. One earlier arm cannot be recovered because
its adapters were deleted.

**Related measurement.** Auditing the training side established that 61 to 79% of
executor probe positions were deadline-cut, leaving a median of one surviving
measurement out of nine behind each `r_task` value, and 10 to 41% of rollouts
received `r_task = 0` by construction. This does not invalidate the training runs
but it does mean the reward signal was far thinner than the design intended, and
we record it as a divergence rather than a defect.

## B.2 A control reused across ten weeks

**The defect.** Our no-memory ALFWorld baseline was measured once, in May 2026,
at 47/140 = 33.6%. It was then treated as canonical and every subsequent arm was
paired against it. An internal note went further and instructed future work to
*always* pair against the fixed file and never against a freshly measured
baseline, on the theory that the baseline had roughly 8pp of run-to-run variance
and the fixed file was therefore the more stable reference.

That reasoning has the sign backwards. The observed 8pp was never sampling
variance around a stable mean. It was the distance between one stale measurement
and the current behaviour of a hosted endpoint.

**The measurement.** Three fresh replicates plus the re-run reference, same 140
games, same week, fixed harness:

| run | SR |
|---|---|
| May 2026 canonical | 33.6% |
| August replicate 1 | 39.3% |
| August replicate 2 | 39.3% |
| August replicate 3 | 39.3% |
| August replicate 4 | 41.4% |

Same-week mean 39.8%, spread 2.1pp. Genuine run-to-run variance is about 2pp.
The May figure lies outside that range, so it reflects a real change in the
measurement conditions rather than a lucky draw. We cannot separate hosted-model
drift from harness-version effects, because both changed between May and August.

**The blast radius.** Every lift reported internally was `arm - 33.6%`, so every
one was inflated by roughly 6 points. Re-paired against a contemporaneous
control, our best seed drops from +13.6pp at p=0.0026 to +3.6pp at p=0.47, and
eight of its twelve checkpoints turn negative. The "significant peak somewhere in
every run" pattern that we had reproduced across three seeds and two RL
frameworks, and had spent weeks trying to explain, was an artifact of the shared
reference.

**Detection.** Also not by us. After the harness fix the re-run numbers came back
lower than the originals, and the human author asked whether the results had
genuinely got worse. Answering that required a fresh baseline, and the fresh
baseline was 6 points higher than the one in use.

**Fix.** Controls are now re-measured alongside every batch of arms, and the
comparison scripts refuse to pair arms from different measurement epochs. The
instruction to pin to a fixed canonical baseline has been deleted and replaced
with its opposite.

## B.3 A third defect that biases nothing but depresses everything

The executor's action parser returns the first admissible command when it cannot
parse the model's output. Unlike B.1 this is a deliberate design choice rather
than a bug, and because it applies identically to every arm including the
baselines it cannot bias a paired comparison. We report it because we had never
measured it, and once instrumented it turned out to be substantial and
arm-dependent: 2.1% of actions with no memory, 7.0% with a trained curator, and
9.1% with a Gemini 2.5 Pro curator on the same games.

That ordering is itself a finding, discussed in Section 5.5. Longer prompts
containing retrieved skills make this executor *less* able to emit a parseable
action, which is a concrete mechanism by which memory can fail to help a small
model. It is also a candidate contributor to our 8pp absolute baseline gap,
though it cannot be the whole story, since the no-memory rate is only 2.1%.
