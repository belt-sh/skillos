# 6. Threats to validity

## 6.1 We may simply have implemented it wrong

This is the leading hypothesis for any failed reproduction and we do not
discount it.

The strongest evidence for it is the 8 point absolute gap in our no-memory
baseline (39.8% against the paper's 47.9%). A frozen Qwen3-8B on ALFWorld
`valid_seen` with no skills should be a setup with very few degrees of freedom,
and we are 8 points below. We eliminated six candidate causes (Section 5.7) and
found none. Something about our executor loop differs from theirs.

The gap is worth stating carefully, because it shrank as a side effect of the
corrections in Appendix B: against the retired May control it was 14 points, and
against a contemporaneous control it is 8. Half of what we spent months treating
as an unexplained implementation gap was our own stale measurement.

If our executor is systematically weaker, it is possible that it sits below the
competence floor at which skill retrieval starts to pay. A skill that says "put
the mug in the coffee machine before turning it on" only helps an agent that can
already reliably execute "put mug in coffeemachine". Section 5.5 gives direct
evidence for this mechanism: curation *increases* our executor's rate of
unparseable actions from 3.0% to 8.5%. That is consistent with an executor that
is being pushed past its instruction-following capacity by longer context, and
it would be a property of our executor rather than of the method.

We consider this the most likely single explanation for the disagreement.

## 6.2 Framework differences

Neither TRL nor verl-agent is the original stack. They differ from it and from
each other in advantage normalisation, sequence packing, reference model
sharding, and optimiser state handling. Running both bounds this somewhat: two
stacks that disagree with each other on many details agree on the shape of the
result. It does not eliminate the possibility that both differ from the original
in the same consequential way.

The verl run additionally used an effective batch of 64 rather than 32, so that
comparison varies two factors. We report it as "the behaviour survives both a
framework change and a 2x batch change" rather than as a clean single-variable
test.

## 6.3 The study was run by an LLM agent

Section 4.5 discloses this. Here we state the specific risk.

An autonomous experimentalist is a systematic-error generator with an unusual
property: its errors are consistent. A human running a hundred evaluation arms
by hand makes scattered mistakes that mostly add noise. An agent running a
hundred arms from one script makes the same mistake in all hundred, which adds
*bias*, and bias in a paired test is indistinguishable from a treatment effect.

Both of this project's data integrity incidents had that shape:

- The evaluation harness answered an upstream API failure by playing the first
  legal action and recording the episode as a task failure. During one outage
  this affected 52 to 65% of the steps in four arms. Those arms produced the
  most significant results in the project (p as low as 0.0002) and pointed in a
  clean, interpretable direction. They were measuring the outage.
- A control measured once was reused for ten weeks against arms measured later.
  This inflated every lift in the project by roughly 6 points.

Neither was caught by the agent's own review. Both were caught after the human
author asked a sceptical question about a result that looked too clean. We
regard this as the central practical finding about the mode of work, and we
recommend that any agent-conducted evaluation study include (a) a per-arm data
integrity gate that aborts on upstream error rates above a threshold, and (b) a
control re-measured alongside every batch of arms. Both are now enforced in our
harness and both are in the released code.

## 6.4 Multiplicity

We ran roughly one hundred evaluation arms. Under a Bonferroni correction across
the checkpoint sweeps alone, the bar is p < 0.001, and no same-executor ALFWorld
result in this project ever reached it, including in the pre-correction era. Our
negative results are not multiplicity-limited, but any reader tempted to extract
a positive from our tables should apply the correction.

## 6.5 Statistical power

140 paired games gives roughly a ±3 percentage point noise floor. An effect of
5pp is at the edge of what we can resolve, and several of our comparisons land
there. Where a comparison matters we add the 134 game `valid_unseen` split to
roughly double n. Effects smaller than about 5pp are outside this study's
resolution, and we do not claim to have excluded them.

## 6.6 Single benchmark family

ALFWorld carries the main result; the reasoning benchmarks are a secondary arm.
WebShop was not attempted. A method could fail on embodied tasks with a small
executor and succeed elsewhere. We make no claim beyond what we measured.

## 6.7 Hosted-model non-stationarity

The executor and judge were served over a hosted API for the entire study. We
have shown that this endpoint's behaviour on our benchmark moved by 5.7 points
over ten weeks. Within-epoch comparisons are protected by re-measured controls,
but any comparison we make across epochs, including the comparison between our
numbers and the original paper's, inherits this uncertainty.
