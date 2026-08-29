# 7. Discussion

## 7.1 What we think is actually true

The method is implementable and the machinery works. A curator trained with GRPO
against a composite reward produces coherent skill repositories, the reward
gradient is dominated by downstream task success as intended, and the training
runs converge without pathology.

One result reproduces cleanly. Cross-executor transfer, the paper's strongest
practical claim, holds at parity: an 8B-trained curator lifts a 32B executor to
62.1% against the paper's 61.2%.

What we could not find is a stable curator lift of any kind on the 8B training
executor. Over eight training runs and roughly one hundred evaluation arms, no
ALFWorld-trained curator produced a significant improvement, and the cross-domain
result that appeared to survive Holm correction on one seed (+11.2pp) does not
replicate across three seeds. The effect is not absent; it is smaller than this
protocol can resolve.

Three readings are consistent with our data, and we cannot distinguish them.

**The effect is real and smaller than reported.** If the true lift is 3 to 5pp,
every observation in this paper is expected: individual arms scatter around it,
nothing reaches significance at n=140, and the original's +13.3pp would be a
favourable draw from a sweep. This is our leading hypothesis, and it is the one
most consistent with our reward analysis, where the optimiser moves training
reward by +0.035 over 60 steps.

**The effect requires a stronger executor than ours.** Our absolute baseline is
8pp below the original's after eliminating six causes, and Section 5.5 shows our
executor gets *worse* at emitting valid actions when given skills. An executor at
the edge of its instruction-following capacity may be unable to convert good notes
into good actions. The 32B transfer family, once re-paired, speaks to this.

**We implemented something subtly different.** Always possible, and the
unexplained baseline gap is evidence for it. We have released everything needed
to find such a difference.

## 7.2 Cross-domain transfer

The seed-1 result that appeared strongest — a mathematics-trained curator lifting
held-out ALFWorld by +11.2pp — does not replicate. Across three seeds and twelve
checkpoint arms on `valid_unseen`, no arm reaches p<0.05 against a
contemporaneous control (Section 5.3). The direction is positive more often than
not (8 of 12 arms), and the best single arm is +6.7pp (p=0.12), so we do not
claim the effect is absent. But it is not the clean positive we reported in
earlier drafts.

The hypothesis from Section 5.5 still fits: a reasoning-trained curator writes
shorter, more general notes that cost the executor less attention than
ALFWorld-specific procedures do. But with no significant result to explain, we
leave it as speculation.

## 7.3 The measurement result is the transferable one

The most useful output of this project is not about skill repositories.

A control measured against a hosted model API is a measurement of that endpoint
on that day. Ours moved 5.7pp in ten weeks, which is larger than the effects most
of the papers in Section 2.1 claim. Reusing it converted endpoint drift into
apparent treatment effect across seven training runs, three seeds and two
frameworks, and produced a consistent, replicable, entirely spurious pattern that
we spent four additional training runs trying to explain.

The pattern was replicable because the error was shared. This is worth stating
plainly: **agreement across seeds and frameworks does not protect you from a
shared reference.** Every one of those runs was independently trained and
independently evaluated, and they all agreed, because they were all subtracting
the same wrong number.

Combined with Section 5.10, where the standard 140-game protocol turns out to have
80% power to detect exactly the effect size the field reports and no less, the
practical recommendations are unglamorous and cheap:

1. Measure the control in the same batch as the arms. Always.
2. Report a confidence interval and a minimum detectable effect next to every
   claimed improvement. One line each.
3. Report the whole sweep, not the best checkpoint, and correct within the family.
4. Instrument the harness for upstream failures and abort rather than score them.

Any one of these would have saved this project weeks.

## 7.4 On having an agent run the study

We disclose in Section 4.5 and Appendix A that this work was executed by an LLM
agent under human direction. Two observations seem worth generalising.

**The agent was strong at execution and weak at doubt.** It implemented, launched,
supervised, recovered, and analysed competently across three months and seven
training runs. It did not catch either of the errors that mattered. In both cases
its analysis was internally consistent, arithmetically correct, and wrong in the
same direction as its pipeline. It produced confident, well-argued write-ups of
artifacts.

**Agent errors are biases, not noise.** A human running a hundred evaluation arms
by hand makes scattered mistakes that mostly cancel. An agent running a hundred
arms from one script makes the same mistake in all hundred. In a paired
significance test, a consistent error is indistinguishable from a treatment
effect, and a *severe* consistent error looks like a strong treatment effect. Our
most significant finding, at p=0.0002, was an eight-hour credential outage.

The practical implication is not that agents should not run experiments. It is
that the review has to target the apparatus rather than the analysis, because the
analysis will be clean. In this project every correction that mattered came from a
human asking why a result looked too good, and none came from the agent's own
review.

## 7.5 Limitations we would fix with more time

WebShop was never attempted. Our absolute baseline gap is unexplained. And the deepest
limitation is one we can now quantify rather than fix: ALFWorld provides 274
valid games in total, which caps a paired comparison at roughly 9pp resolution.
Serious measurement of a 5pp agent-memory effect needs either a larger benchmark
or many rollouts per game, and the field has been reporting effects near or below
that threshold for several years.
