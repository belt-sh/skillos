# 4. The pattern: agent errors are biases, not noise

A human researcher running a hundred evaluation arms by hand makes scattered
mistakes. They misconfigure one run, misread one table, forget to record one
result. The errors are roughly random and they mostly cancel: some inflate the
effect, some deflate it, and the aggregate is noisy but centered.

An agent running a hundred evaluation arms from one script makes the same
mistake in all hundred. If the baseline is wrong, every arm inherits the same
bias. If the harness scores an outage as a failure, every concurrent arm is
equally affected. The errors are perfectly correlated.

In a paired significance test, a consistent error is indistinguishable from a
treatment effect. Worse: a *severe* consistent error looks like a *strong*
treatment effect, because it is reliable, and reliability is what a significance
test rewards.

Our most significant finding was p=0.0002. It was an eight-hour credential
outage.

Our most replicated finding — consistent across three seeds and two RL
frameworks, which we spent four additional training runs trying to explain — was
a 5.7-point baseline drift that every arm inherited equally.

Both looked like good science. The statistics were correct. The replication was
genuine. The write-ups were clear. The only thing wrong was the premise, and the
agent had no mechanism to question a premise.

This is the structural risk of autonomous research that we think generalizes
beyond our case study: **the agent will not make random errors that average out;
it will make systematic errors that accumulate, and those errors will produce
more significant results, not fewer.**
