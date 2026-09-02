# Abstract

We gave an LLM agent a GPU cluster, a paper to reproduce, and the instruction
"replicate this." It ran for three months, launched ten training runs across two
RL frameworks, evaluated roughly one hundred experimental arms, found and fixed
its own infrastructure bugs, and wrote up its results with confidence.

Nearly every conclusion it reached was wrong.

Not because the agent was incompetent. The code worked. The statistics were
correct. The write-ups were clear and well-argued. The problem was simpler and
worse: the agent reused a baseline it measured once, and that baseline drifted
5.7 percentage points over ten weeks. Every training run, every seed, every
framework comparison was subtracting the same wrong number. The result was a
consistent, replicable, statistically significant artifact that looked like
science and was not.

This paper is a postmortem. We report 185 audited failures from the three-month
run, classify them by type and cost, and identify a pattern that we believe
generalizes: an autonomous agent's errors are biases, not noise. A human running
a hundred experiments by hand makes scattered mistakes that mostly cancel. An
agent running a hundred experiments from one script makes the same mistake in all
hundred, and a consistent mistake in a paired significance test is
indistinguishable from a treatment effect.

Both corrections that saved the project — the stale baseline and an eval harness
that scored API outages as task failures — came from a human asking why a result
looked too good. None came from the agent's own review. We describe seven
concrete gates that would have caught the expensive failures automatically, and
we argue that the right frame for autonomous research is not "will the agent
make mistakes" but "will the mistakes be visible before they compound."
