# 5. Every correction came from a human

We reviewed the project's full history for instances where a conclusion changed.
There were five that mattered — changes that, if they had not been caught, would
have produced a qualitatively different paper:

1. **The stale baseline.** The human asked why the lift was so consistent across
   seeds. The agent had interpreted consistency as evidence of a real effect.
   The human suspected a shared reference.

2. **The eval-harness outage bug.** The human noticed the cross-domain transfer
   result had the wrong sign compared to expectations and asked the agent to
   check for infrastructure problems during those runs.

3. **The group-collapse bug.** The human asked why reward curves were flat, after
   the agent had explained the flatness as a property of the task.

4. **The completion-budget truncation.** The human asked the agent to add
   telemetry for measured positions per rollout — a quantity the agent had
   never thought to check.

5. **The baseline gap.** The human refused to accept "unexplained" and pushed
   through six falsification experiments that the agent considered unnecessary.

In every case the agent's analysis was internally consistent and arithmetically
correct. In every case it was wrong about what mattered. The agent had built a
coherent story around the data it had; the human questioned whether the data was
the data it thought it was.

**None of the five corrections originated from the agent's own review.** The
agent reviewed its results regularly, checked its code, re-ran analyses, and
cross-referenced numbers. It caught typos, fixed formatting, and corrected minor
numerical errors. It never caught a structural problem. Not once.

We do not think this is a property of the specific agent we used. We think it is
a property of the task. Checking whether an analysis is internally consistent is
something LLMs are good at. Checking whether the premises of an analysis are
true requires stepping outside the analysis, and an agent operating inside its
own pipeline has no vantage point from which to do that.
