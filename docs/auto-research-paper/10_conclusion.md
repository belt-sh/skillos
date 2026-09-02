# 9. The instruction we'd give next time

"Replicate this and don't stop" produced three months of confident, fast,
systematically wrong work. The agent did exactly what we asked. The instruction
was the problem.

If we were starting again:

**"Replicate this. Before you report any result, check that the control was
measured this week, that less than 2% of episodes errored, that every
configured position was measured, and that within-group reward variance is
nonzero. If any check fails, fix it before you run another arm. After every
positive result, tell me the three most likely ways it could be an artifact,
and test the cheapest one."**

That is a paragraph of instructions. It encodes the specific things we learned
the hard way. A different project would need a different paragraph, because the
failure modes depend on the pipeline — but the principle is the same: the agent
needs to be told what to doubt, not just what to do.

The deeper lesson is less actionable. An autonomous agent operating inside its
own pipeline cannot question the pipeline's premises from outside. It can check
internal consistency — and it does, well. It cannot check whether the world
matches its assumptions, because its assumptions are the lens it uses to look at
the world. The human's role is not to oversee the execution. The execution is
fine. The human's role is to be the person who was not there when the baseline
was measured and who asks, with fresh eyes, "are you sure that number is still
right?"

That question saved this project twice. The agent never asked it.
