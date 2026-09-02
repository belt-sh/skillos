# 7. Seven gates

Every expensive failure in Section 3 had a cheap prevention. These are not
general principles about AI safety. They are specific, implementable checks that
we wish we had enforced from day one. All are now in our released harness.

## Gate 1: Re-measure the control alongside every batch of arms

Cost: one extra evaluation per sweep (~2 hours). Catches: baseline drift of any
origin — model updates, harness changes, infrastructure shifts. The control
measured in May was a measurement of that endpoint on that day. Treating it as a
fixed constant for ten weeks is the single decision that cost us the most.

## Gate 2: Abort on upstream error rate

If more than 2% of episodes in an arm are lost to API errors, kill the arm and
re-run. Cost: five lines of code. Catches: the outage-as-data bug that produced
our most significant result.

## Gate 3: Assert measured positions match configured group size

After every rollout, check that the number of measured positions equals the
training configuration. Cost: one assertion. Catches: the silent truncation that
made six training runs measure 26% of what they were supposed to.

## Gate 4: Log within-group reward variance

If within-group variance of the task reward is zero, the GRPO advantage is dead.
Cost: one summary statistic per batch. Catches: the group-collapse bug that
voided three training runs.

## Gate 5: Timestamp every measurement and refuse cross-epoch comparisons

Tag every evaluation result with the date it was measured. Refuse to compute a
lift between arms measured more than 7 days apart without an explicit override.
Cost: one metadata field and one check. Catches: stale baselines systematically.

## Gate 6: Instrument the quantity you are optimizing

If the reward depends on measured positions, log how many you measured. If
retrieval is supposed to help, log the retrieval hit rate. If the executor is
supposed to follow a format, log the parse failure rate. The agent never
spontaneously measured these; every diagnostic that mattered was added after a
human asked.

## Gate 7: A skeptic pass that asks "what if the premise is wrong"

After every major result, run a second agent (or a second prompt) whose job is
to list the premises the result depends on and check each one. Not "is the
analysis correct" — the agent already does that. "Is the input data what you
think it is." We did not implement this, and we suspect it is the hardest gate
to build well, but it is the one that would have caught both of our top-two
failures.
