# 2. What the agent was good at

Before the failures, the successes — because they are real and they matter for
calibrating what to delegate.

**Implementation.** The agent read the paper, wrote a working GRPO training loop
with composite rewards, plugged in ALFWorld, wired up a Qwen3-8B curator against
a frozen executor, and produced a training run that converged on the first
attempt. It ported the same setup to a second RL framework (verl/GiGPO) without
being asked, to control for framework effects. The code was clean, tested where
it mattered, and ran on the first try more often than not.

**Infrastructure recovery.** Over three months the agent handled NCCL timeouts,
DNS resolution loops, API rate storms (OpenRouter 429s), OOM crashes, and a
credential rotation — all without human intervention. It wrote supervisor scripts
that resumed training from the last checkpoint, built storm-resilient evaluation
sweeps that backed off and retried, and instrumented its own logs for the failure
modes it had already seen.

**Throughput.** Ten training runs, a hundred eval arms, three benchmarks, six
falsification experiments, and a 36-page paper draft — in three months, with one
part-time human. A human researcher could not have run this volume of experiments
in the same time. The agent's pace was its genuine advantage, and also the source
of its most expensive failure: it ran fast enough to compound mistakes before
anyone checked.

**Analysis.** Given clean data, the agent's statistical analysis was correct. It
computed McNemar tests, bootstrap intervals, Holm corrections, and minimum
detectable effects accurately. Its write-ups were clear, its figures were
reproducible, and its tables matched the underlying JSONLs.

The pattern: the agent was excellent at *executing* and *analyzing*. It was
terrible at *doubting*.
