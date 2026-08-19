# Abstract

SkillOS trains a curator model with GRPO to maintain a markdown skill repository
for a frozen executor, and reports a 13.3 percentage point improvement in
ALFWorld task success. We reproduced the method on 8xH100, across seven complete
60-step training runs in two independent RL frameworks (TRL with ZeRO-3 and
verl-agent/GiGPO with FSDP), three seeds, two executor scales, and approximately
one hundred evaluation arms of 140 paired games each.

The method works. Curators trained with GRPO produce coherent skill repositories,
the reward gradient is dominated by task success, and the training converges
without pathology. On the paper's strongest claim, cross-executor transfer, an
8B-trained curator lifts a 32B executor to 62.1% (+12.9pp, p=0.006), at parity
with the paper's 61.2%. A cross-domain curator trained on mathematics yields
+11.2pp on a held-out ALFWorld split (p=0.003, survives Holm correction).

What we could not reproduce is the stability. Across 50 checkpoint arms on the
training executor, no same-agent improvement survives multiplicity correction,
the lift oscillates between runs with peaks moving across seeds, and the final
checkpoint is never the best. We traced part of this to a control measured once
and reused for ten weeks; re-measuring it moved it 5.7pp, which is larger than
most effects claimed in this literature. Re-paired against a contemporaneous
control, our strongest ALFWorld-on-ALFWorld result (+13.6pp, p=0.0026) becomes
+3.6pp, p=0.47.

We additionally report: (1) substituting Gemini 2.5 Pro for the trained curator,
at 84x the cost, yields no improvement over writing no notes at all, confirming
the paper's directional claim that a small trained curator is competitive;
(2) supplying retrieved skills increases this executor's unparseable-action rate
from 2.1% to 7-9%, a concrete mechanism for why memory can hurt a small model;
(3) the standard 140-game ALFWorld protocol has 80% power to detect a 13pp
effect and no less, placing the field's headline effects at the instrument's
resolution limit.

Our absolute baseline remains 8pp below the original's after eliminating six
candidate causes. We make no claim that the original result is wrong; we report
what a faithful implementation yielded under contemporaneous controls and release
all checkpoints, evaluation records, and code so that every test can be
recomputed.

This study was conducted almost entirely by an LLM agent under human direction.
We report 185 audited failures from that process, a taxonomy of what went wrong,
and the concrete gates that would have prevented the expensive ones. Both of the
agent's consequential errors produced *more* significant results, not fewer,
because a systematically degraded arm is reliably degraded and reliability is
what a significance test rewards.
