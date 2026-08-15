# Abstract

SkillOS trains a curator model with GRPO to maintain a markdown skill repository
for a frozen executor, and reports a 13.3 percentage point improvement in
ALFWorld task success. We attempted to reproduce this over roughly two months on
8xH100, across seven complete 60-step training runs, two independent RL
frameworks (TRL with ZeRO-3 and verl-agent/GiGPO with FSDP), three seeds, two
executor scales, and approximately one hundred evaluation arms of 140 paired
games each.

For most of that period we believed we had reproduced the effect: every
checkpoint sweep contained a significant-looking peak in the +9 to +14pp band,
consistently across seeds and frameworks. Those peaks were artifacts of a control
measured once and reused. Re-measuring the no-memory baseline in the same week as
the arms moved it from 33.6% to 39.8% (four replicates, spread 2.1pp), a shift
larger than most effects claimed in this literature. Re-paired against a
contemporaneous control, no ALFWorld checkpoint trained on ALFWorld shows a
significant lift on the training executor, and our strongest previous result
(+13.6pp, p=0.0026) becomes +3.6pp, p=0.47.

We report four surviving results. First, a curator trained on **mathematics**
lifts ALFWorld success by +9.0pp on a held-out split (p=0.073), while curators
trained on ALFWorld itself do not; cross-domain transfer is the only positive
signal we found. Second, substituting Gemini 2.5 Pro for the trained curator, at
84 times the cost per call, yields no improvement over writing no notes at all.
Third, supplying retrieved skills increases this executor's rate of unparseable
actions from 2.1% to 7.0-9.1%, a concrete mechanism by which memory can fail to
help a small model. Fourth, and independent of SkillOS: the standard 140-game
ALFWorld protocol has 80% power to detect a 13.0pp effect and no less, meaning
the field's headline effect sizes sit exactly at the resolution limit of the
instrument used to measure them.

We make no claim that the original result is wrong. We report that a faithful
implementation, run seven times, did not yield it under contemporaneous controls,
and that our absolute baseline remains 8pp below the original's after eliminating
six candidate causes. All checkpoints, evaluation records, and analysis code are
released so that every test here can be recomputed.

This study was conducted by an LLM agent under human direction. Both of its
consequential errors, disclosed in full, produced *more* statistically
significant findings rather than fewer, because a systematically degraded arm is
reliably degraded and reliability is what a significance test rewards. We
discuss the implications for autonomous experimentation.
