# 1. Introduction

An agent that writes notes to itself and gets better at its job is an appealing
idea. It promises improvement without touching the weights of the model doing
the work, which means it should apply to any frozen or closed model you can
call over an API.

SkillOS (Ouyang et al., 2026) makes this concrete. A *curator* model, trained
with GRPO, maintains a repository of markdown skills. A separate *executor*
model, frozen throughout, reads the retrieved skills and acts. The reported
result on ALFWorld is a 13.3 percentage point improvement in task success rate
over the same executor with no skills, and 61.2% absolute with a 32B executor.

We set out to reproduce this. The motivation was practical rather than critical:
if a small trained curator can lift a frozen agent by 13pp, that is a cheap and
deployable technique, and we wanted it working before building on it.

This paper reports what happened over roughly three months of 8xH100 time and
ten complete 60-step training runs. The short version is that the method works,
the cross-executor transfer claim reproduces at parity, but the same-agent
ALFWorld lift does not survive contemporaneous controls or multiplicity
correction, and the most useful thing we learned was about measurement.

## 1.1 What we did

We implemented the method in two independent reinforcement learning stacks: TRL
with DeepSpeed ZeRO-3, and verl-agent (GiGPO) with FSDP. We trained a Qwen3-8B
curator against a frozen Qwen3-8B executor on ALFWorld, with a Qwen3-32B judge
supplying the consistency reward term, following the paper's hyperparameters. We
swept checkpoints every five steps and evaluated each one on 140 held-out
ALFWorld games, paired by game file and tested with McNemar.

Along the way we ran the paper's own ablations (task-type distribution,
within-group curriculum ordering), a LoRA-versus-full-fine-tuning comparison, a
decode-parameter sweep, a cross-executor-scale transfer study, a cross-domain
study training the curator on mathematics and testing on ALFWorld, and an
additional arm the paper motivates but does not run: substituting a frontier
model, Gemini 2.5 Pro, for the trained curator.

## 1.2 What we found

The method works and the machinery is healthy. Within GRPO groups, task success
supplies 79% of the reward variance that reaches the gradient, the training
converges without pathology, and the curators produce coherent skill
repositories.

One of the paper's claims reproduces cleanly. **Cross-executor transfer**
reproduces at parity: an 8B-trained curator lifts a 32B executor to 62.1% (the
paper reports 61.2%). **Cross-domain transfer** is directionally positive — a
mathematics-trained curator helps an ALFWorld executor in two of three seeds —
but no arm reaches significance across three seeds and twelve checkpoint arms,
and the one result that previously survived Holm correction (+11.2pp) does not
replicate.

What does not reproduce is a stable same-agent ALFWorld lift. For most of those
three months we believed we had one: every sweep contained a significant-looking
peak in the +9 to +14pp band. Three seeds and two frameworks agreed. They
agreed because they were all subtracting the same control number, measured once,
ten weeks before the arms it was being compared to.

Re-measured in the same week, the control moved from 33.6% to 39.8% (four
replicates, spread 2.1pp). Re-paired, our best seed goes from +13.6pp at
p=0.0026 to +3.6pp at p=0.47. We have not shown the curator does nothing; we
have shown that any same-agent ALFWorld effect is smaller than this protocol can
resolve.

Additional findings:

1. **Reusing a control measured against a hosted API is a significance factory.**
   The drift (5.7pp) exceeds most effects claimed in this literature.

2. **Gemini 2.5 Pro, at 84x the cost, does no better than writing no notes at
   all** (p=0.86). The paper's directional claim that a small trained curator
   competes with a frontier model survives.

3. **Retrieved skills increase unparseable-action rate** from 2.1% to 7-9%, a
   concrete mechanism for memory hurting a small model.

4. **Six candidate causes of the null are falsified**, each with a full training
   run, including both halves of the paper's own grouping ablation.

## 1.3 What this paper is not

It is not a refutation. Reproduction failures have many causes, and the most
likely one is always that the reproducers got something wrong. We list what we
know we did differently in Section 4.4 and what we suspect but cannot rule out
in Section 6. Our absolute baseline sits 8pp below the paper's and we could not
close that gap after eliminating six causes, which is itself evidence that
something in our setup differs from theirs in a way we have not identified.

What we can offer is the full apparatus: seven trained models, roughly one
hundred evaluation arms of 140 paired games, the training and evaluation code,
and every JSONL needed to recompute every test in this paper. If the effect is
there and we missed it, the artifacts should make it findable.
