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

This paper reports what happened over roughly two months of 8xH100 time and
seven complete 60-step training runs. The short version is that we could not
obtain the effect, and that the most useful thing we learned was about
measurement rather than about skill repositories.

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

For most of those two months we believed we were reproducing the paper. Every
sweep contained a checkpoint with a significant-looking lift, in the +9 to +14pp
band that brackets the reported +13.3pp. Three seeds and two frameworks agreed.

They agreed because they were all being compared against the same control
number, measured once, ten weeks before the arms that were being compared to it.

When we re-measured that control in the same week as the arms, it moved from
33.6% to 39.8% (four replicates, spread 2.1pp). The old figure sits outside the
replicate spread, so it is not sampling noise. Re-paired against a
contemporaneous baseline, the lifts disappear: our best seed goes from +13.6pp
at p=0.0026 to +3.6pp at p=0.47, and eight of its twelve checkpoints are
negative.

Our principal findings are therefore:

1. **The curator lift does not reproduce under a same-epoch control.** We make
   no claim that the original result is wrong. We report that a faithful
   implementation, run seven times, did not yield it once the control was
   measured correctly.

2. **Reusing a control measured against a hosted model API is a significance
   factory.** The drift we measured (5.7pp) is larger than most effects being
   claimed in this literature. This is a methodological result independent of
   SkillOS.

3. **A frontier curator does no better.** Gemini 2.5 Pro, at 84 times the cost
   per call, scores 1.4pp below writing no notes at all (p=0.86). The
   directional part of the paper's claim, that a small trained curator is
   competitive with a much larger one at this task, survives. The part that
   would make it exciting does not, because neither of them beats the baseline.

4. **Adding skills makes a small executor produce more unparseable actions**,
   from 2.1% with no memory to 7.0-9.1% with curation. This is a concrete mechanism
   for why memory can fail to help, or actively hurt, a small model.

5. **Six candidate explanations for the null are falsified**, each with a full
   training run or a full sweep behind it, including both halves of the paper's
   own grouping ablation.

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
