# 1. The setup

In May 2026 we pointed an LLM agent (Claude, Anthropic) at a recent ML paper
and told it to reproduce the results. The paper was SkillOS (Ouyang et al.,
2026), which trains a small curator model with GRPO to maintain a skill
repository for a frozen executor agent. The headline claim: a 13.3 percentage
point improvement on ALFWorld, a household-task benchmark.

We chose it because the claim was practically useful — if true, a cheap
fine-tuned 8B model could lift any frozen agent by double digits — and because
the method was self-contained enough that an agent could plausibly implement it
end to end.

The hardware was 8 H100 GPUs running locally and a hosted inference API for the
executor and judge models. The human author (one person) set objectives every few
days, approved major decisions, and otherwise let the agent run. The agent wrote
the training code, built the evaluation harness, launched and supervised training
runs, ran the analyses, and drafted the paper.

The project ran for roughly three months. It consumed ten complete 60-step
training runs (each 2-10 days of wall time), approximately one hundred
evaluation arms of 140 paired games each, and an amount of inference API
spend that we will detail in Section 7.

We did not set out to study autonomous research. We set out to reproduce a paper.
The auto-research findings are a byproduct of having tried.
