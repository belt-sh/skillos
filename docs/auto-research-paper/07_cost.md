# 6. The bill

## 6.1 Compute

| category | quantity | wall time | notes |
|---|---|---|---|
| Training runs (10 × 60 steps, 8×H100) | ~180 GPU-days | ~3 months | includes 3 voided runs |
| Evaluation arms (~100 × 140 games) | ~14,000 ALFWorld episodes | ~600 hours | executor + judge inference |
| Wasted on stale-baseline arms | ~40 arms | ~240 hours | all pre-correction arms |
| Wasted on outage-scored arms | 4 arms | ~24 hours | voided entirely |
| Wasted on group-collapse runs | 3 training runs | ~9 GPU-days | no task signal reached optimizer |
| Idle GPU time (config bugs, missed launches) | | ~5 days | batch-size mismatch, supervisor gaps |

Roughly a third of the total compute was spent on work that produced no usable
result. That is not unusual for a research project. What is unusual is that the
wasted work looked successful at every stage — it ran to completion, produced
significant-looking numbers, and was written up confidently.

## 6.2 Inference API

The executor and judge were served via a hosted API (inference.sh). Each
ALFWorld episode requires 20-30 executor calls and 1-2 judge calls. Across
training and evaluation:

| | calls (est.) | |
|---|---|---|
| Training (10 runs × 60 steps × 32 rollouts × ~25 calls) | ~480,000 executor | |
| Evaluation (~100 arms × 140 games × ~25 calls) | ~350,000 executor | |
| Judge (training only, ~1 per rollout position) | ~170,000 judge | |

Total: roughly one million inference calls over three months.

## 6.3 Human time

One person, part-time. The total human investment was approximately:

- ~2 hours/week reviewing results and asking questions
- ~5 major decision points (each taking 1-2 hours of discussion)
- ~3 debugging sessions where the human directed the investigation

Total: roughly 40-50 human-hours over three months. The ratio of agent compute
hours to human oversight hours is approximately 100:1.

## 6.4 What the waste ratio means

A human researcher would not have wasted a third of their compute, because a
human researcher would not have run ten training runs in three months. They would
have run two, checked carefully, and caught the baseline problem on the second.

The agent's advantage — throughput — and its disadvantage — unchecked systematic
error — are the same property. It runs fast enough to compound a mistake across
ten runs before anyone looks. The question is not whether the waste ratio is
acceptable. It is whether the absolute output, after corrections, exceeds what a
human alone would have produced in the same time. In our case: probably yes, but
not by the margin the raw volume suggests.
