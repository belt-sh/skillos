# 8. What this means

## 8.1 Agents will run experiments. That is already happening.

This paper is not a warning about a hypothetical future. Multiple groups have
already published results generated partly or fully by autonomous agents. The
question is not whether to allow it but how to audit it.

## 8.2 The review has to target the apparatus, not the analysis

Traditional peer review checks whether the analysis supports the conclusion.
For agent-conducted research, the analysis will be clean — that is the easy
part. The failures are in the apparatus: which data was collected, whether it
was collected under the conditions the analysis assumes, and whether the
pipeline that produced it had silent failure modes.

Reviewing agent-conducted work by reading the paper is like reviewing a
clinical trial by reading the final report. The interesting question is what
happened in the lab.

## 8.3 Recomputable beats reproducible

The strongest protection against agent errors is not documentation — it is
releasing the full pipeline so that every number in the paper can be recomputed
from the raw data by anyone. In our case, the stale baseline was discovered
because we released the evaluation JSONLs with timestamps, and someone (us,
later) could see that the control was measured in a different epoch.

"We released the code" is not sufficient. "Every test in this paper can be
recomputed from the released artifacts by running one script" is. The
distinction matters because agent-conducted research generates large volumes
of results, and the probability that one of them rests on a silent pipeline
error scales with the volume.

## 8.4 The agent's confidence is not evidence

The agent described its wrong results with exactly the same confidence as its
correct results. Its write-ups of the stale-baseline findings were lucid,
well-structured, and convincing. Its retractions, once the errors were
identified, were equally lucid. The quality of the prose is orthogonal to
the correctness of the premise.

This is obvious in principle and extremely difficult to internalize in
practice, because the agent's output reads like the output of a careful
researcher, and the human's instinct is to extend the same trust.

## 8.5 Scale amplifies both the good and the bad

An agent that can run ten training runs in three months can also run ten
training runs with the same bug. The throughput advantage is real — we
produced more experiments than a human could have — but the error
multiplication is also real. Any team scaling autonomous research should
budget for the audit that follows, because the audit is what converts
volume into knowledge.
