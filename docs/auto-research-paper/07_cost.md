# 6. The bill

All GPU numbers below come from wandb run logs (47 logged runs). The
8xH100 box was provisioned for the full 109-day project duration. GPU cost
is estimated at $3.29/GPU/hr (Lambda Labs H100 80GB rate at time of
writing). Inference API costs are from the inference.sh transaction ledger.

## 6.1 Compute

| category | wall hours | GPU-hours | cost (est.) |
|---|---|---|---|
| Completed training runs | 913 | 7,307 | $24,041 |
| Crashed/failed runs | 414 | 3,314 | $10,905 |
| Idle (no training running) | 1,288 | 10,302 | $33,905 |
| **Total (109 days, 8xH100)** | **2,616** | **20,928** | **$68,853** |

The completed-run number overstates useful compute. During training, the
GPUs wait on the remote executor to finish each rollout before computing
the gradient update. We measured the actual GPU utilization during a
representative training step at 1.4% -- the gradient update takes 2.5
minutes in a step that runs 3-7 hours. The other 98.6% is idle wait time
while the executor plays ALFWorld episodes via the inference API.

By that measure, the eight H100s performed roughly 149 GPU-hours of actual
gradient computation across the entire project. At the same rate, that work
could have run on a single GPU in under a week. The remaining $68,000 paid
for a machine to sit in a loop calling `curl` and waiting for responses.

## 6.2 Inference API

The executor and judge were served via a hosted API (inference.sh). Each
ALFWorld episode requires 20-30 executor calls and 1-2 judge calls. Across
training and evaluation:

| | calls (est.) | |
|---|---|---|
| Training (10 runs x 60 steps x 32 rollouts x ~25 calls) | ~480,000 executor | |
| Evaluation (~100 arms x 140 games x ~25 calls) | ~350,000 executor | |
| Judge (training only, ~1 per rollout position) | ~170,000 judge | |

Total: roughly one million inference calls over three months. The
inference.sh transaction ledger does not currently expose a per-period
summary endpoint (a feature gap the project surfaced -- more on this
below), so we report the current account balance of $91.63 and note that
the total API spend was small relative to GPU cost. At inference.sh's
pay-per-token pricing for Qwen3-8B and Qwen3-32B, one million calls with
typical ALFWorld context lengths costs on the order of tens of dollars.

**A note on billing observability.** The inference.sh API exposes
`/v1/transactions` with cursor-based pagination at 10 items per page, and
`/v1/usage/summary` with lifetime aggregates by hardware type. Neither
supports date-range filtering. To answer "how much did this project cost in
API calls," we would need to paginate through tens of thousands of
transaction records -- roughly 20,000 pages. A `/v1/usage/summary?from=&to=`
endpoint, or a downloadable CSV of the transaction ledger, would make
per-project cost analysis trivial. This is the kind of tooling that
autonomous research agents will need: when a training run makes 50,000 API
calls over four days, the human paying for it should be able to see the
cost without writing a pagination script.

## 6.3 Human time

One person, part-time. The total human investment was approximately:

- ~2 hours/week reviewing results and asking questions
- ~5 major decision points (each taking 1-2 hours of discussion)
- ~3 debugging sessions where the human directed the investigation

Total: roughly 40-50 human-hours over three months. The ratio of agent
compute hours to human oversight hours is approximately 100:1.

## 6.4 What the waste ratio means

35% of GPU cost went to useful training. 16% went to crashed runs. 49%
went to idle GPUs -- the box sitting cold because a run finished overnight
and the agent could not autonomously start the next one.

A human researcher would not have wasted half their compute on idle time,
because a human researcher would not have been unable to start a job at
3 AM. But a human researcher also would not have run ten training runs in
three months. They would have run two, checked carefully, and caught the
baseline problem on the second.

The agent's advantage -- throughput -- and its disadvantage -- unchecked
systematic error -- are the same property. It runs fast enough to compound
a mistake across ten runs before anyone looks. The question is not whether
the waste ratio is acceptable. It is whether the absolute output, after
corrections, exceeds what a human alone would have produced in the same
time. In our case: probably yes, but not by the margin the raw volume
suggests.
