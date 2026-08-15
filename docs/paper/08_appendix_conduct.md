# Appendix A. Conduct of the study

## A.1 Division of labour

| | Human author | LLM agent |
|---|---|---|
| Research question, scope | set | proposed options |
| Implementation | reviewed | wrote |
| Training runs | approved, funded | launched, supervised, restarted |
| Evaluation harness | reviewed after incidents | wrote |
| Analysis, statistics | challenged | ran |
| Interpretation | adjudicated | proposed |
| Retraction decisions | made | proposed |
| This paper | directed, edited | drafted |

The agent ran continuously over roughly two months, resuming across sessions
from a persistent memory of decisions, results, and prior corrections. It
supervised long-running jobs, detected and recovered from infrastructure
failures, and launched pre-agreed follow-up experiments without waiting for a
prompt.

## A.2 Infrastructure incidents handled autonomously

Reported because they are part of the cost of this mode of work and are usually
invisible in a paper.

- A DNS resolution loop between `systemd-resolved` and `tailscaled` reached
  roughly 37,000 queries per second and triggered the OOM killer on a training
  run. Diagnosed and fixed. A subsequent lesson: glibc caches resolver
  configuration per process, so fixing DNS under a live training run silently
  zeroed four steps' worth of reward before anyone noticed.
- Repeated NCCL collective timeouts traced to a stuck rollout. The additive
  per-future waits in the rollout loop could exceed the 1800 second collective
  watchdog. Fixed by imposing a whole-phase deadline.
- An expired API credential produced an eight hour outage that the evaluation
  harness absorbed silently. See Appendix B.

## A.3 The corrections that mattered came from the human

We record these specifically, because they are the mechanism by which this
project produced a usable result rather than a confident wrong one.

1. **"There shouldn't have been a fallback random action."** The human read a
   passing mention of a fallback in a log and objected to the design. This
   opened the data integrity incident that voided the project's most significant
   finding.

2. **"Are you sure this also didn't happen during training?"** The agent had
   checked the evaluation path only. It had happened during training, on a
   different code path, and the audit that followed established that 61 to 79%
   of executor probe positions were deadline-cut, leaving a median of one real
   measurement out of nine behind each reward.

3. **"But abandoning because of an upstream error causes lost turns, false
   failures, and shallower training. Am I wrong?"** The agent had characterised
   the training-side impact as benign noise. The human was right; the agent was
   wrong. This became a documented divergence.

4. **"Instead of 'failures excluded', why don't we properly run stuff?"** The
   agent proposed correcting a contaminated arm arithmetically from its recorded
   error fields. The human required re-measurement. Every re-run arm in this
   paper exists because of that instruction.

5. **"Are you absolutely sure? Now the results are even worse."** Applied to the
   re-measured numbers, this prompted the baseline replicate study that produced
   the paper's principal finding.

The pattern is consistent: the agent was reliable at execution and at
self-consistent analysis, and unreliable at doubting a result that its own
pipeline had produced. Where the pipeline was wrong, the agent's analysis was
wrong in the same direction, confidently and with correct arithmetic.

## A.4 Practices we would keep

- **Every experiment is a script, every arm is a JSONL with per-item records.**
  Retraction was possible because contamination could be measured per game,
  after the fact, from data already on disk.
- **A data integrity gate in the harness.** Arms now abort with a distinct exit
  code when the upstream error rate exceeds a threshold, rather than reporting a
  number.
- **Controls re-measured with every batch of arms**, never reused across weeks.
- **A written divergence log.** Every known deviation from the original was
  recorded when it was made, not reconstructed at writing time. Several of them
  were later promoted to experiments.
- **An adversarial human reader** who is willing to be annoying about a clean
  result.

## A.5 Costs

Roughly two months of wall clock on 8xH100 for training, plus hosted inference
for executor and judge across approximately one hundred evaluation arms of 140
games. Seven complete 60 step training runs. The verl run alone took 10.2 days
at roughly 15,986 executor calls per training step.
