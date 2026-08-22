# 4. Methodology

## 4.1 The method under test

SkillOS trains a curator policy to maintain a repository of markdown skills that
a frozen executor retrieves at inference time. We implemented Algorithm 1 as
described in the paper.

For each training prompt the curator observes a task and the current repository,
then emits repository operations (add, edit, delete) as function calls. The
edited repository is then used by the frozen executor on a sequence of probe
tasks, and the executor's success rate becomes the primary reward term.

The composite reward is the paper's Equation 1:

```
r = r_task + λ_f · r_fc + λ_u · r_cnt + λ_c · r_comp
λ_f = 1.0,  λ_u = 0.1,  λ_c = 0.05
```

where `r_task` is mean executor success over probe positions 2..|G|, `r_fc`
scores function-call validity, `r_cnt` is a judged consistency score from a
Qwen3-32B judge, and `r_comp` scores repository compactness.

## 4.2 Models, hyperparameters, and hardware

| | |
|---|---|
| Curator | Qwen3-8B, trained. LoRA r=32 and full fine-tuning |
| Executor | Qwen3-8B, frozen. Also evaluated with Qwen3-32B |
| Judge (`r_cnt`) | Qwen3-32B |
| Algorithm | GRPO, 60 steps, group size N=8, data grouping size \|G\|=10 |
| Effective batch | 32 prompts per optimizer update (paper Table 4) |
| Learning rate | 1.0e-6 for full fine-tuning; 1.0e-5 for LoRA |
| KL coefficient | β = 0.001 |
| Frameworks | TRL 1.4 + DeepSpeed ZeRO-3 + vLLM colocate; verl-agent (GiGPO) + FSDP |
| Hardware | 8xH100 local for the curator; hosted API for executor and judge inference |

The paper used 16 H100s. We used 8 locally and moved executor and judge
inference to a hosted endpoint, which we treat as equivalent in capability but
not in wall clock.

## 4.3 Evaluation protocol

Every ALFWorld arm is 140 games from the `valid_seen` split, run under streaming
curation: the curator updates the repository between games, so the arm measures
the curator as it would actually be deployed. A second protocol, 134 games from
`valid_unseen`, is used where noted as a true held-out test for checkpoints
selected on `valid_seen`.

Arms are compared **paired by game file** and tested with McNemar's test on the
discordant pairs. Pairing matters more than it usually does here: ALFWorld game
difficulty varies enormously by task type, so an unpaired comparison of two 140
game runs is dominated by which games each happened to draw.

Reasoning arms are AIME24 (30 problems), AIME25 (30), and GPQA-Diamond (198).
Per the dataset's access terms we report aggregate accuracies only; no problem
text, options, gold answers, or model responses appear in this paper or in the
released artifacts.

### 4.3.1 Contemporaneous baselines

Every arm reported in Section 5 is paired against a no-memory control measured
**in the same week, on the same games, under the same harness build**. This is
not a routine detail. It is the correction that changed our conclusions, and
Section 5.1 reports what happened when we did not do it.

We recommend the practice generally. Executors served over a hosted API are not
fixed instruments. A control measured against one is a measurement of that
endpoint on that day, and reusing it later silently converts endpoint drift into
apparent treatment effect.

## 4.4 Known deviations from the original

We list these up front because a reproduction failure has to be read against
them.

1. **Framework.** The paper's stack is not ours. TRL with ZeRO-3 and verl-agent
   with FSDP differ from each other and from the original in advantage
   normalisation, sequence packing, and optimiser sharding. We ran both partly
   to bound this.
2. **LoRA in some runs.** Three runs use LoRA r=32 with a 10x learning rate
   rather than full fine-tuning. Full fine-tuning runs are also reported and
   behave the same way.
3. **Effective batch in the verl run.** The verl run used an effective batch of
   64 against the paper's 32. This was our error. It means the framework
   comparison varies two things at once.
4. **Executor and judge served remotely** rather than colocated.
5. **WebShop not attempted.** The paper's third benchmark is absent.
6. **Absolute baseline gap.** Our no-memory Qwen3-8B ALFWorld baseline is 39.8%
   against the paper's 47.9%. We eliminated prompt wording, retrieval, seeds,
   numerical precision and serving stack, and decode parameters as causes. The
   gap remains unexplained and is the strongest single indication that something
   in our setup differs from theirs in a way we have not found.

### 4.4.1 The completion-budget truncation and the paper-faithful rerun

Our first six TRL training runs shared an undetected fidelity gap: TRL enforces
`max_completion_length` against the accumulated multi-turn completion, not per
response. At the paper's 4,096-token setting, a ten-position rollout was
truncated at roughly three positions, so the curator was optimised on 2.3 of
the 9 informed positions the paper specifies. The issue was invisible because a
truncated rollout is indistinguishable from a completed one.

The final TRL run (`dense10`) corrected both causes — raising the completion
budget to 16,384 tokens and the phase deadline to 5 hours — and completed the
full 60-step schedule. Its training-time health: median 9 of 9 informed
positions measured in 85% of batches, action coercion 0.09%, zero early exits.
This run is the paper-faithful TRL witness to the same-agent result, and its
evaluation is reported in Section 5.2.

## 4.5 Conduct of the study: an LLM agent as the running experimentalist

This study was executed by a large language model agent (Claude, Anthropic)
operating under human direction over approximately three months, with the human
author setting objectives, approving experiments, and adjudicating disputes. The
agent wrote the implementation, launched and supervised training runs, built the
evaluation harness, ran the analyses, and drafted this paper.

We disclose this for three reasons.

**It is a threat to validity.** Two of the three most consequential errors in
this project were introduced by the agent and went undetected for weeks: an
evaluation harness that scored an upstream API failure as an ordinary task
failure, and the reuse of a stale control. Both produced *more* statistically
significant results, not fewer, because a systematically degraded arm is
reliably degraded, and reliability is what a significance test rewards. An
autonomous experimentalist optimising for "get a result" is exposed to a failure
mode that a human under publication pressure is also exposed to, with less
friction in the way. Section 6.3 develops this.

**It changes what the artifact set is worth.** Because every experiment was
launched from a script rather than a notebook, and every arm wrote a JSONL with
its full per-game record, the entire study is recomputable. That is a direct
consequence of the mode of work, and it is why the retractions in this paper
could be diagnosed precisely rather than argued about.

**It is a data point on autonomous research.** Seven training runs, roughly one
hundred evaluation arms, six falsified hypotheses, and two self-caught data
integrity incidents is a meaningful amount of experimental work. It was also
possible only because the human author repeatedly refused conclusions the agent
proposed. The specific pattern that mattered is recorded in Appendix A: on at
least four occasions the agent presented a result as settled and the human's
scepticism was correct. The corrections that produced this paper's actual
findings came from that loop, not from the agent's own review.

We do not draw a general conclusion about agent-run science from a single study.
We report the conditions under which this one produced usable output: cheap
recomputation, adversarial human review, and a willingness to retract in public.
