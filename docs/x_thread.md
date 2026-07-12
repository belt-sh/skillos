# SkillOS reproduction — X thread draft

Chain of 5 posts + a reply for the link. Voice: senior-engineer postmortem, lowercase, concrete numbers, falsifiable claims, no hype. Each post stands on its own.

## post 1 — hook

skillos reports a monotone-to-60 training curve on alfworld. reproduced it twice on trl and got a bimodal shape both times. peak mid-run, checkpoint 60 lower than the mid-run peak, same recipe both seeds → peak index moves from step 20 to step 35, shape holds. that's the finding, not a bug in our stack.

## post 2 — numbers

seed 1 fft: peak ckpt20 +10.7pp mcnemar p=0.032, ckpt60 falls to +5.7pp. seed 2 fft: peak ckpt35 +13.6pp p=0.003, ckpt60 falls to +4.3pp. n=140 paired-by-gamefile vs canonical no-memory baseline.

tested lora vs full fine-tune: both bimodal. tested uniform vs natural type distribution: both bimodal, natural kills the lift entirely. tested random vs easy→hard within-group curriculum (paper table 5): both bimodal, curriculum shows no significant lift at any checkpoint.

those last two are exactly the paper's grouping ablation. both halves, both null. surviving suspect: trl ≠ verl at the framework layer — advantage normalization, sampling semantics.

## post 3 — headline

the paper's cross-executor generalisation claim reproduces. 8b-trained lora curator driving qwen3-32b executor on alfworld valid_seen: 62.1% absolute, +12.9pp vs no-memory, p=0.0064. n=140 paired mcnemar.

single run per arm and 32b no-memory has ~4pp draw variance, so call it at-parity with the paper's 61.2% headline, not "beats." the important part is that the direction holds without any 32b in the training loop.

## post 4 — the surprise

on-8b checkpoint ranking inverts on 32b. the fft curators that top 8b transfer weakly (+5.7pp p=0.26) or negatively (−2.1pp). the lora curator that ranked third on 8b transfers best on 32b.

reading: fft skills overfit to 8b executor quirks; lora's constrained update stays more generic. n=1 per arm, so hypothesis, not established. worth its own paper if it holds up at n=3.

separately, heat unlocks at 32b — the 8b microwave role-play pathology (executor pretends to open a microwave instead of using the `heat X with microwave` atomic verb) disappears at scale. same for the −14pp alfworld baseline gap we hit on 8b: reasoning baseline on the same executor reproduces the paper within 1σ (aime24 73.3%, aime25 60.0%, gpqa-d 59.6%, avg 64.3% vs paper 69.6±4.7). the gap is environment-specific, not a broad executor quality issue.

## post 5 — what's shipped

full training + eval code for three benchmarks (alfworld, aime24/25, gpqa-diamond) in one repo. runs on 8× h100 + remote executor/judge on inference.sh, no local vllm required for the executor. every deviation from the paper logged in divergences.md. training uses ~$40 of inference.sh credit per 60-step run.

infrastructure worth stealing: a storm-resilient sweep supervisor that survived a 12-hour openrouter provider outage mid-run by matching gate concurrency to sweep concurrency (single + 10-burst + 40-burst probes, two consecutive passes) and auto-restarting killed waves without losing completed arms.

not a "we did skillos better" post. this is peer review as a gift — code you can grpo small models with, on your own domain, with paper-checkable numbers on the way in.

## post 6 — link (reply to post 5)

repo + full report: github.com/belt-sh/skillos

writeup with all numbers: docs/repro_report.md

running narrative including the dead ends: JOURNAL.md

## posting notes

- send post 6 as a **reply** to post 5, not as post 6 in the chain. links in the main chain suppress dwell.
- if a screenshot would help, use post 2 (bimodal training curve overlaid seed-1 and seed-2). image posts unlock the photo_expand signal channel.
- if a video would help, use post 5 (30s screen recording of a sweep launching and the supervisor gate probing). video posts unlock vqv signal but only above a minimum duration threshold.
- best-case dwell: post 2 or post 4 (dense, rewards re-reading). rewrite those to pack more per line if signal is soft on first attempt.
- topic consistency: keep replies on this thread technical (reproduction methodology, grpo, agentic eval). off-topic replies fragment the two-tower embedding.
