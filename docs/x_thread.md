# SkillOS: What's behind the "8B curator beats Gemini-2.5-Pro" claim?

**TL;DR:** SkillOS is a technique for training a small "curator" model that maintains a markdown skill repo for a frozen executor. Google + UIUC + MIT dropped the paper with a striking claim: an 8B-trained curator outperforms Gemini-2.5-Pro at skill curation across every benchmark. No official code. We reproduced the pipeline on TRL instead of verl, on 8 H100s instead of 16, across three benchmarks: ALFWorld, AIME24/25, GPQA-Diamond. The core generalisation claim reproduces — an 8B-trained LoRA curator drives a 32B executor to 62.1% on ALFWorld valid_seen (+12.9pp vs no-memory, McNemar p=0.0064), above the paper's headline 61.2%. But two findings don't match the paper: the training curve is non-monotone-with-post-peak-regression (reproducible across three seeds, peak indices at ckpt 20 / 35 / 55), and the ALFWorld no-memory baseline sits 14pp below the paper on the same executor that reproduces the reasoning baseline within 1σ. Full training + eval code, three-benchmark harness, and a storm-resilient sweep supervisor are on GitHub. This is peer review as a gift, not a "we did SkillOS better" post.

The article is easier to read on GitHub: [github.com/belt-sh/skillos](https://github.com/belt-sh/skillos)

————————

When the SkillOS paper landed in late May, we were excited. The claim was concrete and specific: freeze the executor (the agent that actually solves tasks), train only a tiny curator LLM whose job is to maintain a markdown skill repo, and beat frontier-model curators with RL on the curation decisions themselves. If it held up, it was a real "GPU-poor continuous learning" story — the kind of targeted RL result that makes small open models a serious alternative to frontier APIs for specific workflows.

There was no official code release.

We wanted it to be good. Skills as markdown with YAML frontmatter is the same format used by [Anthropic's Skills](https://docs.anthropic.com/en/docs/agents/skills) and the [belt CLI](https://github.com/belt-sh/cli). If you're already running agents with markdown skills, this method plugs in without changing your storage layer. And if a small trained curator really does beat Gemini-2.5-Pro at managing your agent's memory, that's a very different economics story than "call the frontier model every time you need to update a skill."

So we spent six weeks reproducing it. On TRL 1.4 instead of the paper's verl. On 8 H100s instead of 16, with the executor and judge running remotely on inference.sh. Across three benchmarks — the same three the paper reports.

Then we looked closer.

## About me

I'm okaris, founder of [inference.sh](https://inference.sh). Before this I spent two years at StyleOf working on generative-AI personalisation — over 100,000 fine-tunings and 10,000 training experiments. Agent skill curation is adjacent to that work in the sense that both are about making a small artifact (a LoRA, a skill markdown file) do something a big general model can't be bothered to do well.

I'm not neutral on this paper. The whole "small model wins on a targeted task via RL" thesis lines up with what I'd want to be true for the platform we're building. So this reproduction is deliberately unforgiving. Every number below is McNemar-paired against a fixed baseline you can reproduce from the JSONLs in the repo.

## What SkillOS actually does

The setup is clean:

1. A **frozen executor** (Qwen3-8B) solves a task using its retrieved skills.
2. A **curator** (also Qwen3-8B, this is the one we train) observes the trajectory and emits a `curate_and_advance` tool call that writes, revises, or deletes skills in the repo.
3. Rinse and repeat over Algorithm 1's evolving `|G|=10` same-type task groups.
4. GRPO optimises the curator against a composite reward: `r = r_task + λ_f · r_fc + λ_u · r_cnt + λ_c · r_comp`. Task success + valid tool calls + judge-scored content quality + repo compression.

The idea is that after training, the curator has learned to write skills that are *actually useful to this particular executor on this particular kind of task* — even if the individual skill markdown looks anodyne to a human. The whole loop is executor-grounded: reward comes from downstream task success, not from a proxy signal.

Reasonable expectation: if the method works, we should see (a) held-out task lift, (b) monotone training curves as reward climbs to a stable optimum, and (c) the trained curator transferring to executors it never saw during training.

Let's take those one at a time.

## Setup

- **Hardware.** 8×H100 (paper: 16×H100).
- **Framework.** TRL 1.4 + accelerate + DeepSpeed ZeRO-3 + vLLM colocate. Paper uses verl-agent. This is our first divergence and a known confound — advantage normalisation, sampling semantics, and buffer handling differ between the two. Every deviation from the paper is enumerated in [DIVERGENCES.md](https://github.com/belt-sh/skillos/blob/main/DIVERGENCES.md).
- **Curator.** Qwen3-8B, LoRA r=32 and full fine-tuning (both tested independently).
- **Executor.** Qwen3-8B during training. At eval time we also test on Qwen3-32B, matching the paper's transfer experiment. Executor runs remotely on inference.sh (`openrouter/qwen3-8b` / `openrouter/qwen3-32b`), which lets us keep all 8 local GPUs on the curator.
- **Judge.** Qwen3-32B, also on inference.sh.
- **Benchmarks.** ALFWorld (agentic), AIME24 + AIME25 (numeric reasoning), GPQA-Diamond (multiple-choice science reasoning). Same three the paper reports.

Full training run: ~70 min/step FFT × 60 steps ≈ 3 days on the local box, plus about $40 of inference.sh credit for the executor and judge calls. Per run.

## The good news, first: the generalisation claim reproduces

The paper's headline generalisation claim is that a curator trained against an 8B executor transfers to a 32B executor without any 32B in the training loop. This is the strongest thing SkillOS claims, and the most important to test.

We ran the sweep: our three best curator checkpoints (one LoRA, two FFT) driving `openrouter/qwen3-32b` on 140 paired ALFWorld valid_seen games. Fresh 32B no-memory baseline at 49.3%.

| curator (8B-trained) → 32B executor | abs SR | Δ vs no-memory | McNemar p |
|---|---|---|---|
| v8 LoRA ckpt30 | **62.1%** | **+12.9pp** | **0.0064** |
| FFT seed-2 ckpt35 | 55.0% | +5.7pp | 0.26 |
| FFT seed-1 ckpt20 | 47.1% | −2.1pp | 0.74 |
| paper SkillOS (Qwen3-32B executor) | 61.2% | ~+13pp | — |

62.1% on a single run is above the paper's headline 61.2%. I'm not going to call it "beats the paper" — 32B no-memory has ~4pp draw variance at n=140, and we ran each arm once. Call it at parity. The direction and the order of magnitude match, and the LoRA-curator artifact clears McNemar significance.

The important part is that this generalisation direction is real. It's not a training artifact — the curator was optimised against 8B executor rewards, produced markdown skills, and those markdown skills lifted a 32B executor it had never seen. On a task family where the executor by itself hits 49% no-memory, this is the result the paper is selling and it holds.

**The surprising twist:** the ranking inverts. The FFT curators that outperform on 8B held-out eval transfer worst on 32B. The LoRA curator that ranked third on 8B transfers best on 32B. My reading is that FFT skills overfit to specific 8B executor quirks — the "how this particular model likes its skills phrased" surface — while LoRA's constrained update produces more generic skills that survive an executor swap. Single run per arm, so hypothesis, not established. Worth its own follow-up.

## The bad news: the training curve is non-monotone and peak indices are wild

The paper reports monotone-to-60 training curves — reward climbs, held-out lift climbs, ship checkpoint-60. Our curves look nothing like that.

We ran three independent 60-step FFT runs with different seeds, everything else identical. Then we swept every saved checkpoint (5-step cadence) through paired-by-gamefile McNemar against a fixed 33.6% no-memory baseline.

| run | peak ckpt | peak lift | p | ckpt60 lift |
|---|---|---|---|---|
| FFT seed-1 (seed=42) | 20 | +10.7pp | 0.032 | +5.7pp |
| FFT seed-2 (seed=123) | 35 | +13.6pp | 0.0026 | +4.3pp |
| FFT seed-3 (seed=456) | 55 | +11.4pp | 0.011 | +3.6pp |

Three independent runs. Three significant peaks. Peak indices span **ckpt 20, 35, and 55** — half the training run. ckpt60 lands 4–9pp below the peak on every seed. If you follow the "ship checkpoint-60" rule the paper implies, you leave meaningful lift on the table on every seed.

I want to be precise about what actually reproduces across N=3, because "bimodal" in the strict "two peaks with a clear trough between" sense doesn't fit seed-3 — it's a late-peaking curve with post-peak regression, not a two-peak shape. What does robustly hold across all three seeds is: statistically significant lift somewhere in the run, peak at ckpt < 60, ckpt60 materially below peak, peak lift in the +10 to +14pp band. The name for that is **non-monotone-with-post-peak-regression**, not necessarily bimodal.

The practical consequence is the same either way. **Ship best-of-heldout from a sweep, not `checkpoint-60`.**

This is a real reproducibility finding, not a bug in our stack. The natural next question is *why*. The paper's own ablation says grouping is the single most impactful design lever, so I spent the next month burning training runs on the two halves of that grouping recipe.

**Test 1: type distribution.** The paper's grouping trains on ALFWorld's natural type frequencies (Pick-heavy). Our default is uniform round-robin across the 6 types. We flipped exactly one knob and re-ran the full 60-step training + sweep.

Result: natural distribution *kills* the lift. No arm significant, best +5.7pp p=0.20. The oscillating shape persists. Uniform's balanced exposure to high-headroom types (Clean, Cool, Heat — where the baseline is 19–25%) is load-bearing. Distribution-matching the eval set isn't the win; balanced exposure to where the headroom lives is.

**Test 2: within-group ordering.** Paper Table 5 specifies a soft easy→hard curriculum (p↑=0.80, difficulty = expert-plan length). Ours is random. Same experiment: flip one knob, run 60 steps, sweep.

Result: curriculum shows no significant lift at any checkpoint (best +4.3pp p=0.36). Same oscillating shape. (This run also got taken out by a 12-hour OpenRouter provider outage at step 49; more on the ops side later.)

With both halves of the grouping ablation independently falsified on our stack, grouping is fully exonerated as the driver of our bimodal curve. That leaves exactly one surviving suspect: **TRL ≠ verl**. Advantage normalisation, sampling, buffer semantics — the framework layer.

I'm not testing that one. A verl-agent port is ~1 week of engineering that doesn't change any confirmed result. It's the right next step for someone who wants to publish a follow-up specifically on the framework hypothesis. If you're reading this and thinking about doing it, the code is open and the reproduction infrastructure is built.

## The baseline gap: same executor, different story per benchmark

There's one more thing worth flagging.

Our ALFWorld no-memory baseline is 33.6%. The paper reports 47.9% for the same executor. Same Qwen3-8B, same decode settings (temp 0.6, top_p 0.95, top_k 20, reasoning on), same prompt from Figure 9 verbatim. 14pp below.

That's a big gap. But it doesn't reproduce on reasoning:

| dataset | ours (no-memory) | paper | delta |
|---|---|---|---|
| AIME24 | 22/30 = 73.3% | 76.0±6.9 | −2.7pp (0.4σ) |
| AIME25 | 18/30 = 60.0% | 71.1±10.7 | −11.1pp (1.0σ) |
| GPQA-Diamond | 118/198 = 59.6% | 61.8±1.1 | −2.2pp (2.0σ) |
| **average** | **64.3%** | **69.6±4.7** | **−5.3pp (1.1σ)** |

*(GPQA-D reported aggregate-only per the dataset owner's access condition. No per-problem content in git or web-visible files.)*

Reasoning baseline reproduces within 1.1σ on the average across three datasets. Same executor. If it were a broad model-quality problem, we'd see it on reasoning too. We don't. The gap is environment-specific.

Trace-level evidence points at the culprit: on ALFWorld's compound-verb tasks (Heat, Clean, Cool), the Qwen3-8B executor tends to role-play a physical microwave — *"I open the microwave and place the item inside"* — instead of emitting ALFWorld's atomic verb, `heat X with microwave`. Heat SR on 8B is 25%, matches the paper. On 32B it unlocks to 56–62% — the role-play pathology dissolves at scale. So the 14pp gap on 8B ALFWorld is a specific ReAct + atomic-verb interaction, not model quality.

Practical consequence: read all ALFWorld absolute numbers relative to their own baseline, not the paper's. McNemar-paired lifts are unaffected because they compare arms drawn from the same executor stack.

## Ops, briefly, because it mattered

Two things bit us hard enough to be worth calling out:

**The naive resubmission pattern feeds provider outages.** Mid-run, a two-day OpenRouter outage from the Alibaba provider hit our executor with 10,852 × 429 responses. Our per-call retry loop kept resubmitting; every rollout in step 49 exhausted its 2-shot backoff; every position became a `DEADLINE CUT`; ranks diverged waiting on nothing; process silently exited. Not code, not OOM, not NCCL — an executor storm.

The fix is a **concurrency-matched probe gate**. Before launching a sweep, probe the executor at three concurrencies: single-call, 10-burst, 40-burst. All three must pass, twice in a row, five minutes apart. Otherwise wait. Sweeps get launched in 4-arm waves instead of 8. A storm auto-abort kills the wave and rolls back to the gate, preserving completed arms. That pattern lives in `scripts/natural_sweep_supervisor.sh` in the repo and it saved the curriculum sweep during the outage — 15 storms killed and re-gated over 19 hours, all 12 kept arms verified clean (140/140 games, zero executor-failure markers).

**Watchers that poll `/tmp` will silently die.** I lost 44 hours of GPU time waiting on a completion signal that never came, because the supervisor logged to `/tmp` and the tmp-cleaner deleted the file mid-run. Rule now: supervisors log to `logs/`, watchers poll durable `output/` artifacts. Never `/tmp`. That change is committed.

Neither of those is a SkillOS problem. Both are the kind of thing you only learn by leaving a 3-day training run unattended.

## What actually got shipped

Everything's in the repo:

- **Full Algorithm 1 training loop** in TRL — curator with `curate_and_advance` mega-tool, `r_task = mean executor success over positions 2..|G|`, judge-scored `r_cnt`, `|G|=10` groups.
- **Three-benchmark eval harness** — ALFWorld closed-loop streaming curation, AIME24/25 numeric grading, GPQA-D multiple-choice.
- **Paired-by-gamefile McNemar comparator** — the honest way to report lift over n=140.
- **Storm-resilient sweep supervisor** — the probe-gate pattern above, works against any inference.sh executor app.
- **Every deviation from the paper logged** in `DIVERGENCES.md`, with status: forced, temporary, tested-and-null, resolved.
- **Full narrative log** in `JOURNAL.md`, including the dead ends. If you want to know what didn't work, that's where.

Runs on 8×H100 with the executor and judge on inference.sh, no local vLLM required. If you want to try SkillOS on your own task family with your own executor, this is the smallest working footprint I know of.

## What I'd want next

The paper's cross-domain claim — a reasoning-trained curator transferring to ALFWorld with +13.3pp — is the most striking generalisation number in the paper and we haven't tested it yet. Closed-loop reasoning training is a ~3-day run once seed-3 (n=3 for bimodality) finishes. That's the next thing I'd spend GPU time on if the goal is to close the paper's headline claim beyond just the executor axis.

The verl port is real work and would settle the last open hypothesis (bimodality driven by framework). Someone else's follow-up.

And I'd want to see this method run on a domain that isn't a text-adventure environment or a math benchmark — something with an open-ended skill space and a checkable reward. Tool-use, coding tasks, browser automation. That's where "skills as markdown, curator as trainable meta-model" starts to matter for real applications. ALFWorld is a proof of the mechanism, not a proof of the ceiling.

## Conclusion

SkillOS works. The method reproduces qualitatively across three benchmarks. The transfer claim reproduces at parity on a single run. The training-curve shape doesn't reproduce, and the paper's own grouping ablation is null on our stack — both point at TRL ≠ verl at the framework layer, which we haven't tested.

Nobody built a new foundation model here — not the paper's authors, not us. What SkillOS actually is: **a training recipe for a small executor-grounded curator, plus a runtime pattern of storing agent skills as markdown files and retrieving them via BM25.** The recipe reproduces. The runtime pattern is what belt and Anthropic and a growing pile of agent frameworks are already using.

If your agent uses markdown skills — and if you're on the SKILL.md track, it will — SkillOS's contribution is the RL recipe for training the curator that maintains those skills. It's small, it runs on 8 H100s, it hits paper parity on the generalisation direction, and the code is now open.

That's the honest read.

————————

Repo: [github.com/belt-sh/skillos](https://github.com/belt-sh/skillos)

Full reproducibility report: [docs/repro_report.md](https://github.com/belt-sh/skillos/blob/main/docs/repro_report.md)

Running narrative including the dead ends: [JOURNAL.md](https://github.com/belt-sh/skillos/blob/main/JOURNAL.md)

Every deviation from the paper, categorised: [DIVERGENCES.md](https://github.com/belt-sh/skillos/blob/main/DIVERGENCES.md)

This research was conducted between May 26 and July 12, 2026, using TRL 1.4 + DeepSpeed ZeRO-3 + vLLM colocate on 8× H100, with `openrouter/qwen3-8b` and `openrouter/qwen3-32b` as executors and `openrouter/qwen3-32b` as the judge, all via inference.sh.
