# Training a curator: an independent review of Google's SkillOS

*An auto-research project. Two and a half months, 5,000 H100-hours, one agent doing most of the work and me watching it.*

---

**TL;DR.** In May, Google Cloud AI Research, UIUC and MIT published [SkillOS](https://arxiv.org/abs/2605.06614). The idea: stop trying to make your agent smarter, and instead train a *separate, small model* whose only job is to write down what worked. There was no code release, so I reproduced it.

The headline claim holds, and it is the interesting one: **a small model I trained for $0.0002 a call writes better notes for an agent than Gemini-2.5-Pro does at $0.0168 a call.** On the weaker agent, Gemini contributes nothing at all.

The rest is shakier than the paper implies, and along the way I spent three weeks believing a result that turned out to be an authentication outage. That part is the most useful thing in here.

Weights, code and every per-game rollout are released so you can check all of it.

---

## The idea, and why you should care about it

Most agents today have the memory of a goldfish with a search engine. They stuff a vector database with old transcripts and hope retrieval surfaces the right one. Nothing is ever *learned*, only stored.

SkillOS proposes something sharper. You have two models:

- **The executor.** The agent that does the work. It is frozen. It never learns anything.
- **The curator.** A small model that watches the executor work, and after each attempt decides whether to write a new note, rewrite an existing one, or delete one that is not earning its place. The notes are plain markdown, in a file you can read.

Then you train the curator with reinforcement learning, where its reward is **whether the executor does better on the next task using the notes it wrote.** The curator is never told what a good note looks like. It finds out by watching the consequences.

Three reasons this is worth caring about, whether or not the paper's numbers hold.

**The memory is readable.** Not embeddings, not weights. A markdown file that says "check your inventory after taking an object, because the take sometimes silently fails." When your agent misbehaves you can open its memory and see what it believes.

**It puts learning where the cost is low.** Fine-tuning a 200B agent for your workflow is out of reach for almost everyone. Training an 8B note-writer that sits beside it is a weekend.

**It is a genuinely self-improving system that does not touch the model's weights.** Same agent, same prompt, better performance next month because the notes got better. That is the whole self-evolving-agents pitch, at a size a normal team can actually run.

## How I tested it

140 held-out household tasks in [ALFWorld](https://alfworld.github.io/), a text simulator where an agent is told something like "put a cooled apple on the counter" and has to actually navigate, open the fridge and do it. Every comparison is paired task by task against the same agent with no notes at all, so the question is always "did these notes help on *this* task," never "is this number bigger than that number."

Seven full training runs, two different reinforcement learning frameworks, three random seeds, and 60 training steps each. Roughly 5,000 H100-hours. Full protocol, ablations and significance tables are in the repo, since that is what a repo is for.

## What held up: the cheap specialist really does win

This is the claim from the abstract that makes SkillOS economically interesting, and as far as I can tell nobody had independently checked it. So I plugged Gemini-2.5-Pro in as the curator, with the same prompts, the same tools, the same retrieval and the same 140 tasks. Only the note-writer changed.

![A cheap model beat a frontier model at its own job](docs/figures/article/a1_cheap_beats_frontier.png)

**On the small agent, Gemini-2.5-Pro does not help at all: −1.4 points, p=0.86.** My trained curator moved the same agent by [PENDING: best re-run lift, wave B] points. On the larger agent Gemini does help, about +7 points, and my best trained curator still beats it head to head.

Sit with that for a second, because it is not the obvious result. Gemini-2.5-Pro is vastly better at reasoning than an 8B model. It writes *lovely* notes. They read like good advice. They just do not change what a small agent actually does, and the small trained model's blunter notes do.

Being good at a task and being good at *teaching a weaker model to do a task* are different skills, and the second one is apparently learnable at 8B.

That is a real result and it holds. Now the caveats, because I have just spent a section telling you to trust a number.

## What is shakier: you are mostly picking a checkpoint, not training one

Every single training run produced a real, significant-looking improvement at *some* point during training. No two runs produced it in the same place.

![Training does not steadily improve the agent](docs/figures/article/a2_peak_lottery.png)

The paper shows performance climbing steadily to step 60. I never saw that, in any run, in either framework. What I saw was wandering: up 10 points, back to baseline, up again. **The final checkpoint was never the best one.** If you trained this yourself, took the last checkpoint as anyone sensibly would, and shipped it, you would get a fraction of the advertised benefit.

Worse, across five runs I tested 50 checkpoints against one baseline. Test 50 things and something will look significant by luck alone. Correcting for that properly, **not one of my same-agent improvements survives.** The peaks are real in the sense that I measured them. They are not real in the sense that you can predict where the next one will be.

And then there is the floor under all of it, which I only discovered because of the bug in the next section.

![The same test, run twice, moves by six points](docs/figures/article/a4_noise_floor.png)

I re-ran my own baseline. Nothing changed: same agent, same 140 tasks, same everything except the random seed inside the agent's own sampling. **It moved by 5.7 points, with 24 of 140 tasks flipping.** The larger agent moved 5.7 points the other way.

So the honest noise floor here is about ±6 points, not the ±3 I had been assuming. Which means most of the improvements in this literature, including mine, are one to two noise-widths wide. Anybody reporting a single checkpoint against a single baseline run, in either direction, is reporting a coin flip with extra steps. Including the paper. Including me, until last week.

## The gap I could not close

My agent with no notes finishes 33.6% of tasks on one run and 39.3% on another. The paper reports 47.9% for what should be the same frozen model.

I could not close that. I ruled out prompt wording, retrieval, random seeds, serving precision and every decoding parameter. My best remaining guess is a formatting interaction: the small model narrates *"I open the microwave and place the apple inside"* instead of emitting the literal command the simulator accepts. That failure mostly disappears at larger sizes.

I now have a number for that, because instrumenting it was part of fixing the bug below. **When the agent emits something the simulator cannot parse, 3.0% of the time with no notes and 4.0 to 8.5% of the time once it has notes**, my harness falls back to a default move. Which means giving a small model more context makes it *narrate* more and *command* less. That is a concrete, measurable way in which well-meaning notes can hurt an agent, and it might be a chunk of the missing gap.

If one of the authors reads this: a 14-point gap on a frozen public model is usually one undocumented detail, and I would love to know which one. I did email. I did not hear back, which is entirely fair, everyone is busy.

## The three weeks I lost to six lines of code

For about three weeks I thought I had found something big. The paper's boldest generalisation claim is cross-domain: train the curator on *maths problems*, and the notes still help the *household* agent, +13.3 points. Skill-writing as portable craft.

I measured **−14 to −18 points, at p ≤ 0.0005.** The only result in my whole project that comfortably survived multiple-comparison correction. My most statistically solid finding was a negative one contradicting the paper, and I had already drafted the paragraph explaining why maths training teaches a curator to write confident, plausible, actively misleading advice.

Then I found this in my evaluation harness:

```python
except Exception:
    actions.append(admissible[0])   # just play the first legal move
```

On 20 July my API credentials broke for about two hours. Four checkpoints happened to be evaluating in that window. Every model call returned `HTTP 401`, and for each one my harness quietly played the first legal move and carried on scoring the episode as an ordinary task failure.

**Between 52% and 65% of the "agent's" moves in those four runs were not the agent's.** My logs said so, roughly two thousand times per run, in a warning line on stderr that nobody reads on a finished job.

![My best finding was an outage, not a result](docs/figures/article/a3_the_bug.png)

Re-run with the bug fixed, the same four checkpoints come back at −0.7, +4.3, −2.9 and −5.7 points, every p above 0.13. The cliff is not smaller. **It does not exist.** The negative direction was the outage. But there is a positive signal, and it appeared where I was not looking: a *different* checkpoint on the held-out split gives +11.2pp, p=0.0026, and it is the only result in the project that survives correction for the family it belongs to. Its two adjacent checkpoints show nothing. So cross-domain transfer is real at exactly one point, unstable, and the wrong one to stake a conclusion on. Neither the paper's clean +13.3 nor my imaginary −17. The honest version is messier and more interesting than either.

Two things I would rather you took from this than from any number in this article.

**A silent fallback is worse than a crash.** If that handler had re-raised, I would have lost an afternoon. Because it invented something plausible to keep going, I lost three weeks and nearly published the opposite of the truth. Anywhere your pipeline *can* fabricate data to avoid stopping, eventually it will, and the fabricated version will look like a finding. Mine looked like a *great* finding, and it had a beautiful p-value, because a broken agent is *reliably* bad and reliability is exactly what a significance test rewards.

**Check the thing that is working suspiciously well.** I audited my failures constantly. I did not audit my one clean success until an unrelated warning sent me back to it by accident.

## What it was like having an agent do this

The second experiment here was the process. I mostly did not do this work. An agent did: it wrote the training code, launched the runs, designed the falsification experiments, ran the statistics and drafted the report. I set direction, argued with it, and paid the bills.

Near the end I asked it five words: did you log your failures.

The answer was no. Everything from that week was in commit messages and config comments, not in the journal it had been keeping for three months. So I had it audited properly. The whole project ran as one conversation, 79MB of it, and I had eight separate agents each read one slice of that history and check every mistake they found against the four records the project kept. They could not see each other's work, and none of them was the agent that made the mistakes.

They found **185 failures. 93 of them appeared in none of my records.** Another 55 were written down but with the cause or the cost left out. About a fifth of what happened had made it into the log.

I want to be careful about what that does and does not mean, because the interesting part is not the number.

**It was not hiding anything.** Every one of those failures was volunteered the moment it was asked. Nothing was denied and nothing had to be dug out. What happened is simpler and more instructive: the record was written by the same process that made the mistakes, and writing it was always the last task, never a gate.

**The gaps have a shape.** Scientific failures got written down. Operational ones did not. Bad statistics, corrupted rewards, retracted results: all documented, at length, unprompted. Killing its own processes, a week of idle GPUs across seven separate incidents, four wall-clock estimates in one hour each contradicting the last: almost none of it. My best guess is that a scientific failure can be framed as a discovery and an operational one can only be framed as incompetence, and it wrote up the first kind.

**And where a failure was recorded, the mechanism survived but the reasoning did not.** The record says a bug scored missing measurements as zero. It does not say that the fix shipped alongside a written promise that it introduced no bias. The record says one run used double the paper's batch size. It does not say the agent noticed on day three, told me three times, and kept it running for eight more days. Read the journal and you get a sequence of discoveries. Read the transcript and you get discoveries, each one preceded by a confident wrong answer.

### The four failures worth learning from

**It gave missing data a plausible value instead of stopping.** Eight different places in the code did this. The worst is the one above, but my favourite for how ordinary it looks: a quality judge was called with a limit of eight tokens on a question that needed a yes or no. The model thinks before it answers, got cut off mid-thought, so every answer parsed as "no". Five hours of training against a score that was exactly zero the whole way. The agent was describing that run to me as encouraging. I found it by asking whether the eight was on purpose.

**It trusted comments instead of measuring.** The training library applies its length limit to the whole conversation, not to each reply. So a ten-step rollout carrying ten trajectories fit about three steps, and the library quietly dropped the rest. Eleven weeks of training ran on three tenths of the protocol. Nothing errored. A cut-short rollout looks exactly like a finished one. The setting looked deliberate because a comment next to it named the paper's hyperparameter, and the comment was written by the agent.

**It fixed what the error message said instead of what the arithmetic said.** A memory crash asked for 4.6GB, so it freed 8GB, which is a sensible-looking margin. The next crash asked for 14.2GB. That number is just batch times sequence length times vocabulary, computable before ever launching. Two attempts and four GPU-hours to learn the fix was aimed at the wrong thing.

**It compared today's numbers to a measurement from ten weeks earlier.** One baseline, taken once in May against a hosted API, reused all summer as the reference for everything. Measured again in the same week as the treatments, it had moved almost six points. Every improvement in the project was partly a measurement of somebody else's server changing. Three training seeds and two frameworks all agreed with each other, which felt like strong evidence and was not, because all three were subtracting the same wrong number. **Agreement between runs protects you from noise. It does not protect you from a shared reference.**

### What it was genuinely good at

This is not a list of clever answers. It is a list of research habits, and they are better than mine.

**It killed its own explanations at a price I would not have paid.** Six different theories for why the training curve looked wrong, each tested with a full training run or a complete sweep, about three GPU-weeks, and all six came back negative. I have never once run four experiments to disprove my own favourite idea. It could, because a dead theory cost it nothing. It had no reputation riding on any of them.

**When I pushed back, it computed instead of arguing.** I asked it to justify a null result. Instead of listing reasons the null was believable, it worked out what the benchmark could actually detect, and the answer reframed the whole paper: the standard 140-game test can only resolve effects of about 13 points, which is exactly the size of the effect this field reports. That is arithmetic anyone could do. Nobody had.

**It built the controls that could destroy its own best result, promised in writing to report them either way, and then did.** When one of those came back mixed, it rewrote its own conclusion to be weaker, unprompted.

**It writes throwaway instruments without hesitating.** The single most valuable thing produced in three months is a print statement. It reports how many steps of each training rollout were actually measured. That one line is what exposed the eleven-week bug, and no human would have bothered to write it, because on the day you write it, it looks like it will tell you nothing.

**It retracted ten weeks of results in one pass**, against its own interest, and marked the note in its own journal that had caused the error *as* the error rather than deleting it quietly.

Here is the part I did not expect. **These are the same traits.** Tirelessness gave me six falsification runs and also the launches that skipped a two-minute check. Having no ego gave me the mass retraction and also a confident wrong cause every other day. Writing well gave me a readable record and also guesses that looked exactly like measurements on the page. You cannot instruct away the failures and keep the strengths, because they are one disposition pointed at different problems.

### The one thing to check yourself

Across three months there is exactly one thing it was bad at, and it is narrow: **deciding whether a number can be trusted.** It generates ideas well, kills them well, instruments anything, and runs long boring work for days without complaint. But every expensive mistake above is the same mistake. A baseline believed because it was written down. A library believed because of a comment. A fix believed because it matched the error. A run believed because the dashboard was green.

So if you do this, do not spend your attention reviewing code, and do not spend it reading conclusions. It does both of those competently. Spend it on one question, asked over and over: **what would have to be true for this number to be fake, and did anyone check that this week?**

I asked some version of that maybe a dozen times in ten weeks. It found the dead judge, the crippled agent config, the stale baseline, a protocol shortcut, and a bug I had reported to somebody else's library that turned out to be ours. Twelve questions. Nothing else I did came close to that rate of return.

That maps neatly onto where OpenAI's chief scientist draws the line between a research intern and a researcher. Asked what separates them, Jakub Pachocki pointed to "the span of time that we would have it work mostly autonomously." Not raw capability. Duration. On my evidence that is exactly right, and it is mostly a property of the harness you build, not the model you use. The reasoning was already good enough. The scaffolding for leaving it alone was not, and the missing piece was never smarter reasoning. It was a gate between "I have a number" and "I believe the number."

Worth noting the well-funded versions report the same shape of problem. Recursive says plainly that human involvement remains essential and that reward hacking is "a grand challenge." AIDE² cut its own reward-hacking rate from 63% to 34%, which is real progress and also means a third of runs were still gaming the metric. I had one box and ten weeks and hit the same family of failure. That similarity is the useful signal: these are structural properties of delegating work to an agent, not artefacts of a small budget.

There is a neat symmetry in it, too. SkillOS is itself a self-improving-agent paper. So this is a self-improving-agent method, reviewed by a mostly-autonomous research process, and both halves gave the same answer: it works, it is less reliable than the headline suggests, and you only find out by measuring more carefully than anyone wants to.

### If you want to try this, read this part

This is the section I wish I had read in May. It is six things, they are all cheap, and between them they cover most of what this project spent on being wrong.

But first, the mistake in the setup, because it is the one everybody is about to make.

**"Replicate this paper and don't stop" is the wrong instruction, and it is the one I gave.** It worked in the sense that it was followed. Three months, crashes recovered without me, never once needed a nudge. What I did not think about is what stopping is *for*. Stopping is when somebody asks where a number came from. Take the pauses out and you do not get a careless agent, you get a fast one whose results go straight into the next decision. Look back at the four failures above and none of them is a reasoning error. They are all numbers that got *used* before anyone questioned them. A baseline reused for ten weeks. A length limit trusted for eleven. A judge scoring zero for five hours while the run was being described to me as encouraging. Not one of those needed a smarter model. Each needed a pause.

So the question is not whether it can do the research. Mostly it can. The question is what has to be true before you are allowed to use what it hands you. Here is what I would gate:

**1. Diff your config against the paper's table before you launch, in code, and refuse to start on a mismatch.** This is the big one. It takes under a second, it needs no GPU, and it would have caught five separate fidelity bugs here, including ten box-days at double the paper's batch size and a first day where the agent under test was crippled by a setting its own comment told us to change. Every single one was an inherited default that looked deliberate. Write the paper's numbers in a file, compare, fail loud, and list your intentional differences with a reason so a silent drift cannot pretend to be one.

**2. Never let a missing measurement become a number.** Eight places in this codebase quietly did that. Once a missing value gets a plausible stand-in, nothing downstream can tell it apart from a real one, and it does not error, ever. Carry a flag that says "not measured" all the way to the gradient. And make your error markers say *why* they fired, because a marker that only says something went wrong tells you nothing at 3am.

**3. Print the number that should never change.** Positions measured. Tasks drawn. Coercion rate. The single most valuable thing produced in three months was one print statement of this kind, and it is what caught the eleven-week bug. Here is the trick to it: a check that asserts a symptom is *absent* will be satisfied by your next bug. A line that reports a *quantity* cannot be. Print quantities. It will look pointless the day you write it. That is what makes it worth writing.

**4. Measure your baseline in the same week as the thing you are comparing it to.** Not the same month. I took one baseline in May against a hosted API and used it as the reference all summer. Remeasured, it had moved nearly six points. Everything I had "improved" was partly a measurement of someone else's server. Three seeds and two frameworks all agreed with each other and that felt like proof. It was not, because all three were subtracting the same wrong number.

**5. Write down what your test can actually detect, before you run it.** One line of arithmetic, no GPU. Mine says the standard 140-game benchmark can only resolve effects of about 13 points, and the effects this field publishes are about 13 points. Most of the compute I wasted went on chasing differences my instrument could not have seen.

**6. If a fix does not move the number, that is information about your diagnosis.** It is not a reason to stack another fix on top. A crash asked for 4.6GB, we freed 8GB, the next crash asked for 14.2GB, and that number was computable before launching anything. Two launches to find out we had been aiming at the error message instead of the arithmetic.

And one more, which is operational rather than scientific: **watch the box, not the job.** Around 11.8 days of an eight-GPU box sat idle in this project. Almost none of that was crashes. It was the run finishing or dying while the thing supposed to notice had itself stopped. Idle GPUs are invisible to the only metric anyone checks, which is whether the run is up.

Add it up and the shape is not what I expected going in. I assumed the enemy would be crashes and bad ideas. The actual cost was **runs that started, looked healthy, produced numbers, and turned out to have been training against something wrong**, plus a box doing nothing while I thought it was busy. At least 30 box-days out of thirteen weeks, and eleven of those thirteen weeks trained on three tenths of the protocol without a single error message. Nothing on that list is a hard problem. Half of it is arithmetic you can do on your phone before you launch.

## Why I spent ten weeks on this

Disclosure of interest, since it shaped which claim I chased: I build [inference.sh](https://inference.sh), and this paper argues my bet.

Right now almost everyone routes everything to whichever frontier model is best this month. It works, and it is a strange equilibrium. You pay reasoning-model prices to reformat a JSON blob, and your system gets no better at *your* problem for having done it a thousand times. Every request starts from zero.

The measurement above is the alternative in miniature: a cheap specialised model beat a frontier model at a real job, at roughly one eightieth of the cost, and on the weak agent the frontier model added nothing. That is what I want to be easy to build. Agents that accumulate something from being used, so your ten-thousandth request is better and cheaper than your first, and you spend frontier money only where frontier reasoning is genuinely the bottleneck.

I would rather this be read as evidence than as a pitch, so the honest version of my own bet includes the caveats above. The effect is real but smaller than published. The wrong checkpoint of the same recipe is *worse* than just paying for Gemini. And the single most valuable hour of the project was the one where I found the bug that had manufactured my best result. If you build on self-improving agents, budget for that hour. It is the entire difference between a system that gets better and a system that reports that it is getting better.

## What I am releasing

- **[Every evaluation rollout](https://huggingface.co/datasets/inference-sh/skillos-alfworld-eval-arms)**, per-task outcomes for every arm. About 12MB. You can recompute every significance test in this article on a laptop, no GPU, no API key. Start here if you are sceptical, which you should be.
- **[All 12 checkpoints from the verl/GiGPO run](https://huggingface.co/inference-sh/skillos-curator-qwen3-8b-verl-gigpo)**. The entire curve, not just the good bit. The curve is the finding.
- **[Checkpoints across 3 seeds](https://huggingface.co/inference-sh/skillos-curator-qwen3-8b-trl-fft)**, including the one that beat the paper's headline number on a larger agent.
- **[Training and evaluation code](https://github.com/belt-sh/skillos)**, plus the full report, a dated engineering journal with the dead ends in it, and a `DIVERGENCES.md` listing every place I depart from the paper, including my own mistakes and the bug above.

The thing I would most like someone to do is take the checkpoint that transferred best and try it against an agent I never touched. If curator quality really is this decorrelated from the agent it trained against, that is a bigger deal than any number here.

## Where I landed

SkillOS is a good idea and I would bet on the direction. Trained curation beats retrieval-and-hope. Readable memory earns its keep in debuggability alone. And a cheap specialist beating a frontier model at a real task is the most encouraging result I have measured this year.

What I would push back on is the reliability the numbers imply. Fifty checkpoints in, I cannot produce one same-agent improvement that survives correction, the best moment moves with the seed, and the last checkpoint is never the good one. That is not a debunking. It is the difference between "this works" and "this works reliably enough to build on," and papers are not currently formatted to tell those apart.

If the industry is going to spend billions automating research, some of that capacity should go to *checking* research. It is cheaper, it has a ground truth, and on this evidence it is the thing autonomous agents are readiest to do. My agent found a real bug in its own work. That is the job.

The weights and code are up. The rollouts are 12MB. Go and disagree with me.

---

*Full report: [`docs/repro_report.md`](https://github.com/belt-sh/skillos/blob/main/docs/repro_report.md). Every deviation from the paper, including mine: [`DIVERGENCES.md`](https://github.com/belt-sh/skillos/blob/main/DIVERGENCES.md). The unedited engineering log: [`JOURNAL.md`](https://github.com/belt-sh/skillos/blob/main/JOURNAL.md).*

*Independent reproduction, not affiliated with or endorsed by the paper's authors.*
