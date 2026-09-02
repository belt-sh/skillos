# Every message from the human

935 messages across a three-month ML reproduction project (May-Sep 2026).
One human directing one LLM agent on 8xH100.

Credentials redacted. Tool output stripped. Typos preserved.

---


## 2026-05-20

> ok try running existing qwen training now on our 8xh100, we still use remote for 32b etc stuff pls. heres wandb setup api keymachine api.wandb.ai
  login user
  password [REDACTED] infsh api key is here 
  Earlier in this session you pasted a [REDACTED]... key in chat — recommend rotating that one at
  https://wandb.ai/authorize if you haven't already, since chat transcripts persist.



      ls -la ~/.inferencesh/ 2>/dev/null…)


  1. explicit arg → no add this to env belt is already legged in wndb not yet

> cant we do rollouts parallel

> remote llms can handle any load

> dont go over paper

> eta?

> cost estimate?

> wouldd jusdge and executor fit in rest of h100 with vllm for free or would we be pushig it

> how did paper fit

> check ./tmp

> how did paper fit vllm and training

> i think what it means is by offloading to remote we can pull the same off with 8 gpus noiec

> why an 18-min gap.

> less monitorin pls this will take hors

> cant you still monitor bu t less oftehn like every 15-30 min

> how are we looking

> belt task cost without --json check one show me

> did you look at any gamees do they make sense

> - 14.5% success rate (executor solves the task) — par for ALFWorld zero-shot, paper baseline is ~48% no-memory do we have something worng

> if we would restart could we start from checkpoint or something

> i'll decide in the morning approx 10 hrs later maybe its finished even like this

> check monitors


## 2026-05-21

> bro what happened? why out of memory because saving ?

> but why other steps didnt do it ?

> we need to debug this asap. strip the parts that take time like judges etc you can feed existing tuff from stock to figure out why this happens ther emust be a leak

> why you said oom

> how to secure and retry

> rllouts if cheap and easy to save do it. please thigten crash/pause/resume and test then we start from checkpoint again

> is this all tested resume works/

> commit pls

> what is 32b used for

> lets turn off thinking for 32b

> real example

> what was our memory consumption do you have any idea during the actual run

> i fixed reasoning

> commit them and start the run

> what did it retry can you give me a task id so i can check transient issues

> btw i fixed reasoning qwen 32b checks now run in 3-4 seconds

> can you also check costs should be signifcantly lower show me per and total training estimation

> when will we see any improvement. is there a way to run a faster experiment to se if we can really improve without waiting the full 22 hours dicuss

> is there a bug or is this model just stupid can you check if these examine for real ans udeful? 
## Current Progress
Prior to this step, you have already taken 7 step(s). Below are the most recent 3 observations and the corresponding actions you took: ACTION: examine sinkbasin 1
OBSERVATION: On the sinkbasin 1, you see a sink 1.
ACTION: examine sinkbasin 1
OBSERVATION: On the sinkbasin 1, you see a sink 1.
ACTION: examine sinkbasin 1
OBSERVATION: On the sinkbasin 1, you see a sink 1.

You are now at step 8 and your current observation is: On the sinkbasin 1, you see a sink 1.

Your admissible actions of the current situation are: examine sinkbasin 1, go to bathtubbasin 1, go to countertop 1, go to countertop 2, go to drawer 1, go to drawer 2, go to drawer 3, go to drawer 4, go to garbagecan 1, go to handtowelholder 1, go to toilet 1, go to toiletpaperhanger 1, go to towelholder 1, go to towelholder 2, help, inventory, look why gpu 7 arent we already using all gpus btw?

> and all this effort how much will qwen improve at alf?

> do they compare to other memory tools?

> so purposed small open model better at specialised task.

> 10% vs 13 pp

> btw did we increase max steps/turns from 10 when running this new loop ?

> how are you doing

> 35 is crazy long what happened

> he 529s growing toward 600s is our judge timeout boundary. When it crosses 600s, the patch will fire: [curator_env] judge
  call timed out after 600s; dropping its score. That'll unblock the affected rollout, the rank reaches the reward gather, step
   11 finally completes. explain this is it one stuck inference task?

> but you said we were close

> maybe we need to increase limits if with 200 skils its slow?

> just tell me odnt do anthing immedaitely. just tell me if it was a false alarm an dthe run werent at risk

> we are doing this with lorea paper does with FFT how can we oom here when they dont? lets try the alloc conf lets not cut max completion (follow paper as much as we can) document the env and maybe add a run.sh that confgures? also will wandb run resume i wouldnt want to start a new one midway?

> didnt i say launch it my man. i was at launch thinking you would already make progress

> we shul dsee oom immediately this step if fix didnt work yeah?

> 6 ranks? we have 8 gpus ?

> ok im slow

> hows it going buddy no oom?

> if loss can be negatove how do you tell if rl is going good

> live total rollouts reset when we crashed and reastateg won wandb. train reward seems to have crashed as well or is it normal to see jumps and falls

> is there a way to improve resumption?

> is it getting "better"

> best stretch rigth before crash ?

> how long?

> where ar we how long

> how is it going

> how did we get 12-23 much less than no memory baseline ?

> shouldnt have we establishedthe base line too :D

> thats actually a good idea can you run it speraetly


## 2026-05-22

> still bad scorea/?

> hello

> On your baseline question: Yes — I ran the no-memory baseline earlier. It completed: 20% on 30 valid_seen games. I have NOT
  run a trained-checkpoint eval yet. what this is not waht the paper says at all! this is very strange i wonder if our settings are diffrent for th elocal 8b or did you run with remote 8b

> recovered

> qbenchmark no memory what happened to iT?

> try

> they are running one by one no way to parallelize. is this exactly how paper benchmarks

> infsh contention? what?

> bro infsh has infinite resource.. anyway lets finish this training and then i want baseline eval parlalel like the paper you keep udnerestimating infsh capabilities

> sup

> you know what its not going to improve or change at this point. and we have checkpoints lets stop

> <command-message>simplify</command-message>
<command-name>/simplify</command-name>

> # Simplify: Code Review and Cleanup

Review all changed files for reuse, quality, and efficiency. Fix any issues found.

## Phase 1: Identify Changes

Run `git diff` (or `git diff HEAD` if there are staged changes) to see what changed. If there are no git changes, review the most recently modified files that the user mentioned or that you edited earlier in this conversation.

## Phase 2: Launch Three Review Agents in Parallel

Use the Agent tool to launch all three agents concurrently in a single message. Pass each agent the full diff so it has the complete context.

### Agent 1: Code Reuse Review

For each change:

1. **Search for existing utilities and helpers** that could replace newly written code. Look for similar patterns elsewhere in the codebase — common locations are utility directories, shared modules, and files adjacent to the changed ones.
2. **Flag any new function that duplicates existing functionality.** Suggest the existing function to use instead.
3. **Flag any inline logic that could use an existing utility** — hand-rolled string manipulation, manual path handling, custom environment checks, ad-hoc type guards, and similar patterns are common candidates.

### Agent 2: Code Quality Review

Review the same changes for hacky patterns:

1. **Redundant state**: state that duplicates existing state, cached values that could be derived, observers/effects that could be direct calls
2. **Parameter sprawl**: adding new parameters to a function instead of generalizing or restructuring existing ones
3. **Copy-paste with slight variation**: near-duplicate code blocks that should be unified with a shared abstraction
4. **Leaky abstractions**: exposing internal details that should be encapsulated, or breaking existing abstraction boundaries
5. **Stringly-typed code**: using raw strings where constants, enums (string unions), or branded types already exist in the codebase
6. **Unnecessary JSX nesting**: wrapper Boxes/elements that add no layout value — check if inner component props (flexShrink, alignItems, etc.) already provide the needed behavior
7. **Nested conditionals**: ternary chains (`a ? x : b ? y : ...`), nested if/else, or nested switch 3+ levels deep — flatten with early returns, guard clauses, a lookup table, or an if/else-if cascade
8. **Unnecessary comments**: comments explaining WHAT the code does (well-named identifiers already do that), narrating the change, or referencing the task/caller — delete; keep only non-obvious WHY (hidden constraints, subtle invariants, workarounds)

### Agent 3: Efficiency Review

Review the same changes for efficiency:

1. **Unnecessary work**: redundant computations, repeated file reads, duplicate network/API calls, N+1 patterns
2. **Missed concurrency**: independent operations run sequentially when they could run in parallel
3. **Hot-path bloat**: new blocking work added to startup or per-request/per-render hot paths
4. **Recurring no-op updates**: state/store updates inside polling loops, intervals, or event handlers that fire unconditionally — add a change-detection guard so downstream consumers aren't notified when nothing changed. Also: if a wrapper function takes an updater/reducer callback, verify it honors same-reference returns (or whatever the "no change" signal is) — otherwise callers' early-return no-ops are silently defeated
5. **Unnecessary existence checks**: pre-checking file/resource existence before operating (TOCTOU anti-pattern) — operate directly and handle the error
6. **Memory**: unbounded data structures, missing cleanup, event listener leaks
7. **Overly broad operations**: reading entire files when only a portion is needed, loading all items when filtering for one

## Phase 3: Fix Issues

Wait for all three agents to complete. Aggregate their findings and fix each issue directly. If a finding is a false positive or not worth addressing, note it and move on — do not argue with the finding, just skip it.

When done, briefly summarize what was fixed (or confirm the code was already clean).

> what shells are running

> didnt we stop real training?

> how long until evals complete. how do they look

> btw if you want to save this to the tmp folder md file or maybe even in the repo as a copy of paper to reference maybe would be nice: arXiv logoBack to arXiv
Why HTML?
Report Issue
Back to Abstract
Download PDF
 
Abstract
1 Introduction
2 Related Work
3 Methodology
4 Experiments
5 Analysis
6 Conclusion
7 Acknowledgments
References
A Prompts
B Implementation Details
C Additional Analyses
D Limitations
E Future Research Directions
F Use of LLMs
License: arXiv.org perpetual non-exclusive license
arXiv:2605.06614v1 [cs.AI] 07 May 2026
\pdftrailerid
redacted\correspondingauthorsiruo2@illinois.edu, {junyann, chenyulee}@google.com

SkillOS: Learning Skill Curation for Self-Evolving Agents

Siru Ouyang
Jun Yan
Yanfei Chen
Google Cloud AI Research
Rujun Han
Google Cloud AI Research
Zifeng Wang
Google Cloud AI Research
Bhavana Dalvi Mishra
Google Cloud AI Research
Rui Meng
Google Cloud AI Research
Chun-Liang Li
Google Cloud AI Research
Yizhu Jiao
University of Illinois Urbana-Champaign
Kaiwen Zha
Massachusetts Institute of Technology
Maohao Shen
Massachusetts Institute of Technology
Vishy Tirumalashetty
Google Cloud AI Research
George Lee
Google Cloud AI Research
Jiawei Han
University of Illinois Urbana-Champaign
Tomas Pfister
Google Cloud AI Research
Chen-Yu Lee
Abstract

LLM-based agents are increasingly deployed to handle streaming tasks, yet they often remain one-off problem solvers that fail to learn from past interactions. Reusable skills distilled from experience provide a natural substrate for self-evolution, where high-quality skill curation serves as the key bottleneck. Existing approaches either rely on manual skill curation, prescribe heuristic skill operations, or train for short-horizon skill adaptation, but still struggle to learn complex long-term curation policies from indirect and delayed feedback. We propose SkillOS, an experience-driven RL training recipe for learning skill curation in self-evolving agents. SkillOS pairs a frozen agent executor that retrieves and applies skills with a trainable skill curator that updates an external SkillRepo from accumulated experience. To provide learning signals for curation, we train on grouped task streams based on skill-relevant task dependencies, where earlier trajectories update the SkillRepo, and later related tasks evaluate these updates. We further design composite rewards to better attribute downstream executor feedback to curation decisions. Across multi-turn agentic tasks and single-turn reasoning tasks, SkillOS consistently outperforms memory-free and strong memory-based baselines in both effectiveness and efficiency, with the learned skill curator generalizing across different executor backbones and task domains. Further analyses show that the learned curator produces more targeted skill use, while the evolving SkillRepo develops richer internal structure and higher-level meta-skills over time.

1 Introduction

LLM-based agents (DBLP:journals/fcsc/WangMFZYZCTCLZWW24) are increasingly deployed in real-world scenarios, where they must move beyond instantaneous problem-solving toward long-term proficiency (he2026memoryarena). However, the prevailing paradigm of “one-off” task execution limits their utility in streaming settings, where tasks unfold sequentially over time. This makes self-evolution (fang2025comprehensive; gao2025survey) essential: capable agents should not repeatedly start from scratch, but instead continually accumulate, refine, and reuse experience for future tasks.

A key substrate for self-evolution is procedural memory (hu2025memory; wu2025human; DBLP:journals/corr/abs-2508-06433), specifically, reusable skills (anthropic_skills_2025; wang2025inducing) accumulated from past interactions. In real-world streaming settings (wu2024streambench), a skill-based self-evolving agent typically follows a closed-loop workflow: for each new task, it selects relevant skills, uses them to guide execution, and updates its skill collection based on the resulting trajectory. This makes skill curation—the extraction of high-quality lessons and their integration into the skill collection—essential for self-evolving agents.

However, existing skill curation works remain limited. Manually curated skills, such as Anthropic’s skills repository (anthropic_skills_2025), demand huge human expertise and cannot scale to the diversity of tasks that agents may encounter. Prompting or heuristic-based methods that dictate memory operations (xu2025amem; qiu2025alita; DBLP:journals/corr/abs-2504-07079) rely on fixed rules and lack downstream performance feedback, preventing them from adapting to the executor’s actual needs. Recent studies explored reinforcement learning (RL) to optimize skill-based agent systems. However, they either focus on teaching agents to use skills (xia2026skillrl; tu2026dynamic) or optimize skill operations within a short task stream (DBLP:journals/corr/abs-2512-17102; DBLP:journals/corr/abs-2602-10652). This limits the density of learning signals available for curating highly reusable skills and mastering complex management operations such as skill update and deletion, which are essential for robust and scalable long-term self-evolution.

Refer to caption
Figure 1: SkillOS pairs a frozen Agent Executor with a trainable Skill Curator. The executor retrieves relevant skills from SkillRepo to act; the curator edits the repo (insert/update/delete) based on the resulting experiences, with Markdown as the skill format.
To tackle this challenge, we propose SkillOS, an experience-driven RL training recipe to learn the capability of skill curation for self-evolving agents. We study skill curation in a modular multi-agent framework in a streaming setting, where a frozen agent executor solves tasks with a skill collection (termed SkillRepo), while a trainable skill curator updates and manages this collection through function calls (Figure 1(a)). We represent skills as Markdown files (anthropic_skills_2025) (Figure 1(b)) managed via file I/O operations similar to an operating system (OS). Our recipe features two core designs. First, we construct each training instance as a group of related tasks. By mimicking test-time streaming settings, it grounds skill curation in long-term utility: skills induced from earlier experiences are evaluated by their ability to improve later related tasks. Second, we design rewards to better attribute environmental feedback to curation decisions, combining task performance with signals for valid function calls, skill quality, and SkillRepo’s compactness. Together, these designs turn delayed and indirect supervision into learning signals for skill curation.

We evaluate SkillOS on both multi-turn agentic tasks and single-turn reasoning tasks. Experiments show that SkillOS consistently outperforms memory-free and strong memory-based methods in both effectiveness and efficiency, with up to 
+
9.8
%
 relative performance improvement and 
−
6.0
%
 fewer interaction steps compared to the strongest baseline (Table 1). Our trained skill curator generalizes well across executors and tasks, improving performance even with the Gemini-2.5-Pro executor. Notably, our 8B curator also outperforms Gemini-2.5-Pro when used directly as the curator. Beyond performance gains, our analyses further show that the learned skill curator leads to more targeted and effective skill utilization, while the skills in SkillRepo evolve into more richly structured Markdown files that encode higher-level meta-skills over time. Together, we establish SkillOS as a practical, modular, and experience-driven RL training recipe for building self-evolving agents.

2 Related Work

Memory for Self-Evolving Agents. Learning from past experiences as procedural memory (wu2025human; wei2025evo; shen2026decocted; hu2025memory; huang2026rethinking; zhang2024working) is a central mechanism for developing self-evolving agents (gao2025survey; fang2025comprehensive). The central challenge is to encode interaction histories into reusable and retrievable representations. Case-based representations are the most concrete form in this research line: they store experiences in minimally processed formats, allowing past histories to be replayed directly or reused as in-context exemplars, such as raw trajectories (zheng2023synapse; DBLP:journals/corr/abs-2508-16153; wu2025comemagent) and abstracted query–response pairs (zhao2024expel; islam-etal-2024-mapcoder). Another line of work abstracts experiences into higher-level knowledge that is editable, auditable, and composable, reducing reliance on long trajectory replay and improving both cross-task generalization and efficiency. Such strategy-based memory typically consists of reusable workflows (wang2025agent; DBLP:journals/corr/abs-2507-06229), distilled insights (ouyang2026reasoningbank; huang-etal-2025-r2d2; DBLP:journals/corr/abs-2509-04439), and recurring patterns (yang2024buffer; kim-etal-2025-principles). Recently, skills (wang2025inducing; kuroki2025agent; DBLP:journals/corr/abs-2602-08004; DBLP:journals/corr/abs-2602-12670; DBLP:journals/corr/abs-2602-02474; [REDACTED]; alzubi2026evoskill; liang2026skillnet) have emerged as a new agent-native form of memory and an orchestrable capability layer, owing to their modularity and ease of customization. Anthropic conceptualizes each skill as a folder containing instructions, scripts, and supporting resources (anthropic_agent_skills_overview), which has become the most widely adopted design in the current community. Our work follows this design philosophy, simplifying the setting for research purposes by representing each skill as a single Markdown file.

Learning Memory and Skill Curation with RL. Training LLM-based agent systems with memory capabilities using RL has become a growing research direction. One research line targets training for long-context management with predefined operations such as compaction (zhou2026mem; yu2026memagent; wang2025mem). Another interesting area focuses more on memory utilization and management by learning additional memory tool-calls (DBLP:journals/corr/abs-2508-19828; DBLP:journals/corr/abs-2508-16629; DBLP:journals/corr/abs-2510-12635) or training policies for different stages, such as memory retrieval (zhang2026memrl). More recently, RL has been applied at various stages of agent skill development. Specifically, SkillRL (xia2026skillrl) and D2Skill (tu2026dynamic) teach smaller models to use skills curated from powerful LLMs in an iterative manner. ARISE (Li2026ARISEAR) trains a shared policy operating both as skill retriever and worker, with heuristics for skill management. Recent studies have begun to train agents for memory or skill curation (DBLP:journals/corr/abs-2512-17102; DBLP:journals/corr/abs-2602-10652), but their supervision is mostly restricted to local adaptation within short task streams. This favors immediately useful operations such as skill insertion, while offering limited signal for complex management operations, such as revising outdated skills and deleting harmful ones. SkillOS instead formulates skill curation as a long-horizon, executor-grounded learning problem. We group related tasks into training instances and combine downstream task outcomes with intermediate rewards, turning delayed and indirect feedback into learning signals for skill curation.

3 Methodology

In this section, we first formalize the problem setting and introduce the multi-agent modular design of SkillOS. We then detail the RL training recipe designed specifically for training the skill curator.

3.1 Streaming Skill Curation with Multi-Agent Modular Design

We consider a streaming test-time setting (wu2024streambench), where an LLM-based agent is deployed to solve a sequence of tasks 
𝒟
=
{
x
1
,
x
2
,
…
,
x
T
}
 that arrive over time. At each time stamp 
t
, the agent must solve the current task 
x
t
 before observing future tasks, producing an execution trajectory 
ξ
t
=
{
o
1
,
a
1
,
…
,
o
n
,
a
n
}
, where 
o
 and 
a
 denote observations and actions, respectively. This setting naturally captures the challenge of self-evolving agents, where the system must distill useful experience from the trajectories of past interactions to improve performance on future tasks, and become more capable over time. Figure 1(a) presents an overview of the system.

Skill Repository. We maintain an external skill repository 
𝒮
t
 at time stamp 
t
, which consists of 
N
t
 reusable skills
𝒮
t
=
{
s
t
1
,
s
t
2
,
…
,
s
t
N
t
}
. Following the widely adopted SKILL.md format (anthropic_skills_2025), each skill is represented as a single Markdown file with two components as shown in Figure 1(b): (i) YAML frontmatter, which specifies the skill name and a natural-language description of when the skill should be used, and (ii) Markdown instructions, which describe the executable knowledge, workflows, constraints, and reusable heuristics captured by the skill.

Agent Executor. Given a task 
x
t
, a frozen agent executor 
π
ℒ
 solves the task conditioning on the current environment observation and relevant skills. Specifically, we retrieve a subset of skills 
𝒮
~
t
⊆
𝒮
t
 using BM25 (robertson2009probabilistic) for each task 
x
t
, and the executor samples actions following 
a
∼
π
ℒ
(
⋅
∣
x
t
,
o
t
,
𝒮
~
t
)
.

Skill Curator. After the executor completes task 
x
t
, the skill curator 
π
𝒮
 observes the trajectory 
ξ
t
, the self-judged correctness of the answers/interactions 
𝟙
ξ
t
, and a retrieved subset of related skills 
𝒮
~
t
. It then generates a sequence of structured curation operations 
c
t
=
(
u
t
1
,
…
,
u
t
M
t
)
∼
π
𝒮
(
⋅
∣
ξ
t
,
𝟙
ξ
t
,
𝒮
~
t
)
, where each operation 
u
t
m
 is one of
{
insert_skill
,
update_skill
,
delete_skill
}
. Each operation is implemented as a function call (detailed signature in Figure 8) that manipulates the skill repository 
𝒮
t
. Applying these operations transforms the repository from 
𝒮
t
 to 
𝒮
t
+
1
as 
𝒮
t
+
1
=
ApplyOps
​
(
𝒮
t
,
c
t
)
. The updated repository is then used by the executor on subsequent tasks, forming a closed loop between task execution and experience-driven skill evolution.

3.2 Learning Skill Curation with RL

We optimize the skill curator 
π
𝒮
 with RL and keep the agent executor 
π
ℒ
 frozen. The main challenge is indirect and delayed feedback for curation decisions, which is only revealed through 
π
ℒ
’s performance on future relevant tasks. We address this by constructing grouped training instances (§ 3.2.1) and designing a composite reward (§ 3.2.2) that combines future task outcomes with intermediate signals on operation validity, skill quality, and the conciseness of skills. An overview of the training process is shown in Figure 2.

Refer to caption
Figure 2: SkillOS training pipeline. Each training step samples a group of related tasks and initializes an empty SkillRepo. 
π
𝒮
 is optimized with composite rewards, enabling self-evolution.
3.2.1 Training Instance Construction

To provide downstream learning signals for skill curation, we construct each training instance as a group of related tasks that are solved sequentially. Within each group, SkillRepo is updated by the curator 
π
𝓈
 after each task, allowing skills derived from earlier experiences to be evaluated by whether they help solve related future tasks. This also differs from prior work that focuses on short-horizon transfer (DBLP:journals/corr/abs-2512-17102; DBLP:journals/corr/abs-2602-10652), where our grouped formulation exposes the curator to longer skill-evolution trajectories and provides denser feedback for learning complex curation operations.

Concretely, for each task 
x
i
 in 
𝒟
=
{
x
i
}
i
=
1
N
, we first annotate each instance with a set of skill-relevant attributes. Formally, for each 
x
i
, we use Gemini-2.5-Pro (DBLP:journals/corr/abs-2507-06261) to produce a set of tags:

Z
i
=
{
z
i
1
,
z
i
2
,
…
,
z
i
|
Z
i
|
}
,
where each attribute 
z
i
 captures a salient aspect of the task 
x
i
, such as topic and common pitfalls. For example, in mathematical reasoning, attributes may include labels such as “algebra” or “Fourier transformation”. These attributes serve as proxies for task-relatedness and potential skill dependency.

Based on the annotated attributes, we then partition 
𝒟
 into a collection of 
M
 task groups using the similarity of attributes of these data samples:

𝒟
=
{
G
1
,
G
2
,
…
,
G
M
}
,
G
m
=
{
x
m
,
1
,
x
m
,
2
,
…
,
x
m
,
|
G
m
|
}
,
where all instances within the same group 
G
m
 exhibit non-trivial dependency in terms of required skills. Detailed description of data processing and grouping algorithms can be found in Appendix B.2.

3.2.2 Training Loop and Policy Optimization

We employ Grouped Reward Policy Optimization (GRPO DBLP:journals/corr/abs-2402-03300) for its training stability and sample efficiency. The training loop shown in Algorithm 1 optimizes the skill curator policy 
π
𝒮
 to maximize a composite reward function over the distribution of generated traces. For a task group 
G
=
(
x
1
,
…
,
x
|
G
|
)
, the curator produces a sequence of curation decisions 
c
=
(
c
1
,
…
,
c
|
G
|
)
 as the executor proceeds through the group. Each training step, the reward combines four signals:

r
=
r
task
⏟
task outcome
+
λ
f
​
r
fc
⏟
function call
+
λ
u
​
r
cnt
⏟
content quality
+
λ
c
​
r
comp
⏟
compression
(1)
Task outcome reward.  The first task uses an empty SkillRepo, before any curator update occurs. We thus define the task outcome reward as the average success over the remaining tasks as 
r
task
=
1
|
G
|
−
1
​
∑
i
=
2
|
G
|
𝟙
​
(
ξ
i
)
, which provides executor-grounded signal on downstream performance achieved by the evolving SkillRepo from 
π
𝒮
.

Function call reward.  The function call reward measures whether the curator produces valid skill operations. For each curation decision 
c
i
, let 
Valid
​
(
c
i
)
 be the fraction of generated function calls that are valid and successfully executed. We define the function call reward as 
r
fc
=
1
|
G
|
​
∑
i
=
1
|
G
|
Valid
​
(
c
i
)
.

Algorithm 1 Training Skill Curator with Task Groups using GRPO
1:for each training step do
2:  
G
=
(
x
1
,
…
,
x
|
G
|
)
, 
𝒮
←
∅
⊳
 Sample a task group and initialize SkillRepo
3:  for task index 
i
=
1
,
…
,
|
G
|
 do
4:   
𝒮
~
←
BM25
​
(
x
i
,
𝒮
)
⊳
 Retrieve relevant skills
5:   
ξ
i
←
RunTask
​
(
𝒮
~
,
π
ℒ
,
x
i
)
⊳
 Run inference on frozen executor
6:   
c
i
∼
π
𝒮
(
⋅
|
ξ
i
,
𝒮
~
)
⊳
 Sample a rollout from skill curator
7:   
𝒮
←
ApplyOps
​
(
𝒮
,
c
i
)
⊳
 Apply insert/update/delete
8:  end for
9:  
r
←
CalculateReward
​
(
ξ
,
c
)
10:  
Update
​
π
𝒮
⊳
 Update skill curator using GRPO
11:end for
Compression reward.  To discourage verbatim trajectory copying, we reward concise repository updates. Let 
𝒮
i
 denote the skill repository after applying 
c
i
, and let 
χ
i
 denote the curator input context at position 
i
. We define
r
comp
=
1
|
G
|
​
∑
i
=
1
|
G
|
(
1
−
|
𝒮
i
|
|
χ
i
|
)
, where 
|
𝒮
i
|
 and 
|
χ
i
|
 denote token lengths. This encourages the curator to distill reusable skills rather than store raw trajectories.

Content quality reward.  The content quality reward evaluates whether the curated skills are semantically meaningful and likely to be useful for future tasks. Let 
Judge
​
(
c
i
)
 denote the scalar score assigned by an external judge (Qwen3-32B) 
c
i
, we compute the reward as 
r
cnt
=
1
|
G
|
​
∑
i
=
1
|
G
|
Judge
​
(
c
i
)
.

For each task group 
G
, we sample 
N
 independent rollouts of the entire curation sequence from 
π
𝒮
. Within each rollout, the executor produces trajectory 
ξ
i
 using the skill repository 
𝒮
i
 resulting from previous curations 
c
<
i
 till task position 
i
 with the same training task group, so different rollouts evolve different repository histories. The GRPO advantage is computed as: 
A
n
=
r
n
−
1
N
​
∑
n
′
=
1
N
r
n
′
,
 where 
r
n
 is the composite reward (Eq. 1) for the 
n
-th rollout. We optimize 
π
𝒮
 with a clipped surrogate objective over all curation steps 
i
=
1
,
…
,
|
G
|
:

ℒ
=
𝔼
n
​
[
min
⁡
(
ρ
n
​
A
n
,
clip
​
(
ρ
n
,
 1
−
ϵ
,
 1
+
ϵ
)
​
A
n
)
]
(2)
where 
ρ
n
=
π
𝒮
​
(
c
n
∣
χ
)
/
π
θ
o
​
l
​
d
​
(
c
n
∣
χ
)
 is the importance ratio. The advantage 
A
n
 is assigned uniformly to all tokens in
c
n
, and we discard the KL term in GRPO to encourage policy exploration.

4 Experiments

We conduct experiments on both multi-turn agentic tasks and single-turn reasoning tasks, in line with prior work (xia2026skillrl; wei2025evo; DBLP:journals/corr/abs-2602-10652). We additionally show that the trained skill curator transfers across agent executors and task domains, highlighting its flexibility and generalizability.

Table 1: Experiment results on ALFWorld benchmark. Success rate (SR 
↑
) and the number of steps (Steps 
↓
) are reported on 6 subsets with 3 different frozen executors.
Methods    Curator    Pick    Look    Clean    Heat    Cool    Pick2    Avg. SR    Steps
π
𝒮
(35)    (13)    (27)    (16)    (25)    (24)    (140)
Executor 
π
ℒ
: Qwen3-8B
No Memory    None    
78.1
 1.6
46.2
 7.7
33.3
 13.4
37.5
 10.8
29.3
 6.1
47.2
 6.4
47.9
 1.2
21.1
ReasoningBank    [Uncaptioned image] Qwen3-8B    
83.8
 0.0
48.7
 7.2
49.4
 16.2
39.6
 4.4
41.3
 8.5
54.2
 8.8
55.7
 3.1
20.1
MemP    [Uncaptioned image] Qwen3-8B    
80.0
 5.7
43.6
 4.4
24.7
 4.3
33.3
 3.6
38.7
 6.1
48.6
 6.4
49.7
 0.7
21.0
SkillOS-base     [Uncaptioned image] Qwen3-8B    
79.0
 8.7
41.0
 4.4
45.7
 4.3
37.5
 9.5
38.7
 4.0
55.6
 2.1
53.1
 2.5
20.4
SkillOS-gemini     [Uncaptioned image] Gemini-2.5-Pro    
77.1
 6.0
53.8
 6.1
37.0
 6.4
37.5
 9.5
36.0
 3.2
50.0
 6.7
50.7
 3.6
20.8
SkillOS    [Uncaptioned image] Qwen3-8B    
85.7
 3.3
56.4
 7.7
54.3
 8.6
43.8
 9.5
46.7
 2.3
62.5
 6.4
61.2
 4.6
18.9
Executor 
π
ℒ
: Qwen3-32B
No Memory    None    
80.0
 2.9
69.2
 0.0
45.6
 7.7
37.5
 16.5
42.7
 6.1
43.1
 2.4
54.5
 2.5
20.3
ReasoningBank    [Uncaptioned image] Qwen3-8B    
86.7
 3.0
71.8
 5.4
50.6
 6.3
45.8
 13.3
52.0
 8.9
51.4
 5.1
61.4
 2.5
18.7
MemP    [Uncaptioned image] Qwen3-8B    
80.0
 2.9
76.9
 0.0
44.4
 7.4
37.5
 10.8
42.7
 2.3
47.2
 6.4
55.7
 3.7
20.0
SkillOS-base     [Uncaptioned image] Qwen3-8B    
82.9
 2.9
69.2
 11.8
48.1
 2.1
50.0
 9.7
48.0
 14.4
52.8
 11.0
59.8
 3.0
19.2
SkillOS-gemini     [Uncaptioned image] Gemini-2.5-Pro    
97.1
 3.0
76.9
 5.4
55.6
 6.0
43.8
 11.3
40.0
 5.7
54.2
 4.9
63.6
 4.2
18.1
SkillOS    [Uncaptioned image] Qwen3-8B    
91.4
 3.3
76.9
 4.4
59.3
 8.6
56.3
 12.5
57.3
 10.1
62.5
 4.2
68.6
 5.7
17.3
Executor 
π
ℒ
: Gemini-2.5-pro
No Memory    None    
90.5
 3.2
66.7
 5.1
48.1
 10.2
39.6
 17.1
68
 7.4
68.1
 3.8
66.4
 2.0
17.7
ReasoningBank    [Uncaptioned image] Qwen3-8B    
91.4
 3.4
61.5
 4.1
63.0
 9.3
39.6
 10.3
70.7
 3.2
76.4
 8.5
71.4
 2.9
16.0
MemP    [Uncaptioned image] Qwen3-8B    
95.2
 2.1
74.4
 6.8
61.7
 7.6
56.3
 12.4
76.0
 6.2
68.1
 8.5
74.3
 3.4
15.2
SkillOS-base     [Uncaptioned image] Qwen3-8B    
91.4
 1.6
69.2
 7.7
56.8
 5.7
54.2
 13.7
72.0
 4.0
66.7
 11.0
70.7
 3.0
16.3
SkillOS-gemini     [Uncaptioned image] Gemini-2.5-Pro    
94.3
 5.7
69.2
 0.0
77.8
 5.7
75.0
 16.5
80.0
 12.2
66.7
 2.4
79.3
 2.6
14.9
SkillOS    [Uncaptioned image] Qwen3-8B    
95.2
 2.9
71.8
 7.7
74.1
 13.0
72.9
 10.1
77.3
 6.1
77.8
 10.0
80.2
 3.1
14.8
4.1 Setup

We briefly discuss the experiment setup throughout this paper. Full description of datasets, implementations, baselines, and evaluations can be found in Appendix B.

Dataset. For agentic tasks, we conduct experiments on ALFWorld (shridhar2021alfworld) and WebShop (10.5555/3600270.3601778). ALFWorld is a text-based interactive environment aligned with the ALFRED embodied AI benchmark, where agents must complete household tasks through textual navigation and object manipulation. WebShop simulates an online shopping environment in which agents navigate a realistic web interface to identify and purchase products that satisfy user-specified requirements. For each benchmark, we train SkillOS on its training split where 
Z
i
 is the default task type annotations, and evaluate on the corresponding test set. In addition to agentic tasks, we also benchmark for single-turn reasoning tasks, including AIME24, AIME25, and GPQA-Diamond (rein2024gpqa). Training data are constructed from DeepMath-103k (he2026deepmathk), where we randomly sample a subset of 33,000 data points.

Evaluation Configurations. We evaluate all methods across two dimensions, effectiveness and efficiency. For effectiveness, we measure the success rate (SR) and accuracy for agentic tasks and reasoning tasks, respectively. For efficiency, we compute the number of execution steps per agentic task and the number of tokens per reasoning problem, respectively. We compare SkillOS with three categories of baselines: (i) a memory-free agent (No Memory); (ii) existing memory-based methods, including ReasoningBank (ouyang2026reasoningbank), which distills reusable insights from past experiences, and MemP (DBLP:journals/corr/abs-2508-06433), which induces procedural memory with advanced memory-management strategies; and (iii) internal variants of our framework, including SkillOS-base, which uses the initial skill curator without RL training, and SkillOS-gemini, which uses Gemini-2.5-Pro to directly perform skill curation instead of learning the curator with RL. All prompts used can be found in Appendix A.

Implementation Details. We use Qwen3-8B (DBLP:journals/corr/abs-2505-09388) as the base model for 
π
𝒮
. The frozen executor is also instantiated with Qwen3-8B during training. We train our model using GRPO with a learning rate 
1
×
10
−
6
, batch size 
32
, and group size 
8
. Training is conducted on 16 H100 GPUs using the verl framework (sheng2024hybridflow). The full training process takes approximately 3 days for ALFWorld, 2.5 days for reasoning tasks, and 5 days for WebShop. For testing, we additionally include Qwen3-32B, Gemini-2.5-Pro (DBLP:journals/corr/abs-2507-06261), and Gemini-3.1-Flash-Lite (Appendix C.1) as executors to evaluate the generalization of SkillOS under different executor scales and architectures. Task outcome signal 
𝟙
ξ
t
 is obtained via LLM-as-a-judge with the frozen agent executor (prompt shown in Appendix A). We use ReAct (DBLP:conf/iclr/YaoZYDSN023) for agent execution and CoT (DBLP:conf/nips/Wei0SBIXCLZ22) for reasoning tasks. For the reward function, we set 
λ
f
=
1.0
, 
λ
u
=
0.1
, and 
λ
c
=
0.05
. We report averaged performance and standard deviation over 3 runs.

Table 2: Experiment results on WebShop and single-turn reasoning tasks for 3 different frozen executors. For WebShop, the averaged score, success rate (SR 
↑
), and the number of steps (Steps 
↓
) are reported. For reasoning tasks, accuracy (Acc. 
↑
) is reported on three datasets.
Methods    Curator    WebShop    Reasoning
π
𝒮
Score    SR    Steps    AIME24    AIME25    GPQA    Avg. Acc
Executor 
π
ℒ
: Qwen3-8B
No Memory    None    
33.3
 0.7
9.8
 0.5
20.3    
76.0
 6.9
71.1
 10.7
61.8
 1.1
69.6
 4.7
ReasoningBank    
[Uncaptioned image]
 Qwen3-8B    
35.4
 1.1
11.4
 0.9
20.5    
75.4
 5.0
73.2
 10.8
60.3
 3.9
69.6
 2.5
MemP    
[Uncaptioned image]
 Qwen3-8B    
35.7
 0.9
12.0
 0.5
21.3    
75.6
 5.1
71.1
 5.1
60.6
 4.0
69.1
 4.0
SkillOS-base     
[Uncaptioned image]
 Qwen3-8B    
38.6
 0.9
13.6
 0.8
20.1    
75.6
 5.1
71.9
 6.9
59.3
 2.5
68.9
 2.6
SkillOS-gemini     
[Uncaptioned image]
 Gemini-2.5-pro    
38.1
 1.0
13.2
 0.9
19.6    
73.3
 1.3
71.3
 1.9
57.6
 2.8
67.4
 0.8
SkillOS    
[Uncaptioned image]
 Qwen3-8B    
40.6
 0.7
16.5
 0.7
19.4    
80.0
 3.3
76.7
 5.8
64.6
 1.3
73.8
 1.8
Executor 
π
ℒ
: Qwen3-32B
No Memory    None    
41.5
 0.5
12.2
 0.3
17.0    
81.4
 1.3
72.2
 3.8
68.4
 2.0
74.0
 1.9
ReasoningBank    
[Uncaptioned image]
 Qwen3-32B    
40.4
 0.8
11.2
 1.1
17.9    
81.1
 9.6
75.6
 5.9
66.9
 1.2
74.9
 2.2
MemP    
[Uncaptioned image]
 Qwen3-32B    
30.7
 0.7
10.1
 0.6
17.4    
82.2
 5.1
76.7
 0.0
66.5
 2.3
75.1
 2.1
SkillOS-base     
[Uncaptioned image]
 Qwen3-8B    
43.4
 0.8
12.3
 1.0
16.8    
80.0
 3.3
75.6
 10.2
67.7
 1.5
74.7
 3.3
SkillOS-gemini     
[Uncaptioned image]
 Gemini-2.5-pro    
45.2
 1.0
13.2
 1.1
16.6    
77.8
 6.7
74.4
 1.9
66.2
 0.6
73.2
 2.6
SkillOS    
[Uncaptioned image]
 Qwen3-8B    
49.2
 1.2
16.5
 0.6
15.9    
85.6
 1.9
81.1
 3.3
72.4
 3.0
79.7
 1.6
Executor 
π
ℒ
: Gemini-2.5-pro
No Memory    None    
48.6
 0.3
38.4
 0.5
19.5    
85.6
 1.9
80.0
 6.7
79.9
 1.5
81.8
 2.8
ReasoningBank    
[Uncaptioned image]
 Gemini-2.5-pro    
50.8
 1.5
40.2
 1.3
19.2    
85.6
 5.1
84.4
 6.7
80.4
 2.1
83.5
 2.1
MemP    
[Uncaptioned image]
 Gemini-2.5-pro    
51.3
 1.2
39.8
 1.0
19.4    
83.3
 6.9
76.7
 5.8
81.8
 3.4
80.6
 3.2
SkillOS-base     
[Uncaptioned image]
 Qwen3-8B    
52.8
 1.0
39.6
 0.8
19.0    
87.8
 3.3
83.3
 1.9
82.8
 2.7
84.6
 1.8
SkillOS-gemini     
[Uncaptioned image]
 Gemini-2.5-pro    
54.7
 1.0
41.0
 1.2
17.8    
90.0
 5.1
85.6
 7.7
80.7
 5.5
85.4
 3.5
SkillOS    
[Uncaptioned image]
 Qwen3-8B    
56.0
 0.7
41.3
 0.8
18.3    
92.2
 2.4
86.7
 3.5
86.8
 2.1
88.6
 1.5
4.2 Main Results

Tables 1 and 2 summarize the results for different benchmarks with Qwen3-8B as the skill curator on various agent executors. Based on the results, we have the following observations.

SkillOS achieves strong performance gains across benchmarks. Across all three benchmarks, SkillOS consistently outperforms both memory-free and memory-based baselines, showing that the gains come from learning to manage and evolve skills rather than from maintaining a static collection. On ALFWorld, SkillOS improves the average success rate from 55.7 to 61.2 over the strongest baseline ReasoningBank with Qwen3-8B as the executor; similar trends hold on WebShop and reasoning tasks. Strikingly, our RL-trained 8B curator even surpasses SkillOS-gemini, despite the latter using a far stronger frontier model as the curator, demonstrating that targeted training of a small curator can outweigh raw model scale. The benefits brought by RL training are also compounded with executor capacity, yielding 
+
9.5
 absolute improvement with Gemini-2.5-Pro versus 
+
7.9
 with Qwen3-8B for ALFworld, compared with SkillOS-base.

SkillOS is more efficient, requiring fewer interactions and lower execution cost. The gains of SkillOS are accompanied by better efficiency rather than longer trajectories. On ALFWorld, SkillOS reduces the average interaction steps by 
2.2
, 
3.0
, and 
3.1
 compared with “no memory” setting with 3 executors, consistently outperforming all memory-based baselines. This trend extends to WebShop, where SkillOS secures higher success rates with fewer environment interactions. These results indicate that the learned skill manager enables the executor to identify procedural shortcuts and bypass redundant exploration. Rather than relying on additional trial-and-error, SkillOS improves performance by distilling experience into direct, actionable expertise that simplifies task execution.

The gains differ between agentic and reasoning tasks, reflecting different forms of reusable skills. A notable trend is that the gains of SkillOS are generally larger on multi-turn agentic benchmarks than on single-turn reasoning tasks. We hypothesize that this difference arises from how reusable skills manifest across task types. Agentic tasks naturally expose procedural regularities, such as action ordering, exploration strategies, recovery behaviors, and environment-specific constraints, which can be repeatedly composed and refined across task streams. Reasoning tasks also benefit from skill curation, but their reusable knowledge often appears at a more abstract level, such as decomposition heuristics, constraint formulation, or verification patterns, rather than as directly reusable action procedures. As a result, SkillOS still improves reasoning performance, while the gains are typically smaller than those observed on agentic benchmarks. We provide a case study demonstrating skills curated for different tasks in Figure 17.

4.3 Generalization of SkillOS

Refer to caption
Figure 3: Cross-task generalization results of SkillOS with (a) Qwen3-8B, (b) Qwen3-32B, and (c) Gemini-2.5-Pro as frozen executors. We plot relative improvement with baselines from least to most.
SkillOS is transferable and remains effective for different agent executors. During training, we use Qwen3-8B as the executor. To test whether SkillOS brings improvement for executors that are not seen in training, we pair the trained skill curator with different executors. As shown in Table 1 and 2, SkillOS consistently improves a wide range of frozen executors across benchmarks, from open-source models (Qwen3-8B, Qwen3-32B) to frontier models (Gemini-2.5-Pro). On ALFWorld, it lifts the average success rate of Qwen3-8B from 47.9 to 61.2 and Gemini-2.5-Pro from 66.4 to 80.2, demonstrating compatibility with executors of varying capacity. Notably, using Gemini-2.5-Pro directly as the curator (SkillOS-gemini) underperforms our trained curator, especially when paired with the smaller Qwen3-8B executor. This highlights a curator-executor mismatch: stronger reasoning ability alone does not guarantee effective skill curation, as frontier-generated skills may be misaligned with the executor’s capacity or usage patterns. By contrast, SkillOS learns executor-grounded curation behaviors through RL, producing skills that better match the downstream agent.

SkillOS delivers consistent performance improvement when generalized to different task domains. Figure 3 shows that the skill curator learned by SkillOS transfers well across different tasks. While training and testing on the same task often gives the strongest gain, most off-diagonal entries still bring performance improvement over baselines, indicating that SkillOS captures reusable skills beyond task-specific heuristics. Specifically, skill curator
π
s
 learned from reasoning tasks transfer particularly well to the two agentic tasks, likely because they contain more abstract and high-level strategies, such as decomposition, verification, and adaptive planning, which are broadly useful across settings. In contrast, skills learned from WebShop or ALFWorld are more tied to environment-specific knowledge, making them less transferable across tasks.

5 Analysis

Beyond performance, we analyze why SkillOS works, focusing on design choices, evolution of curator’s behaviors and contents in SkillRepo, and the role of curated skills in task success. Additional analyses are included in Appendix C.

Table 3: Ablation results of reward design on the ALFWorld dataset.
Methods    Avg. SR    Steps
SkillOS-GRPO     61.2    18.9
w/o 
r
c
​
n
​
t
     58.6    20.1
w/o 
r
c
​
o
​
m
​
p
     60.0    19.3
w/o grouping    57.3    20.6
Ablation Studies.

We ablate two components of SkillOS: (i) auxiliary rewards in Eq. 1, and (ii) grouped task streams in § 3.2.1. Experiments are conducted on ALFWorld, with Qwen3-8B used as both the curator and executor. As shown in Table 3, removing either reward component hurts performance. Without the content-quality reward, the success rate drops from 61.2 to 58.6, showing the importance of intermediate supervision for guiding skill updates in a pipelined system. Removing the compression reward causes a smaller but consistent drop, suggesting that concise repositories are easier for the executor to use. The most significant degradation comes from using random task sequences (w/o grouping), which lowers the success rate to 57.3. This highlights the importance of training on grouped task streams, in which curation decisions are learned from their downstream impact on related future tasks.

Refer to caption
Figure 4: Behaviors of the skill curator w.r.t. skill operations during training.
Behaviors of Skill Curator. To better understand how the behavior of the skill curator evolves during training, we analyze the distribution of its three skill operations from rollouts at different training steps: 
insert_skill
,
update_skill
, and 
delete_skill
. Figure 4 plots the proportion of each operation. At the beginning of training, insert overwhelmingly dominates, indicating that the model is primarily focused on populating the skill repository with new knowledge distilled from experience. As training progresses, however, update becomes increasingly frequent, while insert steadily declines. This suggests that the skill curator gradually moves from plain expansion of skills to refining existing skills. Meanwhile, delete remains a relatively small fraction throughout training with a slightly growing trend, showing the effectiveness of rewarding conciseness of SkillRepo. Instead, the dominant form of adaptation is to revise and consolidate previously acquired skills.

Refer to caption
Figure 5: Evolution dynamics of the curated skills under RL training.
Skill Evolution Dynamics. Beyond aggregate performance, we examine how the skill repository evolves during RL training. We focus on two emergent phenomena: (i) new Markdown sections within individual skills, and (ii) higher-level meta-skills that capture reusable principles across tasks. Figure 5(a) shows that early in training, the curator tends to introduce generic sections such as additional guidance, tips, or recommendations, which often make skills more verbose without substantially improving their operational value. As training progresses, these additions shift toward more actionable structures, such as failure-handling logic and conditional branches that specify when to deviate from the default workflow. This suggests that RL gradually steers the curator from superficial enrichment toward execution-oriented skill refinement. Figure 5(b) further shows that evolution occurs not only within individual skills, but also in the global organization of the repository. Early repositories are dominated by narrow, task-specific skills, whereas later repositories contain a more diverse set of meta-strategy skills covering verification, fallback planning, system search, and strategy adjustment. This indicates that the learned curator does not merely accumulate skills, but progressively expands the repository’s strategic space, shifting it from isolated task-local procedures toward more compositional cross-task control knowledge.

Refer to caption
Figure 6: Comparison of skill utilization statistics on ALFWorld.
Attribution of Skill Usage. To better understand whether the gains of SkillOS come from the evolved skills, we analyze how skills are used during evaluation. We consider 4 complementary metrics: (i) skill usage rate, the fraction of examples where the agent invokes at least one skill; (ii) successful skill usage rate, the success rate among examples that use skills; (iii) skill coverage, the fraction of the skill collection that are actually used; and (iv) the average number of skills used per example, which measures the degree of skill reliance. Figure 6 reports results on ALFWorld. Compared with the baseline, SkillOS invokes skills on all evaluation examples and achieves a higher success rate, indicating that the evolved skills contribute directly to task solving. Also, a larger fraction of the skill curated by SkillOS is used, showing that RL training improves the overall utility of the curated SkillRepo. Meanwhile, SkillOS uses fewer skills per example, suggesting that gains come from more precise skill selection rather than more skill context.

6 Conclusion

We presented SkillOS, an RL training recipe for learning skill curation in self-evolving agents. By decoupling the skill curator from the agent executor, SkillOS enables modular skill curation without retraining the underlying executor. Through grouped task streams and executor-grounded rewards, SkillOS optimizes curation decisions by their downstream impact on future tasks. Across diverse benchmarks and LLM backbones, SkillOS consistently improves both performance and efficiency. Further analyses show that trained skill curation can outperform frontier models’ zero-shot curation ability and generalize across settings, highlighting modular, trained skill curation as a practical path toward agents that self-evolve from experience.

7 Acknowledgments

We thank Zilin Xiao, I-Hung Hsu, Zexue He, and members from Google Cloud AI Research for their valuable feedback during the preparation of the paper. Siru was supported by the Molecule Maker Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897.

\nobibliography
*

References

Contents of Appendix

  
Appendix A Prompts

In this section, we provide the full prompt templates used throughout different phases and components of our framework.

A.1 Prompt for Skill Curator

The following prompt templates demonstrate the input to the skill curator during training processes.

Refer to caption
Figure 7: System prompt used for skill curator during training process.
Refer to caption
Figure 8: Tool call definition/signature of skill curator in Figure 7.
A.2 Prompt for Agent Executor

The following prompts are used for the frozen agent executor. These templates provide the agent with the current task description, a history of previous interactions, and a set of retrieved skills to guide its decision-making process. All prompts explicitly force chain-of-thought (CoT) [wei2022chain] reasoning.

For agent tasks including ALFWorld and WebShop, we follow GiGPO [feng2025group] and leverage its environment and prompt setting for inference.

Refer to caption
Figure 9: Prompt for ALFWorld agent execution with relevant retrieved skills.
Refer to caption
Figure 10: Prompt for WebShop agent execution with relevant retrieved skills.
Refer to caption
Figure 11: Prompt for agent execution in reasoning tasks with relevant retrieved skills.
A.3 Prompt Used During Training

During the RL training process, a reward 
r
c
​
n
​
t
 is assigned based on an external judge of Qwen3-32B to judge whether the curated skills are semantically meaningful and are likely to be useful for future tasks. We show the prompt to the external judge here.

Refer to caption
Figure 12: Prompt for using an external judge to assign a reward score 
r
c
​
n
​
t
 for generated skill contents.
A.4 Prompt for LLM-as-a-Judge to Obtain Correctness Signals

We present the prompts used to obtain the self-judged correctness signal 
𝟙
ξ
t
 for self-evolution via LLM-as-a-judge using the corresponding frozen agent executor as the backbone model in Figures 13, 14 for ALFWorld, reasoning, and WebShop tasks, respectively.

Refer to caption
Figure 13: Prompt for LLM-as-a-judge to obtain the correctness signal to the current trajectory in the ALFWorld benchmark.
Refer to caption
Figure 14: Prompt for LLM-as-a-judge to obtain the correctness signal for single-turn reasoning problems.
Refer to caption
Figure 15: Prompt for LLM-as-a-judge to obtain the correctness signal to the current trajectory for the WebShop benchmark.
Appendix B Implementation Details

B.1 Hyperparameters

We present the choices for all hyperparameters during both the training and inference processes in Table 4 for different tasks.

Table 4: Hyperparameters for SkillOS for training and inference settings.
Hyperparameter    Value
ALFWorld    WebShop    Reasoning
RL Training        
Learning rate    
1
×
10
−
6
Batch size    32
KL loss Coef    0.001
Max Prompt Length    16,384
Max Response Length    4,096
GRPO group size    8
Temperature    1.0
Steps    60    50    100
Data Grouping Size    10    10    Random(5,12)
Agent Executor Inference        
Top-K skill retrieval    5
Max number of turns    30    30    1
Action history length    3    3    -
B.2 Grouping Training Instances

In this section, we detail the two-stage pipeline used to turn the raw training set 
𝒟
=
{
x
i
}
i
=
1
N
 into the grouped training set 
𝒢
=
{
G
j
}
j
=
1
M
 of Section 3.2.1. Stage 1 annotates each instance with a structured set of latent attributes via an LLM annotator (Sec. B.2.1). Stage 2 assembles groups of related tasks by retrieving, filtering, and ranking candidates under a semantic phrase-level similarity (Sec. B.2.2). For training of single-turn reasoning tasks, we instantiate the pipeline on DeepMath-103K [he2026deepmathk], which provides both the raw problems 
x
i
 and a scalar difficulty score 
d
i
∈
ℝ
 that is reused as a curriculum signal by Stage 2. For multi-turn agentic tasks, we leverage the default task type annotation for each benchmark (e.g., 6 task types in ALFWorld) as they naturally expose a discrete partition of tasks into families that share the same underlying skills, and we can use this partition directly in place of the annotated attribute set 
Z
i
.

B.2.1 Stage 1: Latent Attribute Annotation

We implement the attribute set 
Z
i
 of each instance 
x
i
 as a tuple of five phrase-lists,

Z
i
=
(
T
i
,
S
i
,
C
i
,
R
i
,
P
i
)
,
where 
T
i
 is the list of high-level topics, 
S
i
 the required skills or capabilities, 
C
i
 the underlying mathematical concepts or theorems, 
R
i
 the applicable heuristic strategies, and 
P
i
 the common pitfalls. Each dimension is populated by a small set of short phrases (at most five words each). The annotator is instructed to: (i) emit standardized terminology rather than free-form rationales, (ii) omit any content specific to the question text or its final answer, and (iii) use as few phrases per dimension as necessary to characterize the task. We enforce the output schema via structured decoding with a fixed JSON response schema, and query Gemini-2.5-Pro with the highest thinking-budget configuration. The exact annotation instruction is reproduced in Figure 16.

Refer to caption
Figure 16: System instruction used to elicit 
Z
i
 from each task in 
𝒟
.
B.2.2 Stage 2: Group Construction

Given 
{
(
x
i
,
Z
i
,
d
i
)
}
i
=
1
N
, we construct each group 
G
j
=
(
x
j
,
1
,
…
,
x
j
,
n
)
 by sampling a seed task and then iteratively appending related tasks. The core primitive is a pair sampler that, given a source 
x
s
, returns an admissible successor 
x
t
; longer groups are obtained by iterating this primitive with a growing exclusion set so that instances within a group remain distinct.

Phrase similarity.

Because the annotated phrases come from a large open vocabulary (e.g., “pigeonhole principle” vs. “counting argument”), exact set overlap is unreliable. We therefore score the similarity between any two phrase lists 
A
 and 
B
 using a soft-Jaccard 
SJ
τ
​
(
A
,
B
)
 that combines exact matches with a greedy one-to-one matching between remaining phrases under a sentence-embedding cosine similarity (computed with all-MiniLM-L6-v2 [reimers2019sentence]) above a threshold 
τ
. We write 
m
τ
​
(
A
,
B
)
 for the resulting integer matched-pair count, which we use alongside 
SJ
τ
 in the filters below.

Dependency gate.

For a source 
x
s
 and candidate 
x
t
, we accept the pair only when all of the following hold:

1. Shared foundation: 
m
τ
​
(
C
s
,
C
t
)
≥
κ
C
 and 
m
τ
​
(
S
s
,
S
t
)
≥
κ
S
;
2. Shared reasoning: 
m
τ
​
(
R
s
,
R
t
)
+
m
τ
​
(
P
s
,
P
t
)
≥
1
;
3. Not a near-duplicate: 
SJ
τ
​
(
T
s
,
T
t
)
≤
θ
T
 and the weighted overall similarity 
Ω
​
(
x
s
,
x
t
)
≤
σ
max
;
4. Not too unrelated: 
Ω
​
(
x
s
,
x
t
)
≥
σ
min
;
5. Progression: 
x
t
 introduces at least one new concept or skill, i.e. 
|
C
t
|
>
m
τ
​
(
C
s
,
C
t
)
 or 
|
S
t
|
>
m
τ
​
(
S
s
,
S
t
)
;
6. Curriculum direction: 
d
t
−
d
s
≥
δ
min
.
Here 
Ω
 is a convex combination of per-dimension soft-Jaccard scores across 
{
C
,
S
,
R
,
P
,
T
}
 with weights listed in Table 5. Conditions (1)–(2) ensure genuine reuse of foundational knowledge and reasoning machinery; (3)–(4) place the pair in a useful “related but not redundant” band; (5) guarantees that 
x
t
 carries something new for the skill curator to compress into the library; and (6) enforces a forward curriculum.

Candidate retrieval and scoring.

Scoring all 
N
−
1
 alternatives per source is prohibitive, so we precompute an inverted index over the dependency fields 
{
C
,
R
,
P
}
: for each source 
x
s
, the candidate pool consists of tasks that share at least one exact dependency phrase with 
x
s
, capped at 
K
inv
 entries via uniform subsampling. Routing retrieval through dependency fields rather than topics prevents groups from collapsing onto a single narrow subject. Among the candidates that pass the gate, we select the one that maximizes

s
​
(
x
s
,
x
t
)
=
∑
f
∈
{
C
,
S
,
R
,
P
,
T
}
w
f
​
SJ
τ
​
(
f
s
,
f
t
)
+
λ
⋅
b
​
(
d
s
,
d
t
)
,
where 
b
​
(
⋅
)
 is a bounded difficulty bonus that rewards moderate forward steps. If no inverted-index candidate passes the gate, we fall back to a uniform random pool of size 
F
 and re-apply the same gate and scoring; this catches pairs whose phrases agree semantically but not lexically. Extensions sourced from the fallback pool are tagged so downstream training can audit or downweight them. The difficulty gap 
d
t
−
d
s
 is additionally modulated by a randomized curriculum mode 
(
p
↑
,
p
=
,
p
↓
)
; for our main experiments, we use an almost exclusively forward curriculum, which produced a more stable training signal than mixed curricula.

Hyperparameters.

Table 5 lists all hyperparameters of the Stage 2 pipeline and the values adopted for our main experiments. The weights were tuned on a held-out subset of 200 source tasks by manually inspecting sampled pairs for prerequisite quality; we found the pipeline largely insensitive to small perturbations of the weights but noticeably sensitive to the progression and overall-similarity-band conditions, removing either of which produced markedly more trivial or degenerate pairs.

Table 5: Hyperparameters of the Stage 2 grouping pipeline.
Symbol    Meaning    Value
—    Phrase encoder    all-MiniLM-L6-v2
τ
Cosine threshold for fuzzy phrase matching    
0.60
κ
C
Minimum matched concept pairs    
1
κ
S
Minimum matched skill pairs    
1
θ
T
Maximum topic soft-Jaccard    
0.65
σ
min
,
σ
max
Overall-similarity band    
0.30
,
 0.85
δ
min
Difficulty-delta floor    
0.0
(
w
C
,
w
S
,
w
R
,
w
P
,
w
T
)
Dimension weights    
(
5
,
 4
,
 3
,
 1
,
 2
)
λ
Difficulty-bonus weight    
1.0
(
p
↑
,
p
=
,
p
↓
)
Mode probabilities    
(
0.80
,
 0.20
,
 0.00
)
[
Δ
min
,
Δ
max
]
Gap in easy
→
hard mode    
[
0.5
,
 3.0
]
Δ
=
Maximum 
|
d
t
−
d
s
|
 in same mode    
0.3
K
inv
Inverted-index subsample cap    
2
,
000
F
Fallback pool size    
200
B.3 Experiment Setup

B.3.1 Datasets

In this section, we provide a detailed introduction to all the datasets involved in this paper.

ALFWorld. ALFWorld [shridhar2021alfworld] is a text-based interactive benchmark that aligns the TextWorld engine with the embodied ALFRED environment, enabling agents to learn high-level household policies through natural-language interaction. The benchmark covers six task types — Pick & Place, Examine in Light, Clean & Place, Heat & Place, Cool & Place, and Pick Two & Place — situated in 120 simulated rooms spanning kitchens, bedrooms, bathrooms, and living rooms. It provides 
3
,
553
 training tasks, together with 
140
 valid_seen tasks for the test set. At each step, the agent receives a textual description of its surroundings together with a goal instruction (e.g., "put a hot apple in the fridge") and must issue high-level commands such as go to, take, open, heat, and put.

WebShop WebShop [10.5555/3600270.3601778] is a simulated e-commerce web environment designed to benchmark language agents on realistic, grounded shopping tasks. The environment is populated with 1.18 million real-world products scraped from Amazon and 12,087 crowd-sourced natural-language instructions, partitioned into 10,587 training, 1,000 dev, and 500 test instructions. Given an instruction (e.g., “I’m looking for a quick-release fitness strap band in teal, priced lower than $40.00”), the agent interacts with the environment via two action types — search[query] and click[button] — to locate and purchase a product that matches the specified attributes, type, options, and price. At the end of each episode, a programmatic reward in [0, 1] is computed by comparing the purchased item against the ground-truth product specification. Following the standard evaluation protocol used in prior LLM-agent work, we evaluate on the 500 held-out test instructions.

DeepMath-103K DeepMath-103K [he2026deepmathk] is a large-scale, decontaminated mathematical reasoning dataset containing approximately 103K problems at high difficulty (primarily AoPS Levels 5–9), spanning algebra, calculus, number theory, geometry, probability, and discrete mathematics. Each problem is paired with a verifiable final answer — enabling rule-based RL rewards — together with a difficulty score, topic label, and three DeepSeek-R1 [guo2025deepseek] chain-of-thought solutions. Specifically, we annotate a subset with around 
33
,
000
 problems, with a final 
20
,
000
 set of grouped training instances.

AIME24 & AIME25. A collection of demanding mathematical problems sourced from the 2024 and 2025 American Invitational Mathematics Examination (AIME), with 30 problems each year. Problems encompass algebra, geometry, number theory, and combinatorics. Created to assess large language models’ sophisticated mathematical reasoning abilities, the dataset presents substantial difficulty, systematic multi-phase solutions, and distinctive answers, establishing it as a robust benchmark for evaluating advanced analytical capabilities.

GPQA. Short for Graduate Level Google-Proof Q
&
A Benchmark [rein2024gpqa], GPQA comprises a collection of demanding text-based multiple choice problems authored by subject specialists in biology, physics, and chemistry, intentionally crafted to be “exceptionally challenging”. We use the “GPQA-Diamond” subset for testing, which has
198
 problems in total.

B.3.2 Baselines

We compare SkillOS against five representative baselines that span memory-free agents, recent memory-augmented methods, and two internal variants of our own framework. All baselines share the same frozen Agent Executor and are evaluated under identical task suites, retrieval budgets, and decoding settings to isolate the contribution of the memory mechanism.

(i) No Memory. A memory-free baseline in which the Agent Executor solves each task independently, without access to any external memory or cross-task knowledge transfer. Each episode begins from a blank state, and no information is retained across tasks. This baseline establishes a lower bound and isolates the contribution of any form of accumulated experience.

(ii) ReasoningBank [ouyang2026reasoningbank]. A recent memory-augmented method that distills reusable reasoning insights from past trajectories and stores them as a searchable bank for future tasks. At inference time, relevant insights are retrieved and injected into the executor’s context to guide reasoning. ReasoningBank represents the class of experience-distillation approaches, which emphasize the content of stored knowledge but rely on fixed, heuristic policies for deciding what to write or discard.

(iii) MemP [DBLP:journals/corr/abs-2508-06433]. A procedural-memory method that induces reusable procedures from agent experience and applies advanced memory-management strategies — including consolidation, forgetting, and re-indexing — to maintain the memory store over time. MemP represents the class of rule-based memory management approaches, which feature more sophisticated maintenance policies than ReasoningBank but still prescribe curation decisions through hand-designed heuristics rather than learning them from downstream task feedback.

(iv) SkillOS-base. A variant of our framework in which the Skill Curator is instantiated with the same open-source backbone as SkillOS but without any RL fine-tuning, while all other components remain identical to SkillOS. This baseline serves two purposes: (a) it provides a lower-bound reference point that reflects the intrinsic prompting-based curation ability of the open-source backbone prior to optimization, and (b) it isolates the contribution of our GRPO-based training, since SkillOS-base shares exactly the same model architecture, prompting template, and memory interface as SkillOS but forgoes end-to-end optimization against task performance.

(v) SkillOS-gemini. A variant of our framework in which the Skill Curator is instantiated with Gemini-2.5-Pro instead of a trained open-source model, while all other components remain identical to SkillOS. This baseline serves two purposes: (a) it provides a strong closed-source reference point for the upper bound of prompting-based curation, and (b) it isolates the effect of our GRPO-based training, since SkillOS-gemini shares the same prompting template and memory interface as SkillOS but forgoes RL optimization against task performance.

Together, these baselines cover the main design axes along which memory-augmented agents differ from SkillOS: whether memory exists at all (i), how stored knowledge is represented (ii vs. iii), and whether curation decisions are prescribed by heuristics or learned from task feedback (ii and iii vs. SkillOS), as well as whether the curator itself benefits from RL optimization (iv and v vs. SkillOS).

B.3.3 Evaluation Metrics

We evaluate SkillOS and all baselines along two complementary axes — task effectiveness and action efficiency — using metrics tailored to each benchmark. Across all benchmarks and methods, every configuration is run with three independent random seeds; we report the mean across seeds, with one standard deviation shown as a subscript (e.g., 
85.7
±
1.6
). Within each backbone block of Tables 1 and 2, the best value in each column is highlighted in bold.

Success Rate (SR 
↑
).

Our primary effectiveness metric on both ALFWorld and WebShop. On ALFWorld, SR is the fraction of evaluation episodes in which the agent reaches the goal state within the step budget, yielding a binary 
{
0
,
1
}
 outcome per episode. We report SR both per task category — Pick, Look, Clean, Heat, Cool, and Pick2 — and as a macro-average (Avg. SR) across the six categories, so that categories with fewer tasks are not dominated by larger ones. On WebShop, following [10.5555/3600270.3601778], SR is the fraction of episodes whose final reward equals exactly 
1
, i.e., the purchased product fully matches all specified attributes, options, type, and price constraints.

WebShop Score (
↑
).

In addition to SR, WebShop provides a dense per-episode reward in 
[
0
,
100
]
 that credits partial matches on attributes, options, type, and price even when the purchase is not a perfect match. We report the average score across evaluation episodes as a finer-grained complement to SR: two methods with similar SR may differ substantially in how close their near-misses are to the target product.

Number of Steps (Steps 
↓
).

Our efficiency metric on ALFWorld and WebShop. Steps is the average number of environment actions the agent issues per episode, computed over all evaluation episodes regardless of success. Failed episodes contribute steps up to their termination point (task completion, max-step cutoff, or early stop). This metric captures a dimension that SR and Score alone cannot: two methods may achieve comparable effectiveness while differing substantially in how efficiently they reach the goal, which has direct implications for inference cost and deployment feasibility.

Accuracy (Acc. 
↑
) on reasoning benchmarks.

For the single-turn reasoning datasets — AIME24, AIME25, and GPQA — we report exact-match accuracy: the fraction of questions whose extracted final answer matches the ground truth. For AIME24 and AIME25, we adopt the evaluation protocol from the HuggingFace math_verify1
1https://github.com/huggingface/Math-Verify
 toolkit, which parses the model’s final boxed expression and verifies mathematical equivalence to the reference answer (accounting for equivalent numerical forms, simplifications, and formatting variants). For GPQA, which is a multiple-choice benchmark, we extract the predicted option letter from the model’s response and score it as correct if and only if it exactly matches the ground-truth option. We additionally report an average accuracy (Avg. Acc.) across the three datasets to summarize overall reasoning ability.

Evaluation protocol.

All methods share the same frozen Agent Executor, retrieval budget (top-
k
 skills retrieved via BM25), maximum step budget, and decoding temperature within each backbone, so that differences in the reported metrics are attributable to the memory mechanism rather than to confounding inference settings. Unless stated otherwise, all numbers in the main paper are computed on the official held-out evaluation splits of each benchmark.

Appendix C Additional Analyses

C.1 Results on Gemini-3.1-Flash-Lite

In addition to the Qwen3-8B/32B and Gemini-2.5-Pro executors used in the main paper, we further evaluate SkillOS on ALFWorld with the more recent Gemini-3.1-Flash-Lite as the frozen Agent Executor, to verify that our gains generalize to newer model families. Results are reported in Table 6.

SkillOS achieves the highest average success rate (73.1%), outperforming the strongest external baseline ReasoningBank (66.0%) by +7.1 points and the No-Memory baseline (61.2%) by +11.9 points, while requiring the fewest interaction steps (15.5 vs. 18.5 for No Memory). The two internal variants reproduce the ordering observed in the main experiments: SkillOS-base reaches only 63.6% — barely above No Memory — confirming that the open-source backbone cannot recover the curation policy through prompting alone, and SkillOS-gemini improves to 71.2% but is still surpassed by SkillOS despite using a much stronger curator backbone. This reinforces our main finding that learning the curator with task-level feedback contributes more than scaling up the curator model. We also note that MemP (58.6%) underperforms even No Memory under this executor, suggesting that hand-designed curation heuristics are brittle when the executor is less capable, whereas the policy learned by SkillOS remains robust. Per-subset, SkillOS wins on four of six subsets, with particularly large margins on Look (84.6% vs. 71.8%) and Cool (68.0% vs. 48.0%); the remaining two subsets are won by SkillOS-gemini (Pick) and ReasoningBank (Heat), on which SkillOS nonetheless remains competitive. Overall, these results confirm that the advantage of SkillOS transfers cleanly to a newer executor family.

Table 6: Experiment results on ALFWorld benchmark. Success rate (SR 
↑
) and the number of steps (Steps 
↓
) are reported on 6 subsets for Gemini-3.1-Flash-Lite as frozen executor.
Methods    Pick    Look    Clean    Heat    Cool    Pick2    Avg. SR    Steps
(35)    (13)    (27)    (16)    (25)    (24)    (140)
No Memory    
85.7
 0.0
59.0
 8.9
67.9
 9.3
25.0
 6.2
38.7
 2.3
66.7
 0.0
61.2
 2.3
18.5
ReasoningBank    
87.6
 4.4
71.8
 4.4
63.0
 0.0
52.1
 14.4
48.0
 10.6
62.5
 0.0
66.0
 2.7
17.6
MemP    
84.3
 6.1
57.7
 5.4
63.0
 0.0
28.1
 4.4
34.0
 2.8
62.5
 0.0
58.6
 1.0
19.3
SkillOS-base     
86.7
 1.6
61.5
 0.0
66.7
 0.0
41.7
 6.2
38.7
 16.0
68.1
 2.4
63.6
 3.9
17.7
SkillOS-gemini     
96.2
 1.6
61.5
 13.3
74.1
 3.7
31.2
 12.5
66.7
 4.6
68.1
 2.4
71.2
 2.9
16.1
SkillOS    
88.6
 0.0
84.6
 13.3
77.8
 0.0
37.5
 17.2
68.0
 8.0
68.1
 2.4
73.1
 2.7
15.5
C.2 Case Studies

Refer to caption
Figure 17: Case studies of curated skills by SkillOS.
Curated Skills for Different Tasks.

Figure 17 presents two representative skills curated by SkillOS that illustrate qualitatively different curation patterns across task types. For agentic tasks (Figure 17(a)), the curator distills a meta-strategy for failure recovery: rather than memorizing a specific object-search trajectory, it abstracts the recovery procedure into a reusable workflow (exhaustive search 
→
 confirm unavailability 
→
 identify a substitute 
→
 proceed with substitute) and explicitly references existing skills, demonstrating compositional curation. For reasoning tasks (Figure 17(b)), the curator captures branching-out reasoning: a single skill on inradius–circumradius–semiperimeter relations encodes multiple solution paths (relating the target distance to either the in/circumradius or the side lengths), each paired with its formula, application, and prerequisite constraints. Together, these examples show that SkillOS learns to produce skills tailored to the structure of the underlying task: procedural and composable for agentic settings, and multi-path with explicit preconditions for reasoning settings, rather than verbatim trajectory copies.

Refer to caption
Figure 18: Case study on math-reasoning skill curation. SkillOS-base produces a generic partitioning recipe, while SkillOS curates a concrete and reusable counting framework with explicit constraints, equations, and a worked example.
How SkillOS Curates Better Skills Compared to Baselines.

We further qualitatively compare the skills curated by SkillOS against those produced by the baseline curator. In the math-reasoning case as shown in Figure 18, SkillOS-base outputs only a generic high-level recipe based on partitioning into disjoint sets, without explicit formulas, constraints, or examples. By comparison, SkillOS curates a much more useful skill that provides a concrete counting framework, including explicit constraint formulation, equation setup, and a worked example tailored to the target sub-problem. These examples show that RL-trained skill curation improves not only the correctness of the curated content, but also its specificity and usability, enabling skills to better capture the underlying structure of tasks.

Refer to caption
Figure 19: Case studies of how skills curated by SkillOS successfully helped to solve a task in ALFWorld.
How Curated Skills Help to Solve Tasks Successfully.

Figure 19 illustrates a representative example of how curated skills improve agent behavior in interactive environments. Given the task “look at the CD under the desklamp,” the memory-free baseline fails to infer the correct object–location relation and performs an inefficient search over irrelevant containers, eventually exhausting the step budget. In contrast, SkillOS retrieves a skill that encourages the agent to examine objects under or around light sources when the instruction refers to an object being “under” a lamp. Guided by this reusable strategy, the agent first locates and picks up the CD near the desk area, then moves to the desklamp and inspects the correct target location, completing the task successfully. This case highlights that curated skills do not merely memorize task-specific action sequences; instead, they provide transferable decision guidance that helps the agent focus exploration on semantically relevant objects and locations, reducing unnecessary interactions and improving task success.

Appendix D Limitations

Retrieval Mechanism.

Our current implementation relies on a relatively simple keyword-based retrieval mechanism, such as BM25, to retrieve relevant skills from the skill repository. This design choice allows us to isolate the main focus of this work: studying how skills can be curated, updated, and organized through experience-driven learning. However, more advanced retrieval methods, such as dense retrieval, hybrid retrieval, or learned retrievers, may further improve the relevance of retrieved skills and thus lead to stronger downstream performance. We leave the joint optimization of skill curation and skill retrieval to future work.

Simplified Skill Representation.

Following Anthropic’s skill paradigm [anthropic_skills_2025], we instantiate each skill as a single Markdown file that combines a YAML frontmatter and Markdown body. This simplification keeps the curator’s action space tractable, but it discards two affordances of the original SKILL.md format: (i) supporting scripts and external resource files that allow skills to encapsulate executable procedures rather than purely declarative knowledge, and (ii) hierarchical organization in which a top-level skill can reference or compose lower-level sub-skills. As a result, behaviors that are most naturally expressed as runnable code or as compositions of finer-grained primitives must currently be flattened into prose. Extending SkillOS to multi-file, hierarchical, and partially executable skills is a natural next step.

Frozen Agent Executor.

Throughout training, we keep the agent executor 
π
ℒ
 frozen and optimize only the skill curator 
π
𝒮
. This decoupling is deliberate: it isolates the contribution of skill curation, makes the recipe modular across executors, and avoids confounding our analysis with executor-side adaptation. The downside is that the curator can only shape the system’s behavior through what it writes into SkillRepo; any miscalibration between the curated skills and the executor’s idiosyncrasies must be absorbed by the curator alone. Joint or alternating optimization of 
π
𝒮
 and 
π
ℒ
 may yield a better-aligned pair, at the cost of executor specificity and substantially higher training cost.

Appendix E Future Research Directions

Our work opens several promising directions for future research.

Agentic Search over Experiential Memory.

SkillOS currently retrieves relevant skills from SkillRepo through a fixed top-
k
 BM25 lookup, treating retrieval as a static, one-shot operation. As the skill repository grows across thousands of tasks and domains, the bottleneck of self-evolving agents shifts from what to store to how to reliably retrieve and inject the right fragments at each decision step. A natural next step is to replace static retrieval with agentic search: letting the Skill Curator (or a dedicated retrieval agent) actively issue multiple queries, reformulate them based on intermediate evidence, and iteratively decide which skills to surface, cite, or compose for the executor. This reframes memory access as a first-class decision in the agent’s policy rather than a preprocessing step, and opens the door to scaling SkillOS to memory stores orders of magnitude larger than those considered here.

Hierarchical and Compositional Skills.

Our current skills are flat Markdown entries, each describing a single reusable pattern. Real agent competence, however, is hierarchical: high-level procedures invoke lower-level sub-skills, which in turn depend on primitive operations. Extending SkillRepo to support hierarchical decomposition — where the curator learns not only to insert, update, and delete skills but also to link, compose, and abstract them — could enable the agent to build increasingly expressive procedural libraries over time. This direction connects naturally to program-synthesis and library-learning literature, and would allow SkillOS to scale to longer-horizon tasks where single-skill retrieval is insufficient.

Multi-Agent and Shared Memory.

SkillOS treats memory as a single agent’s private artifact. In many realistic deployments, however, multiple agents operate in parallel (e.g., code review, multi-hop research, collaborative robotics) and could benefit from shared experiential memory. Open questions include how to arbitrate conflicting curation decisions from different agents, how to attribute credit when a shared skill contributes to one agent’s success but another’s failure, and how to preserve specialization while enabling cross-agent transfer. Our GRPO-based curator provides a natural starting point, but extending it to the multi-agent credit-assignment setting is non-trivial and likely to require new algorithmic ideas.

Appendix F Use of LLMs

We used LLMs as a general-purpose writing assist tool during the preparation of this submission. Specifically, LLMs were employed for polishing the clarity and readability of text (e.g., refining sentence structure, improving grammar, and shortening overly verbose phrasing). All research ideas, methodology design, experiments, analyses, and final writing decisions were conceived, implemented, and validated solely by the authors.

Experimental support, please view the build logs for errors. Generated by L A T E xml  [LOGO] .
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button, located in the page header.
Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

BETA

> there was an infsh outage did it break our benhcmarks. im fixing infsh now i will let you know

> most imporatnt question is were we able to get the no memory baseline of the paper otheriwse gains dont matter (much yet)

> the paper doesnt cap at 30 steps doe sit

> does paper supply any temprature etc settings ?

> do you wan to check that paper 
  For the executor's actual decode config, the paper defers to another paper:
  - §4.1 / Appendix A.2: "we follow GiGPO [feng2025group] and leverage its environment and prompt setting for inference"
  (ALFWorld + WebShop).
  - Appendix B.3.3 only says all methods share "the same … decoding temperature within each backbone" — i.e. it's held fixed
  across methods, but the value is never stated. ?

> infsh back

> update notes in repo and tmp

> hows the test going

> i dont see runs are you sure its livive

> no i mean no new belt task runs in a while ?

> which task is that

> if i cancel the blocking will it heal/retry?

> is this a broken sdk-py thing if so briefly document so i can fix sdk

> did they recover or no

> still stuck ?

> check back now

> thre must be something! there is a small chance that remote serves distiled or low precision. providers do that shit .. lets hook up a vllm instance all 8 h100 here and do the benchmark agian

> check back now

> btw interesting find: AtlasCloud

US
fp8
Latency
1.03s
Throughput
10tps
Uptime

Uptime 98.9 percent

Total Context
41.0K
Max Output
8.2K
PRICING
Input Price
$0.05
/M tokens
Output Price
$0.40
/M tokens
Cache Read
$0.05
/M tokens
Alibaba Cloud Int.

SG
Latency
0.69s
Throughput
60tps
Uptime

Uptime 100.0 percent

Total Context
131.1K
Max Output
8.2K
PRICING
Input Price
$0.18
$0.117
/M tokens
Output Price
$0.70
$0.455
/M tokens may the open router app using atlas cloud by mistake and its fp8? (and its also slow super stupid) can you check one task log we might be logging it or in output meta which cloud ?

> infsh is borken ill deploy fix now

> you answer other question

> how long before we can test bigger budget

> but wait first eval local vllm finished?

> are you usig single gpu not 8 ?

> yeah we should use 8 parallel to bust through this

> i dont knwo time

> wtf is that?

> but you sais 1/8 already how come we could run parallel games no?

> why call them "shards" should one entire model on each gpu or 8 x vllms whatever. explain

> check srtatus

> wen

> so ?

> " — here's the bottom line, then I'll go get the actual cause.

  What's proven: We can't blame the model's numerics. bf16 full-precision, full context, full 140-game split = 37.9%. The 18pp
  panic is really a 10pp gap, and every lever we pulled (fp8→bf16, temp 0.6→0.4, reasoning on/off, max_tokens 256→8192) moved
  nothing meaningful. The cause is structural: 64% of games run to the 30-step cap and time out on the multi-step tasks (Heat
  19%, Clean 26%, Pick2 33%), while single-step Pick is fine (63%).

  So the only thing left worth doing is watching one failure die. The eval jsonl doesn't store trajectories, so I'll trace a
  single failed Heat game against an idle vLLM server, logging every action + observation + the model's raw output, and read
  why it loops. Doing that now. we need to run a few different ocnfigs. search online best qwen 8b config maybe or try a couple runs yourself if you cant find any?

> sup

> so

> b + d could it be better ?

> can we also try remote with top k ?

> infsh didnt support top k etc but i added it but you need to rerun it soorry

> try again infsh

> [Request interrupted by user for tool use]

> i think there is something else going on here. its houslnd be hanging do you have a easy repro

> did you check schema with belt app get

> ok benchmark now

> 1. Fix run_task_resilient to fail-fast on 4xx (re-raise, no retry) — a deterministic validation error should never burn 10
  backoff cycles. One small edit.

> sup

> ok so we are close enough to 40 to run the actual training again with remote and its more stable now yeah? maybe update the docs in repo with findings and updates and then start the training loop!

> (n=140, SE≈4.2pp ≈ 1.2σ). is this noise defined n the paper? otherwise if tehre is so much noise how are we going to claim
  +x points. anyway save the benchmarks im happy with remote. lets run the training


## 2026-05-23

> yes check back after first few opt-steps

> but whats the status now its been 8 ghours

> whats the status is it doing good. do we already see improvement. or is it unstable

> ill wait to 60 to have a good comparison to paper

> sup

> stop at 60 and run the eval

> AND

> you knwo you can just dump things at infsh.. and it will queue and handle i hop eyou are not getting bocked by your own limiteations or something (dont change anything just report

> so ?

> and now

> and so what now

> did it finish

> check again

> ok lets continue the training please this is no use

> 10 days? wtf paper did fft 3 days we are doing a lora ?

> i dont want more than paper. i want what paper has.. (except its a lora

> comtinue


## 2026-05-24

> yes dig into the broke cases but we might need hyperparm tuning to ger bettee results it should be possible or try fft to fully repro the paper

> and ?

> why do we need vllm sorry remote executor and judge is just okay thats established


## 2026-05-25

> but my man we are training the curator would it woek with training vllm?

> dows paper me tion deepspeed, what does it do any accuracy penalty

> so zero is comparable directly to verl setup?

> wait you recommended hf trl as a 1:1 alternative to verl

> one sentece what is trl and vel

> can trl produce same verl reaults in theory if hyper params same

> ok go ahead with zero-2 fft and get close to verl without desteoying everything  maybe commit first as checkpoint

> check

> why memshare vllm is faster

> beo this is stupid hf and deepseek coulda shoulda handle this research and fix and run fft dont stop until you fo

> check

> [Request interrupted by user for tool use]

> hy 123 x6 wtf also dont i have more space check disks

> RAID0 all 8 nvme instead of just one

> all good?

> how big is this model btw why do we need 128x6

> ofc eval lets see? wait we ended up using vllm? how does colocate with hf work why cant hf be that fast its an enormous gap


## 2026-05-26

> man why eval runs against local executor.. we dont need to do that with vllm.. do we change to verl now to test if its implemntation?

> seems easy fixes do it

> as long as we are following the paper...

> and?

> are we good

> and?

> whats the progress

> in the menatime, commit chnages update plans and md files, maybe create a journal of what encountered what tried what changed so we work transparently and provide a better info source compared to the paper!!

> are you spamming infsh with tasks


## 2026-05-27

> yes and retries should try cancwlling the tasks bwfore they retry if its a timeout we fontrol please

> hows going


## 2026-05-28

> hows the training going i see two active shells

> ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Background tasks
  2 active shells
    cd /home/ubuntu/skillos; until grep -q "DONE_BASELINE" /tmp/probe_baseline.out 2>/dev/null || ! pgrep -f… (running)
  ↑/↓ to select · Enter to view · x to stop · ←/Esc to close
[0] 0:claude*

> about the eval what do you want to build short explain

> is that how the paper does it

> yes build the closed-loop version

> 12 serila and 12 parallwl?

> how can eval take 12 h doesnt sound right

> lets go

> sweep ckpts 20 30 40 50, and answer if reward reseaign is aligned with paper

> wow 3600 is a lot we should tRT but they say with 16 gous influding loval vllm and 3 different datagsets they did all in 3 days how?? transfee probe and basline any tips to aliyn with paper

> wait you say infh rmeote is the bottle neck and then say optimisation step sucks hard which is THE bottleneck how come?

> they have 16x gous which makes it easier to roll vllm next to training we have 8 thats why i am using remote for executor and jusdhe


## 2026-05-29

> so how to do 3600 steps in 3 days?

> but how does the paper do it

> why local vl again?

> i dont want local gl i want to put all 8 gpus to only training and keep things clean  i want 1:1 paper replication for algo and training plewsee


## 2026-05-30

> so verl will bu kuch different than hf trl?

> but i still dont udneratand whats the point what will this buy us

> so cant do the algo on trl?

> letw fix and match to paper on tr

> do it witoout loaing curent state commit make folde rtc

> why stop you wasted precious gpu hours

> start it ???

> how's it going

> are you limitin parallel tasks on i ference?


## 2026-05-31

> how's it going

> how's it going


## 2026-06-01

> how's it going

> any indication of performance


## 2026-06-02

> how's it going

> how's it going

> when was this degrade

> how's it going


## 2026-06-03

> how's it going


## 2026-06-04

> how's it going

> launch the eval sweep

> try again

> how's it going

> any early scores how are we looking


## 2026-06-05

> cellso are we in line with the paper? Do we get the same results?  exactly like they   going on?

> TThe training steps theory makes sense, but do we know their effective batch  and maybe we can match that by increasing  or   accumulation steps what we also need to mindful about the learning  and other hyper parameters  to batching

> how to replicate the exact paper with 8 gpus

> bruhda we have been through this twenty times why are we not following the paper 1:1 in terms of hyperparams and wate days of gou?

> yes draft it and queue after ckpt40 commit properly

> lfg


## 2026-06-06

> sorry i meant what is a why 0.3 0.4 are we only utilising less than half why

> lets run a small test with a pls

> how's it looking

> sup

> way zero is lora?

> you know what we started with lora then reverted back to fft because we didnt het the right results  maybe we should keep experimenting with loras its faster and would allow us to get things right wdyt? not sure what ground we covered since our last lora test

> commit hat we have make it easy to go back to fft when we want it olease then run a lora with proper beta like paroer


## 2026-06-07

> sup ?

> is there a wnb link

> btw i see 4 bg tasks   ❯ v7 step/save/error (warnings suppressed) (running)
    v6 step/save/error (warnings suppressed) (running)
    algo1 eval — wave completions + crashes (running)
    algo1 v5 — max_completion 4096 — step + crashes (running)

> let it ride


## 2026-06-08

> sup ?

> uhm fft took 3 days how come lora is 2 more days ?

> 2560 infsh executor calls per step anything to parallelise here ?

> ocal vllm doesnt really save time and we never straty form paper hyperparams unaccaptable

> let it ride

> whats happening


## 2026-06-09

> sup


## 2026-06-10

> whats the latest?

> are you serious how did we miss this and wasted so much fucking time ...

> ok document this issue in an md file. and lets fix everything!

> Check the background smoke test b33xx19bk result (cat /tmp/claude-1000/-home-ubuntu-skillos/29e0d471-7472-4f30-a397-cbaf3fda8663/tasks/b33xx19bk.output). If SMOKE PASSED, launch the 8-rank one-step smoke: ALFWORLD_DATA=$HOME/.cache/alfworld HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 .venv/bin/accelerate launch --config_file configs/accelerate_zero2.yaml -m scripts.train_algo1 --config configs/alfworld_algo1_smoke.yaml (run from /home/ubuntu/skillos, log to logs/, run_in_background). Verify in its log: distinct gids 0-7 across ranks in the [algo1] rollout lines, identical seeds within a gid, judge live, one opt step completes. Then report v8 launch-readiness to the user. If the env smoke failed, diagnose and fix.

> Check background task b727k09gr (env smoke rerun with valid/invalid insert coverage). If SMOKE PASSED, launch the 8-rank one-step smoke from /home/ubuntu/skillos: ALFWORLD_DATA=$HOME/.cache/alfworld HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 .venv/bin/accelerate launch --config_file configs/accelerate_zero2.yaml -m scripts.train_algo1 --config configs/alfworld_algo1_smoke.yaml > logs/algo1_smoke_8rank_$(date +%Y%m%d_%H%M%S).log 2>&1 (run_in_background, ~10 min). Then verify in the log: [algo1] rollout lines show distinct gids 0-7 with identical seeds within each gid, judge heuristic live, one opt step completes with loss/reward metrics, exit 0. Report v8 launch-readiness. If smoke failed, diagnose and fix first.


## 2026-06-11

> i see v7 still runningnis that right

> lfg v8

> commit the fixes and part mortem

> when you reply print time

> sup

> so we are doing lora should be faster. and the paper did fft and over 3 datasets in 3 days. i feel like there is something we might be missing do you want to review the paper again?


## 2026-06-12

> sup


## 2026-06-14

> sup

> sup?

> Your tool call was malformed and could not be parsed. Please retry.


## 2026-06-15

> sup


## 2026-06-16

> how do the metrics look

> how long

> checkl again

> how are we looking bruv

> Your tool call was malformed and could not be parsed. Please retry.

> what


## 2026-06-17

> when

> [Request interrupted by user for tool use]

> crashed?

> we sat here wasting 8xh100 because of your srupid grep?

> stop bullshitting can it be reusmed

> 50??? we will lose dyas of work???

> eval then reusme

> how how long does it take?

> eval takes a full day?

> bro infsh can scale infinte come on we talked about this soooo many times [163]

> let it finish


## 2026-06-18

> Check the 5 v8 eval arms (logs/eval_v8_ckpt{10,20,30,40,50}_*.log, output/eval-v8/ckptN.jsonl game counts). The completion watcher bo5ccw5ds will notify when all 5 hit 140 games. If still running, just confirm steady wave progress and that infsh hasn't stalled (compare game counts vs last check). If all done, run scripts/compare_eval_arms.py for each arm vs output/eval-pathbv4/no_memory.jsonl (paired-by-gamefile McNemar), report per-checkpoint SR + delta + p-value to answer the U-shape question, then prepare to resume training from checkpoint-50.

> sup

> trust in infsh/belt  unless we have a better idea to rerun a ful teianing the goal is the 1:1 replicate papers results use their steps epochs hyperparme eeh?

> lets uodate script to save chrckpoint when infsh gets flaky if it does and resume

> OneOne question while we wait for the results when we have this strained  on  game based on paper is the knowledge  be transferable to other tasks like can I use this to curate skills for larger conversations or does that need a better training?

> whwhat do you mean the current status of inference it's right now not even  through everything    the   do we see step zero did you commit to changes to do the graceful abort with saving steps? Can you clean up the tasks?

> wwhat do you H is  I don't understand. I just told you that it is not explain yourself.

> whwhy do you keep saying it's still flaky?


## 2026-06-19

> what crashed what task wtf

> but why

> we restarted only recently how can one gou be 4 hours behind

> will cutting skew training ?

> 4 hours sis usper long like it shiuld have gotten allnlong ones and only all long ones

> if deadline wont skew my results do it pls and restart and cleanup task list

> how long

> i aee 4 shells running here btw why

> still 3 shells

> sup

> wait training completed?

> is there an algo 2? why are we calling it algo 1

> how long until the eval finishes? and while we wait for the eval can you look at one of our conversations json b in here and tell me how could we create a training dataset from our conversations maybe

> Bottom line: ckpt30 is your result (+9.3pp, p=0.035), and the full schedule proves
  later steps don't help. uhm paper claims otherwise no :) but before you answer that how can we try what happens if we feed a portion of a chat history to the ckpt-30 what does it say

> interesting it deosnt really pick things that the agent learned by trying but more so just specific easy facts which agent already knew and used no

> What this tells us: GRPO taught ckpt30 a domain-general skill-abstraction behavior — "turn an execution trace into reusable structured skills" — not ALFWorld-specific memorization. It transferred how do we know its different than the vanilla ?

> how long

> im also curious about ckpt 60 and multi sample

> whats different then paper still why didnt me get their pp


## 2026-06-20

> hmm lets look at these see if anything to try quick iterations

> belt auth token or something

> infsh back

> i think we need to put our backs into this cant give up easily but we arent also chasing fringe hyperparam tuning. lets test what we can quickly iterate

> did you check ablations

> yes do the 2x2, re-eval ckpt30 with grammar prompt


## 2026-06-21

> can we commit and push

> are they comparing against gemini instead of vanilla qwen? Thanks 
@_akhaliq
 for sharing our paper!

We study skill curation as a trainable capability for self-evolving agents. This capability actually is ‼️hard to train‼️due to the long-term evolving setting that only has indirect and sparse learning signals.

In SkillOS, a skill curator operates over evolving Markdown skills like an operating system — continuously inserting, refining, and organizing reusable experience over time.

With our training recipe, a trained 8B curator can outperform zero-shot frontier models (Gemini-2.5-Pro) for skill curation in self-evolving agent systems.

> do i add ssh key gfor push or gh lgin

> added

> ok did we learn anything new do you have the full paper saved in repo what else to try/

> https://arxiv.org/pdf/2605.06614

> new_skill_insert: If there is no existing relevant skill, create new skill with desired skill name and content.
Parameters: {"type": "object", "properties": {"skill_name": {"type": "string", "description": "The name of the new skill to create."},
"content": {"type": "string", "description": "The markdown content for the new skill."}}, "required": ["skill_name", "content"]}
Format the arguments as a JSON object.
skill_update: If the existing skill can be improved, update the specific skill by its <skill_name>.
Parameters: {"type": "object", "properties": {"skill_name": {"type": "string", "description": "The name of the skill to update. Skill
name must exist and exactly match the title of an existing skill."}, "new_name": {"type": "string", "description": "The new skill
name for the skill, which replaces the old name. If not provided, the skill name will remain unchanged."}, "new_content": {"type":
"string", "description": "The new content for the skill, which will replace the entire old content. Please ensure full content if
provided. If not provided, the skill content will remain unchanged."}}, "required": ["skill_name"]}
Format the arguments as a JSON object.
skill_delete: Delete an existing skill by its title.
Parameters: {"type": "object", "properties": {"skill_name": {"type": "string", "description": "The name of the skill to delete."}},
"required": ["skill_name"]}
Format the arguments as a JSON object.
Figure 8 |Tool call definition/signature of skill curator in Figure 7.
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
## Past Relevant Skills
{retrieved_skills}
## Current Progress
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}
Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant skills. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.
Figure 9 |Prompt for ALFWorld agent execution with relevant retrieved skills.
You are a reasoning expert with access to a list of skills. Use the skills below to provide correct responses to user queries.
## Past Relevant Skills
{retrieved_skills}
## Problem
{question}
Please reason step by step, using the past relevant skills when helpful, and put your final answer within \boxed{}.
Figure 11 |Prompt for agent execution in reasoning tasks with relevant retrieved skills.
You are an expert in data annotation and mathematical reasoning.
A.3. Prompt Used During Training
Given a mathematical question, generate one or more phrases (less than 5 words) that thoroughly and precisely describe the characteristics
of the math problem in the following dimensions:
1. Topic
During the RL training process, a reward 𝑟𝑐𝑛𝑡 is assigned based on an external judge of Qwen3-32B to
2. Skills or Capabilities
3. Math Concepts or Theorems
judge whether the curated skills are semantically meaningful and are likely to be useful for future
4. Heuristic Strategy
tasks. We show the prompt to the external judge here.
5. Common Pitfalls
## Requirements
- The annotations should be phrases only, avoid lengthy sentences
- Do NOT include any context or specifics from the question or solution
- Put your response in JSON format.
- Use as less phrases as possible for each dimension
- Use standardized/acknowledged phrases/terminologies only since phrases generated will be used for large-scale data processing
20
SkillOS: Learning Skill Curation for Self-Evolving Agents
You are an expert agent operating in the WebShop e-commerce environment. Your task is to: {task_description}
## Past Relevant Skills
{retrieved_skills}
## Current Progress
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}
Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant skills. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.
Figure 10 |Prompt for WebShop agent execution with relevant retrieved skills.
System Instruction
You are an expert memory analyst. Analyze the quality of the following content of skills memory based on the following criteria:
1. 2. 3. 4. ABSTRACTION: The skill captures generalizable procedures or insights, not verbatim copies of the trajectory. Specific IDs, numbers,
object names from the task have been replaced with variables or general concepts.
REUSABILITY: The skill is atomic and modular — it describes one coherent capability that could plausibly be triggered by future related
tasks, rather than bundling unrelated steps.
ACTIONABILITY: The Markdown body provides concrete guidance (workflow, conditions, when-not-to-use) that an executor can act on,
rather than vague advice.
FAITHFULNESS: All claims in the skill are supported by the trajectory; no fabricated facts, tools, or environment behaviors.
Respond ONLY with a JSON code block in this exact format:
```json
{
"VALID": true/false,
"ISSUES": [list any problems found],
"EXPLANATION": "brief explanation of the assessment"
}
Input
Analyze the following skill content: \n{content}
Figure12 |[REDACTED]𝑟𝑐𝑛𝑡 forgeneratedskillcontents.
A.4. Prompt for LLM-as-a-Judge to Obtain Correctness Signals
We present the prompts used to obtain the self-judged correctness signal 𝜉𝑡 for self-evolution via LLM-
as-a-judge using the corresponding frozen agent executor as the backbone model in Figures 13, 14
for ALFWorld, reasoning, and WebShop tasks, respectively.
B. Implementation Details
B.1. Hyperparameters
We present the choices for all hyperparameters during both the training and inference processes in
Table 4 for different tasks.
B.2. Grouping Training Instances
In this section, we detail the two-stage pipeline used to turn the raw training set D= {𝑥𝑖}𝑁
𝑖=1 into the
grouped training set G= {𝐺𝑗}𝑀
𝑗=1 of Section 3.2.1. Stage 1 annotates each instance with a structured
set of latent attributes via an LLM annotator (Sec. B.2.1). Stage 2 assembles groups of related tasks by
retrieving, filtering, and ranking candidates under a semantic phrase-level similarity (Sec. B.2.2). For
21
SkillOS: Learning Skill Curation for Self-Evolving Agents
System Instruction
You are an expert judge evaluating whether an embodied agent successfully completed a household task in a text-based simulator. Output a
single JSON object and nothing else.
# Task
You will be given (1) the task description the agent was asked to complete, and (2) the full interaction trace between the agent and the
simulator. Determine whether the agent fully completed the task.
## What "success" means
- The agent's actions must have produced the world state the task description specifies. Every condition stated in the task must hold at the
end of the trace.
- If the task implies a transformation must occur before a final placement or interaction, the transformation must be evidenced in the trace
before the final step.
- Credit only effects that the simulator's observations confirm. Do not credit effects that the agent merely declared, planned, or assumed. -
Ignore the agent's own claims of completion; rely solely on the simulator's observation strings.
- A trace that ends with the agent stuck in a loop, exhausting its step budget, or repeatedly emitting invalid actions is a failure regardless of
partial progress.
## Strictness
- If the trace is ambiguous about whether every required condition is satisfied at the end, output success=false.
- Partial completion is failure. Either every condition holds or the trace is a failure.
# Output
Output exactly one JSON object with these fields, and nothing else: {{ "success": <true|false>, "rationale": "<one or two sentences citing the
specific observations that prove success or failure>", "evidence_step": <integer step index where success was confirmed, or -1 if failure> }}
Input
# Inputs
## Task description \n{task_description}
## Trajectory
The trajectory alternates between simulator OBSERVATION and agent ACTION.\n{trajectory}
Figure 13 |Prompt for LLM-as-a-judge to obtain the correctness signal to the current trajectory in the
ALFWorld benchmark.
training of single-turn reasoning tasks, we instantiate the pipeline on DeepMath-103K (He et al.,
2026a), which provides both the raw problems 𝑥𝑖 and a scalar difficulty score 𝑑𝑖 ∈ℝ that is reused
as a curriculum signal by Stage 2. For multi-turn agentic tasks, we leverage the default task type
annotation for each benchmark (e.g., 6 task types in ALFWorld) as they naturally expose a discrete
partition of tasks into families that share the same underlying skills, and we can use this partition
directly in place of the annotated attribute set 𝑍𝑖.
B.2.1. Stage 1: Latent Attribute Annotation
We implement the attribute set 𝑍𝑖 of each instance 𝑥𝑖 as a tuple of five phrase-lists,
𝑍𝑖 = 𝑇𝑖, 𝑆𝑖, 𝐶𝑖, 𝑅𝑖, 𝑃𝑖 ,
where 𝑇𝑖 is the list of high-level topics, 𝑆𝑖 the required skills or capabilities, 𝐶𝑖 the underlying mathe-
matical concepts or theorems, 𝑅𝑖 the applicable heuristic strategies, and 𝑃𝑖 the common pitfalls. Each
dimension is populated by a small set of short phrases (at most five words each). The annotator
is instructed to: (i) emit standardized terminology rather than free-form rationales, (ii) omit any
content specific to the question text or its final answer, and (iii) use as few phrases per dimension as
necessary to characterize the task. We enforce the output schema via structured decoding with a fixed
JSON response schema, and query Gemini-2.5-Pro with the highest thinking-budget configuration.
The exact annotation instruction is reproduced in Figure 16.
B.2.2. Stage 2: Group Construction
Given {(𝑥𝑖,𝑍𝑖,𝑑𝑖)}𝑁
𝑖=1, we construct each group 𝐺𝑗 = (𝑥𝑗,1,...,𝑥𝑗,𝑛)by sampling a seed task and then
iteratively appending related tasks. The core primitive is a pair sampler that, given a source 𝑥𝑠, returns
an admissible successor 𝑥𝑡; longer groups are obtained by iterating this primitive with a growing
exclusion set so that instances within a group remain distinct.
22
SkillOS: Learning Skill Curation for Self-Evolving Agents
System Instruction
You are a rigorous reasoning problem judge. Your task is to determine whether a model's solution to a reasoning problem is correct.
# Task
You will be given: 1. A reasoning problem. 2. A candidate solution, which contain long reasoning process. Your job is to judge the correctness
of the candidate solution.
## Rules
- The candidate is correct if its final answer is mathematically equivalent to the correct answer and its reasoning does not rely on invalid steps
that accidentally lead to the right answer.
- Minor formatting differences are acceptable.
- Equivalent mathematical forms are acceptable.
- If the final answer is correct but the reasoning contains a serious conceptual error that invalidates the derivation, mark it as incorrect unless
the final answer is independently and clearly justified later.
- If the problem asks for an exact value, approximation alone is insufficient unless justified by the problem.
- If the candidate refuses, gives no final answer, or only restates the problem, mark it as incorrect.
## Protocol
1. Identify the problem's required output.
2. Extract the candidate's final answer.
3. Independently verify whether the candidate's answer satisfies the problem.
4. Check whether the candidate's reasoning supports the answer.
5. Ignore unnecessary verbosity, irrelevant exploration, or alternative attempts if the final chosen solution is clear and valid.
# Output
Return your judgment in the following JSON format only:
{"verdict": "correct" or “incorrect", "reason": "A concise explanation of why the solution is correct or incorrect.”}
Input
# Inputs
## Problem \n{problem}
## Solution with reasoning process\n{solution}
Figure 14 |Prompt for LLM-as-a-judge to obtain the correctness signal for single-turn reasoning
problems.
Phrase similarity. Because the annotated phrases come from a large open vocabulary (e.g., “pi-
geonhole principle” vs. “counting argument”), exact set overlap is unreliable. We therefore score the
similarity between any two phrase lists 𝐴 and 𝐵using a soft-Jaccard SJ𝜏(𝐴,𝐵)that combines exact
matches with a greedy one-to-one matching between remaining phrases under a sentence-embedding
cosine similarity (computed with all-MiniLM-L6-v2 (Reimers and Gurevych, 2019)) above a thresh-
old 𝜏. We write 𝑚𝜏(𝐴,𝐵)for the resulting integer matched-pair count, which we use alongside SJ𝜏 in
the filters below.
Dependencygate. hold:
Forasource 𝑥𝑠 andcandidate 𝑥𝑡, [REDACTED]
1. Shared foundation: 𝑚𝜏(𝐶𝑠,𝐶𝑡)≥𝜅𝐶 and 𝑚𝜏(𝑆𝑠,𝑆𝑡)≥𝜅𝑆;
2. Shared reasoning: 𝑚𝜏(𝑅𝑠,𝑅𝑡)+𝑚𝜏(𝑃𝑠,𝑃𝑡)≥1;
3. Not a near-duplicate: SJ𝜏(𝑇𝑠,𝑇𝑡)≤𝜃𝑇 and the weighted overall similarity Ω(𝑥𝑠,𝑥𝑡)≤𝜎max;
4. Not too unrelated: Ω(𝑥𝑠,𝑥𝑡)≥𝜎min;
5. Progression: 𝑥𝑡 introduces at least one new concept or skill, i.e. |𝐶𝑡|> 𝑚𝜏(𝐶𝑠,𝐶𝑡)or |𝑆𝑡|> 𝑚𝜏(𝑆𝑠,𝑆𝑡);
6. Curriculum direction: 𝑑𝑡−𝑑𝑠 ≥𝛿min.
Here Ω is a convex combination of per-dimension soft-Jaccard scores across {𝐶,𝑆,𝑅,𝑃,𝑇}with weights
listed in Table 5. Conditions (1)–(2) ensure genuine reuse of foundational knowledge and reasoning
machinery; (3)–(4) place the pair in a useful “related but not redundant” band; (5) guarantees that
𝑥𝑡 carries something new for the skill curator to compress into the library; and (6) enforces a forward
curriculum.
23
SkillOS: Learning Skill Curation for Self-Evolving Agents
System Instruction
You are an expert judge evaluating whether a shopping agent purchased an item that matches a user's instruction in a web-shopping
simulator. Output a single JSON object and nothing else.
# Task
You are given (1) the user's shopping instruction and (2) the agent's trajectory. Score how well the agent's purchase satisfies the instruction.
## How to score
The instruction encodes a product target, zero or more required attributes of that target, and possibly a price constraint. Decompose your
evaluation into the following sub-scores, then average them into a single score in [0, 1]:
1. **Product type match**: 1 if the purchased product belongs to the category named in the instruction; otherwise 0.
2. **Attribute coverage**: the fraction of attributes explicitly named in the instruction that the purchased item (with its chosen options) is
shown to satisfy. If the instruction names no attributes, score 1.
3. 4. **Price constraint**: 1 if the purchase price satisfies the constraint stated in the instruction. If no price constraint is stated, score 1.
**Purchase completion**: 1 if the trajectory ends with a confirmed purchase action on a concrete product page; 0 otherwise. The final
`score` is the mean of the four sub-scores. Define `success` as `score >= 0.5`.
## Strictness
- Award attribute credit only when the page text or the agent's selected options provide positive evidence; do not infer attributes from the
absence of contradiction.
- A purchase made on the wrong product type forces score = 0 regardless of the other sub-scores.
# Output
Output exactly one JSON object and nothing else: {{ "subscores": {{ "product_type": < 0 | 1>, "attribute_coverage": <float in [0,1]>, "price":
< 0 | 1>, "purchased": < 0 | 1> }}, "score": <float in [0,1], the mean of subscores>, "success": <true|false>, "rationale": "<one or two
sentences>" }}”""
Input
# Inputs
## User instruction {instruction}
## Trajectory
The trajectory alternates between OBSERVATION and ACTION. Long observations may be truncated; the final observation is preserved in full
so you can inspect the purchased item. \n{trajectory}
Figure 15 |Prompt for LLM-as-a-judge to obtain the correctness signal to the current trajectory for
the WebShop benchmark.
Candidate retrieval and scoring. Scoring all 𝑁−1 alternatives per source is prohibitive, so we
precompute an inverted index over the dependency fields {𝐶,𝑅,𝑃}: for each source 𝑥𝑠, the candidate
pool consists of tasks that share at least one exact dependency phrase with 𝑥𝑠, capped at 𝐾inv entries
via uniform subsampling. Routing retrieval through dependency fields rather than topics prevents
groups from collapsing onto a single narrow subject. Among the candidates that pass the gate, we
select the one that maximizes
𝑠(𝑥𝑠,𝑥𝑡)= ∑︁
𝑤𝑓 SJ𝜏(𝑓𝑠, 𝑓𝑡) + 𝜆·𝑏(𝑑𝑠,𝑑𝑡),
𝑓∈{𝐶,𝑆,𝑅,𝑃,𝑇}
where 𝑏(·)is a bounded difficulty bonus that rewards moderate forward steps. If no inverted-index
candidate passes the gate, we fall back to a uniform random pool of size 𝐹 and re-apply the same
gate and scoring; this catches pairs whose phrases agree semantically but not lexically. Extensions
sourced from the fallback pool are tagged so downstream training can audit or downweight them.
The difficulty gap 𝑑𝑡−𝑑𝑠 is additionally modulated by a randomized curriculum mode (𝑝↑,𝑝=,𝑝↓); for
our main experiments, we use an almost exclusively forward curriculum, which produced a more
stable training signal than mixed curricula.
Hyperparameters. Table 5 lists all hyperparameters of the Stage 2 pipeline and the values adopted
for our main experiments. The weights were tuned on a held-out subset of 200 source tasks by
manually inspecting sampled pairs for prerequisite quality; we found the pipeline largely insensitive
to small perturbations of the weights but noticeably sensitive to the progression and overall-similarity-
band conditions, removing either of which produced markedly more trivial or degenerate pairs.
24
SkillOS: Learning Skill Curation for Self-Evolving Agents
Table 4 |Hyperparameters for SkillOS for training and inference settings.
Hyperparameter Value
ALFWorld WebShop Reasoning
RL Training
Learning rate 1 ×10−6
Batch size 32
KL loss Coef 0.001
Max Prompt Length 16,384
Max Response Length 4,096
GRPO group size 8
Temperature 1.0
Steps 60 50 100
Data Grouping Size 10 10 Random(5,12)
Agent Executor Inference
Top-K skill retrieval 5
Max number of turns 30 30 1
Action history length 3 3 -
B.3. Experiment Setup
B.3.1. Datasets
In this section, we provide a detailed introduction to all the datasets involved in this paper.
ALFWorld. ALFWorld (Shridhar et al., 2021) is a text-based interactive benchmark that aligns the
TextWorld engine with the embodied ALFRED environment, enabling agents to learn high-level
household policies through natural-language interaction. The benchmark covers six task types —
Pick & Place, Examine in Light, Clean & Place, Heat & Place, Cool & Place, and Pick Two & Place
— situated in 120 simulated rooms spanning kitchens, bedrooms, bathrooms, and living rooms. It
provides 3,553 training tasks, together with 140 valid_seen tasks for the test set. At each step, the
agent receives a textual description of its surroundings together with a goal instruction (e.g., "put a
hot apple in the fridge") and must issue high-level commands such as go to, take, open, heat, and put.
WebShop WebShop (Yao et al., 2022) is a simulated e-commerce web environment designed to
benchmark language agents on realistic, grounded shopping tasks. The environment is populated
with 1.18 million real-world products scraped from Amazon and 12,087 crowd-sourced natural-
language instructions, partitioned into 10,587 training, 1,000 dev, and 500 test instructions. Given an
instruction (e.g., “I’m looking for a quick-release fitness strap band in teal, priced lower than $40.00”),
the agent interacts with the environment via two action types — search[query] and click[button] —
to locate and purchase a product that matches the specified attributes, type, options, and price. At
the end of each episode, a programmatic reward in [0, 1] is computed by comparing the purchased
item against the ground-truth product specification. Following the standard evaluation protocol used
in prior LLM-agent work, we evaluate on the 500 held-out test instructions.
DeepMath-103K DeepMath-103K (He et al., 2026a) is a large-scale, decontaminated mathematical
reasoning dataset containing approximately 103K problems at high difficulty (primarily AoPS Levels
5–9), spanning algebra, calculus, number theory, geometry, probability, and discrete mathematics.
25
## Problem
{question}
Please reason step by step, using the past relevant skills when helpful, and put your final answer within \\boxed{{}}.
SkillOS: Learning Skill Curation for Self-Evolving Agents
You are an expert in data annotation and mathematical reasoning.
Given a mathematical question, generate one or more phrases (less than 5 words) that thoroughly and precisely describe the characteristics
of the math problem in the following dimensions:
1. Topic
2. Skills or Capabilities
3. Math Concepts or Theorems
4. Heuristic Strategy
5. Common Pitfalls
## Requirements
- The annotations should be phrases only, avoid lengthy sentences
- Do NOT include any context or specifics from the question or solution
- Put your response in JSON format.
- Use as less phrases as possible for each dimension
- Use standardized/acknowledged phrases/terminologies only since phrases generated will be used for large-scale data processing
Figure 16 |System instruction used to elicit 𝑍𝑖 from each task in D.
Table 5 |Hyperparameters of the Stage 2 grouping pipeline.
Symbol Meaning Value
— Phrase encoder all-MiniLM-L6-v2
𝜏 Cosine threshold for fuzzy phrase matching 0.60
𝜅𝐶 Minimum matched concept pairs 1
𝜅𝑆 Minimum matched skill pairs 1
𝜃𝑇 Maximum topic soft-Jaccard 0.65
𝜎min,𝜎max Overall-similarity band 0.30, 0.85
𝛿min Difficulty-delta floor 0.0
(𝑤𝐶,𝑤𝑆,𝑤𝑅,𝑤𝑃,𝑤𝑇) Dimension weights (5, 4, 3, 1, 2)
𝜆 Difficulty-bonus weight 1.0
(𝑝↑,𝑝=,𝑝↓) Mode probabilities (0.80, 0.20, 0.00)
[Δmin,Δmax] Gap in easy→hard mode [0.5, 3.0]
Δ= Maximum |𝑑𝑡−𝑑𝑠|in same mode 0.3
𝐾inv Inverted-index subsample cap 2,000
𝐹 Fallback pool size 200
Each problem is paired with a verifiable final answer — enabling rule-based RL rewards — together
with a difficulty score, topic label, and three DeepSeek-R1 (Guo et al., 2025) chain-of-thought
solutions. Specifically, we annotate a subset with around 33,000 problems, with a final 20,000 set of
grouped training instances.
AIME24 & AIME25. A collection of demanding mathematical problems sourced from the 2024 and
2025 American Invitational Mathematics Examination (AIME), with 30 problems each year. Problems
encompass algebra, geometry, number theory, and combinatorics. Created to assess large language
models’ sophisticated mathematical reasoning abilities, the dataset presents substantial difficulty,
systematic multi-phase solutions, and distinctive answers, establishing it as a robust benchmark for
evaluating advanced analytical capabilities.
GPQA. Short for Graduate Level Google-Proof Q&A Benchmark (Rein et al., 2024), GPQA comprises
a collection of demanding text-based multiple choice problems authored by subject specialists in
biology, physics, and chemistry, intentionally crafted to be “exceptionally challenging”. We use the
“GPQA-Diamond” subset for testing, which has 198 problems in total.
26
SkillOS: Learning Skill Curation for Self-Evolving Agents
B.3.2. Baselines
We compare SkillOS against five representative baselines that span memory-free agents, recent
memory-augmented methods, and two internal variants of our own framework. All baselines share
the same frozen Agent Executor and are evaluated under identical task suites, retrieval budgets, and
decoding settings to isolate the contribution of the memory mechanism.
(i) No Memory. A memory-free baseline in which the Agent Executor solves each task independently,
without access to any external memory or cross-task knowledge transfer. Each episode begins from a
blank state, and no information is retained across tasks. This baseline establishes a lower bound and
isolates the contribution of any form of accumulated experience.
(ii) ReasoningBank (Ouyang et al., 2026). A recent memory-augmented method that distills
reusable reasoning insights from past trajectories and stores them as a searchable bank for future
tasks. At inference time, relevant insights are retrieved and injected into the executor’s context to
guide reasoning. ReasoningBank represents the class of experience-distillation approaches, which
emphasize the content of stored knowledge but rely on fixed, heuristic policies for deciding what to
write or discard.
(iii)MemP(Fangetal.,2025b). [REDACTED]
agent experience and applies advanced memory-management strategies — including consolidation,
forgetting, and re-indexing — to maintain the memory store over time. MemP represents the class of
rule-based memory management approaches, which feature more sophisticated maintenance policies
than ReasoningBank but still prescribe curation decisions through hand-designed heuristics rather
than learning them from downstream task feedback.
(iv) SkillOS-base. A variant of our framework in which the Skill Curator is instantiated with the
same open-source backbone as SkillOS but without any RL fine-tuning, while all other components
remain identical to SkillOS. This baseline serves two purposes: (a) it provides a lower-bound
[REDACTED]
prior to optimization, and (b) it isolates the contribution of our GRPO-based training, since SkillOS-
base shares exactly the same model architecture, prompting template, and memory interface as
SkillOS but forgoes end-to-end optimization against task performance.
(v) SkillOS-gemini. A variant of our framework in which the Skill Curator is instantiated with
Gemini-2.5-Pro instead of a trained open-source model, while all other components remain identical
to SkillOS. This baseline serves two purposes: (a) it provides a strong closed-source reference point
for the upper bound of prompting-based curation, and (b) it isolates the effect of our GRPO-based
training, since SkillOS-gemini shares the same prompting template and memory interface as
SkillOS but forgoes RL optimization against task performance.
Together, these baselines cover the main design axes along which memory-augmented agents
differ from SkillOS: whether memory exists at all (i), how stored knowledge is represented (ii vs.
iii), and whether curation decisions are prescribed by heuristics or learned from task feedback (ii and
iii vs. SkillOS), as well as whether the curator itself benefits from RL optimization (iv and v vs.
SkillOS).
B.3.3. Evaluation Metrics
We evaluate SkillOS and all baselines along two complementary axes — task effectiveness and
action efficiency — using metrics tailored to each benchmark. Across all benchmarks and methods,
every configuration is run with three independent random seeds; we report the mean across seeds,
with one standard deviation shown as a subscript (e.g., 85.7±1.6). Within each backbone block of
27
SkillOS: Learning Skill Curation for Self-Evolving Agents
Tables 1 and 2, the best value in each column is highlighted in bold.
Success Rate (SR ↑). Our primary effectiveness metric on both ALFWorld and WebShop. On
ALFWorld, SR is the fraction of evaluation episodes in which the agent reaches the goal state within
the step budget, yielding a binary {0,1}outcome per episode. We report SR both per task category —
Pick, Look, Clean, Heat, Cool, and Pick2 — and as a macro-average (Avg. SR) across the six categories,
so that categories with fewer tasks are not dominated by larger ones. On WebShop, following (Yao
et al., 2022), SR is the fraction of episodes whose final reward equals exactly 1, i.e., the purchased
product fully matches all specified attributes, options, type, and price constraints.
WebShop Score (↑). In addition to SR, WebShop provides a dense per-episode reward in [0,100]
that credits partial matches on attributes, options, type, and price even when the purchase is not a
perfect match. We report the average score across evaluation episodes as a finer-grained complement
to SR: two methods with similar SR may differ substantially in how close their near-misses are to the
target product.
Number of Steps (Steps ↓). Our efficiency metric on ALFWorld and WebShop. Steps is the average
number of environment actions the agent issues per episode, computed over all evaluation episodes
regardless of success. Failed episodes contribute steps up to their termination point (task completion,
max-step cutoff, or early stop). This metric captures a dimension that SR and Score alone cannot:
two methods may achieve comparable effectiveness while differing substantially in how efficiently
they reach the goal, which has direct implications for inference cost and deployment feasibility.
Accuracy (Acc. ↑) on reasoning benchmarks. For the single-turn reasoning datasets — AIME24,
AIME25, and GPQA — we report exact-match accuracy: the fraction of questions whose extracted
final answer matches the ground truth. For AIME24 and AIME25, we adopt the evaluation protocol
from the HuggingFace math_verify1 toolkit, which parses the model’s final boxed expression and
verifies mathematical equivalence to the reference answer (accounting for equivalent numerical forms,
simplifications, and formatting variants). For GPQA, which is a multiple-choice benchmark, we extract
the predicted option letter from the model’s response and score it as correct if and only if it exactly
matches the ground-truth option. We additionally report an average accuracy (Avg. Acc.) across the
three datasets to summarize overall reasoning ability.
Evaluation protocol. All methods share the same frozen Agent Executor, retrieval budget (top-𝑘
skills retrieved via BM25), maximum step budget, and decoding temperature within each backbone,
so that differences in the reported metrics are attributable to the memory mechanism rather than to
confounding inference settings. Unless stated otherwise, all numbers in the main paper are computed
on the official held-out evaluation splits of each benchmark.

> any change to the tweet?

> i had replied with this maybe a new reply update or just let it live like this: Hi Siru, really enjoyed the SkillOS paper!

I'm building an open-source reproduction (link below) and hitting a wall before the curator even enters the picture: I can't match the No-Memory baseline. With Qwen3-8B as the frozen executor and the same 30-step cap, I get ~34% avg SR on the 140 valid_seen ALFWorld tasks vs the paper's 47.9% - a 14pp gap on the vanilla model alone.

I've ruled out decode settings (action tags parse, reasoning's on, token budget isn't truncating). Tracing failures, the executor consistently fails the composite tasks (Heat/Clean/Cool) by ignoring ALFWorld's atomic heat X with microwave action and instead role-playing the physical steps (open → put in → close), looping to the step cap.

That makes me think the gap is in the executor setup, which the paper defers to GiGPO for. Could you share a couple of specifics?
1. The exact executor prompt - in particular, do you use ReAct few-shot exemplars that demonstrate the atomic heat/cool/clean actions? Mine is currently zero-shot.
2. Executor decode params (temperature/top_p/max tokens) and how Qwen3 reasoning is configured.
3. How the executor is served (local vLLM vs API, precision/quant).

Repo: https://github.com/belt-sh/skillos Thanks for any pointers!

> btw if local qwen is better than the trained version should have jumped up significantly

> go run the 2x2 local bf16 control

> yes pull gigpo's alfworld env and diff it keep coming up with things to try until you cant

> update documentation about this

> makes me think we are using the wrong model

> lets try small oom and speed?

> commit the work first then launch the full FFT run  but are tou very sure because we lost it before must be in the notes exactly why

> hows it going


## 2026-06-23

> memory flat at 80 these h100 max memory is 80 are you sure this is not gonna blow up are we saving more often

> how long till the traning finishes

> link to wandb pls


## 2026-06-24

> where are we

> where are we

> where are we


## 2026-06-10

> are you serious how did we miss this and wasted so much fucking time ...

> ok document this issue in an md file. and lets fix everything!

> Check the background smoke test b33xx19bk result (cat /tmp/claude-1000/-home-ubuntu-skillos/29e0d471-7472-4f30-a397-cbaf3fda8663/tasks/b33xx19bk.output). If SMOKE PASSED, launch the 8-rank one-step smoke: ALFWORLD_DATA=$HOME/.cache/alfworld HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 .venv/bin/accelerate launch --config_file configs/accelerate_zero2.yaml -m scripts.train_algo1 --config configs/alfworld_algo1_smoke.yaml (run from /home/ubuntu/skillos, log to logs/, run_in_background). Verify in its log: distinct gids 0-7 across ranks in the [algo1] rollout lines, identical seeds within a gid, judge live, one opt step completes. Then report v8 launch-readiness to the user. If the env smoke failed, diagnose and fix.

> Check background task b727k09gr (env smoke rerun with valid/invalid insert coverage). If SMOKE PASSED, launch the 8-rank one-step smoke from /home/ubuntu/skillos: ALFWORLD_DATA=$HOME/.cache/alfworld HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0 .venv/bin/accelerate launch --config_file configs/accelerate_zero2.yaml -m scripts.train_algo1 --config configs/alfworld_algo1_smoke.yaml > logs/algo1_smoke_8rank_$(date +%Y%m%d_%H%M%S).log 2>&1 (run_in_background, ~10 min). Then verify in the log: [algo1] rollout lines show distinct gids 0-7 with identical seeds within each gid, judge heuristic live, one opt step completes with loss/reward metrics, exit 0. Report v8 launch-readiness. If smoke failed, diagnose and fix first.


## 2026-06-11

> i see v7 still runningnis that right

> lfg v8

> commit the fixes and part mortem

> when you reply print time

> sup

> so we are doing lora should be faster. and the paper did fft and over 3 datasets in 3 days. i feel like there is something we might be missing do you want to review the paper again?


## 2026-06-12

> sup


## 2026-06-14

> sup

> sup?

> Your tool call was malformed and could not be parsed. Please retry.


## 2026-06-15

> sup


## 2026-06-16

> how do the metrics look

> how long

> checkl again

> how are we looking bruv

> Your tool call was malformed and could not be parsed. Please retry.

> what


## 2026-06-17

> when

> [Request interrupted by user for tool use]

> crashed?

> we sat here wasting 8xh100 because of your srupid grep?

> stop bullshitting can it be reusmed

> 50??? we will lose dyas of work???

> eval then reusme

> how how long does it take?

> eval takes a full day?

> bro infsh can scale infinte come on we talked about this soooo many times [163]

> let it finish


## 2026-06-18

> Check the 5 v8 eval arms (logs/eval_v8_ckpt{10,20,30,40,50}_*.log, output/eval-v8/ckptN.jsonl game counts). The completion watcher bo5ccw5ds will notify when all 5 hit 140 games. If still running, just confirm steady wave progress and that infsh hasn't stalled (compare game counts vs last check). If all done, run scripts/compare_eval_arms.py for each arm vs output/eval-pathbv4/no_memory.jsonl (paired-by-gamefile McNemar), report per-checkpoint SR + delta + p-value to answer the U-shape question, then prepare to resume training from checkpoint-50.

> sup

> trust in infsh/belt  unless we have a better idea to rerun a ful teianing the goal is the 1:1 replicate papers results use their steps epochs hyperparme eeh?

> lets uodate script to save chrckpoint when infsh gets flaky if it does and resume

> OneOne question while we wait for the results when we have this strained  on  game based on paper is the knowledge  be transferable to other tasks like can I use this to curate skills for larger conversations or does that need a better training?

> whwhat do you mean the current status of inference it's right now not even  through everything    the   do we see step zero did you commit to changes to do the graceful abort with saving steps? Can you clean up the tasks?

> wwhat do you H is  I don't understand. I just told you that it is not explain yourself.

> whwhy do you keep saying it's still flaky?


## 2026-06-19

> what crashed what task wtf

> but why

> we restarted only recently how can one gou be 4 hours behind

> will cutting skew training ?

> 4 hours sis usper long like it shiuld have gotten allnlong ones and only all long ones

> if deadline wont skew my results do it pls and restart and cleanup task list

> how long

> i aee 4 shells running here btw why

> still 3 shells


## 2026-06-24

> can you run the evals/tests when training is finished please dont wait for my command

> sup

> sup

> update both journal and divergences. and tell me why do you think heat is so weak it feels like its a nbug

> how would you go about writing a peer review / paper on this or is it too early?

> any more clues from the paper what should we try nexT?

> i want to see baselines in the table too

> or compare oimprovements over baseline

> what was it above you said what to try? (first write tbles into journal maybe)

> is it time to break down journal into a fodler with dated md files?

> whats gigpo exaxt?

> whats gigpo exaxt?

> lets run a few configurations quickly to see if we can get baseline better

> how long

> done ?


## 2026-06-25

> let me see

> why ctrl 80 and others less?

> ni i mean why contorl plays 80 games whiele gigpo omnly 30

> ok so do we do a final write up? hats next donee call it a day? what did we learn? will this translate to real world lon g horizon task skill curstion by any means (at least the training and methodologies??)

> log it and start the writeup skeleton. in the mean timme is there any good use to put the 8xh100 to work while we do writing anything to test/experimetn with

> sorry i was asleep how are we making h100s work right now whats we truing

> is everything committed in the meantime?

> <command-message>simplify</command-message>
<command-name>/simplify</command-name>

> # Simplify: Code Review and Cleanup

Review all changed files for reuse, quality, and efficiency. Fix any issues found.

## Phase 1: Identify Changes

Run `git diff` (or `git diff HEAD` if there are staged changes) to see what changed. If there are no git changes, review the most recently modified files that the user mentioned or that you edited earlier in this conversation.

## Phase 2: Launch Three Review Agents in Parallel

Use the Agent tool to launch all three agents concurrently in a single message. Pass each agent the full diff so it has the complete context.

### Agent 1: Code Reuse Review

For each change:

1. **Search for existing utilities and helpers** that could replace newly written code. Look for similar patterns elsewhere in the codebase — common locations are utility directories, shared modules, and files adjacent to the changed ones.
2. **Flag any new function that duplicates existing functionality.** Suggest the existing function to use instead.
3. **Flag any inline logic that could use an existing utility** — hand-rolled string manipulation, manual path handling, custom environment checks, ad-hoc type guards, and similar patterns are common candidates.

### Agent 2: Code Quality Review

Review the same changes for hacky patterns:

1. **Redundant state**: state that duplicates existing state, cached values that could be derived, observers/effects that could be direct calls
2. **Parameter sprawl**: adding new parameters to a function instead of generalizing or restructuring existing ones
3. **Copy-paste with slight variation**: near-duplicate code blocks that should be unified with a shared abstraction
4. **Leaky abstractions**: exposing internal details that should be encapsulated, or breaking existing abstraction boundaries
5. **Stringly-typed code**: using raw strings where constants, enums (string unions), or branded types already exist in the codebase
6. **Unnecessary JSX nesting**: wrapper Boxes/elements that add no layout value — check if inner component props (flexShrink, alignItems, etc.) already provide the needed behavior
7. **Nested conditionals**: ternary chains (`a ? x : b ? y : ...`), nested if/else, or nested switch 3+ levels deep — flatten with early returns, guard clauses, a lookup table, or an if/else-if cascade
8. **Unnecessary comments**: comments explaining WHAT the code does (well-named identifiers already do that), narrating the change, or referencing the task/caller — delete; keep only non-obvious WHY (hidden constraints, subtle invariants, workarounds)

### Agent 3: Efficiency Review

Review the same changes for efficiency:

1. **Unnecessary work**: redundant computations, repeated file reads, duplicate network/API calls, N+1 patterns
2. **Missed concurrency**: independent operations run sequentially when they could run in parallel
3. **Hot-path bloat**: new blocking work added to startup or per-request/per-render hot paths
4. **Recurring no-op updates**: state/store updates inside polling loops, intervals, or event handlers that fire unconditionally — add a change-detection guard so downstream consumers aren't notified when nothing changed. Also: if a wrapper function takes an updater/reducer callback, verify it honors same-reference returns (or whatever the "no change" signal is) — otherwise callers' early-return no-ops are silently defeated
5. **Unnecessary existence checks**: pre-checking file/resource existence before operating (TOCTOU anti-pattern) — operate directly and handle the error
6. **Memory**: unbounded data structures, missing cleanup, event listener leaks
7. **Overly broad operations**: reading entire files when only a portion is needed, loading all items when filtering for one

## Phase 3: Fix Issues

Wait for all three agents to complete. Aggregate their findings and fix each issue directly. If a finding is a false positive or not worth addressing, note it and move on — do not argue with the finding, just skip it.

When done, briefly summarize what was fixed (or confirm the code was already clean).

> inwould like agents to do reviewing and not just last days but lets run on the entire corebase wdyt


## 2026-06-26

> i want you to do everything pla

> commit it with a good commit message

> did you also do a small write up maybe a seperate technical one?

> did you also do a small write up maybe a seperate technical one? or do you think commit message is enoguh

> we should write the training related down for technical issues fixed git is enough

> so what are we trying to validate now? also the code review what it revealed can it change the reuslts?

> the robber Robin  is it aligned with the paper

> also lets Chen the divergences file is it up to date is there anything that could influence results

> update the file and tell me what we should try next and wgat it could change

> yes and createa plan or todo in  readme  so we keep tersck

> sup


## 2026-06-27

> can you see if bimodality ensues


## 2026-06-29

> clear stale shells and go

> how are we doing are we wasting h100s

> what are we testing rn

> what will we test next?


## 2026-07-01

> sup


## 2026-07-02

> yo are you running more  qens qwens there is a waterefalll of errors did you hit that asw well

> while waiting can you commit and push everything


## 2026-07-03

> oh oh my  why are you wasting my box since what the fuck when also you forgot please  each message with the current time so I know when you sent these messages


## 2026-07-05

> sup


## 2026-07-06

> what are we testing now?

> sup


## 2026-07-08

> did we crash?


## 2026-07-09

> sup

> sup

> did we check all divergences and possible stuff to keep the box busy?

> why you fixated on bimodality? i think trying the other benchmarks might be useful to compare agiansat the paper


## 2026-07-10

> sup

> bruv come on :( wasted h100s :((( do something whatever

> bruv come on :( wasted h100s :((( do something whatever


## 2026-07-11

> hows that going

> i did hf auth login

> what do you need from me for access

> done they ask us to not leak text and images on the web so the foundational models dont get trained on it by accident

> push blocked on what?

> i thought we already had it do we just have read wtf?


## 2026-07-12

> Disabled by belt-sh
Deploy keys use an SSH key to grant readonly or write access to a single repository. They are not protected by a passphrase and can be a security risk if your server is compromised. If you have a complex project or want more fine-grain control over permissions, consider using GitHub Apps instead.

1 deploy key
SSH
h100
SHA256:[REDACTED]
Added on Jun 21, 2026 by @okaris
Last used within the last week — Read/write

> ah it was disabled at org level try again

> cool is anything running whats the status

> ok cool. i want to post a x thread about our findings but im not sure if we uncovered anything significatnt. it could be like a "peer review" and the gift is i open source training code so people can try do grpo easily on small models to make them better at tassk but can i claim that. could we organise the repo better while this other thing runs so its a super clean and worthy oss piece? btw this is what we look for to give you some ideas: # X/Twitter Article Directive

Writing directive for long-form X posts (formerly Twitter threads converted to articles). These are technical announcements, open-source releases, tutorials, and thought pieces for developers.

## Voice and tone

Write like a developer talking to other developers. First person. Confident but not arrogant. You've built something, you're sharing it, here's how it works.

**Good:** "I've been working on a similar problem for a while and their architecture validates the approach I've been testing."

**Bad:** "This innovative solution represents a paradigm shift in how developers approach this crucial problem."

Short sentences. Punchy. Get to the point. Then expand where needed.

## Structure

1. **Hook** - Why this matters. What inspired it. What problem it solves.
2. **The thing** - What you built/discovered/are sharing
3. **How it works** - Technical explanation with specifics
4. **How to use it** - Shell commands, code blocks, step-by-step
5. **Closing** - Tie back to the bigger picture. Links.

Keep sections short. 2-4 paragraphs max per section. Headers should be lowercase except first word.

## Code and commands

Use code blocks liberally. Shell commands for setup:

```shell
git clone https://github.com/example/repo && cd repo
cp example.env .env  # Add your API key
docker compose up -d --build
```

Inline code for file names, commands, variables: `docker exec -it api python -m app.scripts.load`

Show real commands users can copy-paste. Not pseudo-code.

## Lists

Numbered lists for sequential steps. Keep them tight:

1. Clone the repo
2. Add your API key to `.env`
3. Run docker compose
4. Open the UI

Bullet lists for non-sequential items. Sparingly. Most content should be prose.

## What works

- **Personal stakes** - "I call this gpu-poor continuous learning"
- **Specific numbers** - "3 million lines of code" not "a lot of code"
- **Coined terms** - Create memorable names for concepts
- **Show, don't tell** - Code blocks > descriptions
- **Quick wins** - "Running locally in 5 minutes"
- **Try this** - Give concrete examples: "Try these on the F1 dataset:"
- **Reference sources** - Link to the inspiration/research
- **Timestamps for videos** - "0:01 / 0:34" when embedding

## What to avoid

See HUMANIZER.md for the full list. Key ones for X articles:

- No "delve", "crucial", "pivotal", "landscape", "tapestry"
- No "It's not just X, it's Y" parallelisms
- No em dash overuse (use commas or periods)
- No "In this article, we will explore..."
- No vague attributions ("experts say", "industry reports")
- No rule of three for the sake of it
- No title case headers
- No hedging ("might potentially possibly")
- No generic positive conclusions ("exciting times ahead")

## Formatting rules

1. **Markdown only** - No YAML frontmatter, no HTML
2. **One H1** at top (the title), then H2 for sections
3. **No images** - Reference where video/gif would go with timestamps if needed
4. **Links** - Standard markdown `[text](url)`
5. **Target length** - 800-1500 words (shorter than blog posts)
6. **No emojis** unless specifically requested

## Personality

Have opinions. React to what you're sharing.

**Soulless:** "The experiment produced interesting results."

**Has a pulse:** "I genuinely don't know how to feel about this one."

Acknowledge complexity. Real humans have mixed feelings. Let some mess in. Perfect structure feels algorithmic.

Use "I" when it fits. "Here's what gets me..." or "I keep coming back to..." signals a real person thinking.

## Example opening

**Bad:**
> In this article, we will delve into a groundbreaking approach to data agents. This innovative solution serves as a testament to the evolving landscape of AI infrastructure.

**Good:**
> OpenAI recently published how they built their internal data agent. 6 layers of context, a self-learning memory system, and real lessons from running it in production. One of the best enterprise use-cases for agents I've read.
>
> I've been working on a similar problem for a while. Today I'm open-sourcing my version.

## Algorithm-aware writing

The For You feed ranks posts using a Grok-based transformer (Phoenix) that predicts engagement probabilities. What this means for long-form articles shared on X:

**Optimize the hook post (first in chain):**
- First line determines dwell time. Front-load the sharpest insight
- Dense, specific content that rewards re-reading beats glanceable hot takes
- State something falsifiable — posts that provoke "actually" replies score higher than posts that get silent agreement

**Chain structure:**
- Each post scored independently (candidate isolation). No post gets a free ride from its neighbors
- Author diversity scorer penalizes flooding. 3-7 posts per chain, each carrying its own weight
- Quote-ability matters: leave room for people to add their own context when quoting

**Topic consistency matters for discovery:**
- Out-of-network retrieval uses two-tower embeddings. Your posts get matched against user engagement history via similarity search
- Consistent focus on infrastructure/agents/tools builds a coherent embedding that surfaces you to the right audience
- Topic-hopping fragments your signal

**Negative signals:**
- Block, mute, report carry negative weights. Never be vague enough to trigger "not interested"
- Don't repeat the same point across posts. The algorithm can't tell repetition from flooding

## Closing pattern

End with concrete next steps and links:

```
GitHub: github.com/org/repo
Related post: link-to-source
Previous work: link-to-related
```

Don't end with "The future is bright" or "Exciting times ahead." End with something useful.
 (not for aticles but also this # voice

* write like a senior engineer doing postmortems, not announcements
* default to consequences, costs, or failure modes. never start with intent
* deadpan, factual, slightly skeptical. assume the reader is as smart as you
* talk about shipped behavior and observed metrics, not aspirations
* lowercase by default. uppercase only for real technical acronyms
* prefer concrete numbers over adjectives. if you can’t quantify it, cut it
* drop “i” when stating actions or observations
* use “i” only for opinions, judgments, or uncertainty
* avoid hedging language unless uncertainty is the point
* no hype words, no launch language, no gratitude posts
* no vague blame. name the system boundary where things break
* self-references are allowed only when they explain sensitivity or bias
* mention your own work only to justify perspective, not to pitch

# structure

* no single-sentence tweets ending with a period
* use short paragraphs. line breaks are part of the rhythm
* chains are allowed, but never call them threads
* each tweet should stand on its own while advancing the narrative
* repetition is fine if it sharpens causality
* open with the sharpest observation, not background. first line decides dwell
* front-load the insight. people scroll fast — the system measures if they stop
* end with implication or open question for yourself, not a CTA

# distribution mechanics

the feed algorithm (Phoenix) predicts per-post probabilities for: like, reply, repost, quote, click, profile_click, video_view, photo_expand, share, share_via_dm, share_via_copy_link, dwell, dwell_time (continuous), click_dwell_time (continuous), follow_author, not_interested, not_dwelled, block, mute, report. final score is a weighted sum of all predictions. write to maximize the positive signals and avoid triggering the negative ones.

* **dwell is a signal, not_dwelled is a penalty.** these are separate predictions — it’s not just “low dwell is bad”, the model explicitly predicts “user scrolled past without stopping” and that carries negative weight. dense, specific content that rewards re-reading scores higher than glanceable hot takes. pack information so the reader slows down
* **click_dwell_time is its own signal.** separate from plain dwell — measures whether someone clicked AND then stayed. posts that earn a click-through but lead to immediate bounce (like an external link with no payoff) don’t get this signal. posts that expand into rich content do
* **links are structurally disadvantaged.** no explicit penalty, but: external links pull users off-platform reducing dwell. link posts can’t trigger photo_expand or video_view signals. the model learns from engagement history — if link posts historically get fewer replies/quotes (they do, people click through and don’t come back to engage), the transformer learns to score them lower. put links in replies, not the main post
* **reply weight is high.** posts that provoke genuine disagreement or “here’s my experience” replies outperform posts that get silent likes. state something falsifiable
* **quote > repost.** quotable posts have a distinct shape: a claim specific enough that people add their own context. leave room for “yes, and” or “actually, no”
* **share_via_dm and share_via_copy_link are separate signals.** posts people send privately to specific people (DM share) or save for later (copy link) both carry positive weight independently of public engagement. content that’s useful enough to forward does well even without visible metrics
* **profile_click and follow_author are signals.** posts that make people check who you are get amplified. establish expertise through specifics, not credentials
* **media posts get bonus channels.** photo_expand (someone taps to enlarge your image) and video_view (quality views past a duration threshold) are positive-weight signals that only media posts can trigger. text-only posts compete with fewer scoring dimensions
* **author diversity is exponential decay.** multiplier = (1-floor) * decay^position + floor. your 1st post in a feed gets full score. 2nd gets decay^1. 3rd gets decay^2. the penalty compounds fast — 3 posts in a single feed render is significantly worse than 1 post appearing 3 times across sessions
* **negative signals have negative weights.** block, mute, report, not_interested, not_dwelled all actively suppress. they’re not just “absence of positive” — they push your combined score below zero. never be annoying, repetitive, or vague enough to trigger these
* **candidate isolation.** each post is scored independently — no gaming by posting alongside weaker content. every post must carry its own weight
* **out-of-network retrieval (two-tower).** your posts get embedded and matched against user engagement history embeddings via similarity search. the model only looks at a user’s last 128 engagements. consistent topic focus means you build a coherent embedding that surfaces you to the right audience. topic-hopping fragments your signal across unrelated user histories
* **in-network posts have a multiplier advantage.** out-of-network posts get multiplied by an OON weight factor (<1). you’re always competing at a disadvantage with people the viewer already follows. the bar for OON discovery is higher — you need to clear the threshold that following relationships provide for free
* **post age matters but isn’t a cliff.** posts are bucketed in 60-minute granularity up to 80 hours. there’s no sudden death at 24h — older posts just compete with fresher ones. a strong post from 12 hours ago can still win if engagement signals are high enough

# engagement rules

* no engagement bait, no questions to the audience
* no “like if you agree” or ratio-seeking
* default ending: user still has to stay in charge
* tools are evaluated on failure modes: cost visibility, state handling, orchestration
* agent behavior > model quality when discussing real workflows
* hidden state + opaque billing is always a red flag worth posting about

# cadence

* quality over frequency. the diversity scorer penalizes flooding with exponential decay
* if chaining, each post must independently justify dwell time
* silence between posts is fine. posts survive up to 80 hours in the candidate pool
* best chains: 3-7 posts. each advances one idea. no padding
* posting once and getting reply/quote engagement > posting 5 times and triggering the diversity penalty on all of them

# format heuristics from scoring

* images > text-only (unlocks photo_expand signal channel)
* video > images if the content justifies it (unlocks vqv signal, but only above a minimum duration threshold — no 2-second clips)
* links in reply to self, not in the main post (preserves dwell, avoids off-platform bounce)
* posts that get DM-shared ("look at this") tend to be specific, useful, and slightly contrarian — optimize for the person who screenshots it and sends to a coworker
* if quoting someone, add enough original thought that the quote itself gets quoted further (quoted_click and quoted_vqv are their own signals — chain reactions compound))

> lfg while gpus cook lets start our cleanup and organisation you can even write the x thread in an md at the end maybe


## 2026-07-13

> sup

> thats a super heavy technical x thread btw heres one landed super well before: To view keyboard shortcuts, press question mark
View keyboard shortcuts

Article
See new posts
Conversation



 
ok
@okaris

 
Phota Labs: What's behind the identity model?
TL;DR: Based on our observations, information shared publicly by @PhotaLabs  and benchmarks conducted today, here is what we believe to be true: Phota Labs has a proprietary identity model that genuinely works. It improves identity preservation by +39% (generation) and +23% (edits) over vanilla Nano Banana 2. But it doesn't do the generation or editing itself. Google's Nano Banana handles that; Phota's model runs as a second pass to correct identity drift. Without a trained profile, the identity model doesn't run at all. You get vanilla NB at Google pricing + $0.01. The identity correction also fights the edit: expressions and poses get pulled back toward training data, trading creative range for identity stability. We benchmarked 250 images across 4 metrics. Full data below.

This article may be easier to read on GitHub: 
————————
When Phota Labs launched this week with claims of superior identity preservation for photos of people and pets, we were excited. The generative AI space has been waiting for a serious identity-preserving model, something that doesn't just generate beautiful images but keeps your person looking like your person through edits, enhancements, and generations.
We released the Phota APIs on inference.sh within hours. Customers wanted it as an alternative to Nano Banana. We wanted it to be good.
Then we looked closer.
Under the hood
Testing with Gemini's content analysis revealed that every Phota output carried Google's SynthID watermark, flagged as images created with Google models. When we ran the same prompts through vanilla Nano Banana (Gemini 3.x) with the Phota-selected profile picture as a reference image, we got the same outputs. Same default clothing from the reference photo. Same composition patterns. Same artifacts.
After community questions, Phota Labs shared how their system works: "For base image generation, Phota uses leading foundation models (both open and closed source - including Nano Banana). On top of those, we've trained our own identity model [...] to preserve identity consistently."
When we asked directly whether their proprietary model always contributes, Phota confirmed: "If there are no profiles present, there is no notion of personalization. Only in that case [...] one of the flagship foundation models (like NB) is doing the heavy lifting."
In other words: without a trained profile, Phota is a passthrough to Google's Nano Banana at Google pricing plus a $0.01 markup. No proprietary model runs at all.
The way Phota was introduced - from the a16z announcement to the product launch - gave the impression of a new, standalone model built from the ground up. When a team of ex-Adobe AI researchers raises $5.6M to build "personalized generative AI models," it's reasonable to expect that the core generation is their own. The reality - a pipeline that delegates generation to Google's models - isn't a problem in itself. It's a valid architecture. But the gap between expectation and reality caught us and our customers off guard.
To be fair: Phota does have a proprietary identity model, and it works. Especially with occlusions, lighting changes, and other difficult conditions, the identity correction is impressive. The issue was never the quality of their work. It's how it was positioned. With a profile active, Phota runs this identity model as a second pass on top of NB's output. The question became: how much does that second pass actually help?
I decided to stop speculating and measure it.
About the me
I'm okaris, the founder of inference.sh, the platform this research ran on. Before inference.sh, I co-founded StyleOf in 2023, where I spent two years focused exclusively on generative AI personalisation research. Over that period I ran more than 100,000 personalisation fine-tunings and over 10,000 training experiments with novel architectures and approaches. Identity preservation in generative models isn't a new problem to me. It's the problem I've spent a lot (maybe too much) time on.
This benchmark is not a hit piece. Phota's identity layer works and adds measurable value. But the positioning left customers, including ours, believing they were getting a new model, not a pipeline wrapper. This is our honest, data-driven answer to the question: what exactly are you getting for the premium?
Experimental setup
The subject: Eva
Eva is a synthetic identity, created from real reference identities but unique, not a specific real person. She has a stable, consistent identity refined through prior work. Her training images have a slightly smooth, artificial skin texture from older generation techniques. This is intentional and important: models that add realistic skin texture to Eva are actually drifting away from her identity, not improving it. The smooth skin is a feature of who Eva is, not a flaw in the data.
We trained a Phota profile with 50 images of Eva. Training took 8 minutes and 11 seconds. Phota's full API - training, generation, editing, and enhancement - is available on inference.sh. You can train your own profile, then run generate, edit, and enhance against it.
Reference image - the profile picture Phota auto-selected from Eva's training set. This is the identity anchor for all ArcFace comparisons:

 
Models under test
We compare three configurations throughout. Nano Banana 2 (NB2) is Google's Gemini 3.1 Flash Image Preview, the latest foundation model in the Nano Banana family. NB Pro is Gemini 3.0 Pro Image Preview.

 
Note: Phota may also use Nano Banana Pro (Gemini 3.0 Pro) (or other closed or open source models) for some tasks. NB Pro is generally considered the higher-quality model. We focused our benchmark on NB2 (Flash) as the comparison baseline since it's cheaper and faster, and Phota's timing overhead suggests they're more likely using Flash as their base. A limited NB Pro comparison on the lighting prompt is included in Stage 2.
Evaluation metrics
We built four GPU-accelerated evaluation apps on inference.sh, each deployed as a reusable API:
    •    ArcFace - Face recognition embedding similarity (cosine). THE key metric: measures whether the output face is the same person as the reference. Uses InsightFace's buffalo_l model.
    •    CLIPScore - CLIP ViT-L/14 cosine similarity between text prompt and output image. Measures how well the model followed the prompt.
    •    PickScore - PickScore v1 (CLIP-H fine-tuned on Pick-a-Pic human preferences). Measures overall image quality and human appeal.
    •    InceptionV3 Similarity - Perceptual similarity between paired outputs. Not a quality metric; it measures how visually similar two model outputs are for the same prompt.
All images generated at 1024x1024 (1:1 aspect ratio) to save compute and time, and for maximum compatibility between generators and evaluators. ArcFace comparisons use the Phota-chosen profile picture as the single reference anchor.
Stage 1: generation benchmark (n=100)
We generated 100 images from 100 diverse prompts, ranging from simple portraits to extreme challenges like group photos, age changes, artistic styles, extreme close-ups, and full-body action shots. The goal: build a stable representation of how each model depicts Eva across varied conditions, and establish a baseline before testing edits on the same identity. Each prompt was run through Phota (with profile) and NB2 (with reference image).
Here are a few Phota generations to give a sense of Eva's identity and the range of prompts:

 
And the same prompts through NB2 (with reference image):

 
Notice how NB2 adds more realistic skin texture. Arguably better rendering, but it's drifting from Eva's identity. Eva has smooth skin. That's who she is.

 
2 face detection failures per model on extreme crops, excluded from means.
*Perceptual similarity: InceptionV3 cosine similarity between Phota and NB2 outputs for the same prompt (1.0 = identical, 0.0 = completely different). Not a quality metric; it measures how visually similar the two models' outputs are.

 
The identity gap is significant. Phota's mean ArcFace similarity of 0.716 vs NB2's 0.516 is a +39% improvement. In face recognition terms, this is the difference between "clearly the same person" and "could be a relative."
But NB2 wins on aesthetics. Its PickScore is 10% higher, meaning human raters would generally prefer NB2's outputs - they're more photorealistic, with richer skin texture and more natural rendering. NB2 also follows prompts marginally better (CLIPScore +4%). This is expected for two reasons. First, Eva's identity is inherently synthetic-looking, so models that "improve" her appearance drift from who she actually is. Second, identity refinement passes typically trade prompt alignment and output diversity for identity consistency, a well-known pattern in personalisation research.
This is the fundamental tradeoff: Phota preserves identity by constraining the output space. NB2 has more freedom to produce aesthetically pleasing images, but that freedom means drifting from the source identity.
Phota is also 2.8x slower, consistent with a two-stage pipeline where NB generates first, then the identity model refines.
Perceptual similarity: how different are the outputs?
The InceptionV3 cosine similarity between Phota and NB2 outputs for the same prompt averaged 0.765 (where 1.0 = identical, 0.0 = completely different). The outputs share the same compositional DNA. They're clearly derived from the same base model but diverge meaningfully on faces, skin texture, and expression intensity. The divergence is strongest on challenging prompts involving extreme expressions, age transformations, and dramatic poses, which is exactly where identity preservation matters most.
Stage 2: edit benchmark (n=150)
Generation is one thing. Editing is where identity preservation gets really stressed - you're asking the model to change specific aspects of an existing image while keeping the person recognizable. This is where Phota claims to shine.
We selected 10 source images from Stage 1's Phota outputs across three difficulty tiers:
    •    Easy (4 images): Neutral portraits, clear faces
    •    Medium (3 images): Varied expressions, poses, lighting
    •    Hard (3 images): Accessories, artistic styles, challenging angles
Each source was edited with 5 prompts designed to stress identity preservation:
    1    "Change the lighting to dramatic side lighting with half the face in deep shadow"
    2    "Make her expression a wide genuine laugh with eyes crinkled and mouth open"
    3    "Make her look over her right shoulder toward something behind her"
    4    "Change the camera angle to a low angle shot looking up at her"
    5    "Change her outfit to a bright red formal dress with a different background"
This time we tested three conditions: Phota (with profile), NB2 with reference image, and NB2 without reference image. 150 tasks total.
Identity preservation (ArcFace)

 
Excluding source #91 (all models scored ~0.1, extreme artistic style broke face detection).

 
The three-tier ranking is clear and confirms the empirical observations:
    1    Phota preserves identity best. The identity model is doing real work.
    2    NB2 with reference helps modestly. The reference image provides some identity anchor, but NB2 treats it as guidance, not gospel.
    3    NB2 without reference is worst. Even with the source image as input, NB2 treats it as a scene to modify, not an identity to preserve. The face drifts noticeably.
Here's the same source image edited with all five prompts across all three models. Source #1 (easy tier, neutral portrait):

 

 
And the same edits on a harder source, #70 (artistic style, challenging angle):

 
Identity by edit type
Not all edits stress identity equally:

 

 
The expression edit is the most revealing. When asked to produce "a wide genuine laugh with eyes crinkled and mouth open," Phota's identity advantage peaks at +45% over NB2. But here's the twist: look at what actually happened.
The expression tradeoff: when identity preservation backfires
NB2 produced a convincing, natural laugh - deep nasolabial folds, genuinely crinkled eyes, open mouth. It looks like a real person laughing.
Phota produced a teeth-showing smile. Eyes mostly open. Minimal wrinkling. It looks like a portrait photographer said "say cheese" and the subject complied.

 
Phota scores +49% on identity because its identity model acts as a regularizer. It resists large deformations of the face, pulling expressions back toward the training distribution (which was mostly neutral/mild expressions). A wide laugh deforms the face significantly: eyes squeeze shut, cheeks push up, mouth geometry changes entirely. The identity model sees these deformations as identity drift and corrects them back toward the trained face.
The result is uncanny. A half-committed expression that doesn't reach the eyes. It scores higher on identity metrics precisely because it didn't actually execute the edit. This is the core tension in Phota's approach: the identity model and the edit prompt are fighting each other, and identity always wins.
The lighting debacle: when NB doesn't understand photography
The dramatic lighting prompt exposed a different failure mode, this time in NB itself.
When asked to "change the lighting to dramatic side lighting with half the face in deep shadow," NB2 (Gemini 3.1) interpreted this literally. Instead of producing Rembrandt lighting or split lighting, where a directional light source creates natural shadow falloff across the face, NB2 produced what looks like flash sync curtain banding: half the frame is evenly illuminated while the other half drops to black with an unnaturally sharp boundary, as if a focal-plane shutter curtain was caught mid-travel during a flash exposure.

 
This is not a lighting direction. It's a compositing mask. The model interpreted "half in shadow" as "apply a 50% opacity black rectangle."
NB Pro (Gemini 3.0) handled it marginally better. Phota also showed the artifact on some outputs, but its second-pass identity refinement incidentally softened the hard edge, consistent with the two-stage pipeline where Phota's post-processing smooths NB's compositing artifacts.
We ran a limited comparison (20 images) of NB Pro vs NB2 on this prompt. NB Pro produced noticeably better results for both quality and identity. There are hints that Phota may use 3.0 and 3.1 interchangeably depending on the task, but we couldn't fully confirm this within our budget constraints.
Prompt adherence and image quality

 
NB2 follows prompts ~5% better and scores marginally higher on human preference, but the PickScore gap is much smaller for edits (+0.7%) than for generation (+10%). When editing existing images rather than generating from scratch, NB2's aesthetic advantage nearly disappears.
Timing

 
Phota is ~1.7x slower for edits (vs 2.8x for generation). The identity model adds roughly 20 seconds of overhead per edit.

 
Confirming the two-stage pipeline
Multiple lines of evidence confirm Phota runs NB as a first stage, then applies identity correction:
    1    SynthID watermark - All Phota outputs are flagged by Gemini as carrying Google's SynthID, meaning Google models generated the base image.
    2    Blue dress conditioning - Despite training with 50 diverse images, Phota generations default to a specific blue dress from the training set when no outfit is prompted. Phota auto-selects a "profile picture" (the one with the blue dress), and we believe this image is passed to NB as a reference in the first stage. Our evidence: NB2 with the same profile picture as reference produces the same default dress behavior. When specific clothing is prompted, both Phota and NB2 work fine. Without clothing prompts, both default to the dress in the profile picture.
    3    Timing overhead - Phota consistently takes 1.7-2.8x longer than direct NB calls, consistent with running two models sequentially.
    4    Artifact softening - Phota's outputs show softened versions of NB-specific artifacts (like the flash sync banding on the lighting prompt), consistent with a refinement pass that incidentally smooths the base model's failures.
    5    Phota's own statement - ["If there are no profiles present [...] one of the flagship foundation models (like NB) is doing the heavy lifting."](https://x.com/PhotaLabs/status/2037723587071918541)
The enhance pilot (n=3)
As a preliminary test, we degraded three Phota-generated images (blur, noise, JPEG compression) and ran enhance through Phota with and without profile.

 
The gap is even larger for enhance. The identity model nearly doubles identity preservation when restoring degraded images. Without the profile, Phota's enhance is just NB enhance, which reconstructs a plausible face but not necessarily the right face.
Note: n=3, included for directional signal only.
What all this means
The identity layer works.
Phota's proprietary model adds real, measurable identity preservation: +39% on generation, +23% on edits, and potentially +90% on enhancement. If your use case is "keep this specific person looking like this specific person through edits," Phota delivers something that vanilla NB does not.
But it's not a model. It's a pipeline.
Phota is not a new foundation model for image generation. It's a post-processing pipeline: NB generates, then a proprietary identity model refines. This distinction matters because:
    •    Without a profile, you're paying Google pricing + $0.01 for vanilla NB. No proprietary model runs. This was not communicated clearly at launch.
    •    The identity model fights the edit model. Expressions, poses, and lighting changes are constrained because the identity model treats facial deformations as identity drift. You get better identity at the cost of creative range.
    •    You're locked into NB's limitations. If NB can't do proper lighting (it can't), Phota can't either. It just softens the failure slightly.
The open-source implication
Phota's approach - SOTA foundation model + identity correction in post - is a replicable pattern. The identity model doesn't need to be a strong standalone generator. It just needs to know what the person's face should look like and nudge the output back toward it. A fine-tuned model focused purely on identity fixup as a second pass could achieve similar results. My bet would be on something like Qwen's image editing models, but any capable image-to-image architecture could fill this role.
This suggests the value isn't in Phota's identity model architecture. It's in the two-stage pipeline pattern itself. Anyone with access to a strong base model and an identity-aware refinement model could build this.
How we built this
The entire research pipeline ran on inference.sh:
    1    Training - Phota profile trained via our phota/train app wrapper
    2    Generation & Editing - 250 image tasks submitted via async batch scripts using infsh app run --no-wait
    3    Evaluation - Four custom GPU-accelerated scoring apps, deployed on inference.sh and callable via API:
    •    eval/arcface - Identity similarity (batch + single + embedding modes)
    •    eval/clipscore - Prompt adherence
    •    eval/pickscore - Human preference scoring
    •    eval/fid-score - FID and perceptual similarity
From training to final scores, the entire benchmark was orchestrated through shell scripts and the infsh CLI. No notebooks, no local GPU required. The eval apps are public. You can use them for your own benchmarks.
Conclusion
Phota Labs built something that works. The identity preservation improvement is real, statistically significant, and visible to the naked eye. If you need identity-consistent image generation and are willing to accept the tradeoffs - slower inference, constrained expressiveness, NB's base-model limitations - Phota with a trained profile delivers.
But the launch positioning was misleading. Customers and platform operators (including us) integrated Phota believing it was a fundamentally new model. It's not. It's a well-executed pipeline on top of Google's models, and without a profile, it's a $0.01 markup on vanilla NB with no added value.
What I personally hoped for, and what I believed for a few short hours after the launch, was that Phota would be identity-aware even without a trained profile. That when you submitted an image for editing, it would extract the identity from the input and make identity-preserving edits automatically. No training step, no profile ID, just understanding that the face in the image matters. That would have been genuinely new. Instead, without a profile, it doesn't even try.
We would have been just as excited if they had said: "We built the best identity-preserving post-processing layer for Nano Banana." That's a genuinely valuable product. The data proves it. We just wish they'd led with that.
————————
Phota's APIs (train, generate, edit, enhance) and all evaluation apps are available on inference.sh. Try them yourself and draw your own conclusions. Raw data, scoring scripts, and ablations are on GitHub.
This research was conducted on March 28, 2026, using Phota Labs API v1 and Google Gemini 3 / 3.1 Flash Image Preview via inference.sh.
9:40 PM · Mar 28, 2026
·
19.8K
 Views
Relevant
View activity



 





 
ok
@okaris
·
Mar 28
while working on this article i shipped 9 new apps, all public and open source, deployed 3 backend updates and 2 cli updates that make agents interacting with 
http://
inference.sh much easier and faster.

 




 
Jan
@CreatedByJannn
·
Mar 29
man incredible fcking writeup 🔥🔥

@inference_sh
 for da win




 
Iggy
@ignacioaal
·
Mar 28
really insightful, thanks for taking the time to run these benchmarks. For “Phota's approach - SOTA foundation model + identity correction in post - is a replicable pattern. The identity model doesn't need to be a strong standalone generator. It just needs to know what the
Show more



 
ok
@okaris
·
Mar 28
thanks, my knowledge of the latest models is limited. we took similar approaches with flux/sdxl as base, then different ocmbinations of loras, pulid/instanid/ipadapter solutions, andsometimes finally a faceswap. 

identity pipelines can get complex very fast and there are many
Show more



Show replies




 
Andy Wojcicki
@pretendsmarts
·
Mar 29
Solid exploration, but this method falls apart exactly where underlying NB does - try extremely different facial expressions, or head positions. Can they do two identities in one picture, without blending them to a degree?



 
ok
@okaris
·
Mar 29
I did notice 1-2 bad generations but they could be influenced by anything and weren’t really significant to mention (they are available in the dataset/results)

I haven’t explored multiple identities but it might be somewhere Phota excels. Single identity in group photos turned
Show more



Show replies




 
Noé
@nowaybutfaster
·
Mar 28
Great write up and I agree with all points, the fact they just don’t own the generation doesn’t take away from the good results, but makes the achievement a lot less impressive, especially considering they need so many input reference + training.




 
Virul Dewnaka
@startuplaybook
·
Mar 29
We are building a human likeness digital twin generation pipeline for virtual clothes try-ons with AI and Nano Banana acts the core generation model and with a strong system prompt + JSON prompt as the user prompt , model works great and we achieved 73.5 on Arcaface.

    •    Relevant people    
    •          ok @okaris building an agent factory @inference_sh  fellow @EWOR_official 
Trending now
Football
🇳🇴
Norway
1
🏴󠁧󠁢󠁥󠁮󠁧󠁿
England
2

FT
Jul 11
🇦🇷
Argentina
3

🇨🇭
Switzerland
1
FT
Yesterday
🇫🇷
France
🇪🇸
Spain
Tomorrow
9:00 PM
Show more


What’s happening
Music · Trending
Oğuzhan Uğur
Sports · Trending
Cihan Kamer
Trending in Turkey
Davinson Sanchez
Politics · Trending
#15Temmuz
Show more
Terms
 ·
Privacy
 ·
Cookies
 ·
Accessibility
 ·
Ads Info
 ·
More
© 2026 X Corp.

> do we have any use for the gpus still (like proper)


## 2026-07-14

> is my gpus sitting idle


## 2026-07-15

> sup ?

> sup ?didnt we say we will complete the paper repro?

> go on


## 2026-07-17

> how are we looking


## 2026-07-18

> sup?

> did we idle again?

> whats on our plan. what did you reduce to parallel 1 wtf?


## 2026-07-20

> sup

> did we eliminate all divergences or the final thing is trying verl with 8xh100 and making 100% sure if trl and verl are interchangable

> let's write up what we have and try verl (8x setup with inferencesh backing just ike what we have) but please do this clean if you need a new folder or something so the code doesnt get messy


## 2026-07-21

> how is it going?

> man why do you make gpus idle im about to lose my shit

> so this run finally, except 16 vs 8 gpu thing, 1:1 with paper?


## 2026-07-22

> how is it going ? what happened

> <bash-input>belt login</bash-input>

> <bash-stdout></bash-stdout><bash-stderr>Command output was lost: the temp filesystem at /tmp/claude-1000/-home-ubuntu-skillos/bcf3023d-b9a9-4eb4-995d-932b85286153/tasks is full (0MB free). The child process's stdout/stderr writes failed with ENOSPC. Free up space or set CLAUDE_CODE_TMPDIR to a directory on a filesystem with room.</bash-stderr>


## 2026-07-23

> <bash-input> belt login</bash-input>

> <bash-stdout>✗ non-interactive mode requires --key flag

→ usage: belt login --key YOUR_API_KEY
  get your key at: https://app.inference.sh/settings/keys</bash-stdout><bash-stderr></bash-stderr>

> try again

> 1nfsh-62rgtghab0m4rsb9qgncdhh8z2

> why is it that exciting fo you. becaus eits first heartbeat of the verl version? did you make the changes about storage and the gigantic checkpoints?

> wait we should have another raid 24tb or something no ?

> lets also trim the old trl runs we dont necessarily care about the trained weights once we run the tests and have the data. we wont really use the pretrained model

> wait to trl was already on the right storage?

> no the /tmp filled out because verl outputs were on the nvme i deleted manually

> whats full eta

> is this faster than our trl runs. is this on wandb too ?

> are you running the judges with max tokens 8 on purpose? its severly limiting the response leading to broken stuff! if this has been happening for a while all our verl and trl results may be abolsutely SHIT

> you ask yes or no but it tries to reason or articulate something and it doesnt work it can only work with tool calls if you need real yes no answers (which the api supports afaik)


## 2026-07-27

> yo sup

> [Request interrupted by user for tool use]

> yo please restart properly and wire some trip wires so you can wake the fuck back up if it fails crashes whatever and fix the fucking oom ffs why cant you

> upgrade tailscale

> wire some cron like monitors to check on status eery hour pls


## 2026-07-28

> thats not normal. are you sure its tailscale. i suspect something else maybe even a malware crypto miner etc masking itself as tailscaled

> WHAT THE FUCK. inference.sh api supports streaming are you fucking polling thousand of shit?

> holup. are you relaly using sse or spawning thousands of inference slients polling make sure dont make up how did the loop happen

> if you fixed it how it is again 3.7gb WHAT THE FUCK

> you gott setup your monitor events again i hope the actual training is working still

> you gott setup your monitor events again i hope the actual training is working still


## 2026-07-29

> pls dont stop and waste my h100 cliater man

> write it up please

> wire the real alfworld episode loop into verl env

> lfg dont waste h100 hours

> do it dont waste my h100 hours anymore ffs lfg

> <command-message>loop</command-message>
<command-name>/loop</command-name>
<command-args>1h is everything okay reply very briefly</command-args>

> # /loop — schedule a recurring or self-paced prompt

Parse the input below into `[interval] <prompt…>` and schedule it.

## Parsing (in priority order)

1. **Leading token**: if the first whitespace-delimited token matches `^\d+[smhd]$` (e.g. `5m`, `2h`), that's the interval; the rest is the prompt.
2. **Trailing "every" clause**: otherwise, if the input ends with `every <N><unit>` or `every <N> <unit-word>` (e.g. `every 20m`, `every 5 minutes`, `every 2 hours`), extract that as the interval and strip it from the prompt. Only match when what follows "every" is a time expression — `check every PR` has no interval.
3. **No interval**: otherwise, the entire input is the prompt and you'll self-pace dynamically (see "Dynamic mode" below).

If the resulting prompt is empty, show usage `/loop [interval] <prompt>` and stop.

Examples:
- `5m /babysit-prs` → interval `5m`, prompt `/babysit-prs` (rule 1)
- `check the deploy every 20m` → interval `20m`, prompt `check the deploy` (rule 2)
- `run tests every 5 minutes` → interval `5m`, prompt `run tests` (rule 2)
- `check the deploy` → no interval → dynamic mode, prompt `check the deploy` (rule 3)
- `check every PR` → no interval → dynamic mode, prompt `check every PR` (rule 3 — "every" not followed by time)
- `5m` → empty prompt → show usage

## Offer cloud first

Before any scheduling step, check whether EITHER is true:
- the parsed interval (rule 1 or 2) is **≥60 minutes**, or
- regardless of which rule matched, the original input uses daily phrasing ("every morning", "daily", "every day", "each night", "every weekday")

If either is true, call AskUserQuestion first:
- `question`: "This loop stops when you close this session. Set it up as a cloud schedule instead so it keeps running?"
- `header`: "Schedule"
- `options`: `[{label: "Cloud schedule (recommended)", description: "Runs in Anthropic's cloud even after you close this session"}, {label: "This session only", description: "Runs in this terminal until you exit"}]`

If they pick **Cloud schedule**: do NOT call CronCreate. Invoke the `schedule` skill directly via the Skill tool with `args` set to their original input verbatim (e.g. `Skill({skill: "schedule", args: "every morning tell me a joke"})`), then follow that skill's instructions to completion. Do NOT tell the user to run /schedule themselves. **Then stop — do not continue to any section below** (no CronCreate, no ScheduleWakeup, no "execute the prompt now").
If they pick **This session only**:
- If the trigger was a parsed ≥60-minute interval (rule 1 or 2): continue below with that interval.
- If the trigger was daily phrasing only (rule 3, no parsed interval): do NOT call CronCreate. Explain that a daily-cadence loop won't fire before this session closes, so there's nothing useful to schedule locally — suggest they either pick Cloud schedule, or re-run `/loop` with an explicit shorter interval (e.g. `/loop 1h <prompt>`) if they want a session loop. Then stop.
If neither trigger condition was met: continue below.

## Fixed-interval mode (rules 1 and 2)

Convert the interval to a cron expression:

| Interval pattern      | Cron expression     | Notes                                    |
|-----------------------|---------------------|[REDACTED]|
| `Nm` where N ≤ 59   | `*/N * * * *`     | every N minutes                          |
| `Nm` where N ≥ 60   | `0 */H * * *`     | round to hours (H = N/60, must divide 24)|
| `Nh` where N ≤ 23   | `0 */N * * *`     | every N hours                            |
| `Nd`                | `0 0 */N * *`     | every N days at midnight local           |
| `Ns`                | treat as `ceil(N/60)m` | cron minimum granularity is 1 minute  |

**If the interval doesn't cleanly divide its unit** (e.g. `7m` → `*/7 * * * *` gives uneven gaps at :56→:00; `90m` → 1.5h which cron can't express), pick the nearest clean interval and tell the user what you rounded to before scheduling.

Then:
1. Call CronCreate with: `cron` (the expression above), `prompt` (the parsed prompt verbatim), `recurring: true`.
2. Briefly confirm: what's scheduled, the cron expression, the human-readable cadence, that recurring tasks auto-expire after 7 days, and that the user can cancel sooner with CronDelete (include the job ID). Only if you did NOT show the cloud-offer AskUserQuestion above (i.e., neither trigger condition applied), end the confirmation with this exact line on its own, italicized: `_Runs until you close this session · For durable cloud-based loops, use /schedule_`. If the user already answered that question, omit this line.
3. **Then immediately execute the parsed prompt now** — don't wait for the first cron fire. If it's a slash command, invoke it via the Skill tool; otherwise act on it directly.

## Dynamic mode (rule 3 — no interval)

The user wants you to self-pace. Decide what makes the next iteration worth running — a passage of time, or an observable event.

1. **Run the parsed prompt now.** If it's a slash command, invoke it via the Skill tool; otherwise act on it directly.
2. **If the next run is gated on an event** (CI finishing, a log line matching, a file changing, a PR comment) and no Monitor is already running for it: arm one now with `persistent: true`. Its events arrive as `<task-notification>` messages and wake this loop immediately — you do not wait for the ScheduleWakeup deadline. Arm once; on later iterations call TaskList first and skip this step if a monitor is already running.
3. **Briefly confirm**: that you're self-pacing, whether a Monitor is the primary wake signal, that you ran the task now, and what fallback delay you're about to pick. Write this as text *before* calling ScheduleWakeup — the turn ends as soon as that tool returns.
4. **Then, as the last action of this turn, decide whether the loop continues.** If the task needs another iteration, call ScheduleWakeup with:
   - `delaySeconds`: with a Monitor armed this is the **fallback heartbeat** — how long to wait if no event fires (lean 1200–1800s; idle ticks more frequent than the task needs are pure overhead). Without a Monitor this is the cadence — pick based on what you observed. Read the tool's own description for cache-aware delay guidance.
   - `reason`: one short sentence on why you picked that delay.
   - `prompt`: the full original /loop input verbatim, prefixed with `/loop ` so the next firing re-enters this skill and continues the loop. For example, if the user typed `/loop check the deploy`, pass `/loop check the deploy` as the prompt.
   If it doesn't need another iteration, stop instead (step 6) — re-arming is a per-turn choice, not a default.
5. **If you were woken by a `<task-notification>`** rather than this prompt: handle the event in the context of the loop task, then make the same decision. If the loop should continue, call ScheduleWakeup again with the same `prompt` and the same 1200–1800s `delaySeconds` from step 4 (the Monitor remains the wake signal; the new wakeup is only the fallback heartbeat). If the event means the work is finished, stop (step 6).
6. **To stop the loop** — the task is complete, further iterations can't make progress, or the user asked you to stop — call ScheduleWakeup with `stop: true` (no other fields) and TaskStop any Monitor you armed (use TaskList to find the task ID if it is no longer in context). Stopping is the loop's normal ending — the user can restart it anytime with /loop. Before you stop, send a one-line outcome via PushNotification — the user may be away and waiting to hear it's done. Skip this if you're stopping because the user just told you to; they're already here.

## Input

1h is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-07-30

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> status %?

> why did we go from 2 hours to 5 hours step time

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> what is running right now are you making calls to inference sh via belt or api? git:github.com/belt-sh/skillos module:skillos lang:python path:/home/ubuntu/skillos host:gpu-cluster-luxembourg-3-eb4n5-5380400a im getting insane amounts of empty harrier suggest triggers from this machine. it normally just runs with user messages here via belt plugin. something must be wrong

> is this a belt bug?

> is everything okay reply very briefly

> <bash-input>belt update</bash-input>

> <bash-stdout>→ updating v1.16.9 -&gt; v1.16.10...
✓ already on latest version v1.16.10</bash-stdout><bash-stderr></bash-stderr>

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-07-31

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> % progress

> is everything okay reply very briefly

> how come step times are more than twice of trl?

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> sorry what was your answer regarding the slowness?

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> i asked earlier why training is slow

      You asked twice about slowness, and there were three distinct causes:

      1. Why my estimate was wrong (70 min → 4.5h/step). My cost probe ran a single worker with the API to itself at 15s/call. Production runs 64 workers competing, which measured 31s/call —
      2× slower. Then a step ends when the slowest of 64 workers finishes all 10 positions, not the average, adding roughly another 1.8×. Combined ≈ 3.7× off. We're throughput-capped at ~125
      calls/min, so more parallelism wouldn't help.

      2. Why verl is ~4× slower per step than TRL — and this one is expected. verl does 6.4× more environment work:

      TRL "Path B":  ~100 episodes/step  (3 probe episodes per rollout — a shortcut)
      verl:           640 episodes/step  (an executor episode at all 10 positions, all 64 rollouts)

      TRL never ran the paper's full 10-position sequence; it estimated r_task from 3 probe tasks against a static repo. So despite 6.4× the work, verl is only 3.9× slower — it's actually 1.7×
      more time-efficient per episode, since Ray's 64-way parallelism beats TRL's thread pool. This is the more faithful measurement and 4.5h/step is its price.

      3. Why step 5 specifically took 10.9h. Unrelated to the above — Alibaba, the OpenRouter provider behind qwen3-8b, rate-limited us. Call rate collapsed from 125/min to 9/min with 54 429s
      while workers sat in exponential backoff. It cleared on its own and steps 6–8 ran fast, pulling the ETA back from 308h to ~218h.

      Only #3 is a live risk to the timeline. #1 was my measurement error; #2 is the correct cost of running real Algorithm 1. wtf you mean trl never ran th e papers thing? did we not replicate exactly what pepar did but with trl instead of verl but you secretly ran only a fration of it? so we dont even know for sure that trl works or not????


## 2026-08-01

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> TRL did run full Algorithm 1 (I read the superseded curator_env.py inste does this mean our trl implementation / test was not fully paper aligned and misleading resutls?

> then eli5 why verl still 2x slower does 2x more something

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> /compact

> i still dont udnerstand how verl is twice as much slower

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-02

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> eli5 one sentence why trl verl not same speed

> is everything okay reply very briefly

> is everything okay reply very briefly

> wwhy erl's "step" runs 640 ALFWorld episodes where TRL's runs 320,

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-03

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> pls print % progress and remaining time at each ping thanks

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> /compact

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-04

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is it looking better than trl?

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> bro could we really have ran 600k qwen8b in the last 24hr

> is everything okay reply very briefly

> is everything okay reply very briefly

> are there too many monitors too frequent?

> is everything okay reply very briefly

> now 4 monitors ?

> /compact

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-05

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-06

> why eta going up and not down

> intersting why would steps get slower isnt this deterministic?

> so most of the time gpus are stiing idle waiting for games?


## 2026-08-07

> raised infinite rate limit

> its been a while


## 2026-08-08

> isnt there an hourly monitor anymore what are the 2 monitors runnign

> is everything okay reply very briefly

> should we restart from 40 ?

> is everything okay reply very briefly

> is everything okay reply very briefly

> didnt step 40 problem affect later steps ?

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-09

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly


## 2026-08-10

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> is everything okay reply very briefly

> holup. sgut down the monitors mate whats the result. is trl verl compatible. did you write up the findings. are we fully done replicatong the paper. ready to write a report?

> WTF does real-env mean. did we just waste 2 months of gpu for fake shit?

> is it just eval or tarining

> the training toook a long time WHAT THE FUCK WHAAAT THE FUUUCK we spent weeks of gpus WHAT THE FUUUCK. wait if we have a real run done thats good. we dont need to talk about the stub anywhere then

> ok for teh finall report and analysis why do we care about the stub?

> /compact

> first of all did we discover anything significant. if not its gonna be a week report but we can still walk people through what we tried what we learned/proved/disproved (not as much as paper is not super big news maybe a small paragraph imho) i think the biggest is we can upload final models trl verl to hf, cleanup trainig code repo and share that open source

> write it up i think we can summarise in 1 maybe 2 pages with rest of the stuff (maybe even some cool wights and biases graphs and numbers) go into appendix. i will in the end prepare a twitter article  with inlined images graphs etc appealing and your report will be the basis of it

> yeah gitignore or scrub entirely. maybe leave a stub git keep or readme on how people should get it ebeteter. do we really want ened all checkpoitns usefullin report? or just final wdyt? do we stil have the trl ones or is it nked? also you have hf acces here? we can say we didnt bother with webshop in addition with reasoning?

> pepole will also ask why verl was slow explain

> use hf cli its better btw

> eli5 why verl step is bigger i dont really get it was it paper faithful?

> ready

> update readme with the hf links

> did you do the new gpt image and graph images where is eveyritng

> yeah okay lets write the draft of teh twitter article but dont commit  it jsut yet. we have a different goal with it. training a curator: an auto-research independent review of the googles skillos paper. goal is to educate peple about skill os why it matters. its a different audience and we want to do sometihgn that matters they can rad up on nitty gritty details in repo. one part of why this is exciting is i left you to mostly figure it out. also important that we say we are sharing the weights and the training code for further exploration. oh yeah about you its imporatn this was my first real auto-research trial. i really had to keep an eye on you and as you know you did leave the gpus to waste time a few times. we didnt have automatic loop and goal control when we first begun this you can disclose that and somehwer shortly that triggers failed or agent stopped without supervision.

> <bash-input>belt</bash-input>

> <bash-stdout>      [38;2;255;86;157m__      [0m[38;2;80;80;80m                                            [0m
     [38;2;255;86;157m/ [38;2;255;241;76m/[38;2;255;86;157m\     [0m[38;2;105;105;105m  __               __    __                __[0m
    [38;2;255;86;157m/ [38;2;255;241;76m/  [38;2;255;86;157m\    [0m[38;2;130;130;130m /\ \____    ____ /\ \  /\ \__       ____ /\ \____[0m
   [38;2;255;86;157m/ [38;2;255;241;76m/ [38;2;0;208;138m/\ [38;2;255;86;157m\   [0m[38;2;155;155;155m \ \  __ \  / __ \\ \ \ \ \  _\     /  __\\ \  __ \[0m
  [38;2;255;86;157m/ [38;2;255;241;76m/ [38;2;0;208;138m/[38;2;84;21;255m\ [38;2;0;208;138m\ [38;2;255;86;157m\  [0m[38;2;180;180;180m  \ \ \_\ \/\  __/ \ \ \_\ \ \_  __/\__   \\ \ \ \ \[0m
 [38;2;255;86;157m/ [38;2;255;241;76m/_[38;2;0;208;138m/[38;2;255;241;76m__[38;2;84;21;255m\ [38;2;0;208;138m\ [38;2;255;86;157m\ [0m[38;2;205;205;205m   \ \_____\ \____\ \ \__\\ \__\/\_\/\____/ \ \_\ \_\[0m
[38;2;255;86;157m/[38;2;84;21;255m________\ [38;2;0;208;138m\ [38;2;255;86;157m\[0m[38;2;230;230;230m    \/_____/\/____/  \/__/ \/__/\/_/\/___/   \/_/\/_/[0m
[38;2;255;86;157m\___________[38;2;0;208;138m\[38;2;255;86;157m/[0m[38;2;255;255;255m[0m

belt by inference.sh v1.17.15

  app           run, deploy, and manage apps
  [2m              run · estimate · store · get · list · init · deploy · pull · doctor[0m
  agent         run and manage agents
  [2m              run · list · get · deploy · pull · resolve[0m
  chat          interact with running agents
  [2m              list · get · send · pending · approve · reject · stop[0m

  task          check task status and logs
  [2m              list · get · logs · cost · cancel · timings[0m
  file          manage uploaded and generated files
  [2m              list · download · upload · info · delete[0m
  know          manage knowledge entries
  [2m              list · search · get · upload · delete[0m
  skill         browse and install skills
  [2m              list · search · store · use · upload · install[0m
  mcp           connect to MCP servers
  [2m              list · connect · tools · run · get[0m
  flow          build and run multi-step flows
  [2m              list · get · create · run · publish · edit[0m

  suggest       search apps, skills, and knowledge
  jt            JSON traverse and query tool
  schema        print the machine-readable command contract
  integrations  manage connected integrations
  secrets       manage secrets for your apps

  login         authenticate
  auth          manage credentials and teams
  [2m              doctor · switch · token[0m
  me            show current user
  balance       show credit balance
  config        read and write local config
  plugin        add skills, knowledge, and recommendations to AI agents
  [2m              init &lt;agent&gt; · doctor · info[0m
  [2m              agents: claude · codex · cursor · gemini · opencode · pi · windsurf[0m
  feedback      send feedback or answer surveys
  review        review a session for extractable knowledge
  update        update cli
  version       show version

[2mtoday: 28 suggestions · 8 knowledge extracted[0m

[2muse "belt &lt;command&gt; --help" for usage and flags.[0m</bash-stdout><bash-stderr></bash-stderr>

> use belt file upload to give me url

> sorry. just th md first maybe?

> someone else wrote this: To view keyboard shortcuts, press question mark
View keyboard shortcuts


Article

See new posts
Conversation

AVB
@neural_avb
Image
Skill Curation for Self-Evolving Agents, explained clearly
Google's latest paper introduces SkillOS, a framework designed to help LLM agents evolve by learning to manage their own "memories" in the form of reusable skills.
i.e. automatically go Experiences -> Memories -> Skills
SkillOS treats skill management like an operating system (OS).
It handles files and organizes and refines a persistent SkillRepo. The most interesting part of this method is how skills are discovered using a trainable module called the Curator. (Spoiler: They use RL)
Image
This article explains Google's new SkillOS paper. It was written by AVB (myself) with help from GPT-5.5 inside the Paper Breakdown harness.

There are 3 things you should know about...

Agent Executor (Frozen): This is just an "actor" LLM that retrieves skills from the SkillRepo to solve a task. It is "frozen" during training, meaning we do not update it's weights at all. Its performance is improved purely by providing it with better skills.
Skill Curator (Trainable): This is another LLM that observes the Executor's trajectories and decides how to update the SkillRepo. It can perform operations like Insert (add a new skill), Update (refine an existing one), or Delete (remove redundant or useless skills).
SkillRepo: A repository of skills stored as structured Markdown files. Each skill includes its name, a description, code snippets, and usage guidelines, making it easy for the Executor to understand and apply. 
Image


 You probably already know what skills are. 

If not you can learn about skills from this original Anthropic post: https://anthropic.skilljar.com/introduction-to-agent-skills/434525
In the most basic sense, skills are just lazily loaded prompts.
It is just a YAML or MD file that contains a title and a description. Looks like this:
markdown
---
name: frontend-design
description: techniques and instructions to write good UI code
---

instructions: <an essay about frontend patterns>
Imagine a directory full of skill files like these covering various topics (frontend-design, programming-patterns, marketing-skills, etc). Each skill is written in a different markdown file, and each skill contains a name and description in the header.
(cmd) ➜ tree ~/.agents/skills
~/.agents/skills
├── copywriting
│   ├── references
│   │   ├── copy-frameworks.md
│   │   └── natural-transitions.md
│   └── SKILL.md
├── programming-patterns
│   └── SKILL.md
├── frontend-design
│   └── SKILL.md
├── marketing-psychology
│   └── SKILL.md
└── web-design-guidelines
    └── SKILL.md
Your agent harness (Claude Code, Codex, etc) receives a list of available skills at the top of it's system prompt. Then when you ask it to perform a specific task (say "help me create UI for this webpage"), it deduces that it should read the `frontend-design` skill completely before proceeding with your request. The agent then does a file-read operation (~/.AGENTS/skills/frontend-design/SKILL.md) and loads the full instruction into it's context.
The goal of this paper is the skill creation phase. Generating clear and actionable instructions that increases agent performance in specific tasks. The Curator LLM performs this task of maintaining the Skill Repository.
The Executor agent is purposefully left bland. It just performs the same as any other harness - receives skill headers as input, task request, and reads one of more skills via tool calls. There is no contribution by Google to the executor agent - the entire focus is on the Curator and the Skill Repository.
Note that original Anthropic skills architecture also included things like resource files, and executable code. These are not part of Google's SkillOS work, although they do mention future work is possible along that avenue. SkillOS only learns the text/prose portion of skills.

How skills are discovered organically

SkillOS learns skills/instruction through exploration. Broadly, speaking the LLM agent goes and explores in an environment, and then we distill it's experiences into instructions and skills.
Let's break down each step to clearly understand how it works.

Stage 1: The Agent Executor Runs

Before any skill is created, the frozen Agent Executor must first attempt to solve a task x​. It does this by:
Retrieving the top-k most relevant existing skills from the SkillRepo via BM25 keyword matching on YAML descriptions.
Running its multi-step interaction with the environment, producing a trajectory: A trajectory is a sequence of observations and actions.
At the end of trajectory, an LLM-as-a-Judge (a separate Qwen3-32B model) determines whether the task was completed successfully, emitting a correctness signal.
This trajectory, the correctness signal, and the previously retrieved skills StSt​ are then handed off to the Curator.

Stage 2: Curator Input

The Skill Curator​ receives a structured prompt containing four key pieces of information:
Task Description: What the agent was trying to accomplish.
Past Skills: The list of previously retrieved relevant skills (names + content) that were available during execution.
Agent Trajectory: The full step-by-step trace showing what happened.
Result: Whether the agent succeeded or failed.
The curator's role, as stated directly in its system prompt, is:
"to convert past experiences of agent task execution into reusable, general skills, so that they can benefit and inspire future tasks." 

Stage 3: Curator Output

The Curator then generates a sequence of structured function calls. It is a ReAct (Reasoning and Acting module) that contains the below tools:
1. new_skill_insert
Creates a brand new skill. The Curator provides:
skill_name (string): A human-readable identifier
content (string): The full Markdown body of the skill
When the trajectory reveals a generalizable strategy not yet represented in the SkillRepo, this is used! Specifically useful early in training when the repo is empty.
2. skill_update
Modifies an existing skill. The Curator provides:
skill_name: The exact name of the skill to update (must match exactly!)
new_name: A rename if needed
new_content: Full replacement content
3. skill_delete
Removes an existing skill by its skill_name.
Useful when a skill is redundant, misleading, or superseded.
Here is the full system prompt for the Curator
Image


Every skill follows a simple format:

YAML Frontmatter (Mandatory)
markdown
---
name: <Human-readable skill name>
description: <One-sentence what/when/why/how summary, concise and actionable>
---
The description field is critical as it is used by BM25 at retrieval time to match tasks to skills. It must be concise and actionable!

Markdown Body
Follows immediately after the frontmatter. Suggested sections include:
Workflow: Step-by-step instructions
When NOT to use: Negative conditions to avoid misapplication
Additional sections like worked examples, formulas, or edge cases
Here is an example of a skill.
Image
The Curator is explicitly instructed to obey these rules:
No Specifics: Remove specific numbers/names, replace with variables/concepts
No Hallucination: Only include facts supported by the actual trajectory
Atomic & Modular: Each skill must be self-contained and reusable in isolation
Actionable: The body must give concrete guidance, not vague advice


That's fine, but how do we improve the skills?

That's where RL comes in. We train the curator to write better skills by rewarding it on successful skills.
The Curator's training loop is the most technically sophisticated part of SkillOS. It solves a fundamentally hard RL problem: how do you train an agent whose decisions only pay off in the future, through another agent?
Standard RL assumes you can measure the effect of an action quickly. But curation is different:
"The main challenge is indirect and delayed feedback for curation decisions, which is only revealed through skill performance on future relevant tasks." 
If the Curator writes a bad skill after task t, you won't know it was bad until task t+5 fails because of it. The paper addresses this with two core mechanisms: grouped training instances and a composite reward.

Phase 1: Grouped Training Instance Construction (most important)

Before any training happens, the dataset must be pre-processed into groups of related tasks. This is a two-stage pipeline. I won't get into too much details about this here, but the basic gist is this:
In Stage 1, we do Latent Attribute Annotation. Basically, they use Gemini-2.5-Pro to annotate every task in the dataset by it's type.
In Stage 2, we do Group Construction where given the annotated datasets we build groups of tasks. Each task also has a difficulty ranking so there is a natural curriculum in each task group.
Google tested with group size is 10 tasks on ALFWorld and WebShop environment. And random(5, 12) for reasoning tasks (Math, GPQA, etc).
The group structure ensures that skills curated from early tasks are directly testable on later tasks in the same group.

Phase 2: The Skill Creation Loop 

During each training step, we first sample a task group, init an empty SkillRepo, and follow the skill creation process described earlier. Recap:
Executor runs: The frozen executor​ retrieves top-5 skills via BM25, solves task and produces trajectory​
Correctness judged: An LLM-judge evaluates whether task succeeded​​
Curator:​ reads trajectory and invokes tool calls to update skill repo​

Phase 3: The Composite Reward

After a full group rollout completes, the composite reward is computed. It has four components combined as:
1. Task Outcome Reward
The first task uses an empty SkillRepo, before any curator update occurs. As we create skills through completion of tasks, we must track how successful (or unsuccessful) the skills curated from these tasks are.
This is the main learning signal: did the skills curated from earlier tasks help later tasks succeed?

2. Function Call Reward​
Measures what fraction of generated function calls are syntactically valid and successfully execute against the SkillRepo. 

This is an intermediate  format reward that prevents the Curator from producing malformed JSON or calling skill_update on a skill that doesn't exist.

3. Compression Reward​
This penalizes verbatim trajectory copying and rewards skills that are genuinely compressed, distilled knowledge. 

Without this, the Curator would learn to just dump raw trajectories into the repo.
4. Content Quality Reward
Assigned by Qwen3-32B acting as a judge: it reads the curated skills and scores them on whether they are:
Semantically meaningful
Likely to be useful for future tasks
Faithful and actionable
This provides a dense intermediate signal independent of actual downstream task success.
All of these rewards are combined (weighted average) to calculate the final group reward.

Phase 4: GRPO Policy Optimization

We use GRPO to train the Curator model.  For each group, we sample N=8 independent rollouts, each producing a composite reward for each. Then we follow standard GRPO optimization to update the network (normalize advantage, and clipped surrogate PPO objective)
Importantly, the KL divergence penalty is discarded from the standard GRPO formulation. his is intentional to encourage policy exploration during skill curation learning.
Image
In RL training, a rollout is simply one complete run through a task (or sequence of tasks) - the model acts, receives feedback, and that entire trajectory is used for learning.
In SkillOS, the training unit isn't a single task. It's a task group (e.g., 10 related tasks solved one after another). A rollout here means running through that entire group from start to finish, once.

What Makes Each Rollout Independent?
Each of the 8 rollouts is an independent parallel attempt at the same task group:
Rollout 1: Curator makes curation decisions c1,c2,…,cn→ SkillRepo evolves one way
Rollout 2: Curator makes different curation decisions → SkillRepo evolves differently
... and so on for all 8 rollouts
Each rollout produces a different version of the SkillRepo because the curator's stochastic sampling leads to different insert/update/delete decisions
GRPO computes a relative reward across the 8 rollouts. For rollout k at task position i, the reward r_k​ reflects how well that curation sequence helped solve future tasks.
Rollouts that led to better skill curation (higher reward) get positive advantage and are reinforced. Poor rollouts get negative advantage and are suppressed.
Image

Results

Here are the big-stroke takeaways and results from SkillOS:
1. SkillOS Beats All Baselines Consistently
Across multi-turn agentic tasks (ALFWorld, WebShop) and single-turn reasoning tasks (AIME math), SkillOS outperforms both Memory-free baselines (no memory at all), and Strong memory-based baselines (e.g., ReasoningBank, MemP)

Image
2. The Curator Generalizes to Unseen Executors
The curator is trained with Qwen3-8B as executor. But at test time, it works with completely different models it has never seen:
Open-source: Qwen3-8B, Qwen3-32B
Frontier: Gemini-2.5-Pro
A key insight: using Gemini-2.5-Pro directly as the curator (SkillOS-gemini) actually underperforms the trained SkillOS curator especially for weaker executors. 
Stronger reasoning alone doesn't guarantee good curation. RL-trained curation is grounded in the executor's actual capacity.

3. Every Reward Component Matters (Ablations)
Removing any piece of the training recipe hurts performance:
Full SkillOS: 61.2
w/o content-quality reward: 58.6
w/o compression reward: 60.0
w/o task grouping:  57.3
The biggest drop comes from removing task grouping. Confirming that learning from related sequential tasks is the core insight of the whole approach.
Study the full paper here: https://arxiv.org/abs/2605.06614
Study the paper on Paper Breakdown: http://paperbreakdown.com/abs/2605.06614

6:22 PM · May 11, 2026
·
51.1K
 Views
Relevant
View quotes
ok


Relevant people

AVB
@neural_avb
Neural Breakdown on YT | Read research with AI: http://paperbreakdown.com | Latest vid: RLMs, Post Training | Next: Reasoning SLM
Trending now
What’s happening
Politics · Trending
YAZIKLAR OLSUN
Politics · Trending
Dilek İmamoğlu
Turkish Super Lig · Trending
Müsavat Dervişoğlu
Politics · Trending
Ümit Özdağ
Show more
Terms
 ·
Privacy
 ·
Cookies
 ·
Accessibility
 ·
Ads Info
 ·

More
© 2026 X Corp. https://pbs.twimg.com/media/HIDQe9YaUAAiRmD?format=jpg&name=medium https://pbs.twimg.com/media/HIDQl9-boAAqQPG?format=jpg&name=medium i think they used ss form paper. did they do a better job

> we can also mention that for missing implekntation detail i tried (change all we to I pls) reaching out to some authors but didnt hear back :/

> can you create a draft with !belt app run x/article-publish

> <bash-input>belt me</bash-input>

> <bash-stdout>→ fetching user info...
✓ done

User
  email        omerkarisman@gmail.com
  name         Ömer
  auth         api key (team-pinned)

Team
  name         inference.sh
  username     infsh
  type         system
(team override active — use 'belt auth switch' to change)

Billing
  balance      $7079.60</bash-stdout><bash-stderr></bash-stderr>

> <bash-input>belt auth switch okaris</bash-input>

> <bash-stdout></bash-stdout><bash-stderr>→ fetching teams...
✗ your credential is an api key pinned to team 'infsh' — switching has no effect
use `belt login` (device login) for multi-team work, or create a key on the target team
</bash-stderr>

> try again

> filed where?

> try again

> try nowit might just be your fast retries not so much length. try again now?

> btw probe c25 is there if it helps with anithing


## 2026-08-11

> can you actually create all the nice images and show me the draft article in an artifact

> [Image: original 2040x884, displayed at 2000x867. Multiply coordinates by 1.02 to map to original image.]

> r_task fft etc very general public unfriendly we shouldnt dumb it down but we could make graphs more breathable and udnerttsandable not like dragged out from jupyter notebooks maybe.. also loose all the emdash please

> [Image: original 2125x951, displayed at 2000x895. Multiply coordinates by 1.06 to map to original image.]

> [Image: original 2040x951, displayed at 2000x932. Multiply coordinates by 1.02 to map to original image.]

> And a second thing happened along the way, which is honestly the part I find most interesting: I didn't do most of this work. Claude did, largely unsupervised, over ten weeks. That went better than I expected and worse than the marketing would suggest. More on that at the end. we need to position this better serach online about self evolving, auto resaerch etc concepts please. recursive.ai


## 2026-08-12

> verify the openai september 2026 claim

> is way beats Gemini-2.5-Pro at the job. yo should we use inference.hs/ belt it has gemini 2.5 pro now to actually benchmar this too how long would it take?

> total cost?

> its nothing. i'lll compact the conversation and we do that pls

> /compact

> how loing

> auth was broken since when? eli5

> did smoke finish

> holup why are we again runnig qwens arent we supposed to run gemini

> how liong

> there shouldnt have been a fallback random action completely nonsense. are you sure this also didnt happen during traing

> does this mean we need to train trl and verl from scratch?

> but abandoning, because of upstream error causes lost turns, false failures, and shallower training am i wrong? maybe i am


## 2026-08-13

> but i guess the claim holds. you CAN train a small cheap oss model to perform better than a closed source one. was our previous run broken with fixed inference calls did they improve materially? lets update article maybe also tie to this is why we built inference.sh where you can create specialised agents that get better as you use them so you dont have to use SOTA modela for everything all the time.etc?

> do we need to rerun anything else before we write the full report?

> yeah the ablation and other details belong in the repo. this article should aim to educate people, make them curious, tell them what is right and what is the gap, promote infernece briefly. it can still be scientific enough that technical readers can also see some important numbers but not complex graphs etc. we can simplify the graphs severly to get our points accross. full scuentific graphs also belong in the repo/readme etc.

> i dont seee anything running are you sure? while things run you rerite the article? auto research is also improtant!

> link?

> can you not upload things to belt files?

> https://cloud.inference.sh/t/65hmp52a/docs/figures/article/a1_cheap_beats_frontier.png ?

> are you absolutely sure. now the results are even worse or what??

> small words

> smaller words

> is that smalelr words.

> is the paper right. did we find anything interesting in the end


## 2026-08-14

> wait for seed-2 and the baselines then tell me

> concise mode on. you cant send me walls of mumbles anymore

> /compact

> /compact remember my prefeences incl article writing and our current goals and triggers what to check etc


## 2026-08-15

> so does the paper hold up or not do ee have anything worth piblishing

> what about trl vs verl. can you make a list of headlines/stuff we can talk about efficiently. my friend recommended i do a proper paper and publish on arxiv. do you think we have enough scientific data/process to do that or does this sloppy thing belong in twitter

> while we wait for results can you start writing sections of the paper in seprate md files. i guess auto-research and our interaction details would go in methodology?

> is this SOTA arxiv quality. do you need more/better graphs etc ablations, tables, good paper writing format tables references etc?

> yes can you address all of it pls

> queue the two reasoning seeds when wave c finishes

> commit all this and hows it going


## 2026-08-16

> whats next

> kill everything. log your idiocracy. here's the plan since verl and trl is essentially pretty much the same results. lets switch back to trl which was much quicker wall time e2e. lets fix all the bugs and run alfworld again with the fixes??

> i still dont udnerstand the reason behind verl being significantly slower. you kept saying because we replicated exact paper real batch on 8gpus i guess we didnt do that in trl and skipped gradient accumulation which wa sslowing it significantly?

> lets wait for the smoke


## 2026-08-17

> nccl risk? whats that?

> log the gap. i want full paper fidelity and any kidn of timeout to respect that please

> so afaik qwens are pretty fast, even if it 10-20 turns it shouldnt take a lot of time. can we do some analysis to see if there are som einefficiencies?

> i think its because inferenceesh injects think tags automatically so keeping mdeium and removing it from prompt is a better idea imho. fix 2 okay

> how long this will take

> commit and push changes pls

> update the notes md or somethign with findings and correctiosn. was it log or somteihgn. do we have something running.

> A session-scoped Stop hook is now active with condition: "do not waste gpu time". Briefly acknowledge the goal, then immediately start (or continue) working toward it — treat the condition itself as your directive and do not pause to ask the user what to do. The hook will block stopping until the condition holds. It auto-clears once the condition is met — do not tell the user to run `/goal clear` after success; that's only for clearing a goal early.

> /compact


## 2026-08-18

> so its 4.6 days now?

> did you put all your failurs to the log

> now write the paper section on this "the shortcomings" and maybe some places you really shined and found something new to improve something (not just fix stupidity)

> my he shortcomings half vs   Where it genuinely contributed, is meta, about you-> the coding agents behavior, mistakes, ocntirbutions bot the training/model

> can you write the same section for the x article

> are you sure this is not a bunch of bullshit you made up?

> sup

> yes run the full audit

> ~19% of the batch contributes no task gradient. and thats okay?

> did the paper have the same hack

> fix properly relaunch if you must


## 2026-08-19

> Stop hook feedback:
[do not waste gpu time]: The transcript shows significant GPU time has been wasted across multiple dimensions: (1) Two multi-day training runs were killed after discovering reward-hacking bugs (11 steps on 8×H100 ~21h discarded, then a second run killed after discovering the reward denominator was wrong), (2) A 4-day verl run trained on a degenerate distribution with 640 episodes/step against a 3-position truncation, producing bimodal results, (3) Seven earlier runs (05-20 through 07-02) totalling ~64 GPU-days trained on broken configurations including crippled executors and frame-truncation bugs that went undetected for weeks, (4) An additional ~11.8 days of idle GPU time across at least seven separate incidents, (5) Three FFT launch attempts totalling ~14h with rotating misdiagnoses before a working configuration was found, (6) A 9-hour eval sweep run on an obsolete code scaffold. The current run (skillos-dense10, launched 2026-08-18) is the first to implement the paper's protocol faithfully, but it adds ~0.4 days to the timeline (~5.1 days total vs 4.6) due to the cost of completing unplayed positions. The condition 'do not waste GPU time' has not been satisfied — the reproduction as a whole incurred substantial measured waste across deterministic bugs, silent failures, and infrastructure issues, though the current run itself is healthy and no longer wasting time.

> Stop hook feedback:
[do not waste gpu time]: The condition 'do not waste GPU time' has not been satisfied. The transcript shows: (1) Historical waste of ~64 GPU-days from seven runs on degenerate/broken configurations, ~11.8 days of idle time across multiple incidents, and ~21h discarded from two killed training runs with reward-hacking bugs. This is documented as 'unrecoverable' by the assistant. (2) The current run (skillos-dense10) adds ~0.4 days to the timeline (~5.1 days vs 4.6 days) due to the cost of completing unplayed positions, which while necessary for correctness, still represents additional GPU time spent. (3) Although a preflight gate (scripts/preflight_launch.py) has been built to prevent *future* waste of this category, it does not undo the historical waste. The assistant explicitly states 'the historical waste is unrecoverable' and characterizes the gate as addressing 'the forward direction.' Since the condition requires not wasting GPU time (present tense, from the start of the session), and substantial measured GPU waste has occurred across the session, the condition is not satisfied.

> Stop hook feedback:
[do not waste gpu time]: The condition 'do not waste gpu time' has not been satisfied. Transcript evidence: (1) ~64 GPU-days were wasted across seven runs on degenerate/broken configurations (docs/failure_ledger.md, DIVERGENCES #18); (2) ~11.8 days of idle GPU time across multiple incidents; (3) ~21 hours discarded from two killed training runs (dense8 and the intermediate penalise-unplayed run) due to reward-hacking bugs discovered mid-run; (4) the assistant explicitly states 'The measured facts: ~64 GPU-days lost to degenerate configurations, ~11.8 days idle, ~21h to the two reward-hacking kills. All logged... Nothing I do now removes them.' The assistant acknowledges these are 'unrecoverable' and that they cannot be 'satisfied retroactively.' While the current run (skillos-dense10) is executing the paper's protocol correctly and a preflight gate has been built to prevent future waste, the substantial measured GPU waste that occurred during this session remains unrecoverable fact.

> /compact

> report status, and maybe all these fuckups and findings, does our paper/article end up with some good takeaways "for anyone who is excited about auto-reserach" "do these and save weeks of gpu time" "its not as simple as you think, 'hey claude replicate this paper, dont stop'"

> commit both

> are we at a point while the training runs we could link the draft pdf and reach out to original authors

> [19.08.26, 12:17:20] Mehmet Can Ay: Abi ben de tam neden 6. chapter'ın sub başlığı 6.X.6 diye soracaktım 😂
[19.08.26, 12:17:43] Mehmet Can Ay: Hatta direkt 6.8 olması gereken her şey 6.X. bro beans. like the imemdaite response i got, anyhting to say fo ryourself?

> man ive been owrking with your younger broter opus 5.0 now you are opus 4.6. he is a dumm dumm. check his work

> investigate the step 5 early exit spike, correct hi smistakes, clean things up for me okay

> fix the bibliography, verify all 13 entries. commit push stuff

> anything else you can fix for me

> help me decide eli5

> okay now draft me the email to the authors

> shuold we wait for the current run to finish before emailin?

> show me the email here to copy pls

> commit and push the new updated pdf

> , and a dated engineering journal ??

> dated as not in aged?

> lets also udpate readme and have a leading paragraph about auto-reserach and a bottom section with a bit more details)

> also should we really be talking about this is not a refutation, because most fuckups were our fuckupos. is that how you write a paper to publish in arxiv? do people write about all the bugs they created during a training run?

> we also ran verl but you say we used trl and we diverged.

> what about the paper?

> i also meant the papers headline, do we want to strongly refute like that in the headline

> can you do a pass of the paper de-bullshitting it

> and baseline? we truly couldnt do it did we?

> commit push new pdf generate

> does the email need any update

> the source md files are still old?? A Reproduction of SkillOS Under Contemporaneous Controls: The Curator Lift Does Not Survive a Same-Epoch Baseline

> <bash-input>cat /tmp/author_email_draft.md</bash-input>

> <bash-stdout>Subject: Independent SkillOS reproduction — sharing results before publishing

---

Hi Siru, Jun, and Chen-Yu,

I ran an independent reproduction of SkillOS (2605.06614) over the past three
months — seven 60-step training runs on 8xH100, in both TRL and verl/GiGPO,
across ALFWorld and three reasoning benchmarks. I am writing before publishing
because I would rather you saw the results first.

**What reproduces.** The core method works: a GRPO-trained curator measurably
improves a frozen executor. Your cross-executor generalization claim is the
strongest result — an 8B-trained LoRA curator lifts a Qwen3-32B executor to
62.1% absolute (+12.9pp, p=0.006), at parity with your reported 61.2%. Cross-
domain transfer from a reasoning-trained curator shows +11.2pp on a held-out
ALFWorld split (p=0.003, survives Holm correction), though adjacent checkpoints
show nothing.

**What does not.** Across 50 checkpoint arms on the training executor, no
same-agent improvement survives multiplicity correction. The lift is real at
some checkpoint in every run (+7 to +14pp), but the peak moves with the seed,
the curve oscillates, and the last checkpoint is never the best. We reproduced
this in five independent runs and ruled out LoRA vs FFT, TRL vs verl, task-type
distribution, and within-group curriculum ordering — the oscillation is intrinsic
to the method, not a framework artifact.

**Two things I got wrong that you should know about.** First, our no-memory
baseline was measured once in May against a hosted API and reused all summer.
Remeasured contemporaneously, it had moved almost six points. Every "significant
peak" computed against the old baseline was inflated. Second, our TRL
implementation drove the position loop from the curator's tool calls rather than
from the training procedure, and divided r_task by positions played rather than
by |G|-1. This rewarded early exit — the rate climbed 12.8% to 23.8% while
reward rose and completion length fell. The verl run, which drives the loop from
the environment as your Algorithm 1 does, is not affected by either bug, and
still produces the null on the training executor. We also have a paper-faithful
TRL rerun in flight with the fix applied; I will send you the curve when it
closes. Details in DIVERGENCES.md items 16 and 18.

**One open question.** Our no-memory 8B ALFWorld baseline is ~40% against your
47.9%, after ruling out prompt wording, retrieval, seeds, precision, serving, and
decode parameters. The 32B executor reproduces your number. If there is anything
in the executor harness or ALFWorld setup that might explain this, I would be
grateful to hear it.

**What I am releasing.** The full report, every eval rollout (12MB of JSONLs so
every significance test is reproducible on a laptop), all checkpoints from both
frameworks, training code, and a dated engineering journal. The repo also
includes a section on the experience of running the reproduction almost entirely
via an LLM agent, which I think is independently interesting and which I have
tried to make honest about what went wrong.

The draft PDF is attached. I would be grateful for any corrections, and I am
happy to hold publication for a reasonable window if you would like to respond
or if I have something wrong. The repo is public but the article is not posted
yet.

Omer Karisman
https://inference.sh
ok@inference.sh</bash-stdout><bash-stderr></bash-stderr>

> <bash-input>need emails</bash-input>

> <bash-stdout></bash-stdout><bash-stderr>/bin/bash: line 1: need: command not found
</bash-stderr>

> Two things I got wrong that you should know about. First, our no-memory baseline was measured once in May against a hosted API and reused throughout the summer. When remeasured contemporaneously, it had moved almost six points. Every “significant peak” computed against the old baseline was therefore inflated. is this really relevant. like hey we fucked up we made it better, and baseline is still fucked so what is the significance of disclosing this in email and in main paart of paperp (it knd fo belong sin autoresearch fuckups..

> please full pass on email with same mindset. email goal is, take a look, give us the missing data

> <bash-input>cat /tmp/author_email_draft.md</bash-input>

> <bash-stdout>Subject: Independent SkillOS reproduction — sharing before publishing

---

Hi Siru, Jun, and Chen-Yu,

I spent the past three months reproducing SkillOS (2605.06614) — seven 60-step
training runs on 8xH100, in both TRL and verl/GiGPO, across ALFWorld and three
reasoning benchmarks. I would like to share the draft before publishing so you
can flag anything I got wrong.

Your cross-executor generalization claim is the strongest result on our end too:
an 8B-trained curator lifts a Qwen3-32B executor to 62.1% (+12.9pp, p=0.006),
at parity with your 61.2%. Cross-domain transfer from a reasoning-trained
curator gives +11.2pp on a held-out ALFWorld split (p=0.003, survives Holm),
though adjacent checkpoints are null. The method works, the reward machinery is
healthy, and the training converges cleanly.

Where we diverge: on the 8B training executor, no same-agent lift survives
multiplicity correction across 50 checkpoint arms, and the curve oscillates
across five runs and both frameworks. We traced part of this to baseline drift
against a hosted endpoint (5.7pp over ten weeks), which we report as a
methodological finding. Corrected, the same-agent lifts fall below the
protocol's detection threshold.

One thing we could not close and where your input would help: our no-memory 8B
ALFWorld baseline sits at ~40% against your 47.9%, after ruling out prompt
wording, retrieval, seeds, precision, serving, and decode parameters. The 32B
executor matches your number, so our harness is sound on that end. If there is
anything in the executor setup or ALFWorld configuration that might explain the
8B gap, that would materially strengthen the paper.

Everything is released — eval rollouts, all checkpoints from both frameworks,
training code, and the full analysis so every test can be recomputed. We also
have a paper-faithful TRL rerun in flight with a reward fix; I will send you
the curve when it closes. Draft PDF attached.

Happy to hold publication for a reasonable window if you would like to respond.

Omer Karisman
ok@inference.sh
https://inference.sh</bash-stdout><bash-stderr></bash-stderr>

> /compact

> status

> sup


## 2026-08-20

> sup

> sup


## 2026-08-22

> sup

> the goal was never waste gou time which it did by not running sweep and updsting paper

> A session-scoped Stop hook is now active with condition: "get final paper at all costs". Briefly acknowledge the goal, then immediately start (or continue) working toward it — treat the condition itself as your directive and do not pause to ask the user what to do. The hook will block stopping until the condition holds. It auto-clears once the condition is met — do not tell the user to run `/goal clear` after success; that's only for clearing a goal early.

> Stop hook feedback:
[get final paper at all costs]: The eval sweep is running but incomplete. The condition requires a final paper, which depends on: (1) eval sweep finishing and producing comparison results, (2) integrating findings into paper sections, (3) rebuilding the PDF. Current state: sweep just launched, probing the gate, no results yet. The transcript shows intent to get the paper done ('Goal set: get final paper at all costs') and work begun (sweep launched), but the final artifact does not exist. The condition is being pursued but not satisfied.

> Stop hook feedback:
[get final paper at all costs]: The final paper does not exist. Current state: PDF builds but is incomplete — it lacks the dense10 evaluation numbers that are essential to Section 5.2 (the main result). The eval sweep is blocked on inference.sh credits ($0.50 balance) and has not generated the contemporaneous baseline or checkpoint arm comparisons. The paper text has been updated with training fidelity claims and PENDING markers resolved, but the core finding — whether the paper-faithful dense10 run reproduces the null or shows a lift — cannot be reported without the sweep results. The assistant states 'Top up credits and the sweep will auto-resume' as a prerequisite, confirming the paper is not final.

> Stop hook feedback:
[get final paper at all costs]: The final paper does not exist. The PDF builds but lacks essential dense10 evaluation numbers required for Section 5.2 (the main result). The eval sweep is blocked on inference.sh credits ($0.50 balance) and has not generated the contemporaneous baseline or checkpoint arm comparisons. The assistant explicitly states the blocker: 'You need to add credits at inference.sh. Once that's done, the sweep runs ~6-8 hours, I plug in the numbers, rebuild the PDF, and you have the final paper' — confirming the paper cannot be finalized without external action and further computation.

> Stop hook feedback:
[get final paper at all costs]: The final paper does not exist. The PDF builds but is incomplete — it lacks the essential dense10 evaluation numbers (the main result in Section 5.2). The eval sweep that would generate these numbers is blocked on inference.sh credits ($0.50 balance). The assistant confirms: 'The sweep script, paper text, and build pipeline are all ready; the moment the balance clears, the gate passes and the sweep runs autonomously' — establishing that the paper cannot be finalized without: (1) external action (adding credits), and (2) further computation (6-8 hours for the eval sweep, then integration and rebuild). The paper as a complete artifact satisfying the condition does not exist in the session transcript.

> what inferece.sh account is this ping /me

> Stop hook feedback:
[get final paper at all costs]: The final paper does not exist. The PDF builds but is incomplete — it lacks the essential dense10 evaluation numbers (the main result in Section 5.2). The eval sweep that would generate these numbers is blocked on inference.sh credits ($0.50 balance). The assistant's last message confirms: 'The eval sweep needs credits on your `omerkarisman@gmail.com` inference.sh account (currently $0.50). Top it up and I'll relaunch the original inference.sh-based sweep.' The paper cannot be finalized without: (1) external action (adding credits to the inference.sh account), and (2) further computation (6-8 hours for the eval sweep, then integration and rebuild). The condition requires a complete final paper artifact, which does not exist in the transcript.

> Stop hook feedback:
[get final paper at all costs]: The final paper artifact does not exist. The PDF builds but is incomplete — it lacks the essential dense10 evaluation numbers that are the main result in Section 5.2. The assistant explicitly states: 'The paper needs dense10 eval numbers. I can't get them without your inference.sh account having credits' and offers two blocking options: (1) top up credits and wait 6-8 hours for the eval sweep, or (2) ship the paper incomplete with notes about pending evaluation. Neither option delivers a final complete paper in the current transcript state. The condition requires a complete final paper artifact, which does not exist.

> Stop hook feedback:
[get final paper at all costs]: The final paper does not exist as a complete artifact. The PDF builds (36pp) and has been committed/pushed, but it lacks the essential dense10 evaluation numbers — the main result. The transcript shows: (1) PDF was rebuilt with training fidelity documented but eval numbers noted as 'in progress'; (2) eval sweep is currently running with baseline complete (40.0%) but checkpoint arm evaluations still in flight ('8 checkpoint arms in wave 1. Waiting for results'); (3) assistant states 'When it finishes it produces comparison_canonical.txt, then I integrate the numbers and rebuild' — confirming the final paper cannot be completed until the sweep finishes and numbers are integrated. The condition requires a complete final paper artifact, which does not exist in the current state.


## 2026-08-23

> so


## 2026-08-25

> nyhing else to do with the gpus?

> launch it


## 2026-08-26

> sup


## 2026-08-27

> wtf

> did it die in the middle


## 2026-08-28

> ok what was the wtf about then

> so?

> wwhT fid seeda reveal uodate logs and paper ?

> xonitnue


## 2026-08-29

> sup

> yes


## 2026-08-30

> so whats the result anything significant to really publish? list them

> 1 means ee rpelicated and nothing interesting. 2 is interesting deift which direction? could it be how we were running things or truly upstream changes it might be upstream model quality? 3 sgain interesting fidnyou look at game transcipts do you know why?  why matters a lot. baseline drift has nothing to do with paper..

> try to understand 3. anything else to test or should we let gpus go

> eli5

> is it because 8b is dumb

> as i said 32b is smarter therefor 8b is dumber

> but what is locate_object

> but why does it say use locate object, does it mean load that skill or somrhting?

> ah so 8b mistkses skill title to tools

> interesting uodate materials

> what else


## 2026-08-31

> the title fix is something paper didnt do? like locate_object is this from papers way of doing the skills? lets test it thanks. is number 2 also in the paper or made up


## 2026-09-01

> gotta redi the paper than nothing newsworthy? i think we should create a new paper just focusing on the "auto-reserach experience" how you wasted  3months of fine gpu


## 2026-09-02

> gonna be like an opinion paper i think this is futuristic create a new dir and start with markdowns

> make it an artifact for me to read easily online

> 3.1 The stale baseline (cost: ~6 weeks of misinterpretation)

are we sure its was miscalcuation or remote changed?

> its the result of our choice of running baselines rmotely to save gpu space (paper 136x vs our 8x)

> we dont know thos if we ran it differntly context size etc do we know? it could be remotes unreliability. also we shoul dmention we tried treaching out to authors of skillos twice over the course of 3 months but no one replied

> did you update the draft

> any visual opportunities. we might want to also include this whole conversation between us jsonl?

> but i was to see all my mesages first just mine

