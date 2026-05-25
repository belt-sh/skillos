# SkillOS: Learning Skill Curation for Self-Evolving Agents

> **Reference copy** transcribed from the arXiv HTML (arXiv:2605.06614v1 [cs.AI], 07 May 2026)
> for use while implementing this repo. Mangled LaTeX math/tables have been reconstructed
> into clean Markdown; numbers in tables are written as `value ± std`.

**Authors:** Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra,
Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee,
Jiawei Han, Tomas Pfister, Chen-Yu Lee. (Google Cloud AI Research; UIUC; MIT)

Correspondence: siruo2@illinois.edu, {junyann, chenyulee}@google.com

---

## TL;DR — numbers that matter for this repo

These are the headline reference points we compare our run against (ALFWorld, **Qwen3-8B
executor** — our setup):

| Method | Avg. SR | Steps |
|---|---|---|
| No Memory (baseline) | **47.9 ± 1.2** | 21.1 |
| ReasoningBank (strongest external baseline) | 55.7 ± 3.1 | 20.1 |
| SkillOS-base (curator, no RL) | 53.1 ± 2.5 | 20.4 |
| **SkillOS (RL-trained Qwen3-8B curator)** | **61.2 ± 4.6** | **18.9** |

Key implementation facts (from Appendix B.1, Table 4) that pin our config:

- **Training steps for ALFWorld: 60** (not ~111). WebShop: 50. Reasoning: 100.
- Batch size **32**, GRPO group size **8**, lr **1e-6**, **KL coef 0.001**, temperature **1.0**.
- Max Prompt Length **16,384**, Max Response Length **4,096**.
- **Data Grouping Size 10** for ALFWorld (each training instance = a group of 10 related tasks;
  SkillRepo starts empty per group; `r_task` averages success over tasks 2..|G|).
- Executor inference: **Top-K skill retrieval = 5**, **Max turns = 30**, **action history length = 3**.
- Executor uses **ReAct with forced chain-of-thought reasoning** (Appendix A.2). The frozen
  executor *reasons* — a reasoning-off executor is off-spec.
- **Executor decode settings are NOT in this paper** — they come from GiGPO (temp **0.4**,
  do_sample, max_response **512**, top_p 1.0). See "Executor decode settings" under Appendix B.1.
- Trained on **16 H100 GPUs** with **verl**; ALFWorld training takes **~3 days**.
- ALFWorld test set = **140 valid_seen tasks**; per-type counts: Pick 35, Look 13, Clean 27,
  Heat 16, Cool 25, Pick2 24. Training split = **3,553 tasks**.
- Judge for `r_cnt` = **Qwen3-32B** (LLM-as-a-judge). Correctness signal `1_ξt` via
  LLM-as-a-judge using the frozen executor backbone.
- Reward weights: `λ_f = 1.0` (function call), `λ_u = 0.1` (content quality), `λ_c = 0.05` (compression).

---

## Abstract

LLM-based agents are increasingly deployed to handle streaming tasks, yet they often remain
one-off problem solvers that fail to learn from past interactions. Reusable skills distilled from
experience provide a natural substrate for self-evolution, where high-quality skill curation
serves as the key bottleneck. Existing approaches either rely on manual skill curation, prescribe
heuristic skill operations, or train for short-horizon skill adaptation, but still struggle to
learn complex long-term curation policies from indirect and delayed feedback. We propose
**SkillOS**, an experience-driven RL training recipe for learning skill curation in self-evolving
agents. SkillOS pairs a **frozen agent executor** that retrieves and applies skills with a
**trainable skill curator** that updates an external **SkillRepo** from accumulated experience. To
provide learning signals for curation, we train on grouped task streams based on skill-relevant
task dependencies, where earlier trajectories update the SkillRepo, and later related tasks
evaluate these updates. We further design composite rewards to better attribute downstream
executor feedback to curation decisions. Across multi-turn agentic tasks and single-turn reasoning
tasks, SkillOS consistently outperforms memory-free and strong memory-based baselines in both
effectiveness and efficiency, with the learned skill curator generalizing across different executor
backbones and task domains. Further analyses show that the learned curator produces more targeted
skill use, while the evolving SkillRepo develops richer internal structure and higher-level
meta-skills over time.

---

## 1 Introduction

LLM-based agents are increasingly deployed in real-world scenarios, where they must move beyond
instantaneous problem-solving toward long-term proficiency. However, the prevailing paradigm of
"one-off" task execution limits their utility in streaming settings, where tasks unfold
sequentially over time. This makes **self-evolution** essential: capable agents should not
repeatedly start from scratch, but instead continually accumulate, refine, and reuse experience
for future tasks.

A key substrate for self-evolution is **procedural memory**, specifically reusable **skills**
accumulated from past interactions. In real-world streaming settings, a skill-based self-evolving
agent typically follows a closed-loop workflow: for each new task, it selects relevant skills, uses
them to guide execution, and updates its skill collection based on the resulting trajectory. This
makes **skill curation** — the extraction of high-quality lessons and their integration into the
skill collection — essential.

However, existing skill curation works remain limited. Manually curated skills (e.g. Anthropic's
skills repository) demand huge human expertise and cannot scale. Prompting/heuristic-based methods
that dictate memory operations rely on fixed rules and lack downstream performance feedback. Recent
RL studies either focus on teaching agents to *use* skills, or optimize skill operations within a
short task stream — limiting the density of learning signals for curating highly reusable skills
and mastering complex management operations such as update and deletion.

**SkillOS** is an experience-driven RL training recipe to learn skill curation. A frozen agent
executor solves tasks with a skill collection (**SkillRepo**), while a trainable skill curator
updates and manages this collection through function calls. Skills are **Markdown files** managed
via file I/O operations (like an OS). Two core designs:

1. Each training instance is a **group of related tasks** solved sequentially — grounding curation
   in long-term utility (skills from earlier tasks evaluated by whether they help later tasks).
2. **Composite rewards** that attribute environmental feedback to curation decisions: task
   performance + valid function calls + skill quality + SkillRepo compactness.

Results: up to **+9.8% relative** performance improvement and **−6.0% fewer interaction steps** vs
the strongest baseline. The trained curator generalizes across executors (incl. Gemini-2.5-Pro) and
tasks. Notably, the 8B curator outperforms Gemini-2.5-Pro used directly as curator.

---

## 2 Related Work

**Memory for Self-Evolving Agents.** Encoding interaction histories into reusable, retrievable
representations. *Case-based*: raw trajectories, abstracted query–response pairs. *Strategy-based*:
reusable workflows, distilled insights, recurring patterns. **Skills** have emerged as an
agent-native memory form and orchestrable capability layer. Anthropic conceptualizes each skill as
a folder of instructions/scripts/resources; this work simplifies each skill to a single Markdown
file.

**Learning Memory and Skill Curation with RL.** One line trains long-context management with
predefined ops (compaction). Another focuses on memory utilization/management via memory tool-calls
or stage-specific policies (e.g. retrieval). RL has been applied across agent skill development:
SkillRL and D2Skill teach smaller models to *use* skills curated from powerful LLMs; ARISE trains a
shared retriever+worker policy with heuristic management. Prior curation training is mostly
restricted to local adaptation within short task streams — favoring insertion, with limited signal
for revising/deleting. SkillOS instead formulates curation as a **long-horizon, executor-grounded**
learning problem with grouped tasks and combined downstream + intermediate rewards.

---

## 3 Methodology

### 3.1 Streaming Skill Curation with Multi-Agent Modular Design

Streaming test-time setting: an agent solves a sequence of tasks `D = {x_1, …, x_T}` arriving over
time. At each timestamp `t`, it must solve `x_t` before observing future tasks, producing trajectory
`ξ_t = {o_1, a_1, …, o_n, a_n}`.

**Skill Repository.** An external repo `S_t` of `N_t` skills. Following the SKILL.md format, each
skill is a single Markdown file with (i) **YAML frontmatter** (skill name + natural-language
description of *when* to use it) and (ii) **Markdown instructions** (executable knowledge,
workflows, constraints, reusable heuristics).

**Agent Executor.** A frozen executor `π_L` solves task `x_t` conditioned on the current observation
and relevant skills. Skills `S̃_t ⊆ S_t` are retrieved via **BM25**; the executor samples
`a ∼ π_L(· | x_t, o_t, S̃_t)`.

**Skill Curator.** After the executor completes `x_t`, the curator `π_S` observes the trajectory
`ξ_t`, the self-judged correctness `1_{ξ_t}`, and retrieved related skills `S̃_t`. It generates a
sequence of structured curation operations `c_t = (u_t^1, …, u_t^{M_t}) ∼ π_S(· | ξ_t, 1_{ξ_t}, S̃_t)`,
where each `u_t^m ∈ {insert_skill, update_skill, delete_skill}`. Each op is a function call that
manipulates `S_t`. Applying ops gives `S_{t+1} = ApplyOps(S_t, c_t)`. The updated repo is used by the
executor on subsequent tasks — a closed loop.

### 3.2 Learning Skill Curation with RL

Optimize `π_S` with RL; keep `π_L` frozen. Challenge: indirect, delayed feedback (revealed only
through `π_L`'s performance on future relevant tasks). Addressed via grouped training instances
(§3.2.1) and a composite reward (§3.2.2).

#### 3.2.1 Training Instance Construction

Each training instance = a **group of related tasks** solved sequentially. Within a group, SkillRepo
is updated by the curator after each task; skills from earlier experiences are evaluated by whether
they help later related tasks. This exposes the curator to longer skill-evolution trajectories and
denser feedback than short-horizon transfer work.

For each task `x_i`, annotate with skill-relevant attributes `Z_i = {z_i^1, …}` via Gemini-2.5-Pro
(topic, common pitfalls, etc.). Partition `D` into `M` task groups by attribute similarity; all
instances in a group share non-trivial skill dependency. For ALFWorld, the **default 6 task types**
are used directly as the partition (no LLM annotation needed).

#### 3.2.2 Training Loop and Policy Optimization

**GRPO** is used. For a task group `G = (x_1, …, x_|G|)`, the curator produces curation decisions
`c = (c_1, …, c_|G|)` as the executor proceeds through the group. The reward combines four signals:

```
r = r_task              (task outcome)
  + λ_f · r_fc          (function call validity)
  + λ_u · r_cnt         (content quality, judged by Qwen3-32B)
  + λ_c · r_comp        (compression / conciseness)
```

- **Task outcome reward.** The *first* task uses an empty SkillRepo (before any curator update). So
  `r_task = (1 / (|G|−1)) · Σ_{i=2}^{|G|} 1(ξ_i)` — average success over the *remaining* tasks.
- **Function call reward.** `r_fc = (1/|G|) · Σ_i Valid(c_i)`, the fraction of generated function
  calls that are valid and successfully executed.
- **Compression reward.** `r_comp = (1/|G|) · Σ_i (1 − |S_i| / |χ_i|)` where `|S_i|` is the
  post-update repo token length and `|χ_i|` is the curator input context token length. Discourages
  verbatim trajectory copying.
- **Content quality reward.** `r_cnt = (1/|G|) · Σ_i Judge(c_i)`, where `Judge` is an external
  Qwen3-32B judge scoring whether curated skills are semantically meaningful / likely useful.

For each group, sample `N` independent rollouts of the entire curation sequence. GRPO advantage:
`A_n = r_n − (1/N) Σ_{n'} r_{n'}`. Clipped surrogate objective over all curation steps:

```
L = E_n[ min( ρ_n A_n, clip(ρ_n, 1−ε, 1+ε) A_n ) ]
```

with importance ratio `ρ_n = π_S(c_n|χ) / π_{θ_old}(c_n|χ)`. Advantage assigned uniformly to all
tokens in `c_n`. **The KL term is discarded** to encourage exploration. (Note: Table 4 still lists a
KL loss coef 0.001 — see hyperparameters.)

**Algorithm 1 (Training Skill Curator with Task Groups using GRPO):**

```
for each training step do
  G = (x_1, …, x_|G|);  S ← ∅                    # sample a task group, init empty SkillRepo
  for task index i = 1, …, |G| do
    S̃ ← BM25(x_i, S)                             # retrieve relevant skills
    ξ_i ← RunTask(S̃, π_L, x_i)                   # run inference on frozen executor
    c_i ∼ π_S(· | ξ_i, S̃)                        # sample a rollout from skill curator
    S ← ApplyOps(S, c_i)                          # apply insert/update/delete
  end for
  r ← CalculateReward(ξ, c)
  Update π_S                                       # GRPO update
end for
```

---

## 4 Experiments

### Table 1 — ALFWorld (SR ↑ / Steps ↓), 3 frozen executors

Per-type SR with subset sizes: Pick(35) Look(13) Clean(27) Heat(16) Cool(25) Pick2(24); Avg over 140.

**Executor π_L: Qwen3-8B**

| Method | Curator | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. SR | Steps |
|---|---|---|---|---|---|---|---|---|---|
| No Memory | None | 78.1±1.6 | 46.2±7.7 | 33.3±13.4 | 37.5±10.8 | 29.3±6.1 | 47.2±6.4 | **47.9±1.2** | 21.1 |
| ReasoningBank | Qwen3-8B | 83.8±0.0 | 48.7±7.2 | 49.4±16.2 | 39.6±4.4 | 41.3±8.5 | 54.2±8.8 | 55.7±3.1 | 20.1 |
| MemP | Qwen3-8B | 80.0±5.7 | 43.6±4.4 | 24.7±4.3 | 33.3±3.6 | 38.7±6.1 | 48.6±6.4 | 49.7±0.7 | 21.0 |
| SkillOS-base | Qwen3-8B | 79.0±8.7 | 41.0±4.4 | 45.7±4.3 | 37.5±9.5 | 38.7±4.0 | 55.6±2.1 | 53.1±2.5 | 20.4 |
| SkillOS-gemini | Gemini-2.5-Pro | 77.1±6.0 | 53.8±6.1 | 37.0±6.4 | 37.5±9.5 | 36.0±3.2 | 50.0±6.7 | 50.7±3.6 | 20.8 |
| **SkillOS** | Qwen3-8B | 85.7±3.3 | 56.4±7.7 | 54.3±8.6 | 43.8±9.5 | 46.7±2.3 | 62.5±6.4 | **61.2±4.6** | **18.9** |

**Executor π_L: Qwen3-32B**

| Method | Curator | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. SR | Steps |
|---|---|---|---|---|---|---|---|---|---|
| No Memory | None | 80.0±2.9 | 69.2±0.0 | 45.6±7.7 | 37.5±16.5 | 42.7±6.1 | 43.1±2.4 | 54.5±2.5 | 20.3 |
| ReasoningBank | Qwen3-8B | 86.7±3.0 | 71.8±5.4 | 50.6±6.3 | 45.8±13.3 | 52.0±8.9 | 51.4±5.1 | 61.4±2.5 | 18.7 |
| MemP | Qwen3-8B | 80.0±2.9 | 76.9±0.0 | 44.4±7.4 | 37.5±10.8 | 42.7±2.3 | 47.2±6.4 | 55.7±3.7 | 20.0 |
| SkillOS-base | Qwen3-8B | 82.9±2.9 | 69.2±11.8 | 48.1±2.1 | 50.0±9.7 | 48.0±14.4 | 52.8±11.0 | 59.8±3.0 | 19.2 |
| SkillOS-gemini | Gemini-2.5-Pro | 97.1±3.0 | 76.9±5.4 | 55.6±6.0 | 43.8±11.3 | 40.0±5.7 | 54.2±4.9 | 63.6±4.2 | 18.1 |
| **SkillOS** | Qwen3-8B | 91.4±3.3 | 76.9±4.4 | 59.3±8.6 | 56.3±12.5 | 57.3±10.1 | 62.5±4.2 | **68.6±5.7** | **17.3** |

**Executor π_L: Gemini-2.5-Pro**

| Method | Curator | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. SR | Steps |
|---|---|---|---|---|---|---|---|---|---|
| No Memory | None | 90.5±3.2 | 66.7±5.1 | 48.1±10.2 | 39.6±17.1 | 68.0±7.4 | 68.1±3.8 | 66.4±2.0 | 17.7 |
| ReasoningBank | Qwen3-8B | 91.4±3.4 | 61.5±4.1 | 63.0±9.3 | 39.6±10.3 | 70.7±3.2 | 76.4±8.5 | 71.4±2.9 | 16.0 |
| MemP | Qwen3-8B | 95.2±2.1 | 74.4±6.8 | 61.7±7.6 | 56.3±12.4 | 76.0±6.2 | 68.1±8.5 | 74.3±3.4 | 15.2 |
| SkillOS-base | Qwen3-8B | 91.4±1.6 | 69.2±7.7 | 56.8±5.7 | 54.2±13.7 | 72.0±4.0 | 66.7±11.0 | 70.7±3.0 | 16.3 |
| SkillOS-gemini | Gemini-2.5-Pro | 94.3±5.7 | 69.2±0.0 | 77.8±5.7 | 75.0±16.5 | 80.0±12.2 | 66.7±2.4 | 79.3±2.6 | 14.9 |
| **SkillOS** | Qwen3-8B | 95.2±2.9 | 71.8±7.7 | 74.1±13.0 | 72.9±10.1 | 77.3±6.1 | 77.8±10.0 | **80.2±3.1** | **14.8** |

### Table 2 — WebShop (Score / SR ↑ / Steps ↓) and Reasoning (Acc. ↑)

**Executor π_L: Qwen3-8B**

| Method | Curator | Score | SR | Steps | AIME24 | AIME25 | GPQA | Avg. Acc |
|---|---|---|---|---|---|---|---|---|
| No Memory | None | 33.3±0.7 | 9.8±0.5 | 20.3 | 76.0±6.9 | 71.1±10.7 | 61.8±1.1 | 69.6±4.7 |
| ReasoningBank | Qwen3-8B | 35.4±1.1 | 11.4±0.9 | 20.5 | 75.4±5.0 | 73.2±10.8 | 60.3±3.9 | 69.6±2.5 |
| MemP | Qwen3-8B | 35.7±0.9 | 12.0±0.5 | 21.3 | 75.6±5.1 | 71.1±5.1 | 60.6±4.0 | 69.1±4.0 |
| SkillOS-base | Qwen3-8B | 38.6±0.9 | 13.6±0.8 | 20.1 | 75.6±5.1 | 71.9±6.9 | 59.3±2.5 | 68.9±2.6 |
| SkillOS-gemini | Gemini-2.5-pro | 38.1±1.0 | 13.2±0.9 | 19.6 | 73.3±1.3 | 71.3±1.9 | 57.6±2.8 | 67.4±0.8 |
| **SkillOS** | Qwen3-8B | 40.6±0.7 | 16.5±0.7 | 19.4 | 80.0±3.3 | 76.7±5.8 | 64.6±1.3 | 73.8±1.8 |

**Executor π_L: Qwen3-32B**

| Method | Curator | Score | SR | Steps | AIME24 | AIME25 | GPQA | Avg. Acc |
|---|---|---|---|---|---|---|---|---|
| No Memory | None | 41.5±0.5 | 12.2±0.3 | 17.0 | 81.4±1.3 | 72.2±3.8 | 68.4±2.0 | 74.0±1.9 |
| ReasoningBank | Qwen3-32B | 40.4±0.8 | 11.2±1.1 | 17.9 | 81.1±9.6 | 75.6±5.9 | 66.9±1.2 | 74.9±2.2 |
| MemP | Qwen3-32B | 30.7±0.7 | 10.1±0.6 | 17.4 | 82.2±5.1 | 76.7±0.0 | 66.5±2.3 | 75.1±2.1 |
| SkillOS-base | Qwen3-8B | 43.4±0.8 | 12.3±1.0 | 16.8 | 80.0±3.3 | 75.6±10.2 | 67.7±1.5 | 74.7±3.3 |
| SkillOS-gemini | Gemini-2.5-pro | 45.2±1.0 | 13.2±1.1 | 16.6 | 77.8±6.7 | 74.4±1.9 | 66.2±0.6 | 73.2±2.6 |
| **SkillOS** | Qwen3-8B | 49.2±1.2 | 16.5±0.6 | 15.9 | 85.6±1.9 | 81.1±3.3 | 72.4±3.0 | 79.7±1.6 |

**Executor π_L: Gemini-2.5-pro**

| Method | Curator | Score | SR | Steps | AIME24 | AIME25 | GPQA | Avg. Acc |
|---|---|---|---|---|---|---|---|---|
| No Memory | None | 48.6±0.3 | 38.4±0.5 | 19.5 | 85.6±1.9 | 80.0±6.7 | 79.9±1.5 | 81.8±2.8 |
| ReasoningBank | Gemini-2.5-pro | 50.8±1.5 | 40.2±1.3 | 19.2 | 85.6±5.1 | 84.4±6.7 | 80.4±2.1 | 83.5±2.1 |
| MemP | Gemini-2.5-pro | 51.3±1.2 | 39.8±1.0 | 19.4 | 83.3±6.9 | 76.7±5.8 | 81.8±3.4 | 80.6±3.2 |
| SkillOS-base | Qwen3-8B | 52.8±1.0 | 39.6±0.8 | 19.0 | 87.8±3.3 | 83.3±1.9 | 82.8±2.7 | 84.6±1.8 |
| SkillOS-gemini | Gemini-2.5-pro | 54.7±1.0 | 41.0±1.2 | 17.8 | 90.0±5.1 | 85.6±7.7 | 80.7±5.5 | 85.4±3.5 |
| **SkillOS** | Qwen3-8B | 56.0±0.7 | 41.3±0.8 | 18.3 | 92.2±2.4 | 86.7±3.5 | 86.8±2.1 | 88.6±1.5 |

### 4.1 Setup (summary)

- **Datasets.** Agentic: ALFWorld, WebShop. Reasoning: AIME24, AIME25, GPQA-Diamond. Reasoning
  training data from DeepMath-103K (~33K sampled → 20K grouped instances).
- **Metrics.** Effectiveness: SR (agentic), accuracy (reasoning). Efficiency: steps/task (agentic),
  tokens/problem (reasoning).
- **Baselines.** (i) No Memory; (ii) ReasoningBank (distilled insights), MemP (procedural memory +
  management); (iii) SkillOS-base (curator, no RL), SkillOS-gemini (Gemini-2.5-Pro as curator).
- **Implementation.** `π_S` = Qwen3-8B. Frozen executor = Qwen3-8B during training. GRPO, lr 1e-6,
  batch 32, group 8, on **16 H100s** with **verl**. ALFWorld ~3 days, reasoning ~2.5 days,
  WebShop ~5 days. Test executors also include Qwen3-32B, Gemini-2.5-Pro, Gemini-3.1-Flash-Lite.
  **ReAct** for agentic, **CoT** for reasoning. Reward weights `λ_f=1.0, λ_u=0.1, λ_c=0.05`.
  Reported as mean ± std over 3 runs.

### 4.2 Main Results (key claims)

- **Strong gains across benchmarks.** ALFWorld avg SR 55.7 → 61.2 over strongest baseline
  (ReasoningBank, Qwen3-8B executor). The RL-trained 8B curator surpasses SkillOS-gemini despite the
  latter using a frontier model as curator.
- **More efficient.** ALFWorld reduces avg steps by 2.2 / 3.0 / 3.1 vs No Memory across the 3
  executors — gains come from procedural shortcuts, not longer trajectories.
- **Agentic > reasoning gains.** Agentic tasks expose procedural regularities (action ordering,
  exploration, recovery, constraints) that compose across streams; reasoning skills are more
  abstract, so gains are smaller.

### 4.3 Generalization

- **Transfers across executors.** Trained on Qwen3-8B executor; still lifts Qwen3-32B and
  Gemini-2.5-Pro (e.g. Qwen3-8B 47.9 → 61.2; Gemini-2.5-Pro 66.4 → 80.2). Using Gemini-2.5-Pro
  directly as curator (SkillOS-gemini) underperforms the trained curator — a **curator–executor
  mismatch**: stronger reasoning ≠ better curation; RL learns executor-grounded curation.
- **Transfers across task domains** (Figure 3). Curators trained on reasoning transfer especially
  well to agentic tasks (abstract strategies); WebShop/ALFWorld skills are more environment-specific.

---

## 5 Analysis

### Table 3 — Ablation of reward / grouping design (ALFWorld, Qwen3-8B curator+executor)

| Method | Avg. SR | Steps |
|---|---|---|
| SkillOS-GRPO (full) | 61.2 | 18.9 |
| w/o `r_cnt` (content quality) | 58.6 | 20.1 |
| w/o `r_comp` (compression) | 60.0 | 19.3 |
| w/o grouping (random task seq) | 57.3 | 20.6 |

Removing either reward hurts; **grouping is the most important** (largest drop, 61.2 → 57.3).

- **Behaviors of Skill Curator (Fig 4).** Early training: `insert` dominates (populating the repo).
  As training progresses: `update` rises and `insert` declines (refining existing skills); `delete`
  stays small with a slight growing trend. Adaptation shifts toward revising/consolidating.
- **Skill Evolution Dynamics (Fig 5).** (a) Within skills: early additions are generic
  (guidance/tips); later they become actionable (failure-handling, conditional branches). (b) Across
  the repo: early = narrow task-specific skills; later = diverse **meta-strategy** skills
  (verification, fallback planning, system search, strategy adjustment).
- **Attribution of Skill Usage (Fig 6).** SkillOS invokes skills on all eval examples, higher
  success among skill-using examples, higher skill coverage, and **fewer skills used per example** —
  more precise selection, not more context.

---

## 6 Conclusion

SkillOS decouples a trainable skill curator from a frozen executor, enabling modular skill curation
without retraining the executor. Through grouped task streams and executor-grounded rewards, it
optimizes curation by downstream impact on future tasks. It improves both performance and
efficiency across benchmarks and backbones, can outperform frontier zero-shot curation, and
generalizes across settings.

---

## Appendix A — Prompts (pointers)

- **A.1 Skill Curator** — system prompt (Fig 7) and tool-call signatures for
  `insert_skill / update_skill / delete_skill` (Fig 8).
- **A.2 Agent Executor** — ALFWorld (Fig 9), WebShop (Fig 10), reasoning (Fig 11). **All prompts
  explicitly force chain-of-thought reasoning.** For ALFWorld/WebShop they follow **GiGPO**'s
  environment and prompt setting.
- **A.3 Content-quality judge** — Qwen3-32B judge prompt for `r_cnt` (Fig 12).
- **A.4 LLM-as-a-judge correctness** — `1_{ξt}` prompts: ALFWorld (Fig 13), reasoning (Fig 14),
  WebShop (Fig 15), using the frozen executor backbone as judge.

## Appendix B — Implementation Details

### Table 4 — Hyperparameters (training + inference)

| Hyperparameter | ALFWorld | WebShop | Reasoning |
|---|---|---|---|
| **RL Training** | | | |
| Learning rate | 1×10⁻⁶ | 1×10⁻⁶ | 1×10⁻⁶ |
| Batch size | 32 | 32 | 32 |
| KL loss Coef | 0.001 | 0.001 | 0.001 |
| Max Prompt Length | 16,384 | 16,384 | 16,384 |
| Max Response Length | 4,096 | 4,096 | 4,096 |
| GRPO group size | 8 | 8 | 8 |
| Temperature | 1.0 | 1.0 | 1.0 |
| **Steps** | **60** | **50** | **100** |
| Data Grouping Size | 10 | 10 | Random(5,12) |
| **Agent Executor Inference** | | | |
| Top-K skill retrieval | 5 | 5 | 5 |
| Max number of turns | 30 | 30 | 1 |
| Action history length | 3 | 3 | - |

> **NOTE — Temperature 1.0 above is the CURATOR's RL-rollout temperature, not the executor's.**
> The "Agent Executor Inference" block gives no temperature/top_p/max_tokens. The paper says
> executor inference "follows GiGPO" (§4.1 / A.2), so the executor decode config lives in GiGPO's
> code, not this paper. See below.

### Executor decode settings — inherited from GiGPO (NOT printed in this paper)

SkillOS §4.1 / Appendix A.2: *"we follow GiGPO and leverage its environment and prompt setting for
inference."* GiGPO = **arXiv:2505.10978** (Feng et al., NeurIPS 2025); code:
**github.com/langfengQ/verl-agent**. The authoritative executor decode config is in
`examples/gigpo_trainer/run_alfworld.sh` (verbatim):

| Setting | GiGPO value | Notes |
|---|---|---|
| `val_kwargs.temperature` | **0.4** | evaluation/inference temperature |
| `val_kwargs.do_sample` | **True** | sampling on (not greedy) |
| top_p / top_k | (unset → verl defaults: top_p 1.0, top_k off) | |
| `data.max_response_length` | **512** | max output tokens per step |
| `data.max_prompt_length` | 2048 | |
| `env.max_steps` | 50 | **SkillOS overrode this to 30** (Table 4) |
| base model | Qwen2.5-1.5B / 7B-Instruct | **NOT a reasoning model** — `<think>` is prompt-induced CoT in the response text |

**Prompt/format (GiGPO, matches ours):** the agent emits chain-of-thought in `<think> </think>`
then the action in `<action> </action>`. Our `ALFWORLD_EXECUTOR` prompt + `_parse_action`
(`skillos/executor/executor.py`) already match this format.

**Critical Qwen3 caveat (our setup, learned this session):** GiGPO used Qwen2.5-Instruct (no native
reasoning channel), so its `<think>` CoT is plain text inside the 512-token response. We run
**Qwen3-8B** on `openrouter/qwen3-8b`, which has a *separate native reasoning channel*. Audited via
`scripts/debug_executor_audit.py` on a real ALFWorld step:
- `reasoning_effort="none"` is **silently ignored** by the app (reasoning still generated).
- Only `reasoning_max_tokens=0` + `reasoning_exclude=True` truly disables it — **but that also kills
  the CoT entirely** (model emits a bare `<action>` with no reasoning), which is off-spec and hurts.
- So for Qwen3 we **keep native reasoning ON** (CoT happens in the `reasoning` field; the `<action>`
  lands in `response` and parses fine). The faithful adaptation is: **temp 0.4, top_p 1.0,
  do_sample, reasoning on, generous token budget** — NOT GiGPO's 512 (which would truncate Qwen3's
  longer native reasoning).

### B.2 Grouping Training Instances (summary)

Two-stage pipeline to turn raw `D` into grouped `G`:

- **Stage 1 — Latent Attribute Annotation.** `Z_i = (T_i, S_i, C_i, R_i, P_i)` = phrase-lists for
  topics, required skills, concepts/theorems, heuristic strategies, common pitfalls. Annotated via
  Gemini-2.5-Pro with structured JSON decoding (highest thinking-budget). For ALFWorld, the default
  6 task types are used directly instead of LLM annotation.
- **Stage 2 — Group Construction.** Seed a task, iteratively append related successors. Phrase
  similarity via **soft-Jaccard** over `all-MiniLM-L6-v2` cosine (threshold τ). A **dependency gate**
  enforces: shared foundation (concepts+skills), shared reasoning, not-near-duplicate,
  not-too-unrelated, progression (new concept/skill), and curriculum direction (difficulty delta).
  Candidate retrieval via an inverted index over dependency fields {C,R,P}, fallback to random pool.

### Table 5 — Stage 2 grouping hyperparameters

| Symbol | Meaning | Value |
|---|---|---|
| — | Phrase encoder | all-MiniLM-L6-v2 |
| τ | Cosine threshold for fuzzy phrase matching | 0.60 |
| κ_C | Min matched concept pairs | 1 |
| κ_S | Min matched skill pairs | 1 |
| θ_T | Max topic soft-Jaccard | 0.65 |
| σ_min, σ_max | Overall-similarity band | 0.30, 0.85 |
| δ_min | Difficulty-delta floor | 0.0 |
| (w_C, w_S, w_R, w_P, w_T) | Dimension weights | (5, 4, 3, 1, 2) |
| λ | Difficulty-bonus weight | 1.0 |
| (p↑, p=, p↓) | Mode probabilities | (0.80, 0.20, 0.00) |
| [Δ_min, Δ_max] | Gap in easy→hard mode | [0.5, 3.0] |
| Δ_= | Max \|d_t − d_s\| in same mode | 0.3 |
| K_inv | Inverted-index subsample cap | 2,000 |
| F | Fallback pool size | 200 |

### B.3 Datasets (key facts)

- **ALFWorld.** Text-based; TextWorld engine + embodied ALFRED. **6 task types** (Pick & Place,
  Examine in Light, Clean & Place, Heat & Place, Cool & Place, Pick Two & Place) across 120 rooms.
  **3,553 training tasks; 140 valid_seen test tasks.** High-level commands: go to, take, open, heat,
  put.
- **WebShop.** Simulated e-commerce; 1.18M products, 12,087 instructions (10,587 train / 1,000 dev /
  500 test). Actions: `search[query]`, `click[button]`. Programmatic reward in [0,1]. Eval on 500
  held-out test instructions.
- **DeepMath-103K.** ~103K hard math problems with verifiable answers + difficulty + topic + 3
  DeepSeek-R1 CoT solutions. ~33K annotated → 20K grouped instances.
- **AIME24/25.** 30 problems each. **GPQA-Diamond.** 198 multiple-choice problems.

### B.3.3 Evaluation Metrics (summary)

- **SR (↑).** ALFWorld: fraction of episodes reaching goal within step budget (binary per episode);
  reported per task category and as **macro-average over the 6 categories** (Avg. SR). WebShop:
  fraction with final reward exactly 1.
- **WebShop Score (↑).** Dense per-episode reward in [0,100] for partial matches.
- **Steps (↓).** Avg environment actions per episode over all eval episodes (failures count up to
  termination).
- **Accuracy (↑).** Reasoning: exact-match (math_verify for AIME, option-letter for GPQA).
- **Protocol.** All methods share the same frozen executor, retrieval budget (top-k BM25), max step
  budget, and decoding temperature within each backbone; numbers on official held-out splits.

## Appendix C — Additional Analyses

### C.1 Table 6 — ALFWorld with Gemini-3.1-Flash-Lite executor

SkillOS achieves the highest avg SR (73.1%), beating ReasoningBank (66.0) by +7.1 and No Memory
(61.2) by +11.9, with the fewest steps (15.5 vs 18.5). MemP (58.6) underperforms even No Memory here
— hand-designed heuristics are brittle when the executor is weaker.

| Method | Pick(35) | Look(13) | Clean(27) | Heat(16) | Cool(25) | Pick2(24) | Avg. SR(140) | Steps |
|---|---|---|---|---|---|---|---|---|
| No Memory | 85.7±0.0 | 59.0±8.9 | 67.9±9.3 | 25.0±6.2 | 38.7±2.3 | 66.7±0.0 | 61.2±2.3 | 18.5 |
| ReasoningBank | 87.6±4.4 | 71.8±4.4 | 63.0±0.0 | 52.1±14.4 | 48.0±10.6 | 62.5±0.0 | 66.0±2.7 | 17.6 |
| MemP | 84.3±6.1 | 57.7±5.4 | 63.0±0.0 | 28.1±4.4 | 34.0±2.8 | 62.5±0.0 | 58.6±1.0 | 19.3 |
| SkillOS-base | 86.7±1.6 | 61.5±0.0 | 66.7±0.0 | 41.7±6.2 | 38.7±16.0 | 68.1±2.4 | 63.6±3.9 | 17.7 |
| SkillOS-gemini | 96.2±1.6 | 61.5±13.3 | 74.1±3.7 | 31.2±12.5 | 66.7±4.6 | 68.1±2.4 | 71.2±2.9 | 16.1 |
| **SkillOS** | 88.6±0.0 | 84.6±13.3 | 77.8±0.0 | 37.5±17.2 | 68.0±8.0 | 68.1±2.4 | **73.1±2.7** | **15.5** |

### C.2 Case Studies (summary)

- **Fig 17.** Agentic: curator distills a *failure-recovery meta-strategy* (exhaustive search →
  confirm unavailability → identify substitute → proceed), referencing existing skills
  (compositional). Reasoning: a single skill encodes *multiple solution paths* with formulas and
  preconditions.
- **Fig 18.** SkillOS-base produces a generic partitioning recipe; SkillOS curates a concrete,
  reusable counting framework with explicit constraints, equations, and a worked example.
- **Fig 19.** ALFWorld "look at the CD under the desklamp": No-Memory does an inefficient search and
  exhausts the budget; SkillOS retrieves a skill ("examine objects under/around light sources") and
  completes the task efficiently.

## Appendix D — Limitations

- **Retrieval.** Uses simple keyword-based BM25 to isolate the curation question; dense/hybrid/learned
  retrievers may help. Joint curation+retrieval optimization left to future work.
- **Simplified skill representation.** Single Markdown file (YAML + body) — discards SKILL.md
  affordances: supporting scripts/resources and hierarchical sub-skill composition.
- **Frozen executor.** Only the curator is trained; miscalibration between skills and executor
  idiosyncrasies must be absorbed by the curator. Joint/alternating optimization may align the pair
  better at higher cost.

## Appendix E — Future Directions

Agentic search over experiential memory (replace static top-k with iterative query/reformulation);
hierarchical/compositional skills (link/compose/abstract); multi-agent and shared memory (conflict
arbitration, credit assignment, specialization vs transfer).

## Appendix F — Use of LLMs

LLMs used only as a writing-assist tool (clarity/grammar/concision). All research ideas, methodology,
experiments, analyses, and final writing decisions were by the authors.
