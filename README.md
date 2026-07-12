# SkillOS

Open reproduction of ["SkillOS: Learning Skill Curation for Self-Evolving Agents"](https://arxiv.org/abs/2605.06614) (Google Cloud AI Research + UIUC + MIT, 2026) on [HuggingFace TRL](https://github.com/huggingface/trl). The paper trains a curator with GRPO on 16 H100s using verl; this repo reproduces the same result on 8 H100s using TRL, with all training and eval code, three benchmarks (ALFWorld + AIME + GPQA-Diamond), and every deviation logged.

<p align="center">
  <img src="assets/banner.png" alt="SkillOS Training Loop" width="720" />
</p>

## What is SkillOS

The idea: freeze the executor (the agent that actually solves tasks), train only a **curator** (a separate LLM) whose job is to maintain a markdown skill repo. The curator writes/updates/deletes skills after each rollout via a `curate_and_advance` tool call. The executor retrieves the top-k relevant skills before each new task. GRPO optimises the curator against a composite reward: task success + valid tool calls + judge-scored content quality + repo compression.

Skill files are markdown with YAML frontmatter, the same format used by [Anthropic's Skills](https://docs.anthropic.com/en/docs/agents/skills) and the [belt CLI](https://github.com/belt-sh/cli).

## Headline results

Best single-run numbers, paired McNemar vs no-memory baseline unless noted. Full narrative in [`JOURNAL.md`](JOURNAL.md), every deviation from the paper in [`DIVERGENCES.md`](DIVERGENCES.md).

**Cross-executor transfer (paper's generalisation claim):**

| curator (8B-trained) → executor | abs SR | Δ vs no-memory | p (McNemar, n=140) |
|---|---|---|---|
| v8-lora ckpt30 → Qwen3-32B | **62.1%** | **+12.9pp** | **0.0064** |
| paper headline (SkillOS, Qwen3-32B executor) | 61.2% | ~+13pp | — |

Above the paper's headline on this single run. Baseline stochasticity is ~±4pp, so treat as "reproduces at parity" rather than "beats."

**Reasoning baselines (no-memory, Qwen3-8B executor):**

| dataset | ours | paper (Qwen3-8B) | delta |
|---|---|---|---|
| AIME24 | 22/30 = 73.3% | 76.0±6.9 | −2.7pp (0.4σ) |
| AIME25 | 18/30 = 60.0% | 71.1±10.7 | −11.1pp (1.0σ) |
| GPQA-Diamond | 118/198 = 59.6% | 61.8±1.1 | −2.2pp (2.0σ) |
| **average** | **64.3%** | **69.6±4.7** | **−5.3pp (1.1σ)** |

**ALFWorld baselines, same executor:**

| method | ours | paper | delta |
|---|---|---|---|
| No Memory | 33.6% | 47.9% | −14.3pp |
| SkillOS (best 8B curator, on 8B executor) | 47.1% (seed-2 ckpt35) | 61.2% | −14.1pp |

The −14pp baseline gap is **environment-specific**: same executor reproduces the paper on reasoning within noise, so the ALFWorld gap is the ReAct/atomic-verb interaction, not model quality. Details: [`DIVERGENCES.md`](DIVERGENCES.md) #13, gotcha `executor-atomic-verb-gap`.

## Training trajectory is bimodal on TRL, not monotone

Held-out lift over 60 training steps, sweep vs canonical 33.6% baseline:

| run | peak ckpt | peak lift | p |
|---|---|---|---|
| seed-1 FFT | ckpt20 | +10.7pp | 0.032 |
| seed-2 FFT (seed=123) | ckpt35 | +13.6pp | 0.0026 |
| seed-3 FFT (seed=456) | *training as of 2026-07-12* | — | — |

Peak index is RNG-dependent, shape holds. The paper reports monotone-to-60. Systematically falsified as causes of the shape: uniform vs natural type distribution, easy→hard within-group curriculum (both halves of DIVERGENCES #0). Only surviving suspect: TRL vs verl (advantage normalisation, sampling, buffer semantics). See gotcha `fft-bimodal-not-lora`.

## Quick start

```bash
# Clone + install
git clone https://github.com/belt-sh/skillos && cd skillos
pip install -e ".[dev]"

# ALFWorld data (one-time)
alfworld-download -f
export ALFWORLD_DATA=~/.cache/alfworld

# Smoke: verify all deps + heuristic pipeline
python -m skillos.smoke_test

# Train (single GPU, LoRA, heuristic executor — no API needed)
python -m skillos.train --config configs/alfworld_single_gpu.yaml
```

Reasoning benchmark, no local GPU needed (executor is remote):

```bash
# GPQA-Diamond gated: huggingface-cli login (needs dataset access request)
python -m scripts.eval_reasoning --mode no_memory --dataset aime \
  --executor infsh --executor-app openrouter/qwen3-8b \
  --parallel 12 --out output/eval-reasoning/nomem_aime.jsonl
```

Paper-faithful full training (8×H100, executor + judge remote on inference.sh):

```bash
belt login --key <YOUR_INFERENCE_SH_KEY>   # from https://belt.sh
./run_algo1_fft.sh                          # FFT + ZeRO-3, ~70 min/step, 60 steps
# or LoRA:
./run_algo1_v8_lora_kl.sh                   # LoRA r=32, ~40 min/step
```

## Pluggable backends

Executor and judge both support four backends (heuristic / local / vLLM / inference.sh API):

```yaml
executor:
  type: infsh
  app: openrouter/qwen3-8b
  reasoning_effort: medium

judge:
  type: infsh
  app: openrouter/qwen3-32b
```

Full options and the config precedence chain: see any file under `configs/`.

## Project layout

```
skillos/
  algo1/env.py           # Algorithm 1: |G|=10 evolving task groups, mega-tool
  envs/curator_env.py    # Curator env: runs frozen executor, exposes skill tools
  envs/task_types.py     # ALFWorld task-type taxonomy (single source of truth)
  executor/executor.py   # Pluggable frozen executor (heuristic/local/vLLM/infsh)
  curator/prompts.py     # All prompts verbatim from paper Appendix A
  skills/repo.py         # Markdown skill store + BM25 retrieval
  rewards/composite.py   # r = r_task + λ_f r_fc + λ_u r_cnt + λ_c r_comp
  rewards/judge.py       # Pluggable content quality judge
  reasoning/             # AIME + GPQA eval harness (datasets, prompts, grading)
  train.py               # Path B training (legacy but working)
scripts/
  train_algo1.py         # Algorithm 1 training entrypoint
  eval_streaming_curation.py   # ALFWorld closed-loop eval (no_memory + closed_loop)
  eval_reasoning.py            # Reasoning eval (no_memory today; closed_loop stubbed)
  compare_eval_arms.py         # Paired McNemar comparator over any set of JSONLs
  {natural,curriculum,transfer}_sweep_supervisor.sh   # storm-resilient sweep runners
configs/
  accelerate_zero3.yaml
  alfworld_paper.yaml           # 8×H100 paper-faithful (remote executor + judge)
  alfworld_8xh100_algo1_fft.yaml     # canonical FFT config
  alfworld_8xh100_algo1_v8_lora_kl.yaml   # canonical LoRA config
  alfworld_single_gpu.yaml, alfworld_multi_gpu.yaml   # dev configs
docs/
  repro_report.md        # findings write-up
  training_notes.md      # engineering notes (ZeRO-2 hang, NCCL, storm handling)
  skillos_paper.md       # reconstructed paper text for cross-reference
legacy/                  # superseded launchers and configs, kept for provenance
```

## What's confirmed vs open

Confirmed:
- Algorithm 1 with paper-faithful reward composition, executor/judge/decoder settings
- Held-out lift, ALFWorld: LoRA +9.3pp (p=0.035), FFT seed-1 +10.7pp (p=0.032), seed-2 +13.6pp (p=0.0026)
- Cross-executor transfer: 8B-trained LoRA curator lifts 32B executor +12.9pp (p=0.0064), 62.1% absolute
- Reasoning baseline reproduces paper within 1.1σ on average across AIME24/25 + GPQA-D
- ALFWorld baseline gap is env-specific (same executor matches paper on reasoning)

Open:
- Bimodal trajectory: n=3 in progress. Cause narrowed to TRL vs verl framework difference
- Closed-loop reasoning eval (curator inference blocked on GPUs currently held by seed-3)
- WebShop untouched
- Reasoning-trained curator (paper's headline generalisation direction)

## Hardware

| setup | hardware | use case |
|---|---|---|
| smoke | 1× 8GB+ GPU | pipeline validation, heuristic executor |
| LoRA | 1× H100 (80GB) | single-GPU LoRA training |
| this repo's paper-faithful | 8× H100 + inference.sh (remote 8B executor / 32B judge) | ~3 days for 60 GRPO steps |
| paper original | 16× H100 (verl) | ~3 days |

## Stack

- [TRL](https://github.com/huggingface/trl) — GRPOTrainer with `environment_factory`, multi-turn RL
- [vLLM](https://github.com/vllm-project/vllm) — colocate generation
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) — base curator (Apache 2.0)
- [ALFWorld](https://github.com/alfworld/alfworld) — household tasks
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — skill retrieval
- [inference.sh](https://inference.sh) — remote executor + judge

## References

- [SkillOS: Learning Skill Curation for Self-Evolving Agents](https://arxiv.org/abs/2605.06614) — Ouyang et al., 2026
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) — Shao et al., 2024
- [Anthropic SKILL.md format](https://docs.anthropic.com/en/docs/agents/skills)
- [belt CLI](https://github.com/belt-sh/cli) — agent skill management

## License

Apache 2.0
