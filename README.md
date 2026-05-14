# SkillOS

PyTorch implementation of ["SkillOS: Learning Skill Curation for Self-Evolving Agents"](https://arxiv.org/abs/2605.06614) (Google Cloud AI Research + UIUC + MIT, 2026) using [HuggingFace TRL](https://github.com/huggingface/trl). The paper has no official code release — this repo provides a clean, reproducible implementation with open weights.

**The paper's key finding:** An 8B model trained with RL specifically for skill curation outperforms Gemini-2.5-Pro doing the same thing via prompting. Targeted training on curation decisions > raw model scale.

**What this means:** You can run better agent memory management locally on consumer hardware than you can get from the largest API models.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Executor   │────▶│  Environment │────▶│  Trajectory │
│  (frozen)    │     │  (ALFWorld)  │     │  (results)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
       ▲                                        │
       │ retrieves skills                       ▼
┌──────┴──────┐                          ┌──────────────┐
│  SkillRepo  │◀─────────────────────────│   Curator    │
│  (markdown) │   insert/update/delete   │  (trained)   │
└─────────────┘                          └──────────────┘
```

1. **Frozen executor** solves tasks using retrieved skills
2. **Curator** (the model we train) observes the trajectory and decides what to do with the skill repo
3. **GRPO** optimizes the curator based on whether its curation decisions help future tasks
4. Skills are markdown files with YAML frontmatter — the same format used by [Anthropic](https://docs.anthropic.com/en/docs/agents/skills), [belt](https://belt.sh), and others

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Download ALFWorld data
alfworld-download -f

# Smoke test — verify everything loads
python -m skillos.smoke_test

# Train (single GPU, LoRA)
python -m skillos.train --config configs/alfworld_single_gpu.yaml
```

## Project Structure

```
skillos/
  envs/
    curator_env.py   # Curator's environment: runs frozen executor, exposes skill tools
    alfworld.py      # ALFWorld wrapped as TRL environment (executor-only mode)
  curator/
    model.py         # Parse curator tool calls, apply to skill repo
    prompts.py       # All prompts verbatim from paper Appendix A
  skills/
    repo.py          # Markdown skill store + BM25 retrieval
  rewards/
    composite.py     # r_task + r_fc + r_cnt + r_comp (Eq. 1)
    judge.py         # Content quality judge (heuristic + LLM prompt)
  data/
    grouping.py      # Grouped task stream construction
  train.py           # Main training script (trains curator, not executor)
  smoke_test.py      # Verify setup
configs/
  alfworld_env.yaml            # ALFWorld environment config
  alfworld_single_gpu.yaml     # Single GPU training (LoRA)
  alfworld_multi_gpu.yaml      # Multi GPU training (matches paper)
```

## Roadmap

### v0.1 — Reproduce SkillOS on ALFWorld (in progress)

Reproduce the paper's core result: an RL-trained 8B curator that manages a skill repo for a frozen executor, evaluated on ALFWorld household tasks.

- [ ] ALFWorld environment wrapped as TRL `environment_factory`
- [ ] BM25 skill retrieval + markdown skill store
- [ ] Curator model with insert/update/delete tool calling
- [ ] Composite reward function (task + validity + quality + compression)
- [ ] Grouped task stream data loader
- [ ] Single-GPU training with LoRA
- [ ] Match paper's ALFWorld results (target: 61.2% success rate with Qwen3-8B executor)

### v0.2 — Reproduce SkillOS on Reasoning + Cross-Domain Transfer

Reproduce the remaining benchmarks and the paper's key transfer result: a curator trained on reasoning tasks improves agentic task performance.

- [ ] Reasoning environment (DeepMath-103K + GPQA-Diamond)
- [ ] Match paper's reasoning results (target: 73.8% avg accuracy)
- [ ] Cross-domain transfer: reasoning-trained curator on ALFWorld (+13.3%)
- [ ] WebShop environment integration
- [ ] Multi-GPU scaling configs (8x/16x H100)

### v0.3 — Open Weights Release

Publish trained curator weights so anyone can use a skill curator without training from scratch.

- [ ] Full training runs on all benchmarks
- [ ] Publish trained curator weights on HuggingFace
- [ ] Evaluation suite for comparing curator quality
- [ ] Guide for training on custom task domains

### v1.0 — Production Curator

Take the trained curator from research benchmark to real-world agent memory management.

- [ ] Real-world trajectory extraction (long agent transcripts to structured traces)
- [ ] Curator serving via vLLM/Ollama for local inference
- [ ] Continuous training pipeline on user telemetry
- [ ] Integration with agent frameworks ([belt](https://github.com/belt-sh/cli), Claude Code, LangChain)

## Hardware Requirements

| Setup | Hardware | Use Case |
|---|---|---|
| Development | 1x RTX 6000 Pro (96GB) | Pipeline validation, small-batch training |
| Training | 2x H100 (80GB) | Full training with LoRA, reduced batch |
| Paper config | 16x H100 (80GB) | Full replication, 3-5 days |
| Inference only | Any GPU 8GB+ | Run trained curator in 4-bit quantization |

## Stack

- **[TRL](https://github.com/huggingface/trl)** — GRPOTrainer with `environment_factory` for multi-turn RL
- **[vLLM](https://github.com/vllm-project/vllm)** — Fast inference during rollout generation
- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** — Base model (Apache 2.0)
- **[ALFWorld](https://github.com/alfworld/alfworld)** — Household task environment
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — Skill retrieval

All components are permissively licensed (Apache 2.0 / MIT). The trained model is fully commercial-use.

## Key Insight

The paper's biggest finding: **a trained 8B curator beats Gemini-2.5-Pro at zero-shot curation.** Targeted RL training on curation-specific signals matters more than raw model capability. This means you can run better skill management locally on consumer hardware than you can get from the largest API models.

## References

- [SkillOS paper](https://arxiv.org/abs/2605.06614) — Ouyang et al., 2026
- [GRPO](https://arxiv.org/abs/2402.03300) — DeepSeek-Math, Shao et al., 2024
- [Anthropic SKILL.md format](https://docs.anthropic.com/en/docs/agents/skills)

## License

Apache 2.0
