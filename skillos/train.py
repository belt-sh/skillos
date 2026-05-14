"""Main training script — train a skill curator with GRPO.

The SkillOS training loop:
1. Frozen executor solves ALFWorld tasks (inference only, not trained)
2. Curator (THIS model, being trained) sees trajectories and curates the skill repo
3. GRPO optimizes curator based on composite reward

Usage:
    # Smoke test on CPU (tiny model, verifies full loop)
    python -m skillos.train --smoke

    # Single GPU training with LoRA
    python -m skillos.train --config configs/alfworld_single_gpu.yaml

    # Multi-GPU training
    accelerate launch -m skillos.train --config configs/alfworld_multi_gpu.yaml
"""

from __future__ import annotations

import argparse
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from skillos.envs.curator_env import CuratorEnv


def _has_vllm() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False


def build_dataset(num_episodes: int = 1000) -> Dataset:
    """Build a dataset of curator prompts.

    Each row triggers one curator episode:
    1. CuratorEnv.reset() runs the frozen executor on an ALFWorld task
    2. Returns the trajectory as observation
    3. Curator decides insert/update/delete via tool calls
    """
    return Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": (
                "You are a skill curator. Analyze the agent trajectory below and decide "
                "whether to insert a new skill, update an existing skill, or delete a skill "
                "from the skill repository. Use the available tools to manage skills."
            )}]
        ] * num_episodes,
    })


def reward_func(environments: list[CuratorEnv], **kwargs) -> list[float]:
    """Compute composite reward from each curator environment."""
    rewards = []
    for env in environments:
        env.compute_reward()
        rewards.append(env.reward)
    return rewards


def train(config: dict):
    model_name = config.get("model", "Qwen/Qwen3-8B")
    num_episodes = config.get("num_episodes", 1000)
    has_cuda = torch.cuda.is_available()
    has_vllm = _has_vllm()

    # Auto-detect: disable vLLM on CPU or when not installed
    use_vllm = config.get("use_vllm", True) and has_cuda and has_vllm
    if config.get("use_vllm", True) and not use_vllm:
        reason = "no CUDA" if not has_cuda else "vllm not installed"
        print(f"Note: vLLM disabled ({reason}), using native generation")

    dataset = build_dataset(num_episodes)

    grpo_kwargs = dict(
        output_dir=config.get("output_dir", "./output/curator"),
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 1e-6),
        num_generations=config.get("num_generations", 4),
        generation_batch_size=config.get("generation_batch_size", config.get("num_generations", 4)),
        max_completion_length=config.get("max_completion_length", 4096),
        temperature=config.get("temperature", 1.0),
        # Logging
        logging_steps=config.get("logging_steps", 1),
        log_completions=True,
        report_to=config.get("report_to", "none"),
        # Chat
        chat_template_kwargs={"enable_thinking": config.get("enable_thinking", False)},
        # CPU fallback
        use_cpu=not has_cuda,
        bf16=has_cuda,
        gradient_checkpointing=has_cuda,
    )

    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.get("vllm_mode", "colocate")

    grpo_args = GRPOConfig(**grpo_kwargs)

    # LoRA config via peft
    peft_config = None
    if config.get("use_lora", True):
        peft_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_args,
        peft_config=peft_config,
        environment_factory=CuratorEnv,
    )

    trainer.train()
    trainer.save_model(config.get("output_dir", "./output/curator"))
    print(f"Curator model saved to {config.get('output_dir', './output/curator')}")


def main():
    parser = argparse.ArgumentParser(description="Train SkillOS skill curator")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true", help="Run a minimal smoke test")
    args = parser.parse_args()

    if args.smoke:
        config = {
            "model": "Qwen/Qwen3-0.6B",
            "num_episodes": 4,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 2,
            "max_completion_length": 128,
            "use_vllm": False,
            "use_lora": True,
            "lora_r": 8,
            "output_dir": "./output/smoke",
            "report_to": "none",
        }
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        parser.print_help()
        sys.exit(1)

    train(config)


if __name__ == "__main__":
    main()
