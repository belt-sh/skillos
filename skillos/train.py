"""Main training script — train a skill curator with GRPO on ALFWorld.

Usage:
    # Smoke test (tiny model, no real training)
    python -m skillos.train --smoke

    # Single GPU training with LoRA
    python -m skillos.train --config configs/alfworld_single_gpu.yaml

    # Multi-GPU training
    accelerate launch -m skillos.train --config configs/alfworld_multi_gpu.yaml
"""

from __future__ import annotations

import argparse
import sys

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from skillos.envs.alfworld import ALFWorldEnv
from skillos.skills.repo import SkillRepo


def build_dataset(num_episodes: int = 1000) -> Dataset:
    """Build a dataset of ALFWorld task prompts.

    Each row is one episode. The environment handles task selection in reset().
    """
    return Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": "Complete the household task in the environment. Use the act tool to take actions."}]
        ] * num_episodes,
    })


def reward_func(environments: list[ALFWorldEnv], **kwargs) -> list[float]:
    """Reward function — reads binary success from each environment."""
    return [env.reward for env in environments]


def train(config: dict):
    model_name = config.get("model", "Qwen/Qwen3-8B")
    num_episodes = config.get("num_episodes", 1000)

    dataset = build_dataset(num_episodes)

    grpo_args = GRPOConfig(
        output_dir=config.get("output_dir", "./output/alfworld"),
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 1e-6),
        num_generations=config.get("num_generations", 4),
        max_completion_length=config.get("max_completion_length", 4096),
        max_prompt_length=config.get("max_prompt_length", 16384),
        temperature=config.get("temperature", 1.0),
        # vLLM
        use_vllm=config.get("use_vllm", True),
        vllm_mode=config.get("vllm_mode", "colocate"),
        # LoRA
        use_lora=config.get("use_lora", True),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        # Logging
        logging_steps=config.get("logging_steps", 1),
        log_completions=True,
        report_to=config.get("report_to", "none"),
        # Chat
        chat_template_kwargs={"enable_thinking": config.get("enable_thinking", False)},
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_args,
        environment_factory=ALFWorldEnv,
    )

    trainer.train()
    trainer.save_model(config.get("output_dir", "./output/alfworld"))
    print(f"Model saved to {config.get('output_dir', './output/alfworld')}")


def main():
    parser = argparse.ArgumentParser(description="Train SkillOS curator")
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
            "max_completion_length": 512,
            "use_vllm": True,
            "vllm_mode": "colocate",
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
