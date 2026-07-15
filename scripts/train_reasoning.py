"""Algorithm 1 GRPO training entrypoint for the reasoning benchmark.

Mirrors scripts/train_algo1.py; swaps in ReasoningCuratorEnv (DeepMath-103K
problems, single-turn CoT executor via inference.sh) as the environment_factory.

Usage:
    accelerate launch -m scripts.train_reasoning --config configs/reasoning_8xh100_algo1_fft.yaml
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys

import torch
import torch.distributed as dist

# Same 4-hour NCCL default as train_algo1.
if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(
            seconds=int(os.environ.get("SKILLOS_NCCL_TIMEOUT_S", "14400"))
        ),
    )

import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from skillos.algo1 import configure as configure_algo1
from skillos.envs.curator_env import configure as configure_classic_env
from skillos.reasoning.env import ReasoningCuratorEnv, configure as configure_reasoning
from skillos.reasoning.train_data import DEEPMATH_TOPICS, build_topic_index


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _assign_group_topics(num_groups: int, seed: int = 42) -> list[str]:
    """Round-robin across the 9 DeepMath topic buckets — same "uniform is the
    tested winner" rationale as ALFWorld (natural-distribution was falsified,
    see `natural-distribution-uniform-wins`). Shuffled so the data sampler
    doesn't see topic-blocks."""
    topics = [DEEPMATH_TOPICS[i % len(DEEPMATH_TOPICS)] for i in range(num_groups)]
    random.Random(seed).shuffle(topics)
    return topics


def build_dataset(num_episodes: int, group_size: int, seed: int = 42) -> Dataset:
    from skillos.curator.prompts import CURATOR_SYSTEM
    num_groups = max(1, num_episodes // group_size)
    return Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": CURATOR_SYSTEM},
                {"role": "user", "content": ""},
            ]
        ] * num_groups,
        "group_id": list(range(num_groups)),
        "task_type": _assign_group_topics(num_groups, seed),
    })


def reward_func(environments: list[ReasoningCuratorEnv], **kwargs) -> list[float]:
    return [env._finalize_reward() for env in environments]


def train(config: dict) -> None:
    model_name = config.get("model", "Qwen/Qwen3-8B")
    num_episodes = config.get("num_episodes", 1000)
    group_size = config.get("group_size", 10)
    num_generations = config.get("num_generations", 8)
    has_cuda = torch.cuda.is_available()
    has_vllm = _has_vllm()
    use_vllm = config.get("use_vllm", True) and has_cuda and has_vllm

    # Warm the DeepMath topic index once on rank-0 before workers spawn.
    build_topic_index()

    # We need the classic env's judge submit + rollout pool infra (reasoning env
    # inherits from Algo1CuratorEnv which uses it), so wire the same primitives.
    configure_classic_env(
        executor_config=config.get("executor_placeholder", {"type": "heuristic"}),
        judge_config=config.get("judge", {"type": "heuristic"}),
        num_generations=num_generations,
        num_probe_tasks=0,
    )
    from skillos.envs import curator_env as classic_env
    configure_algo1(
        judge_submit=classic_env._submit_judge,
        num_generations=num_generations,
        group_size=group_size,
        curriculum=False,
    )
    configure_reasoning(
        executor_app=config.get("executor_app", "openrouter/qwen3-8b"),
    )

    output_dir = config.get("output_dir", "./output/reasoning-algo1")

    if config.get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config.get("wandb_project", "skillos"))
        os.environ.setdefault("WANDB_ENTITY", config.get("wandb_entity", "okaris"))
        run_name = config.get("wandb_run_name") or output_dir.rsplit("/", 1)[-1]
        os.environ.setdefault("WANDB_NAME", run_name)

    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 1e-6),
        num_generations=num_generations,
        max_completion_length=config.get("max_completion_length", 4096),
        temperature=config.get("temperature", 1.0),
        beta=config.get("beta", 0.0),
        loss_type=config.get("loss_type", "grpo"),
        lr_scheduler_type=config.get("lr_scheduler_type", "constant"),
        max_tool_calling_iterations=group_size + 1,
        logging_steps=config.get("logging_steps", 1),
        log_completions=True,
        report_to=config.get("report_to", "none"),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 10),
        save_total_limit=config.get("save_total_limit", 6),
        chat_template_kwargs={"enable_thinking": config.get("enable_thinking", False)},
        use_cpu=not has_cuda,
        bf16=has_cuda,
        gradient_checkpointing=has_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = config.get("vllm_mode", "colocate")
        for k in ("vllm_gpu_memory_utilization", "vllm_max_model_length",
                  "vllm_tensor_parallel_size", "vllm_enable_sleep_mode"):
            if k in config:
                grpo_kwargs[k] = config[k]

    if "generation_batch_size" in config:
        grpo_kwargs["generation_batch_size"] = config["generation_batch_size"]
    if config.get("max_steps"):
        grpo_kwargs["max_steps"] = config["max_steps"]
    grpo_kwargs["seed"] = config.get("seed", 42)

    dataset = build_dataset(num_episodes, group_size, seed=config.get("seed", 42))
    args = GRPOConfig(**grpo_kwargs)

    peft_config = None
    if config.get("use_lora", False):
        peft_config = LoraConfig(
            r=config.get("lora_r", 32),
            lora_alpha=config.get("lora_alpha", 64),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=args,
        peft_config=peft_config,
        environment_factory=ReasoningCuratorEnv,
    )

    resume_ckpt = (
        config.get("resume_from_checkpoint")
        or os.environ.get("SKILLOS_RESUME_FROM_CHECKPOINT")
        or None
    )
    if resume_ckpt:
        print(f"[reasoning.train] resuming from checkpoint: {resume_ckpt}")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        print("Interrupted — best-effort save…")
    try:
        trainer.save_model(output_dir)
    except Exception as e:
        print(f"[reasoning.train] save_model failed: {type(e).__name__}: {e}",
              file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
