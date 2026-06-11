"""Algorithm 1 GRPO training entrypoint — paper §3.1-3.2, single mega-tool.

Mirrors skillos.train.py but swaps in `skillos.algo1.Algo1CuratorEnv` as the
environment_factory and pins `max_tool_calling_iterations = group_size` so
TRL's tool-call loop walks exactly |G| positions per rollout. Reward is
computed by each env at end-of-rollout (no separate batched probe phase).

Usage:
    accelerate launch -m scripts.train_algo1 --config configs/alfworld_8xh100_algo1.yaml
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist

# Initialize the default NCCL process group with a 4-hour collective
# timeout BEFORE accelerate's own init runs. accelerate's state.py checks
# `dist.is_initialized()` and skips init if we got there first, so our
# timeout sticks. The default 30-min timeout trips during Algorithm 1's
# G=10 tool-loop iterations because each iteration's tool execution can
# take 10-15 min, and per-iteration rank skew accumulates across the
# G+1=11 iterations into >30-min waits at the post-_generate gather.
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

from skillos.algo1 import Algo1CuratorEnv, configure as configure_algo1
from skillos.envs.curator_env import configure as configure_classic_env


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


ALFWORLD_TASK_TYPES = ["pick", "clean", "heat", "cool", "look", "pick2"]


def build_dataset(num_episodes: int, group_size: int) -> Dataset:
    """One row per GRPO *group* (paper: 3553 episodes / |G|=10 ≈ 355 groups),
    carrying explicit `group_id`/`task_type` columns. TRL repeats each row
    num_generations times and passes the row to `env.reset(**row)`, so all N
    generations of a group share the same task sequence regardless of how
    completions are sharded across ranks — group identity comes from the
    data, never from env-slot arithmetic (which collapses because TRL reuses
    env instances; see docs/postmortem-2026-06-10-algo1-group-collapse.md).

    Task types round-robin over the six ALFWorld types so training covers
    all of them (uniform by group, not split-proportional — an interpretive
    choice; the paper groups "related" tasks without specifying frequencies).
    """
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
        "task_type": [ALFWORLD_TASK_TYPES[i % len(ALFWORLD_TASK_TYPES)]
                      for i in range(num_groups)],
    })


def reward_func(environments: list[Algo1CuratorEnv], **kwargs) -> list[float]:
    """Algorithm 1 reward: each env recorded |G| executor results during its
    own tool-loop. We just finalize per-env — no shared probe phase."""
    return [env._finalize_reward() for env in environments]


def train(config: dict) -> None:
    model_name = config.get("model", "Qwen/Qwen3-8B")
    num_episodes = config.get("num_episodes", 1000)
    group_size = config.get("group_size", 10)
    num_generations = config.get("num_generations", 8)
    has_cuda = torch.cuda.is_available()
    has_vllm = _has_vllm()
    use_vllm = config.get("use_vllm", True) and has_cuda and has_vllm

    # Configure classic env primitives (executor pool, ALFWorld env factory,
    # judge). algo1's env reuses _run_probe and the seed index from this
    # module.
    configure_classic_env(
        executor_config=config.get("executor", {"type": "heuristic"}),
        judge_config=config.get("judge", {"type": "heuristic"}),
        num_generations=num_generations,
        num_probe_tasks=0,   # Algorithm 1 doesn't use Path B probes
    )

    # Algorithm 1 hyperparams. judge_submit must be wired explicitly —
    # passing None silently zeroes the paper's λ_u·r_cnt reward term
    # (postmortem 2026-06-10, bug 3).
    from skillos.envs import curator_env as classic_env
    configure_algo1(
        executor=None,
        judge_submit=classic_env._submit_judge,
        num_generations=num_generations,
        group_size=group_size,
    )

    output_dir = config.get("output_dir", "./output/curator-algo1")

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
        # TRL 1.4.0 defaults loss_type="dapo" and HF defaults a linear-decay
        # LR schedule — both undeclared deviations from the paper's GRPO setup
        # (postmortem 2026-06-10, bug 4). Pin paper-faithful defaults.
        loss_type=config.get("loss_type", "grpo"),
        lr_scheduler_type=config.get("lr_scheduler_type", "constant"),
        # Tool loop iterations: G+1 because the first generation is a
        # priming "empty ops" call (curator hasn't seen any trajectory yet
        # — reset returns only the session-start instructional prompt),
        # then G informed generations each emit ops based on the previous
        # position's trajectory. All G informed generations need to land,
        # which requires G+1 tool calls. The (G+1)th call runs no executor
        # — it just applies the final position's ops and terminates.
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

    dataset = build_dataset(num_episodes, group_size)
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
        environment_factory=Algo1CuratorEnv,
    )

    resume_ckpt = (
        config.get("resume_from_checkpoint")
        or os.environ.get("SKILLOS_RESUME_FROM_CHECKPOINT")
        or None
    )
    if resume_ckpt:
        print(f"[algo1.train] resuming from checkpoint: {resume_ckpt}")
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        print("Interrupted — best-effort save…")
    try:
        trainer.save_model(output_dir)
    except Exception as e:
        print(f"[algo1.train] save_model failed: {type(e).__name__}: {e}",
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
