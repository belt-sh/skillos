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

from skillos.envs.curator_env import CuratorEnv, configure as configure_env


def _has_vllm() -> bool:
    try:
        import vllm
        return True
    except ImportError:
        return False


def build_dataset(num_episodes: int = 1000) -> Dataset:
    """Build a dataset of curator prompts (paper Appendix A.1, 1:1).

    Each row triggers one curator episode:
    1. CuratorEnv.reset() runs the frozen executor on an ALFWorld task
    2. Returns the CURATOR_INPUT_TEMPLATE-formatted trajectory text, which
       TRL appends to the user message
    3. Curator decides insert/update/delete via tool calls

    Paper structure:
      system: CURATOR_SYSTEM (Role + Input Data + Critical Constraints +
              Skill Markdown Format + Action Guidelines, verbatim)
      user:   CURATOR_INPUT_TEMPLATE (Task / Past Skills / Trajectory / Result,
              filled in by env.reset() — TRL appends env observation to user)
    """
    from skillos.curator.prompts import CURATOR_SYSTEM
    return Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": CURATOR_SYSTEM},
                {"role": "user", "content": ""},
            ]
        ] * num_episodes,
    })


def reward_func(environments: list[CuratorEnv], **kwargs) -> list[float]:
    """Compute composite reward from each curator environment."""
    rewards = []
    for env in environments:
        env._compute_reward()
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

    # Configure executor, judge, and GRPO group size (so the env can share
    # one frozen-executor trajectory across the N curator samples per prompt).
    configure_env(
        executor_config=config.get("executor", {"type": "heuristic"}),
        judge_config=config.get("judge", {"type": "heuristic"}),
        num_generations=config.get("num_generations", 4),
    )

    output_dir = config.get("output_dir", "./output/curator")
    dataset = build_dataset(num_episodes)

    # Wandb defaults (override via env vars or `wandb_*` config keys)
    import os
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
        num_generations=config.get("num_generations", 4),
        max_completion_length=config.get("max_completion_length", 4096),
        temperature=config.get("temperature", 1.0),
        beta=config.get("beta", 0.0),  # GRPO KL coefficient (paper: 0.001)
        # Logging
        logging_steps=config.get("logging_steps", 1),
        log_completions=True,
        report_to=config.get("report_to", "none"),
        # Checkpointing — write intermediate adapters so a crashed run doesn't lose everything
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 25),
        save_total_limit=config.get("save_total_limit", 3),
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

    # TRL requires generation_batch_size to be a multiple of the global batch
    # (per_device × num_processes). Only forward an explicit value; otherwise
    # let TRL pick its own default to avoid hardcoded mismatches.
    if "generation_batch_size" in config:
        grpo_kwargs["generation_batch_size"] = config["generation_batch_size"]

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

    # Persist the skill repo + rollouts + judge cache alongside every
    # checkpoint, and load them back when resuming. The adapter alone is
    # useless at eval time — the curator's learned behavior is *what
    # curates*, but evaluation (and the next opt-step of a resumed run)
    # consumes the repo's skills. Without the load-on-resume hook a
    # resumed run starts with an empty repo and trains a different problem
    # than the pre-crash run.
    from transformers import TrainerCallback
    from skillos.envs import curator_env as _ce
    import json as _json
    import shutil as _shutil

    class SkillRepoSaver(TrainerCallback):
        """Save skill repo + rollouts.jsonl + judge cache on each checkpoint.

        Cost: <100 KB skills + jsonl copy + JSON dump of judge cache. Cheap
        enough to do every step (drop save_steps to 1 in the config).
        """

        def on_save(self, args, state, control, **kwargs):
            ckpt_root = f"{args.output_dir}/checkpoint-{state.global_step}"
            _ce._shared_skill_repo.save(f"{ckpt_root}/skills")
            # Snapshot rollouts.jsonl so post-crash analytics / reward
            # histories aren't lost.
            try:
                if os.path.exists(_ce._rollouts_jsonl_path):
                    _shutil.copyfile(
                        _ce._rollouts_jsonl_path,
                        f"{ckpt_root}/rollouts.jsonl",
                    )
            except Exception as e:
                print(f"[ckpt] rollouts snapshot failed: {e}")
            # Persist the judge cache so a resumed run doesn't re-pay for
            # judge calls on content it has already scored.
            try:
                with _ce._judge_cache_lock:
                    cache_snap = dict(_ce._judge_cache)
                with open(f"{ckpt_root}/judge_cache.json", "w") as f:
                    _json.dump(cache_snap, f)
            except Exception as e:
                print(f"[ckpt] judge cache snapshot failed: {e}")

    class SkillRepoLoader(TrainerCallback):
        """When resuming from a checkpoint, restore the skill repo + judge
        cache before training resumes. Without this the resumed run trains
        a different problem than the pre-crash run.

        HF TrainerCallback isn't told the resume path directly — the only
        signal in on_train_begin is `state.global_step > 0`. When that's
        the case, the corresponding checkpoint is at
        `{output_dir}/checkpoint-{global_step}`.

        Rollouts.jsonl is intentionally *not* copied back to the live path —
        it's snapshotted into the checkpoint for analytics; the live file
        keeps growing append-only across runs.
        """

        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step <= 0:
                return  # fresh run, nothing to restore
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.isdir(ckpt):
                print(f"[resume] no checkpoint at {ckpt}; skipping skill/cache restore")
                return
            from skillos.skills.repo import SkillRepo
            skills_dir = os.path.join(ckpt, "skills")
            if os.path.isdir(skills_dir):
                loaded = SkillRepo.load(skills_dir)
                _ce._shared_skill_repo.skills = loaded.skills
                _ce._shared_skill_repo._bm25_dirty = True
                print(f"[resume] loaded {len(loaded.skills)} skills from {skills_dir}")
            cache_path = os.path.join(ckpt, "judge_cache.json")
            if os.path.isfile(cache_path):
                try:
                    with open(cache_path) as f:
                        cache = _json.load(f)
                    with _ce._judge_cache_lock:
                        _ce._judge_cache.update(cache)
                    print(f"[resume] restored judge cache: {len(cache)} entries")
                except Exception as e:
                    print(f"[resume] judge cache restore failed: {e}")

    trainer.add_callback(SkillRepoSaver())
    trainer.add_callback(SkillRepoLoader())

    # Per-opt-step observability: reset rollout counters + (re)start heartbeat.
    rollouts_per_step = (
        config.get("batch_size", 1)
        * config.get("gradient_accumulation_steps", 8)
        * config.get("num_generations", 4)
    )
    class StepBoundaryObserver(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            _ce.set_step_expected_rollouts(rollouts_per_step)
        def on_step_begin(self, args, state, control, **kwargs):
            _ce.set_step_expected_rollouts(rollouts_per_step)
    trainer.add_callback(StepBoundaryObserver())

    try:
        trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    except KeyboardInterrupt:
        print("Interrupted — attempting to save current state…")

    # Save no matter how training exited. Best-effort: even if model-card
    # generation throws (e.g. wandb/comet stub bugs), the adapter still lands.
    final_dir = output_dir
    try:
        trainer.save_model(final_dir)
        _ce._shared_skill_repo.save(f"{final_dir}/skills")
        print(f"Curator model + {len(_ce._shared_skill_repo)} skills saved to {final_dir}")
    except Exception as e:
        emergency_dir = f"{final_dir}-emergency"
        print(f"save_model() raised {type(e).__name__}: {e}")
        print(f"Falling back to direct state dump at {emergency_dir}")
        try:
            trainer.model.save_pretrained(emergency_dir)
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(emergency_dir)
            _ce._shared_skill_repo.save(f"{emergency_dir}/skills")
            print(f"Adapter + skills rescued to {emergency_dir}")
        except Exception as e2:
            print(f"Emergency save also failed: {type(e2).__name__}: {e2}")
            raise


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
