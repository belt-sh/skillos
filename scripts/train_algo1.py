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
import random
import sys
import time

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
from skillos.algo1 import env as algo1_env  # for _num_generations at reward time
from skillos.envs.curator_env import configure as configure_classic_env


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


ALFWORLD_TASK_TYPES = ["pick", "clean", "heat", "cool", "look", "pick2"]


def _assign_group_types(num_groups: int, distribution: str, seed: int) -> list[str]:
    """Pick a task type for each of the `num_groups` training groups.

    - "uniform" (default): round-robin over the 6 types — equal counts. The
      original interpretive choice; the paper doesn't print frequencies.
    - "natural": counts proportional to ALFWorld's real type frequencies
      (DIVERGENCES #0). Frequencies come from the same one-time seed-bucket scan
      the env uses (`_type_seeds` bucket sizes), so train matches the natural
      distribution the held-out eval is drawn from. Deterministic per `seed`.
    """
    if distribution != "natural":
        return [ALFWORLD_TASK_TYPES[i % len(ALFWORLD_TASK_TYPES)]
                for i in range(num_groups)]

    from skillos.envs import curator_env as _ce
    if not _ce._type_seeds:
        _ce._build_type_seed_index()
    weights = {t: len(_ce._type_seeds.get(t, [])) for t in ALFWORLD_TASK_TYPES}
    total = sum(weights.values()) or 1
    counts = {t: round(num_groups * w / total) for t, w in weights.items()}
    # Fix rounding drift against num_groups by nudging the largest buckets.
    order = sorted(ALFWORLD_TASK_TYPES, key=lambda t: weights[t], reverse=True)
    i = 0
    while sum(counts.values()) != num_groups:
        t = order[i % len(order)]
        if sum(counts.values()) < num_groups:
            counts[t] += 1
        elif counts[t] > 0:
            counts[t] -= 1
        i += 1
    types: list[str] = []
    for t in ALFWORLD_TASK_TYPES:
        types += [t] * counts[t]
    random.Random(seed).shuffle(types)  # interleave so the data sampler doesn't see type-blocks
    return types


def build_dataset(num_episodes: int, group_size: int,
                  type_distribution: str = "uniform", seed: int = 42) -> Dataset:
    """One row per GRPO *group* (paper: 3553 episodes / |G|=10 ≈ 355 groups),
    carrying explicit `group_id`/`task_type` columns. TRL repeats each row
    num_generations times and passes the row to `env.reset(**row)`, so all N
    generations of a group share the same task sequence regardless of how
    completions are sharded across ranks — group identity comes from the
    data, never from env-slot arithmetic (which collapses because TRL reuses
    env instances; see docs/postmortem-2026-06-10-algo1-group-collapse.md).

    `type_distribution` selects how the 6 ALFWorld types are spread over groups
    (see _assign_group_types / DIVERGENCES #0).
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
        "task_type": _assign_group_types(num_groups, type_distribution, seed),
    })


def _complete_the_protocol(environments: list[Algo1CuratorEnv]) -> None:
    """Run every position the curator left unplayed, across all envs at once.

    Envs are completed concurrently with each other, and each env completes its
    own missing positions concurrently, because a frozen repo makes positions
    independent. Real parallelism is bounded by the alfworld env pool, which
    blocks rather than over-subscribing.
    """
    import concurrent.futures as _cf

    todo = [e for e in environments if e.missing_positions()]
    if not todo:
        return
    budget = float(os.environ.get("SKILLOS_COMPLETION_BUDGET_S", "3600"))
    deadline = time.time() + budget
    n_missing = sum(len(e.missing_positions()) for e in todo)
    print(f"[algo1] completing the protocol: {len(todo)}/{len(environments)} "
          f"rollouts ended early, {n_missing} positions to run, "
          f"budget {budget:.0f}s", file=sys.stderr, flush=True)

    with _cf.ThreadPoolExecutor(max_workers=len(todo)) as pool:
        futs = [pool.submit(e.complete_unplayed_positions, deadline) for e in todo]
        for f in futs:
            try:
                f.result(timeout=max(1.0, deadline - time.time()) + 60)
            except Exception as exc:  # never let this kill the step
                print(f"[algo1] protocol completion failed for one rollout: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def reward_func(environments: list[Algo1CuratorEnv], **kwargs) -> list[float]:
    """Algorithm 1 reward: each env recorded |G| executor results during its
    own tool-loop. We finalize per-env, then neutralise rollouts that produced
    no measurement at all.

    WHY THE SECOND STEP EXISTS. A rollout whose every informed position was
    deadline-cut has no evidence about its curator. It used to receive
    r_task = 0.0, i.e. an infrastructure failure scored as bad curation. That is
    the same error class as the retracted eval bug, and it is worse inside GRPO:
    advantages are centred within the group, so a spuriously-zeroed rollout
    pushes the gradient away from whatever that curator wrote, at random, in the
    exact term that reaches the parameters. It affected 10 to 41% of rollouts.

    The fix: give such rollouts the mean reward of the MEASURED rollouts in their
    own group. Their advantage becomes ~0, so they neither help nor hurt. This is
    the standard treatment for a missing observation in a group-relative
    estimator, and it degrades gracefully: if every rollout in a group is
    unmeasured, the whole group is flat and contributes nothing, which is the
    correct outcome for a group we failed to measure.

    Grouping: TRL repeats each prompt `num_generations` times consecutively, so
    environments arrive as contiguous blocks of that size. We read the size off
    the env module rather than assuming, and fall back to treating the batch as
    one group if it does not divide evenly.
    """
    # STEP 0: finish the protocol before scoring it (DIVERGENCES #18).
    #
    # The paper's Algorithm 1 runs the executor at all |G| positions; the curator
    # does not choose whether position i+1 happens. In this TRL port it advances
    # the loop with a tool call, so it can stop early, and 19-24% of rollouts did.
    # Scoring only what it chose to play made quitting after a success the
    # highest-reward action available; scoring the rest as failures is harsher
    # than the paper. So we run them, with S frozen at wherever the curator left
    # it, which is exactly "the curator wrote nothing further".
    #
    # Bounded by SKILLOS_COMPLETION_BUDGET_S so a rank with many early exits
    # cannot skew the next collective. Positions we abandon to that budget are
    # marked as infrastructure losses and leave the denominator.
    _complete_the_protocol(environments)

    rewards = [env._finalize_reward() for env in environments]
    unmeasured = [bool(getattr(env, "r_task_unmeasured", False))
                  for env in environments]

    if not any(unmeasured):
        _log_measurement_health(environments, rewards, n_neutralised=0)
        return rewards

    # Group by the env's OWN group id, not by block arithmetic over
    # num_generations. Measured on the 2026-08-16 smoke: reward_func receives 4
    # envs per call (32 completions sharded over 8 ranks) while num_generations
    # is 8, so block arithmetic would have fallen back to "treat the batch as one
    # group". That fallback happened to be right — all 4 envs in a call shared
    # gid=229 — but relying on it is correct by accident. Group identity already
    # lives on the env, put there by the dataset row precisely because slot
    # arithmetic is untrustworthy here (see the 2026-06-10 group-collapse
    # postmortem). Use it.
    groups: dict[object, list[int]] = {}
    for i, env in enumerate(environments):
        groups.setdefault(getattr(env, "_group_id", None), []).append(i)

    n_neutralised = 0
    for gid, idxs in groups.items():
        measured = [rewards[i] for i in idxs if not unmeasured[i]]
        if not measured:
            # Whole group unmeasured: flatten it so it contributes no gradient.
            for i in idxs:
                rewards[i] = 0.0
            n_neutralised += len(idxs)
            print(f"[algo1] group {gid}: ALL {len(idxs)} rollouts unmeasured — "
                  f"flattened, contributes no gradient",
                  file=sys.stderr, flush=True)
            continue
        group_mean = sum(measured) / len(measured)
        for i in idxs:
            if unmeasured[i]:
                rewards[i] = group_mean
                n_neutralised += 1

    _log_measurement_health(environments, rewards, n_neutralised)
    return rewards


def _log_measurement_health(environments, rewards, n_neutralised: int) -> None:
    """Per-step visibility into how much of the reward was actually measured.

    DIVERGENCES #16 was discovered months late because nothing printed this. The
    numbers to watch: `measured` well below |G|-1 means the deadline budget is
    too tight and r_task is being estimated from one or two episodes, which is
    the real reason held-out lift came out small.
    """
    n = len(environments)
    counts = [int(getattr(env, "n_task_measured", 0)) for env in environments]
    counts_sorted = sorted(counts)
    median = counts_sorted[n // 2] if n else 0
    try:
        from skillos.executor.executor import get_parse_stats, reset_parse_stats
        calls, coerced = get_parse_stats()
        reset_parse_stats()
        coercion = f"{coerced}/{calls} actions coerced ({coerced / max(calls, 1):.1%})"
    except Exception:
        coercion = "coercion telemetry unavailable"
    # EARLY-ENDED share is the number that would have caught the r_task
    # denominator bug on step 2 instead of step 11. A rollout that ends after the
    # seed position played no informed position at all; before 2026-08-18 that
    # scored r_task over a denominator of whatever it *did* play, so stopping
    # early after one success paid better than playing the protocol out. The
    # share rose 12.8% -> 23.8% across 11 steps while mean reward rose and mean
    # completion length fell, and nothing printed it.
    early = sum(1 for env in environments
                if int(getattr(env, "n_task_measured", 0)) == 0
                and int(getattr(env, "n_task_denominator", 0)) > 0)
    denoms = [int(getattr(env, "n_task_denominator", 0)) for env in environments]
    median_denom = sorted(denoms)[n // 2] if n else 0
    print(f"[algo1] reward health: {n} rollouts, "
          f"r_task measured from median {median} positions "
          f"(min {counts_sorted[0] if n else 0}, max {counts_sorted[-1] if n else 0}) "
          f"over median denominator {median_denom}, "
          f"{early} ended early with 0 played, "
          f"{n_neutralised} neutralised as unmeasured; {coercion}",
          file=sys.stderr, flush=True)


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
        judge_submit=classic_env._submit_judge,
        num_generations=num_generations,
        group_size=group_size,
        curriculum=config.get("group_curriculum", False),
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

    grpo_kwargs["seed"] = config.get("seed", 42)

    dataset = build_dataset(
        num_episodes, group_size,
        type_distribution=config.get("group_type_distribution", "uniform"),
        seed=config.get("seed", 42),
    )
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
