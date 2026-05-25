"""Parallel ALFWorld eval — paper-style batched throughput.

Same evaluation protocol as scripts/eval_alfworld.py (frozen executor +
retrieved skills, success rate per task type), but runs `batch_size` games
concurrently: at each step-round, every active game's executor action is
requested concurrently (thread pool over the remote inference.sh calls),
then all games are advanced in a single batched env.step(). inference.sh
autoscales, so wall-clock collapses from hours (serial) to ~minutes.

The ALFWorld env is stepped single-threaded (textworld/tatsu hold
module-global parser state and aren't thread-safe); only the network-bound
executor calls run concurrently — mirroring the training rollout pattern.

Usage:
    python -m scripts.eval_alfworld_parallel \\
        --checkpoint output/alfworld-8xh100/checkpoint-111 \\
        --split valid_seen --num-games 50 --batch-size 25 --max-steps 30
    # no-memory baseline: point --checkpoint at a dir with an empty skills/
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from collections import deque
from pathlib import Path

from scripts.eval_alfworld import (
    classify_task, extract_task_description, report_eval_results,
    resolve_skills_dir,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="valid_seen",
                   choices=["valid_seen", "valid_unseen", "train"])
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=25,
                   help="games run concurrently per batch (executor calls fired in parallel)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--executor", default="infsh",
                   choices=["heuristic", "infsh", "vllm", "api"])
    p.add_argument("--executor-app", default="openrouter/qwen3-8b")
    p.add_argument("--base-url", default="http://localhost:8001/v1",
                   help="OpenAI-compatible endpoint for --executor vllm/api (local vLLM)")
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="served model name for --executor vllm/api")
    # Executor decode overrides (default None = use InfshExecutor defaults).
    # Used to audit the no-memory baseline against the paper's GiGPO config.
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None,
                   help="vLLM Qwen3 thinking mode ON (default: model default)")
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                   help="vLLM Qwen3 thinking mode OFF (paper-analog non-reasoning executor)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--game-offset", type=int, default=0,
                   help="skip this many games before evaluating (shard across processes)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    skills_dir = resolve_skills_dir(ckpt)

    from skillos.skills.repo import SkillRepo
    repo = SkillRepo.load(str(skills_dir))
    print(f"Loaded {len(repo)} skills from {skills_dir}")
    if len(repo) == 0:
        print("WARNING: empty repo — No Memory baseline.")

    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    bs = min(args.batch_size, args.num_games)
    env = make_alfworld_env(SPLIT_MAP[args.split], batch_size=bs)
    # Shard: skip this process's offset of games (resets advance through the
    # split sequentially). Lets 8 processes each play a distinct slice in
    # parallel on their own replica/GPU — no shared lockstep barrier.
    skipped = 0
    while skipped < args.game_offset:
        env.reset()
        skipped += bs
    if args.game_offset:
        print(f"[shard] skipped {skipped} games (offset {args.game_offset})", flush=True)

    from skillos.executor.executor import create_executor
    # --base-url may be a comma-separated list of replicas (one vLLM server per
    # GPU); we build one executor per URL and round-robin requests across them.
    base_urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    executors = []
    for url in (base_urls or [None]):
        exec_cfg = {"type": args.executor}
        if args.executor == "infsh":
            exec_cfg["app"] = args.executor_app
            for k, v in (("temperature", args.temperature), ("top_p", args.top_p),
                         ("top_k", args.top_k), ("min_p", args.min_p),
                         ("max_tokens", args.max_tokens),
                         ("reasoning_effort", args.reasoning_effort)):
                if v is not None:
                    exec_cfg[k] = v
        elif args.executor in ("vllm", "api"):
            exec_cfg["base_url"] = url
            exec_cfg["model"] = args.model
            for k, v in (("temperature", args.temperature), ("max_tokens", args.max_tokens),
                         ("top_p", args.top_p), ("top_k", args.top_k), ("min_p", args.min_p),
                         ("presence_penalty", args.presence_penalty),
                         ("enable_thinking", args.enable_thinking)):
                if v is not None:
                    exec_cfg[k] = v
        executors.append(create_executor(exec_cfg))
    print(f"executor: {args.executor} x{len(executors)} replicas; cfg base_urls={base_urls}", flush=True)

    out_path = Path(args.out) if args.out else ckpt / f"eval-parallel-{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    games_done = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=bs, thread_name_prefix="eval")

    fout = open(out_path, "w")
    while games_done < args.num_games:
        obs, infos = env.reset()
        n = len(obs)
        observation = list(obs)
        admissible = [infos.get("admissible_commands", [[]])[i] for i in range(n)]
        task = [extract_task_description(observation[i]) for i in range(n)]
        gamefile = [(infos.get("extra.gamefile") or [""] * n)[i] for i in range(n)]
        task_type = [classify_task(gamefile[i]) for i in range(n)]
        # Retrieve skills once per game (task is fixed for the episode).
        skills_text = [repo.format_skills(repo.retrieve(task[i], top_k=5)) for i in range(n)]
        history = [deque(maxlen=3) for _ in range(n)]
        done = [False] * n
        success = [False] * n
        steps = [0] * n

        rnd = 0
        while not all(done) and rnd < args.max_steps:
            rnd += 1
            # Fire executor calls for all active games concurrently.
            futs = {}
            for i in range(n):
                if done[i]:
                    continue
                futs[i] = pool.submit(
                    executors[i % len(executors)].act,
                    task_description=task[i], observation=observation[i],
                    admissible_actions=admissible[i], step_count=steps[i],
                    action_history="\n".join(history[i]), retrieved_skills=skills_text[i],
                )
            actions = []
            for i in range(n):
                if done[i]:
                    actions.append("look")  # batched step needs an action per slot
                    continue
                try:
                    actions.append(futs[i].result())
                except Exception as e:
                    print(f"  [warn] executor failed game {games_done+i}: {type(e).__name__}: {e}")
                    actions.append(admissible[i][0] if admissible[i] else "look")
            obs_n, scores, dones, infos = env.step(actions)
            for i in range(n):
                if done[i]:
                    continue
                observation[i] = obs_n[i]
                admissible[i] = infos.get("admissible_commands", [[]])[i]
                steps[i] += 1
                history[i].append(f"ACTION: {actions[i]}\nOBSERVATION: {observation[i]}")
                if dones[i]:
                    done[i] = True
                    success[i] = scores[i] > 0

        for i in range(n):
            if games_done >= args.num_games:
                break
            r = {"task_type": task_type[i], "success": bool(success[i]),
                 "steps": steps[i], "task": task[i], "gamefile": gamefile[i]}
            results.append(r)
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            games_done += 1
            print(f"  game {games_done:3d}/{args.num_games}  "
                  f"{'OK ' if r['success'] else 'XX '} {r['task_type']:6s} "
                  f"{r['steps']:2d} steps  {r['task'][:55]}")
    fout.close()
    pool.shutdown(wait=False)

    report_eval_results(results, args.split, len(repo), out_path,
                        time.time() - t0, checkpoint=str(ckpt))


if __name__ == "__main__":
    main()
