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
from collections import defaultdict, deque
from pathlib import Path

from scripts.eval_alfworld import classify_task, extract_task_description


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="valid_seen",
                   choices=["valid_seen", "valid_unseen", "train"])
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=25,
                   help="games run concurrently per batch (executor calls fired in parallel)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--executor", default="infsh", choices=["heuristic", "infsh"])
    p.add_argument("--executor-app", default="openrouter/qwen3-8b")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    skills_dir = ckpt / "skills"
    if not skills_dir.is_dir():
        skills_dir = ckpt.parent / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"No skills/ under {ckpt} or {ckpt.parent}")

    from skillos.skills.repo import SkillRepo
    repo = SkillRepo.load(str(skills_dir))
    print(f"Loaded {len(repo)} skills from {skills_dir}")
    if len(repo) == 0:
        print("WARNING: empty repo — No Memory baseline.")

    from skillos.envs.config import load_alfworld_config
    from alfworld.agents.environment import get_environment
    cfg = load_alfworld_config()
    AlfredTWEnv = get_environment("AlfredTWEnv")
    split_map = {"valid_seen": "eval_in_distribution",
                 "valid_unseen": "eval_out_of_distribution", "train": "train"}
    bs = min(args.batch_size, args.num_games)
    env = AlfredTWEnv(cfg, train_eval=split_map[args.split]).init_env(batch_size=bs)

    from skillos.executor.executor import create_executor
    exec_cfg = {"type": args.executor}
    if args.executor == "infsh":
        exec_cfg["app"] = args.executor_app
    executor = create_executor(exec_cfg)

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
                    executor.act,
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

    # aggregate
    by_type = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)
    cols = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]
    print("\n=== ALFWorld parallel eval — Table 1 row ===")
    print(f"  {args.split:14s}" + "".join(f" {c:>7s}" for c in cols) + "   Avg SR   Steps")
    row = f"  skills={len(repo):4d}  "
    ts, tn, tstep = 0, 0, 0
    for c in cols:
        b = by_type.get(c, [])
        if not b:
            row += f" {'-':>7s}"; continue
        sr = sum(1 for x in b if x["success"]) / len(b)
        row += f" {sr*100:6.1f}%"
        ts += sum(1 for x in b if x["success"]); tn += len(b); tstep += sum(x["steps"] for x in b)
    avg_sr = ts / tn if tn else 0
    row += f"   {avg_sr*100:5.1f}%   {tstep/tn if tn else 0:5.1f}"
    print(row)
    print(f"\n  total {len(results)} games in {time.time()-t0:.0f}s  ({(time.time()-t0)/max(len(results),1):.1f}s/game)")
    print(f"  log: {out_path}")
    summary = out_path.with_suffix(".summary.json")
    summary.write_text(json.dumps({
        "split": args.split, "skills": len(repo), "n_games": len(results),
        "batch_size": bs, "wall_seconds": time.time() - t0,
        "avg_success_rate": avg_sr,
        "by_type": {c: {"n": len(by_type.get(c, [])),
                        "success_rate": (sum(1 for x in by_type[c] if x["success"]) / len(by_type[c]))
                        if by_type.get(c) else None} for c in cols},
    }, indent=2))
    print(f"  summary: {summary}")


if __name__ == "__main__":
    main()
