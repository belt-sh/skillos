"""ALFWorld benchmark eval — produces the paper's Table 1 row.

Loads a trained curator checkpoint (LoRA adapter + persisted skill repo) and
runs the frozen executor on held-out ALFWorld games (valid_seen for in-dist,
valid_unseen for out-of-dist). Aggregates success rate per task type
(Pick, Look, Clean, Heat, Cool, Pick2) and average steps to completion.

The trained curator is NOT invoked during eval — only the skills it produced.
The executor retrieves top-k skills via BM25 and decides ALFWorld actions.
This matches the paper's evaluation protocol (curator artifact = skill repo).

Usage:
    python -m scripts.eval_alfworld \\
        --checkpoint output/alfworld-lora-pilot/checkpoint-10 \\
        --split valid_seen \\
        --num-games 50 \\
        --max-steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from skillos.envs.task_types import DISPLAY, classify_gamefile

def classify_task(gamefile: str) -> str:
    """Map an alfworld gamefile path to its Table-1 task-type column."""
    return DISPLAY[classify_gamefile(gamefile)]


def extract_task_description(observation: str) -> str:
    if not observation:
        return "Unknown task"
    for line in observation.splitlines():
        s = line.strip()
        if s.lower().startswith("your task is"):
            return s
    return observation.splitlines()[0].strip()


# The 6 paper task types, in Table 1 column order.
TASK_COLS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]


def resolve_skills_dir(ckpt: Path) -> Path:
    """Find the skills/ dir for a checkpoint, tolerating the layout where
    skills were saved at <output_dir>/skills rather than under the checkpoint."""
    skills_dir = ckpt / "skills"
    if not skills_dir.is_dir():
        skills_dir = ckpt.parent / "skills"
    if not skills_dir.is_dir():
        raise FileNotFoundError(
            f"No skills/ directory under {ckpt} or {ckpt.parent}. "
            "Did training run with the SkillRepoSaver callback?"
        )
    return skills_dir


def report_eval_results(results: list[dict], split: str, n_skills: int,
                        out_path: Path, wall_seconds: float,
                        checkpoint: str | None = None) -> dict:
    """Aggregate per-episode results into the paper's Table 1 row, print it,
    and write a <out>.summary.json. Shared by the serial and parallel evals."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["task_type"]].append(r)

    def _stats(bucket: list[dict]) -> tuple[float | None, float | None]:
        if not bucket:
            return None, None
        sr = sum(1 for x in bucket if x["success"]) / len(bucket)
        steps = sum(x["steps"] for x in bucket) / len(bucket)
        return sr, steps

    print("\n=== ALFWorld eval — Table 1 row ===")
    print(f"  {split:14s}" + "".join(f" {c:>7s}" for c in TASK_COLS) + "   Avg SR   Steps")
    row = f"  skills={n_skills:4d}  "
    totals_succ = totals_n = total_steps = 0
    per_type: dict[str, dict] = {}
    for c in TASK_COLS:
        bucket = by_type.get(c, [])
        sr, steps = _stats(bucket)
        per_type[c] = {"n": len(bucket), "success_rate": sr, "avg_steps": steps}
        row += f" {sr*100:6.1f}%" if sr is not None else f" {'-':>7s}"
        totals_succ += sum(1 for x in bucket if x["success"])
        totals_n += len(bucket)
        total_steps += sum(x["steps"] for x in bucket)
    avg_sr = totals_succ / totals_n if totals_n else 0.0
    avg_st = total_steps / totals_n if totals_n else 0.0
    row += f"   {avg_sr*100:5.1f}%   {avg_st:5.1f}"
    print(row)

    summary = {
        "split": split, "skills": n_skills, "n_games": len(results),
        "wall_seconds": wall_seconds, "by_type": per_type,
        "avg_success_rate": avg_sr, "avg_steps": avg_st,
    }
    if checkpoint is not None:
        summary["checkpoint"] = checkpoint
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nPer-episode log:  {out_path}")
    print(f"Summary:          {summary_path}")
    print(f"Wall time:        {wall_seconds:.0f}s "
          f"({wall_seconds/max(len(results),1):.1f}s/game)")
    return summary


def run_episode(env, executor, repo, max_steps: int, history_length: int = 3) -> dict:
    """Run one held-out game with the frozen executor + retrieved skills."""
    obs, infos = env.reset()
    observation = obs[0]
    admissible = infos.get("admissible_commands", [[]])[0]
    task = extract_task_description(observation)
    gamefile = (infos.get("extra.gamefile") or [""])[0]
    task_type = classify_task(gamefile)

    retrieved = repo.retrieve(task, top_k=5)
    skills_text = repo.format_skills(retrieved)

    history: deque[str] = deque(maxlen=history_length)
    done = False
    success = False
    step = 0
    while not done and step < max_steps:
        action = executor.act(
            task_description=task,
            observation=observation,
            admissible_actions=admissible,
            step_count=step,
            action_history="\n".join(history),
            retrieved_skills=skills_text,
        )
        obs_n, scores, dones, infos = env.step([action])
        observation = obs_n[0]
        admissible = infos.get("admissible_commands", [[]])[0]
        done = dones[0]
        step += 1
        history.append(f"ACTION: {action}\nOBSERVATION: {observation}")
        if done:
            success = scores[0] > 0
    return {
        "task_type": task_type,
        "success": success,
        "steps": step,
        "task": task,
        "gamefile": gamefile,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint dir containing skills/ and adapter_model.safetensors")
    p.add_argument("--split", default="valid_seen",
                   choices=["valid_seen", "valid_unseen", "train"],
                   help="ALFWorld split to evaluate on")
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--executor", default="infsh",
                   choices=["heuristic", "infsh"])
    p.add_argument("--executor-app", default="openrouter/qwen3-8b")
    p.add_argument("--out", default=None,
                   help="Where to write the per-episode JSONL log (default: <ckpt>/eval-<split>.jsonl)")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    skills_dir = resolve_skills_dir(ckpt)

    from skillos.skills.repo import SkillRepo
    repo = SkillRepo.load(str(skills_dir))
    print(f"Loaded {len(repo)} skills from {skills_dir}")
    if len(repo) == 0:
        print("WARNING: repo is empty — eval will use no retrieved skills "
              "(equivalent to the 'No Memory' baseline).")

    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    env = make_alfworld_env(SPLIT_MAP[args.split], batch_size=1)

    from skillos.executor.executor import create_executor
    exec_cfg = {"type": args.executor}
    if args.executor == "infsh":
        exec_cfg["app"] = args.executor_app
    executor = create_executor(exec_cfg)

    out_path = Path(args.out) if args.out else ckpt / f"eval-{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    with open(out_path, "w") as f:
        for i in range(args.num_games):
            try:
                r = run_episode(env, executor, repo, args.max_steps)
            except Exception as e:
                print(f"  game {i}: error — {type(e).__name__}: {e}")
                continue
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            status = "✓" if r["success"] else "✗"
            print(f"  game {i+1:3d}/{args.num_games}  {status}  {r['task_type']:6s}  "
                  f"{r['steps']:2d} steps  {r['task'][:60]}")

    report_eval_results(results, args.split, len(repo), out_path,
                        time.time() - t0, checkpoint=str(ckpt))


if __name__ == "__main__":
    main()
