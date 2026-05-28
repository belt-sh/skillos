"""Closed-loop streaming-curation eval — the paper's actual ALFWorld eval protocol.

Per task in a deterministic split iteration:
  1. Executor solves the task with top-5 BM25 retrieval from current SkillRepo.
     → record success / step count.
  2. (closed_loop mode only) Trained curator runs on the trajectory → parse tool
     calls (`new_skill_insert` / `skill_update` / `skill_delete`) → mutate repo.
  3. Advance to next task.

The SAME 140 valid_seen tasks both contribute to memory AND are scored — early
tasks score low (sparse repo), later tasks score higher (accumulated memory).
SR is the aggregate over all tasks. Matches paper §3.1 / §B.3.3.

Three modes for a paired held-out comparison (run all three with the SAME split
+ num-games + ordering, then join the JSONLs on `gamefile` for McNemar):

  --mode no_memory  : empty repo, curator never invoked  (arm A: baseline)
  --mode closed_loop --curator-checkpoint <ckpt>         (arm B / arm C)

Same `infsh` Qwen3-8B executor as training — do NOT swap in the vLLM +
presence_penalty=1.5 config from the prior 41% eval; that train/eval mismatch
is what corrupted the previous read. See DIVERGENCES.md #12.

Usage:
  python -m scripts.eval_streaming_curation --mode no_memory \\
      --num-games 140 --split valid_seen \\
      --out output/eval-pathbv4/no_memory.jsonl

  python -m scripts.eval_streaming_curation --mode closed_loop \\
      --curator-checkpoint output/alfworld-8xh100-v4-pathb \\
      --num-games 140 --split valid_seen \\
      --out output/eval-pathbv4/ckpt60.jsonl

  python -m scripts.eval_streaming_curation --mode closed_loop \\
      --curator-checkpoint output/alfworld-8xh100-v4-pathb/checkpoint-10 \\
      --num-games 140 --split valid_seen \\
      --out output/eval-pathbv4/ckpt10.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Reuse the existing eval's executor-episode building blocks.
from scripts.eval_alfworld import classify_task, extract_task_description

# Tool schemas exposed to the curator. Mirrors the methods on CuratorEnv that
# TRL auto-discovered during training (new_skill_insert / skill_update /
# skill_delete) — the curator was trained to emit calls to exactly these.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "new_skill_insert",
            "description": "Insert a new skill into the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "Short, descriptive name for the skill."},
                    "content": {"type": "string", "description": "Skill body in markdown."},
                },
                "required": ["skill_name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_update",
            "description": "Update an existing skill's name or content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "new_name": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_delete",
            "description": "Delete an existing skill from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"skill_name": {"type": "string"}},
                "required": ["skill_name"],
            },
        },
    },
]


def _format_trajectory(traj_steps: list[dict]) -> str:
    parts = []
    for s in traj_steps:
        parts.append(f"Step {s['step']}: ACTION: {s['action']}")
        parts.append(f"        OBSERVATION: {s['observation']}")
    return "\n".join(parts)


def run_executor_episode_with_trace(env, executor, repo, max_steps: int,
                                    history_length: int = 3) -> dict:
    """Same protocol as scripts.eval_alfworld.run_episode but also captures the
    per-step trajectory text so we can hand it to the curator afterwards."""
    obs, infos = env.reset()
    observation = obs[0]
    admissible = infos.get("admissible_commands", [[]])[0]
    task = extract_task_description(observation)
    gamefile = (infos.get("extra.gamefile") or [""])[0]
    task_type = classify_task(gamefile)

    retrieved = repo.retrieve(task, top_k=5)
    skills_text = repo.format_skills(retrieved) if retrieved else ""
    n_retrieved = len(retrieved)

    history: deque[str] = deque(maxlen=history_length)
    traj: list[dict] = []
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
        new_obs = obs_n[0]
        step += 1
        traj.append({"step": step, "action": action, "observation": new_obs})
        observation = new_obs
        admissible = infos.get("admissible_commands", [[]])[0]
        done = dones[0]
        history.append(f"ACTION: {action}\nOBSERVATION: {observation}")
        if done:
            success = scores[0] > 0
    return {
        "task_type": task_type, "success": success, "steps": step,
        "task": task, "gamefile": gamefile, "trajectory": traj,
        "n_retrieved": n_retrieved,
    }


class CuratorInference:
    """Wraps the trained curator for closed-loop eval: build prompt, generate,
    parse tool calls, mutate the repo. Mirrors training-time inputs (prompts
    from skillos.curator.prompts, tools from CuratorEnv) so the model sees the
    same format it was trained on."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda",
                 max_new_tokens: int = 4096, temperature: float = 1.0,
                 enable_thinking: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch
        print(f"[curator] loading tokenizer + model from {checkpoint_dir}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        # Lazy-imported here so the parser code path matches training exactly.
        from skillos.curator.model import parse_tool_calls, apply_curation_ops
        from skillos.curator.prompts import CURATOR_SYSTEM, CURATOR_INPUT_TEMPLATE
        self._parse = parse_tool_calls
        self._apply = apply_curation_ops
        self._system = CURATOR_SYSTEM
        self._template = CURATOR_INPUT_TEMPLATE

    def curate(self, repo, traj_result: dict) -> dict:
        """Generate curation ops for one trajectory and apply them to `repo`."""
        # Past-skills view = what the curator would see retrieved for THIS task.
        past = repo.retrieve(traj_result["task"], top_k=5)
        past_text = repo.format_skills(past) if past else ""
        traj_text = _format_trajectory(traj_result["trajectory"])
        user = self._template.format(
            task_description=traj_result["task"],
            past_skills=past_text,
            agent_trajectory=traj_text,
            result="Success" if traj_result["success"] else "Failure",
        )
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": user},
        ]
        input_ids = self.tok.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        ).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
        )
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(self.temperature))
        else:
            gen_kwargs.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(input_ids, **gen_kwargs)
        gen = out[0, input_ids.shape[1]:]
        response = self.tok.decode(gen, skip_special_tokens=True)
        ops = self._parse(response)
        size_before = len(repo)
        applied = self._apply(repo, ops)
        return {
            "ops_parsed": len(ops),
            "ops_executed": sum(1 for o in applied if o.executed),
            "repo_size_before": size_before,
            "repo_size_after": len(repo),
            "repo_tokens_after": repo.total_tokens(),
            "response_chars": len(response),
        }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["no_memory", "closed_loop"])
    p.add_argument("--curator-checkpoint", default=None,
                   help="Path to the trained curator dir (required for closed_loop). "
                        "May be the run root (final model) or a checkpoint-N subdir.")
    p.add_argument("--split", default="valid_seen",
                   choices=["valid_seen", "valid_unseen", "train"])
    p.add_argument("--num-games", type=int, default=140,
                   help="Paper's ALFWorld test set = 140 valid_seen tasks.")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--executor", default="infsh", choices=["heuristic", "infsh"])
    p.add_argument("--executor-app", default="openrouter/qwen3-8b")
    p.add_argument("--curator-temperature", type=float, default=1.0,
                   help="Curator decode temperature (training was 1.0). Use 0 for greedy.")
    p.add_argument("--curator-max-new-tokens", type=int, default=4096)
    p.add_argument("--curator-device", default="cuda",
                   help="Device for the curator model (e.g. cuda, cuda:0).")
    p.add_argument("--out", required=True,
                   help="Per-game JSONL output. Compare arms by joining on `gamefile`.")
    args = p.parse_args()

    if args.mode == "closed_loop" and not args.curator_checkpoint:
        p.error("--curator-checkpoint is required in closed_loop mode")

    # Build executor + env. Executor settings mirror training (reasoning_effort
    # medium, max_tokens 8192) so we don't reintroduce a train/eval mismatch.
    exec_cfg = {"type": args.executor}
    if args.executor == "infsh":
        exec_cfg.update({
            "app": args.executor_app,
            "history_length": 3,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "context_size": 32768,
            "reasoning_effort": "medium",
        })
    from skillos.executor.executor import create_executor
    executor = create_executor(exec_cfg)

    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    env = make_alfworld_env(SPLIT_MAP[args.split], batch_size=1)

    from skillos.skills.repo import SkillRepo
    repo = SkillRepo()  # always starts empty — paper protocol

    curator = None
    if args.mode == "closed_loop":
        curator = CuratorInference(
            args.curator_checkpoint,
            device=args.curator_device,
            max_new_tokens=args.curator_max_new_tokens,
            temperature=args.curator_temperature,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] mode={args.mode}  split={args.split}  num_games={args.num_games}", flush=True)
    print(f"[eval] curator={args.curator_checkpoint or '<none>'}", flush=True)
    print(f"[eval] executor={args.executor}/{args.executor_app}", flush=True)
    print(f"[eval] out={out_path}", flush=True)

    records: list[dict] = []
    wall_start = time.time()
    with open(out_path, "w") as fh:
        for i in range(args.num_games):
            t0 = time.time()
            try:
                result = run_executor_episode_with_trace(env, executor, repo, args.max_steps)
            except Exception as e:
                print(f"  game {i}: executor episode FAILED — {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            ep_seconds = time.time() - t0

            curation_meta: dict = {}
            if args.mode == "closed_loop":
                tc = time.time()
                try:
                    curation_meta = curator.curate(repo, result)
                except Exception as e:
                    print(f"  game {i}: CURATOR error — {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    curation_meta = {"error": f"{type(e).__name__}: {e}"}
                curation_meta["curate_seconds"] = time.time() - tc

            rec = {
                "game_idx": i,
                "gamefile": result["gamefile"],
                "task_type": result["task_type"],
                "task": result["task"],
                "success": bool(result["success"]),
                "steps": result["steps"],
                "n_retrieved_at_eval": result["n_retrieved"],
                "episode_seconds": round(ep_seconds, 2),
                "repo_size_at_episode_start": curation_meta.get("repo_size_before",
                                                                  len(repo) - curation_meta.get("ops_executed", 0)),
                **{f"curator/{k}": v for k, v in curation_meta.items()},
            }
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            n_ok = sum(1 for r in records if r["success"])
            running_sr = n_ok / len(records)
            print(f"  [{i + 1:3d}/{args.num_games}] {result['task_type']:6s}  "
                  f"{'✓' if result['success'] else '✗'}  steps={result['steps']:2d}  "
                  f"repo={len(repo):3d}  ep={ep_seconds:5.0f}s  "
                  f"SR_so_far={n_ok}/{len(records)}={running_sr:.1%}", flush=True)

    wall = time.time() - wall_start
    n = len(records)
    n_ok = sum(1 for r in records if r["success"])
    overall = n_ok / max(n, 1)

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        by_type[r["task_type"]][0] += int(r["success"])
        by_type[r["task_type"]][1] += 1

    print(f"\n=== {args.mode}  ckpt={args.curator_checkpoint or '<none>'}  "
          f"split={args.split} ===")
    print(f"  TOTAL: {n_ok}/{n} = {overall:.1%}  "
          f"({wall:.0f}s total, {wall / max(n, 1):.0f}s/game avg)")
    for t in sorted(by_type):
        s, total = by_type[t]
        print(f"  {t:6s}: {s}/{total} = {s / max(total, 1):.1%}")
    if args.mode == "closed_loop":
        print(f"  final repo: {len(repo)} skills, {repo.total_tokens()} tokens")
        ops_total = sum(r.get("curator/ops_executed", 0) for r in records)
        print(f"  curator ops executed across run: {ops_total}")
    print(f"  JSONL: {out_path}")


if __name__ == "__main__":
    main()
