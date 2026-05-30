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

`--batch-size K` (default 1 = strict per-task serial, paper-literal) runs K
games concurrently against the SAME repo snapshot, then runs the curator
serially over the wave's K trajectories before the next wave. This is a
documented deviation from strict per-task ordering — memory updates happen at
wave boundaries instead of after every task — for a ~K× wall-clock speedup.
Within-wave games don't see each other's curation; across waves the repo still
accumulates. Set K=1 for paper-literal serial.

Usage:
  python -m scripts.eval_streaming_curation --mode no_memory \\
      --num-games 140 --split valid_seen --batch-size 20 \\
      --out output/eval-pathbv4/no_memory.jsonl

  python -m scripts.eval_streaming_curation --mode closed_loop \\
      --curator-checkpoint output/alfworld-8xh100-v4-pathb \\
      --num-games 140 --split valid_seen --batch-size 20 \\
      --out output/eval-pathbv4/ckpt60.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
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


def run_executor_wave_with_trace(env, executors, repo, max_steps: int, pool,
                                 history_length: int = 3) -> list[dict]:
    """Run one wave of `env.batch_size` games concurrently against the current
    repo snapshot. Mirrors scripts.eval_alfworld_parallel: the ALFWorld env is
    stepped single-threaded; only the network-bound executor calls run
    concurrently (one inference.sh request per active game per step round).

    Returns an ordered list of per-game result dicts (one entry per env slot)
    with the same schema as the prior serial run_executor_episode_with_trace.
    All games in a wave retrieve skills from the SAME repo snapshot — paper-
    faithful per-task ordering only holds at wave boundaries.
    """
    obs, infos = env.reset()
    n = len(obs)
    observation = list(obs)
    admissible = [infos.get("admissible_commands", [[]])[i] for i in range(n)]
    task = [extract_task_description(observation[i]) for i in range(n)]
    gamefile = [(infos.get("extra.gamefile") or [""] * n)[i] for i in range(n)]
    task_type = [classify_task(gamefile[i]) for i in range(n)]
    retrieved_lists = [repo.retrieve(task[i], top_k=5) for i in range(n)]
    skills_text = [repo.format_skills(r) if r else "" for r in retrieved_lists]
    history = [deque(maxlen=history_length) for _ in range(n)]
    traj: list[list[dict]] = [[] for _ in range(n)]
    done = [False] * n
    success = [False] * n
    steps = [0] * n

    rnd = 0
    while not all(done) and rnd < max_steps:
        rnd += 1
        futs: dict[int, concurrent.futures.Future] = {}
        for i in range(n):
            if done[i]:
                continue
            futs[i] = pool.submit(
                executors[i % len(executors)].act,
                task_description=task[i], observation=observation[i],
                admissible_actions=admissible[i], step_count=steps[i],
                action_history="\n".join(history[i]), retrieved_skills=skills_text[i],
            )
        actions: list[str] = []
        for i in range(n):
            if done[i]:
                actions.append("look")  # batched env.step needs an action per slot
                continue
            try:
                actions.append(futs[i].result())
            except Exception as e:
                print(f"  [warn] executor failed slot {i}: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                actions.append(admissible[i][0] if admissible[i] else "look")
        obs_n, scores, dones, infos = env.step(actions)
        for i in range(n):
            if done[i]:
                continue
            observation[i] = obs_n[i]
            admissible[i] = infos.get("admissible_commands", [[]])[i]
            steps[i] += 1
            traj[i].append({"step": steps[i], "action": actions[i], "observation": observation[i]})
            history[i].append(f"ACTION: {actions[i]}\nOBSERVATION: {observation[i]}")
            if dones[i]:
                done[i] = True
                success[i] = scores[i] > 0
    return [
        {
            "task_type": task_type[i], "success": success[i], "steps": steps[i],
            "task": task[i], "gamefile": gamefile[i], "trajectory": traj[i],
            "n_retrieved": len(retrieved_lists[i]),
        }
        for i in range(n)
    ]


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
        # Use TRL's response parser — it reads tokenizer.response_schema to
        # decode <tool_call>...</tool_call> blocks the same way training did.
        # The legacy regex parse_tool_calls in skillos.curator.model never
        # matched real Qwen3 tool-call output (lookahead required `{` or EOF
        # after `}`, but the model emits `}\n</tool_call>`).
        from trl.chat_template_utils import add_response_schema, parse_response
        add_response_schema(self.tok)
        self._parse_response = parse_response
        from skillos.curator.model import apply_curation_ops, CurationOp
        from skillos.curator.prompts import CURATOR_SYSTEM, CURATOR_INPUT_TEMPLATE
        self._apply = apply_curation_ops
        self._CurationOp = CurationOp
        self._system = CURATOR_SYSTEM
        self._template = CURATOR_INPUT_TEMPLATE

    def curate(self, repo, traj_result: dict) -> dict:
        """Generate curation ops for one trajectory and apply them to `repo`."""
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
        # With `tools=`, apply_chat_template returns a BatchEncoding (dict-like)
        # whose __getattr__ raises empty AttributeError on .shape — passing it
        # straight to generate() fails. Pull input_ids + attention_mask out.
        enc = self.tok.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
        )
        if attn_mask is not None:
            gen_kwargs["attention_mask"] = attn_mask
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(self.temperature))
        else:
            gen_kwargs.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(input_ids, **gen_kwargs)
        gen_ids = out[0, input_ids.shape[1]:].tolist()
        parsed = self._parse_response(self.tok, gen_ids)
        tool_calls = parsed.get("tool_calls") or []
        ops = []
        for tc in tool_calls:
            fn = tc.get("function") if tc.get("type") == "function" else None
            if not fn:
                continue
            name = fn.get("name", "")
            if name not in ("new_skill_insert", "skill_update", "skill_delete"):
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                # Tolerate arguments accidentally serialized as a JSON string.
                import json as _json
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            ops.append(self._CurationOp(name=name, arguments=args))
        size_before = len(repo)
        applied = self._apply(repo, ops)
        return {
            "ops_parsed": len(ops),
            "ops_executed": sum(1 for o in applied if o.executed),
            "repo_size_before": size_before,
            "repo_size_after": len(repo),
            "repo_tokens_after": repo.total_tokens(),
            "response_chars": len(parsed.get("content", "")),
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
    p.add_argument("--batch-size", type=int, default=1,
                   help="Games run concurrently per wave. K=1 = strict paper-literal serial. "
                        "K>1 = wave-batched: K games share the same repo snapshot, curator "
                        "runs serially between waves. ~K× faster, small deviation.")
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

    # Executor settings mirror training (reasoning_effort medium, max_tokens
    # 8192) so we don't reintroduce a train/eval mismatch.
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
    bs = max(1, min(args.batch_size, args.num_games))
    env = make_alfworld_env(SPLIT_MAP[args.split], batch_size=bs)

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

    print(f"[eval] mode={args.mode}  split={args.split}  num_games={args.num_games}  "
          f"batch_size={bs}", flush=True)
    print(f"[eval] curator={args.curator_checkpoint or '<none>'}", flush=True)
    print(f"[eval] executor={args.executor}/{args.executor_app}", flush=True)
    print(f"[eval] out={out_path}", flush=True)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=bs, thread_name_prefix="eval-wave")
    records: list[dict] = []
    wall_start = time.time()
    games_done = 0
    wave_idx = 0
    with open(out_path, "w") as fh:
        while games_done < args.num_games:
            wave_idx += 1
            wave_repo_size_before = len(repo)
            t_exec = time.time()
            wave_results = run_executor_wave_with_trace(
                env, [executor], repo, args.max_steps, pool)
            executor_wave_seconds = time.time() - t_exec

            curator_wave_seconds = 0.0
            for slot, result in enumerate(wave_results):
                if games_done >= args.num_games:
                    break

                curation_meta: dict = {}
                if args.mode == "closed_loop":
                    tc = time.time()
                    try:
                        curation_meta = curator.curate(repo, result)
                    except Exception as e:
                        import traceback
                        msg = f"{type(e).__name__}: {e!r}"
                        print(f"  game {games_done}: CURATOR error — {msg}",
                              file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        sys.stderr.flush()
                        curation_meta = {"error": msg}
                    curation_meta["curate_seconds"] = round(time.time() - tc, 2)
                    curator_wave_seconds += curation_meta["curate_seconds"]

                rec = {
                    "game_idx": games_done,
                    "wave_idx": wave_idx,
                    "wave_slot": slot,
                    "gamefile": result["gamefile"],
                    "task_type": result["task_type"],
                    "task": result["task"],
                    "success": bool(result["success"]),
                    "steps": result["steps"],
                    "n_retrieved_at_eval": result["n_retrieved"],
                    "executor_wave_seconds": round(executor_wave_seconds, 2),
                    "repo_size_at_episode_start": wave_repo_size_before,
                    **{f"curator/{k}": v for k, v in curation_meta.items()},
                }
                records.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                games_done += 1

                n_ok = sum(1 for r in records if r["success"])
                running_sr = n_ok / len(records)
                print(f"  [{games_done:3d}/{args.num_games}] w{wave_idx:02d}.{slot:02d} "
                      f"{result['task_type']:6s}  "
                      f"{'OK' if result['success'] else 'XX'}  steps={result['steps']:2d}  "
                      f"repo={len(repo):3d}  "
                      f"SR_so_far={n_ok}/{len(records)}={running_sr:.1%}", flush=True)
            print(f"  --- wave {wave_idx} done: exec={executor_wave_seconds:.0f}s  "
                  f"curator={curator_wave_seconds:.0f}s  repo_now={len(repo)} ---", flush=True)
    pool.shutdown(wait=False)

    wall = time.time() - wall_start
    n = len(records)
    n_ok = sum(1 for r in records if r["success"])
    overall = n_ok / max(n, 1)

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in records:
        by_type[r["task_type"]][0] += int(r["success"])
        by_type[r["task_type"]][1] += 1

    print(f"\n=== {args.mode}  ckpt={args.curator_checkpoint or '<none>'}  "
          f"split={args.split}  bs={bs} ===")
    print(f"  TOTAL: {n_ok}/{n} = {overall:.1%}  "
          f"({wall:.0f}s total, {wall / max(n, 1):.0f}s/game avg, "
          f"{wave_idx} waves)")
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
