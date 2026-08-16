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
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

# Reuse the existing eval's executor-episode building blocks.
from scripts.eval_alfworld import classify_task, extract_task_description
from skillos.curator.prompts import format_trajectory
from skillos.executor.executor import get_parse_stats, get_reformat_stats

# Data-integrity gate. Above this share of episodes lost to upstream errors the
# arm is unusable, so stop instead of writing a file that looks comparable.
_ABORT_ERR_RATE = float(os.environ.get("SKILLOS_EVAL_MAX_ERROR_RATE", "0.02"))
_ABORT_MIN_GAMES = int(os.environ.get("SKILLOS_EVAL_ABORT_MIN_GAMES", "20"))

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
    # An upstream failure is NOT a task failure. Previously this loop caught the
    # exception and played admissible[0] instead, so a rate-limited or 401'd arm
    # kept "playing" with invented actions and scored the result as if the agent
    # had tried and lost. That silently voided four eval arms (r2a ckpt45-60,
    # 52-65% invented actions) and dragged eval-v8 ckpt60 down by 12%. Now the
    # episode is abandoned and flagged, and the caller drops it from the rate.
    errored = [False] * n
    n_exec_errors = [0] * n
    coerced_before = get_parse_stats()[1]

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
                # Abandon this episode. "look" is only to satisfy the batched
                # env.step contract; done[i] is set first so nothing about this
                # slot is recorded past the failure point.
                print(f"  [error] executor failed slot {i} at step {steps[i]}: "
                      f"{type(e).__name__}: {e} — ABANDONING episode "
                      f"(excluded from success rate)", file=sys.stderr, flush=True)
                errored[i] = True
                n_exec_errors[i] += 1
                done[i] = True
                actions.append("look")
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
    # Parse coercions are counted process-wide, so attribute the wave's delta
    # across its slots rather than pretending to a per-episode number we can't
    # get from a shared counter.
    coerced_wave = get_parse_stats()[1] - coerced_before
    return [
        {
            "task_type": task_type[i], "success": success[i], "steps": steps[i],
            "task": task[i], "gamefile": gamefile[i], "trajectory": traj[i],
            "n_retrieved": len(retrieved_lists[i]),
            "errored": errored[i], "n_exec_errors": n_exec_errors[i],
            "wave_action_coercions": coerced_wave,
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
        traj_text = format_trajectory(traj_result["trajectory"])
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
        # `enable_thinking` must be a direct kwarg: apply_chat_template has no
        # `chat_template_kwargs` parameter, so the dict form was silently
        # ignored and the curator ran in thinking mode despite being trained
        # with thinking disabled (postmortem 2026-06-10, eval findings).
        enc = self.tok.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=self.enable_thinking,
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
        return apply_tool_calls_to_repo(
            repo, parsed.get("tool_calls") or [],
            self._apply, self._CurationOp,
            response_chars=len(parsed.get("content", "")),
        )


def build_curator_messages(repo, traj_result: dict, system: str, template: str):
    """The curator's prompt, built identically for every backend.

    Factored out so a remote curator cannot accidentally be evaluated on a
    different prompt than the trained one: same top-5 BM25 retrieval, same
    trajectory formatting, same system prompt and user template.
    """
    past = repo.retrieve(traj_result["task"], top_k=5)
    past_text = repo.format_skills(past) if past else ""
    traj_text = format_trajectory(traj_result["trajectory"])
    user = template.format(
        task_description=traj_result["task"],
        past_skills=past_text,
        agent_trajectory=traj_text,
        result="Success" if traj_result["success"] else "Failure",
    )
    return system, user


def apply_tool_calls_to_repo(repo, tool_calls, apply_fn, op_cls,
                             response_chars: int = 0) -> dict:
    """Turn OpenAI-style tool calls into curation ops and apply them.

    Shared by the local and remote curators so the two differ *only* in how the
    tool calls were generated, never in how they are interpreted.
    """
    ops = []
    for tc in tool_calls or []:
        # Local (TRL parser) always sets type=="function"; some remote providers
        # omit it, so accept a bare {"name", "arguments"} shape too.
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = (fn or {}).get("name", "")
        if name not in ("new_skill_insert", "skill_update", "skill_delete"):
            continue
        args = (fn or {}).get("arguments") or {}
        if isinstance(args, str):
            # Tolerate arguments accidentally serialized as a JSON string.
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            continue
        ops.append(op_cls(name=name, arguments=args))
    size_before = len(repo)
    applied = apply_fn(repo, ops)
    return {
        "ops_parsed": len(ops),
        "ops_executed": sum(1 for o in applied if o.executed),
        "repo_size_before": size_before,
        "repo_size_after": len(repo),
        "repo_tokens_after": repo.total_tokens(),
        "response_chars": response_chars,
    }


class RemoteCuratorInference:
    """Curator served over the inference.sh API instead of a local checkpoint.

    Exists to test the paper's headline economic claim: that an RL-trained 8B
    curator beats a frontier model used directly as the curator. For that
    comparison to mean anything the frontier model must see *exactly* what the
    trained one saw, so prompts, tool schema and retrieval all come from the
    same helpers the local path uses. Only generation differs.

    Drop-in for CuratorInference: same .curate(repo, traj_result) signature.
    """

    def __init__(self, app: str = "google/gemini-2-5-pro",
                 temperature: float = 0.0, max_tokens: int = 8192,
                 reasoning_effort: str | None = "medium",
                 infra: str = "cloud", variant: str = "default"):
        from inferencesh import inference
        from skillos.utils.infsh_auth import resolve_infsh_api_key
        self.app = app
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.infra = infra
        self.variant = variant
        self.client = inference(api_key=resolve_infsh_api_key())
        from skillos.curator.model import apply_curation_ops, CurationOp
        from skillos.curator.prompts import CURATOR_SYSTEM, CURATOR_INPUT_TEMPLATE
        self._apply = apply_curation_ops
        self._CurationOp = CurationOp
        self._system = CURATOR_SYSTEM
        self._template = CURATOR_INPUT_TEMPLATE
        # Providers that ignore the tools field would silently produce zero ops
        # and look like a curator that chose to do nothing. Count the fallbacks
        # so the run can be audited instead of quietly scoring as a null.
        self.n_calls = 0
        self.n_text_fallback = 0
        print(f"[curator] remote backend: {app} "
              f"(temp={temperature}, reasoning={reasoning_effort})", flush=True)

    def curate(self, repo, traj_result: dict) -> dict:
        system, user = build_curator_messages(
            repo, traj_result, self._system, self._template)
        payload = {
            "text": user,
            "system_prompt": system,
            "tools": TOOLS_SCHEMA,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        from skillos.utils.infsh_client import run_task_resilient
        from skillos.executor.executor import _log_infsh_task
        result = run_task_resilient(
            self.client,
            {"app": self.app, "infra": self.infra,
             "variant": self.variant, "input": payload},
            on_task_id=lambda tid: _log_infsh_task("curator", self.app, tid),
        )
        output = (result or {}).get("output") or {}
        tool_calls = output.get("tool_calls") or []
        text = output.get("response") or ""
        self.n_calls += 1
        if not tool_calls and text:
            # Fall back to the <tool_call>{...}</tool_call> convention Qwen3 uses,
            # in case the provider returned the calls as prose.
            found = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
            for blob in found:
                try:
                    tool_calls.append({"function": json.loads(blob)})
                except Exception:
                    pass
            if tool_calls:
                self.n_text_fallback += 1
        meta = apply_tool_calls_to_repo(
            repo, tool_calls, self._apply, self._CurationOp,
            response_chars=len(text),
        )
        usage = output.get("usage") or {}
        if isinstance(usage, dict):
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                if k in usage:
                    meta[f"usage_{k}"] = usage[k]
        return meta


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True, choices=["no_memory", "closed_loop"])
    p.add_argument("--curator-checkpoint", default=None,
                   help="Path to the trained curator dir (required for closed_loop). "
                        "May be the run root (final model) or a checkpoint-N subdir.")
    p.add_argument("--static-repo", default=None,
                   help="Preload this directory of markdown skills and disable curation. "
                        "Used for the hand-written oracle upper-bound arm.")
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
    p.add_argument("--executor-temperature", type=float, default=0.6,
                   help="Executor decode temp. Eval baseline 0.6; GiGPO/paper-faithful 0.4.")
    p.add_argument("--executor-top-p", type=float, default=0.95)
    p.add_argument("--executor-top-k", type=int, default=20,
                   help="Executor top_k. Set <=0 to disable (GiGPO-faithful).")
    p.add_argument("--executor-reasoning", default="medium",
                   help="reasoning_effort for the infsh executor (low/medium/high).")
    p.add_argument("--curator-temperature", type=float, default=1.0,
                   help="Curator decode temperature (training was 1.0). Use 0 for greedy.")
    p.add_argument("--curator-max-new-tokens", type=int, default=4096)
    p.add_argument("--curator-device", default="cuda",
                   help="Device for the curator model (e.g. cuda, cuda:0).")
    p.add_argument("--curator-backend", default="local", choices=["local", "remote"],
                   help="local = trained checkpoint on GPU (default, unchanged). "
                        "remote = a hosted model as the curator, for the paper's "
                        "'trained 8B beats a frontier curator' comparison. Remote "
                        "needs no GPU and no --curator-checkpoint.")
    p.add_argument("--curator-app", default="google/gemini-2-5-pro",
                   help="inference.sh app id for --curator-backend remote.")
    p.add_argument("--curator-reasoning-effort", default="medium",
                   help="Thinking budget for a remote curator; 'none' to disable.")
    p.add_argument("--out", required=True,
                   help="Per-game JSONL output. Compare arms by joining on `gamefile`.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow clobbering an existing --out file.")
    args = p.parse_args()

    if args.mode == "closed_loop" and args.curator_backend == "local" \
            and not args.curator_checkpoint:
        p.error("--curator-checkpoint is required in closed_loop mode "
                "with --curator-backend local")
    if Path(args.out).exists() and not args.overwrite:
        p.error(f"{args.out} already exists — a crashed/finished arm would be "
                "silently truncated. Pass --overwrite to clobber.")

    # Executor settings mirror training (reasoning_effort medium, max_tokens
    # 8192) so we don't reintroduce a train/eval mismatch.
    exec_cfg = {"type": args.executor}
    if args.executor == "infsh":
        exec_cfg.update({
            "app": args.executor_app,
            "history_length": 3,
            "temperature": args.executor_temperature,
            "top_p": args.executor_top_p,
            "top_k": args.executor_top_k if args.executor_top_k > 0 else None,
            "max_tokens": 8192,
            "context_size": 32768,
            "reasoning_effort": args.executor_reasoning,
        })
    from skillos.executor.executor import create_executor
    executor = create_executor(exec_cfg)

    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    bs = max(1, min(args.batch_size, args.num_games))
    env = make_alfworld_env(SPLIT_MAP[args.split], batch_size=bs)

    from skillos.skills.repo import SkillRepo
    repo = SkillRepo()  # always starts empty — paper protocol

    # `--static-repo DIR` preloads a fixed skill set and never curates. Used for
    # the hand-written oracle arm, which bounds what curation could be worth on
    # this executor: same retrieval, same prompt, human-authored content.
    if args.static_repo:
        n_loaded = 0
        for md in sorted(Path(args.static_repo).glob("*.md")):
            text = md.read_text()
            if not text.lstrip().startswith("---"):
                continue  # skip README and other non-skill files
            repo.insert(md.stem, text)
            n_loaded += 1
        if n_loaded == 0:
            raise SystemExit(f"--static-repo {args.static_repo} contained no skills")
        print(f"[eval] static repo: {n_loaded} hand-written skills, curation disabled",
              flush=True)

    curator = None
    if args.mode == "closed_loop" and args.curator_backend == "remote":
        curator = RemoteCuratorInference(
            app=args.curator_app,
            temperature=args.curator_temperature,
            reasoning_effort=(None if args.curator_reasoning_effort in ("none", "")
                              else args.curator_reasoning_effort),
        )
    elif args.mode == "closed_loop":
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
    _cur_desc = (args.curator_app if args.curator_backend == "remote"
                 else (args.curator_checkpoint or "<none>"))
    print(f"[eval] curator={_cur_desc} (backend={args.curator_backend})", flush=True)
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
                    "errored": bool(result["errored"]),
                    "n_exec_errors": result["n_exec_errors"],
                    "wave_action_coercions": result["wave_action_coercions"],
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

                scored = [r for r in records if not r["errored"]]
                n_ok = sum(1 for r in scored if r["success"])
                running_sr = n_ok / max(len(scored), 1)
                mark = "ER" if result["errored"] else ("OK" if result["success"] else "XX")
                print(f"  [{games_done:3d}/{args.num_games}] w{wave_idx:02d}.{slot:02d} "
                      f"{result['task_type']:6s}  "
                      f"{mark}  steps={result['steps']:2d}  "
                      f"repo={len(repo):3d}  "
                      f"SR_so_far={n_ok}/{len(scored)}={running_sr:.1%}", flush=True)
            n_err = sum(1 for r in records if r["errored"])
            err_rate = n_err / max(len(records), 1)
            calls, coerced = get_parse_stats()
            print(f"  --- wave {wave_idx} done: exec={executor_wave_seconds:.0f}s  "
                  f"curator={curator_wave_seconds:.0f}s  repo_now={len(repo)}  "
                  f"abandoned={n_err}/{len(records)}={err_rate:.1%}  "
                  f"coerced_actions={coerced}/{calls}="
                  f"{coerced / max(calls, 1):.1%} ---", flush=True)
            # Hard stop rather than write a quietly-corrupt arm. An arm that has
            # lost this much of its data cannot be compared to a clean baseline,
            # and the only thing worse than losing it is publishing it.
            if len(records) >= _ABORT_MIN_GAMES and err_rate > _ABORT_ERR_RATE:
                print(f"\n[ABORT] {err_rate:.1%} of episodes abandoned to upstream "
                      f"errors (limit {_ABORT_ERR_RATE:.1%}). This arm is not "
                      f"usable; fix the upstream problem and re-run. Partial "
                      f"output left at {out_path} for diagnosis.",
                      file=sys.stderr, flush=True)
                raise SystemExit(3)
    pool.shutdown(wait=False)

    wall = time.time() - wall_start
    n_all = len(records)
    scored = [r for r in records if not r["errored"]]
    n = len(scored)
    n_ok = sum(1 for r in scored if r["success"])
    overall = n_ok / max(n, 1)
    n_err = n_all - n

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in scored:
        by_type[r["task_type"]][0] += int(r["success"])
        by_type[r["task_type"]][1] += 1

    print(f"\n=== {args.mode}  curator={_cur_desc}  "
          f"split={args.split}  bs={bs} ===")
    print(f"  TOTAL: {n_ok}/{n} = {overall:.1%}  "
          f"({wall:.0f}s total, {wall / max(n, 1):.0f}s/game avg, "
          f"{wave_idx} waves)")
    calls, coerced = get_parse_stats()
    tries, recovered = get_reformat_stats()
    print(f"  data integrity: {n_err}/{n_all} episodes abandoned to upstream "
          f"errors; {coerced}/{calls} actions coerced to admissible[0] "
          f"({coerced / max(calls, 1):.1%})")
    if tries:
        print(f"  reformat retries: {tries} unparseable outputs re-asked, "
              f"{recovered} recovered ({recovered / max(tries, 1):.0%}); "
              f"without the retry, coercion would have been "
              f"{(coerced + recovered) / max(calls, 1):.1%}")
    for t in sorted(by_type):
        s, total = by_type[t]
        print(f"  {t:6s}: {s}/{total} = {s / max(total, 1):.1%}")
    if args.mode == "closed_loop":
        print(f"  final repo: {len(repo)} skills, {repo.total_tokens()} tokens")
        ops_total = sum(r.get("curator/ops_executed", 0) for r in records)
        print(f"  curator ops executed across run: {ops_total}")

    # Always dump the end-of-run repository next to the JSONL. A curator arm's
    # success rate says nothing about WHAT the curator wrote, and without the
    # text there is no way to analyse mechanism after the fact. Cheap to keep.
    repo_path = out_path.with_suffix(".repo.md")
    with open(repo_path, "w") as rf:
        rf.write(f"<!-- final repo: {len(repo)} skills, {repo.total_tokens()} tokens\n")
        rf.write(f"     arm: {args.mode} curator={_cur_desc} split={args.split} -->\n\n")
        for name, skill in sorted(repo.skills.items()):
            rf.write(f"{'=' * 70}\n# {name}\n{'=' * 70}\n{skill.content}\n\n")
    print(f"  repo dump: {repo_path}")
    print(f"  JSONL: {out_path}")


if __name__ == "__main__":
    main()
