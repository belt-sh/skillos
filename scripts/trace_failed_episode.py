"""Trace ONE full ALFWorld episode step-by-step to find the STRUCTURAL gap
behind the no-memory baseline shortfall (33.6% vs paper 47.9%).

Per the paper-repro-decode-settings-audit skill: decode knobs are exhausted
(reasoning ON, 8192-token budget not truncating, <action> parses). The terminal
move is to read one failed episode and look for the model IGNORING an available
atomic action and improvising a wrong procedure — invisible to any sampler.

Key instrumentation: at each step we log the RAW <action> the model emitted,
whether it was admissible, and what _parse_action actually executed. A large
"raw != executed" rate means the model is off-grammar and getting silently
coerced to admissible[0] (a 'go to ...' default) — the structural smoking gun.

Run:
  python -m scripts.trace_failed_episode --task-type Heat --max-steps 30
"""
from __future__ import annotations

import argparse
import os
import re
import sys

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-type", default="Heat",
                    help="Heat/Clean/Cool/Pick/Look/Pick2 — composite verbs are the failing ones.")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-resets", type=int, default=40,
                    help="How many games to skip looking for the requested task-type.")
    args = ap.parse_args()

    from inferencesh import inference
    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    from skillos.executor.executor import InfshExecutor, _parse_action
    from skillos.utils.infsh_auth import resolve_infsh_api_key
    from skillos.utils.infsh_client import run_task_resilient
    from scripts.eval_alfworld import extract_task_description, classify_task

    APP = "openrouter/qwen3-8b"
    env = make_alfworld_env(SPLIT_MAP["valid_seen"], batch_size=1)

    # Skip to a game of the requested task type.
    obs = infos = None
    for _ in range(args.max_resets):
        obs, infos = env.reset()
        gamefile = (infos.get("extra.gamefile") or [""])[0]
        if classify_task(gamefile).lower() == args.task_type.lower():
            break
    else:
        print(f"Could not find a {args.task_type} game in {args.max_resets} resets.", file=sys.stderr)
        return 1

    observation = obs[0]
    admissible = infos.get("admissible_commands", [[]])[0]
    task = extract_task_description(observation)
    gamefile = (infos.get("extra.gamefile") or [""])[0]

    ex = InfshExecutor(app=APP, temperature=args.temperature, top_p=args.top_p,
                       max_tokens=args.max_tokens, reasoning_effort=args.reasoning)
    client = inference(api_key=resolve_infsh_api_key(None))

    print("=" * 78)
    print(f"TASK: {task}")
    print(f"type={classify_task(gamefile)}  game={os.path.basename(os.path.dirname(os.path.dirname(gamefile)))}")
    print(f"decode: t={args.temperature} top_p={args.top_p} reasoning={args.reasoning} max_tokens={args.max_tokens}")
    print("=" * 78)

    history = []
    offgrammar = 0
    success = False
    for step in range(args.max_steps):
        prompt = ex._build_prompt(task, observation, admissible, step,
                                  "\n".join(history[-3:]),
                                  "No relevant skills found.", 3)
        payload = {"text": prompt, "temperature": args.temperature,
                   "top_p": args.top_p, "max_tokens": args.max_tokens,
                   "context_size": 32768, "reasoning_effort": args.reasoning}
        params = {"app": APP, "infra": "cloud", "variant": "default", "input": payload}
        res = run_task_resilient(client, params)
        out = (res or {}).get("output") or {}
        resp = out.get("response") or ""
        reas = out.get("reasoning") or ""
        m = ACTION_RE.search(resp) or ACTION_RE.search(reas)
        raw_action = m.group(1).strip() if m else "(no <action> tag)"
        executed = _parse_action(resp or reas, admissible)
        raw_admissible = m is not None and raw_action in admissible
        coerced = (raw_action != executed)
        if coerced:
            offgrammar += 1

        print(f"\n--- step {step+1}  (#admissible={len(admissible)}) ---")
        print(f"  obs: {observation[:240].strip()!r}")
        print(f"  RAW <action>:  {raw_action!r}   admissible? {raw_admissible}")
        print(f"  EXECUTED:      {executed!r}" + ("   <-- COERCED (off-grammar!)" if coerced else ""))
        if reas:
            print(f"  reasoning tail: ...{reas[-220:].strip()!r}")
        # show a few admissible to judge grammar mismatch
        interesting = [a for a in admissible if any(
            k in a for k in ("heat", "cool", "clean", "put", "take", "open", "microwave", "fridge", "sink"))]
        if interesting:
            print(f"  notable admissible: {interesting[:6]}")

        obs_n, scores, dones, infos = env.step([executed])
        observation = obs_n[0]
        admissible = infos.get("admissible_commands", [[]])[0]
        history.append(f"ACTION: {executed}\nOBSERVATION: {observation}")
        if dones[0]:
            success = scores[0] > 0
            print(f"\n=== EPISODE DONE at step {step+1}: success={success} ===")
            break
    else:
        print(f"\n=== HIT STEP CAP {args.max_steps} without completing ===")

    print("\n" + "=" * 78)
    print(f"SUMMARY: success={success}  steps={step+1}  "
          f"off-grammar/coerced steps = {offgrammar}/{step+1} "
          f"({100*offgrammar/(step+1):.0f}%)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
