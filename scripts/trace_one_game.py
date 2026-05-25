"""Trace ONE ALFWorld game end-to-end against local vLLM, logging every step's
raw model output + parsed action + observation. Used to diagnose WHY the
multi-step tasks (Heat/Clean/Pick2) loop to the 30-step cap.

    python -m scripts.trace_one_game --task-type Heat --base-url http://localhost:8001/v1
"""
from __future__ import annotations

import argparse
from collections import deque

from scripts.eval_alfworld import classify_task, extract_task_description


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-type", default="Heat")
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-scan", type=int, default=40, help="games to scan for the target type")
    args = p.parse_args()

    from skillos.envs.config import make_alfworld_env, SPLIT_MAP
    from skillos.executor.executor import VLLMExecutor
    from skillos.utils.http import openai_chat

    env = make_alfworld_env(SPLIT_MAP["valid_seen"], batch_size=1)
    ex = VLLMExecutor(base_url=args.base_url, model=args.model, temperature=0.6,
                      max_tokens=8192, top_p=0.95, top_k=20, min_p=0, enable_thinking=True)

    # scan resets until we land on the requested task type
    for _ in range(args.max_scan):
        obs, infos = env.reset()
        gamefile = (infos.get("extra.gamefile") or [""])[0]
        tt = classify_task(gamefile)
        if tt == args.task_type:
            break
    else:
        print(f"no {args.task_type} game found in {args.max_scan} resets")
        return

    observation = obs[0]
    admissible = infos.get("admissible_commands", [[]])[0]
    task = extract_task_description(observation)
    history: deque = deque(maxlen=3)
    print(f"==== {tt} game ====\nTASK: {task}\nGAMEFILE: {gamefile}\n")

    for step in range(args.max_steps):
        prompt = ex._build_prompt(task, observation, admissible, step,
                                  "\n".join(history), "None", ex.history_length)
        raw = openai_chat(args.base_url, args.model, prompt, temperature=0.6,
                          max_tokens=8192, top_p=0.95,
                          extra_body={"top_k": 20, "min_p": 0,
                                      "chat_template_kwargs": {"enable_thinking": True}})
        from skillos.executor.executor import _parse_action
        action = _parse_action(raw, admissible)
        print(f"----- STEP {step+1} -----")
        print(f"OBS: {observation[:300]}")
        print(f"ADMISSIBLE ({len(admissible)}): {admissible[:18]}")
        print(f"RAW MODEL OUTPUT:\n{raw.strip()}")
        print(f">>> PARSED ACTION: {action!r}")
        obs_n, scores, dones, infos = env.step([action])
        observation = obs_n[0]
        admissible = infos.get("admissible_commands", [[]])[0]
        history.append(f"ACTION: {action}\nOBSERVATION: {observation}")
        print(f"RESULT OBS: {observation[:300]}\n")
        if dones[0]:
            print(f"==== DONE step {step+1}, success={scores[0] > 0} ====")
            return
    print(f"==== HIT {args.max_steps}-STEP CAP — TIMEOUT FAILURE ====")


if __name__ == "__main__":
    main()
