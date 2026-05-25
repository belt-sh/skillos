"""Audit the infsh Qwen3-8B executor's RAW output on a real ALFWorld step.

Why: the no-memory baseline lands ~30% vs the paper's 47.9%, and 70% of games
flail to the 30-step cap — the signature of the executor failing to emit a
parseable <action> tag. The paper's executor decode config comes from GiGPO
(verl-agent run_alfworld.sh): temp 0.4, max_response 512, NON-reasoning model
emitting prompt-CoT (<think>…</think><action>…</action>) in the response.

We're running Qwen3 with reasoning_effort=medium + 8192 tokens. The reasoning
flag is known-unreliable on openrouter/qwen3 apps, so we don't trust it — we
pull a real generation and LOOK at what each config actually returns.

Run: source .venv/bin/activate && python scripts/debug_executor_audit.py
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

from inferencesh import inference
from skillos.envs.config import make_alfworld_env, SPLIT_MAP
from skillos.executor.executor import InfshExecutor, _parse_action
from skillos.utils.infsh_auth import resolve_infsh_api_key
from skillos.utils.infsh_client import run_task_resilient
from scripts.eval_alfworld import extract_task_description

APP = "openrouter/qwen3-8b"


def main():
    # One real ALFWorld observation (no-memory: empty skills).
    env = make_alfworld_env(SPLIT_MAP["valid_seen"], batch_size=1)
    obs, infos = env.reset()
    observation = obs[0]
    admissible = infos.get("admissible_commands", [[]])[0]
    task = extract_task_description(observation)
    print(f"TASK: {task}")
    print(f"#admissible actions: {len(admissible)}  e.g. {admissible[:5]}\n")

    ex = InfshExecutor(app=APP)  # only used to build the prompt identically
    prompt = ex._build_prompt(task, observation, admissible, 0, "",
                              "No relevant skills found.", 3)
    client = inference(api_key=resolve_infsh_api_key(None))

    def probe(label: str, extra: dict):
        payload = {
            "text": prompt,
            "temperature": extra.get("temperature", 0.4),
            "top_p": extra.get("top_p", 1.0),
            "max_tokens": extra.get("max_tokens", 512),
            "context_size": extra.get("context_size", 4096),
        }
        for k in ("reasoning_effort", "reasoning_max_tokens", "reasoning_exclude"):
            if k in extra:
                payload[k] = extra[k]
        params = {"app": APP, "infra": "cloud", "variant": "default", "input": payload}
        res = run_task_resilient(client, params)
        out = (res or {}).get("output") or {}
        resp = (out.get("response") or "") if isinstance(out, dict) else ""
        reas = (out.get("reasoning") or "") if isinstance(out, dict) else ""
        m = re.search(r"<action>\s*(.*?)\s*</action>", resp, re.DOTALL)
        print(f"=== {label} ===")
        print(f"  output keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")
        print(f"  response len={len(resp)}  reasoning len={len(reas)}")
        print(f"  response[:500]={resp[:500]!r}")
        if reas:
            print(f"  reasoning[:200]={reas[:200]!r}")
        print(f"  <action> tag in response? {bool(m)} -> {m.group(1).strip() if m else None}")
        print(f"  _parse_action result: {_parse_action(resp or reas, admissible)!r}")
        print()

    probe("A current (reasoning=medium, 8192, t0.6, tp0.95)",
          dict(temperature=0.6, top_p=0.95, max_tokens=8192,
               context_size=32768, reasoning_effort="medium"))
    probe("B gigpo-match (reasoning=none, 512, t0.4, tp1.0)",
          dict(temperature=0.4, top_p=1.0, max_tokens=512,
               context_size=4096, reasoning_effort="none"))
    probe("C force-off (none + rmt=0 + rexclude, 512, t0.4)",
          dict(temperature=0.4, top_p=1.0, max_tokens=512, context_size=4096,
               reasoning_effort="none", reasoning_max_tokens=0,
               reasoning_exclude=True))


if __name__ == "__main__":
    main()
