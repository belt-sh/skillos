"""Probe: feed a portion of a Claude Code chat transcript to the trained curator
and print what skills it proposes.

The curator (ckpt30) was trained on ALFWorld (ACTION/OBSERVATION) trajectories.
This linearizes a session .jsonl into the same shape — assistant `tool_use` ->
ACTION, the following `tool_result` -> OBSERVATION — so the model sees its
training format on out-of-domain (software-engineering) input.

Usage:
  python -m scripts.probe_curator_on_chat \
    --transcript ~/.claude/projects/-home-ubuntu-skillos/<id>.jsonl \
    --checkpoint output/alfworld-8xh100-algo1-v8-lora-kl/checkpoint-30 \
    --max-steps 8 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                out.append(c.get("text", ""))
            elif c.get("type") == "tool_result":
                inner = c.get("content")
                out.append(_text_of(inner) if not isinstance(inner, str) else inner)
        return "\n".join(out)
    return ""


def linearize(path: str, max_steps: int):
    """Return (task_description, [ {step, action, observation} ]) from a session."""
    task = None
    steps = []
    pending = None  # (tool_name, args_str)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") not in ("user", "assistant"):
            continue
        if o.get("isMeta") or o.get("isSidechain"):
            continue
        msg = o.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")

        if role == "user" and task is None:
            t = _text_of(content).strip()
            # skip system-injected / command / caveat noise; want a real ask
            if t and not t.startswith("<") and "command-name" not in t \
                    and "session is being continued" not in t:
                task = t[:600]

        if role == "assistant" and isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    args = c.get("input", {})
                    arg_str = json.dumps(args)[:200]
                    pending = (c.get("name", "?"), arg_str)
                    break
        elif role == "user" and pending is not None:
            obs = _text_of(content).strip()[:300] or "(no output)"
            steps.append({
                "step": len(steps) + 1,
                "action": f"{pending[0]}({pending[1]})",
                "observation": obs,
            })
            pending = None
            if len(steps) >= max_steps:
                break
    return task or "(unknown task)", steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--checkpoint",
                    default="output/alfworld-8xh100-algo1-v8-lora-kl/checkpoint-30")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--result", default="Success",
                    help="Outcome label fed to the curator (Success/Failure).")
    args = ap.parse_args()

    task, steps = linearize(args.transcript, args.max_steps)
    if not steps:
        print("No tool_use/tool_result steps found in transcript.", file=sys.stderr)
        return 1

    traj_text = "\n".join(
        f"Step {s['step']}: ACTION: {s['action']}\n        OBSERVATION: {s['observation']}"
        for s in steps
    )

    print("=" * 70)
    print(f"TASK (from transcript): {task[:200]}")
    print(f"PORTION FED: {len(steps)} steps")
    print("=" * 70)
    print(traj_text[:1500])
    print("=" * 70)

    from scripts.eval_streaming_curation import CuratorInference, TOOLS_SCHEMA
    ci = CuratorInference(args.checkpoint, device=args.device,
                          temperature=args.temperature, enable_thinking=False)

    user = ci._template.format(
        task_description=task, past_skills="(none)",
        agent_trajectory=traj_text, result=args.result,
    )
    messages = [{"role": "system", "content": ci._system},
                {"role": "user", "content": user}]
    enc = ci.tok.apply_chat_template(
        messages, tools=TOOLS_SCHEMA, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, enable_thinking=False)
    input_ids = enc["input_ids"].to(ci.model.device)
    attn = enc.get("attention_mask")
    attn = attn.to(ci.model.device) if attn is not None else None
    gen_kwargs = dict(max_new_tokens=ci.max_new_tokens, pad_token_id=ci.tok.eos_token_id,
                      do_sample=args.temperature > 0, temperature=args.temperature)
    if attn is not None:
        gen_kwargs["attention_mask"] = attn
    import torch
    with torch.inference_mode():
        out = ci.model.generate(input_ids, **gen_kwargs)
    gen_ids = out[0, input_ids.shape[1]:].tolist()

    raw = ci.tok.decode(gen_ids, skip_special_tokens=False)
    print("\n========== RAW CURATOR OUTPUT ==========")
    print(raw)
    print("\n========== PARSED SKILL OPS ==========")
    parsed = ci._parse_response(ci.tok, gen_ids)
    for tc in parsed.get("tool_calls") or []:
        fn = tc.get("function") or {}
        print(f"- {fn.get('name')}: {json.dumps(fn.get('arguments'), indent=2)[:800]}")
    if not (parsed.get("tool_calls")):
        print("(no tool calls parsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
