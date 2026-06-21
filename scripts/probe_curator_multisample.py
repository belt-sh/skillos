"""Multi-sample A/B/C probe: compare base vs trained curator checkpoints on the
same chat-transcript slice, N samples each, with aggregate stats so granularity
differences can be separated from sampling noise.

Usage:
  python -m scripts.probe_curator_multisample \
    --transcript <session.jsonl> \
    --checkpoints "vanilla=Qwen/Qwen3-8B,ckpt30=output/.../checkpoint-30,ckpt60=output/.../checkpoint-60" \
    --max-steps 8 --num-samples 8 --temperature 0.7 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from scripts.probe_curator_on_chat import linearize


def _summarize(tok, model, parse_response, system, template, tools, user_text,
               n: int, temperature: float, max_new_tokens: int):
    import torch
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_text}]
    enc = tok.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, enable_thinking=False)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask")
    attn = attn.to(model.device) if attn is not None else None
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                      do_sample=True, temperature=temperature,
                      num_return_sequences=n)
    if attn is not None:
        gen_kwargs["attention_mask"] = attn
    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)
    plen = input_ids.shape[1]

    per_sample = []   # (n_valid_skills, n_format_ok, merged_ls_wc, names)
    name_freq = {}
    for s in range(out.shape[0]):
        gen_ids = out[s, plen:].tolist()
        parsed = parse_response(tok, gen_ids)
        calls = [tc for tc in (parsed.get("tool_calls") or [])
                 if (tc.get("function") or {}).get("name") in
                 ("new_skill_insert", "skill_update", "skill_delete")]
        names = []
        fmt_ok = 0
        merged = False
        for tc in calls:
            args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            nm = args.get("skill_name", "")
            content = args.get("content", "") or ""
            names.append(nm)
            if content.lstrip().startswith("---"):
                fmt_ok += 1
            low = (nm + " " + content).lower()
            if "ls" in low and "wc" in low:
                merged = True
        for nm in names:
            name_freq[nm] = name_freq.get(nm, 0) + 1
        per_sample.append((len(calls), fmt_ok, merged, names))
    return per_sample, name_freq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="comma list of label=path")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    args = ap.parse_args()

    task, steps = linearize(args.transcript, args.max_steps)
    traj_text = "\n".join(
        f"Step {s['step']}: ACTION: {s['action']}\n        OBSERVATION: {s['observation']}"
        for s in steps)

    from scripts.eval_streaming_curation import TOOLS_SCHEMA
    from skillos.curator.prompts import CURATOR_SYSTEM, CURATOR_INPUT_TEMPLATE
    from trl.chat_template_utils import add_response_schema, parse_response
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    user_text = CURATOR_INPUT_TEMPLATE.format(
        task_description=task, past_skills="(none)",
        agent_trajectory=traj_text, result="Success")

    print("=" * 72)
    print(f"TASK: {task[:120]}")
    print(f"slice={len(steps)} steps | N={args.num_samples} samples/arm | T={args.temperature}")
    print("=" * 72)

    arms = []
    for spec in args.checkpoints.split(","):
        label, path = spec.split("=", 1)
        arms.append((label.strip(), path.strip()))

    results = {}
    for label, path in arms:
        print(f"\n[load] {label} <- {path}", flush=True)
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        add_response_schema(tok)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=args.device,
            trust_remote_code=True)
        model.eval()
        per_sample, name_freq = _summarize(
            tok, model, parse_response, CURATOR_SYSTEM, CURATOR_INPUT_TEMPLATE,
            TOOLS_SCHEMA, user_text, args.num_samples, args.temperature,
            args.max_new_tokens)
        results[label] = (per_sample, name_freq)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    print(f"{'arm':<10} {'skills/sample min/med/max/mean':<32} {'fmt_ok%':>8} {'merged%':>8}")
    for label, _ in arms:
        per_sample, _ = results[label]
        counts = [p[0] for p in per_sample]
        total_skills = sum(counts)
        fmt = sum(p[1] for p in per_sample)
        merged = sum(1 for p in per_sample if p[2])
        med = statistics.median(counts) if counts else 0
        mean = statistics.mean(counts) if counts else 0
        fmt_pct = 100.0 * fmt / total_skills if total_skills else 0.0
        merged_pct = 100.0 * merged / len(per_sample) if per_sample else 0.0
        dist = f"{min(counts)}/{med}/{max(counts)}/{mean:.1f}"
        print(f"{label:<10} {dist:<32} {fmt_pct:>7.0f}% {merged_pct:>7.0f}%")

    print("\n--- skill-name frequency per arm (top 12) ---")
    for label, _ in arms:
        _, name_freq = results[label]
        top = sorted(name_freq.items(), key=lambda kv: -kv[1])[:12]
        print(f"\n[{label}]")
        for nm, c in top:
            print(f"  {c:>2}x  {nm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
