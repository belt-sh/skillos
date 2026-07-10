"""Reasoning-benchmark evaluator — paper §4.1 / Table 2.

Two modes, per-problem JSONL output:

  --mode no_memory    empty repo, curator never invoked (baseline)
  --mode closed_loop  streaming curation with a curator checkpoint

The executor is inference.sh Qwen3-8B (same app as the ALFWorld executor).
Datasets: AIME24, AIME25, GPQA-Diamond. GPQA is gated — set HF_TOKEN or run
`huggingface-cli login` first, or pass --dataset aime.

JSONL row: {id, kind, correct(bool), pred, gold, response, dataset, n_skills}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillos.reasoning.datasets import load
from skillos.reasoning.grading import grade
from skillos.reasoning.prompts import build_messages
from skillos.skills.repo import SkillRepo


def _messages_to_text(msgs: list[dict]) -> str:
    parts = []
    for m in msgs:
        role = m["role"].upper()
        parts.append(f"<{role}>\n{m['content']}\n</{role}>")
    return "\n\n".join(parts)


def _call_infsh(app: str, messages: list[dict], max_tokens: int,
                temperature: float, top_p: float, reasoning: str) -> str:
    from inferencesh import inference
    from skillos.utils.infsh_auth import resolve_infsh_api_key
    client = inference(api_key=resolve_infsh_api_key())
    r = client.tasks.run({
        "app": app, "infra": "cloud", "variant": "default",
        "input": {
            "text": _messages_to_text(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning,
        },
    })
    out = (r or {}).get("output") or {}
    return (out.get("response") or "").strip()


def solve_one(row: dict, repo: SkillRepo, app: str, max_tokens: int,
              temperature: float, top_p: float, reasoning: str) -> dict:
    skills = repo.retrieve(row["problem"], top_k=5) if len(repo) else []
    past = repo.format_skills(skills) if skills else ""
    messages = build_messages(row["problem"], past, row["kind"])
    t0 = time.time()
    try:
        resp = _call_infsh(app, messages, max_tokens, temperature, top_p, reasoning)
    except Exception as e:
        return {**{k: row[k] for k in ("id", "kind", "answer")},
                "gold": row["answer"], "correct": False, "pred": None,
                "response": None, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "n_skills": len(skills), "wall_s": time.time() - t0}
    ok, pred = grade(resp, row["answer"], row["kind"])
    return {"id": row["id"], "kind": row["kind"], "correct": ok,
            "pred": pred, "gold": row["answer"], "response": resp,
            "n_skills": len(skills), "wall_s": time.time() - t0}


def run_no_memory(rows: list[dict], out_path: str, app: str, max_tokens: int,
                  temperature: float, top_p: float, reasoning: str,
                  parallel: int) -> None:
    repo = SkillRepo()
    lock = threading.Lock()
    done = 0
    n = len(rows)
    with open(out_path, "w") as f, ThreadPoolExecutor(parallel) as pool:
        futs = [pool.submit(solve_one, r, repo, app, max_tokens,
                            temperature, top_p, reasoning) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done += 1
                marker = "OK" if rec["correct"] else "XX"
                print(f"[{done:4d}/{n}] {rec['id']:<15s} {marker} pred={rec['pred']!s:<6s} "
                      f"gold={rec['gold']:<4s} {rec['wall_s']:.1f}s", flush=True)


def summarize(out_path: str) -> None:
    rows = [json.loads(l) for l in open(out_path)]
    by_ds: dict[str, list[bool]] = {}
    for r in rows:
        ds = r["id"].split("-")[0]
        by_ds.setdefault(ds, []).append(bool(r["correct"]))
    print()
    print(f"=== {out_path} ===")
    for ds in sorted(by_ds):
        ok = sum(by_ds[ds]); n = len(by_ds[ds])
        print(f"  {ds:<8s} {ok:3d}/{n} = {100*ok/n:5.1f}%")
    overall_ok = sum(1 for r in rows if r["correct"])
    print(f"  TOTAL:   {overall_ok:3d}/{len(rows)} = {100*overall_ok/len(rows):5.1f}%")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["no_memory", "closed_loop"], required=True)
    p.add_argument("--dataset", default="aime",
                   help="'aime' (AIME24+25), 'aime24', 'aime25', 'gpqa', 'all'")
    p.add_argument("--executor", default="infsh", choices=["infsh"])
    p.add_argument("--executor-app", default="openrouter/qwen3-8b")
    p.add_argument("--curator-checkpoint", default=None,
                   help="required for closed_loop (not yet implemented in MVP)")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--reasoning", default="medium")
    p.add_argument("--parallel", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.dataset == "aime":
        names = ["aime24", "aime25"]
    elif args.dataset == "all":
        names = ["aime24", "aime25", "gpqa"]
    else:
        names = [args.dataset]
    rows: list[dict] = []
    for n in names:
        try:
            rows.extend(load(n))
        except Exception as e:
            print(f"[eval_reasoning] SKIP {n}: {type(e).__name__}: {e}",
                  file=sys.stderr)
    if not rows:
        print("no rows loaded", file=sys.stderr); return 1
    print(f"[eval_reasoning] mode={args.mode} datasets={names} n={len(rows)} "
          f"parallel={args.parallel} out={args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.mode == "no_memory":
        run_no_memory(rows, args.out, args.executor_app, args.max_tokens,
                      args.temperature, args.top_p, args.reasoning, args.parallel)
    else:
        print("closed_loop not implemented in MVP; blocked on GPUs anyway "
              "(seed-3 is training). Coming next.", file=sys.stderr)
        return 2

    summarize(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
