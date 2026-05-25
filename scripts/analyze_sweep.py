"""Summarize the 2x2 Qwen3-8B config sweep (sweep-{B,C,D,E}-sh{0,1}.jsonl)."""
import glob
import json
from collections import defaultdict

CONFIGS = {
    "B": "think + official (t.6/p.95/k20)",
    "C": "no-think + official (t.7/p.8/k20)",
    "D": "think + anti-loop (pp1.5)",
    "E": "no-think + anti-loop (pp1.5)",
}
TASKS = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]


def load(name):
    rows = []
    for f in sorted(glob.glob(f"output/sweep-{name}-sh*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


print(f"{'cfg':>3}  {'desc':38s}  {'n':>3}  {'SR':>6}  {'steps':>5}  {'cap%':>5}  per-task SR")
print("-" * 110)
# reference A from the prior 140-game baseline run (games 0-71 ~ shards 0-3)
ref = []
for f in sorted(glob.glob("output/eval-nomem-shard[0-3].jsonl")):
    ref += [json.loads(l) for l in open(f)]
for name, rows in [("A", ref)] + [(k, load(k)) for k in CONFIGS]:
    if not rows:
        print(f"{name:>3}  (no data yet)")
        continue
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    steps = sum(r["steps"] for r in rows) / n
    cap = 100 * sum(1 for r in rows if r["steps"] >= 30) / n
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["task_type"]][0] += r["success"]
        by[r["task_type"]][1] += 1
    pt = "  ".join(f"{t}:{by[t][0]}/{by[t][1]}" for t in TASKS if by[t][1])
    desc = "prior 140-run games0-71 (no top_k)" if name == "A" else CONFIGS[name]
    print(f"{name:>3}  {desc:38s}  {n:>3}  {100*ok/n:5.1f}%  {steps:5.1f}  {cap:4.0f}%  {pt}")
print("\npaper no-memory: 47.9% SR, 21.1 steps")
