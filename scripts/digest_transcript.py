#!/usr/bin/env python3
"""Compress a Claude Code session transcript into readable chunks for review.

The raw transcript is 79 MB, almost all of it tool-result payloads (training
logs, eval JSONLs, file dumps). What matters for auditing how the research went
is the PROSE: what the user asked, what I claimed, and what commands I ran.

Output goes to /tmp by default and MUST stay there: transcripts can contain
GPQA problem text and credentials pasted in chat, neither of which may reach a
git-tracked or web-visible file.
"""
import json, sys, pathlib, re

SRC = pathlib.Path(sys.argv[1])
OUTDIR = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/skillos_digest")
NCHUNKS = int(sys.argv[3]) if len(sys.argv) > 3 else 8

OUTDIR.mkdir(parents=True, exist_ok=True)

# Redact obvious secrets even though this stays in /tmp — cheap insurance
# against a subagent quoting one into a report.
SECRETS = [
    (re.compile(r"\b[0-9a-f]{40}\b"), "<REDACTED-40HEX>"),          # wandb keys
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), "<REDACTED-SK>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "<REDACTED-HF>"),
]

def scrub(s: str) -> str:
    for pat, rep in SECRETS:
        s = pat.sub(rep, s)
    return s

def clip(s, n):
    s = scrub(" ".join(str(s).split()))
    return s if len(s) <= n else s[: n // 2] + f" …[{len(s)-n} chars cut]… " + s[-n // 2 :]

entries = []
with SRC.open(errors="replace") as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        if d.get("isSidechain"):
            continue
        msg = d.get("message") or {}
        role = msg.get("role", d["type"])
        content = msg.get("content")
        ts = (d.get("timestamp") or "")[:16]
        parts = []
        if isinstance(content, str):
            parts.append(("text", content))
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    parts.append(("text", b.get("text", "")))
                elif t == "thinking":
                    parts.append(("thinking", b.get("thinking", "")))
                elif t == "tool_use":
                    inp = b.get("input", {})
                    key = next((k for k in ("command", "file_path", "prompt", "pattern") if k in inp), None)
                    parts.append(("tool", f"{b.get('name')}: {clip(inp.get(key, inp), 300)}"))
                elif t == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    parts.append(("result", clip(c or "", 400)))
        if parts:
            entries.append((ts, role, parts))

print(f"{len(entries)} conversational entries", file=sys.stderr)

# Chunk by entry count so each reviewer gets a contiguous slice of history.
size = -(-len(entries) // NCHUNKS)
for ci in range(NCHUNKS):
    sl = entries[ci * size : (ci + 1) * size]
    if not sl:
        continue
    p = OUTDIR / f"chunk_{ci+1:02d}.md"
    with p.open("w") as out:
        out.write(f"# Transcript chunk {ci+1}/{NCHUNKS} — entries {ci*size}..{ci*size+len(sl)-1}\n")
        out.write(f"# Dates: {sl[0][0]} .. {sl[-1][0]}\n\n")
        for i, (ts, role, parts) in enumerate(sl, start=ci * size):
            for kind, val in parts:
                if kind == "text":
                    if not val.strip():
                        continue
                    tag = "USER" if role == "user" else "ME"
                    out.write(f"\n[{i} {ts}] {tag}: {clip(val, 4000)}\n")
                elif kind == "thinking":
                    out.write(f"    (thinking) {clip(val, 700)}\n")
                elif kind == "tool":
                    out.write(f"    $ {clip(val, 300)}\n")
                elif kind == "result":
                    out.write(f"    -> {clip(val, 400)}\n")
    print(f"{p}  {p.stat().st_size/1e6:.1f} MB", file=sys.stderr)
