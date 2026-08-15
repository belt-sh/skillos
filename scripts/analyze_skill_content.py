"""Quantify what a curator actually wrote.

A curator arm's success rate says whether it helped. It does not say why, and
without a mechanism a positive result is an anecdote. This script characterises
a dumped repository (`*.repo.md`, written by scripts.eval_streaming_curation)
along the dimensions a reviewer would ask about:

- **Size and growth**: skills, tokens, tokens per skill.
- **Domain vocabulary**: how much of the text is ALFWorld action grammar
  (`go to`, `take ... from`, `heat ... with`) versus generic advice. This is the
  key measurement for the cross-domain result: a curator trained on mathematics
  that helps an embodied agent either learned the embodied vocabulary from its
  probe rollouts, or is helping through something other than domain content.
- **Actionability**: share of lines that are imperative procedure versus
  narration.
- **Retrievability**: for each ALFWorld task type, whether BM25 over this repo
  surfaces a skill whose text mentions that task type's defining verb. A repo
  can hold the right knowledge and still never retrieve it.
- **Redundancy**: pairwise token-set Jaccard between skills, to detect a repo
  that is one idea restated N times.

Usage:
    python -m scripts.analyze_skill_content output/reeval/.../arm.repo.md [...]
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# The environment's complete action grammar. Anything outside this is rejected by
# ALFWorld, so a skill's value to the executor is bounded by how much of this it
# uses correctly.
ACTION_VERBS = ["go to", "open", "close", "take", "put", "clean", "heat", "cool",
                "use", "examine", "look"]

# Defining verb per task type: a skill is "relevant" to the type if it mentions it.
TYPE_VERB = {"Clean": "clean", "Heat": "heat", "Cool": "cool",
             "Look": "use", "Pick": "take", "Pick2": "two"}

IMPERATIVE = re.compile(
    # Allow list markers, numbering, and inline-code/quote marks before the verb:
    # markdown procedure lines are frequently written as "1. `take X from Y`".
    r"^\s*(?:[-*\d.)\s]+)?[`'\"*_]*(?:"
    r"go|open|close|take|put|clean|heat|cool|use|examine|look|find|search|check|"
    r"first|then|next|always|never|do not|don't|avoid|make sure|ensure|verify"
    r")\b", re.IGNORECASE)


def parse_repo(path: Path) -> dict[str, str]:
    """Split a repo dump back into {skill_name: text}."""
    text = path.read_text()
    skills: dict[str, str] = {}
    for block in text.split("=" * 70):
        block = block.strip()
        if not block or block.startswith("<!--"):
            continue
        lines = block.splitlines()
        if lines[0].startswith("# "):
            name = lines[0][2:].strip()
            skills[name] = "\n".join(lines[1:]).strip()
        elif skills:  # continuation body after a name-only block
            last = list(skills)[-1]
            skills[last] = (skills[last] + "\n" + block).strip()
    return skills


def tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def analyze(path: Path) -> dict:
    skills = parse_repo(path)
    if not skills:
        return {"path": str(path), "n_skills": 0}

    all_text = "\n".join(skills.values())
    toks = tokens(all_text)
    low = all_text.lower()

    verb_hits = {v: low.count(v) for v in ACTION_VERBS}
    n_verb_mentions = sum(verb_hits.values())

    lines = [l for l in all_text.splitlines() if l.strip()]
    n_imperative = sum(1 for l in lines if IMPERATIVE.match(l))

    # Redundancy: mean pairwise Jaccard over token sets.
    sets = [set(tokens(t)) for t in skills.values()]
    jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    mean_jac = sum(jac) / len(jac) if jac else 0.0

    # Coverage: does any skill mention each task type's defining verb?
    covered = {t: any(v in tokens(s) for s in skills.values())
               for t, v in TYPE_VERB.items()}

    return {
        "path": str(path),
        "n_skills": len(skills),
        "n_tokens": len(toks),
        "tokens_per_skill": len(toks) / len(skills),
        "action_verb_mentions": n_verb_mentions,
        "action_verbs_per_100_tok": n_verb_mentions / max(len(toks), 1) * 100,
        "distinct_action_verbs": sum(1 for v, c in verb_hits.items() if c > 0),
        "imperative_line_share": n_imperative / max(len(lines), 1),
        "mean_pairwise_jaccard": mean_jac,
        "types_covered": sum(covered.values()),
        "coverage_detail": covered,
        "top_terms": [w for w, _ in Counter(
            t for t in toks if len(t) > 3 and t not in _STOP).most_common(12)],
    }


_STOP = {"this", "that", "with", "from", "have", "will", "when", "then", "your",
         "into", "them", "task", "skill", "step", "steps", "name", "description",
         "which", "there", "these", "such", "some", "must", "should", "been",
         "before", "after", "also", "make", "sure", "need", "used", "using"}


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    rows = [analyze(p) for p in paths]

    hdr = (f"{'arm':<34}{'skills':>7}{'tokens':>8}{'tok/sk':>8}"
           f"{'verbs/100':>10}{'nverbs':>7}{'imper':>7}{'jaccard':>8}{'types':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r["n_skills"]:
            print(f"{Path(r['path']).stem:<34}  (empty repo)")
            continue
        print(f"{Path(r['path']).stem:<34}{r['n_skills']:>7}{r['n_tokens']:>8}"
              f"{r['tokens_per_skill']:>8.0f}{r['action_verbs_per_100_tok']:>10.2f}"
              f"{r['distinct_action_verbs']:>7}{r['imperative_line_share']:>7.2f}"
              f"{r['mean_pairwise_jaccard']:>8.2f}{r['types_covered']:>4}/6")
    print("\nverbs/100 = ALFWorld action-grammar mentions per 100 tokens "
          "(domain groundedness)")
    print("nverbs    = how many of the 11 legal verb forms appear at all")
    print("imper     = share of non-blank lines that read as procedure")
    print("jaccard   = mean pairwise token overlap (high = repo restates one idea)")
    print("types     = ALFWorld task types whose defining verb appears anywhere\n")
    for r in rows:
        if r["n_skills"]:
            print(f"{Path(r['path']).stem}: {', '.join(r['top_terms'])}")


if __name__ == "__main__":
    main()
