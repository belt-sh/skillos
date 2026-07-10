"""Paper §4.1 reasoning datasets: AIME24 / AIME25 / GPQA-Diamond.

Unified row format: {"id", "problem", "answer", "kind"} where `kind` is
"aime" (integer answer 0..999) or "gpqa" (one of A/B/C/D). GPQA requires
`huggingface-cli login` (gated dataset). Both AIME sets are open.
"""

from __future__ import annotations

from typing import Iterable

_LOADED: dict[str, list[dict]] = {}


def _aime24() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [
        {"id": f"AIME24-{r['ID']}", "problem": r["Problem"],
         "answer": str(r["Answer"]).strip(), "kind": "aime"}
        for r in ds
    ]


def _aime25() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("MathArena/aime_2025", split="train")
    return [
        {"id": f"AIME25-{r['problem_idx']}", "problem": r["problem"],
         "answer": str(r["answer"]).strip(), "kind": "aime"}
        for r in ds
    ]


def _gpqa_diamond() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    out: list[dict] = []
    for i, r in enumerate(ds):
        # GPQA-Diamond fields: Question, Correct Answer, Incorrect Answer 1..3.
        # We stringify as A/B/C/D with the correct one at a fixed slot per
        # problem id for reproducibility.
        letters = ["A", "B", "C", "D"]
        # Deterministic shuffle keyed by problem index so same question always
        # has the same answer letter across runs.
        rng_order = [(i + k) % 4 for k in range(4)]
        options = [None, None, None, None]
        answer_slot = rng_order[0]
        options[answer_slot] = r["Correct Answer"]
        wrong = [r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                 r["Incorrect Answer 3"]]
        for slot, choice in zip(rng_order[1:], wrong):
            options[slot] = choice
        body = r["Question"] + "\n\n" + "\n".join(
            f"{L}. {opt}" for L, opt in zip(letters, options))
        out.append({"id": f"GPQA-{i:03d}", "problem": body,
                    "answer": letters[answer_slot], "kind": "gpqa"})
    return out


_LOADERS = {"aime24": _aime24, "aime25": _aime25, "gpqa": _gpqa_diamond}


def load(name: str) -> list[dict]:
    """Cache-once loader. name in {'aime24','aime25','gpqa'}."""
    if name not in _LOADED:
        _LOADED[name] = _LOADERS[name]()
    return list(_LOADED[name])


def load_all(names: Iterable[str] = ("aime24", "aime25", "gpqa")) -> list[dict]:
    rows: list[dict] = []
    for n in names:
        try:
            rows.extend(load(n))
        except Exception as e:
            print(f"[reasoning.datasets] SKIP {n}: {type(e).__name__}: {e}")
    return rows
