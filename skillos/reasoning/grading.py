"""Answer extraction + grading for the reasoning benchmarks.

AIME  — integer answer 0..999. Prefer the last \\boxed{...} content, fall back
        to the last standalone integer in the response.
GPQA — one of A/B/C/D. Prefer the last \\boxed{...} letter, fall back to the
        last standalone A-D letter in the response.
"""

from __future__ import annotations

import re

_BOX_RE = re.compile(r"\\boxed\s*{\s*([^{}]*?)\s*}")
_INT_RE = re.compile(r"(?<![\w.])(-?\d+)(?![\w.])")
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def _last(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def extract_aime(response: str) -> str | None:
    boxed = _last(_BOX_RE, response)
    if boxed:
        m = _INT_RE.search(boxed)
        if m:
            return str(int(m.group(1)))
    m = _last(_INT_RE, response)
    return str(int(m)) if m is not None else None


def extract_gpqa(response: str) -> str | None:
    boxed = _last(_BOX_RE, response)
    if boxed and (m := _LETTER_RE.search(boxed)):
        return m.group(1)
    return _last(_LETTER_RE, response)


def grade(response: str, gold: str, kind: str) -> tuple[bool, str | None]:
    pred = extract_aime(response) if kind == "aime" else extract_gpqa(response)
    if pred is None:
        return False, None
    if kind == "aime":
        try:
            return int(pred) == int(gold), pred
        except ValueError:
            return False, pred
    return pred.upper() == gold.upper(), pred
