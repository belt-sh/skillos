"""Content quality judge — evaluates curator output for r_cnt reward.

Uses an external model (Qwen3-32B in paper) to judge whether curated skills
are abstract, reusable, actionable, and faithful.

For single-GPU training, this can run as a separate inference call or be
approximated with rule-based heuristics.
"""

from __future__ import annotations

import json
import re

from skillos.curator.prompts import CONTENT_QUALITY_JUDGE


def judge_skill_quality_heuristic(content: str) -> float:
    """Rule-based approximation of content quality judge.

    Checks for the key criteria without needing an external model.
    Good enough for pipeline validation; replace with LLM judge for real training.
    """
    score = 0.0
    total = 4.0

    # 1. Has valid frontmatter (basic format check)
    if re.match(r"^---\s*\n.*?name:.*?\n.*?description:.*?\n---", content, re.DOTALL):
        score += 1.0

    # 2. Has workflow/structure (not just raw text)
    if re.search(r"^#\s+\w", content, re.MULTILINE):
        score += 1.0

    # 3. Conciseness (not a raw trajectory dump)
    word_count = len(content.split())
    if 20 < word_count < 500:
        score += 1.0

    # 4. Abstraction (no specific IDs/numbers dominating)
    digit_ratio = sum(c.isdigit() for c in content) / max(len(content), 1)
    if digit_ratio < 0.1:
        score += 1.0

    return score / total


def format_judge_prompt(skill_content: str) -> str:
    """Format the content quality judge prompt for LLM evaluation."""
    return CONTENT_QUALITY_JUDGE.format(content=skill_content)


def parse_judge_response(response: str) -> tuple[bool, list[str], str]:
    """Parse the judge's JSON response.

    Returns: (valid, issues, explanation)
    """
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*?\})", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            return (
                result.get("VALID", False),
                result.get("ISSUES", []),
                result.get("EXPLANATION", ""),
            )
    except (json.JSONDecodeError, AttributeError):
        pass
    return False, ["Failed to parse judge response"], ""
