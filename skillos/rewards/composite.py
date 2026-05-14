"""Composite reward function from SkillOS paper (Eq. 1).

r = r_task + λ_f * r_fc + λ_u * r_cnt + λ_c * r_comp

Paper weights: λ_f=1.0, λ_u=0.1, λ_c=0.05
"""

from __future__ import annotations

import json
import re


def reward_task(env_rewards: list[float], skip_first: bool = True) -> float:
    """Average success rate over tasks in a group, skipping first (empty repo).

    r_task = 1/(|G|-1) * sum(success_i for i=2..|G|)
    """
    if not env_rewards:
        return 0.0
    rewards = env_rewards[1:] if skip_first and len(env_rewards) > 1 else env_rewards
    return sum(rewards) / max(len(rewards), 1)


def reward_function_call(tool_calls: list[dict], skill_repo) -> float:
    """Fraction of curator function calls that are valid and execute successfully.

    r_fc = 1/|G| * sum(Valid(c_i))
    """
    if not tool_calls:
        return 0.0

    valid = 0
    total = 0
    for call in tool_calls:
        total += 1
        fn_name = call.get("name", "")
        args = call.get("arguments", {})

        if fn_name == "new_skill_insert":
            if "skill_name" in args and "content" in args:
                # Check content has valid frontmatter
                if "---" in args["content"]:
                    valid += 1
        elif fn_name == "skill_update":
            if "skill_name" in args:
                # Check skill exists in repo
                if skill_repo and args["skill_name"] in skill_repo.skills:
                    valid += 1
        elif fn_name == "skill_delete":
            if "skill_name" in args:
                if skill_repo and args["skill_name"] in skill_repo.skills:
                    valid += 1
        # Unknown function names count as invalid

    return valid / max(total, 1)


def reward_compression(repo_tokens: int, input_tokens: int) -> float:
    """Compression reward — penalizes large skill repos relative to input.

    r_comp = 1 - |S_i| / |χ_i|
    """
    if input_tokens == 0:
        return 1.0
    return max(0.0, 1.0 - repo_tokens / input_tokens)


def reward_content_quality(judge_response: str) -> float:
    """Parse judge response to get content quality score.

    r_cnt = 1.0 if VALID else 0.0 (binary from judge)
    """
    try:
        # Extract JSON from response (may be in code block)
        json_match = re.search(r"\{[^}]+\}", judge_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return 1.0 if result.get("VALID", False) else 0.0
    except (json.JSONDecodeError, AttributeError):
        pass
    return 0.0


def composite_reward(
    r_task: float,
    r_fc: float,
    r_cnt: float,
    r_comp: float,
    lambda_f: float = 1.0,
    lambda_u: float = 0.1,
    lambda_c: float = 0.05,
) -> float:
    """Composite reward from Eq. 1 of the paper.

    r = r_task + λ_f * r_fc + λ_u * r_cnt + λ_c * r_comp
    """
    return r_task + lambda_f * r_fc + lambda_u * r_cnt + lambda_c * r_comp
