"""Composite reward function from SkillOS paper (Eq. 1).

r = r_task + λ_f * r_fc + λ_u * r_cnt + λ_c * r_comp

Paper weights: λ_f=1.0, λ_u=0.1, λ_c=0.05
"""

from __future__ import annotations


def reward_task(env_rewards: list[float], skip_first: bool = True) -> float:
    """Average success rate over tasks in a group, skipping first (empty repo).

    r_task = 1/(|G|-1) * sum(success_i for i=2..|G|)
    """
    if not env_rewards:
        return 0.0
    rewards = env_rewards[1:] if skip_first and len(env_rewards) > 1 else env_rewards
    return sum(rewards) / max(len(rewards), 1)


def reward_function_call(ops: list[dict], skill_repo=None) -> float:
    """Fraction of curator function calls that are valid and execute successfully.

    r_fc = 1/|G| * sum(Valid(c_i))

    Each op dict has: {"name": str, "arguments": dict, "valid": bool}
    """
    if not ops:
        return 0.0
    valid = sum(1 for op in ops if op.get("valid", False))
    return valid / len(ops)


def reward_compression(repo_tokens: int, input_tokens: int) -> float:
    """Compression reward. Penalizes large skill repos relative to input.

    r_comp = 1 - |S_i| / |χ_i|
    """
    if input_tokens == 0:
        return 1.0
    return max(0.0, 1.0 - repo_tokens / input_tokens)



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
