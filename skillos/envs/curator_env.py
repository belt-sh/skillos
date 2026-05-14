"""Curator training environment — the environment for the model we actually train.

The SkillOS training loop:
1. Frozen executor solves ALFWorld tasks (inference only)
2. Curator (being trained) sees the trajectory and decides insert/update/delete
3. Skill repo carries across tasks within a group
4. Reward = did curated skills help future tasks?

From the curator's perspective, the "environment" is:
- Observation: executor trajectory + existing skills + task result
- Actions: new_skill_insert / skill_update / skill_delete (tool calls)
- Reward: composite (task outcome + valid ops + content quality + compression)
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from skillos.curator.prompts import CURATOR_INPUT_TEMPLATE
from skillos.executor.executor import Executor, HeuristicExecutor, create_executor
from skillos.rewards.judge import Judge, HeuristicJudge, create_judge
from skillos.skills.repo import SkillRepo


def _load_alfworld_config() -> dict:
    config_path = os.environ.get(
        "ALFWORLD_CONFIG",
        str(Path(__file__).parent.parent.parent / "configs" / "alfworld_env.yaml"),
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


# --- Module-level shared state ---
# These persist across CuratorEnv instances within a training step.
# TRL creates fresh env instances per generation, so we share state here.

_shared_skill_repo = SkillRepo()
_alfworld_env = None
_executor: Executor = HeuristicExecutor()
_judge: Judge = HeuristicJudge()


def configure(executor_config: dict | None = None, judge_config: dict | None = None):
    """Configure the executor and judge backends.

    Call this before training starts. Examples:

        # Pipeline validation (default)
        configure()

        # Local models
        configure(
            executor_config={"type": "local", "model": "Qwen/Qwen3-8B"},
            judge_config={"type": "local", "model": "Qwen/Qwen3-32B"},
        )

        # vLLM servers on dedicated GPUs
        configure(
            executor_config={"type": "vllm", "base_url": "http://localhost:8002/v1"},
            judge_config={"type": "vllm", "base_url": "http://localhost:8001/v1"},
        )

        # Remote API
        configure(
            executor_config={"type": "api", "base_url": "https://api.inference.sh/v1"},
            judge_config={"type": "api", "base_url": "https://api.inference.sh/v1"},
        )
    """
    global _executor, _judge
    if executor_config:
        _executor = create_executor(executor_config)
    if judge_config:
        _judge = create_judge(judge_config)


def reset_shared_state():
    """Reset shared state at the start of a new task group."""
    global _shared_skill_repo, _alfworld_env
    _shared_skill_repo = SkillRepo()
    _alfworld_env = None


def _get_alfworld_env():
    """Get or create ALFWorld environment."""
    global _alfworld_env
    if _alfworld_env is None:
        from alfworld.agents.environment import get_environment
        config = _load_alfworld_config()
        AlfredTWEnv = get_environment("AlfredTWEnv")
        tw_env = AlfredTWEnv(config, train_eval="train")
        _alfworld_env = tw_env.init_env(batch_size=1)
    return _alfworld_env


def _run_frozen_executor(task_description: str, observation: str,
                         admissible_actions: list[str], env,
                         skills_text: str, max_steps: int = 30) -> dict:
    """Run the frozen executor on one ALFWorld task."""
    trajectory = []
    action_history_parts = []
    success = False
    done = False
    step = 0

    while not done and step < max_steps:
        action_history = "\n".join(action_history_parts[-3:]) if action_history_parts else ""

        action = _executor.act(
            task_description=task_description,
            observation=observation,
            admissible_actions=admissible_actions,
            step_count=step,
            action_history=action_history,
            retrieved_skills=skills_text,
        )

        obs, scores, dones, infos = env.step([action])
        observation = obs[0]
        admissible_actions = infos.get("admissible_commands", [[]])[0]
        done = dones[0]
        score = scores[0]
        step += 1

        trajectory.append({"step": step, "action": action, "observation": observation})
        action_history_parts.append(f"ACTION: {action}\nOBSERVATION: {observation}")

        if done:
            success = score > 0

    return {
        "trajectory": trajectory,
        "success": success,
        "steps": step,
        "task_description": task_description,
    }


def _format_trajectory(result: dict) -> str:
    parts = []
    for step in result["trajectory"]:
        parts.append(f"Step {step['step']}: ACTION: {step['action']}")
        parts.append(f"        OBSERVATION: {step['observation']}")
    return "\n".join(parts)


class CuratorEnv:
    """Environment for training the skill curator with TRL GRPOTrainer.

    The curator sees an executor's trajectory and decides how to update
    the skill repo. TRL auto-discovers the tool methods (new_skill_insert,
    skill_update, skill_delete) and exposes them to the model being trained.
    """

    def __init__(self):
        self.reward = 0.0
        self.done = False
        self._executor_result: dict | None = None
        self._ops_applied: list[dict] = []
        self._input_tokens = 0

    def reset(self, **kwargs) -> str:
        """Run frozen executor on an ALFWorld task, return trajectory for curator."""
        self.reward = 0.0
        self.done = False
        self._ops_applied = []

        env = _get_alfworld_env()
        obs, infos = env.reset()
        observation = obs[0]
        admissible_actions = infos.get("admissible_commands", [[]])[0]
        task_description = observation.split("\n")[0] if observation else "Unknown task"

        # Retrieve relevant skills for the executor
        retrieved = _shared_skill_repo.retrieve(task_description, top_k=5)
        skills_text = _shared_skill_repo.format_skills(retrieved)

        # Run frozen executor
        self._executor_result = _run_frozen_executor(
            task_description, observation, admissible_actions, env, skills_text
        )

        # Format curator input
        trajectory_text = _format_trajectory(self._executor_result)
        result_text = "Success" if self._executor_result["success"] else "Failure"

        curator_input = CURATOR_INPUT_TEMPLATE.format(
            task_description=self._executor_result["task_description"],
            past_skills=skills_text,
            agent_trajectory=trajectory_text,
            result=result_text,
        )

        self._input_tokens = len(curator_input.split())
        return curator_input

    def new_skill_insert(self, skill_name: str, content: str) -> str:
        """Create a new skill in the skill repo.

        Args:
            skill_name: The name of the new skill to create.
            content: The markdown content for the new skill, including YAML frontmatter.

        Returns:
            Confirmation message or error.
        """
        success = _shared_skill_repo.insert(skill_name, content)
        self._ops_applied.append({
            "name": "new_skill_insert",
            "arguments": {"skill_name": skill_name, "content": content},
            "valid": success,
        })
        if success:
            return f"Skill '{skill_name}' created. Repo has {len(_shared_skill_repo)} skills."
        return f"Failed to create '{skill_name}'. Already exists or invalid format."

    def skill_update(self, skill_name: str, new_name: str = "", new_content: str = "") -> str:
        """Update an existing skill in the skill repo.

        Args:
            skill_name: The name of the skill to update. Must exactly match an existing skill.
            new_name: Optional new name for the skill.
            new_content: Optional new content to replace the entire skill content.

        Returns:
            Confirmation message or error.
        """
        success = _shared_skill_repo.update(
            skill_name,
            new_name=new_name if new_name else None,
            new_content=new_content if new_content else None,
        )
        self._ops_applied.append({
            "name": "skill_update",
            "arguments": {"skill_name": skill_name},
            "valid": success,
        })
        if success:
            return f"Skill '{new_name or skill_name}' updated."
        return f"Failed to update '{skill_name}'. Does not exist."

    def skill_delete(self, skill_name: str) -> str:
        """Delete an existing skill from the skill repo.

        Args:
            skill_name: The name of the skill to delete.

        Returns:
            Confirmation message or error.
        """
        success = _shared_skill_repo.delete(skill_name)
        self._ops_applied.append({
            "name": "skill_delete",
            "arguments": {"skill_name": skill_name},
            "valid": success,
        })
        if success:
            return f"Skill '{skill_name}' deleted. Repo has {len(_shared_skill_repo)} skills."
        return f"Failed to delete '{skill_name}'. Does not exist."

    def compute_reward(self) -> float:
        """Compute composite reward: r_task + r_fc + r_cnt + r_comp."""
        from skillos.rewards.composite import composite_reward, reward_compression, reward_function_call

        r_task = 1.0 if self._executor_result and self._executor_result["success"] else 0.0
        r_fc = reward_function_call(self._ops_applied)

        # Content quality via pluggable judge
        content_scores = []
        for op in self._ops_applied:
            if not op.get("valid"):
                continue
            content = None
            if op["name"] == "new_skill_insert":
                content = op["arguments"].get("content")
            elif op["name"] == "skill_update":
                content = op["arguments"].get("new_content")
            if content:
                content_scores.append(_judge.score(content))
        r_cnt = sum(content_scores) / len(content_scores) if content_scores else 0.0

        r_comp = reward_compression(_shared_skill_repo.total_tokens(), self._input_tokens)

        self.reward = composite_reward(r_task, r_fc, r_cnt, r_comp)
        return self.reward
