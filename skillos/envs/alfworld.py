"""ALFWorld environment wrapped as TRL environment_factory class.

Follows the TRL OpenEnv pattern: environment class with tool methods,
auto-discovered by GRPOTrainer for multi-turn rollouts.
"""

from __future__ import annotations

import json
import os
import re

import alfworld.agents.environment as alfworld_env
import yaml

from skillos.skills.repo import SkillRepo


def _load_alfworld_config() -> dict:
    """Load ALFWorld config, using default if not specified."""
    config_path = os.environ.get(
        "ALFWORLD_CONFIG",
        os.path.join(os.path.dirname(alfworld_env.__file__), "..", "configs", "base_config.yaml"),
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


# Shared environment pool — ALFWorld envs are expensive to create.
# GRPOTrainer creates one ALFWorldEnv instance per generation,
# but they all share the underlying ALFWorld environment batch.
_env_pool: list | None = None
_env_config: dict | None = None


def _get_env_pool():
    global _env_pool, _env_config
    if _env_pool is None:
        _env_config = _load_alfworld_config()
        env = alfworld_env.AlfredTWEnv(_env_config, train_eval="train")
        env = env.init_env(batch_size=1)
        _env_pool = [env]
    return _env_pool


class ALFWorldEnv:
    """ALFWorld environment for TRL GRPOTrainer.

    Each instance handles one episode. The trainer calls:
    1. reset(**kwargs) -> initial observation
    2. act(action) -> observation (tool method, auto-discovered)
    3. Reads self.reward after episode
    """

    def __init__(self):
        self.reward = 0.0
        self.done = False
        self.steps = 0
        self.max_steps = 50
        self.trajectory: list[dict] = []
        self.task_description = ""
        self.skill_repo: SkillRepo | None = None
        self._env = None
        self._admissible_actions: list[str] = []

    def reset(self, skill_repo: SkillRepo | None = None, **kwargs) -> str:
        """Reset environment for a new episode.

        Args:
            skill_repo: Optional skill repo for skill-augmented execution.

        Returns:
            Initial observation with task description and available actions.
        """
        self.reward = 0.0
        self.done = False
        self.steps = 0
        self.trajectory = []
        self.skill_repo = skill_repo

        # Get an ALFWorld environment
        envs = _get_env_pool()
        self._env = envs[0]
        obs, infos = self._env.reset()

        observation = obs[0]
        self._admissible_actions = infos.get("admissible_commands", [[]])[0]

        # Extract task description from first observation
        self.task_description = observation.split("\n")[0] if observation else "Unknown task"

        # Build initial prompt with skills if available
        parts = [f"Task: {self.task_description}\n"]

        if self.skill_repo:
            retrieved = self.skill_repo.retrieve(self.task_description, top_k=5)
            if retrieved:
                parts.append("## Relevant Skills")
                parts.append(self.skill_repo.format_skills(retrieved))
                parts.append("")

        parts.append(f"Observation: {observation}")
        parts.append(f"\nAdmissible actions: {', '.join(self._admissible_actions)}")

        return "\n".join(parts)

    def act(self, action: str) -> str:
        """Execute an action in the ALFWorld environment.

        Args:
            action: The action to take (e.g. 'go to counter 1', 'take mug 1', 'heat mug 1')

        Returns:
            The observation after taking the action, including admissible next actions.
        """
        if self.done:
            raise ValueError("Episode finished.")

        if self.steps >= self.max_steps:
            self.done = True
            raise ValueError("Maximum steps reached.")

        # Step the environment
        obs, scores, dones, infos = self._env.step([action])

        observation = obs[0]
        self._admissible_actions = infos.get("admissible_commands", [[]])[0]
        done = dones[0]
        score = scores[0]

        self.steps += 1
        self.trajectory.append({"action": action, "observation": observation})

        if done:
            self.done = True
            self.reward = 1.0 if score > 0 else 0.0

        result = f"Observation: {observation}"
        if not done:
            result += f"\nAdmissible actions: {', '.join(self._admissible_actions)}"

        return result

    def format_trajectory(self) -> str:
        """Format the episode trajectory for the curator."""
        parts = []
        for i, step in enumerate(self.trajectory, 1):
            parts.append(f"Step {i}: ACTION: {step['action']}")
            parts.append(f"        OBSERVATION: {step['observation']}")
        return "\n".join(parts)
