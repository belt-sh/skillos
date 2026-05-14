"""ALFWorld environment wrapped as TRL environment_factory class.

Follows the TRL OpenEnv pattern: environment class with tool methods,
auto-discovered by GRPOTrainer for multi-turn rollouts.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def _load_alfworld_config() -> dict:
    """Load ALFWorld config."""
    config_path = os.environ.get(
        "ALFWORLD_CONFIG",
        str(Path(__file__).parent.parent.parent / "configs" / "alfworld_env.yaml"),
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_env():
    """Create a fresh ALFWorld TextWorld environment."""
    from alfworld.agents.environment import get_environment

    config = _load_alfworld_config()
    AlfredTWEnv = get_environment("AlfredTWEnv")
    env = AlfredTWEnv(config, train_eval="train")
    env = env.init_env(batch_size=1)
    return env


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
        self._env = None
        self._admissible_actions: list[str] = []

    def reset(self, **kwargs) -> str:
        """Reset environment for a new episode.

        Returns:
            Initial observation with task description and available actions.
        """
        self.reward = 0.0
        self.done = False
        self.steps = 0
        self.trajectory = []

        self._env = _get_env()
        obs, infos = self._env.reset()

        observation = obs[0]
        self._admissible_actions = infos.get("admissible_commands", [[]])[0]

        # Extract task description from first observation
        self.task_description = observation.split("\n")[0] if observation else "Unknown task"

        parts = [f"Task: {self.task_description}"]
        parts.append(f"\nObservation: {observation}")
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
