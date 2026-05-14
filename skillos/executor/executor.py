"""Frozen executor — solves ALFWorld tasks using retrieved skills.

Pluggable backends:
- HeuristicExecutor: picks first admissible action (pipeline validation)
- LocalExecutor: runs a local model via transformers (same or separate GPU)
- VLLMExecutor: calls a vLLM server (dedicated inference GPU)
- APIExecutor: calls any OpenAI-compatible API (remote executor)

Paper uses frozen Qwen3-8B as executor. The executor is NEVER trained —
it just generates trajectories that the curator learns from.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from skillos.curator.prompts import ALFWORLD_EXECUTOR


class Executor(ABC):
    """Base interface for frozen executors."""

    @abstractmethod
    def act(self, task_description: str, observation: str, admissible_actions: list[str],
            step_count: int, action_history: str, retrieved_skills: str) -> str:
        ...

    def _build_prompt(self, task_description, observation, admissible_actions,
                      step_count, action_history, retrieved_skills, history_length=3) -> str:
        return ALFWORLD_EXECUTOR.format(
            task_description=task_description,
            retrieved_skills=retrieved_skills or "None",
            step_count=step_count,
            history_length=history_length,
            action_history=action_history or "None",
            current_step=step_count + 1,
            current_observation=observation,
            admissible_actions=", ".join(admissible_actions),
        )


class HeuristicExecutor(Executor):
    """Picks the first admissible action. Fast, zero cost."""

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        return admissible_actions[0] if admissible_actions else "look"


class LocalExecutor(Executor):
    """Frozen executor using a local model via transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "auto",
                 history_length: int = 3):
        self.model_name = model_name
        self._pipeline = None
        self._device = device
        self.history_length = history_length

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self._device,
            torch_dtype="auto",
            max_new_tokens=256,
        )

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        self._load()
        prompt = self._build_prompt(
            task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills, self.history_length,
        )
        result = self._pipeline(prompt, return_full_text=False)[0]["generated_text"]
        return _parse_action(result, admissible_actions)


class VLLMExecutor(Executor):
    """Frozen executor calling a vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8002/v1",
                 model: str = "Qwen/Qwen3-8B", history_length: int = 3):
        self.base_url = base_url
        self.model = model
        self.history_length = history_length

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        from skillos.utils.http import openai_chat
        prompt = self._build_prompt(
            task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills, self.history_length,
        )
        result = openai_chat(self.base_url, self.model, prompt, temperature=0.7)
        return _parse_action(result, admissible_actions)


class APIExecutor(Executor):
    """Frozen executor calling any OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "Qwen/Qwen3-8B", history_length: int = 3):
        import os
        self.base_url = base_url or os.environ.get("SKILLOS_EXECUTOR_BASE_URL", "https://api.inference.sh/v1")
        self.api_key = api_key or os.environ.get("SKILLOS_EXECUTOR_API_KEY", "")
        self.model = model
        self.history_length = history_length

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        from skillos.utils.http import openai_chat
        prompt = self._build_prompt(
            task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills, self.history_length,
        )
        result = openai_chat(self.base_url, self.model, prompt, temperature=0.7, api_key=self.api_key)
        return _parse_action(result, admissible_actions)


def _parse_action(model_output: str, admissible_actions: list[str]) -> str:
    """Parse action from model output. Paper uses <action>...</action> tags."""
    match = re.search(r"<action>\s*(.*?)\s*</action>", model_output, re.DOTALL)
    if match:
        action = match.group(1).strip()
        if action in admissible_actions:
            return action
        action_lower = action.lower()
        for a in admissible_actions:
            if a.lower() == action_lower:
                return a

    for a in admissible_actions:
        if a in model_output:
            return a

    return admissible_actions[0] if admissible_actions else "look"


def create_executor(config: dict) -> Executor:
    """Create an executor from config."""
    executor_type = config.get("type", "heuristic")

    if executor_type == "heuristic":
        return HeuristicExecutor()
    elif executor_type == "local":
        return LocalExecutor(
            model_name=config.get("model", "Qwen/Qwen3-8B"),
            device=config.get("device", "auto"),
            history_length=config.get("history_length", 3),
        )
    elif executor_type == "vllm":
        return VLLMExecutor(
            base_url=config.get("base_url", "http://localhost:8002/v1"),
            model=config.get("model", "Qwen/Qwen3-8B"),
            history_length=config.get("history_length", 3),
        )
    elif executor_type == "api":
        return APIExecutor(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config.get("model", "Qwen/Qwen3-8B"),
            history_length=config.get("history_length", 3),
        )
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")
