"""Frozen executor — solves ALFWorld tasks using retrieved skills.

Pluggable backends:
- HeuristicExecutor: picks first admissible action (pipeline validation)
- LocalExecutor: runs a local model via transformers (same or separate GPU)
- VLLMExecutor: calls a vLLM server (dedicated inference GPU)
- APIExecutor: calls any OpenAI-compatible API (remote executor)
- InfshExecutor: calls an inference.sh app via the inferencesh SDK

Paper uses frozen Qwen3-8B as executor. The executor is NEVER trained —
it just generates trajectories that the curator learns from.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from skillos.curator.prompts import ALFWORLD_EXECUTOR


def _log_infsh_task(role: str, app: str, task_id: str) -> None:
    """Append task IDs to a NDJSON log so we can query `belt task cost` later.

    Path overridable via SKILLOS_INFSH_TASKLOG (default: ./output/infsh_tasks.jsonl).
    """
    path = os.environ.get("SKILLOS_INFSH_TASKLOG", "./output/infsh_tasks.jsonl")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import json
        import time
        with open(path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "role": role,
                "app": app,
                "task_id": task_id,
            }) + "\n")
    except Exception:
        # Logging is best-effort — never fail a rollout because of it.
        pass


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
                 model: str = "Qwen/Qwen3-8B", history_length: int = 3,
                 temperature: float = 0.6, max_tokens: int = 2048,
                 top_p: float | None = None, top_k: int | None = None,
                 min_p: float | None = None, presence_penalty: float | None = None,
                 enable_thinking: bool | None = None):
        self.base_url = base_url
        self.model = model
        self.history_length = history_length
        self.temperature = temperature
        # Qwen3 thinking-mode emits <think>…</think> inline before <action>;
        # 256 (openai_chat default) truncates the reasoning. Give it room.
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        from skillos.utils.http import openai_chat
        prompt = self._build_prompt(
            task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills, self.history_length,
        )
        extra: dict = {}
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        if self.min_p is not None:
            extra["min_p"] = self.min_p
        if self.presence_penalty is not None:
            extra["presence_penalty"] = self.presence_penalty
        if self.enable_thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        result = openai_chat(self.base_url, self.model, prompt,
                             temperature=self.temperature, max_tokens=self.max_tokens,
                             top_p=self.top_p, extra_body=extra or None)
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


class InfshExecutor(Executor):
    """Frozen executor calling an inference.sh app (e.g. openrouter/qwen3-8b).

    Uses the inferencesh Python SDK. Stateless tasks: each act() is a fresh
    client.tasks.run() call.
    """

    def __init__(self, app: str = "openrouter/qwen3-8b", api_key: str | None = None,
                 history_length: int = 3, temperature: float = 0.6,
                 max_tokens: int = 8192, context_size: int = 32768,
                 top_p: float = 0.95, top_k: int | None = None,
                 min_p: float | None = None,
                 infra: str = "cloud", variant: str = "default",
                 setup: dict | None = None,
                 reasoning_effort: str | None = "medium"):
        """Defaults follow the Qwen3-8B HF model card's *thinking-mode*
        recommendation (temperature=0.6, top_p=0.95, native 32768 context,
        generous output budget), because the ALFWORLD_EXECUTOR prompt requires
        the model to reason inside <think></think> tags before emitting
        <action></action>. A reasoning-off, token-starved executor (e.g.
        max_tokens=256) silently underperforms here. The card also warns:
        "DO NOT use greedy decoding." Pass reasoning_effort=None to force
        non-thinking behavior.
        """
        from inferencesh import inference
        from skillos.utils.infsh_auth import resolve_infsh_api_key
        self.app = app
        self.api_key = resolve_infsh_api_key(api_key)
        self.client = inference(api_key=self.api_key)
        self.history_length = history_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_size = context_size
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.infra = infra
        self.variant = variant
        self.setup = setup or {}
        self.reasoning_effort = reasoning_effort

    def act(self, task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills) -> str:
        prompt = self._build_prompt(
            task_description, observation, admissible_actions,
            step_count, action_history, retrieved_skills, self.history_length,
        )
        input_payload = {
            "text": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "context_size": self.context_size,
        }
        if self.top_k is not None:
            input_payload["top_k"] = self.top_k
        if self.min_p is not None:
            input_payload["min_p"] = self.min_p
        if self.reasoning_effort is not None:
            input_payload["reasoning_effort"] = self.reasoning_effort
        params = {
            "app": self.app,
            "infra": self.infra,
            "variant": self.variant,
            "input": input_payload,
        }
        if self.setup:
            params["setup"] = self.setup
        from skillos.utils.infsh_client import run_task_resilient
        result = run_task_resilient(
            self.client, params,
            on_task_id=lambda tid: _log_infsh_task("executor", self.app, tid),
        )
        output = (result or {}).get("output") or {}
        # Qwen3 native reasoning lands in a separate "reasoning" field; the
        # final answer (with the <action> tag) is in "response". Concatenate
        # so _parse_action sees the action regardless of where the model put it.
        if isinstance(output, dict):
            text = (output.get("response") or "")
            if not text:
                text = output.get("reasoning") or ""
        else:
            text = ""
        return _parse_action(text, admissible_actions)


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
            temperature=config.get("temperature", 0.6),
            max_tokens=config.get("max_tokens", 2048),
            top_p=config.get("top_p"),
            top_k=config.get("top_k"),
            min_p=config.get("min_p"),
            presence_penalty=config.get("presence_penalty"),
            enable_thinking=config.get("enable_thinking"),
        )
    elif executor_type == "api":
        return APIExecutor(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config.get("model", "Qwen/Qwen3-8B"),
            history_length=config.get("history_length", 3),
        )
    elif executor_type == "infsh":
        return InfshExecutor(
            app=config.get("app", "openrouter/qwen3-8b"),
            api_key=config.get("api_key"),
            history_length=config.get("history_length", 3),
            temperature=config.get("temperature", 0.6),
            max_tokens=config.get("max_tokens", 8192),
            context_size=config.get("context_size", 32768),
            top_p=config.get("top_p", 0.95),
            top_k=config.get("top_k"),
            min_p=config.get("min_p"),
            infra=config.get("infra", "cloud"),
            variant=config.get("variant", "default"),
            setup=config.get("setup"),
            reasoning_effort=config.get("reasoning_effort", "medium"),
        )
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")
