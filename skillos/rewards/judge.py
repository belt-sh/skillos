"""Content quality judge -- evaluates curator output for r_cnt reward.

Pluggable backends:
- HeuristicJudge: rule-based, no model needed (pipeline validation)
- LocalJudge: runs a local model via transformers (single GPU)
- VLLMJudge: calls a vLLM server (multi-GPU, dedicated judge GPU)
- APIJudge: calls any OpenAI-compatible API (remote judge)

Paper uses Qwen3-32B as judge.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from skillos.curator.prompts import CONTENT_QUALITY_JUDGE


class Judge(ABC):
    """Base interface for content quality judges."""

    @abstractmethod
    def score(self, skill_content: str) -> float:
        """Score a skill's content quality. Returns 0.0-1.0."""
        ...

    def score_batch(self, contents: list[str]) -> list[float]:
        """Score multiple skills. Override for true batching."""
        return [self.score(c) for c in contents]


class HeuristicJudge(Judge):
    """Rule-based judge, no model needed. Fast, zero cost."""

    def score(self, skill_content: str) -> float:
        points = 0.0

        if re.match(r"^---\s*\n.*?name:.*?\n.*?description:.*?\n---", skill_content, re.DOTALL):
            points += 1.0

        if re.search(r"^#\s+\w", skill_content, re.MULTILINE):
            points += 1.0

        word_count = len(skill_content.split())
        if 20 < word_count < 500:
            points += 1.0

        digit_ratio = sum(c.isdigit() for c in skill_content) / max(len(skill_content), 1)
        if digit_ratio < 0.1:
            points += 1.0

        return points / 4.0


class LocalJudge(Judge):
    """Judge using a local model via transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-32B", device: str = "auto"):
        self.model_name = model_name
        self._pipeline = None
        self._device = device

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

    def score(self, skill_content: str) -> float:
        self._load()
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        result = self._pipeline(prompt, return_full_text=False)[0]["generated_text"]
        return _parse_judge_score(result)

    def score_batch(self, contents: list[str]) -> list[float]:
        self._load()
        prompts = [CONTENT_QUALITY_JUDGE.format(content=c) for c in contents]
        results = self._pipeline(prompts, return_full_text=False, batch_size=len(prompts))
        return [_parse_judge_score(r[0]["generated_text"]) for r in results]


class VLLMJudge(Judge):
    """Judge calling a vLLM server."""

    def __init__(self, base_url: str = "http://localhost:8001/v1", model: str = "Qwen/Qwen3-32B"):
        self.base_url = base_url
        self.model = model

    def score(self, skill_content: str) -> float:
        from skillos.utils.http import openai_chat
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        return _parse_judge_score(openai_chat(self.base_url, self.model, prompt))

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


class APIJudge(Judge):
    """Judge calling any OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str = "Qwen/Qwen3-32B"):
        import os
        self.base_url = base_url or os.environ.get("SKILLOS_JUDGE_BASE_URL", "https://api.inference.sh/v1")
        self.api_key = api_key or os.environ.get("SKILLOS_JUDGE_API_KEY", "")
        self.model = model

    def score(self, skill_content: str) -> float:
        from skillos.utils.http import openai_chat
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        return _parse_judge_score(openai_chat(self.base_url, self.model, prompt, api_key=self.api_key))

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


class InfshJudge(Judge):
    """Judge calling an inference.sh app (e.g. openrouter/qwen3-32b)."""

    def __init__(self, app: str = "openrouter/qwen3-32b", api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 256,
                 context_size: int = 8192,
                 infra: str = "cloud", variant: str = "default",
                 setup: dict | None = None):
        from inferencesh import inference
        from skillos.utils.infsh_auth import resolve_infsh_api_key
        self.app = app
        self.api_key = resolve_infsh_api_key(api_key)
        self.client = inference(api_key=self.api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_size = context_size
        self.infra = infra
        self.variant = variant
        self.setup = setup or {}

    def score(self, skill_content: str) -> float:
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        params = {
            "app": self.app,
            "infra": self.infra,
            "variant": self.variant,
            "input": {
                "text": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "context_size": self.context_size,
            },
        }
        if self.setup:
            params["setup"] = self.setup
        from skillos.utils.infsh_client import run_task_resilient
        from skillos.executor.executor import _log_infsh_task
        # Judge retry budget must stay well under the env-level judge timeout
        # (_judge_timeout_s, default 600s) and under NCCL's 1800s watchdog.
        # Worst case here: 2 resubmissions × (1 stream reconnect × 120s +
        # 120s poll fallback) + ~10s backoff = ~490s. Bounded; survives normal
        # infsh hiccups; gives up promptly under a real outage so the rank
        # can reach the reward gather.
        import os as _os
        result = run_task_resilient(
            self.client, params,
            on_task_id=lambda tid: _log_infsh_task("judge", self.app, tid),
            max_stream_reconnects=int(_os.environ.get("SKILLOS_JUDGE_MAX_STREAM_RECONNECTS", "1")),
            poll_fallback_max_seconds=float(_os.environ.get("SKILLOS_JUDGE_POLL_MAX_S", "120")),
            max_resubmissions=int(_os.environ.get("SKILLOS_JUDGE_MAX_RESUBS", "2")),
            resubmission_backoff_base=float(_os.environ.get("SKILLOS_JUDGE_BACKOFF_BASE_S", "10")),
            resubmission_backoff_cap=float(_os.environ.get("SKILLOS_JUDGE_BACKOFF_CAP_S", "60")),
        )
        output = (result or {}).get("output") or {}
        text = output.get("response", "") if isinstance(output, dict) else ""
        return _parse_judge_score(text)

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


def _parse_judge_score(response: str) -> float:
    """Parse judge response JSON to get binary VALID score."""
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{[^}]*\"VALID\"[^}]*\})", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            return 1.0 if result.get("VALID", False) else 0.0
    except (json.JSONDecodeError, AttributeError):
        pass
    return 0.0


def create_judge(config: dict) -> Judge:
    """Create a judge from config."""
    judge_type = config.get("type", "heuristic")

    if judge_type == "heuristic":
        return HeuristicJudge()
    elif judge_type == "local":
        return LocalJudge(
            model_name=config.get("model", "Qwen/Qwen3-32B"),
            device=config.get("device", "auto"),
        )
    elif judge_type == "vllm":
        return VLLMJudge(
            base_url=config.get("base_url", "http://localhost:8001/v1"),
            model=config.get("model", "Qwen/Qwen3-32B"),
        )
    elif judge_type == "api":
        return APIJudge(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config.get("model", "Qwen/Qwen3-32B"),
        )
    elif judge_type == "infsh":
        return InfshJudge(
            app=config.get("app", "openrouter/qwen3-32b"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 256),
            context_size=config.get("context_size", 8192),
            infra=config.get("infra", "cloud"),
            variant=config.get("variant", "default"),
            setup=config.get("setup"),
        )
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
