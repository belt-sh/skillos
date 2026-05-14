"""Content quality judge — evaluates curator output for r_cnt reward.

Pluggable backends:
- HeuristicJudge: rule-based, no model needed (pipeline validation)
- LocalJudge: runs a local model via transformers (single GPU)
- VLLMJudge: calls a vLLM server (multi-GPU, dedicated judge GPU)
- APIJudge: calls any OpenAI-compatible API (remote judge)

Paper uses Qwen3-32B as judge. Configure via --judge flag or config.
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

    @abstractmethod
    def score_batch(self, contents: list[str]) -> list[float]:
        """Score multiple skills. Default: loop over score()."""
        ...


class HeuristicJudge(Judge):
    """Rule-based judge — no model needed.

    Checks: valid frontmatter, has structure, conciseness, abstraction.
    Good enough for pipeline validation. Fast, zero cost.
    """

    def score(self, skill_content: str) -> float:
        points = 0.0

        # 1. Has valid frontmatter
        if re.match(r"^---\s*\n.*?name:.*?\n.*?description:.*?\n---", skill_content, re.DOTALL):
            points += 1.0

        # 2. Has workflow/structure (markdown headings)
        if re.search(r"^#\s+\w", skill_content, re.MULTILINE):
            points += 1.0

        # 3. Conciseness (not a raw trajectory dump)
        word_count = len(skill_content.split())
        if 20 < word_count < 500:
            points += 1.0

        # 4. Abstraction (not dominated by specific numbers/IDs)
        digit_ratio = sum(c.isdigit() for c in skill_content) / max(len(skill_content), 1)
        if digit_ratio < 0.1:
            points += 1.0

        return points / 4.0

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


class LocalJudge(Judge):
    """Judge using a local model via transformers.

    Loads a model (e.g. Qwen3-32B) and runs inference locally.
    Use when you have a spare GPU or enough RAM for the judge model.
    """

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
    """Judge calling a vLLM server.

    Run the judge model on a dedicated GPU:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen3-32B --port 8001

    Then point this judge at it.
    """

    def __init__(self, base_url: str = "http://localhost:8001/v1", model: str = "Qwen/Qwen3-32B"):
        self.base_url = base_url
        self.model = model

    def _call(self, prompt: str) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def score(self, skill_content: str) -> float:
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        return _parse_judge_score(self._call(prompt))

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


class APIJudge(Judge):
    """Judge calling any OpenAI-compatible API.

    Works with: inference.sh, OpenRouter, Together, Fireworks, etc.
    Set SKILLOS_JUDGE_API_KEY and SKILLOS_JUDGE_BASE_URL env vars,
    or pass them directly.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "Qwen/Qwen3-32B",
    ):
        import os
        self.base_url = base_url or os.environ.get("SKILLOS_JUDGE_BASE_URL", "https://api.inference.sh/v1")
        self.api_key = api_key or os.environ.get("SKILLOS_JUDGE_API_KEY", "")
        self.model = model

    def _call(self, prompt: str) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def score(self, skill_content: str) -> float:
        prompt = CONTENT_QUALITY_JUDGE.format(content=skill_content)
        return _parse_judge_score(self._call(prompt))

    def score_batch(self, contents: list[str]) -> list[float]:
        return [self.score(c) for c in contents]


# --- Helpers ---

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
    """Create a judge from config.

    Config examples:
        {"type": "heuristic"}
        {"type": "local", "model": "Qwen/Qwen3-32B"}
        {"type": "vllm", "base_url": "http://localhost:8001/v1"}
        {"type": "api", "base_url": "https://api.inference.sh/v1", "api_key": "..."}
    """
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
    else:
        raise ValueError(f"Unknown judge type: {judge_type}")
