"""Shared HTTP helper for OpenAI-compatible API calls."""

from __future__ import annotations

import json
import urllib.request


def openai_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: int = 30,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Used by VLLMExecutor, APIExecutor, VLLMJudge, APIJudge.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]
