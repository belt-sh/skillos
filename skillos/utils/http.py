"""Shared HTTP helper for OpenAI-compatible API calls.

Used by both executor and judge backends that hit a remote inference endpoint
(vLLM sidecar, inference.sh, OpenAI, etc.). Built to survive a long training
run: retries with backoff on 5xx / connection drops, configurable per-request
timeout, and optional concurrent batch dispatch for executor batching across
parallel envs.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = int(os.environ.get("SKILLOS_HTTP_TIMEOUT", "120"))
DEFAULT_RETRIES = int(os.environ.get("SKILLOS_HTTP_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.environ.get("SKILLOS_HTTP_BACKOFF", "1.5"))


def openai_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    top_p: float | None = None,
    extra_body: dict | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint with retries.

    Retries on connection errors and HTTP 5xx with exponential backoff.
    4xx errors return immediately (likely a config bug, not transient).

    extra_body merges vendor sampling knobs into the request (e.g. top_k,
    min_p, presence_penalty, or vLLM's chat_template_kwargs for enable_thinking).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if extra_body:
        payload.update(extra_body)
    body = json.dumps(payload).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url.rstrip('/')}/chat/completions"
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last_err = e
            if 400 <= e.code < 500:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        if attempt < retries:
            time.sleep(DEFAULT_BACKOFF ** attempt)
    raise RuntimeError(f"openai_chat failed after {retries + 1} attempts: {last_err}")


def openai_chat_batch(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int = 256,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_concurrency: int = 16,
) -> list[str]:
    """Issue many chat requests concurrently. Order preserved.

    Use to batch executor actions across parallel envs, or to score multiple
    skills with the judge in parallel.
    """
    if not prompts:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futs = [
            pool.submit(openai_chat, base_url, model, p, max_tokens, temperature, api_key, timeout)
            for p in prompts
        ]
        return [f.result() for f in futs]
