"""Reasoning training env — Algorithm 1 for DeepMath-103K.

Subclasses `skillos.algo1.Algo1CuratorEnv` and overrides just the two ALFWorld-
specific hooks:

- `_ensure_group_sequence`: draw G same-topic problem indices from DeepMath
  instead of ALFWorld game seeds.
- `_run_executor_at`: call the reasoning executor (Qwen3-8B via inference.sh),
  grade the boxed answer, and package the (problem, response, correctness)
  into a curator-consumable trajectory shape.

Everything else — the |G|+1 tool loop, r_task over positions 2..G, r_fc /
r_cnt / r_comp reward composition, judge wiring, GRPO batching — is inherited
unchanged.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import Optional

from skillos.algo1.env import Algo1CuratorEnv
from skillos.reasoning.train_data import (
    DEEPMATH_TOPICS, build_topic_index, get_problem, sample_group_seeds,
    seeds_for_topic,
)

# Same env vars as ALFWorld path — no reason to invent parallel knobs.
_max_tokens = int(os.environ.get("SKILLOS_REASONING_MAX_TOKENS", "8192"))
_temperature = float(os.environ.get("SKILLOS_REASONING_TEMPERATURE", "0.6"))
_top_p = float(os.environ.get("SKILLOS_REASONING_TOP_P", "0.95"))
_reasoning_effort = os.environ.get("SKILLOS_REASONING_EFFORT", "medium")
_executor_app: Optional[str] = None  # set by configure()

# Group-level topic registry — mirrors _group_types in algo1/env.py so the
# per-rollout log surfaces topic distribution the same way.
_group_topics: dict[int, str] = {}
_topic_lock = threading.Lock()


def configure(*, executor_app: str = "openrouter/qwen3-8b") -> None:
    """Wire the reasoning env to a specific executor app on inference.sh."""
    global _executor_app
    _executor_app = executor_app


def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<{m['role'].upper()}>\n{m['content']}\n</{m['role'].upper()}>")
    return "\n\n".join(parts)


def _call_reasoning_executor(problem: str, past_skills: str) -> str:
    """One remote executor call: retrieved skills + problem → CoT + boxed answer."""
    from inferencesh import inference
    from skillos.utils.infsh_auth import resolve_infsh_api_key
    from skillos.reasoning.prompts import build_messages
    messages = build_messages(problem, past_skills, kind="aime")
    client = inference(api_key=resolve_infsh_api_key())
    r = client.tasks.run({
        "app": _executor_app, "infra": "cloud", "variant": "default",
        "input": {
            "text": _messages_to_text(messages),
            "max_tokens": _max_tokens,
            "temperature": _temperature,
            "top_p": _top_p,
            "reasoning_effort": _reasoning_effort,
        },
    })
    out = (r or {}).get("output") or {}
    return (out.get("response") or "").strip()


class ReasoningCuratorEnv(Algo1CuratorEnv):
    """Algo1 env with reasoning problems instead of ALFWorld tasks."""

    # ---- Group sequence sampling --------------------------------------

    def _ensure_group_sequence(self) -> None:
        """Draw G same-topic DeepMath problem indices, deterministic per gid."""
        build_topic_index()
        gid = self._group_id
        topic = self._group_type  # reused slot — carries topic string here
        # `_group_sequences` and `_batch_lock` live in the parent module.
        from skillos.algo1.env import _group_sequences, _batch_lock, _group_size
        with _batch_lock:
            if gid in _group_sequences:
                self._task_seeds = list(_group_sequences[gid])
            else:
                self._task_seeds = sample_group_seeds(
                    group_id=gid, topic=topic, group_size=_group_size)
                _group_sequences[gid] = list(self._task_seeds)
        with _topic_lock:
            _group_topics[gid] = topic
        self._task_descriptions = [""] * _group_size
        print(f"[reasoning] rollout slot={self._slot} gid={gid} topic={topic!r} "
              f"seeds={self._task_seeds}", flush=True)

    # ---- Executor invocation ------------------------------------------

    def _run_executor_at(self, position: int) -> dict:
        """Retrieve → executor CoT → grade → package as a curator trajectory."""
        from skillos.reasoning.grading import grade
        idx = self._task_seeds[position]
        prob = get_problem(idx)
        t0 = time.time()
        past_skills = ""
        try:
            past = self._repo.retrieve(prob["question"], top_k=5)
            if past:
                past_skills = self._repo.format_skills(past)
        except Exception as e:
            print(f"[reasoning] retrieve failed slot={self._slot} "
                  f"gid={self._group_id} pos={position}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        try:
            response = _call_reasoning_executor(prob["question"], past_skills)
        except Exception as e:
            print(f"[reasoning] executor failed slot={self._slot} "
                  f"gid={self._group_id} pos={position}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            response = ""
        ok, pred = grade(response, prob["final_answer"], kind="gpqa_ft") \
            if not prob["final_answer"].lstrip("-+").isdigit() \
            else grade(response, prob["final_answer"], kind="aime")
        traj = [{
            "step": 1,
            "action": (response or "")[:6000],
            "observation": (
                f"correct answer is {prob['final_answer']}. "
                f"executor gave {pred!s} — {'CORRECT' if ok else 'INCORRECT'}."),
        }]
        result = {
            "task_description": prob["question"],
            "trajectory": traj,
            "success": bool(ok),
            "steps": 1,
            "gamefile": f"deepmath:{idx}",
            "skills_text": past_skills,
            "wall_s": time.time() - t0,
        }
        self._task_descriptions[position] = prob["question"]
        return result
