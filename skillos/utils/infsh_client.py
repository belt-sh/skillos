"""Resilient wrapper around inferencesh.client.tasks for long training runs.

The plain `client.tasks.run(params)` submits + waits inline. If the *stream*
drops (e.g. "Stream timed out - no chunks received for 120 seconds") the
SDK raises and the task ID is lost, even though the task itself is still
alive on inference.sh. For a multi-hour training run with thousands of
calls, intermittent stream drops are expected — we want to reconnect and
collect the real result, not silently lose work.

This helper:
1. Submits without waiting (`wait=False`) to capture the task_id up front.
2. Polls `wait_for_completion(task_id)` with bounded reconnects on stream
   timeout. The task continues running on the server between reconnects.
3. Falls back to `tasks.get(task_id)` if the stream is permanently flaky.
4. Logs the task_id immediately on submission so callers can audit cost
   via `belt task cost <id>` even if the local call ultimately gives up.
"""

from __future__ import annotations

import os
import time
from typing import Callable

_STREAM_TIMEOUT_MARKERS = (
    "stream timed out",
    "no chunks received",
    "connection reset",
    "connection aborted",
)


def _is_stream_timeout(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _STREAM_TIMEOUT_MARKERS)


def _attach_to_task(client, task_id: str, max_stream_reconnects: int,
                    poll_fallback_max_seconds: float, poll_fallback_interval: float):
    """Wait for an already-submitted task to finish, with stream reconnects
    and a polling fallback. Returns the completed task dict, or raises if
    the task fails / is cancelled / never completes.
    """
    last_err: Exception | None = None
    # 1. Try the stream a few times — server keeps running across reconnects.
    for _ in range(max_stream_reconnects + 1):
        try:
            return client.tasks.wait_for_completion(task_id)
        except Exception as e:
            last_err = e
            if not _is_stream_timeout(e):
                raise

    # 2. Stream is hopeless — poll directly until the task settles.
    deadline = time.time() + poll_fallback_max_seconds
    while time.time() < deadline:
        try:
            state = client.tasks.get(task_id)
        except Exception as e:
            last_err = e
            time.sleep(poll_fallback_interval)
            continue
        status = state.get("status")
        # TaskStatus: COMPLETED=9, FAILED=10, CANCELLED=11
        if status in (9, "COMPLETED", "completed"):
            return state
        if status in (10, "FAILED", "failed", 11, "CANCELLED", "cancelled"):
            raise RuntimeError(
                f"task {task_id} ended in status {status}: {state.get('error')}"
            )
        time.sleep(poll_fallback_interval)

    raise RuntimeError(
        f"task {task_id} never completed within {poll_fallback_max_seconds:.0f}s "
        f"of fallback polling. last error: {last_err}"
    )


def run_task_resilient(
    client,
    params: dict,
    on_task_id: Callable[[str], None] | None = None,
    max_stream_reconnects: int = 5,
    poll_fallback_max_seconds: float = 900.0,
    poll_fallback_interval: float = 5.0,
    max_resubmissions: int = 4,
    resubmission_backoff_base: float = 5.0,
) -> dict:
    """Submit + wait on an inference.sh task with aggressive retry.

    Failures (stream drops, task FAILED, task CANCELLED, polling exhausted)
    trigger a *fresh resubmission* of the params — a new task_id, a new
    full retry budget. Up to `max_resubmissions` total submissions before
    re-raising. This keeps real reward signal coming through transient
    upstream flakiness instead of silently scoring 0 / dropping samples.

    Wall budget per call (worst case):
        max_resubmissions × (max_stream_reconnects + poll_fallback_max_seconds)
        ≈ 4 × (5 × 120s stream + 900s poll) = ~6000s = 100 min.

    Callers should expect long-tail latency under outage. The point is to
    eventually return a real result, since silent failure pollutes training.
    """
    last_err: Exception | None = None
    for sub_idx in range(max_resubmissions):
        try:
            submission = client.tasks.run(params, wait=False)
        except Exception as e:
            last_err = e
            time.sleep(resubmission_backoff_base * (2 ** sub_idx))
            continue
        task_id = submission.get("id") or submission.get("task_id")
        if not task_id:
            last_err = RuntimeError(f"tasks.run returned no task id: {submission}")
            time.sleep(resubmission_backoff_base * (2 ** sub_idx))
            continue
        if on_task_id is not None:
            try:
                on_task_id(task_id)
            except Exception:
                pass  # never let logging break the call

        try:
            return _attach_to_task(
                client, task_id,
                max_stream_reconnects=max_stream_reconnects,
                poll_fallback_max_seconds=poll_fallback_max_seconds,
                poll_fallback_interval=poll_fallback_interval,
            )
        except Exception as e:
            last_err = e
            import sys
            print(
                f"[infsh] submission {sub_idx + 1}/{max_resubmissions} "
                f"task {task_id} failed: {type(e).__name__}: {e} — resubmitting",
                file=sys.stderr,
            )
            time.sleep(resubmission_backoff_base * (2 ** sub_idx))

    raise RuntimeError(
        f"inference.sh call failed after {max_resubmissions} resubmissions. "
        f"last error: {last_err}"
    )
