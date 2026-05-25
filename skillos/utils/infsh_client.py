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


def _is_client_error(err: Exception) -> bool:
    """4xx errors are deterministic (bad payload, validation) — resubmitting
    the same params just burns backoff cycles, so treat them as fatal. 408
    (timeout) and 429 (rate limit) are transient and stay retryable.
    """
    code = getattr(err, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in (408, 429)


def _attach_to_task(client, task_id: str, max_stream_reconnects: int,
                    poll_fallback_max_seconds: float, poll_fallback_interval: float,
                    wait_timeout: float = 120.0):
    """Wait for an already-submitted task to finish, with stream reconnects
    and a polling fallback. Returns the completed task dict, or raises if
    the task fails / is cancelled / never completes.

    wait_timeout bounds each wait_for_completion call: the SDK (>=0.7.7) takes
    a `timeout` and does a final GET-poll reconciliation on expiry, so a wedged
    stream can no longer block forever. We then loop our own reconnects on top.
    """
    last_err: Exception | None = None
    # 1. Try the stream a few times — server keeps running across reconnects.
    for _ in range(max_stream_reconnects + 1):
        try:
            return client.tasks.wait_for_completion(task_id, timeout=wait_timeout)
        except TypeError:
            # Older SDK without the `timeout` kwarg — fall back (may hang).
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
    max_resubmissions: int = 10,
    resubmission_backoff_base: float = 60.0,
    resubmission_backoff_cap: float = 900.0,
) -> dict:
    """Submit + wait on an inference.sh task with aggressive retry.

    Failures (stream drops, task FAILED, task CANCELLED, polling exhausted)
    trigger a *fresh resubmission* of the params — a new task_id, a new
    full retry budget. Up to `max_resubmissions` total submissions before
    re-raising. This keeps real reward signal coming through transient
    upstream flakiness instead of silently scoring 0 / dropping samples.

    Wall budget per call (worst case):
        per-attempt: ~5 × 120s stream + 900s poll = ~1500s = 25 min
        backoff between attempts: min(60 × 2^idx, 900s) = 60, 120, 240, 480,
          900, 900, 900, 900, 900s (capped from idx=4 onward)
        10 attempts: 10 × 1500s + ~5340s backoff total = ~20340s = ~5.6 hours.

    That's the worst case under a sustained upstream outage. In normal flow
    each attempt completes in seconds and backoff is irrelevant.

    Callers should expect long-tail latency under outage. The point is to
    eventually return a real result, since silent failure pollutes training.
    """
    last_err: Exception | None = None
    for sub_idx in range(max_resubmissions):
        try:
            submission = client.tasks.run(params, wait=False)
        except Exception as e:
            last_err = e
            if _is_client_error(e):
                raise  # bad request — retrying won't help, fail fast
            time.sleep(min(resubmission_backoff_base * (2 ** sub_idx), resubmission_backoff_cap))
            continue
        task_id = submission.get("id") or submission.get("task_id")
        if not task_id:
            last_err = RuntimeError(f"tasks.run returned no task id: {submission}")
            time.sleep(min(resubmission_backoff_base * (2 ** sub_idx), resubmission_backoff_cap))
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
            if _is_client_error(e):
                print(f"[infsh] task {task_id} 4xx client error — not resubmitting: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                raise
            is_last = sub_idx + 1 >= max_resubmissions
            verb = "GIVING UP" if is_last else "resubmitting"
            print(
                f"[infsh] submission {sub_idx + 1}/{max_resubmissions} "
                f"task {task_id} failed: {type(e).__name__}: {e} — {verb}",
                file=sys.stderr,
            )
            if not is_last:
                time.sleep(min(resubmission_backoff_base * (2 ** sub_idx), resubmission_backoff_cap))

    import sys
    print(
        f"[infsh] ALL {max_resubmissions} resubmissions exhausted; raising. "
        f"last error: {last_err}",
        file=sys.stderr,
    )
    raise RuntimeError(
        f"inference.sh call failed after {max_resubmissions} resubmissions. "
        f"last error: {last_err}"
    )
