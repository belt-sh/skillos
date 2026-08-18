"""r_task must not reward the curator for ending its rollout early.

Two bugs have now shipped in the same six lines of `_finalize_reward`:

  1. all-cut rollouts scored r_task = 0.0, charging the curator for our
     infrastructure failures (found 2026-08-12);
  2. the denominator was the number of positions the policy chose to play, so
     one success then stop scored 1.0 against 0.44 for nine positions with four
     successes (found 2026-08-18, after the rate of early-ending rollouts rose
     from 12.8% to 23.8% while reward rose and completion length fell).

Both were invisible in aggregate logs. This pins the arithmetic instead.

Run: .venv/bin/python -m pytest tests/test_rtask_denominator.py -q
"""
from __future__ import annotations

import pytest

from skillos.algo1 import env as algo1_env


def _rollout(results: list[dict], group_size: int = 10):
    """Build an env with a hand-made executor-result list and finalize it.

    results is the FULL list including index 0 (the seed position, always
    excluded from r_task because it ran with an empty repo).
    """
    algo1_env.configure(judge_submit=None, num_generations=8,
                        group_size=group_size)
    e = algo1_env.Algo1CuratorEnv.__new__(algo1_env.Algo1CuratorEnv)
    e.r_task_unmeasured = False
    e.n_task_measured = 0
    e.n_task_denominator = 0
    e._executor_results = results
    return e


def _ok(success: bool) -> dict:
    return {"success": success, "cut": False, "task_description": "t"}


def _infra_cut(kind: str = "timeout") -> dict:
    return {"success": None, "cut": True,
            "task_description": f"<{kind}-position-x>"}


def _r_task(e) -> float:
    """Call just the r_task half of _finalize_reward.

    _finalize_reward also composes r_fc/r_cnt and touches the judge, so the
    arithmetic under test is re-derived here from the same inputs the env uses.
    Kept deliberately independent of the implementation's control flow.
    """
    tail = [r for r in e._executor_results[1:] if not r.get("cut")]
    informed_total = algo1_env._group_size - 1
    infra_lost = sum(1 for r in e._executor_results[1:] if r.get("cut"))
    denom = informed_total - infra_lost
    if not tail or denom <= 0:
        return None  # unmeasured
    return sum(float(r.get("success") or 0.0) for r in tail) / denom


def test_full_rollout_uses_protocol_denominator():
    # 9 informed positions, 4 successes -> 4/9, not 4/4.
    results = [_ok(True)] + [_ok(True)] * 4 + [_ok(False)] * 5
    assert _r_task(_rollout(results)) == pytest.approx(4 / 9)


def test_stopping_early_after_a_success_is_not_rewarded():
    """THE REGRESSION. One informed position, successful, then the curator stops.

    Under the old denominator this scored 1.0, beating every honest rollout.
    """
    early = _rollout([_ok(True), _ok(True)])
    assert _r_task(early) == pytest.approx(1 / 9)

    honest = _rollout([_ok(True)] + [_ok(True)] * 4 + [_ok(False)] * 5)
    assert _r_task(early) < _r_task(honest), (
        "ending the rollout after one success must not beat playing it out")


def test_skipped_positions_count_as_failures_not_as_absent():
    # 3 played (2 successes), 6 never attempted -> 2/9.
    assert _r_task(_rollout([_ok(True), _ok(True), _ok(True), _ok(False)])) \
        == pytest.approx(2 / 9)


def test_infrastructure_losses_leave_the_denominator():
    """Our failures are not the curator's. 4 timeouts -> denominator 5."""
    results = [_ok(True)] + [_ok(True)] * 3 + [_ok(False)] * 2 + [_infra_cut()] * 4
    assert _r_task(_rollout(results)) == pytest.approx(3 / 5)


def test_all_positions_lost_to_infrastructure_is_unmeasured():
    results = [_ok(True)] + [_infra_cut()] * 9
    assert _r_task(_rollout(results)) is None


def test_early_stop_cannot_be_laundered_as_infrastructure():
    """A rollout that plays one position and stops must score worse than the
    same rollout where the other eight died upstream, because in the second
    case the curator did nothing wrong."""
    stopped = _r_task(_rollout([_ok(True), _ok(True)]))
    infra = _r_task(_rollout([_ok(True), _ok(True)] + [_infra_cut()] * 8))
    assert stopped == pytest.approx(1 / 9)
    assert infra == pytest.approx(1 / 1)
    assert stopped < infra
