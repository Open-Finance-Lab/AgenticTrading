"""Read-side classification of backtest_attempts rows (spec decision D3)."""

from datetime import datetime, timedelta, timezone

from dashboard.backend.domain.backtesting.attempts import (
    DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
    INTERRUPTED_GRACE_SECONDS,
    INTERRUPTED_MESSAGE,
    attempt_as_run_entry,
    present_attempt,
    summarize_attempt,
)


def _row(status="running", created_at="2026-08-04 10:00:00", timeout_seconds=1800):
    return {
        "run_id": "att-1", "agent_id": "agent-1", "session_id": "sess-1",
        "agent_name": "My Agent", "start_date": "2026-05-01",
        "end_date": "2026-05-07", "params_json": None, "status": status,
        "error": None, "timeout_seconds": timeout_seconds,
        "created_at": created_at, "finished_at": None,
    }


def _now(created="2026-08-04 10:00:00", plus_seconds=0):
    base = datetime.strptime(created, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return base + timedelta(seconds=plus_seconds)


def test_fresh_running_row_stays_running():
    presented = present_attempt(_row(), now=_now(plus_seconds=60))
    assert presented["status"] == "running"
    assert presented["error"] is None


def test_running_past_own_timeout_plus_grace_presents_interrupted():
    stale = 1800 + INTERRUPTED_GRACE_SECONDS + 1
    presented = present_attempt(_row(), now=_now(plus_seconds=stale))
    assert presented["status"] == "interrupted"
    assert presented["error"] == INTERRUPTED_MESSAGE


def test_running_within_timeout_plus_grace_stays_running():
    inside = 1800 + INTERRUPTED_GRACE_SECONDS - 1
    assert present_attempt(_row(), now=_now(plus_seconds=inside))["status"] == "running"


def test_missing_timeout_falls_back_to_default_budget():
    stale = DEFAULT_ATTEMPT_TIMEOUT_SECONDS + INTERRUPTED_GRACE_SECONDS + 1
    row = _row(timeout_seconds=None)
    assert present_attempt(row, now=_now(plus_seconds=stale))["status"] == "interrupted"


def test_terminal_rows_and_unparseable_timestamps_pass_through():
    assert present_attempt(_row(status="failed"))["status"] == "failed"
    assert present_attempt(_row(status="completed"))["status"] == "completed"
    weird = _row(created_at="not-a-time")
    assert present_attempt(weird, now=_now(plus_seconds=999999))["status"] == "running"


def test_present_attempt_does_not_mutate_its_input():
    row = _row()
    present_attempt(row, now=_now(plus_seconds=10**6))
    assert row["status"] == "running"


def test_summarize_attempt_is_the_five_key_card_payload():
    stale = 1800 + INTERRUPTED_GRACE_SECONDS + 1
    summary = summarize_attempt(_row(), now=_now(plus_seconds=stale))
    assert summary == {
        "run_id": "att-1", "status": "interrupted",
        "error": INTERRUPTED_MESSAGE,
        "created_at": "2026-08-04 10:00:00", "finished_at": None,
    }


def test_attempt_as_run_entry_shapes_a_history_row():
    presented = dict(_row(status="failed"), error="boom")
    entry = attempt_as_run_entry(presented)
    assert entry == {
        "run_id": "att-1", "agent_name": "My Agent", "mode": "backtest",
        "start_date": "2026-05-01", "end_date": "2026-05-07",
        "initial_equity": 0.0, "num_trades": 0,
        "created_at": "2026-08-04 10:00:00",
        "status": "failed", "error": "boom",
    }
    anonymous = attempt_as_run_entry(dict(presented, agent_name=None))
    assert anonymous["agent_name"] == "Agent"
