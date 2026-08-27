"""Explainable Analytics user-state precedence and snapshot persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    calculate_user_state,
    recalculate_user_snapshot,
)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def _fixture(tmp_path, *, created_at=NOW - timedelta(days=1)):
    path = tmp_path / "states.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO users VALUES (1, 'user@example.test', 'User', 'x', 'user', ?)",
            (created_at.isoformat(),),
        )
    analytics = AnalyticsStore(path)
    return AnalyticsService(analytics), AnalyticsStateStore(analytics)


def _run(service, name, index, at, *, error_category=None):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=f"run:{name}:run-{index}",
        source_record_type="run",
        source_record_id=f"run-{index}",
        correlation_id=f"run-{index}",
        error_category=error_category,
        occurred_at=at,
    ).event


def test_blocked_wins_over_needs_attention(tmp_path):
    service, store = _fixture(tmp_path)
    for index in range(3):
        _run(
            service,
            "backtest_failed",
            index,
            NOW - timedelta(hours=index + 2),
            error_category="internal_error",
        )
    blocked = service.record_server_event(
        event_name="safe_error_recorded",
        user_id=1,
        source_event_id="resource:safe_error_recorded:run-blocked:credits_unavailable",
        source_record_type="run",
        source_record_id="run-blocked",
        error_category="credits_unavailable",
        occurred_at=NOW - timedelta(hours=1),
    ).event

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "blocked"
    assert snapshot.reason_code == "billing_lane_unavailable"
    assert snapshot.evidence_event_ids == [blocked.event_id]


def test_new_user_without_run_is_onboarding(tmp_path):
    _service, store = _fixture(tmp_path)

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "onboarding"
    assert snapshot.reason_code == "no_successful_run"


def test_three_newest_terminal_failures_need_attention(tmp_path):
    service, store = _fixture(tmp_path)
    failures = [
        _run(
            service,
            "backtest_failed",
            index,
            NOW - timedelta(hours=index + 1),
            error_category="internal_error",
        )
        for index in range(3)
    ]

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "needs_attention"
    assert snapshot.reason_code == "three_consecutive_failed_runs"
    assert set(snapshot.evidence_event_ids) == {event.event_id for event in failures}


def test_completed_or_cancelled_run_breaks_failure_sequence(tmp_path):
    service, store = _fixture(tmp_path)
    _run(service, "backtest_failed", 1, NOW - timedelta(hours=1))
    _run(service, "backtest_cancelled", 2, NOW - timedelta(hours=2))
    _run(service, "backtest_failed", 3, NOW - timedelta(hours=3))
    _run(service, "backtest_failed", 4, NOW - timedelta(hours=4))

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "onboarding"


def test_dormant_and_active_use_thirty_day_activity(tmp_path):
    service, store = _fixture(tmp_path, created_at=NOW - timedelta(days=60))
    _run(service, "backtest_completed", 1, NOW - timedelta(days=45))

    dormant = calculate_user_state(1, now=NOW, store=store)
    _run(service, "backtest_completed", 2, NOW - timedelta(days=1))
    active = calculate_user_state(1, now=NOW, store=store)

    assert dormant.status == "dormant"
    assert dormant.reason_code == "no_meaningful_activity_30d"
    assert active.status == "active"


def test_recalculate_upserts_one_current_snapshot(tmp_path):
    service, store = _fixture(tmp_path)
    first = recalculate_user_snapshot(1, now=NOW, store=store)
    _run(service, "backtest_completed", 1, NOW - timedelta(hours=1))
    second = recalculate_user_snapshot(1, now=NOW, store=store)

    assert first.status == "onboarding"
    assert second.status == "active"
    assert store.get_snapshot(1) == second
