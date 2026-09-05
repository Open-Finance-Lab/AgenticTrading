"""Daily Analytics rollups use bounded dimensions and idempotent upserts."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    DailyRollup,
    rollup_lifecycle_day,
    rollup_day,
)
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)


def _store(tmp_path):
    path = tmp_path / "rollups.db"
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
            "INSERT INTO users VALUES (1, 'user@example.test', 'User', 'x', 'user', ?) ",
            ("2026-08-01T00:00:00+00:00",),
        )
        conn.execute(
            "INSERT INTO users VALUES (2, 'second@example.test', 'Second', 'x', 'user', ?) ",
            ("2026-08-01T00:00:00+00:00",),
        )
    analytics = AnalyticsStore(path)
    return analytics, AnalyticsRollupStore(analytics)


def test_rollup_day_is_idempotent_and_contains_no_user_dimension(tmp_path):
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:backtest_completed:run-1",
        source_record_type="run",
        source_record_id="run-1",
        occurred_at=at,
    )

    first = rollup_day(date(2026, 8, 25), store=rollups)
    second = rollup_day(date(2026, 8, 25), store=rollups)
    stored = rollups.list_rollups(
        start=date(2026, 8, 25),
        end=date(2026, 8, 26),
    )

    assert first == second
    assert any(
        row.metric_name == "terminal_completed" and row.value_count == 1
        for row in stored
    )
    assert all("user" not in row.model_dump() for row in stored)


def test_rollup_records_platform_cost_as_micro_usd(tmp_path):
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    service.record_server_event(
        event_name="model_usage_recorded",
        user_id=1,
        source_event_id="resource:model_usage_recorded:run-1:0",
        source_record_type="run",
        source_record_id="run-1",
        billing_mode="platform_credits",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        properties={
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_micro_usd": 1_250_000,
        },
        occurred_at=at,
    )

    rollup_day(date(2026, 8, 25), store=rollups)
    cost = next(
        row
        for row in rollups.list_rollups(
            start=date(2026, 8, 25),
            end=date(2026, 8, 26),
        )
        if row.metric_name == "platform_model_cost_usd"
    )

    assert cost.value_sum_micro == 1_250_000
    assert cost.billing_mode == "platform_credits"


def test_lifecycle_rollup_is_bounded_and_preserves_other_metrics(tmp_path):
    analytics, rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    day = date(2026, 8, 25)
    updated_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
    rollups.replace_day(
        day,
        [
            DailyRollup(
                rollup_date=day,
                metric_name="completed_runs",
                value_count=7,
                updated_at=updated_at,
            )
        ],
    )
    for snapshot_date, user_id, segment in (
        (day - timedelta(days=1), 1, "new"),
        (day - timedelta(days=1), 2, "onboarding"),
        (day, 1, "growing"),
        (day, 2, "onboarding"),
    ):
        values.upsert_daily_snapshot(
            UserLifecycleDailySnapshot(
                snapshot_date=snapshot_date,
                user_id=user_id,
                lifecycle_segment=segment,
                lifecycle_reason_code=f"{segment}_reason",
                data_quality="complete",
                calculated_at=updated_at,
            )
        )

    first = rollup_lifecycle_day(day, store=values)
    second = rollup_lifecycle_day(day, store=values)
    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))

    assert first == second
    assert any(
        row.metric_name == "completed_runs" and row.value_count == 7 for row in stored
    )
    assert {
        (row.user_state, row.value_count)
        for row in stored
        if row.metric_name == "lifecycle_segment_count"
    } == {("growing", 1), ("onboarding", 1)}
    assert {
        (row.event_name, row.user_state, row.value_count)
        for row in stored
        if row.metric_name == "lifecycle_transition"
    } == {("new", "growing", 1)}
    assert all("user_id" not in row.model_dump() for row in first)

    rollup_day(day, store=rollups)
    rebuilt = rollups.list_rollups(start=day, end=day + timedelta(days=1))
    assert any(row.metric_name == "lifecycle_transition" for row in rebuilt)
    assert any(row.metric_name == "lifecycle_segment_count" for row in rebuilt)
