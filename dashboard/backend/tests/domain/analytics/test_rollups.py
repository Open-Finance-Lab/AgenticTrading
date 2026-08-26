"""Daily Analytics rollups use bounded dimensions and idempotent upserts."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    rollup_day,
)
from dashboard.backend.domain.analytics.service import AnalyticsService


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
