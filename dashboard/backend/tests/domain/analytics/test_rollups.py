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


def _record_usage(service, index, billing_mode, properties, at, model_id="openai/gpt-5.5"):
    service.record_server_event(
        event_name="model_usage_recorded",
        user_id=1,
        source_event_id=f"resource:model_usage_recorded:run-1:{index}",
        source_record_type="run",
        source_record_id="run-1",
        billing_mode=billing_mode,
        provider_id="openrouter",
        model_id=model_id,
        properties={"input_tokens": 1, "output_tokens": 1, **properties},
        occurred_at=at + timedelta(minutes=index),
    )


def _usage_rows(rollups, metric_name):
    return [
        (row.billing_mode, row.provider_id, row.model_id, row.value_sum_micro, row.value_count)
        for row in rollups.list_rollups(start=date(2026, 8, 25), end=date(2026, 8, 26))
        if row.metric_name == metric_name
    ]


def test_rollup_keeps_the_byok_estimate_out_of_platform_cost(tmp_path):
    """A BYOK call's list-price estimate gets its own metric per provider/model
    and never reaches platform_model_cost_usd, neither the day's total nor any
    per-model row. A BYOK call with no estimate is counted, not summed as 0."""
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    for index, (billing_mode, properties) in enumerate(
        (
            ("platform_credits", {"cost_micro_usd": 1_000_000}),
            ("byok", {"cost_micro_usd": 0, "estimated_cost_micro_usd": 420_000}),
            ("byok", {"cost_micro_usd": 0, "estimated_cost_micro_usd": 80_000}),
            ("byok", {"cost_micro_usd": 0}),
        )
    ):
        _record_usage(service, index, billing_mode, properties, at)

    rollup_day(date(2026, 8, 25), store=rollups)

    assert {
        (row[1], row[2]): row[3] for row in _usage_rows(rollups, "platform_model_cost_usd")
    } == {("", ""): 1_000_000, ("openrouter", "openai/gpt-5.5"): 1_000_000}
    assert _usage_rows(rollups, "byok_estimated_cost_usd") == [
        ("byok", "openrouter", "openai/gpt-5.5", 500_000, 0)
    ]
    assert _usage_rows(rollups, "byok_unpriced_calls") == [
        ("byok", "openrouter", "openai/gpt-5.5", 0, 1)
    ]


def test_rollup_writes_the_unpriced_count_even_when_it_is_zero(tmp_path):
    """Its presence is what tells a reader the day was rolled up after unpriced
    calls began to be counted; a missing row means an older rollup."""
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    _record_usage(
        service, 0, "byok", {"cost_micro_usd": 0, "estimated_cost_micro_usd": 5}, at
    )

    rollup_day(date(2026, 8, 25), store=rollups)

    assert _usage_rows(rollups, "byok_unpriced_calls") == [
        ("byok", "openrouter", "openai/gpt-5.5", 0, 0)
    ]


def test_rollup_reads_the_pr_572_era_byok_estimate_from_cost_micro_usd(tmp_path):
    """Between PR #572 and the property split, a BYOK event carried its estimate
    in cost_micro_usd. Rolled up now it still reads as an estimate -- never as
    platform cost -- re-priced from its own tokens at the listed rate, because
    #572 priced an unlisted model at the $1/$5 fallback: that call reads as
    unpriced, like a zero from before #572."""
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    tokens = {"input_tokens": 1_000, "output_tokens": 100}
    _record_usage(service, 0, "byok", {**tokens, "cost_micro_usd": 30_000}, at)
    _record_usage(service, 1, "byok", {"cost_micro_usd": 0}, at)
    # gpt-4.1-nano is not listed; #572's snapshot priced it at $1/$5.
    _record_usage(
        service, 2, "byok", {**tokens, "cost_micro_usd": 1_500}, at,
        model_id="openai/gpt-4.1-nano",
    )

    rollup_day(date(2026, 8, 25), store=rollups)

    # Only the day's undimensioned platform total, and it is zero.
    assert _usage_rows(rollups, "platform_model_cost_usd") == [
        ("platform_credits", "", "", 0, 0)
    ]
    # gpt-5.5 at $5/$30 per million: 1,000 in + 100 out = $0.008.
    assert _usage_rows(rollups, "byok_estimated_cost_usd") == [
        ("byok", "openrouter", "openai/gpt-4.1-nano", 0, 0),
        ("byok", "openrouter", "openai/gpt-5.5", 8_000, 0),
    ]
    assert _usage_rows(rollups, "byok_unpriced_calls") == [
        ("byok", "openrouter", "openai/gpt-4.1-nano", 0, 1),
        ("byok", "openrouter", "openai/gpt-5.5", 0, 1),
    ]


def test_rollup_survives_a_stored_property_this_build_does_not_know(tmp_path):
    """A stored event is validated again on read. When that was strict, one
    event carrying a property from a later build -- read after a revert, or by
    the old instance mid-deploy -- failed the rollup of its whole day."""
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    at = datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc)
    _record_usage(service, 0, "platform_credits", {"cost_micro_usd": 250}, at)
    with sqlite3.connect(analytics.db_path) as conn:
        conn.execute(
            "UPDATE analytics_events SET properties_json = "
            "json_set(properties_json, '$.from_a_later_build', 1)"
        )

    rollup_day(date(2026, 8, 25), store=rollups)

    assert ("platform_credits", "openrouter", "openai/gpt-5.5", 250, 0) in _usage_rows(
        rollups, "platform_model_cost_usd"
    )


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
