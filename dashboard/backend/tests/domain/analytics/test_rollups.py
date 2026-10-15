"""Daily Analytics rollups use bounded dimensions and idempotent upserts."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    DailyRollup,
    rollup_current_day,
    rollup_lifecycle_day,
    rollup_day,
)
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import ValueAnalyticsStore


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


def test_lifecycle_rollup_reads_facts_and_transitions_and_writes_tier_and_group_families(tmp_path):
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact, transition

    analytics, rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    day = date(2026, 8, 25)
    updated_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
    rollups.replace_day(
        day, [DailyRollup(rollup_date=day, metric_name="completed_runs", value_count=7, updated_at=updated_at)]
    )
    values.upsert_daily_facts(
        [
            fact(1, day, lifecycle_segment="growing", tier="starter", user_group="organic",
                 runs_completed=2, operator_cost_micro=100_000, own_spend_micro=5_000),
            fact(2, day, lifecycle_segment="onboarding", tier="unpaid", user_group="organic",
                 runs_completed=0, operator_cost_micro=0, own_spend_micro=0, data_quality="partial"),
        ]
    )
    values.append_lifecycle_transitions([transition(1, day, "new", "growing")])

    first = rollup_lifecycle_day(day, store=values)
    second = rollup_lifecycle_day(day, store=values)
    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))

    assert first == second
    assert any(row.metric_name == "completed_runs" and row.value_count == 7 for row in stored)
    assert {(r.user_state, r.value_count, r.outcome) for r in stored if r.metric_name == "lifecycle_segment_count"} == {
        ("growing", 1, "complete"), ("onboarding", 1, "partial")
    }
    assert {(r.event_name, r.user_state, r.value_count) for r in stored if r.metric_name == "lifecycle_transition"} == {
        ("new", "growing", 1)
    }
    assert {(r.user_state, r.value_count) for r in stored if r.metric_name == "runs_by_tier"} == {("starter", 2), ("unpaid", 0)}
    assert {(r.user_state, r.value_count) for r in stored if r.metric_name == "runs_by_user_group"} == {("organic", 2)}
    assert {(r.user_state, r.value_count, r.value_sum_micro) for r in stored if r.metric_name == "operator_cost_by_tier"} == {
        ("starter", 1, 100_000), ("unpaid", 1, 0)
    }
    assert {(r.user_state, r.value_sum_micro) for r in stored if r.metric_name == "own_spend_by_user_group"} == {("organic", 5_000)}
    assert all("user_id" not in row.model_dump() for row in first)

    rollup_day(day, store=rollups)
    rebuilt = rollups.list_rollups(start=day, end=day + timedelta(days=1))
    for name in ("lifecycle_transition", "lifecycle_segment_count", "runs_by_tier", "own_spend_by_user_group"):
        assert any(row.metric_name == name for row in rebuilt), name


def test_lifecycle_rollup_of_a_day_with_no_facts_keeps_what_was_already_written(tmp_path):
    """A retention re-run after a day's facts expired must not blank its history."""
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact

    analytics, rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    day = date(2026, 8, 25)
    values.upsert_daily_facts([fact(1, day, tier="starter", user_group="organic", runs_completed=2)])
    written = rollup_lifecycle_day(day, store=values)
    assert written

    assert values.expire_facts_for_date(day) == 1
    assert rollup_lifecycle_day(day, store=values) == ()

    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))
    assert {(r.metric_name, r.user_state, r.value_count) for r in stored} == {
        (r.metric_name, r.user_state, r.value_count) for r in written
    }


def test_replace_lifecycle_rollups_rejects_malformed_dimension_rows(tmp_path):
    import pytest

    analytics, _rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    day = date(2026, 8, 25)
    at = datetime(2026, 8, 26, tzinfo=timezone.utc)

    def row(**overrides):
        fields = dict(
            rollup_date=day, metric_name="own_spend_by_tier", user_state="starter",
            outcome="complete", value_count=1, value_sum_micro=5, updated_at=at,
        )
        fields.update(overrides)
        return DailyRollup(**fields)

    values.replace_lifecycle_rollups(day, [row()])  # the well-formed shape is accepted
    for bad in (
        row(user_state=""),                                  # a family needs its dimension value
        row(event_name="growing"),                           # event_name is not part of this shape
        row(value_sum_micro=-1),                             # cost sums are non-negative
        row(metric_name="runs_by_tier", value_sum_micro=5),  # runs families carry a count only
        row(outcome="unknown"),
        row(metric_name="completed_runs"),                   # not a lifecycle-owned metric
    ):
        with pytest.raises(ValueError):
            values.replace_lifecycle_rollups(day, [bad])


def test_rollup_day_writes_distinct_activation_users_per_event(tmp_path):
    analytics, rollups = _store(tmp_path)
    service = AnalyticsService(analytics)
    day = date(2026, 8, 25)
    for index, (user_id, name, record_type) in enumerate(
        (
            (1, "account_signed_up", "user"),
            (2, "account_signed_up", "user"),
            (1, "agent_created", "agent"),
            (1, "agent_created", "agent"),
            (1, "backtest_completed", "run"),
        )
    ):
        service.record_server_event(
            event_name=name,
            user_id=user_id,
            source_event_id=f"{record_type}:{name}:{index}",
            source_record_type=record_type,
            source_record_id=str(index),
            occurred_at=datetime(2026, 8, 25, 10 + index, 0, tzinfo=timezone.utc),
        )

    rollup_day(day, store=rollups)
    stored = rollups.list_rollups(start=day, end=day + timedelta(days=1))

    assert {
        (row.event_name, row.value_count)
        for row in stored
        if row.metric_name == "activation_users"
    } == {("account_signed_up", 2), ("agent_created", 1), ("backtest_completed", 1)}


def test_rollup_current_day_reads_only_the_current_day(tmp_path):
    analytics, rollups = _store(tmp_path)
    windows = []
    original = rollups.list_events

    def recording(**kwargs):
        windows.append((kwargs["start"], kwargs["end"]))
        return original(**kwargs)

    rollups.list_events = recording
    now = datetime(2026, 8, 25, 15, 30, tzinfo=timezone.utc)

    rollup_current_day(store=rollups, now=now)

    assert windows == [(datetime(2026, 8, 25, tzinfo=timezone.utc), now)]


def test_lifecycle_rollup_warns_when_a_day_has_transitions_but_no_facts(tmp_path, capsys):
    """Unreachable while both expire together; if it happens, the lost movement
    must reach the service log rather than vanish."""
    from dashboard.backend.tests.domain.analytics.test_fact_reads import transition

    analytics, rollups = _store(tmp_path)
    values = ValueAnalyticsStore(
        analytics, credits_base=object(), provider_base=object(), agent_base=object(), run_base=object()
    )
    day = date(2026, 8, 25)
    values.append_lifecycle_transitions([transition(1, day, "new", "growing")])

    assert rollup_lifecycle_day(day, store=values) == ()

    out = capsys.readouterr().out
    assert (
        "WARNING: analytics.rollup.lifecycle_day_without_facts "
        "day=2026-08-25 transitions=1"
    ) in out
    assert rollups.list_rollups(start=day, end=day + timedelta(days=1)) == []
