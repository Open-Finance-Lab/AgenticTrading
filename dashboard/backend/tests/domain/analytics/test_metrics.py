"""Deterministic Analytics metric formulas at fixed UTC boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dashboard.backend.domain.analytics.metrics import (
    backtest_success_rate,
    calculate_overview_metrics,
    repeat_run_rate,
)
from dashboard.backend.domain.analytics.models import (
    EVENT_GROUP_BY_NAME,
    AnalyticsEventRecord,
)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def _event(name: str, user_id: int, at: datetime, **kwargs) -> AnalyticsEventRecord:
    if name in {"page_viewed", "page_hidden", "session_heartbeat"}:
        return AnalyticsEventRecord(
            event_id=str(uuid4()),
            event_name=name,
            event_group=EVENT_GROUP_BY_NAME[name],
            user_id=user_id,
            session_id=str(uuid4()),
            occurred_at=at,
            received_at=at,
            event_source="frontend",
            page_view=kwargs.pop("page_view", "home"),
            properties=kwargs.pop("properties", {}),
            **kwargs,
        )
    return AnalyticsEventRecord(
        event_id=str(uuid4()),
        event_name=name,
        event_group=EVENT_GROUP_BY_NAME[name],
        user_id=user_id,
        occurred_at=at,
        received_at=at,
        event_source="server",
        source_event_id=f"test:{name}:{user_id}:{uuid4()}",
        properties=kwargs.pop("properties", {}),
        **kwargs,
    )


def test_active_users_excludes_unattended_heartbeat():
    events = [
        _event("page_viewed", 1, NOW - timedelta(days=1), page_view="home"),
        AnalyticsEventRecord(
            event_id=str(uuid4()),
            event_name="session_heartbeat",
            event_group="experience",
            user_id=2,
            session_id=str(uuid4()),
            occurred_at=NOW - timedelta(days=1),
            received_at=NOW - timedelta(days=1),
            event_source="frontend",
            page_view="home",
            properties={"visible_ms": 1000},
        ),
    ]

    result = calculate_overview_metrics(
        events,
        start=NOW - timedelta(days=30),
        end=NOW,
    )

    assert result.active_users_7d == 1


def test_first_success_conversion_excludes_immature_cohort():
    events = [
        _event("account_signed_up", 1, NOW - timedelta(days=8)),
        _event("account_signed_up", 2, NOW - timedelta(days=2)),
        _event("backtest_completed", 1, NOW - timedelta(days=7, hours=1)),
    ]

    result = calculate_overview_metrics(
        events,
        start=NOW - timedelta(days=30),
        end=NOW,
    )

    assert result.mature_signup_cohort_users == 1
    assert result.first_success_within_7d_users == 1
    assert result.first_success_conversion == 1.0


def test_success_and_repeat_rate_formulas():
    assert backtest_success_rate(completed=1, failed=1) == 0.5
    assert backtest_success_rate(completed=0, failed=0) is None
    assert repeat_run_rate(users_with_first_success=2, users_with_repeat_success=1) == 0.5
    assert repeat_run_rate(users_with_first_success=0, users_with_repeat_success=0) is None


def test_cancelled_runs_and_byok_cost_are_excluded():
    events = [
        _event("backtest_completed", 1, NOW - timedelta(days=1)),
        _event("backtest_failed", 2, NOW - timedelta(days=1)),
        _event("backtest_cancelled", 3, NOW - timedelta(days=1)),
        _event(
            "model_usage_recorded",
            1,
            NOW - timedelta(days=1),
            billing_mode="platform_credits",
            properties={
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_micro_usd": 1_250_000,
            },
        ),
        _event(
            "model_usage_recorded",
            2,
            NOW - timedelta(days=1),
            billing_mode="byok",
            properties={
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_micro_usd": 0,
            },
        ),
    ]

    result = calculate_overview_metrics(
        events,
        start=NOW - timedelta(days=30),
        end=NOW,
    )

    assert result.backtest_success_rate == 0.5
    assert result.platform_model_cost_usd == 1.25
    assert result.input_tokens == 200
    assert result.output_tokens == 100


def test_excluded_users_do_not_contribute():
    events = [
        _event("backtest_completed", 1, NOW - timedelta(days=1)),
        _event("backtest_failed", 2, NOW - timedelta(days=1)),
    ]

    result = calculate_overview_metrics(
        events,
        start=NOW - timedelta(days=30),
        end=NOW,
        excluded_user_ids={2},
    )

    assert result.completed_runs == 1
    assert result.failed_runs == 0
    assert result.backtest_success_rate == 1.0
