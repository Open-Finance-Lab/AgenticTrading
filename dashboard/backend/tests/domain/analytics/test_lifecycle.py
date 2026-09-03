"""Pure user-value lifecycle, operational, and commercial rules."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from dashboard.backend.domain.analytics.lifecycle import (
    LifecycleInputs,
    OperationalSignals,
    activation_cohort_week,
    calculate_lifecycle,
    calculate_operational_state,
    commercial_tier,
    is_lifecycle_activity,
)
from dashboard.backend.domain.analytics.models import (
    EVENT_GROUP_BY_NAME,
    AnalyticsEventRecord,
)


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _inputs(
    *,
    account_age_days: int = 60,
    inactive_days: int = 0,
    activated: bool = False,
    active_days_30d: int = 1,
    successful_backtests_30d: int = 0,
) -> LifecycleInputs:
    return LifecycleInputs(
        user_id=1,
        created_at=NOW - timedelta(days=account_age_days),
        first_successful_backtest_at=(
            NOW - timedelta(days=max(15, inactive_days + 5)) if activated else None
        ),
        last_meaningful_activity_at=NOW - timedelta(days=inactive_days),
        active_days_30d=active_days_30d,
        successful_backtests_30d=successful_backtests_30d,
    )


def _event(name: str, **kwargs) -> AnalyticsEventRecord:
    occurred_at = kwargs.pop("occurred_at", NOW)
    if name in {"page_viewed", "page_hidden", "session_heartbeat"}:
        return AnalyticsEventRecord(
            event_id=str(uuid4()),
            event_name=name,
            event_group=EVENT_GROUP_BY_NAME[name],
            user_id=1,
            session_id=str(uuid4()),
            occurred_at=occurred_at,
            received_at=occurred_at,
            event_source="frontend",
            page_view="home",
            properties=kwargs.pop("properties", {}),
            **kwargs,
        )
    return AnalyticsEventRecord(
        event_id=str(uuid4()),
        event_name=name,
        event_group=EVENT_GROUP_BY_NAME[name],
        user_id=1,
        occurred_at=occurred_at,
        received_at=occurred_at,
        event_source="server",
        source_event_id=f"test:{name}:{uuid4()}",
        properties=kwargs.pop("properties", {}),
        **kwargs,
    )


def test_lifecycle_boundaries_use_utc_dates():
    assert calculate_lifecycle(_inputs(account_age_days=6), NOW).segment == "new"
    assert calculate_lifecycle(_inputs(account_age_days=7), NOW).segment == "onboarding"
    assert calculate_lifecycle(_inputs(inactive_days=7), NOW).segment == "onboarding"
    assert calculate_lifecycle(_inputs(inactive_days=8), NOW).segment == "at_risk"
    assert calculate_lifecycle(_inputs(inactive_days=29), NOW).segment == "at_risk"
    assert calculate_lifecycle(_inputs(inactive_days=30), NOW).segment == "dormant"


def test_lifecycle_dates_are_normalized_to_utc_before_day_comparison():
    west = timezone(timedelta(hours=-7))
    inputs = LifecycleInputs(
        user_id=1,
        created_at=datetime(2026, 8, 27, 23, 30, tzinfo=west),
        last_meaningful_activity_at=datetime(2026, 9, 2, 23, 30, tzinfo=west),
        active_days_30d=1,
        successful_backtests_30d=0,
    )

    result = calculate_lifecycle(inputs, NOW)

    assert result.segment == "new"
    assert result.inactive_days == 0
    assert result.calculated_at == NOW


def test_core_requires_both_trailing_thresholds():
    assert (
        calculate_lifecycle(
            _inputs(
                activated=True,
                active_days_30d=3,
                successful_backtests_30d=3,
            ),
            NOW,
        ).segment
        == "core"
    )
    assert (
        calculate_lifecycle(
            _inputs(
                activated=True,
                active_days_30d=2,
                successful_backtests_30d=3,
            ),
            NOW,
        ).segment
        == "growing"
    )
    assert (
        calculate_lifecycle(
            _inputs(
                activated=True,
                active_days_30d=3,
                successful_backtests_30d=2,
            ),
            NOW,
        ).segment
        == "growing"
    )


@pytest.mark.parametrize(
    "event_name",
    [
        "agent_created",
        "agent_updated",
        "credential_saved",
        "credential_verified",
        "credential_reverified",
        "credential_defaulted",
        "backtest_requested",
        "backtest_queued",
        "backtest_started",
        "backtest_completed",
        "backtest_failed",
        "backtest_cancelled",
        "model_usage_recorded",
        "credits_reserved",
        "credits_settled",
    ],
)
def test_intentional_setup_run_and_consumption_events_are_lifecycle_activity(
    event_name,
):
    properties = {}
    if event_name == "model_usage_recorded":
        properties = {"input_tokens": 1, "output_tokens": 1, "cost_micro_usd": 1}
    elif event_name in {"credits_reserved", "credits_settled"}:
        properties = {"amount_micro": 1, "bucket": "grant"}

    assert is_lifecycle_activity(_event(event_name, properties=properties)) is True


@pytest.mark.parametrize(
    "event_name",
    [
        "page_viewed",
        "page_hidden",
        "session_heartbeat",
        "account_signed_up",
        "authenticated_session_started",
        "agent_deleted",
        "credential_revoked",
        "credits_refunded",
        "safe_error_recorded",
    ],
)
def test_passive_automatic_or_nonqualifying_events_are_not_lifecycle_activity(
    event_name,
):
    properties = (
        {"amount_micro": 1, "bucket": "grant"}
        if event_name == "credits_refunded"
        else {}
    )
    assert is_lifecycle_activity(_event(event_name, properties=properties)) is False


@pytest.mark.parametrize("event_name", ["account_signed_in", "admin_grant_assigned"])
def test_unknown_login_or_admin_grant_names_are_not_lifecycle_activity(event_name):
    assert is_lifecycle_activity(SimpleNamespace(event_name=event_name)) is False


def test_inactive_reason_distinguishes_activation_history_and_evidence_is_safe():
    never = calculate_lifecycle(
        _inputs(inactive_days=30, activated=False),
        NOW,
    )
    previous = calculate_lifecycle(
        _inputs(inactive_days=30, activated=True),
        NOW,
    )

    assert never.reason_code == "dormant_never_activated"
    assert previous.reason_code == "dormant_previously_activated"
    assert "30 days ago" in " ".join(previous.evidence)
    assert "2026-07-30" in " ".join(previous.evidence)
    assert all("properties" not in item for item in previous.evidence)


def test_operational_state_is_independent_and_blocked_has_precedence():
    operational = calculate_operational_state(
        OperationalSignals(
            user_id=1,
            account_restricted=True,
            failed_terminal_runs_24h=3,
            run_beyond_safe_deadline=True,
        ),
        NOW,
    )

    assert operational.state == "blocked"
    assert operational.reason_code == "account_restricted"


@pytest.mark.parametrize(
    ("signals", "state", "reason_code"),
    [
        (
            {"usable_billing_lane": False},
            "blocked",
            "billing_lane_unavailable",
        ),
        (
            {"selected_provider_enabled": False},
            "blocked",
            "provider_disabled",
        ),
        (
            {"default_credential_status": "invalid"},
            "needs_attention",
            "invalid_default_credential",
        ),
        (
            {"failed_terminal_runs_24h": 3},
            "needs_attention",
            "three_consecutive_failed_runs",
        ),
        (
            {"run_beyond_safe_deadline": True},
            "needs_attention",
            "run_deadline_exceeded",
        ),
        ({}, "healthy", "no_supported_issue"),
    ],
)
def test_operational_reason_precedence(signals, state, reason_code):
    result = calculate_operational_state(
        OperationalSignals(user_id=1, **signals),
        NOW,
    )

    assert result.state == state
    assert result.reason_code == reason_code
    assert result.evidence


@pytest.mark.parametrize(
    ("net_purchased_micro", "expected"),
    [
        (-1, "unpaid"),
        (0, "unpaid"),
        (1, "starter"),
        (4_999_999, "starter"),
        (5_000_000, "invested"),
        (19_999_999, "invested"),
        (20_000_000, "high_value"),
    ],
)
def test_commercial_tier_boundaries(net_purchased_micro, expected):
    assert commercial_tier(net_purchased_micro) == expected


def test_activation_cohort_week_uses_utc_monday():
    west = timezone(timedelta(hours=-7))
    activated_at = datetime(2026, 8, 30, 23, 30, tzinfo=west)

    assert activation_cohort_week(activated_at) == date(2026, 8, 31)


@pytest.mark.parametrize(
    "value",
    [
        LifecycleInputs(
            user_id=1,
            created_at=datetime(2026, 9, 1),
            active_days_30d=0,
            successful_backtests_30d=0,
        ),
        LifecycleInputs(
            user_id=1,
            created_at=NOW,
            last_meaningful_activity_at=datetime(2026, 9, 2),
            active_days_30d=1,
            successful_backtests_30d=0,
        ),
    ],
)
def test_lifecycle_rejects_naive_input_dates(value):
    with pytest.raises(ValueError, match="timezone"):
        calculate_lifecycle(value, NOW)


def test_rule_calculations_reject_naive_calculation_times():
    with pytest.raises(ValueError, match="as_of must include a timezone"):
        calculate_lifecycle(_inputs(), datetime(2026, 9, 3, 12, 0))
    with pytest.raises(ValueError, match="as_of must include a timezone"):
        calculate_operational_state(
            OperationalSignals(user_id=1),
            datetime(2026, 9, 3, 12, 0),
        )
    with pytest.raises(ValueError, match="activated_at must include a timezone"):
        activation_cohort_week(datetime(2026, 9, 3, 12, 0))
