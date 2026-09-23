"""Lifecycle from stored facts, with the rules unchanged."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.lifecycle import calculate_lifecycle
from dashboard.backend.domain.analytics.lifecycle_reads import build_lifecycle_inputs
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    RecentFactTotals,
    UserActivity,
    build_value_analytics_store,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


def _totals(**overrides):
    base = dict(
        active_days=0,
        successful_backtests=0,
        runs_requested=0,
        runs_completed=0,
        runs_failed=0,
        runs_cancelled=0,
        operator_cost_micro=0,
        own_spend_micro=0,
        days_present=30,
    )
    base.update(overrides)
    return RecentFactTotals(**base)


def test_core_needs_three_active_days_and_three_successes():
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=20),
        last_meaningful_activity_at=NOW - timedelta(days=1),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=3),
        as_of=NOW,
    )

    assert calculate_lifecycle(inputs, NOW).segment == "core"

    below = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=3, successful_backtests=2),
        as_of=NOW,
    )
    assert calculate_lifecycle(below, NOW).segment == "growing"


def test_a_user_with_no_activity_row_falls_back_to_signup():
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=2),
        activity=None,
        totals=None,
        as_of=NOW,
    )
    result = calculate_lifecycle(inputs, NOW)

    assert result.segment == "new"
    assert result.active_days_30d == 0


def test_evidence_newer_than_as_of_is_clamped_not_raised():
    """Serving a past day from a table that only knows the present.

    `user_activity` is one row per user, overwritten in place. Asked for
    the end of yesterday while holding a timestamp from this morning,
    `calculate_lifecycle` raises rather than guessing -- so the clamp has to
    happen here, and the fact table is where the answer for that past day
    actually lives.
    """
    end_of_yesterday = datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc)
    activity = UserActivity(
        user_id=1,
        activated_at=NOW,  # activated today
        last_meaningful_activity_at=NOW,  # active today
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=1, last_active_date=date(2026, 9, 3)),
        as_of=end_of_yesterday,
    )

    # Not activated as of yesterday, and last active on the 3rd -- eight
    # whole UTC days before the day being computed, i.e. genuinely at risk.
    # (The plan drafted this fixture with the 4th, which is only seven days
    # back and classifies onboarding under lifecycle.py's >=8 threshold.)
    assert inputs.first_successful_backtest_at is None
    assert inputs.last_meaningful_activity_at == datetime(
        2026, 9, 3, tzinfo=timezone.utc
    )
    result = calculate_lifecycle(inputs, end_of_yesterday)
    assert result.segment == "at_risk"


def test_activity_inside_the_computed_day_beats_the_window_fallback():
    """A user active only on D is not dormant, and the window cannot say so.

    The daily job narrows the fact window to D-29..D-1, so a user whose first
    activity in a month lands on D has no row in it. Falling straight through
    to `last_active_date` reads None and classifies them Dormant on the very
    day they came back.
    """
    end_of_yesterday = datetime(2026, 9, 11, 23, 59, 59, tzinfo=timezone.utc)
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=200),
        last_meaningful_activity_at=NOW,  # newer than as_of, so clamped
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=300),
        activity=activity,
        totals=_totals(active_days=1, last_active_date=None),
        as_of=end_of_yesterday,
        day_activity_at=datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc),
    )

    assert inputs.last_meaningful_activity_at == datetime(
        2026, 9, 11, 18, 0, tzinfo=timezone.utc
    )
    assert calculate_lifecycle(inputs, end_of_yesterday).segment == "growing"


def test_the_live_path_clamps_nothing():
    """as_of=now is the identity case, and must stay cost-free."""
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=3),
        last_meaningful_activity_at=NOW,
        updated_at=NOW,
    )

    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=60),
        activity=activity,
        totals=_totals(active_days=5, successful_backtests=5),
        as_of=NOW,
    )

    assert inputs.first_successful_backtest_at == activity.activated_at
    assert inputs.last_meaningful_activity_at == activity.last_meaningful_activity_at


def test_inactivity_crosses_at_risk_without_any_new_evidence():
    """Time alone moves the segment. That is why nothing needs recomputing."""
    activity = UserActivity(
        user_id=1,
        activated_at=NOW - timedelta(days=40),
        last_meaningful_activity_at=NOW - timedelta(days=7),
        updated_at=NOW,
    )
    inputs = build_lifecycle_inputs(
        1,
        created_at=NOW - timedelta(days=90),
        activity=activity,
        totals=_totals(),
        as_of=NOW,
    )

    assert calculate_lifecycle(inputs, NOW).segment == "growing"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=1)).segment == "at_risk"
    assert calculate_lifecycle(inputs, NOW + timedelta(days=23)).segment == "dormant"


def _fact_store(tmp_path):
    path = tmp_path / "facts.db"
    users = UserStore(db_path=path)
    first = int(users.create_user("a@example.test", "A", "SecurePass1!")["id"])
    second = int(users.create_user("b@example.test", "B", "SecurePass1!")["id"])
    analytics = AnalyticsStore(db_path=path)
    store = build_value_analytics_store(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    with analytics._get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO user_daily_facts (
                snapshot_date, user_id, lifecycle_segment, lifecycle_reason_code,
                operational_state, tier, user_group, active, runs_requested,
                runs_completed, runs_failed, runs_cancelled, operator_cost_micro,
                own_spend_micro, data_quality, calculated_at
            ) VALUES (?, ?, 'growing', 'growing_activated_below_core_threshold',
                      'healthy', 'unpaid', 'unknown', ?, ?, ?, ?, ?, ?, ?,
                      'complete', '2026-09-12T00:05:00+00:00')
            """,
            [
                ("2026-09-01", first, 1, 2, 1, 1, 0, 1_000_000, 0),
                ("2026-09-05", first, 1, 1, 1, 0, 0, 0, 250_000),
                ("2026-09-09", first, 0, 0, 0, 0, 0, 0, 0),
                ("2026-09-11", first, 1, 3, 2, 0, 1, 500_000, 0),
                ("2026-09-11", second, 1, 1, 1, 0, 0, 0, 0),
                ("2026-08-01", first, 1, 9, 9, 0, 0, 0, 0),  # outside the window
            ],
        )
    return store, first, second


def test_sum_recent_facts_groups_the_window_in_one_statement(tmp_path):
    store, first, second = _fact_store(tmp_path)

    totals = store.sum_recent_facts(
        [first, second], start=date(2026, 8, 13), end=date(2026, 9, 11)
    )

    assert totals[first] == RecentFactTotals(
        active_days=3,
        successful_backtests=4,
        runs_requested=6,
        runs_completed=4,
        runs_failed=1,
        runs_cancelled=1,
        operator_cost_micro=1_500_000,
        own_spend_micro=250_000,
        days_present=4,
        last_active_date=date(2026, 9, 11),
    )
    assert totals[second].active_days == 1
    assert totals[second].days_present == 1
    # None means the whole population -- the daily job's shape.
    assert store.sum_recent_facts(
        None, start=date(2026, 8, 13), end=date(2026, 9, 11)
    ) == totals
    assert store.sum_recent_facts([], start=date(2026, 8, 13), end=date(2026, 9, 11)) == {}
