"""Display-safe query composition for Admin user-value analytics."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from dashboard.backend.domain.analytics.lifecycle import commercial_tier
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsStateSummary,
    AnalyticsUserProfile,
)
from dashboard.backend.domain.analytics.value_queries import (
    _MAX_HISTORY_SCAN_DAYS,
    _MOVEMENT_WINDOWS,
    UserValueFilters,
    ValueAnalyticsQueryService,
)
from dashboard.backend.domain.analytics.value_repository import (
    CommercialValueFact,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _user(user_id: int) -> dict[str, object]:
    return {
        "id": user_id,
        "display_name": f"Value User {user_id}",
        "email": f"value-{user_id}@example.test",
        "created_at": (NOW - timedelta(days=90)).isoformat(),
    }


def _snapshot(
    user_id: int,
    *,
    lifecycle: str = "core",
    operational: str = "healthy",
    activated_at: datetime | None = None,
    inactive_days: int = 2,
) -> UserValueSnapshot:
    activated = activated_at if activated_at is not None else NOW - timedelta(days=60)
    return UserValueSnapshot(
        user_id=user_id,
        lifecycle_segment=lifecycle,
        lifecycle_reason_code=f"{lifecycle}_synthetic_reason",
        lifecycle_reason=f"Synthetic {lifecycle} lifecycle reason.",
        lifecycle_evidence=("Synthetic lifecycle evidence.",),
        operational_state=operational,
        operational_reason_code=f"{operational}_synthetic_reason",
        operational_reason=f"Synthetic {operational} operational reason.",
        operational_evidence=("Synthetic operational evidence.",),
        activated_at=(None if lifecycle == "onboarding" else activated),
        last_meaningful_activity_at=NOW - timedelta(days=inactive_days),
        inactive_days=inactive_days,
        active_days_30d=3 if lifecycle == "core" else 1,
        successful_backtests_30d=3 if lifecycle == "core" else 0,
        calculated_at=NOW,
    )


def _commercial(
    user_id: int,
    purchased: int = 0,
    *,
    period_purchased: int = 0,
    refunded: int = 0,
    consumed: int = 0,
    grants: int = 0,
    grant_balance: int = 0,
    purchased_balance: int = 0,
) -> CommercialValueFact:
    return CommercialValueFact(
        user_id=user_id,
        lifetime_net_purchased_micro=purchased,
        commercial_tier=commercial_tier(purchased),
        purchased_micro=period_purchased,
        refunded_micro=refunded,
        consumed_micro=consumed,
        admin_grant_activity_micro=grants,
        grant_available_micro=grant_balance,
        purchased_available_micro=purchased_balance,
        total_available_micro=grant_balance + purchased_balance,
    )


def _daily(
    user_id: int,
    day: date,
    lifecycle: str = "core",
    quality: str = "complete",
) -> UserLifecycleDailySnapshot:
    return UserLifecycleDailySnapshot(
        snapshot_date=day,
        user_id=user_id,
        lifecycle_segment=lifecycle,
        lifecycle_reason_code=f"{lifecycle}_synthetic_reason",
        data_quality=quality,
        calculated_at=NOW,
    )


class FakeUserStore:
    def __init__(self, users):
        self.users = list(users)

    def list_users_admin(self, *, limit, offset):
        return self.users[offset : offset + limit]


class FakeBaseStore:
    def __init__(self, excluded=()):
        self.excluded = set(excluded)

    def list_excluded_user_ids(self, *, include_admin_accounts):
        assert include_admin_accounts is True
        return set(self.excluded)


class FakeValueStore:
    def __init__(self, *, snapshots, commercial, daily=(), credit_activity=None):
        self.snapshots = dict(snapshots)
        self.commercial = dict(commercial)
        self.daily = list(daily)
        self.credit_activity = dict(credit_activity or {})
        self.commercial_windows = []
        self.daily_windows = []

    def list_current_snapshots(self, user_ids):
        return {
            user_id: self.snapshots[user_id]
            for user_id in user_ids
            if user_id in self.snapshots
        }

    def get_current_snapshot(self, user_id):
        return self.snapshots.get(user_id)

    def list_commercial_values(self, user_ids, *, start, end):
        self.commercial_windows.append((start, end))
        return {
            user_id: self.commercial[user_id]
            for user_id in user_ids
            if user_id in self.commercial
        }

    def list_daily_snapshots(self, *, start, end, user_ids=None):
        self.daily_windows.append((start, end))
        selected = None if user_ids is None else set(user_ids)
        return [
            row
            for row in self.daily
            if start <= row.snapshot_date < end
            and (selected is None or row.user_id in selected)
        ]

    def list_credit_activity(self, user_ids, *, start, end):
        return {
            user_id: tuple(
                value
                for value in self.credit_activity.get(user_id, ())
                if start <= value < end
            )
            for user_id in user_ids
        }


class FakeRollups:
    def __init__(self, events=(), rollups=()):
        self.events = list(events)
        self.rollups = list(rollups)

    def list_events(self, *, start, end, include_internal, user_id=None):
        assert include_internal is True
        return [
            event
            for event in self.events
            if start <= event.occurred_at < end
            and (user_id is None or event.user_id == user_id)
        ]

    def list_rollups(self, *, start, end):
        return [row for row in self.rollups if start <= row.rollup_date < end]


class FakeQueryStore:
    def __init__(self, *, events=(), rollups=(), legacy_status=None):
        self.rollups = FakeRollups(events, rollups)
        self.legacy_status = dict(legacy_status or {})

    def list_snapshots(self):
        return {
            user_id: SimpleNamespace(status=status)
            for user_id, status in self.legacy_status.items()
        }


class FakeLegacyService:
    def __init__(self, availability=None):
        self.profile_windows = []
        self.availability = availability or {"growth": True, "friction": True}

    def get_overview(self, *, filters, now):
        return SimpleNamespace(
            availability={
                name: SimpleNamespace(available=available)
                for name, available in self.availability.items()
            },
            backtest_success_rate=0.75,
            completed_runs=9,
            failed_runs=3,
            input_tokens=120,
            output_tokens=80,
            platform_model_cost_usd=0.25,
            top_failure_categories=[],
        )

    def get_user_profile(self, *, user_id, now, start=None, end=None):
        self.profile_windows.append((start, end))
        return AnalyticsUserProfile(
            user_id=user_id,
            display_name=f"Value User {user_id}",
            email=f"value-{user_id}@example.test",
            joined_at=NOW - timedelta(days=90),
            last_meaningful_activity=NOW - timedelta(days=2),
            state=AnalyticsStateSummary(
                status="active",
                reason_code="synthetic_active",
                human_readable_reason="Synthetic active state.",
                calculated_at=NOW,
            ),
            activation_milestones={},
            recent_footprint=[],
            run_summary={},
            billing_lane_mix={},
        )


def _service(
    *,
    snapshots,
    commercial=None,
    daily=(),
    events=(),
    rollups=(),
    excluded=(),
    legacy_availability=None,
):
    facts = commercial or {user_id: _commercial(user_id) for user_id in snapshots}
    value_store = FakeValueStore(
        snapshots=snapshots,
        commercial=facts,
        daily=daily,
    )
    legacy_service = FakeLegacyService(legacy_availability)
    service = ValueAnalyticsQueryService(
        store=FakeBaseStore(excluded),
        user_store=FakeUserStore([_user(user_id) for user_id in snapshots]),
        value_store=value_store,
        query_store=FakeQueryStore(events=events, rollups=rollups),
        legacy_service=legacy_service,
    )
    return service, value_store, legacy_service


def _activity(user_id: int, occurred_at: datetime):
    return SimpleNamespace(
        user_id=user_id,
        event_name="agent_updated",
        occurred_at=occurred_at,
    )


def test_date_filter_changes_history_not_current_lifecycle_identity():
    snapshots = {
        1: _snapshot(1),
        2: _snapshot(2, lifecycle="at_risk", inactive_days=12),
    }
    daily = [
        _daily(1, date(2026, 8, 25), "growing"),
        _daily(2, date(2026, 8, 25), "onboarding"),
        _daily(1, date(2026, 9, 2), "core"),
        _daily(2, date(2026, 9, 2), "at_risk"),
    ]
    service, _value_store, _legacy = _service(snapshots=snapshots, daily=daily)

    short = service.get_lifecycle(
        start=date(2026, 9, 1),
        end=date(2026, 9, 4),
        now=NOW,
    )
    long = service.get_lifecycle(
        start=date(2026, 8, 24),
        end=date(2026, 9, 4),
        now=NOW,
    )

    assert short.headline == long.headline
    assert short.segment_counts == long.segment_counts
    assert short.weekly_segments != long.weekly_segments


@pytest.mark.parametrize(
    ("movement_range", "granularity"),
    [("5d", "day"), ("1w", "day"), ("1m", "week"), ("1y", "month")],
)
def test_lifecycle_movement_returns_selected_range_and_granularity(
    movement_range, granularity
):
    snapshots = {1: _snapshot(1)}
    daily = [
        _daily(1, date(2025, 10, 1) + timedelta(days=offset), "core")
        for offset in range(365)
    ]
    service, _value_store, _legacy = _service(snapshots=snapshots, daily=daily)

    response = service.get_lifecycle(
        start=date(2026, 4, 5),
        end=date(2026, 10, 1),
        movement_range=movement_range,
        now=datetime(2026, 10, 1, 12, tzinfo=UTC),
    )

    assert response.movement_range == movement_range
    assert response.movement_granularity == granularity
    starts = [point.period_start for point in response.movement_segments]
    assert starts
    assert starts == sorted(set(starts))
    # Derived from the window rather than written down: a calendar bucket count
    # depends on where the window falls in the calendar, so a hard-coded ceiling
    # holds only for the dates this case happens to pick.
    window_days = _MOVEMENT_WINDOWS[movement_range][0]
    period_days = {"day": 1, "week": 7, "month": 28}[granularity]
    assert len(starts) <= window_days // period_days + 2
    assert all(
        date(2026, 10, 1) - timedelta(days=window_days) <= day < date(2026, 10, 1)
        for day in starts
    )


def test_retention_uses_nulls_for_immature_cells_and_weighted_mature_summary():
    first_week = date(2026, 7, 6)
    second_week = date(2026, 7, 13)
    immature_week = date(2026, 8, 24)
    snapshots = {
        1: _snapshot(1, activated_at=datetime(2026, 7, 6, 10, tzinfo=UTC)),
        2: _snapshot(2, activated_at=datetime(2026, 7, 7, 10, tzinfo=UTC)),
        3: _snapshot(3, activated_at=datetime(2026, 7, 13, 10, tzinfo=UTC)),
        4: _snapshot(4, activated_at=datetime(2026, 8, 24, 10, tzinfo=UTC)),
    }
    daily = []
    for cohort, members in ((first_week, (1, 2)), (second_week, (3,))):
        for target_week in (1, 2, 4):
            target_start = cohort + timedelta(days=target_week * 7)
            daily.extend(
                _daily(user_id, target_start + timedelta(days=day))
                for user_id in members
                for day in range(7)
            )
    events = [
        _activity(1, datetime(2026, 7, 14, 10, tzinfo=UTC)),
        _activity(3, datetime(2026, 7, 21, 10, tzinfo=UTC)),
    ]
    service, _value_store, _legacy = _service(
        snapshots=snapshots,
        daily=daily,
        events=events,
    )

    response = service.get_retention(
        start=date(2026, 7, 1),
        end=date(2026, 9, 1),
        now=NOW,
    )

    assert response.summary_week_1.mature is True
    assert response.summary_week_1.retained_users == 2
    assert response.summary_week_1.eligible_users == 3
    assert response.summary_week_1.rate == pytest.approx(2 / 3)
    immature = next(
        cohort for cohort in response.cohorts if cohort.cohort_week == immature_week
    )
    assert immature.week_1.mature is False
    assert immature.week_1.retained_users is None
    assert immature.week_1.eligible_users is None
    assert immature.week_1.rate is None


def test_missing_or_partial_retention_history_propagates_partial_quality():
    cohort = date(2026, 7, 6)
    snapshots = {
        1: _snapshot(1, activated_at=datetime(2026, 7, 6, 10, tzinfo=UTC)),
        2: _snapshot(2, activated_at=datetime(2026, 7, 7, 10, tzinfo=UTC)),
    }
    target_start = cohort + timedelta(days=7)
    daily = [_daily(1, target_start + timedelta(days=day)) for day in range(7)]
    daily.extend(
        _daily(2, target_start + timedelta(days=day), quality="partial")
        for day in range(6)
    )
    service, _value_store, _legacy = _service(snapshots=snapshots, daily=daily)

    response = service.get_retention(
        start=date(2026, 7, 1),
        end=date(2026, 7, 20),
        now=NOW,
    )

    assert response.cohorts[0].week_1.data_quality == "partial"
    assert response.summary_week_1.data_quality == "partial"
    assert response.availability.status == "partial"


def test_priority_order_is_group_value_inactivity_then_user_id():
    snapshots = {
        10: _snapshot(10, operational="blocked", inactive_days=3),
        11: _snapshot(11, operational="blocked", inactive_days=20),
        12: _snapshot(12, operational="blocked", inactive_days=8),
        13: _snapshot(13, operational="blocked", inactive_days=8),
        20: _snapshot(20, operational="needs_attention"),
        30: _snapshot(30, lifecycle="at_risk", inactive_days=15),
        40: _snapshot(40, lifecycle="onboarding", inactive_days=4),
        50: _snapshot(50, lifecycle="core"),
    }
    commercial = {
        10: _commercial(10, 6_000_000),
        11: _commercial(11, 4_000_000),
        12: _commercial(12, 3_000_000),
        13: _commercial(13, 3_000_000),
        20: _commercial(20, 30_000_000),
        30: _commercial(30, 30_000_000),
        40: _commercial(40, 30_000_000),
        50: _commercial(50, 30_000_000),
    }
    service, _value_store, _legacy = _service(
        snapshots=snapshots,
        commercial=commercial,
    )

    page = service.list_users(
        filters=UserValueFilters(priority=True),
        limit=25,
        offset=0,
        now=NOW,
    )

    assert [item.user_id for item in page.items] == [10, 11, 12, 13, 20, 30, 40]


def test_user_list_uses_injected_utc_day_for_commercial_window():
    service, value_store, _legacy = _service(snapshots={1: _snapshot(1)})

    service.list_users(
        filters=UserValueFilters(),
        limit=25,
        offset=0,
        now=NOW,
    )

    assert value_store.commercial_windows == [
        (
            datetime(2026, 8, 4, tzinfo=UTC),
            datetime(2026, 9, 4, tzinfo=UTC),
        )
    ]


def test_commercial_response_keeps_revenue_usage_grants_cost_and_balances_separate():
    facts = {
        1: _commercial(
            1,
            7_000_000,
            period_purchased=3_000_000,
            refunded=1_000_000,
            consumed=400_000,
            grants=1_500_000,
            grant_balance=900_000,
            purchased_balance=100_000,
        ),
        2: _commercial(
            2,
            0,
            consumed=100_000,
            grants=500_000,
            grant_balance=400_000,
        ),
    }
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1), 2: _snapshot(2)},
        commercial=facts,
    )

    response = service.get_commercial(
        start=date(2026, 8, 1),
        end=date(2026, 9, 1),
        now=NOW,
    )

    assert response.tier_counts == {
        "unpaid": 1,
        "starter": 0,
        "invested": 1,
        "high_value": 0,
    }
    assert response.lifetime_net_purchased_micro == 7_000_000
    assert response.selected_period.purchased_micro == 3_000_000
    assert response.selected_period.refunded_micro == 1_000_000
    assert response.selected_period.consumed_micro == 500_000
    assert response.selected_period.admin_grant_activity_micro == 2_000_000
    assert response.selected_period.platform_model_cost_micro_usd == 250_000
    assert response.current_balances.model_dump() == {
        "grant_available_micro": 1_300_000,
        "purchased_available_micro": 100_000,
        "total_available_micro": 1_400_000,
    }


def test_missing_operational_subsection_is_reported_as_partial():
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        legacy_availability={"growth": True, "friction": False},
    )

    response = service.get_operational(
        start=date(2026, 8, 1),
        end=date(2026, 9, 1),
        now=NOW,
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"


def test_profile_includes_selected_period_and_lifecycle_transition():
    start = date(2026, 8, 1)
    end = date(2026, 9, 1)
    snapshots = {1: _snapshot(1)}
    daily = [
        _daily(1, start - timedelta(days=1), "growing"),
        _daily(1, start, "core"),
    ]
    service, _value_store, legacy = _service(snapshots=snapshots, daily=daily)

    profile = service.get_user_profile(
        user_id=1,
        start=start,
        end=end,
        now=NOW,
    )

    assert profile.selected_period_start == start
    assert profile.selected_period_end == end
    assert profile.lifecycle.segment == "core"
    assert profile.operational.state == "healthy"
    assert [
        (row.from_segment, row.to_segment)
        for row in profile.recent_lifecycle_transitions
    ] == [("growing", "core")]
    assert legacy.profile_windows == [
        (datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))
    ]


def test_profile_never_uses_anonymous_transition_rollups():
    start = date(2026, 8, 1)
    aggregate = SimpleNamespace(
        rollup_date=start,
        metric_name="lifecycle_transition",
        event_name="growing",
        user_state="core",
        value_count=99,
        outcome="complete",
    )
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        rollups=[aggregate],
    )

    profile = service.get_user_profile(
        user_id=1,
        start=start,
        end=date(2026, 9, 1),
        now=NOW,
    )

    assert profile.recent_lifecycle_transitions == []


def test_internal_accounts_are_excluded_unless_explicitly_included():
    snapshots = {1: _snapshot(1), 2: _snapshot(2)}
    service, _value_store, _legacy = _service(
        snapshots=snapshots,
        excluded={2},
    )

    external = service.list_users(
        filters=UserValueFilters(),
        limit=25,
        offset=0,
        now=NOW,
    )
    all_users = service.list_users(
        filters=UserValueFilters(include_internal=True),
        limit=25,
        offset=0,
        now=NOW,
    )

    assert [item.user_id for item in external.items] == [1]
    assert [item.user_id for item in all_users.items] == [1, 2]


def _transition_rollup(day: date, count: int, outcome: str = "complete"):
    return SimpleNamespace(
        rollup_date=day,
        metric_name="lifecycle_transition",
        event_name="growing",
        user_state="core",
        value_count=count,
        outcome=outcome,
    )


def test_lifecycle_transitions_ignore_rollups_outside_the_requested_window():
    """A long movement range widens the scan; it must not widen the totals.

    `transitions` is stamped with the requested period, so a rollup from before
    `start` that is summed in reports itself as having happened inside a window
    it predates.
    """
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    daily = [
        _daily(1, start + timedelta(days=offset))
        for offset in range(30)
        if start + timedelta(days=offset) != date(2026, 9, 10)
    ]
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        daily=daily,
        rollups=[
            _transition_rollup(date(2026, 9, 10), 2),
            _transition_rollup(date(2026, 1, 15), 97),
        ],
    )

    response = service.get_lifecycle(
        start=start,
        end=end,
        movement_range="1y",
        now=datetime(2026, 10, 1, 12, tzinfo=UTC),
    )

    assert [
        (row.from_segment, row.to_segment, row.users) for row in response.transitions
    ] == [("growing", "core", 2)]


def test_lifecycle_coverage_reports_the_requested_window_not_the_movement_history():
    """Coverage describes the window the admin asked for.

    The movement chart needs a wider scan than the filter range, but
    `availability.history` is what the Lifecycle distribution card renders, so
    it must keep describing `start..end`.
    """
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    daily = [_daily(1, start + timedelta(days=offset)) for offset in range(30)]
    daily.append(_daily(1, date(2025, 12, 1), quality="partial"))
    service, _value_store, _legacy = _service(snapshots={1: _snapshot(1)}, daily=daily)

    response = service.get_lifecycle(
        start=start,
        end=end,
        movement_range="1y",
        now=datetime(2026, 10, 1, 12, tzinfo=UTC),
    )

    history = response.availability["history"]
    assert history.coverage_start == start
    assert history.coverage_end == end - timedelta(days=1)
    assert history.status == "ready"


def test_lifecycle_movement_buckets_never_start_before_the_selected_window():
    """Calendar bucketing must not label a point outside the chart's own range."""
    start = date(2026, 9, 15)
    end = date(2026, 10, 15)
    window_start = end - timedelta(days=365)
    daily = [_daily(1, window_start + timedelta(days=offset)) for offset in range(366)]
    service, _value_store, _legacy = _service(snapshots={1: _snapshot(1)}, daily=daily)

    response = service.get_lifecycle(
        start=start,
        end=end,
        movement_range="1y",
        now=datetime(2026, 10, 15, 12, tzinfo=UTC),
    )

    assert response.movement_segments
    assert min(point.period_start for point in response.movement_segments) >= window_start


@pytest.mark.parametrize("movement_range", sorted(_MOVEMENT_WINDOWS))
def test_lifecycle_daily_scan_stays_within_the_derived_history_bound(movement_range):
    """Pins the widest span of per-user rows one request may read.

    Not a regression test -- `_validate_dates` already bounds the filter range,
    so the scan is bounded today. It is a drift guard: the movement window is
    read from a table, and a longer entry added to that table would widen this
    scan for every eligible user with nothing else noticing.
    """
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    service, value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        daily=[_daily(1, start + timedelta(days=offset)) for offset in range(30)],
    )

    service.get_lifecycle(
        start=start,
        end=end,
        movement_range=movement_range,
        now=datetime(2026, 10, 1, 12, tzinfo=UTC),
    )

    scan_start, scan_end = value_store.daily_windows[-1]
    assert scan_end == end
    assert scan_start >= end - timedelta(days=_MAX_HISTORY_SCAN_DAYS + 1)
