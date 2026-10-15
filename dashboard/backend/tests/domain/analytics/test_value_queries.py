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
from dashboard.backend.domain.analytics.rollups import DailyRollup
from dashboard.backend.domain.analytics.value_queries import (
    UserValueFilters,
    ValueAnalyticsQueryService,
)
from dashboard.backend.domain.analytics.value_repository import (
    CommercialValueFact,
    FactStateCount,
    GroupFactTotals,
    LifecycleTransitionRow,
    RecentFactTotals,
    SegmentDayCount,
    TransitionCount,
    UserActivity,
    UserDailyFact,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def _user(user_id: int, user_group: str = "unknown") -> dict[str, object]:
    return {
        "id": user_id,
        "display_name": f"Value User {user_id}",
        "email": f"value-{user_id}@example.test",
        "created_at": (NOW - timedelta(days=90)).isoformat(),
        "user_group": user_group,
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


def _fact(user_id: int, day: date, **overrides) -> UserDailyFact:
    values = dict(
        snapshot_date=day,
        user_id=user_id,
        lifecycle_segment="core",
        lifecycle_reason_code="core_repeated_value",
        operational_state="healthy",
        operational_reason_code="no_supported_issue",
        tier="unpaid",
        user_group="unknown",
        active=True,
        runs_requested=1,
        runs_completed=1,
        runs_failed=0,
        runs_cancelled=0,
        operator_cost_micro=0,
        own_spend_micro=0,
        data_quality="complete",
        calculated_at=NOW,
    )
    values.update(overrides)
    return UserDailyFact(**values)


def _activity(
    user_id: int, *, activated_days_ago: int | None = 60, active_days_ago: int = 2
) -> UserActivity:
    return UserActivity(
        user_id=user_id,
        activated_at=(
            None
            if activated_days_ago is None
            else NOW - timedelta(days=activated_days_ago)
        ),
        last_meaningful_activity_at=NOW - timedelta(days=active_days_ago),
        updated_at=NOW,
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
    """Fact-backed fake: answers every fact/activity read from in-memory rows.

    ``snapshots``/``daily`` back the snapshot reads the routes not yet moved
    onto facts still make; each goes with the task that moves its route.
    """

    def __init__(
        self,
        *,
        facts=(),
        transitions=(),
        activity=None,
        commercial=None,
        page_rows=(),
        operational_facts=None,
        internal_ids=(),
        credit_activity=None,
        ledger_days=(),
        snapshots=None,
        daily=(),
    ):
        self.facts = list(facts)
        self.transitions = list(transitions)
        self.activity = dict(activity or {})
        self.commercial = dict(commercial or {})
        self.page_rows = list(page_rows)
        self.operational_facts = dict(operational_facts or {})
        # UserActivity is frozen and carries no role, so "internal" is a set of ids.
        self.internal_ids = set(internal_ids)
        self.credit_activity = dict(credit_activity or {})
        # user_id -> [LedgerDayTotal]; None stands for an unconfigured reader.
        self.ledger_days = None if ledger_days is None else dict(ledger_days)
        self.snapshots = dict(snapshots or {})
        self.daily = list(daily)
        self.commercial_windows = []
        self.daily_windows = []
        self.ledger_calls = []
        self.page_calls = []
        self.recent_windows = []

    # -- fact-table reads --
    def count_states_for_date(self, day):
        cells = {}
        for row in self.facts:
            if row.snapshot_date != day:
                continue
            key = (row.operational_state, row.operational_reason_code, row.lifecycle_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            FactStateCount(
                operational_state=key[0],
                operational_reason_code=key[1],
                lifecycle_segment=key[2],
                users=users,
                partial_rows=partial,
            )
            for key, (users, partial) in sorted(
                cells.items(), key=lambda item: tuple(str(v) for v in item[0])
            )
        ]

    def count_segments_by_date(self, *, start, end):
        cells = {}
        for row in self.facts:
            if not start <= row.snapshot_date < end:
                continue
            key = (row.snapshot_date, row.lifecycle_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            SegmentDayCount(
                snapshot_date=key[0],
                lifecycle_segment=key[1],
                users=users,
                partial_rows=partial,
            )
            for key, (users, partial) in sorted(cells.items())
        ]

    def count_transitions(self, *, start, end):
        cells = {}
        for row in self.transitions:
            if not start <= row.snapshot_date < end:
                continue
            key = (row.from_segment, row.to_segment)
            users, partial = cells.get(key, (0, 0))
            cells[key] = (users + 1, partial + (row.data_quality == "partial"))
        return [
            TransitionCount(
                from_segment=key[0], to_segment=key[1], users=users, partial_rows=partial
            )
            for key, (users, partial) in sorted(cells.items())
        ]

    def list_attention_candidates(self, *, day, window_start, limit):
        return []

    def sum_facts_by_user_group(self, *, start, end):
        totals = {}
        completion_days = {}
        for row in self.facts:
            if not start <= row.snapshot_date < end:
                continue
            group = totals.setdefault(
                row.user_group,
                {"total_runs": 0, "operator_cost_micro": 0, "successful": set()},
            )
            group["total_runs"] += row.runs_requested
            group["operator_cost_micro"] += row.operator_cost_micro
            if row.runs_completed > 0:
                group["successful"].add(row.user_id)
                completion_days.setdefault((row.user_group, row.user_id), set()).add(
                    row.snapshot_date
                )
        return {
            name: GroupFactTotals(
                total_runs=values["total_runs"],
                operator_cost_micro=values["operator_cost_micro"],
                successful_run_users=len(values["successful"]),
                repeat_users=sum(
                    1
                    for (group, _user), days in completion_days.items()
                    if group == name and len(days) >= 2
                ),
            )
            for name, values in totals.items()
        }

    def list_active_dates(self, user_ids, *, start, end):
        result = {user_id: [] for user_id in user_ids}
        for row in sorted(self.facts, key=lambda r: (r.user_id, r.snapshot_date)):
            if row.user_id in result and row.active and start <= row.snapshot_date < end:
                result[row.user_id].append(row.snapshot_date)
        return result

    def list_activations(self, *, start, end, include_internal):
        return {
            user_id: activity.activated_at
            for user_id, activity in self.activity.items()
            if activity.activated_at is not None
            and start <= activity.activated_at < end
            and (include_internal or user_id not in self.internal_ids)
        }

    def count_activated_users(self, *, include_internal):
        return sum(
            1
            for user_id, activity in self.activity.items()
            if activity.activated_at is not None
            and (include_internal or user_id not in self.internal_ids)
        )

    def list_user_transitions(self, user_id, *, start, end):
        return [
            row
            for row in self.transitions
            if row.user_id == user_id and start <= row.snapshot_date < end
        ]

    def list_transitions_for_date(self, day):
        return [row for row in self.transitions if row.snapshot_date == day]

    def list_users_page(self, *, query, day, limit, offset):
        self.page_calls.append((query, day, limit, offset))
        rows = list(self.page_rows)
        return rows[offset : offset + limit], len(rows)

    # -- activity clock and trailing windows --
    def get_activity(self, user_id):
        return self.activity.get(user_id)

    def list_activity(self, user_ids):
        return {
            user_id: self.activity[user_id]
            for user_id in user_ids
            if user_id in self.activity
        }

    def sum_recent_facts(self, user_ids, *, start, end):
        self.recent_windows.append((start, end))
        result = {}
        for user_id in user_ids:
            rows = [
                r for r in self.facts if r.user_id == user_id and start <= r.snapshot_date <= end
            ]
            if not rows:
                continue
            active_dates = [r.snapshot_date for r in rows if r.active]
            result[user_id] = RecentFactTotals(
                active_days=len(active_dates),
                successful_backtests=sum(r.runs_completed for r in rows),
                runs_requested=sum(r.runs_requested for r in rows),
                runs_completed=sum(r.runs_completed for r in rows),
                runs_failed=sum(r.runs_failed for r in rows),
                runs_cancelled=sum(r.runs_cancelled for r in rows),
                operator_cost_micro=sum(r.operator_cost_micro for r in rows),
                own_spend_micro=sum(r.own_spend_micro for r in rows),
                days_present=len(rows),
                last_active_date=max(active_dates) if active_dates else None,
            )
        return result

    def get_operational_facts(self, user_id, *, now):
        return self.operational_facts[user_id]

    # -- snapshot reads (routes not yet on facts) --
    def list_current_snapshots(self, user_ids):
        return {
            user_id: self.snapshots[user_id]
            for user_id in user_ids
            if user_id in self.snapshots
        }

    def get_current_snapshot(self, user_id):
        return self.snapshots.get(user_id)

    def list_daily_snapshots(self, *, start, end, user_ids=None):
        self.daily_windows.append((start, end))
        selected = None if user_ids is None else set(user_ids)
        return [
            row
            for row in self.daily
            if start <= row.snapshot_date < end
            and (selected is None or row.user_id in selected)
        ]

    # -- commercial and ledger --
    def list_commercial_values(self, user_ids, *, start, end):
        self.commercial_windows.append((start, end))
        return {
            user_id: self.commercial[user_id]
            for user_id in user_ids
            if user_id in self.commercial
        }

    def list_ledger_days(self, user_ids, *, start, end):
        self.ledger_calls.append((list(user_ids), start, end))
        if self.ledger_days is None:
            raise RuntimeError("credits ledger reader is not configured")
        return [
            row
            for user_id in user_ids
            for row in self.ledger_days.get(user_id, ())
            if start.date() <= row.day < end.date()
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
    def __init__(self, events=None, rollups=()):
        # None: no raw read is expected, and one fails the test. A list serves
        # the routes that still scan raw events until their own task.
        self.events = None if events is None else list(events)
        self.rollups = list(rollups)

    def list_events(self, *, start, end, include_internal, user_id=None):
        if self.events is None:
            raise AssertionError("this read must not scan raw events")
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
    def __init__(self, *, events=None, rollups=(), legacy_status=None):
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
        self.overview_calls = 0
        self.platform_cost_filters = []
        self.availability = availability or {"growth": True, "friction": True}

    def get_platform_model_cost_micro(self, *, filters, now):
        self.platform_cost_filters.append(filters)
        return 250_000 if self.availability.get("growth", True) else None

    def get_overview(self, *, filters, now):
        self.overview_calls += 1
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
            # Deliberately set even when growth is unavailable: the real
            # overview can compute the cost before the block fails.
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
    facts=(),
    transitions=(),
    activity=None,
    commercial=None,
    page_rows=(),
    operational_facts=None,
    internal_ids=(),
    users=(),
    rollups=(),
    excluded=(),
    legacy_availability=None,
    ledger_days=(),
    # Snapshot-model fixtures for the routes not yet on facts: ``users``
    # defaults to the snapshot ids grouped by ``user_groups``, and raw events
    # stay readable. Without ``snapshots`` a raw-event read fails the test.
    snapshots=None,
    daily=(),
    events=None,
    user_groups=None,
):
    if snapshots is not None:
        if not users:
            groups = dict(user_groups or {})
            users = tuple(
                (user_id, groups.get(user_id, "unknown")) for user_id in snapshots
            )
        if events is None:
            events = ()
    user_ids = [user_id for user_id, _group in users]
    value_store = FakeValueStore(
        facts=facts,
        transitions=transitions,
        activity=activity,
        commercial=commercial or {user_id: _commercial(user_id) for user_id in user_ids},
        page_rows=page_rows,
        operational_facts=operational_facts,
        internal_ids=internal_ids,
        ledger_days=ledger_days,
        snapshots=snapshots,
        daily=daily,
    )
    legacy_service = FakeLegacyService(legacy_availability)
    service = ValueAnalyticsQueryService(
        store=FakeBaseStore(excluded),
        user_store=FakeUserStore([_user(user_id, group) for user_id, group in users]),
        value_store=value_store,
        query_store=FakeQueryStore(events=events, rollups=rollups),
        legacy_service=legacy_service,
    )
    return service, value_store, legacy_service


def _metric_event(
    user_id: int,
    event_name: str,
    occurred_at: str | datetime,
    *,
    billing_mode: str | None = None,
    properties: dict[str, object] | None = None,
):
    at = (
        datetime.fromisoformat(occurred_at)
        if isinstance(occurred_at, str)
        else occurred_at
    )
    return SimpleNamespace(
        user_id=user_id,
        event_name=event_name,
        occurred_at=at,
        billing_mode=billing_mode,
        properties=properties or {},
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
        _metric_event(1, "agent_updated", datetime(2026, 7, 14, 10, tzinfo=UTC)),
        _metric_event(3, "agent_updated", datetime(2026, 7, 21, 10, tzinfo=UTC)),
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


def test_group_summary_has_six_rows_zeroes_and_fixed_order():
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        user_groups={1: "organic"},
        events=[
            _metric_event(
                1,
                "backtest_requested",
                "2026-09-01T01:00:00+00:00",
            ),
            _metric_event(
                1,
                "backtest_completed",
                "2026-09-01T02:00:00+00:00",
            ),
            _metric_event(
                1,
                "backtest_completed",
                "2026-09-03T02:00:00+00:00",
            ),
        ],
    )

    result = service.get_groups(
        date(2026, 9, 1), date(2026, 9, 4), now=NOW
    )

    assert [row.group for row in result.groups] == [
        "internal",
        "invited",
        "organic",
        "competition",
        "partner",
        "unknown",
    ]
    organic = result.groups[2]
    assert organic.users == 1
    assert organic.successful_run_users == 1
    assert organic.repeat_users == 1
    assert organic.total_runs == 1
    assert result.groups[0].users == 0
    assert result.availability.status == "ready"


def test_group_summary_cost_excludes_byok_and_paid_uses_lifetime_credits():
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1), 2: _snapshot(2)},
        user_groups={1: "partner", 2: "organic"},
        commercial={
            1: _commercial(1, 5_000_000),
            2: _commercial(2, 0),
        },
        events=[
            _metric_event(
                1,
                "model_usage_recorded",
                "2026-09-01T01:00:00+00:00",
                billing_mode="platform_credits",
                properties={"cost_micro_usd": 1234},
            ),
            _metric_event(
                1,
                "model_usage_recorded",
                "2026-09-01T02:00:00+00:00",
                billing_mode="byok",
                properties={"cost_micro_usd": 9999},
            ),
        ],
    )

    result = service.get_groups(
        date(2026, 9, 1), date(2026, 9, 2), now=NOW
    )

    partner = next(row for row in result.groups if row.group == "partner")
    organic = next(row for row in result.groups if row.group == "organic")
    assert partner.atl_cost_micro_usd == 1234
    assert partner.paid_users == 1
    assert organic.atl_cost_micro_usd == 0
    assert organic.paid_users == 0


def test_user_group_filter_applies_to_priority_users():
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1, operational="blocked"), 2: _snapshot(2)},
        user_groups={1: "partner", 2: "organic"},
    )

    filters = UserValueFilters(user_group="partner", priority=True)
    page = service.list_users(filters=filters, limit=25, offset=0, now=NOW)

    assert filters.user_group == "partner"
    assert [item.user_id for item in page.items] == [1]


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
        users=((1, "unknown"), (2, "unknown")),
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


def test_commercial_platform_cost_is_unknown_not_zero_when_its_read_failed():
    """0 is a real answer ("no platform calls"). When the model-usage read
    behind the figure failed, the field is None so the page draws a dash, and
    the section is partial."""
    service, _value_store, _legacy = _service(
        users=((1, "unknown"),),
        legacy_availability={"growth": False, "friction": True},
    )

    response = service.get_commercial(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)

    assert response.selected_period.platform_model_cost_micro_usd is None
    assert response.availability.status == "partial"


def test_commercial_reads_the_platform_cost_alone_not_a_whole_overview():
    """The figure is one headline of the overview; building the whole overview
    for it scanned the range's raw events, rollups, snapshots and users a
    second time on every page load."""
    service, _value_store, legacy = _service(users=((1, "unknown"),))

    service.get_commercial(
        start=date(2026, 8, 1), end=date(2026, 9, 1), include_internal=True, now=NOW
    )

    assert legacy.overview_calls == 0
    (filters,) = legacy.platform_cost_filters
    assert (filters.start, filters.end, filters.include_internal) == (
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 9, 1, tzinfo=UTC),
        True,
    )


def test_commercial_series_sum_across_users_and_match_their_headlines():
    """purchased_by_day is gross purchases and consumed_by_day is model spend,
    the same two quantities as selected_period.purchased_micro/consumed_micro."""

    from dashboard.backend.domain.credits.repository_common import LedgerDayTotal

    service, value_store, _legacy = _service(
        users=((1, "unknown"), (2, "unknown")),
        ledger_days={
            1: [
                LedgerDayTotal(day=date(2026, 8, 3), purchased_micro=5_000_000, refunded_micro=2_000_000, consumed_micro=100_000),
                LedgerDayTotal(day=date(2026, 8, 5), refunded_micro=1_000_000),
            ],
            2: [LedgerDayTotal(day=date(2026, 8, 3), purchased_micro=1_000_000, consumed_micro=300_000)],
        },
    )

    response = service.get_commercial(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)

    assert [(row.day, row.amount_micro) for row in response.purchased_by_day] == [
        (date(2026, 8, 3), 6_000_000)
    ]
    assert [(row.day, row.amount_micro) for row in response.consumed_by_day] == [
        (date(2026, 8, 3), 400_000)
    ]
    assert response.availability.status == "ready"
    assert value_store.ledger_calls[0][1:] == (
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 9, 1, tzinfo=UTC),
    )


def test_commercial_series_chunk_user_ids_like_the_headline():
    user_ids = range(1, 1202)
    service, value_store, _legacy = _service(
        users=tuple((user_id, "unknown") for user_id in user_ids),
    )

    service.get_commercial(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)

    assert [len(ids) for ids, _start, _end in value_store.ledger_calls] == [500, 500, 201]


def test_commercial_series_failure_is_partial_and_logged_not_an_empty_ready(capsys):
    service, _value_store, _legacy = _service(
        users=((1, "unknown"),),
        ledger_days=None,
    )

    response = service.get_commercial(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)

    # None, not []: [] renders as "No settled purchases in this range".
    assert response.purchased_by_day is None
    assert response.consumed_by_day is None
    assert response.availability.status == "partial"
    assert response.tier_counts["unpaid"] == 1  # the rest is still served
    assert "ERROR commercial per-day ledger unavailable" in capsys.readouterr().out


def test_operational_growth_figures_are_unknown_not_zero_when_their_read_failed():
    """Run counts, tokens and platform cost come from the overview's growth
    read. When it failed they are None, not 0 -- which is the real answer "no
    runs" -- even though the overview may have computed some before failing."""
    service, _value_store, _legacy = _service(
        snapshots={1: _snapshot(1)},
        legacy_availability={"growth": False, "friction": True},
    )

    response = service.get_operational(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)

    assert (
        response.backtest_success_rate,
        response.completed_runs,
        response.failed_runs,
        response.input_tokens,
        response.output_tokens,
        response.platform_model_cost_micro_usd,
    ) == (None, None, None, None, None, None)
    assert response.availability.status == "partial"

    service, _value_store, _legacy = _service(snapshots={1: _snapshot(1)})
    response = service.get_operational(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)
    assert (response.completed_runs, response.platform_model_cost_micro_usd) == (9, 250_000)


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
    snapshots = {1: _snapshot(1), 2: _snapshot(2)}
    transitions = [
        LifecycleTransitionRow(
            user_id=1, snapshot_date=start, from_segment="growing",
            to_segment="core", created_at=NOW,
        ),
        # Another user's move on the same day stays off this profile.
        LifecycleTransitionRow(
            user_id=2, snapshot_date=start, from_segment="growing",
            to_segment="at_risk", created_at=NOW,
        ),
        # Outside the selected period.
        LifecycleTransitionRow(
            user_id=1, snapshot_date=end, from_segment="core",
            to_segment="at_risk", created_at=NOW,
        ),
    ]
    service, _value_store, legacy = _service(
        snapshots=snapshots, transitions=transitions
    )

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
        (row.from_segment, row.to_segment, row.users, row.period_start, row.period_end)
        for row in profile.recent_lifecycle_transitions
    ] == [("growing", "core", 1, start, end)]
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


YESTERDAY = NOW.date() - timedelta(days=1)


def _lifecycle_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, lifecycle_segment="core"),
        _fact(2, YESTERDAY, lifecycle_segment="at_risk", active=False),
        _fact(3, YESTERDAY, lifecycle_segment="onboarding", runs_completed=0),
        _fact(1, YESTERDAY - timedelta(days=1), lifecycle_segment="growing"),
        _fact(
            2,
            YESTERDAY - timedelta(days=1),
            lifecycle_segment="growing",
            data_quality="partial",
        ),
        _fact(
            3, YESTERDAY - timedelta(days=1), lifecycle_segment="new", runs_completed=0
        ),
    ]
    transitions = [
        LifecycleTransitionRow(
            user_id=1, snapshot_date=YESTERDAY, from_segment="growing",
            to_segment="core", inactive_days=0, data_quality="complete",
            created_at=NOW,
        ),
        LifecycleTransitionRow(
            user_id=2, snapshot_date=YESTERDAY, from_segment="growing",
            to_segment="at_risk", inactive_days=8, data_quality="partial",
            created_at=NOW,
        ),
    ]
    activity = {
        1: _activity(1),
        2: _activity(2, active_days_ago=8),
        3: _activity(3, activated_days_ago=None),
        9: _activity(9),
    }
    values = dict(
        facts=facts,
        transitions=transitions,
        activity=activity,
        internal_ids={9},
        users=((1, "organic"), (2, "partner"), (3, "unknown"), (9, "internal")),
        commercial={
            1: _commercial(1, 6_000_000),
            2: _commercial(2),
            3: _commercial(3),
            9: _commercial(9),
        },
    )
    values.update(overrides)
    return _service(**values)[0]


def test_lifecycle_counts_are_yesterdays_facts_and_headline_reads_activity():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.segment_counts == {
        "new": 0, "onboarding": 1, "growing": 0, "core": 1, "at_risk": 1, "dormant": 0
    }
    assert response.headline.core_users == 1
    assert response.headline.at_risk_users == 1
    assert response.headline.activated_users == 2  # user 9 is internal
    assert response.headline.paid_users == 1
    current = response.availability["current"]
    assert (current.status, current.coverage_start, current.coverage_end) == (
        "ready", YESTERDAY, YESTERDAY
    )


def test_lifecycle_current_is_building_before_yesterdays_facts_land():
    service = _lifecycle_service(
        facts=[_fact(1, YESTERDAY - timedelta(days=1))]
    )

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert set(response.segment_counts.values()) == {0}
    assert response.availability["current"].status == "building"


def test_lifecycle_include_internal_counts_internal_activations():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6),
        end=YESTERDAY + timedelta(days=1),
        include_internal=True,
        now=NOW,
    )

    assert response.headline.activated_users == 3


def test_lifecycle_movement_and_transitions_come_from_facts_and_the_transition_log():
    service = _lifecycle_service()

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6),
        end=YESTERDAY + timedelta(days=1),
        movement_range="5d",
        now=NOW,
    )

    assert response.movement_granularity == "day"
    assert [point.period_start for point in response.movement_segments] == [
        YESTERDAY - timedelta(days=1),
        YESTERDAY,
    ]
    assert response.movement_segments[0].segment_counts["growing"] == 2
    assert response.movement_segments[0].data_quality == "partial"
    assert response.movement_segments[1].segment_counts["core"] == 1
    assert {
        (t.from_segment, t.to_segment, t.users, t.data_quality)
        for t in response.transitions
    } == {
        ("growing", "core", 1, "complete"),
        ("growing", "at_risk", 1, "partial"),
    }
    assert response.availability["history"].status == "partial"
    assert response.availability["history"].coverage_start == YESTERDAY - timedelta(
        days=1
    )


def test_lifecycle_history_is_ready_through_yesterday_for_the_default_window():
    """The router's default ``end`` is today + 1, and facts exist only through
    yesterday: a fully covered window must not read as incomplete for lacking
    a row that cannot exist yet."""
    start = YESTERDAY - timedelta(days=6)
    service = _lifecycle_service(
        facts=[_fact(1, start + timedelta(days=offset)) for offset in range(7)],
        transitions=[],
    )

    response = service.get_lifecycle(
        start=start, end=NOW.date() + timedelta(days=1), now=NOW
    )

    history = response.availability["history"]
    assert (history.status, history.coverage_start, history.coverage_end) == (
        "ready", start, YESTERDAY
    )


def test_lifecycle_history_falls_back_to_rollups_for_days_without_facts():
    old_day = YESTERDAY - timedelta(days=3)
    rollups = [
        DailyRollup(
            rollup_date=old_day, metric_name="lifecycle_segment_count",
            user_state="core", outcome="complete", value_count=7, updated_at=NOW,
        ),
        DailyRollup(
            rollup_date=old_day, metric_name="lifecycle_transition",
            event_name="onboarding", user_state="growing", outcome="complete",
            value_count=2, updated_at=NOW,
        ),
    ]
    service = _lifecycle_service(rollups=rollups)

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert any(
        point.period_start == old_day and point.segment_counts["core"] == 7
        for point in response.movement_segments
    )
    assert any(
        t.from_segment == "onboarding" and t.to_segment == "growing" and t.users == 2
        for t in response.transitions
    )

    internal = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6),
        end=YESTERDAY + timedelta(days=1),
        include_internal=True,
        now=NOW,
    )
    assert not any(point.period_start == old_day for point in internal.movement_segments)


def test_lifecycle_reads_no_raw_events():
    service = _lifecycle_service()
    # FakeRollups.list_events raises; a passing call proves no raw read happened.
    service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )


def test_lifecycle_failed_activation_read_is_partial_not_a_ready_zero():
    service = _lifecycle_service()

    def fail(*, include_internal):
        raise RuntimeError("user_activity unavailable")

    service.value_store.count_activated_users = fail

    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.headline.activated_users == 0
    assert response.availability["current"].status == "partial"
    assert response.segment_counts["core"] == 1  # the rest is still served
