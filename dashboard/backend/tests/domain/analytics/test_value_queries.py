"""Display-safe query composition for Admin user-value analytics."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import pytest

from dashboard.backend.domain.analytics.lifecycle import commercial_tier
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsStateSummary,
    AnalyticsUserProfile,
)
from dashboard.backend.domain.analytics.rollups import DailyRollup
from dashboard.backend.domain.analytics.value_queries import (
    OperationalAnalyticsResponse,
    UserGroupSummary,
    UserValueFilters,
    ValueAnalyticsQueryService,
)
from dashboard.backend.domain.analytics.value_repository import (
    CommercialValueFact,
    CurrentOperationalFacts,
    FactStateCount,
    GroupFactTotals,
    LifecycleTransitionRow,
    RecentFactTotals,
    SegmentDayCount,
    TransitionCount,
    UserActivity,
    UserDailyFact,
    UserPageQuery,
    UserPageRow,
)
from dashboard.backend.domain.user_groups import USER_GROUPS


UTC = timezone.utc
NOW = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
YESTERDAY = NOW.date() - timedelta(days=1)


def _user(user_id: int, user_group: str = "unknown") -> dict[str, object]:
    return {
        "id": user_id,
        "display_name": f"Value User {user_id}",
        "email": f"value-{user_id}@example.test",
        "created_at": (NOW - timedelta(days=90)).isoformat(),
        "user_group": user_group,
    }


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
    """Fact-backed fake: answers every fact/activity read from in-memory rows."""

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
        ledger_days=(),
    ):
        self.facts = list(facts)
        self.transitions = list(transitions)
        self.activity = dict(activity or {})
        self.commercial = dict(commercial or {})
        self.page_rows = list(page_rows)
        self.operational_facts = dict(operational_facts or {})
        # UserActivity is frozen and carries no role, so "internal" is a set of ids.
        self.internal_ids = set(internal_ids)
        # user_id -> [LedgerDayTotal]; None stands for an unconfigured reader.
        self.ledger_days = None if ledger_days is None else dict(ledger_days)
        self.commercial_windows = []
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
                {"total_runs": 0, "operator_cost_micro": 0, "successful": set(), "partial": 0},
            )
            group["partial"] += row.data_quality == "partial"
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
                partial_rows=values["partial"],
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

    def latest_fact_date(self, *, on_or_before):
        days = [row.snapshot_date for row in self.facts if row.snapshot_date <= on_or_before]
        return max(days) if days else None

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


class FakeRollups:
    def __init__(self, rollups=()):
        self.rollups = list(rollups)

    def list_events(self, *, start, end, include_internal, user_id=None):
        raise AssertionError("this read must not scan raw events")

    def list_rollups(self, *, start, end):
        return [row for row in self.rollups if start <= row.rollup_date < end]


class FakeQueryStore:
    def __init__(self, *, rollups=()):
        self.rollups = FakeRollups(rollups)


class FakeLegacyService:
    def __init__(self, availability=None, *, operational_facts=None):
        self.profile_windows = []
        self.operational_facts = dict(operational_facts or {})
        self.raise_lookup = False
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

    def get_user_axes(self, user_id, *, now):
        from dashboard.backend.domain.analytics.lifecycle import (
            LifecycleInputs, OperationalSignals, calculate_lifecycle, calculate_operational_state,
        )
        if self.raise_lookup:
            raise LookupError("Analytics user was not found")
        facts = self.operational_facts.get(user_id)
        operational = calculate_operational_state(
            OperationalSignals(**facts.model_dump()) if facts else OperationalSignals(user_id=user_id), now
        )
        lifecycle = calculate_lifecycle(
            LifecycleInputs(user_id=user_id, created_at=now - timedelta(days=90),
                            first_successful_backtest_at=now - timedelta(days=60),
                            last_meaningful_activity_at=now - timedelta(days=2),
                            active_days_30d=3, successful_backtests_30d=3),
            now,
        )
        return lifecycle, operational

    def get_user_profile(self, *, user_id, now, start=None, end=None, axes=None):
        if self.raise_lookup:
            raise LookupError("Analytics user was not found")
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
):
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
    )
    legacy_service = FakeLegacyService(
        legacy_availability, operational_facts=operational_facts
    )
    service = ValueAnalyticsQueryService(
        store=FakeBaseStore(excluded),
        user_store=FakeUserStore([_user(user_id, group) for user_id, group in users]),
        value_store=value_store,
        query_store=FakeQueryStore(rollups=rollups),
        legacy_service=legacy_service,
    )
    return service, value_store, legacy_service


def _retention_service():
    # Cohort week Monday 2026-08-03: users 1 and 2 activated that week; user 3 in the next.
    activity = {
        1: _activity(1, activated_days_ago=(NOW.date() - date(2026, 8, 4)).days),
        2: _activity(2, activated_days_ago=(NOW.date() - date(2026, 8, 7)).days),
        3: _activity(3, activated_days_ago=(NOW.date() - date(2026, 8, 12)).days),
        9: _activity(9, activated_days_ago=(NOW.date() - date(2026, 8, 5)).days),
    }
    facts = []
    # Every day from the cohort start through the latest ``now`` the tests use
    # has at least one row, so no week is partial for lack of data.
    day = date(2026, 8, 3)
    while day <= RETENTION_LAST_FACT_DAY:
        facts.append(_fact(3, day, active=False, runs_completed=0))
        day += timedelta(days=1)
    # Week 1 (2026-08-10..16): user 1 active, user 2 not. Week 2 (08-17..23): both.
    # Week 4 (08-31..09-06): user 2 only.
    facts += [
        _fact(1, date(2026, 8, 11)),
        _fact(1, date(2026, 8, 18)),
        _fact(2, date(2026, 8, 20)),
        _fact(2, date(2026, 9, 1)),
    ]
    return _service(
        facts=facts,
        activity=activity,
        internal_ids={9},
        users=((1, "organic"), (2, "organic"), (3, "organic"), (9, "internal")),
    )[0]


# Week 4 of the 2026-08-03 cohort ends 2026-09-07, so it is mature from 09-07 on.
RETENTION_NOW = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)
RETENTION_LAST_FACT_DAY = RETENTION_NOW.date() - timedelta(days=1)


def test_retention_cells_come_from_activation_weeks_and_active_fact_days():
    service = _retention_service()

    response = service.get_retention(
        start=date(2026, 8, 1), end=date(2026, 8, 20), now=RETENTION_NOW
    )

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.activated_users == 2  # user 9 is internal
    assert (first.week_1.retained_users, first.week_1.eligible_users, first.week_1.rate) == (1, 2, 0.5)
    assert (first.week_2.retained_users, first.week_2.eligible_users) == (2, 2)
    assert (first.week_4.retained_users, first.week_4.eligible_users, first.week_4.rate) == (1, 2, 0.5)
    second = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 10))
    assert second.activated_users == 1
    assert second.week_1.retained_users == 0
    assert response.summary_week_1.rate == 1 / 3
    assert response.availability.status == "ready"


def test_retention_cohort_membership_covers_the_window_end_week():
    # end falls mid-week: the 08-17 cohort starts before it and must still be read.
    service = _retention_service()
    service.value_store.activity[4] = _activity(
        4, activated_days_ago=(NOW.date() - date(2026, 8, 23)).days
    )

    response = service.get_retention(
        start=date(2026, 8, 1), end=date(2026, 8, 19), now=RETENTION_NOW
    )

    assert [c.cohort_week for c in response.cohorts] == [
        date(2026, 8, 3),
        date(2026, 8, 10),
        date(2026, 8, 17),
    ]
    assert response.cohorts[-1].activated_users == 1


def test_retention_marks_immature_weeks_unavailable_not_zero():
    service = _retention_service()
    now = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    response = service.get_retention(start=date(2026, 8, 1), end=date(2026, 8, 20), now=now)

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.week_2.mature is True
    assert first.week_4.mature is False
    assert first.week_4.retained_users is None and first.week_4.rate is None
    assert response.summary_week_4.mature is False


def test_retention_week_with_a_missing_or_partial_fact_day_is_partial():
    service = _retention_service()
    service.value_store.facts = [
        row for row in service.value_store.facts if row.snapshot_date != date(2026, 8, 12)
    ]
    service.value_store.facts.append(
        _fact(3, date(2026, 8, 19), active=False, data_quality="partial")
    )

    response = service.get_retention(
        start=date(2026, 8, 1), end=date(2026, 8, 20), now=RETENTION_NOW
    )

    first = next(c for c in response.cohorts if c.cohort_week == date(2026, 8, 3))
    assert first.week_1.data_quality == "partial"  # 08-12 has no rows
    assert first.week_2.data_quality == "partial"  # 08-19 carries a partial row
    assert first.week_4.data_quality == "complete"
    assert response.availability.status == "partial"


def test_retention_include_internal_does_not_deflate_rates():
    """I4: facts hold no internal users, so an internal activation admitted to a
    cohort would count as eligible and never retained."""
    service = _retention_service()
    window = dict(start=date(2026, 8, 1), end=date(2026, 8, 20), now=RETENTION_NOW)

    default = service.get_retention(**window)
    internal = service.get_retention(include_internal=True, **window)

    assert internal.cohorts == default.cohorts
    assert internal.summary_week_1 == default.summary_week_1
    assert internal.summary_week_4 == default.summary_week_4
    assert next(
        c for c in internal.cohorts if c.cohort_week == date(2026, 8, 3)
    ).activated_users == 2  # user 9 is internal


def test_retention_reads_no_raw_events():
    # FakeRollups.list_events raises; a passing call proves no raw read happened.
    _retention_service().get_retention(
        start=date(2026, 8, 1), end=date(2026, 8, 20), now=RETENTION_NOW
    )


def _page_row(user_id: int, **overrides) -> UserPageRow:
    values = dict(
        user_id=user_id, email=f"value-{user_id}@example.test", display_name=f"Value User {user_id}",
        role="user", user_group="unknown", created_at=NOW - timedelta(days=90),
        activated_at=NOW - timedelta(days=60), last_meaningful_activity_at=NOW - timedelta(days=2),
        lifecycle_segment="core", operational_state="healthy",
        operational_reason_code="no_supported_issue", tier="unpaid", data_quality="complete",
    )
    values.update(overrides)
    return UserPageRow(**values)


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def test_users_list_reads_one_sql_page_and_computes_live_segments():
    rows = [
        _page_row(1, operational_state="blocked", operational_reason_code="account_restricted", tier="invested"),
        _page_row(2, lifecycle_segment="growing", last_meaningful_activity_at=NOW - timedelta(days=9)),
    ]
    facts = [_fact(1, YESTERDAY - timedelta(days=offset)) for offset in range(3)]
    activity = {1: _activity(1), 2: _activity(2, active_days_ago=9)}
    service, value_store, _legacy = _service(
        page_rows=rows, facts=facts, activity=activity,
        commercial={1: _commercial(1, 6_000_000), 2: _commercial(2)},
    )

    response = service.list_users(filters=UserValueFilters(priority=True), limit=25, offset=0, now=NOW)

    assert response.total == 2
    first, second = response.items
    assert first.user_id == 1
    assert first.lifecycle.segment == "core"           # live: 3 active days, 3 successes
    assert first.operational.state == "blocked"        # yesterday's fact row
    assert first.operational.reason == "The Credits account is restricted from model spending."
    assert first.commercial_tier == "invested"
    assert first.lifetime_net_purchased_micro == 6_000_000
    assert first.priority_group == "blocked"
    assert second.lifecycle.segment == "at_risk"       # live: 9 idle days, no fact rows
    assert second.priority_group == "none"             # yesterday's segment was growing
    query, day, limit, offset = value_store.page_calls[0]
    assert (day, limit, offset) == (YESTERDAY, 25, 0)
    assert query.priority is True
    assert value_store.recent_windows == [(YESTERDAY - timedelta(days=29), YESTERDAY)]
    assert value_store.commercial_windows == [
        (_utc_midnight(YESTERDAY), _utc_midnight(YESTERDAY + timedelta(days=1)))
    ]


def test_users_list_passes_every_filter_to_sql_unchanged():
    service, value_store, _legacy = _service(page_rows=[], activity={})
    filters = UserValueFilters(
        q="Ada", lifecycle_segment="core", operational_state="healthy", commercial_tier="starter",
        user_group="partner", activated=True, last_meaningful_activity_from=NOW - timedelta(days=3),
        last_meaningful_activity_to=NOW, priority=False, include_internal=True,
    )

    response = service.list_users(filters=filters, limit=10, offset=20, now=NOW)

    assert response.total == 0 and response.items == []
    query, _day, limit, offset = value_store.page_calls[0]
    assert (limit, offset) == (10, 20)
    assert query == UserPageQuery(
        q="Ada", lifecycle_segment="core", operational_state="healthy", commercial_tier="starter",
        user_group="partner", activated=True, last_meaningful_activity_from=NOW - timedelta(days=3),
        last_meaningful_activity_to=NOW, priority=False, include_internal=True,
    )


def test_users_list_row_without_a_fact_row_reads_as_healthy_unpaid():
    service, _value_store, _legacy = _service(
        page_rows=[_page_row(
            7, lifecycle_segment=None, data_quality=None, activated_at=None,
            last_meaningful_activity_at=None,
        )],
        activity={}, commercial={},
    )

    response = service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)

    item = response.items[0]
    assert item.operational.state == "healthy"
    assert item.commercial_tier == "unpaid"
    assert item.lifetime_net_purchased_micro == 0
    assert item.lifecycle.segment == "dormant"  # created 90 days ago, no activity row
    assert item.priority_group == "none"


def test_users_list_falls_back_to_the_latest_landed_fact_day_and_warns(capsys):
    """I5: before the night's facts land, or while the job fails, the page joins
    the latest landed day and says so, instead of reading everyone as healthy."""
    stale_day = YESTERDAY - timedelta(days=1)
    service, value_store, _legacy = _service(
        facts=[_fact(1, stale_day)], activity={1: _activity(1)}, commercial={},
    )
    blocked = _page_row(
        1, operational_state="blocked", operational_reason_code="account_restricted"
    )

    def page(*, query, day, limit, offset):
        value_store.page_calls.append((query, day, limit, offset))
        # Only the stale day carries the fact join; yesterday has no rows.
        return ([blocked], 1) if day == stale_day else ([_page_row(1)], 1)

    value_store.list_users_page = page

    response = service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)

    assert value_store.page_calls[0][1] == stale_day
    item = response.items[0]
    assert item.operational.state == "blocked"
    assert item.operational.calculated_at == _utc_midnight(YESTERDAY)
    assert (
        "WARNING: analytics.read.users_facts_stale "
        f"latest={stale_day.isoformat()} expected={YESTERDAY.isoformat()}"
    ) in capsys.readouterr().out


def test_users_list_with_no_facts_at_all_keeps_yesterday_and_warns(capsys):
    service, value_store, _legacy = _service(page_rows=[_page_row(1)], activity={}, commercial={})

    response = service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)

    assert value_store.page_calls[0][1] == YESTERDAY
    assert response.items[0].operational.calculated_at == _utc_midnight(NOW.date())
    assert (
        "WARNING: analytics.read.users_facts_stale "
        f"latest=none expected={YESTERDAY.isoformat()}"
    ) in capsys.readouterr().out


def test_users_list_with_yesterdays_facts_landed_is_silent(capsys):
    service, value_store, _legacy = _service(
        page_rows=[_page_row(1)], facts=[_fact(1, YESTERDAY)], activity={}, commercial={},
    )

    service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)

    assert value_store.page_calls[0][1] == YESTERDAY
    assert "users_facts_stale" not in capsys.readouterr().out


def test_users_list_priority_groups_are_a_pure_function_of_state_and_segment():
    # The store maps a NULL code to no_supported_issue before the service sees it.
    rows = [
        _page_row(1),
        _page_row(2, operational_state="needs_attention", operational_reason_code="three_consecutive_failed_runs", lifecycle_segment="at_risk"),
        _page_row(3, lifecycle_segment="at_risk"),
        _page_row(4, lifecycle_segment="onboarding"),
    ]
    service, _value_store, _legacy = _service(page_rows=rows, activity={}, commercial={})

    items = service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW).items

    assert items[0].operational.state == "healthy"
    assert items[0].operational.reason_code == "no_supported_issue"
    assert [item.priority_group for item in items] == [
        "none", "needs_attention", "healthy_at_risk", "healthy_onboarding"
    ]


def _groups_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, user_group="organic", runs_requested=3, runs_completed=2, operator_cost_micro=200_000),
        _fact(1, YESTERDAY - timedelta(days=1), user_group="organic", runs_requested=1, runs_completed=1, operator_cost_micro=50_000),
        _fact(2, YESTERDAY, user_group="organic", runs_requested=2, runs_completed=0, runs_failed=2),
        _fact(3, YESTERDAY, user_group="partner", runs_requested=1, runs_completed=1),
        _fact(4, YESTERDAY - timedelta(days=20), user_group="partner", runs_requested=9, runs_completed=9),
    ]
    values = dict(
        facts=facts,
        users=((1, "organic"), (2, "organic"), (3, "partner"), (4, "partner"), (5, "unknown")),
        commercial={1: _commercial(1, 6_000_000), 2: _commercial(2), 3: _commercial(3, 1), 4: _commercial(4), 5: _commercial(5)},
    )
    values.update(overrides)
    return _service(**values)[0:2]


def test_group_summary_has_six_rows_in_fixed_order_with_facts_totals():
    service, _store = _groups_service()

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert [row.group for row in response.groups] == list(USER_GROUPS)
    by_group = {row.group: row for row in response.groups}
    assert (by_group["organic"].users, by_group["organic"].total_runs) == (2, 6)
    assert by_group["organic"].atl_cost_micro_usd == 250_000
    assert by_group["organic"].successful_run_users == 1
    assert by_group["organic"].repeat_users == 1
    assert by_group["organic"].paid_users == 1
    assert (by_group["partner"].users, by_group["partner"].total_runs) == (2, 1)  # user 4's row is outside the window
    assert by_group["partner"].paid_users == 1
    assert by_group["unknown"].users == 1
    assert by_group["internal"] == UserGroupSummary(
        group="internal", label="Internal", users=0, successful_run_users=0,
        repeat_users=0, total_runs=0, atl_cost_micro_usd=0, paid_users=0,
    )
    assert response.availability.status == "ready"
    assert response.availability.coverage_end == YESTERDAY


def test_group_filter_zeroes_every_other_group():
    service, _store = _groups_service()

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        user_group="partner", now=NOW,
    )

    by_group = {row.group: row for row in response.groups}
    assert response.selected_user_group == "partner"
    assert by_group["partner"].total_runs == 1
    assert by_group["organic"].total_runs == 0
    assert by_group["organic"].users == 0
    assert by_group["organic"].paid_users == 0


def test_group_summary_is_partial_when_the_fact_read_fails():
    service, store = _groups_service()

    def explode(**_kwargs):
        raise RuntimeError("private detail")

    store.sum_facts_by_user_group = explode

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"
    assert all(row.total_runs == 0 for row in response.groups)


def test_group_summary_is_partial_when_any_fact_row_in_the_window_is_partial():
    """I2: the PR A history copy wrote zero runs as partial rows; a window
    reaching them must not label those zeros ready."""
    facts = _groups_service()[1].facts + [
        _fact(5, YESTERDAY - timedelta(days=2), user_group="unknown",
              runs_requested=0, runs_completed=0, data_quality="partial"),
    ]
    service, _store = _groups_service(facts=facts)

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"
    assert response.availability.coverage_end == YESTERDAY
    # Outside the selected group the partial row does not downgrade the answer.
    filtered = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1),
        user_group="organic", now=NOW,
    )
    assert filtered.availability.status == "ready"


def test_group_summary_clamps_coverage_to_yesterday_for_a_window_ending_today():
    service, _store = _groups_service()

    response = service.get_groups(
        start=YESTERDAY - timedelta(days=6), end=NOW.date() + timedelta(days=1), now=NOW
    )

    assert response.availability.status == "ready"
    assert response.availability.coverage_end == YESTERDAY


def test_group_summary_does_not_read_an_unlanded_window_as_zero_runs():
    service, store = _groups_service()
    store.sum_facts_by_user_group = lambda **_kwargs: pytest.fail("no day has landed")

    response = service.get_groups(
        start=NOW.date(), end=NOW.date() + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"
    assert response.availability.coverage_end is None
    assert {row.group: row.users for row in response.groups}["organic"] == 2


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
        facts=[_fact(1, YESTERDAY)],
        users=((1, "unknown"),),
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

    service, _value_store, _legacy = _service(facts=[_fact(1, YESTERDAY)], users=((1, "unknown"),))
    response = service.get_operational(start=date(2026, 8, 1), end=date(2026, 9, 1), now=NOW)
    assert (response.completed_runs, response.platform_model_cost_micro_usd) == (9, 250_000)


def _operational_service(**overrides):
    facts = [
        _fact(1, YESTERDAY, operational_state="blocked", operational_reason_code="billing_lane_unavailable"),
        _fact(2, YESTERDAY, operational_state="blocked", operational_reason_code="billing_lane_unavailable"),
        _fact(3, YESTERDAY, operational_state="needs_attention", operational_reason_code="invalid_default_credential"),
        _fact(4, YESTERDAY),
        _fact(5, YESTERDAY - timedelta(days=1), operational_state="blocked", operational_reason_code="account_restricted"),
    ]
    values = dict(facts=facts, users=((1, "unknown"), (2, "unknown"), (3, "unknown"), (4, "unknown")))
    values.update(overrides)
    return _service(**values)[0]


def test_operational_state_counts_are_yesterdays_facts():
    service = _operational_service()

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.operational_state_counts == {"blocked": 2, "needs_attention": 1, "healthy": 1}
    assert response.completed_runs == 9  # FakeLegacyService
    assert response.availability.status == "ready"


def test_operational_is_partial_when_the_overview_growth_panel_is_down():
    service = _operational_service(legacy_availability={"growth": False, "friction": True})

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"


def test_missing_operational_subsection_is_reported_as_partial():
    service = _operational_service(legacy_availability={"growth": True, "friction": False})

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.availability.available is True
    assert response.availability.status == "partial"


def test_operational_is_building_before_the_first_fact_day():
    service = _operational_service(facts=[])

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert response.operational_state_counts == {"blocked": 0, "needs_attention": 0, "healthy": 0}
    assert response.availability.status == "building"


def test_null_operational_reason_codes_count_by_state_but_are_not_reasons():
    facts = [
        _fact(1, YESTERDAY, operational_state="healthy", operational_reason_code=None),
        _fact(2, YESTERDAY, operational_state="blocked", operational_reason_code=None),
        _fact(3, YESTERDAY, operational_state="blocked", operational_reason_code="billing_lane_unavailable"),
    ]
    service = _operational_service(
        facts=facts, users=((1, "unknown"), (2, "unknown"), (3, "unknown"))
    )

    response = service.get_operational(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )
    reasons = service.top_operational_reasons(day=YESTERDAY)

    assert response.operational_state_counts == {"blocked": 2, "needs_attention": 0, "healthy": 1}
    assert [(r.reason_code, r.users) for r in reasons] == [("billing_lane_unavailable", 1)]


def test_top_operational_reasons_rank_blocking_reasons_by_users():
    service = _operational_service()

    reasons = service.top_operational_reasons(day=YESTERDAY)

    assert [(r.reason_code, r.users) for r in reasons] == [
        ("billing_lane_unavailable", 2),
        ("invalid_default_credential", 1),
    ]
    assert reasons[0].reason == "No usable model billing lane is available."
    assert service.top_operational_reasons(day=YESTERDAY, limit=1)[0].reason_code == "billing_lane_unavailable"
    assert "top_operational_reasons" not in OperationalAnalyticsResponse.model_fields


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=UTC)


def test_value_profile_combines_live_axes_facts_and_user_transitions():
    transitions = [
        LifecycleTransitionRow(user_id=1, snapshot_date=YESTERDAY, from_segment="growing", to_segment="core",
                               inactive_days=0, data_quality="complete", created_at=NOW),
        # Another user's move on the same day stays off this profile.
        LifecycleTransitionRow(user_id=2, snapshot_date=YESTERDAY, from_segment="growing", to_segment="at_risk",
                               inactive_days=8, data_quality="complete", created_at=NOW),
    ]
    service, _value_store, legacy = _service(
        facts=[_fact(1, YESTERDAY - timedelta(days=offset)) for offset in range(3)],
        transitions=transitions, activity={1: _activity(1)}, commercial={1: _commercial(1, 6_000_000)},
        operational_facts={1: CurrentOperationalFacts(user_id=1, account_restricted=True)},
    )

    profile = service.get_user_profile(user_id=1, start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW)

    assert profile.lifecycle.segment == "core"
    assert profile.operational.state == "blocked"
    assert profile.operational.reason_code == "account_restricted"
    assert profile.commercial.commercial_tier == "invested"
    assert (profile.selected_period_start, profile.selected_period_end) == (YESTERDAY - timedelta(days=6), YESTERDAY + timedelta(days=1))
    assert [
        (t.from_segment, t.to_segment, t.users, t.period_start, t.period_end)
        for t in profile.recent_lifecycle_transitions
    ] == [("growing", "core", 1, YESTERDAY, YESTERDAY + timedelta(days=1))]
    assert legacy.profile_windows == [(_day_start(YESTERDAY - timedelta(days=6)), _day_start(YESTERDAY + timedelta(days=1)))]


def test_value_profile_never_uses_anonymous_transition_rollups():
    aggregate = SimpleNamespace(
        rollup_date=YESTERDAY,
        metric_name="lifecycle_transition",
        event_name="growing",
        user_state="core",
        value_count=99,
        outcome="complete",
    )
    service, _value_store, _legacy = _service(users=((1, "unknown"),), rollups=[aggregate])

    profile = service.get_user_profile(
        user_id=1, start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )

    assert profile.recent_lifecycle_transitions == []


def test_value_profile_raises_lookup_error_for_an_unknown_user():
    service, _value_store, legacy = _service(activity={}, commercial={})
    legacy.raise_lookup = True

    with pytest.raises(LookupError):
        service.get_user_profile(user_id=404, start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW)


def test_internal_accounts_are_excluded_unless_explicitly_included():
    service, value_store, _legacy = _service(page_rows=[_page_row(1)], activity={}, commercial={})

    service.list_users(filters=UserValueFilters(), limit=25, offset=0, now=NOW)
    service.list_users(filters=UserValueFilters(include_internal=True), limit=25, offset=0, now=NOW)

    # The exclusion is a predicate of the SQL page; the service only forwards the flag.
    assert [call[0].include_internal for call in value_store.page_calls] == [False, True]


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

    # I4: facts are internal-free just as the rollups are, so the internal
    # toggle has nothing to protect here and must not drop rollup-backed days.
    internal = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6),
        end=YESTERDAY + timedelta(days=1),
        include_internal=True,
        now=NOW,
    )
    assert internal.movement_segments == response.movement_segments
    assert internal.transitions == response.transitions
    assert internal.weekly_segments == response.weekly_segments


def test_lifecycle_reads_no_raw_events():
    service = _lifecycle_service()
    # FakeRollups.list_events raises. get_lifecycle swallows a failed read into
    # an unavailable section, so the proof is that every section is available.
    response = service.get_lifecycle(
        start=YESTERDAY - timedelta(days=6), end=YESTERDAY + timedelta(days=1), now=NOW
    )
    assert all(section.available for section in response.availability.values())


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
