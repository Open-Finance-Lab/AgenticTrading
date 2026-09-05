"""Display-safe query composition for Admin user-value analytics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .lifecycle import (
    CommercialTier,
    LifecycleResult,
    LifecycleSegment,
    OperationalResult,
    OperationalState,
    is_lifecycle_activity,
)
from .metrics import AnalyticsMetricFilters
from .query_service import (
    AnalyticsQueryService,
    AnalyticsQueryStore,
    AnalyticsUserProfile,
    FailureCategoryCount,
)
from .repository import analytics_store
from .repository_common import positive_limit, positive_user_id
from .value_repository import (
    CommercialValueFact,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    ValueAnalyticsStore,
)


UTC = timezone.utc
_LIFECYCLE_SEGMENTS: tuple[LifecycleSegment, ...] = (
    "new",
    "onboarding",
    "growing",
    "core",
    "at_risk",
    "dormant",
)
_OPERATIONAL_STATES: tuple[OperationalState, ...] = (
    "blocked",
    "needs_attention",
    "healthy",
)
_COMMERCIAL_TIERS: tuple[CommercialTier, ...] = (
    "unpaid",
    "starter",
    "invested",
    "high_value",
)
_TIER_RANK = {tier: rank for rank, tier in enumerate(_COMMERCIAL_TIERS)}
_MOVEMENT_WINDOWS: dict[str, tuple[int, Literal["day", "week", "month"]]] = {
    "5d": (5, "day"),
    "1w": (7, "day"),
    "1m": (31, "week"),
    "1y": (365, "month"),
}
MAX_VALUE_RANGE_DAYS = 180
"""Longest filter range a caller may request, in days.

The router enforces the same bound on the query string; this is the copy that
holds for every caller, and both now read it from here. It was duplicated as a
bare literal, which is how the two could drift apart unnoticed.
"""

_MAX_HISTORY_SCAN_DAYS = max(
    MAX_VALUE_RANGE_DAYS, *(days for days, _granularity in _MOVEMENT_WINDOWS.values())
)
"""Widest span of per-user daily rows one lifecycle request may scan.

`MAX_VALUE_RANGE_DAYS` bounds the *filter* range, but the movement chart reads a
window of its own, so the scan is the wider of the two -- 365 days today, double
the filter cap. Derived rather than written down so that adding a longer
movement range cannot silently widen every user's scan.
"""

_PRIORITY_RANK = {
    "blocked": 0,
    "needs_attention": 1,
    "healthy_at_risk": 2,
    "healthy_onboarding": 3,
    "none": 4,
}


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(UTC)


def _day_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _period_start(value: date, granularity: Literal["day", "week", "month"]) -> date:
    if granularity == "day":
        return value
    if granularity == "week":
        return _week_start(value)
    return value.replace(day=1)


def _parse_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_dates(start: date, end: date) -> tuple[date, date]:
    if not isinstance(start, date) or isinstance(start, datetime):
        raise ValueError("start must be a date")
    if not isinstance(end, date) or isinstance(end, datetime):
        raise ValueError("end must be a date")
    if end <= start:
        raise ValueError("end must be later than start")
    if (end - start).days > MAX_VALUE_RANGE_DAYS:
        raise ValueError(
            f"date range must contain at most {MAX_VALUE_RANGE_DAYS} days"
        )
    return start, end


class SectionAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    stale: bool = False
    status: Literal["ready", "building", "partial", "unavailable"]
    coverage_start: date | None = None
    coverage_end: date | None = None


class LifecycleHeadline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activated_users: int = Field(ge=0)
    core_users: int = Field(ge=0)
    at_risk_users: int = Field(ge=0)
    paid_users: int = Field(ge=0)


class WeeklyLifecycleCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    week_start: date
    segment_counts: dict[LifecycleSegment, int]
    data_quality: Literal["complete", "partial"]


class LifecycleMovementPoint(BaseModel):
    """A display-safe lifecycle snapshot at the selected chart granularity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    period_start: date
    segment_counts: dict[LifecycleSegment, int]
    data_quality: Literal["complete", "partial"]


class LifecycleTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_segment: LifecycleSegment
    to_segment: LifecycleSegment
    users: int = Field(ge=0)
    period_start: date
    period_end: date
    data_quality: Literal["complete", "partial"]


class LifecycleAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    headline: LifecycleHeadline
    segment_counts: dict[LifecycleSegment, int]
    weekly_segments: Sequence[WeeklyLifecycleCount]
    movement_range: Literal["5d", "1w", "1m", "1y"] = "5d"
    movement_granularity: Literal["day", "week", "month"] = "day"
    movement_segments: Sequence[LifecycleMovementPoint] = Field(default_factory=tuple)
    transitions: Sequence[LifecycleTransition]
    availability: dict[str, SectionAvailability]


class RetentionCell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    week: Literal[1, 2, 4]
    mature: bool
    retained_users: int | None = Field(default=None, ge=0)
    eligible_users: int | None = Field(default=None, ge=0)
    rate: float | None = Field(default=None, ge=0, le=1)
    data_quality: Literal["complete", "partial"]


class RetentionCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cohort_week: date
    activated_users: int = Field(ge=0)
    week_1: RetentionCell
    week_2: RetentionCell
    week_4: RetentionCell


class RetentionAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    cohorts: Sequence[RetentionCohort]
    summary_week_1: RetentionCell
    summary_week_2: RetentionCell
    summary_week_4: RetentionCell
    availability: SectionAvailability


class CommercialPeriodSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    purchased_micro: int = Field(ge=0)
    refunded_micro: int = Field(ge=0)
    consumed_micro: int = Field(ge=0)
    admin_grant_activity_micro: int = Field(ge=0)
    platform_model_cost_micro_usd: int = Field(ge=0)


class BalanceTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grant_available_micro: int = Field(ge=0)
    purchased_available_micro: int = Field(ge=0)
    total_available_micro: int = Field(ge=0)


class CommercialAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    tier_counts: dict[CommercialTier, int]
    lifetime_net_purchased_micro: int = Field(ge=0)
    selected_period: CommercialPeriodSummary
    current_balances: BalanceTotals
    availability: SectionAvailability


class OperationalAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    operational_state_counts: dict[OperationalState, int]
    backtest_success_rate: float | None = Field(default=None, ge=0, le=1)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    platform_model_cost_micro_usd: int = Field(ge=0)
    top_failure_categories: Sequence[FailureCategoryCount]
    availability: SectionAvailability


class UserValueFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    q: str | None = Field(default=None, max_length=100)
    lifecycle_segment: LifecycleSegment | None = None
    operational_state: OperationalState | None = None
    commercial_tier: CommercialTier | None = None
    activated: bool | None = None
    last_meaningful_activity_from: datetime | None = None
    last_meaningful_activity_to: datetime | None = None
    priority: bool = False
    legacy_status: str | None = None
    include_internal: bool = False

    @field_validator("q")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "last_meaningful_activity_from",
        "last_meaningful_activity_to",
    )
    @classmethod
    def normalize_activity_timestamp(cls, value: datetime | None) -> datetime | None:
        return _utc(value, "last_meaningful_activity") if value is not None else None

    @model_validator(mode="after")
    def validate_activity_range(self) -> "UserValueFilters":
        if (
            self.last_meaningful_activity_from is not None
            and self.last_meaningful_activity_to is not None
            and self.last_meaningful_activity_to < self.last_meaningful_activity_from
        ):
            raise ValueError("last meaningful activity range is reversed")
        legacy = self.legacy_status
        if legacy is not None and legacy not in {
            "blocked",
            "needs_attention",
            "dormant",
            "onboarding",
            "active",
        }:
            raise ValueError("legacy_status is unsupported")
        return self


PriorityGroup = Literal[
    "blocked",
    "needs_attention",
    "healthy_at_risk",
    "healthy_onboarding",
    "none",
]


class ValueUserListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    joined_at: datetime
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial_tier: CommercialTier
    lifetime_net_purchased_micro: int = Field(ge=0)
    priority_group: PriorityGroup
    profile_path: str


class PaginatedValueUsers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: Sequence[ValueUserListItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class ValueUserProfile(AnalyticsUserProfile):
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial: CommercialValueFact
    selected_period_start: date
    selected_period_end: date
    recent_lifecycle_transitions: Sequence[LifecycleTransition]


def _lifecycle(snapshot: UserValueSnapshot) -> LifecycleResult:
    return LifecycleResult(
        segment=snapshot.lifecycle_segment,
        reason_code=snapshot.lifecycle_reason_code,
        reason=snapshot.lifecycle_reason,
        evidence=tuple(snapshot.lifecycle_evidence),
        activated_at=snapshot.activated_at,
        last_meaningful_activity_at=snapshot.last_meaningful_activity_at,
        inactive_days=snapshot.inactive_days,
        active_days_30d=snapshot.active_days_30d,
        successful_backtests_30d=snapshot.successful_backtests_30d,
        calculated_at=snapshot.calculated_at,
    )


def _operational(snapshot: UserValueSnapshot) -> OperationalResult:
    return OperationalResult(
        state=snapshot.operational_state,
        reason_code=snapshot.operational_reason_code,
        reason=snapshot.operational_reason,
        evidence=tuple(snapshot.operational_evidence),
        calculated_at=snapshot.calculated_at,
    )


def _priority_group(snapshot: UserValueSnapshot) -> PriorityGroup:
    if snapshot.operational_state == "blocked":
        return "blocked"
    if snapshot.operational_state == "needs_attention":
        return "needs_attention"
    if snapshot.lifecycle_segment == "at_risk":
        return "healthy_at_risk"
    if snapshot.lifecycle_segment == "onboarding":
        return "healthy_onboarding"
    return "none"


def _all_users(user_store: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = user_store.list_users_admin(limit=500, offset=offset)
        rows.extend(page)
        if len(page) < 500:
            return rows
        offset += len(page)


class ValueAnalyticsQueryService:
    def __init__(
        self,
        *,
        store: Any = analytics_store,
        user_store: Any | None = None,
        value_store: ValueAnalyticsStore | None = None,
        query_store: AnalyticsQueryStore | None = None,
        legacy_service: AnalyticsQueryService | None = None,
    ) -> None:
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        self.store = store
        self.user_store = user_store
        self.value_store = value_store or ValueAnalyticsStore(store)
        self.query_store = query_store or AnalyticsQueryStore(store)
        self.legacy_service = legacy_service or AnalyticsQueryService(
            store=store,
            user_store=user_store,
        )

    def _eligible_users(self, *, include_internal: bool) -> list[dict[str, Any]]:
        users = _all_users(self.user_store)
        if include_internal:
            return users
        excluded = self.store.list_excluded_user_ids(include_admin_accounts=True)
        return [user for user in users if int(user["id"]) not in excluded]

    @staticmethod
    def _ids(users: Sequence[dict[str, Any]]) -> list[int]:
        return [positive_user_id(user["id"]) for user in users]

    def _current(
        self,
        users: Sequence[dict[str, Any]],
    ) -> dict[int, UserValueSnapshot]:
        result: dict[int, UserValueSnapshot] = {}
        ids = self._ids(users)
        for offset in range(0, len(ids), 500):
            result.update(
                self.value_store.list_current_snapshots(ids[offset : offset + 500])
            )
        return result

    def _commercial(
        self,
        users: Sequence[dict[str, Any]],
        *,
        start: date,
        end: date,
    ) -> dict[int, CommercialValueFact]:
        result: dict[int, CommercialValueFact] = {}
        ids = self._ids(users)
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            result.update(
                self.value_store.list_commercial_values(
                    chunk,
                    start=_day_start(start),
                    end=_day_start(end),
                )
            )
        return result

    def _daily(
        self,
        user_ids: Sequence[int],
        *,
        start: date,
        end: date,
    ) -> list[UserLifecycleDailySnapshot]:
        rows: list[UserLifecycleDailySnapshot] = []
        for offset in range(0, len(user_ids), 500):
            rows.extend(
                self.value_store.list_daily_snapshots(
                    start=start,
                    end=end,
                    user_ids=user_ids[offset : offset + 500],
                )
            )
        return rows

    def _history(
        self,
        user_ids: Sequence[int],
        *,
        start: date,
        end: date,
        use_anonymous_rollups: bool,
        movement_start: date | None = None,
        movement_range: str = "5d",
    ) -> tuple[
        list[WeeklyLifecycleCount],
        list[LifecycleMovementPoint],
        list[LifecycleTransition],
        SectionAvailability,
    ]:
        if movement_range not in _MOVEMENT_WINDOWS:
            raise ValueError("unsupported lifecycle movement range")
        window_days, granularity = _MOVEMENT_WINDOWS[movement_range]
        selected_movement_start = movement_start or end - timedelta(days=window_days)
        if selected_movement_start >= end:
            raise ValueError("lifecycle movement range is empty")
        history_start = max(
            min(start, selected_movement_start),
            end - timedelta(days=_MAX_HISTORY_SCAN_DAYS),
        )
        rows = self._daily(
            user_ids,
            start=history_start - timedelta(days=1),
            end=end,
        )
        by_date: dict[date, list[UserLifecycleDailySnapshot]] = defaultdict(list)
        for row in rows:
            by_date[row.snapshot_date].append(row)
        rollups = (
            self.query_store.rollups.list_rollups(start=history_start, end=end)
            if use_anonymous_rollups
            else []
        )
        rollup_counts: dict[date, dict[LifecycleSegment, int]] = defaultdict(dict)
        rollup_quality: dict[date, str] = {}
        for row in rollups:
            if row.metric_name != "lifecycle_segment_count":
                continue
            rollup_counts[row.rollup_date][row.user_state] = row.value_count
            if row.outcome == "partial":
                rollup_quality[row.rollup_date] = "partial"
            else:
                rollup_quality.setdefault(row.rollup_date, "complete")

        daily_counts: dict[date, dict[LifecycleSegment, int]] = {}
        daily_quality: dict[date, str] = {}
        direct_dates = {day for day in by_date if history_start <= day < end}
        for day in direct_dates:
            counts = Counter(row.lifecycle_segment for row in by_date[day])
            daily_counts[day] = {
                segment: int(counts.get(segment, 0)) for segment in _LIFECYCLE_SEGMENTS
            }
            daily_quality[day] = (
                "partial"
                if any(row.data_quality == "partial" for row in by_date[day])
                else "complete"
            )
        for day, counts in rollup_counts.items():
            if day not in direct_dates:
                daily_counts[day] = {
                    segment: int(counts.get(segment, 0))
                    for segment in _LIFECYCLE_SEGMENTS
                }
                daily_quality[day] = rollup_quality.get(day, "partial")

        weekly: list[WeeklyLifecycleCount] = []
        by_week: dict[date, list[date]] = defaultdict(list)
        for day in daily_counts:
            if not start <= day < end:
                continue
            by_week[_week_start(day)].append(day)
        for week, dates in sorted(by_week.items()):
            latest = max(dates)
            weekly.append(
                WeeklyLifecycleCount(
                    week_start=week,
                    segment_counts=daily_counts[latest],
                    data_quality=daily_quality[latest],
                )
            )

        movement: list[LifecycleMovementPoint] = []
        by_period: dict[date, list[date]] = defaultdict(list)
        for day in daily_counts:
            if selected_movement_start <= day < end:
                by_period[_period_start(day, granularity)].append(day)
        for period, dates in sorted(by_period.items()):
            latest = max(dates)
            movement.append(
                LifecycleMovementPoint(
                    period_start=max(period, selected_movement_start),
                    segment_counts=daily_counts[latest],
                    data_quality=daily_quality[latest],
                )
            )

        transition_counts: Counter[tuple[str, str]] = Counter()
        transition_partial: set[tuple[str, str]] = set()
        by_user_date = {(row.user_id, row.snapshot_date): row for row in rows}
        for row in rows:
            if not start <= row.snapshot_date < end:
                continue
            previous = by_user_date.get(
                (row.user_id, row.snapshot_date - timedelta(days=1))
            )
            if previous is None or previous.lifecycle_segment == row.lifecycle_segment:
                continue
            key = (previous.lifecycle_segment, row.lifecycle_segment)
            transition_counts[key] += 1
            if previous.data_quality == "partial" or row.data_quality == "partial":
                transition_partial.add(key)
        for row in rollups:
            if (
                row.metric_name != "lifecycle_transition"
                or not start <= row.rollup_date < end
                or row.rollup_date in direct_dates
            ):
                continue
            key = (row.event_name, row.user_state)
            transition_counts[key] += row.value_count
            if row.outcome == "partial":
                transition_partial.add(key)
        transitions = [
            LifecycleTransition(
                from_segment=from_segment,
                to_segment=to_segment,
                users=count,
                period_start=start,
                period_end=end,
                data_quality=(
                    "partial"
                    if (from_segment, to_segment) in transition_partial
                    else "complete"
                ),
            )
            for (from_segment, to_segment), count in sorted(
                transition_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        coverage = sorted(day for day in daily_counts if start <= day < end)
        if not coverage:
            availability = SectionAvailability(
                available=False,
                status="building",
            )
        else:
            partial = (
                any(daily_quality[day] == "partial" for day in coverage)
                or coverage[0] > start
                or coverage[-1] < end - timedelta(days=1)
            )
            availability = SectionAvailability(
                available=True,
                status="partial" if partial else "ready",
                coverage_start=coverage[0],
                coverage_end=coverage[-1],
            )
        return weekly, movement, transitions, availability

    def get_lifecycle(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        movement_range: str = "5d",
        now: datetime | None = None,
    ) -> LifecycleAnalyticsResponse:
        start, end = _validate_dates(start, end)
        if movement_range not in _MOVEMENT_WINDOWS:
            raise ValueError("unsupported lifecycle movement range")
        window_days, _granularity = _MOVEMENT_WINDOWS[movement_range]
        movement_start = end - timedelta(days=window_days)
        current_time = _utc(now or datetime.now(UTC), "now")
        users = self._eligible_users(include_internal=include_internal)
        current = self._current(users)
        counts = Counter(snapshot.lifecycle_segment for snapshot in current.values())
        segment_counts = {
            segment: int(counts.get(segment, 0)) for segment in _LIFECYCLE_SEGMENTS
        }
        availability: dict[str, SectionAvailability] = {
            "current": SectionAvailability(
                available=bool(current) or not users,
                status="ready" if len(current) == len(users) else "building",
            )
        }
        try:
            commercial = self._commercial(users, start=start, end=end)
            paid = sum(
                fact.lifetime_net_purchased_micro > 0 for fact in commercial.values()
            )
            availability["commercial"] = SectionAvailability(
                available=True,
                status="ready",
            )
        except Exception:
            paid = 0
            availability["commercial"] = SectionAvailability(
                available=False,
                status="unavailable",
            )
        try:
            weekly, movement, transitions, history_availability = self._history(
                self._ids(users),
                start=start,
                end=end,
                use_anonymous_rollups=not include_internal,
                movement_start=movement_start,
                movement_range=movement_range,
            )
        except Exception:
            weekly, movement, transitions = [], [], []
            history_availability = SectionAvailability(
                available=False,
                status="unavailable",
            )
        availability["history"] = history_availability
        return LifecycleAnalyticsResponse(
            as_of=current_time,
            headline=LifecycleHeadline(
                activated_users=sum(
                    snapshot.activated_at is not None for snapshot in current.values()
                ),
                core_users=segment_counts["core"],
                at_risk_users=segment_counts["at_risk"],
                paid_users=paid,
            ),
            segment_counts=segment_counts,
            weekly_segments=weekly,
            movement_range=movement_range,
            movement_granularity=_MOVEMENT_WINDOWS[movement_range][1],
            movement_segments=movement,
            transitions=transitions,
            availability=availability,
        )

    def _retention_quality(
        self,
        rows: Sequence[UserLifecycleDailySnapshot],
        *,
        user_ids: Sequence[int],
        start: date,
        end: date,
    ) -> Literal["complete", "partial"]:
        selected = [row for row in rows if start <= row.snapshot_date < end]
        expected = {
            (user_id, start + timedelta(days=offset))
            for user_id in user_ids
            for offset in range((end - start).days)
        }
        observed = {(row.user_id, row.snapshot_date) for row in selected}
        if expected - observed or any(
            row.data_quality == "partial" for row in selected
        ):
            return "partial"
        return "complete"

    @staticmethod
    def _summary_cell(
        week: Literal[1, 2, 4],
        cells: Sequence[RetentionCell],
    ) -> RetentionCell:
        mature = [cell for cell in cells if cell.mature]
        retained = sum(cell.retained_users or 0 for cell in mature)
        eligible = sum(cell.eligible_users or 0 for cell in mature)
        return RetentionCell(
            week=week,
            mature=bool(mature),
            retained_users=retained if mature else None,
            eligible_users=eligible if mature else None,
            rate=(retained / eligible if eligible else None),
            data_quality=(
                "partial"
                if not mature or any(cell.data_quality == "partial" for cell in mature)
                else "complete"
            ),
        )

    def get_retention(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        now: datetime | None = None,
    ) -> RetentionAnalyticsResponse:
        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        users = self._eligible_users(include_internal=include_internal)
        current = self._current(users)
        cohort_start = _week_start(start)
        cohort_users: dict[date, list[int]] = defaultdict(list)
        for user_id, snapshot in current.items():
            if snapshot.activated_at is None:
                continue
            week = _week_start(snapshot.activated_at.date())
            if cohort_start <= week < end:
                cohort_users[week].append(user_id)
        ids = sorted(
            {user_id for values in cohort_users.values() for user_id in values}
        )
        activity_by_user: dict[int, list[datetime]] = defaultdict(list)
        if ids:
            activity_end = min(_day_start(end + timedelta(days=35)), current_time)
            events = self.query_store.rollups.list_events(
                start=_day_start(cohort_start),
                end=activity_end + timedelta(microseconds=1),
                include_internal=True,
            )
            id_set = set(ids)
            for event in events:
                if event.user_id in id_set and is_lifecycle_activity(event):
                    activity_by_user[event.user_id].append(event.occurred_at)
            for offset in range(0, len(ids), 500):
                chunk = ids[offset : offset + 500]
                credit_activity = self.value_store.list_credit_activity(
                    chunk,
                    start=_day_start(cohort_start),
                    end=activity_end + timedelta(microseconds=1),
                )
                for user_id, timestamps in credit_activity.items():
                    activity_by_user[user_id].extend(timestamps)
        daily_end = min(
            end + timedelta(days=35), current_time.date() + timedelta(days=1)
        )
        daily = (
            self._daily(
                ids,
                start=cohort_start,
                end=daily_end,
            )
            if ids
            else []
        )

        cohorts: list[RetentionCohort] = []
        all_cells: dict[int, list[RetentionCell]] = {1: [], 2: [], 4: []}
        for cohort_week, members in sorted(cohort_users.items()):
            cells: dict[int, RetentionCell] = {}
            for week in (1, 2, 4):
                target_start = cohort_week + timedelta(days=week * 7)
                target_end = target_start + timedelta(days=7)
                mature = current_time >= _day_start(target_end)
                quality = self._retention_quality(
                    daily,
                    user_ids=members,
                    start=target_start,
                    end=target_end,
                )
                retained = None
                eligible = None
                rate = None
                if mature:
                    eligible = len(members)
                    retained = sum(
                        any(
                            _day_start(target_start)
                            <= timestamp
                            < _day_start(target_end)
                            for timestamp in activity_by_user.get(user_id, ())
                        )
                        for user_id in members
                    )
                    rate = retained / eligible if eligible else None
                cell = RetentionCell(
                    week=week,
                    mature=mature,
                    retained_users=retained,
                    eligible_users=eligible,
                    rate=rate,
                    data_quality=quality,
                )
                cells[week] = cell
                all_cells[week].append(cell)
            cohorts.append(
                RetentionCohort(
                    cohort_week=cohort_week,
                    activated_users=len(members),
                    week_1=cells[1],
                    week_2=cells[2],
                    week_4=cells[4],
                )
            )
        partial = any(
            cell.data_quality == "partial"
            for cells in all_cells.values()
            for cell in cells
            if cell.mature
        )
        return RetentionAnalyticsResponse(
            as_of=current_time,
            cohorts=cohorts,
            summary_week_1=self._summary_cell(1, all_cells[1]),
            summary_week_2=self._summary_cell(2, all_cells[2]),
            summary_week_4=self._summary_cell(4, all_cells[4]),
            availability=SectionAvailability(
                available=bool(current) or not users,
                status=(
                    "partial"
                    if partial
                    else "ready" if len(current) == len(users) else "building"
                ),
                coverage_start=cohort_start if cohorts else None,
                coverage_end=end - timedelta(days=1) if cohorts else None,
            ),
        )

    def get_commercial(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        now: datetime | None = None,
    ) -> CommercialAnalyticsResponse:
        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        users = self._eligible_users(include_internal=include_internal)
        facts = self._commercial(users, start=start, end=end)
        tier_counts = Counter(fact.commercial_tier for fact in facts.values())
        overview = self.legacy_service.get_overview(
            filters=AnalyticsMetricFilters(
                start=_day_start(start),
                end=_day_start(end),
                include_internal=include_internal,
            ),
            now=current_time,
        )
        cost_available = overview.availability["growth"].available
        return CommercialAnalyticsResponse(
            as_of=current_time,
            tier_counts={
                tier: int(tier_counts.get(tier, 0)) for tier in _COMMERCIAL_TIERS
            },
            lifetime_net_purchased_micro=sum(
                fact.lifetime_net_purchased_micro for fact in facts.values()
            ),
            selected_period=CommercialPeriodSummary(
                purchased_micro=sum(fact.purchased_micro for fact in facts.values()),
                refunded_micro=sum(fact.refunded_micro for fact in facts.values()),
                consumed_micro=sum(fact.consumed_micro for fact in facts.values()),
                admin_grant_activity_micro=sum(
                    fact.admin_grant_activity_micro for fact in facts.values()
                ),
                platform_model_cost_micro_usd=round(
                    (overview.platform_model_cost_usd or 0) * 1_000_000
                ),
            ),
            current_balances=BalanceTotals(
                grant_available_micro=sum(
                    fact.grant_available_micro for fact in facts.values()
                ),
                purchased_available_micro=sum(
                    fact.purchased_available_micro for fact in facts.values()
                ),
                total_available_micro=sum(
                    fact.total_available_micro for fact in facts.values()
                ),
            ),
            availability=SectionAvailability(
                available=True,
                status="ready" if cost_available else "partial",
            ),
        )

    def get_operational(
        self,
        *,
        start: date,
        end: date,
        include_internal: bool = False,
        provider_id: str | None = None,
        model_id: str | None = None,
        billing_mode: str | None = None,
        now: datetime | None = None,
    ) -> OperationalAnalyticsResponse:
        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        users = self._eligible_users(include_internal=include_internal)
        current = self._current(users)
        counts = Counter(snapshot.operational_state for snapshot in current.values())
        overview = self.legacy_service.get_overview(
            filters=AnalyticsMetricFilters(
                start=_day_start(start),
                end=_day_start(end),
                include_internal=include_internal,
                provider_id=provider_id,
                model_id=model_id,
                billing_mode=billing_mode,
            ),
            now=current_time,
        )
        growth_available = overview.availability["growth"].available
        friction_available = overview.availability["friction"].available
        current_complete = len(current) == len(users)
        available = bool(current) or not users or growth_available or friction_available
        if not current_complete:
            status = "building"
        elif growth_available and friction_available:
            status = "ready"
        else:
            status = "partial" if available else "unavailable"
        return OperationalAnalyticsResponse(
            as_of=current_time,
            operational_state_counts={
                state: int(counts.get(state, 0)) for state in _OPERATIONAL_STATES
            },
            backtest_success_rate=overview.backtest_success_rate,
            completed_runs=overview.completed_runs or 0,
            failed_runs=overview.failed_runs or 0,
            input_tokens=overview.input_tokens or 0,
            output_tokens=overview.output_tokens or 0,
            platform_model_cost_micro_usd=round(
                (overview.platform_model_cost_usd or 0) * 1_000_000
            ),
            top_failure_categories=overview.top_failure_categories,
            availability=SectionAvailability(
                available=available,
                status=status,
            ),
        )

    def list_users(
        self,
        *,
        filters: UserValueFilters,
        limit: int,
        offset: int,
        now: datetime | None = None,
    ) -> PaginatedValueUsers:
        if not isinstance(filters, UserValueFilters):
            filters = UserValueFilters.model_validate(filters)
        page_size = positive_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        current_time = _utc(now or datetime.now(UTC), "now")
        users = self._eligible_users(include_internal=filters.include_internal)
        current = self._current(users)
        commercial = self._commercial(
            users,
            start=current_time.date() - timedelta(days=30),
            end=current_time.date() + timedelta(days=1),
        )
        legacy = self.query_store.list_snapshots() if filters.legacy_status else {}
        selected: list[ValueUserListItem] = []
        needle = filters.q.casefold() if filters.q else None
        for user in users:
            user_id = int(user["id"])
            snapshot = current.get(user_id)
            fact = commercial.get(user_id)
            if snapshot is None or fact is None:
                continue
            group = _priority_group(snapshot)
            if filters.priority and group == "none":
                continue
            if filters.lifecycle_segment not in {None, snapshot.lifecycle_segment}:
                continue
            if filters.operational_state not in {None, snapshot.operational_state}:
                continue
            if filters.commercial_tier not in {None, fact.commercial_tier}:
                continue
            if filters.activated is not None:
                if (snapshot.activated_at is not None) != filters.activated:
                    continue
            if filters.last_meaningful_activity_from is not None and (
                snapshot.last_meaningful_activity_at is None
                or snapshot.last_meaningful_activity_at
                < filters.last_meaningful_activity_from
            ):
                continue
            if filters.last_meaningful_activity_to is not None and (
                snapshot.last_meaningful_activity_at is None
                or snapshot.last_meaningful_activity_at
                > filters.last_meaningful_activity_to
            ):
                continue
            if filters.legacy_status is not None and (
                user_id not in legacy or legacy[user_id].status != filters.legacy_status
            ):
                continue
            if needle is not None and needle not in (
                f"{user.get('display_name', '')} {user.get('email', '')}".casefold()
            ):
                continue
            selected.append(
                ValueUserListItem(
                    user_id=user_id,
                    display_name=str(user.get("display_name") or ""),
                    email=str(user.get("email") or ""),
                    joined_at=_parse_timestamp(user["created_at"]),
                    lifecycle=_lifecycle(snapshot),
                    operational=_operational(snapshot),
                    commercial_tier=fact.commercial_tier,
                    lifetime_net_purchased_micro=fact.lifetime_net_purchased_micro,
                    priority_group=group,
                    profile_path=f"/admin/analytics/users/{user_id}",
                )
            )
        if filters.priority:
            selected.sort(
                key=lambda item: (
                    _PRIORITY_RANK[item.priority_group],
                    -_TIER_RANK[item.commercial_tier],
                    -item.lifetime_net_purchased_micro,
                    -item.lifecycle.inactive_days,
                    item.user_id,
                )
            )
        else:
            selected.sort(key=lambda item: item.user_id)
        return PaginatedValueUsers(
            items=selected[offset : offset + page_size],
            total=len(selected),
            limit=page_size,
            offset=offset,
        )

    def get_user_profile(
        self,
        *,
        user_id: int,
        start: date,
        end: date,
        now: datetime | None = None,
    ) -> ValueUserProfile:
        start, end = _validate_dates(start, end)
        subject_id = positive_user_id(user_id)
        current_time = _utc(now or datetime.now(UTC), "now")
        snapshot = self.value_store.get_current_snapshot(subject_id)
        if snapshot is None:
            raise LookupError("Analytics value snapshot was not found")
        commercial = self.value_store.list_commercial_values(
            [subject_id],
            start=_day_start(start),
            end=_day_start(end),
        )[subject_id]
        legacy = self.legacy_service.get_user_profile(
            user_id=subject_id,
            now=current_time,
            start=_day_start(start),
            end=_day_start(end),
        )
        _weekly, _movement, transitions, _availability = self._history(
            [subject_id],
            start=start,
            end=end,
            use_anonymous_rollups=False,
        )
        return ValueUserProfile(
            **legacy.model_dump(),
            lifecycle=_lifecycle(snapshot),
            operational=_operational(snapshot),
            commercial=commercial,
            selected_period_start=start,
            selected_period_end=end,
            recent_lifecycle_transitions=transitions,
        )


__all__ = [
    "BalanceTotals",
    "CommercialAnalyticsResponse",
    "CommercialPeriodSummary",
    "LifecycleAnalyticsResponse",
    "LifecycleHeadline",
    "LifecycleMovementPoint",
    "LifecycleTransition",
    "OperationalAnalyticsResponse",
    "PaginatedValueUsers",
    "RetentionAnalyticsResponse",
    "RetentionCell",
    "RetentionCohort",
    "SectionAvailability",
    "UserValueFilters",
    "ValueAnalyticsQueryService",
    "ValueUserListItem",
    "ValueUserProfile",
    "WeeklyLifecycleCount",
]
