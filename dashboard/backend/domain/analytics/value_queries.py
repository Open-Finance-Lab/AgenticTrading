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
    activation_cohort_week,
    calculate_lifecycle,
    operational_reason,
    operational_result_from_code,
)
from .lifecycle_reads import build_lifecycle_inputs
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
    GroupFactTotals,
    UserActivity,
    UserPageQuery,
    build_value_analytics_store,
)
from dashboard.backend.domain.user_groups import (
    USER_GROUPS,
    USER_GROUP_LABELS,
    UserGroup,
    coerce_user_group,
    parse_user_group,
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
"""Widest span of daily fact and rollup rows one lifecycle request may read.

`MAX_VALUE_RANGE_DAYS` bounds the *filter* range, but the movement chart reads a
window of its own, so the scan is the wider of the two -- 365 days today, double
the filter cap. Derived rather than written down so that adding a longer
movement range cannot silently widen the scan.
"""

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
    # None when the model-usage read behind it failed: 0 is a real answer
    # ("no platform calls"), so it must not also stand for "unknown".
    platform_model_cost_micro_usd: int | None = Field(ge=0)


class BalanceTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grant_available_micro: int = Field(ge=0)
    purchased_available_micro: int = Field(ge=0)
    total_available_micro: int = Field(ge=0)


class DailyMicroTotals(BaseModel):
    """One UTC day of ledger movement for the /admin Credits & Cost chart."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: date
    amount_micro: int = Field(ge=0)


class CommercialAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    tier_counts: dict[CommercialTier, int]
    lifetime_net_purchased_micro: int = Field(ge=0)
    selected_period: CommercialPeriodSummary
    current_balances: BalanceTotals
    # §9 chart series, one entry per UTC day with movement. Each sums to the
    # selected_period figure of the same name (gross purchases; model spend
    # only), so a chart never disagrees with the headline drawn above it.
    # None when the ledger read failed: "partial" alone cannot say which half
    # of the section degraded, and [] is the real answer "no purchases".
    purchased_by_day: list[DailyMicroTotals] | None
    consumed_by_day: list[DailyMicroTotals] | None
    availability: SectionAvailability


class OperationalAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    operational_state_counts: dict[OperationalState, int]
    backtest_success_rate: float | None = Field(default=None, ge=0, le=1)
    # The growth read's figures, each None when that read failed: 0 is a real
    # answer ("no runs", "no platform calls"), so it must not also stand for
    # "unknown".
    completed_runs: int | None = Field(ge=0)
    failed_runs: int | None = Field(ge=0)
    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    platform_model_cost_micro_usd: int | None = Field(ge=0)
    top_failure_categories: Sequence[FailureCategoryCount]
    availability: SectionAvailability


class OperationalReasonCount(BaseModel):
    """One blocking or attention reason and how many users carried it.

    Service-level only in PR B (design §9: PR D puts `top_operational_reasons`
    on OperationalAnalyticsResponse in the contract re-cut).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    reason: str
    users: int = Field(ge=0)


class UserGroupSummary(BaseModel):
    """Display-safe value metrics for one canonical account-source group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group: UserGroup
    label: str
    users: int = Field(ge=0)
    successful_run_users: int = Field(ge=0)
    repeat_users: int = Field(ge=0)
    total_runs: int = Field(ge=0)
    atl_cost_micro_usd: int = Field(ge=0)
    paid_users: int = Field(ge=0)


class GroupAnalyticsResponse(BaseModel):
    """Fixed-order, admin-only source summary for the selected UTC period."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    groups: Sequence[UserGroupSummary]
    selected_user_group: UserGroup | None = None
    availability: SectionAvailability


class UserValueFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    q: str | None = Field(default=None, max_length=100)
    lifecycle_segment: LifecycleSegment | None = None
    operational_state: OperationalState | None = None
    commercial_tier: CommercialTier | None = None
    user_group: UserGroup | None = None
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


def _priority_group(
    operational_state: str, lifecycle_segment: str | None
) -> PriorityGroup:
    """Yesterday's fact values -> group; the same rule as ``_users_page_sql``'s
    priority predicate and sort, so a row agrees with where SQL placed it."""
    if operational_state == "blocked":
        return "blocked"
    if operational_state == "needs_attention":
        return "needs_attention"
    if lifecycle_segment == "at_risk":
        return "healthy_at_risk"
    if lifecycle_segment == "onboarding":
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


def _transition_models(
    counts: Counter[tuple[str, str]],
    partial: set[tuple[str, str]],
    *,
    start: date,
    end: date,
) -> list[LifecycleTransition]:
    """Window-level transition totals, largest first, stamped with the window."""
    return [
        LifecycleTransition(
            from_segment=from_segment,
            to_segment=to_segment,
            users=count,
            period_start=start,
            period_end=end,
            data_quality=(
                "partial" if (from_segment, to_segment) in partial else "complete"
            ),
        )
        for (from_segment, to_segment), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


class ValueAnalyticsQueryService:
    def __init__(
        self,
        *,
        store: Any = analytics_store,
        user_store: Any | None = None,
        value_store=None,
        query_store: AnalyticsQueryStore | None = None,
        legacy_service: AnalyticsQueryService | None = None,
    ) -> None:
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        self.store = store
        self.user_store = user_store
        self.value_store = value_store or build_value_analytics_store(store)
        self.query_store = query_store or AnalyticsQueryStore(store)
        # A defaulted legacy service shares this service's value store, so the
        # profile's axes and fact sums read the stores this service was given.
        self.legacy_service = legacy_service or AnalyticsQueryService(
            store=store,
            user_store=user_store,
            value_store=self.value_store,
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

    def _ledger_days(
        self,
        users: Sequence[dict[str, Any]],
        *,
        start: date,
        end: date,
    ) -> tuple[list[DailyMicroTotals], list[DailyMicroTotals]]:
        """(purchased_by_day, consumed_by_day) for the Credits & Cost chart.

        Chunked by 500 like ``_commercial`` so the bound-parameter count never
        grows with the user table; per-day sums are additive across chunks.
        """
        purchased: dict[date, int] = defaultdict(int)
        consumed: dict[date, int] = defaultdict(int)
        ids = self._ids(users)
        for offset in range(0, len(ids), 500):
            for row in self.value_store.list_ledger_days(
                ids[offset : offset + 500],
                start=_day_start(start),
                end=_day_start(end),
            ):
                purchased[row.day] += row.purchased_micro
                consumed[row.day] += row.consumed_micro
        return (
            [
                DailyMicroTotals(day=day, amount_micro=amount)
                for day, amount in sorted(purchased.items())
                if amount > 0
            ],
            [
                DailyMicroTotals(day=day, amount_micro=amount)
                for day, amount in sorted(consumed.items())
                if amount > 0
            ],
        )

    def _history(
        self,
        *,
        start: date,
        end: date,
        yesterday: date,
        use_anonymous_rollups: bool,
        movement_start: date | None = None,
        movement_range: str = "5d",
    ) -> tuple[
        list[WeeklyLifecycleCount],
        list[LifecycleMovementPoint],
        list[LifecycleTransition],
        SectionAvailability,
    ]:
        """Population history from ``user_daily_facts`` and ``lifecycle_transitions``.

        Both are grouped in SQL, so the read does not grow with the user table.
        Days the facts table no longer holds fall back to the anonymous
        lifecycle rollups. Facts exist only through ``yesterday``, so coverage
        is judged against that day rather than against ``end``.
        """
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
        daily_counts: dict[date, dict[LifecycleSegment, int]] = {}
        daily_quality: dict[date, str] = {}
        for row in self.value_store.count_segments_by_date(start=history_start, end=end):
            counts = daily_counts.setdefault(
                row.snapshot_date, {segment: 0 for segment in _LIFECYCLE_SEGMENTS}
            )
            counts[row.lifecycle_segment] = row.users
            if row.partial_rows:
                daily_quality[row.snapshot_date] = "partial"
            else:
                daily_quality.setdefault(row.snapshot_date, "complete")
        direct_dates = set(daily_counts)

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
            if start <= day < end:
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
        for row in self.value_store.count_transitions(start=start, end=end):
            key = (row.from_segment, row.to_segment)
            transition_counts[key] += row.users
            if row.partial_rows:
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
        transitions = _transition_models(
            transition_counts, transition_partial, start=start, end=end
        )
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
                or coverage[-1] < min(end - timedelta(days=1), yesterday)
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
        """Segment counts are as of yesterday's fact rows (design §6.3, §6.10).

        Fact rows hold no internal or excluded accounts, so ``include_internal``
        widens only the activation and paid headlines, which read
        ``user_activity`` and the ledger.
        """
        start, end = _validate_dates(start, end)
        if movement_range not in _MOVEMENT_WINDOWS:
            raise ValueError("unsupported lifecycle movement range")
        window_days, _granularity = _MOVEMENT_WINDOWS[movement_range]
        movement_start = end - timedelta(days=window_days)
        current_time = _utc(now or datetime.now(UTC), "now")
        yesterday = current_time.date() - timedelta(days=1)
        cells = self.value_store.count_states_for_date(yesterday)
        segment_counts = {
            segment: sum(
                cell.users for cell in cells if cell.lifecycle_segment == segment
            )
            for segment in _LIFECYCLE_SEGMENTS
        }
        availability: dict[str, SectionAvailability] = {
            "current": SectionAvailability(
                available=True,
                status="ready" if cells else "building",
                coverage_start=yesterday if cells else None,
                coverage_end=yesterday if cells else None,
            )
        }
        try:
            activated_users = self.value_store.count_activated_users(
                include_internal=include_internal
            )
        except Exception:
            activated_users = 0
            availability["current"] = SectionAvailability(
                available=bool(cells),
                status="partial",
            )
        try:
            users = self._eligible_users(include_internal=include_internal)
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
                start=start,
                end=end,
                yesterday=yesterday,
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
                activated_users=activated_users,
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
        cohort_start = _week_start(start)
        # Every Monday below ``end`` is a cohort week, so the last one is the
        # Monday of the window's final day; its members activated before the
        # next Monday.
        activations = self.value_store.list_activations(
            start=_day_start(cohort_start),
            end=_day_start(_week_start(end - timedelta(days=1)) + timedelta(days=7)),
            include_internal=include_internal,
        )
        cohort_users: dict[date, list[int]] = defaultdict(list)
        for user_id, activated_at in activations.items():
            week = activation_cohort_week(activated_at)
            if cohort_start <= week < end:
                cohort_users[week].append(user_id)
        ids = sorted(
            {user_id for members in cohort_users.values() for user_id in members}
        )
        grid_end = min(
            end + timedelta(days=35), current_time.date() + timedelta(days=1)
        )
        active_dates: dict[int, list[date]] = {}
        day_quality: dict[date, str] = {}
        if ids:
            for offset in range(0, len(ids), 500):
                active_dates.update(
                    self.value_store.list_active_dates(
                        ids[offset : offset + 500], start=cohort_start, end=grid_end
                    )
                )
            # A day is trustworthy only when it has fact rows and none is partial.
            for row in self.value_store.count_segments_by_date(
                start=cohort_start, end=grid_end
            ):
                if row.partial_rows:
                    day_quality[row.snapshot_date] = "partial"
                else:
                    day_quality.setdefault(row.snapshot_date, "complete")

        def week_quality(
            target_start: date, target_end: date
        ) -> Literal["complete", "partial"]:
            day = target_start
            while day < target_end:
                if day_quality.get(day) != "complete":
                    return "partial"
                day += timedelta(days=1)
            return "complete"

        cohorts: list[RetentionCohort] = []
        all_cells: dict[int, list[RetentionCell]] = {1: [], 2: [], 4: []}
        for cohort_week, members in sorted(cohort_users.items()):
            cells: dict[int, RetentionCell] = {}
            for week in (1, 2, 4):
                target_start = cohort_week + timedelta(days=week * 7)
                target_end = target_start + timedelta(days=7)
                mature = current_time >= _day_start(target_end)
                retained = None
                eligible = None
                rate = None
                if mature:
                    eligible = len(members)
                    retained = sum(
                        any(
                            target_start <= day < target_end
                            for day in active_dates.get(user_id, ())
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
                    data_quality=week_quality(target_start, target_end),
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
                available=True,
                status="partial" if partial else "ready",
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
        # The overview's platform-cost headline alone, not a whole second
        # overview: the page already draws that figure from /overview.
        platform_cost_micro = self.legacy_service.get_platform_model_cost_micro(
            filters=AnalyticsMetricFilters(
                start=_day_start(start),
                end=_day_start(end),
                include_internal=include_internal,
            ),
            now=current_time,
        )
        cost_available = platform_cost_micro is not None
        try:
            purchased_by_day, consumed_by_day = self._ledger_days(
                users, start=start, end=end
            )
            ledger_available = True
        except Exception as exc:
            # An empty series renders as "No settled purchases in this range",
            # so a failed read must be distinguishable from a quiet one: say so
            # in the log, ship None rather than [], and downgrade the section.
            print(
                "[analytics] ERROR commercial per-day ledger unavailable: "
                f"{type(exc).__name__}"
            )
            purchased_by_day = consumed_by_day = None
            ledger_available = False
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
                # None when the read behind it failed.
                platform_model_cost_micro_usd=platform_cost_micro,
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
            purchased_by_day=purchased_by_day,
            consumed_by_day=consumed_by_day,
            availability=SectionAvailability(
                available=True,
                status="ready" if cost_available and ledger_available else "partial",
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
        yesterday = current_time.date() - timedelta(days=1)
        # Fact rows hold no internal or excluded accounts, so ``include_internal``
        # widens only the overview figures below.
        cells = self.value_store.count_states_for_date(yesterday)
        counts = {
            state: sum(cell.users for cell in cells if cell.operational_state == state)
            for state in _OPERATIONAL_STATES
        }
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
        available = bool(cells) or growth_available or friction_available
        if not cells:
            # Yesterday's facts have not landed: empty counts are not "none blocked".
            status = "building"
        elif growth_available and friction_available:
            status = "ready"
        else:
            status = "partial" if available else "unavailable"
        return OperationalAnalyticsResponse(
            as_of=current_time,
            operational_state_counts=counts,
            # Unknown whenever the growth read failed, even part-way: the
            # overview can compute a figure and then fail the block.
            backtest_success_rate=(
                overview.backtest_success_rate if growth_available else None
            ),
            completed_runs=overview.completed_runs if growth_available else None,
            failed_runs=overview.failed_runs if growth_available else None,
            input_tokens=overview.input_tokens if growth_available else None,
            output_tokens=overview.output_tokens if growth_available else None,
            platform_model_cost_micro_usd=(
                None
                if not growth_available or overview.platform_model_cost_usd is None
                else round(overview.platform_model_cost_usd * 1_000_000)
            ),
            top_failure_categories=overview.top_failure_categories,
            availability=SectionAvailability(
                available=available,
                status=status,
            ),
        )

    def top_operational_reasons(
        self, *, day: date, limit: int = 10
    ) -> list[OperationalReasonCount]:
        """Why users were blocked or needed attention on ``day`` (design §9).

        One grouped read over the fact rows; the reason text is the rule
        table's copy, never the stored code alone.
        """
        page_size = positive_limit(limit)
        totals: Counter[str] = Counter()
        for cell in self.value_store.count_states_for_date(day):
            # A NULL code is not a supported reason (it reads as
            # no_supported_issue); the row still counts under its state.
            if (
                cell.operational_state in {"blocked", "needs_attention"}
                and cell.operational_reason_code is not None
            ):
                totals[cell.operational_reason_code] += cell.users
        ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
        return [
            OperationalReasonCount(
                reason_code=code, reason=operational_reason(code), users=users
            )
            for code, users in ranked[:page_size]
        ]

    def get_groups(
        self,
        start: date,
        end: date,
        include_internal: bool = False,
        user_group: UserGroup | None = None,
        now: datetime | None = None,
    ) -> GroupAnalyticsResponse:
        """Build the fixed-order source summary for one UTC date window.

        The account row is the source of truth for group membership. Run counts
        and operator cost come from one grouped read of ``user_daily_facts``,
        which carries the group each day was written under. That read and the
        Credits read are independently guarded so a missing projection leaves
        six honest rows with partial availability, rather than dropping the
        whole summary.
        """

        start, end = _validate_dates(start, end)
        current_time = _utc(now or datetime.now(UTC), "now")
        selected_group = (
            parse_user_group(user_group) if user_group is not None else None
        )

        # Keep the row shape stable even if the account projection is
        # temporarily unavailable. The endpoint can then expose an explicit
        # unavailable state without leaking a storage exception.
        users_available = True
        try:
            all_users = self._eligible_users(include_internal=include_internal)
        except Exception:
            all_users = []
            users_available = False

        group_by_user: dict[int, UserGroup] = {}
        user_rows: list[dict[str, Any]] = []
        for user in all_users:
            try:
                user_id = positive_user_id(user["id"])
            except (KeyError, TypeError, ValueError):
                # A malformed legacy account must not crash an admin report.
                users_available = False
                continue
            group_by_user[user_id] = coerce_user_group(user.get("user_group"))
            user_rows.append(user)

        if selected_group is not None:
            user_rows = [
                user
                for user in user_rows
                if group_by_user.get(int(user["id"])) == selected_group
            ]
        user_ids = {int(user["id"]) for user in user_rows}

        users_by_group: Counter[UserGroup] = Counter(
            group_by_user[user_id] for user_id in user_ids
        )
        # Facts exist only through yesterday: a window that starts today has
        # nothing landed, and reading it as zero runs would be a false answer.
        landed = start < current_time.date()
        facts_available = True
        totals: dict[str, GroupFactTotals] = {}
        if landed:
            try:
                totals = self.value_store.sum_facts_by_user_group(start=start, end=end)
            except Exception:
                facts_available = False
        if selected_group is not None:
            totals = {
                group: value for group, value in totals.items() if group == selected_group
            }

        commercial_available = True
        commercial: dict[int, CommercialValueFact] = {}
        try:
            commercial = self._commercial(user_rows, start=start, end=end)
            if user_ids and not user_ids.issubset(commercial):
                commercial_available = False
        except Exception:
            commercial_available = False

        paid_by_group: Counter[UserGroup] = Counter()
        for user_id, fact in commercial.items():
            if user_id not in user_ids:
                continue
            if fact.lifetime_net_purchased_micro > 0:
                paid_by_group[group_by_user[user_id]] += 1

        rows: list[UserGroupSummary] = []
        for group in USER_GROUPS:
            group_totals = totals.get(group, GroupFactTotals())
            rows.append(
                UserGroupSummary(
                    group=group,
                    label=USER_GROUP_LABELS[group],
                    users=int(users_by_group[group]),
                    successful_run_users=group_totals.successful_run_users,
                    repeat_users=group_totals.repeat_users,
                    total_runs=group_totals.total_runs,
                    atl_cost_micro_usd=group_totals.operator_cost_micro,
                    paid_users=int(paid_by_group[group]),
                )
            )

        if not users_available:
            availability = SectionAvailability(available=False, status="unavailable")
        else:
            complete = landed and facts_available and commercial_available
            # Facts stop at yesterday; claiming today as covered would read an
            # unlanded day as zero runs.
            coverage_end = min(end - timedelta(days=1), current_time.date() - timedelta(days=1))
            availability = SectionAvailability(
                available=True,
                status="ready" if complete else "partial",
                coverage_start=start if landed else None,
                coverage_end=coverage_end if landed else None,
            )
        return GroupAnalyticsResponse(
            as_of=current_time,
            groups=rows,
            selected_user_group=selected_group,
            availability=availability,
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
        yesterday = current_time.date() - timedelta(days=1)
        query = UserPageQuery(
            q=filters.q,
            lifecycle_segment=filters.lifecycle_segment,
            operational_state=filters.operational_state,
            commercial_tier=filters.commercial_tier,
            user_group=filters.user_group,
            activated=filters.activated,
            last_meaningful_activity_from=filters.last_meaningful_activity_from,
            last_meaningful_activity_to=filters.last_meaningful_activity_to,
            priority=filters.priority,
            legacy_status=filters.legacy_status,
            include_internal=filters.include_internal,
        )
        rows, total = self.value_store.list_users_page(
            query=query, day=yesterday, limit=page_size, offset=offset
        )
        ids = [row.user_id for row in rows]
        totals = (
            self.value_store.sum_recent_facts(
                ids, start=yesterday - timedelta(days=29), end=yesterday
            )
            if ids
            else {}
        )
        commercial = (
            self.value_store.list_commercial_values(
                ids,
                start=_day_start(yesterday),
                end=_day_start(yesterday + timedelta(days=1)),
            )
            if ids
            else {}
        )
        as_of = _day_start(yesterday + timedelta(days=1))
        items: list[ValueUserListItem] = []
        for row in rows:
            activity = (
                UserActivity(
                    user_id=row.user_id,
                    activated_at=row.activated_at,
                    last_meaningful_activity_at=row.last_meaningful_activity_at,
                    updated_at=current_time,
                )
                if row.activated_at is not None
                or row.last_meaningful_activity_at is not None
                else None
            )
            lifecycle = calculate_lifecycle(
                build_lifecycle_inputs(
                    row.user_id,
                    created_at=row.created_at,
                    activity=activity,
                    totals=totals.get(row.user_id),
                    as_of=current_time,
                ),
                current_time,
            )
            fact = commercial.get(row.user_id)
            items.append(
                ValueUserListItem(
                    user_id=row.user_id,
                    display_name=row.display_name,
                    email=row.email,
                    joined_at=row.created_at,
                    lifecycle=lifecycle,
                    operational=operational_result_from_code(
                        row.operational_reason_code, as_of=as_of
                    ),
                    commercial_tier=row.tier,
                    lifetime_net_purchased_micro=(
                        fact.lifetime_net_purchased_micro if fact is not None else 0
                    ),
                    priority_group=_priority_group(
                        row.operational_state, row.lifecycle_segment
                    ),
                    profile_path=f"/admin/analytics/users/{row.user_id}",
                )
            )
        return PaginatedValueUsers(
            items=items, total=total, limit=page_size, offset=offset
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
        axes = self.legacy_service.get_user_axes(subject_id, now=current_time)
        legacy = self.legacy_service.get_user_profile(
            user_id=subject_id,
            now=current_time,
            start=_day_start(start),
            end=_day_start(end),
            axes=axes,
        )
        commercial = self.value_store.list_commercial_values(
            [subject_id],
            start=_day_start(start),
            end=_day_start(end),
        )[subject_id]
        # This user's own moves from the transition log, each stamped with its
        # own day. Never the anonymous rollups: they are population-wide.
        transitions = [
            LifecycleTransition(
                from_segment=row.from_segment,
                to_segment=row.to_segment,
                users=1,
                period_start=row.snapshot_date,
                period_end=row.snapshot_date + timedelta(days=1),
                data_quality=row.data_quality,
            )
            for row in self.value_store.list_user_transitions(
                subject_id, start=start, end=end
            )
        ]
        lifecycle, operational = axes
        return ValueUserProfile(
            **legacy.model_dump(),
            lifecycle=lifecycle,
            operational=operational,
            commercial=commercial,
            selected_period_start=start,
            selected_period_end=end,
            recent_lifecycle_transitions=transitions,
        )


__all__ = [
    "BalanceTotals",
    "CommercialAnalyticsResponse",
    "CommercialPeriodSummary",
    "GroupAnalyticsResponse",
    "LifecycleAnalyticsResponse",
    "LifecycleHeadline",
    "LifecycleMovementPoint",
    "LifecycleTransition",
    "OperationalAnalyticsResponse",
    "OperationalReasonCount",
    "PaginatedValueUsers",
    "RetentionAnalyticsResponse",
    "RetentionCell",
    "RetentionCohort",
    "SectionAvailability",
    "UserGroupSummary",
    "UserValueFilters",
    "ValueAnalyticsQueryService",
    "ValueUserListItem",
    "ValueUserProfile",
    "WeeklyLifecycleCount",
]
