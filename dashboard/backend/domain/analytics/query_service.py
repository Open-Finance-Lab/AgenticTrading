"""Read-only, display-safe Admin Analytics query composition."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .lifecycle import (
    CommercialTier,
    LifecycleResult,
    OperationalResult,
    OperationalSignals,
    calculate_lifecycle,
    calculate_operational_state,
    operational_result_from_code,
    resolve_group_badge,
)
from .lifecycle_reads import build_lifecycle_inputs
from .metrics import (
    ACTIVATION_EVENTS,
    AnalyticsMetricFilters,
    is_meaningful_event,
)
from .models import AnalyticsEventRecord
from .repository import _row_to_event, analytics_store
from .repository_common import (
    AnalyticsStoreError,
    decode_event_cursor,
    encode_event_cursor,
    positive_limit,
    positive_user_id,
    utc_iso,
)
from .rollups import AnalyticsRollupStore, DailyRollup
from .usage_cost import atl_cost_micro, byok_estimate_micro, byok_rollup_lane
from .value_repository import build_value_analytics_store

ActivitySection = Literal["timeline", "runs", "usage", "sessions"]
_ATTENTION_STATES = {"blocked", "needs_attention"}
# The two BillingLaneDay call-count fields; also models.ALLOWED_BILLING_MODES.
_BILLING_LANES = ("platform_credits", "byok")
# Every BillingLaneDay counter: the two call counts, each lane's Credits (the
# platform debit, the BYOK list-price estimate) and the BYOK calls left unpriced.
_LANE_FIELDS = (
    *_BILLING_LANES,
    "platform_cost_micro",
    "byok_estimated_micro",
    "byok_unpriced",
)
_ACTIVATION_EVENTS = ACTIVATION_EVENTS
_USAGE_EVENTS = {
    "model_usage_recorded",
    "credits_reserved",
    "credits_settled",
    "credits_refunded",
}


@dataclass(frozen=True)
class _MetricEvent:
    event_name: str
    event_group: str
    user_id: int
    occurred_at: datetime
    provider_id: str | None
    model_id: str | None
    billing_mode: str | None
    error_category: str | None
    properties: dict[str, Any]


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class PanelAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool = True
    error_code: Literal["temporarily_unavailable"] | None = None

    @model_validator(mode="after")
    def match_error_to_availability(self) -> "PanelAvailability":
        expected = None if self.available else "temporarily_unavailable"
        object.__setattr__(self, "error_code", expected)
        return self


class FailureCategoryCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error_category: str
    event_count: int = Field(ge=0)


class AnalyticsStateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str
    reason_code: str
    human_readable_reason: str
    evidence_event_ids: list[str] = Field(default_factory=list)
    calculated_at: datetime


class AnalyticsUserListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    joined_at: datetime
    status: str
    reason_code: str
    human_readable_reason: str
    last_meaningful_activity: datetime | None = None
    recent_runs: int = Field(default=0, ge=0)
    recent_failures: int = Field(default=0, ge=0)
    profile_path: str






class AnalyticsFootprintItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    event_name: str
    occurred_at: datetime
    page_view: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    billing_mode: str | None = None
    outcome: str | None = None
    error_category: str | None = None


class AnalyticsUserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    joined_at: datetime
    last_meaningful_activity: datetime | None = None
    state: AnalyticsStateSummary
    primary_billing_lane: str | None = None
    default_provider: str | None = None
    country_code: str | None = None
    device_category: str | None = None
    browser_family: str | None = None
    activation_milestones: dict[str, datetime]
    recent_footprint: list[AnalyticsFootprintItem]
    run_summary: dict[str, int]
    billing_lane_mix: dict[str, int]
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    platform_model_cost_usd: float = Field(default=0, ge=0)
    credits_debited_micro: int = Field(default=0, ge=0)
    top_product_page: str | None = None


class AnalyticsActivityItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_type: Literal["event", "run", "usage", "session"]
    event_id: str | None = None
    event_name: str
    occurred_at: datetime
    outcome: str | None = None
    error_category: str | None = None
    page_view: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    billing_mode: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_micro_usd: int | None = Field(default=None, ge=0)
    amount_micro: int | None = Field(default=None, ge=0)
    bucket: str | None = None
    session_event_count: int | None = Field(default=None, ge=0)
    visible_ms: int | None = Field(default=None, ge=0)
    country_code: str | None = None
    device_category: str | None = None
    browser_family: str | None = None


class AnalyticsActivityPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[AnalyticsActivityItem]
    next_cursor: str | None = None


class BillingLaneDay(BaseModel):
    """One UTC day of model calls split by billing lane, for the Credits panel.

    Both lanes count ``model_usage_recorded`` events -- one per model call --
    so the two bars share a unit. ``credits_settled`` is not a run or a call
    counter: it fires once per non-zero credit bucket, and not at all for a
    zero-cost settlement.

    The two ``*_micro`` fields are in micro-Credits ($1 = 1 Credit), off those
    same events (analytics.usage_cost): ``platform_cost_micro`` is what the
    platform lane debited, ``byok_estimated_micro`` what the priced BYOK calls
    would have debited at list price. ``byok_unpriced`` counts the BYOK calls
    that have no estimate -- model not on the price list, no provider usage
    reported, or recorded before estimates existed -- so the estimate's gap is
    a number rather than an inference from a zero.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: str
    platform_credits: int = Field(ge=0)
    byok: int = Field(ge=0)
    platform_cost_micro: int = Field(default=0, ge=0)
    byok_estimated_micro: int = Field(default=0, ge=0)
    byok_unpriced: int = Field(default=0, ge=0)


class AnalyticsOverview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_users_7d: int | None = Field(default=None, ge=0)
    first_success_conversion: float | None = Field(default=None, ge=0, le=1)
    backtest_success_rate: float | None = Field(default=None, ge=0, le=1)
    repeat_run_rate: float | None = Field(default=None, ge=0, le=1)
    platform_model_cost_usd: float | None = Field(default=None, ge=0)
    completed_runs: int | None = Field(default=None, ge=0)
    failed_runs: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    daily_active_users: dict[str, int]
    daily_completed_runs: dict[str, int]
    billing_lane_mix: list[BillingLaneDay]
    activation_funnel: dict[str, int]
    top_failure_categories: list[FailureCategoryCount]
    users_needing_attention: list[AnalyticsUserListItem]
    last_updated: datetime
    filters: AnalyticsMetricFilters
    availability: dict[str, PanelAvailability]


class RunOutcomeCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    cancelled: int = Field(default=0, ge=0)


class UserMetricsShared(BaseModel):
    """One user's numbers as the user may see them (design §6.13).

    ``get_user_metrics`` builds this for ``audience="user"`` and
    ``UserMetricsAdmin`` for ``audience="admin"``. Two models rather than one
    with nullable admin fields: the admin-only fields are *absent* from the
    user projection, not nulled -- a null still tells the subject the field
    exists -- and absence must hold on every serialisation path, including
    ``model_dump_json`` and FastAPI's, which a ``model_dump`` override would
    not reach. A future ``GET /api/me/usage`` is this call with
    ``audience="user"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    facts_as_of: date
    runs_7d: RunOutcomeCounts
    runs_30d: RunOutcomeCounts
    own_spend_micro: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    tier: CommercialTier = "unpaid"
    credit_balance_micro: int = Field(default=0, ge=0)
    data_quality: Literal["complete", "partial"] = "complete"


class UserMetricsAdmin(UserMetricsShared):
    """The admin projection: the shared numbers plus the admin-only axes."""

    active_days_30d: int = Field(ge=0, le=30)
    lifecycle: LifecycleResult
    operational: OperationalResult
    user_group: str
    group_badge: str
    access_log: Sequence[str]


class AnalyticsQueryStore:
    """PR 2 read companions over the PR 1 Analytics tables."""

    def __init__(self, base_store: Any = analytics_store):
        self.base_store = base_store
        self.rollups = AnalyticsRollupStore(base_store)
        self.is_postgres = hasattr(base_store, "database_url")

    def list_metric_events(
        self,
        *,
        start: datetime,
        end: datetime,
        include_internal: bool,
    ) -> list[_MetricEvent]:
        """Load only fields used by Overview formulas.

        Full Analytics event validation is intentionally reserved for detail
        responses. Overview can touch tens of thousands of rows, and rebuilding
        UUID/session/source models for fields it never returns dominated the
        page latency without adding a privacy or correctness check.
        """

        params = (utc_iso(start), utc_iso(end))
        sql = """
            SELECT event_name, event_group, user_id, occurred_at,
                   provider_id, model_id, billing_mode, error_category,
                   properties_json
            FROM analytics_events
            WHERE occurred_at >= {start_placeholder}
              AND occurred_at < {end_placeholder}
            ORDER BY occurred_at ASC, sequence ASC
        """
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.format(
                            start_placeholder="%s",
                            end_placeholder="%s",
                        ),
                        params,
                    )
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    sql.format(
                        start_placeholder="?",
                        end_placeholder="?",
                    ),
                    params,
                ).fetchall()
        excluded = (
            set()
            if include_internal
            else self.base_store.list_excluded_user_ids(
                include_admin_accounts=True
            )
        )
        events = []
        for row in rows:
            user_id = int(row["user_id"])
            if user_id in excluded:
                continue
            event_name = str(row["event_name"])
            properties = (
                json.loads(str(row["properties_json"]))
                if event_name == "model_usage_recorded"
                else {}
            )
            if not isinstance(properties, dict):
                raise ValueError("stored Analytics properties are invalid")
            events.append(
                _MetricEvent(
                    event_name=event_name,
                    event_group=str(row["event_group"]),
                    user_id=user_id,
                    occurred_at=_parse_timestamp(row["occurred_at"]),
                    provider_id=(
                        str(row["provider_id"])
                        if row["provider_id"] is not None
                        else None
                    ),
                    model_id=(
                        str(row["model_id"])
                        if row["model_id"] is not None
                        else None
                    ),
                    billing_mode=(
                        str(row["billing_mode"])
                        if row["billing_mode"] is not None
                        else None
                    ),
                    error_category=(
                        str(row["error_category"])
                        if row["error_category"] is not None
                        else None
                    ),
                    properties=properties,
                )
            )
        return events

    def list_activity_rows(
        self,
        *,
        user_id: int,
        section: ActivitySection,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[tuple[int, AnalyticsEventRecord]], str | None]:
        subject_id = positive_user_id(user_id)
        page_size = positive_limit(limit)
        where = {
            "timeline": "event_name NOT IN ('session_heartbeat', 'page_hidden')",
            "runs": "event_group = 'run'",
            "usage": (
                "event_name IN ('model_usage_recorded', 'credits_reserved', "
                "'credits_settled', 'credits_refunded')"
            ),
        }.get(section)
        if where is None:
            raise ValueError("section must be timeline, runs, or usage")
        cursor_values = decode_event_cursor(cursor) if cursor is not None else None
        params: list[Any] = [subject_id]
        cursor_sql = ""
        if cursor_values is not None:
            occurred_at, sequence = cursor_values
            if self.is_postgres:
                cursor_sql = "AND (occurred_at, sequence) < (%s, %s)"
                params.extend([occurred_at, sequence])
            else:
                cursor_sql = (
                    "AND (occurred_at < ? OR (occurred_at = ? AND sequence < ?))"
                )
                params.extend([occurred_at, occurred_at, sequence])
        params.append(page_size + 1)
        placeholder = "%s" if self.is_postgres else "?"
        sql = f"""
            SELECT * FROM analytics_events
            WHERE user_id = {placeholder} AND {where} {cursor_sql}
            ORDER BY occurred_at DESC, sequence DESC
            LIMIT {placeholder}
        """
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        has_more = len(rows) > page_size
        page_rows = rows[:page_size]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_event_cursor(
                str(last["occurred_at"]),
                int(last["sequence"]),
            )
        return [
            (int(row["sequence"]), _row_to_event(row)) for row in page_rows
        ], next_cursor

    def first_event_times(
        self,
        *,
        user_id: int,
        event_names: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, datetime]:
        """Each named event's first ``occurred_at`` for one user in ``[start, end)``.

        One grouped, indexed read returning at most ``len(event_names)``
        rows: the profile's activation milestones are first occurrences,
        which the newest-100 timeline page cannot answer for a user whose
        first verification is older than their last hundred events.
        """
        subject_id = positive_user_id(user_id)
        names = sorted(set(event_names))
        if not names:
            return {}
        placeholder = "%s" if self.is_postgres else "?"
        sql = f"""
            SELECT event_name, MIN(occurred_at) AS first_at
            FROM analytics_events
            WHERE user_id = {placeholder}
              AND event_name IN ({", ".join(placeholder for _ in names)})
              AND occurred_at >= {placeholder} AND occurred_at < {placeholder}
            GROUP BY event_name
        """
        params = [
            subject_id,
            *names,
            utc_iso(_utc(start, "start")),
            utc_iso(_utc(end, "end")),
        ]
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        return {
            str(row["event_name"]): _parse_timestamp(row["first_at"])
            for row in rows
            if row["first_at"] is not None
        }

    def list_session_rows(
        self,
        *,
        user_id: int,
        limit: int,
        cursor: str | None,
        start: datetime,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Sessions with any experience event at or after ``start``, whole.

        The window selects *sessions*, not events: a session that began
        before ``start`` and continued after it is read with all of its
        events, so its count and visible time are not cut at the boundary.
        Sessions wholly before ``start`` are out of scope (design: the
        activity route's 30-day sessions window).
        """
        subject_id = positive_user_id(user_id)
        page_size = positive_limit(limit)
        window_start = utc_iso(_utc(start, "start"))
        placeholder = "%s" if self.is_postgres else "?"
        sql = f"""
            SELECT * FROM analytics_events
            WHERE user_id = {placeholder}
              AND event_group = 'experience'
              AND session_id IN (
                  SELECT session_id FROM analytics_events
                  WHERE user_id = {placeholder} AND occurred_at >= {placeholder}
                    AND event_group = 'experience'
                    AND session_id IS NOT NULL
              )
            ORDER BY occurred_at DESC, sequence DESC
        """
        params = (subject_id, subject_id, window_start)
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            event = _row_to_event(row)
            session_id = event.session_id
            if session_id is None:
                continue
            group = grouped.setdefault(
                session_id,
                {
                    "occurred_at": event.occurred_at,
                    "sequence": int(row["sequence"]),
                    "event_count": 0,
                    "visible_ms": 0,
                    "country_code": event.country_code,
                    "device_category": event.device_category,
                    "browser_family": event.browser_family,
                },
            )
            group["event_count"] += 1
            group["visible_ms"] += int(event.properties.get("visible_ms", 0))
        sessions = sorted(
            grouped.values(),
            key=lambda row: (row["occurred_at"], row["sequence"]),
            reverse=True,
        )
        if cursor is not None:
            cursor_time, cursor_sequence = decode_event_cursor(cursor)
            cursor_at = _parse_timestamp(cursor_time)
            sessions = [
                row
                for row in sessions
                if (row["occurred_at"], row["sequence"])
                < (cursor_at, cursor_sequence)
            ]
        page_rows = sessions[: page_size + 1]
        has_more = len(page_rows) > page_size
        page_rows = page_rows[:page_size]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_event_cursor(
                last["occurred_at"].isoformat(),
                last["sequence"],
            )
        return page_rows, next_cursor


def _availability(available: bool = True) -> PanelAvailability:
    return PanelAvailability(available=available)


def _warn_read_failed(section: str, exc: BaseException) -> None:
    """One log line for a read the page degrades to unavailable/partial.

    The page already says "incomplete"; this is what lets an operator tell a
    Postgres-only SQL bug from an outage in the Render log. Class name only:
    an exception message can carry row data.
    """
    print(
        f"WARNING: analytics.read.{section}_failed "
        f"category={type(exc).__name__[:80]}",
        flush=True,
    )


def _legacy_state(operational_state: str, lifecycle_segment: str) -> str:
    """Map the live operational state and lifecycle segment onto the legacy
    five-state label (blocked, needs_attention, dormant, onboarding, active)
    that the user profile's ``state`` summary still reports. Operational
    problems win; otherwise the lifecycle segment decides."""

    if operational_state in {"blocked", "needs_attention"}:
        return operational_state
    if lifecycle_segment == "dormant":
        return "dormant"
    if lifecycle_segment in {"new", "onboarding"}:
        return "onboarding"
    return "active"


def _is_total_row(row: DailyRollup) -> bool:
    """A rollup's undimensioned day total, not one of its per-dimension rows."""

    return not (
        row.event_name
        or row.billing_mode
        or row.provider_id
        or row.model_id
        or row.outcome
        or row.error_category
        or row.user_state
    )


def _latest_rollup_value(rows: Sequence[DailyRollup], metric_name: str) -> int | None:
    """The metric's total on the latest rolled-up day: "as of yesterday"."""

    latest: DailyRollup | None = None
    for row in rows:
        if row.metric_name == metric_name and _is_total_row(row):
            if latest is None or row.rollup_date > latest.rollup_date:
                latest = row
    return None if latest is None else latest.value_count


def _sum_rollup(rows: Sequence[DailyRollup], metric_name: str) -> int:
    return sum(
        row.value_count
        for row in rows
        if row.metric_name == metric_name and _is_total_row(row)
    )


def _rollup_token_total(
    rows: Sequence[DailyRollup],
    metric_name: str,
    filters: AnalyticsMetricFilters,
    *,
    dimensional: bool,
) -> int:
    """A token metric over completed days: the day totals, or -- when a
    billing/provider/model filter is set -- the per-usage-dimension rows."""

    if not dimensional:
        return _sum_rollup(rows, metric_name)
    return sum(
        row.value_count
        for row in rows
        if row.metric_name == metric_name
        and (row.billing_mode or row.provider_id or row.model_id)
        and _matches_rollup_dimensions(row, filters)
    )


def _matches_rollup_dimensions(
    row: DailyRollup,
    filters: AnalyticsMetricFilters,
) -> bool:
    if filters.billing_mode is not None and row.billing_mode != filters.billing_mode:
        return False
    if filters.provider_id is not None and row.provider_id != filters.provider_id:
        return False
    if filters.model_id is not None and row.model_id != filters.model_id:
        return False
    return True


def _counts_toward_platform_cost(
    row: DailyRollup,
    filters: AnalyticsMetricFilters,
) -> bool:
    """Which ``platform_model_cost_usd`` rollup rows make up the platform cost.

    A day carries one undimensioned total plus one row per provider/model, so
    exactly one of the two sets is summed: the per-model rows when a provider
    or model filter is set, the total otherwise. Shared by the headline and
    the Credits panel's per-day lane so the two always agree.
    """
    if row.metric_name != "platform_model_cost_usd":
        return False
    if row.billing_mode != "platform_credits" or filters.billing_mode == "byok":
        return False
    if filters.provider_id is not None or filters.model_id is not None:
        return (
            bool(row.provider_id)
            and bool(row.model_id)
            and (filters.provider_id is None or row.provider_id == filters.provider_id)
            and (filters.model_id is None or row.model_id == filters.model_id)
        )
    return not row.provider_id and not row.model_id


def _platform_cost_micro(
    rollups: list[DailyRollup],
    current_events: list[_MetricEvent],
    filters: AnalyticsMetricFilters,
) -> int:
    """The platform model cost of a window: completed days from the rollups,
    today from its raw events (already filtered). The one computation behind
    the overview headline and ``get_platform_model_cost_micro``."""
    return sum(
        row.value_sum_micro
        for row in rollups
        if _counts_toward_platform_cost(row, filters)
    ) + sum(
        atl_cost_micro(event.billing_mode, event.properties)
        for event in current_events
        if event.event_name == "model_usage_recorded"
    )


def _run_outcome_counts(totals: Any) -> RunOutcomeCounts:
    if totals is None:
        return RunOutcomeCounts()
    return RunOutcomeCounts(
        requested=totals.runs_requested,
        completed=totals.runs_completed,
        failed=totals.runs_failed,
        cancelled=totals.runs_cancelled,
    )


def landed_fact_day(
    value_store: Any,
    *,
    yesterday: date,
    warned: set[tuple[date, date | None]],
) -> date | None:
    """The fact day a read "as of yesterday" should use, or None if none landed.

    Facts land once a night. Until they do -- or all day, when the job's
    facts step fails -- yesterday has no rows, and a read of it returns empty
    counts indistinguishable from "nobody is blocked". Every such read
    joins the newest landed day instead and labels it (``stale`` plus the
    coverage dates; ``/users`` stamps ``as_of``), or, where its response has
    no field to label it with, refuses a day older than the normal gap.

    The WARNING is printed once per (expected, latest) pair for the calling
    service, not once per request: a failed job would otherwise write it on
    every page load, filter change and pagination click all day.
    """
    latest = value_store.latest_fact_date(on_or_before=yesterday)
    if latest != yesterday and (yesterday, latest) not in warned:
        warned.add((yesterday, latest))
        print(
            "WARNING: analytics.read.facts_stale "
            f"latest={latest.isoformat() if latest else 'none'} "
            f"expected={yesterday.isoformat()}",
            flush=True,
        )
    return latest


class AnalyticsQueryService:
    def __init__(
        self,
        *,
        store: Any = analytics_store,
        user_store: Any | None = None,
        value_store: Any | None = None,
    ):
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        self.store = store
        self.user_store = user_store
        self.query_store = AnalyticsQueryStore(store)
        self._value_store = value_store
        self._stale_fact_warnings: set[tuple[date, date | None]] = set()

    @property
    def value_store(self) -> Any:
        """The fact-table store, built on first use.

        Not in ``__init__``: the module-level ``analytics_query_service`` is
        constructed at import, and building the default twin resolves the
        credits / provider / agent / run singletons -- on the web boot path
        and in every backtest child that imports this module.
        """
        if self._value_store is None:
            self._value_store = build_value_analytics_store(analytics_base=self.store)
        return self._value_store

    @value_store.setter
    def value_store(self, value: Any) -> None:
        self._value_store = value

    def get_overview(
        self,
        *,
        filters: AnalyticsMetricFilters,
        now: datetime | None = None,
    ) -> AnalyticsOverview:
        if not isinstance(filters, AnalyticsMetricFilters):
            filters = AnalyticsMetricFilters.model_validate(filters)
        current = _utc(now or datetime.now(timezone.utc), "now")
        effective_end = min(filters.end, current)
        availability = {
            "snapshot": _availability(),
            "growth": _availability(),
            "funnel": _availability(),
            "friction": _availability(),
            "attention": _availability(),
        }
        active_users: int | None = None
        conversion: float | None = None
        success_rate: float | None = None
        repeat_rate: float | None = None
        completed: int | None = None
        failed: int | None = None
        platform_cost: float | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        daily_active: dict[str, int] = {}
        daily_completed: dict[str, int] = {}
        lane_days: dict[str, dict[str, int]] = {}
        funnel: dict[str, int] = {}
        failures: list[FailureCategoryCount] = []
        dimensional = any(
            value is not None
            for value in (
                filters.billing_mode,
                filters.provider_id,
                filters.model_id,
            )
        )
        # The one raw-event read: the current UTC day, only the part of it the
        # window covers (design §6.15 item 2). Completed days come from the
        # rollups, which already hold the trailing context the old
        # window-wide scan existed to supply.
        #
        # Always without internal accounts: ``rollup_day`` writes the
        # completed days internal-free, so admitting them into today's slice
        # under ``include_internal`` summed two populations into one figure
        # that matched neither setting. The flag is accepted and echoed, and
        # widens nothing here (as on /retention, /groups and /lifecycle).
        today_start = datetime.combine(
            current.date(), datetime.min.time(), tzinfo=timezone.utc
        )
        scan_start = max(filters.start, today_start)
        today_events: list[_MetricEvent] = []
        # An empty slice has nothing to read, which is not a failed read.
        raw_events_read = effective_end <= scan_start
        if not raw_events_read:
            try:
                today_events = self.query_store.list_metric_events(
                    start=scan_start,
                    end=effective_end,
                    include_internal=False,
                )
                raw_events_read = True
            except Exception as exc:
                _warn_read_failed("overview_events", exc)
                today_events = []
        current_events = [
            event for event in today_events if _event_matches_filters(event, filters)
        ]
        rollups: list[DailyRollup] = []
        rollups_read = False
        try:
            historical_end = min(effective_end.date(), current.date())
            rollups = self.query_store.rollups.list_rollups(
                start=filters.start.date(),
                end=historical_end,
            )
            rollups_read = True
        except Exception as exc:
            _warn_read_failed("overview_rollups", exc)
            rollups = []

        # Trailing context, as of the latest completed day in the window. The
        # rollups hold these as day totals only, so a billing/provider/model
        # filter does not narrow them.
        if rollups_read:
            active_users = _latest_rollup_value(rollups, "rolling_active_users_7d")
            mature = _sum_rollup(rollups, "mature_signup_cohort_users")
            converted = _sum_rollup(rollups, "first_success_within_7d_users")
            conversion = None if mature == 0 else converted / mature
            first_success = _latest_rollup_value(rollups, "users_with_first_success")
            repeat = _latest_rollup_value(rollups, "users_with_repeat_success_24h_30d")
            if first_success:
                repeat_rate = (repeat or 0) / first_success
        else:
            availability["snapshot"] = _availability(False)

        if rollups_read and raw_events_read:
            # Completed days sum their distinct users per day, so a user who
            # repeats a step on two days counts twice (the daily tier's trade,
            # design §6.3). activation_users rows carry no billing/provider/
            # model, so today's half is unfiltered too.
            funnel = {
                event_name: sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "activation_users"
                    and row.event_name == event_name
                )
                + len(
                    {
                        event.user_id
                        for event in today_events
                        if event.event_name == event_name
                    }
                )
                for event_name in _ACTIVATION_EVENTS
            }
            # Design D13: events per category, filtered or not. A distinct-user
            # count over the range would need a raw read or a rollup keyed by
            # user, and summing per-day distinct users counts user-days, so
            # the one figure the rollups give exactly is the event count
            # (#647).
            failure_counts: dict[str, int] = defaultdict(int)
            for row in rollups:
                if (
                    row.error_category
                    and row.metric_name == "event_count"
                    and _matches_rollup_dimensions(row, filters)
                ):
                    failure_counts[row.error_category] += row.value_count
            for event in current_events:
                if event.error_category:
                    failure_counts[event.error_category] += 1
            failures = [
                FailureCategoryCount(error_category=category, event_count=count)
                for category, count in sorted(
                    failure_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
            ]
            # A completed day rolled up before activation_users existed has
            # every other rollup row and none of these, and summed as zero.
            # rollup_day now writes them zero or not and the startup
            # migration fills the old days inside raw retention; a day still
            # without them makes the funnel an undercount, which it must not
            # publish as a figure.
            rolled_up_days = {
                row.rollup_date for row in rollups if row.metric_name == "daily_active_users"
            }
            activation_days = {
                row.rollup_date for row in rollups if row.metric_name == "activation_users"
            }
            if not rolled_up_days <= activation_days:
                funnel = {}
                availability["funnel"] = _availability(False)
        else:
            availability["funnel"] = _availability(False)
            availability["friction"] = _availability(False)

        try:
            if not raw_events_read:
                # Today's lanes, today's platform cost and the token totals are
                # all read off the raw events. Without them this block would
                # publish a rollup-only figure as the whole range -- the quiet
                # answer a failed read must never be mistaken for.
                raise AnalyticsStoreError("raw analytics events are unavailable")
            if not rollups_read:
                raise AnalyticsStoreError("analytics rollups are unavailable")
            if dimensional:
                historical_completed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "event_count"
                    and row.event_name == "backtest_completed"
                    and _matches_rollup_dimensions(row, filters)
                )
                historical_failed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "event_count"
                    and row.event_name == "backtest_failed"
                    and _matches_rollup_dimensions(row, filters)
                )
            else:
                historical_completed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "terminal_completed"
                    and not row.event_name
                    and not row.provider_id
                    and not row.model_id
                )
                historical_failed = sum(
                    row.value_count
                    for row in rollups
                    if row.metric_name == "terminal_failed"
                    and not row.event_name
                    and not row.provider_id
                    and not row.model_id
                )
            current_completed = sum(
                event.event_name == "backtest_completed" for event in current_events
            )
            current_failed = sum(
                event.event_name == "backtest_failed" for event in current_events
            )
            completed = historical_completed + current_completed
            failed = historical_failed + current_failed
            denominator = completed + failed
            success_rate = None if denominator == 0 else completed / denominator

            platform_cost = (
                _platform_cost_micro(rollups, current_events, filters) / 1_000_000
            )
            input_tokens, output_tokens = (
                _rollup_token_total(
                    rollups, metric_name, filters, dimensional=dimensional
                )
                + sum(
                    int(event.properties.get(metric_name, 0))
                    for event in current_events
                    if event.event_name == "model_usage_recorded"
                )
                for metric_name in ("input_tokens", "output_tokens")
            )
            for row in rollups:
                key = row.rollup_date.isoformat()
                if row.metric_name == "daily_active_users" and not dimensional:
                    daily_active[key] = row.value_count
                if row.metric_name in {"completed_runs", "terminal_completed"} and not dimensional:
                    daily_completed[key] = row.value_count
                if (
                    dimensional
                    and row.metric_name == "event_count"
                    and row.event_name == "backtest_completed"
                    and _matches_rollup_dimensions(row, filters)
                ):
                    daily_completed[key] = daily_completed.get(key, 0) + row.value_count
            current_day_events = [
                event for event in current_events if is_meaningful_event(event)
            ]
            if current_events:
                day_key = current.date().isoformat()
                daily_active[day_key] = len(
                    {event.user_id for event in current_day_events}
                )
                daily_completed[day_key] = current_completed
            # Billing-lane mix (design D14): completed days from the rollups'
            # billing_mode dimension, today from the same filtered raw events
            # as the rest of this block. Built locally and published only on
            # success, so a failure part-way never ships a half-counted series.
            # Each lane also carries its cost in Credits, from the same
            # model_usage_recorded events: the platform debit
            # (platform_model_cost_usd, the headline's own rows) and the BYOK
            # list-price estimate (byok_estimated_cost_usd), plus the BYOK
            # calls with no estimate (byok_unpriced_calls). Never
            # credits_settled, which has no provider or model to filter on.
            lanes: dict[str, dict[str, int]] = defaultdict(
                lambda: dict.fromkeys(_LANE_FIELDS, 0)
            )
            # BYOK is gathered per provider/model before it is folded into a
            # day: whether a legacy rollup's estimate can stand depends on the
            # model, so deciding it per day made the unfiltered total and a
            # per-model filter disagree about the same calls.
            byok_rollups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
                lambda: {"calls": 0, "estimate": 0, "unpriced": 0, "counted": 0}
            )
            for row in rollups:
                day_key = row.rollup_date.isoformat()
                byok_key = (day_key, row.provider_id or "", row.model_id or "")
                if row.metric_name == "event_count":
                    if (
                        row.event_name == "model_usage_recorded"
                        and row.billing_mode in _BILLING_LANES
                        and _matches_rollup_dimensions(row, filters)
                    ):
                        lanes[day_key][row.billing_mode] += row.value_count
                        if row.billing_mode == "byok":
                            byok_rollups[byok_key]["calls"] += row.value_count
                elif _counts_toward_platform_cost(row, filters):
                    lanes[day_key]["platform_cost_micro"] += row.value_sum_micro
                elif row.billing_mode == "byok" and _matches_rollup_dimensions(
                    row, filters
                ):
                    if row.metric_name == "byok_estimated_cost_usd":
                        byok_rollups[byok_key]["estimate"] += row.value_sum_micro
                    elif row.metric_name == "byok_unpriced_calls":
                        byok_rollups[byok_key]["unpriced"] += row.value_count
                        byok_rollups[byok_key]["counted"] = 1
            for (day_key, _provider, model_id), totals in byok_rollups.items():
                estimate, unpriced = byok_rollup_lane(
                    model_id,
                    calls=totals["calls"],
                    estimate_micro=totals["estimate"],
                    unpriced_calls=(
                        totals["unpriced"] if totals["counted"] else None
                    ),
                )
                lanes[day_key]["byok_estimated_micro"] += estimate
                lanes[day_key]["byok_unpriced"] += unpriced
            for event in current_events:
                if (
                    event.event_name != "model_usage_recorded"
                    or event.billing_mode not in _BILLING_LANES
                ):
                    continue
                lane = lanes[event.occurred_at.date().isoformat()]
                lane[event.billing_mode] += 1
                if event.billing_mode == "byok":
                    estimate = byok_estimate_micro(event.properties, event.model_id)
                    if estimate is None:
                        lane["byok_unpriced"] += 1
                    else:
                        lane["byok_estimated_micro"] += estimate
                else:
                    lane["platform_cost_micro"] += atl_cost_micro(
                        event.billing_mode, event.properties
                    )
            # Every rolled-up day carries a platform cost total, zero or not;
            # a day with neither calls nor cost is not activity, and keeping
            # it would hide the panel's "no activity" state behind flat lines.
            lane_days = {
                day: lane for day, lane in lanes.items() if any(lane.values())
            }
        except Exception as exc:
            _warn_read_failed("overview_growth", exc)
            availability["growth"] = _availability(False)

        # The friction panel (top_failure_categories) is computed from the
        # funnel's reads, so it is unavailable exactly when they failed (set
        # above) -- not when only the funnel's activation rows are missing.

        # Who needs attention, as of the latest landed fact day (design D20):
        # yesterday once the night's job lands, the day before until then.
        # The daily job writes facts for non-internal accounts only, so
        # include_internal does not widen it.
        yesterday = current.date() - timedelta(days=1)
        attention: list[AnalyticsUserListItem] = []
        try:
            facts_day = landed_fact_day(
                self.value_store,
                yesterday=yesterday,
                warned=self._stale_fact_warnings,
            )
            if facts_day is None or facts_day < yesterday - timedelta(days=1):
                # No fact day has landed, or only one older than the normal
                # "tonight's job has not run yet" gap. An empty list would read
                # as "nobody needs attention", and this panel has no as-of
                # field to label an old list with, so it says unavailable.
                raise AnalyticsStoreError("no recent analytics fact day has landed")
            candidates = self.value_store.list_attention_candidates(
                day=facts_day,
                window_start=facts_day - timedelta(days=29),
                limit=10,
            )
            activity = (
                self.value_store.list_activity(
                    [candidate.user_id for candidate in candidates]
                )
                if candidates
                else {}
            )
            for candidate in candidates:
                reason = operational_result_from_code(
                    candidate.operational_reason_code, as_of=current
                )
                clock = activity.get(candidate.user_id)
                attention.append(
                    AnalyticsUserListItem(
                        user_id=candidate.user_id,
                        display_name=candidate.display_name,
                        email=candidate.email,
                        joined_at=candidate.created_at,
                        status=candidate.operational_state,
                        reason_code=reason.reason_code,
                        human_readable_reason=reason.reason,
                        last_meaningful_activity=(
                            clock.last_meaningful_activity_at if clock else None
                        ),
                        recent_runs=candidate.recent_runs,
                        recent_failures=candidate.recent_failures,
                        profile_path=f"/admin/analytics/users/{candidate.user_id}",
                    )
                )
        except Exception as exc:
            _warn_read_failed("overview_attention", exc)
            attention = []
            availability["attention"] = _availability(False)

        return AnalyticsOverview(
            active_users_7d=active_users,
            first_success_conversion=conversion,
            backtest_success_rate=success_rate,
            repeat_run_rate=repeat_rate,
            platform_model_cost_usd=platform_cost,
            completed_runs=completed,
            failed_runs=failed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            daily_active_users=daily_active,
            daily_completed_runs=daily_completed,
            billing_lane_mix=[
                BillingLaneDay(
                    day=day,
                    platform_credits=counts["platform_credits"],
                    byok=counts["byok"],
                    platform_cost_micro=counts["platform_cost_micro"],
                    byok_estimated_micro=counts["byok_estimated_micro"],
                    byok_unpriced=counts["byok_unpriced"],
                )
                for day, counts in sorted(lane_days.items())
            ],
            activation_funnel=funnel,
            top_failure_categories=failures,
            users_needing_attention=attention,
            last_updated=current,
            filters=filters,
            availability=availability,
        )

    def get_platform_model_cost_micro(
        self,
        *,
        filters: AnalyticsMetricFilters,
        now: datetime | None = None,
    ) -> int | None:
        """``get_overview``'s platform-cost headline alone, in micro-Credits.

        Reads what the headline reads -- the rollups for completed days, the
        raw events for today only -- and none of the snapshots, users or
        range-wide events the rest of the overview needs. None when either read
        fails: 0 is a real answer ("no platform calls"), not an unknown one.
        """
        if not isinstance(filters, AnalyticsMetricFilters):
            filters = AnalyticsMetricFilters.model_validate(filters)
        current = _utc(now or datetime.now(timezone.utc), "now")
        effective_end = min(filters.end, current)
        if effective_end <= filters.start:
            return 0
        today_start = datetime.combine(
            current.date(), datetime.min.time(), tzinfo=timezone.utc
        )
        current_start = max(filters.start, today_start)
        try:
            rollups = self.query_store.rollups.list_rollups(
                start=filters.start.date(),
                end=min(effective_end.date(), current.date()),
            )
            current_events = (
                [
                    event
                    # Internal-free, like the rollups it is added to -- the
                    # overview's own figure reads the same way (see there).
                    for event in self.query_store.list_metric_events(
                        start=current_start,
                        end=effective_end,
                        include_internal=False,
                    )
                    if _event_matches_filters(event, filters)
                ]
                if effective_end > current_start
                else []
            )
            return _platform_cost_micro(rollups, current_events, filters)
        except Exception as exc:
            print(
                "[analytics] ERROR platform model cost unavailable: "
                f"{type(exc).__name__}"
            )
            return None

    def get_user_axes(
        self, user_id: int, *, now: datetime
    ) -> tuple[LifecycleResult, OperationalResult]:
        """Live segment and live operational state for exactly one user.

        The segment's inputs are ``user_activity`` plus the trailing 30 fact
        days through yesterday, evaluated ``as_of=now``; today's runs reach
        the segment tomorrow (the accepted one-day lag, design §6.3).
        """
        subject_id = positive_user_id(user_id)
        current = _utc(now, "now")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        yesterday = current.date() - timedelta(days=1)
        activity = self.value_store.get_activity(subject_id)
        totals = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=29), end=yesterday
        ).get(subject_id)
        lifecycle = calculate_lifecycle(
            build_lifecycle_inputs(
                subject_id,
                created_at=_parse_timestamp(user["created_at"]),
                activity=activity,
                totals=totals,
                as_of=current,
            ),
            current,
        )
        facts = self.value_store.get_operational_facts(subject_id, now=current)
        operational = calculate_operational_state(
            OperationalSignals(**facts.model_dump()), current
        )
        return lifecycle, operational

    def get_user_profile(
        self,
        *,
        user_id: int,
        now: datetime | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        axes: tuple[LifecycleResult, OperationalResult] | None = None,
    ) -> AnalyticsUserProfile:
        """One user's profile: live axes, fact-window sums, one timeline page.

        The run, cost and spend figures sum ``user_daily_facts`` over the
        selected window clipped to yesterday ("as of yesterday", design
        §6.3). The raw-event scan is the first timeline page (design §6.12),
        and every field computed from it -- lane mix, provider, client, top
        page, tokens -- is bounded by that page, not by the window. The two
        event milestones are first occurrences, so they come from one grouped
        MIN() read over the selected range instead.
        """
        subject_id = positive_user_id(user_id)
        current = _utc(now or datetime.now(timezone.utc), "now")
        range_start = (
            _utc(start, "start") if start is not None else current - timedelta(days=180)
        )
        range_end = (
            _utc(end, "end") if end is not None else current + timedelta(microseconds=1)
        )
        if range_end <= range_start:
            raise ValueError("end must be after start")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        lifecycle, operational = axes or self.get_user_axes(subject_id, now=current)
        yesterday = current.date() - timedelta(days=1)
        window_start = range_start.date()
        window_end = min((range_end - timedelta(microseconds=1)).date(), yesterday)
        totals = (
            self.value_store.sum_recent_facts(
                [subject_id], start=window_start, end=window_end
            ).get(subject_id)
            if window_end >= window_start
            else None
        )
        activity = self.value_store.get_activity(subject_id)

        # The profile's one raw-event read: the first timeline page (design §6.12).
        rows, _next_cursor = self.query_store.list_activity_rows(
            user_id=subject_id, section="timeline", limit=100, cursor=None
        )
        events = [event for _sequence, event in rows]  # newest first
        meaningful = [event for event in events if is_meaningful_event(event)]
        joined_at = _parse_timestamp(user["created_at"])
        milestones: dict[str, datetime] = {"account_signed_up": joined_at}
        if activity is not None and activity.activated_at is not None:
            milestones["backtest_completed"] = activity.activated_at
        # First occurrences in the selected range, as main computed them --
        # never min() over the timeline page, which holds only the newest
        # 100 events and so loses (or misdates) an older first verification.
        milestones.update(
            self.query_store.first_event_times(
                user_id=subject_id,
                event_names=("credential_verified", "agent_created"),
                start=range_start,
                end=range_end,
            )
        )
        usage = [event for event in events if event.event_name == "model_usage_recorded"]
        lane_counts = Counter(event.billing_mode for event in usage if event.billing_mode)
        page_counts = Counter(
            event.page_view
            for event in events
            if event.event_name == "page_viewed" and event.page_view
        )
        provider_event = next(
            (
                event
                for event in events
                if event.provider_id
                and event.event_name
                in {"credential_defaulted", "credential_verified", "model_usage_recorded"}
            ),
            None,
        )
        client_event = next(
            (
                event
                for event in events
                if event.country_code or event.device_category or event.browser_family
            ),
            None,
        )
        state_status = _legacy_state(operational.state, lifecycle.segment)
        state_reason = (
            operational if state_status in _ATTENTION_STATES else lifecycle
        )
        return AnalyticsUserProfile(
            user_id=subject_id,
            display_name=str(user.get("display_name") or ""),
            email=str(user.get("email") or ""),
            joined_at=joined_at,
            last_meaningful_activity=(
                activity.last_meaningful_activity_at if activity is not None else None
            ),
            state=AnalyticsStateSummary(
                status=state_status,
                reason_code=state_reason.reason_code,
                human_readable_reason=state_reason.reason,
                # The five-state evidence list had one source,
                # calculate_user_state, and the profile no longer calls it.
                evidence_event_ids=[],
                calculated_at=current,
            ),
            primary_billing_lane=(lane_counts.most_common(1)[0][0] if lane_counts else None),
            default_provider=(provider_event.provider_id if provider_event else None),
            country_code=(client_event.country_code if client_event else None),
            device_category=(client_event.device_category if client_event else None),
            browser_family=(client_event.browser_family if client_event else None),
            activation_milestones=milestones,
            recent_footprint=[
                AnalyticsFootprintItem(
                    event_id=event.event_id,
                    event_name=event.event_name,
                    occurred_at=event.occurred_at,
                    page_view=event.page_view,
                    provider_id=event.provider_id,
                    model_id=event.model_id,
                    billing_mode=event.billing_mode,
                    outcome=event.outcome,
                    error_category=event.error_category,
                )
                for event in meaningful[:20]
            ],
            run_summary={
                "requested": totals.runs_requested if totals else 0,
                "completed": totals.runs_completed if totals else 0,
                "failed": totals.runs_failed if totals else 0,
                "cancelled": totals.runs_cancelled if totals else 0,
            },
            billing_lane_mix=dict(lane_counts),
            input_tokens=sum(
                int(event.properties.get("input_tokens", 0)) for event in usage
            ),
            output_tokens=sum(
                int(event.properties.get("output_tokens", 0)) for event in usage
            ),
            platform_model_cost_usd=(totals.operator_cost_micro if totals else 0)
            / 1_000_000,
            credits_debited_micro=totals.own_spend_micro if totals else 0,
            top_product_page=(page_counts.most_common(1)[0][0] if page_counts else None),
        )

    def get_user_metrics(
        self,
        user_id: int,
        *,
        audience: Literal["admin", "user"],
        now: datetime | None = None,
    ) -> UserMetricsShared:
        """Build one user's metric struct for one audience (design §6.13).

        ``audience="user"`` returns ``UserMetricsShared`` and never reads the
        admin-only sources (the axes, the access log); ``audience="admin"``
        returns ``UserMetricsAdmin``. The shared numbers come from the same
        reads either way, so the two projections cannot disagree on them.
        Nothing calls this yet and it is on no response model.
        """
        if audience not in {"admin", "user"}:
            raise ValueError("audience must be admin or user")
        subject_id = positive_user_id(user_id)
        current = _utc(now or datetime.now(timezone.utc), "now")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        yesterday = current.date() - timedelta(days=1)
        week = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=6), end=yesterday
        ).get(subject_id)
        month = self.value_store.sum_recent_facts(
            [subject_id], start=yesterday - timedelta(days=29), end=yesterday
        ).get(subject_id)
        yesterday_start = datetime.combine(yesterday, datetime.min.time(), tzinfo=timezone.utc)
        commercial = self.value_store.list_commercial_values(
            [subject_id],
            start=yesterday_start,
            end=yesterday_start + timedelta(days=1),
        ).get(subject_id)
        tier: CommercialTier = (
            commercial.commercial_tier if commercial is not None else "unpaid"
        )
        shared = dict(
            user_id=subject_id,
            facts_as_of=yesterday,
            runs_7d=_run_outcome_counts(week),
            runs_30d=_run_outcome_counts(month),
            own_spend_micro=month.own_spend_micro if month else 0,
            operator_cost_micro=month.operator_cost_micro if month else 0,
            tier=tier,
            credit_balance_micro=(
                commercial.total_available_micro if commercial is not None else 0
            ),
            data_quality=(
                "complete" if month is not None and month.days_present >= 30 else "partial"
            ),
        )
        if audience == "user":
            return UserMetricsShared(**shared)
        lifecycle, operational = self.get_user_axes(subject_id, now=current)
        user_group = str(user.get("user_group") or "unknown")
        return UserMetricsAdmin(
            **shared,
            active_days_30d=min(30, month.active_days) if month else 0,
            lifecycle=lifecycle,
            operational=operational,
            user_group=user_group,
            group_badge=resolve_group_badge(
                role=str(user.get("role") or "user"), user_group=user_group, tier=tier
            ),
            access_log=[
                str(row["section"])
                for row in self.store.list_admin_access(subject_id, limit=10)
            ],
        )

    def get_user_activity(
        self,
        *,
        user_id: int,
        section: ActivitySection,
        limit: int,
        cursor: str | None,
        now: datetime | None = None,
    ) -> AnalyticsActivityPage:
        if section not in {"timeline", "runs", "usage", "sessions"}:
            raise ValueError("section must be a supported Analytics activity section")
        page_size = positive_limit(limit)
        if section == "sessions":
            current = _utc(now or datetime.now(timezone.utc), "now")
            rows, next_cursor = self.query_store.list_session_rows(
                user_id=user_id,
                limit=page_size,
                cursor=cursor,
                start=current - timedelta(days=30),
            )
            return AnalyticsActivityPage(
                items=[
                    AnalyticsActivityItem(
                        item_type="session",
                        event_name="session",
                        occurred_at=row["occurred_at"],
                        session_event_count=row["event_count"],
                        visible_ms=row["visible_ms"],
                        country_code=row["country_code"],
                        device_category=row["device_category"],
                        browser_family=row["browser_family"],
                    )
                    for row in rows
                ],
                next_cursor=next_cursor,
            )
        rows, next_cursor = self.query_store.list_activity_rows(
            user_id=user_id,
            section=section,
            limit=page_size,
            cursor=cursor,
        )
        items = []
        for _sequence, event in rows:
            properties = event.properties
            items.append(
                AnalyticsActivityItem(
                    item_type=(
                        "run" if section == "runs" else "usage" if section == "usage" else "event"
                    ),
                    event_id=event.event_id,
                    event_name=event.event_name,
                    occurred_at=event.occurred_at,
                    outcome=event.outcome,
                    error_category=event.error_category,
                    page_view=event.page_view,
                    provider_id=event.provider_id,
                    model_id=event.model_id,
                    billing_mode=event.billing_mode,
                    input_tokens=(
                        int(properties.get("input_tokens", 0))
                        if event.event_name == "model_usage_recorded"
                        else None
                    ),
                    output_tokens=(
                        int(properties.get("output_tokens", 0))
                        if event.event_name == "model_usage_recorded"
                        else None
                    ),
                    # ATL cost only: a BYOK call's list-price estimate is not
                    # spend, and must not read as spend to a reader that does
                    # not know to check billing_mode first.
                    cost_micro_usd=(
                        atl_cost_micro(event.billing_mode, properties)
                        if event.event_name == "model_usage_recorded"
                        else None
                    ),
                    amount_micro=(
                        int(properties.get("amount_micro", 0))
                        if event.event_name in _USAGE_EVENTS
                        and event.event_name != "model_usage_recorded"
                        else None
                    ),
                    bucket=(
                        str(properties.get("bucket"))
                        if event.event_name in _USAGE_EVENTS
                        and event.event_name != "model_usage_recorded"
                        else None
                    ),
                )
            )
        return AnalyticsActivityPage(items=items, next_cursor=next_cursor)


def _event_matches_filters(
    event: AnalyticsEventRecord | _MetricEvent,
    filters: AnalyticsMetricFilters,
) -> bool:
    if filters.billing_mode is not None and event.billing_mode != filters.billing_mode:
        return False
    if filters.provider_id is not None and event.provider_id != filters.provider_id:
        return False
    if filters.model_id is not None and event.model_id != filters.model_id:
        return False
    return True


analytics_query_service = AnalyticsQueryService()
_value_analytics_query_service: ValueAnalyticsQueryService | None = None


def get_analytics_query_service() -> AnalyticsQueryService:
    return analytics_query_service


def get_value_analytics_query_service() -> ValueAnalyticsQueryService:
    """Lazily construct the value service without creating an import cycle."""

    global _value_analytics_query_service
    if _value_analytics_query_service is None:
        from .value_queries import ValueAnalyticsQueryService

        _value_analytics_query_service = ValueAnalyticsQueryService()
    return _value_analytics_query_service


__all__ = [
    "ActivitySection",
    "AnalyticsActivityItem",
    "AnalyticsActivityPage",
    "AnalyticsFootprintItem",
    "AnalyticsOverview",
    "AnalyticsQueryService",
    "AnalyticsStateSummary",
    "AnalyticsUserListItem",
    "AnalyticsUserProfile",
    "FailureCategoryCount",
    "PanelAvailability",
    "RunOutcomeCounts",
    "UserMetricsAdmin",
    "UserMetricsShared",
    "get_analytics_query_service",
    "get_value_analytics_query_service",
]
