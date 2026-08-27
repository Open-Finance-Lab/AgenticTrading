"""Read-only, display-safe Admin Analytics query composition."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .metrics import (
    AnalyticsMetricFilters,
    calculate_overview_metrics,
    is_meaningful_event,
)
from .models import AnalyticsEventRecord
from .repository import _row_to_event, analytics_store
from .repository_common import (
    decode_event_cursor,
    encode_event_cursor,
    positive_limit,
    positive_user_id,
    utc_iso,
)
from .rollups import AnalyticsRollupStore, DailyRollup
from .states import (
    AnalyticsStateStore,
    UserAnalyticsSnapshot,
    calculate_user_state,
)


ActivitySection = Literal["timeline", "runs", "usage", "sessions"]
_USER_STATES = {"blocked", "needs_attention", "dormant", "onboarding", "active"}
_ATTENTION_STATES = {"blocked", "needs_attention"}
_ACTIVATION_EVENTS = (
    "account_signed_up",
    "credential_verified",
    "agent_created",
    "backtest_completed",
)
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
    affected_users: int = Field(ge=0)


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


class AnalyticsUserFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    q: str | None = Field(default=None, max_length=100)
    status: str | None = None
    last_activity_from: datetime | None = None
    last_activity_to: datetime | None = None
    sort: Literal[
        "last_activity", "joined_at", "recent_runs", "recent_failures"
    ] = "last_activity"
    order: Literal["asc", "desc"] = "desc"
    include_internal: bool = False

    @field_validator("q")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is not None and value not in _USER_STATES:
            raise ValueError("status must be a supported Analytics user state")
        return value

    @model_validator(mode="after")
    def validate_activity_range(self) -> "AnalyticsUserFilters":
        start = self.last_activity_from
        end = self.last_activity_to
        if start is not None:
            object.__setattr__(self, "last_activity_from", _utc(start, "last_activity_from"))
        if end is not None:
            object.__setattr__(self, "last_activity_to", _utc(end, "last_activity_to"))
        if start is not None and end is not None and _utc(end, "last_activity_to") < _utc(start, "last_activity_from"):
            raise ValueError("last_activity_to cannot be before last_activity_from")
        return self


class PaginatedUsers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[AnalyticsUserListItem]
    total: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


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
    activation_funnel: dict[str, int]
    user_state_counts: dict[str, int]
    top_failure_categories: list[FailureCategoryCount]
    users_needing_attention: list[AnalyticsUserListItem]
    last_updated: datetime
    filters: AnalyticsMetricFilters
    availability: dict[str, PanelAvailability]


def _all_users(user_store: Any) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = user_store.list_users_admin(limit=500, offset=offset)
        users.extend(page)
        if len(page) < 500:
            return users
        offset += len(page)


class AnalyticsQueryStore:
    """PR 2 read companions over the PR 1 Analytics tables."""

    def __init__(self, base_store: Any = analytics_store):
        self.base_store = base_store
        self.rollups = AnalyticsRollupStore(base_store)
        self.states = AnalyticsStateStore(base_store)
        self.is_postgres = hasattr(base_store, "database_url")

    def list_snapshots(self) -> dict[int, UserAnalyticsSnapshot]:
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM user_analytics_snapshots ORDER BY user_id")
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM user_analytics_snapshots ORDER BY user_id"
                ).fetchall()
        result = {}
        for row in rows:
            snapshot = UserAnalyticsSnapshot(
                user_id=int(row["user_id"]),
                status=str(row["status"]),
                reason_code=str(row["reason_code"]),
                human_readable_reason=str(row["human_readable_reason"]),
                evidence_event_ids=list(
                    json.loads(str(row["evidence_event_ids_json"]))
                ),
                calculated_at=_parse_timestamp(row["calculated_at"]),
            )
            result[snapshot.user_id] = snapshot
        return result

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

    def summarize_users(
        self,
        *,
        user_ids: set[int],
        start: datetime,
        end: datetime,
    ) -> dict[int, dict[str, Any]]:
        """Return bounded 30-day summaries without hydrating event models."""

        if not user_ids:
            return {}
        result: dict[int, dict[str, Any]] = {}
        ordered_ids = sorted(positive_user_id(user_id) for user_id in user_ids)
        for offset in range(0, len(ordered_ids), 500):
            chunk = ordered_ids[offset : offset + 500]
            placeholder = "%s" if self.is_postgres else "?"
            id_placeholders = ", ".join(placeholder for _ in chunk)
            params: list[Any] = [utc_iso(start), utc_iso(end), *chunk]
            sql = f"""
                SELECT user_id,
                       MAX(CASE
                           WHEN event_group <> 'experience'
                                OR event_name = 'page_viewed'
                           THEN occurred_at
                       END) AS last_meaningful_activity,
                       SUM(CASE WHEN event_group = 'run' THEN 1 ELSE 0 END)
                           AS recent_runs,
                       SUM(CASE WHEN event_name = 'backtest_failed' THEN 1 ELSE 0 END)
                           AS recent_failures
                FROM analytics_events
                WHERE occurred_at >= {placeholder}
                  AND occurred_at < {placeholder}
                  AND user_id IN ({id_placeholders})
                GROUP BY user_id
            """
            if self.is_postgres:
                with self.base_store._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
            else:
                with self.base_store._get_connection() as conn:
                    rows = conn.execute(sql, params).fetchall()
            for row in rows:
                result[int(row["user_id"])] = {
                    "last_meaningful_activity": (
                        _parse_timestamp(row["last_meaningful_activity"])
                        if row["last_meaningful_activity"] is not None
                        else None
                    ),
                    "recent_runs": int(row["recent_runs"] or 0),
                    "recent_failures": int(row["recent_failures"] or 0),
                }
        return result

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

    def list_session_rows(
        self,
        *,
        user_id: int,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        subject_id = positive_user_id(user_id)
        page_size = positive_limit(limit)
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM analytics_events
                        WHERE user_id = %s AND event_group = 'experience'
                          AND session_id IS NOT NULL
                        ORDER BY occurred_at DESC, sequence DESC
                        """,
                        (subject_id,),
                    )
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM analytics_events
                    WHERE user_id = ? AND event_group = 'experience'
                      AND session_id IS NOT NULL
                    ORDER BY occurred_at DESC, sequence DESC
                    """,
                    (subject_id,),
                ).fetchall()
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


def _snapshot_summary(snapshot: UserAnalyticsSnapshot) -> AnalyticsStateSummary:
    return AnalyticsStateSummary(
        status=snapshot.status,
        reason_code=snapshot.reason_code,
        human_readable_reason=snapshot.human_readable_reason,
        evidence_event_ids=snapshot.evidence_event_ids,
        calculated_at=snapshot.calculated_at,
    )


def _availability(available: bool = True) -> PanelAvailability:
    return PanelAvailability(available=available)


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


class AnalyticsQueryService:
    def __init__(
        self,
        *,
        store: Any = analytics_store,
        user_store: Any | None = None,
    ):
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        self.store = store
        self.user_store = user_store
        self.query_store = AnalyticsQueryStore(store)

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
        funnel: dict[str, int] = {}
        failures: list[FailureCategoryCount] = []
        raw_events: list[_MetricEvent] = []

        if effective_end > filters.start:
            try:
                raw_events = self.query_store.list_metric_events(
                    start=filters.start,
                    end=effective_end,
                    include_internal=filters.include_internal,
                )
                raw_metrics = calculate_overview_metrics(
                    raw_events,
                    start=filters.start,
                    end=effective_end,
                    billing_mode=filters.billing_mode,
                    provider_id=filters.provider_id,
                    model_id=filters.model_id,
                )
                active_users = raw_metrics.active_users_7d
                conversion = raw_metrics.first_success_conversion
                repeat_rate = raw_metrics.repeat_run_rate
                funnel = {
                    event_name: len(
                        {
                            event.user_id
                            for event in raw_events
                            if event.event_name == event_name
                            and _event_matches_filters(event, filters)
                        }
                    )
                    for event_name in _ACTIVATION_EVENTS
                }
                failure_users: dict[str, set[int]] = defaultdict(set)
                for event in raw_events:
                    if event.error_category and _event_matches_filters(event, filters):
                        failure_users[event.error_category].add(event.user_id)
                failures = [
                    FailureCategoryCount(
                        error_category=category,
                        affected_users=len(users),
                    )
                    for category, users in sorted(
                        failure_users.items(),
                        key=lambda item: (-len(item[1]), item[0]),
                    )[:10]
                ]
            except Exception:
                availability["snapshot"] = _availability(False)
                availability["funnel"] = _availability(False)
                availability["friction"] = _availability(False)

        try:
            historical_end = min(effective_end.date(), current.date())
            rollups = self.query_store.rollups.list_rollups(
                start=filters.start.date(),
                end=historical_end,
            )
            dimensional = any(
                value is not None
                for value in (
                    filters.billing_mode,
                    filters.provider_id,
                    filters.model_id,
                )
            )
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
            today_start = datetime.combine(
                current.date(), datetime.min.time(), tzinfo=timezone.utc
            )
            current_events = [
                event
                for event in raw_events
                if max(filters.start, today_start)
                <= event.occurred_at
                < effective_end
                and _event_matches_filters(event, filters)
            ]
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

            if filters.billing_mode == "byok":
                platform_micro = 0
            elif filters.provider_id is not None or filters.model_id is not None:
                platform_micro = sum(
                    row.value_sum_micro
                    for row in rollups
                    if row.metric_name == "platform_model_cost_usd"
                    and row.billing_mode == "platform_credits"
                    and bool(row.provider_id)
                    and bool(row.model_id)
                    and (
                        filters.provider_id is None
                        or row.provider_id == filters.provider_id
                    )
                    and (
                        filters.model_id is None or row.model_id == filters.model_id
                    )
                )
            else:
                platform_micro = sum(
                    row.value_sum_micro
                    for row in rollups
                    if row.metric_name == "platform_model_cost_usd"
                    and row.billing_mode == "platform_credits"
                    and not row.provider_id
                    and not row.model_id
                )
            platform_micro += sum(
                int(event.properties.get("cost_micro_usd", 0))
                for event in current_events
                if event.event_name == "model_usage_recorded"
                and event.billing_mode == "platform_credits"
            )
            platform_cost = platform_micro / 1_000_000
            input_tokens = sum(
                int(event.properties.get("input_tokens", 0))
                for event in raw_events
                if event.event_name == "model_usage_recorded"
                and _event_matches_filters(event, filters)
            )
            output_tokens = sum(
                int(event.properties.get("output_tokens", 0))
                for event in raw_events
                if event.event_name == "model_usage_recorded"
                and _event_matches_filters(event, filters)
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
        except Exception:
            availability["growth"] = _availability(False)

        state_counts: dict[str, int] = {}
        attention: list[AnalyticsUserListItem] = []
        snapshots: dict[int, UserAnalyticsSnapshot] = {}
        excluded: set[int] = set()
        snapshots_available = False
        try:
            snapshots = self.query_store.list_snapshots()
            excluded = (
                set()
                if filters.include_internal
                else self.store.list_excluded_user_ids(include_admin_accounts=True)
            )
            state_counts = dict(
                Counter(
                    snapshot.status
                    for user_id, snapshot in snapshots.items()
                    if user_id not in excluded
                )
            )
            snapshots_available = True
        except Exception:
            availability["friction"] = _availability(False)
        try:
            if not snapshots_available:
                raise RuntimeError("Analytics snapshots are unavailable")
            attention_ids = {
                user_id
                for user_id, snapshot in snapshots.items()
                if user_id not in excluded
                and snapshot.status in _ATTENTION_STATES
            }
            summaries = self.query_store.summarize_users(
                user_ids=attention_ids,
                start=current - timedelta(days=30),
                end=current + timedelta(microseconds=1),
            )
            identities = {
                int(user["id"]): user
                for user in _all_users(self.user_store)
                if int(user["id"]) in attention_ids
            }
            candidates = []
            for user_id in attention_ids:
                user = identities.get(user_id)
                if user is None:
                    continue
                snapshot = snapshots[user_id]
                summary = summaries.get(user_id, {})
                candidates.append(
                    AnalyticsUserListItem(
                        user_id=user_id,
                        display_name=str(user.get("display_name") or ""),
                        email=str(user.get("email") or ""),
                        joined_at=_parse_timestamp(user["created_at"]),
                        status=snapshot.status,
                        reason_code=snapshot.reason_code,
                        human_readable_reason=snapshot.human_readable_reason,
                        last_meaningful_activity=summary.get(
                            "last_meaningful_activity"
                        ),
                        recent_runs=int(summary.get("recent_runs", 0)),
                        recent_failures=int(summary.get("recent_failures", 0)),
                        profile_path=f"/admin/analytics/users/{user_id}",
                    )
                )
            candidates.sort(
                key=lambda item: (item.recent_failures, item.user_id),
                reverse=True,
            )
            attention = candidates[:10]
        except Exception:
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
            activation_funnel=funnel,
            user_state_counts=state_counts,
            top_failure_categories=failures,
            users_needing_attention=attention,
            last_updated=current,
            filters=filters,
            availability=availability,
        )

    def list_users(
        self,
        *,
        filters: AnalyticsUserFilters,
        limit: int,
        offset: int,
        now: datetime | None = None,
    ) -> PaginatedUsers:
        if not isinstance(filters, AnalyticsUserFilters):
            filters = AnalyticsUserFilters.model_validate(filters)
        page_size = positive_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        current = _utc(now or datetime.now(timezone.utc), "now")
        users = _all_users(self.user_store)
        excluded = (
            set()
            if filters.include_internal
            else self.store.list_excluded_user_ids(include_admin_accounts=True)
        )
        snapshots = self.query_store.list_snapshots()
        events = self.query_store.rollups.list_events(
            start=current - timedelta(days=30),
            end=current + timedelta(microseconds=1),
            include_internal=True,
        )
        by_user: dict[int, list[AnalyticsEventRecord]] = defaultdict(list)
        for event in events:
            by_user[event.user_id].append(event)

        items: list[AnalyticsUserListItem] = []
        for user in users:
            user_id = int(user["id"])
            if user_id in excluded:
                continue
            user_events = sorted(
                by_user.get(user_id, []),
                key=lambda event: event.occurred_at,
                reverse=True,
            )
            meaningful = [event for event in user_events if is_meaningful_event(event)]
            last_activity = meaningful[0].occurred_at if meaningful else None
            snapshot = snapshots.get(user_id)
            if snapshot is None:
                snapshot = calculate_user_state(
                    user_id,
                    now=current,
                    store=self.query_store.states,
                )
            item = AnalyticsUserListItem(
                user_id=user_id,
                display_name=str(user.get("display_name") or ""),
                email=str(user.get("email") or ""),
                joined_at=_parse_timestamp(user["created_at"]),
                status=snapshot.status,
                reason_code=snapshot.reason_code,
                human_readable_reason=snapshot.human_readable_reason,
                last_meaningful_activity=last_activity,
                recent_runs=sum(event.event_group == "run" for event in user_events),
                recent_failures=sum(
                    event.event_name == "backtest_failed" for event in user_events
                ),
                profile_path=f"/admin/analytics/users/{user_id}",
            )
            if filters.q is not None:
                needle = filters.q.lower()
                if needle not in item.email.lower() and needle not in item.display_name.lower():
                    continue
            if filters.status is not None and item.status != filters.status:
                continue
            if (
                filters.last_activity_from is not None
                and (
                    item.last_meaningful_activity is None
                    or item.last_meaningful_activity < filters.last_activity_from
                )
            ):
                continue
            if (
                filters.last_activity_to is not None
                and (
                    item.last_meaningful_activity is None
                    or item.last_meaningful_activity > filters.last_activity_to
                )
            ):
                continue
            items.append(item)

        def sort_value(item: AnalyticsUserListItem):
            if filters.sort == "joined_at":
                return item.joined_at
            if filters.sort == "recent_runs":
                return item.recent_runs
            if filters.sort == "recent_failures":
                return item.recent_failures
            return item.last_meaningful_activity or datetime.min.replace(
                tzinfo=timezone.utc
            )

        items.sort(
            key=lambda item: (sort_value(item), item.user_id),
            reverse=filters.order == "desc",
        )
        return PaginatedUsers(
            items=items[offset : offset + page_size],
            total=len(items),
            limit=page_size,
            offset=offset,
        )

    def get_user_profile(
        self,
        *,
        user_id: int,
        now: datetime | None = None,
    ) -> AnalyticsUserProfile:
        subject_id = positive_user_id(user_id)
        current = _utc(now or datetime.now(timezone.utc), "now")
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        snapshot = self.query_store.states.get_snapshot(subject_id)
        if snapshot is None:
            snapshot = calculate_user_state(
                subject_id,
                now=current,
                store=self.query_store.states,
            )
        events = self.query_store.rollups.list_events(
            start=current - timedelta(days=180),
            end=current + timedelta(microseconds=1),
            include_internal=True,
            user_id=subject_id,
        )
        ordered = sorted(events, key=lambda event: event.occurred_at, reverse=True)
        meaningful = [event for event in ordered if is_meaningful_event(event)]
        milestones: dict[str, datetime] = {}
        for event_name in _ACTIVATION_EVENTS:
            matches = [
                event.occurred_at for event in events if event.event_name == event_name
            ]
            if matches:
                milestones[event_name] = min(matches)
        lane_counts = Counter(
            event.billing_mode
            for event in events
            if event.event_name == "model_usage_recorded" and event.billing_mode
        )
        page_counts = Counter(
            event.page_view
            for event in events
            if event.event_name == "page_viewed" and event.page_view
        )
        provider_event = next(
            (
                event
                for event in ordered
                if event.provider_id
                and event.event_name
                in {"credential_defaulted", "credential_verified", "model_usage_recorded"}
            ),
            None,
        )
        client_event = next(
            (
                event
                for event in ordered
                if event.country_code or event.device_category or event.browser_family
            ),
            None,
        )
        recent = [
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
        ]
        return AnalyticsUserProfile(
            user_id=subject_id,
            display_name=str(user.get("display_name") or ""),
            email=str(user.get("email") or ""),
            joined_at=_parse_timestamp(user["created_at"]),
            last_meaningful_activity=(meaningful[0].occurred_at if meaningful else None),
            state=_snapshot_summary(snapshot),
            primary_billing_lane=(lane_counts.most_common(1)[0][0] if lane_counts else None),
            default_provider=(provider_event.provider_id if provider_event else None),
            country_code=(client_event.country_code if client_event else None),
            device_category=(client_event.device_category if client_event else None),
            browser_family=(client_event.browser_family if client_event else None),
            activation_milestones=milestones,
            recent_footprint=recent,
            run_summary={
                "requested": sum(event.event_name == "backtest_requested" for event in events),
                "completed": sum(event.event_name == "backtest_completed" for event in events),
                "failed": sum(event.event_name == "backtest_failed" for event in events),
                "cancelled": sum(event.event_name == "backtest_cancelled" for event in events),
            },
            billing_lane_mix=dict(lane_counts),
            input_tokens=sum(
                int(event.properties.get("input_tokens", 0))
                for event in events
                if event.event_name == "model_usage_recorded"
            ),
            output_tokens=sum(
                int(event.properties.get("output_tokens", 0))
                for event in events
                if event.event_name == "model_usage_recorded"
            ),
            platform_model_cost_usd=sum(
                int(event.properties.get("cost_micro_usd", 0))
                for event in events
                if event.event_name == "model_usage_recorded"
                and event.billing_mode == "platform_credits"
            )
            / 1_000_000,
            credits_debited_micro=sum(
                int(event.properties.get("amount_micro", 0))
                for event in events
                if event.event_name == "credits_settled"
            ),
            top_product_page=(page_counts.most_common(1)[0][0] if page_counts else None),
        )

    def get_user_activity(
        self,
        *,
        user_id: int,
        section: ActivitySection,
        limit: int,
        cursor: str | None,
    ) -> AnalyticsActivityPage:
        if section not in {"timeline", "runs", "usage", "sessions"}:
            raise ValueError("section must be a supported Analytics activity section")
        page_size = positive_limit(limit)
        if section == "sessions":
            rows, next_cursor = self.query_store.list_session_rows(
                user_id=user_id,
                limit=page_size,
                cursor=cursor,
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
                    cost_micro_usd=(
                        int(properties.get("cost_micro_usd", 0))
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


def get_analytics_query_service() -> AnalyticsQueryService:
    return analytics_query_service


__all__ = [
    "ActivitySection",
    "AnalyticsActivityItem",
    "AnalyticsActivityPage",
    "AnalyticsFootprintItem",
    "AnalyticsOverview",
    "AnalyticsQueryService",
    "AnalyticsStateSummary",
    "AnalyticsUserFilters",
    "AnalyticsUserListItem",
    "AnalyticsUserProfile",
    "FailureCategoryCount",
    "PaginatedUsers",
    "PanelAvailability",
    "get_analytics_query_service",
]
