"""Companion repository and deterministic daily Analytics rollups.

PR 1 owns the tables.  PR 2 owns these bounded aggregate operations so the
Foundation repository contract remains unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .metrics import calculate_overview_metrics
from .models import AnalyticsEventRecord
from .repository import _row_to_event, analytics_store
from .repository_common import utc_iso


class DailyRollup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollup_date: date
    metric_name: str = Field(min_length=1, max_length=64)
    event_name: str = ""
    billing_mode: str = ""
    provider_id: str = ""
    model_id: str = ""
    outcome: str = ""
    error_category: str = ""
    user_state: str = ""
    value_count: int = Field(default=0, ge=0)
    value_sum_micro: int = 0
    updated_at: datetime


def _day_start(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


class AnalyticsRollupStore:
    """SQLite/PostgreSQL twin operations over PR 1 aggregate tables."""

    def __init__(self, base_store=analytics_store):
        self.base_store = base_store
        self.is_postgres = hasattr(base_store, "database_url")

    def list_events(
        self,
        *,
        start: datetime,
        end: datetime,
        include_internal: bool = False,
        user_id: int | None = None,
    ) -> list[AnalyticsEventRecord]:
        params: list[Any] = [utc_iso(start), utc_iso(end)]
        user_sql = ""
        if user_id is not None:
            user_sql = "AND user_id = " + ("%s" if self.is_postgres else "?")
            params.append(int(user_id))
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT * FROM analytics_events
                        WHERE occurred_at >= %s AND occurred_at < %s {user_sql}
                        ORDER BY occurred_at ASC, sequence ASC
                        """,
                        params,
                    )
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM analytics_events
                    WHERE occurred_at >= ? AND occurred_at < ? {user_sql}
                    ORDER BY occurred_at ASC, sequence ASC
                    """,
                    params,
                ).fetchall()
        events = [_row_to_event(row) for row in rows]
        if include_internal:
            return events
        excluded = self.base_store.list_excluded_user_ids(
            include_admin_accounts=True
        )
        return [event for event in events if event.user_id not in excluded]

    def replace_day(self, day: date, rows: Iterable[DailyRollup]) -> None:
        values = list(rows)
        columns = (
            "rollup_date",
            "metric_name",
            "event_name",
            "billing_mode",
            "provider_id",
            "model_id",
            "outcome",
            "error_category",
            "user_state",
            "value_count",
            "value_sum_micro",
            "updated_at",
        )
        payloads = [
            (
                row.rollup_date.isoformat(),
                row.metric_name,
                row.event_name,
                row.billing_mode,
                row.provider_id,
                row.model_id,
                row.outcome,
                row.error_category,
                row.user_state,
                row.value_count,
                row.value_sum_micro,
                utc_iso(row.updated_at),
            )
            for row in values
        ]
        if self.is_postgres:
            placeholders = ", ".join("%s" for _ in columns)
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM analytics_daily_rollups WHERE rollup_date = %s",
                        (day.isoformat(),),
                    )
                    if payloads:
                        cur.executemany(
                            f"INSERT INTO analytics_daily_rollups ({', '.join(columns)}) "
                            f"VALUES ({placeholders})",
                            payloads,
                        )
        else:
            placeholders = ", ".join("?" for _ in columns)
            with self.base_store._get_connection() as conn:
                conn.execute(
                    "DELETE FROM analytics_daily_rollups WHERE rollup_date = ?",
                    (day.isoformat(),),
                )
                if payloads:
                    conn.executemany(
                        f"INSERT INTO analytics_daily_rollups ({', '.join(columns)}) "
                        f"VALUES ({placeholders})",
                        payloads,
                    )

    def list_rollups(self, *, start: date, end: date) -> list[DailyRollup]:
        params = (start.isoformat(), end.isoformat())
        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT * FROM analytics_daily_rollups
                        WHERE rollup_date >= %s AND rollup_date < %s
                        ORDER BY rollup_date, metric_name, event_name,
                                 billing_mode, provider_id, model_id,
                                 outcome, error_category, user_state
                        """,
                        params,
                    )
                    rows = cur.fetchall()
        else:
            with self.base_store._get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM analytics_daily_rollups
                    WHERE rollup_date >= ? AND rollup_date < ?
                    ORDER BY rollup_date, metric_name, event_name,
                             billing_mode, provider_id, model_id,
                             outcome, error_category, user_state
                    """,
                    params,
                ).fetchall()
        return [
            DailyRollup(
                rollup_date=date.fromisoformat(str(row["rollup_date"])),
                metric_name=str(row["metric_name"]),
                event_name=str(row["event_name"]),
                billing_mode=str(row["billing_mode"]),
                provider_id=str(row["provider_id"]),
                model_id=str(row["model_id"]),
                outcome=str(row["outcome"]),
                error_category=str(row["error_category"]),
                user_state=str(row["user_state"]),
                value_count=int(row["value_count"]),
                value_sum_micro=int(row["value_sum_micro"]),
                updated_at=datetime.fromisoformat(str(row["updated_at"])),
            )
            for row in rows
        ]


def _row(
    day: date,
    metric_name: str,
    *,
    count: int = 0,
    sum_micro: int = 0,
    updated_at: datetime,
    **dimensions: str,
) -> DailyRollup:
    return DailyRollup(
        rollup_date=day,
        metric_name=metric_name,
        value_count=count,
        value_sum_micro=sum_micro,
        updated_at=updated_at,
        **dimensions,
    )


def rollup_day(
    day: date,
    *,
    store: AnalyticsRollupStore | None = None,
    state_counts: Mapping[str, int] | None = None,
    include_internal: bool = False,
    now: datetime | None = None,
) -> list[DailyRollup]:
    aggregate_store = store or AnalyticsRollupStore()
    start = _day_start(day)
    end = start + timedelta(days=1)
    # Completed-day rebuilds are deterministic: rerunning the same source rows
    # yields byte-equivalent aggregate values, including the stored timestamp.
    current = now or end
    events = aggregate_store.list_events(
        start=start - timedelta(days=30),
        end=end,
        include_internal=include_internal,
    )
    day_events = [event for event in events if start <= event.occurred_at < end]
    overview = calculate_overview_metrics(
        events,
        start=start,
        end=end,
    )
    cohort_start = start - timedelta(days=7)
    signups = {
        event.user_id: event.occurred_at
        for event in events
        if event.event_name == "account_signed_up"
        and cohort_start <= event.occurred_at < cohort_start + timedelta(days=1)
    }
    successes_by_user: dict[int, list[datetime]] = defaultdict(list)
    for event in events:
        if event.event_name == "backtest_completed":
            successes_by_user[event.user_id].append(event.occurred_at)
    mature_converted = sum(
        any(
            signup_at <= success_at <= signup_at + timedelta(days=7)
            for success_at in successes_by_user.get(user_id, [])
        )
        for user_id, signup_at in signups.items()
    )
    rows: list[DailyRollup] = [
        _row(day, "daily_active_users", count=overview.affected_users, updated_at=current),
        _row(day, "rolling_active_users_7d", count=overview.active_users_7d, updated_at=current),
        _row(day, "completed_runs", count=overview.completed_runs, updated_at=current),
        _row(day, "terminal_completed", count=overview.completed_runs, updated_at=current),
        _row(day, "terminal_failed", count=overview.failed_runs, updated_at=current),
        _row(
            day,
            "mature_signup_cohort_users",
            count=len(signups),
            updated_at=current,
        ),
        _row(
            day,
            "first_success_within_7d_users",
            count=mature_converted,
            updated_at=current,
        ),
        _row(
            day,
            "users_with_first_success",
            count=overview.users_with_first_success,
            updated_at=current,
        ),
        _row(
            day,
            "users_with_repeat_success_24h_30d",
            count=overview.users_with_repeat_success,
            updated_at=current,
        ),
        _row(
            day,
            "platform_model_cost_usd",
            sum_micro=round(overview.platform_model_cost_usd * 1_000_000),
            billing_mode="platform_credits",
            updated_at=current,
        ),
        _row(day, "input_tokens", count=overview.input_tokens, updated_at=current),
        _row(day, "output_tokens", count=overview.output_tokens, updated_at=current),
        _row(day, "affected_users", count=overview.affected_users, updated_at=current),
    ]

    event_dimensions = Counter(
        (
            event.event_name,
            event.billing_mode or "",
            event.provider_id or "",
            event.model_id or "",
            event.outcome or "",
            event.error_category or "",
        )
        for event in day_events
    )
    for dimensions, count in sorted(event_dimensions.items()):
        (
            event_name,
            event_billing_mode,
            event_provider_id,
            event_model_id,
            event_outcome,
            event_error_category,
        ) = dimensions
        rows.append(
            _row(
                day,
                "event_count",
                count=count,
                event_name=event_name,
                billing_mode=event_billing_mode,
                provider_id=event_provider_id,
                model_id=event_model_id,
                outcome=event_outcome,
                error_category=event_error_category,
                updated_at=current,
            )
        )
    usage_dimensions: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"input_tokens": 0, "output_tokens": 0, "cost_micro_usd": 0}
    )
    for event in day_events:
        if event.event_name != "model_usage_recorded":
            continue
        key = (
            event.billing_mode or "",
            event.provider_id or "",
            event.model_id or "",
        )
        usage_dimensions[key]["input_tokens"] += int(
            event.properties.get("input_tokens", 0)
        )
        usage_dimensions[key]["output_tokens"] += int(
            event.properties.get("output_tokens", 0)
        )
        if event.billing_mode == "platform_credits":
            usage_dimensions[key]["cost_micro_usd"] += int(
                event.properties.get("cost_micro_usd", 0)
            )
    for (mode, provider, model), values in sorted(usage_dimensions.items()):
        rows.extend(
            [
                _row(
                    day,
                    "input_tokens",
                    count=values["input_tokens"],
                    billing_mode=mode,
                    provider_id=provider,
                    model_id=model,
                    updated_at=current,
                ),
                _row(
                    day,
                    "output_tokens",
                    count=values["output_tokens"],
                    billing_mode=mode,
                    provider_id=provider,
                    model_id=model,
                    updated_at=current,
                ),
            ]
        )
        if mode == "platform_credits":
            rows.append(
                _row(
                    day,
                    "platform_model_cost_usd",
                    sum_micro=values["cost_micro_usd"],
                    billing_mode=mode,
                    provider_id=provider,
                    model_id=model,
                    updated_at=current,
                )
            )
    failure_users: dict[str, set[int]] = defaultdict(set)
    for event in day_events:
        if event.error_category:
            failure_users[event.error_category].add(event.user_id)
    for category, users in sorted(failure_users.items()):
        rows.append(
            _row(
                day,
                "affected_users",
                count=len(users),
                error_category=category,
                updated_at=current,
            )
        )
    for status, count in sorted((state_counts or {}).items()):
        rows.append(
            _row(
                day,
                "user_state_count",
                count=int(count),
                user_state=status,
                updated_at=current,
            )
        )

    rows.sort(
        key=lambda row: (
            row.metric_name,
            row.event_name,
            row.billing_mode,
            row.provider_id,
            row.model_id,
            row.outcome,
            row.error_category,
            row.user_state,
        )
    )
    aggregate_store.replace_day(day, rows)
    return rows


def rollup_current_day(
    *,
    store: AnalyticsRollupStore | None = None,
    now: datetime | None = None,
    include_internal: bool = False,
):
    current = now or datetime.now(timezone.utc)
    start = _day_start(current.astimezone(timezone.utc).date())
    aggregate_store = store or AnalyticsRollupStore()
    events = aggregate_store.list_events(
        start=start - timedelta(days=30),
        end=current,
        include_internal=include_internal,
    )
    return calculate_overview_metrics(events, start=start, end=current)


__all__ = [
    "AnalyticsRollupStore",
    "DailyRollup",
    "rollup_current_day",
    "rollup_day",
]
