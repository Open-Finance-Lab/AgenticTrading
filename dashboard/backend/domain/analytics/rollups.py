"""Companion repository and deterministic daily Analytics rollups.

PR 1 owns the tables.  PR 2 owns these bounded aggregate operations so the
Foundation repository contract remains unchanged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .metrics import ACTIVATION_EVENTS, calculate_overview_metrics
from .models import AnalyticsEventRecord
from .repository import _row_to_event, analytics_store
from .repository_common import utc_iso
from .usage_cost import atl_cost_micro, byok_estimate_micro
from .value_repository import LIFECYCLE_ROLLUP_METRICS


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
        columns = _ROLLUP_TABLE_COLUMNS
        payloads = [_rollup_payload(row) for row in rows]
        preserved = sorted(LIFECYCLE_ROLLUP_METRICS)
        if self.is_postgres:
            placeholders = ", ".join("%s" for _ in columns)
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        DELETE FROM analytics_daily_rollups
                        WHERE rollup_date = %s
                          AND metric_name NOT IN ({", ".join("%s" for _ in preserved)})
                        """,
                        (day.isoformat(), *preserved),
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
                    f"""
                    DELETE FROM analytics_daily_rollups
                    WHERE rollup_date = ?
                      AND metric_name NOT IN ({", ".join("?" for _ in preserved)})
                    """,
                    (day.isoformat(), *preserved),
                )
                if payloads:
                    conn.executemany(
                        f"INSERT INTO analytics_daily_rollups ({', '.join(columns)}) "
                        f"VALUES ({placeholders})",
                        payloads,
                    )

    def backfill_activation_users(self, *, since: date, until: date) -> tuple[date, ...]:
        """Write ``activation_users`` for rolled-up days that predate the metric.

        ``rollup_day`` gained the metric in PR B, so every day rolled up
        before it holds none, and the overview's funnel summed those days as
        zero. One grouped read over the raw events of the missing days --
        distinct users per step per day, counted in SQL -- fills them, where
        a ``rollup_day`` rebuild per day would re-read 31 days of events for
        each one. Only days in ``[since, until)`` that were rolled up (they
        hold ``daily_active_users``) are written, so a day whose raw events
        have expired is never filled with zeros; admin and excluded accounts
        are left out, as ``list_events`` leaves them out. A filled day no
        longer matches, so a re-run writes nothing. Returns the days filled.
        """
        p = "%s" if self.is_postgres else "?"
        missing_sql = f"""
            SELECT DISTINCT r.rollup_date FROM analytics_daily_rollups AS r
            WHERE r.metric_name = 'daily_active_users'
              AND r.rollup_date >= {p} AND r.rollup_date < {p}
              AND NOT EXISTS (
                  SELECT 1 FROM analytics_daily_rollups AS a
                  WHERE a.rollup_date = r.rollup_date
                    AND a.metric_name = 'activation_users'
              )
            ORDER BY r.rollup_date
        """
        events = sorted(ACTIVATION_EVENTS)
        counts_sql = f"""
            SELECT substr(occurred_at, 1, 10) AS day, event_name,
                   COUNT(DISTINCT user_id) AS users
            FROM analytics_events
            WHERE occurred_at >= {p} AND occurred_at < {p}
              AND event_name IN ({", ".join(p for _ in events)})
              AND user_id NOT IN (SELECT id FROM users WHERE role = 'admin')
              AND user_id NOT IN (
                  SELECT user_id FROM analytics_subject_settings WHERE excluded
              )
            GROUP BY substr(occurred_at, 1, 10), event_name
        """
        insert_sql = (
            f"INSERT INTO analytics_daily_rollups ({', '.join(_ROLLUP_TABLE_COLUMNS)}) "
            f"VALUES ({', '.join(p for _ in _ROLLUP_TABLE_COLUMNS)})"
        )

        def fill(run: Any) -> tuple[date, ...]:
            days = tuple(
                date.fromisoformat(str(row["rollup_date"]))
                for row in run(missing_sql, (since.isoformat(), until.isoformat()))
            )
            if not days:
                return ()
            counts = {
                (str(row["day"]), str(row["event_name"])): int(row["users"])
                for row in run(
                    counts_sql,
                    (
                        utc_iso(_day_start(days[0])),
                        utc_iso(_day_start(days[-1] + timedelta(days=1))),
                        *events,
                    ),
                )
            }
            payloads = [
                _rollup_payload(
                    _row(
                        day,
                        "activation_users",
                        count=counts.get((day.isoformat(), event_name), 0),
                        event_name=event_name,
                        updated_at=_day_start(day + timedelta(days=1)),
                    )
                )
                for day in days
                for event_name in ACTIVATION_EVENTS
            ]
            run(insert_sql, payloads, many=True)
            return days

        if self.is_postgres:
            with self.base_store._get_connection() as conn:
                with conn.cursor() as cur:

                    def run_postgres(sql: str, params: Any, many: bool = False) -> Any:
                        if many:
                            cur.executemany(sql, params)
                            return []
                        cur.execute(sql, params)
                        return cur.fetchall()

                    return fill(run_postgres)
        with self.base_store._get_connection() as conn:

            def run_sqlite(sql: str, params: Any, many: bool = False) -> Any:
                if many:
                    conn.executemany(sql, params)
                    return []
                return conn.execute(sql, params).fetchall()

            return fill(run_sqlite)

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


_ROLLUP_TABLE_COLUMNS = (
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


def _rollup_payload(row: DailyRollup) -> tuple[Any, ...]:
    return (
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
        lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_micro_usd": 0,
            "unpriced_calls": 0,
        }
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
        # One cost per lane, published under separate metrics: the platform
        # figure is a debit, the BYOK one a list-price estimate
        # (analytics.usage_cost), and the two must never add into one another.
        if event.billing_mode == "byok":
            estimate = byok_estimate_micro(event.properties, event.model_id)
            if estimate is None:
                usage_dimensions[key]["unpriced_calls"] += 1
            else:
                usage_dimensions[key]["cost_micro_usd"] += estimate
        else:
            usage_dimensions[key]["cost_micro_usd"] += atl_cost_micro(
                event.billing_mode, event.properties
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
        elif mode == "byok":
            rows.extend(
                [
                    _row(
                        day,
                        "byok_estimated_cost_usd",
                        sum_micro=values["cost_micro_usd"],
                        billing_mode=mode,
                        provider_id=provider,
                        model_id=model,
                        updated_at=current,
                    ),
                    # Written zero or not: its presence is how a reader tells
                    # a day rolled up before unpriced calls were counted.
                    _row(
                        day,
                        "byok_unpriced_calls",
                        count=values["unpriced_calls"],
                        billing_mode=mode,
                        provider_id=provider,
                        model_id=model,
                        updated_at=current,
                    ),
                ]
            )
    # Distinct users per activation step per day: the overview's funnel sums
    # these across completed days instead of rescanning the window. Written
    # zero or not, like byok_unpriced_calls: presence is how the funnel tells
    # a day with no activations from a day rolled up before this metric
    # existed, which it must not read as zero.
    activation_users: dict[str, set[int]] = defaultdict(set)
    for event in day_events:
        if event.event_name in ACTIVATION_EVENTS:
            activation_users[event.event_name].add(event.user_id)
    for event_name in ACTIVATION_EVENTS:
        rows.append(
            _row(
                day,
                "activation_users",
                count=len(activation_users[event_name]),
                event_name=event_name,
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


def _quality(rows: Sequence[Any]) -> str:
    return "partial" if any(row.data_quality == "partial" for row in rows) else "complete"


def rollup_lifecycle_day(
    day: date,
    *,
    store,
) -> tuple[DailyRollup, ...]:
    """Persist anonymous lifecycle counts, transitions and the long-term
    tier / user_group families for one day (design SS6.7, SS11).

    Reads the day's fact rows and transition log; writes nothing keyed by a
    user. The retention sweep calls it for every expiring day before that
    day's user-keyed rows are deleted. ``runs_by_tier`` /
    ``runs_by_user_group`` count ``runs_completed``, not the
    ``runs_requested`` that ``/groups`` reports as ``total_runs``. A day with
    no facts writes nothing:
    its facts may already have expired, and replacing its lifecycle rollups
    with an empty set would erase the history this function exists to keep.
    """

    facts = store.list_facts_for_date(day)
    transitions = store.list_transitions_for_date(day)
    if not facts:
        if transitions:
            # Unreachable while facts and transitions expire together; if it
            # ever happens the day's movement is lost, so say so.
            print(
                "WARNING: analytics.rollup.lifecycle_day_without_facts "
                f"day={day.isoformat()} transitions={len(transitions)}",
                flush=True,
            )
        return ()
    updated_at = _day_start(day + timedelta(days=1))
    rows: list[DailyRollup] = []

    by_segment: dict[str, list[Any]] = defaultdict(list)
    for fact in facts:
        by_segment[fact.lifecycle_segment].append(fact)
    for segment, group in sorted(by_segment.items()):
        rows.append(
            _row(
                day,
                "lifecycle_segment_count",
                count=len(group),
                user_state=segment,
                outcome=_quality(group),
                updated_at=updated_at,
            )
        )

    by_transition: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for transition in transitions:
        by_transition[(transition.from_segment, transition.to_segment)].append(
            transition
        )
    for (from_segment, to_segment), group in sorted(by_transition.items()):
        rows.append(
            _row(
                day,
                "lifecycle_transition",
                count=len(group),
                event_name=from_segment,
                user_state=to_segment,
                outcome=_quality(group),
                updated_at=updated_at,
            )
        )

    # ``runs_by_*`` count runs_completed -- finished runs -- unlike /groups'
    # ``total_runs``, which counts runs_requested. These rows are permanent
    # once the day's facts expire, so the semantics are fixed here.
    for dimension in ("tier", "user_group"):
        by_value: dict[str, list[Any]] = defaultdict(list)
        for fact in facts:
            by_value[getattr(fact, dimension)].append(fact)
        for value, group in sorted(by_value.items()):
            quality = _quality(group)
            rows.append(
                _row(
                    day,
                    f"runs_by_{dimension}",
                    count=sum(fact.runs_completed for fact in group),
                    user_state=value,
                    outcome=quality,
                    updated_at=updated_at,
                )
            )
            rows.append(
                _row(
                    day,
                    f"operator_cost_by_{dimension}",
                    count=len(group),
                    sum_micro=sum(fact.operator_cost_micro for fact in group),
                    user_state=value,
                    outcome=quality,
                    updated_at=updated_at,
                )
            )
            rows.append(
                _row(
                    day,
                    f"own_spend_by_{dimension}",
                    count=len(group),
                    sum_micro=sum(fact.own_spend_micro for fact in group),
                    user_state=value,
                    outcome=quality,
                    updated_at=updated_at,
                )
            )

    rows.sort(key=lambda row: (row.metric_name, row.event_name, row.user_state))
    # Transitions come from their own table for ``day``, so there is no
    # "previous day missing" case left to protect against.
    store.replace_lifecycle_rollups(day, rows, replace_transitions=True)
    return tuple(rows)


def _legacy_lifecycle_rows(
    day: date, current: Sequence[Any], previous: Sequence[Any]
) -> list[DailyRollup]:
    """Main's ``rollup_lifecycle_day`` over legacy snapshot rows, unchanged in shape:
    a count per segment, and a transition per (previous, current) segment pair
    for users present on both days whose segment moved."""
    updated_at = _day_start(day + timedelta(days=1))
    rows: list[DailyRollup] = []
    by_segment: dict[str, list[Any]] = defaultdict(list)
    for snapshot in current:
        by_segment[snapshot.lifecycle_segment].append(snapshot)
    for segment, group in sorted(by_segment.items()):
        rows.append(
            _row(
                day,
                "lifecycle_segment_count",
                count=len(group),
                user_state=segment,
                outcome=_quality(group),
                updated_at=updated_at,
            )
        )
    previous_by_user = {snapshot.user_id: snapshot for snapshot in previous}
    moved: dict[tuple[str, str], list[Any]] = defaultdict(list)
    moved_quality: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for snapshot in current:
        prior = previous_by_user.get(snapshot.user_id)
        if prior is None or prior.lifecycle_segment == snapshot.lifecycle_segment:
            continue
        key = (prior.lifecycle_segment, snapshot.lifecycle_segment)
        moved[key].append(snapshot)
        moved_quality[key].extend((prior, snapshot))
    for (from_segment, to_segment), group in sorted(moved.items()):
        rows.append(
            _row(
                day,
                "lifecycle_transition",
                count=len(group),
                event_name=from_segment,
                user_state=to_segment,
                outcome=_quality(moved_quality[(from_segment, to_segment)]),
                updated_at=updated_at,
            )
        )
    rows.sort(key=lambda row: (row.metric_name, row.event_name, row.user_state))
    return rows


def rollup_legacy_lifecycle_days(*, store) -> tuple[date, ...]:
    """Roll the legacy days no fact row covers into anonymous lifecycle rollups.

    The startup migration's step between the ``user_activity`` seed and the
    drop of ``user_lifecycle_daily_snapshots`` (design SS6.9, SS11): the PR A
    history copy reached back only 56 days, so older legacy days live in that
    table alone, and ``_history`` already falls back to these two families for
    any day without facts. Days that already hold a lifecycle-owned rollup are
    excluded by the store's read, so nothing is overwritten and a re-run (a
    boot whose drop failed) writes nothing new. Transitions are written only
    when the previous legacy day exists, exactly as main's writer did.
    Returns the days rolled up.
    """
    history = store.list_legacy_lifecycle_history()
    by_day: dict[date, list[Any]] = defaultdict(list)
    for snapshot in history.snapshots:
        by_day[snapshot.snapshot_date].append(snapshot)
    for day in history.days:
        previous = by_day.get(day - timedelta(days=1), [])
        store.replace_lifecycle_rollups(
            day,
            _legacy_lifecycle_rows(day, by_day.get(day, []), previous),
            replace_transitions=bool(previous),
        )
    return history.days


__all__ = [
    "AnalyticsRollupStore",
    "DailyRollup",
    "rollup_day",
    "rollup_legacy_lifecycle_days",
    "rollup_lifecycle_day",
]
