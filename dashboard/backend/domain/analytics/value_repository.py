"""Persistence primitives for the user-value Analytics projection.

This module deliberately keeps the Credits ledger authoritative.  Analytics
stores only calculated lifecycle history and reads commercial facts in batches.

SQLite twin. The PostgreSQL twin is ``value_repository_postgres.py``
(``PostgresValueAnalyticsStore``); ``build_value_analytics_store()`` below
selects between them the same way ``repository.py``'s ``_build_analytics_store()``
selects between ``AnalyticsStore`` and ``PostgresAnalyticsStore``. Every method
that reads or writes this store's own tables (``user_analytics_snapshots``,
``user_lifecycle_daily_snapshots``, ``analytics_projection_jobs``) has exactly
one code path here; the two methods that branch on a *different* store's
dialect (``list_commercial_values``, ``list_credit_activity``, both reading
``self.credits_base``) are unchanged and identical on both twins, because that
branch was never about which twin this class is -- a caller can pair either
``analytics_base`` with either ``credits_base``, and both twins must handle it.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .lifecycle import (
    CommercialTier,
    LifecycleSegment,
    OperationalState,
    commercial_tier,
)
from .repository import analytics_store
from .repository_common import positive_limit, positive_user_id, utc_iso


MAX_USER_BATCH = 500
RUN_SAFE_DEADLINE = timedelta(minutes=60)
LIFECYCLE_SEGMENTS = frozenset(
    {"new", "onboarding", "growing", "core", "at_risk", "dormant"}
)
LIFECYCLE_ROLLUP_METRICS = frozenset(
    {"lifecycle_segment_count", "lifecycle_transition"}
)
_ACTIVE_RUN_STATUSES = frozenset({"created", "loading", "running"})
_TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "cancelled", "closed", "timed_out"}
)


def _utc(value: datetime, name: str = "timestamp") -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc)


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    try:
        return row[name]
    except (IndexError, KeyError, TypeError):
        return default


def _object_value(value: object, name: str, default: Any = 0) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return _timestamp(value)
    except (TypeError, ValueError):
        return None


class UserValueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str = Field(min_length=1, max_length=100)
    lifecycle_reason: str = Field(min_length=1, max_length=500)
    lifecycle_evidence: Sequence[str] = Field(default_factory=tuple, max_length=10)
    operational_state: OperationalState
    operational_reason_code: str = Field(min_length=1, max_length=100)
    operational_reason: str = Field(min_length=1, max_length=500)
    operational_evidence: Sequence[str] = Field(default_factory=tuple, max_length=10)
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    inactive_days: int = Field(ge=0)
    active_days_30d: int = Field(ge=0, le=30)
    successful_backtests_30d: int = Field(ge=0)
    calculated_at: datetime

    @field_validator("activated_at", "last_meaningful_activity_at", "calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


class UserLifecycleDailySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date
    user_id: int = Field(gt=0)
    lifecycle_segment: LifecycleSegment
    lifecycle_reason_code: str = Field(min_length=1, max_length=100)
    data_quality: Literal["complete", "partial"]
    calculated_at: datetime

    @field_validator("calculated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)


class UserActivity(BaseModel):
    """One user's current activity clock. One row, overwritten in place.

    Design SS6.5: ``activated_at`` is the first accepted ``backtest_completed``
    and is set once; ``last_meaningful_activity_at`` is the greatest
    ``occurred_at`` of any accepted event in the lifecycle activity set. It is
    current state, not history, and is never swept (SS11).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    activated_at: datetime | None = None
    last_meaningful_activity_at: datetime | None = None
    updated_at: datetime

    @field_validator("activated_at", "last_meaningful_activity_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


def _activity_from_row(row: Any) -> UserActivity:
    """Shared by both twins."""
    return UserActivity(
        user_id=int(_row_value(row, "user_id")),
        activated_at=_optional_timestamp(_row_value(row, "activated_at")),
        last_meaningful_activity_at=_optional_timestamp(
            _row_value(row, "last_meaningful_activity_at")
        ),
        updated_at=_timestamp(_row_value(row, "updated_at")),
    )


class RecentFactTotals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    active_days: int = Field(default=0, ge=0)
    successful_backtests: int = Field(default=0, ge=0)
    runs_requested: int = Field(default=0, ge=0)
    runs_completed: int = Field(default=0, ge=0)
    runs_failed: int = Field(default=0, ge=0)
    runs_cancelled: int = Field(default=0, ge=0)
    operator_cost_micro: int = Field(default=0, ge=0)
    own_spend_micro: int = Field(default=0, ge=0)
    # How many of the requested dates actually have a row. Fewer than
    # requested means the window is incomplete, which the UI labels rather
    # than rendering as zero.
    days_present: int = Field(default=0, ge=0)
    # The latest date inside the window on which this user was active, or
    # None. The daily job needs it to answer "what did this user's activity
    # look like as of the end of day D" -- `user_activity` holds one row
    # overwritten in place and cannot answer a question about the past.
    last_active_date: date | None = None


_RECENT_FACTS_SQL = """
    SELECT user_id,
           SUM(CASE WHEN active THEN 1 ELSE 0 END) AS active_days,
           SUM(runs_completed) AS successful_backtests,
           SUM(runs_requested) AS runs_requested,
           SUM(runs_failed) AS runs_failed,
           SUM(runs_cancelled) AS runs_cancelled,
           SUM(operator_cost_micro) AS operator_cost_micro,
           SUM(own_spend_micro) AS own_spend_micro,
           COUNT(*) AS days_present,
           MAX(CASE WHEN active THEN snapshot_date END) AS last_active_date
    FROM user_daily_facts
    WHERE snapshot_date >= {p} AND snapshot_date <= {p}{user_clause}
    GROUP BY user_id
"""


def _recent_totals_from_row(row: Any) -> RecentFactTotals:
    """Shared by both twins."""
    last_active = _row_value(row, "last_active_date")
    completed = int(_row_value(row, "successful_backtests", 0) or 0)
    return RecentFactTotals(
        active_days=int(_row_value(row, "active_days", 0) or 0),
        successful_backtests=completed,
        runs_requested=int(_row_value(row, "runs_requested", 0) or 0),
        runs_completed=completed,
        runs_failed=int(_row_value(row, "runs_failed", 0) or 0),
        runs_cancelled=int(_row_value(row, "runs_cancelled", 0) or 0),
        operator_cost_micro=int(_row_value(row, "operator_cost_micro", 0) or 0),
        own_spend_micro=int(_row_value(row, "own_spend_micro", 0) or 0),
        days_present=int(_row_value(row, "days_present", 0) or 0),
        last_active_date=(
            date.fromisoformat(str(last_active)) if last_active else None
        ),
    )


class CommercialValueFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    lifetime_net_purchased_micro: int = Field(ge=0)
    commercial_tier: CommercialTier
    purchased_micro: int = Field(ge=0)
    refunded_micro: int = Field(ge=0)
    consumed_micro: int = Field(ge=0)
    admin_grant_activity_micro: int = Field(ge=0)
    grant_available_micro: int = Field(ge=0)
    purchased_available_micro: int = Field(ge=0)
    total_available_micro: int = Field(ge=0)


class CurrentOperationalFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    account_restricted: bool = False
    usable_billing_lane: bool = True
    selected_provider_enabled: bool = True
    default_credential_status: Literal[
        "verified", "invalid", "verification_unavailable", "missing"
    ] = "verified"
    failed_terminal_runs_24h: int = Field(default=0, ge=0)
    run_beyond_safe_deadline: bool = False


class ProjectionJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_name: str = Field(min_length=1, max_length=100)
    window_start: date
    window_end: date
    cursor: str | None = None
    status: Literal["pending", "running", "complete"]
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_window(self) -> "ProjectionJob":
        if self.window_end < self.window_start:
            raise ValueError("window_end must not precede window_start")
        return self


def _ids(user_ids: Sequence[int]) -> list[int]:
    if not isinstance(user_ids, (list, tuple)):
        raise ValueError("user_ids must be a list or tuple")
    values = list(dict.fromkeys(positive_user_id(item) for item in user_ids))
    if len(values) > MAX_USER_BATCH:
        raise ValueError(f"user_ids must contain at most {MAX_USER_BATCH} users")
    return values


def _validate_window(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    window_start = _utc(start, "start")
    window_end = _utc(end, "end")
    if window_end <= window_start:
        raise ValueError("end must be later than start")
    return window_start, window_end


def _legacy_seed(snapshot: UserValueSnapshot) -> tuple[str, str, str]:
    """Seed compatibility fields only when no legacy projection exists yet."""

    if snapshot.operational_state == "blocked":
        status = "blocked"
        reason_code = snapshot.operational_reason_code
        reason = snapshot.operational_reason
    elif snapshot.operational_state == "needs_attention":
        status = "needs_attention"
        reason_code = snapshot.operational_reason_code
        reason = snapshot.operational_reason
    elif snapshot.lifecycle_segment == "dormant":
        status = "dormant"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    elif snapshot.lifecycle_segment in {"new", "onboarding"}:
        status = "onboarding"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    else:
        status = "active"
        reason_code = snapshot.lifecycle_reason_code
        reason = snapshot.lifecycle_reason
    return status, reason_code, reason


def _current_snapshot_from_row(row: Any) -> UserValueSnapshot | None:
    """Shared by both twins.

    Moved out of the class (was ``ValueAnalyticsStore._current_snapshot_from_row``,
    a ``@staticmethod``) so ``value_repository_postgres.py`` can import it
    directly, mirroring how ``repository_postgres.py`` imports bare functions
    (``_row_to_event``) from ``repository.py``. No caller outside this module
    referenced the staticmethod, so this is not a behaviour change.
    """
    if row is None or _row_value(row, "lifecycle_segment") is None:
        return None

    def seq(name: str) -> tuple[str, ...]:
        try:
            value = json.loads(_row_value(row, name, "[]"))
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return ()
            return tuple(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()

    return UserValueSnapshot(
        user_id=int(_row_value(row, "user_id")),
        lifecycle_segment=_row_value(row, "lifecycle_segment"),
        lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
        lifecycle_reason=_row_value(row, "lifecycle_reason"),
        lifecycle_evidence=seq("lifecycle_evidence_json"),
        operational_state=_row_value(row, "operational_state") or "healthy",
        operational_reason_code=(
            _row_value(row, "operational_reason_code") or "no_supported_issue"
        ),
        operational_reason=(
            _row_value(row, "operational_reason")
            or "No supported current operational issue was detected."
        ),
        operational_evidence=seq("operational_evidence_json"),
        activated_at=_optional_timestamp(_row_value(row, "activated_at")),
        last_meaningful_activity_at=_optional_timestamp(
            _row_value(row, "last_meaningful_activity_at")
        ),
        inactive_days=int(_row_value(row, "inactive_days", 0)),
        active_days_30d=int(_row_value(row, "active_days_30d", 0)),
        successful_backtests_30d=int(
            _row_value(row, "successful_backtests_30d", 0)
        ),
        calculated_at=_timestamp(_row_value(row, "calculated_at")),
    )


def _fetchall(conn: Any, postgres: bool, sql: str, params: Sequence[Any]):
    """Shared by both twins.

    Still dialect-parameterised: it serves ``list_commercial_values``/
    ``list_credit_activity``, which branch on the *credits* store's dialect,
    never on this class's own -- see the module docstring.
    """
    if postgres:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    return conn.execute(sql, params).fetchall()


def _user_clause(ids: list[int], postgres: bool) -> tuple[str, list[Any]]:
    """Shared by both twins; see ``_fetchall`` above."""
    if postgres:
        return "user_id = ANY(%s)", [ids]
    placeholders = ", ".join("?" for _ in ids)
    return f"user_id IN ({placeholders})", list(ids)


def _projection_job_name(value: object) -> str:
    """Shared by both twins; see ``_current_snapshot_from_row`` above."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 100
    ):
        raise ValueError("job_name must be a trimmed non-empty string")
    return value


class ValueAnalyticsStore:
    """SQLite value projection storage.

    See ``value_repository_postgres.py`` for the PostgreSQL twin;
    ``build_value_analytics_store()`` below picks between them.

    ``credits_base`` and the optional operational stores are injectable to keep
    contract tests synthetic and to avoid importing production singletons.
    """

    def __init__(
        self,
        analytics_base: Any | None = None,
        credits_base: Any | None = None,
        provider_base: Any | None = None,
        agent_base: Any | None = None,
        run_base: Any | None = None,
    ) -> None:
        # The mirror of PostgresValueAnalyticsStore's guard, and the reason
        # this class needs one at all: until PR T it served both dialects, so
        # `ValueAnalyticsStore(postgres_base)` was correct and is still the
        # natural thing to write. It now emits `?` placeholders, which raise
        # psycopg.errors.SyntaxError on the Postgres deployment *only* -- CI
        # and local runs are both SQLite, so a caller that reintroduced it
        # would keep a green suite all the way to prod.
        self.analytics_base = analytics_base or analytics_store
        if hasattr(self.analytics_base, "database_url"):
            raise TypeError(
                "ValueAnalyticsStore is the SQLite twin and emits `?` "
                "placeholders, but the resolved analytics base is "
                "PostgreSQL. Build the pair through "
                "build_value_analytics_store(), which resolves the base and "
                "returns the matching twin."
            )
        if credits_base is None:
            from dashboard.backend.domain.credits.repository import credits_store

            credits_base = credits_store
        self.credits_base = credits_base
        if provider_base is None:
            from dashboard.backend.domain.model_providers.repository import (
                model_provider_store,
            )

            provider_base = model_provider_store
        self.provider_base = provider_base
        if agent_base is None:
            from dashboard.backend.domain.agents.repository import agent_store

            agent_base = agent_store
        self.agent_base = agent_base
        if run_base is None:
            from dashboard.backend.domain.runs.repository import run_store

            run_base = run_store
        self.run_base = run_base

    def _analytics_connection(self):
        return self.analytics_base._get_connection()

    def upsert_current_snapshot(self, snapshot: UserValueSnapshot) -> UserValueSnapshot:
        def evidence(value: Sequence[str]) -> str:
            return json.dumps(
                list(value),
                separators=(",", ":"),
                ensure_ascii=True,
            )

        legacy_status, legacy_reason_code, legacy_reason = _legacy_seed(snapshot)
        values = (
            snapshot.user_id,
            legacy_status,
            legacy_reason_code,
            legacy_reason,
            "[]",
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.lifecycle_reason,
            evidence(snapshot.lifecycle_evidence),
            snapshot.operational_state,
            snapshot.operational_reason_code,
            snapshot.operational_reason,
            evidence(snapshot.operational_evidence),
            utc_iso(snapshot.activated_at) if snapshot.activated_at else None,
            (
                utc_iso(snapshot.last_meaningful_activity_at)
                if snapshot.last_meaningful_activity_at
                else None
            ),
            snapshot.inactive_days,
            snapshot.active_days_30d,
            snapshot.successful_backtests_30d,
            utc_iso(snapshot.calculated_at),
        )
        columns = """
            user_id, status, reason_code, human_readable_reason,
            evidence_event_ids_json, lifecycle_segment, lifecycle_reason_code,
            lifecycle_reason, lifecycle_evidence_json, operational_state,
            operational_reason_code, operational_reason,
            operational_evidence_json, activated_at,
            last_meaningful_activity_at, inactive_days, active_days_30d,
            successful_backtests_30d, calculated_at
        """
        updates = """
            lifecycle_segment=excluded.lifecycle_segment,
            lifecycle_reason_code=excluded.lifecycle_reason_code,
            lifecycle_reason=excluded.lifecycle_reason,
            lifecycle_evidence_json=excluded.lifecycle_evidence_json,
            operational_state=excluded.operational_state,
            operational_reason_code=excluded.operational_reason_code,
            operational_reason=excluded.operational_reason,
            operational_evidence_json=excluded.operational_evidence_json,
            activated_at=excluded.activated_at,
            last_meaningful_activity_at=excluded.last_meaningful_activity_at,
            inactive_days=excluded.inactive_days,
            active_days_30d=excluded.active_days_30d,
            successful_backtests_30d=excluded.successful_backtests_30d,
            calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO user_analytics_snapshots ({columns})
                VALUES ({", ".join(["?"] * len(values))})
                ON CONFLICT(user_id) DO UPDATE SET {updates}
                """,
                values,
            )
        return snapshot

    def get_current_snapshot(self, user_id: int) -> UserValueSnapshot | None:
        subject = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_analytics_snapshots WHERE user_id=?", (subject,)
            ).fetchone()
        return _current_snapshot_from_row(row)

    def list_current_snapshots(
        self,
        user_ids: Sequence[int],
    ) -> dict[int, UserValueSnapshot]:
        ids = _ids(user_ids)
        if not ids:
            return {}
        result: dict[int, UserValueSnapshot] = {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            clause = f"user_id IN ({', '.join('?' for _ in chunk)})"
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM user_analytics_snapshots
                    WHERE {clause} AND lifecycle_segment IS NOT NULL
                    ORDER BY user_id
                    """,
                    chunk,
                ).fetchall()
            for row in rows:
                user_id = int(_row_value(row, "user_id"))
                snapshot = _current_snapshot_from_row(row)
                if snapshot is not None:
                    result[user_id] = snapshot
        return result

    def upsert_daily_snapshot(
        self,
        snapshot: UserLifecycleDailySnapshot,
    ) -> UserLifecycleDailySnapshot:
        values = (
            snapshot.snapshot_date.isoformat(),
            snapshot.user_id,
            snapshot.lifecycle_segment,
            snapshot.lifecycle_reason_code,
            snapshot.data_quality,
            utc_iso(snapshot.calculated_at),
        )
        sql = """
            INSERT INTO user_lifecycle_daily_snapshots (
                snapshot_date, user_id, lifecycle_segment,
                lifecycle_reason_code, data_quality, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET
                lifecycle_segment=excluded.lifecycle_segment,
                lifecycle_reason_code=excluded.lifecycle_reason_code,
                data_quality=excluded.data_quality,
                calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)
        return snapshot

    def list_daily_snapshots(
        self,
        *,
        start: date,
        end: date,
        user_ids: Sequence[int] | None = None,
    ) -> list[UserLifecycleDailySnapshot]:
        if end <= start:
            raise ValueError("end must be later than start")
        ids = _ids(user_ids) if user_ids is not None else None
        if ids == []:
            return []
        params: list[Any] = [start.isoformat(), end.isoformat()]
        clause = ""
        if ids:
            clause = f" AND user_id IN ({','.join('?' for _ in ids)})"
            params.extend(ids)
        sql = f"""
            SELECT *
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date >= ?
              AND snapshot_date < ?
              {clause}
            ORDER BY snapshot_date, user_id
        """
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            UserLifecycleDailySnapshot(
                snapshot_date=date.fromisoformat(str(_row_value(row, "snapshot_date"))),
                user_id=int(_row_value(row, "user_id")),
                lifecycle_segment=_row_value(row, "lifecycle_segment"),
                lifecycle_reason_code=_row_value(row, "lifecycle_reason_code"),
                data_quality=_row_value(row, "data_quality"),
                calculated_at=_timestamp(_row_value(row, "calculated_at")),
            )
            for row in rows
        ]

    def replace_lifecycle_rollups(
        self,
        day: date,
        rows: Sequence[Any],
        *,
        replace_transitions: bool = True,
    ) -> None:
        """Replace only lifecycle aggregates, preserving other daily metrics."""

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
        payloads = []
        for row in values:
            if (
                row.rollup_date != day
                or row.metric_name not in LIFECYCLE_ROLLUP_METRICS
            ):
                raise ValueError("invalid lifecycle rollup row")
            if row.metric_name == "lifecycle_segment_count":
                valid_dimensions = (
                    not row.event_name and row.user_state in LIFECYCLE_SEGMENTS
                )
            else:
                valid_dimensions = (
                    replace_transitions
                    and row.event_name in LIFECYCLE_SEGMENTS
                    and row.user_state in LIFECYCLE_SEGMENTS
                    and row.event_name != row.user_state
                )
            unused_dimensions = (
                row.billing_mode,
                row.provider_id,
                row.model_id,
                row.error_category,
            )
            if (
                not valid_dimensions
                or any(unused_dimensions)
                or row.outcome not in {"complete", "partial"}
                or row.value_sum_micro != 0
            ):
                raise ValueError("invalid lifecycle rollup dimensions")
            payloads.append(
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
            )
        placeholders = ", ".join(["?"] * len(columns))
        metrics = ["lifecycle_segment_count"]
        if replace_transitions:
            metrics.append("lifecycle_transition")
        metric_placeholders = ", ".join(["?"] * len(metrics))
        with self._analytics_connection() as conn:
            conn.execute(
                f"""
                DELETE FROM analytics_daily_rollups
                WHERE rollup_date = ?
                  AND metric_name IN ({metric_placeholders})
                """,
                (day.isoformat(), *metrics),
            )
            if payloads:
                conn.executemany(
                    f"""
                    INSERT INTO analytics_daily_rollups ({', '.join(columns)})
                    VALUES ({placeholders})
                    """,
                    payloads,
                )

    def list_expiring_daily_dates(
        self,
        *,
        before: date,
        limit: int,
    ) -> list[date]:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        page_size = positive_limit(limit, maximum=1000)
        sql = """
            SELECT DISTINCT snapshot_date
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < ?
            ORDER BY snapshot_date
            LIMIT ?
        """
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, (before.isoformat(), page_size)).fetchall()
        return [
            date.fromisoformat(str(_row_value(row, "snapshot_date"))) for row in rows
        ]

    def delete_daily_snapshots_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM user_lifecycle_daily_snapshots
                WHERE snapshot_date = ?
                """,
                (day.isoformat(),),
            )
            return max(0, int(cursor.rowcount))

    def has_daily_before(self, before: date) -> bool:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        sql = """
            SELECT 1
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < ?
            LIMIT 1
        """
        with self._analytics_connection() as conn:
            row = conn.execute(sql, (before.isoformat(),)).fetchone()
        return row is not None

    def record_activity(
        self,
        user_id: int,
        *,
        occurred_at: datetime,
        activating: bool,
        now: datetime,
    ) -> None:
        """Advance one user's activity timestamps. One statement, no read.

        The two columns move in opposite directions, on purpose.

        ``activated_at`` keeps the **earliest** success, because activation
        is defined as the first server-authoritative ``backtest_completed``
        by ``occurred_at`` -- not by arrival order. Ingestion does not see
        events in occurred_at order: a completion can be appended late,
        replayed, or backdated up to the 24 hours ``service.py:96-97``
        accepts. A plain ``COALESCE(stored, incoming)`` would freeze
        whichever row happened to land first and make the value
        uncorrectable afterwards, which would also silently turn the daily
        job's ``events`` step repair into a no-op. A non-activating event
        passes NULL and the COALESCE pair leaves the stored value alone.

        ``last_meaningful_activity_at`` only ever advances, so a
        late-arriving old event cannot make a user look more dormant than
        they are.

        Timestamps are ISO-8601 UTC text throughout this schema, which orders
        lexicographically, so MIN and MAX over the text are the same
        comparisons as over the instants.

        SQLite's scalar ``max(X, Y)`` and ``min(X, Y)`` both return NULL if
        **either** argument is NULL, which is why each stored value is
        COALESCEd against the incoming one before the comparison rather than
        passed raw.
        """
        subject_id = positive_user_id(user_id)
        occurred = utc_iso(_utc(occurred_at, "occurred_at"))
        values = (
            subject_id,
            occurred if activating else None,
            occurred,
            utc_iso(_utc(now, "now")),
        )
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = MIN(
                    COALESCE(user_activity.activated_at, excluded.activated_at),
                    COALESCE(excluded.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = MAX(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        excluded.last_meaningful_activity_at
                    ),
                    excluded.last_meaningful_activity_at
                ),
                updated_at = excluded.updated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)

    def get_activity(self, user_id: int) -> UserActivity | None:
        """One user's stored activity row, or None if they have none yet."""
        subject_id = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_activity WHERE user_id = ?", (subject_id,)
            ).fetchone()
        return _activity_from_row(row) if row is not None else None

    def list_activity(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, UserActivity]:
        """Activity rows for many users, or for everyone when ``user_ids`` is None.

        ``None`` is the daily job's shape: one statement over the whole
        population, parameterised by nothing. A sequence is batched by
        ``MAX_USER_BATCH`` the way ``list_current_snapshots`` already is.
        """
        result: dict[int, UserActivity] = {}
        if user_ids is None:
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM user_activity ORDER BY user_id"
                ).fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
            return result
        ids = _ids(user_ids)
        if not ids:
            return {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            clause = f"user_id IN ({', '.join('?' for _ in chunk)})"
            with self._analytics_connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM user_activity WHERE {clause} ORDER BY user_id",
                    chunk,
                ).fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
        return result

    def seed_activity_from_snapshots(self, *, now: datetime) -> int:
        """Copy ``activated_at`` / ``last_meaningful_activity_at`` from the legacy row.

        Design SS11: ``user_analytics_snapshots`` is the only table that knows
        when an existing user first activated, and it is dropped in PR B.
        Ingestion only maintains ``user_activity`` for events arriving after
        this deploy, so without this copy every pre-existing activation date
        would vanish on the night of the drop.

        Idempotent by construction -- the same MIN/MAX upsert
        ``record_activity`` uses -- so PR B can re-run it immediately before
        the drop (the legacy row keeps being maintained by the throttled
        repair between the two PRs). Returns the number of rows touched.
        """
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            )
            SELECT user_id, activated_at, last_meaningful_activity_at, ?
            FROM user_analytics_snapshots
            WHERE activated_at IS NOT NULL
               OR last_meaningful_activity_at IS NOT NULL
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = MIN(
                    COALESCE(user_activity.activated_at, excluded.activated_at),
                    COALESCE(excluded.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = MAX(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        excluded.last_meaningful_activity_at
                    ),
                    COALESCE(
                        excluded.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = excluded.updated_at
        """
        with self._analytics_connection() as conn:
            cursor = conn.execute(sql, (stamp,))
            return max(0, int(cursor.rowcount))

    def sum_recent_facts(
        self,
        user_ids: Sequence[int] | None,
        *,
        start: date,
        end: date,
    ) -> dict[int, RecentFactTotals]:
        """Trailing-window totals for many users in one query.

        ``start`` and ``end`` are inclusive UTC dates. ``None`` for
        ``user_ids`` means the whole population -- the daily job's shape,
        parameterised by two dates and nothing else. One statement for the
        whole batch: a per-user loop here is the shape that caused the
        outage, and the read-budget test fails on it.
        """
        if end < start:
            raise ValueError("end must not precede start")
        params: list[Any] = [start.isoformat(), end.isoformat()]
        user_clause = ""
        if user_ids is not None:
            ids = _ids(user_ids)
            if not ids:
                return {}
            user_clause = f" AND user_id IN ({', '.join('?' for _ in ids)})"
            params.extend(ids)
        sql = _RECENT_FACTS_SQL.format(p="?", user_clause=user_clause)
        with self._analytics_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {
            int(_row_value(row, "user_id")): _recent_totals_from_row(row)
            for row in rows
        }

    def list_commercial_values(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, CommercialValueFact]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        if not ids:
            return {}

        lifetime_by_user: dict[int, tuple[int, int]] = {}
        period_by_user: dict[int, tuple[int, int, int]] = {}
        usage_by_user: dict[int, int] = {}
        if hasattr(self.credits_base, "_get_connection"):
            postgres = hasattr(self.credits_base, "database_url")
            user_clause, user_params = _user_clause(ids, postgres)
            placeholder = "%s" if postgres else "?"
            window_params = [
                *user_params,
                utc_iso(window_start),
                utc_iso(window_end),
            ]
            with self.credits_base._get_connection() as conn:
                lifetime_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND entry_type IN ('purchase', 'refund')
                    GROUP BY user_id
                    """,
                    user_params,
                )
                period_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(CASE WHEN entry_type = 'purchase'
                               THEN amount_micro ELSE 0 END), 0) AS purchased_micro,
                           COALESCE(SUM(CASE WHEN entry_type = 'refund'
                               THEN -amount_micro ELSE 0 END), 0) AS refunded_micro,
                           COALESCE(SUM(CASE
                               WHEN entry_type = 'admin_grant_assign' THEN amount_micro
                               WHEN entry_type = 'admin_grant_reclaim' THEN -amount_micro
                               ELSE 0 END), 0) AS grant_activity_micro
                    FROM credit_ledger_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
                usage_rows = _fetchall(
                    conn,
                    postgres,
                    f"""
                    SELECT user_id,
                           COALESCE(SUM(-amount_micro), 0) AS consumed_micro
                    FROM credit_llm_usage_entries
                    WHERE {user_clause}
                      AND created_at >= {placeholder}
                      AND created_at < {placeholder}
                    GROUP BY user_id
                    """,
                    window_params,
                )
            lifetime_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                )
                for row in lifetime_rows
            }
            period_by_user = {
                int(_row_value(row, "user_id")): (
                    max(int(_row_value(row, "purchased_micro", 0)), 0),
                    max(int(_row_value(row, "refunded_micro", 0)), 0),
                    max(int(_row_value(row, "grant_activity_micro", 0)), 0),
                )
                for row in period_rows
            }
            usage_by_user = {
                int(_row_value(row, "user_id")): max(
                    int(_row_value(row, "consumed_micro", 0)), 0
                )
                for row in usage_rows
            }

        balances = (
            self.credits_base.get_balance_projections(ids)
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        result: dict[int, CommercialValueFact] = {}
        for user_id in ids:
            lifetime_purchased, lifetime_refunded = lifetime_by_user.get(
                user_id, (0, 0)
            )
            purchased, refunded, grant_activity = period_by_user.get(user_id, (0, 0, 0))
            net_purchased = max(lifetime_purchased - lifetime_refunded, 0)
            balance = balances.get(user_id, {})
            result[user_id] = CommercialValueFact(
                user_id=user_id,
                lifetime_net_purchased_micro=net_purchased,
                commercial_tier=commercial_tier(net_purchased),
                purchased_micro=purchased,
                refunded_micro=refunded,
                consumed_micro=usage_by_user.get(user_id, 0),
                admin_grant_activity_micro=grant_activity,
                grant_available_micro=max(
                    int(_object_value(balance, "grant_available_micro")), 0
                ),
                purchased_available_micro=max(
                    int(_object_value(balance, "purchased_available_micro")), 0
                ),
                total_available_micro=max(
                    int(_object_value(balance, "total_available_micro")), 0
                ),
            )
        return result

    def list_credit_activity(
        self,
        user_ids: Sequence[int],
        *,
        start: datetime,
        end: datetime,
    ) -> dict[int, Sequence[datetime]]:
        ids = _ids(user_ids)
        window_start, window_end = _validate_window(start, end)
        result: dict[int, list[datetime]] = {user_id: [] for user_id in ids}
        if not ids or not hasattr(self.credits_base, "_get_connection"):
            return {user_id: () for user_id in ids}

        postgres = hasattr(self.credits_base, "database_url")
        user_clause, user_params = _user_clause(ids, postgres)
        placeholder = "%s" if postgres else "?"
        params = [*user_params, utc_iso(window_start), utc_iso(window_end)]
        with self.credits_base._get_connection() as conn:
            purchase_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_ledger_entries
                WHERE {user_clause}
                  AND entry_type = 'purchase'
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
            usage_rows = _fetchall(
                conn,
                postgres,
                f"""
                SELECT user_id, created_at
                FROM credit_llm_usage_entries
                WHERE {user_clause}
                  AND created_at >= {placeholder}
                  AND created_at < {placeholder}
                """,
                params,
            )
        for row in (*purchase_rows, *usage_rows):
            user_id = int(_row_value(row, "user_id"))
            if user_id in result:
                result[user_id].append(_timestamp(_row_value(row, "created_at")))
        return {
            user_id: tuple(sorted(timestamps)) for user_id, timestamps in result.items()
        }

    def _run_health(self, user_id: int, now: datetime) -> tuple[int, bool]:
        if self.agent_base is None or self.run_base is None:
            return 0, False
        agents = self.agent_base.list_agents(owner_user_id=user_id)
        agent_ids = [str(row.get("agent_id") or "") for row in agents]
        runs = [
            run
            for agent_id in agent_ids
            if agent_id
            for run in self.run_base.list_runs(agent_id)
        ]
        ordered = sorted(
            runs,
            key=lambda run: (
                _optional_timestamp(run.get("updated_at"))
                or _optional_timestamp(run.get("created_at"))
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
            reverse=True,
        )
        terminal_24h = [
            run
            for run in ordered
            if str(run.get("status")) in _TERMINAL_RUN_STATUSES
            and (
                timestamp := (
                    _optional_timestamp(run.get("updated_at"))
                    or _optional_timestamp(run.get("created_at"))
                )
            )
            and now - timedelta(hours=24) <= timestamp <= now
        ]
        consecutive_failures = 0
        for run in terminal_24h:
            if str(run.get("status")) not in {"failed", "timed_out"}:
                break
            consecutive_failures += 1

        beyond_deadline = False
        for run in ordered:
            if str(run.get("status")) not in _ACTIVE_RUN_STATUSES:
                continue
            explicit_deadline = _optional_timestamp(run.get("deadline_at"))
            created_at = _optional_timestamp(run.get("created_at"))
            if (explicit_deadline is not None and explicit_deadline < now) or (
                explicit_deadline is None
                and created_at is not None
                and created_at + RUN_SAFE_DEADLINE < now
            ):
                beyond_deadline = True
                break
        return consecutive_failures, beyond_deadline

    def get_operational_facts(
        self,
        user_id: int,
        *,
        now: datetime,
    ) -> CurrentOperationalFacts:
        user_id = positive_user_id(user_id)
        current = _utc(now, "now")
        billing = (
            self.credits_base.get_account_billing_state(user_id)
            if hasattr(self.credits_base, "get_account_billing_state")
            else {}
        )
        balances = (
            self.credits_base.get_balance_projections([user_id])
            if hasattr(self.credits_base, "get_balance_projections")
            else {}
        )
        total_available = int(
            _object_value(balances.get(user_id, {}), "total_available_micro")
        )

        credential_status: Literal[
            "verified", "invalid", "verification_unavailable", "missing"
        ] = "missing"
        selected_provider_enabled = True
        verified_byok_lane = False
        platform_lane = total_available > 0
        if self.provider_base is not None:
            credentials = self.provider_base.list_user_credentials(user_id)
            providers = {
                str(row.get("provider_id")): row
                for row in self.provider_base.list_all_providers()
            }
            defaults = [row for row in credentials if row.get("is_default")]
            default_statuses = {str(row.get("status")) for row in defaults}
            if "invalid" in default_statuses:
                credential_status = "invalid"
            elif "verification_unavailable" in default_statuses:
                credential_status = "verification_unavailable"
            elif "verified" in default_statuses:
                credential_status = "verified"

            def provider_supports(row: Mapping[str, Any], mode: str) -> bool:
                provider = providers.get(str(row.get("provider_id")), {})
                return provider.get("status") == "enabled" and bool(
                    provider.get(f"{mode}_enabled")
                )

            selected_provider_enabled = all(
                providers.get(str(row.get("provider_id")), {}).get("status")
                == "enabled"
                for row in defaults
            )
            verified_byok_lane = any(
                row.get("status") == "verified"
                and row.get("is_default")
                and provider_supports(row, "byok")
                for row in credentials
            )
            platform_lane = total_available > 0 and any(
                row.get("status") == "enabled" and row.get("platform_enabled")
                for row in providers.values()
            )
            if hasattr(self.provider_base, "get_platform_credential_public"):
                from dashboard.backend.domain.model_providers.service import (
                    ModelProviderService,
                )

                options = ModelProviderService(
                    store=self.provider_base
                ).list_execution_options(user_id)
                verified_byok_lane = any(option.byok_available for option in options)
                platform_lane = total_available > 0 and any(
                    option.platform_credits_available for option in options
                )

        failures, beyond_deadline = self._run_health(user_id, current)
        return CurrentOperationalFacts(
            user_id=user_id,
            account_restricted=billing.get("account_status") == "restricted",
            usable_billing_lane=platform_lane or verified_byok_lane,
            selected_provider_enabled=selected_provider_enabled,
            default_credential_status=credential_status,
            failed_terminal_runs_24h=failures,
            run_beyond_safe_deadline=beyond_deadline,
        )

    def get_projection_job(self, job_name: str) -> ProjectionJob | None:
        name = _projection_job_name(job_name)
        with self._analytics_connection() as conn:
            row = conn.execute(
                "SELECT * FROM analytics_projection_jobs WHERE job_name=?",
                (name,),
            ).fetchone()
        if row is None:
            return None
        return ProjectionJob(
            job_name=_row_value(row, "job_name"),
            window_start=date.fromisoformat(str(_row_value(row, "window_start"))),
            window_end=date.fromisoformat(str(_row_value(row, "window_end"))),
            cursor=_row_value(row, "cursor"),
            status=_row_value(row, "status"),
            updated_at=_timestamp(_row_value(row, "updated_at")),
        )

    def save_projection_job(self, job: ProjectionJob) -> ProjectionJob:
        _projection_job_name(job.job_name)
        values = (
            job.job_name,
            job.window_start.isoformat(),
            job.window_end.isoformat(),
            job.cursor,
            job.status,
            utc_iso(job.updated_at),
        )
        sql = """
            INSERT INTO analytics_projection_jobs (
                job_name, window_start, window_end, cursor, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_name) DO UPDATE SET
                window_start=excluded.window_start,
                window_end=excluded.window_end,
                cursor=excluded.cursor,
                status=excluded.status,
                updated_at=excluded.updated_at
        """
        with self._analytics_connection() as conn:
            conn.execute(sql, values)
        return job


def build_value_analytics_store(
    analytics_base: Any | None = None,
    credits_base: Any | None = None,
    provider_base: Any | None = None,
    agent_base: Any | None = None,
    run_base: Any | None = None,
):
    """Pick the SQLite or PostgreSQL value-analytics twin.

    Mirrors ``repository.py``'s ``_build_analytics_store()``: the decision is
    made from the resolved ``analytics_base`` alone -- the same object every
    caller already passes, or, if none, the same ``analytics_store`` singleton
    that decision is based on. ``credits_base``/``provider_base``/
    ``agent_base``/``run_base`` are forwarded unexamined; their own dialect,
    if any, is handled inside ``list_commercial_values``/``list_credit_activity``
    on either twin, independently of this choice.
    """
    resolved_analytics_base = analytics_base or analytics_store
    if hasattr(resolved_analytics_base, "database_url"):
        from .value_repository_postgres import PostgresValueAnalyticsStore

        return PostgresValueAnalyticsStore(
            resolved_analytics_base,
            credits_base=credits_base,
            provider_base=provider_base,
            agent_base=agent_base,
            run_base=run_base,
        )
    return ValueAnalyticsStore(
        resolved_analytics_base,
        credits_base=credits_base,
        provider_base=provider_base,
        agent_base=agent_base,
        run_base=run_base,
    )


__all__ = [
    "CommercialValueFact",
    "CurrentOperationalFacts",
    "ProjectionJob",
    "RecentFactTotals",
    "UserActivity",
    "UserLifecycleDailySnapshot",
    "UserValueSnapshot",
    "ValueAnalyticsStore",
    "build_value_analytics_store",
]
