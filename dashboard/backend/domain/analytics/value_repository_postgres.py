"""PostgreSQL twin of the user-value Analytics projection store.

See ``value_repository.py`` for the SQLite twin and its module docstring,
which explains which methods branch on this class's own dialect (all
rewritten below, one path each) versus a *different* store's dialect
(``list_commercial_values``, ``list_credit_activity`` -- unchanged here,
and identical to the SQLite twin, because that branch was never about
which twin this class is).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence

from .repository_common import positive_limit, positive_user_id, utc_iso
from .value_repository import (
    _ACTIVE_RUN_STATUSES,
    _TERMINAL_RUN_STATUSES,
    _activity_from_row,
    LIFECYCLE_ROLLUP_METRICS,
    LIFECYCLE_SEGMENTS,
    MAX_USER_BATCH,
    RUN_SAFE_DEADLINE,
    CommercialValueFact,
    CurrentOperationalFacts,
    ProjectionJob,
    UserActivity,
    UserLifecycleDailySnapshot,
    UserValueSnapshot,
    _current_snapshot_from_row,
    _fetchall,
    _ids,
    _legacy_seed,
    _object_value,
    _optional_timestamp,
    _projection_job_name,
    _row_value,
    _timestamp,
    _user_clause,
    _utc,
    _validate_window,
    analytics_store,
    commercial_tier,
)


class PostgresValueAnalyticsStore:
    """PostgreSQL value projection storage.

    See ``value_repository.py`` for the SQLite twin;
    ``build_value_analytics_store()`` there picks between them.
    """

    def __init__(
        self,
        analytics_base: Any | None = None,
        credits_base: Any | None = None,
        provider_base: Any | None = None,
        agent_base: Any | None = None,
        run_base: Any | None = None,
    ) -> None:
        # `or analytics_store` keeps this signature identical to the SQLite
        # twin's. The parity tests do pin the signatures, but they compare
        # *public* methods only -- both build their name list from `dir(cls)`
        # and skip `name.startswith("_")` -- so neither sees `__init__`, and a
        # pinned signature would say nothing about this body in any case.
        # Unguarded, the fallback resolves to whichever dialect the module
        # singleton happens to be: Postgres on prod, SQLite locally and under
        # pytest, where every `%s` query below would reach sqlite3 and fail at
        # the first cursor rather than here. Hence the explicit check.
        self.analytics_base = analytics_base or analytics_store
        if not hasattr(self.analytics_base, "database_url"):
            raise TypeError(
                "PostgresValueAnalyticsStore requires a PostgreSQL analytics "
                "base, but the resolved base exposes no database_url. Build "
                "the pair through build_value_analytics_store(), which "
                "resolves the base and returns the matching twin."
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
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO user_analytics_snapshots ({columns})
                    VALUES ({", ".join(["%s"] * len(values))})
                    ON CONFLICT(user_id) DO UPDATE SET {updates}
                    """,
                    values,
                )
        return snapshot

    def get_current_snapshot(self, user_id: int) -> UserValueSnapshot | None:
        subject = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_analytics_snapshots WHERE user_id=%s",
                    (subject,),
                )
                row = cur.fetchone()
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
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT *
                        FROM user_analytics_snapshots
                        WHERE user_id = ANY(%s) AND lifecycle_segment IS NOT NULL
                        ORDER BY user_id
                        """,
                        [chunk],
                    )
                    rows = cur.fetchall()
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
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(snapshot_date, user_id) DO UPDATE SET
                lifecycle_segment=excluded.lifecycle_segment,
                lifecycle_reason_code=excluded.lifecycle_reason_code,
                data_quality=excluded.data_quality,
                calculated_at=excluded.calculated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
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
            clause = " AND user_id = ANY(%s)"
            params.append(ids)
        sql = f"""
            SELECT *
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date >= %s
              AND snapshot_date < %s
              {clause}
            ORDER BY snapshot_date, user_id
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
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
        placeholders = ", ".join(["%s"] * len(columns))
        metrics = ["lifecycle_segment_count"]
        if replace_transitions:
            metrics.append("lifecycle_transition")
        metric_placeholders = ", ".join(["%s"] * len(metrics))
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM analytics_daily_rollups
                    WHERE rollup_date = %s
                      AND metric_name IN ({metric_placeholders})
                    """,
                    (day.isoformat(), *metrics),
                )
                if payloads:
                    cur.executemany(
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
            WHERE snapshot_date < %s
            ORDER BY snapshot_date
            LIMIT %s
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (before.isoformat(), page_size))
                rows = cur.fetchall()
        return [
            date.fromisoformat(str(_row_value(row, "snapshot_date"))) for row in rows
        ]

    def delete_daily_snapshots_for_date(self, day: date) -> int:
        if not isinstance(day, date) or isinstance(day, datetime):
            raise ValueError("day must be a date")
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM user_lifecycle_daily_snapshots
                    WHERE snapshot_date = %s
                    RETURNING user_id
                    """,
                    (day.isoformat(),),
                )
                return len(cur.fetchall())

    def has_daily_before(self, before: date) -> bool:
        if not isinstance(before, date) or isinstance(before, datetime):
            raise ValueError("before must be a date")
        sql = """
            SELECT 1
            FROM user_lifecycle_daily_snapshots
            WHERE snapshot_date < %s
            LIMIT 1
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (before.isoformat(),))
                row = cur.fetchone()
        return row is not None

    def record_activity(
        self,
        user_id: int,
        *,
        occurred_at: datetime,
        activating: bool,
        now: datetime,
    ) -> None:
        """See the SQLite twin. Postgres ``LEAST``/``GREATEST`` skip NULLs, so
        the COALESCE pair is redundant here but harmless; keeping both dialects
        spelled the same way is worth more than saving two lines. ``%s::text``
        on the nullable ``activated_at`` gives psycopg a type for the ``None``
        it would otherwise send as OID 0.
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
            ) VALUES (%s, %s::text, %s, %s)
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = LEAST(
                    COALESCE(user_activity.activated_at, EXCLUDED.activated_at),
                    COALESCE(EXCLUDED.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = GREATEST(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    EXCLUDED.last_meaningful_activity_at
                ),
                updated_at = EXCLUDED.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)

    def get_activity(self, user_id: int) -> UserActivity | None:
        """One user's stored activity row, or None if they have none yet."""
        subject_id = positive_user_id(user_id)
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_activity WHERE user_id = %s", (subject_id,)
                )
                row = cur.fetchone()
        return _activity_from_row(row) if row is not None else None

    def list_activity(
        self,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, UserActivity]:
        """See the SQLite twin."""
        result: dict[int, UserActivity] = {}
        if user_ids is None:
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM user_activity ORDER BY user_id")
                    rows = cur.fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
            return result
        ids = _ids(user_ids)
        if not ids:
            return {}
        for offset in range(0, len(ids), MAX_USER_BATCH):
            chunk = ids[offset : offset + MAX_USER_BATCH]
            with self._analytics_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM user_activity WHERE user_id = ANY(%s) "
                        "ORDER BY user_id",
                        (chunk,),
                    )
                    rows = cur.fetchall()
            for row in rows:
                activity = _activity_from_row(row)
                result[activity.user_id] = activity
        return result

    def seed_activity_from_snapshots(self, *, now: datetime) -> int:
        """See the SQLite twin."""
        stamp = utc_iso(_utc(now, "now"))
        sql = """
            INSERT INTO user_activity (
                user_id, activated_at, last_meaningful_activity_at, updated_at
            )
            SELECT user_id, activated_at, last_meaningful_activity_at, %s
            FROM user_analytics_snapshots
            WHERE activated_at IS NOT NULL
               OR last_meaningful_activity_at IS NOT NULL
            ON CONFLICT(user_id) DO UPDATE SET
                activated_at = LEAST(
                    COALESCE(user_activity.activated_at, EXCLUDED.activated_at),
                    COALESCE(EXCLUDED.activated_at, user_activity.activated_at)
                ),
                last_meaningful_activity_at = GREATEST(
                    COALESCE(
                        user_activity.last_meaningful_activity_at,
                        EXCLUDED.last_meaningful_activity_at
                    ),
                    COALESCE(
                        EXCLUDED.last_meaningful_activity_at,
                        user_activity.last_meaningful_activity_at
                    )
                ),
                updated_at = EXCLUDED.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (stamp,))
                return max(0, int(cur.rowcount))

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
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_projection_jobs WHERE job_name=%s",
                    (name,),
                )
                row = cur.fetchone()
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
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(job_name) DO UPDATE SET
                window_start=excluded.window_start,
                window_end=excluded.window_end,
                cursor=excluded.cursor,
                status=excluded.status,
                updated_at=excluded.updated_at
        """
        with self._analytics_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
        return job
