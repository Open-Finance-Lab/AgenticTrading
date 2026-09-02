"""PostgreSQL twin of the privacy-safe Analytics repository."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import psycopg

from dashboard.backend.db_url import require_postgres_url

from .models import AnalyticsEventRecord, AppendEventResult, RetentionResult
from .repository import _EVENT_COLUMNS, _event_values, _row_to_event
from .repository_common import (
    AnalyticsIdempotencyConflictError,
    AnalyticsStoreError,
    canonical_event_payload,
    decode_event_cursor,
    encode_event_cursor,
    positive_limit,
    positive_user_id,
    required_reason,
    utc_iso,
    utcnow_iso,
    validate_access_section,
)


ANALYTICS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS analytics_events (
    sequence BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) = 36),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    event_name TEXT NOT NULL CHECK (length(event_name) BETWEEN 1 AND 64),
    event_group TEXT NOT NULL
        CHECK (event_group IN ('experience', 'account', 'credential', 'agent', 'run', 'resource')),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT CHECK (session_id IS NULL OR length(session_id) = 36),
    occurred_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    event_source TEXT NOT NULL
        CHECK (event_source IN ('frontend', 'server', 'backfill')),
    source_event_id TEXT UNIQUE
        CHECK (source_event_id IS NULL OR length(source_event_id) BETWEEN 1 AND 200),
    source_record_type TEXT
        CHECK (source_record_type IS NULL OR length(source_record_type) BETWEEN 1 AND 64),
    source_record_id TEXT
        CHECK (source_record_id IS NULL OR length(source_record_id) BETWEEN 1 AND 200),
    correlation_id TEXT
        CHECK (correlation_id IS NULL OR length(correlation_id) BETWEEN 1 AND 200),
    page_view TEXT CHECK (page_view IS NULL OR length(page_view) BETWEEN 1 AND 64),
    provider_id TEXT CHECK (provider_id IS NULL OR length(provider_id) BETWEEN 1 AND 128),
    model_id TEXT CHECK (model_id IS NULL OR length(model_id) BETWEEN 1 AND 256),
    billing_mode TEXT
        CHECK (billing_mode IS NULL OR billing_mode IN ('byok', 'platform_credits')),
    outcome TEXT
        CHECK (outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')),
    error_category TEXT CHECK (
        error_category IS NULL OR error_category IN (
            'credential_invalid', 'credential_missing', 'provider_timeout',
            'provider_unavailable', 'provider_quota_exhausted',
            'credits_unavailable',
            'model_not_allowed', 'internal_error'
        )
    ),
    country_code TEXT CHECK (country_code IS NULL OR length(country_code) = 2),
    device_category TEXT CHECK (
        device_category IS NULL OR device_category IN (
            'mobile', 'tablet', 'desktop', 'unknown'
        )
    ),
    browser_family TEXT CHECK (
        browser_family IS NULL OR browser_family IN (
            'Edge', 'Chrome', 'Firefox', 'Safari', 'Other'
        )
    ),
    network_hash TEXT CHECK (network_hash IS NULL OR length(network_hash) = 64),
    properties_json TEXT NOT NULL DEFAULT '{}'
        CHECK (length(properties_json) <= 1024)
);

CREATE INDEX IF NOT EXISTS idx_analytics_events_user_time
    ON analytics_events(user_id, occurred_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_name_time
    ON analytics_events(event_name, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_events_session_time
    ON analytics_events(session_id, occurred_at DESC)
    WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_outcome_time
    ON analytics_events(outcome, occurred_at DESC)
    WHERE outcome IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analytics_events_error_time
    ON analytics_events(error_category, occurred_at DESC)
    WHERE error_category IS NOT NULL;

CREATE TABLE IF NOT EXISTS analytics_daily_rollups (
    rollup_date TEXT NOT NULL CHECK (length(rollup_date) = 10),
    metric_name TEXT NOT NULL CHECK (length(metric_name) BETWEEN 1 AND 64),
    event_name TEXT NOT NULL DEFAULT '' CHECK (length(event_name) <= 64),
    billing_mode TEXT NOT NULL DEFAULT '' CHECK (length(billing_mode) <= 32),
    provider_id TEXT NOT NULL DEFAULT '' CHECK (length(provider_id) <= 128),
    model_id TEXT NOT NULL DEFAULT '' CHECK (length(model_id) <= 256),
    outcome TEXT NOT NULL DEFAULT '' CHECK (length(outcome) <= 32),
    error_category TEXT NOT NULL DEFAULT '' CHECK (length(error_category) <= 64),
    user_state TEXT NOT NULL DEFAULT '' CHECK (length(user_state) <= 32),
    value_count BIGINT NOT NULL DEFAULT 0 CHECK (value_count >= 0),
    value_sum_micro BIGINT NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (
        rollup_date, metric_name, event_name, billing_mode, provider_id,
        model_id, outcome, error_category, user_state
    )
);

CREATE TABLE IF NOT EXISTS user_analytics_snapshots (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (
        status IN ('blocked', 'needs_attention', 'dormant', 'onboarding', 'active')
    ),
    reason_code TEXT NOT NULL CHECK (length(reason_code) BETWEEN 1 AND 100),
    human_readable_reason TEXT NOT NULL
        CHECK (length(human_readable_reason) BETWEEN 1 AND 500),
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (length(evidence_event_ids_json) <= 4096),
    calculated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analytics_subject_settings (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    excluded BOOLEAN NOT NULL,
    actor_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS admin_analytics_access_log (
    sequence BIGSERIAL PRIMARY KEY,
    admin_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    subject_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    section TEXT NOT NULL CHECK (
        section IN ('overview', 'timeline', 'runs', 'usage', 'sessions')
    ),
    accessed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_subject_time
    ON admin_analytics_access_log(subject_user_id, accessed_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_admin_time
    ON admin_analytics_access_log(admin_user_id, accessed_at DESC, sequence DESC);
"""


class PostgresAnalyticsStore:
    """Account-scoped Analytics persistence backed by PostgreSQL."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        from dashboard.backend.db_pool import get_pool

        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ANALYTICS_POSTGRES_DDL)
                cur.execute(
                    "ALTER TABLE analytics_events "
                    "DROP CONSTRAINT IF EXISTS analytics_events_error_category_check"
                )
                cur.execute(
                    """
                    ALTER TABLE analytics_events
                    ADD CONSTRAINT analytics_events_error_category_check
                    CHECK (
                        error_category IS NULL OR error_category IN (
                            'credential_invalid', 'credential_missing',
                            'provider_timeout', 'provider_unavailable',
                            'provider_quota_exhausted', 'credits_unavailable',
                            'model_not_allowed', 'internal_error'
                        )
                    )
                    """
                )

    @staticmethod
    def _select_event(cur, field: str, value: str | None):
        if value is None:
            return None
        if field not in {"event_id", "source_event_id"}:
            raise ValueError("unsupported Analytics event lookup")
        cur.execute(
            f"SELECT * FROM analytics_events WHERE {field} = %s",
            (value,),
        )
        return cur.fetchone()

    def append_event(self, event: AnalyticsEventRecord) -> AppendEventResult:
        if not isinstance(event, AnalyticsEventRecord):
            event = AnalyticsEventRecord.model_validate(event)
        columns = ", ".join(_EVENT_COLUMNS)
        placeholders = ", ".join("%s" for _ in _EVENT_COLUMNS)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        f"""
                        INSERT INTO analytics_events ({columns})
                        VALUES ({placeholders})
                        ON CONFLICT DO NOTHING
                        RETURNING *
                        """,
                        _event_values(event),
                    )
                except psycopg.IntegrityError:
                    raise AnalyticsStoreError(
                        "Analytics event could not be persisted"
                    ) from None
                created_row = cur.fetchone()
                if created_row is not None:
                    return AppendEventResult(
                        event=_row_to_event(created_row),
                        created=True,
                    )

                by_event = self._select_event(cur, "event_id", event.event_id)
                by_source = self._select_event(
                    cur,
                    "source_event_id",
                    event.source_event_id,
                )
                if (
                    by_event is not None
                    and by_source is not None
                    and int(by_event["sequence"]) != int(by_source["sequence"])
                ):
                    raise AnalyticsIdempotencyConflictError(
                        "analytics event idempotency conflict"
                    )
                existing_row = by_source or by_event
                if existing_row is None:
                    raise AnalyticsStoreError(
                        "Analytics event could not be persisted"
                    )
                existing = _row_to_event(existing_row)
                ignore_event_id = by_source is not None
                if canonical_event_payload(
                    existing,
                    ignore_event_id=ignore_event_id,
                ) != canonical_event_payload(
                    event,
                    ignore_event_id=ignore_event_id,
                ):
                    raise AnalyticsIdempotencyConflictError(
                        "analytics event idempotency conflict"
                    )
                return AppendEventResult(event=existing, created=False)

    def get_event(self, event_id: str) -> AnalyticsEventRecord | None:
        if not isinstance(event_id, str) or len(event_id) != 36:
            raise ValueError("event_id must be a canonical UUID")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_events WHERE event_id = %s",
                    (event_id,),
                )
                row = cur.fetchone()
        return _row_to_event(row) if row is not None else None

    def list_user_events(
        self,
        user_id: int,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        subject_id = positive_user_id(user_id)
        page_size = positive_limit(limit)
        params: list[object] = [subject_id]
        cursor_sql = ""
        if cursor is not None:
            occurred_at, sequence = decode_event_cursor(cursor)
            cursor_sql = "AND (occurred_at, sequence) < (%s, %s)"
            params.extend([occurred_at, sequence])
        params.append(page_size + 1)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT * FROM analytics_events
                    WHERE user_id = %s {cursor_sql}
                    ORDER BY occurred_at DESC, sequence DESC
                    LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        has_more = len(rows) > page_size
        page_rows = rows[:page_size]
        next_cursor = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_event_cursor(
                str(last["occurred_at"]),
                int(last["sequence"]),
            )
        return {
            "items": [_row_to_event(row) for row in page_rows],
            "next_cursor": next_cursor,
        }

    def set_subject_exclusion(
        self,
        user_id: int,
        *,
        excluded: bool,
        actor_user_id: int,
        reason: str,
    ) -> dict[str, Any]:
        subject_id = positive_user_id(user_id)
        actor_id = positive_user_id(actor_user_id, "actor_user_id")
        if not isinstance(excluded, bool):
            raise ValueError("excluded must be a boolean")
        safe_reason = required_reason(reason)
        now = utcnow_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO analytics_subject_settings (
                            user_id, excluded, actor_user_id, reason,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT(user_id) DO UPDATE SET
                            excluded = EXCLUDED.excluded,
                            actor_user_id = EXCLUDED.actor_user_id,
                            reason = EXCLUDED.reason,
                            updated_at = EXCLUDED.updated_at
                        RETURNING *
                        """,
                        (subject_id, excluded, actor_id, safe_reason, now, now),
                    )
                except psycopg.IntegrityError:
                    raise AnalyticsStoreError(
                        "Analytics subject setting could not be persisted"
                    ) from None
                row = cur.fetchone()
        if row is None:
            raise AnalyticsStoreError("Analytics subject setting could not be loaded")
        result = dict(row)
        result["excluded"] = bool(result["excluded"])
        return result

    def get_subject_setting(self, user_id: int) -> dict[str, Any] | None:
        subject_id = positive_user_id(user_id)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM analytics_subject_settings WHERE user_id = %s",
                    (subject_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        result = dict(row)
        result["excluded"] = bool(result["excluded"])
        return result

    def list_excluded_user_ids(
        self,
        *,
        include_admin_accounts: bool = True,
    ) -> set[int]:
        if not isinstance(include_admin_accounts, bool):
            raise ValueError("include_admin_accounts must be a boolean")
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM analytics_subject_settings WHERE excluded = TRUE"
                )
                excluded = {int(row["user_id"]) for row in cur.fetchall()}
                if include_admin_accounts:
                    cur.execute("SELECT id FROM users WHERE role = 'admin'")
                    excluded.update(int(row["id"]) for row in cur.fetchall())
        return excluded

    def record_admin_access(
        self,
        admin_user_id: int,
        subject_user_id: int,
        section: str,
    ) -> dict[str, Any]:
        admin_id = positive_user_id(admin_user_id, "admin_user_id")
        subject_id = positive_user_id(subject_user_id, "subject_user_id")
        safe_section = validate_access_section(section)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO admin_analytics_access_log (
                            admin_user_id, subject_user_id, section, accessed_at
                        ) VALUES (%s, %s, %s, %s)
                        RETURNING *
                        """,
                        (admin_id, subject_id, safe_section, utcnow_iso()),
                    )
                except psycopg.IntegrityError:
                    raise AnalyticsStoreError(
                        "Admin Analytics access could not be recorded"
                    ) from None
                row = cur.fetchone()
        if row is None:
            raise AnalyticsStoreError("Admin Analytics access could not be loaded")
        return dict(row)

    def list_admin_access(
        self,
        subject_user_id: int,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        subject_id = positive_user_id(subject_user_id, "subject_user_id")
        page_size = positive_limit(limit)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM admin_analytics_access_log
                    WHERE subject_user_id = %s
                    ORDER BY accessed_at DESC, sequence DESC
                    LIMIT %s
                    """,
                    (subject_id, page_size),
                )
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def delete_expired(
        self,
        *,
        raw_before: datetime,
        access_before: datetime,
        batch_size: int,
    ) -> RetentionResult:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 10_000
        ):
            raise ValueError("batch_size must be an integer from 1 through 10000")
        raw_cutoff = utc_iso(raw_before)
        access_cutoff = utc_iso(access_before)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH doomed AS (
                        SELECT sequence FROM analytics_events
                        WHERE received_at < %s
                        ORDER BY received_at, sequence
                        LIMIT %s
                    )
                    DELETE FROM analytics_events AS events
                    USING doomed
                    WHERE events.sequence = doomed.sequence
                    RETURNING events.sequence
                    """,
                    (raw_cutoff, batch_size),
                )
                raw_deleted = len(cur.fetchall())
                cur.execute(
                    """
                    WITH doomed AS (
                        SELECT sequence FROM admin_analytics_access_log
                        WHERE accessed_at < %s
                        ORDER BY accessed_at, sequence
                        LIMIT %s
                    )
                    DELETE FROM admin_analytics_access_log AS access_log
                    USING doomed
                    WHERE access_log.sequence = doomed.sequence
                    RETURNING access_log.sequence
                    """,
                    (access_cutoff, batch_size),
                )
                access_deleted = len(cur.fetchall())
                cur.execute(
                    "SELECT 1 FROM analytics_events WHERE received_at < %s LIMIT 1",
                    (raw_cutoff,),
                )
                has_more_raw = cur.fetchone() is not None
                cur.execute(
                    """
                    SELECT 1 FROM admin_analytics_access_log
                    WHERE accessed_at < %s LIMIT 1
                    """,
                    (access_cutoff,),
                )
                has_more_access = cur.fetchone() is not None
        return RetentionResult(
            raw_events_deleted=raw_deleted,
            access_rows_deleted=access_deleted,
            has_more_raw_events=has_more_raw,
            has_more_access_rows=has_more_access,
        )
