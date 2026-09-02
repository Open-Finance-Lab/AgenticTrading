"""SQLite persistence for privacy-safe first-party Analytics data."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from dashboard.backend.database import DB_PATH
from dashboard.backend.db_url import describe_database_url

from .models import AnalyticsEventRecord, AppendEventResult, RetentionResult
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


ANALYTICS_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS analytics_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE CHECK (length(event_id) = 36),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    event_name TEXT NOT NULL CHECK (length(event_name) BETWEEN 1 AND 64),
    event_group TEXT NOT NULL
        CHECK (event_group IN ('experience', 'account', 'credential', 'agent', 'run', 'resource')),
    user_id INTEGER NOT NULL,
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
        CHECK (length(properties_json) <= 1024),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
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
    value_count INTEGER NOT NULL DEFAULT 0 CHECK (value_count >= 0),
    value_sum_micro INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (
        rollup_date, metric_name, event_name, billing_mode, provider_id,
        model_id, outcome, error_category, user_state
    )
);

CREATE TABLE IF NOT EXISTS user_analytics_snapshots (
    user_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('blocked', 'needs_attention', 'dormant', 'onboarding', 'active')
    ),
    reason_code TEXT NOT NULL CHECK (length(reason_code) BETWEEN 1 AND 100),
    human_readable_reason TEXT NOT NULL
        CHECK (length(human_readable_reason) BETWEEN 1 AND 500),
    evidence_event_ids_json TEXT NOT NULL DEFAULT '[]'
        CHECK (length(evidence_event_ids_json) <= 4096),
    calculated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analytics_subject_settings (
    user_id INTEGER PRIMARY KEY,
    excluded INTEGER NOT NULL CHECK (excluded IN (0, 1)),
    actor_user_id INTEGER NOT NULL,
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS admin_analytics_access_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_user_id INTEGER NOT NULL,
    subject_user_id INTEGER NOT NULL,
    section TEXT NOT NULL CHECK (
        section IN ('overview', 'timeline', 'runs', 'usage', 'sessions')
    ),
    accessed_at TEXT NOT NULL,
    FOREIGN KEY (admin_user_id) REFERENCES users(id) ON DELETE RESTRICT,
    FOREIGN KEY (subject_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_subject_time
    ON admin_analytics_access_log(subject_user_id, accessed_at DESC, sequence DESC);
CREATE INDEX IF NOT EXISTS idx_admin_analytics_access_admin_time
    ON admin_analytics_access_log(admin_user_id, accessed_at DESC, sequence DESC);
"""


_EVENT_COLUMNS = (
    "event_id",
    "schema_version",
    "event_name",
    "event_group",
    "user_id",
    "session_id",
    "occurred_at",
    "received_at",
    "event_source",
    "source_event_id",
    "source_record_type",
    "source_record_id",
    "correlation_id",
    "page_view",
    "provider_id",
    "model_id",
    "billing_mode",
    "outcome",
    "error_category",
    "country_code",
    "device_category",
    "browser_family",
    "network_hash",
    "properties_json",
)


def _event_values(event: AnalyticsEventRecord) -> tuple[object, ...]:
    data = event.model_dump(mode="json")
    data["occurred_at"] = utc_iso(event.occurred_at)
    data["received_at"] = utc_iso(event.received_at)
    data["properties_json"] = json.dumps(
        event.properties,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return tuple(data.get(column) for column in _EVENT_COLUMNS)


def _row_to_event(row: sqlite3.Row | dict[str, Any]) -> AnalyticsEventRecord:
    data = dict(row)
    data.pop("sequence", None)
    raw_properties = data.pop("properties_json", "{}")
    try:
        properties = json.loads(raw_properties)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AnalyticsStoreError("stored Analytics properties are invalid") from exc
    data["properties"] = properties
    return AnalyticsEventRecord.model_validate(data)


class AnalyticsStore:
    """Account-scoped Analytics persistence backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(ANALYTICS_SQLITE_DDL)
            self._migrate_error_category_constraint(conn)

    @staticmethod
    def _migrate_error_category_constraint(conn: sqlite3.Connection) -> None:
        """Rebuild the events table when upgrading its closed category check."""

        users_table = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'users'"
        ).fetchone()
        if users_table is None:
            return

        row = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'analytics_events'"
        ).fetchone()
        table_sql = str(row[0] or "").lower() if row else ""
        if "provider_quota_exhausted" in table_sql:
            return

        # Index names are schema-global, so remove the old table's indexes
        # before creating the replacement table and its indexes.
        for index_name in (
            "idx_analytics_events_user_time",
            "idx_analytics_events_name_time",
            "idx_analytics_events_session_time",
            "idx_analytics_events_outcome_time",
            "idx_analytics_events_error_time",
        ):
            conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        conn.execute(
            "ALTER TABLE analytics_events "
            "RENAME TO analytics_events_migration_legacy"
        )
        conn.executescript(ANALYTICS_SQLITE_DDL)
        columns = (
            "sequence, "
            + ", ".join(_EVENT_COLUMNS)
            + ", country_code, device_category, browser_family, "
            "network_hash, properties_json"
        )
        conn.execute(
            f"INSERT INTO analytics_events ({columns}) "
            f"SELECT {columns} FROM analytics_events_migration_legacy"
        )
        conn.execute("DROP TABLE analytics_events_migration_legacy")

    @staticmethod
    def _existing_rows_for_event(
        conn: sqlite3.Connection,
        event: AnalyticsEventRecord,
    ) -> tuple[sqlite3.Row | None, sqlite3.Row | None]:
        by_event = conn.execute(
            "SELECT * FROM analytics_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        by_source = None
        if event.source_event_id is not None:
            by_source = conn.execute(
                "SELECT * FROM analytics_events WHERE source_event_id = ?",
                (event.source_event_id,),
            ).fetchone()
        return by_event, by_source

    def append_event(self, event: AnalyticsEventRecord) -> AppendEventResult:
        if not isinstance(event, AnalyticsEventRecord):
            event = AnalyticsEventRecord.model_validate(event)
        placeholders = ", ".join("?" for _ in _EVENT_COLUMNS)
        columns = ", ".join(_EVENT_COLUMNS)
        with self._get_connection() as conn:
            self._migrate_error_category_constraint(conn)
            try:
                cursor = conn.execute(
                    f"INSERT INTO analytics_events ({columns}) VALUES ({placeholders})",
                    _event_values(event),
                )
            except sqlite3.IntegrityError as exc:
                by_event, by_source = self._existing_rows_for_event(conn, event)
                if (
                    by_event is not None
                    and by_source is not None
                    and int(by_event["sequence"]) != int(by_source["sequence"])
                ):
                    raise AnalyticsIdempotencyConflictError(
                        "analytics event idempotency conflict"
                    ) from None
                existing_row = by_source or by_event
                if existing_row is None:
                    raise AnalyticsStoreError(
                        "Analytics event could not be persisted"
                    ) from None
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
                    ) from None
                return AppendEventResult(event=existing, created=False)

            row = conn.execute(
                "SELECT * FROM analytics_events WHERE sequence = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            if row is None:
                raise AnalyticsStoreError("Analytics event could not be loaded")
            return AppendEventResult(event=_row_to_event(row), created=True)

    def get_event(self, event_id: str) -> AnalyticsEventRecord | None:
        if not isinstance(event_id, str) or len(event_id) != 36:
            raise ValueError("event_id must be a canonical UUID")
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM analytics_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
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
            cursor_sql = (
                "AND (occurred_at < ? OR (occurred_at = ? AND sequence < ?))"
            )
            params.extend([occurred_at, occurred_at, sequence])
        params.append(page_size + 1)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM analytics_events
                WHERE user_id = ? {cursor_sql}
                ORDER BY occurred_at DESC, sequence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
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
            try:
                conn.execute(
                    """
                    INSERT INTO analytics_subject_settings (
                        user_id, excluded, actor_user_id, reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        excluded = excluded.excluded,
                        actor_user_id = excluded.actor_user_id,
                        reason = excluded.reason,
                        updated_at = excluded.updated_at
                    """,
                    (subject_id, int(excluded), actor_id, safe_reason, now, now),
                )
            except sqlite3.IntegrityError:
                raise AnalyticsStoreError(
                    "Analytics subject setting could not be persisted"
                ) from None
            row = conn.execute(
                "SELECT * FROM analytics_subject_settings WHERE user_id = ?",
                (subject_id,),
            ).fetchone()
        if row is None:
            raise AnalyticsStoreError("Analytics subject setting could not be loaded")
        result = dict(row)
        result["excluded"] = bool(result["excluded"])
        return result

    def get_subject_setting(self, user_id: int) -> dict[str, Any] | None:
        subject_id = positive_user_id(user_id)
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM analytics_subject_settings WHERE user_id = ?",
                (subject_id,),
            ).fetchone()
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
            rows = conn.execute(
                "SELECT user_id FROM analytics_subject_settings WHERE excluded = 1"
            ).fetchall()
            excluded = {int(row["user_id"]) for row in rows}
            if include_admin_accounts:
                admin_rows = conn.execute(
                    "SELECT id FROM users WHERE role = 'admin'"
                ).fetchall()
                excluded.update(int(row["id"]) for row in admin_rows)
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
        accessed_at = utcnow_iso()
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO admin_analytics_access_log (
                        admin_user_id, subject_user_id, section, accessed_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (admin_id, subject_id, safe_section, accessed_at),
                )
            except sqlite3.IntegrityError:
                raise AnalyticsStoreError(
                    "Admin Analytics access could not be recorded"
                ) from None
            row = conn.execute(
                "SELECT * FROM admin_analytics_access_log WHERE sequence = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
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
            rows = conn.execute(
                """
                SELECT * FROM admin_analytics_access_log
                WHERE subject_user_id = ?
                ORDER BY accessed_at DESC, sequence DESC
                LIMIT ?
                """,
                (subject_id, page_size),
            ).fetchall()
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
            raw_cursor = conn.execute(
                """
                DELETE FROM analytics_events
                WHERE sequence IN (
                    SELECT sequence FROM analytics_events
                    WHERE received_at < ?
                    ORDER BY received_at, sequence
                    LIMIT ?
                )
                """,
                (raw_cutoff, batch_size),
            )
            access_cursor = conn.execute(
                """
                DELETE FROM admin_analytics_access_log
                WHERE sequence IN (
                    SELECT sequence FROM admin_analytics_access_log
                    WHERE accessed_at < ?
                    ORDER BY accessed_at, sequence
                    LIMIT ?
                )
                """,
                (access_cutoff, batch_size),
            )
            has_more_raw = conn.execute(
                "SELECT 1 FROM analytics_events WHERE received_at < ? LIMIT 1",
                (raw_cutoff,),
            ).fetchone()
            has_more_access = conn.execute(
                "SELECT 1 FROM admin_analytics_access_log WHERE accessed_at < ? LIMIT 1",
                (access_cutoff,),
            ).fetchone()
        return RetentionResult(
            raw_events_deleted=max(0, int(raw_cursor.rowcount)),
            access_rows_deleted=max(0, int(access_cursor.rowcount)),
            has_more_raw_events=has_more_raw is not None,
            has_more_access_rows=has_more_access is not None,
        )


def _build_analytics_store():
    """Bind Analytics to the account database and no other persistence plane."""

    database_url = (os.getenv("USERS_DATABASE_URL") or "").strip()
    if database_url:
        from .repository_postgres import PostgresAnalyticsStore

        print(
            "analytics_store backend: postgres "
            f"({describe_database_url(database_url)})"
        )
        return PostgresAnalyticsStore(database_url)
    print("analytics_store backend: sqlite (ephemeral on Render)")
    return AnalyticsStore()


analytics_store = _build_analytics_store()
