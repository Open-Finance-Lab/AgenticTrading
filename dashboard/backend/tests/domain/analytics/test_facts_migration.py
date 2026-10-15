"""The last read of the five-state snapshot table, then its drop (design SS11)."""

from __future__ import annotations

import inspect
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics import repository, repository_postgres
from dashboard.backend.domain.analytics.facts_migration import (
    StartupMigrationReport,
    run_startup_migrations,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)
ACTIVATED_AT = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
LAST_ACTIVITY_AT = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
LEGACY_TABLES = ("user_analytics_snapshots", "user_lifecycle_daily_snapshots")


def create_legacy_snapshot_tables(analytics, rows=()) -> None:
    """Recreate the legacy tables the DDL no longer declares, as prod still has them.

    ``rows`` holds ``(user_id, activated_at, last_meaningful_activity_at)``.
    Only the columns the seed reads plus the NOT NULL ones the five-state
    writer filled are declared; the seed never reads anything else.
    """
    with analytics._get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE user_analytics_snapshots (
                user_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                human_readable_reason TEXT NOT NULL,
                evidence_event_ids_json TEXT NOT NULL DEFAULT '[]',
                calculated_at TEXT NOT NULL,
                activated_at TEXT,
                last_meaningful_activity_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE user_lifecycle_daily_snapshots (
                snapshot_date TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                PRIMARY KEY (snapshot_date, user_id)
            )
            """
        )
        for user_id, activated_at, last_activity_at in rows:
            conn.execute(
                "INSERT INTO user_analytics_snapshots (user_id, status, reason_code, "
                "human_readable_reason, calculated_at, activated_at, "
                "last_meaningful_activity_at) VALUES (?, 'active', 'x', 'x', ?, ?, ?)",
                (
                    user_id,
                    NOW.isoformat(),
                    activated_at.isoformat() if activated_at else None,
                    last_activity_at.isoformat() if last_activity_at else None,
                ),
            )


def _table_names(analytics) -> set[str]:
    with analytics._get_connection() as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def _fixture(tmp_path):
    path = tmp_path / "migration.db"
    users = UserStore(db_path=path)
    user_id = int(users.create_user("m@example.test", "M", "SecurePass1!")["id"])
    analytics = AnalyticsStore(db_path=path)
    value_store = build_value_analytics_store(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    return user_id, analytics, value_store


def test_reseed_then_drop_carries_activation_across_and_removes_the_tables(tmp_path):
    """Design SS11: the seed runs again immediately before the drop, because the
    legacy row kept being maintained between PR A and PR B."""
    user_id, analytics, value_store = _fixture(tmp_path)
    create_legacy_snapshot_tables(
        analytics, [(user_id, ACTIVATED_AT, LAST_ACTIVITY_AT)]
    )

    report = run_startup_migrations(value_store=value_store, now=NOW)

    assert report == StartupMigrationReport(activity_seeded=1)
    activity = value_store.get_activity(user_id)
    assert activity.activated_at == ACTIVATED_AT
    assert activity.last_meaningful_activity_at == LAST_ACTIVITY_AT
    assert not set(LEGACY_TABLES) & _table_names(analytics)


def test_second_boot_after_the_drop_is_a_clean_no_op(tmp_path):
    """run_startup_migrations runs twice per boot (app startup and the worker's
    prepare) and on every boot after: once the table is gone the seed reads
    nothing and the drop is IF EXISTS."""
    user_id, analytics, value_store = _fixture(tmp_path)
    create_legacy_snapshot_tables(
        analytics, [(user_id, ACTIVATED_AT, LAST_ACTIVITY_AT)]
    )
    run_startup_migrations(value_store=value_store, now=NOW)

    again = run_startup_migrations(value_store=value_store, now=NOW + timedelta(hours=1))
    analytics.drop_legacy_snapshot_tables()

    assert again == StartupMigrationReport(activity_seeded=0)
    assert value_store.seed_activity_from_snapshots(now=NOW) == 0
    assert value_store.get_activity(user_id).activated_at == ACTIVATED_AT


def test_fresh_install_boots_without_the_legacy_tables(tmp_path):
    _user_id, analytics, value_store = _fixture(tmp_path)

    assert not set(LEGACY_TABLES) & _table_names(analytics)
    assert run_startup_migrations(value_store=value_store, now=NOW) == (
        StartupMigrationReport(activity_seeded=0)
    )


def test_seed_still_raises_on_errors_other_than_a_missing_table(tmp_path):
    """Only the dropped table is tolerated; a broken legacy row must stay loud."""
    _user_id, analytics, value_store = _fixture(tmp_path)
    with analytics._get_connection() as conn:
        # The table exists but lacks the columns the seed reads.
        conn.execute("CREATE TABLE user_analytics_snapshots (user_id INTEGER PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        value_store.seed_activity_from_snapshots(now=NOW)


def test_fresh_schema_declares_no_snapshot_tables():
    for ddl in (repository.ANALYTICS_SQLITE_DDL, repository_postgres.ANALYTICS_POSTGRES_DDL):
        for table in LEGACY_TABLES:
            assert table not in ddl
    for init_schema in (
        repository.AnalyticsStore._init_schema,
        repository_postgres.PostgresAnalyticsStore._init_schema,
    ):
        assert "user_analytics_snapshots" not in inspect.getsource(init_schema)
