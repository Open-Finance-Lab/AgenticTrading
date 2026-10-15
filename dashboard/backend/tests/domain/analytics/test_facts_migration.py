"""The last read of the five-state snapshot table, then its drop (design SS11)."""

from __future__ import annotations

import inspect
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics import repository, repository_postgres
from dashboard.backend.domain.analytics.facts_migration import (
    StartupMigrationReport,
    run_startup_migrations,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.rollups import AnalyticsRollupStore, DailyRollup
from dashboard.backend.domain.analytics.value_repository import (
    UserDailyFact,
    build_value_analytics_store,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 12, 0, 5, tzinfo=timezone.utc)
ACTIVATED_AT = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
LAST_ACTIVITY_AT = datetime(2026, 9, 11, 18, 0, tzinfo=timezone.utc)
LEGACY_TABLES = ("user_analytics_snapshots", "user_lifecycle_daily_snapshots")
# The legacy daily table as main declared it (minus the users FK).
LEGACY_DAILY_DDL = """
    CREATE TABLE user_lifecycle_daily_snapshots (
        snapshot_date TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        lifecycle_segment TEXT NOT NULL,
        lifecycle_reason_code TEXT NOT NULL,
        data_quality TEXT NOT NULL,
        calculated_at TEXT NOT NULL,
        PRIMARY KEY (snapshot_date, user_id)
    )
"""
# Two uncovered legacy days (07-11, 07-12) and one the facts already cover
# (07-13). ``(snapshot_date, user_id, lifecycle_segment, data_quality)``.
UNCOVERED_DAYS = (date(2026, 7, 11), date(2026, 7, 12))
COVERED_DAY = date(2026, 7, 13)
LEGACY_DAILY_ROWS = (
    ("2026-07-11", 101, "new", "complete"),
    ("2026-07-11", 102, "core", "complete"),
    ("2026-07-12", 101, "growing", "complete"),
    ("2026-07-12", 102, "core", "complete"),
    ("2026-07-12", 103, "new", "partial"),
    ("2026-07-13", 101, "core", "complete"),
)


def legacy_lifecycle_rollups(day: date) -> list[tuple]:
    """What main's ``rollup_lifecycle_day`` wrote for the fixture's two days."""
    return {
        UNCOVERED_DAYS[0]: [
            ("lifecycle_segment_count", "", "core", 1, "complete"),
            ("lifecycle_segment_count", "", "new", 1, "complete"),
        ],
        UNCOVERED_DAYS[1]: [
            ("lifecycle_segment_count", "", "core", 1, "complete"),
            ("lifecycle_segment_count", "", "growing", 1, "complete"),
            ("lifecycle_segment_count", "", "new", 1, "partial"),
            ("lifecycle_transition", "new", "growing", 1, "complete"),
        ],
    }[day]


def _rollup_shape(rows) -> list[tuple]:
    return sorted(
        (row.metric_name, row.event_name, row.user_state, row.value_count, row.outcome)
        for row in rows
    )


def covering_fact(user_id: int, day: date = COVERED_DAY) -> UserDailyFact:
    return UserDailyFact(
        snapshot_date=day,
        user_id=user_id,
        lifecycle_segment="core",
        lifecycle_reason_code="x",
        operational_state="healthy",
        tier="unpaid",
        user_group="organic",
        data_quality="complete",
        calculated_at=NOW,
    )


def create_legacy_snapshot_tables(analytics, rows=(), daily_rows=()) -> None:
    """Recreate the legacy tables the DDL no longer declares, as prod still has them.

    ``rows`` holds ``(user_id, activated_at, last_meaningful_activity_at)``.
    Only the columns the seed reads plus the NOT NULL ones the five-state
    writer filled are declared; the seed never reads anything else. The daily
    table carries main's columns, since the legacy-day rollup reads them.
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
        conn.execute(LEGACY_DAILY_DDL)
        for snapshot_date, user_id, segment, quality in daily_rows:
            conn.execute(
                "INSERT INTO user_lifecycle_daily_snapshots VALUES (?, ?, ?, 'x', ?, ?)",
                (snapshot_date, user_id, segment, quality, NOW.isoformat()),
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


def test_a_failed_seed_keeps_both_legacy_tables(tmp_path):
    """The drop is irreversible, so it must never run after a seed that did not
    finish -- not from a ``finally``, not behind a swallowed exception."""
    _user_id, analytics, value_store = _fixture(tmp_path)
    with analytics._get_connection() as conn:
        # Present but lacking the columns the seed reads: a real failure, not
        # the tolerated missing table.
        conn.execute("CREATE TABLE user_analytics_snapshots (user_id INTEGER PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE user_lifecycle_daily_snapshots ("
            "snapshot_date TEXT NOT NULL, user_id INTEGER NOT NULL)"
        )

    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        run_startup_migrations(value_store=value_store, now=NOW)

    assert set(LEGACY_TABLES) <= _table_names(analytics)


def test_fresh_schema_declares_no_snapshot_tables():
    for ddl in (repository.ANALYTICS_SQLITE_DDL, repository_postgres.ANALYTICS_POSTGRES_DDL):
        for table in LEGACY_TABLES:
            assert table not in ddl
    for init_schema in (
        repository.AnalyticsStore._init_schema,
        repository_postgres.PostgresAnalyticsStore._init_schema,
    ):
        assert "user_analytics_snapshots" not in inspect.getsource(init_schema)


def _lifecycle_rollups(analytics, day: date) -> list[tuple]:
    return _rollup_shape(
        AnalyticsRollupStore(analytics).list_rollups(start=day, end=day + timedelta(days=1))
    )


def test_uncovered_legacy_days_are_rolled_up_before_the_drop(tmp_path):
    """I1: legacy days with no fact row survive the drop as anonymous lifecycle
    rollups, in the shape main's rollup_lifecycle_day wrote; a covered day is
    left to its facts, and the second boot is a no-op."""
    user_id, analytics, value_store = _fixture(tmp_path)
    create_legacy_snapshot_tables(analytics, daily_rows=LEGACY_DAILY_ROWS)
    value_store.upsert_daily_facts([covering_fact(user_id)])

    report = run_startup_migrations(value_store=value_store, now=NOW)

    assert report.legacy_days_rolled_up == 2
    for day in UNCOVERED_DAYS:
        assert _lifecycle_rollups(analytics, day) == legacy_lifecycle_rollups(day)
    assert _lifecycle_rollups(analytics, COVERED_DAY) == []
    assert not set(LEGACY_TABLES) & _table_names(analytics)

    again = run_startup_migrations(value_store=value_store, now=NOW)
    assert again == StartupMigrationReport()
    for day in UNCOVERED_DAYS:
        assert _lifecycle_rollups(analytics, day) == legacy_lifecycle_rollups(day)


def test_legacy_rollup_never_overwrites_a_day_that_has_lifecycle_rollups(tmp_path):
    """A day already rolled up (an earlier boot whose drop failed, or another
    writer) keeps what it has; other metric families on any day are untouched."""
    user_id, analytics, value_store = _fixture(tmp_path)
    create_legacy_snapshot_tables(analytics, daily_rows=LEGACY_DAILY_ROWS)
    value_store.upsert_daily_facts([covering_fact(user_id)])
    existing = DailyRollup(
        rollup_date=UNCOVERED_DAYS[0],
        metric_name="lifecycle_segment_count",
        outcome="complete",
        user_state="dormant",
        value_count=9,
        updated_at=NOW,
    )
    value_store.replace_lifecycle_rollups(UNCOVERED_DAYS[0], [existing])
    AnalyticsRollupStore(analytics).replace_day(
        UNCOVERED_DAYS[1],
        [
            DailyRollup(
                rollup_date=UNCOVERED_DAYS[1],
                metric_name="daily_active_users",
                value_count=4,
                updated_at=NOW,
            )
        ],
    )

    report = run_startup_migrations(value_store=value_store, now=NOW)

    assert report.legacy_days_rolled_up == 1
    assert _lifecycle_rollups(analytics, UNCOVERED_DAYS[0]) == [
        ("lifecycle_segment_count", "", "dormant", 9, "complete")
    ]
    day_two = _lifecycle_rollups(analytics, UNCOVERED_DAYS[1])
    assert ("daily_active_users", "", "", 4, "") in day_two
    assert [row for row in day_two if row[0] != "daily_active_users"] == (
        legacy_lifecycle_rollups(UNCOVERED_DAYS[1])
    )


def test_a_failed_legacy_rollup_keeps_both_legacy_tables(tmp_path):
    """The rollup is the last read of the daily table: it failing must skip the
    drop exactly as a failed seed does."""
    _user_id, analytics, value_store = _fixture(tmp_path)
    create_legacy_snapshot_tables(
        analytics,
        # A five-state label, not a lifecycle segment: the rollup refuses it.
        daily_rows=[("2026-07-11", 101, "active", "complete")],
    )

    with pytest.raises(ValueError):
        run_startup_migrations(value_store=value_store, now=NOW)

    assert set(LEGACY_TABLES) <= _table_names(analytics)
    assert _lifecycle_rollups(analytics, UNCOVERED_DAYS[0]) == []


def test_missing_legacy_table_is_detected_from_the_catalogue_not_message_text():
    """Error text follows the server's lc_messages; a non-English Postgres would
    turn the tolerated missing table into a boot-time failure every boot."""
    from dashboard.backend.domain.analytics import value_repository, value_repository_postgres

    for store in (
        value_repository.ValueAnalyticsStore,
        value_repository_postgres.PostgresValueAnalyticsStore,
    ):
        for method in (store.seed_activity_from_snapshots, store.list_legacy_lifecycle_history):
            source = inspect.getsource(method)
            assert "str(exc)" not in source and "except" not in source, method
            assert "_table_exists(" in source, method
    assert "to_regclass" in inspect.getsource(value_repository_postgres._postgres_table_exists)
    assert "sqlite_master" in inspect.getsource(value_repository._sqlite_table_exists)
