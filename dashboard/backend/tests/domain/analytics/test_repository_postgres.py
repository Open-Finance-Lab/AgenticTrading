"""Dispatch and live-PostgreSQL tests for the Analytics store twin."""

from __future__ import annotations

import inspect
import os
import uuid
from datetime import timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from dashboard.backend import db_pool
from dashboard.backend.domain.analytics import repository as repo_module
from dashboard.backend.domain.analytics import repository_postgres as pg_module
from dashboard.backend.domain.analytics.repository_postgres import (
    PostgresAnalyticsStore,
)
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    DailyRollup,
    rollup_lifecycle_day,
)
from dashboard.backend.domain.analytics.value_repository import (
    ActivityUpdate,
    LifecycleTransitionRow,
    ProjectionJob,
    UserDailyFact,
)
from dashboard.backend.domain.analytics.value_repository_postgres import (
    PostgresValueAnalyticsStore,
)
from dashboard.backend.tests._postgres_testing import require_local_postgres_url
from dashboard.backend.tests.domain.analytics.test_repository_contract import (
    assert_cursor_contract,
    assert_event_idempotency_contract,
    assert_pr2_query_contract,
    assert_source_event_idempotency_contract,
    assert_subject_and_access_contract,
    assert_error_category_contract,
    event_record,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    NOW,
    SyntheticAgentStore,
    SyntheticCreditsStore,
    SyntheticProviderStore,
    SyntheticRunStore,
)


TEST_POSTGRES_URL = require_local_postgres_url(os.getenv("TEST_POSTGRES_URL"))
pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL is not configured",
)


def _schema_url(database_url: str, schema: str) -> str:
    parts = urlsplit(database_url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


@pytest.fixture
def postgres_contract_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    schema = f"analytics_{uuid.uuid4().hex}"
    with psycopg.connect(TEST_POSTGRES_URL, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    scoped_url = _schema_url(TEST_POSTGRES_URL, schema)
    try:
        with psycopg.connect(scoped_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        email TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO users (
                        id, email, display_name, password_hash, role, created_at
                    ) VALUES
                        (1, 'admin@example.test', 'Admin', 'x', 'admin',
                         '2026-08-01T00:00:00+00:00'),
                        (2, 'analytics-user@example.test', 'Analytics User', 'x',
                         'user', '2026-08-06T12:00:00+00:00')
                    """
                )
        yield PostgresAnalyticsStore(scoped_url), 1, 2
    finally:
        db_pool._reset_for_tests()
        with psycopg.connect(TEST_POSTGRES_URL, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
                )


def test_build_analytics_store_defaults_to_sqlite(monkeypatch, capsys):
    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    store = repo_module._build_analytics_store()
    assert isinstance(store, repo_module.AnalyticsStore)
    assert "analytics_store backend: sqlite" in capsys.readouterr().out


def test_build_analytics_store_uses_only_users_database_url(monkeypatch, capsys):
    created = {}

    class FakePostgresAnalyticsStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(
        pg_module,
        "PostgresAnalyticsStore",
        FakePostgresAnalyticsStore,
    )
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/accounts")
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://ignored/content")
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://ignored/runs")

    store = repo_module._build_analytics_store()

    assert isinstance(store, FakePostgresAnalyticsStore)
    assert created["database_url"] == "postgresql://fake/accounts"
    assert "analytics_store backend: postgres (fake/accounts)" in capsys.readouterr().out


def test_dispatch_log_never_contains_database_password(monkeypatch, capsys):
    monkeypatch.setattr(
        pg_module,
        "PostgresAnalyticsStore",
        lambda database_url: object(),
    )
    monkeypatch.setenv(
        "USERS_DATABASE_URL",
        "postgresql://admin:synthetic-password@host/accounts",
    )
    repo_module._build_analytics_store()
    output = capsys.readouterr().out
    assert "synthetic-password" not in output
    assert "host/accounts" in output


def test_postgres_store_rejects_non_postgres_url_before_connecting():
    with pytest.raises(ValueError, match="postgresql://"):
        PostgresAnalyticsStore("sqlite:///tmp/not-postgres.db")


def test_public_repository_methods_match():
    def public_methods(cls):
        return {
            name
            for name, value in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith("_")
        }

    assert public_methods(repo_module.AnalyticsStore) == public_methods(
        PostgresAnalyticsStore
    )


@pg_only
def test_postgres_runs_shared_event_contracts(postgres_contract_store):
    store, _admin_id, user_id = postgres_contract_store
    assert_event_idempotency_contract(store, user_id)
    assert_source_event_idempotency_contract(store, user_id)
    assert_error_category_contract(store, user_id)


@pg_only
def test_postgres_runs_shared_cursor_contract(postgres_contract_store):
    store, _admin_id, user_id = postgres_contract_store
    assert_cursor_contract(store, user_id)


@pg_only
def test_postgres_runs_shared_subject_and_access_contract(postgres_contract_store):
    store, admin_id, user_id = postgres_contract_store
    assert_subject_and_access_contract(store, admin_id, user_id)


@pg_only
def test_postgres_runs_pr2_query_contract(postgres_contract_store):
    store, _admin_id, user_id = postgres_contract_store
    assert_pr2_query_contract(store, user_id)


@pg_only
def test_postgres_projection_job_and_lifecycle_rollups_round_trip(
    postgres_contract_store,
    tmp_path,
):
    analytics, _admin_id, _user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-value-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
    job = ProjectionJob(
        job_name="postgres-lifecycle-backfill",
        window_start=NOW.date(),
        window_end=NOW.date(),
        status="complete",
        updated_at=NOW,
    )

    value_store.save_projection_job(job)
    value_store.save_projection_job(job)

    assert value_store.get_projection_job(job.job_name) == job

    value_store.replace_lifecycle_rollups(
        NOW.date(),
        [
            DailyRollup(
                rollup_date=NOW.date(),
                metric_name="lifecycle_segment_count",
                outcome="complete",
                user_state="core",
                value_count=1,
                updated_at=NOW,
            )
        ],
    )
    lifecycle_rollups = AnalyticsRollupStore(analytics).list_rollups(
        start=NOW.date(),
        end=NOW.date() + timedelta(days=1),
    )
    assert lifecycle_rollups[0].metric_name == "lifecycle_segment_count"


@pg_only
def test_postgres_reseed_then_drop_legacy_snapshot_tables(
    postgres_contract_store,
):
    """The prod path of design SS11: re-seed from the live legacy row, drop the
    two tables, and boot cleanly again once they are gone (UndefinedTable)."""
    from dashboard.backend.domain.analytics.facts_migration import (
        StartupMigrationReport,
        run_startup_migrations,
    )
    from dashboard.backend.tests.domain.analytics.test_facts_migration import (
        ACTIVATED_AT,
        LAST_ACTIVITY_AT,
        LEGACY_TABLES,
    )

    analytics, _admin_id, user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    with analytics._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE user_analytics_snapshots (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    calculated_at TEXT NOT NULL,
                    activated_at TEXT,
                    last_meaningful_activity_at TEXT
                )
                """
            )
            cur.execute(
                "CREATE TABLE user_lifecycle_daily_snapshots ("
                "snapshot_date TEXT NOT NULL, user_id INTEGER NOT NULL)"
            )
            cur.execute(
                "INSERT INTO user_analytics_snapshots (user_id, status, calculated_at, "
                "activated_at, last_meaningful_activity_at) VALUES (%s, 'active', %s, %s, %s)",
                (user_id, NOW.isoformat(), ACTIVATED_AT.isoformat(), LAST_ACTIVITY_AT.isoformat()),
            )

    def legacy_tables_present() -> set[str]:
        with analytics._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
                    (list(LEGACY_TABLES),),
                )
                return {row["table_name"] for row in cur.fetchall()}  # dict_row pool

    assert legacy_tables_present() == set(LEGACY_TABLES)

    report = run_startup_migrations(value_store=value_store, now=NOW)

    assert report == StartupMigrationReport(activity_seeded=1)
    activity = value_store.get_activity(user_id)
    assert activity.activated_at == ACTIVATED_AT
    assert activity.last_meaningful_activity_at == LAST_ACTIVITY_AT
    assert legacy_tables_present() == set()

    # Second boot: the seed hits UndefinedTable and reads nothing; the drop is
    # IF EXISTS. The failed SELECT must not poison the pooled connection.
    analytics.drop_legacy_snapshot_tables()
    assert value_store.seed_activity_from_snapshots(now=NOW) == 0
    assert run_startup_migrations(value_store=value_store, now=NOW) == (
        StartupMigrationReport(activity_seeded=0)
    )
    assert value_store.get_activity(user_id).activated_at == ACTIVATED_AT


@pg_only
def test_postgres_rolls_up_uncovered_legacy_days_before_the_drop(
    postgres_contract_store,
):
    """I1 on the prod dialect: uncovered legacy days become anonymous lifecycle
    rollups, a covered day is left alone, the drop follows, and the second boot
    (to_regclass IS NULL) is a no-op."""
    from dashboard.backend.domain.analytics.facts_migration import (
        StartupMigrationReport,
        run_startup_migrations,
    )
    from dashboard.backend.tests.domain.analytics.test_facts_migration import (
        COVERED_DAY,
        LEGACY_DAILY_DDL,
        LEGACY_DAILY_ROWS,
        UNCOVERED_DAYS,
        _rollup_shape,
        covering_fact,
        legacy_lifecycle_rollups,
    )

    analytics, _admin_id, user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    with analytics._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE user_analytics_snapshots (user_id INTEGER PRIMARY KEY, "
                "activated_at TEXT, last_meaningful_activity_at TEXT)"
            )
            cur.execute(LEGACY_DAILY_DDL)
            for snapshot_date, legacy_user, segment, quality in LEGACY_DAILY_ROWS:
                cur.execute(
                    "INSERT INTO user_lifecycle_daily_snapshots "
                    "VALUES (%s, %s, %s, 'x', %s, %s)",
                    (snapshot_date, legacy_user, segment, quality, NOW.isoformat()),
                )
    value_store.upsert_daily_facts([covering_fact(user_id)])

    def lifecycle_rollups(day):
        return _rollup_shape(
            AnalyticsRollupStore(analytics).list_rollups(
                start=day, end=day + timedelta(days=1)
            )
        )

    report = run_startup_migrations(value_store=value_store, now=NOW)

    assert report == StartupMigrationReport(legacy_days_rolled_up=2)
    for day in UNCOVERED_DAYS:
        assert lifecycle_rollups(day) == legacy_lifecycle_rollups(day)
    assert lifecycle_rollups(COVERED_DAY) == []
    assert value_store.list_legacy_lifecycle_history().days == ()
    assert run_startup_migrations(value_store=value_store, now=NOW) == (
        StartupMigrationReport()
    )


@pg_only
def test_postgres_failed_seed_keeps_both_legacy_tables(postgres_contract_store):
    from dashboard.backend.domain.analytics.facts_migration import (
        run_startup_migrations,
    )
    from dashboard.backend.tests.domain.analytics.test_facts_migration import (
        LEGACY_TABLES,
    )

    analytics, _admin_id, _user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    with analytics._get_connection() as conn:
        with conn.cursor() as cur:
            # Lacks activated_at: UndefinedColumn, not the tolerated UndefinedTable.
            cur.execute("CREATE TABLE user_analytics_snapshots (user_id INTEGER PRIMARY KEY)")
            cur.execute(
                "CREATE TABLE user_lifecycle_daily_snapshots ("
                "snapshot_date TEXT NOT NULL, user_id INTEGER NOT NULL)"
            )

    with pytest.raises(psycopg.errors.UndefinedColumn):
        run_startup_migrations(value_store=value_store, now=NOW)

    with analytics._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = ANY(%s)",
                (list(LEGACY_TABLES),),
            )
            present = {row["table_name"] for row in cur.fetchall()}
    assert present == set(LEGACY_TABLES)


@pg_only
def test_postgres_fact_expiry_and_long_term_rollups_round_trip(postgres_contract_store, tmp_path):
    from dashboard.backend.tests.domain.analytics.test_fact_reads import fact, transition

    analytics, _admin_id, user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-expiry-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
    old, recent = NOW.date() - timedelta(days=200), NOW.date() - timedelta(days=1)
    value_store.upsert_daily_facts(
        [
            fact(user_id, old, tier="starter", user_group="partner", runs_completed=3, operator_cost_micro=90_000),
            fact(user_id, recent),
        ]
    )
    value_store.append_lifecycle_transitions([transition(user_id, old, "onboarding", "growing")])
    before = NOW.date() - timedelta(days=180)

    # Rollup first (the long-term families), then expire the day.
    rollups = rollup_lifecycle_day(old, store=value_store)
    stored = AnalyticsRollupStore(analytics).list_rollups(start=old, end=old + timedelta(days=1))
    assert {(r.metric_name, r.user_state, r.value_count, r.value_sum_micro) for r in stored} >= {
        ("runs_by_tier", "starter", 3, 0),
        ("operator_cost_by_user_group", "partner", 1, 90_000),
        ("own_spend_by_tier", "starter", 1, 0),
    }
    assert {r.metric_name for r in rollups} == {r.metric_name for r in stored}

    assert value_store.list_expiring_fact_dates(before=before, limit=10) == [old]
    assert value_store.has_facts_before(before) is True
    assert value_store.expire_facts_for_date(old) == 2  # one fact row + one transition
    assert value_store.list_facts_for_date(old) == []
    assert value_store.list_transitions_for_date(old) == []
    assert value_store.list_facts_for_date(recent) != []  # inside the window, untouched
    assert value_store.list_expiring_fact_dates(before=before, limit=10) == []
    assert value_store.has_facts_before(before) is False

    # A re-run finds no facts and leaves the day's rollups alone.
    assert rollup_lifecycle_day(old, store=value_store) == ()
    again = AnalyticsRollupStore(analytics).list_rollups(start=old, end=old + timedelta(days=1))
    assert again == stored


@pg_only
def test_postgres_fact_reads_follow_the_contract(postgres_contract_store):
    from dashboard.backend.tests.domain.analytics.test_fact_reads import (
        assert_fact_read_contract,
        assert_priority_sort_contract,
        assert_users_page_contract,
        seed_population,
    )

    store, admin_id, _user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics_base=store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    next_id = iter(range(100, 200))
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            # The fixture's users table is a stub with no default id and no
            # user_group column; the users page filters and returns it. The
            # pre-seeded plain user is not part of the seeded population the
            # contract counts, so drop it rather than loosen every total.
            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS user_group TEXT "
                "NOT NULL DEFAULT 'unknown'"
            )
            cur.execute("DELETE FROM users WHERE id = 2")

    def create_user(key: str, email: str, group: str) -> int:
        user_id = next(next_id)
        with store._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (id, email, display_name, password_hash, "
                    "role, created_at, user_group) VALUES (%s, %s, %s, 'x', "
                    "'user', '2026-08-01T00:00:00+00:00', %s)",
                    (user_id, email, key.title(), group),
                )
        return user_id

    ids = seed_population(value_store, create_user, admin_id=admin_id)
    assert_fact_read_contract(value_store, ids)
    assert_users_page_contract(value_store, ids)
    assert_priority_sort_contract(value_store, create_user)


@pg_only
def test_postgres_daily_facts_methods_round_trip(postgres_contract_store, tmp_path):
    # Every method the daily-facts job calls on the Postgres twin. These
    # raised NameError in prod from 2026-09-23 (helpers used but never
    # imported from the SQLite twin) while the SQLite suite stayed green.
    analytics, _admin_id, user_id = postgres_contract_store
    value_store = PostgresValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-daily-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
    day = NOW.date()
    at = NOW.replace(hour=12, minute=0, second=0, microsecond=0)

    analytics.append_event(
        event_record(
            user_id,
            event_name="backtest_completed",
            event_group="run",
            event_source="server",
            source_event_id="run:backtest_completed:daily-facts-pg",
            source_record_type="run",
            source_record_id="daily-facts-pg",
            page_view=None,
            session_id=None,
            device_category=None,
            browser_family=None,
            outcome="succeeded",
            occurred_at=at,
            received_at=at + timedelta(seconds=1),
        )
    )
    totals = value_store.aggregate_events_for_day(day)
    assert totals[user_id].runs_completed == 1

    assert value_store.record_activity_batch(
        [ActivityUpdate(user_id=user_id, activated_at=at, last_activity_at=at)],
        now=NOW,
    ) == 1
    activity = value_store.get_activity(user_id)
    assert activity.activated_at == at
    assert activity.last_meaningful_activity_at == at

    fact = UserDailyFact(
        snapshot_date=day,
        user_id=user_id,
        lifecycle_segment="growing",
        lifecycle_reason_code="growing_activated_below_core_threshold",
        operational_state="healthy",
        tier="unpaid",
        user_group="unknown",
        active=True,
        runs_requested=1,
        runs_completed=1,
        data_quality="complete",
        calculated_at=at,
    )
    assert value_store.upsert_daily_facts([fact]) == 1
    assert value_store.list_facts_for_date(day) == [fact]

    assert value_store.append_lifecycle_transitions(
        [
            LifecycleTransitionRow(
                user_id=user_id,
                snapshot_date=day,
                from_segment="new",
                to_segment="growing",
                created_at=at,
            )
        ]
    ) == 1

    # The event was received after the facts were calculated, so the day is
    # stale and due for a recompute.
    assert value_store.list_days_needing_recompute(since=day, until=day) == [day]


def test_postgres_ddl_declares_the_daily_fact_tables():
    ddl = pg_module.ANALYTICS_POSTGRES_DDL
    assert "CREATE TABLE IF NOT EXISTS user_activity" in ddl
    assert "CREATE TABLE IF NOT EXISTS user_daily_facts" in ddl
    assert "CREATE TABLE IF NOT EXISTS lifecycle_transitions" in ddl
    for column in (
        "operator_cost_micro",
        "own_spend_micro",
        "runs_completed",
        "data_quality",
        "operational_reason_code",
        "user_group TEXT NOT NULL DEFAULT 'unknown'",
        "tier",
    ):
        assert column in ddl
    assert "cohort" not in ddl
    # Wide counters are BIGINT on Postgres, matching analytics_daily_rollups.
    assert "operator_cost_micro BIGINT" in ddl
    assert "own_spend_micro BIGINT" in ddl
