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
from dashboard.backend.domain.analytics.value_repository import (
    ProjectionJob,
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)
from dashboard.backend.tests._postgres_testing import require_local_postgres_url
from dashboard.backend.tests.domain.analytics.test_repository_contract import (
    assert_cursor_contract,
    assert_event_idempotency_contract,
    assert_pr2_query_contract,
    assert_source_event_idempotency_contract,
    assert_subject_and_access_contract,
    assert_error_category_contract,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    NOW,
    SyntheticAgentStore,
    SyntheticCreditsStore,
    SyntheticProviderStore,
    SyntheticRunStore,
    _value_snapshot,
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


def test_postgres_ddl_declares_user_value_projection_storage():
    ddl = pg_module.ANALYTICS_POSTGRES_DDL
    assert "CREATE TABLE IF NOT EXISTS user_lifecycle_daily_snapshots" in ddl
    assert "CREATE TABLE IF NOT EXISTS analytics_projection_jobs" in ddl
    for column in (
        "lifecycle_segment",
        "operational_state",
        "activated_at",
        "last_meaningful_activity_at",
        "inactive_days",
        "active_days_30d",
        "successful_backtests_30d",
    ):
        assert column in ddl or f'"{column}"' in ddl
    assert "ADD COLUMN IF NOT EXISTS" in inspect.getsource(
        PostgresAnalyticsStore._init_schema
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
def test_postgres_user_value_projection_round_trip(
    postgres_contract_store,
    tmp_path,
):
    analytics, _admin_id, user_id = postgres_contract_store
    value_store = ValueAnalyticsStore(
        analytics,
        SyntheticCreditsStore(tmp_path / "postgres-value-credits.db"),
        provider_base=SyntheticProviderStore({}, []),
        agent_base=SyntheticAgentStore(),
        run_base=SyntheticRunStore({}),
    )
    snapshot = _value_snapshot(user_id)
    daily = UserLifecycleDailySnapshot(
        snapshot_date=NOW.date(),
        user_id=user_id,
        lifecycle_segment="core",
        lifecycle_reason_code="core_repeated_value",
        data_quality="complete",
        calculated_at=NOW,
    )
    job = ProjectionJob(
        job_name="postgres-lifecycle-backfill",
        window_start=NOW.date(),
        window_end=NOW.date(),
        status="complete",
        updated_at=NOW,
    )

    value_store.upsert_current_snapshot(snapshot)
    value_store.upsert_daily_snapshot(daily)
    value_store.save_projection_job(job)

    assert value_store.get_current_snapshot(user_id) == snapshot
    assert value_store.list_daily_snapshots(
        start=NOW.date(),
        end=NOW.date() + timedelta(days=1),
        user_ids=[user_id],
    ) == [daily]
    assert value_store.get_projection_job(job.job_name) == job
