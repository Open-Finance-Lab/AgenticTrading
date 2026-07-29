"""PostgresBacktestDatabase dispatch tests (run-history backend factory).

Two tiers, mirroring test_agent_store_postgres.py:
1. Dispatch-logic tests (no live Postgres needed) -- verify _build_backtest_db
   picks the right database class based on AGENT_RUNS_DATABASE_URL.
2. Behavioral tests against a real Postgres -- skipped unless TEST_POSTGRES_URL
   is set. Point it at a throwaway database, e.g.:
     docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=atl_test \
       -p 5433:5432 postgres:18-alpine
     export TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/atl_test

Task 9 appends the @pg_only behavioral half below the dispatch tests in this
same file.
"""

import os

import pytest

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


# --- dispatch tests (backtest db) --------------------------------------------

def test_build_backtest_db_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.database as database_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    store = database_module._build_backtest_db()

    assert isinstance(store, database_module.BacktestDatabase)
    assert (
        "run history backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_build_backtest_db_picks_postgres_when_url_set(monkeypatch, capsys):
    import dashboard.backend.database as database_module
    import dashboard.backend.database_postgres as database_pg_module

    created = {}

    class FakePostgresBacktestDatabase:
        def __init__(self, database_url):
            created["database_url"] = database_url

    # _build_backtest_db imports PostgresBacktestDatabase *inside the function*
    # from database_postgres, so the name is never bound on database_module --
    # patching it there would be a no-op. Patch the source module instead.
    monkeypatch.setattr(
        database_pg_module, "PostgresBacktestDatabase", FakePostgresBacktestDatabase
    )
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://fake/db")

    store = database_module._build_backtest_db()

    assert isinstance(store, FakePostgresBacktestDatabase)
    assert created["database_url"] == "postgresql://fake/db"
    assert "run history backend: postgres (fake/db)" in capsys.readouterr().out


def test_build_backtest_db_ignores_content_and_users_database_url(monkeypatch, capsys):
    """AGENT_RUNS_DATABASE_URL is the ONLY var allowed to select the Postgres
    run-history backend. CONTENT_DATABASE_URL (agents/versions/strategies) and
    USERS_DATABASE_URL (accounts) are scoped to their own stores and must never
    leak into this decision (spec, Decision 3) -- neither may substitute for
    AGENT_RUNS_DATABASE_URL, not even when it is unset.

    This is the no-fallback-chain guarantee: it is what would catch someone
    later "simplifying" the factory into falling back to a sibling database's
    URL, a one-line change that reads like an improvement and keeps the rest
    of the suite green while silently binding run history to the wrong
    database. Both siblings are set here and SQLite must still be chosen.
    """
    import dashboard.backend.database as database_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/content")
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/users")

    store = database_module._build_backtest_db()

    assert isinstance(store, database_module.BacktestDatabase)
    assert (
        "run history backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_build_backtest_db_never_prints_the_credentials(monkeypatch, capsys):
    """The printed line is the design's only misconfiguration tripwire (see
    describe_database_url in db_url.py) -- assert BOTH halves: the secret is
    absent, AND the exact host/db line is present. Asserting only absence
    would keep passing even if the whole line silently disappeared, which
    would delete the tripwire without any test noticing.
    """
    import dashboard.backend.database as database_module
    import dashboard.backend.database_postgres as database_pg_module

    class FakePostgresBacktestDatabase:
        def __init__(self, database_url):
            pass

    monkeypatch.setattr(
        database_pg_module, "PostgresBacktestDatabase", FakePostgresBacktestDatabase
    )
    monkeypatch.setenv(
        "AGENT_RUNS_DATABASE_URL", "postgresql://admin:sup3r-s3cret@host/db"
    )

    database_module._build_backtest_db()

    out = capsys.readouterr().out
    assert "sup3r-s3cret" not in out
    assert "run history backend: postgres (host/db)" in out


def test_unreachable_postgres_raises_instead_of_falling_back():
    """Fail loud: a set-but-unreachable AGENT_RUNS_DATABASE_URL must not
    silently degrade to ephemeral SQLite. A silent fallback here would be
    exactly the "absent vs. broken" failure shape CLAUDE.md's "Fail-closed is
    not fail-visible" section warns about -- run history would just look
    empty, with nothing in the logs saying why.

    What this proves is "it fails loudly", not "psycopg raises one particular
    class": the pool checkout (db_pool.py) actually raises
    psycopg_pool.PoolTimeout, a subclass of psycopg.OperationalError, so a
    future psycopg upgrade that changes the concrete subclass should not read
    as a product regression here -- only a change in fail-loud-vs-silent
    behavior should.

    Timing: db_pool.get_pool() retries the connection for
    POOL_TIMEOUT_SECONDS (prod default 10s) before giving up.
    conftest.py's autouse `_reset_shared_scale_state` fixture already
    monkeypatches that constant down to 1.0s for every test in this suite, so
    this raises in ~1s here rather than paying the full prod timeout -- no
    additional patching needed in this test.
    """
    import psycopg

    from dashboard.backend.database_postgres import PostgresBacktestDatabase

    with pytest.raises(psycopg.OperationalError):
        PostgresBacktestDatabase("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """A typo'd AGENT_RUNS_DATABASE_URL must not put the password in the log.

    psycopg parses anything not starting with postgresql:// as a keyword DSN
    and quotes the whole input back ('missing "=" after "<the entire URL>"').
    This runs at import time with no try/except, so that message is the boot
    failure and it lands in Render's log. require_postgres_url must therefore
    run before psycopg ever sees the value.
    """
    from dashboard.backend.database_postgres import PostgresBacktestDatabase

    with pytest.raises(ValueError) as excinfo:
        PostgresBacktestDatabase('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)
