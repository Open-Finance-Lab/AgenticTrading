"""PostgresStrategyStore tests.

Dispatch tests need no live Postgres. Behavioral tests run only when
TEST_POSTGRES_URL is set (see test_users_postgres.py's docstring for the
docker recipe). The collision tests mirror test_strategy_store.py exactly:
the Postgres retry loop is structurally different (ON CONFLICT DO NOTHING +
rowcount, because a UniqueViolation would abort the transaction) and must
behave identically to SQLite's catch-and-retry.
"""

import os

import pytest

from dashboard.backend.tests._postgres_testing import require_local_postgres_url

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


# --- dispatch tests ----------------------------------------------------------

def test_build_strategy_store_defaults_to_sqlite(monkeypatch, capsys):
    from dashboard.backend.domain.strategies.repository import (
        StrategyStore,
        _build_strategy_store,
    )

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = _build_strategy_store()
    assert isinstance(store, StrategyStore)
    assert "strategy_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out


def test_build_strategy_store_picks_postgres_when_url_set(monkeypatch, capsys):
    from dashboard.backend.domain.strategies.repository import _build_strategy_store

    created = {}

    class FakePostgresStrategyStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(
        "dashboard.backend.domain.strategies.repository_postgres.PostgresStrategyStore",
        FakePostgresStrategyStore,
    )
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/db")

    store = _build_strategy_store()

    assert isinstance(store, FakePostgresStrategyStore)
    assert created["database_url"] == "postgresql://fake/db"
    assert "strategy_store backend: postgres (fake/db)" in capsys.readouterr().out


def test_build_strategy_store_ignores_users_database_url(monkeypatch, capsys):
    """See the agent-store twin of this test (test_agent_store_postgres.py)."""
    from dashboard.backend.domain.strategies.repository import (
        StrategyStore,
        _build_strategy_store,
    )

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/users")

    store = _build_strategy_store()

    assert isinstance(store, StrategyStore)
    assert (
        "strategy_store backend: sqlite (ephemeral on Render)"
        in capsys.readouterr().out
    )


def test_unreachable_postgres_strategy_store_raises_instead_of_falling_back():
    """Fail loud — see the agent-store twin of this test."""
    import psycopg

    from dashboard.backend.domain.strategies.repository_postgres import (
        PostgresStrategyStore,
    )

    with pytest.raises(psycopg.OperationalError):
        PostgresStrategyStore("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """See the agent-store twin of this test (test_agent_store_postgres.py)."""
    from dashboard.backend.domain.strategies.repository_postgres import (
        PostgresStrategyStore,
    )

    with pytest.raises(ValueError) as excinfo:
        PostgresStrategyStore('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)


# --- live-Postgres behavioral tests ------------------------------------------

@pytest.fixture
def pg_strategy_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    from dashboard.backend.domain.strategies.repository_postgres import (
        PostgresStrategyStore,
    )

    store = PostgresStrategyStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM strategies")
    yield store


@pg_only
def test_create_get_set_last_run_postgres(pg_strategy_store):
    created = pg_strategy_store.create(
        prompt="buy the dip", description="  classic  ", owner="discord:123"
    )
    assert len(created["code"]) == 8
    assert created["description"] == "classic"
    assert created["last_run_id"] is None

    fetched = pg_strategy_store.get(created["code"])
    assert fetched == created
    assert pg_strategy_store.get("missing0") is None
    assert pg_strategy_store.get("") is None

    updated = pg_strategy_store.set_last_run(created["code"], "run_abc")
    assert updated["last_run_id"] == "run_abc"
    assert updated["last_run_at"] is not None
    assert pg_strategy_store.set_last_run("missing0", "run_abc") is None

    with pytest.raises(ValueError):
        pg_strategy_store.create(prompt="   ")


@pg_only
def test_create_retries_past_a_code_collision_postgres(pg_strategy_store, monkeypatch):
    from dashboard.backend.domain.strategies.repository_postgres import secrets

    first = pg_strategy_store.create(prompt="first strategy")

    codes = iter([first["code"], "fresh456"])
    monkeypatch.setattr(
        secrets, "token_hex", lambda nbytes: next(codes)
    )

    second = pg_strategy_store.create(prompt="second strategy")
    assert second["code"] == "fresh456"
    assert pg_strategy_store.get(first["code"])["prompt"] == "first strategy"


@pg_only
def test_create_widens_code_space_after_20_collisions_postgres(
    pg_strategy_store, monkeypatch
):
    from dashboard.backend.domain.strategies.repository import _CODE_LENGTH
    from dashboard.backend.domain.strategies.repository_postgres import secrets

    first = pg_strategy_store.create(prompt="first strategy")

    calls = {"n": 0, "nbytes": []}

    def fake_token_hex(nbytes):
        calls["n"] += 1
        calls["nbytes"].append(nbytes)
        if calls["n"] <= 20:
            return first["code"]
        return "w" * 16

    monkeypatch.setattr(secrets, "token_hex", fake_token_hex)

    second = pg_strategy_store.create(prompt="second strategy")
    assert second["code"] == "w" * 16
    assert calls["n"] == 21
    # #137 gap 5: the 20 narrow attempts request _CODE_LENGTH // 2 bytes; only the
    # widened 21st requests the full _CODE_LENGTH. Swapping the two call sites is
    # otherwise invisible -- both still produce a valid code.
    assert calls["nbytes"][:20] == [_CODE_LENGTH // 2] * 20
    assert calls["nbytes"][20] == _CODE_LENGTH


@pg_only
def test_create_raises_when_even_widened_code_collides_postgres(
    pg_strategy_store, monkeypatch
):
    from dashboard.backend.domain.strategies.repository_postgres import secrets

    first = pg_strategy_store.create(prompt="first strategy")

    monkeypatch.setattr(
        secrets, "token_hex", lambda nbytes: first["code"]
    )

    with pytest.raises(RuntimeError):
        pg_strategy_store.create(prompt="second strategy")
