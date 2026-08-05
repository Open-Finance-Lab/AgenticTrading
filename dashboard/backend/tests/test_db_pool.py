"""Shared per-URL psycopg pool (T4). Unit tests need no live Postgres; the
@pg_only round-trip follows the established local-postgres fixture rules."""

import os

import pytest

pytest.importorskip("psycopg_pool")

from dashboard.backend import db_pool


class _FakePool:
    instances = []

    # Mirrors psycopg_pool.ConnectionPool.check_connection: production passes
    # ``check=ConnectionPool.check_connection`` and resolves it off whatever
    # class the module holds, so the fake must expose one too.
    @staticmethod
    def check_connection(conn):
        raise AssertionError("not used in dispatch tests")

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.closed = False
        type(self).instances.append(self)

    def connection(self):
        raise AssertionError("not used in dispatch tests")

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(db_pool, "ConnectionPool", _FakePool)
    db_pool._reset_for_tests()
    _FakePool.instances = []
    yield
    db_pool._reset_for_tests()


def test_one_pool_per_url_cached():
    p1 = db_pool.get_pool("postgresql://u@h/db1")
    p2 = db_pool.get_pool("postgresql://u@h/db1")
    p3 = db_pool.get_pool("postgresql://u@h/db2")
    assert p1 is p2
    assert p1 is not p3
    assert len(_FakePool.instances) == 2


def test_pool_configured_for_neon_and_dict_rows():
    from psycopg.rows import dict_row

    db_pool.get_pool("postgresql://u@h/db")
    kwargs = _FakePool.instances[0].kwargs
    assert kwargs["max_size"] == 5
    # Well below Neon's ~300s scale-to-zero suspend. At exactly 300 the pool
    # and Neon race: whichever side's timer fires first decides whether the
    # next caller gets a live socket or a dead one.
    assert kwargs["max_idle"] == 120
    # Pre-ping at checkout: a connection Neon killed during suspend costs one
    # silent reconnect instead of a user-visible 500 on the first request
    # after idle.
    assert kwargs["check"] is _FakePool.check_connection
    assert kwargs["kwargs"] == {"row_factory": dict_row}


def test_close_all_pools_closes_and_forgets():
    """Short-lived CLIs call this so psycopg_pool's own teardown does not print
    'couldn't stop thread ...' warnings after a successful run."""
    p1 = db_pool.get_pool("postgresql://u@h/db1")
    p2 = db_pool.get_pool("postgresql://u@h/db2")

    db_pool.close_all_pools()

    assert p1.closed and p2.closed
    # Forgotten, not merely closed: a closed pool left in the cache would be
    # handed to the next get_pool() caller and raise on use.
    assert db_pool.get_pool("postgresql://u@h/db1") is not p1


def test_close_all_pools_survives_a_pool_that_wont_close():
    """Teardown is best-effort -- one broken pool must not strand the others."""
    class _Stubborn(_FakePool):
        def close(self):
            raise RuntimeError("nope")

    # Registered *first*, so an uncaught raise here would abort the loop before
    # it ever reaches the healthy pool -- that is what this test pins.
    db_pool._pools["postgresql://u@h/bad"] = _Stubborn("postgresql://u@h/bad")
    healthy = db_pool.get_pool("postgresql://u@h/ok")

    db_pool.close_all_pools()

    assert healthy.closed
    assert db_pool._pools == {}


TEST_PG = os.getenv("TEST_POSTGRES_URL")
pg_only = pytest.mark.skipif(not TEST_PG, reason="TEST_POSTGRES_URL not set")


@pg_only
def test_pooled_agent_store_round_trip(monkeypatch):
    """A twin resolves through the real pool. Guard: never a prod URL."""
    from psycopg_pool import ConnectionPool as RealPool

    from dashboard.backend.tests._postgres_testing import require_local_postgres_url

    require_local_postgres_url(TEST_PG)
    monkeypatch.setattr(db_pool, "ConnectionPool", RealPool)  # replace the fake
    db_pool._reset_for_tests()
    from dashboard.backend.domain.agents.repository_postgres import PostgresAgentStore

    store = PostgresAgentStore(TEST_PG)
    created = store.create_agent(name="pool-probe", model_name="m",
                                 agent_type="external", description="")
    resolved = store.resolve_api_key(created["api_key"])
    assert resolved and resolved["agent_id"] == created["agent_id"]
    store.delete_agent(created["agent_id"])
    db_pool._reset_for_tests()  # close the real pool before the fake returns
