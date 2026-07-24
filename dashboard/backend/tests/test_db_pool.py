"""Shared per-URL psycopg pool (T4). Unit tests need no live Postgres; the
@pg_only round-trip follows the established local-postgres fixture rules."""

import os

import pytest

pytest.importorskip("psycopg_pool")

from dashboard.backend import db_pool


class _FakePool:
    instances = []

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs
        type(self).instances.append(self)

    def connection(self):
        raise AssertionError("not used in dispatch tests")


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
    assert kwargs["max_idle"] == 300          # < Neon scale-to-zero idle window
    assert kwargs["kwargs"] == {"row_factory": dict_row}


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
