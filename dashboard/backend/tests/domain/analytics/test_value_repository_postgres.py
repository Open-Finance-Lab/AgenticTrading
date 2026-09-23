"""Structural parity checks for PostgresValueAnalyticsStore.

No test here needs TEST_POSTGRES_URL: they check the class shape, the
factory's dispatch logic and the two constructor dialect guards with plain
Python objects, never a live connection. The behavioural round-trip against
a real database lives in
tests/domain/analytics/test_repository_postgres.py
(test_postgres_user_value_projection_round_trip, @pg_only), which already
had a live-Postgres tier before this store had a twin to put on it -- this
file only has to point that existing test at the new class.
"""

from __future__ import annotations

import inspect

import pytest

from dashboard.backend.domain.analytics.value_repository import (
    ValueAnalyticsStore,
    build_value_analytics_store,
)


def _public_methods(cls) -> set[str]:
    return {
        name
        for name in dir(cls)
        if not name.startswith("_") and callable(getattr(cls, name, None))
    }


def test_postgres_value_analytics_store_matches_sqlite_public_surface():
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    sqlite_methods = _public_methods(ValueAnalyticsStore)
    postgres_methods = _public_methods(PostgresValueAnalyticsStore)
    assert sqlite_methods == postgres_methods, (
        f"sqlite-only={sorted(sqlite_methods - postgres_methods)} "
        f"postgres-only={sorted(postgres_methods - sqlite_methods)}"
    )
    for name in sqlite_methods:
        assert inspect.signature(
            getattr(ValueAnalyticsStore, name)
        ) == inspect.signature(getattr(PostgresValueAnalyticsStore, name)), (
            f"{name} signature diverges between the twins"
        )


class _FakePostgresBase:
    """A stand-in analytics_base with no real connection.

    build_value_analytics_store() dispatches on ``hasattr(base,
    "database_url")`` alone, so a plain attribute exercises the branch
    without a live database -- exactly like the ``object()`` sentinels
    domain/analytics/retention.py already passes for credits_base/
    provider_base/agent_base/run_base when it never calls those stores.
    """

    database_url = "postgresql://example/test"


def test_build_value_analytics_store_selects_postgres_twin_for_a_postgres_base():
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    store = build_value_analytics_store(
        _FakePostgresBase(),
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    assert isinstance(store, PostgresValueAnalyticsStore)


def test_build_value_analytics_store_selects_sqlite_twin_by_default():
    class _FakeSqliteBase:
        pass

    store = build_value_analytics_store(
        _FakeSqliteBase(),
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    assert isinstance(store, ValueAnalyticsStore)


# --------------------------------------------------------------------------
# Constructor dialect guards
# --------------------------------------------------------------------------
#
# The two parity tests above compare *public* methods: both build their name
# list from ``dir(cls)`` and skip ``name.startswith("_")``, so neither one
# sees ``__init__``. Nothing pinned either constructor until these tests, and
# the mismatch they catch is invisible everywhere it would be run -- CI and
# local pytest are both SQLite, so a twin paired with the wrong dialect only
# raises on the Postgres deployment.


def test_sqlite_twin_rejects_a_postgres_base():
    """``ValueAnalyticsStore(postgres_base)`` was correct before PR T.

    It now emits ``?`` placeholders, so it has to refuse rather than fail
    later with a psycopg syntax error on prod alone.
    """
    with pytest.raises(TypeError, match="SQLite twin"):
        ValueAnalyticsStore(
            _FakePostgresBase(),
            credits_base=object(),
            provider_base=object(),
            agent_base=object(),
            run_base=object(),
        )


def test_postgres_twin_rejects_a_sqlite_base():
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    class _FakeSqliteBase:
        pass

    with pytest.raises(TypeError, match="requires a PostgreSQL analytics base"):
        PostgresValueAnalyticsStore(
            _FakeSqliteBase(),
            credits_base=object(),
            provider_base=object(),
            agent_base=object(),
            run_base=object(),
        )


def test_dialect_guards_quote_no_part_of_the_connection_string():
    """Same rule as ``require_postgres_url``: the message carries no URL.

    These constructors are reached at import time by the store factories, so
    whatever they raise lands in the deploy log verbatim.
    """
    from dashboard.backend.domain.analytics.value_repository_postgres import (
        PostgresValueAnalyticsStore,
    )

    secret = "postgresql://user:hunter2@ep-example.neon.tech/atl"

    class _SecretBase:
        database_url = secret

    base = _SecretBase()
    with pytest.raises(TypeError) as excinfo:
        ValueAnalyticsStore(base, credits_base=object())

    message = str(excinfo.value)
    assert secret not in message
    assert "hunter2" not in message
    assert "neon.tech" not in message
    # repr() of the base would carry no URL today, but a base that grows a
    # __repr__ naming its DSN is exactly how this leaks later.
    assert repr(base) not in message

    # The Postgres twin only ever refuses a base with *no* database_url, so
    # it has no URL to leak; assert the shape rather than a secret.
    with pytest.raises(TypeError) as pg_excinfo:
        PostgresValueAnalyticsStore(object(), credits_base=object())
    assert "database_url" in str(pg_excinfo.value)
