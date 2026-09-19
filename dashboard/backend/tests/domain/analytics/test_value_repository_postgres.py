"""Structural parity checks for PostgresValueAnalyticsStore.

Neither test here needs TEST_POSTGRES_URL: both check the class shape and
the factory's dispatch logic with plain Python objects, never a live
connection. The behavioural round-trip against a real database lives in
tests/domain/analytics/test_repository_postgres.py
(test_postgres_user_value_projection_round_trip, @pg_only), which already
had a live-Postgres tier before this store had a twin to put on it -- this
file only has to point that existing test at the new class.
"""

from __future__ import annotations

import inspect

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
