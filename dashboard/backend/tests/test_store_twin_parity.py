"""Signature parity between each SQLite store and its Postgres twin.

Every dual-backend store is selected by a factory at import time, and the
service layer calls one interface against whichever twin got built. Callers
pass sentinel kwargs (``_UNSET``) on every call, so a parameter that exists on
the SQLite store but not the Postgres twin is not a feature gap — it is a
``TypeError`` raised on prod before any SQL runs, on every call to that
method. #227 added ``live_trading_enabled`` to ``AgentStore.update_agent``
only, and every agent Configure PATCH on prod 500'd bare (an unhandled
exception escapes CORSMiddleware, so browsers reported it as a CORS block)
while the SQLite-backed test suite stayed green.

Comparing ``inspect.signature`` needs no live Postgres, so this tier stays
active where the @pg_only behavioral tier fails open (TEST_POSTGRES_URL
unset — local dev and any CI lane without the service container).
"""

import inspect

import pytest


def _twin_pairs():
    from dashboard.backend.domain.agents.repository import AgentStore
    from dashboard.backend.domain.agents.repository_postgres import PostgresAgentStore
    from dashboard.backend.domain.agents.version_repository import AgentVersionStore
    from dashboard.backend.domain.agents.version_repository_postgres import (
        PostgresAgentVersionStore,
    )
    from dashboard.backend.domain.portfolios.repository import PortfolioStore
    from dashboard.backend.domain.portfolios.repository_postgres import (
        PostgresPortfolioStore,
    )
    from dashboard.backend.domain.strategies.repository import StrategyStore
    from dashboard.backend.domain.strategies.repository_postgres import (
        PostgresStrategyStore,
    )
    from dashboard.backend.users import UserStore
    from dashboard.backend.users_postgres import PostgresUserStore

    return [
        (AgentStore, PostgresAgentStore),
        (AgentVersionStore, PostgresAgentVersionStore),
        (PortfolioStore, PostgresPortfolioStore),
        (StrategyStore, PostgresStrategyStore),
        (UserStore, PostgresUserStore),
    ]


@pytest.mark.parametrize(
    "sqlite_cls,postgres_cls",
    _twin_pairs(),
    ids=lambda cls: cls.__name__,
)
def test_postgres_twin_signatures_match_sqlite(sqlite_cls, postgres_cls):
    mismatches = []
    for name in dir(sqlite_cls):
        if name.startswith("_"):
            continue
        sqlite_attr = getattr(sqlite_cls, name)
        if not callable(sqlite_attr) or not hasattr(postgres_cls, name):
            # Twins may expose different helper surfaces; only methods present
            # on BOTH classes are interchangeable-interface claims.
            continue
        sqlite_params = set(inspect.signature(sqlite_attr).parameters)
        postgres_params = set(
            inspect.signature(getattr(postgres_cls, name)).parameters
        )
        if sqlite_params != postgres_params:
            mismatches.append(
                f"{name}: sqlite-only={sorted(sqlite_params - postgres_params)} "
                f"postgres-only={sorted(postgres_params - sqlite_params)}"
            )
    assert not mismatches, (
        f"{postgres_cls.__name__} diverges from {sqlite_cls.__name__} — a kwarg "
        f"missing on the Postgres twin TypeErrors every call on prod:\n"
        + "\n".join(mismatches)
    )
