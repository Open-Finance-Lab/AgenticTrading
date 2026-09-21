"""A dashboard backtest child must not repeat the parent's schema DDL.

The parent runs every store's `_init_schema` at import, so the schema exists
before it spawns anything. A child that ran it again paid a pooled Neon
checkout plus a batch of DDL round trips per store before reading a single
bar -- pure start-up waste, and invisible because nothing before the bar loop
was measured. The flag names the process role, not the DDL, so no operator
sets it globally and later child-only behaviour has a home.

Eleven twins, not the six a child imports today: which stores a child
constructs is an accident of the import graph, so the invariant worth pinning
is "no Postgres twin runs DDL outside the guard", not "these six".
"""
import importlib
from pathlib import Path

import pytest

from dashboard.backend import db_url

_URL = "postgresql://atl:not-a-secret@db.example.invalid:5432/atl_test"
_BACKEND = Path(__file__).resolve().parents[1]

# (module, class, the label that store's factory already prints in its
# "<label> backend: postgres (...)" boot line). Imported inside the test
# bodies, not here: the registry is plain strings, so an import error fails
# one case instead of erroring collection and aborting the session -- the
# convention test_store_twin_parity.py documents for the same reason.
_TWINS = [
    ("dashboard.backend.database_postgres", "PostgresBacktestDatabase", "run history"),
    ("dashboard.backend.users_postgres", "PostgresUserStore", "user_store"),
    ("dashboard.backend.domain.agents.repository_postgres", "PostgresAgentStore", "agent_store"),
    ("dashboard.backend.domain.agents.version_repository_postgres", "PostgresAgentVersionStore", "agent_version_store"),
    ("dashboard.backend.domain.agents.credential_store_postgres", "PostgresAgentCredentialStore", "agent_credential_store"),
    ("dashboard.backend.domain.analytics.repository_postgres", "PostgresAnalyticsStore", "analytics_store"),
    ("dashboard.backend.domain.brokers.repository_postgres", "BrokerConnectionStorePostgres", "broker_connections"),
    ("dashboard.backend.domain.credits.repository_postgres", "PostgresCreditsStore", "credits_store"),
    ("dashboard.backend.domain.model_providers.repository_postgres", "PostgresModelProviderStore", "model_provider_store"),
    ("dashboard.backend.domain.portfolios.repository_postgres", "PostgresPortfolioStore", "portfolio_store"),
    ("dashboard.backend.domain.strategies.repository_postgres", "PostgresStrategyStore", "strategy_store"),
]
_IDS = [name for _m, name, _l in _TWINS]


def test_only_the_literal_one_arms_the_flag(monkeypatch):
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)
    assert db_url.schema_init_skipped() is False
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")
    assert db_url.schema_init_skipped() is True
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "true")
    assert db_url.schema_init_skipped() is False


@pytest.mark.parametrize(("module", "name", "label"), _TWINS, ids=_IDS)
def test_a_worker_skips_schema_init(monkeypatch, capsys, module, name, label):
    calls = []
    cls = getattr(importlib.import_module(module), name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")

    cls(_URL)

    assert calls == []
    # The label is the token that store's factory already prints in its
    # "<label> backend: postgres (...)" line, so one grep finds both halves
    # of a store's start-up story.
    assert f"{label} backend: schema init skipped (backtest worker)" in capsys.readouterr().out


@pytest.mark.parametrize(("module", "name", "label"), _TWINS, ids=_IDS)
def test_the_parent_still_runs_schema_init(monkeypatch, capsys, module, name, label):
    calls = []
    cls = getattr(importlib.import_module(module), name)
    monkeypatch.setattr(cls, "_init_schema", lambda self: calls.append("ddl"))
    monkeypatch.delenv(db_url.BACKTEST_WORKER_ENV, raising=False)

    cls(_URL)

    assert calls == ["ddl"]
    # Timed, not merely run. This line in the parent's Render boot log is the
    # only pre-change number the deploy that removes the child's copy can still
    # produce -- the parent runs the same DDL against the same databases and is
    # never a worker.
    out = capsys.readouterr().out
    assert f"{label} backend: schema init " in out and "skipped" not in out


def test_a_worker_accumulates_no_schema_time(monkeypatch):
    """The accumulator is the measurement, so its zero has to be a real zero.

    `starting.schema_init_seconds` is what tells the Final verification table
    how much of the start was DDL. If a skipped init still charged time, the
    prod run's 0.0 would stop being evidence that the flag fired. Read as a
    delta, not an absolute: the counter is process-global and never resets.
    """
    monkeypatch.setenv(db_url.BACKTEST_WORKER_ENV, "1")
    before = db_url.schema_init_seconds()
    for module, name, _label in _TWINS:
        cls = getattr(importlib.import_module(module), name)
        monkeypatch.setattr(cls, "_init_schema", lambda self: None)
        cls(_URL)
    assert db_url.schema_init_seconds() == before


def test_no_postgres_twin_runs_ddl_outside_the_guard():
    """A twelfth twin must not be able to arrive unguarded.

    Which twins a backtest child constructs is an accident of the import
    graph -- six of the eleven on the graph as it stands -- so the invariant
    worth pinning is not "the six" but "no Postgres twin calls _init_schema
    from __init__ directly". Eleven non-test modules matched before this
    change and none may after. The twelfth file under this glob,
    value_repository_postgres.py, has no _init_schema at all.
    """
    offenders = sorted(
        str(path.relative_to(_BACKEND))
        for path in _BACKEND.rglob("*_postgres.py")
        if "tests" not in path.parts
        and "self._init_schema()" in path.read_text(encoding="utf-8")
    )
    assert offenders == [], (
        "these Postgres twins run schema DDL outside "
        "db_url.init_schema_unless_worker, so a backtest child repeats it: "
        f"{offenders}"
    )


def test_the_guarded_list_accounts_for_every_twin():
    """Guarded + deliberately-exempt must equal the parity registry.

    The sentence this plan writes into CLAUDE.md claims a scope. A claim
    about "every twin" that is not checked against the list of twins is how
    the four-versus-twelve error got written down in the first place.

    The exempt half is read from test_store_twin_parity's own
    _NO_OWN_DDL_TWINS rather than restated here: that registry already names
    PostgresValueAnalyticsStore with the reason (it composes the other
    stores), and a second copy of the exemption is a second owner that can
    disagree with the first. Imported in the body, not at module scope, so a
    rename there fails this one case instead of aborting collection.
    """
    from dashboard.backend.tests.test_store_twin_parity import (
        _NO_OWN_DDL_TWINS,
        _TWIN_IDS,
    )

    assert set(_IDS) | set(_NO_OWN_DDL_TWINS) == set(_TWIN_IDS)
    assert not (set(_IDS) & set(_NO_OWN_DDL_TWINS)), (
        "a twin cannot be both guarded and exempt from owning DDL"
    )
