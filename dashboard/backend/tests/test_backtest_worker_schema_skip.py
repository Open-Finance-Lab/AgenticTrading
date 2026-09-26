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
import ast
import importlib
import re
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


def _twin_sources() -> list[Path]:
    """Every non-test Postgres twin module, as paths."""
    return sorted(
        path for path in _BACKEND.rglob("*_postgres.py") if "tests" not in path.parts
    )


def _schema_calls_in_init(path: Path) -> list[str]:
    """`self.<something with "schema" in it>()` called straight from __init__.

    Parsed, not string-matched. The first version of this guard tested for the
    literal `"self._init_schema()"`, which is a check on one *spelling*: a twin
    that called `self._ensure_schema()` from its constructor passed it while
    doing exactly the thing the guard exists to forbid, and the docstring
    conceded as much. Matching the call shape in the AST covers every name, and
    distinguishes the call `self._init_schema()` from the reference
    `self._init_schema` that the guarded form passes to
    init_schema_unless_worker -- a distinction no substring search can make,
    since the guarded line contains the unguarded one as a prefix.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for klass in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        for init in (
            node
            for node in klass.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ):
            for node in ast.walk(init):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and "schema" in node.func.attr.lower()
                ):
                    offenders.append(
                        f"{path.relative_to(_BACKEND)}:{node.lineno} "
                        f"{klass.name}.__init__ calls self.{node.func.attr}()"
                    )
    return offenders


def test_no_postgres_twin_runs_ddl_outside_the_guard():
    """A twelfth twin must not be able to arrive unguarded.

    Which twins a backtest child constructs is an accident of the import
    graph -- six of the eleven on the graph as it stands -- so the invariant
    worth pinning is not "the six" but "no Postgres twin runs its schema
    method straight from __init__". The twelfth file under this glob,
    value_repository_postgres.py, has no schema method at all.
    """
    offenders = sorted(
        call for path in _twin_sources() for call in _schema_calls_in_init(path)
    )
    assert offenders == [], (
        "these Postgres twins run schema DDL outside "
        "db_url.init_schema_unless_worker, so a backtest child repeats it: "
        f"{offenders}"
    )


def test_a_twin_that_owns_a_schema_method_routes_it_through_the_guard():
    """The rename half: owning schema DDL means naming the helper.

    The call-shape guard above answers "does __init__ call it directly"; it
    says nothing about a twin that stops calling init_schema_unless_worker
    altogether -- inlining the DDL into a helper invoked some other way, or
    quietly dropping the guard during a refactor. Both leave __init__ clean
    and both re-add the cost this flag exists to remove. Keyed on defining a
    schema method rather than on the file list, so a twelfth twin is covered
    the day it is written.
    """
    unguarded = sorted(
        str(path.relative_to(_BACKEND))
        for path in _twin_sources()
        if any(
            isinstance(node, ast.FunctionDef) and "schema" in node.name.lower()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
        and "init_schema_unless_worker" not in path.read_text(encoding="utf-8")
    )
    assert unguarded == [], (
        "these Postgres twins define a schema method but never mention "
        "db_url.init_schema_unless_worker: "
        f"{unguarded}"
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


# Guarded `_init_schema` bodies that also MUTATE DATA, with the reason each is
# safe to skip in a child. Declared rather than discovered, because the whole
# point is that adding a fourth is a decision somebody makes on purpose.
#
# `init_schema_unless_worker` reads as "skip some CREATE TABLE IF NOT EXISTS",
# and for eight of the eleven twins that is all it is. For these three the
# guard also skips a backfill, a seed or a credential scrub -- statements whose
# effect is not idempotent-by-construction the way DDL is, and which a reader
# of the helper's name would not expect to be conditional on a process role.
#
# All three are safe **today for one reason only**: the parent runs them at
# boot before it spawns any child, so a child that skips them observes a
# database they have already been applied to. That is a property of the
# deployment, not of the code -- a worker that ever runs without a parent boot
# ahead of it (a bare CLI run with the flag exported, a worker-only service)
# would silently not apply them.
_DATA_MUTATING_SCHEMA_BODIES = {
    "domain/credits/repository_postgres.py": (
        "seeds the 'default' grant pool and backfills settled reservation "
        "amounts plus ledger bucket/operation columns"
    ),
    "domain/model_providers/repository_postgres.py": (
        "scrubs api_key_enc on revoked credentials, seeds SEEDED_PROVIDERS "
        "and backfills the CommonStack model allowlist"
    ),
    "users_postgres.py": (
        "repairs users.user_group values outside the allowed set"
    ),
}

_MUTATION = re.compile(r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)\b", re.I)


def _mutates_data_in_schema_init(path: Path) -> bool:
    """True when this twin's `_init_schema` runs a non-DDL statement.

    Covers SQL written inline in the method *and* SQL reached through a
    module-level string constant the method executes, because the migrations
    that matter here live in both shapes -- the credits twin keeps its
    backfills in CREDITS_POSTGRES_GRANT_MIGRATION_DDL, a name that says DDL and
    carries three UPDATEs.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_init_schema"):
            continue
        body = ast.get_source_segment(source, node) or ""
        if _MUTATION.search(body):
            return True
        for name in set(re.findall(r"\b[A-Z_][A-Z0-9_]{4,}\b", body)):
            if name in constants and _MUTATION.search(constants[name]):
                return True
    return False


def test_the_guard_skips_exactly_the_declared_data_migrations():
    """The helper's name says DDL; for three twins it also gates data changes.

    Fails in both directions on purpose. A twin that *gains* a backfill inside
    `_init_schema` turns this red until somebody adds it to the registry above
    with a reason -- which is the review the name would otherwise not prompt,
    since nothing about `init_schema_unless_worker(...)` at a call site hints
    that a child will skip a credential scrub. A twin that *loses* one turns it
    red too, so the registry cannot rot into a list of things that used to be
    true.
    """
    detected = {
        str(path.relative_to(_BACKEND)).replace("\\", "/")
        for path in _twin_sources()
        if _mutates_data_in_schema_init(path)
    }
    assert detected == set(_DATA_MUTATING_SCHEMA_BODIES), (
        "the set of guarded _init_schema bodies that mutate data has changed; "
        "add or remove it in _DATA_MUTATING_SCHEMA_BODIES with the reason a "
        "backtest child may skip it.\n"
        f"  detected: {sorted(detected)}\n"
        f"  declared: {sorted(_DATA_MUTATING_SCHEMA_BODIES)}"
    )
