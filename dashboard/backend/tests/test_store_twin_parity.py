"""Parity between each SQLite store and its Postgres twin.

Every dual-backend store is selected by a factory at import time, and the
service layer calls one interface against whichever twin got built. Two
independent ways a twin can diverge both surface only on prod:

* **Call signature.** Callers pass sentinel kwargs (``_UNSET``) on every call,
  so a parameter that exists on the SQLite store but not the Postgres twin is
  not a feature gap -- it is a ``TypeError`` raised before any SQL runs, on
  every call to that method. A method missing from the twin outright is the
  same defect one step worse (``AttributeError``).
* **Table schema.** ``CREATE TABLE IF NOT EXISTS`` silently no-ops once the
  table exists, so a column added to only one twin -- or added to the twin's
  ``CREATE`` but not to an ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` --
  leaves every query naming it raising ``UndefinedColumn``. Both halves are
  checked: the twins must declare the same columns, *and* the Postgres twin
  must repeat every lazy migration the SQLite store performs (declaring a
  column in ``CREATE`` alone reaches a fresh database but never a deployed
  one, which is the failure mode the twin's own header comment warns about).

#227 hit the first axis: it added ``live_trading_enabled`` to
``AgentStore.update_agent`` only, and every agent Configure PATCH on prod
500'd bare (an unhandled exception escapes CORSMiddleware, so browsers
reported it as a CORS block) while the SQLite-backed test suite stayed green.
It hit the second axis too -- the column was missing from the Postgres table
-- which would have been the *next* 500 had the kwarg alone been fixed.

Both checks are static: signatures come from ``inspect``, columns from
parsing the module source. Neither needs a live Postgres, so this tier stays
active where the @pg_only behavioral tier fails open (TEST_POSTGRES_URL
unset -- local dev and any CI lane without the service container;
test_ci_postgres_wired.py asserts CI itself never lands in that state).

Imports happen inside the test bodies, not at collection: the registry below
is plain strings, so an import error here fails one test rather than erroring
collection and aborting the whole session.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import NamedTuple

import pytest

# (sqlite module, sqlite class, postgres module, postgres class)
_TWINS = [
    (
        "dashboard.backend.domain.agents.credential_store",
        "AgentCredentialStore",
        "dashboard.backend.domain.agents.credential_store_postgres",
        "PostgresAgentCredentialStore",
    ),
    (
        "dashboard.backend.domain.agents.repository",
        "AgentStore",
        "dashboard.backend.domain.agents.repository_postgres",
        "PostgresAgentStore",
    ),
    (
        "dashboard.backend.domain.agents.version_repository",
        "AgentVersionStore",
        "dashboard.backend.domain.agents.version_repository_postgres",
        "PostgresAgentVersionStore",
    ),
    (
        "dashboard.backend.domain.brokers.repository",
        "BrokerConnectionStore",
        "dashboard.backend.domain.brokers.repository_postgres",
        "BrokerConnectionStorePostgres",
    ),
    (
        "dashboard.backend.domain.portfolios.repository",
        "PortfolioStore",
        "dashboard.backend.domain.portfolios.repository_postgres",
        "PostgresPortfolioStore",
    ),
    (
        "dashboard.backend.domain.strategies.repository",
        "StrategyStore",
        "dashboard.backend.domain.strategies.repository_postgres",
        "PostgresStrategyStore",
    ),
    (
        "dashboard.backend.users",
        "UserStore",
        "dashboard.backend.users_postgres",
        "PostgresUserStore",
    ),
    (
        "dashboard.backend.database",
        "BacktestDatabase",
        "dashboard.backend.database_postgres",
        "PostgresBacktestDatabase",
    ),
]

_TWIN_IDS = [pg_cls for _, _, _, pg_cls in _TWINS]

# tests/ -> backend/ -> dashboard/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(module_name: str, class_name: str):
    return getattr(importlib.import_module(module_name), class_name)


def _module_source_path(module_name: str) -> Path:
    """Locate a module's source without importing it (or its parents)."""
    return (_REPO_ROOT / Path(*module_name.split("."))).with_suffix(".py")


def test_every_postgres_twin_module_is_registered():
    """The registry above must not silently stop covering new twins.

    A parity guard that quietly skips a store is worth less than no guard,
    because the green run reads as "all twins checked". This caught
    domain/brokers, which shipped uncovered.

    Discovery keys on the ``*_postgres.py`` filename convention every twin
    follows; a twin named some other way still needs adding by hand.
    """
    backend = _REPO_ROOT / "dashboard" / "backend"
    on_disk = {
        ".".join(path.relative_to(_REPO_ROOT).with_suffix("").parts)
        for path in backend.rglob("*_postgres.py")
        if "tests" not in path.parts
    }
    assert on_disk, f"no *_postgres.py modules found under {backend}"

    registered = {postgres_mod for _, _, postgres_mod, _ in _TWINS}
    unregistered = sorted(on_disk - registered)

    assert not unregistered, (
        "Postgres twin module(s) not covered by the parity tests in this file. "
        "Add each to _TWINS with its SQLite counterpart, or the twin ships "
        f"with no drift guard at all: {unregistered}"
    )


# --------------------------------------------------------------------------
# Axis 1: call signatures
# --------------------------------------------------------------------------


def _public_methods(cls) -> list[str]:
    names = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if callable(getattr(cls, name, None)):
            names.append(name)
    return sorted(names)


def _default_token(default) -> str:
    """Comparable stand-in for a parameter default.

    Literals compare by value, so ``limit=50`` vs ``limit=100`` is caught.
    Everything else collapses to its type name, so two module-level sentinels
    (``_UNSET = object()``) compare equal instead of by memory address.
    """
    if default is inspect.Parameter.empty:
        return "<required>"
    if isinstance(default, (bool, int, float, str, bytes, type(None))):
        return repr(default)
    return f"<{type(default).__name__}>"


def _signature_shape(cls, name: str) -> list[tuple[str, str, str]]:
    """Ordered (name, kind, default) triples -- order and kind are part of it."""
    return [
        (p.name, p.kind.name, _default_token(p.default))
        for p in inspect.signature(getattr(cls, name)).parameters.values()
    ]


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_postgres_twin_exposes_every_sqlite_method(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """A twin may add helpers; it may never *drop* one the service calls.

    The reverse direction is deliberately allowed: Postgres-only helpers
    (pool plumbing, dialect shims) are never reached through the shared
    interface, so they cannot break a caller.
    """
    sqlite_type = _load(sqlite_mod, sqlite_cls)
    postgres_type = _load(postgres_mod, postgres_cls)

    missing = [n for n in _public_methods(sqlite_type) if not hasattr(postgres_type, n)]

    assert not missing, (
        f"{postgres_cls} is missing {len(missing)} method(s) that exist on "
        f"{sqlite_cls}; the factory hands either one to the same callers, so "
        f"each is an AttributeError on prod: {missing}"
    )


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_postgres_twin_signatures_match_sqlite(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    sqlite_type = _load(sqlite_mod, sqlite_cls)
    postgres_type = _load(postgres_mod, postgres_cls)

    mismatches = []
    for name in _public_methods(sqlite_type):
        if not hasattr(postgres_type, name):
            continue  # reported by the missing-method test above
        sqlite_shape = _signature_shape(sqlite_type, name)
        postgres_shape = _signature_shape(postgres_type, name)
        if sqlite_shape != postgres_shape:
            mismatches.append(
                f"  {name}\n"
                f"    sqlite:   {sqlite_shape}\n"
                f"    postgres: {postgres_shape}"
            )

    assert not mismatches, (
        f"{postgres_cls} diverges from {sqlite_cls}. Parameter name, order, "
        f"kind and default must all match -- callers always pass optional "
        f"kwargs by name, so any difference TypeErrors or silently changes "
        f"behaviour on prod:\n" + "\n".join(mismatches)
    )


# --------------------------------------------------------------------------
# Axis 2: table schemas, parsed from source (no live Postgres needed)
# --------------------------------------------------------------------------

_EXPR = "__EXPR__"  # stands in for an f-string interpolation

_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.IGNORECASE,
)
_ADD_COLUMN = re.compile(
    r"ALTER\s+TABLE\s+([A-Za-z_][A-Za-z0-9_]*)\s+ADD\s+COLUMN\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
# Tables a Postgres twin deliberately never creates, keyed by twin class name.
# The default is that both twins declare the same tables -- a divergence is
# normally the #227 bug -- so every entry needs its reason recorded here, and
# the guard below checks each exempted table really exists on the SQLite side
# so a stale or misspelled name cannot quietly widen the exemption.
_DELIBERATELY_POSTGRES_ABSENT_TABLES: dict[str, set[str]] = {
    # idempotency_keys is the *hot* v2 table: every decision submission reads
    # and writes it. PostgresBacktestDatabase keeps that half local, delegating
    # get_idempotency/put_idempotency to an embedded SQLite BacktestDatabase so
    # a per-step agent request never gains a network round-trip. The table is
    # therefore never created in Postgres, by design -- see the module
    # docstring of dashboard/backend/database_postgres.py. Adding a CREATE the
    # twin never executes, purely to satisfy this assertion, would be worse:
    # the guard would then be reading a claim rather than the schema.
    "PostgresBacktestDatabase": {"idempotency_keys"},
}

# Definitions opening with one of these describe a table constraint, not a column.
_CONSTRAINT_KEYWORDS = {
    "primary",
    "foreign",
    "unique",
    "check",
    "constraint",
    "exclude",
    "like",
}


def _string_literals(source: str) -> list[str]:
    """Every string literal in a module, with f-strings reassembled.

    Adjacent plain literals are folded by the parser, so a statement split
    across source lines arrives as one string. f-strings become one JoinedStr
    whose interpolations collapse to a placeholder -- enough to read the
    column name, which is never interpolated.
    """
    tree = ast.parse(source)

    nested = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            nested.update(id(inner) for inner in ast.walk(node) if inner is not node)

    literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            literals.append(
                "".join(
                    part.value
                    if isinstance(part, ast.Constant) and isinstance(part.value, str)
                    else _EXPR
                    for part in node.values
                )
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in nested
        ):
            literals.append(node.value)
    return literals


def _skip_quoted(text: str, i: int) -> int:
    """Index just past the single-quoted literal starting at ``i`` ('' escapes)."""
    n = len(text)
    i += 1
    while i < n:
        if text[i] == "'":
            if i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _balanced_body(text: str, open_paren: int) -> str | None:
    """Text between ``open_paren`` and its match, ignoring quotes and comments."""
    depth = 0
    i, n = open_paren, len(text)
    while i < n:
        if text[i] == "'":
            i = _skip_quoted(text, i)
            continue
        if text.startswith("--", i):
            newline = text.find("\n", i)
            i = n if newline == -1 else newline
            continue
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
        i += 1
    return None


def _split_definitions(body: str) -> list[str]:
    """Split a CREATE TABLE body on top-level commas only.

    Quote-aware because a default can contain commas (the ``scopes`` default
    is a comma-separated scope list), paren-aware because of
    ``REFERENCES users(id)``, and comment-aware because of ``--`` notes.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    i, n = 0, len(body)
    while i < n:
        char = body[i]
        if char == "'":
            end = _skip_quoted(body, i)
            buf.append(body[i:end])
            i = end
            continue
        if body.startswith("--", i):
            newline = body.find("\n", i)
            i = n if newline == -1 else newline
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(char)
        i += 1
    parts.append("".join(buf))
    return [part.strip() for part in parts if part.strip()]


def _column_names(body: str) -> set[str]:
    columns = set()
    for definition in _split_definitions(body):
        # Split on whitespace *or* an opening paren. A constraint written with
        # no space before its paren -- ``UNIQUE(run_id, timestamp)``, which
        # database.py writes -- would otherwise yield the first token
        # "UNIQUE(run_id," and be recorded as a column, so the same constraint
        # spelled with and without a space would read as schema drift.
        first = re.split(r"[\s(]", definition, maxsplit=1)[0]
        if first.lower() in _CONSTRAINT_KEYWORDS:
            continue
        columns.add(first.strip('"').lower())
    return columns


class _Schema(NamedTuple):
    #: columns per table as a fresh database ends up: CREATE union every ALTER
    declared: dict[str, set[str]]
    #: columns per table reachable by an *existing* table: ALTER only
    migrated: dict[str, set[str]]


def _parse_ddl(source: str) -> _Schema:
    declared: dict[str, set[str]] = {}
    migrated: dict[str, set[str]] = {}
    for literal in _string_literals(source):
        for match in _CREATE_TABLE.finditer(literal):
            body = _balanced_body(literal, match.end() - 1)
            if body is None:
                continue
            declared.setdefault(match.group(1).lower(), set()).update(
                _column_names(body)
            )
        for match in _ADD_COLUMN.finditer(literal):
            table, column = match.group(1).lower(), match.group(2).lower()
            declared.setdefault(table, set()).add(column)
            migrated.setdefault(table, set()).add(column)
    return _Schema(declared, migrated)


def test_ddl_parser_extracts_columns_from_tricky_sql():
    """Guards the comparisons below from passing vacuously.

    The fixture is written against SQL shapes, not against this repo's field
    names, so it cannot drift into agreement with the code it checks. Every
    hazard here is one the real store modules actually contain.
    """
    source = '''
def _init_schema(self):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS widgets (
            widget_id TEXT PRIMARY KEY,
            -- a SQL comment, with a comma, that must not become a column
            tags TEXT NOT NULL DEFAULT 'alpha,beta,gamma',
            owner_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            label TEXT,
            PRIMARY KEY (widget_id, label),
            FOREIGN KEY (owner_id) REFERENCES people(id),
            -- no space before the paren: database.py spells its natural key
            -- this way while the Postgres twin spells it with a space, and
            -- reading either as a column invents drift out of formatting
            UNIQUE(widget_id, tags),
            CHECK(owner_id > 0)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_widgets_owner ON widgets(owner_id)
        """
    )
    cur.execute(
        "ALTER TABLE widgets "
        "ADD COLUMN IF NOT EXISTS retired BOOLEAN NOT NULL DEFAULT FALSE"
    )
    cur.execute(
        "ALTER TABLE widgets "
        f"ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT '{DEFAULT_MODE}'"
    )
    cur.execute("ALTER TABLE widgets ADD COLUMN note TEXT")
'''
    schema = _parse_ddl(source)
    assert schema.declared == {
        "widgets": {
            "widget_id",
            "tags",
            "owner_id",
            "label",
            "retired",
            "mode",
            "note",
        }
    }
    # The ALTER-only view must not absorb the CREATE's columns: the check that
    # a deployed table still gains new columns depends on the two being distinct.
    assert schema.migrated == {"widgets": {"retired", "mode", "note"}}


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_postgres_twin_schema_columns_match_sqlite(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """The column half of the #227 bug, which signature parity cannot see.

    Compares column *names* only: types legitimately differ per dialect
    (REAL/DOUBLE PRECISION, INTEGER/BOOLEAN, TIMESTAMP/TEXT).
    """
    sqlite_path = _module_source_path(sqlite_mod)
    postgres_path = _module_source_path(postgres_mod)
    assert sqlite_path.is_file(), f"twin registry points at a missing {sqlite_path}"
    assert postgres_path.is_file(), f"twin registry points at a missing {postgres_path}"

    sqlite_schema = _parse_ddl(sqlite_path.read_text(encoding="utf-8")).declared
    postgres_schema = _parse_ddl(postgres_path.read_text(encoding="utf-8")).declared

    # Non-vacuity: a parser that silently extracted nothing would agree with
    # itself on every pair and report no drift forever.
    assert sqlite_schema, f"no CREATE TABLE parsed from {sqlite_path}"
    assert postgres_schema, f"no CREATE TABLE parsed from {postgres_path}"
    for table, columns in (*sqlite_schema.items(), *postgres_schema.items()):
        assert len(columns) >= 2, (
            f"suspiciously empty parse of {table}: {columns}. A table or "
            f"column named {_EXPR.lower()} means DDL was assembled with an "
            f"f-string, which this source-text parser cannot see through -- "
            f"write the ALTER/CREATE as a literal string instead."
        )

    # Deliberate divergences are narrowed here, never by deleting the assert.
    exempt = _DELIBERATELY_POSTGRES_ABSENT_TABLES.get(postgres_cls, set())
    stale_exemptions = sorted(exempt - set(sqlite_schema))
    assert not stale_exemptions, (
        f"_DELIBERATELY_POSTGRES_ABSENT_TABLES exempts table(s) that "
        f"{sqlite_cls} does not declare, so the exemption is obsolete or "
        f"misspelled and is silently widening this guard: {stale_exemptions}"
    )
    expected_tables = set(sqlite_schema) - exempt

    assert expected_tables == set(postgres_schema), (
        f"{postgres_cls} and {sqlite_cls} declare different tables -- "
        f"sqlite-only={sorted(expected_tables - set(postgres_schema))} "
        f"postgres-only={sorted(set(postgres_schema) - expected_tables)}"
        + (f" (exempted by design: {sorted(exempt)})" if exempt else "")
    )

    drift = []
    for table in sorted(expected_tables):
        sqlite_columns = sqlite_schema[table]
        postgres_columns = postgres_schema[table]
        if sqlite_columns != postgres_columns:
            drift.append(
                f"  {table}: "
                f"sqlite-only={sorted(sqlite_columns - postgres_columns)} "
                f"postgres-only={sorted(postgres_columns - sqlite_columns)}"
            )

    assert not drift, (
        f"{postgres_cls} and {sqlite_cls} declare different columns. A column "
        f"present on one twin only makes every query naming it raise on the "
        f"other -- and adding it to the Postgres CREATE TABLE alone is not "
        f"enough, since CREATE TABLE IF NOT EXISTS no-ops on the deployed "
        f"table: it needs an ALTER TABLE ... ADD COLUMN IF NOT EXISTS too. If "
        f"a divergence is ever deliberate, narrow this assertion explicitly "
        f"rather than deleting it:\n" + "\n".join(drift)
    )


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_postgres_twin_repeats_every_sqlite_lazy_migration(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """Column sets agreeing is not enough -- the twin must also *migrate*.

    A column added only to the Postgres CREATE TABLE passes the comparison
    above (both twins declare it) yet never reaches a deployed table, because
    CREATE TABLE IF NOT EXISTS no-ops once the table exists. Prod then raises
    UndefinedColumn while a fresh test database looks perfect.

    The SQLite store's ALTERs are the ground truth for "added after the
    original schema shipped": needing a lazy migration there means deployed
    tables predate the column, so the Postgres deployment needs one too.
    Extra Postgres ADD COLUMNs are fine -- they no-op on a current table.
    """
    sqlite_migrated = _parse_ddl(
        _module_source_path(sqlite_mod).read_text(encoding="utf-8")
    ).migrated
    postgres_migrated = _parse_ddl(
        _module_source_path(postgres_mod).read_text(encoding="utf-8")
    ).migrated

    gaps = []
    for table, columns in sorted(sqlite_migrated.items()):
        missing = columns - postgres_migrated.get(table, set())
        if missing:
            gaps.append(f"  {table}: {sorted(missing)}")

    assert not gaps, (
        f"{sqlite_cls} lazily adds columns that {postgres_cls} never adds to an "
        f"existing table. Declaring them in the Postgres CREATE TABLE alone is "
        f"not enough -- add `ALTER TABLE <t> ADD COLUMN IF NOT EXISTS <c> ...` "
        f"beside the others in its _init_schema:\n" + "\n".join(gaps)
    )
