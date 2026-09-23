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
* **Index order.** The same no-op bites a ``CREATE INDEX`` that names a
  column the twin only adds by ``ALTER TABLE`` further down: on a deployed
  table the index runs first and raises ``UndefinedColumn`` at import (#432
  killed the Render boot this way). Each index must sit below every ADD
  COLUMN it depends on.

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

from dashboard.backend.domain.user_groups import DEFAULT_USER_GROUP, USER_GROUPS

# (sqlite module, sqlite class, postgres module, postgres class)
_TWINS = [
    (
        "dashboard.backend.domain.analytics.repository",
        "AnalyticsStore",
        "dashboard.backend.domain.analytics.repository_postgres",
        "PostgresAnalyticsStore",
    ),
    (
        "dashboard.backend.domain.model_providers.repository",
        "ModelProviderStore",
        "dashboard.backend.domain.model_providers.repository_postgres",
        "PostgresModelProviderStore",
    ),
    (
        "dashboard.backend.domain.credits.repository",
        "CreditsStore",
        "dashboard.backend.domain.credits.repository_postgres",
        "PostgresCreditsStore",
    ),
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
    (
        "dashboard.backend.domain.analytics.value_repository",
        "ValueAnalyticsStore",
        "dashboard.backend.domain.analytics.value_repository_postgres",
        "PostgresValueAnalyticsStore",
    ),
]

_TWIN_IDS = [pg_cls for _, _, _, pg_cls in _TWINS]


def test_twins_registry_has_no_duplicate_pairs():
    """`_TWINS` listed the AnalyticsStore pair twice (the tuple that sat
    between `StrategyStore` and `UserStore`).

    Harmless today -- both instances of a duplicate tuple pass or fail
    together -- but it is the exact list PR T's absence-direction check and
    every future twin extends, so a reader counting entries gets 12 when
    there are 11 distinct stores. `_TWIN_IDS` uses the postgres class name
    as the parametrize id, so a duplicate also means two test instances
    sharing one id in every parametrized case above.
    """
    assert len(_TWINS) == len(set(_TWINS)), (
        f"_TWINS has {len(_TWINS) - len(set(_TWINS))} duplicate tuple(s); "
        "each store pair belongs in the registry exactly once."
    )


def test_value_analytics_store_pair_is_registered():
    """PR T's whole point: a store with no *_postgres.py file is invisible
    to test_every_postgres_twin_module_is_registered, which starts from
    files on disk. This asserts the registry side directly.
    """
    assert (
        "dashboard.backend.domain.analytics.value_repository",
        "ValueAnalyticsStore",
        "dashboard.backend.domain.analytics.value_repository_postgres",
        "PostgresValueAnalyticsStore",
    ) in _TWINS


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
# Absence-direction check: a dialect branch outside a registered twin
# --------------------------------------------------------------------------
#
# The registry above and the rglob discovery test are both existence-first:
# they start from a *_postgres.py file or a _TWINS entry and check it is
# complete. Neither can catch a class that branches SQLite-vs-Postgres
# inline -- exactly what ValueAnalyticsStore did until PR T -- because such
# a class owns no *_postgres.py file to discover. This test starts from the
# other end: every occurrence of the two known dialect-branch idioms --
# `is_postgres`, and a `hasattr` check for `database_url` -- across every
# non-test module under dashboard/backend. Each hit is a twin extraction
# that has not happened yet, a non-twin helper reading a table a registered
# twin already owns, or a deliberate branch on some *other* store's dialect.
# All three need a name below with a reason -- silence here is exactly how
# ValueAnalyticsStore's 21 branches went unnoticed for as long as they did.
#
# Registered *_postgres.py twins are scanned too, and deliberately so. They
# were exempt until the fix for PR #498's review: being a registered twin
# says the file is the Postgres *side* of a pair, which is no reason for it
# to contain a dialect *branch* -- a well-formed twin just writes `%s` and
# tests nothing. The exemption meant value_repository_postgres.py's own
# branches needed no reason while the identical ones in its SQLite twin got
# a long one, and it left the single blind spot this whole test exists to
# close: a genuinely missing extraction added inside a registered twin.
#
# The match is textual, not semantic, and that limit is worth stating rather
# than leaving a reader to assume otherwise: a branch written a third way --
# an `isinstance(store, PostgresAnalyticsStore)` test, a
# `getattr(store, "database_url", None)` default -- evades this scan. Both
# matched forms are what every dialect branch in the tree uses today, so
# this guard covers the established idiom; it is not a proof that no
# dialect branch can ever hide again.
#
# The allowlist below is keyed by *file*, which bounds it the same way: a
# new dialect branch added to a file that already has an entry trips
# neither assertion -- not `unlisted`, because the file is listed, and not
# `stale`, because the file still has hits. Only a branch in an unlisted
# file is caught. A per-entry hit count would close that and is deliberately
# not used: it churns on every unrelated edit to these files, so it would be
# updated reflexively and stop meaning anything. What each reason can do
# instead is name the branching symbols, as `backfill.py`'s does.
_DIALECT_BRANCH_PATTERN = re.compile(
    r"is_postgres|hasattr\([^)]*[\"']database_url[\"']"
)

_DIALECT_BRANCH_ALLOWLIST: dict[str, str] = {
    "dashboard/backend/domain/analytics/value_repository.py": (
        "list_commercial_values and list_credit_activity branch on the "
        "injected credits_base's own dialect (hasattr(self.credits_base, "
        "'database_url')), not on ValueAnalyticsStore's -- a caller can (and "
        "in tests does) pair either analytics_base with either credits_base. "
        "PostgresValueAnalyticsStore carries the identical branch for the "
        "same reason (see value_repository_postgres.py's module docstring); "
        "both are correct, not a missing extraction. A third hit lives in "
        "build_value_analytics_store's own selection branch "
        "(hasattr(resolved_analytics_base, 'database_url')): that factory's "
        "whole job is to pick SQLite vs. Postgres, and it must do so by "
        "reading the resolved analytics_base object rather than an "
        "os.getenv(...) check, because it receives an already-constructed "
        "base and must follow that object's dialect -- reading the "
        "environment instead would hand a caller who injects a Postgres "
        "base the SQLite twin whenever the var happens to be unset. That "
        "makes this a dialect selection, the same idiom every other "
        "store's _build_*_store() performs on os.getenv(...) directly, not "
        "a missing extraction. A fourth hit is ValueAnalyticsStore.__init__'s "
        "guard (PR #498 review), which refuses a Postgres analytics_base "
        "rather than silently emitting `?` placeholders against it; it reads "
        "the same attribute the factory dispatches on, on purpose, so guard "
        "and factory cannot disagree about which twin a base belongs to."
    ),
    "dashboard/backend/domain/analytics/value_repository_postgres.py": (
        "The Postgres twin's two hits are the same credits_base branch its "
        "SQLite counterpart carries (list_commercial_values and "
        "list_credit_activity read the *injected* credits store's dialect, "
        "not this class's -- see the module docstring), plus "
        "PostgresValueAnalyticsStore.__init__'s mirror of the guard "
        "described in the value_repository.py entry above: it refuses an "
        "analytics_base with no database_url, because `or analytics_store` "
        "otherwise resolves to the SQLite singleton under pytest and every "
        "`%s` query in the file would reach sqlite3. None is a missing "
        "extraction: this file *is* the extraction."
    ),
    "dashboard/backend/domain/analytics/states.py": (
        "AnalyticsStateStore dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for the legacy "
        "user_analytics_snapshots table, not a store of its own -- the "
        "branched methods also read users (get_user; that table belongs to "
        "another registered twin, UserStore/PostgresUserStore, so this is a "
        "cross-domain read, not a missing extraction) and, via "
        "list_stale_user_ids, analytics_subject_settings and "
        "user_lifecycle_daily_snapshots. Admin layer redesign PR A/PR B "
        "(docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md "
        "§6.5) deletes the five-state columns and repair path this class "
        "serves, which removes or shrinks this branch -- tracked there, not "
        "in PR T."
    ),
    "dashboard/backend/domain/analytics/rollups.py": (
        "AnalyticsRollupStore dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for "
        "analytics_daily_rollups, a table that pair already declares and "
        "parity-checks (repository.py:100-117 / repository_postgres.py:96-113). "
        "Not a store of its own; out of scope for PR T."
    ),
    "dashboard/backend/domain/analytics/query_service.py": (
        "AnalyticsQueryStore dialect-branches over the already-twinned "
        "AnalyticsStore/PostgresAnalyticsStore base_store for the legacy "
        "overview/users query surface. Note the class: AnalyticsQueryService, "
        "further down the same file, is a wrapper holding an "
        "AnalyticsQueryStore and does not itself branch -- an auditor "
        "grepping for the service name will not find the branch. Not a store "
        "of its own; that surface is rewritten in admin layer redesign PR A/"
        "PR B (design doc §6.10), which is where this branch is next touched."
    ),
    "dashboard/backend/domain/analytics/lifecycle_backfill.py": (
        "LifecycleBackfillSource dialect-branches over the already-twinned "
        "analytics_base for the historical 8-week lifecycle reconstruction "
        "job. Not a store of its own; admin layer redesign PR A's daily job "
        "and migration (design doc §6.9) replace this reconstruction path."
    ),
    "dashboard/backend/domain/analytics/backfill.py": (
        "_query_all dialect-branches over injected credits/agent stores for "
        "the authoritative-history backfill; admin layer redesign PR A "
        "moves those two ledger and run reads onto CreditsStore/"
        "PostgresCreditsStore and the run-history store (design doc §12 "
        "item 1), which removes this branch. _existing_source_event_ids "
        "separately dialect-branches over the already-twinned analytics_store "
        "to deduplicate against analytics_events, the analytics domain's own "
        "table -- a different case, out of scope for PR T."
    ),
}


def test_dialect_branches_outside_a_registered_twin_are_allowlisted():
    """A store that inlines SQLite/Postgres branching owns no *_postgres.py
    file, so test_every_postgres_twin_module_is_registered above cannot see
    it -- that test starts from files on disk, and there is no second file
    for a branch like this. This test starts from the branch instead, across
    every non-test module under ``dashboard/backend``, and requires each hit
    to be named here.

    ValueAnalyticsStore had 21 such branches until PR T split it into a real
    twin. Nothing before this test would have caught it going in, and
    nothing would catch the next one either.

    The name says "outside a registered twin" for the case that motivated
    it; the scan itself no longer excludes registered ``*_postgres.py``
    twins (see the note above the allowlist), so a branch inside one needs
    an entry exactly like a branch anywhere else.
    """
    backend = _REPO_ROOT / "dashboard" / "backend"
    hits: set[str] = set()
    for path in backend.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if _DIALECT_BRANCH_PATTERN.search(source):
            hits.add(str(path.relative_to(_REPO_ROOT)))

    unlisted = sorted(hits - set(_DIALECT_BRANCH_ALLOWLIST))
    assert not unlisted, (
        "Dialect-branch idiom (is_postgres / hasattr(*, 'database_url')) "
        "found with no allowlist entry. Registered *_postgres.py twins are "
        "scanned too: being the Postgres side of a pair is not a reason to "
        "branch on a dialect. Either extract a real Postgres twin and add "
        "it to _TWINS, or add this file to _DIALECT_BRANCH_ALLOWLIST with a "
        f"reason: {unlisted}"
    )

    stale = sorted(set(_DIALECT_BRANCH_ALLOWLIST) - hits)
    assert not stale, (
        "_DIALECT_BRANCH_ALLOWLIST names file(s) with no dialect-branch "
        "idiom left in them -- the exemption is stale and silently widens "
        f"this guard for whatever gets added there next: {stale}"
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

# A bare or double-quoted identifier, optionally schema-qualified. Matching only
# the bare form (as this did until the #433 review) is not a narrower guard, it
# is a *silent* one: `ON public.t(a)` and `ON "t"(a)` simply do not match, so a
# twin written that way would sail past the ordering check below with zero
# indexes parsed and nothing to show for it. _CREATE_INDEX_KEYWORD /
# test_ddl_parser_sees_every_create_index exist to make that failure loud.
_SQL_IDENT = r'(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)'
_SQL_QUALIFIED = rf"(?:{_SQL_IDENT}\s*\.\s*)*({_SQL_IDENT})"
_CREATE_INDEX_KEYWORD = re.compile(r"CREATE\s+(?:UNIQUE\s+)?INDEX\b", re.IGNORECASE)

_CREATE_TABLE = re.compile(
    rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_SQL_QUALIFIED}\s*\(",
    re.IGNORECASE,
)
_ADD_COLUMN = re.compile(
    rf"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{_SQL_QUALIFIED}"
    rf"\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?{_SQL_QUALIFIED}",
    re.IGNORECASE,
)
_CREATE_INDEX = re.compile(
    rf"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?"
    rf"(?:IF\s+NOT\s+EXISTS\s+)?{_SQL_QUALIFIED}\s+ON\s+(?:ONLY\s+)?"
    rf"{_SQL_QUALIFIED}\s*(?:USING\s+[A-Za-z_]+\s*)?\(",
    re.IGNORECASE,
)


def _name(raw: str) -> str:
    """Fold one captured identifier to its comparison key.

    Postgres treats a quoted identifier as case-sensitive and an unquoted one
    as folded to lower case, so ``"T"`` and ``T`` really are different tables.
    This guard collapses them anyway: over-matching makes it flag an ordering
    it should not (loud, and fixable), while under-matching makes it miss the
    #432 boot crash (silent, and shipped).
    """
    return raw.strip().strip('"').lower()
_SQL_STRING = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Twins whose class owns no table of its own: it reads and writes tables
# declared by a *different*, already-registered twin through an injected
# connection, rather than declaring CREATE TABLE/CREATE INDEX itself. For
# such a pair, test_postgres_twin_schema_columns_match_sqlite's non-vacuity
# assert ("no CREATE TABLE parsed from ...") would fail on every entry, not
# because a column drifted but because there is nothing to parse -- exactly
# the false positive that assert exists to prevent for a store that *does*
# own tables. Each entry names which already-registered pair actually owns
# the schema, so this cannot silently swallow a future twin that adds its
# own DDL. The other three DDL-adjacent tests in this file
# (test_ddl_parser_sees_every_create_index,
# test_postgres_twin_indexes_a_migrated_column_only_after_adding_it,
# test_postgres_twin_repeats_every_sqlite_lazy_migration) need no matching
# entry: each is vacuously true for zero DDL statements by construction, and
# that vacuity is not a blind spot the way the non-vacuity assert would be.
_NO_OWN_DDL_TWINS: dict[str, str] = {
    "PostgresValueAnalyticsStore": (
        "ValueAnalyticsStore/PostgresValueAnalyticsStore own no CREATE TABLE "
        "or CREATE INDEX: both read and write user_analytics_snapshots, "
        "user_lifecycle_daily_snapshots and analytics_projection_jobs "
        "through an injected analytics_base connection, and that base is "
        "AnalyticsStore/PostgresAnalyticsStore -- already a registered pair "
        "above, whose own schema-column test covers those tables."
    ),
}

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


def _blank_sql_comments(literal: str) -> str:
    """``literal`` with every ``--`` comment replaced by spaces, same length.

    Two reasons this is not cosmetic. A commented-out ``CREATE INDEX`` would
    otherwise be read as real DDL, and prose inside a comment ("CREATE INDEX
    IF NOT EXISTS matches by name ...", which the credits twin really does
    say) would be counted as a statement the parser failed to understand.
    Length is preserved so every offset the callers compare stays valid.
    """
    out = list(literal)
    i, n = 0, len(literal)
    while i < n:
        if literal[i] == "'":
            i = _skip_quoted(literal, i)
            continue
        if literal.startswith("--", i):
            while i < n and literal[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _ddl_literals(source: str) -> list[str]:
    """``_string_literals`` with SQL comments blanked -- the SQL readers' view."""
    return [_blank_sql_comments(literal) for literal in _string_literals(source)]


def _string_literals(source: str) -> list[str]:
    """Every string literal in a module, in source order, f-strings reassembled.

    Adjacent plain literals are folded by the parser, so a statement split
    across source lines arrives as one string. f-strings become one JoinedStr
    whose interpolations collapse to a placeholder -- enough to read the
    column name, which is never interpolated.

    Source order is what the index-ordering guard reads as a *proxy for*
    execution order -- see that test's docstring for where the two come apart.
    ``ast.walk`` is breadth-first, so a statement nested one level deeper
    (inside an ``if``, say) would otherwise sort after a shallower statement
    that follows it in the file.
    """
    tree = ast.parse(source)

    nested = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            nested.update(id(inner) for inner in ast.walk(node) if inner is not node)

    positioned: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            positioned.append(
                (
                    node.lineno,
                    node.col_offset,
                    "".join(
                        (
                            part.value
                            if isinstance(part, ast.Constant)
                            and isinstance(part.value, str)
                            else _EXPR
                        )
                        for part in node.values
                    ),
                )
            )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in nested
        ):
            positioned.append((node.lineno, node.col_offset, node.value))
    positioned.sort(key=lambda item: item[:2])
    return [value for _, _, value in positioned]


class _IndexReference(NamedTuple):
    name: str
    table: str
    #: every identifier the index names -- key columns, INCLUDE, the partial
    #: WHERE predicate -- because a missing column anywhere in it is fatal
    columns: frozenset[str]


def _index_references(literal: str) -> list[tuple[int, _IndexReference]]:
    """``(offset, reference)`` for every ``CREATE INDEX`` in one literal.

    Identifiers are collected from the whole statement after ``ON <table>``
    up to its ``;`` (or the literal's end), minus string literals, so a
    partial index's predicate counts too. Keywords (DESC, WHERE, TRUE ...)
    come along; the guard only ever intersects this set with a table's
    migrated columns, so they cost nothing.
    """
    references = []
    for match in _CREATE_INDEX.finditer(literal):
        open_paren = match.end() - 1
        body = _balanced_body(literal, open_paren)
        if body is None:
            continue
        close_paren = open_paren + len(body) + 1
        terminator = literal.find(";", close_paren)
        predicate = literal[close_paren + 1 : None if terminator == -1 else terminator]
        clause = _SQL_STRING.sub(" ", f"{body} {predicate}")
        references.append(
            (
                match.start(),
                _IndexReference(
                    _name(match.group(1)),
                    _name(match.group(2)),
                    frozenset(tok.lower() for tok in _IDENTIFIER.findall(clause)),
                ),
            )
        )
    return references


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
    for literal in _ddl_literals(source):
        for match in _CREATE_TABLE.finditer(literal):
            body = _balanced_body(literal, match.end() - 1)
            if body is None:
                continue
            declared.setdefault(_name(match.group(1)), set()).update(
                _column_names(body)
            )
        for match in _ADD_COLUMN.finditer(literal):
            table, column = _name(match.group(1)), _name(match.group(2))
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


def test_string_literals_come_back_in_source_order():
    """The ordering guard below reads position as execution order."""
    source = (
        "def f(x):\n"
        "    if x:\n"
        "        a = 'nested first'\n"
        "    b = 'shallow second'\n"
    )

    assert _string_literals(source) == ["nested first", "shallow second"]


def test_ddl_parser_extracts_index_references():
    """Guards the ordering check below from passing vacuously."""
    source = '''
cur.execute(
    "CREATE INDEX IF NOT EXISTS idx_widgets_owner "
    "ON widgets(owner_id, updated_at DESC)"
)
cur.execute(
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_widgets_default
    ON widgets (owner_id) WHERE is_default = TRUE AND label <> 'retired';
    CREATE INDEX idx_widgets_note ON widgets USING btree (lower(note));
    """
)
'''
    references = [
        reference
        for literal in _ddl_literals(source)
        for _, reference in _index_references(literal)
    ]

    assert [reference.name for reference in references] == [
        "idx_widgets_owner",
        "uq_widgets_default",
        "idx_widgets_note",
    ]
    assert {reference.table for reference in references} == {"widgets"}
    assert {"owner_id", "updated_at"} <= references[0].columns
    # The partial-index predicate counts: a column missing there is just as
    # fatal. The quoted value must not, or a column literally named "retired"
    # would be reported as referenced.
    assert {"owner_id", "is_default", "label"} <= references[1].columns
    assert "retired" not in references[1].columns
    assert "note" in references[2].columns


def test_ddl_parser_reads_qualified_and_quoted_identifiers():
    """Schema-qualified and quoted DDL must parse, not silently vanish.

    Before the #433 review the identifier patterns accepted a bare name only,
    so ``ON public.t(a)`` and ``ON "t"(a)`` matched nothing at all -- the
    ordering guard would have reported zero indexes for such a twin and passed.
    """
    source = '''
cur.execute(
    """
    ALTER TABLE public.widgets ADD COLUMN IF NOT EXISTS "owner_id" INTEGER;
    CREATE INDEX IF NOT EXISTS idx_q ON public.widgets(owner_id);
    CREATE UNIQUE INDEX uq_q ON "widgets"("owner_id", label);
    """
)
'''
    literal = _ddl_literals(source)[0]

    references = [reference for _, reference in _index_references(literal)]
    assert [reference.name for reference in references] == ["idx_q", "uq_q"]
    assert {reference.table for reference in references} == {"widgets"}
    assert "owner_id" in references[0].columns
    assert {"owner_id", "label"} <= references[1].columns

    added = [
        (_name(match.group(1)), _name(match.group(2)))
        for match in _ADD_COLUMN.finditer(literal)
    ]
    assert added == [("widgets", "owner_id")]


def test_ddl_parser_ignores_sql_comments():
    """A ``--`` comment is prose, not DDL -- in both directions.

    The credits twin's migration really does contain the sentence "CREATE
    INDEX IF NOT EXISTS matches by name" inside a comment, so a parser that
    reads comments both invents an index and, via the coverage guard below,
    accuses itself of failing to parse one.
    """
    source = '''
cur.execute(
    """
    -- CREATE INDEX IF NOT EXISTS idx_commented ON widgets(owner_id);
    CREATE INDEX IF NOT EXISTS idx_real ON widgets(owner_id); -- trailing note
    """
)
'''
    literal = _ddl_literals(source)[0]
    assert [name for _, (name, *_rest) in _index_references(literal)] == ["idx_real"]
    assert len(_CREATE_INDEX_KEYWORD.findall(literal)) == 1


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_ddl_parser_sees_every_create_index(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """Every CREATE INDEX in a twin's DDL must actually parse.

    This is the anti-vacuity guard for the ordering test below. That test can
    only report an index it managed to read, so an unparsed spelling does not
    fail it -- it empties it. Counting the keyword and the parsed statements
    separately is the only way a regex blind spot shows up as a failure rather
    than as a green run over an unchecked twin.
    """
    literals = _ddl_literals(_module_source_path(postgres_mod).read_text("utf-8"))

    keyword_hits = sum(len(_CREATE_INDEX_KEYWORD.findall(lit)) for lit in literals)
    parsed = sum(len(_index_references(lit)) for lit in literals)

    assert parsed == keyword_hits, (
        f"{postgres_cls}: {keyword_hits} CREATE INDEX statement(s) in the DDL "
        f"but only {parsed} parsed. _CREATE_INDEX has a blind spot, and the "
        f"ordering guard silently skips whatever it cannot read."
    )


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_postgres_twin_indexes_a_migrated_column_only_after_adding_it(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    """The #432 Render boot crash, as a rule rather than a credits-only guard.

    An ``ALTER TABLE t ADD COLUMN IF NOT EXISTS c`` exists because some
    deployed table predates ``c``. On that deployment ``CREATE TABLE IF NOT
    EXISTS`` no-ops, so a ``CREATE INDEX`` naming ``c`` that runs *before*
    the ALTER raises UndefinedColumn at import -- fatal for a store built at
    module scope, and invisible to CI, whose Postgres is empty on every run
    and therefore only ever exercises the CREATE path.

    Scope: this reads *source* position, which is a proxy for execution order,
    not the thing itself. It holds for a twin whose DDL literals run where they
    are written, which is every twin today for the columns that matter. It
    already does not hold in general: ``users_postgres.py`` defines
    ``AUTH_SESSIONS_DDL`` near the top of the module but executes it *after*
    the inline ``ALTER TABLE users ADD COLUMN`` statements further down.
    Nothing crosses tables there, so the proxy costs nothing today -- but a
    future ``CREATE INDEX`` inside a hoisted constant, over a column added by
    an ALTER executed earlier, would be reported as too-early when it is fine,
    and the mirror case would pass while being the #432 bug. Read a failure
    here as "check the execution order", not as proof of one.
    """
    source = _module_source_path(postgres_mod).read_text(encoding="utf-8")
    literals = _ddl_literals(source)

    first_added: dict[tuple[str, str], tuple[int, int]] = {}
    for position, literal in enumerate(literals):
        for match in _ADD_COLUMN.finditer(literal):
            key = (_name(match.group(1)), _name(match.group(2)))
            first_added.setdefault(key, (position, match.start()))

    too_early = []
    for position, literal in enumerate(literals):
        for offset, reference in _index_references(literal):
            for column in sorted(reference.columns):
                added_at = first_added.get((reference.table, column))
                if added_at is not None and added_at > (position, offset):
                    too_early.append(
                        f"  {reference.name} indexes {reference.table}.{column} "
                        f"before ALTER TABLE {reference.table} ADD COLUMN IF NOT "
                        f"EXISTS {column}"
                    )

    assert not too_early, (
        f"{postgres_cls} creates an index on a column its own migration adds "
        f"later. On a deployment whose table predates that column the CREATE "
        f"TABLE no-ops and the CREATE INDEX raises UndefinedColumn at import "
        f"(the #432 Render boot crash). Move the CREATE INDEX below the ADD "
        f"COLUMN:\n" + "\n".join(too_early)
    )


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
    if postgres_cls in _NO_OWN_DDL_TWINS:
        sqlite_ddl = _parse_ddl(
            _module_source_path(sqlite_mod).read_text(encoding="utf-8")
        ).declared
        postgres_ddl = _parse_ddl(
            _module_source_path(postgres_mod).read_text(encoding="utf-8")
        ).declared
        assert not sqlite_ddl and not postgres_ddl, (
            f"_NO_OWN_DDL_TWINS exempts {postgres_cls} as owning no DDL, but "
            f"CREATE TABLE was parsed from it (sqlite={sorted(sqlite_ddl)} "
            f"postgres={sorted(postgres_ddl)}). Remove the entry so the column "
            f"parity check runs."
        )
        pytest.skip(_NO_OWN_DDL_TWINS[postgres_cls])
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


def test_user_store_twins_explicitly_migrate_user_group():
    """Pin the account-group migration in both source-level twin guards.

    The generic schema checks above catch drift, but this focused assertion
    keeps the admin-only dimension's deployed-table migration visible in the
    parity suite's failure output and prevents a future refactor from hiding
    it behind a dynamically assembled SQL fragment.
    """
    sqlite_source = _module_source_path("dashboard.backend.users").read_text(
        encoding="utf-8"
    )
    postgres_source = _module_source_path(
        "dashboard.backend.users_postgres"
    ).read_text(encoding="utf-8")

    sqlite_schema = _parse_ddl(sqlite_source)
    postgres_schema = _parse_ddl(postgres_source)

    assert "user_group" in sqlite_schema.declared["users"]
    assert "user_group" in postgres_schema.declared["users"]
    assert "user_group" in sqlite_schema.migrated["users"]
    assert "user_group" in postgres_schema.migrated["users"]

    folded_postgres = re.sub(r"\s+", " ", postgres_source)
    folded_sqlite = re.sub(r"\s+", " ", sqlite_source)
    assert (
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS user_group "
        f"TEXT NOT NULL DEFAULT '{DEFAULT_USER_GROUP}'"
    ) in folded_postgres

    # The two statements that make the Python constants the owners of this
    # column, pinned in both twins. Neither is decoration.
    #
    # ADD COLUMN IF NOT EXISTS is skipped entirely once the column exists, so it
    # can never restate a default -- the deployed catalog keeps whatever the
    # column was created with, however the line above reads. SET DEFAULT is
    # where Postgres actually applies it. And a row holding a value
    # parse_user_group() rejects stays put forever otherwise, because every read
    # path coerces it out of sight.
    #
    # Both are written as plain SQL strings for the same reason the DDL is: this
    # guard reads source text, and an f-string collapses to nothing it can see.
    assert (
        f"ALTER TABLE users ALTER COLUMN user_group SET DEFAULT '{DEFAULT_USER_GROUP}'"
    ) in folded_postgres
    repair = (
        f"UPDATE users SET user_group = '{DEFAULT_USER_GROUP}' "
        f"WHERE user_group NOT IN ({', '.join(repr(group) for group in USER_GROUPS)})"
    )
    assert repair in folded_postgres, repair
    assert repair in folded_sqlite, repair


def test_credits_postgres_migrates_every_column_added_by_sqlite_rebuild():
    """Credits rebuilds its ledger instead of using SQLite ADD COLUMN statements.

    The generic lazy-migration guard above cannot infer those added columns from
    ALTER syntax, so pin the shipped pre-Grant baseline and require every newer
    SQLite ledger column to have an explicit PostgreSQL ADD COLUMN migration.
    """

    pre_grant_columns = {
        "id",
        "user_id",
        "entry_type",
        "amount_micro",
        "payment_order_id",
        "refund_request_id",
        "stripe_event_id",
        "operation_key",
        "created_at",
    }
    sqlite_source = _module_source_path(
        "dashboard.backend.domain.credits.repository"
    ).read_text(encoding="utf-8")
    postgres_source = _module_source_path(
        "dashboard.backend.domain.credits.repository_postgres"
    ).read_text(encoding="utf-8")
    sqlite_columns = _parse_ddl(sqlite_source).declared["credit_ledger_entries"]
    postgres_migrations = _parse_ddl(postgres_source).migrated.get(
        "credit_ledger_entries", set()
    )
    expected = sqlite_columns - pre_grant_columns

    assert expected <= postgres_migrations, (
        "CreditsStore rebuilds credit_ledger_entries with columns that the "
        "Postgres deployed-table migration never adds: "
        f"{sorted(expected - postgres_migrations)}"
    )


def test_credits_postgres_migrates_pool_columns_added_by_sqlite_rebuild():
    pre_snapshot_columns = {
        "id",
        "pool_id",
        "entry_type",
        "amount_micro",
        "operation_id",
        "idempotency_key",
        "request_digest",
        "actor_user_id",
        "source",
        "reason",
        "user_id",
        "user_ledger_entry_id",
        "created_at",
    }
    sqlite_source = _module_source_path(
        "dashboard.backend.domain.credits.repository"
    ).read_text(encoding="utf-8")
    postgres_source = _module_source_path(
        "dashboard.backend.domain.credits.repository_postgres"
    ).read_text(encoding="utf-8")
    table = "credit_grant_pool_ledger_entries"
    sqlite_columns = _parse_ddl(sqlite_source).declared[table]
    postgres_migrations = _parse_ddl(postgres_source).migrated.get(table, set())
    expected = sqlite_columns - pre_snapshot_columns

    assert expected <= postgres_migrations, (
        "CreditsStore rebuilds the Grant Pool ledger with columns that the "
        "Postgres deployed-table migration never adds: "
        f"{sorted(expected - postgres_migrations)}"
    )


def test_credits_twins_and_postgres_migration_reject_blank_operation_keys():
    sqlite_source = _module_source_path(
        "dashboard.backend.domain.credits.repository"
    ).read_text(encoding="utf-8")
    postgres_source = _module_source_path(
        "dashboard.backend.domain.credits.repository_postgres"
    ).read_text(encoding="utf-8")
    sqlite_sql = re.sub(r"\s+", " ", sqlite_source)
    postgres_sql = re.sub(r"\s+", " ", postgres_source)
    constraint = (
        "operation_key TEXT NOT NULL UNIQUE " "CHECK (length(trim(operation_key)) > 0)"
    )

    assert constraint in sqlite_sql
    assert constraint in postgres_sql
    assert (
        "DROP CONSTRAINT IF EXISTS credit_ledger_entries_operation_key_check"
        in postgres_sql
    )
    assert (
        "ADD CONSTRAINT credit_ledger_entries_operation_key_check "
        "CHECK (length(trim(operation_key)) > 0)" in postgres_sql
    )


# --------------------------------------------------------------------------
# Axis 4: method bodies duplicated across a twin pair
# --------------------------------------------------------------------------
#
# Every axis above compares *shape* -- signatures, columns, index order --
# because shape is cheap to compare. What actually drifts is behaviour. A
# method copied verbatim into both twins has no dialect difference to justify
# it, so the next fix applied to one copy and not the other ships a
# Postgres-only behaviour divergence with a fully green suite: the signature
# axis still matches, the column axis still matches, and nothing reads a body.
#
# PR T is what made this worth guarding. Extracting PostgresValueAnalyticsStore
# duplicated 321 lines across five methods holding no SQL and no dialect
# branching of their own -- more than every other twin in this registry
# combined, the next largest being 17 lines -- because that split was
# deliberately scoped to "change no query". The duplication is therefore a
# known, accepted cost, not an oversight; what was missing is anything that
# notices when a copy stops matching.
#
# The declaration below is deliberately two-way, and that is what keeps it
# from rotting the way a one-way allowlist does:
#
#   * A declared method whose two copies stop matching fails as a divergence
#     -- the behaviour drift above, caught at the commit that introduces it.
#   * A method that *becomes* identical without being declared fails as new
#     duplication, so a twin cannot quietly accumulate more copied code than
#     it admits to.
#
# Comparison is `ast.unparse`, not source text: it ignores comments and
# formatting and compares the code that actually runs. Two copies that differ
# only in a comment are still one behaviour in two places, which is precisely
# what this axis is about.
#
# Shrinking an entry is always safe -- de-duplicate onto a shared mixin or
# helper and delete the name. Growing one is the decision that deserves the
# thought.
_DUPLICATED_BODIES: dict[str, frozenset[str]] = {
    "PostgresAnalyticsStore": frozenset(),
    "PostgresModelProviderStore": frozenset({"revoke_user_credential"}),
    "PostgresCreditsStore": frozenset(
        {
            "_validate_utc_boundary",
            "assign_grant",
            "fund_grant_pool",
            "get_balance_micro",
            "reclaim_grant",
            "reduce_grant_pool",
        }
    ),
    "PostgresAgentCredentialStore": frozenset(),
    "PostgresAgentStore": frozenset(),
    "PostgresAgentVersionStore": frozenset(),
    "BrokerConnectionStorePostgres": frozenset(),
    "PostgresPortfolioStore": frozenset(),
    "PostgresStrategyStore": frozenset(),
    "PostgresUserStore": frozenset(
        {
            "_email_change_expiry",
            "_password_reset_expiry",
            "authenticate",
            "get_user_admin",
        }
    ),
    "PostgresBacktestDatabase": frozenset(),
    # PR T. None of these five contain SQL or branch on this class's own
    # dialect, which is why they could be copied unchanged; list_commercial_values
    # and list_credit_activity branch on the *injected credits_base*, a
    # different store's dialect (see value_repository_postgres.py's module
    # docstring). PR A adds ~12 methods to this pair -- anything it copies
    # verbatim lands here rather than passing unremarked.
    "PostgresValueAnalyticsStore": frozenset(
        {
            "_analytics_connection",
            "_run_health",
            "get_operational_facts",
            "list_commercial_values",
            "list_credit_activity",
        }
    ),
}


def _method_code(module_name: str, class_name: str) -> dict[str, str]:
    """Normalised source of each method, keyed by name.

    Parsed from disk rather than imported: this mirrors the column axis, and
    an import error here would abort collection for the whole session.
    """
    source = _module_source_path(module_name).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name: ast.unparse(item)
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {module_name}")


def test_every_twin_declares_its_duplicated_bodies():
    """The registry and this declaration must not drift apart.

    A twin added to ``_TWINS`` with no entry here would be exempt from the
    body axis while looking covered -- the same silence the dialect-branch
    allowlist exists to prevent.
    """
    missing = sorted(set(_TWIN_IDS) - set(_DUPLICATED_BODIES))
    extra = sorted(set(_DUPLICATED_BODIES) - set(_TWIN_IDS))
    assert not missing and not extra, (
        f"_DUPLICATED_BODIES is missing {missing} and has stale entries "
        f"{extra}; every pair in _TWINS needs a declaration, even an empty "
        "frozenset()."
    )


@pytest.mark.parametrize(
    "sqlite_mod,sqlite_cls,postgres_mod,postgres_cls", _TWINS, ids=_TWIN_IDS
)
def test_duplicated_bodies_match_their_declaration(
    sqlite_mod, sqlite_cls, postgres_mod, postgres_cls
):
    sqlite_code = _method_code(sqlite_mod, sqlite_cls)
    postgres_code = _method_code(postgres_mod, postgres_cls)

    actual = frozenset(
        name
        for name in set(sqlite_code) & set(postgres_code)
        if sqlite_code[name] == postgres_code[name]
    )
    declared = _DUPLICATED_BODIES[postgres_cls]

    diverged = sorted(declared - actual)
    assert not diverged, (
        f"{postgres_cls}: {diverged} are declared identical to {sqlite_cls} "
        "but no longer are. A dialect-free method fixed on one twin only is "
        "a Postgres-only behaviour divergence that every other axis here "
        "passes. Either apply the change to both copies, or de-duplicate the "
        "method and drop it from _DUPLICATED_BODIES -- do not simply remove "
        "the name to quiet this."
    )

    undeclared = sorted(actual - declared)
    assert not undeclared, (
        f"{postgres_cls}: {undeclared} are now byte-identical to "
        f"{sqlite_cls} but undeclared. Duplicated bodies drift silently, so "
        "either share the implementation between the twins or add the "
        "name(s) to _DUPLICATED_BODIES with a reason."
    )
