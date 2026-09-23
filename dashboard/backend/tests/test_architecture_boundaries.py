"""Permanent architecture-boundary contract (Phase 4C).

Durable, structural guard rails for the dashboard backend. These tests encode
the layering contract that the migration established and should keep passing for
the lifetime of the codebase:

    API / CLI / Discord  ->  Domain services/logic  ->  Infrastructure adapters

They prefer AST / import inspection over brittle raw-string matching. Test
modules (``dashboard/backend/tests``) are intentionally excluded from the
production-only checks: tests may legitimately manipulate ``sys.path`` and import
across layers.
"""

import ast
import importlib
import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

_BACKEND = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "dashboard" / "scripts"

_MODEL_EXECUTION_TREE = "dashboard/backend/domain/model_execution"
_MODEL_EXECUTION_MODULE_STEMS = (
    _MODEL_EXECUTION_TREE,
    "dashboard/backend/infrastructure/llm/provider_executor",
    "dashboard/backend/infrastructure/llm/routed_client",
)
_IMPORTABLE_SUFFIXES = tuple(
    importlib.machinery.SOURCE_SUFFIXES
    + importlib.machinery.BYTECODE_SUFFIXES
    + importlib.machinery.EXTENSION_SUFFIXES
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _production_py_files():
    """All backend production modules (excludes tests/ and __pycache__)."""
    for path in _BACKEND.rglob("*.py"):
        parts = path.relative_to(_BACKEND).parts
        if "tests" in parts or "__pycache__" in parts:
            continue
        yield path


def _ignored_model_execution_runtime_files(repo_root: Path) -> set[str]:
    """Find importable artifacts that Git inventory intentionally ignores."""

    violations: set[str] = set()
    tree = repo_root / _MODEL_EXECUTION_TREE
    if tree.is_dir():
        violations.update(
            str(path.relative_to(repo_root))
            for path in tree.rglob("*")
            if path.is_file() and path.name.endswith(_IMPORTABLE_SUFFIXES)
        )
    for stem_value in _MODEL_EXECUTION_MODULE_STEMS:
        stem = repo_root / stem_value
        for suffix in _IMPORTABLE_SUFFIXES:
            candidate = stem.with_name(f"{stem.name}{suffix}")
            if candidate.is_file():
                violations.add(str(candidate.relative_to(repo_root)))
    return violations


def _imported_modules(path: Path):
    """Set of absolute module names imported by a file (skips relative imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module)
    return mods


def _backend_top_level_names():
    """First-party top-level module/package names that live under backend/."""
    names = set()
    for entry in _BACKEND.iterdir():
        if entry.name in {"tests", "__pycache__"}:
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.suffix == ".py" and entry.stem != "__init__":
            names.add(entry.stem)
    return names


# ---------------------------------------------------------------------------
# Canonical imports
# ---------------------------------------------------------------------------

def test_first_party_imports_are_canonical():
    """Production backend code imports first-party modules via dashboard.backend.*

    No bare imports such as ``from database import db`` or ``import paths``.
    """
    first_party = _backend_top_level_names()
    offenders = []
    for path in _production_py_files():
        for mod in _imported_modules(path):
            top = mod.split(".")[0]
            if top in first_party:  # bare first-party import (not dashboard.*)
                offenders.append((path.relative_to(_REPO_ROOT).as_posix(), mod))
    assert offenders == [], f"non-canonical first-party imports: {offenders}"


# ---------------------------------------------------------------------------
# Backend must not depend on scripts
# ---------------------------------------------------------------------------

def test_backend_does_not_import_scripts():
    script_basenames = {p.stem for p in _SCRIPTS.glob("*.py")}
    offenders = []
    for path in _production_py_files():
        for mod in _imported_modules(path):
            top = mod.split(".")[0]
            if mod.startswith("dashboard.scripts") or top in script_basenames:
                offenders.append((path.relative_to(_REPO_ROOT).as_posix(), mod))
    assert offenders == [], f"backend imports scripts: {offenders}"


# ---------------------------------------------------------------------------
# sys.path mutation is confined to scripts/_bootstrap.py
# ---------------------------------------------------------------------------

def test_sys_path_mutation_only_in_bootstrap():
    allowed = _SCRIPTS / "_bootstrap.py"
    offenders = []

    def _mutates_sys_path(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"insert", "append", "extend"}:
                    target = node.func.value
                    # sys.path.<attr>(...)
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "path"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "sys"
                    ):
                        return True
        return False

    scan = list(_production_py_files()) + list(_SCRIPTS.glob("*.py"))
    for path in scan:
        if _mutates_sys_path(path) and path != allowed:
            offenders.append(path.relative_to(_REPO_ROOT).as_posix())
    assert offenders == [], f"unexpected sys.path mutation: {offenders}"


# ---------------------------------------------------------------------------
# app.py is a thin composition root (no backend API route bodies)
# ---------------------------------------------------------------------------

def test_app_defines_no_backend_api_routes():
    app_file = _BACKEND / "app.py"
    tree = ast.parse(app_file.read_text(encoding="utf-8"), filename=str(app_file))
    bad = []
    has_include_router = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "include_router":
                has_include_router = True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                    continue
                if not (isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app"):
                    continue
                if dec.func.attr not in {"get", "post", "put", "delete", "patch", "websocket"}:
                    continue
                path_arg = dec.args[0] if dec.args else None
                if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                    p = path_arg.value
                    if p.startswith("/api") or p.startswith("/paper") or p.startswith("/v1"):
                        bad.append((node.name, p))
    assert has_include_router, "app.py must register routers via include_router"
    assert bad == [], f"app.py defines backend API routes inline: {bad}"


# ---------------------------------------------------------------------------
# Route registration contract
# ---------------------------------------------------------------------------

def _app_route_pairs():
    from dashboard.backend.app import app

    pairs = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if method == "HEAD":
                continue
            pairs.append((method, route.path))
    return pairs


def test_every_route_registered_exactly_once():
    pairs = _app_route_pairs()
    dupes = {p for p in pairs if pairs.count(p) > 1}
    assert dupes == set(), f"routes registered more than once: {dupes}"


def test_key_vault_pr_has_no_model_execution_runtime():
    files = set(
        subprocess.check_output(
            ["git", "ls-files"], cwd=_REPO_ROOT, text=True
        ).splitlines()
    )
    files.update(
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=_REPO_ROOT,
            text=True,
        ).splitlines()
    )
    violations = {
        path
        for path in files
        if path == _MODEL_EXECUTION_TREE
        or path.startswith(f"{_MODEL_EXECUTION_TREE}/")
        or any(
            path == f"{stem}{suffix}"
            for stem in _MODEL_EXECUTION_MODULE_STEMS
            for suffix in _IMPORTABLE_SUFFIXES
        )
    }
    violations.update(_ignored_model_execution_runtime_files(_REPO_ROOT))
    assert violations == set(), f"key vault scope includes model execution: {violations}"


@pytest.mark.parametrize(
    "relative_path",
    [
        "dashboard/backend/domain/model_execution/__pycache__/service.cpython-313.pyc",
        "dashboard/backend/infrastructure/llm/provider_executor.pyc",
    ],
)
def test_key_vault_scope_guard_detects_ignored_bytecode(tmp_path, relative_path):
    artifact = tmp_path / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"sourceless bytecode placeholder")

    assert relative_path in _ignored_model_execution_runtime_files(tmp_path)


def test_paper_routes_stay_outside_api_prefix():
    from dashboard.backend.app import app

    paths = {r.path for r in app.routes if isinstance(r, APIRoute)}
    assert "/paper/account" in paths, "/paper/account must remain registered"
    leaked = {p for p in paths if p.startswith("/api/paper")}
    assert leaked == set(), f"/paper routes leaked under /api: {leaked}"


# ---------------------------------------------------------------------------
# Layering: domain & infrastructure never depend on API or the app
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer", ["domain", "infrastructure"])
def test_lower_layers_do_not_import_api_or_app(layer):
    layer_dir = _BACKEND / layer
    offenders = []
    for path in layer_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for mod in _imported_modules(path):
            if mod.startswith("dashboard.backend.api") or mod == "dashboard.backend.app":
                offenders.append((path.relative_to(_REPO_ROOT).as_posix(), mod))
    assert offenders == [], f"{layer} imports API/app: {offenders}"


def test_execution_layer_imports_only_the_v2_contract_from_api():
    """execution/ is application-layer glue: it binds domain engines to the v2
    API contract, so it may import the contract modules (api.v2.models /
    api.v2.errors) but must not reach routers, dependencies, auth, or the app
    composition root — otherwise it silently becomes a second api layer."""
    exec_dir = _BACKEND / "execution"
    allowed = {
        "dashboard.backend.api.v2.models",
        "dashboard.backend.api.v2.errors",
    }
    offenders = []
    for path in exec_dir.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for mod in _imported_modules(path):
            if mod == "dashboard.backend.app" or mod.startswith("dashboard.backend.app."):
                offenders.append((path.name, mod))
            elif (
                mod == "dashboard.backend.api"
                or mod.startswith("dashboard.backend.api.")
            ) and mod not in allowed:
                offenders.append((path.name, mod))
    assert offenders == [], f"execution/ imports beyond the v2 contract: {offenders}"


# ---------------------------------------------------------------------------
# Removed compatibility shims stay gone
# ---------------------------------------------------------------------------

_DELETED_SHIMS = [
    "dashboard.backend.agent_store",
    "dashboard.backend.agent_version_store",
    "dashboard.backend.algo_prompt",
    "dashboard.backend.algo_service",
    "dashboard.backend.environments",
    "dashboard.backend.external_backtest_service",
    "dashboard.backend.llm_validator",
    "dashboard.backend.market_data",
    "dashboard.backend.paper_baselines",
    "dashboard.backend.paper_trading",
    "dashboard.backend.protocol",
    "dashboard.backend.run_service",
    "dashboard.backend.run_store",
    "dashboard.backend.token_cost",
    "dashboard.backend.baselines",
    "dashboard.backend.baseline_data",
    "dashboard.backend.api.agents",
    "dashboard.backend.api.agent_versions",
    "dashboard.backend.api.algo",
    "dashboard.backend.api.environments",
    "dashboard.backend.api.external_backtest",
    "dashboard.backend.api.leaderboard",
    "dashboard.backend.api.runs",
    "dashboard.backend.engines",
    "dashboard.backend.engines.leaderboard_baselines",
    "dashboard.backend.engines.strategies",
    "dashboard.backend.services",
    "dashboard.backend.services.agent_chat_service",
    "dashboard.backend.services.leaderboard_service",
]


@pytest.mark.parametrize("module", _DELETED_SHIMS)
def test_deleted_shim_is_not_importable(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_deleted_shims_absent_from_production_and_scripts():
    deleted = set(_DELETED_SHIMS)
    offenders = []
    scan = list(_production_py_files()) + list(_SCRIPTS.glob("*.py"))
    for path in scan:
        for mod in _imported_modules(path):
            if mod in deleted:
                offenders.append((path.relative_to(_REPO_ROOT).as_posix(), mod))
    assert offenders == [], f"deleted shim paths still imported: {offenders}"


# ---------------------------------------------------------------------------
# Chat / Discord import safety: no secrets, no network at import time
# ---------------------------------------------------------------------------

def _import_clean(module: str) -> subprocess.CompletedProcess:
    code = (
        "import os, sys\n"
        "for v in ('ANTHROPIC_API_KEY','ANTHROPIC_MODEL','DISCORD_BOT_TOKEN','DISCORD_GUILD_ID'):\n"
        "    os.environ.pop(v, None)\n"
        f"import {module}\n"
        "print('import-ok')\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in {"ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                        "DISCORD_BOT_TOKEN", "DISCORD_GUILD_ID"}}
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_chat_service_imports_without_secrets():
    proc = _import_clean("dashboard.backend.domain.chat.service")
    assert proc.returncode == 0, proc.stderr
    assert "import-ok" in proc.stdout
    # The Anthropic client must not be constructed at import time.
    code = (
        "import dashboard.backend.domain.chat.service as s\n"
        "assert s._claude_client is None\n"
        "print('no-client')\n"
    )
    proc2 = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_REPO_ROOT),
        capture_output=True, text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    assert "no-client" in proc2.stdout


def test_discord_bot_imports_without_secrets():
    try:
        import discord  # noqa: F401
    except Exception:
        pytest.skip("discord.py not installed in this environment")
    proc = _import_clean("dashboard.backend.integrations.discord_bot")
    assert proc.returncode == 0, proc.stderr
    assert "import-ok" in proc.stdout


# ---------------------------------------------------------------------------
# Settlement semantics come from the market profile, never a constructor default
# ---------------------------------------------------------------------------

def test_provenance_is_not_reachable_from_the_leaderboard_package():
    """The import cycle this module sits inside, kept open at one point.

    ``provenance`` imports ``leaderboard/service.py`` for the threshold, and
    the leaderboard already imports back into ``domain/backtesting``
    (``strategies/llm_agent`` -> ``portfolio_manager``; ``baselines`` ->
    ``constants``/``metrics``). ``domain/backtesting`` <-> ``domain/leaderboard``
    is therefore a package-level cycle already, and it is survivable only
    because it does not close on *this* module: nothing the leaderboard can
    reach imports ``provenance``.

    One import from ``portfolio_manager`` -- the most natural place someone
    would reach for a verdict helper, since that is where the fallbacks happen
    -- closes it, and the app dies at import with a traceback naming neither of
    the two files whose relationship caused it. This fails first, and prints
    the chain.

    Deliberately a reachability walk rather than a deny-list of three module
    names: the invariant is "nothing the leaderboard can reach", which widens
    by itself the day the leaderboard imports another backtesting module.

    **What this can and cannot catch, measured rather than assumed.** A plain
    module-level import closing the cycle takes the whole pytest session down
    in conftest setup, before any test body runs -- this one included. That
    case needs no guard: it is an unmissable ImportError storm, and the
    recognisable line is "cannot import name 'MIN_LLM_DECISION_COVERAGE' from
    partially initialized module".

    The case this guard is actually for is the *deferred* one: a
    function-local ``from ...provenance import ...`` inside a module the
    leaderboard reaches. That collects green, passes CI, and raises on the
    first fallback step in production. ``ast.walk`` sees imports at any depth,
    so it is caught here; verified by adding exactly that import and watching
    this fail while the rest of the suite stayed green.
    """
    backend, repo_root = _BACKEND, _REPO_ROOT

    graph: dict[str, set[str]] = {}
    for path in backend.rglob("*.py"):
        parts = path.relative_to(backend).parts
        if "tests" in parts or "__pycache__" in parts:
            continue
        module = ".".join(path.relative_to(repo_root).with_suffix("").parts)
        edges = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                edges.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                edges.add(node.module)
                edges.update(f"{node.module}.{a.name}" for a in node.names)
        graph[module] = {e for e in edges if e.startswith("dashboard.backend")}

    # A package import reaches its __init__, which may re-export submodules.
    def _targets(name: str) -> set[str]:
        return {m for m in graph if m == name or m == f"{name}.__init__"}

    parent: dict[str, str] = {}
    frontier = [m for m in graph if m.startswith("dashboard.backend.domain.leaderboard")]
    seen = set(frontier)
    while frontier:
        module = frontier.pop()
        for edge in graph.get(module, ()):
            for target in _targets(edge):
                if target not in seen:
                    seen.add(target)
                    parent[target] = module
                    frontier.append(target)

    # Non-vacuity: the reverse edge really is there, so an empty walk (a broken
    # parser, a renamed package) cannot pass this as "no cycle".
    assert "dashboard.backend.domain.backtesting.portfolio_manager" in seen, (
        "the leaderboard no longer reaches portfolio_manager -- this walk is "
        "not measuring what it claims to"
    )

    offender = "dashboard.backend.domain.backtesting.provenance"
    if offender in seen:
        chain, cursor = [offender], offender
        while cursor in parent:
            cursor = parent[cursor]
            chain.append(cursor)
        pytest.fail(
            "provenance imports domain/leaderboard for MIN_LLM_DECISION_COVERAGE, "
            "so the leaderboard must not reach provenance -- that closes the "
            "cycle and kills app import. Chain:\n  "
            + "\n  ".join(reversed(chain))
        )


def test_portfolio_manager_construction_always_declares_settlement():
    """Every production PortfolioManager must pass ``t_plus_one_enabled``.

    ``t_plus_one_enabled`` defaults to ``False``, which is correct for US
    markets and silently wrong for A-shares. A site that omits it therefore
    fails *closed but invisible*: no error, no log, just same-day settlement on
    a market that does not have it. Requiring the argument at every call site
    turns that into a test failure at the moment the site is written.

    Pass ``get_market_profile(...).t_plus_one_enabled`` — resolving it from the
    registry means wiring a new data source carries its settlement rules along
    automatically. Hardcoding ``False`` satisfies this test but re-opens the
    hole, so keep the profile lookup.

    Known limits, so nobody reads a green run as full coverage. The match is by
    **name**, so a call through an aliased import (``PortfolioManager as PM``)
    is never inspected; ``**kwargs`` splats pass, since the keyword set is
    unknowable statically; and a purely positional third argument is *flagged*
    even though it is correct. The check exists to catch the one form anyone
    actually writes — the keyword call that simply omits the argument.
    """
    offenders = []
    for path in _production_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name != "PortfolioManager":
                continue
            passed = {kw.arg for kw in node.keywords}
            if "t_plus_one_enabled" not in passed and None not in passed:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")

    assert offenders == [], (
        "PortfolioManager built without an explicit t_plus_one_enabled at: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Event-log discipline (admin layer redesign design doc SS6.11, rules 1-7)
# ---------------------------------------------------------------------------

_ANALYTICS = _BACKEND / "domain" / "analytics"
_EVENT_READ_NAMES = {"list_events", "list_user_events", "list_metric_events"}
# Files that may call a raw-event reader, each with the reason. Rules 3-5: the
# paginated timeline and the overview's one-day scan are the only permitted raw
# reads; everything else reads rollups or user_daily_facts. Every entry must
# still contain a hit (the stale check below), so PR B removes the last two as
# it deletes the files and narrows the others as it moves the reads.
_EVENT_READ_ALLOWLIST = {
    "dashboard/backend/domain/analytics/rollups.py": (
        "defines AnalyticsRollupStore.list_events; rollup_day / rollup_current_day "
        "are the day rollups and the overview's current-day scan (rule 5), "
        "narrowed to one UTC day in PR B"
    ),
    "dashboard/backend/domain/analytics/query_service.py": (
        "the paginated timeline (rule 3) and the overview's current-day read "
        "(rule 5); PR B narrows the overview read and gives sessions a 30-day window"
    ),
    "dashboard/backend/domain/analytics/value_queries.py": (
        "the retention grid's per-activation-week scan and the one-user profile "
        "scan; PR B moves both onto user_daily_facts / user_activity"
    ),
    "dashboard/backend/domain/analytics/states.py": (
        "legacy five-state calculator: full-history per-user reads. DELETED IN "
        "PR B (design SS13 row B); allowlisted, not fixed, because PR A creates "
        "and PR B drops"
    ),
    "dashboard/backend/domain/analytics/lifecycle_backfill.py": (
        "historical eight-week reconstruction, per-user history reads. DELETED "
        "IN PR B; the copy in facts_migration.py carries the history now"
    ),
}
_OWN_CONNECTION_RECEIVERS = {"self", "self.analytics_base", "self.base_store"}
_OWN_DIALECT_RECEIVERS = {
    "base_store",
    "self.base_store",
    "analytics_base",
    "self.analytics_base",
    "resolved_analytics_base",
}
# Rule 7 tolerates nothing after PR A. A future entry needs a file path and a
# reason, exactly like _EVENT_READ_ALLOWLIST -- and a design-doc amendment,
# because the rule it relaxes is SS6.14's first row.
_CROSS_DOMAIN_READ_ALLOWLIST: dict[str, str] = {}
_PER_USER_QUERY_PREFIXES = (
    "aggregate_", "list_", "upsert_", "record_", "sum_", "claim_",
    "complete_", "release_", "get_", "append_", "copy_", "seed_",
)


def _analytics_sources():
    for path in sorted(_ANALYTICS.glob("*.py")):
        yield path.relative_to(_REPO_ROOT).as_posix(), ast.parse(
            path.read_text(encoding="utf-8")
        )


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def test_event_log_rule_1_and_2_constants_hold():
    from dashboard.backend.domain.analytics import models, retention

    assert models.MAX_PROPERTIES_BYTES == 1024
    assert isinstance(models.ALLOWED_SERVER_EVENT_NAMES, (set, frozenset))
    assert retention.RAW_EVENT_RETENTION_DAYS == 180
    mutators = []
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"add", "update", "discard", "remove"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id.startswith("ALLOWED_")
            ):
                mutators.append((relative, node.lineno))
    assert mutators == [], f"the event allowlist is mutated at runtime: {mutators}"


def test_raw_event_reads_stay_inside_the_allowlist():
    """No new caller may read the event log (rules 3, 4, 5).

    On 2026-09-11 a maintenance sweep read every user's 180-day history
    every fifteen minutes, exhausted a 5 GB monthly egress allowance, and
    500'd login for five hours. Nothing in the suite noticed, because every
    behavioural assertion still passed. This is the assertion that would have.
    """
    hits: dict[str, list[tuple[str, int]]] = {}
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _EVENT_READ_NAMES
            ):
                hits.setdefault(relative, []).append((node.func.attr, node.lineno))
    unapproved = {path: calls for path, calls in hits.items() if path not in _EVENT_READ_ALLOWLIST}
    assert unapproved == {}, f"unapproved raw-event reads: {unapproved}"
    stale = sorted(set(_EVENT_READ_ALLOWLIST) - set(hits))
    assert stale == [], (
        "allowlist entries with no raw-event read left in them -- delete the entry "
        f"so the exemption cannot be inherited by the next reader: {stale}"
    )


def test_the_daily_job_never_loops_over_users():
    """A for-loop issuing a query per user is the shape that caused the outage (rule 6)."""
    source = (_ANALYTICS / "daily_facts.py").read_text(encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        # The loop's own iterable may be one set-based read; its body may not
        # issue any.
        for statement in node.body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr.startswith(_PER_USER_QUERY_PREFIXES)
                ):
                    offenders.append((inner.func.attr, inner.lineno))
    assert offenders == [], f"store calls inside a loop in daily_facts.py: {offenders}"


def test_the_analytics_package_opens_only_its_own_connection():
    """Rule 7 (design SS6.11, SS6.14 row 1): `_get_connection()` is called only
    on the analytics domain's own store, and no analytics module sniffs another
    store's dialect. Other domains' tables are read through public methods on
    the store that owns them -- CreditsStore.aggregate_ledger_for_day, not SQL
    against credit_ledger_entries from inside this package."""
    offenders = []
    for relative, tree in _analytics_sources():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_get_connection":
                receiver = _dotted(node.func.value)
                if receiver not in _OWN_CONNECTION_RECEIVERS:
                    offenders.append((relative, node.lineno, f"{receiver}._get_connection()"))
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "hasattr"
                and len(node.args) == 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "database_url"
            ):
                receiver = _dotted(node.args[0])
                if receiver not in _OWN_DIALECT_RECEIVERS:
                    offenders.append((relative, node.lineno, f'hasattr({receiver}, "database_url")'))
    unlisted = [entry for entry in offenders if entry[0] not in _CROSS_DOMAIN_READ_ALLOWLIST]
    assert unlisted == [], f"cross-domain connection or dialect sniff in domain/analytics: {unlisted}"
    stale = sorted(set(_CROSS_DOMAIN_READ_ALLOWLIST) - {entry[0] for entry in offenders})
    assert stale == [], f"stale rule-7 allowlist entries: {stale}"
