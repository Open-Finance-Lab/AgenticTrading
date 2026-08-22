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
