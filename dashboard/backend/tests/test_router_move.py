"""Phase 3A3 / 3B3 / 3C2 — HTTP router move verification.

Confirms the agent, run, environment, external-backtest, and algo HTTP routers
live in the canonical ``dashboard.backend.api.routers`` package, with identical
route registration and no duplicate routes. (The legacy ``api/<name>.py`` shims
have been removed.)
"""

import ast
import subprocess
import sys
from pathlib import Path

from fastapi.routing import APIRoute

from dashboard.backend.api import dependencies as deps
from dashboard.backend.api import protocol_auth
from dashboard.backend.api import router as router_module
from dashboard.backend.api.routers import agent_versions as versions_canon
from dashboard.backend.api.routers import agents as agents_canon
from dashboard.backend.api.routers import algo as algo_canon
from dashboard.backend.api.routers import environments as environments_canon
from dashboard.backend.api.routers import external_backtest as external_backtest_canon
from dashboard.backend.api.routers import leaderboard as leaderboard_canon
from dashboard.backend.api.routers import news as news_canon
from dashboard.backend.api.routers import paper_trading as paper_trading_canon
from dashboard.backend.api.routers import runs as runs_canon
from dashboard.backend.app import app

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APP_FILE = Path(__file__).resolve().parents[1] / "app.py"

# (method, full path, endpoint name) — the contract that must not change.
EXPECTED_AGENT_ROUTES = {
    ("POST", "/v1/agents", "create_agent"),
    ("GET", "/v1/agents", "list_agents"),
    ("GET", "/v1/agents/builtin", "list_builtin_agents"),
    ("GET", "/v1/agents/marketplace", "list_marketplace_agents"),
    ("POST", "/v1/agents/marketplace/{template_id}/clone", "clone_marketplace_agent"),
    ("POST", "/v1/agents/claim-account", "claim_account_agents"),
    ("POST", "/v1/agents/import-session", "import_session_agent"),
    ("GET", "/v1/agents/resolve", "resolve_api_key"),
    ("GET", "/v1/agents/{agent_id}/runs", "list_agent_runs"),
    ("GET", "/v1/agents/{agent_id}", "get_agent"),
    ("PATCH", "/v1/agents/{agent_id}", "update_agent"),
    (
        "GET",
        "/v1/agents/{agent_id}/credentials/financial-datasets",
        "get_financial_datasets_credential_status",
    ),
    (
        "PUT",
        "/v1/agents/{agent_id}/credentials/financial-datasets",
        "set_financial_datasets_credential",
    ),
    (
        "DELETE",
        "/v1/agents/{agent_id}/credentials/financial-datasets",
        "delete_financial_datasets_credential",
    ),
    ("DELETE", "/v1/agents/{agent_id}", "delete_agent"),
    ("POST", "/v1/agents/{agent_id}/rotate-api-key", "rotate_agent_api_key"),
    ("POST", "/v1/agents/{agent_id}/duplicate", "duplicate_agent"),
    ("POST", "/v1/agents/{agent_id}/activate", "activate_agent"),
}

EXPECTED_VERSION_ROUTES = {
    ("POST", "/v1/agents/{agent_id}/versions", "create_agent_version"),
    ("GET", "/v1/agents/{agent_id}/versions", "list_agent_versions"),
    ("GET", "/v1/agent-versions/{agent_version_id}", "get_agent_version"),
}

EXPECTED_RUN_ROUTES = {
    ("POST", "/v1/runs", "create_run"),
    ("GET", "/v1/runs/{run_id}", "get_run"),
    ("GET", "/v1/runs/{run_id}/status", "get_run_status"),
    ("GET", "/v1/runs/{run_id}/steps/next", "get_next_step"),
    ("GET", "/v1/runs/{run_id}/steps/{step_id}", "get_step"),
    ("POST", "/v1/runs/{run_id}/steps/{step_id}/decision", "submit_step_decision"),
    ("GET", "/v1/runs/{run_id}/steps", "list_steps"),
    ("GET", "/v1/runs/{run_id}/decisions", "list_decisions"),
    ("GET", "/v1/runs/{run_id}/trades", "list_trades"),
    ("GET", "/v1/runs/{run_id}/metrics", "get_metrics"),
    ("GET", "/v1/runs/{run_id}/result", "get_result"),
}

EXPECTED_ENV_ROUTES = {
    ("GET", "/v1/environments", "api_list_environments"),
    ("GET", "/v1/environments/{environment_id}", "api_get_environment"),
}

EXPECTED_EXTERNAL_BACKTEST_ROUTES = {
    ("GET", "/v1/backtest/schema", "api_decision_schema"),
    ("POST", "/v1/backtest/start", "api_start_backtest"),
    ("GET", "/v1/backtest/runs/{run_id}/result", "api_run_result"),
    ("GET", "/v1/backtest/runs/{run_id}/trades", "api_run_trades"),
    ("GET", "/v1/backtest/runs/{run_id}/decisions", "api_run_decisions"),
    ("GET", "/v1/backtest/{backtest_id}/status", "api_backtest_status"),
    ("GET", "/v1/backtest/{backtest_id}/decisions", "api_backtest_decisions"),
    ("GET", "/v1/backtest/{backtest_id}/steps/current", "api_get_current_step"),
    ("POST", "/v1/backtest/{backtest_id}/steps/current/decisions", "api_submit_decisions"),
}

EXPECTED_ALGO_ROUTES = {
    ("GET", "/algo/setup", "algo_setup_status"),
    ("GET", "/algo/defaults", "algo_defaults"),
    ("POST", "/algo/chat", "algo_chat"),
    ("POST", "/algo/execute", "algo_execute"),
    ("GET", "/algo/status", "algo_execution_status"),
    ("GET", "/algo/submissions", "list_submissions"),
}

EXPECTED_LEADERBOARD_ROUTES = {
    ("GET", "/v1/leaderboard", "api_get_leaderboard"),
    ("POST", "/v1/leaderboard/daily/refresh", "api_refresh_daily_leaderboard"),
}

EXPECTED_NEWS_ROUTES = {
    ("GET", "/news/signals", "latest_news_signals"),
}

# Paper Trading routes are registered directly on the app (no /api prefix);
# external paths must stay exactly /paper/...
EXPECTED_PAPER_ROUTES = {
    ("GET", "/paper/account", "get_paper_account"),
    ("GET", "/paper/positions", "get_paper_positions"),
    ("GET", "/paper/trades", "get_paper_trades"),
    ("GET", "/paper/portfolio-history", "get_paper_portfolio_history"),
    ("POST", "/paper/start-session", "start_paper_session"),
    ("GET", "/paper/baselines", "get_paper_baselines"),
}


def _route_triples(router):
    triples = set()
    for route in router.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if method == "HEAD":
                continue
            triples.add((method, route.path, route.name))
    return triples


# ---------------------------------------------------------------------------
# Canonical import
# ---------------------------------------------------------------------------

def test_canonical_modules_import():
    assert agents_canon.router is not None
    assert versions_canon.router is not None
    assert agents_canon.router.__class__.__name__ == "APIRouter"
    # agent_service singleton identity shared across canonical routers.
    assert agents_canon.agent_service is versions_canon.agent_service


def test_shared_auth_helpers_live_in_dependencies():
    # Phase 3A4: the shared owner-context/agent-access helpers moved to the
    # canonical dependencies module; the agents router imports them from there.
    assert agents_canon._owner_context is deps._owner_context
    assert agents_canon._require_agent_access is deps._require_agent_access
    assert agents_canon._require_owner_context is deps._require_owner_context


def test_protocol_auth_sources_helpers_from_dependencies():
    # protocol_auth must source the helpers from dependencies, not the old shim.
    src = Path(protocol_auth.__file__).read_text(encoding="utf-8")
    assert "from dashboard.backend.api.dependencies import" in src
    assert "from dashboard.backend.api.agents import" not in src


# ---------------------------------------------------------------------------
# Route identity (paths, methods, names, tags)
# ---------------------------------------------------------------------------

def test_agent_router_route_contract_unchanged():
    assert _route_triples(agents_canon.router) == EXPECTED_AGENT_ROUTES


def test_version_router_route_contract_unchanged():
    assert _route_triples(versions_canon.router) == EXPECTED_VERSION_ROUTES


def test_run_router_route_contract_unchanged():
    assert _route_triples(runs_canon.router) == EXPECTED_RUN_ROUTES


def test_environment_router_route_contract_unchanged():
    assert _route_triples(environments_canon.router) == EXPECTED_ENV_ROUTES


def test_router_prefixes_and_tags_unchanged():
    assert agents_canon.router.prefix == "/v1/agents"
    assert agents_canon.router.tags == ["agents"]
    assert versions_canon.router.prefix == "/v1"
    assert versions_canon.router.tags == ["agent-versions"]
    assert runs_canon.router.prefix == "/v1/runs"
    assert runs_canon.router.tags == ["runs"]
    assert environments_canon.router.prefix == "/v1/environments"
    assert environments_canon.router.tags == ["environments"]
    assert external_backtest_canon.router.prefix == "/v1/backtest"
    assert external_backtest_canon.router.tags == ["external-backtest"]
    assert algo_canon.router.prefix == "/algo"
    assert algo_canon.router.tags == ["algo"]
    assert leaderboard_canon.router.prefix == "/v1/leaderboard"
    assert leaderboard_canon.router.tags == ["leaderboard"]


# ---------------------------------------------------------------------------
# Run / Environment router move (Phase 3B3)
# ---------------------------------------------------------------------------

def test_run_env_canonical_modules_import():
    assert runs_canon.router.__class__.__name__ == "APIRouter"
    assert environments_canon.router.__class__.__name__ == "APIRouter"


def _all_imported_modules(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_canonical_run_router_imports_domain_modules():
    modules = _all_imported_modules(runs_canon.__file__)
    assert "dashboard.backend.domain.runs.service" in modules
    assert "dashboard.backend.domain.runs.repository" in modules
    assert "dashboard.backend.domain.runs.protocol" in modules
    assert "dashboard.backend.domain.runs.environment" in modules


def test_canonical_environment_router_imports_domain_module():
    modules = _all_imported_modules(environments_canon.__file__)
    assert "dashboard.backend.domain.runs.environment" in modules


# ---------------------------------------------------------------------------
# External-backtest / Algo router move (Phase 3C2)
# ---------------------------------------------------------------------------

def test_backtesting_canonical_modules_import():
    assert external_backtest_canon.router.__class__.__name__ == "APIRouter"
    assert algo_canon.router.__class__.__name__ == "APIRouter"


def test_external_backtest_router_route_contract_unchanged():
    assert _route_triples(external_backtest_canon.router) == EXPECTED_EXTERNAL_BACKTEST_ROUTES


def test_algo_router_route_contract_unchanged():
    assert _route_triples(algo_canon.router) == EXPECTED_ALGO_ROUTES


def test_canonical_backtesting_routers_use_canonical_services():
    ext_modules = _all_imported_modules(external_backtest_canon.__file__)
    assert "dashboard.backend.domain.backtesting.external_run_service" in ext_modules
    algo_modules = _all_imported_modules(algo_canon.__file__)
    assert "dashboard.backend.domain.backtesting.algo_service" in algo_modules


# ---------------------------------------------------------------------------
# Leaderboard router move (Phase 3C4)
# ---------------------------------------------------------------------------

def test_leaderboard_canonical_module_imports():
    assert leaderboard_canon.router.__class__.__name__ == "APIRouter"


def test_leaderboard_router_route_contract_unchanged():
    assert _route_triples(leaderboard_canon.router) == EXPECTED_LEADERBOARD_ROUTES


def test_canonical_leaderboard_router_uses_canonical_service():
    modules = _all_imported_modules(leaderboard_canon.__file__)
    assert "dashboard.backend.domain.leaderboard.service" in modules


# ---------------------------------------------------------------------------
# News panel proxy router (Task 5)
# ---------------------------------------------------------------------------

def test_news_router_route_contract_unchanged():
    assert _route_triples(news_canon.router) == EXPECTED_NEWS_ROUTES


# ---------------------------------------------------------------------------
# Paper Trading router move (Phase 3C5C)
# ---------------------------------------------------------------------------

def test_paper_canonical_module_imports():
    assert paper_trading_canon.router.__class__.__name__ == "APIRouter"


def test_paper_router_prefix_and_route_contract_unchanged():
    assert paper_trading_canon.router.prefix == "/paper"
    assert _route_triples(paper_trading_canon.router) == EXPECTED_PAPER_ROUTES


def test_paper_routes_registered_exactly_once_in_app():
    counts = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if method == "HEAD":
                continue
            counts[(method, route.path, route.name)] = (
                counts.get((method, route.path, route.name), 0) + 1
            )
    for triple in EXPECTED_PAPER_ROUTES:
        assert counts.get(triple) == 1, (triple, counts.get(triple))


def test_paper_routes_keep_external_paths_without_api_prefix():
    paper_paths = {
        (m, p)
        for route in app.routes
        if isinstance(route, APIRoute)
        for m in route.methods
        if m != "HEAD"
        for p in [route.path]
        if p.startswith("/paper")
    }
    expected = {(m, p) for (m, p, _) in EXPECTED_PAPER_ROUTES}
    assert paper_paths == expected
    # None of the paper routes should have leaked under the /api prefix.
    api_paper = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/paper")
    }
    assert api_paper == set()


def test_canonical_paper_router_uses_canonical_modules():
    modules = _all_imported_modules(paper_trading_canon.__file__)
    assert "dashboard.backend.infrastructure.brokers.alpaca_paper" in modules
    assert "dashboard.backend.domain.trading.paper_session" in modules


def test_app_no_longer_defines_paper_routes():
    src = _APP_FILE.read_text(encoding="utf-8")
    # No inline paper route decorators or handler bodies remain.
    assert '"/paper/' not in src
    assert "async def get_paper_account" not in src
    assert "async def get_paper_positions" not in src
    assert "async def get_paper_trades" not in src
    assert "async def get_paper_portfolio_history" not in src
    assert "async def start_paper_session" not in src
    assert "async def get_paper_baselines" not in src
    # app must register the canonical router instead.
    assert "paper_trading_router" in src


def test_app_imports_canonical_paper_router():
    src = Path(_APP_FILE).read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "dashboard.backend.api.routers.paper_trading" in modules


def test_each_endpoint_registered_exactly_once_in_app():
    counts = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if method == "HEAD":
                continue
            counts[(method, route.path)] = counts.get((method, route.path), 0) + 1

    expected_full = set()
    all_routes = (
        EXPECTED_AGENT_ROUTES | EXPECTED_VERSION_ROUTES
        | EXPECTED_RUN_ROUTES | EXPECTED_ENV_ROUTES
        | EXPECTED_EXTERNAL_BACKTEST_ROUTES | EXPECTED_ALGO_ROUTES
        | EXPECTED_LEADERBOARD_ROUTES
    )
    for method, path, _ in all_routes:
        expected_full.add((method, f"/api{path}"))

    for key in expected_full:
        assert counts.get(key) == 1, (key, counts.get(key))


# ---------------------------------------------------------------------------
# router.py uses canonical imports
# ---------------------------------------------------------------------------

def test_router_py_uses_canonical_imports():
    src = Path(router_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "dashboard.backend.api.routers.agents" in modules
    assert "dashboard.backend.api.routers.agent_versions" in modules
    assert "dashboard.backend.api.routers.runs" in modules
    assert "dashboard.backend.api.routers.environments" in modules
    assert "dashboard.backend.api.routers.external_backtest" in modules
    assert "dashboard.backend.api.routers.algo" in modules
    assert "dashboard.backend.api.routers.leaderboard" in modules
    # Must not import the routers from the legacy shim locations.
    assert "dashboard.backend.api.agents" not in modules
    assert "dashboard.backend.api.agent_versions" not in modules
    assert "dashboard.backend.api.runs" not in modules
    assert "dashboard.backend.api.environments" not in modules
    assert "dashboard.backend.api.external_backtest" not in modules
    assert "dashboard.backend.api.algo" not in modules
    assert "dashboard.backend.api.leaderboard" not in modules


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------

def test_no_circular_imports():
    code = (
        "import dashboard.backend.api.routers.agents\n"
        "import dashboard.backend.api.routers.agent_versions\n"
        "import dashboard.backend.api.routers.runs\n"
        "import dashboard.backend.api.routers.environments\n"
        "import dashboard.backend.api.routers.external_backtest\n"
        "import dashboard.backend.api.routers.algo\n"
        "import dashboard.backend.api.routers.leaderboard\n"
        "import dashboard.backend.api.router\n"
        "import dashboard.backend.api.protocol_auth\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
