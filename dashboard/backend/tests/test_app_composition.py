"""Phase 3D4A/3D4B — application composition-root verification.

Confirms the remaining inline backend API routes moved to canonical router
modules under ``dashboard.backend.api.routers`` with an unchanged route
contract, that each route is registered exactly once, that the reusable CSP
middleware moved into ``dashboard.backend.middleware``, and that ``app.py`` no
longer contains backend route bodies / business logic.
"""

import ast
from pathlib import Path

from fastapi.routing import APIRoute

from dashboard.backend.api.routers import admin as admin_canon
from dashboard.backend.api.routers import backtests as backtests_canon
from dashboard.backend.api.routers import config as config_canon
from dashboard.backend.api.routers import health as health_canon
from dashboard.backend.api.routers import market as market_canon
from dashboard.backend import middleware as middleware_mod
from dashboard.backend.app import app

_BACKEND = Path(__file__).resolve().parents[1]
_APP_FILE = _BACKEND / "app.py"

EXPECTED_HEALTH_ROUTES = {("GET", "/health", "health")}
EXPECTED_MARKET_ROUTES = {("GET", "/ticker", "get_ticker")}
EXPECTED_CONFIG_ROUTES = {
    ("GET", "/config/defaults", "get_defaults"),
    ("GET", "/config/features", "get_features"),
}
# ``/admin/clear`` is absent on purpose: an unauthenticated DELETE that called
# db.clear_all() became unrecoverable once AGENT_RUNS_DATABASE_URL made run
# history durable, and nothing called it. See api/routers/admin.py's docstring.
EXPECTED_ADMIN_ROUTES = {
    ("DELETE", "/admin/runs/{run_id}", "admin_delete_run"),
}
EXPECTED_BACKTESTS_ROUTES = {
    ("POST", "/backtest/run", "run_backtest_endpoint"),
    ("GET", "/backtest/status", "get_backtest_status"),
    ("GET", "/api/backtest/runs", "get_backtest_runs"),
    ("GET", "/api/backtest/compare/latest", "compare_latest_backtests"),
    ("GET", "/api/backtest/{run_id}/chart-data", "get_backtest_chart_data"),
    ("GET", "/api/backtest/{run_id}", "get_backtest_run"),
    ("GET", "/runs/latest/metrics", "get_latest_metrics"),
    ("GET", "/runs", "get_runs"),
    ("GET", "/runs/{run_id}", "get_run"),
    ("GET", "/runs/{run_id}/equity", "get_equity_curve"),
    ("GET", "/runs/{run_id}/trades", "get_run_trades"),
    ("GET", "/runs/{run_id}/plot.png", "get_run_plot"),
    ("GET", "/compare", "compare_runs"),
}

# The complete, frozen external route contract (method, path) — no HEAD.
EXPECTED_FULL_CONTRACT = {
    ("GET", "/api/strategies/{code}"),
    ("POST", "/api/strategies"),
    ("GET", "/"),
    ("GET", "/app"),
    ("GET", "/app/"),
    ("GET", "/assets/{file_name}"),
    ("DELETE", "/admin/runs/{run_id}"),
    ("POST", "/api/algo/chat"),
    ("GET", "/api/algo/defaults"),
    ("POST", "/api/algo/execute"),
    ("GET", "/api/algo/setup"),
    ("GET", "/api/algo/status"),
    ("GET", "/api/algo/submissions"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("GET", "/api/auth/me"),
    ("POST", "/api/auth/signup"),
    ("POST", "/api/auth/change-password"),
    ("PUT", "/api/auth/display-name"),
    ("POST", "/api/auth/email-change"),
    ("GET", "/api/auth/email-change"),
    ("DELETE", "/api/auth/email-change"),
    ("POST", "/api/auth/email-change/verify"),
    ("PUT", "/api/auth/avatar"),
    ("DELETE", "/api/auth/avatar"),
    ("GET", "/api/auth/discord/callback"),
    ("POST", "/api/auth/discord/start"),
    ("GET", "/api/auth/robinhood/callback"),
    ("POST", "/api/auth/robinhood/complete"),
    ("POST", "/api/auth/robinhood/start"),
    ("GET", "/api/backtest/compare/latest"),
    ("GET", "/api/backtest/runs"),
    ("GET", "/api/backtest/{run_id}"),
    ("GET", "/api/backtest/{run_id}/chart-data"),
    ("GET", "/api/health"),
    ("GET", "/api/news/signals"),
    ("GET", "/api/v1/agent-versions/{agent_version_id}"),
    ("GET", "/api/v1/agents"),
    ("POST", "/api/v1/agents"),
    ("GET", "/api/v1/agents/builtin"),
    ("GET", "/api/v1/agents/marketplace"),
    ("POST", "/api/v1/agents/marketplace/{template_id}/clone"),
    ("POST", "/api/v1/agents/claim-account"),
    ("POST", "/api/v1/agents/import-session"),
    ("GET", "/api/v1/agents/resolve"),
    ("GET", "/api/v1/discord/agents"),
    ("DELETE", "/api/v1/agents/{agent_id}"),
    ("GET", "/api/v1/agents/{agent_id}"),
    ("PATCH", "/api/v1/agents/{agent_id}"),
    ("DELETE", "/api/v1/agents/{agent_id}/credentials/financial-datasets"),
    ("GET", "/api/v1/agents/{agent_id}/credentials/financial-datasets"),
    ("PUT", "/api/v1/agents/{agent_id}/credentials/financial-datasets"),
    ("POST", "/api/v1/agents/{agent_id}/activate"),
    ("POST", "/api/v1/agents/{agent_id}/rotate-api-key"),
    ("GET", "/api/v1/agents/{agent_id}/runs"),
    ("GET", "/api/v1/agents/{agent_id}/versions"),
    ("POST", "/api/v1/agents/{agent_id}/versions"),
    ("GET", "/api/v1/backtest/runs/{run_id}/decisions"),
    ("GET", "/api/v1/backtest/runs/{run_id}/result"),
    ("GET", "/api/v1/backtest/runs/{run_id}/trades"),
    ("GET", "/api/v1/backtest/schema"),
    ("POST", "/api/v1/backtest/start"),
    ("GET", "/api/v1/backtest/{backtest_id}/decisions"),
    ("GET", "/api/v1/backtest/{backtest_id}/status"),
    ("GET", "/api/v1/backtest/{backtest_id}/steps/current"),
    ("POST", "/api/v1/backtest/{backtest_id}/steps/current/decisions"),
    ("GET", "/api/v1/environments"),
    ("GET", "/api/v1/environments/{environment_id}"),
    ("GET", "/api/v1/leaderboard"),
    ("GET", "/api/v1/portfolio"),
    ("POST", "/api/v1/portfolio/allocate"),
    ("POST", "/api/v1/portfolio/reclaim"),
    ("POST", "/api/v1/robinhood/agents/{agent_id}/live-run"),
    ("DELETE", "/api/v1/robinhood/disconnect"),
    ("GET", "/api/v1/robinhood/status"),
    ("POST", "/api/v1/runs"),
    ("GET", "/api/v1/runs/{run_id}"),
    ("GET", "/api/v1/runs/{run_id}/decisions"),
    ("GET", "/api/v1/runs/{run_id}/metrics"),
    ("GET", "/api/v1/runs/{run_id}/result"),
    ("GET", "/api/v1/runs/{run_id}/status"),
    ("GET", "/api/v1/runs/{run_id}/steps"),
    ("GET", "/api/v1/runs/{run_id}/steps/next"),
    ("GET", "/api/v1/runs/{run_id}/steps/{step_id}"),
    ("POST", "/api/v1/runs/{run_id}/steps/{step_id}/decision"),
    ("GET", "/api/v1/runs/{run_id}/trades"),
    ("POST", "/api/v2/agents"),
    ("GET", "/api/v2/agents/me"),
    ("POST", "/api/v2/agents/{agent_id}/rotate-key"),
    ("GET", "/api/v2/leaderboard"),
    ("POST", "/api/v2/runs"),
    ("GET", "/api/v2/runs/{run_id}"),
    ("POST", "/api/v2/runs/{run_id}/cancel"),
    ("GET", "/api/v2/runs/{run_id}/context"),
    ("GET", "/api/v2/runs/{run_id}/decisions"),
    ("POST", "/api/v2/runs/{run_id}/decisions"),
    ("GET", "/api/v2/runs/{run_id}/result"),
    ("GET", "/api/v2/schema"),
    ("GET", "/app.js"),
    ("POST", "/backtest/run"),
    ("GET", "/backtest/status"),
    ("GET", "/compare"),
    ("GET", "/config/defaults"),
    ("GET", "/config/features"),
    ("GET", "/health"),
    ("GET", "/favicon.ico"),
    ("GET", "/favicon.svg"),
    ("GET", "/home-news-signals.js"),
    ("GET", "/home-page.js"),
    ("GET", "/images/{file_name}"),
    ("GET", "/js/{file_name}"),
    ("GET", "/market-events/{file_name}"),
    ("GET", "/paper/account"),
    ("GET", "/paper/baselines"),
    ("GET", "/paper/portfolio-history"),
    ("GET", "/paper/positions"),
    ("POST", "/paper/start-session"),
    ("GET", "/paper/trades"),
    ("GET", "/runs"),
    ("GET", "/runs/latest/metrics"),
    ("GET", "/runs/{run_id}"),
    ("GET", "/runs/{run_id}/equity"),
    ("GET", "/runs/{run_id}/plot.png"),
    ("GET", "/runs/{run_id}/trades"),
    ("GET", "/strategy"),
    ("GET", "/styles.css"),
    ("GET", "/ticker"),
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


def _app_method_path_counts():
    counts = {}
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods:
            if method == "HEAD":
                continue
            counts[(method, route.path)] = counts.get((method, route.path), 0) + 1
    return counts


def _imported_modules(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


# ---------------------------------------------------------------------------
# Canonical modules import + per-router contract
# ---------------------------------------------------------------------------

def test_canonical_router_modules_import():
    for mod in (health_canon, market_canon, config_canon, admin_canon, backtests_canon):
        assert mod.router.__class__.__name__ == "APIRouter"


def test_health_router_contract():
    assert _route_triples(health_canon.router) == EXPECTED_HEALTH_ROUTES


def test_market_router_contract():
    assert _route_triples(market_canon.router) == EXPECTED_MARKET_ROUTES


def test_config_router_contract():
    assert _route_triples(config_canon.router) == EXPECTED_CONFIG_ROUTES


def test_admin_router_contract():
    assert _route_triples(admin_canon.router) == EXPECTED_ADMIN_ROUTES


def test_backtests_router_contract():
    assert _route_triples(backtests_canon.router) == EXPECTED_BACKTESTS_ROUTES


# ---------------------------------------------------------------------------
# Full app contract + single registration
# ---------------------------------------------------------------------------

def test_full_route_contract_unchanged():
    actual = {
        (m, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for m in route.methods
        if m != "HEAD"
    }
    assert actual == EXPECTED_FULL_CONTRACT


def test_extracted_routes_registered_exactly_once():
    counts = _app_method_path_counts()
    extracted = (
        EXPECTED_HEALTH_ROUTES | EXPECTED_MARKET_ROUTES | EXPECTED_CONFIG_ROUTES
        | EXPECTED_ADMIN_ROUTES | EXPECTED_BACKTESTS_ROUTES
    )
    for method, path, _name in extracted:
        assert counts.get((method, path)) == 1, (method, path, counts.get((method, path)))


# ---------------------------------------------------------------------------
# app.py is a thin composition root
# ---------------------------------------------------------------------------

def test_app_no_longer_defines_extracted_handlers_or_logic():
    src = _APP_FILE.read_text(encoding="utf-8")
    for marker in (
        "async def health(",
        "async def get_ticker(",
        "async def get_defaults(",
        "async def admin_clear_all(",
        "async def get_runs(",
        "async def compare_runs(",
        "async def run_backtest_endpoint(",
        "def filter_market_hours(",
        "def run_backtest_background(",
        "class EquityPoint(",
        "class RunMetadata(",
        "class CSPHeaderMiddleware(",
        "backtest_status",
    ):
        assert marker not in src, marker


def test_app_imports_canonical_routers():
    modules = _imported_modules(_APP_FILE)
    for m in (
        "dashboard.backend.api.routers.health",
        "dashboard.backend.api.routers.backtests",
        "dashboard.backend.api.routers.config",
        "dashboard.backend.api.routers.market",
        "dashboard.backend.api.routers.admin",
        "dashboard.backend.api.routers.paper_trading",
    ):
        assert m in modules, m


def test_app_still_serves_frontend_and_startup():
    src = _APP_FILE.read_text(encoding="utf-8")
    assert "async def serve_root(" in src
    assert "async def startup_event(" in src


# ---------------------------------------------------------------------------
# Extracted middleware + ordering
# ---------------------------------------------------------------------------

def test_csp_middleware_lives_in_middleware_module():
    assert hasattr(middleware_mod, "CSPHeaderMiddleware")
    from dashboard.backend.app import CSPHeaderMiddleware as app_csp
    assert app_csp is middleware_mod.CSPHeaderMiddleware


def test_csp_header_omits_unsafe_eval():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    response = TestClient(app).get("/api/health")
    csp = response.headers.get("content-security-policy", "")
    assert "script-src" in csp
    assert "unsafe-eval" not in csp


def test_middleware_order_preserved():
    names = [m.cls.__name__ for m in app.user_middleware]
    assert names == ["CSPHeaderMiddleware", "SessionMiddleware", "CORSMiddleware"]


def test_cors_preflight_allows_every_routed_method():
    """A routed method missing from ``allow_methods`` is unreachable in prod.

    The frontend (Vercel) and the API (Render) are separate origins, so any
    request the browser preflights -- PATCH among them -- dies at the preflight
    when the method is absent, even though the route exists and answers fine to
    curl. ``PATCH /api/v1/agents/{id}`` shipped that way: the only PATCH route
    in the app, and the one behind the agent Configure screen's Save.
    """
    from fastapi.testclient import TestClient

    routed = {
        method
        for route in app.routes
        for method in (getattr(route, "methods", None) or set())
    } - {"HEAD", "OPTIONS"}
    assert "PATCH" in routed, "guard the guard: the PATCH route must still exist"

    client = TestClient(app)
    for method in sorted(routed):
        response = client.options(
            "/api/v1/agents/some-agent",
            headers={
                "Origin": "https://agentic-trading-lab.vercel.app",
                "Access-Control-Request-Method": method,
            },
        )
        assert response.status_code == 200, (
            f"{method} preflight rejected ({response.status_code}): {response.text}"
        )
        allowed = response.headers.get("access-control-allow-methods", "")
        assert method in allowed, f"{method} missing from allow_methods: {allowed!r}"


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_canonical_routers_do_not_import_scripts():
    for mod in (health_canon, market_canon, config_canon, admin_canon, backtests_canon):
        modules = _imported_modules(Path(mod.__file__))
        for m in modules:
            assert not m.startswith("dashboard.scripts"), (mod.__name__, m)


def test_market_router_uses_canonical_market_data():
    modules = _imported_modules(Path(market_canon.__file__))
    assert "dashboard.backend.infrastructure.market_data.quotes" in modules


# ---------------------------------------------------------------------------
# Composition root has no path manipulation (Phase 3D4B)
# ---------------------------------------------------------------------------

def test_app_has_no_sys_path_mutation():
    src = _APP_FILE.read_text(encoding="utf-8")
    assert "sys.path.insert" not in src
    assert "sys.path.append" not in src


def test_app_first_party_imports_are_canonical():
    modules = _imported_modules(_APP_FILE)
    first_party = {m for m in modules if "backend" in m or m.startswith("dashboard")}
    for m in first_party:
        assert m.startswith("dashboard.backend"), m
