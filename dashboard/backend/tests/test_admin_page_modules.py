"""Source-shape pins shared by the four /admin modules (design §7.4, §7.6, §13 row C).

Harvested from test_admin_analytics_frontend.py's `test_client_uses_exact_pr2_
endpoints_and_query_names` and `test_analytics_is_read_only_and_uses_safe_dom_
rendering`, re-targeted at js/admin-*.js, plus the pins the design's "Must not"
column for PR C asks for: no #live route, no Chart.js, no client-side group
badge, and none of the D15 display fields read.
"""

import re
from pathlib import Path

from dashboard.backend.tests._frontend_source import fn_body

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
NAMES = ("admin-shell.js", "admin-live.js", "admin-overview.js", "admin-users.js")
ABSORBED_MODULES = ("admin-bridge.js", "admin-credits.js", "admin-model-providers.js")
MODULES = {name: (FRONTEND / "js" / name).read_text(encoding="utf-8") for name in NAMES}
ALL = "\n".join(MODULES.values())
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
GLOBALS = {
    "admin-shell.js": "AdminShell",
    "admin-live.js": "AdminLive",
    "admin-overview.js": "AdminOverview",
    "admin-users.js": "AdminUsers",
}


def test_each_module_is_an_iife_exposing_exactly_one_global():
    for name, source in MODULES.items():
        assert source.lstrip().startswith("/**"), name
        assert "(function () {\n  'use strict';" in source, name
        assigned = set(re.findall(r"^\s*window\.(\w+) = ", source, re.M))
        assert assigned == {GLOBALS[name]}, (name, assigned)


def test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded():
    srcs = re.findall(r'<script src="js/([^?"]+)\?v=\d+" defer></script>', ADMIN_HTML)
    # D5 absorbed the old console's modules too: admin-bridge.js stands in for
    # the app.js globals they consumed, and admin-credits/admin-model-providers
    # keep their own routers, stores and mutations behind the same gate.
    assert set(srcs) == set(NAMES) | set(ABSORBED_MODULES) | {"credit-format.js"}
    for name in srcs:
        assert (FRONTEND / "js" / name).exists(), name


def test_absorbed_modules_still_own_their_mutating_requests():
    """The four analytics modules are read-only (§13 row C); the absorbed old
    console never was. Pin the difference instead of pretending it away: the
    absorbed modules keep their mutating verbs, the bridge stays transport-only."""
    bridge = (FRONTEND / "js" / "admin-bridge.js").read_text(encoding="utf-8")
    assert "fetch(" in bridge
    assert not re.findall(r"method:\s*'(\w+)'", bridge)
    for name in ABSORBED_MODULES[1:]:  # the bridge carries no verbs of its own
        source = (FRONTEND / "js" / name).read_text(encoding="utf-8")
        verbs = set(re.findall(r"method:\s*'(\w+)'", source))
        assert verbs, name
        assert verbs <= {"POST", "PATCH", "PUT", "DELETE"}, (name, verbs)


def test_rendering_is_text_content_only():
    for name, source in MODULES.items():
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function("):
            assert forbidden not in source, (name, forbidden)
        assert "textContent" in source, name


def test_every_request_is_a_credentialed_get_made_by_the_shell():
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET"}
    for name, source in MODULES.items():
        if name == "admin-shell.js":
            assert source.count("fetch(") == 2  # request() and gate()
            assert "credentials: 'include'" in source
        else:
            assert "fetch(" not in source, name
            assert "XMLHttpRequest" not in source, name
    for verb in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL


def test_exact_endpoints_and_query_names():
    assert "/api/auth/me" in MODULES["admin-shell.js"]
    assert "/api/admin/stats" in MODULES["admin-live.js"]
    for endpoint in (
        "/api/admin/analytics/overview", "/api/admin/analytics/lifecycle", "/api/admin/analytics/retention",
        "/api/admin/analytics/commercial", "/api/admin/analytics/operational", "/api/admin/analytics/groups",
    ):
        assert endpoint in MODULES["admin-overview.js"], endpoint
    assert "/api/admin/analytics/users" in MODULES["admin-users.js"]
    assert "/activity?" in MODULES["admin-users.js"]
    for query_name in (
        "'from'", "'to'", "'include_internal'", "'user_group'", "'lifecycle_segment'",
        "'commercial_tier'", "'q'", "'priority'", "'limit'", "'offset'", "cursor",
    ):
        assert query_name in ALL, query_name
    assert "section, limit:" in MODULES["admin-users.js"]
    for rejected in ("'status'", "start_date", "provider_id'", "model_id'", "'cohort'", "analyticsUser", "adminTab=analytics"):
        assert rejected not in ALL, rejected


def test_no_live_route_and_no_chartjs():
    assert "#live" not in ALL and "#live" not in ADMIN_HTML
    routes_line = re.search(r"const ROUTES = \[([^\]]+)\];", MODULES["admin-shell.js"]).group(1)
    assert "'live'" not in routes_line
    assert "'usage'" not in routes_line and "'revenue'" not in routes_line and "'profiles'" not in routes_line
    for forbidden in ("window.Chart", "new Chart(", "chart.js", "cdn.jsdelivr.net"):
        assert forbidden not in ALL, forbidden
        assert forbidden not in ADMIN_HTML, forbidden
    assert "createElementNS" in MODULES["admin-overview.js"]  # the revenue line is generated SVG


def test_group_badge_is_never_computed_client_side():
    """D11: `group_badge` is rendered verbatim; the precedence rule has one owner, the server."""
    assert "group_badge" in MODULES["admin-users.js"]
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        source = MODULES[name]
        for forbidden in ("role === 'admin'", "=== 'unpaid'", "'paid'", "'free'", "user_group ===", "!== 'unknown'"):
            assert forbidden not in source, (name, forbidden)
    # The shell's gate is the one place `role` is compared, and only against 'admin' for the redirect.
    assert MODULES["admin-shell.js"].count("user.role !== 'admin'") == 1


def test_d15_display_fields_are_not_read():
    for field in ("country_code", "device_category", "browser_family", "top_product_page"):
        assert field not in ALL, field


def test_prohibited_field_names_and_local_storage_are_absent():
    for prohibited in (
        "api_key", "session_id", "network_hash", "provider_response_body",
        "credential_ciphertext", "prompt", "strategy", "portfolio", "password", "raw_user_agent",
    ):
        assert prohibited not in ALL, prohibited
    assert "localStorage" not in ALL
    assert "sessionStorage" not in ALL


def test_formatting_goes_through_intl_and_the_shared_credit_formatter():
    shell = MODULES["admin-shell.js"]
    assert "window.CreditFormat.formatCreditsMicro(value)" in shell
    assert "Intl.NumberFormat" in shell and "Intl.DateTimeFormat" in shell
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        assert "formatCreditsMicro" not in MODULES[name], name  # only via AdminShell.formatCredits
        assert ".toFixed(6)" not in MODULES[name], name


def test_renderers_can_be_lifted_with_fn_body():
    """§7.6: every renderer is a named function the shared harness can slice."""
    for signature in (
        "function renderAttention(", "function renderActiveUsers(", "function renderActivation(",
        "function renderSources(", "function renderRetention(", "function renderValue(",
        "function renderLifecycle(", "function renderCredits(", "function renderRevenue(",
        "function detailSources(", "function detailRetention(", "function detailCredits(",
        "function detailLifecycle(", "function detailHealth(",
    ):
        body = fn_body(signature, MODULES["admin-overview.js"])
        assert "innerHTML" not in body
    for signature in (
        "function renderUserRows(", "function renderPager(", "function renderEvidence(",
        "function renderProfileHeader(", "function renderProfileOverview(", "function renderActivityItems(",
    ):
        fn_body(signature, MODULES["admin-users.js"])
    body = fn_body("function renderTiles(", MODULES["admin-live.js"])
    assert "max_active_dashboard_backtests" in body
