"""Design D5 absorption pins: the old console's Users/Providers/Activity now
live on the /admin page behind its single gate, and every /app?view=admin deep
link hands off to the absorbed section instead of rendering the legacy console.

The modules themselves keep their own test files (the grant-credits console has
its behaviours pinned elsewhere); what this file owns is the *seam*: the page
carries the absorbed sections, the shell routes to them, the bridge stands in
for app.js's globals, and both the server (Render origin) and the client
(Vercel's static /app) redirect the legacy links.
"""

import re
from pathlib import Path

import pytest

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
APP_PY = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")

CONSOLE_ROUTES = ("account", "providers", "activity")


def test_admin_html_carries_the_absorbed_sections():
    for route in CONSOLE_ROUTES:
        assert f'data-panel="{route}"' in ADMIN_HTML, route
        assert f'href="#{route}" data-route="{route}"' in ADMIN_HTML, route
    for module_id in (
        "accountView", "providersView", "activityView",
        "adminCreditsSection", "adminGrantPoolForm", "adminCreditsUserQuery",
        "adminProviderForm", "adminCreditsActivityBody",
    ):
        assert f'id="{module_id}"' in ADMIN_HTML, module_id


def test_admin_html_carries_the_absorbed_dialog_and_icon_sprite():
    # The grant-reason dialog the credits console opens comes along; the icons
    # the absorbed markup and modules reference are mirrored in a sprite.
    assert 'id="adminGrantReasonDialog"' in ADMIN_HTML
    for symbol in ("icon-refresh", "icon-wallet", "icon-check-circle", "icon-search", "icon-x", "icon-minus"):
        assert f'id="{symbol}"' in ADMIN_HTML, symbol


def test_admin_console_css_exists_and_is_loaded():
    stylesheet = FRONTEND / "admin-console.css"
    assert stylesheet.exists()
    rules = stylesheet.read_text(encoding="utf-8")
    # A sample of the ported component families the absorbed markup needs.
    for selector in (".admin-stats", ".admin-credits-pool-ring", ".admin-provider-grid", ".auth-btn", ".credits-status"):
        assert selector in rules, selector
    assert re.search(r'<link rel="stylesheet" href="admin-console.css\?v=\d+">', ADMIN_HTML)
    # The shell keeps its own stylesheet; the page still does not import the
    # app-wide styles.css (design §7.2).
    assert "styles.css" not in ADMIN_HTML


def test_absorbed_modules_listen_for_the_shell_route():
    credits = source("admin-credits.js")
    providers = source("admin-model-providers.js")
    assert "admin:route" in credits
    assert "admin:route" in providers
    # The #account?user=… hand-off replaces the old adminUserQuery deep link.
    assert "detail.route !== 'account'" in credits
    assert "query?.user" in credits


def test_bridge_stands_in_for_the_app_globals():
    bridge = source("admin-bridge.js")
    assert "window.API = API" in bridge
    assert "window.getStoredAuthUser" in bridge
    assert "atl_csrf" in bridge and "__Host-atl_csrf" in bridge
    assert "AdminShell?.state?.user" in bridge
    # Loaded before the modules that consume it.
    bridge_at = ADMIN_HTML.index("js/admin-bridge.js")
    for consumer in ("js/admin-credits.js", "js/admin-model-providers.js"):
        assert ADMIN_HTML.index(consumer) > bridge_at, consumer


def test_server_redirects_the_legacy_admin_view():
    assert 'request.query_params.get("view") == "admin"' in APP_PY
    for tab, route in (("users", "account"), ("providers", "providers"), ("activity", "activity")):
        assert f'"{tab}": "{route}"' in APP_PY, tab
    assert "adminUserQuery" in APP_PY
    assert '?user=' in APP_PY
    assert "status_code=307" in APP_PY


def test_client_redirect_covers_the_static_host():
    """Vercel serves /app as a static file: no session access, no server
    redirect. The boot resolver must hand ?view=admin off before the gate can
    bounce it on a stale cache."""
    assert "view === 'admin'" in APP_JS
    assert "`/admin#${route}" in APP_JS
    assert "encodeURIComponent(userQuery)" in APP_JS


def test_profile_deep_link_keeps_working_end_to_end():
    """The hand-off the profile and evidence links emit is the one the server
    redirect writes and AdminCredits applies — three spellings, one contract."""
    users_js = source("admin-users.js")
    assert "accountManagementHref" in users_js
    assert "#account${" in users_js
    assert "?user=${encodeURIComponent(query)}" in users_js
    assert "#account" in ADMIN_HTML  # evidence dialog's Open account management
    assert "query?.user" in source("admin-credits.js")


def test_bridge_request_parses_json_errors_with_status():
    BRIDGE = source("admin-bridge.js")
    out = run_node(
        BRIDGE,
        r"""
        globalThis.document = { cookie: 'atl_csrf=tok123' };
        globalThis.fetch = async (endpoint, options) => ({
          ok: false,
          status: 403,
          headers: { get: () => 'application/json' },
          json: async () => ({ detail: 'Admin access required.' }),
          text: async () => '',
        });
        Promise.resolve(window.API.request('/api/admin/credits/grant-pool'))
          .then((v) => console.log(JSON.stringify({ ok: true, v })))
          .catch((error) => console.log(JSON.stringify({ ok: false, message: error.message, status: error.status })));
        """,
    )
    parsed = out
    assert parsed["ok"] is False
    assert parsed["message"] == "Admin access required."
    assert parsed["status"] == 403
