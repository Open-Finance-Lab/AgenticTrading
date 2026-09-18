"""Design N2/PR2 absorption pins: the old console's remaining tabs (Account
Management, Activity) live on the /admin page behind its single gate, and every
/app?view=admin deep link hands off to the absorbed section instead of rendering
the legacy console.

Rebased onto the 09-17 nav consolidation (#488): Providers had already landed
there with the app rail and the account menu, so this file owns only the seam
this branch added — the credits console's two routes, the hand-off query, the
ticker, and the redirects.
"""

from pathlib import Path

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
APP_PY = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")

CONSOLE_ROUTES = ("account", "activity")


def test_admin_html_carries_the_absorbed_sections():
    for route in CONSOLE_ROUTES:
        assert f'data-panel="{route}"' in ADMIN_HTML, route
        assert f'data-rail="{route}" href="#{route}"' in ADMIN_HTML, route
    for module_id in (
        "accountView", "activityView",
        "adminCreditsSection", "adminGrantPoolForm", "adminCreditsUserQuery",
        "adminCreditsActivityBody",
    ):
        assert f'id="{module_id}"' in ADMIN_HTML, module_id


def test_admin_html_carries_the_absorbed_dialog_and_extra_sprite_symbols():
    # The grant-reason dialog the credits console opens comes along; the icons
    # the absorbed markup and modules reference are mirrored in the sprite.
    assert 'id="adminGrantReasonDialog"' in ADMIN_HTML
    for symbol in ("icon-wallet", "icon-search", "icon-minus"):
        assert f'id="{symbol}"' in ADMIN_HTML, symbol


def test_absorbed_modules_listen_for_the_shell_route():
    credits = source("admin-credits.js")
    assert "admin:route" in credits
    # The #account?user=… hand-off replaces the old adminUserQuery deep link.
    assert "detail.query?.user" in credits


def test_absorbed_module_uses_the_shells_write_path():
    """#488 established the shell's write() as the page's ONE CSRF-bearing write
    path; the absorbed credits console goes through it like providers does, and
    the retired window.API bridge is gone."""
    credits = source("admin-credits.js")
    assert "AdminShell.write(" in credits
    assert "AdminShell.request(" in credits
    assert "AdminShell.handleAccessLost(" in credits
    assert "window.API" not in credits
    assert not (FRONTEND / "js" / "admin-bridge.js").exists()
    assert not (FRONTEND / "js" / "admin-chrome.js").exists()


def test_ticker_module_is_loaded_and_pinned_at_its_fixed_version():
    assert 'src="js/admin-ticker.js?v=2"' in ADMIN_HTML
    ticker = source("admin-ticker.js")
    assert "/ticker?symbols=" in ticker
    assert "requestAnimationFrame" in ticker
    # The repeat estimate divides by the estimated single-pass width — the
    # left-to-right reading once built ~380k DOM nodes and froze the page.
    assert "singlePassWidth" in ticker


def test_server_redirects_the_legacy_admin_view():
    assert 'request.query_params.get("view") == "admin"' in APP_PY
    for tab, route in (("users", "account"), ("providers", "providers"), ("activity", "activity")):
        assert f'"{tab}": "{route}"' in APP_PY, tab
    assert "adminUserQuery" in APP_PY
    assert "status_code=307" in APP_PY


def test_client_redirect_covers_the_static_host():
    """Vercel serves /app as a static file: no session access, no server
    redirect. The boot resolver must hand ?view=admin off before the cached-role
    gate can bounce it."""
    assert "view === 'admin'" in APP_JS
    assert "`/admin#${route}" in APP_JS


def test_profile_deep_link_handoff_spells_one_contract():
    """The hand-off the profile and evidence links emit is the one the server
    redirect writes and AdminCredits applies — three spellings, one contract."""
    users_js = source("admin-users.js")
    assert "#account${" in users_js
    assert "?user=${encodeURIComponent(query)}" in users_js
    assert 'id="evidenceAccount"' in ADMIN_HTML and 'href="#account"' in ADMIN_HTML
    assert "query?.user" in source("admin-credits.js")


def _status_after(scenario: str) -> object:
    """Load admin-credits.js over a registered status node, then run `scenario`.

    The module is an IIFE that only registers listeners at load, so nothing
    touches window.AdminShell until a handler runs -- which is why a route the
    module ignores needs no shell stub.
    """
    return run_node(
        # admin-credits.js destructures window.CreditFormat at load, so the
        # formatter has to precede it here exactly as it does in the page.
        source("credit-format.js"),
        source("admin-credits.js"),
        "const status = register('adminCreditsStatus', new Node('p'));",
        # The class list the page actually ships (admin.html / app.html).
        "status.className = 'credits-status admin-credits-status';",
        "status.textContent = 'Grant Pool funded.';",
        scenario,
        "console.log(JSON.stringify({ text: status.textContent,"
        " classes: status.className.split(' ').filter(Boolean).sort() }));",
    )


def test_status_line_is_cleared_when_the_shell_routes_away():
    """#adminCreditsStatus lives outside both absorbed sections, because the two
    of them share it (Account search/pagination and Activity load both write it).

    Outside means no view can hide it: on /app the enclosing #adminView was
    display:none, so a message died with the view. Here only the module can
    retire its own message, and the route event is the only signal it gets.
    Without this, "Grant Pool funded." stays painted at the top of <main> on
    Overview, Providers and Activity -- attached to nothing on screen, and
    announced by aria-live as current.
    """
    result = _status_after(
        "document.dispatchEvent(new CustomEvent('admin:route',"
        " { detail: { route: 'overview', query: {} } }));"
    )
    assert result["text"] == ""


def test_status_writes_keep_the_layout_class_the_markup_ships():
    """setStatus assigns className outright, so every write re-states the whole
    list. The markup ships two classes and only `credits-status` was being
    re-stated, which dropped `.admin-credits-status`'s margin on the first
    write -- on /admin (admin.css) and /app (styles.css) alike.
    """
    result = _status_after("window.AdminCredits.syncAuth({ role: 'user' });")
    assert result["classes"] == ["admin-credits-status", "credits-status"]
