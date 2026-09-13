"""The Analytics-tab redirect fires only on admin intent, never on page load.

PR #467 moved the admin Analytics tab to the standalone /admin-analytics page by
redirecting from the tab controller's `setTab`. But `setTab('analytics')` is also
what the controller runs on every `/app` DOMContentLoaded and on every popstate,
purely to initialise hidden panel state — so after #467 *every* view of the app
bounced to /admin-analytics the moment it loaded. Those lifecycle calls must set
panel state without leaving the page; only entering the admin view (the Admin
menu item, or a `?view=admin` deep link routed by app.js) or clicking the
Analytics tab may redirect.

The controller is an IIFE over `window`/`document`, so it runs under node against
a minimal DOM stub that records every navigation.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ADMIN_TABS_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "admin-tabs.js"
).read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_STUB = r"""
const nav = [];
const docListeners = {};
const winListeners = {};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
globalThis.Event = class { constructor(type) { this.type = type; } };
globalThis.document = {
  addEventListener(name, fn) { (docListeners[name] ||= []).push(fn); },
  dispatchEvent() {},
  getElementById() { return null; },
  querySelectorAll() { return []; },
};
globalThis.window = {
  location: {
    href: 'https://atl.example/app?view=community',
    assign(u) { nav.push(['assign', u]); },
    replace(u) { nav.push(['replace', u]); },
  },
  history: { state: null, replaceState() {}, pushState() {} },
  addEventListener(name, fn) { (winListeners[name] ||= []).push(fn); },
};
const fire = (map, name) => (map[name] || []).forEach((fn) => fn({}));
const setHref = (h) => { window.location.href = h; };
"""


def _run(scenario: str) -> list:
    script = "\n".join([_STUB, ADMIN_TABS_JS, scenario, "console.log(JSON.stringify(nav));"])
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_page_load_on_a_non_admin_view_does_not_redirect():
    assert _run("fire(docListeners, 'DOMContentLoaded');") == []


def test_popstate_does_not_redirect():
    # app.js owns view routing on popstate; a back/forward that lands on the
    # admin view reaches onEnter through navigateToPage, not through here.
    assert _run("fire(docListeners, 'DOMContentLoaded'); fire(winListeners, 'popstate');") == []


def test_entering_admin_view_redirects_to_the_analytics_page():
    nav = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        "setHref('https://atl.example/app?view=admin');"
        "window.AdminTabs.onEnter();"
    )
    assert nav == [["replace", "/admin-analytics"]]


def test_entering_admin_view_on_another_tab_stays_in_the_console():
    nav = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        "setHref('https://atl.example/app?view=admin&adminTab=providers');"
        "window.AdminTabs.onEnter();"
    )
    assert nav == []


def test_redirect_replaces_history_so_back_does_not_loop():
    # `assign` leaves `/app?view=admin` behind /admin-analytics; Back reloads it,
    # app.js routes to admin, and the user is redirected forward again forever.
    assert "window.location.assign('/admin-analytics')" not in ADMIN_TABS_JS
    assert "window.location.replace('/admin-analytics')" in ADMIN_TABS_JS
