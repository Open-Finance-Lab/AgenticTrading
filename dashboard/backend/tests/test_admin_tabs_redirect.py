"""Entering the admin view never navigates away; the Analytics tab is gone.

PR #467 sent the console's Analytics tab to a standalone mock via
`window.location.replace('/admin-analytics')` from `setTab`, and PR #468 gated
that redirect on admin intent after it had bounced every /app load. PR C
(design §7.5) removes the tab and the redirect: the analytics page is reached
from Profile → Admin in app.js, and this controller only switches the Users,
Providers and Activity panels. It also reads `adminUserQuery`, the pre-fill the
/admin profile's "Open account management" link carries, and hands it to
`openAccountManagement` the way the in-process evidence dialog used to.

The controller is an IIFE over `window`/`document`, so it runs under node
against a minimal DOM stub that records every navigation.
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
const submitted = [];
const docListeners = {};
const winListeners = {};
const input = { value: '', focus() {} };
const form = { dispatchEvent(event) { submitted.push(event.type); return true; } };
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
globalThis.Event = class { constructor(type) { this.type = type; } };
globalThis.document = {
  addEventListener(name, fn) { (docListeners[name] ||= []).push(fn); },
  dispatchEvent() {},
  getElementById(id) { return id === 'adminCreditsUserQuery' ? input : id === 'adminCreditsUserSearch' ? form : null; },
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


def _run(scenario: str) -> dict:
    script = "\n".join([
        _STUB, ADMIN_TABS_JS, scenario,
        "console.log(JSON.stringify({nav, submitted, query: input.value}));",
    ])
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_page_load_never_navigates():
    assert _run("fire(docListeners, 'DOMContentLoaded');")["nav"] == []


def test_entering_the_admin_view_never_navigates():
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        "setHref('https://atl.example/app?view=admin');"
        "window.AdminTabs.onEnter();"
    )
    assert result["nav"] == []
    assert result["submitted"] == []


@pytest.mark.parametrize("tab", ["users", "providers", "activity", "analytics", "grant-pool"])
def test_entering_on_any_tab_stays_in_the_console(tab):
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        f"setHref('https://atl.example/app?view=admin&adminTab={tab}');"
        "window.AdminTabs.onEnter();"
    )
    assert result["nav"] == []


def test_default_tab_is_users_and_analytics_is_not_a_tab():
    assert "DEFAULT_TAB = 'users'" in ADMIN_TABS_JS
    assert "'analytics'" not in ADMIN_TABS_JS
    assert "admin-analytics" not in ADMIN_TABS_JS
    assert "window.location.replace" not in ADMIN_TABS_JS
    assert "window.location.assign" not in ADMIN_TABS_JS
    assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS


def test_admin_user_query_prefills_account_management():
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        "setHref('https://atl.example/app?view=admin&adminTab=users&adminUserQuery=ada%40example.test');"
        "window.AdminTabs.onEnter();"
    )
    assert result == {"nav": [], "submitted": ["submit"], "query": "ada@example.test"}
