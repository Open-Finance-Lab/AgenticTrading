"""Entering the admin view never navigates away, except off a retired tab.

PR #467 sent the console's Analytics tab to a standalone mock via
`window.location.replace('/admin-analytics')` from `setTab`, and PR #468 gated
that redirect on admin intent after it had bounced every /app load. PR C
(design §7.5) removes the tab and the redirect: the analytics page is reached
from Profile → Admin in app.js, and this controller only switches the Users
and Activity panels. Providers left the same way in the 09-17 consolidation
(N2, N5): a URL that still names it hops to its new home on /admin instead of
falling back silently to Account Management. It also reads `adminUserQuery`,
the pre-fill the /admin profile's "Open account management" link carries, and
hands it to `openAccountManagement` the way the in-process evidence dialog
used to.

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


# The five-way sweep stays five-way and gains the new expectation rather than
# losing the `providers` case: "providers no longer navigates" and "providers
# navigates to /admin" are different claims, and dropping the row would leave
# only the second one covered.
@pytest.mark.parametrize(
    "tab, expected_nav",
    [
        ("users", []),
        ("activity", []),
        ("analytics", []),
        ("grant-pool", []),
        # N5. The URL is in browser histories and possibly in bookmarks, and
        # normalizeTab would otherwise coerce it silently to Account Management
        # -- a different page than the one the operator asked for, with nothing
        # on screen saying so. The /admin-analytics -> /admin 308 is the
        # precedent for paying this courtesy.
        ("providers", [["replace", "/admin#providers"]]),
    ],
)
def test_entering_on_a_tab_stays_in_the_console_unless_the_tab_has_moved(tab, expected_nav):
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        f"setHref('https://atl.example/app?view=admin&adminTab={tab}');"
        "window.AdminTabs.onEnter();"
    )
    assert result["nav"] == expected_nav


def test_a_stale_providers_bookmark_redirects_on_page_load():
    """The only way a real operator reaches this redirect. Nobody calls setTab by
    hand; they follow a link or a bookmark still naming ?adminTab=providers, and
    `init` runs it off DOMContentLoaded before `onEnter` is ever reached. The
    sweep above enters through onEnter, so without this case the entry point that
    actually happens in a browser is covered by nothing."""
    result = _run(
        "setHref('https://atl.example/app?view=admin&adminTab=providers');"
        "fire(docListeners, 'DOMContentLoaded');"
    )
    assert result["nav"] == [["replace", "/admin#providers"]]


def test_a_providers_url_redirects_before_it_paints_a_different_tab():
    """The redirect short-circuits setTab rather than following it: painting
    Account Management first and navigating afterwards flashes the wrong panel
    and rewrites adminTab in the history entry on the way out."""
    assert "ALLOWED_TABS = new Set(['users', 'activity'])" in ADMIN_TABS_JS
    assert "RETIRED_TABS = Object.freeze({ providers: '/admin#providers' })" in ADMIN_TABS_JS
    # The guard runs before normalizeTab, which is the whole point.
    body = ADMIN_TABS_JS[ADMIN_TABS_JS.index("function setTab("):]
    assert body.index("retiredDestination(value)") < body.index("normalizeTab(value)")


def test_default_tab_is_users_and_analytics_is_not_a_tab():
    assert "DEFAULT_TAB = 'users'" in ADMIN_TABS_JS
    assert "'analytics'" not in ADMIN_TABS_JS
    assert "admin-analytics" not in ADMIN_TABS_JS
    # window.location.assign is banned and replace is the one navigation this
    # controller may make, only for the retired Providers tab (N5). replace
    # drops the *current* entry -- the retired ?adminTab=providers URL -- and
    # leaves the entry the operator came from, one further back, untouched.
    # assign kept the retired URL reachable by Back, which either re-ran the
    # redirect or restored the pre-redirect console from bfcache showing Account
    # Management, the silent fallback N5 exists to prevent.
    assert "window.location.assign" not in ADMIN_TABS_JS
    assert ADMIN_TABS_JS.count("window.location.replace") == 1
    assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS


def test_admin_user_query_prefills_account_management():
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        "setHref('https://atl.example/app?view=admin&adminTab=users&adminUserQuery=ada%40example.test');"
        "window.AdminTabs.onEnter();"
    )
    assert result == {"nav": [], "submitted": ["submit"], "query": "ada@example.test"}
