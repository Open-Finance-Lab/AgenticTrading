"""Guards for the Portfolio Management subtab (the recorded weekly-rebalancing demo).

The tab plays a fixed recording inside a Shadow DOM root (js/portfolio-management.js),
so these guards pin the three seams where it could leak into the rest of /app:
the subtab wiring (My Agents must stay the default), what loading the two scripts
registers on other tabs (nothing), and which keys the demo is allowed to take.
"""

import json
import re
import shutil

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    FRONTEND,
    fn_body,
    run_node,
    strip_comments,
)

requires_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


# --- Task 1: ATL wiring ---------------------------------------------------------


def test_subtab_sits_before_my_agents_and_is_not_active():
    row = APP_HTML[APP_HTML.index('<div class="subtab-bar playground-subtabs"'):]  # not the boot CSS
    row = row[: row.index("</div>")]
    tabs = re.findall(r'<button class="([^"]*)" data-playground-tab="([^"]+)"[^>]*>([^<]+)</button>', row)
    assert [(t[1], t[2]) for t in tabs[:2]] == [
        ("portfolio-management", "Portfolio Management"), ("agents", "My Agents")]
    assert "active" not in tabs[0][0].split() and "active" in tabs[1][0].split()


def test_view_map_routes_the_slug_and_reserves_no_pm_keys():
    assert "'portfolio-management': { page: 'playground', playgroundTab: 'portfolio-management' }," in APP_HTML
    nav_map = APP_HTML[APP_HTML.index("var NAV_VIEW_MAP"): APP_HTML.index("window.NAV_VIEW_MAP")]
    assert not re.search(r"^\s*'pm-", nav_map, re.M)  # the step link (#pm-step-N) must never route


@requires_node
def test_view_param_round_trips_and_keeps_the_agents_default():
    fn = fn_body("function viewParamForNavState(")
    out = run_node(fn + "; console.log(JSON.stringify(["
        "viewParamForNavState({page:'playground', playgroundTab:'portfolio-management'}),"
        "viewParamForNavState({page:'playground'}),"
        "viewParamForNavState({page:'playground', playgroundTab:'agents'})]))")
    assert out == ["portfolio-management", "agents", "agents"]


def test_default_tab_is_still_my_agents():
    assert 'let playgroundTab = "agents";' in APP_JS
    assert "nav.playgroundTab || 'agents'" in APP_HTML
    assert "playgroundTab: initial.playgroundTab || 'agents'" in fn_body("function applyInitialNavigation(")


def test_boot_css_paints_the_tab_without_flashing_my_agents():
    sel = 'html[data-nav-boot][data-nav-page="playground"][data-nav-playground-tab="portfolio-management"]'
    assert sel + ' .playground-subtabs [data-playground-tab="portfolio-management"]' in APP_HTML
    assert sel + ' #playgroundAgentsPanel' in APP_HTML  # joined to the existing display:none group
    assert sel + ' #playgroundPortfolioManagementPanel { display: block !important; }' in APP_HTML


def test_show_playground_panel_shows_the_demo_and_hides_it_elsewhere():
    body = strip_comments(fn_body("function showPlaygroundPanel("))
    branch = body[body.index("tab === 'portfolio-management') {"):]
    branch = branch[: branch.index("} else {")]
    assert "showPortfolioManagementDemo(" in branch and "loadAgents" not in branch
    assert "if (tab !== 'portfolio-management') window.PortfolioManagementDemo?.hide();" in body
    helper = strip_comments(fn_body("function showPortfolioManagementDemo("))
    assert ".show(panel)" in helper and "console.error" in helper
    assert "Portfolio Management did not load. Reload the page to try again." in helper  # fail-visible


def test_navigate_to_page_hides_the_panel_and_stops_the_demo_off_playground():
    body = strip_comments(fn_body("function navigateToPage("))
    assert "hide(document.getElementById('playgroundPortfolioManagementPanel'));" in body
    assert "if (page !== 'playground') window.PortfolioManagementDemo?.hide();" in body


# --- Task 2: the recording ------------------------------------------------------


def _HARNESS(files, attrs, extra=""):
    """JS that loads `files` (from frontend/js/) into a stub browser and prints what they registered.

    The stub records every listener, timer and animation-frame request, so a script that
    reaches for the document at load (instead of on first show) shows up in `events`.
    Paths go in as string literals and the files are read by node, which keeps the
    `node -e` argument far under the 128 KB argv limit.
    """
    paths = [str(FRONTEND / "js" / f) for f in files]
    return """
const fs = require('fs'), vm = require('vm');
const events = [];
const rec = (kind) => function (type) { events.push(kind + (typeof type === 'string' ? ':' + type : '')); return 0; };
const ATTRS = %s;
const document = {
  documentElement: { getAttribute: (k) => (Object.prototype.hasOwnProperty.call(ATTRS, k) ? ATTRS[k] : null) },
  body: { tagName: 'BODY' }, readyState: 'loading', activeElement: null,
  getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  addEventListener: rec('document.addEventListener'),
  fonts: { ready: { then: rec('fonts.ready.then') } },
};
const window = { addEventListener: rec('window.addEventListener') };
const ctx = { window, document, console, location: { hash: '' },
  setTimeout: rec('setTimeout'), setInterval: rec('setInterval'),
  requestAnimationFrame: rec('requestAnimationFrame'), clearTimeout() {}, clearInterval() {} };
Object.assign(window, { document, location: ctx.location, setTimeout: ctx.setTimeout,
  setInterval: ctx.setInterval, requestAnimationFrame: ctx.requestAnimationFrame });
vm.createContext(ctx);
const ctxBefore = new Set(Object.keys(ctx)), winBefore = new Set(Object.keys(window));
for (const p of %s) vm.runInContext(fs.readFileSync(p, 'utf8'), ctx, { filename: p });
const extra = %s ? vm.runInContext(%s, ctx) : null;
const R = window.PortfolioManagementRecording;
console.log(JSON.stringify({ events,
  windowKeys: Object.keys(window).filter((k) => !winBefore.has(k)).sort(),
  strayGlobals: Object.keys(ctx).filter((k) => !ctxBefore.has(k)),
  rec: R ? [R.run.id, R.run.calls] : null, extra }));
""" % (json.dumps(attrs), json.dumps(paths), json.dumps(bool(extra)), json.dumps(extra))


@requires_node
def test_recording_defines_one_global_and_nothing_else():
    out = run_node(_HARNESS(["portfolio-management-data.js"], {"data-nav-page": "home"}))
    assert out["windowKeys"] == ["PortfolioManagementRecording"]
    assert out["strayGlobals"] == [] and out["events"] == []
    assert out["rec"] == ["20261104T050945Z-3a2f26", 8]


def test_recording_header_says_generated_do_not_edit():
    head = (FRONTEND / "js" / "portfolio-management-data.js").read_text(encoding="utf-8").splitlines()[0]
    assert "Generated, do not edit" in head and "20261104T050945Z-3a2f26" in head


# --- Task 3: the page module ----------------------------------------------------

PAGE = ["portfolio-management-data.js", "portfolio-management.js"]
_AGENTS_TAB = {"data-nav-page": "playground", "data-nav-playground-tab": "agents"}


def _page_src() -> str:  # read lazily: a missing file fails the test, not collection
    return (FRONTEND / "js" / "portfolio-management.js").read_text(encoding="utf-8")


def _data_src() -> str:
    return (FRONTEND / "js" / "portfolio-management-data.js").read_text(encoding="utf-8")


@requires_node
def test_loading_on_another_tab_registers_nothing():
    for attrs in (_AGENTS_TAB, {"data-nav-page": "home"}):
        out = run_node(_HARNESS(PAGE, attrs))
        assert out["events"] == [] and out["strayGlobals"] == []
        assert out["windowKeys"] == ["PortfolioManagementDemo", "PortfolioManagementRecording"]


@requires_node
def test_auto_show_tolerates_a_missing_panel():  # boot says PM, getElementById -> null
    out = run_node(_HARNESS(PAGE, {"data-nav-page": "playground", "data-nav-playground-tab": "portfolio-management"}))
    assert out["events"] == []


KEY_CASES = [  # (event JS literal, activeEl: body|host|inside|outside|navbtn, expected)
    ("{key:'ArrowRight'}", "body", "next"), ("{key:'PageDown'}", "host", "next"),
    ("{key:'ArrowLeft'}", "inside", "prev"), ("{key:'PageUp'}", "body", "prev"),
    ("{key:' '}", "body", "toggle"), ("{key:'Escape'}", "body", "escape"), ("{key:'a'}", "body", None),
    ("{key:'ArrowRight', target:{tagName:'INPUT'}}", "body", None),
    ("{key:' ', target:{tagName:'TEXTAREA'}}", "body", None),
    ("{key:'ArrowLeft', target:{tagName:'SELECT'}}", "body", None),
    ("{key:' ', target:{tagName:'DIV', isContentEditable:true}}", "body", None),
    ("{key:' ', target:HOST, composedPath:()=>[{tagName:'INPUT'}]}", "host", None),  # shadow retargeting
    ("{key:'ArrowRight', ctrlKey:true}", "body", None), ("{key:'ArrowRight', metaKey:true}", "body", None),
    ("{key:'ArrowRight', altKey:true}", "body", None), ("{key:'ArrowRight', defaultPrevented:true}", "body", None),
    ("{key:'ArrowRight'}", "outside", None),  # e.g. focus in an ATL header control
    # Clicking the Portfolio Management subtab leaves focus on that ATL button.
    ("{key:'ArrowRight', target:NAVBTN}", "navbtn", "next"),
    ("{key:' ', target:NAVBTN}", "navbtn", None),  # Space belongs to the focused button
    ("{key:'ArrowRight', target:{tagName:'BUTTON'}}", "inside", "next"),  # a demo button
    ("{key:' ', target:{tagName:'BUTTON'}}", "inside", None),
    ("{key:' ', target:{tagName:'SUMMARY'}}", "inside", None),
    ("{key:' ', target:{tagName:'A'}}", "inside", None),
]


@requires_node
def test_key_action_table():
    cases = ",".join("{ev:%s, active:%s}" % (ev, json.dumps(active)) for ev, active, _ in KEY_CASES)
    extra = """(() => {
  const INSIDE = {tagName:'DIV'}, OUTSIDE = {tagName:'INPUT'};
  const HOST = {tagName:'DIV', offsetParent:{}, contains: (n) => n === INSIDE};
  const NAVBTN = {tagName:'BUTTON', closest: (s) => (s.indexOf('.subtab-btn') >= 0 ? NAVBTN : null)};
  const HIDDEN = {tagName:'DIV', offsetParent:null, contains: () => false};
  const ACTIVE = {body: document.body, host: HOST, inside: INSIDE, outside: OUTSIDE, navbtn: NAVBTN};
  const ka = window.PortfolioManagementDemo.keyAction;
  return {
    table: [%s].map((c) => ka(Object.assign({target: document.body}, c.ev), HOST, ACTIVE[c.active])),
    hidden: ['ArrowRight', ' ', 'Escape'].map((k) => ka({key:k, target: document.body}, HIDDEN, document.body)),
  };
})()""" % cases
    out = run_node(_HARNESS(PAGE, _AGENTS_TAB, extra))["extra"]
    assert out["table"] == [want for _, _, want in KEY_CASES]
    assert out["hidden"] == [None, None, None]  # a redirect that skipped hide() must not eat keys


def test_keydown_handler_prevents_default_only_for_an_action():
    body = strip_comments(fn_body("function onKeydown(", _page_src()))
    assert "keyAction(e, panel, document.activeElement)" in body
    assert body.index("if (!action) return;") < body.index("preventDefault")


def test_document_and_window_listeners_come_off_on_hide():
    src = _page_src()
    pat = r"(document|window)\.%sEventListener\(\s*'(\w+)',\s*(\w+)\s*\)"
    adds, removes = set(re.findall(pat % "add", src)), set(re.findall(pat % "remove", src))
    assert ("document", "keydown", "onKeydown") in adds and adds == removes
    assert not re.search(r"(document|window)\.addEventListener\(\s*'\w+',\s*function", src)


def test_hide_settles_every_timer():
    body = fn_body("function hide(", _page_src())
    for name in ("setPlaying(false)", "typeTimer", "sendTimer", "phaseTimer"):
        assert name in body


def test_never_touches_url_storage_or_network():
    src = _page_src()
    for pat in (r"location\.hash\s*=(?!=)", r"\bhistory\.", r"replaceState", r"pushState", r"localStorage",
                r"sessionStorage", r"\bfetch\(", r"XMLHttpRequest", r"\beval\(", r"new Function", r"document\.write",
                r"hashchange"):
        assert not re.search(pat, src), pat


def test_styles_and_markup_live_in_a_shadow_root_without_chrome_or_stage():
    src = _page_src()
    assert "attachShadow({ mode: 'open' })" in src and ":host" in src
    for gone in (".atl-", "atl-header", "subtab-bar", 'data-mode="stage"', ":root", "modeBtn",
                 "detailsClose", "__LOGO__", "renderTicker", "S.mode"):
        assert gone not in src, gone
    assert re.search(r"\.about\s*\{[^}]*z-index:\s*1000", src)


def test_no_banned_copy():
    src, data = _page_src(), _data_src()
    text = (src + data).lower()
    for phrase in ("certified rebalance", "deterministic", "paper trading", "audit", "sample"):
        assert phrase not in text, phrase
    # "live" is banned in visible copy only. The ported logic has `{ live: true }`, so scan
    # the recording's string literals and the MARKUP text nodes, never identifiers.
    literals = re.findall(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"", data)
    markup = re.search(r"var MARKUP = String\.raw`([^`]*)`;", src).group(1)  # not the logic's `{ live: true }`
    markup_text = re.findall(r">([^<>]+)<", markup)
    visible = " ".join(a or b for a, b in literals) + " " + " ".join(markup_text)
    assert not re.search(r"\blive\b", visible, re.I), "banned word 'live' in visible copy"


def test_scripts_load_after_app_js_data_first():
    i_app = APP_HTML.index('<script src="app.js?v=')
    i_data = re.search(r'<script src="js/portfolio-management-data\.js\?v=\d+" defer></script>', APP_HTML).start()
    i_page = re.search(r'<script src="js/portfolio-management\.js\?v=\d+" defer></script>', APP_HTML).start()
    assert i_app < i_data < i_page
