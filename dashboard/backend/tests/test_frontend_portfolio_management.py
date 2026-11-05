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


@requires_node
def test_failure_text_shows_even_after_the_shadow_root_exists():
    # A shadow root with no <slot> hides the panel's light DOM, so a first show() that throws
    # after attachShadow must write the message into the root, or the tab is just blank.
    js = """
var window = {};
%s
var msg = 'Portfolio Management did not load. Reload the page to try again.';
var missing = {shadowRoot: null, textContent: ''};
showPortfolioManagementDemo(missing);
var broken = {shadowRoot: null, textContent: ''};
window.PortfolioManagementDemo = {show: function (p) { p.shadowRoot = {textContent: ''}; throw new Error('bind'); }};
console.error = function () {};
showPortfolioManagementDemo(broken);
console.log(JSON.stringify({missing: missing.textContent === msg, broken: broken.shadowRoot.textContent === msg}));
""" % fn_body("function showPortfolioManagementDemo(")
    assert run_node(js) == {"missing": True, "broken": True}


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


@requires_node
def test_escape_closes_about_without_rebuilding_the_page():
    # Rebuilding the page collapses an opened "show all" block or certificate tree, resets the
    # chat scroll and replays step 3's row reveal: Esc with About closed must do nothing, and
    # closing About must touch the overlay only.
    js = """
var renders = 0, syncs = 0, S = {about: false, playing: false};
function render() { renders++; }
function syncAbout() { syncs++; }
function keyAction() { return 'escape'; }
function setPlaying() {} function next() {} function prev() {}
var panel = {offsetParent: {}}, document = {activeElement: null};
%s
var ev = {preventDefault: function () {}};
onKeydown(ev); var closed = [renders, syncs];
S.about = true; onKeydown(ev);
console.log(JSON.stringify({closed: closed, renders: renders, syncs: syncs, about: S.about}));
""" % fn_body("function onKeydown(", _page_src())
    assert run_node(js) == {"closed": [0, 0], "renders": 0, "syncs": 1, "about": False}


def _key_action(cases, about, modal):
    js = """
var S = {about: %s};
var document = {body: {tagName: 'BODY'}, documentElement: {}, querySelector: function (sel) {
  return %s && sel.indexOf('.auth-modal') >= 0 ? {} : null; }};
var HOST = {offsetParent: {}, contains: function () { return true; }};
%s
console.log(JSON.stringify(%s.map(function (k) {
  return keyAction({key: k, target: document.body}, HOST, document.body); })));
""" % (json.dumps(about), json.dumps(modal), fn_body("function keyAction(", _page_src()), json.dumps(cases))
    return run_node(js)


@requires_node
def test_keys_belong_to_about_and_to_atl_dialogs_while_they_are_open():
    keys = ["ArrowRight", "PageDown", "ArrowLeft", "PageUp", " ", "Escape", "a"]
    assert _key_action(keys, about=False, modal=False) == ["next", "next", "prev", "prev", "toggle", "escape", None]
    # About open: the page behind it must not step, scroll or toggle autoplay; only Esc closes it.
    assert _key_action(keys, about=True, modal=False) == [None, None, None, None, None, "escape", None]
    # An ATL dialog (.auth-modal without [hidden]) owns every key.
    assert _key_action(keys, about=False, modal=True) == [None] * 7


@requires_node
def test_goto_cancels_a_queued_send():
    # typeThenSend queues the send 350 ms after the last character; a step change inside that
    # window must cancel it, or it blanks the composer and replays the chat on the new step.
    js = """
var cleared = [], S = {step: 1, phase: 'a', sendTimer: 7, typeTimer: null, phaseTimer: null};
var LAST = 9, CHAT = [], R = {msg: {parts: []}}, composer = {value: 'x'};
function $() { return composer; }
function clearTimeout(t) { cleared.push(t); } function clearInterval() {}
function render() {} function renderExplain() {} function typeThenSend() {}
%s
goto(2, 'a', {live: true});
console.log(JSON.stringify({cleared: cleared, sendTimer: S.sendTimer}));
""" % fn_body("function goto(", _page_src())
    out = run_node(js)
    assert 7 in out["cleared"] and out["sendTimer"] is None


@requires_node
def test_failed_first_build_is_retried_not_stuck():
    # show() used to set the shadow root before building, so a throw left it set: every later
    # show() skipped the build and the tab stayed broken (with key and resize listeners live).
    js = """
var built = false, stale = false, active = false, tries = 0, rendered = 0, listeners = [];
var document = {readyState: 'complete', addEventListener: function (t) { listeners.push('doc:' + t); }};
var window = {addEventListener: function (t) { listeners.push('win:' + t); }};
function requestAnimationFrame() {}
function onKeydown() {} function onResize() {} function onWindowLoad() {} function redraw() {}
function render() { rendered++; }
function build() { if (++tries === 1) throw new Error('bind'); }
%s
var first = 'no error';
try { show({}); } catch (e) { first = e.message; }
var afterFail = {active: active, built: built, listeners: listeners.length};
show({});
console.log(JSON.stringify({first: first, afterFail: afterFail, active: active, built: built,
  tries: tries, listeners: listeners}));
""" % fn_body("function show(", _page_src())
    out = run_node(js)
    assert out["first"] == "bind" and out["afterFail"] == {"active": False, "built": False, "listeners": 0}
    assert out["active"] and out["built"] and out["tries"] == 2
    assert out["listeners"] == ["doc:keydown", "win:resize"]


@requires_node
def test_build_reuses_a_root_left_by_a_failed_build():
    # attachShadow throws on an element that already has a root, so a retry must reuse it.
    js = """
var kids = ['stale-style', 'stale-markup'], attached = 0, appended = [];
var root = {get firstChild() { return kids[0] || null; }, removeChild: function () { kids.shift(); },
  appendChild: function (n) { appended.push(n); }};
var panel;
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
var CSS = 'css', MARKUP = '<p></p>', location = {hash: ''}, shadow;
var document = {createElement: function () { return {content: 'content'}; }, fonts: null};
function bindElements() {} function goto() {} function redraw() {}
%s
build({shadowRoot: root, attachShadow: function () { attached++; throw new Error('already attached'); }});
console.log(JSON.stringify({attached: attached, left: kids.length, appended: appended.length}));
""" % fn_body("function build(", _page_src())
    assert run_node(js) == {"attached": 0, "left": 0, "appended": 2}


@requires_node
def test_hide_settles_state_without_drawing_into_a_hidden_panel():
    # The panel is display:none when hide() runs, so every box measures 0. hide() marks the
    # page stale and show() redraws once it is laid out.
    js = """
var active = true, stale = false, rendered = 0, playing = null;
var S = {typeTimer: 5, sendTimer: null, phaseTimer: 9, phase: 'a'};
var composer = {value: 'half typed'};
function $() { return composer; } function render() { rendered++; }
function setPlaying(on) { playing = on; } function clearInterval() {} function clearTimeout() {}
var document = {removeEventListener: function () {}}, window = {removeEventListener: function () {}};
function onKeydown() {} function onResize() {} function onWindowLoad() {}
%s
hide();
console.log(JSON.stringify({rendered: rendered, stale: stale, active: active, phase: S.phase,
  composer: composer.value, playing: playing}));
""" % fn_body("function hide(", _page_src())
    assert run_node(js) == {"rendered": 0, "stale": True, "active": False, "phase": "b",
                            "composer": "", "playing": False}


@requires_node
def test_resize_redraws_lanes_and_the_evidence_chart_once_per_frame():
    # The chart takes its width from its box at draw time, so a resize that redraws only the
    # lanes leaves it letterboxed. Resize events fire many times a frame: coalesce them.
    js = """
var active = true, resizeFrame = 0, frames = [], calls = [];
function requestAnimationFrame(fn) { frames.push(fn); return frames.length; }
function loopState() { return {ev: 'chart', lane: 'return'}; }
function renderEvidence(k) { calls.push('ev:' + k); } function drawLanes(k) { calls.push('lanes:' + k); }
%s
onResize(); onResize(); onResize();
var queued = frames.length;
frames[0]();
active = false; onResize(); frames[1](); // a frame that lands after hide() draws nothing
console.log(JSON.stringify({queued: queued, calls: calls}));
""" % "\n".join(fn_body(sig, _page_src()) for sig in ("function relayout(", "function onResize("))
    assert run_node(js) == {"queued": 1, "calls": ["ev:chart", "lanes:return"]}


def test_about_and_details_clicks_do_not_rebuild_the_page():
    src = _page_src()
    for sig in ("function bindElements(", "function openTab("):
        body = strip_comments(fn_body(sig, src))
        assert not re.search(r"(?<![A-Za-z])render\(", body), sig  # renderDetails() / syncAbout() only
    assert "syncAbout()" in fn_body("function bindElements(", src)
    assert "renderDetails()" in fn_body("function openTab(", src)


@requires_node
def test_live_region_is_written_only_when_the_announcement_changes():
    js = """
var writes = 0, text = '', live = {get textContent() { return text; }, set textContent(v) { writes++; text = v; }};
var S = {step: 3, phase: 'a'}, LAST = 9, STEPS = [{}, {}, {title: 'T', line: 'L', next: 'N'}];
var nodes = {'ex-title': {}, 'ex-line': {}, 'ex-next': {}, live: live};
function $(id) { return nodes[id]; } function each() {}
%s
renderExplain(); renderExplain(); S.step = 3; renderExplain();
console.log(JSON.stringify({writes: writes, text: text}));
""" % fn_body("function renderExplain(", _page_src())
    assert run_node(js) == {"writes": 1, "text": "Step 3: T. L"}


# --- narrative numbers come from the recording -----------------------------------


def _recording_facts():
    paths = json.dumps(str(FRONTEND / "js" / "portfolio-management-data.js"))
    js = """
var window = {}; require('vm').runInThisContext(require('fs').readFileSync(%s, 'utf8'));
var R = window.PortfolioManagementRecording;
console.log(JSON.stringify({
  stocks: R.ticker.length, d1: R.d1.map(function (d) { return d.ticker; }), hon: R.hon.length,
  codeOnly: R.runs.code_only.map(function (r) { return [r.final, r.orders]; }),
  finalRuns: R.runs.final.length, control: R.runs.control.length, orders: R.fills.n_orders,
  fill: R.checks.fill.map(function (c) { return c[1]; }),
}));
""" % paths
    return run_node(js)


@requires_node
def test_prose_counts_are_facts_about_the_recording():
    # The page derives its counts from the data; these pin what the sentences around them
    # assert, so a re-recorded run that breaks one fails here rather than on stage.
    f = _recording_facts()
    assert f["d1"] == ["APH"] + ["HON"] * f["hon"]            # "APH's split and N HON spin-offs"
    assert f["stocks"] == 31 and f["finalRuns"] == 5 and f["control"] == 1
    assert {tuple(r) for r in f["codeOnly"]} == {("EXECUTE", f["orders"])}  # "the same fix and the same N orders"
    assert set(f["fill"]) == {"ok"}                            # "After-fill checks: N of N pass"


def test_no_recording_count_is_spelled_out_in_the_page_logic():
    code = strip_comments(_page_src())
    markup = re.search(r"var MARKUP = String\.raw`([^`]*)`;", code).group(1)
    logic = code.replace(markup, "")
    for stale in (r"\bthe 31 stocks", r"\bThree days\b", r"\bthose three events", r"\bTwo HON adjustments",
                  r"\bIn five runs", r"'✕ 3 days", r"\b4 of 4 pass", r"\(1\.39%", r"\(1\.53%",
                  r"'5 recorded runs", r"'5 runs, critic", r"'1 control"):
        assert not re.search(stale, logic), stale


@requires_node
def test_about_fill_words_match_the_markup_fallbacks():
    # MARKUP is static, so each <span data-fill> holds a fallback word. It must equal what the
    # page fills in, or the text flashes a wrong number before the fill (or when it fails).
    f = _recording_facts()
    words = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    want = {"code-only-n": words[len(f["codeOnly"])], "orders-n": words[f["orders"]], "hon-n": words[f["hon"]]}
    markup = re.search(r"var MARKUP = String\.raw`([^`]*)`;", _page_src()).group(1)
    got = dict(re.findall(r'<span data-fill="([a-z-]+)">([^<]*)</span>', markup))
    assert got == want
    assert re.search(r"var FILL = \{[^}]*'code-only-n'[^}]*'orders-n'[^}]*'hon-n'", _page_src())


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
