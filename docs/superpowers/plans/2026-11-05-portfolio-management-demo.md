# Portfolio Management Demo Subtab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a **Portfolio Management** subtab before My Agents that plays the approved weekly-v2 case-2 recording, identical to the standalone build's shell view, without changing any other ATL behaviour.

**Architecture:** Two new deferred scripts. One is the recording, copied verbatim. The other is the ported page, which mounts the demo's markup and CSS into a **Shadow DOM root** attached to an empty panel div. ATL's existing subtab code calls `show(panel)` and `hide()` on it. The shadow root isolates styles both ways: the demo's global rules (`*`, `button`, `[hidden]`, `:focus-visible`, keyframes, ids) cannot reach ATL, and ATL's `* { margin: 0; padding: 0 }` (`styles.css:6`) and bare `.chip` rule (`styles.css:1463`) cannot reach the demo. So the 40 KB stylesheet and the markup port nearly verbatim, with no id or class renames.

**Tech Stack:** Vanilla JS (ES5 style, like the source), Shadow DOM, pytest plus `node` (via `tests/_frontend_source.py`), and Windows headless Chrome for the parity renders.

**Spec:** `docs/superpowers/specs/2026-11-04-portfolio-management-demo-design.md`

## Global Constraints

- **Names:**
  - slug `portfolio-management`;
  - tab label `Portfolio Management`;
  - panel id `playgroundPortfolioManagementPanel`;
  - files `dashboard/frontend/js/portfolio-management-data.js` and `dashboard/frontend/js/portfolio-management.js`;
  - the only globals are `window.PortfolioManagementRecording` and `window.PortfolioManagementDemo`.
- **Default tab:** My Agents stays the default. Every `'agents'` default and fallback in `app.html` and `app.js` stays as it is.
- **Source (read-only):** `/mnt/d/AgenticTradingLab/Events/IBM-Nov-6/atl-demo/weekly-v2/src/`, recorder run `20261104T050945Z-3a2f26`. Never regenerate `data.js`.
- **Out of bounds:** no new routes. No edits to `styles.css`, `middleware.py`, `vercel.json` or the Google Fonts link.
- **Cache-buster:** bump `app.js?v=164` to `165` in `app.html` and in all 4 pinning tests (`test_frontend_fast_boot.py`, `test_admin_analytics_frontend.py`, `test_analytics_frontend.py`, `test_backtest_comparison_frontend.py`). Grep, never recall.
  - New `?v=1` pins go only in `test_frontend_fast_boot.py`, which calls itself the single owner of cache-buster pins.
  - Feature tests match `?v=\d+`.
- **Banned visible copy:** "Certified Rebalance", "deterministic", "live", "paper trading", "audit", "sample".
- **The demo never touches:** the network, storage or `history`, and it never writes `location.hash`.
- **Platform:** desktop Chrome and Edge only.
- **Ad-hoc scripts:** use `node`, or `/mnt/d/CENTRAL-DATABASE/.venv/Scripts/python.exe -I` (stdlib only). Trees and screenshots go on ext4 or under `D:\`, never under `/tmp`.
- **Commits:** one local commit per task on `feat/portfolio-management-demo`.
  - No push, PR or merge until the owner asks.
  - The spec and this plan stay uncommitted until the owner says.
- **User-facing docs:** never edit them. `docs/source/lab/getting_started.rst` is logged as a follow-up (spec §9).

### Deliberate deviations from the spec (for the owner's review)

1. **Style isolation uses a Shadow DOM root, not selector prefixing (§7.6).** It meets every item on §7.6's list: no rules outside the root, tokens move to `:host`, `color-scheme` is scoped, and colliding names are confined. It also blocks the reverse leak (ATL into the demo), which prefixing cannot.
2. **`data-mode="shell"` stays as a static attribute on `.wr`, and its `.wr[data-mode="shell"]` rules stay verbatim (§7.1 says to remove `data-mode`).** Rewriting them would change selector specificity, and with it the cascade. Nothing reads or toggles the attribute. Every stage rule, `S.mode` and `#modeBtn` still go.
3. **A read-only step link, `#pm-step-N` or `#pm-step-9a`/`#pm-step-9b`, is read once on first show (§8 left it open).**
   - It exists so the parity renders can reach steps 5–9. It also lets a presenter rehearse from any step.
   - The demo never writes it, and it matches no `NAV_VIEW_MAP` key.
   - ATL keeps the hash in its URLs, so a reload with the link present reopens that step. Without it, a reload opens step 1.
4. **The page script shows itself at load when the boot attributes already say Portfolio Management.** On a reload, `app.js` navigates only after its auth awaits (`app.js:6697–6751`), so without this the panel would sit empty for that long. Otherwise loading the script registers nothing.

## Review Focus

1. **Typing in ATL controls while the tab shows** (any ATL input, the auth modal, the account menu): keys type normally. Only arrows, PgUp/PgDn, Space and Esc act, and only when focus is on the page body or inside the demo. → Task 3, `keyAction` table.
2. **A reload or direct link** `/app?view=portfolio-management` (a presenter mid-talk): the Portfolio Management tab is highlighted and the demo paints at step 1. My Agents never flashes first, and the panel is never empty while auth loads. → Task 1 boot-CSS test, Task 3 auto-show tests, Task 4 renders.
3. **Leaving mid-autoplay, mid-typing or during the step-9 pause, then coming back:** nothing ran while away, the step is unchanged and settled, and the button shows ▶ Play. → Task 3 `hide` pin plus the Task 4 manual check.
4. **Back and forward between Portfolio Management and other pages:** the panel hides with its listeners removed, and coming back resumes where it was. → Task 1 `navigateToPage` test plus the Task 4 manual check.
5. **The About panel:** it covers ATL's header and ticker, and Esc, × and a backdrop click all close it. ATL's own Esc handlers (account menu, run-backtest modal) still work. → Task 3 z-index pin plus the Task 4 manual check.

---

### Task 1: ATL wiring for the new subtab

**Files:**
- Modify: `dashboard/frontend/app.html`:
  - `NAV_VIEW_MAP` at lines 28–56;
  - boot CSS at lines 157–163;
  - the subtab row at line 954;
  - the new panel just before `#playgroundAgentsPanel` (line 958);
  - the `app.js?v=` tag at line 2519.
- Modify: `dashboard/frontend/app.js`:
  - `viewParamForNavState` at 11282;
  - `showPlaygroundPanel` at 11563;
  - the `navigateToPage` hide list at about 11755–11770.
- Modify (pin bump 164 to 165): the 4 test files listed in Global Constraints.
- Create: `dashboard/backend/tests/test_frontend_portfolio_management.py`

**Interfaces:**
- Consumes `window.PortfolioManagementDemo?.show(panel)` and `.hide()` (Task 3). Optional chaining keeps this task working before Task 3 lands.
- Produces:
  - the panel `<div id="playgroundPortfolioManagementPanel" class="playground-panel" style="display: none;"><p>Loading Portfolio Management…</p></div>`, inside `#playgroundView`, right after the subtab row;
  - `playgroundTab === 'portfolio-management'`.
- Not changed, on purpose: the boot sanitiser (`resolveBootNavigation`, `migrateSavedNavState`) and `resolveInitialNavigation`. Both validate only the page and pass the tab through, so a saved `portfolio-management` tab survives a reload as it is.

- [ ] **Step 1: Write the failing tests** in `test_frontend_portfolio_management.py`. Use `APP_HTML`, `APP_JS`, `fn_body`, `run_node` and `strip_comments` from `_frontend_source`, and a `requires_node` skipif.

```python
def test_subtab_sits_before_my_agents_and_is_not_active():
    row = APP_HTML[APP_HTML.index('<div class="subtab-bar playground-subtabs"'):]  # not the boot CSS at ~line 157
    row = row[: row.index("</div>")]
    tabs = re.findall(r'<button class="([^"]*)" data-playground-tab="([^"]+)"[^>]*>([^<]+)</button>', row)
    assert [(t[1], t[2]) for t in tabs[:2]] == [
        ("portfolio-management", "Portfolio Management"), ("agents", "My Agents")]
    assert "active" not in tabs[0][0].split() and "active" in tabs[1][0].split()

def test_view_map_routes_the_slug_and_reserves_no_pm_keys():
    assert "'portfolio-management': { page: 'playground', playgroundTab: 'portfolio-management' }," in APP_HTML
    nav_map = APP_HTML[APP_HTML.index("var NAV_VIEW_MAP"): APP_HTML.index("window.NAV_VIEW_MAP")]
    assert not re.search(r"^\s*'pm-", nav_map, re.M)   # the step link (#pm-step-N) must never route

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
    assert sel + ' #playgroundAgentsPanel' in APP_HTML      # joined to the existing display:none group
    assert sel + ' #playgroundPortfolioManagementPanel { display: block !important; }' in APP_HTML

def test_show_playground_panel_shows_the_demo_and_hides_it_elsewhere():
    body = strip_comments(fn_body("function showPlaygroundPanel("))
    branch = body[body.index("tab === 'portfolio-management') {"):]
    branch = branch[: branch.index("} else {")]
    assert "showPortfolioManagementDemo(" in branch and "loadAgents" not in branch
    assert "if (tab !== 'portfolio-management') window.PortfolioManagementDemo?.hide();" in body
    helper = strip_comments(fn_body("function showPortfolioManagementDemo("))
    assert ".show(panel)" in helper and "console.error" in helper
    assert "Portfolio Management did not load. Reload the page to try again." in helper   # fail-visible

def test_navigate_to_page_hides_the_panel_and_stops_the_demo_off_playground():
    body = strip_comments(fn_body("function navigateToPage("))
    assert "hide(document.getElementById('playgroundPortfolioManagementPanel'));" in body
    assert "if (page !== 'playground') window.PortfolioManagementDemo?.hide();" in body
```

- [ ] **Step 2: Run the tests and confirm they fail.**
Run: `pytest dashboard/backend/tests/test_frontend_portfolio_management.py -v`
Expected: every test FAILs, with a missing string or "substring not found".

- [ ] **Step 3: Implement the wiring.**
  - **`app.html`:**
    - add the `NAV_VIEW_MAP` entry after `agents:`;
    - add the three boot rules: join the button highlight to the existing `agents/backtest/paper` group, join `#playgroundAgentsPanel` to the existing `display: none` group, and add a new `display: block !important` rule for the panel;
    - put the button `<button class="subtab-btn" data-playground-tab="portfolio-management">Portfolio Management</button>` before My Agents;
    - add the panel div (Interfaces).
  - **`app.js`:**
    - `viewParamForNavState`: add `if (state.playgroundTab === 'portfolio-management') return 'portfolio-management';` before `return 'agents'`.
    - `showPlaygroundPanel`: set the panel's `style.display` alongside the other three panels. Then add the `hide()` line above the `if (tab === 'backtest')` chain, and a new `else if (tab === 'portfolio-management')` branch before the final `else`.
    - That branch only sets `currentMode = 'portfolio-management'` and calls `showPortfolioManagementDemo(panel)`. Keep it free of an inner `else`, which the test's slice relies on.
    - Add a new top-level `function showPortfolioManagementDemo(panel)` beside `showPlaygroundPanel`. If `window.PortfolioManagementDemo` is missing, it sets `panel.textContent = 'Portfolio Management did not load. Reload the page to try again.'`. Otherwise it calls `window.PortfolioManagementDemo.show(panel)` inside a `try` whose `catch` logs `console.error` and sets the same text.
    - `navigateToPage`: add the panel to the hide list, and add the `page !== 'playground'` hide line right after the list.
  - **Pins:** bump `app.js?v=164` to `165` in `app.html` and in the 4 tests.
  - **Note (analytics):** the tab is NOT logged. `analytics.js` `resolvePageView` reads `PAGE_VIEW_MAP[`playground_${tab}`]`, which has no entry here, so it returns null, no page view is recorded and the previous view's dwell timer ends. Accepted: adding a map entry is new instrumentation, a non-goal. Do not count demo visits from analytics.

- [ ] **Step 4: Run and verify.**
Run: `node --check dashboard/frontend/app.js && pytest dashboard/backend/tests/test_frontend_portfolio_management.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_frontend_marketplace_placement.py dashboard/backend/tests/test_frontend_live_trading_board.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit.** Message: `feat(portfolio-management): wire the Portfolio Management subtab before My Agents`. Stage only `app.html`, `app.js` and the 5 test files.

---

### Task 2: The recording file

**Files:**
- Create: `dashboard/frontend/js/portfolio-management-data.js`
- Test: `dashboard/backend/tests/test_frontend_portfolio_management.py` (append)

**Interfaces:**
- Produces `window.PortfolioManagementRecording`, the object `src/data.js` assigns to `var REC`, byte for byte (keys such as `ticker` that nothing reads any more stay in).

- [ ] **Step 1: Write the failing test.** Add a module-level node harness, `_HARNESS(files, attrs, extra="")`. It returns JS that:
  - builds a `vm` context whose stub `document` and `window` record every `addEventListener`, `setTimeout`, `setInterval` and `requestAnimationFrame` call into `events`;
  - gives `document.documentElement.getAttribute` the boot attributes `attrs`;
  - makes `getElementById` return `null`;
  - sets `document.body` to `{tagName: 'BODY'}`;
  - runs each file with `fs.readFileSync` plus `vm.runInContext`;
  - evaluates `extra` (JS) in the same context after the files, if given;
  - prints `{events, windowKeys, strayGlobals, rec, extra}`:
    - `windowKeys` are the keys the scripts added to the stub `window`, sorted;
    - `strayGlobals` are the context keys the scripts added, which would come from a top-level `var`;
    - `rec` is `[R.run.id, R.run.calls]` when the recording is present;
    - `extra` is the value of the `extra` code.

  Task 3 reuses it. The file paths go in as JS string literals, so the `node -e` argument stays small.

```python
@requires_node
def test_recording_defines_one_global_and_nothing_else():
    out = run_node(_HARNESS(["portfolio-management-data.js"], {"data-nav-page": "home"}))
    assert out["windowKeys"] == ["PortfolioManagementRecording"]
    assert out["strayGlobals"] == [] and out["events"] == []
    assert out["rec"] == ["20261104T050945Z-3a2f26", 8]

def test_recording_header_says_generated_do_not_edit():
    head = (FRONTEND / "js" / "portfolio-management-data.js").read_text(encoding="utf-8").splitlines()[0]
    assert "Generated, do not edit" in head and "20261104T050945Z-3a2f26" in head
```

- [ ] **Step 2: Run it and confirm it fails** (`FileNotFoundError`).
Run: `pytest dashboard/backend/tests/test_frontend_portfolio_management.py -k recording -v`

- [ ] **Step 3: Generate the file with a shell pipeline, not by hand.**
  - Line 1: `/* Portfolio Management demo recording. Generated, do not edit: copied on 2026-11-05 from the standalone build weekly-v2 src/data.js (content/build_case2.py, recorder run 20261104T050945Z-3a2f26). Only the declaration changed: var REC -> window.PortfolioManagementRecording. */`
  - Line 2: source line 2 with its leading `var REC = ` replaced by `window.PortfolioManagementRecording = `.

- [ ] **Step 4: Verify.**
  - `node --check dashboard/frontend/js/portfolio-management-data.js`.
  - The byte-identity check, `cmp <(sed -n '2s/^var REC = //p' "$SRC/data.js") <(sed -n '2s/^window.PortfolioManagementRecording = //p' dashboard/frontend/js/portfolio-management-data.js)`, prints nothing.
  - The Step 2 tests PASS.

- [ ] **Step 5: Commit.** Message: `feat(portfolio-management): add the recorded case-2 run`.

---

### Task 3: The page module

**Files:**
- Create: `dashboard/frontend/js/portfolio-management.js`
- Modify: `dashboard/frontend/app.html`. Put the two script tags directly after `home-news-signals.js` (line 2533): the data file first, then the page, both `?v=1 defer`. That keeps them clear of #708's hunk at lines 2526–2527.
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py`. Add `assert "js/portfolio-management-data.js?v=1" in APP_HTML` and the matching line for `portfolio-management.js?v=1`, directly after the `app.js?v=165` line, which avoids #708's change at line 200.
- Test: `test_frontend_portfolio_management.py` (append)

**Interfaces:**
- Consumes `window.PortfolioManagementRecording` (Task 2) and the panel div (Task 1).
- Produces `window.PortfolioManagementDemo = { show, hide, keyAction }`:
  - `show(panel: HTMLElement|null): void`:
    - it is a no-op when `panel` is null;
    - the first call attaches `panel.attachShadow({ mode: 'open' })` and inserts `<style>` + `CSS` and `MARKUP`. It wires the element listeners once, then calls `goto` with the step from the step link, or `goto(1, 'a')`;
    - any call made while inactive adds the document and window listeners, sets the module active, and redraws (`scrollBottom(false)` + `drawLanes(loopState().lane)`) in `requestAnimationFrame`. A call made while already active does nothing.
  - `hide(): void`: does nothing when inactive. Otherwise it settles the timers (Step 3, item 4), removes every document and window listener, and marks the module inactive.
  - `keyAction(e, host, activeEl): 'next'|'prev'|'toggle'|'escape'|null` is pure. It returns null when any of these holds:
    - `e.defaultPrevented`, or any of `ctrlKey`, `metaKey`, `altKey`;
    - the origin (`e.composedPath ? e.composedPath()[0] : e.target`) is an INPUT, TEXTAREA or SELECT, or is `isContentEditable`;
    - `activeEl` is set and is none of `document.body`, `document.documentElement`, `host`, a node with `host.contains(activeEl)`, or an ATL nav control (`activeEl.closest('.subtab-btn, .mode-btn, [data-playground-tab]')`). Clicking the Portfolio Management subtab leaves focus on that ATL button, and the keys must work right after;
    - the key is `' '` and the origin is a BUTTON or SUMMARY (or an `<a>`): Space then belongs to the native control, so return null. The arrows, PgUp/PgDn and Escape still work from a demo button.

    Add table cases for: focus on the ATL subtab button (arrows work), Space on a demo BUTTON and on a SUMMARY (null), and Space on body (`'toggle'`). Otherwise ArrowRight or PageDown returns `'next'`, ArrowLeft or PageUp returns `'prev'`, `' '` returns `'toggle'`, Escape returns `'escape'`, and any other key returns null.
  - `onKeydown` and `keyAction` also return early when `!panel.offsetParent` (panel hidden): a redirect that skips `navigateToPage` must not leave the handler eating Space and the arrows on another page. Test: panel `display:none` with a keydown dispatched gives no `preventDefault`.
  - At load: if `data-nav-page` is `playground` and `data-nav-playground-tab` is `portfolio-management` on `<html>`, it calls `show(document.getElementById('playgroundPortfolioManagementPanel'))`. Otherwise it does nothing.

- [ ] **Step 1: Write the failing tests.** Append them, and reuse `_HARNESS`. Read `SRC` (the page file's text) and `DATA_SRC` (the data file's text) lazily, so a missing file fails the test rather than collection. Set `PAGE = ["portfolio-management-data.js", "portfolio-management.js"]`.
  - `test_key_action_table` makes one `run_node(_HARNESS(PAGE, agents_attrs, extra))` call.
  - `extra` defines `INSIDE = {tagName:'DIV'}`, `HOST = {contains: n => n === INSIDE}` and `OUTSIDE = {tagName:'INPUT'}`. It maps `'body'|'host'|'inside'|'outside'` to `document.body|HOST|INSIDE|OUTSIDE`.
  - It returns `KEY_CASES.map(c => window.PortfolioManagementDemo.keyAction(Object.assign({target: document.body}, EVENT), HOST, ACTIVE))`.
  - The test asserts that list equals the expected column.

```python
@requires_node
def test_loading_on_another_tab_registers_nothing():
    for attrs in ({"data-nav-page": "playground", "data-nav-playground-tab": "agents"}, {"data-nav-page": "home"}):
        out = run_node(_HARNESS(PAGE, attrs))
        assert out["events"] == [] and out["strayGlobals"] == []
        assert out["windowKeys"] == ["PortfolioManagementDemo", "PortfolioManagementRecording"]

@requires_node
def test_auto_show_tolerates_a_missing_panel():   # boot says PM, getElementById -> null
    out = run_node(_HARNESS(PAGE, {"data-nav-page": "playground", "data-nav-playground-tab": "portfolio-management"}))
    assert out["events"] == []

KEY_CASES = [  # (event JS literal, activeEl: 'body'|'host'|'inside'|'outside', expected)
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
    ("{key:'ArrowRight'}", "outside", None),   # e.g. focus in an ATL header control
]

def test_document_and_window_listeners_come_off_on_hide():
    pat = r"(document|window)\.%sEventListener\(\s*'(\w+)',\s*(\w+)\s*\)"
    adds, removes = set(re.findall(pat % "add", SRC)), set(re.findall(pat % "remove", SRC))
    assert ("document", "keydown", "onKeydown") in adds and adds == removes
    assert not re.search(r"(document|window)\.addEventListener\(\s*'\w+',\s*function", SRC)

def test_hide_settles_every_timer():
    body = fn_body("function hide(", SRC)
    for name in ("setPlaying(false)", "typeTimer", "sendTimer", "phaseTimer"):
        assert name in body

def test_never_touches_url_storage_or_network():
    for pat in (r"location\.hash\s*=(?!=)", r"\bhistory\.", r"replaceState", r"pushState", r"localStorage",
                r"sessionStorage", r"\bfetch\(", r"XMLHttpRequest", r"\beval\(", r"new Function", r"document\.write",
                r"hashchange"):
        assert not re.search(pat, SRC), pat

def test_styles_and_markup_live_in_a_shadow_root_without_chrome_or_stage():
    assert "attachShadow({ mode: 'open' })" in SRC and ":host" in SRC
    for gone in (".atl-", "atl-header", "subtab-bar", 'data-mode="stage"', ":root", "modeBtn",
                 "detailsClose", "__LOGO__", "renderTicker", "S.mode"):
        assert gone not in SRC, gone
    assert re.search(r"\.about\s*\{[^}]*z-index:\s*1000", SRC)

def test_no_banned_copy():
    text = (SRC + DATA_SRC).lower()
    for phrase in ("certified rebalance", "deterministic", "paper trading", "audit", "sample"):
        assert phrase not in text, phrase
    # "live" is banned in visible copy only. The ported logic has `{ live: true }` (app.js:685), so scan
    # the DATA_SRC string literals and the MARKUP template text nodes, never identifiers.
    literals = re.findall(r"'((?:[^'\\\n]|\\.)*)'|\"((?:[^\"\\\n]|\\.)*)\"", DATA_SRC)
    markup_text = re.findall(r">([^<>]+)<", SRC)
    visible = " ".join(a or b for a, b in literals) + " " + " ".join(markup_text)
    assert not re.search(r"\blive\b", visible, re.I), "banned word 'live' in visible copy"

def test_scripts_load_after_app_js_data_first():
    i_app = APP_HTML.index('<script src="app.js?v=')
    i_data = re.search(r'<script src="js/portfolio-management-data\.js\?v=\d+" defer></script>', APP_HTML).start()
    i_page = re.search(r'<script src="js/portfolio-management\.js\?v=\d+" defer></script>', APP_HTML).start()
    assert i_app < i_data < i_page
```

- [ ] **Step 2: Run them and confirm they fail** (the file is missing).
Run: `pytest dashboard/backend/tests/test_frontend_portfolio_management.py -v`

- [ ] **Step 3: Build `js/portfolio-management.js`.**
  - **Layout:**
    - a provenance header: ported on 2026-11-05 from weekly-v2 Version 6 (`src/app.js`, `src/style.css`, `src/markup.html`), recorder run `20261104T050945Z-3a2f26`. It says this file is now the source, and the published standalone page is the reference rendering;
    - then one IIFE with `'use strict'` holding `var CSS = String.raw\`…\``, `var MARKUP = String.raw\`…\`` and the ported logic. Neither source file contains a backtick or `${` (checked).
  - **Tooling:** do the CSS and markup edits with a throwaway node script, kept beside `build.sh` as `weekly-v2/atl-port.mjs`, outside the repo. Never retype content.

  **CSS (from `src/style.css`).**
  - Delete:
    - the chrome rules: every selector starting `.atl-` and every `.subtab-bar` rule, including those inside `@media` blocks;
    - every `[data-mode="stage"]` rule.
  - Keep the `.wr[data-mode="shell"]` rules verbatim (deviation 2).
  - Fold both `:root { … }` token blocks into `:host`, and turn the `html`/`body` rules into `:host { all: initial; display: block; color: var(--text); font-family: var(--font-sans); font-size: 16px; line-height: 1.45; -webkit-font-smoothing: antialiased; color-scheme: dark; }`.
  - Drop `height`, `margin` and `background`. ATL's page background shows through. Restore `background: var(--bg)` only if a Task 4 render shows a seam.
  - Change `.about`'s `z-index: 40` to `1000` (ATL's modal tier is `.auth-modal`, 1000; the header is 2).
  - Keep everything else verbatim. The `*`, `button, input`, `[hidden]`, `:focus-visible` and keyframe rules now reach only the shadow tree.

  **Markup (from `src/markup.html`).**
  - Delete lines 2–23 (header, ticker, ticker-info, blank) and lines 25–30 (the fake subtab bar). KEEP line 24, the `<div class="atl-page" id="page">` wrapper, whose closing `</div>` stays; deleting 2–30 as one range removes its opening tag. With them goes the only use of `__LOGO__`, the GitHub link and the "Paper Trading" copy.
  - Delete `#modeBtn` and `#detailsClose`. The second is stage-only, hidden in shell by `.dt-x`.
  - Rename the wrapper `<div class="atl-page" id="page">` to class `page`. Its CSS rule keeps `display: flex; flex-direction: column; min-height: 0;` and drops the padding, because ATL's `.page-view` supplies the outer padding. The `@media` padding rule for `.atl-page` goes too.
  - Ids stay unchanged; the shadow root scopes them.

  **Logic (from `src/app.js`): only these changes.**
  1. `$` becomes `shadow.getElementById(id)`, and `each` uses `shadow.querySelectorAll`. `var R = window.PortfolioManagementRecording; if (!R) return;` runs before the derived constants, so a missing recording leaves `PortfolioManagementDemo` undefined, and Task 1's branch then shows its failure text.
  2. Remove:
     - `renderTicker` and its call;
     - `S.mode`, the `#modeBtn` listener, and every `S.mode === 'stage'` branch. Take the shell side each time: `aphChart(…, 96)`, `openTab` always scrolls, and the `details.open` toggle goes;
     - `S.details` and the `#detailsClose` listener, since nothing else reads them;
     - `writeHash`, `readHash`, `writing`, every `writeHash()` call and the `hashchange` listener.
  3. Replace `init()` and its DOMContentLoaded bootstrap with `show`/`hide` (Interfaces).
     - Element listeners bind once, on first show.
     - The document and window listeners are named handlers: `onKeydown` on document; `onResize` on window, which calls `drawLanes`; and `onWindowLoad` on window, which redraws. `onWindowLoad` is added only when `document.readyState !== 'complete'` at show.
     - `document.fonts.ready.then(…)` runs once, on first show, and its callback redraws only while active.
  4. **Timers:**
     - keep the `setTimeout` in `typeThenSend` (`src/app.js:326`) as `S.sendTimer`;
     - `hide()` calls `setPlaying(false)`;
     - if `S.typeTimer` or `S.sendTimer` is pending, it clears both, empties `$('composer')` and calls `render()`;
     - if `S.phaseTimer` is pending, it clears it, sets `S.phase = 'b'` and calls `render()`;
     - each cleared timer is set to null.
  5. `onKeydown(e)` maps `keyAction(e, panel, document.activeElement)`:
     - next and prev call `setPlaying(false)` first, as the source does;
     - toggle calls `setPlaying(!S.playing)`;
     - escape closes About (if it is open) and calls `render()`;
     - call `e.preventDefault()` only for next, prev and toggle.
  6. **Step link (read-only, goes stale by design):** ATL's `syncNavigationHistory` appends `location.hash` to every URL and the demo never rewrites it, so after stepping the URL still says the entry step. A reload or a copied URL reopens that step, not the current one. Rehearsal links must not be bookmarked or shared. "Reload opens step 1" holds only with no hash. On first show only, `var m = /^#pm-step-([1-9])([ab])?$/.exec(location.hash)`. If it matches, call `goto(+m[1], m[2] || 'a', { hold: !!m[2] })`; otherwise call `goto(1, 'a')`.
  7. Define `window.PortfolioManagementDemo`, then run the load-time auto-show check (Interfaces).

- [ ] **Step 4: Add the script tags and the two `test_frontend_fast_boot.py` pins** (see Files).

- [ ] **Step 5: Verify.**
  - Run: `node --check dashboard/frontend/js/portfolio-management.js && pytest dashboard/backend/tests/test_frontend_portfolio_management.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_frontend_api_base.py -v`
  - Expected: all PASS.

- [ ] **Step 6: Check content drift locally** (the source is outside CI).
  - `atl-port.mjs --check` strips tags and attributes from `MARKUP` and from `src/markup.html` minus the deleted parts, then compares the remaining text. Expected: identical.
  - Save `diff -u src/app.js <the logic part>` to `weekly-v2/atl-port.diff` for the owner's review. Expected: only items 1–7 above.

- [ ] **Step 7: Commit.** Message: `feat(portfolio-management): port the weekly rebalancing demo into a shadow-root panel`.

---

### Task 4: Parity renders, full suite, manual pass

**Files:**
- Create (outside the repo): `weekly-v2/shoot-atl.sh`, which takes `<name> <width> <height> <step>`. Model it on `shoot.sh` (fresh `--user-data-dir` per shot, `--virtual-time-budget=7000`).

**Interfaces:**
- Consumes the `#pm-step-N` link (Task 3) and `?view=portfolio-management` (Task 1).

- [ ] **Step 1: Run the full suite in the background,** controller-owned, from the worktree root: `pytest dashboard/backend/tests/ --timeout=180 -p no:cacheprovider`.
  - Expected: green, except the known local-only failures in `test_report_pdf.py` (no reportlab) and `test_robinhood_mcp_transport.py` (mcp 1.28.1 against the 1.30.0 pin).
  - If anything else fails, rerun that file on clean `origin/main` before calling it a regression.

- [ ] **Step 2: Serve the frontend on Windows** so headless Chrome reaches `localhost`.
  - Command: `/mnt/d/CENTRAL-DATABASE/.venv/Scripts/python.exe -I <scratchpad>/serve.py "$(wslpath -w dashboard/frontend)" 8765`, run in the background.
  - `serve.py` is stdlib only. It starts a `ThreadingHTTPServer` on `127.0.0.1` with `SimpleHTTPRequestHandler(directory=argv[1])`, and a `threading.Timer(900, server.shutdown)`, so it exits on its own.
  - Never stop it with `taskkill /IM python.exe`: the central-DB MCP servers run on Windows Python too.
  - There is no API on this server, so ATL's other panels and its ticker show empty or error states. That is expected. The Portfolio Management panel needs no API.

- [ ] **Step 3: Shoot both sides.**
  - Steps: 1, 5, 6, 7 and 9b.
  - Sizes: 1536×864 and 1920×1080.
  - ATL side: `http://localhost:8765/app.html?view=portfolio-management#pm-step-<N>`.
  - Standalone side: `shoot.sh` with `shell-<N>`.
  - Expected:
    - inside the panel, the content, layout, wrapping and colours match the standalone shell view;
    - ATL's header, ticker bar and subtab row sit above it, with Portfolio Management highlighted, placed first, and no My Agents flash;
    - accepted differences: JetBrains Mono 600 renders as 700, because ATL loads weights 400/500/700 and the Fonts link stays untouched; and the stand-in chrome is gone.
  - Shots go to `weekly-v2/shots/atl-*.png`.

- [ ] **Step 4: Manual pass by a person** (Chrome, the same local server, or the PR's Vercel preview once the owner asks for a PR). Check each item:
  1. A plain `/app` load lands on its usual page, never Portfolio Management unless that was the last tab.
  2. Click Portfolio Management: step 1 shows. → and ← move steps. The keyboard arrows, PgUp/PgDn and Space work with focus on the page.
  3. Click into any ATL input while on the tab: arrows and Space type normally.
  4. Start ▶ Play, switch to My Agents mid-typing, wait 15 s, come back: the same step, settled, with ▶ Play showing.
  5. On My Agents, Space and the arrows scroll the page as before.
  6. Back and Forward move between Portfolio Management and other pages correctly.
  7. Reload on Portfolio Management: it returns at step 1.
  8. About: covers ATL's header. Esc, × and a backdrop click each close it. The account menu still closes on Esc.
  9. Resize the window on the tab and while away: the lanes redraw correctly on return.

- [ ] **Step 5: Report to the owner.**
  - Include: the shot paths, the suite result, the manual-pass status, and `weekly-v2/atl-port.diff`.
  - Do not push. A PR, its body (concise, ending with the attribution line) and any merge wait for the owner.
  - Remind them of the follow-ups: `getting_started.rst` (user-facing, theirs to coordinate) and the deck's slide 6 split figures.
