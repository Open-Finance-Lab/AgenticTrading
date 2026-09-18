# Admin Navigation Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `/admin` the application's navigation idiom and an account menu, move the Providers surface out of `/app?view=admin` into `/admin`, and rename the old console's "Users" tab to "Account Management".

**Architecture:** `/admin` is a no-build-step page — one HTML file, one CSS file, and deferred vanilla-JS IIFE modules that each expose exactly one global. `admin-shell.js` owns the gate, the hash router, and every network call; the other modules render. This plan adds a sixth module (`admin-providers.js`), gives the shell its one write path (`AdminShell.write`), retypes the legacy rail's CSS into `admin.css` against that file's own tokens, and removes the Providers tab from the old in-app console.

**Tech Stack:** Vanilla ES2022 (no bundler, no framework), CSS custom properties, FastAPI + pytest. Frontend behaviour is tested by lifting named functions out of the shipped `.js` with `dashboard/backend/tests/_admin_dom_stub.py` and running them under `node -e`.

**Spec:** `docs/superpowers/specs/2026-09-17-admin-nav-consolidation-design.md`

---

## Global Constraints

Copied from the spec and from `CLAUDE.md`. Every task's requirements implicitly include this section.

**Read/write discipline**
- `AdminShell.request(path)` stays hardcoded `method: 'GET'`. It gains no options bag. (N4)
- `AdminShell.write(path, {method, body})` is the **only** write path on `/admin`. No module calls `fetch` directly except `admin-shell.js`. (N4)
- Every write carries the CSRF double-submit header. See Task 1 — without it every Providers write and the logout 403 in production.
- Rendering is `textContent` only. `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`, `eval(`, `new Function(` are banned in every `js/admin-*.js`.
- `localStorage` and `sessionStorage` are banned in **every** admin module, read or write.
- Prohibited field names in read modules: `api_key`, `session_id`, `network_hash`, `provider_response_body`, `credential_ciphertext`, `prompt`, `strategy`, `portfolio`, `password`, `raw_user_agent`. In a write module `api_key` is permitted **only** on a line that also contains `JSON.stringify(` — as a request-body key, never anywhere else.
- No line in a write module may both name a credential (`api_key`, `secret`, `credential`, `token`, `password`, case-insensitive) and reach a DOM sink (`.textContent =`, `.innerHTML =`, or `.value =` with a non-empty right-hand side; or an argument to `setAttribute` / `append` / `appendChild` / `replaceChildren` / `createTextNode`).

**Page shape**
- `admin.html` carries **no inline script**. Every `<script>` is `src=` + `defer` + `?v=`.
- `admin.css` does **not** `@import` and never references `styles.css`. Rules moved between the two files are **retyped against the destination's tokens**. Token map: `--border-color`→`--line`, `--text-primary`→`--text`, `--text-secondary`/`--text-muted`→`--muted`, `--bg-card`/`--bg-surface`/`--bg-input`→`--surface`, `--info-color`→`--accent`, `#67e8f9`→`var(--accent)`.
- No panel region in `admin.html` may contain a numeric or percentage literal in its rendered text. Placeholders are the em dash `—`.
- Rail entries are `<a href="#route">` carrying `aria-current="page"`. They are **not** `role="tab"` / `aria-selected`. (N1)

**Cache-buster pins — use exactly these values, in the task that first touches the file**

| File | Old | New | Task |
|---|---|---|---|
| `js/admin-shell.js` | `?v=1` | `?v=2` | 1 |
| `admin.css` | `?v=1` | `?v=2` | 2 |
| `js/admin-providers.js` | — | `?v=1` (new) | 3 |
| `js/admin-tabs.js` | `?v=9` | `?v=10` | 4 |
| `app.js` | `?v=133` | `?v=134` | 4 |
| `styles.css` | `?v=141` | `?v=142` | 4 |
| `js/admin-credits.js` | `?v=6` | unchanged | — |

Project memory records that these pins are mirrored in **five** test files, not the one the docs name. Grep before assuming a pin has one owner:

```bash
grep -rn 'admin-shell.js?v=\|admin.css?v=\|admin-tabs.js?v=\|app.js?v=\|styles.css?v=' dashboard/backend/tests/
```

**Process**
- Run pytest from the repo root: `pytest dashboard/backend/tests/ -q`.
- Stage files **by name**. Never `git add -A`, `git add .`, `git add -u`, or `git add -f`.
- Never commit `dashboard/storage/data/backtest.db`.
- Never use bare `git stash` / `git stash pop` — the stack is shared across worktrees.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
  ```

---

## File Structure

**Created**
- `dashboard/frontend/js/admin-providers.js` — the ported Providers surface. One IIFE, one global `window.AdminProviders`, reads via `AdminShell.request`, writes via `AdminShell.write`.
- `dashboard/backend/tests/test_admin_providers_frontend.py` — node-driven behaviour tests for that module, in the idiom of `test_admin_users_frontend.py`.

**Modified**
- `dashboard/frontend/js/admin-shell.js` — gains `readCsrfToken`, `write`, `user`, the account-menu controller, `'providers'` in `ROUTES`, and rail active-state handling in `route()`.
- `dashboard/frontend/admin.html` — icon sprite, account menu, rail rebase, `<section id="providers">`, sixth script tag.
- `dashboard/frontend/admin.css` — rail, account menu and Providers rules retyped in; the flat `aside a` list rules removed. Roughly doubles in size.
- `dashboard/frontend/app.html` — Providers rail button and panel removed; "Users" tab relabelled "Account Management".
- `dashboard/frontend/app.js` — three `window.AdminModelProviders` call sites removed.
- `dashboard/frontend/js/admin-tabs.js` — `'providers'` out of `ALLOWED_TABS`, N5 redirect in.
- `dashboard/frontend/styles.css` — `.admin-provider-*` / `.admin-platform-key-*` families deleted.
- `dashboard/backend/tests/test_admin_page_modules.py` — the §9 guard rescope.
- `test_admin_page_shell.py`, `test_admin_shell_frontend.py`, `test_admin_tabs_redirect.py`, `test_admin_analytics_frontend.py`, `test_admin_credits_frontend.py` — assertions that pin what this PR changes.
- `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — the four §10 amendments.

**Deleted**
- `dashboard/frontend/js/admin-model-providers.js` — superseded by `admin-providers.js`.
- `dashboard/backend/tests/test_admin_model_providers_frontend.py` — all four of its tests assert against artifacts this PR removes (the `?view=admin` markup, the old module, the `app.js` wiring, the `styles.css` families). Its still-true content moves into `test_admin_providers_frontend.py`.

---

## Three spec gaps this plan closes

Found while reading source, not present in the approved spec. Each is load-bearing; none changes a decision.

1. **CSRF.** `dashboard/backend/csrf.py`'s `CsrfMiddleware` requires a matching `atl_csrf` / `__Host-atl_csrf` cookie and `X-CSRF-Token` header on every unsafe method that carries a session cookie. `/api/auth/logout` is **not** in `_CSRF_EXEMPT_PATHS`. The old providers module got this for free from `window.API.request` → `csrfHeaders()` in `app.js`; `/admin` cannot reach that function. `AdminShell.write` must read the cookie itself or every write 403s in production while passing every test that stubs `fetch`.
2. **The sprite needs seven symbols, not five.** Spec §6 names `icon-chart`, `icon-users`, `icon-network`, `icon-activity`, `icon-refresh`. The ported markup also references `#icon-check-circle` (both submit buttons) and the module calls `appendIcon(revoke, 'icon-x')`. A missing symbol renders an empty box with no error.
3. **The providers section must not nest a `<section>`.** `test_admin_page_shell.py`'s `PANEL_REGION` is non-greedy, so an outer `<section data-panel="providers">` wrapping an inner `<section>` matches only as far as the inner closing tag — the guard would silently scan a fragment. The port flattens the wrapper to a `<div>`.

---

## Task 1: `AdminShell` gains its write path

Adds the CSRF-bearing `write()`, the `user()` accessor N8 asks for, and nothing else. No markup, no CSS, no new module. Two shipped assertions change because the shell's `fetch` count moves from 2 to 3.

**Files:**
- Modify: `dashboard/frontend/js/admin-shell.js` (`state` at `:49`, new functions after `request()` at `:276`, `gate()` at `:311-326`, `api` object at `:536-545`)
- Modify: `dashboard/frontend/admin.html:13` (pin bump)
- Test: `dashboard/backend/tests/test_admin_shell_frontend.py`
- Test: `dashboard/backend/tests/test_admin_page_modules.py:50-60`
- Test: `dashboard/backend/tests/test_admin_page_shell.py:17-23, 78-84`
- Test: `dashboard/backend/tests/test_admin_analytics_frontend.py:149-162`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AdminShell.write(path, {method, body}) -> Promise<object|null>` — `method` is the verb string, `body` an already-serialised JSON string or `undefined`. Rejects with an `Error` carrying `.status`. Returns `null` on 204 or a non-JSON body.
  - `AdminShell.user() -> {display_name: string, email: string} | null` — the account `gate()` resolved, or `null` before the gate runs / after access is lost.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_admin_shell_frontend.py`:

```python
def test_write_sends_the_verb_the_body_and_the_csrf_double_submit_header():
    """The /app copy of this lives in app.js's csrfHeaders(); /admin cannot reach it,
    so the shell carries its own reader. Without the header CsrfMiddleware answers 403
    to every provider write and to logout -- and a fetch stub would never notice."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {saved: true}});"
        "  const saved = await window.AdminShell.write('/api/admin/model-providers/x', "
        "    {method: 'PUT', body: JSON.stringify({display_name: 'X'})});"
        "  const [url, options] = fetchCalls[0];"
        "  return {saved, url, method: options.method, credentials: options.credentials,"
        "          csrf: options.headers['X-CSRF-Token'], type: options.headers['Content-Type'],"
        "          body: options.body};"
        "})()",
        "globalThis.document.cookie = 'other=1; atl_csrf=tok%20en; more=2';",
    )
    assert result == {
        "saved": {"saved": True},
        "url": "/api/admin/model-providers/x",
        "method": "PUT",
        "credentials": "include",
        "csrf": "tok en",
        "type": "application/json",
        "body": '{"display_name":"X"}',
    }


def test_write_reads_the_host_prefixed_csrf_cookie_production_sets():
    """cookie_secure() picks __Host-atl_csrf in prod and atl_csrf in dev
    (backend/csrf.py:51-52). Reading only one name works in exactly one of them."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 204, body: null});"
        "  const body = await window.AdminShell.write('/api/auth/logout', {method: 'POST'});"
        "  return {body, csrf: fetchCalls[0][1].headers['X-CSRF-Token'], sent: 'body' in fetchCalls[0][1]};"
        "})()",
        "globalThis.document.cookie = '__Host-atl_csrf=prodtoken';",
    )
    assert result == {"body": None, "csrf": "prodtoken", "sent": False}


def test_write_without_a_csrf_cookie_omits_the_header_rather_than_sending_empty():
    """An empty header is not the same request as no header: csrf_tokens_match()
    rejects both, but only the absent one lets the middleware fall through to the
    cookie-less agent lane it was written for. Sending '' claims a token we lack."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {}});"
        "  await window.AdminShell.write('/api/admin/x', {method: 'POST', body: '{}'});"
        "  return {present: 'X-CSRF-Token' in fetchCalls[0][1].headers};"
        "})()",
        "globalThis.document.cookie = '';",
    )
    assert result == {"present": False}


def test_write_throws_the_servers_detail_with_its_status():
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: false, status: 403, body: {detail: 'CSRF token missing or invalid'}});"
        "  try { await window.AdminShell.write('/api/admin/x', {method: 'DELETE', body: '{}'}); }"
        "  catch (error) { return {message: error.message, status: error.status}; }"
        "  return 'did not throw';"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert result == {"message": "CSRF token missing or invalid", "status": 403}


def test_gate_retains_the_user_it_already_fetched():
    """N8: the account menu needs the display name and email, and the page has
    already paid for /api/auth/me. A second call for the same body is the thing
    this pin exists to stop."""
    granted = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin',"
        "    display_name: 'Ada Admin', email: 'ada@example.test'}}});"
        "  const admin = await window.AdminShell.gate();"
        "  return {admin, user: window.AdminShell.user(), calls: fetchCalls.length};"
        "})()"
    )
    assert granted == {
        "admin": True,
        "user": {"display_name": "Ada Admin", "email": "ada@example.test"},
        "calls": 1,
    }
    refused = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'user', email: 'bo@example.test'}}});"
        "  await window.AdminShell.gate();"
        "  return window.AdminShell.user();"
        "})()"
    )
    assert refused is None


def test_losing_access_drops_the_retained_user():
    """handleAccessLost redirects, but a redirect is not instantaneous: anything
    that reads user() in the same tick must not still see the identity."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin', display_name: 'A', email: 'a@x.test'}}});"
        "  await window.AdminShell.gate();"
        "  await window.AdminShell.handleAccessLost({status: 401});"
        "  return {user: window.AdminShell.user(), nav};"
        "})()"
    )
    assert result == {"user": None, "nav": [["replace", "/app"]]}
```

- [ ] **Step 2: Run them to verify they fail**

```bash
pytest dashboard/backend/tests/test_admin_shell_frontend.py -q -k "write or retains or losing"
```

Expected: FAIL. The first four report `TypeError: window.AdminShell.write is not a function` surfacing as a non-zero node exit through `run_node`'s `assert result.returncode == 0`; the last two report `AdminShell.user is not a function`.

- [ ] **Step 3: Add the CSRF reader and `write()` to `admin-shell.js`**

Insert immediately after `request()` (which ends at `:276`), before `nextSeq`:

```js
  // The /app copy of this is app.js's readCsrfToken(); /admin is a separate
  // document with no access to it, so the two are deliberate twins rather than
  // a shared helper -- linking them would mean importing app.js, which is the
  // 15,000-line inheritance this page exists to avoid. Both cookie names because
  // cookie_secure() picks __Host-atl_csrf in prod and atl_csrf in dev
  // (backend/csrf.py:51-52); reading one name works in exactly one environment.
  // `document.cookie || ''` is load-bearing, not defensive noise: the node test
  // stub has no cookie property at all.
  function readCsrfToken() {
    try {
      const raw = document.cookie || '';
      for (const name of ['atl_csrf', '__Host-atl_csrf']) {  // CORRECTED 2026-09-18 (finding I6): ships as ['__Host-atl_csrf', 'atl_csrf'] to match backend/csrf.py:79 precedence.
        const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const match = raw.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
        if (match) return decodeURIComponent(match[1]);
      }
    } catch (_error) { /* a document with no cookie access is a no-token document */ }
    return null;
  }

  // The page's ONE write path (design N4). Deliberately a second, differently
  // named function rather than an options bag on request(): "does this module
  // write?" has to stay answerable by grep, because that is exactly what
  // test_admin_page_modules.py asserts over the module set. An options bag would
  // make the two indistinguishable in source and leave the guard asserting
  // nothing.
  //
  // CsrfMiddleware requires the double-submit header on every unsafe method that
  // carries a session cookie, and /api/auth/logout is not exempt, so a write
  // without this header is a 403 in production that no fetch-stubbed test can
  // see. The header is omitted rather than sent empty when there is no cookie:
  // an empty token claims one we do not have.
  async function write(path, { method, body } = {}) {
    const token = readCsrfToken();
    const headers = { Accept: 'application/json', 'Content-Type': 'application/json' };
    if (token) headers['X-CSRF-Token'] = token;
    const init = { method, credentials: 'include', headers };
    if (body !== undefined) init.body = body;
    const response = await fetch(path, init);
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail = payload?.detail || payload?.error;
      const error = new Error(
        typeof detail === 'string' ? detail : `Request failed with status ${response.status}`,
      );
      error.status = response.status;
      throw error;
    }
    if (response.status === 204) return null;
    return response.json().catch(() => null);
  }
```

- [ ] **Step 4: Retain the user in `gate()` and expose it**

In `state` (`:49-56`), add `user: null` after `admin: false,`:

```js
  const state = {
    route: 'overview',
    routeId: null,
    range: '1W',
    filters: { group: '', segment: '', tier: '', internal: false, q: '', priority: false },
    admin: false,
    user: null,
    seq: {},
  };
```

Replace the body of `gate()` (`:311-326`) with:

```js
  async function gate() {
    try {
      const response = await fetch('/api/auth/me', { method: 'GET', credentials: 'include', headers: { Accept: 'application/json' } });
      const body = response.ok ? await response.json() : null;
      const account = body && body.user;
      if (!account || account.role !== 'admin') {
        state.user = null;
        window.location.replace('/app');
        return false;
      }
      state.admin = true;
      // N8: kept, not discarded. The account menu needs the display name and
      // the email, and this request has already been paid for -- the next reader
      // should not add a second /api/auth/me for the same two strings. Only
      // these two fields are retained: nothing on this page renders a role or
      // an id, and a wider copy is a wider thing to leak into a DOM sink.
      state.user = { display_name: account.display_name || '', email: account.email || '' };
      return true;
    } catch (_error) {
      state.user = null;
      window.location.replace('/app');
      return false;
    }
  }

  function user() {
    return state.user;
  }
```

In `handleAccessLost` (`:298-304`), add the drop after `state.admin = false;`:

```js
  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    invalidateAll();
    state.admin = false;
    state.user = null;
    window.location.replace('/app');
    return true;
  }
```

- [ ] **Step 5: Export the two new functions**

In the `api` object, change the `request` line (`:543`):

```js
    request, write, user, nextSeq, isCurrent, invalidateAll, handleAccessLost, gate,
```

- [ ] **Step 6: Bump the shell's pin**

`dashboard/frontend/admin.html:13`:

```html
  <script src="js/admin-shell.js?v=2" defer></script>
```

- [ ] **Step 7: Update the three shipped assertions the new `fetch` breaks**

`test_admin_page_modules.py:50-60` — the shell now holds three `fetch(` calls, and the added one is the write path. The verb-set assertion is untouched: `write()` takes its method from the caller, so there is no new `method: '<VERB>'` literal anywhere in the shell.

```python
def test_every_request_is_a_credentialed_get_made_by_the_shell():
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET"}
    for name, source in MODULES.items():
        if name == "admin-shell.js":
            # request(), gate() and write() -- the third is the page's single
            # write path (N4) and is deliberately a separate named function so
            # this file can tell readers from writers by grep.
            assert source.count("fetch(") == 3
            assert "credentials: 'include'" in source
        else:
            assert "fetch(" not in source, name
            assert "XMLHttpRequest" not in source, name
    for verb in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL


def test_the_write_path_carries_the_csrf_double_submit_header():
    """CsrfMiddleware 403s an unsafe method that carries a session cookie without
    a matching X-CSRF-Token. A fetch-stubbed behaviour test cannot see that, so
    the header is pinned in source too."""
    shell = MODULES["admin-shell.js"]
    assert "X-CSRF-Token" in shell
    assert "__Host-atl_csrf" in shell and "'atl_csrf'" in shell
    assert "document.cookie" in shell
```

`test_admin_page_shell.py:17-23` and `:78-84` — the pin moves to `?v=2`:

```python
EXPECTED_SCRIPTS = [
    "js/admin-shell.js?v=2",
    "js/credit-format.js?v=1",
    "js/admin-live.js?v=1",
    "js/admin-overview.js?v=1",
    "js/admin-users.js?v=1",
]
```

and in `test_gate_module_loads_first_and_every_script_is_pinned`:

```python
    assert srcs[0] == "js/admin-shell.js?v=2"
```

`test_admin_analytics_frontend.py:159` — inside `test_app_lifecycle_and_cache_versions_are_wired`, change the shell entry of the `for tag in (...)` tuple to `'src="js/admin-shell.js?v=2"'`.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_page_modules.py dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_analytics_frontend.py -q
```

Expected: PASS, no skips (`node` must be on `PATH`; a skip here is a silent loss of the whole layer).

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/js/admin-shell.js dashboard/frontend/admin.html dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_page_modules.py dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "$(cat <<'EOF'
feat: give the admin shell a CSRF-bearing write path

AdminShell.write(path, {method, body}) is the page's single write path
(design N4), separately named from request() so the module guard can tell
readers from writers by grep. It carries the double-submit X-CSRF-Token
CsrfMiddleware requires; app.js's csrfHeaders() is unreachable from /admin,
so the shell reads the cookie itself.

gate() now retains the display name and email it already fetched (N8)
instead of discarding them, and drops them when access is lost.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
EOF
)"
```

---

## Task 2: The rail and the account menu

Rebases `/admin`'s aside onto the legacy rail idiom (18px icons, 13px/600 labels, sticky card) and gives the page an account menu, closing defect 1. Providers still links out to the old console at the end of this task — Task 3 flips that one `href`.

**Files:**
- Modify: `dashboard/frontend/admin.html` (`:8` pin, new sprite after `<body>` at `:19`, header at `:20-23`, aside at `:25-42`)
- Modify: `dashboard/frontend/admin.css` (`:6` `:root`, `:32-40` aside rules, `:273` the 680px media query, new blocks at the end)
- Modify: `dashboard/frontend/js/admin-shell.js` (`route()` at `:427-447`, `bindControls()` at `:489-523`, `boot()` at `:525-534`, `api` at `:536-545`)
- Test: `dashboard/backend/tests/test_admin_page_shell.py`
- Test: `dashboard/backend/tests/test_admin_shell_frontend.py`
- Test: `dashboard/backend/tests/test_admin_analytics_frontend.py:149-162`

**Interfaces:**
- Consumes: `AdminShell.user()` from Task 1.
- Produces:
  - DOM ids `adminRail`, `accountMenuWrap`, `accountBtn`, `accountMenu`, `accountMenuName`, `accountMenuEmail`, `accountLabel`, `accountAvatar`, `accountMenuLogoutBtn`.
  - `data-rail="analytics|account-management|providers|activity"` on the four top-level rail entries. `route()` sets `aria-current="page"` + `.is-active` on the matching one.
  - `ANALYTICS_ROUTES` as the single owner of the seven analytics routes; `ROUTES` is derived from it in Task 3.
  - A `<svg hidden>` sprite defining `icon-chart`, `icon-users`, `icon-network`, `icon-activity`, `icon-refresh`, `icon-x`, `icon-check-circle`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_admin_page_shell.py`:

```python
RAIL_ICONS = ("icon-chart", "icon-users", "icon-network", "icon-activity")
SPRITE_ICONS = RAIL_ICONS + ("icon-refresh", "icon-x", "icon-check-circle")


def test_the_rail_is_anchors_with_icons_not_a_tablist():
    """N1: /admin navigates by hash and #users/{id} is a real linkable location,
    so the rail imitates the legacy rail's look and not its ARIA. role="tab" on a
    control that changes the URL misreports the interaction, and an anchor gets
    middle-click, copy-link and keyboard handling with no JavaScript."""
    start = ADMIN_HTML.index('<aside id="adminRail"')
    end = ADMIN_HTML.index("</aside>", start)
    rail = ADMIN_HTML[start:end]
    entries = re.findall(r'<a class="admin-tab[^"]*"[^>]*data-rail="([a-z-]+)"[^>]*>(.*?)</a>', rail, re.S)
    assert [name for name, _ in entries] == ["analytics", "account-management", "providers", "activity"]
    for name, body in entries:
        assert "<use href=\"#icon-" in body, name          # every entry has an icon
        assert re.search(r"<span>[^<]+</span>", body), name  # ...and a text label
    assert 'role="tab"' not in rail
    assert "aria-selected" not in rail
    assert ">Account Management<" in rail
    assert ">Users<" not in rail


def test_the_sprite_defines_every_symbol_the_page_references():
    """A <use href="#icon-x"> with no matching <symbol> renders an empty box and
    raises nothing -- the one failure mode a source-shape guard can actually catch."""
    defined = set(re.findall(r'<symbol id="([a-z-]+)"', ADMIN_HTML))
    assert defined == set(SPRITE_ICONS), defined
    referenced = set(re.findall(r'<use href="#([a-z-]+)"', ADMIN_HTML))
    assert referenced <= defined, referenced - defined
    # The sprite is markup, not script, so D6's no-inline-script guard is untouched.
    assert "<script" not in ADMIN_HTML[ADMIN_HTML.index("<svg") : ADMIN_HTML.index("</svg>")]


def test_the_account_menu_exists_and_carries_no_identity_text():
    """The menu renders only after the gate resolves. A name baked into the markup
    would be a different account's name for the duration of the probe."""
    start = ADMIN_HTML.index('<div id="accountMenuWrap"')
    end = ADMIN_HTML.index("</header>", start)
    menu = ADMIN_HTML[start:end]
    for control in ("accountBtn", "accountMenu", "accountMenuName", "accountMenuEmail", "accountMenuLogoutBtn"):
        assert f'id="{control}"' in menu, control
    assert '<span id="accountMenuName" class="account-menu-name"></span>' in menu
    assert '<span id="accountMenuEmail" class="account-menu-email"></span>' in menu
    assert 'href="/app?view=account"' in menu
    assert 'href="/app?view=credits"' in menu
    # N7: no Admin item (it would link to the page it is on) and no ticker.
    assert ">Admin<" not in menu
    assert "ticker" not in ADMIN_HTML.lower()


def test_the_rail_css_is_retyped_against_admin_tokens_not_copied():
    """admin.css does not import styles.css and the two use different token
    names (design §4.2), so a rule that arrived by copy-paste is a rule that
    renders unstyled. These are styles.css token names; none may appear here."""
    for foreign in ("--border-color", "--text-primary", "--text-secondary", "--bg-card", "--info-color", "#67e8f9"):
        assert foreign not in ADMIN_CSS, foreign
    for retyped in (".admin-rail", ".admin-tab", ".admin-tab svg", ".admin-tab.is-active::before",
                    ".account-menu", ".account-menu-item", ".account-menu-item--danger"):
        assert retyped in ADMIN_CSS, retyped
    assert "min-height:43px" in ADMIN_CSS      # the legacy pill height
    assert "font-size:13px;font-weight:600" in ADMIN_CSS
    assert "width:18px;height:18px" in ADMIN_CSS


def test_the_flat_text_list_rules_are_gone():
    """The old aside styled every anchor with one `aside a` rule and shrank the
    labels to font-size:0 on phones. Leaving either behind lets a stale rule win
    over the rail by source order."""
    assert "aside a{" not in ADMIN_CSS
    assert "aside a:hover{" not in ADMIN_CSS
    assert "aside a::first-letter" not in ADMIN_CSS
    assert ".analytics-parent{" not in ADMIN_CSS
```

Replace `test_subnav_routes_match_the_shell_router_and_the_aside_links_back` (`:187-198`) — the subnav is unchanged, but two of the three outbound links survive and Providers becomes a hash route in Task 3. Until then it is still an outbound link, so this task asserts all three and Task 3 moves one:

```python
def test_subnav_routes_match_the_shell_router_and_the_aside_links_back():
    start = ADMIN_HTML.index('<nav class="analytics-subnav"')
    end = ADMIN_HTML.index("</nav>", start)
    hrefs = re.findall(r'href="(#[a-z]+)"', ADMIN_HTML[start:end])
    assert hrefs == ["#overview", "#sources", "#retention", "#credits", "#lifecycle", "#health", "#users"]
    assert "#live" not in ADMIN_HTML
    for href in (
        "/app?view=admin&amp;adminTab=users",
        "/app?view=admin&amp;adminTab=providers",
        "/app?view=admin&amp;adminTab=activity",
    ):
        assert href in ADMIN_HTML, href
```

Append to `dashboard/backend/tests/test_admin_shell_frontend.py`:

```python
def test_the_account_menu_renders_only_after_the_gate_resolves():
    result = _eval(
        "(async () => {"
        "  const wrap = register('accountMenuWrap', new Node('div'));"
        "  const name = register('accountMenuName', new Node('span'));"
        "  const email = register('accountMenuEmail', new Node('span'));"
        "  const label = register('accountLabel', new Node('span'));"
        "  const avatar = register('accountAvatar', new Node('span'));"
        "  wrap.hidden = true;"
        "  const before = wrap.hidden;"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin',"
        "    display_name: 'Ada Admin', email: 'ada@example.test'}}});"
        "  await window.AdminShell.gate();"
        "  window.AdminShell.renderAccountMenu();"
        "  return {before, after: wrap.hidden, name: name.textContent, email: email.textContent,"
        "          label: label.textContent, avatar: avatar.textContent};"
        "})()"
    )
    assert result == {
        "before": True, "after": False,
        "name": "Ada Admin", "email": "ada@example.test",
        "label": "Ada Admin", "avatar": "A",
    }


def test_an_account_with_no_display_name_falls_back_to_the_email():
    result = _eval(
        "(async () => {"
        "  register('accountMenuWrap', new Node('div'));"
        "  const name = register('accountMenuName', new Node('span'));"
        "  const avatar = register('accountAvatar', new Node('span'));"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin', email: 'bo@example.test'}}});"
        "  await window.AdminShell.gate();"
        "  window.AdminShell.renderAccountMenu();"
        "  return {name: name.textContent, avatar: avatar.textContent};"
        "})()"
    )
    assert result == {"name": "bo@example.test", "avatar": "B"}


def test_logout_writes_then_leaves_and_leaves_even_when_the_write_fails():
    """A failed logout still leaves the page: the session may already be gone,
    and stranding an admin on a console they can no longer read is worse than a
    redirect that turns out to be redundant."""
    ok = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 204, body: null});"
        "  await window.AdminShell.logout();"
        "  return {method: fetchCalls[0][1].method, url: fetchCalls[0][0], nav};"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert ok == {"method": "POST", "url": "/api/auth/logout", "nav": [["assign", "/app"]]}
    failed = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: false, status: 500, body: {detail: 'boom'}});"
        "  await window.AdminShell.logout();"
        "  return nav;"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert failed == [["assign", "/app"]]


def test_the_rail_marks_the_active_entry_with_aria_current_not_aria_selected():
    """N1. Every analytics route lights the one Analytics entry; a sibling route
    lights its own. aria-current is removed rather than set to "false", which is
    a value assistive technology treats as present."""
    result = _eval(
        "(() => {"
        "  const made = {};"
        "  const entries = ['analytics', 'account-management', 'providers', 'activity'].map((rail) => {"
        "    const a = new Node('a'); a.dataset.rail = rail; made[rail] = a; return a;"
        "  });"
        "  document.querySelectorAll = (selector) => (selector.includes('data-rail') ? entries : []);"
        "  const seen = {};"
        "  ['overview', 'users', 'health'].forEach((route) => {"
        "    window.location.hash = '#' + route;"
        "    window.AdminShell.syncRail(route);"
        "    seen[route] = Object.fromEntries(entries.map((a) => [a.dataset.rail,"
        "      [a.getAttribute('aria-current'), a.classList.contains('is-active')]]));"
        "  });"
        "  return seen;"
        "})()"
    )
    for route in ("overview", "users", "health"):
        assert result[route]["analytics"] == ["page", True], route
        for other in ("account-management", "providers", "activity"):
            assert result[route][other] == [None, False], (route, other)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
pytest dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_shell_frontend.py -q
```

Expected: FAIL — `ValueError: substring not found` on `'<aside id="adminRail"'` and on `'<div id="accountMenuWrap"'`, and node exits non-zero on `window.AdminShell.renderAccountMenu is not a function`.

- [ ] **Step 3: Add the icon sprite to `admin.html`**

Insert immediately after `<body>` (`:19`), before `<header>`. The seven symbols are copied byte-for-byte from `app.html:188, 191, 192, 202, 207, 216, 218`; none needs drawing:

```html
  <!-- Icon sprite (design §6). Markup, not script, so the no-inline-script
       guard (D6) is unaffected. Seven symbols, not the five the design named:
       the ported providers panel also references #icon-check-circle on both
       submit buttons and #icon-x on the revoke action, and a <use> with no
       matching <symbol> renders an empty box without raising. -->
  <svg xmlns="http://www.w3.org/2000/svg" style="position:absolute;width:0;height:0;overflow:hidden" aria-hidden="true">
    <symbol id="icon-chart" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></symbol>
    <symbol id="icon-users" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></symbol>
    <symbol id="icon-network" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/><rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/><path d="M12 12V8"/></symbol>
    <symbol id="icon-activity" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/></symbol>
    <symbol id="icon-refresh" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M20 11a8.1 8.1 0 0 0-15.5-2M4 4v5h5"/><path d="M4 13a8.1 8.1 0 0 0 15.5 2M20 20v-5h-5"/></symbol>
    <symbol id="icon-x" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></symbol>
    <symbol id="icon-check-circle" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></symbol>
  </svg>
```

Note for the reviewer: the sprite's `viewBox="0 0 24 24"` and `points="22 4 …"` are attribute values, and `test_panel_regions_carry_no_numeric_or_percentage_literal` strips tags before scanning, so these digits are invisible to that guard. The sprite also sits outside every `data-panel` region.

- [ ] **Step 4: Add the account menu to the header**

Replace `admin.html:20-23` with:

```html
  <header>
    <a class="brand" href="#overview"><img src="images/atltransparent.png" alt="ATL">Agentic Trading Lab</a>
    <nav class="product-nav" aria-label="Product"><a href="/app">Home</a><a href="/app?view=playground&amp;playgroundTab=agents">My Agents</a><a href="/app?view=competition">Competition</a><a href="/app?view=community">Community</a></nav>
    <!-- Design §7. Hidden until gate() resolves: before that the page knows no
         identity, and a placeholder name is a different account's name for the
         duration of the probe. No Admin item and no ticker (N7). -->
    <div id="accountMenuWrap" class="account-menu-wrap" hidden>
      <button id="accountBtn" class="account-btn" type="button" aria-haspopup="true" aria-expanded="false" aria-label="Account menu">
        <span id="accountAvatar" class="account-avatar" aria-hidden="true"></span>
        <span id="accountLabel" class="account-label"></span>
        <span class="account-caret" aria-hidden="true">▾</span>
      </button>
      <div id="accountMenu" class="account-menu" hidden>
        <div class="account-menu-identity">
          <span id="accountMenuName" class="account-menu-name"></span>
          <span id="accountMenuEmail" class="account-menu-email"></span>
        </div>
        <a class="account-menu-item" href="/app?view=account">Account</a>
        <a class="account-menu-item" href="/app?view=credits">Credits &amp; Billing</a>
        <button id="accountMenuLogoutBtn" class="account-menu-item account-menu-item--danger" type="button">Log out</button>
      </div>
    </div>
  </header>
```

- [ ] **Step 5: Rebase the aside onto the rail**

Replace `admin.html:25-42` with:

```html
    <aside id="adminRail" class="admin-rail" aria-label="Administration">
      <strong>Administration</strong>
      <a class="admin-tab analytics-parent" id="analyticsParent" data-rail="analytics" href="#overview" aria-expanded="true"><svg aria-hidden="true"><use href="#icon-chart"/></svg><span>Analytics</span><i class="admin-tab-caret" aria-hidden="true">⌄</i></a>
      <nav class="analytics-subnav" id="analyticsSubnav" aria-label="Analytics modules">
        <a class="active" href="#overview" data-route="overview">Overview</a>
        <a href="#sources" data-route="sources">User sources</a>
        <a href="#retention" data-route="retention">Retention</a>
        <a href="#credits" data-route="credits">Credits &amp; revenue</a>
        <a href="#lifecycle" data-route="lifecycle">User lifecycle</a>
        <a href="#health" data-route="health">System health</a>
        <a href="#users" data-route="users">Users</a>
      </nav>
      <!-- Account Management and Activity stay in the old console until PR2
           (design N2: they are one 712-line module and port together). -->
      <a class="admin-tab" data-rail="account-management" href="/app?view=admin&amp;adminTab=users"><svg aria-hidden="true"><use href="#icon-users"/></svg><span>Account Management</span></a>
      <a class="admin-tab" data-rail="providers" href="/app?view=admin&amp;adminTab=providers"><svg aria-hidden="true"><use href="#icon-network"/></svg><span>Providers</span></a>
      <a class="admin-tab" data-rail="activity" href="/app?view=admin&amp;adminTab=activity"><svg aria-hidden="true"><use href="#icon-activity"/></svg><span>Activity</span></a>
    </aside>
```

The `<aside>` keeps `class="admin-rail"` so one element is both the grid column and the card. The subnav's `<a>` elements keep `data-route`, which `route()` already drives.

- [ ] **Step 6: Retype the rail and account-menu CSS into `admin.css`**

Delete `admin.css:35-40` (`aside a`, `aside a:hover`, `.analytics-parent`, `.analytics-subnav*`) and replace with nothing — the replacements go at the end of the file. Widen the grid column at `:32`:

```css
.workspace{display:grid;grid-template-columns:224px minmax(0,1fr);min-height:calc(100vh - 80px)}
```

Change `:33-34` to drop the border (the card supplies the edge now):

```css
aside{padding:24px 14px}
aside>strong{display:block;margin:0 4px 14px;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
```

Append at the end of `admin.css`, before the three media queries:

```css
/* —— Navigation rail (design §6) ——
   The application's idiom, retyped against this file's tokens. styles.css's
   .admin-rail/.admin-tab family is the source; its --border-color,
   --text-primary and --bg-card do not exist here, so this is a retype and
   never a copy (§4.2). The active bar is var(--accent) because that IS this
   page's cyan -- styles.css's #67e8f9 is the same role in the other palette. */
.admin-rail{position:sticky;top:24px;display:flex;flex-direction:column;gap:5px;margin:0;padding:7px;border:1px solid var(--line);border-radius:12px;background:linear-gradient(180deg,rgba(0,191,255,.055),transparent 42%),var(--surface);box-shadow:0 16px 38px rgba(2,8,23,.2)}
.admin-tab{position:relative;display:flex;align-items:center;gap:10px;width:100%;min-height:43px;padding:9px 11px;border:0;border-radius:8px;background:transparent;color:var(--muted);font-size:13px;font-weight:600;white-space:nowrap}
.admin-tab svg{flex:0 0 auto;width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8}
.admin-tab::before{content:"";position:absolute;inset:9px auto 9px -7px;width:2px;border-radius:999px;background:var(--accent);opacity:0;transform:scaleY(.35);transition:opacity 140ms ease,transform 140ms ease}
.admin-tab:hover,.admin-tab:focus-visible{color:var(--text);background:rgba(148,163,184,.12)}
.admin-tab.is-active{color:var(--text);background:rgba(0,191,255,.12);box-shadow:inset 0 0 0 1px rgba(0,191,255,.12)}
.admin-tab.is-active::before{opacity:1;transform:scaleY(1)}
.admin-tab-caret{margin-left:auto;font-style:normal;font-size:12px}
/* The subnav rises from 12px to the rail's 13px (§6) and keeps its indent. */
.analytics-subnav{margin:2px 0 6px 25px;padding-left:10px;border-left:1px solid var(--line)}
.analytics-subnav a{display:block;margin-bottom:2px;padding:7px 10px;border-radius:6px;color:var(--muted);font-size:13px}
.analytics-subnav a:hover{color:var(--text);background:rgba(148,163,184,.12)}
.analytics-subnav a.active{color:var(--accent);background:rgba(0,191,255,.12)}

/* —— Header account menu (design §7) ——
   Retyped from styles.css's .account-menu* family. Note what is NOT here:
   styles.css needs a `.account-menu-item[hidden]{display:none!important}`
   override because its author `display:block` beats the UA's [hidden] rule.
   This file has a global `[hidden]{display:none!important}` at :18, which
   already carries the !important, so the override would be dead weight. */
.account-menu-wrap{position:relative}
.account-btn{display:inline-flex;align-items:center;gap:8px;max-width:200px;min-height:38px;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--text);cursor:pointer}
.account-btn:hover{border-color:var(--accent);color:var(--accent)}
.account-avatar{display:inline-grid;place-items:center;width:24px;height:24px;flex:0 0 24px;border-radius:50%;background:#153c48;color:var(--accent);font-size:12px;font-weight:700}
.account-label{overflow:hidden;max-width:118px;font-size:14px;font-weight:600;text-overflow:ellipsis;white-space:nowrap}
.account-caret{color:var(--muted);font-size:10px}
.account-menu{position:absolute;top:calc(100% + 6px);right:0;z-index:60;display:flex;flex-direction:column;gap:2px;min-width:220px;padding:6px;border:1px solid var(--line);border-radius:8px;background:var(--surface);box-shadow:0 10px 28px rgba(0,0,0,.35)}
.account-menu-identity{display:flex;flex-direction:column;gap:2px;margin-bottom:4px;padding:8px 10px;border-bottom:1px solid var(--line)}
.account-menu-name{color:var(--text);font-size:12px;font-weight:700}
.account-menu-email{color:var(--muted);font-size:11px;word-break:break-all}
.account-menu-item{display:block;width:100%;min-height:0;padding:8px 10px;border:0;border-radius:6px;background:transparent;color:var(--text);font-size:12px;font-weight:600;text-align:left;cursor:pointer}
.account-menu-item:hover{color:var(--text);background:rgba(148,163,184,.12)}
.account-menu-item--danger,.account-menu-item--danger:hover{color:var(--red)}
```

Replace the aside clauses inside `@media(max-width:680px)` (`:273`). That query is one long line; the run to replace starts at `.workspace{` and ends just before `main{padding:24px 16px 40px}`. Replace exactly this substring:

```css
.workspace{grid-template-columns:66px minmax(0,1fr)}aside{padding:18px 6px}aside>strong{display:none}aside a{overflow:hidden;padding:10px 7px;font-size:0;text-align:center}aside a::first-letter{font-size:14px}.analytics-parent span{display:none}.analytics-subnav{margin-left:9px;padding-left:3px}
```

with exactly this one:

```css
.workspace{grid-template-columns:64px minmax(0,1fr)}aside{padding:18px 6px}aside>strong{display:none}.admin-rail{padding:5px}.admin-tab{justify-content:center;min-height:42px;padding:9px}.admin-tab span,.admin-tab-caret{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}.analytics-subnav{display:none}
```

One substitution, so the old `66px` column and the `font-size:0` label trick both go in the same edit — leaving either behind gives the rail two column widths and a set of invisible labels that win by source order.

Seven sub-entries cannot be reduced to icons, so the subnav is hidden rather than shrunk at phone width; the Analytics entry still routes to `#overview`.

- [ ] **Step 7: Bump the stylesheet pin**

`admin.html:8`:

```html
  <link rel="stylesheet" href="admin.css?v=2">
```

- [ ] **Step 8: Wire the rail and the account menu in `admin-shell.js`**

Split the routes constant at `:7` so one list owns the analytics routes:

```js
  // Design §7.2: seven analytics routes plus the profile (#users/{id}), and --
  // since the 09-17 consolidation -- Providers alongside them. Split rather than
  // one list because route() has to ask "is this an analytics route?" to light
  // the rail's Analytics entry, and a second hand-maintained copy of the seven
  // is how the two drift.
  const ANALYTICS_ROUTES = ['overview', 'sources', 'retention', 'credits', 'lifecycle', 'health', 'users'];
  const ROUTES = [...ANALYTICS_ROUTES];
```

(Task 3 appends `'providers'`; this task leaves `ROUTES` at seven so `test_hash_router_knows_exactly_the_seven_routes_and_the_profile` stays green.)

Add, just before `route()`:

```js
  // N1: the rail navigates by hash, so its entries are anchors carrying
  // aria-current="page" -- not role="tab"/aria-selected, which would tell
  // assistive technology this is an in-place panel swap when the URL actually
  // changes and #users/{id} is a real, linkable, back-button-able location.
  // Removed rather than set to "false": aria-current="false" is a value that is
  // *present*, and screen readers announce the attribute, not its truthiness.
  function syncRail(route) {
    document.querySelectorAll('#adminRail a[data-rail]').forEach((link) => {
      const target = link.dataset.rail;
      const active = target === route || (target === 'analytics' && ANALYTICS_ROUTES.includes(route));
      if (active) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
      link.classList.toggle('is-active', active);
    });
  }
```

In `route()`, add the call right after the existing subnav loop (`:442-444`):

```js
    syncRail(parsed.route);
```

Add the account-menu controller, after `syncRail`:

```js
  function renderAccountMenu() {
    const wrap = document.getElementById('accountMenuWrap');
    const account = state.user;
    if (!wrap || !account) return;
    const label = account.display_name || account.email || '';
    const nameNode = document.getElementById('accountMenuName');
    const emailNode = document.getElementById('accountMenuEmail');
    const labelNode = document.getElementById('accountLabel');
    const avatarNode = document.getElementById('accountAvatar');
    if (nameNode) nameNode.textContent = label;
    if (emailNode) emailNode.textContent = account.email || '';
    if (labelNode) labelNode.textContent = label;
    if (avatarNode) avatarNode.textContent = (label.trim()[0] || '?').toUpperCase();
    wrap.hidden = false;
  }

  function setAccountMenuOpen(open) {
    const menu = document.getElementById('accountMenu');
    const button = document.getElementById('accountBtn');
    if (!menu || !button) return;
    menu.hidden = !open;
    button.setAttribute('aria-expanded', String(open));
  }

  async function logout() {
    try {
      await write('/api/auth/logout', { method: 'POST' });
    } catch (_error) {
      // Deliberately swallowed. The session may already be gone (401), or the
      // server may be cold (5xx); either way, leaving an admin sitting on a
      // console they can no longer read is worse than a redirect that turns out
      // to have been redundant. The cookie is HttpOnly, so there is nothing
      // this page could clear locally as a consolation.
    }
    window.location.assign('/app');  // CORRECTED 2026-09-18 (finding I5): logout() ships as replace('/app') -- assign leaves the painted console restorable from bfcache, where gate() never re-runs.
  }
```

In `bindControls()`, append before the `[data-retry]` loop:

```js
    document.getElementById('accountBtn')?.addEventListener('click', (event) => {
      event.stopPropagation();
      setAccountMenuOpen(document.getElementById('accountMenu')?.hidden !== false);
    });
    document.getElementById('accountMenuLogoutBtn')?.addEventListener('click', () => { logout(); });
    document.addEventListener('click', (event) => {
      const wrap = document.getElementById('accountMenuWrap');
      // `typeof … === 'function'` rather than an optional call: the node DOM stub
      // has no contains(), and `wrap?.contains?.(t)` returning undefined would
      // read as "outside" and close a menu the test just opened.
      if (wrap && typeof wrap.contains === 'function' && wrap.contains(event.target)) return;
      setAccountMenuOpen(false);
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') setAccountMenuOpen(false);
    });
```

In `boot()`, render the menu once the gate has resolved:

```js
  async function boot() {
    bindControls();
    const admin = await gate();
    if (!admin) return;
    renderAccountMenu();
    readUrlIntoState();
    syncControls();
    window.addEventListener('hashchange', handleLocationChange);
    window.addEventListener('popstate', handleLocationChange);
    route();
  }
```

Export the three the tests call, in the `api` object:

```js
    setPanelState, openDialog, closeDialog, openRules, setFilters,
    syncRail, renderAccountMenu, logout,
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_page_modules.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add dashboard/frontend/admin.html dashboard/frontend/admin.css dashboard/frontend/js/admin-shell.js dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_shell_frontend.py
git commit -m "$(cat <<'EOF'
feat: rebase the admin rail and add the account menu

/admin's aside was a flat list of text links with a 12px subnav; the rest of
the application uses a sticky rail card with 18px icons and 13px/600 labels.
That idiom is retyped into admin.css against its own tokens (the two
stylesheets share no token names), and the aside becomes anchors carrying
aria-current="page" -- not role="tab", which would misreport a URL change.

The page also had no account menu at all: an admin on /admin could not see
who they were signed in as, reach Account, or log out. It has one now, fed
by the user gate() already fetches.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
EOF
)"
```

---

## Task 3: Port Providers into `/admin`

Creates `js/admin-providers.js`, its section and CSS, registers the `#providers` route, and rescopes the module guard (§9) in the same commit — the guard's module list is hardcoded, so a new script tag in `admin.html` fails `test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded` the moment it lands.

**Files:**
- Create: `dashboard/frontend/js/admin-providers.js`
- Create: `dashboard/backend/tests/test_admin_providers_frontend.py`
- Modify: `dashboard/frontend/admin.html` (script list, rail `href`, new section)
- Modify: `dashboard/frontend/admin.css` (providers CSS)
- Modify: `dashboard/frontend/js/admin-shell.js` (`ROUTES`, `route()`)
- Modify: `dashboard/backend/tests/test_admin_page_modules.py` (the §9 rescope)
- Modify: `dashboard/backend/tests/test_admin_page_shell.py`, `test_admin_shell_frontend.py`, `test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `AdminShell.request`, `AdminShell.write`, `AdminShell.user`, `AdminShell.el`, `AdminShell.clear`, `AdminShell.nextSeq`, `AdminShell.isCurrent`, `AdminShell.handleAccessLost` (Task 1); the `admin:route` CustomEvent `announce()` dispatches; the sprite symbols (Task 2).
- Produces: `window.AdminProviders = { renderProviderList, renderProviderOptions, platformLabel, load, savePlatformKey, confirmRevoke, onRoute }`. `renderProviderList(payload)` returns the built `<div>` rather than only mutating, so a node test can assert on it without a live document — the same shape `AdminUsers.renderUserRows` uses.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_admin_providers_frontend.py`:

```python
"""js/admin-providers.js under node: registry rows, platform-key states, writes.

Replaces test_admin_model_providers_frontend.py, whose four cases all asserted
against artifacts the 09-17 consolidation removes (the ?view=admin markup, the
old module, the app.js wiring, the styles.css families). The assertions that are
still true about the surface -- admin routes only, the secret field cleared on
every exit, no localStorage, no innerHTML -- are carried forward below.
"""

import re
from pathlib import Path

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source
from dashboard.backend.tests._frontend_source import fn_body, strip_comments

pytestmark = requires_node

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
ADMIN_CSS = (FRONTEND / "admin.css").read_text(encoding="utf-8")
SHELL = source("admin-shell.js")
PROVIDERS_JS = source("admin-providers.js")
# Every source-shape assertion below runs on the stripped copy. This module's
# header comment names what it replaced -- window.API.request,
# getStoredAuthUser -- so a raw scan for those strings would fail on the prose
# explaining their absence, which is the inverse of the trap `strip_comments`
# was written for (a guard satisfied by a comment rather than by the code).
PROVIDERS_CODE = strip_comments(PROVIDERS_JS)

PROVIDERS = """{"providers": [
  {"provider_id": "openrouter", "display_name": "OpenRouter", "adapter_type": "openrouter",
   "approved_base_url": "https://openrouter.ai/api/v1", "status": "enabled",
   "byok_enabled": true, "platform_enabled": true,
   "platform_credential": {"status": "verified", "key_last_four": "9f21"}},
  {"provider_id": "anthropic", "display_name": "Anthropic", "adapter_type": "anthropic",
   "approved_base_url": "https://api.anthropic.com/v1", "status": "disabled",
   "byok_enabled": true, "platform_enabled": false, "platform_credential": null},
  {"provider_id": "gemini", "display_name": "Gemini", "adapter_type": "gemini",
   "approved_base_url": "https://generativelanguage.googleapis.com/v1", "status": "enabled",
   "byok_enabled": false, "platform_enabled": true,
   "platform_credential": {"status": "pending_verification", "key_last_four": "0c4d"}}
]}"""


def _eval(expression: str, *setup: str) -> object:
    return run_node(
        SHELL, PROVIDERS_JS, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_registry_rows_show_name_state_identity_and_key_status():
    result = _eval(
        "(() => {"
        f"  const list = window.AdminProviders.renderProviderList({PROVIDERS});"
        "  return list.children.map((row) => ["
        "    byTag(row, 'STRONG')[0].textContent,"
        "    byClass(row, 'admin-provider-state')[0].textContent,"
        "    byClass(row, 'admin-provider-row-meta')[0].textContent,"
        "    byClass(row, 'admin-provider-credential')[0].textContent,"
        "  ]);"
        "})()"
    )
    assert result == [
        ["OpenRouter", "enabled", "openrouter · openrouter · https://openrouter.ai/api/v1", "Verified •••• 9f21"],
        ["Anthropic", "disabled", "anthropic · anthropic · https://api.anthropic.com/v1", "No platform key"],
        ["Gemini", "enabled", "gemini · gemini · https://generativelanguage.googleapis.com/v1",
         "pending verification •••• 0c4d"],
    ]


def test_only_a_live_key_offers_reverify_and_revoke():
    """A provider with no key, or a revoked one, must not offer actions that
    would 404 or 409 -- the row is the only place the operator learns which."""
    result = _eval(
        "(() => {"
        f"  const payload = {PROVIDERS};"
        "  payload.providers.push({provider_id: 'dead', display_name: 'Dead', adapter_type: 'openai',"
        "    approved_base_url: 'https://x.test/v1', status: 'disabled',"
        "    platform_credential: {status: 'revoked', key_last_four: 'aaaa'}});"
        "  const list = window.AdminProviders.renderProviderList(payload);"
        "  return list.children.map((row) => byTag(row, 'BUTTON').map((b) => b.textContent));"
        "})()"
    )
    assert result == [
        ["Edit provider", "Reverify key", "Revoke key"],
        ["Edit provider"],
        ["Edit provider", "Reverify key", "Revoke key"],
        ["Edit provider"],
    ]


def test_an_empty_registry_says_so_rather_than_rendering_nothing():
    result = _eval(
        "(() => {"
        "  const list = window.AdminProviders.renderProviderList({providers: []});"
        "  return [list.children.length, list.textContent];"
        "})()"
    )
    assert result == [1, "No providers registered."]


def test_a_non_array_providers_field_is_treated_as_empty_not_crashed_on():
    result = _eval(
        "(() => window.AdminProviders.renderProviderList({providers: null}).textContent)()"
    )
    assert result == "No providers registered."


def test_the_provider_select_keeps_the_operators_choice_across_a_refresh():
    """Reloading the registry after a save must not silently move the platform-key
    form to a different provider -- the next Save would write the key to it.

    This pins the *restore*, not the vanished-provider case: a real <select>
    whose value no longer matches any option falls back to the first one on its
    own, and the node stub models `value` as a plain property, so asserting on
    it there would pin the stub rather than the browser."""
    result = _eval(
        "(() => {"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        "  select.value = 'anthropic';"
        f"  window.AdminProviders.renderProviderOptions({PROVIDERS});"
        "  const kept = select.value;"
        f"  const payload = {PROVIDERS};"
        "  payload.providers = payload.providers.filter((p) => p.provider_id !== 'anthropic');"
        "  window.AdminProviders.renderProviderOptions(payload);"
        "  return {kept, options: select.children.map((o) => o.textContent)};"
        "})()"
    )
    assert result["kept"] == "anthropic"
    assert result["options"] == ["Select a provider", "OpenRouter", "Gemini"]
    # The restore is guarded rather than unconditional; an unguarded
    # `select.value = current` is what would point the form at a gone provider.
    assert "providers.some((provider) => provider.provider_id === current)" in PROVIDERS_CODE


def test_reads_go_through_the_shell_get_and_writes_through_the_shell_write():
    result = _eval(
        "(async () => {"
        f"  fetchQueue.push({{ok: true, status: 200, body: {PROVIDERS}}});"
        "  register('adminProviderList', new Node('div'));"
        "  register('adminPlatformProvider', new Node('select'));"
        "  await window.AdminProviders.load();"
        "  return fetchCalls.map(([url, options]) => [url, options.method]);"
        "})()"
    )
    assert result == [["/api/admin/model-providers", "GET"]]


def test_the_platform_key_secret_is_cleared_on_every_exit():
    """Carried forward from test_admin_model_providers_frontend.py. The field is
    cleared on the refusal path as well as the success path: a secret left in a
    live input survives every later navigation on a single-page console."""
    assert PROVIDERS_CODE.count("secretInput.value = ''") >= 2
    refused = _eval(
        "(async () => {"
        "  const secretInput = register('adminPlatformKeySecret', new Node('input'));"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        "  register('adminPlatformKeyStatus', new Node('p'));"
        "  secretInput.value = 'sk-should-not-survive';"
        "  select.value = '';"
        "  await window.AdminProviders.savePlatformKey({preventDefault() {}});"
        "  return {left: secretInput.value, calls: fetchCalls.length};"
        "})()"
    )
    assert refused == {"left": "", "calls": 0}


def test_revoking_asks_first_and_does_nothing_when_the_operator_declines():
    result = _eval(
        "(async () => {"
        "  register('adminPlatformKeyStatus', new Node('p'));"
        "  window.confirm = () => false;"
        "  await window.AdminProviders.confirmRevoke('openrouter', 'OpenRouter');"
        "  return fetchCalls.length;"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert result == 0


def test_the_module_is_an_iife_with_one_global_and_no_fetch_of_its_own():
    assert PROVIDERS_JS.lstrip().startswith("/**")
    assert "(function () {\n  'use strict';" in PROVIDERS_JS
    assert set(re.findall(r"^\s*window\.(\w+) = ", PROVIDERS_CODE, re.M)) == {"AdminProviders"}
    # Stripped, so the header comment naming the two seams this port replaced
    # cannot satisfy -- or fail -- the assertion that they are gone.
    assert "fetch(" not in PROVIDERS_CODE
    assert "window.API" not in PROVIDERS_CODE
    assert "getStoredAuthUser" not in PROVIDERS_CODE
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "localStorage", "sessionStorage"):
        assert forbidden not in PROVIDERS_CODE, forbidden


def test_every_renderer_is_a_named_function_the_shared_harness_can_slice():
    """§7.6 / §12: the other five admin modules are testable because every
    renderer is a named function `fn_body` can brace-match out. An arrow
    assigned to a const, or a closure inside load(), is not liftable and the
    whole module then has to be exercised through the DOM."""
    for signature in (
        "function renderProviderList(", "function renderProviderOptions(",
        "function platformLabel(", "function fillProviderForm(",
    ):
        body = fn_body(signature, PROVIDERS_JS)
        assert "innerHTML" not in body, signature
    assert "textContent" in fn_body("function setStatus(", PROVIDERS_JS)


def test_the_providers_section_is_flat_so_the_panel_guard_can_see_all_of_it():
    """PANEL_REGION in test_admin_page_shell.py is non-greedy, so an outer
    <section data-panel> wrapping an inner <section> matches only as far as the
    inner closing tag and the numeric-literal guard silently scans a fragment.
    The port flattens the app.html wrapper to a <div> for exactly that reason."""
    start = ADMIN_HTML.index('<section id="providers"')
    end = ADMIN_HTML.index("</section>", start)
    markup = ADMIN_HTML[start:end]
    assert "<section" not in markup[1:]
    for control in ("adminProviderList", "adminProviderForm", "adminPlatformKeyForm",
                    "adminPlatformKeySecret", "adminProviderRefreshBtn"):
        assert f'id="{control}"' in markup, control
    assert 'type="password"' in markup and 'autocomplete="new-password"' in markup


def test_the_provider_css_is_retyped_here_and_not_imported():
    for family in (".admin-provider-row", ".admin-provider-state", ".admin-provider-grid",
                   ".admin-platform-key-form", ".auth-btn", ".control-select",
                   ".credits-key-field", ".credits-key-action", ".credits-icon-btn"):
        assert family in ADMIN_CSS, family
    assert "@import" not in ADMIN_CSS
```

Extend `test_admin_shell_frontend.py`'s router test (`:26-38`) — `ROUTES` is eight now:

```python
def test_hash_router_knows_the_analytics_routes_plus_providers_and_the_profile():
    assert _eval("window.AdminShell.ROUTES") == [
        "overview", "sources", "retention", "credits", "lifecycle", "health", "users", "providers",
    ]
    assert _eval("window.AdminShell.parseHash('')") == {"route": "overview", "id": None}
    assert _eval("window.AdminShell.parseHash('#health')") == {"route": "health", "id": None}
    assert _eval("window.AdminShell.parseHash('#providers')") == {"route": "providers", "id": None}
    assert _eval("window.AdminShell.parseHash('#users')") == {"route": "users", "id": None}
    assert _eval("window.AdminShell.parseHash('#users/42')") == {"route": "users", "id": "42"}
    assert _eval("window.AdminShell.parseHash('#users/abc')") == {"route": "users", "id": None}
    # No #live route (design §8.2, D16) and no orphan routes: unknown → overview.
    for unknown in ("#live", "#usage", "#revenue", "#profiles", "#funnel", "#nonsense"):
        assert _eval(f"window.AdminShell.parseHash('{unknown}')") == {"route": "overview", "id": None}, unknown


def test_providers_lights_its_own_rail_entry_not_analytics():
    result = _eval(
        "(() => {"
        "  const made = ['analytics', 'account-management', 'providers', 'activity'].map((rail) => {"
        "    const a = new Node('a'); a.dataset.rail = rail; return a;"
        "  });"
        "  document.querySelectorAll = (selector) => (selector.includes('data-rail') ? made : []);"
        "  window.AdminShell.syncRail('providers');"
        "  return Object.fromEntries(made.map((a) => [a.dataset.rail, a.getAttribute('aria-current')]));"
        "})()"
    )
    assert result == {"analytics": None, "account-management": None, "providers": "page", "activity": None}
```

- [ ] **Step 2: Run them to verify they fail**

```bash
pytest dashboard/backend/tests/test_admin_providers_frontend.py -q
```

Expected: collection error — `FileNotFoundError: .../frontend/js/admin-providers.js`, raised at import time by `source()`.

- [ ] **Step 3: Write `js/admin-providers.js`**

Create `dashboard/frontend/js/admin-providers.js`:

```js
/** /admin Providers: the approved provider registry and the platform credential.
 *
 * Ported from js/admin-model-providers.js (design §8), which reached the host
 * page through window.API.request and window.getStoredAuthUser -- both defined
 * in app.js, neither reachable from /admin. Every seam now goes through
 * AdminShell: reads through request() (GET-hardcoded), writes through write()
 * (the page's single write path, N4), identity through user().
 *
 * This is the page's ONLY write surface. Two things follow from that and are
 * load-bearing rather than stylistic:
 *   - no line may both name a credential and touch a DOM sink, which is why the
 *     masked-key line is called keyState and not credentialLine;
 *   - `api_key` appears exactly once, inside a JSON.stringify body.
 * test_admin_page_modules.py asserts both.
 */
(function () {
  'use strict';

  const SURFACE = 'providers';
  const state = { providers: [], bound: false };

  function shell() {
    return window.AdminShell;
  }

  function element(id) {
    return document.getElementById(id);
  }

  function textNode(tag, className, text) {
    return shell().el(tag, className, text);
  }

  function appendIcon(button, iconId) {
    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    icon.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#${iconId}`);
    icon.appendChild(use);
    button.appendChild(icon);
  }

  function setStatus(target, message, tone = '') {
    if (!target) return;
    target.textContent = message || '';
    target.classList.toggle('is-error', tone === 'error');
    target.classList.toggle('is-success', tone === 'success');
    target.classList.toggle('is-pending', tone === 'pending');
  }

  function operationKey(prefix) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  function actionPayload(reason) {
    return {
      source: 'admin-console',
      reason,
      idempotency_key: operationKey('admin-provider'),
    };
  }

  function fillProviderForm(provider) {
    if (!provider) return;
    element('adminProviderId').value = provider.provider_id || '';
    element('adminProviderDisplayName').value = provider.display_name || '';
    element('adminProviderAdapter').value = provider.adapter_type || 'openai_compatible';
    element('adminProviderBaseUrl').value = provider.approved_base_url || '';
    element('adminProviderByok').checked = provider.byok_enabled !== false;
    element('adminProviderPlatform').checked = provider.platform_enabled === true;
    element('adminProviderStatus').value = provider.status || 'enabled';
    const select = element('adminPlatformProvider');
    if (select) select.value = provider.provider_id || '';
  }

  // The masked label. Named for the *state* it reports, not for the credential
  // it reports on: the caller appends its node to the DOM, and a line that both
  // says "credential" and calls .append() is what the DOM-sink guard refuses.
  function platformLabel(credential) {
    if (!credential) return 'No platform key';
    if (credential.status === 'verified') return `Verified •••• ${credential.key_last_four}`;
    if (credential.status === 'revoked') return 'Revoked';
    return `${String(credential.status).replaceAll('_', ' ')} •••• ${credential.key_last_four}`;
  }

  function renderProviderList(payload) {
    const providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const list = shell().el('div', 'admin-provider-list');
    if (!providers.length) {
      list.appendChild(textNode('p', 'credits-muted', 'No providers registered.'));
      return list;
    }
    providers.forEach((provider) => {
      const row = shell().el('article', 'admin-provider-row');
      const head = shell().el('div', 'admin-provider-row-head');
      head.appendChild(textNode('strong', '', provider.display_name));
      head.appendChild(textNode('span', `admin-provider-state is-${provider.status}`, provider.status));
      const meta = textNode(
        'p',
        'admin-provider-row-meta',
        `${provider.provider_id} · ${provider.adapter_type} · ${provider.approved_base_url}`,
      );
      const key = provider.platform_credential;
      const keyState = textNode('p', 'admin-provider-credential', platformLabel(key));
      const actions = shell().el('div', 'admin-provider-row-actions');
      const edit = textNode('button', 'credits-key-action', 'Edit provider');
      edit.type = 'button';
      edit.addEventListener('click', () => fillProviderForm(provider));
      actions.appendChild(edit);
      // A provider with no key, or a revoked one, is offered neither action:
      // both would refuse server-side, and the row is the only place the
      // operator finds out which providers actually hold one.
      if (key && key.status !== 'revoked') {
        const verify = textNode('button', 'credits-key-action', 'Reverify key');
        verify.type = 'button';
        appendIcon(verify, 'icon-refresh');
        verify.addEventListener('click', () => reverifyPlatformKey(provider.provider_id));
        actions.appendChild(verify);
        const revoke = textNode('button', 'credits-key-action is-danger', 'Revoke key');
        revoke.type = 'button';
        appendIcon(revoke, 'icon-x');
        revoke.addEventListener('click', () => confirmRevoke(provider.provider_id, provider.display_name));
        actions.appendChild(revoke);
      }
      row.append(head, meta, keyState, actions);
      list.appendChild(row);
    });
    return list;
  }

  function renderProviderOptions(payload) {
    const select = element('adminPlatformProvider');
    if (!select) return;
    const providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const current = select.value;
    shell().clear(select);
    select.appendChild(textNode('option', '', 'Select a provider'));
    providers.forEach((provider) => {
      const option = shell().el('option', '', provider.display_name);
      option.value = provider.provider_id;
      select.appendChild(option);
    });
    // Restored only when it still exists. Forcing a vanished id back onto the
    // select would leave the platform-key form pointed at a provider the
    // registry no longer has, and the next Save would post the secret to it.
    if (providers.some((provider) => provider.provider_id === current)) select.value = current;
  }

  function paint(payload) {
    state.providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const host = element('adminProviderList');
    if (host) host.replaceChildren(...renderProviderList(payload).children);
    renderProviderOptions(payload);
  }

  async function load() {
    const seq = shell().nextSeq(SURFACE);
    try {
      const data = await shell().request('/api/admin/model-providers');
      if (!shell().isCurrent(SURFACE, seq)) return;
      paint(data);
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq)) return;
      paint({ providers: [] });
      setStatus(element('adminProviderStatusMessage'), error.message || 'Providers could not be loaded.', 'error');
    }
  }

  async function saveProvider(event) {
    event.preventDefault();
    const providerId = element('adminProviderId')?.value.trim();
    if (!providerId) {
      setStatus(element('adminProviderStatusMessage'), 'Provider ID is required.', 'error');
      return;
    }
    const existing = state.providers.find((provider) => provider.provider_id === providerId);
    const payload = {
      display_name: element('adminProviderDisplayName').value.trim(),
      adapter_type: element('adminProviderAdapter').value,
      approved_base_url: element('adminProviderBaseUrl').value.trim(),
      capabilities: existing?.capabilities || { model_discovery: true },
      byok_enabled: element('adminProviderByok').checked,
      platform_enabled: element('adminProviderPlatform').checked,
      status: element('adminProviderStatus').value,
      source: 'admin-console',
      reason: element('adminProviderReason').value.trim() || 'Provider registry update.',
      idempotency_key: operationKey('provider-registry'),
    };
    setStatus(element('adminProviderStatusMessage'), 'Saving provider…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
      setStatus(element('adminProviderStatusMessage'), 'Provider saved.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminProviderStatusMessage'), error.message || 'Provider could not be saved.', 'error');
    }
  }

  async function savePlatformKey(event) {
    event.preventDefault();
    const providerId = element('adminPlatformProvider')?.value;
    const secretInput = element('adminPlatformKeySecret');
    const secret = secretInput?.value || '';
    if (!providerId || !secret) {
      setStatus(element('adminPlatformKeyStatus'), 'Select a provider and enter the key once.', 'error');
      if (secretInput) secretInput.value = '';
      return;
    }
    setStatus(element('adminPlatformKeyStatus'), 'Saving and verifying…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'PUT',
        body: JSON.stringify({ api_key: secret, ...actionPayload('Configure platform model access.') }),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key saved and verification requested.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be saved.', 'error');
    } finally {
      // In `finally`, not on each branch: the field must be empty after a
      // refusal and after a redirect, not only after a success.
      if (secretInput) secretInput.value = '';
    }
  }

  async function reverifyPlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Reverifying platform key…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential/verify`, {
        method: 'POST',
        body: JSON.stringify(actionPayload('Retry platform provider verification.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key verification updated.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be verified.', 'error');
    }
  }

  // window.confirm is kept for this PR (design §8). Replacing it with
  // AdminShell.openDialog is a real improvement and a real scope increase; it
  // is recorded in the design's §11 rather than smuggled in here.
  async function confirmRevoke(providerId, displayName) {
    if (!window.confirm(`Revoke the platform key for ${displayName}?`)) return;
    await revokePlatformKey(providerId);
  }

  async function revokePlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Revoking platform key…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'DELETE',
        body: JSON.stringify(actionPayload('Revoke platform model access.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key revoked.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be revoked.', 'error');
    }
  }

  function bind() {
    if (state.bound) return;
    state.bound = true;
    element('adminProviderForm')?.addEventListener('submit', saveProvider);
    element('adminPlatformKeyForm')?.addEventListener('submit', savePlatformKey);
    element('adminProviderRefreshBtn')?.addEventListener('click', () => load());
  }

  function onRoute(detail) {
    if (detail?.route !== SURFACE) {
      // Leaving the surface clears the field: a secret typed and not submitted
      // must not still be sitting in a live input when the operator comes back
      // three routes later.
      const secretInput = element('adminPlatformKeySecret');
      if (secretInput) secretInput.value = '';
      return;
    }
    if (!shell().user()) return;
    bind();
    load();
  }

  window.AdminProviders = {
    renderProviderList, renderProviderOptions, platformLabel,
    load, savePlatformKey, confirmRevoke, onRoute,
  };
  document.addEventListener('admin:route', (event) => onRoute(event.detail));
})();
```

- [ ] **Step 4: Add the section, the script tag and the rail href to `admin.html`**

Add the module to the script list (after `admin-users.js` at `:17`):

```html
  <script src="js/admin-providers.js?v=1" defer></script>
```

Flip the rail's Providers entry from the outbound link to the hash route:

```html
      <a class="admin-tab" data-rail="providers" href="#providers"><svg aria-hidden="true"><use href="#icon-network"/></svg><span>Providers</span></a>
```

Insert the section after `<section id="profile" …>` (`:182`), before the `<footer>`. One flat `<section>`, no nested `<section>` — see the third spec gap above:

```html
      <section id="providers" class="detail-view" data-panel="providers" hidden aria-labelledby="providersTitle">
        <div class="page-head"><div><h1 id="providersTitle" tabindex="-1">Providers</h1><p class="muted">The approved model-provider registry and the platform credential</p></div><span class="credits-admin-badge">Admin only</span></div>
        <div class="admin-provider-grid">
          <div class="admin-provider-list-wrap">
            <div class="admin-provider-list-head">
              <span class="credits-section-kicker">Approved registry</span>
              <button id="adminProviderRefreshBtn" class="credits-icon-btn" type="button" aria-label="Refresh providers" title="Refresh providers"><svg aria-hidden="true"><use href="#icon-refresh"/></svg></button>
            </div>
            <div id="adminProviderList" class="admin-provider-list" aria-live="polite"><p class="credits-muted">Loading providers…</p></div>
          </div>
          <form id="adminProviderForm" class="admin-provider-form">
            <p class="credits-section-kicker">Provider registry</p>
            <label class="credits-key-field" for="adminProviderId"><span>Provider ID</span><input id="adminProviderId" type="text" maxlength="64" pattern="[a-z0-9_]{2,64}" autocomplete="off" required></label>
            <label class="credits-key-field" for="adminProviderDisplayName"><span>Display name</span><input id="adminProviderDisplayName" type="text" maxlength="100" autocomplete="off" required></label>
            <label class="credits-key-field" for="adminProviderAdapter"><span>Adapter</span><select id="adminProviderAdapter" class="control-select" autocomplete="off"><option value="openai_compatible">OpenAI-compatible</option><option value="openrouter">OpenRouter</option><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="gemini">Gemini</option></select></label>
            <label class="credits-key-field" for="adminProviderBaseUrl"><span>Approved base URL</span><input id="adminProviderBaseUrl" type="url" maxlength="500" placeholder="https://provider.example/v1" autocomplete="off" required></label>
            <div class="admin-provider-checks"><label><input id="adminProviderByok" type="checkbox" checked> BYOK enabled</label><label><input id="adminProviderPlatform" type="checkbox"> Platform enabled</label></div>
            <label class="credits-key-field" for="adminProviderStatus"><span>Status</span><select id="adminProviderStatus" class="control-select"><option value="enabled">Enabled</option><option value="disabled">Disabled</option></select></label>
            <label class="credits-key-field" for="adminProviderReason"><span>Reason</span><input id="adminProviderReason" type="text" maxlength="500" value="Provider registry update." autocomplete="off" required></label>
            <button class="auth-btn auth-btn-primary" type="submit"><svg aria-hidden="true"><use href="#icon-check-circle"/></svg><span>Save provider</span></button>
            <p id="adminProviderStatusMessage" class="credits-status" role="status" aria-live="polite"></p>
          </form>
        </div>
        <div class="admin-platform-key-area">
          <div class="admin-provider-heading"><div><p class="credits-section-kicker">Platform credential</p><h3>ATL model access</h3></div></div>
          <form id="adminPlatformKeyForm" class="admin-platform-key-form">
            <label class="credits-key-field" for="adminPlatformProvider"><span>Provider</span><select id="adminPlatformProvider" class="control-select" required><option value="">Select a provider</option></select></label>
            <label class="credits-key-field" for="adminPlatformKeySecret"><span>API key</span><input id="adminPlatformKeySecret" type="password" maxlength="4096" autocomplete="new-password" spellcheck="false" required></label>
            <button class="auth-btn auth-btn-primary" type="submit"><svg aria-hidden="true"><use href="#icon-check-circle"/></svg><span>Save and verify</span></button>
          </form>
          <p id="adminPlatformKeyStatus" class="credits-status" role="status" aria-live="polite"></p>
        </div>
      </section>
```

- [ ] **Step 5: Retype the Providers CSS into `admin.css`**

Append at the end of `admin.css`, before the media queries. Class names are kept as-is — the design's §8.1 counts these six families by their existing names, and PR2 moves `admin-credits.js`, which reuses the same `credits-*` vocabulary. Renaming now would give the two halves of one move different names.

```css
/* —— Providers (design §8, §8.1) ——
   Six class families retyped from styles.css against this file's tokens. The
   `credits-*` vocabulary keeps its names: PR2 moves admin-credits.js, which
   uses the same names, and renaming half a move is how two files end up
   describing the same control differently. The duplication with styles.css is
   accepted for two releases (§11); those copies die with ?view=admin. */
.admin-provider-grid{display:grid;grid-template-columns:minmax(300px,1.12fr) minmax(300px,.88fr);gap:24px;align-items:start;margin-top:8px}
.admin-provider-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;margin-bottom:16px}
.admin-provider-list-wrap,.admin-provider-form,.admin-platform-key-form{min-width:0;padding:16px;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
.admin-provider-list-head{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:34px}
.admin-provider-list-head .credits-section-kicker{margin-bottom:0}
.admin-provider-list{margin-top:8px;border-top:1px solid var(--line)}
.admin-provider-row{padding:13px 0;border-bottom:1px solid rgba(148,163,184,.12)}
.admin-provider-row:last-child{border-bottom:0}
.admin-provider-row-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
.admin-provider-row-head strong{min-width:0;overflow:hidden;color:var(--text);font-size:13px;text-overflow:ellipsis;white-space:nowrap}
.admin-provider-state{display:inline-flex;align-items:center;min-height:20px;padding:2px 7px;border:1px solid var(--line);border-radius:4px;color:var(--muted);font-size:10px;font-weight:700;text-transform:capitalize;white-space:nowrap}
.admin-provider-state.is-enabled{border-color:rgba(66,204,139,.38);color:var(--green)}
.admin-provider-state.is-disabled{border-color:rgba(240,128,128,.42);color:var(--red)}
.admin-provider-row-meta,.admin-provider-credential{margin:6px 0 0;color:var(--muted);font-family:var(--font-mono);font-size:10px;line-height:1.45;overflow-wrap:anywhere}
.admin-provider-credential{color:var(--muted)}
.admin-provider-row-actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.admin-provider-form{display:grid;gap:13px}
.admin-provider-form>.credits-section-kicker{margin-bottom:-2px}
.admin-provider-checks{display:flex;flex-wrap:wrap;gap:12px 18px;color:var(--muted);font-size:11px}
.admin-provider-checks label{display:inline-flex;align-items:center;gap:7px}
.admin-provider-checks input{width:auto;min-height:0;accent-color:var(--accent)}
.admin-platform-key-area{margin-top:24px;padding-top:24px;border-top:1px solid var(--line)}
.admin-platform-key-area .admin-provider-heading{margin-bottom:12px}
.admin-platform-key-form{display:grid;grid-template-columns:minmax(180px,.7fr) minmax(240px,1fr) auto;gap:12px;align-items:end}
.admin-provider-form>.auth-btn,.admin-platform-key-form>.auth-btn{display:inline-flex;align-items:center;justify-content:center;justify-self:start;width:max-content;gap:8px;min-height:36px;padding:7px 12px;white-space:nowrap}
.admin-provider-form .auth-btn svg,.admin-platform-key-form .auth-btn svg{width:15px;height:15px;flex:0 0 15px;fill:none;stroke:currentColor}

/* Buttons and the select. .auth-btn* and .control-select are (0,1,0), so they
   beat this file's bare `button{…}` / `select{…}` element rules at :10-13. */
.auth-btn{padding:7px 14px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text);cursor:pointer;font-size:14px;font-weight:600;white-space:nowrap;flex-shrink:0}
.auth-btn:hover{border-color:var(--accent);color:var(--accent)}
.auth-btn-primary{border-color:rgba(0,191,255,.55);background:linear-gradient(135deg,#00bfff,#38bdf8);color:#041018;box-shadow:0 4px 14px rgba(0,191,255,.2)}
.auth-btn-primary:hover{filter:brightness(1.05);border-color:rgba(0,191,255,.75);color:#041018}
.auth-btn-secondary{color:var(--muted)}
.auth-btn-danger{border-color:rgba(240,128,128,.55);background:rgba(240,128,128,.1);color:var(--red)}
.auth-btn-danger:hover{border-color:var(--red);background:rgba(240,128,128,.2);color:#fca5a5}
.control-select{width:100%;height:auto;min-height:38px;padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text);font-size:13px;color-scheme:dark}
.control-select option{background:var(--surface);color:var(--text)}
.control-select:focus{border-color:var(--accent);outline:none}
.control-select:disabled{cursor:not-allowed;color:var(--muted);opacity:.62}

/* Shared credits-* vocabulary (§8.1). */
.credits-section-kicker{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase}
.credits-admin-badge{display:inline-flex;align-items:center;min-height:22px;padding:3px 8px;border:1px solid rgba(0,191,255,.35);border-radius:4px;background:rgba(0,191,255,.08);color:var(--accent);font-size:11px;font-weight:700;white-space:nowrap}
.credits-icon-btn{display:inline-grid;place-items:center;width:34px;height:34px;flex:0 0 34px;min-height:34px;padding:0;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--muted);cursor:pointer}
.credits-icon-btn:hover,.credits-icon-btn:focus-visible{border-color:var(--accent);color:var(--accent)}
.credits-icon-btn svg{width:17px;height:17px;fill:none;stroke:currentColor}
.credits-key-field{display:grid;gap:7px;color:var(--text);font-size:12px;font-weight:600}
.credits-key-field input,.credits-key-field select{width:100%;min-height:38px;padding:8px 10px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--text);font-size:13px}
.credits-key-field input:focus,.credits-key-field select:focus{border-color:var(--accent);outline:none;box-shadow:0 0 0 2px rgba(0,191,255,.12)}
.credits-key-action{display:inline-flex;align-items:center;gap:5px;min-height:30px;padding:5px 8px;border:1px solid var(--line);border-radius:4px;background:transparent;color:var(--muted);cursor:pointer;font-size:10px;font-weight:700}
.credits-key-action:hover,.credits-key-action:focus-visible{border-color:var(--accent);color:var(--text);outline:none}
.credits-key-action.is-danger:hover,.credits-key-action.is-danger:focus-visible{border-color:rgba(240,128,128,.6);color:var(--red)}
.credits-key-action svg{width:15px;height:15px;fill:none;stroke:currentColor}
.credits-status,.credits-muted{margin:7px 0 0;color:var(--muted);font-size:12px;line-height:1.5}
.credits-status.is-success{color:var(--green)}
.credits-status.is-error{color:var(--red)}
.credits-status.is-pending{color:var(--amber)}
```

Add the monospace token the two `font-family:var(--font-mono)` rules need. `admin.css` has no such token today; append it to `:root` at `:6`, immediately before `--font-base`:

```css
--font-mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
```

Add the providers clauses to the two media queries. In `@media(max-width:1040px)`, append:

```css
.admin-provider-grid{grid-template-columns:1fr}.admin-platform-key-form{grid-template-columns:repeat(2,minmax(0,1fr))}.admin-platform-key-form .auth-btn{grid-column:1/-1}
```

In `@media(max-width:680px)`, append:

```css
.admin-platform-key-form{grid-template-columns:1fr}
```

- [ ] **Step 6: Register the route in `admin-shell.js`**

Change the `ROUTES` line added in Task 2:

```js
  const ROUTES = [...ANALYTICS_ROUTES, 'providers'];
```

In `route()`, add the view toggle after `showView('profile', …)` (`:441`):

```js
    showView('providers', parsed.route === 'providers');
    // The range group, the filter form and the freshness legend all describe
    // *daily analytics* figures. On Providers they describe nothing on screen,
    // and a range control that scopes nothing is worse than no control -- it
    // invites the operator to believe the registry is being filtered.
    showView('pageControls', parsed.route !== 'providers');
    showView('freshnessLegend', parsed.route !== 'providers');
```

- [ ] **Step 7: Rescope the module guard (design §9)**

First extend the import at `test_admin_page_modules.py:13`:

```python
from dashboard.backend.tests._frontend_source import fn_body, strip_comments
```

Then replace the module constants at `:16-25`:

```python
# Design N3: the read-only property is kept and its subject is narrowed. The
# analytics modules stay pinned GET-only and free of credential field names.
# Write modules are enumerated by name with their EXACT permitted verb set --
# exact, not a subset, so adding a verb is a test failure rather than a silent
# pass. Deleting the verb list and dropping api_key from the prohibited names
# would have left two tests that run, pass, and protect nothing.
READ_MODULES = ("admin-shell.js", "admin-live.js", "admin-overview.js", "admin-users.js")
WRITE_MODULES = {"admin-providers.js": {"PUT", "POST", "DELETE"}}
NAMES = READ_MODULES + tuple(WRITE_MODULES)
MODULES = {name: (FRONTEND / "js" / name).read_text(encoding="utf-8") for name in NAMES}
# The write-module scans below run on the stripped copy. A write module's header
# explains the rules it obeys, and every one of those sentences names a
# credential -- `strip_comments` is what keeps the guard asserting about the
# code rather than about the prose describing the code.
CODE = {name: strip_comments(source) for name, source in MODULES.items()}
READS = {name: MODULES[name] for name in READ_MODULES}
ALL = "\n".join(READS.values())
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
GLOBALS = {
    "admin-shell.js": "AdminShell",
    "admin-live.js": "AdminLive",
    "admin-overview.js": "AdminOverview",
    "admin-users.js": "AdminUsers",
    "admin-providers.js": "AdminProviders",
}

CREDENTIAL = re.compile(r"\w*(?:api_key|secret|credential|token|password)\w*", re.I)
SINK_ASSIGN = re.compile(r"\.(?:textContent|innerHTML|value)\s*=\s*(?P<rhs>[^;]+);")
SINK_CALL = re.compile(r"\.(?:setAttribute|append|appendChild|replaceChildren|createTextNode)\s*\(")
EMPTY_RHS = {"''", '""', "``"}
```

`ALL` now spans the read modules only, so `test_exact_endpoints_and_query_names`, `test_no_live_route_and_no_chartjs`, `test_d15_display_fields_are_not_read` and `test_prohibited_field_names_and_local_storage_are_absent` keep their existing meaning with no edit.

That narrowing is doing real work, not just avoiding churn. `test_exact_endpoints_and_query_names` asserts `"provider_id'" not in ALL` — a pin that the *analytics* client never asks a question about a provider. `admin-providers.js` names `provider_id` on nearly every line, so folding it into `ALL` would have forced that assertion to be deleted, and the property it protects would have gone with it. Keeping `ALL` as the read set is what lets the pin survive a page that now has a provider surface.

Replace `test_every_request_is_a_credentialed_get_made_by_the_shell` (as amended in Task 1) and add the three new cases:

```python
def test_every_request_is_a_credentialed_get_made_by_the_shell():
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET"}
    for name, source in READS.items():
        if name == "admin-shell.js":
            # request(), gate() and write() -- the third is the page's single
            # write path (N4), separately named so this file can tell readers
            # from writers by grep.
            assert source.count("fetch(") == 3
            assert "credentials: 'include'" in source
        else:
            assert "fetch(" not in source, name
            assert "XMLHttpRequest" not in source, name
            # A read module that reaches for the write path is a read module no
            # longer, and nothing else in this file would notice.
            assert "AdminShell.write" not in source, name
            assert "shell().write" not in source, name
    for verb in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL


def test_only_the_shell_owns_fetch():
    for name, code in CODE.items():
        if name == "admin-shell.js":
            continue
        assert "fetch(" not in code, name
        assert "XMLHttpRequest" not in code, name
    # request() stays GET-hardcoded and gains no options bag (N4). The exact
    # signature, not just the name: `request(path, options = {})` would satisfy
    # a substring check while being precisely the change N4 forbids.
    assert "async function request(path) {" in CODE["admin-shell.js"]
    assert "method: 'GET'" in CODE["admin-shell.js"]


def test_write_modules_declare_their_exact_verbs():
    """Exact, not a subset: a module that grows a verb has grown a capability,
    and the point of enumerating them is that the growth is visible here."""
    for name, permitted in WRITE_MODULES.items():
        code = CODE[name]
        found = set(re.findall(r"method:\s*'(\w+)'", code))
        assert found == permitted, (name, found, permitted)
        assert "AdminShell" in code or "shell()" in code, name


def test_credential_names_in_a_write_module_appear_only_as_request_body_keys():
    for name in WRITE_MODULES:
        code = CODE[name]
        occurrences = [line for line in code.splitlines() if "api_key" in line]
        assert len(occurrences) == 1, (name, occurrences)
        assert "JSON.stringify(" in occurrences[0], (name, occurrences[0].strip())
        for prohibited in (
            "session_id", "network_hash", "provider_response_body",
            "credential_ciphertext", "prompt", "strategy", "portfolio", "raw_user_agent",
        ):
            assert prohibited not in code, (name, prohibited)
        assert "localStorage" not in code, name
        assert "sessionStorage" not in code, name


def test_no_credential_field_reaches_a_dom_sink():
    """The pin a write surface needs and a read surface never did. A line may
    name a credential, or it may write to the DOM; not both. Clearing a field
    (`= ''`) is exempt -- that is the control, not the leak."""
    for name in WRITE_MODULES:
        for number, line in enumerate(CODE[name].splitlines(), start=1):
            if not CREDENTIAL.search(line):
                continue
            assigned = SINK_ASSIGN.search(line)
            if assigned:
                assert assigned.group("rhs").strip() in EMPTY_RHS, (name, number, line.strip())
            assert not SINK_CALL.search(line), (name, number, line.strip())
```

Extend the IIFE and script-list tests to cover the new module — both already iterate `MODULES` / `NAMES`, so `test_each_module_is_an_iife_exposing_exactly_one_global` and `test_rendering_is_text_content_only` need no edit. `test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded` also needs none: `set(srcs) == set(NAMES) | {"credit-format.js"}` now includes `admin-providers.js` because `NAMES` does.

- [ ] **Step 8: Update the two shell guards the new route and section touch**

`test_admin_page_shell.py` — add `"providers"` to the expected panel-region names in `test_panel_regions_carry_no_numeric_or_percentage_literal`:

```python
    assert names == [
        "live", "attention", "active-users", "activation", "sources", "retention",
        "value", "lifecycle", "credits", "revenue", "detail", "users", "profile", "providers",
    ]
```

In `test_subnav_routes_match_the_shell_router_and_the_aside_links_back`, drop the Providers outbound link and pin the hash route:

```python
    for href in (
        "/app?view=admin&amp;adminTab=users",
        "/app?view=admin&amp;adminTab=activity",
    ):
        assert href in ADMIN_HTML, href
    # Providers is no longer an outbound link; it is a route on this page (§8).
    assert "adminTab=providers" not in ADMIN_HTML
    assert 'data-rail="providers" href="#providers"' in ADMIN_HTML
```

In `EXPECTED_SCRIPTS`, add the sixth entry after `admin-users.js`:

```python
    "js/admin-providers.js?v=1",
```

`test_admin_analytics_frontend.py` — add `'src="js/admin-providers.js?v=1"'` to the `for tag in (...)` tuple in `test_app_lifecycle_and_cache_versions_are_wired`.

- [ ] **Step 9: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_admin_providers_frontend.py dashboard/backend/tests/test_admin_page_modules.py dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add dashboard/frontend/js/admin-providers.js dashboard/frontend/admin.html dashboard/frontend/admin.css dashboard/frontend/js/admin-shell.js dashboard/backend/tests/test_admin_providers_frontend.py dashboard/backend/tests/test_admin_page_modules.py dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "$(cat <<'EOF'
feat: port Providers into the /admin console

js/admin-providers.js replaces admin-model-providers.js's two host seams
(window.API.request, window.getStoredAuthUser) with AdminShell, and #providers
joins the hash router. The panel's markup and its six CSS families are retyped
into admin.html/admin.css; the wrapper is flattened to a div because the panel
guard's region regex is non-greedy and a nested section truncates it.

This is the page's first write surface, so the module guard is rescoped rather
than relaxed (design N3): READ_MODULES stay GET-only, WRITE_MODULES declare an
exact verb set, credential names are allowed only as request-body keys, and a
new pin refuses any line that both names a credential and reaches a DOM sink.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
EOF
)"
```

---

## Task 4: Remove Providers from `/app` and relabel Users

Deletes the old console's Providers tab, panel, module and CSS; redirects the old URL (N5); relabels the remaining tab (N6).

**Files:**
- Modify: `dashboard/frontend/app.html:2142` (rail button), `:2266-2316` (panel), `:2399` (script tag), `:16` + `:2394` + `:2401` (pins), `:2141` (label)
- Modify: `dashboard/frontend/app.js:5029-5031`, `:5407-5409`, `:10841-10843`
- Modify: `dashboard/frontend/js/admin-tabs.js:8-15`
- Modify: `dashboard/frontend/styles.css:12770-12960`, `:12991-13001`
- Delete: `dashboard/frontend/js/admin-model-providers.js`
- Delete: `dashboard/backend/tests/test_admin_model_providers_frontend.py`
- Test: `dashboard/backend/tests/test_admin_tabs_redirect.py`, `test_admin_analytics_frontend.py`, `test_admin_credits_frontend.py`, `test_frontend_fast_boot.py`, `test_credit_format_frontend.py`

**Interfaces:**
- Consumes: `#providers` on `/admin` (Task 3).
- Produces: `AdminTabs.ALLOWED_TABS` = `{users, activity}`; `?adminTab=providers` navigates to `/admin#providers`.

- [ ] **Step 1: Write the failing tests**

Rewrite `test_admin_tabs_redirect.py`'s two affected cases and add the N5 pin. Replace `test_entering_on_any_tab_stays_in_the_console` (`:84-91`) and `test_default_tab_is_users_and_analytics_is_not_a_tab` (`:94-100`):

```python
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
        ("providers", [["assign", "/admin#providers"]]),
    ],
)
def test_entering_on_a_tab_stays_in_the_console_unless_the_tab_has_moved(tab, expected_nav):
    result = _run(
        "fire(docListeners, 'DOMContentLoaded');"
        f"setHref('https://atl.example/app?view=admin&adminTab={tab}');"
        "window.AdminTabs.onEnter();"
    )
    assert result["nav"] == expected_nav


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
    # window.location.replace is still banned -- a *replace* would destroy the  # CORRECTED 2026-09-18 (finding I1): location.replace() drops the CURRENT entry, not the referrer -- shipped code uses replace. (prose continues on :2015)
    # history entry the operator came from. assign is the one navigation this
    # controller may make, and only for the retired Providers tab (N5).
    assert "window.location.replace" not in ADMIN_TABS_JS  # CORRECTED 2026-09-18 (finding I1): location.replace() drops the CURRENT entry, not the referrer -- shipped code uses replace.
    assert ADMIN_TABS_JS.count("window.location.assign") == 1  # CORRECTED 2026-09-18 (finding I1): location.replace() drops the CURRENT entry, not the referrer -- shipped code uses replace.
    assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS
```

Update `test_admin_analytics_frontend.py::test_admin_rail_has_three_tabs_defaulting_to_users`:

```python
def test_admin_rail_has_two_tabs_defaulting_to_account_management():
    """Providers moved to /admin in the 09-17 consolidation (design N2); Account
    Management and Activity are one module and port together in PR2."""
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index("</nav>", nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    expected = ["users", "activity"]
    assert nav_markup.count("data-admin-tab=") == 2
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="activity"')
    assert 'data-admin-tab="providers"' not in nav_markup
    assert 'aria-orientation="vertical"' in nav_markup
    assert 'id="adminTabUsers" class="admin-tab is-active"' in nav_markup
    # N6: label-only. The tab id, the data attribute, the panel id and the
    # query value are all still "users", so no URL breaks and admin-tabs.js
    # needs no new normalisation.
    assert "<span>Account Management</span>" in nav_markup
    assert "<span>Users</span>" not in nav_markup
    assert 'aria-label="Account Management"' in nav_markup
    assert '<section id="adminPanelUsers" class="admin-tab-panel" role="tabpanel" aria-labelledby="adminTabUsers" data-admin-panel="users">' in APP_HTML
```

Update the two tab counts in `test_admin_credits_frontend.py` (`:41-50` and `:136-146`) the same way, and its pin test (`:148-150`):

```python
    assert 'data-admin-tab="providers"' not in admin_markup
    assert 'data-admin-tab="users"' in admin_markup
    assert 'data-admin-tab="activity"' in admin_markup
    nav_start = admin_markup.index('<nav id="adminTabs"')
    nav_end = admin_markup.index('</nav>', nav_start)
    nav_markup = admin_markup[nav_start:nav_end]
    assert nav_markup.count('data-admin-tab=') == 2
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="activity"')
```

```python
def test_admin_tabs_have_two_tabs_in_usage_order_and_legacy_alias():
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index('</nav>', nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    assert nav_markup.count('data-admin-tab=') == 2
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="activity"')
    assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS


def test_admin_visual_assets_use_fresh_cache_versions():
    assert 'js/admin-credits.js?v=6' in APP_HTML
    assert 'js/admin-tabs.js?v=10' in APP_HTML
```

Add to `test_admin_analytics_frontend.py`, beside the existing family guard:

```python
def test_the_provider_families_left_styles_css_with_their_markup():
    """Dead CSS for markup that no longer exists is how a 14,000-line stylesheet
    is grown. The families move to admin.css in the same PR as the panel."""
    for family in (".admin-provider-", ".admin-platform-key-"):
        assert family not in STYLES, family
    for kept in (".admin-workspace", ".admin-rail", ".admin-tab:focus-visible", ".admin-rail button"):
        assert kept in STYLES, kept


def test_the_old_provider_module_is_gone_from_the_page_and_the_tree():
    assert "admin-model-providers" not in APP_HTML
    assert "AdminModelProviders" not in APP_JS
    assert not (FRONTEND / "js" / "admin-model-providers.js").exists()
```

`test_admin_analytics_frontend.py` has no `FRONTEND` of its own — it imports `APP_HTML`, `APP_JS`, `STYLES` from `_frontend_source`. Extend that import (`:19`) rather than defining a second `Path` root:

```python
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, FRONTEND, STYLES
```

- [ ] **Step 2: Run them to verify they fail**

```bash
pytest dashboard/backend/tests/test_admin_tabs_redirect.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py -q
```

Expected: FAIL — `assert result["nav"] == [["assign", "/admin#providers"]]` gets `[]`; the tab-count assertions get 3.

- [ ] **Step 3: Remove the Providers tab and panel from `app.html`**

Delete line `:2142` (the `adminTabProviders` button) and the whole `<section id="adminPanelProviders" …>` block at `:2266-2316`, including its nested `<section id="adminModelProvidersSection">`. Delete the script tag at `:2399`.

Relabel the remaining Users button (`:2141`) — label, `aria-label` and `title` only:

```html
            <button id="adminTabUsers" class="admin-tab is-active" type="button" role="tab" aria-label="Account Management" title="Account Management" aria-selected="true" aria-controls="adminPanelUsers" tabindex="0" data-admin-tab="users"><svg aria-hidden="true"><use href="#icon-users"/></svg><span>Account Management</span></button>
```

Bump the three pins:

```html
    <link rel="stylesheet" href="styles.css?v=142">
    <script src="app.js?v=134" defer></script>
    <script src="js/admin-tabs.js?v=10" defer></script>
```

- [ ] **Step 4: Remove the three `app.js` call sites**

Delete `:5029-5031`:

```js
  if (window.AdminModelProviders) {
    window.AdminModelProviders.syncAuth(user);
  }
```

Delete `:5407-5409` (inside the `adminRefreshBtn` handler) and `:10841-10843` (inside `navigateToPage('admin')`):

```js
    if (window.AdminModelProviders) {
      window.AdminModelProviders.onEnter();
    }
```

- [ ] **Step 5: Redirect the retired tab in `admin-tabs.js`**

Replace `:5-15`:

```js
  // Analytics lives on the standalone /admin page (design §7.5), and Providers
  // joined it in the 09-17 consolidation (N2). This console keeps Users (account
  // management + grant pool) and Activity until the follow-up port -- they are
  // one 712-line module and move together.
  const DEFAULT_TAB = 'users';
  const ALLOWED_TABS = new Set(['users', 'activity']);
  // N5: an explicit hop, not a silent fallback. normalizeTab would otherwise
  // coerce this to DEFAULT_TAB and land the operator on Account Management with
  // nothing on screen saying the tab they asked for lives elsewhere now. assign
  // rather than replace: the entry they came from stays in history.  // CORRECTED 2026-09-18 (finding I1): location.replace() drops the CURRENT entry, not the referrer -- shipped code uses replace.
  const RETIRED_TABS = Object.freeze({ providers: '/admin#providers' });
  let initialized = false;

  function retiredDestination(value) {
    return Object.hasOwn(RETIRED_TABS, value) ? RETIRED_TABS[value] : null;
  }

  function normalizeTab(value) {
    const normalizedValue = value === 'grant-pool' ? 'users' : value;
    return ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
  }
```

In `setTab`, short-circuit before any painting:

```js
  function setTab(value, { updateUrl = true } = {}) {
    const retired = retiredDestination(value);
    if (retired) {
      window.location.assign(retired);  // CORRECTED 2026-09-18 (finding I1): location.replace() drops the CURRENT entry, not the referrer -- shipped code uses replace.
      return value;
    }
    const tab = normalizeTab(value);
```

- [ ] **Step 6: Delete the old module, its test, and its CSS**

```bash
git rm dashboard/frontend/js/admin-model-providers.js dashboard/backend/tests/test_admin_model_providers_frontend.py
```

In `styles.css`, delete the block from the `/* Admin provider controls stay separate … */` comment (`:12769`) through the `.admin-provider-form .auth-btn svg, .admin-platform-key-form .auth-btn svg { … }` rule (`:12961`), and the three provider clauses inside `@media (max-width: 760px)` (`:12991-13001`):

```css
    .admin-provider-grid {
        grid-template-columns: 1fr;
    }

    .admin-platform-key-form {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .admin-platform-key-form .auth-btn {
        grid-column: 1 / -1;
    }
```

Leave `.admin-workspace`, `.admin-rail`, `.admin-tab*` and the `credits-*` families in `styles.css`: the console still has a two-entry rail and still runs `admin-credits.js`. They die with `?view=admin` in PR2 (design §11).

- [ ] **Step 7: Sweep the cache-buster pins across all five owners**

```bash
grep -rn 'styles.css?v=\|app.js?v=\|admin-tabs.js?v=' dashboard/backend/tests/
```

Update every hit to `?v=142`, `?v=134`, `?v=10` respectively. The known set is `test_admin_analytics_frontend.py:152-154`, `test_admin_credits_frontend.py:150`, `test_credit_format_frontend.py`, `test_frontend_fast_boot.py`. Trust the grep, not this list — project memory records that this pin family has surprised a previous session.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/ -q -k "admin or frontend or composition"
```

Expected: PASS, with `test_admin_model_providers_frontend.py` no longer collected.

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/js/admin-tabs.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_tabs_redirect.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "$(cat <<'EOF'
feat: retire the in-app Providers tab and rename Users

Providers now lives on /admin, so the old console loses its tab, its panel,
its module and its CSS families. ?adminTab=providers redirects to
/admin#providers rather than falling back silently to Account Management --
the URL is in histories and bookmarks, and a silent fallback lands the
operator on a different page with nothing saying so.

The remaining tab is relabelled Account Management (N6). The rename is
label-only: data-admin-tab, the panel id and the query value are unchanged,
so no URL breaks and admin-tabs.js needs no new normalisation.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
EOF
)"
```

---

## Task 5: Amend the 2026-09-15 design and verify the whole suite

The four §10 amendments, applied in place and marked, so a later reader of the older document is not told to undo this one. Then the full suite.

**Files:**
- Modify: `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` (§7.2, §7.5, §13, §14)

**Interfaces:** none.

- [ ] **Step 1: Locate the four passages**

```bash
grep -n 'admin-users.js\|The old console keeps\|Port Users, Providers or Activity\|Providers and Activity absorption' docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md
```

- [ ] **Step 2: Amend §7.2's module table**

Add a row for the new module, immediately after the `js/admin-users.js` row. Keep the existing rows byte-identical — the `admin-users.js` parenthetical about account management not being ported is still true:

```markdown
| `js/admin-providers.js` | `AdminProviders` | The approved provider registry and the platform credential. *(Amended 2026-09-17: added by the navigation consolidation; the page's only write surface, reaching the network through `AdminShell.write`.)* |
```

- [ ] **Step 3: Amend §7.5**

Replace the sentence "The old console keeps **Users**, **Providers** and **Activity**" with:

```markdown
The old console keeps **Users** (relabelled **Account Management**) and
**Activity**. *(Amended 2026-09-17: Providers ported to `/admin` in the
navigation consolidation; `?adminTab=providers` now redirects to
`/admin#providers`.)*
```

And in the following sentence, drop `providers` from the list of aside links:

```markdown
The new page's aside links to `…=users` and `…=activity` until the follow-up
port. *(Amended 2026-09-17: `…=providers` removed — it is `#providers` on this
page now.)*
```

- [ ] **Step 4: Amend §13's PR C row**

The "Must not" column currently reads "Port Users, Providers or Activity." A reader hitting that line after this PR would conclude the port was a violation. Append to that cell:

```markdown
*(Amended 2026-09-17: this prohibition bound **PR C only**. Providers was ported
to `/admin` by the 09-17 navigation consolidation, which executes D5's staged
absorption ahead of schedule. Account Management and Activity remain unported.)*
```

- [ ] **Step 5: Amend §14's follow-ups**

Replace "Providers and Activity absorption into `/admin` (D5)" with:

```markdown
Activity and Account Management absorption into `/admin` (D5). *(Amended
2026-09-17: Providers landed in the navigation consolidation; the two that
remain are one 712-line module and port together.)*
```

- [ ] **Step 6: Verify the amendments say what they claim**

```bash
grep -c 'Amended 2026-09-17' docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md
```

Expected: `5` (§7.2, both §7.5 sentences, §13, §14).

- [ ] **Step 7: Run the full suite**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: `4864 + N passed`, 162 skipped, **0 failed**, where the 162 skips are 114 Postgres, 43 landing `node_modules`, 4 vnpy, 1 pg_only. **A skip in any `test_admin_*_frontend.py` means `node` is missing from `PATH` and this PR's entire new test layer silently did not run** — that is a failure to report, not a pass.

Sanity-check the count moved in the right direction:

```bash
pytest dashboard/backend/tests/ -q --collect-only | tail -1
```

- [ ] **Step 8: Commit**

```bash
git add docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md
git commit -m "$(cat <<'EOF'
docs: amend the admin-layer design for the 09-17 consolidation

Four passages of the 2026-09-15 design describe a split the 09-17 navigation
consolidation closes for Providers. Each is marked in place rather than
rewritten, so the original reasoning stays legible -- in particular §13's PR C
row, whose "Must not: Port Users, Providers or Activity" would otherwise tell
a later reader the port was a violation.

D5 itself needs no amendment: staged absorption with full absorption as the
committed end state is exactly what this was.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J9HL9GfiFhxPKJ6YybDaSZ
EOF
)"
```

---

## Verification checklist before opening the PR

- [ ] `pytest dashboard/backend/tests/ -q` — 0 failed, and **zero** skips in any `test_admin_*` module.
- [ ] `grep -rn 'AdminModelProviders\|admin-model-providers' dashboard/` returns nothing.
- [ ] `grep -c 'fetch(' dashboard/frontend/js/admin-shell.js` returns 3; `grep -l 'fetch(' dashboard/frontend/js/admin-*.js` returns only `admin-shell.js`.
- [ ] `grep -n 'api_key' dashboard/frontend/js/admin-providers.js` returns exactly one line, and that line contains `JSON.stringify(`.
- [ ] `grep -n -- '--border-color\|--text-primary\|--bg-card\|--info-color' dashboard/frontend/admin.css` returns nothing.
- [ ] Every `<use href="#icon-…">` in `admin.html` has a matching `<symbol id="…">` in the same file.
- [ ] `git status --short` shows no `dashboard/storage/data/backtest.db`.
- [ ] Branch is `feat/admin-nav-consolidation` and is **not** the merged `feat/admin-prC-admin-page`.

## Deliberately out of scope (design §11)

- PR2: Account Management and Activity. `admin-credits.js` ported whole, `/app?view=admin` deleted, `adminView` removed from `app.html`, the ~475-line admin block removed from `app.js` (`:4065-4540`), `admin-tabs.js` deleted, and the now-dead `admin-*` / `credits-admin-*` families removed from `styles.css`.
- `window.confirm` → `AdminShell.openDialog` in the providers module.
- A back-link from `/app?view=admin` to `/admin` — named so its absence reads as a decision, not an oversight. PR2 deletes that view.
