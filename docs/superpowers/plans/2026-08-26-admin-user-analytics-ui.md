# Admin User Analytics UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the read-only Admin Analytics overview and dedicated User Analytics Profile, backed only by the PR 2 Admin Analytics query APIs.

**Architecture:** Keep the existing server-rendered Admin shell and add one isolated vanilla-JavaScript module, `admin-analytics.js`, that owns Analytics URL state, API reads, rendering, pagination, and partial errors. The module consumes explicit display-safe PR 2 response contracts captured as synthetic JSON fixtures; it does not add repositories, metrics, instrumentation, schemas, or backend routes. Existing account mutations remain in the Users tab, reached through a small `AdminTabs.openAccountManagement()` bridge.

**Tech Stack:** HTML, vanilla JavaScript, CSS, Chart.js 4.4.0 already loaded by `app.html`, pytest static/frontend contract tests, synthetic JSON fixtures.

**Spec:** `docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`

## Global Constraints

- PR 3 is UI-only: do not modify PR 1 Analytics foundation or PR 2 instrumentation, metric, query-service, schema, repository, retention, backfill, authorization, or API implementation code.
- Rebase this worktree onto the completed PR 2 branch before implementation. The four `/api/admin/analytics/*` routes and their centralized Admin authorization are hard dependencies, not work to recreate in PR 3.
- Use only display-safe fields returned by PR 2. Never render, log, fixture, persist, or add URL state for API keys, authentication tokens, passwords, verification codes, prompt/instruction/strategy/portfolio/form text, raw provider bodies, full IP addresses, raw User-Agent headers, encrypted credential ciphertext, network hashes, or raw analytics session identifiers.
- Keep Analytics read-only. Role changes, Grant Credits operations, provider configuration, and every other mutation remain under the existing Users or Providers surfaces.
- Default Analytics queries exclude Admin accounts and accounts marked `analytics_excluded`; `Include internal accounts` is an explicit temporary filter.
- Preserve the existing Admin URL key `adminTab`; `adminTab=analytics` is first and default, while legacy `adminTab=grant-pool` continues to normalize to `users`.
- Preserve ARIA tab roles, `aria-selected`, roving `tabindex`, left/right keyboard cycling, URL synchronization, and global page Back/Forward behavior. Add Home/End support for both Admin tabs and profile section tabs.
- Detailed Timeline, Runs, Usage, and Sessions data loads independently with opaque cursors. A section failure retains already-rendered data and affects only that section.
- Overview panel failures render exactly `This metric is temporarily unavailable.` in the failed panel while all successful panels and the attention table remain usable.
- All dates sent to PR 2 use inclusive UTC `YYYY-MM-DD` query values. All timestamps displayed by the browser use the administrator's locale and retain the source timestamp in a `<time datetime="2026-08-26T12:34:56Z">` attribute.
- Use `textContent`, DOM node creation, and existing `escapeHtml()` only where an existing table-string pattern makes it unavoidable. Do not introduce unsanitized response-driven `innerHTML`.
- Use synthetic identities, fake provider/model IDs, and zero-secret fixtures only. Do not use or commit a real API key, local database, `.superpowers/`, or `work/` content.
- Do not add a third-party analytics, charting, state-management, or testing dependency.

## PR 2 API Contract Required by PR 3

PR 3 must first rebase onto PR 2 and compare these shapes with the committed router/schema tests. If PR 2 deliberately uses a different display-safe field name, update the fixture and frontend in the same PR 3 commit; do not add compatibility guesses, silently accept multiple names, or modify PR 2 backend behavior from this branch.

### `GET /api/admin/analytics/overview`

Query parameters:

```text
start_date=2026-07-28
end_date=2026-08-26
billing_mode=all|byok|platform_credits
provider_id=<safe provider id, omitted for all>
model_id=<safe model id, omitted for all>
include_internal=false|true
```

Required response:

```json
{
  "last_updated_at": "2026-08-26T12:34:56Z",
  "filter_options": {
    "billing_modes": ["byok", "platform_credits"],
    "providers": [{"provider_id": "openrouter", "display_name": "OpenRouter"}],
    "models": [{"model_id": "openai/gpt-5.5", "display_name": "GPT-5.5"}]
  },
  "panels": {
    "snapshot": {
      "data": {
        "active_users_7d": 42,
        "first_successful_run_conversion": 0.625,
        "backtest_success_rate": 0.8,
        "platform_model_cost": {"amount": "12.340000", "currency": "USD"}
      },
      "error": null
    },
    "engagement": {
      "data": {
        "trend": [{"date": "2026-08-26", "active_users": 12, "completed_runs": 18}],
        "activation_funnel": [
          {"key": "signed_up", "label": "Signed up", "users": 80, "conversion_rate": 1.0},
          {"key": "first_success", "label": "First successful run", "users": 50, "conversion_rate": 0.625}
        ]
      },
      "error": null
    },
    "health": {
      "data": {
        "states": [
          {"status": "blocked", "label": "Blocked", "users": 2},
          {"status": "needs_attention", "label": "Needs Attention", "users": 3},
          {"status": "dormant", "label": "Dormant", "users": 8},
          {"status": "onboarding", "label": "Onboarding", "users": 11},
          {"status": "active", "label": "Active", "users": 37}
        ],
        "friction": [
          {"error_category": "credential_invalid", "label": "Invalid credential", "affected_users": 4, "failures": 9}
        ]
      },
      "error": null
    }
  }
}
```

Each `panels.*` member is an independent envelope. PR 2 may return `data: null` and a safe non-empty `error` for one member while returning data for the others. The UI never displays the backend error text; it maps any non-null panel error to the approved temporary-unavailable copy.

### `GET /api/admin/analytics/users`

Query parameters:

```text
query=<display name or email, omitted when empty>
status=all|blocked|needs_attention|dormant|onboarding|active
last_activity_from=2026-07-28
last_activity_to=2026-08-26
sort=attention|last_activity_desc|failures_desc|runs_desc
limit=25
cursor=<opaque cursor, omitted for the first page>
billing_mode=all|byok|platform_credits
provider_id=<safe provider id, omitted for all>
model_id=<safe model id, omitted for all>
include_internal=false|true
```

Required response:

```json
{
  "last_updated_at": "2026-08-26T12:34:56Z",
  "users": [
    {
      "user_id": 101,
      "display_name": "Synthetic Ada",
      "email": "ada.synthetic@example.test",
      "status": "needs_attention",
      "reason_code": "three_failed_runs_24h",
      "human_readable_reason": "Three consecutive runs failed in the last 24 hours.",
      "last_meaningful_activity_at": "2026-08-26T11:20:00Z",
      "recent_run_count": 6,
      "recent_failed_run_count": 3
    }
  ],
  "next_cursor": "synthetic-next-page",
  "total": 1
}
```

### `GET /api/admin/analytics/users/{user_id}`

Required response:

```json
{
  "last_updated_at": "2026-08-26T12:34:56Z",
  "user": {
    "user_id": 101,
    "display_name": "Synthetic Ada",
    "email": "ada.synthetic@example.test",
    "joined_at": "2026-07-01T09:00:00Z",
    "last_meaningful_activity_at": "2026-08-26T11:20:00Z",
    "primary_billing_mode": "platform_credits",
    "default_provider": {"provider_id": "openrouter", "display_name": "OpenRouter"},
    "country_code": "US",
    "device_category": "desktop",
    "browser_family": "Chrome"
  },
  "state": {
    "status": "needs_attention",
    "reason_code": "three_failed_runs_24h",
    "human_readable_reason": "Three consecutive runs failed in the last 24 hours.",
    "evidence_event_ids": ["evt_synthetic_001", "evt_synthetic_002", "evt_synthetic_003"],
    "calculated_at": "2026-08-26T12:33:00Z"
  },
  "milestones": [
    {"key": "signup", "label": "Signed up", "occurred_at": "2026-07-01T09:00:00Z"},
    {"key": "first_successful_run", "label": "First successful run", "occurred_at": "2026-07-02T10:00:00Z"}
  ],
  "summary": {
    "recent_footprint_events": [
      {"event_name": "page_viewed", "label": "Viewed Agents", "occurred_at": "2026-08-26T11:20:00Z"}
    ],
    "runs": {"total": 12, "completed": 8, "failed": 3, "cancelled": 1},
    "billing_lane_mix": [
      {"billing_mode": "platform_credits", "runs": 7},
      {"billing_mode": "byok", "runs": 5}
    ],
    "model_usage": {"input_tokens": 12000, "output_tokens": 3000, "total_tokens": 15000},
    "atl_model_cost": {"amount": "4.250000", "currency": "USD"},
    "top_product_page": {"page_view": "agents", "label": "Agents", "views": 19}
  }
}
```

Calling this base detail endpoint is the PR 2 operation that creates the required Admin analytics access-log record. PR 3 must not add a second logging request.

### `GET /api/admin/analytics/users/{user_id}/activity`

Query parameters:

```text
section=timeline|runs|usage|sessions
limit=50
cursor=<opaque cursor, omitted for the first page>
```

Every section response uses:

```json
{
  "section": "timeline",
  "items": [],
  "next_cursor": null,
  "last_updated_at": "2026-08-26T12:34:56Z"
}
```

Section item contracts:

```json
{
  "timeline": {
    "event_id": "evt_synthetic_004",
    "occurred_at": "2026-08-26T11:20:00Z",
    "event_name": "backtest_failed",
    "label": "Backtest failed",
    "event_group": "run",
    "outcome": "failed",
    "error_category": "provider_timeout",
    "provider_id": "openrouter",
    "model_id": "openai/gpt-5.5",
    "billing_mode": "platform_credits"
  },
  "runs": {
    "run_id": "run_synthetic_001",
    "requested_at": "2026-08-26T11:00:00Z",
    "completed_at": "2026-08-26T11:02:00Z",
    "outcome": "failed",
    "error_category": "provider_timeout",
    "provider_id": "openrouter",
    "model_id": "openai/gpt-5.5",
    "billing_mode": "platform_credits",
    "duration_ms": 120000
  },
  "usage": {
    "usage_id": "usage_synthetic_001",
    "occurred_at": "2026-08-26T11:02:00Z",
    "provider_id": "openrouter",
    "model_id": "openai/gpt-5.5",
    "billing_mode": "platform_credits",
    "input_tokens": 1400,
    "output_tokens": 350,
    "total_tokens": 1750,
    "atl_model_cost": {"amount": "0.430000", "currency": "USD"},
    "atl_credits_debited": "1.000000"
  },
  "sessions": {
    "started_at": "2026-08-26T10:30:00Z",
    "last_activity_at": "2026-08-26T11:20:00Z",
    "duration_seconds": 3000,
    "page_views": 7,
    "top_page": {"page_view": "agents", "label": "Agents"},
    "country_code": "US",
    "device_category": "desktop",
    "browser_family": "Chrome"
  }
}
```

For BYOK usage, `atl_model_cost.amount` and `atl_credits_debited` must both be zero. The UI labels those fields as ATL cost and ATL Credits debited rather than implying that token usage is a platform charge.

## File Structure

- `dashboard/frontend/app.html` — adds the first Analytics Admin tab, overview/profile semantic markup, filters, panel fallbacks, tables, profile section tabs, and script include.
- `dashboard/frontend/js/admin-tabs.js` — owns four-tab selection, keyboard behavior, `adminTab` URL state, tab-change notification, and the account-management bridge.
- `dashboard/frontend/js/admin-analytics.js` — owns the read-only Analytics API client, filter/profile URL state, rendering, independent request state, pagination, refresh, and access-loss behavior.
- `dashboard/frontend/app.js` — calls `AdminAnalytics.onEnter()`, `syncAuth()`, and `refresh()` from the existing Admin lifecycle without moving any account-management logic.
- `dashboard/frontend/styles.css` — adds scoped Analytics hierarchy, chart, health, table, profile, state, responsive, focus, and reduced-motion styles.
- `dashboard/backend/tests/test_admin_analytics_frontend.py` — static frontend contracts for tab order, API/query names, URL state, read-only behavior, partial failures, cursors, ARIA, safe rendering, and cache versions.
- `dashboard/backend/tests/fixtures/admin_analytics/overview.json` — successful synthetic overview response.
- `dashboard/backend/tests/fixtures/admin_analytics/overview_partial_error.json` — one failed overview panel with two successful siblings.
- `dashboard/backend/tests/fixtures/admin_analytics/users.json` — synthetic attention-list response.
- `dashboard/backend/tests/fixtures/admin_analytics/user_detail.json` — synthetic base profile response.
- `dashboard/backend/tests/fixtures/admin_analytics/activity_timeline.json` — synthetic Timeline page.
- `dashboard/backend/tests/fixtures/admin_analytics/activity_runs.json` — synthetic Runs page.
- `dashboard/backend/tests/fixtures/admin_analytics/activity_usage.json` — synthetic Usage page with BYOK and Platform Credits rows.
- `dashboard/backend/tests/fixtures/admin_analytics/activity_sessions.json` — synthetic Sessions page without a raw session identifier.

---

### Task 1: Lock the PR 2 display contract with safe frontend fixtures

**Files:**
- Create: `dashboard/backend/tests/fixtures/admin_analytics/overview.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/overview_partial_error.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/users.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/user_detail.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/activity_timeline.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/activity_runs.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/activity_usage.json`
- Create: `dashboard/backend/tests/fixtures/admin_analytics/activity_sessions.json`
- Create: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: The four PR 2 Admin Analytics endpoints and exact response members defined above.
- Produces: `load_fixture(name: str) -> dict`, a canonical safe fixture set, and failing source-contract tests that later tasks satisfy.

- [ ] **Step 1: Rebase onto PR 2 and prove the routes exist before UI work**

Run:

```bash
git rebase feature/admin-user-analytics-metrics
rg -n '/api/admin/analytics/(overview|users)' dashboard/backend
```

Expected: PR 2 route/schema tests identify all four routes. If the PR 2 branch still points at the design-only commit, stop implementation because the required backend dependency is absent; do not implement backend substitutes in PR 3.

- [ ] **Step 2: Create the successful synthetic fixtures**

Create fixtures using the exact response shapes in **PR 2 API Contract Required by PR 3**. Expand array examples so each renderer has at least two rows and include these deliberate cases:

```json
{
  "identity_cases": [
    {"display_name": "Synthetic Ada", "email": "ada.synthetic@example.test"},
    {"display_name": "<Synthetic & Grace>", "email": "grace.synthetic@example.test"}
  ],
  "nullable_display_cases": {
    "default_provider": null,
    "country_code": null,
    "device_category": null,
    "browser_family": null,
    "completed_at": null,
    "error_category": null
  }
}
```

Represent these cases in their normal endpoint records rather than adding the wrapper above to an API response. The special-character display name proves response text is escaped; nullable fields prove the UI renders `Unknown` or `—` instead of throwing.

- [ ] **Step 3: Create the partial-error and billing-lane fixtures**

In `overview_partial_error.json`, copy the successful overview response, then set only the engagement envelope to:

```json
{
  "data": null,
  "error": "synthetic_query_failure"
}
```

In `activity_usage.json`, include one `platform_credits` row with non-zero `atl_model_cost.amount` and `atl_credits_debited`, plus one `byok` row with both values set to zero.

- [ ] **Step 4: Write the failing fixture and source-contract tests**

Create `dashboard/backend/tests/test_admin_analytics_frontend.py` with this foundation:

```python
"""Frontend and PR2 response contracts for the read-only Admin Analytics UI."""

import json
from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, STYLES


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"
ANALYTICS_JS_PATH = FRONTEND / "js" / "admin-analytics.js"
ADMIN_TABS_JS_PATH = FRONTEND / "js" / "admin-tabs.js"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def test_safe_fixtures_have_no_prohibited_response_fields():
    prohibited = {
        "api_key", "auth_token", "password", "verification_code",
        "prompt", "instruction", "strategy", "portfolio", "form_value",
        "provider_response_body", "ip_address", "user_agent",
        "credential_ciphertext", "network_hash", "session_id",
    }
    for path in sorted(FIXTURES.glob("*.json")):
        payload = load_fixture(path.name)
        assert prohibited.isdisjoint(set(walk_keys(payload))), path.name


def test_partial_error_fixture_fails_only_engagement():
    payload = load_fixture("overview_partial_error.json")
    assert payload["panels"]["snapshot"]["data"]
    assert payload["panels"]["engagement"] == {
        "data": None,
        "error": "synthetic_query_failure",
    }
    assert payload["panels"]["health"]["data"]


def test_byok_fixture_never_reports_atl_cost_or_credits_debit():
    payload = load_fixture("activity_usage.json")
    byok = next(item for item in payload["items"] if item["billing_mode"] == "byok")
    assert byok["atl_model_cost"]["amount"] == "0.000000"
    assert byok["atl_credits_debited"] == "0.000000"
```

Also add failing tests asserting that `app.html` contains `adminTabAnalytics`, `adminPanelAnalytics`, `adminAnalyticsProfile`, and `js/admin-analytics.js`; the JavaScript module contains all four endpoint strings; and the stylesheet contains `.admin-analytics-overview` and `.admin-analytics-profile`.

- [ ] **Step 5: Run the contract file and confirm the UI assertions fail**

Run:

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: fixture-safety tests pass; UI/source tests fail because the Analytics markup, module, and styles do not exist yet.

- [ ] **Step 6: Commit the contract boundary**

```bash
git add dashboard/backend/tests/fixtures/admin_analytics dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "test: define admin analytics frontend contract"
```

---

### Task 2: Add Analytics as the first and default accessible Admin tab

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/admin-tabs.js`
- Modify: `dashboard/backend/tests/test_admin_credits_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: Existing `[data-admin-tab]`, `[data-admin-panel]`, `adminTab` URL behavior, and the `grant-pool -> users` legacy alias.
- Produces: `AdminTabs.setTab(value, options) -> string`, `AdminTabs.onEnter()`, `AdminTabs.openAccountManagement({userId, email})`, and the `admin:tabchange` document event with `{detail: {tab}}`.

- [ ] **Step 1: Strengthen the failing tab contract**

Add assertions for exactly four tabs in this order:

```python
expected = ["analytics", "users", "providers", "activity"]
positions = [nav_markup.index(f'data-admin-tab="{value}"') for value in expected]
assert positions == sorted(positions)
assert "DEFAULT_TAB = 'analytics'" in ADMIN_TABS_JS
assert "ArrowRight" in ADMIN_TABS_JS and "ArrowLeft" in ADMIN_TABS_JS
assert "Home" in ADMIN_TABS_JS and "End" in ADMIN_TABS_JS
assert "admin:tabchange" in ADMIN_TABS_JS
assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS
```

Update the older Grant Credits assertions from three tabs to four without weakening their checks that Grant Pool is not a tab and Users precedes Providers and Activity.

- [ ] **Step 2: Add the Analytics tab and panel shell**

Change the Admin page description to `Read platform analytics and manage Grant Credits, accounts, approved providers, and the audit trail.`

Insert before Users:

```html
<button id="adminTabAnalytics" class="admin-tab is-active" type="button"
        role="tab" aria-selected="true" aria-controls="adminPanelAnalytics"
        tabindex="0" data-admin-tab="analytics">Analytics</button>
```

Remove the active state from Users, set `aria-selected="false"` and `tabindex="-1"` on Users, Providers, and Activity, and add before `adminPanelUsers`:

```html
<section id="adminPanelAnalytics" class="admin-tab-panel" role="tabpanel"
         aria-labelledby="adminTabAnalytics" data-admin-panel="analytics">
  <div id="adminAnalyticsOverview" class="admin-analytics-overview"></div>
  <div id="adminAnalyticsProfile" class="admin-analytics-profile" hidden></div>
</section>
```

Set the Users panel `hidden` initially. The empty containers are temporary and are replaced by semantic markup in Task 3.

- [ ] **Step 3: Update tab state and keyboard behavior**

Change `admin-tabs.js` to use:

```javascript
const DEFAULT_TAB = 'analytics';
const ALLOWED_TABS = new Set(['analytics', 'users', 'providers', 'activity']);

function normalizeTab(value) {
  const normalized = value === 'grant-pool' ? 'users' : value;
  return ALLOWED_TABS.has(normalized) ? normalized : DEFAULT_TAB;
}
```

Have `setTab()` return the normalized tab and dispatch after DOM and URL state are synchronized:

```javascript
document.dispatchEvent(new CustomEvent('admin:tabchange', { detail: { tab } }));
return tab;
```

Replace the current two-key condition with an ordered key map:

```javascript
const keys = new Set(['ArrowRight', 'ArrowLeft', 'Home', 'End']);
if (!keys.has(event.key)) return;
event.preventDefault();
const buttons = [...document.querySelectorAll('[data-admin-tab]')];
const index = buttons.indexOf(button);
const next = event.key === 'Home'
  ? buttons[0]
  : event.key === 'End'
    ? buttons[buttons.length - 1]
    : event.key === 'ArrowRight'
      ? buttons[(index + 1) % buttons.length]
      : buttons[(index - 1 + buttons.length) % buttons.length];
next.focus();
setTab(next.dataset.adminTab);
```

- [ ] **Step 4: Add the account-management bridge**

Add:

```javascript
function openAccountManagement({ userId, email } = {}) {
  setTab('users');
  const url = new URL(window.location.href);
  url.searchParams.delete('analyticsUser');
  url.searchParams.delete('analyticsSection');
  window.history.replaceState(window.history.state, '', url);
  const input = document.getElementById('adminCreditsUserQuery');
  const form = document.getElementById('adminCreditsUserSearch');
  if (input) input.value = String(email || userId || '');
  form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  input?.focus();
}
```

Export it as `window.AdminTabs = { onEnter, openAccountManagement, setTab };`. This bridge searches the existing account-management table and does not duplicate or move any mutation control.

- [ ] **Step 5: Run focused tab contracts**

```bash
pytest -q \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: four-tab, default, alias, ARIA, keyboard, and account-management bridge assertions pass; Analytics content/module assertions remain red.

- [ ] **Step 6: Commit the Admin navigation change**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-tabs.js \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add default admin analytics tab"
```

---

### Task 3: Build the Analytics shell, filters, API client, and URL state

**Files:**
- Modify: `dashboard/frontend/app.html`
- Create: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `window.API.request(path, options)`, `window.getStoredAuthUser()`, `window.AdminTabs`, PR 2 overview/user endpoints, and the global `admin:tabchange` event.
- Produces: `window.AdminAnalytics = { onEnter, refresh, syncAuth }`; URL keys `analyticsStart`, `analyticsEnd`, `analyticsBilling`, `analyticsProvider`, `analyticsModel`, `analyticsInternal`, `analyticsUser`, and `analyticsSection`.

- [ ] **Step 1: Add failing shell and URL-state assertions**

Assert that the markup contains labels and controls for the date range, billing mode, provider, model, internal-account toggle, manual refresh, and Last updated. Assert the source owns the exact URL keys above, uses `URLSearchParams`, calls `history.replaceState`, sends `include_internal`, and does not use `localStorage`.

- [ ] **Step 2: Replace the temporary Analytics containers with semantic overview/profile shells**

Build the overview in this vertical order:

```html
<div id="adminAnalyticsOverview" class="admin-analytics-overview">
  <header class="admin-analytics-heading">
    <div>
      <p class="credits-section-kicker">Platform intelligence</p>
      <h3>Analytics overview</h3>
      <p>Activation, usage, model execution, cost, and product friction.</p>
    </div>
    <div class="admin-analytics-update">
      <button id="adminAnalyticsRefreshBtn" class="auth-btn auth-btn-secondary" type="button">Refresh analytics</button>
      <p>Last updated <time id="adminAnalyticsLastUpdated">—</time></p>
    </div>
  </header>
  <form id="adminAnalyticsFilters" class="admin-analytics-filters" aria-label="Analytics filters">
    <label>Start date<input id="adminAnalyticsStart" type="date" required></label>
    <label>End date<input id="adminAnalyticsEnd" type="date" required></label>
    <label>Billing mode<select id="adminAnalyticsBilling"><option value="all">All billing modes</option></select></label>
    <label>Provider<select id="adminAnalyticsProvider"><option value="">All providers</option></select></label>
    <label>Model<select id="adminAnalyticsModel"><option value="">All models</option></select></label>
    <label class="admin-analytics-check"><input id="adminAnalyticsInternal" type="checkbox"> Include internal accounts</label>
    <button class="auth-btn auth-btn-primary" type="submit">Apply filters</button>
  </form>
  <section id="adminAnalyticsSnapshot" data-analytics-panel="snapshot" aria-labelledby="adminAnalyticsSnapshotHeading"></section>
  <section id="adminAnalyticsEngagement" data-analytics-panel="engagement" aria-labelledby="adminAnalyticsEngagementHeading"></section>
  <section id="adminAnalyticsHealth" data-analytics-panel="health" aria-labelledby="adminAnalyticsHealthHeading"></section>
  <section id="adminAnalyticsAttention" data-analytics-panel="attention" aria-labelledby="adminAnalyticsAttentionHeading"></section>
</div>
<article id="adminAnalyticsProfile" class="admin-analytics-profile" aria-labelledby="adminAnalyticsProfileTitle" hidden></article>
```

Use complete headings, loading paragraphs with `role="status"`, error paragraphs with `role="alert"`, chart/table containers, and table headers in the final markup; do not leave the section bodies empty.

- [ ] **Step 3: Implement the module state and safe request wrapper**

Start `admin-analytics.js` as an IIFE with one state object:

```javascript
const state = {
  initialized: false,
  active: false,
  overviewRequestSeq: 0,
  usersRequestSeq: 0,
  userRequestSeq: 0,
  filters: null,
  filterOptionsLoaded: false,
  attention: { items: [], nextCursor: null, history: [], total: 0 },
  profile: { userId: null, detail: null, section: 'overview', sections: {} },
  trendChart: null,
};

function request(path) {
  if (!window.API || typeof window.API.request !== 'function') {
    return Promise.reject(new Error('Admin Analytics API is not ready yet.'));
  }
  return window.API.request(path, { method: 'GET' });
}
```

Use monotonically increasing request sequence numbers for overview, attention, and base profile requests so stale filter/profile responses cannot repaint the current view.

- [ ] **Step 4: Implement deterministic default filters and validation**

Use a 30-day inclusive UTC window ending today:

```javascript
function defaultDateRange(now = new Date()) {
  const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const start = new Date(end);
  start.setUTCDate(start.getUTCDate() - 29);
  return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
}
```

Reject empty dates and `start > end` in the UI before making a request. Read URL values only when they match `^\d{4}-\d{2}-\d{2}$`, an allowed billing mode, and the checkbox literal `true`; otherwise use defaults.

- [ ] **Step 5: Serialize filters and profile state without losing Admin navigation**

Implement:

```javascript
function replaceAnalyticsUrl({ userId = state.profile.userId, section = state.profile.section } = {}) {
  const url = new URL(window.location.href);
  url.searchParams.set('adminTab', 'analytics');
  url.searchParams.set('analyticsStart', state.filters.start);
  url.searchParams.set('analyticsEnd', state.filters.end);
  url.searchParams.set('analyticsBilling', state.filters.billingMode);
  setOptionalParam(url, 'analyticsProvider', state.filters.providerId);
  setOptionalParam(url, 'analyticsModel', state.filters.modelId);
  if (state.filters.includeInternal) url.searchParams.set('analyticsInternal', 'true');
  else url.searchParams.delete('analyticsInternal');
  setOptionalParam(url, 'analyticsUser', userId);
  if (userId) url.searchParams.set('analyticsSection', section || 'overview');
  else url.searchParams.delete('analyticsSection');
  window.history.replaceState(window.history.state, '', url);
}

function setOptionalParam(url, key, value) {
  if (value == null || value === '') url.searchParams.delete(key);
  else url.searchParams.set(key, String(value));
}
```

URL state contains only IDs, enum values, and dates—never email, display name, reasons, evidence IDs, or activity data.

Add a shared access-loss handler inside the module so 401/403 responses do not leave a stale Admin page visible:

```javascript
async function handleAccessLost(error) {
  if (error?.status !== 401 && error?.status !== 403) return false;
  if (typeof refreshAuthUser === 'function') await refreshAuthUser();
  if (typeof navigateToPage === 'function') navigateToPage('home');
  return true;
}
```

`syncAuth(user)` clears all in-memory Analytics response data when `user?.role !== 'admin'`; it never persists profile or activity data in browser storage.

- [ ] **Step 6: Build exact API query strings**

Overview query construction must set `start_date`, `end_date`, `billing_mode`, and `include_internal`; add `provider_id` and `model_id` only when non-empty. Attention query construction additionally sets `status`, `sort`, `last_activity_from`, `last_activity_to`, `limit=25`, and an opaque `cursor` only when paging forward.

Encode user IDs with `encodeURIComponent(String(userId))`. Encode cursors through `URLSearchParams`; never parse, inspect, or manufacture them.

- [ ] **Step 7: Populate filter options from PR 2 without overwriting a valid selection**

Create `<option>` nodes with `textContent`; preserve the current provider/model selection if it is present in `filter_options`, otherwise reset only that unavailable facet to All. Billing options remain the fixed `all`, `byok`, and `platform_credits` values even if the result window has no rows for one lane.

- [ ] **Step 8: Wire Admin lifecycle and cache-busted script loading**

Load `js/admin-analytics.js?v=1` after `app.js` and before `admin-tabs.js`. Because this task changes `app.js`, increment its existing cache key from `app.js?v=115` to `app.js?v=116`. In `app.js`:

```javascript
if (window.AdminAnalytics) window.AdminAnalytics.syncAuth(user);
```

Add `window.AdminAnalytics.onEnter()` to the Admin entry branch. Add `window.AdminAnalytics.refresh()` to `adminRefreshBtn` while retaining every existing Admin refresh call; `refresh()` returns immediately when Analytics is not the active Admin tab. In the Analytics module, listen for `admin:tabchange` and fetch only when `detail.tab === 'analytics'`. Bind one `popstate` listener during initialization; when Analytics is active it calls `restoreAnalyticsUrlState()`, which re-reads filters, `analyticsUser`, and `analyticsSection` before selecting the visible overview/profile state.

- [ ] **Step 9: Run the shell/client contracts**

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: shell, filter, URL-key, endpoint, lifecycle, safe-DOM, and script-order tests pass; overview and profile rendering tests added in later tasks remain red.

- [ ] **Step 10: Commit the Analytics client shell**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js \
  dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add admin analytics client shell"
```

---

### Task 4: Render Overview metrics, engagement, health, and partial errors

**Files:**
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `overview.panels.snapshot`, `overview.panels.engagement`, `overview.panels.health`, and `overview.last_updated_at` from Task 1.
- Produces: `loadOverview()`, `renderSnapshot(data)`, `renderEngagement(data)`, `renderHealth(data)`, and `renderPanelEnvelope(panelName, envelope, renderData)`.

- [ ] **Step 1: Add failing overview-rendering contracts**

Assert the source references every required metric and collection field, the exact partial-error copy, `Promise.allSettled`, `aria-busy`, a Chart.js destroy-before-create guard, and an accessible trend-data table. Assert no overview catch handler hides `adminPanelAnalytics` or clears unrelated panels.

- [ ] **Step 2: Implement shared panel state without exposing backend errors**

Use:

```javascript
const TEMPORARY_UNAVAILABLE = 'This metric is temporarily unavailable.';

function renderPanelEnvelope(panelName, envelope, renderData) {
  const panel = document.querySelector(`[data-analytics-panel="${panelName}"]`);
  if (!panel) return;
  panel.setAttribute('aria-busy', 'false');
  const error = panel.querySelector('[data-panel-error]');
  if (!envelope || envelope.error || !envelope.data) {
    if (error) {
      error.textContent = TEMPORARY_UNAVAILABLE;
      error.hidden = false;
    }
    return;
  }
  if (error) error.hidden = true;
  renderData(envelope.data);
}
```

Before each request set only the three Overview panels to `aria-busy="true"`; do not clear their previous successful content while refreshing.

- [ ] **Step 3: Render the four snapshot cards**

Render:

- Active users, rolling 7 days as an integer.
- First successful run conversion as `formatPercent(value)`.
- Backtest success rate as `formatPercent(value)`.
- Platform model cost as `formatMoney(amount, currency)` with the visible label `ATL platform model cost`.

`formatPercent()` accepts only finite values in `[0, 1]`; invalid values render `—`. `formatMoney()` uses `Intl.NumberFormat` and does not coerce missing values to zero.

- [ ] **Step 4: Render engagement trend accessibly**

Destroy `state.trendChart` before creating a replacement. Create a two-series Chart.js line chart for daily active users and completed runs. Give the canvas `role="img"` and a concise range-aware `aria-label`, and populate a sibling visually-hidden table with Date, Active users, and Completed runs so the values are not canvas-only.

If `window.Chart` is unavailable, keep the table available and replace the canvas region with `Trend chart is unavailable; values are listed in the table.` without failing the funnel.

- [ ] **Step 5: Render the activation funnel**

Create one ordered-list item per `activation_funnel` row. Each item shows the label, user count, and conversion percentage, and sets a CSS custom property only after clamping the finite conversion rate to `[0, 1]`:

```javascript
row.style.setProperty('--analytics-progress', `${Math.round(rate * 100)}%`);
```

Do not infer missing funnel stages or recalculate PR 2 metrics in the browser.

- [ ] **Step 6: Render health states and friction**

Render all five statuses in the backend-provided precedence order. Use stable `data-status` values for CSS but backend-provided labels for visible text. Render friction rows with safe label, affected-user count, and failure count; an empty list reads `No actionable failure categories in this period.`

- [ ] **Step 7: Load independent overview and attention requests**

Have `refresh()` call:

```javascript
await Promise.allSettled([
  loadOverview(),
  loadAttention({ reset: true }),
]);
```

`loadOverview()` catches only its own network failure and marks snapshot, engagement, and health unavailable. `loadAttention()` owns only the attention panel. A 401/403 from either routes through one access-loss handler; other errors remain panel-local.

- [ ] **Step 8: Run overview and fixture contracts**

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: metric-field, Chart.js fallback, accessible table, five-state, friction, partial-error, and independent-request assertions pass.

- [ ] **Step 9: Commit the Overview renderers**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: render admin analytics overview"
```

---

### Task 5: Add the filterable and pageable Users Needing Attention table

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `GET /api/admin/analytics/users`, current global filters, and opaque `next_cursor`.
- Produces: `loadAttention({reset, direction})`, attention query/status/sort controls, cursor-history pagination, and profile-open actions.

- [ ] **Step 1: Add failing attention-table contracts**

Assert the markup has a search input, status select with all five states, sort select, Previous/Next buttons, live result range, table headers for Account, State/reason, Last meaningful activity, Recent runs, Failures, and Profile, plus a button carrying only `data-analytics-user-id`.

Assert the JavaScript uses `next_cursor`, retains a cursor-history stack for Previous, resets cursors when filters change, and opens profiles without placing email in the URL.

- [ ] **Step 2: Add semantic controls and table markup**

Use a separate form inside `adminAnalyticsAttention`, with:

```html
<input id="adminAnalyticsUserQuery" type="search" maxlength="120"
       autocomplete="off" placeholder="Search name or email">
<select id="adminAnalyticsUserStatus">
  <option value="all">All attention states</option>
  <option value="blocked">Blocked</option>
  <option value="needs_attention">Needs Attention</option>
  <option value="dormant">Dormant</option>
  <option value="onboarding">Onboarding</option>
  <option value="active">Active</option>
</select>
<select id="adminAnalyticsUserSort">
  <option value="attention">Most actionable</option>
  <option value="last_activity_desc">Latest activity</option>
  <option value="failures_desc">Most failures</option>
  <option value="runs_desc">Most runs</option>
</select>
```

The table body starts with a single `Loading users…` cell and the pager has `aria-label="Users needing attention pages"`.

- [ ] **Step 3: Implement opaque forward/back cursor state**

For the first page, clear `history` and omit `cursor`. Before moving Next, push the cursor used for the current page to `history`; for Previous, pop the prior cursor and request it. Disable Previous when history is empty and Next when `next_cursor` is null.

Never derive a cursor from a user ID or row count. On a filter/search/sort change, call `loadAttention({ reset: true })`.

- [ ] **Step 4: Render safe table rows**

Create cells and buttons with DOM APIs and `textContent`. The state cell shows a state badge and the full human-readable reason. The account cell shows display name, email, and `User #<id>`. The profile button is:

```javascript
button.type = 'button';
button.className = 'credits-key-action';
button.dataset.analyticsUserId = String(user.user_id);
button.textContent = 'View analytics';
button.setAttribute('aria-label', `View analytics for ${displayIdentity(user)}`);
```

Define the label helper next to the renderer:

```javascript
function displayIdentity(user) {
  return String(user?.display_name || user?.email || `User #${user?.user_id ?? 'unknown'}`);
}
```

Use one delegated click listener on the table body to call `openProfile(userId)`.

- [ ] **Step 5: Keep failures local and existing rows stable**

On the first-page failure, show `This metric is temporarily unavailable.` in the attention error region. On a later-page failure, retain the current rows and pager, announce `The next user page is temporarily unavailable.`, and roll back the cursor-history mutation.

- [ ] **Step 6: Run attention contracts**

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: search/status/sort fields, exact query names, cursor history, row safety, local error, and profile-open tests pass.

- [ ] **Step 7: Commit the attention queue**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add analytics attention queue"
```

---

### Task 6: Build the dedicated User Analytics Profile and Overview

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `GET /api/admin/analytics/users/{user_id}`, `AdminTabs.openAccountManagement()`, `analyticsUser`, and `analyticsSection` URL state.
- Produces: `openProfile(userId)`, `closeProfile()`, `loadProfile(userId)`, stable two-column profile markup, and the profile Overview renderer.

- [ ] **Step 1: Add failing profile contracts**

Assert the profile has a Back to overview button, left account summary, state/reason, billing/provider/region/device/browser fields, an `Open account management` anchor, right-side section tabs, an Overview panel, and URL restoration for `analyticsUser` and `analyticsSection`. Assert the base-detail request fires once per explicit profile open and no separate access-log route is called.

- [ ] **Step 2: Add the stable profile skeleton**

Inside `adminAnalyticsProfile`, add:

```html
<button id="adminAnalyticsProfileBack" class="credits-key-action" type="button">Back to analytics overview</button>
<div class="admin-analytics-profile-layout">
  <aside class="admin-analytics-profile-summary" aria-labelledby="adminAnalyticsProfileTitle">
    <p class="credits-section-kicker">User analytics profile</p>
    <h3 id="adminAnalyticsProfileTitle" tabindex="-1">Loading user analytics</h3>
    <p id="adminAnalyticsProfileEmail">—</p>
    <dl class="admin-analytics-profile-facts">
      <div><dt>User ID</dt><dd id="adminAnalyticsProfileUserId">—</dd></div>
      <div><dt>Joined</dt><dd id="adminAnalyticsProfileJoined">—</dd></div>
      <div><dt>Last meaningful activity</dt><dd id="adminAnalyticsProfileLastActivity">—</dd></div>
      <div><dt>State</dt><dd id="adminAnalyticsProfileState">—</dd></div>
      <div><dt>Reason</dt><dd id="adminAnalyticsProfileReason">—</dd></div>
      <div><dt>Primary billing lane</dt><dd id="adminAnalyticsProfileBilling">—</dd></div>
      <div><dt>Default provider</dt><dd id="adminAnalyticsProfileProvider">—</dd></div>
      <div><dt>Region</dt><dd id="adminAnalyticsProfileRegion">Unknown</dd></div>
      <div><dt>Device</dt><dd id="adminAnalyticsProfileDevice">Unknown</dd></div>
      <div><dt>Browser</dt><dd id="adminAnalyticsProfileBrowser">Unknown</dd></div>
    </dl>
    <details id="adminAnalyticsProfileEvidence"><summary>Evidence event IDs</summary><ul></ul></details>
    <a id="adminAnalyticsOpenAccount" href="/app?view=admin&amp;adminTab=users">Open account management</a>
  </aside>
  <div class="admin-analytics-profile-content">
    <nav id="adminAnalyticsProfileTabs" role="tablist" aria-label="User analytics sections">
      <button id="adminAnalyticsProfileTabOverview" role="tab" aria-selected="true" aria-controls="adminAnalyticsSectionOverview" tabindex="0" data-analytics-section-tab="overview" type="button">Overview</button>
      <button id="adminAnalyticsProfileTabTimeline" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionTimeline" tabindex="-1" data-analytics-section-tab="timeline" type="button">Timeline</button>
      <button id="adminAnalyticsProfileTabRuns" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionRuns" tabindex="-1" data-analytics-section-tab="runs" type="button">Runs</button>
      <button id="adminAnalyticsProfileTabUsage" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionUsage" tabindex="-1" data-analytics-section-tab="usage" type="button">Usage</button>
      <button id="adminAnalyticsProfileTabSessions" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionSessions" tabindex="-1" data-analytics-section-tab="sessions" type="button">Sessions</button>
    </nav>
    <section id="adminAnalyticsSectionOverview" data-analytics-section-panel="overview" role="tabpanel" aria-labelledby="adminAnalyticsProfileTabOverview">
      <p data-section-status role="status" aria-live="polite">Loading profile overview…</p>
      <p data-section-error role="alert" hidden></p>
      <div id="adminAnalyticsProfileOverviewContent"></div>
    </section>
    <section id="adminAnalyticsSectionTimeline" data-analytics-section-panel="timeline" role="tabpanel" aria-labelledby="adminAnalyticsProfileTabTimeline" hidden>
      <p data-section-status role="status" aria-live="polite"></p><p data-section-error role="alert" hidden></p>
      <div data-section-items></div><button data-section-more type="button" hidden>Load more timeline</button>
    </section>
    <section id="adminAnalyticsSectionRuns" data-analytics-section-panel="runs" role="tabpanel" aria-labelledby="adminAnalyticsProfileTabRuns" hidden>
      <p data-section-status role="status" aria-live="polite"></p><p data-section-error role="alert" hidden></p>
      <div data-section-items></div><button data-section-more type="button" hidden>Load more runs</button>
    </section>
    <section id="adminAnalyticsSectionUsage" data-analytics-section-panel="usage" role="tabpanel" aria-labelledby="adminAnalyticsProfileTabUsage" hidden>
      <p data-section-status role="status" aria-live="polite"></p><p data-section-error role="alert" hidden></p>
      <div data-section-items></div><button data-section-more type="button" hidden>Load more usage</button>
    </section>
    <section id="adminAnalyticsSectionSessions" data-analytics-section-panel="sessions" role="tabpanel" aria-labelledby="adminAnalyticsProfileTabSessions" hidden>
      <p data-section-status role="status" aria-live="polite"></p><p data-section-error role="alert" hidden></p>
      <div data-section-items></div><button data-section-more type="button" hidden>Load more sessions</button>
    </section>
  </div>
</div>
```

Give every tab a stable ID, `aria-controls`, `aria-selected`, and roving `tabindex`. Each non-Overview panel contains its own status, error, items container, and Load more button.

- [ ] **Step 3: Implement profile open, close, and stale-response protection**

`openProfile(userId)` validates a non-empty scalar ID, increments `userRequestSeq`, clears prior section caches, shows the profile, hides the overview, focuses the profile heading, sets `analyticsUser` and `analyticsSection=overview`, then requests the base detail.

`closeProfile()` increments the request sequence, destroys section state, hides the profile, shows the overview, removes `analyticsUser` and `analyticsSection`, and focuses the attention heading.

On initial module entry, a valid `analyticsUser` URL value opens that profile directly. The Task 3 `popstate` listener and Admin tab re-entry both call `restoreAnalyticsUrlState()` instead of keeping a stale in-memory profile selected.

- [ ] **Step 4: Render the left account summary**

Render display name, email, user ID, join date, last meaningful activity, current state badge, human-readable reason, primary billing lane, default provider, country/region, device category, and browser family. Use `Unknown` for optional coarse environment fields and `—` for absent billing/provider values.

Evidence event IDs support the explanation but are not useful account data; show them only in a collapsed `<details>` element labeled `Evidence event IDs`, using text nodes.

- [ ] **Step 5: Wire Open account management**

Keep a real anchor so the destination remains discoverable and works without the click handler. Intercept an ordinary unmodified click for the in-app handoff:

```javascript
accountLink.addEventListener('click', (event) => {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  const user = state.profile.detail?.user;
  if (!user) return;
  closeProfile();
  window.AdminTabs?.openAccountManagement({
    userId: user.user_id,
    email: user.email,
  });
});
```

The Analytics module closes its profile state before the handoff. The Admin Tabs bridge removes profile-only URL keys and searches the existing Users account-management table; PR 3 adds no mutation action to Analytics.

- [ ] **Step 6: Render profile Overview summaries**

Render activation milestones in chronological order, recent footprint events, run totals by outcome, billing-lane mix, input/output/total tokens, ATL model cost, and top product page. Do not calculate authoritative totals from activity pages; use only the base detail `summary` response.

Label BYOK explicitly as `BYOK usage — no ATL Credits debit`. Label platform cost explicitly as `ATL platform model cost`.

- [ ] **Step 7: Keep profile failures isolated**

A base-detail failure leaves the profile shell and Back button available, shows `User analytics are temporarily unavailable.`, and does not hide the global Analytics overview permanently. A 401/403 calls the shared Admin access-loss handler and returns Home.

- [ ] **Step 8: Run profile contracts**

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: dedicated-view, two-column markup, detail endpoint, URL state, summary fields, safe evidence, Back, and account-management tests pass.

- [ ] **Step 9: Commit the base profile**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add user analytics profile"
```

---

### Task 7: Add independently pageable Timeline, Runs, Usage, and Sessions

**Files:**
- Modify: `dashboard/frontend/js/admin-analytics.js`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: `GET /api/admin/analytics/users/{user_id}/activity` with `section`, `limit=50`, and optional `cursor`, plus the four activity fixtures.
- Produces: `selectProfileSection(section)`, `loadProfileSection(section, {append})`, one independent `{items, nextCursor, loading, loaded, error, requestSeq}` state per detailed section, and keyboard-operable profile tabs.

- [ ] **Step 1: Add failing pagination and keyboard contracts**

Assert all five section tabs are present in order, Timeline/Runs/Usage/Sessions use the activity endpoint with exact `section` values, each has a separate `nextCursor` and request sequence, cursors are appended only for Load more, and tab keys support ArrowLeft, ArrowRight, Home, and End with `preventDefault()`.

- [ ] **Step 2: Initialize independent section state**

Use:

```javascript
const PROFILE_SECTIONS = ['overview', 'timeline', 'runs', 'usage', 'sessions'];

function emptySectionState() {
  return { items: [], nextCursor: null, loading: false, loaded: false, error: null, requestSeq: 0 };
}

function resetProfileSections() {
  state.profile.sections = {
    timeline: emptySectionState(),
    runs: emptySectionState(),
    usage: emptySectionState(),
    sessions: emptySectionState(),
  };
}
```

Do not share a cursor or item array between sections.

- [ ] **Step 3: Implement section selection and URL synchronization**

`selectProfileSection(section)` normalizes unknown values to `overview`, updates ARIA/hidden state, focuses only when invoked by keyboard/click rather than initial restoration, writes `analyticsSection`, and lazy-loads a detailed section only when `loaded === false`.

Use the same roving-tabindex keyboard algorithm as Task 2 over `PROFILE_SECTIONS`.

- [ ] **Step 4: Implement append-only cursor loading**

For a first request, omit the cursor and replace the empty item list. For Load more, require a non-null `nextCursor`, pass it unchanged through `URLSearchParams`, append returned items, and replace `nextCursor` with the response value.

Capture both the current profile user ID and section request sequence. Ignore a response when either no longer matches, preventing a slow Ada Timeline response from painting into Grace's profile or the Runs panel.

- [ ] **Step 5: Render Timeline safely**

Render a chronological list with `<time>`, safe label, group/outcome, provider, model, billing lane, and safe error-category label. Do not show arbitrary `properties`, correlation IDs, source record IDs, or raw error text even if an unexpected backend response contains them.

- [ ] **Step 6: Render Runs safely**

Render a table with requested/completed time, outcome, provider/model, billing lane, duration, and safe error category. `completed_at: null` reads `In progress`; `duration_ms` is formatted into bounded seconds/minutes without recalculating run outcomes.

- [ ] **Step 7: Render Usage with billing-lane truthfulness**

Render input, output, and total tokens, ATL model cost, and ATL Credits debited. For BYOK rows, show `—` in ATL cost/debit cells when both values are zero and add `BYOK — no ATL charge` to the billing cell. Do not sum usage pages into the Overview cards.

- [ ] **Step 8: Render Sessions without raw identifiers**

Render start, last activity, duration, page views, top page, coarse region, device category, and browser family. The renderer and fixture must not reference `session_id` or `network_hash`.

- [ ] **Step 9: Keep page errors section-local**

If a first section request fails, show `This section is temporarily unavailable.` only in that section. If Load more fails, retain existing items and show `More activity is temporarily unavailable.` The other section caches, profile summary, and Back/account-management actions remain usable.

- [ ] **Step 10: Run section contracts**

```bash
pytest -q dashboard/backend/tests/test_admin_analytics_frontend.py
```

Expected: section URL, independent cursor, lazy-load, stale-response, field, BYOK-zero, no-session-identifier, local-error, and profile-tab keyboard assertions pass.

- [ ] **Step 11: Commit detailed profile sections**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: add paged user analytics activity"
```

---

### Task 8: Style the vertical overview and responsive profile accessibly

**Files:**
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: The scoped `.admin-analytics-*` markup from Tasks 3–7 and existing CSS variables.
- Produces: A clear vertical overview, stable desktop profile columns, stacked mobile profile, scroll-safe tables, visible focus, status differentiation that does not rely on color alone, and reduced-motion behavior.

- [ ] **Step 1: Add failing visual-source contracts**

Assert CSS contains scoped rules for the overview, filters, snapshot cards, trend/funnel, state badges, friction, attention table, profile layout, profile tabs, activity tables, `:focus-visible`, `@media (max-width: 900px)`, `@media (max-width: 600px)`, and `@media (prefers-reduced-motion: reduce)`.

- [ ] **Step 2: Set the vertical hierarchy and readable width**

Keep `.admin-view` at its existing maximum width. Use a single-column `.admin-analytics-overview` with 24–32px section gaps. Use four equal snapshot cards at wide widths, two columns below 900px, and one column below 600px.

- [ ] **Step 3: Style filters and tables without clipping controls**

Use a responsive CSS grid with `minmax(150px, 1fr)` filter cells, aligned labels, 44px minimum interactive height, and full-width Apply on small screens. Wrap attention and activity tables in `overflow-x: auto`; do not hide columns with CSS because their values are required administrative evidence.

- [ ] **Step 4: Style state and friction semantics**

Give each state a shared badge shape plus distinct text label and border treatment. Color may reinforce the state but may not be its only signal. Keep human-readable reasons visible rather than tooltip-only.

- [ ] **Step 5: Create the stable profile layout**

Use:

```css
.admin-analytics-profile-layout {
  display: grid;
  grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
  gap: 24px;
  align-items: start;
}
```

Make the summary column sticky only above 900px and only beneath the app header. At 900px and below, switch to one column and remove sticky positioning.

- [ ] **Step 6: Add focus and reduced-motion rules**

Use a clear `outline` plus `outline-offset` for Analytics buttons/tabs/links. Disable chart/funnel transition effects under reduced motion. Do not remove focus outlines globally.

- [ ] **Step 7: Update asset cache versions**

Change `styles.css?v=124` to `styles.css?v=125`. Keep the Task 3 `app.js?v=116` and `js/admin-analytics.js?v=1` values. Change `js/admin-tabs.js?v=2` to `js/admin-tabs.js?v=3` because its behavior changed. Update static tests to pin all four values.

- [ ] **Step 8: Run focused frontend contracts**

```bash
pytest -q \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_console_frontend.py \
  dashboard/backend/tests/test_admin_model_providers_frontend.py
```

Expected: all pass.

- [ ] **Step 9: Commit the Analytics presentation**

```bash
git add dashboard/frontend/app.html dashboard/frontend/styles.css \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_credits_frontend.py
git commit -m "style: finish admin analytics layouts"
```

---

### Task 9: Verify privacy, partial failures, navigation, and PR-scope integrity

**Files:**
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`
- Verify: `dashboard/frontend/app.html`
- Verify: `dashboard/frontend/app.js`
- Verify: `dashboard/frontend/js/admin-tabs.js`
- Verify: `dashboard/frontend/js/admin-analytics.js`
- Verify: `dashboard/frontend/styles.css`
- Verify: `dashboard/backend/tests/fixtures/admin_analytics/*.json`

**Interfaces:**
- Consumes: All PR 3 UI behavior and the rebased PR 2 API contract.
- Produces: A fully green focused suite, documented synthetic acceptance checks, and a diff containing no PR 1/PR 2 implementation changes or sensitive artifacts.

- [ ] **Step 1: Complete the privacy/read-only contract**

Add tests that recursively scan fixture keys, scan new Analytics source for forbidden field access, and assert the Analytics markup contains no form with a mutating method and no controls labeled Save, Grant, Assign, Reclaim, Revoke, Delete, Suspend, Role, or API key.

Permit only the filter form and attention-search form, both handled as GET-style client filtering with `event.preventDefault()`.

- [ ] **Step 2: Prove partial failure behavior from source boundaries**

Assert `loadOverview()` and `loadAttention()` are separate functions called with `Promise.allSettled`, `renderPanelEnvelope()` maps backend errors to the exact approved copy, and detailed section errors do not assign to the Overview or profile-root `hidden` properties.

- [ ] **Step 3: Prove URL and keyboard coverage**

Assert Admin tab order/default/alias, profile tab order, ArrowLeft/ArrowRight/Home/End handling, `preventDefault()`, `aria-selected`, roving `tabindex`, `aria-controls`, URL keys, and removal of `analyticsUser`/`analyticsSection` on Back and account-management handoff.

- [ ] **Step 4: Run the full relevant automated suite**

```bash
pytest -q \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_credits_frontend.py \
  dashboard/backend/tests/test_admin_console_frontend.py \
  dashboard/backend/tests/test_admin_model_providers_frontend.py \
  dashboard/backend/tests/test_frontend_xss_guards.py \
  dashboard/backend/tests/test_static_routes.py \
  dashboard/backend/tests/test_app_composition.py
```

Expected: all pass without a real credential or external provider.

- [ ] **Step 5: Run the synthetic end-to-end acceptance against PR 2**

Use PR 2's synthetic analytics test setup to load the same safe scenario represented by the fixtures: signup, major page visits, fake BYOK credential verification, failed then successful run, and one Platform Credits run with usage evidence. In the browser verify:

1. `?view=admin` opens `adminTab=analytics`, and four Admin tabs cycle with arrow/Home/End keys.
2. Filters change only documented query parameters; Refresh updates Last updated; a forced engagement-panel error leaves snapshot, health, and attention usable.
3. The attention row opens the dedicated profile; Overview values match the synthetic source; Timeline/Runs/Usage/Sessions page independently.
4. BYOK rows display tokens but no ATL cost or Credits debit; Platform Credits rows display the synthetic authoritative values.
5. Open account management switches to Users and searches the same synthetic email; no mutation control exists on Analytics.

Do not enter a real credential or connect to a real provider during this check.

- [ ] **Step 6: Inspect scope and secret hygiene**

Run:

```bash
git status --short
git diff --check
git diff --name-only feature/admin-user-analytics-metrics...HEAD
git grep -n -I -E '(sk-[A-Za-z0-9]|AKIA[0-9A-Z]{16}|postgres(ql)?://[^ ]+:[^ ]+@)' -- \
  dashboard/frontend dashboard/backend/tests/fixtures/admin_analytics
```

Expected changed paths are limited to the PR 3 frontend files, frontend tests/fixtures, and this plan. The secret-pattern scan prints nothing. No database, `.superpowers/`, or `work/` path is staged.

- [ ] **Step 7: Commit final contract hardening if Step 1–3 changed tests**

```bash
git add dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "test: harden admin analytics UI boundaries"
```

- [ ] **Step 8: Stop for review before opening or merging PR 3**

Provide the commit list, focused test output, synthetic acceptance results, and the exact PR 2 commit used for the rebase. Do not alter PR 1 or PR 2 code to resolve review feedback unless the user explicitly moves that work into the corresponding PR conversation.
