# Admin Analytics Demo Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Admin Analytics overview with the approved `analytics-layout-demo.html` hierarchy and visual treatment while preserving real data, safe rendering, accessible navigation, partial failures, and the dedicated User Analytics Profile.

**Architecture:** Keep the existing FastAPI analytics endpoints and vanilla JavaScript controllers. Extend the value-user list response with one selected-range run count needed by the Action queue, then reshape `app.html`, `admin-analytics-value.js`, and scoped CSS around the approved five-section story. Keep `admin-analytics.js` responsible for the dedicated profile and activity tabs; the overview controller only owns aggregate panels and drill-down entry points.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL analytics repositories, server-rendered HTML, vanilla JavaScript, Chart.js, CSS, pytest source-contract tests, safe JSON fixtures.

**Spec:** `docs/superpowers/specs/2026-09-07-admin-analytics-demo-alignment-design.md`

## Global Constraints

- The local `analytics-layout-demo.html` is the visual baseline; demo fixtures and sample names must never enter runtime code.
- The global ATL header remains unchanged; alignment begins inside the Admin workspace.
- The only Analytics date ranges are `1D`, `1W`, `1M`, and `1Y`; `1W` is the default.
- Pulse contains exactly Active users, Task completed, Repeat users, and ATL Credits settled.
- Acquisition contains exactly Group, Users, Active, Task completed, Repeat, Runs / user, ATL Credits, Paid intent, and Paid.
- User detail remains a dedicated URL-backed profile with Overview, Timeline, Runs, Usage, and Sessions.
- Do not render token totals, raw prompts, strategy content, provider response bodies, credentials, or automatically selected provider details.
- Keep safe DOM construction, independent request settlement, local retry states, admin access-loss handling, ARIA, and keyboard navigation.
- Do not commit `.superpowers/`, `dashboard/storage/backtest.db`, credentials, API keys, or local QA databases.

## File Map

- Modify `dashboard/backend/domain/analytics/value_queries.py`: add the selected-range accepted-run count to each value-user list row and load its existing acquisition fact for the queue.
- Modify `dashboard/backend/tests/fixtures/admin_analytics/users.json`: provide a safe contract example for the new count.
- Modify `dashboard/backend/tests/domain/analytics/test_value_queries.py`: prove counts are selected-range and default priority requests receive them.
- Modify `dashboard/backend/tests/test_admin_analytics_api.py`: pin the public response field.
- Modify `dashboard/backend/tests/test_admin_analytics_value_frontend.py`: replace old lifecycle-layout assertions with demo-alignment, partial-state, and profile contracts.
- Modify `dashboard/backend/tests/test_admin_acquisition_frontend.py`: pin expanded-by-default acquisition markup and the exact nine columns.
- Modify `dashboard/backend/tests/test_admin_analytics_frontend.py`: retain rail, profile, safe-rendering, and cache-version contracts while removing obsolete hidden-overview expectations.
- Modify `dashboard/frontend/app.html`: replace the old Admin heading and Analytics overview markup with the approved rail, header, filters, five overview sections, and simplified profile shell.
- Modify `dashboard/frontend/js/admin-analytics-value.js`: make acquisition the aggregate source of truth, render the new modules and filtered Users directory, auto-apply filters, and preserve partial data.
- Modify `dashboard/frontend/js/admin-analytics.js`: keep the dedicated profile navigation and render only approved profile/run facts.
- Modify `dashboard/frontend/js/admin-tabs.js`: let Analytics open the Users workspace without losing its URL-backed filters.
- Modify `dashboard/frontend/styles.css`: port the approved demo proportions and visual language into scoped production selectors.

---

### Task 1: Supply the Action Queue Run Count

**Files:**
- Modify: `dashboard/backend/domain/analytics/value_queries.py:361-375`
- Modify: `dashboard/backend/domain/analytics/value_queries.py:1193-1361`
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/users.json`
- Modify: `dashboard/backend/tests/domain/analytics/test_value_queries.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py`

**Interfaces:**
- Consumes: `ValueAnalyticsStore.list_acquisition_facts(user_ids, start, end) -> dict[int, AcquisitionGroupFacts]`.
- Produces: `ValueUserListItem.accepted_runs_in_range: int`, always non-negative and scoped by `date_range` / `acquisition_start` / `acquisition_end`.

- [ ] **Step 1: Write failing domain and API contract tests**

Import `AcquisitionGroupFacts`, extend the existing `FakeValueStore`, and update
the existing `_service` helper so the list-user contract is tested through its
real composition boundary:

```python
class FakeValueStore:
    def __init__(self, *, snapshots, commercial, daily=(), credit_activity=None, acquisition=None):
        self.snapshots = dict(snapshots)
        self.commercial = dict(commercial)
        self.daily = list(daily)
        self.credit_activity = dict(credit_activity or {})
        self.acquisition = dict(acquisition or {})
        self.commercial_windows = []
        self.acquisition_windows = []

    def list_acquisition_facts(self, user_ids, *, start, end):
        self.acquisition_windows.append((start, end))
        return {
            user_id: self.acquisition.get(user_id, AcquisitionGroupFacts())
            for user_id in user_ids
        }


def _service(
    *,
    snapshots,
    commercial=None,
    daily=(),
    events=(),
    rollups=(),
    excluded=(),
    legacy_availability=None,
    acquisition=None,
):
    facts = commercial or {user_id: _commercial(user_id) for user_id in snapshots}
    value_store = FakeValueStore(
        snapshots=snapshots,
        commercial=facts,
        daily=daily,
        acquisition=acquisition,
    )
    legacy_service = FakeLegacyService(legacy_availability)
    service = ValueAnalyticsQueryService(
        store=FakeBaseStore(excluded),
        user_store=FakeUserStore([_user(user_id) for user_id in snapshots]),
        value_store=value_store,
        query_store=FakeQueryStore(events=events, rollups=rollups),
        legacy_service=legacy_service,
    )
    return service, value_store, legacy_service
def test_priority_users_include_selected_range_accepted_runs():
    start = datetime(2026, 9, 1, tzinfo=UTC)
    end = datetime(2026, 9, 8, tzinfo=UTC)
    service, value_store, _legacy = _service(
        snapshots={101: _snapshot(101, lifecycle="at_risk")},
        acquisition={101: AcquisitionGroupFacts(runs=2)},
    )

    page = service.list_users(
        filters=UserValueFilters(
            priority=True,
            acquisition_start=start,
            acquisition_end=end,
        ),
        limit=25,
        offset=0,
        now=datetime(2026, 9, 8, tzinfo=UTC),
    )

    assert page.items[0].accepted_runs_in_range == 2
    assert value_store.acquisition_windows == [(start, end)]
```

In the API test, assert the serialized field exists and is numeric:

```python
payload = response.json()
assert payload["items"][0]["accepted_runs_in_range"] >= 0
```

Update `users.json` with safe values such as:

```json
"accepted_runs_in_range": 6
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/domain/analytics/test_value_queries.py \
  dashboard/backend/tests/test_admin_analytics_api.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models \
  -q
```

Expected: FAIL because `ValueUserListItem` does not expose `accepted_runs_in_range`.

- [ ] **Step 3: Extend the value-user contract without adding a query type**

Add the field:

```python
class ValueUserListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    joined_at: datetime
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial_tier: CommercialTier
    lifetime_net_purchased_micro: int = Field(ge=0)
    accepted_runs_in_range: int = Field(default=0, ge=0)
    priority_group: PriorityGroup
    profile_path: str
    acquisition: AcquisitionAttribution | None = None
```

Load the already-defined acquisition facts for every user-list request, because
the queue always displays the run count. The API router already converts
`date_range=1d|1w|1m|1y` into the UTC start and exclusive UTC end passed here,
so the service must not derive a second date window:

```python
acquisition_facts = (
    self.value_store.list_acquisition_facts(
        self._ids(users),
        start=acquisition_start,
        end=acquisition_end,
    )
)
```

The production `ValueAnalyticsStore` already implements this method. Add the
same method to the `FakeValueStore` in `test_value_queries.py`; do not hide a
missing implementation behind `hasattr`, because the queue's `Runs` column is
part of the required contract.

Pass the selected-range count into the response:

```python
ValueUserListItem(
    user_id=user_id,
    display_name=str(user.get("display_name") or ""),
    email=str(user.get("email") or ""),
    joined_at=_parse_timestamp(user["created_at"]),
    lifecycle=_lifecycle(snapshot),
    operational=_operational(snapshot),
    commercial_tier=fact.commercial_tier,
    lifetime_net_purchased_micro=fact.lifetime_net_purchased_micro,
    accepted_runs_in_range=acquisition_fact.runs,
    priority_group=group,
    profile_path=f"/admin/analytics/users/{user_id}",
    acquisition=attribution,
)
```

- [ ] **Step 4: Re-run the focused tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Commit the contract extension**

```bash
git add \
  dashboard/backend/domain/analytics/value_queries.py \
  dashboard/backend/tests/fixtures/admin_analytics/users.json \
  dashboard/backend/tests/domain/analytics/test_value_queries.py \
  dashboard/backend/tests/test_admin_analytics_api.py
git commit -m "feat: expose analytics queue run counts"
```

---

### Task 2: Replace the Overview Markup With the Approved Story

**Files:**
- Modify: `dashboard/frontend/app.html:2103-2270`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_acquisition_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`

**Interfaces:**
- Consumes: the existing Admin tab IDs and `adminTab` navigation behavior from `js/admin-tabs.js`.
- Produces: stable DOM targets for `renderPulse`, `renderAcquisition`, `renderReturnPanel`, `renderValueExchange`, `renderActionQueue`, and the filtered Users directory.

- [ ] **Step 1: Replace old static expectations with the approved markup contract**

Add an ordered-section test:

```python
def test_demo_aligned_overview_has_the_approved_story_order():
    overview_start = APP_HTML.index('id="adminAnalyticsOverview"')
    overview_end = APP_HTML.index('id="adminAnalyticsDeepSections"', overview_start)
    markup = APP_HTML[overview_start:overview_end]
    ordered = [
        'id="adminAnalyticsPulse"',
        'id="adminAnalyticsAcquisition"',
        'id="adminAnalyticsReturn"',
        'id="adminAnalyticsValueExchange"',
        'id="adminAnalyticsActionQueue"',
    ]
    assert [markup.index(value) for value in ordered] == sorted(
        markup.index(value) for value in ordered
    )
    for label in (
        "Active users",
        "Task completed",
        "Repeat users",
        "ATL Credits settled",
    ):
        assert label in markup
    for removed in ("Core users", "Lifecycle distribution", "Recent 5-day movement"):
        assert removed not in markup
    for removed_id in ("adminAcquisitionBlocked", "adminAcquisitionGroupBy", "adminPriorityFilters"):
        assert removed_id not in markup
```

Update the acquisition contract to assert `aria-expanded="true"`, no Blocked or
Group-by top-level controls, and this exact header order:

```python
def test_acquisition_markup_is_expanded_with_exact_demo_columns():
    overview_start = APP_HTML.index('id="adminAnalyticsOverview"')
    overview_end = APP_HTML.index('id="adminAnalyticsDeepSections"', overview_start)
    markup = APP_HTML[overview_start:overview_end]
    assert 'aria-expanded="true"' in markup
    for removed_id in ("adminAcquisitionBlocked", "adminAcquisitionGroupBy", "adminPriorityFilters"):
        assert removed_id not in markup
    table_start = markup.index('class="admin-analytics-acquisition-table"')
    table_markup = markup[table_start:table_start + 1800]
    headers = [
        "Group", "Users", "Active", "Task completed", "Repeat",
        "Runs / user", "ATL Credits", "Paid intent", "Paid",
    ]
    assert [table_markup.index(f">{label}<") for label in headers] == sorted(
        table_markup.index(f">{label}<") for label in headers
    )
```

- [ ] **Step 2: Run the frontend contract tests and verify failure**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  -q
```

Expected: FAIL on the old headline, old lifecycle cards, collapsed acquisition,
and old priority-user layout.

- [ ] **Step 3: Build the Admin rail shell**

Wrap the existing tablist in a semantic rail without changing tab IDs:

```html
<aside class="admin-rail-shell" aria-label="Admin workspace">
    <a class="admin-rail-brand" href="/app?view=admin&amp;adminTab=analytics">
        <span class="admin-rail-mark" aria-hidden="true"></span>
        <span>ATL / admin</span>
    </a>
    <p class="admin-rail-kicker">Workspace</p>
    <nav id="adminTabs" class="admin-tabs admin-rail" aria-label="Admin sections" role="tablist" aria-orientation="vertical">
        <button id="adminTabAnalytics" class="admin-tab is-active" type="button" role="tab" aria-label="Analytics" title="Analytics" aria-selected="true" aria-controls="adminPanelAnalytics" tabindex="0" data-admin-tab="analytics"><svg aria-hidden="true"><use href="#icon-chart"/></svg><span>Analytics</span></button>
        <button id="adminTabUsers" class="admin-tab" type="button" role="tab" aria-label="Users" title="Users" aria-selected="false" aria-controls="adminPanelUsers" tabindex="-1" data-admin-tab="users"><svg aria-hidden="true"><use href="#icon-users"/></svg><span>Users</span></button>
        <button id="adminTabProviders" class="admin-tab" type="button" role="tab" aria-label="Providers" title="Providers" aria-selected="false" aria-controls="adminPanelProviders" tabindex="-1" data-admin-tab="providers"><svg aria-hidden="true"><use href="#icon-network"/></svg><span>Providers</span></button>
        <button id="adminTabActivity" class="admin-tab" type="button" role="tab" aria-label="Activity" title="Activity" aria-selected="false" aria-controls="adminPanelActivity" tabindex="-1" data-admin-tab="activity"><svg aria-hidden="true"><use href="#icon-activity"/></svg><span>Activity</span></button>
    </nav>
    <p class="admin-rail-footer"><strong>Operator view</strong><span>Read-only analytics</span><span id="adminRailInternalState">Internal accounts hidden</span></p>
</aside>
```

Keep `adminError`, `adminSuccess`, and `adminRefreshBtn` as compatibility hooks,
but move the first two into `admin-workspace-content` and give the old global
refresh button the `sr-only` class. Panel-specific refresh actions remain
visible.

At the start of `adminPanelUsers`, before the existing Grant Pool and Account
Management console, add the Analytics-filtered directory. It stays hidden on a
normal Users visit and does not replace or mutate account management:

```html
<section id="adminAnalyticsUsersDirectory" class="admin-analytics-users-directory" aria-labelledby="adminAnalyticsUsersDirectoryTitle" hidden>
    <header class="admin-analytics-section-head">
        <div><p class="admin-analytics-eyebrow">Analytics selection</p><h3 id="adminAnalyticsUsersDirectoryTitle" tabindex="-1">Filtered users</h3><p id="adminAnalyticsUsersDirectoryScope">—</p></div>
        <button id="adminAnalyticsUsersDirectoryClose" type="button">Close filtered view</button>
    </header>
    <p id="adminAnalyticsUsersDirectoryStatus" role="status" aria-live="polite"></p>
    <p id="adminAnalyticsUsersDirectoryError" class="auth-error" role="alert" hidden></p>
    <button id="adminAnalyticsUsersDirectoryRetry" class="credits-key-action" type="button" hidden>Retry filtered users</button>
    <div class="admin-analytics-table-wrap">
        <table class="admin-analytics-directory-table">
            <caption>Users matching the Analytics filters</caption>
            <thead><tr><th scope="col">User</th><th scope="col">Source</th><th scope="col">Lifecycle</th><th scope="col">Operational</th><th scope="col">Last active</th><th scope="col">Runs</th><th scope="col"><span class="sr-only">Manage account</span></th></tr></thead>
            <tbody id="adminAnalyticsUsersDirectoryBody"></tbody>
        </table>
    </div>
    <div class="admin-analytics-pager" aria-label="Filtered user pages"><span id="adminAnalyticsUsersDirectoryRange" aria-live="polite">Showing 0 of 0</span><button id="adminAnalyticsUsersDirectoryPrev" type="button" disabled>Previous</button><button id="adminAnalyticsUsersDirectoryNext" type="button" disabled>Next</button></div>
</section>
```

- [ ] **Step 4: Replace the Analytics overview block**

Use the approved structure and stable IDs:

```html
<div id="adminAnalyticsOverview" class="admin-analytics-overview">
    <header class="admin-analytics-page-head">
        <div>
            <p class="admin-analytics-eyebrow">Operator cockpit / user intelligence</p>
            <h3 id="adminAnalyticsValueTitle">Analytics</h3>
            <p>A clear path from where users arrive to the moment they reach value, return, consume platform resources, and become ready to pay.</p>
        </div>
        <div class="admin-analytics-head-tools">
            <div id="adminAnalyticsRanges" class="admin-analytics-range" role="group" aria-label="Date range">
                <button type="button" data-analytics-range="1d" aria-pressed="false" tabindex="-1">1D</button>
                <button type="button" data-analytics-range="1w" aria-pressed="true">1W</button>
                <button type="button" data-analytics-range="1m" aria-pressed="false" tabindex="-1">1M</button>
                <button type="button" data-analytics-range="1y" aria-pressed="false" tabindex="-1">1Y</button>
            </div>
            <p class="admin-analytics-updated">Last updated <time id="adminAnalyticsLastUpdated">—</time></p>
            <button id="adminAnalyticsValueRefresh" class="admin-analytics-refresh" type="button">
                <svg aria-hidden="true"><use href="#icon-refresh"/></svg><span>Refresh data</span>
            </button>
        </div>
    </header>

    <p id="adminAnalyticsContext" class="admin-analytics-context">
        <span>All acquisition sources · all cohorts · all users</span>
        <span>Progress is lifetime · activity is selected range</span>
    </p>

    <form id="adminAnalyticsValueFilters" class="admin-analytics-filter-grid" aria-label="Analytics filters">
        <label for="adminAcquisitionSource"><span>Source</span><select id="adminAcquisitionSource" name="acquisition_source" class="filter-select"><option value="">All sources</option><option value="student">Student</option><option value="community">Community</option><option value="friend">Friend</option><option value="competition">Competition team</option><option value="unknown">Unknown</option></select></label>
        <label for="adminAcquisitionCohort"><span>Operating cohort</span><input id="adminAcquisitionCohort" name="acquisition_cohort" maxlength="64" autocomplete="off" spellcheck="false" placeholder="All cohorts"></label>
        <label for="adminAcquisitionLifecycle"><span>Lifecycle</span><select id="adminAcquisitionLifecycle" name="lifecycle" class="filter-select"><option value="">All lifecycle stages</option><option value="new">New</option><option value="active">Active</option><option value="at_risk">At risk</option><option value="dormant">Dormant</option></select></label>
        <label for="adminAcquisitionPaid"><span>Paid status</span><select id="adminAcquisitionPaid" name="paid" class="filter-select"><option value="">All paid states</option><option value="true">Paid</option><option value="false">Unpaid</option></select></label>
        <label class="admin-analytics-filter-toggle" for="adminValueInternal"><input id="adminValueInternal" name="include_internal" type="checkbox"><span>Include internal accounts</span></label>
        <p id="adminValueFilterError" class="auth-error" role="alert" tabindex="-1" hidden></p>
    </form>

    <section id="adminAnalyticsPulse" class="admin-analytics-story-section" aria-labelledby="adminAnalyticsPulseTitle">
        <div class="admin-analytics-section-head"><div><p class="admin-analytics-eyebrow">01 / pulse</p><h4 id="adminAnalyticsPulseTitle">What is happening now?</h4></div><p id="adminAnalyticsSelectedRange">Selected range: —</p></div>
        <div id="adminAnalyticsHeadline" class="admin-analytics-metrics" aria-busy="true">
            <article><span>Active users</span><strong data-admin-value-metric="active-users">—</strong><small>Accepted backtest request · selected range</small></article>
            <article><span>Task completed</span><strong data-admin-value-metric="task-completed">—</strong><small>First successful result · lifetime</small></article>
            <article><span>Repeat users</span><strong data-admin-value-metric="repeat-users">—</strong><small>2 successful runs · different dates</small></article>
            <article><span>ATL Credits settled</span><strong data-admin-value-metric="credits-settled">—</strong><small>Platform consumption · selected range</small></article>
        </div>
        <p id="adminValuePrimaryStatus" class="credits-status" role="status" aria-live="polite"></p>
        <p id="adminValuePrimaryError" class="auth-error" role="alert" hidden></p>
        <button id="adminPulseRetry" class="credits-key-action" data-admin-primary-retry="acquisition" type="button" hidden>Retry Pulse</button>
    </section>

    <section id="adminAnalyticsAcquisition" class="admin-analytics-story-section" aria-labelledby="adminAcquisitionTitle">
        <div class="admin-analytics-section-head"><div><p class="admin-analytics-eyebrow">02 / acquisition</p><h4 id="adminAcquisitionTitle">Where users come from</h4></div><p>Click a row to open the filtered Users list</p></div>
        <div class="admin-analytics-panel admin-analytics-acquisition-panel">
            <div class="admin-analytics-panel-head"><div><p class="admin-analytics-panel-kicker">Acquisition groups</p><h5>Source quality at a glance</h5><p>The same source label follows a user through activation, return, resource use, and purchase.</p></div><button id="adminAcquisitionToggle" type="button" aria-expanded="true" aria-controls="adminAcquisitionPanel" aria-label="Collapse acquisition groups"><svg aria-hidden="true"><use href="#icon-chevron-down"/></svg></button></div>
            <div id="adminAcquisitionPanel" data-admin-value-panel="acquisition" aria-busy="true">
                <p data-admin-value-status role="status" aria-live="polite"></p><p data-admin-value-error class="auth-error" role="alert" hidden></p><button id="adminAcquisitionRetry" class="credits-key-action" data-admin-primary-retry="acquisition" type="button" hidden>Retry Acquisition</button><p id="adminAcquisitionMaturity" role="status"></p>
                <div id="adminAcquisitionGroups" class="admin-analytics-table-wrap" data-admin-value-content><table class="admin-analytics-acquisition-table"><caption>Acquisition groups by source</caption><thead><tr><th scope="col">Group</th><th scope="col">Users</th><th scope="col">Active</th><th scope="col">Task completed</th><th scope="col">Repeat</th><th scope="col">Runs / user</th><th scope="col">ATL Credits</th><th scope="col">Paid intent</th><th scope="col">Paid</th></tr></thead><tbody><tr><td colspan="9">Loading acquisition groups…</td></tr></tbody></table></div>
                <p class="admin-analytics-table-foot"><span id="adminAcquisitionRange">—</span><strong>Select a group → Users</strong></p>
            </div>
        </div>
    </section>

    <section class="admin-analytics-story-section admin-analytics-two-col" aria-label="Retention and resource usage">
        <article id="adminAnalyticsReturn" class="admin-analytics-panel" data-admin-value-panel="overview" aria-labelledby="adminAnalyticsReturnTitle"><div class="admin-analytics-panel-head"><div><p class="admin-analytics-panel-kicker">03 / return</p><h5 id="adminAnalyticsReturnTitle">Do they come back?</h5><p>A compact view of accepted requests across the selected range.</p></div><button type="button" data-open-analysis="retention">Open retention →</button></div><p data-admin-value-status role="status" aria-live="polite"></p><p data-admin-value-error class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-primary-retry="overview" type="button" hidden>Retry Return</button><div id="adminAnalyticsReturnContent" data-admin-value-content></div></article>
        <article id="adminAnalyticsValueExchange" class="admin-analytics-panel" aria-labelledby="adminAnalyticsValueExchangeTitle"><div class="admin-analytics-panel-head"><div><p class="admin-analytics-panel-kicker">04 / value exchange</p><h5 id="adminAnalyticsValueExchangeTitle">Resource and purchase signal</h5><p>What each source costs and whether users move toward payment.</p></div></div><p id="adminValueExchangeStatus" role="status" aria-live="polite"></p><p id="adminValueExchangeError" class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-primary-retry="acquisition" type="button" hidden>Retry Value exchange</button><div id="adminAnalyticsValueExchangeContent"></div></article>
    </section>

    <section id="adminAnalyticsActionQueue" class="admin-analytics-story-section" aria-labelledby="adminPriorityUsersTitle"><div class="admin-analytics-section-head"><div><p class="admin-analytics-eyebrow">05 / action queue</p><h4 id="adminPriorityUsersTitle" tabindex="-1">Who needs attention?</h4></div><button id="adminAnalyticsOpenAllUsers" type="button">Open all users →</button></div><div class="admin-analytics-panel"><div class="admin-analytics-panel-head"><div><p class="admin-analytics-panel-kicker">Operations</p><h5>Next useful action</h5><p>The overview starts with understanding and ends with action.</p></div><span id="adminPriorityUsersRange" aria-live="polite">—</span></div><p id="adminPriorityStatus" role="status" aria-live="polite"></p><p id="adminPriorityError" class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-primary-retry="users" type="button" hidden>Retry Action queue</button><div class="admin-analytics-table-wrap"><table class="admin-analytics-action-table"><caption>Users needing attention</caption><thead><tr><th scope="col">User</th><th scope="col">Source</th><th scope="col">Status</th><th scope="col">Reason</th><th scope="col">Last active</th><th scope="col">Runs</th><th scope="col"><span class="sr-only">Open profile</span></th></tr></thead><tbody id="adminPriorityUsers"><tr><td colspan="7">Loading users…</td></tr></tbody></table></div></div></section>

</div>
```

When applying the snippet, add this hidden detail region immediately after
`adminAnalyticsActionQueue`:

```html
<div id="adminAnalyticsDeepSections" class="admin-analytics-deep-sections" hidden>
    <header class="admin-analytics-deep-head">
        <button id="adminAnalyticsDeepBack" class="admin-analytics-back" type="button">
            <svg aria-hidden="true"><use href="#icon-chevron-left"/></svg>
            <span>Back to analytics overview</span>
        </button>
        <h4 id="adminAnalyticsDeepTitle" tabindex="-1">Deeper analysis</h4>
    </header>
    <div id="adminAnalyticsDeepPanels" class="admin-value-disclosures" aria-labelledby="adminAnalyticsDeepTitle">
        <section class="admin-value-disclosure">
            <button class="admin-value-disclosure-toggle" type="button" data-admin-value-disclosure="retention" aria-expanded="false" aria-controls="adminRetentionPanel"><span><small>Return behavior</small><strong>Activation retention</strong></span><svg aria-hidden="true"><use href="#icon-chevron-down"/></svg></button>
            <div id="adminRetentionPanel" class="admin-value-disclosure-panel" data-admin-value-panel="retention" hidden><p data-admin-value-status role="status" aria-live="polite"></p><p data-admin-value-error class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-value-retry type="button" hidden>Retry section</button><div data-admin-value-content></div></div>
        </section>
        <section class="admin-value-disclosure">
            <button class="admin-value-disclosure-toggle" type="button" data-admin-value-disclosure="commercial" aria-expanded="false" aria-controls="adminCommercialPanel"><span><small>Economic signal</small><strong>Commercial value</strong></span><svg aria-hidden="true"><use href="#icon-chevron-down"/></svg></button>
            <div id="adminCommercialPanel" class="admin-value-disclosure-panel" data-admin-value-panel="commercial" hidden><p data-admin-value-status role="status" aria-live="polite"></p><p data-admin-value-error class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-value-retry type="button" hidden>Retry section</button><div data-admin-value-content></div></div>
        </section>
        <section class="admin-value-disclosure">
            <button class="admin-value-disclosure-toggle" type="button" data-admin-value-disclosure="operational" aria-expanded="false" aria-controls="adminOperationalPanel"><span><small>Friction signal</small><strong>Operational health</strong></span><svg aria-hidden="true"><use href="#icon-chevron-down"/></svg></button>
            <div id="adminOperationalPanel" class="admin-value-disclosure-panel" data-admin-value-panel="operational" hidden><p data-admin-value-status role="status" aria-live="polite"></p><p data-admin-value-error class="auth-error" role="alert" hidden></p><button class="credits-key-action" data-admin-value-retry type="button" hidden>Retry section</button><div data-admin-value-content></div></div>
        </section>
    </div>
</div>
```

Insert only the existing Retention, Commercial, and Operational disclosure
sections inside `adminAnalyticsDeepPanels`, keeping their IDs, toggle buttons,
`data-admin-value-disclosure` values, and `data-admin-value-panel` hooks so
`ensureDisclosureLoaded()` can load them on demand. Do not move the old
lifecycle distribution, movement chart, acquisition disclosure, or
priority-user cards into this container; remove those nodes entirely. The
container must not contain `adminLifecycleDistribution`,
`adminLifecycleMovementChart`, `adminPriorityFilters`, or the old `Core users`
headline. The visible overview controller requests the small legacy overview
payload directly for the Return panel.

- [ ] **Step 5: Re-run markup contracts**

Run the Step 2 command.

Expected: markup tests PASS; renderer tests may still fail until Task 3.

- [ ] **Step 6: Commit the markup migration**

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "feat: align analytics overview structure"
```

---

### Task 3: Connect Real Data to the Five Overview Sections

**Files:**
- Modify: `dashboard/frontend/js/admin-analytics-value.js`
- Modify: `dashboard/frontend/js/admin-tabs.js`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_acquisition_frontend.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- Consumes: `/api/admin/analytics/acquisition`, `/api/admin/analytics/overview`, `/api/admin/analytics/users`, `/api/admin/analytics/retention`, `/api/admin/analytics/commercial`, and `/api/admin/analytics/operational`.
- Produces: `summarizeAcquisition(payload) -> { users, activeUsers, taskCompleted, repeatUsers, runs, creditsSettledMicro, paidIntent, paid }`, independent render functions for all five visible sections, and `openAnalyticsUsers(filters)` navigation into the Users workspace.

- [ ] **Step 1: Write failing controller source contracts**

Pin the required behavior:

```python
def test_demo_aligned_controller_uses_acquisition_as_the_pulse_source():
    source = value_source()
    for contract in (
        "summarizeAcquisition",
        "renderPulse",
        "renderReturnPanel",
        "renderValueExchange",
        "renderActionQueue",
        "/api/admin/analytics/overview",
        "accepted_runs_in_range",
    ):
        assert contract in source
    assert "analyticsRange: '1w'" in source
    assert "Promise.allSettled" in source
    assert "innerHTML" not in source


def test_filters_apply_without_a_visible_apply_button():
    source = value_source()
    assert "scheduleFilterRefresh" in source
    assert "adminAnalyticsValueFilters" in source
    assert "change" in source
    overview_start = APP_HTML.index('id="adminAnalyticsOverview"')
    overview_end = APP_HTML.index('id="adminAnalyticsProfile"', overview_start)
    assert "Apply filters" not in APP_HTML[overview_start:overview_end]


def test_acquisition_drilldown_opens_a_url_backed_users_directory():
    source = value_source()
    for contract in (
        "openAnalyticsUsers",
        "adminAnalyticsUsersDirectory",
        "analyticsUsersView",
        "AdminTabs?.setTab('users')",
    ):
        assert contract in source or contract in APP_HTML
    assert 'id="adminAnalyticsUsersDirectoryBody"' in APP_HTML
    assert 'id="adminAnalyticsUsersDirectoryPrev"' in APP_HTML
    assert 'id="adminAnalyticsUsersDirectoryNext"' in APP_HTML
```

- [ ] **Step 2: Run the controller tests and verify failure**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_app_composition.py \
  -q
```

Expected: FAIL because the old controller renders lifecycle distribution and
does not render the approved modules.

- [ ] **Step 3: Simplify state and set the correct default range**

Keep the API endpoints and safe helpers, add `overview`, set `1w`, and remove
movement from the visible overview state:

```javascript
const API_ENDPOINTS = Object.freeze({
  overview: '/api/admin/analytics/overview',
  retention: '/api/admin/analytics/retention',
  commercial: '/api/admin/analytics/commercial',
  operational: '/api/admin/analytics/operational',
  acquisition: '/api/admin/analytics/acquisition',
  users: '/api/admin/analytics/users',
});

const state = {
  initialized: false,
  active: false,
  requestSeq: 0,
  range: null,
  analyticsRange: '1w',
  acquisitionOpen: true,
  includeInternal: false,
  acquisition: { source: '', cohort: '', lifecycle: '', paid: '', metric: '' },
  directory: { active: false, offset: 0, limit: 50, total: 0, opener: null },
  sections: {
    overview: { loaded: false, data: null, error: null, stale: false },
    acquisition: { loaded: false, data: null, error: null, stale: false },
    users: { loaded: false, data: null, error: null, stale: false },
    retention: { loaded: false, data: null, error: null, stale: false },
    commercial: { loaded: false, data: null, error: null, stale: false },
    operational: { loaded: false, data: null, error: null, stale: false },
  },
};
```

Rewrite `setControls()` so it touches only elements that remain in the new
markup and guards every lookup:

```javascript
function setControls() {
  const values = {
    adminValueInternal: state.includeInternal,
    adminAcquisitionSource: state.acquisition.source,
    adminAcquisitionCohort: state.acquisition.cohort,
    adminAcquisitionLifecycle: state.acquisition.lifecycle,
    adminAcquisitionPaid: state.acquisition.paid,
  };
  Object.entries(values).forEach(([id, value]) => {
    const control = element(id);
    if (!control) return;
    if (control.type === 'checkbox') control.checked = Boolean(value);
    else control.value = value;
  });
  document.querySelectorAll('[data-analytics-range]').forEach((button) => {
    const selected = button.dataset.analyticsRange === state.analyticsRange;
    button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    button.tabIndex = selected ? 0 : -1;
  });
}
```

Read the existing URL keys once during `onEnter`. A missing
`analyticsRange` resolves to `1w`; a missing `analyticsAcquisitionOpen`
resolves to expanded. When writing state, delete obsolete movement/provider
overview keys and only persist the selected range, filters, acquisition-open
state, disclosure state, and profile identifiers. The acquisition collapse
state is written as `analyticsAcquisitionOpen=false` only when collapsed, so a
fresh Analytics visit is expanded by default.

Add `usersView: 'analyticsUsersView'` and
`usersOffset: 'analyticsUsersOffset'` to `URL_KEYS`. Parse the
directory flag only when it equals `true`; validate offset as a non-negative
base-10 integer. `writeUrlState()` preserves those keys while the directory is
active and deletes them when it closes:

```javascript
state.directory.active = params.get(URL_KEYS.usersView) === 'true';
const directoryOffset = Number(params.get(URL_KEYS.usersOffset) || 0);
state.directory.offset = Number.isSafeInteger(directoryOffset) && directoryOffset >= 0
  ? directoryOffset : 0;

setOrDelete(url.searchParams, URL_KEYS.usersView, state.directory.active ? 'true' : '');
setOrDelete(url.searchParams, URL_KEYS.usersOffset, state.directory.active && state.directory.offset
  ? String(state.directory.offset) : '');
```

Replace acquisition and user query builders so removed controls cannot leak
back into requests:

```javascript
function appendAcquisitionFilters(params, { includeMetric = false } = {}) {
  if (state.acquisition.source) params.set('acquisition_source', state.acquisition.source);
  if (state.acquisition.cohort) params.set('acquisition_cohort', state.acquisition.cohort);
  if (state.acquisition.lifecycle) params.set('lifecycle', state.acquisition.lifecycle);
  if (state.acquisition.paid) params.set('paid', state.acquisition.paid);
  if (includeMetric && state.acquisition.metric) {
    const key = {
      active: 'acquisition_active',
      task_completed: 'acquisition_task_completed',
      repeat: 'acquisition_repeat',
      paid_intent: 'acquisition_paid_intent',
      paid: 'paid',
    }[state.acquisition.metric];
    if (key) params.set(key, 'true');
  }
  return params;
}

function acquisitionParams() {
  const params = appendAcquisitionFilters(rangeParams());
  params.set('group_by', 'source');
  return params;
}

function userParams({ priority = true, limit = 25, offset = 0 } = {}) {
  const params = new URLSearchParams({
    priority: priority ? 'true' : 'false',
    include_internal: state.includeInternal ? 'true' : 'false',
    limit: String(limit),
    offset: String(offset),
    date_range: state.analyticsRange,
  });
  return appendAcquisitionFilters(params, { includeMetric: true });
}
```

The `users` request always receives `date_range: state.analyticsRange` so its
`accepted_runs_in_range` field uses the same window as Acquisition. The
Action queue calls `userParams({ priority: true })`, so global filters narrow
the urgent users without turning the queue into a generic directory. The Users
workspace calls `userParams({ priority: false, limit: 50, offset })` to show all
matching users. The
`overview` request is separate because its legacy contract only supplies the
daily active-user series used by the Return panel; source, cohort, lifecycle,
and paid filters therefore do not get silently applied to that series.

Preserve old query parameters when reading a link, but stop writing obsolete
`analyticsMovementRange`, `analyticsOperational`, and `analyticsCommercial`
parameters from the overview.

- [ ] **Step 4: Aggregate the authoritative acquisition response**

Add a pure reducer and fixed metric setters:

```javascript
function summarizeAcquisition(payload) {
  return (Array.isArray(payload?.groups) ? payload.groups : []).reduce((total, group) => ({
    users: total.users + Number(group.users || 0),
    activeUsers: total.activeUsers + Number(group.active || 0),
    taskCompleted: total.taskCompleted + Number(group.task_completed || 0),
    repeatUsers: total.repeatUsers + Number(group.repeat || 0),
    runs: total.runs + Number(group.runs || 0),
    creditsSettledMicro: total.creditsSettledMicro + Number(group.atl_credits_settled_micro || 0),
    paidIntent: total.paidIntent + Number(group.paid_intent || 0),
    paid: total.paid + Number(group.paid || 0),
  }), {
    users: 0,
    activeUsers: 0,
    taskCompleted: 0,
    repeatUsers: 0,
    runs: 0,
    creditsSettledMicro: 0,
    paidIntent: 0,
    paid: 0,
  });
}

function setPulseMetric(name, value) {
  const target = document.querySelector(`[data-admin-value-metric="${name}"]`);
  if (target) target.textContent = value;
}

function renderPulse(summary) {
  setPulseMetric('active-users', number(summary.activeUsers));
  setPulseMetric('task-completed', number(summary.taskCompleted));
  setPulseMetric('repeat-users', number(summary.repeatUsers));
  setPulseMetric('credits-settled', credits(summary.creditsSettledMicro));
  element('adminAnalyticsHeadline')?.setAttribute('aria-busy', 'false');
}
```

- [ ] **Step 5: Render Return and Value Exchange without invented data**

Render daily active data from the overview response. If no authoritative
returning-user series exists, label it unavailable rather than substituting
completed runs:

```javascript
function renderReturnPanel(payload) {
  const target = element('adminAnalyticsReturnContent');
  clear(target);
  const acquisitionScoped = Boolean(
    state.acquisition.source
    || state.acquisition.cohort
    || state.acquisition.lifecycle
    || state.acquisition.paid
  );
  if (acquisitionScoped) {
    target.appendChild(node('p', 'admin-value-empty', 'Unavailable for acquisition-filtered slices.'));
    return;
  }
  const daily = Object.entries(payload?.daily_active_users || {}).sort(([left], [right]) => left.localeCompare(right));
  const chart = node('div', 'admin-analytics-return-chart');
  const max = Math.max(1, ...daily.map(([, value]) => Number(value || 0)));
  daily.forEach(([day, value]) => {
    const column = node('div', 'admin-analytics-return-column');
    const bar = node('span', 'admin-analytics-return-bar');
    bar.style.setProperty('--bar-ratio', String(Number(value || 0) / max));
    bar.setAttribute('aria-label', `${formatDate(day)}: ${number(value)} active users`);
    column.append(bar, node('small', '', new Intl.DateTimeFormat(undefined, { weekday: 'short', timeZone: 'UTC' }).format(new Date(`${day}T00:00:00Z`))));
    chart.appendChild(column);
  });
  target.appendChild(daily.length ? chart : node('p', 'admin-value-empty', 'No activity in this range.'));
  target.appendChild(node('p', 'admin-analytics-legend', 'Active users · Returning-user series unavailable'));
}

function metricList(entries) {
  const list = node('ul', 'admin-analytics-signal-list');
  entries.forEach(([label, value]) => {
    const item = document.createElement('li');
    item.append(node('span', '', label), node('strong', '', value));
    list.appendChild(item);
  });
  return list;
}

function funnelList(entries) {
  const list = node('ol', 'admin-analytics-funnel');
  entries.forEach(([label, value]) => {
    const item = document.createElement('li');
    item.append(node('span', '', label), node('strong', '', value));
    list.appendChild(item);
  });
  return list;
}

function renderValueExchange(summary) {
  const target = element('adminAnalyticsValueExchangeContent');
  clear(target);
  const runsPerActive = summary.activeUsers ? summary.runs / summary.activeUsers : null;
  const creditsPerUser = summary.users ? summary.creditsSettledMicro / summary.users : null;
  target.appendChild(metricList([
    ['Runs per active user', formatRatio(runsPerActive)],
    ['Credits settled / user', creditsPerUser == null ? '—' : credits(creditsPerUser)],
    ['Checkout starters', number(summary.paidIntent)],
  ]));
  target.appendChild(funnelList([
    ['Credits page viewed', 'Unavailable'],
    ['Checkout started', number(summary.paidIntent)],
    ['Purchase settled', number(summary.paid)],
  ]));
}
```

- [ ] **Step 6: Render the Action queue as the approved table**

Replace the card renderer with row construction:

```javascript
function actionStatus(user) {
  if (user.operational?.state !== 'healthy') {
    return OPERATIONAL_LABELS[user.operational?.state] || 'Needs attention';
  }
  const segment = user.lifecycle?.segment;
  if (segment === 'at_risk') return 'At risk';
  if (segment === 'dormant') return 'Dormant';
  if (Number(user.accepted_runs_in_range || 0) > 0 || ['growing', 'core'].includes(segment)) return 'Active';
  return 'New';
}

function relativeTime(value, now = Date.now()) {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return 'No activity';
  const elapsed = Math.max(0, now - timestamp);
  const minutes = Math.floor(elapsed / 60000);
  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function renderActionQueue(payload) {
  const body = element('adminPriorityUsers');
  clear(body);
  const items = Array.isArray(payload?.items) ? payload.items : [];
  items.slice(0, 12).forEach((user) => {
    const row = document.createElement('tr');
    const identity = document.createElement('th');
    identity.scope = 'row';
    identity.append(profileLink(user), node('span', 'admin-analytics-user-email', user.email));
    row.appendChild(identity);
    row.appendChild(node('td', '', ACQUISITION_SOURCE_LABELS[user.acquisition?.source] || 'Unknown'));
    row.appendChild(node('td', '', actionStatus(user)));
    row.appendChild(node('td', '', user.operational?.state === 'healthy' ? user.lifecycle?.reason : user.operational?.reason));
    row.appendChild(node('td', '', relativeTime(user.lifecycle?.last_meaningful_activity_at)));
    row.appendChild(node('td', 'admin-analytics-number', number(user.accepted_runs_in_range)));
    const action = document.createElement('td');
    action.appendChild(profileLink(user, { iconOnly: true }));
    row.appendChild(action);
    body.appendChild(row);
  });
  if (!items.length) {
    const row = document.createElement('tr');
    const cell = node('td', 'admin-value-empty', 'No users need attention for these filters.');
    cell.colSpan = 7;
    row.appendChild(cell);
    body.appendChild(row);
  }
  body.setAttribute('aria-busy', 'false');
  element('adminPriorityUsersRange').textContent = `${number(payload?.total || 0)} users`;
}
```

Extend `profileLink(user, { iconOnly = false } = {})` with an optional
icon-only action while preserving normal link behavior, modified clicks, focus,
and the dedicated profile URL:

```javascript
function profileLink(user, { iconOnly = false } = {}) {
  const label = user.display_name || user.email || `User #${user.user_id}`;
  const link = node('a', `admin-priority-profile-link${iconOnly ? ' is-icon-only' : ''}`);
  link.href = profileHref(user.user_id);
  link.setAttribute('aria-label', `Open analytics profile for ${label}`);
  if (iconOnly) {
    link.title = `Open analytics profile for ${label}`;
    const icon = document.createElement('svg');
    icon.setAttribute('aria-hidden', 'true');
    const use = document.createElement('use');
    use.setAttribute('href', '#icon-chevron-right');
    icon.appendChild(use);
    link.append(icon, node('span', 'sr-only', `Open ${label}`));
  } else {
    link.textContent = label;
  }
  link.addEventListener('click', (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof window.AdminAnalytics?.openProfile !== 'function') return;
    event.preventDefault();
    window.AdminAnalytics.openProfile(user.user_id);
  });
  return link;
}
```

The action-queue header's `Open all users` button calls the existing
`window.AdminTabs.openAccountManagement()` with no user argument. That opens
the Users workspace with an empty account search. Acquisition group and metric
links update the real `analyticsAcquisition*` URL keys, refresh the queue with
matching `/api/admin/analytics/users` parameters, and move focus to the queue
heading; they do not invent a second user-list endpoint. Returning to Analytics
through the rail restores the range and filter query parameters still present
in the URL.

Route acquisition groups and the header action through a URL-backed Users
directory. Keep metric drill-down values in the existing acquisition query
keys, and keep Action queue rendering separate:

```javascript
function openAnalyticsUsers({ group = null, metric = '', opener = document.activeElement } = {}) {
  if (group?.kind === 'source') state.acquisition.source = group.value;
  if (group?.kind === 'cohort') state.acquisition.cohort = group.value;
  state.acquisition.metric = metric;
  if (metric === 'paid') state.acquisition.paid = 'true';
  state.directory.active = true;
  state.directory.offset = 0;
  state.directory.opener = opener;
  writeUrlState();
  window.AdminTabs?.setTab('users');
  syncUsersDirectory();
}

function closeUsersDirectory({ updateUrl = true, focus = true } = {}) {
  state.directory.active = false;
  state.directory.offset = 0;
  element('adminAnalyticsUsersDirectory')?.setAttribute('hidden', '');
  if (updateUrl) writeUrlState();
  if (focus) element('adminCreditsUserQuery')?.focus();
}

element('adminAnalyticsOpenAllUsers')?.addEventListener('click', (event) => {
  openAnalyticsUsers({ opener: event.currentTarget });
});
```

Change `acquisitionLink()` to call
`openAnalyticsUsers({ group, metric, opener: event.currentTarget })`. Render and
page the directory with the same safe DOM helpers:

```javascript
function renderUsersDirectory(payload) {
  const body = element('adminAnalyticsUsersDirectoryBody');
  clear(body);
  const items = Array.isArray(payload?.items) ? payload.items : [];
  items.forEach((user) => {
    const row = document.createElement('tr');
    const identity = document.createElement('th');
    identity.scope = 'row';
    identity.append(node('strong', '', user.display_name || user.email), node('span', '', user.email));
    row.appendChild(identity);
    row.appendChild(node('td', '', ACQUISITION_SOURCE_LABELS[user.acquisition?.source] || 'Unknown'));
    row.appendChild(node('td', '', LIFECYCLE_LABELS[user.lifecycle?.segment] || 'Unknown'));
    row.appendChild(node('td', '', OPERATIONAL_LABELS[user.operational?.state] || 'Unknown'));
    row.appendChild(node('td', '', relativeTime(user.lifecycle?.last_meaningful_activity_at)));
    row.appendChild(node('td', 'admin-analytics-number', number(user.accepted_runs_in_range)));
    const action = document.createElement('td');
    const manage = node('button', 'credits-key-action', 'Manage');
    manage.type = 'button';
    manage.setAttribute('aria-label', `Manage ${user.display_name || user.email}`);
    manage.addEventListener('click', () => {
      closeUsersDirectory({ updateUrl: false, focus: false });
      window.AdminTabs?.openAccountManagement({ userId: user.user_id, email: user.email });
    });
    action.appendChild(manage);
    row.appendChild(action);
    body.appendChild(row);
  });
  if (!items.length) {
    const row = document.createElement('tr');
    const cell = node('td', 'admin-value-empty', 'No users match these Analytics filters.');
    cell.colSpan = 7;
    row.appendChild(cell);
    body.appendChild(row);
  }
  state.directory.total = Number(payload?.total || 0);
  const start = state.directory.total ? state.directory.offset + 1 : 0;
  const end = Math.min(state.directory.offset + items.length, state.directory.total);
  element('adminAnalyticsUsersDirectoryRange').textContent = `Showing ${start}–${end} of ${state.directory.total}`;
  element('adminAnalyticsUsersDirectoryPrev').disabled = state.directory.offset <= 0;
  element('adminAnalyticsUsersDirectoryNext').disabled = end >= state.directory.total;
}

async function refreshUsersDirectory() {
  const status = element('adminAnalyticsUsersDirectoryStatus');
  const error = element('adminAnalyticsUsersDirectoryError');
  const retry = element('adminAnalyticsUsersDirectoryRetry');
  status.textContent = 'Loading filtered users...';
  error.hidden = true;
  retry.hidden = true;
  try {
    const params = userParams({ priority: false, limit: state.directory.limit, offset: state.directory.offset });
    const payload = await request(`${API_ENDPOINTS.users}?${params}`);
    renderUsersDirectory(payload);
    status.textContent = '';
  } catch (requestError) {
    if (await handleAccessLost(requestError)) return;
    status.textContent = '';
    error.textContent = SECTION_UNAVAILABLE;
    error.hidden = false;
    retry.hidden = false;
  }
}

function syncUsersDirectory() {
  const visible = state.directory.active
    && new URL(window.location.href).searchParams.get('adminTab') === 'users';
  const directory = element('adminAnalyticsUsersDirectory');
  if (directory) directory.hidden = !visible;
  if (!visible) return;
  updateScopeCopy();
  refreshUsersDirectory();
  element('adminAnalyticsUsersDirectoryTitle')?.focus();
}
```

Bind close, retry, Previous, and Next. Previous/Next change offset by
`state.directory.limit`, call `writeUrlState()`, and call
`refreshUsersDirectory()`. On `admin:tabchange`, render the Analytics overview
when the tab is `analytics`, and call `syncUsersDirectory()` when it is `users`.
On `popstate`, re-read URL state before the same synchronization.

In `admin-tabs.js`, extend `openAccountManagement()` to call
`window.AdminAnalyticsValue?.closeUsersDirectory({ updateUrl: false, focus: false })`
and delete `analyticsUsersView` plus `analyticsUsersOffset` from the URL before
submitting the existing account search. Export `closeUsersDirectory` and
`openAnalyticsUsers` from `window.AdminAnalyticsValue`; no Credits API or
account-mutation behavior changes.

- [ ] **Step 7: Fetch visible panels independently and auto-apply filters**

Build the legacy overview query using the backend's existing `date_range`
contract. Do not send ISO timestamps in `from` or `to`: the router accepts
`YYYY-MM-DD` for explicit dates and rejects full timestamp strings.

```javascript
function overviewParams() {
  return new URLSearchParams({
    date_range: state.analyticsRange,
    include_internal: state.includeInternal ? 'true' : 'false',
  });
}
```

Refresh acquisition, overview, and users together through settled results:

```javascript
async function refreshPrimary() {
  const requestSeq = ++state.requestSeq;
  setPrimaryLoading(true);
  const results = await Promise.allSettled([
    request(`${API_ENDPOINTS.acquisition}?${acquisitionParams()}`),
    request(`${API_ENDPOINTS.overview}?${overviewParams()}`),
    request(`${API_ENDPOINTS.users}?${userParams()}`),
  ]);
  if (requestSeq !== state.requestSeq) return;
  await applyPrimaryResult('acquisition', results[0]);
  await applyPrimaryResult('overview', results[1]);
  await applyPrimaryResult('users', results[2]);
  renderCombinedOverview();
  setPrimaryLoading(false);
}
```

Use these settled-result helpers so one failed request does not erase the other
two visible panels:

```javascript
async function applyPrimaryResult(name, result) {
  const section = state.sections[name];
  if (result.status === 'fulfilled') {
    section.data = result.value;
    section.loaded = true;
    section.error = null;
    section.stale = false;
    if (name === 'overview' && result.value?.last_updated) {
      const updated = element('adminAnalyticsLastUpdated');
      if (updated) {
        updated.dateTime = String(result.value.last_updated);
        updated.textContent = formatTimestamp(result.value.last_updated);
      }
    }
    return;
  }
  if (await handleAccessLost(result.reason)) return;
  section.error = SECTION_UNAVAILABLE;
  section.stale = Boolean(section.data);
}

async function refreshPrimarySection(name) {
  const path = name === 'acquisition'
    ? `${API_ENDPOINTS.acquisition}?${acquisitionParams()}`
    : name === 'overview'
      ? `${API_ENDPOINTS.overview}?${overviewParams()}`
      : `${API_ENDPOINTS.users}?${userParams()}`;
  const result = await Promise.allSettled([request(path)]);
  await applyPrimaryResult(name, result[0]);
  renderCombinedOverview();
}

function setPrimaryLoading(loading) {
  element('adminAnalyticsHeadline')?.setAttribute('aria-busy', String(loading));
  element('adminAcquisitionPanel')?.setAttribute('aria-busy', String(loading));
  element('adminPriorityUsers')?.setAttribute('aria-busy', String(loading));
  if (loading) element('adminValuePrimaryStatus').textContent = 'Refreshing analytics...';
}

function setPrimaryRetry(name, visible) {
  document.querySelectorAll(`[data-admin-primary-retry="${name}"]`).forEach((button) => {
    button.hidden = !visible;
  });
}

function renderCombinedOverview() {
  const acquisition = state.sections.acquisition;
  const overview = state.sections.overview;
  const users = state.sections.users;
  if (acquisition.data) {
    const summary = summarizeAcquisition(acquisition.data);
    renderPulse(summary);
    renderAcquisition(acquisition.data, element('adminAcquisitionPanel'));
    renderValueExchange(summary);
  }
  if (overview.data) renderReturnPanel(overview.data);
  if (users.data) renderActionQueue(users.data);
  renderPrimaryErrors();
  setPrimaryRetry('acquisition', Boolean(acquisition.error));
  setPrimaryRetry('overview', Boolean(overview.error));
  setPrimaryRetry('users', Boolean(users.error));
  updateScopeCopy();
}

function renderPrimaryErrors() {
  const mappings = [
    ['acquisition', element('adminAcquisitionPanel')?.querySelector('[data-admin-value-error]')],
    ['overview', element('adminAnalyticsReturn')?.querySelector('[data-admin-value-error]')],
    ['users', element('adminPriorityError')],
  ];
  mappings.forEach(([name, target]) => {
    if (!target) return;
    target.textContent = state.sections[name].error || '';
    target.hidden = !state.sections[name].error;
  });
  const pulseError = element('adminValuePrimaryError');
  if (pulseError) {
    pulseError.textContent = state.sections.acquisition.error || '';
    pulseError.hidden = !state.sections.acquisition.error;
  }
}

function updateScopeCopy() {
  const source = state.acquisition.source
    ? (ACQUISITION_SOURCE_LABELS[state.acquisition.source] || state.acquisition.source)
    : 'All acquisition sources';
  const cohort = state.acquisition.cohort || 'all cohorts';
  const internal = state.includeInternal ? 'internal accounts included' : 'internal accounts hidden';
  const lifecycle = state.acquisition.lifecycle
    ? (LIFECYCLE_LABELS[state.acquisition.lifecycle] || state.acquisition.lifecycle)
    : 'all lifecycle stages';
  const paid = state.acquisition.paid === 'true'
    ? 'paid users'
    : state.acquisition.paid === 'false' ? 'unpaid users' : 'all paid states';
  const scope = `${source} · ${cohort} · ${lifecycle} · ${paid}`;
  const context = element('adminAnalyticsContext');
  if (context) {
    clear(context);
    context.append(
      node('span', '', scope),
      node('span', '', `Progress is lifetime · activity is ${ANALYTICS_RANGES[state.analyticsRange].label}`),
    );
  }
  const directoryScope = element('adminAnalyticsUsersDirectoryScope');
  if (directoryScope) directoryScope.textContent = scope;
  const range = element('adminAnalyticsSelectedRange');
  if (range) range.textContent = `Selected range: ${ANALYTICS_RANGES[state.analyticsRange].label}`;
  const railState = element('adminRailInternalState');
  if (railState) railState.textContent = internal;
}
```

For filter controls, refresh immediately on select/checkbox changes, on Enter
for the cohort input, and after a 350 ms debounce while typing:

```javascript
let filterTimer = null;
function applyOverviewFilters() {
  const form = element('adminAnalyticsValueFilters');
  if (!form?.reportValidity()) return;
  state.includeInternal = Boolean(element('adminValueInternal')?.checked);
  state.acquisition.source = element('adminAcquisitionSource')?.value || '';
  state.acquisition.cohort = element('adminAcquisitionCohort')?.value.trim() || '';
  state.acquisition.lifecycle = element('adminAcquisitionLifecycle')?.value || '';
  state.acquisition.paid = element('adminAcquisitionPaid')?.value || '';
  state.acquisition.metric = '';
  writeUrlState();
  refresh();
}

function scheduleFilterRefresh() {
  window.clearTimeout(filterTimer);
  filterTimer = window.setTimeout(applyOverviewFilters, 350);
}
```

`setAnalyticsRange` must call `writeUrlState()` and `refresh()` after updating
the range. Bind `change` on the selects and internal-account checkbox, bind
`keydown` Enter on the cohort field, and bind `input` to
`scheduleFilterRefresh()`. Bind `[data-admin-primary-retry]` so the failed
section alone is re-requested: `acquisition` and `users` call their existing
fetchers; `overview` calls the legacy overview request. Update the context line,
selected-range copy, rail internal-account copy, and last-updated `<time>` after
each successful request. The last-updated time comes from the fulfilled
overview `last_updated` field when available; otherwise retain the previous
visible time and mark the affected panel stale.

- [ ] **Step 8: Preserve acquisition collapse and on-demand detail behavior**

Set acquisition expanded by default. The collapse control hides only
`adminAcquisitionPanel` and synchronizes `aria-expanded`, its accessible label,
the `+`/`−` icon text, and `state.acquisitionOpen`:

```javascript
function setAcquisitionExpanded(expanded, { writeUrl = true } = {}) {
  state.acquisitionOpen = Boolean(expanded);
  const panel = element('adminAcquisitionPanel');
  const toggle = element('adminAcquisitionToggle');
  if (panel) panel.hidden = !state.acquisitionOpen;
  if (toggle) {
    toggle.setAttribute('aria-expanded', String(state.acquisitionOpen));
    toggle.setAttribute(
      'aria-label',
      state.acquisitionOpen ? 'Collapse acquisition groups' : 'Expand acquisition groups'
    );
  }
  if (writeUrl) writeUrlState();
}
```

Bind `adminAcquisitionToggle` to `setAcquisitionExpanded(!state.acquisitionOpen)`.
The `Open retention` action reveals `adminAnalyticsDeepSections`, moves focus to
`adminAnalyticsDeepTitle`, calls `ensureDisclosureLoaded('retention')`, and
leaves the overview's five sections in the DOM. The deep-region Back action
hides the region, restores focus to the `Open retention` button, and does not
clear loaded panel data. Retention, Commercial, and Operational toggles remain
independently collapsible inside the deep region.

- [ ] **Step 9: Re-run the focused frontend tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 10: Commit the real-data controller**

```bash
git add \
  dashboard/frontend/js/admin-analytics-value.js \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_app_composition.py
git commit -m "feat: connect demo analytics to real data"
```

---

### Task 4: Simplify and Restyle the Dedicated User Profile

**Files:**
- Modify: `dashboard/frontend/app.html:2360-2439`
- Modify: `dashboard/frontend/js/admin-analytics.js:60-1260` (remove the old overview loader and retain profile routing/activity rendering)
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`

**Interfaces:**
- Consumes: existing `ValueUserProfile`, activity pagination endpoints, `openProfile`, and `closeProfile`.
- Produces: a progress-first dedicated profile with the five existing tabs and approved core run facts.

- [ ] **Step 1: Write failing profile hierarchy tests**

Add:

```python
def test_profile_is_progress_first_and_keeps_five_sections():
    start = APP_HTML.index('id="adminAnalyticsProfile"')
    end = APP_HTML.index('id="adminPanelUsers"', start)
    markup = APP_HTML[start:end]
    progress = markup.index('id="adminAnalyticsMilestones"')
    tabs = markup.index('id="adminAnalyticsProfileTabs"')
    assert progress < tabs
    for section in ("Overview", "Timeline", "Runs", "Usage", "Sessions"):
        assert f">{section}<" in markup
    assert "Default provider" not in markup
    assert "Legacy status" not in markup


def test_run_renderer_excludes_removed_details():
    source = profile_source()
    assert "LLM calls" not in source
    assert "Token usage" not in source
    assert "Input tokens" not in source
    assert "Output tokens" not in source
```

- [ ] **Step 2: Run profile tests and verify failure**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  -q
```

Expected: FAIL because the current profile begins with a dense summary sidebar
and renders obsolete default-provider and legacy-state rows.

- [ ] **Step 3: Reorder the profile shell**

Keep the existing profile and tab IDs. Build this hierarchy:

```html
<article id="adminAnalyticsProfile" class="admin-analytics-profile" aria-labelledby="adminAnalyticsProfileTitle" hidden>
    <nav class="admin-analytics-breadcrumbs" aria-label="Breadcrumb"><a id="adminAnalyticsProfileBreadcrumbParent" href="/app?view=admin&amp;adminTab=analytics">Analytics</a><span aria-hidden="true">/</span><span id="adminAnalyticsProfileBreadcrumbCurrent">User analytics profile</span></nav>
    <button id="adminAnalyticsProfileBack" class="admin-analytics-back" type="button"><svg aria-hidden="true"><use href="#icon-chevron-left"/></svg><span>Back to analytics overview</span></button>
    <p id="adminAnalyticsProfileError" class="auth-error" role="alert" hidden></p>
    <header class="admin-analytics-profile-head"><div><p class="admin-analytics-eyebrow">User analytics profile</p><h3 id="adminAnalyticsProfileTitle" tabindex="-1">Loading user analytics</h3><p id="adminAnalyticsProfileEmail">—</p></div><div class="admin-analytics-profile-tags"><span id="adminAnalyticsProfileSource">Unknown</span><span id="adminAnalyticsProfileCohort">Unknown</span><span id="adminAnalyticsProfileLifecycle">—</span><span id="adminAnalyticsProfileOperational">—</span></div></header>
    <section class="admin-analytics-progress" aria-labelledby="adminAnalyticsProgressTitle"><div class="admin-analytics-section-head"><div><p class="admin-analytics-eyebrow">Progress</p><h4 id="adminAnalyticsProgressTitle">Path to first value</h4></div></div><ol id="adminAnalyticsMilestones" aria-label="User progress"></ol></section>
    <nav id="adminAnalyticsProfileTabs" class="admin-analytics-profile-tabs" role="tablist" aria-label="User analytics sections">
        <button id="adminAnalyticsProfileTabOverview" role="tab" aria-selected="true" aria-controls="adminAnalyticsSectionOverview" tabindex="0" data-analytics-section-tab="overview" type="button">Overview</button>
        <button id="adminAnalyticsProfileTabTimeline" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionTimeline" tabindex="-1" data-analytics-section-tab="timeline" type="button">Timeline</button>
        <button id="adminAnalyticsProfileTabRuns" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionRuns" tabindex="-1" data-analytics-section-tab="runs" type="button">Runs</button>
        <button id="adminAnalyticsProfileTabUsage" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionUsage" tabindex="-1" data-analytics-section-tab="usage" type="button">Usage</button>
        <button id="adminAnalyticsProfileTabSessions" role="tab" aria-selected="false" aria-controls="adminAnalyticsSectionSessions" tabindex="-1" data-analytics-section-tab="sessions" type="button">Sessions</button>
    </nav>
    <a id="adminAnalyticsOpenAccount" href="/app?view=admin&amp;adminTab=users">Open account management →</a>
</article>
```

Place the five existing tab panels, with their current IDs and ARIA ownership,
between `adminAnalyticsProfileTabs` and `adminAnalyticsOpenAccount`. The
collapsed account details inside Overview keep User ID, Joined, Last meaningful activity,
attribution method/timestamps, and the audited Source/Cohort form. Remove
Default provider, Region, Device, Browser, Legacy status, raw evidence-event ID
presentation, and repeated summary fields from the visible overview.

- [ ] **Step 4: Render the fixed four-step progress indicator**

Replace the generic event loop with fixed milestones:

```javascript
const PROGRESS_STEPS = Object.freeze([
  ['account_signed_up', 'Account created'],
  ['agent_created', 'Agent created'],
  ['backtest_requested', 'Backtest attempted'],
  ['backtest_completed', 'First successful result'],
]);

function renderMilestones(profile) {
  const list = element('adminAnalyticsMilestones');
  clearChildren(list);
  const milestones = profile.activation_milestones || {};
  let waiting = false;
  PROGRESS_STEPS.forEach(([eventName, label]) => {
    const occurredAt = milestones[eventName];
    const item = document.createElement('li');
    const complete = Boolean(occurredAt) && !waiting;
    waiting = waiting || !complete;
    item.className = complete ? 'is-complete' : 'is-pending';
    item.appendChild(textNode('span', 'admin-analytics-progress-dot', complete ? '✓' : ''));
    const copy = document.createElement('div');
    copy.appendChild(textNode('strong', '', label));
    copy.appendChild(occurredAt ? makeTime(occurredAt) : textNode('small', '', 'Not completed'));
    item.appendChild(copy);
    list.appendChild(item);
  });
}
```

- [ ] **Step 5: Keep only approved activity and run fields**

Continue building all DOM nodes with `textContent`. Runs show time, model,
billing mode, market-data source/universe or symbols when present, date range,
terminal status, result summary, and safe error category. When the current
safe activity contract lacks a requested field, omit the row rather than
showing provider or token details as a substitute.

Remove provider IDs from the visible Runs and Usage table cell composition:

```javascript
const modelLabel = item.model_id || '—';
row.appendChild(textNode('td', '', modelLabel));
```

Do not remove provider IDs from the underlying response model or safe fixture;
this task changes presentation only.

Remove the obsolete overview, attention-filter, and chart rendering paths from
`admin-analytics.js`; `admin-analytics-value.js` now owns those requests. Keep
the exported controller focused on profile routing and refresh behavior:

```javascript
async function refresh() {
  if (!state.active || !state.profile.userId) return;
  await loadProfile(state.profile.userId);
  const section = state.profile.section;
  if (section !== 'overview') await loadProfileSection(section, { append: false });
}

window.AdminAnalytics = { onEnter, refresh, syncAuth, openProfile, closeProfile };
```

Update the frontend source contract so `/api/admin/analytics/overview` is
required in `admin-analytics-value.js`, while `admin-analytics.js` is required
only to use the profile and `/activity` endpoints.

- [ ] **Step 6: Re-run profile tests**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 7: Commit the profile simplification**

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/frontend/js/admin-analytics.js \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py
git commit -m "feat: simplify analytics user profile"
```

---

### Task 5: Port the Demo Visual System and Responsive Layout

**Files:**
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`

**Interfaces:**
- Consumes: the demo-aligned semantic classes created in Tasks 2 and 4.
- Produces: a scoped border-led Analytics interface at 1440, 1024, 768, and 390 CSS pixels.

- [ ] **Step 1: Write failing CSS contract assertions**

Add:

```python
def test_demo_aligned_visual_system_and_responsive_breakpoints_exist():
    for selector in (
        ".admin-rail-shell",
        ".admin-analytics-page-head",
        ".admin-analytics-filter-grid",
        ".admin-analytics-metrics",
        ".admin-analytics-story-section",
        ".admin-analytics-two-col",
        ".admin-analytics-acquisition-table",
        ".admin-analytics-action-table",
        ".admin-analytics-progress",
    ):
        assert selector in STYLES
    assert "@media (max-width: 760px)" in STYLES
    assert "@media (max-width: 470px)" in STYLES
    assert "prefers-reduced-motion: reduce" in STYLES
```

- [ ] **Step 2: Run CSS contracts and verify failure**

Run:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  -q
```

Expected: FAIL because the production stylesheet lacks the demo-aligned class
system.

- [ ] **Step 3: Define scoped Analytics tokens**

Map the demo palette to existing app tokens and keep border-led depth:

```css
.admin-view {
    --analytics-surface: color-mix(in srgb, var(--bg-card) 94%, transparent);
    --analytics-surface-soft: color-mix(in srgb, var(--bg-secondary) 88%, transparent);
    --analytics-line: color-mix(in srgb, var(--border-color) 78%, transparent);
    --analytics-accent: #67e8f9;
    --analytics-lime: #a3e635;
    --analytics-amber: #fbbf24;
    --analytics-danger: #fb7185;
    max-width: 1440px;
}
```

Do not use decorative gradients, floating cards, or nested card shadows. Use
unframed sections with thin dividers; reserve bordered panels for the table and
two compact analysis tools.

- [ ] **Step 4: Port the approved layout proportions**

Implement the core geometry:

```css
.admin-workspace {
    display: grid;
    grid-template-columns: 210px minmax(0, 1fr);
    align-items: stretch;
    gap: 0;
    margin-top: 0;
}

.admin-rail-shell {
    position: sticky;
    top: 64px;
    min-height: calc(100vh - 64px);
    padding: 24px 16px;
    border-right: 1px solid var(--analytics-line);
    background: var(--analytics-surface-soft);
}

.admin-workspace-content {
    min-width: 0;
    padding: 34px clamp(22px, 4vw, 56px) 64px;
}

.admin-analytics-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border-block: 1px solid var(--analytics-line);
}

.admin-analytics-two-col {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}
```

Tables use tabular numerals, sticky or clear headers, compact 12-14 px copy,
and horizontal overflow on constrained widths. Interactive controls receive
visible hover, pressed, focus-visible, disabled, loading, and retry states.

- [ ] **Step 5: Add responsive and motion rules**

```css
@media (max-width: 760px) {
    .admin-workspace { grid-template-columns: 58px minmax(0, 1fr); }
    .admin-rail-shell { padding: 18px 7px; }
    .admin-rail-brand span:last-child,
    .admin-rail-kicker,
    .admin-tab span,
    .admin-rail-footer { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0, 0, 0, 0); }
    .admin-workspace-content { padding: 24px 16px 55px; }
    .admin-analytics-page-head,
    .admin-analytics-section-head { align-items: flex-start; flex-direction: column; }
    .admin-analytics-filter-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .admin-analytics-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .admin-analytics-two-col { grid-template-columns: 1fr; }
}

@media (max-width: 470px) {
    .admin-analytics-filter-grid,
    .admin-analytics-metrics { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
    .admin-view *, .admin-view *::before, .admin-view *::after {
        scroll-behavior: auto;
        transition-duration: 0.01ms;
        animation-duration: 0.01ms;
    }
}
```

- [ ] **Step 6: Remove obsolete Analytics-only style blocks**

Delete selectors used only by removed lifecycle distribution, movement chart,
priority cards, and hidden legacy overview. Keep shared `.admin-analytics-*`
styles still used by the profile activity tables. Confirm no selector removal
affects Users, Providers, Activity, Credits, or the global app shell.

- [ ] **Step 7: Bump static asset cache versions**

Increment the `styles.css`, `admin-analytics.js`, and
`admin-analytics-value.js` query versions in `app.html`, then update exact
version assertions in frontend tests.

- [ ] **Step 8: Run focused contracts**

Run the Step 2 command plus:

```bash
python -m pytest \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Commit the visual alignment**

```bash
git add \
  dashboard/frontend/styles.css \
  dashboard/frontend/app.html \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_frontend_fast_boot.py
git commit -m "style: match the approved analytics demo"
```

---

### Task 6: Verify Behavior, Visual Fidelity, and Repository Hygiene

**Files:**
- Modify only if verification reveals a scoped defect in files already listed above.

**Interfaces:**
- Consumes: the complete implementation from Tasks 1-5.
- Produces: passing regression tests and verified desktop/mobile screenshots of the real authenticated `/app` route.

- [ ] **Step 1: Run the complete Analytics regression set**

```bash
python -m pytest \
  dashboard/backend/tests/domain/analytics \
  dashboard/backend/tests/test_admin_analytics_api.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_api.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_profile_frontend.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the full backend test suite**

```bash
python -m pytest dashboard/backend/tests -q
```

Expected: PASS with no new failures.

- [ ] **Step 3: Start the authenticated local application**

Use a temporary database outside the repository and never print secrets:

```bash
ATL_QA_DIR="$(mktemp -d /tmp/atl-analytics-demo-alignment.XXXXXX)"
DATABASE_PATH="$ATL_QA_DIR/qa.db" \
  python -m uvicorn dashboard.backend.app:app \
  --host 127.0.0.1 \
  --port 8766
```

Open:

```text
http://127.0.0.1:8766/app?view=admin&adminTab=analytics&analyticsRange=1w
```

- [ ] **Step 4: Verify desktop visual fidelity at 1440 x 1000**

Compare the real route with
`http://127.0.0.1:8765/analytics-layout-demo.html`. Confirm:

- the rail, page header, filters, section numbering, metric strip, table, two
  compact panels, and Action queue follow the same composition;
- the real page shows no old lifecycle distribution or movement chart;
- loading, empty, partial, stale, and retry states reserve stable space;
- tables do not clip and no text overlaps.

- [ ] **Step 5: Verify responsive layouts**

Capture and inspect the real page at 1024 x 900, 768 x 900, and 390 x 844.
Confirm the left rail persists, labels collapse accessibly, the header and
filters stack, metric cells become 2-column then 1-column, the two analysis
panels stack, and tables scroll horizontally without widening the viewport.

- [ ] **Step 6: Verify keyboard and profile navigation**

Using only the keyboard:

1. Move through Admin tabs with Up/Down/Home/End.
2. Move through ranges with Left/Right/Home/End.
3. Change filters and confirm URL state and visible data update.
4. Collapse and expand Acquisition and confirm `aria-expanded` changes.
5. Open a user profile from the queue, move through the five profile tabs,
   then return and confirm the prior filters and focus are restored.
6. Open account management and confirm the exact user remains selected.

- [ ] **Step 7: Check the final diff and ignored local artifacts**

```bash
git diff --check
git status --short
git diff --stat origin/main...HEAD
```

Expected: no whitespace errors; `.superpowers/`, `analytics-layout-demo.html`,
and `dashboard/storage/backtest.db` remain untracked and unstaged; no credential
or temporary QA file appears in the diff.

- [ ] **Step 8: Commit only verification fixes, if any**

If visual or regression verification required scoped corrections:

```bash
git add \
  dashboard/backend/domain/analytics/value_queries.py \
  dashboard/backend/tests/fixtures/admin_analytics/users.json \
  dashboard/backend/tests/domain/analytics/test_value_queries.py \
  dashboard/backend/tests/test_admin_analytics_api.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_value_frontend.py \
  dashboard/backend/tests/test_admin_acquisition_frontend.py \
  dashboard/backend/tests/test_app_composition.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/frontend/app.html \
  dashboard/frontend/js/admin-analytics.js \
  dashboard/frontend/js/admin-analytics-value.js \
  dashboard/frontend/styles.css
git commit -m "fix: complete analytics demo alignment"
```
