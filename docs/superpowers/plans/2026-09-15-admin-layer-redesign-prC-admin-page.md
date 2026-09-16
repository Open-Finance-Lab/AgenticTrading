# Admin Layer PR C: The /admin Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three coexisting admin analytics surfaces (the synthetic `/admin-analytics` mock, the in-app value overview, and the hidden in-app legacy overview) with one standalone page at `/admin` built from the mock's *surviving* elements (design §8.2), wired to the nine live `/api/admin/analytics/*` endpoints plus `GET /api/admin/stats`, on the repo's testable IIFE + `?v=N` convention with no inline scripts — and, in the same PR, re-cut the analytics API contract once (§9 fields on, D15 display fields off the page, declared FastAPI query parameters, `_query_values()` gone), rename the route, fix the deploy topology, delete the mock and the two wired in-app modules, and drop the Analytics tab from the old console.

**Architecture:** Four IIFE modules under `dashboard/frontend/js/` (`admin-shell.js` owns the `/api/auth/me` courtesy gate, hash routing, URL-backed range/filter state, the shared `request()` with per-surface `requestSeq`, the loading/empty/error/stale helpers and both dialogs; `admin-live.js` renders the live-operations row from `GET /api/admin/stats` only; `admin-overview.js` renders the nine overview panels and the five detail routes from the analytics endpoints; `admin-users.js` renders the SQL-filtered users list and the lazy, cursor-paged profile) plus `admin.html` (a shell that carries no data — every value slot is `—`) and `admin.css` (the mock's inline styles externalised and pruned of the funnel and live-detail passes the final DOM never renders). The backend side is a contract re-cut on the existing routers: new response-model fields wired to the service methods PR B shipped, hand-parsed query strings replaced by declared `Query(...)` parameters that keep today's names and today's display-safe 422s, and one added key on `/api/admin/stats`. Every renderer is a pure function from payload to DOM built with `createElement`/`textContent` (never `innerHTML`), so pytest can run it under `node -e` against the committed fixtures in `tests/fixtures/admin_analytics/`.

**Tech Stack:** FastAPI + Pydantic v2 (backend), vanilla-JS IIFE modules with no build step, CSS bars and generated SVG (no Chart.js), pytest run from the repo root, Node.js via `subprocess` for the frontend tests (`skipif(shutil.which("node") is None)`), `vercel.json` (Vercel static host) and `app.py` `FileResponse` routes (Render).

**Spec:** `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — implements §7 in full (§7.1 one page at `/admin`; §7.2 the four modules and the panel-ownership table; §7.3 the gate; §7.4 the harvest map; §7.5 what PR C removes and what the old console keeps; §7.6 the testing contract), §8 in full (§8.1 how the mock renders, §8.2 the survival table, which is this plan's element list, §8.3 the forced additions), §9 (exposing the added fields on the response models and dropping the D15 fields from display), §10.3–10.4 (the contract re-cut and the live-operations route), §13 row **C** including its "Must not" column, and decisions D3–D8, D12, D15–D19. §4.1, §4.2 and §4.6 describe what is being replaced; §15 is the source of the copy the rules dialog and tooltips carry.

## Global Constraints

- Never commit `dashboard/storage/data/backtest.db` — a bare backend import rewrites it (`CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH`); stage files by name (`git add <path> <path>`), never `-A`, never `.`, never a bare `-u`.
- Run every `pytest` invocation from the repo root.
- Node-driven frontend tests are `pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")` — a skip on a machine without `node`, not a pass; install a current LTS Node to actually exercise Tasks 4–10 locally.
- Every `?v=` bump updates `test_admin_analytics_frontend.py::test_app_lifecycle_and_cache_versions_are_wired` in lockstep (Task 9 rewrites that test to pin the new page's five `?v=1` tags and the bumped `app.js` / `styles.css` / `js/admin-tabs.js` values), and every other pin of the same literal listed in Task 9 Step 1.
- Both store twins change together and CI's Postgres tier (`ci.yml`, `TEST_POSTGRES_URL`, `@pg_only`) must be green before merge. This plan adds no store method and no table (§13 row C changes routers, response models and the frontend); the constraint binds because PR B's `CreditsStore.sum_ledger_by_day` / `resolve_group_badge` / `top_operational_reasons` paths are exercised for the first time through HTTP here, so a twin that PR B left one-sided surfaces in this PR's Postgres run, not PR B's.
- Ordering (§13): PR 0, PR T, PR A and PR B are merged before this plan starts. PR B must have run in prod long enough that the read paths have data (§13 "B must not land until A's daily job has written at least eight days"); nothing here depends on that beyond the page showing "Incomplete data" honestly.
- The nine `/api/admin/analytics/*` routes are re-cut **once** here (D19): fields listed in §9 land, D15 fields stay in the payload but leave the page, parameter names stay exactly `role` (new), `commercial_tier`, `user_group`, `lifecycle_segment`, `operational_state`, `include_internal`, `from`, `to`, `movement_range`, `limit`, `offset`, `cursor`, `section`, `q`, `priority`, `activated`, `last_meaningful_activity_from`, `last_meaningful_activity_to`, `billing_mode`, `provider`, `model`. No other route changes shape.
- The group badge is **server-computed** (D11). No frontend file in this plan derives a badge from `role`/`user_group`/`commercial_tier`; the client renders `group_badge` as the string it receives, pinned by Task 10.
- No inline `<script>` in `admin.html` (D6); no `#live` route (§8.2, D16); no Chart.js on the page (§7.2); Users/Providers/Activity are **not** ported (D5) — the aside links to `/app?view=admin&adminTab=users|providers|activity`.
- Every UI element built here appears in §8.2 as **keep**, **keep, relabelled** or **added**; nothing marked **cut** is built. Self-review notes state the check.
- Read-only client: every request is a `GET`; every renderer writes `textContent`, never `innerHTML`; no prohibited field name (`api_key`, `session_id`, `network_hash`, `provider_response_body`, `credential_ciphertext`, `prompt`, `strategy`, `portfolio`) appears in any new module (harvested pin, §7.4).

## File Map

Backend:
- `dashboard/backend/domain/analytics/query_service.py`: `BillingLaneDay` model; `AnalyticsOverview.billing_lane_mix`.
- `dashboard/backend/domain/analytics/value_queries.py`: `OperationalReasonCount`, `LedgerDayTotal`; `OperationalAnalyticsResponse.top_operational_reasons`; `CommercialAnalyticsResponse.purchased_by_day` / `consumed_by_day`; `UserValueFilters.role`; `ValueUserListItem` and `ValueUserProfile` gain `user_group`, `role`, `group_badge`, `last_meaningful_activity_at`; the two construction sites populate them.
- `dashboard/backend/api/routers/admin_analytics.py`: declared `Query(...)` dependencies replace `_query_values()`/`_invalid_query()`; every validation rule and every 422 body stays byte-identical.
- `dashboard/backend/api/routers/admin_users.py`: `admin_stats` gains `max_active_dashboard_backtests`.
- `dashboard/backend/app.py`: `GET /admin`, `GET /admin.css`, `GET /admin-analytics` → 308.
- `dashboard/backend/middleware.py`: `EXEMPT_PATHS` gains `/admin`, keeps `/admin-analytics`.
- `dashboard/backend/tests/fixtures/admin_analytics/{overview,overview_partial_error,operational,commercial,users,user_detail}.json`: new fields; new `admin_stats.json`.

Frontend:
- `dashboard/frontend/admin.html` (new), `dashboard/frontend/admin.css` (new).
- `dashboard/frontend/js/admin-shell.js`, `js/admin-live.js`, `js/admin-overview.js`, `js/admin-users.js` (new).
- `dashboard/frontend/js/admin-tabs.js`: redirect branch and `analytics` tab removed; `DEFAULT_TAB = 'users'`; `adminUserQuery` prefill.
- `dashboard/frontend/app.html`: Analytics rail button, `#adminPanelAnalytics`, the two analytics dialogs and the two module `<script>` tags removed; `?v=` bumps.
- `dashboard/frontend/app.js`: `AdminAnalytics`/`AdminAnalyticsValue` calls removed; Profile → Admin navigates to `/admin`.
- `dashboard/frontend/styles.css`: `.admin-analytics-*`, `.admin-value-*`, `.admin-priority-*`, `.admin-lifecycle-*`, `.admin-group-*`, `.admin-help-btn`, `.admin-profile-evidence-grid`, `.admin-commercial-tier-grid` families removed; `.admin-rail button` and `.admin-tab:focus-visible` kept as standalone rules.
- `dashboard/frontend/vercel.json`: `/admin/:path*` rewrite deleted; `redirects` added; `/admin` and `/admin.html` header rules added after the API no-store rule.
- Deleted: `dashboard/frontend/admin-analytics.html`, `js/admin-analytics.js`, `js/admin-analytics-value.js`.

Tests:
- Modified: `test_admin_analytics_api.py`, `test_admin_users.py`, `test_app_composition.py`, `test_vercel_cache_headers.py`, `test_admin_analytics_frontend.py`, `test_admin_tabs_redirect.py` (rewritten), `test_admin_credits_frontend.py`, `test_frontend_fast_boot.py`, `test_credit_format_frontend.py`, `test_analytics_frontend.py`, `test_backtest_comparison_frontend.py`.
- New: `test_middleware_exemptions.py`, `test_admin_page_shell.py`, `_admin_dom_stub.py`, `test_admin_shell_frontend.py`, `test_admin_live_frontend.py`, `test_admin_overview_frontend.py`, `test_admin_users_frontend.py`, `test_admin_page_modules.py`.
- Deleted: `test_admin_analytics_value_frontend.py`.

---

### Task 1: Backend contract re-cut — §9 fields on the response models, declared query parameters, `_query_values()` deleted

**Files:**
- Modify: `dashboard/backend/domain/analytics/query_service.py` — imports (line 8), `FailureCategoryCount` (lines 90-94, add `BillingLaneDay` after it), `AnalyticsOverview` (lines 244-264).
- Modify: `dashboard/backend/domain/analytics/value_queries.py` — imports (lines 9-27), `CommercialAnalyticsResponse` (lines 253-261), `OperationalAnalyticsResponse` (lines 264-276), `UserValueFilters` (lines 305-350), `ValueUserListItem` (lines 365-377), `ValueUserProfile` (lines 389-395), the `ValueUserListItem(` construction inside `list_users` (lines 1319-1331), `get_user_profile` (lines 1351-1390), `__all__` (lines 1395-1414).
- Modify: `dashboard/backend/api/routers/admin_analytics.py` — the whole module (596 lines).
- Modify: `dashboard/backend/tests/fixtures/admin_analytics/overview.json`, `overview_partial_error.json`, `operational.json`, `commercial.json`, `users.json`, `user_detail.json`.
- Modify: `dashboard/backend/tests/test_admin_analytics_api.py` — `test_admin_user_list_accepts_documented_filters` (lines 703-739), `test_value_routes_reject_unknown_duplicate_and_unsafe_queries` (lines 823-844); new tests appended.
- Test: `dashboard/backend/tests/test_admin_analytics_api.py`, `dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models` (unchanged, must stay green).

**Interfaces:**
- Consumes (PR B service methods, named in design §9/§13 row B; PR B's plan file is not in this checkout, so the names below are the ones this plan assumes — verify each with `rg -n "def billing_lane_mix|def top_operational_reasons|def purchased_by_day|def consumed_by_day|def resolve_group_badge" dashboard/backend/domain` before Step 3 and substitute at the single call site named for each):
  - `AnalyticsQueryService.billing_lane_mix(*, filters: AnalyticsMetricFilters) -> dict[str, dict[str, int]]` — ISO-date key → `{"platform_credits": n, "byok": n}` from rollups `billing_mode` (D14).
  - `ValueAnalyticsQueryService.top_operational_reasons(*, include_internal: bool, limit: int = 5) -> list[tuple[str, str, int]]` — `(reason_code, state, users)` from `user_daily_facts.operational_reason_code`.
  - `ValueAnalyticsQueryService.purchased_by_day(*, start: date, end: date, include_internal: bool) -> dict[str, int]` and `consumed_by_day(...)` — ISO-date key → micro amount, via `CreditsStore.sum_ledger_by_day`.
  - `dashboard.backend.domain.analytics.lifecycle.resolve_group_badge(*, role: str, user_group: UserGroup, tier: CommercialTier) -> str` — the superseded 2026-09-12 plan's Task B5 signature (`git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md`, line 2336: `resolve_group_badge(*, role, cohort, tier)`) with `cohort` renamed `user_group` per D9; precedence `admin` → `user_group` when not `unknown` → `paid`/`free` (§6.2, D11).
- Produces:
  - `BillingLaneDay(day: date, platform_credits: int, byok: int)`; `AnalyticsOverview.billing_lane_mix: list[BillingLaneDay]`.
  - `OperationalReasonCount(reason_code: str, state: OperationalState, users: int)`; `OperationalAnalyticsResponse.top_operational_reasons: Sequence[OperationalReasonCount]`.
  - `LedgerDayTotal(day: date, amount_micro: int)`; `CommercialAnalyticsResponse.purchased_by_day` / `consumed_by_day: Sequence[LedgerDayTotal]`.
  - `UserValueFilters.role: Literal["user", "admin"] | None`.
  - `ValueUserListItem` and `ValueUserProfile`: `user_group: UserGroup`, `role: Literal["user", "admin"]`, `group_badge: str`, `last_meaningful_activity_at: datetime | None`.
  - Router: every handler takes its filters through a `Depends(...)` dependency whose parameters are declared `Query(...)`, so each appears in `app.openapi()`; `_query_values` and `_invalid_query` no longer exist; `InvalidAnalyticsQuery` (an `HTTPException` subclass with the fixed detail) replaces the latter.

Two things about the re-cut that a reader of `main@c3bbf2ed` would otherwise get wrong. First, PR B (D20) removed `user_state_counts` from `AnalyticsOverview` and the `status` query parameter (with `UserValueFilters.legacy_status`) from `/users`; the classes quoted below as "before" are the post-PR-B shapes, i.e. `main`'s text minus those two items. If your checkout still shows `user_state_counts` or `legacy_status`, PR B has not merged and this plan is not yet runnable (§13 ordering). Second, the display-safe 422 contract is load-bearing: `app.py:62` registers `validation_error_handler`, which for non-`/api/v2` paths returns `{"detail": exc.errors()}` — and Pydantic's error list **echoes the offending input**. `test_admin_analytics_rejects_invalid_queries_without_echo` asserts the canary never appears in the body. So every parameter is declared as `str | None = Query(default=None, ...)` — a shape FastAPI's own validation can never reject — and the value rules stay in Python, raising `InvalidAnalyticsQuery`. Do not "tidy" `limit: str | None` into `limit: int = Query(50, ge=1, le=100)`: the first out-of-range value would echo through the global handler.

- [ ] **Step 1: Write the failing tests**

In `dashboard/backend/tests/test_admin_analytics_api.py`, replace `test_admin_user_list_accepts_documented_filters` (lines 703-739; PR B already removed its `"status": "active"` param and `legacy_status` assertion) with:

```python
def test_admin_user_list_accepts_documented_filters(admin_analytics_api):
    api = admin_analytics_api
    today = datetime.now(timezone.utc).date()
    response = api["client"].get(
        "/api/admin/analytics/users",
        params={
            "q": "Subject",
            "role": "user",
            "lifecycle_segment": "core",
            "operational_state": "healthy",
            "commercial_tier": "invested",
            "user_group": "partner",
            "activated": "true",
            "last_meaningful_activity_from": (today - timedelta(days=1)).isoformat(),
            "last_meaningful_activity_to": today.isoformat(),
            "priority": "false",
            "limit": "1",
            "offset": "0",
        },
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    item = response.json()["items"][0]
    assert item["user_id"] == api["subject"]["id"]
    assert {"user_group", "role", "group_badge", "last_meaningful_activity_at"} <= item.keys()
    name, call = api["value_query_service"].calls[-1]
    assert name == "users"
    assert call["limit"] == 1
    assert call["offset"] == 0
    filters = call["filters"]
    assert filters.role == "user"
    assert filters.lifecycle_segment == "core"
    assert filters.operational_state == "healthy"
    assert filters.commercial_tier == "invested"
    assert filters.user_group == "partner"
    assert filters.activated is True
```

Replace `test_value_routes_reject_unknown_duplicate_and_unsafe_queries` (lines 823-844) with the three tests below. The `{"unknown": "value"}` case and the duplicated-`from` case go: declared parameters are FastAPI's declarative surface (§6.14 row 4), and FastAPI ignores undeclared query keys and reads the last of a repeated key; rejecting them was the hand-rolled parser's behaviour, and the parser is what this task deletes.

```python
@pytest.mark.parametrize(
    "path,params",
    [
        ("/api/admin/analytics/commercial", {"from": "2026-01-01", "to": "2026-08-01"}),
        ("/api/admin/analytics/operational", {"provider": "synthetic secret!"}),
        ("/api/admin/analytics/users", {"commercial_tier": "unsupported"}),
        ("/api/admin/analytics/users", {"role": "synthetic secret owner"}),
        ("/api/admin/analytics/users", {"limit": "synthetic secret 999"}),
        ("/api/admin/analytics/lifecycle", {"movement_range": "synthetic secret"}),
    ],
)
def test_value_routes_reject_unsafe_queries_without_echo(
    admin_analytics_api,
    path,
    params,
):
    api = admin_analytics_api
    response = api["client"].get(path, params=params, headers=api["admin_headers"])

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid Analytics query."}
    assert "synthetic secret" not in response.text


def test_undeclared_query_keys_are_ignored_not_rejected(admin_analytics_api):
    """Declared parameters mean FastAPI owns the query string (design §6.14).

    The hand-rolled parser 422'd on any key it did not know. That was the
    parser's rule, not the API's; with declared parameters an unknown key is
    simply not a parameter, and the route answers as if it were absent.
    """
    api = admin_analytics_api
    response = api["client"].get(
        "/api/admin/analytics/lifecycle",
        params={"unknown": "value"},
        headers=api["admin_headers"],
    )
    assert response.status_code == 200, response.text


def test_analytics_query_parameters_are_declared_in_openapi():
    schema = app.openapi()
    expected = {
        "/api/admin/analytics/overview": {"from", "to", "billing_mode", "provider", "model", "include_internal"},
        "/api/admin/analytics/lifecycle": {"from", "to", "include_internal", "movement_range"},
        "/api/admin/analytics/retention": {"from", "to", "include_internal"},
        "/api/admin/analytics/commercial": {"from", "to", "include_internal"},
        "/api/admin/analytics/operational": {"from", "to", "include_internal", "billing_mode", "provider", "model"},
        "/api/admin/analytics/groups": {"from", "to", "include_internal", "user_group"},
        "/api/admin/analytics/users": {
            "q", "role", "lifecycle_segment", "operational_state", "commercial_tier",
            "user_group", "activated", "last_meaningful_activity_from",
            "last_meaningful_activity_to", "priority", "limit", "offset", "include_internal",
        },
        "/api/admin/analytics/users/{user_id}": {"from", "to"},
        "/api/admin/analytics/users/{user_id}/activity": {"section", "limit", "cursor"},
    }
    for path, names in expected.items():
        parameters = schema["paths"][path]["get"].get("parameters", [])
        declared = {p["name"] for p in parameters if p["in"] == "query"}
        assert declared == names, (path, declared)
    assert "status" not in {
        p["name"] for p in schema["paths"]["/api/admin/analytics/users"]["get"]["parameters"]
    }


def test_recut_fields_are_on_every_response(admin_analytics_api):
    api = admin_analytics_api
    subject_id = api["subject"]["id"]
    overview = api["client"].get(
        "/api/admin/analytics/overview", headers=api["admin_headers"]
    ).json()
    assert isinstance(overview["billing_lane_mix"], list)
    for row in overview["billing_lane_mix"]:
        assert set(row) == {"day", "platform_credits", "byok"}
    operational = api["client"].get(
        "/api/admin/analytics/operational", headers=api["admin_headers"]
    ).json()
    assert operational["top_operational_reasons"] == [
        {"reason_code": "no_usable_billing_lane", "state": "blocked", "users": 2},
        {"reason_code": "credential_invalid", "state": "needs_attention", "users": 4},
    ]
    commercial = api["client"].get(
        "/api/admin/analytics/commercial", headers=api["admin_headers"]
    ).json()
    assert commercial["purchased_by_day"] == [{"day": "2026-09-01", "amount_micro": 5000000}, {"day": "2026-09-02", "amount_micro": 7000000}]
    assert commercial["consumed_by_day"] == [{"day": "2026-09-01", "amount_micro": 1800000}, {"day": "2026-09-02", "amount_micro": 3000000}]
    profile = api["client"].get(
        f"/api/admin/analytics/users/{subject_id}", headers=api["admin_headers"]
    ).json()
    assert profile["user_group"] == "invited"
    assert profile["role"] == "user"
    assert profile["group_badge"] == "invited"
    assert profile["last_meaningful_activity_at"] == "2026-08-22T11:20:00Z"
    # D15: the fields stay in the payload (collection is a separate decision, §14);
    # only their display leaves the page (Task 8 renders none of them).
    assert {"country_code", "device_category", "browser_family", "top_product_page"} <= profile.keys()
```

The `overview` assertion is structural because the fixture's real `AnalyticsQueryService` runs over a store with no `model_usage_recorded` rows, so the list may be empty; the value routes go through `FixtureValueQueryService`, whose fixtures gain the exact rows asserted above in Step 3.

- [ ] **Step 2: Run them and confirm the expected failures**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py -v -k "documented_filters or unsafe_queries or undeclared or openapi or recut"
```

Expected: **FAIL**. `test_admin_user_list_accepts_documented_filters` fails at the request with 422 (`role` is not in `_value_user_filters`' allowed set, so `_query_values` rejects it). `test_undeclared_query_keys_are_ignored_not_rejected` fails with 422 for the same reason. `test_analytics_query_parameters_are_declared_in_openapi` fails at the first path with `declared == set()` (today every handler reads `request.query_params`; nothing is declared). `test_recut_fields_are_on_every_response` fails with `KeyError: 'billing_lane_mix'`. `test_value_routes_reject_unsafe_queries_without_echo[role]` fails with 422 for the wrong reason today (unknown key) — it goes green only once `role` is declared and validated.

- [ ] **Step 3: Add the models and populate them**

`dashboard/backend/domain/analytics/query_service.py`, line 8, change

```python
from datetime import datetime, timedelta, timezone
```

to

```python
from datetime import date, datetime, timedelta, timezone
```

After `FailureCategoryCount` (lines 90-94) add:

```python
class BillingLaneDay(BaseModel):
    """One UTC day of run counts by billing lane, from rollups `billing_mode` (D14)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    day: date
    platform_credits: int = Field(ge=0)
    byok: int = Field(ge=0)
```

In `AnalyticsOverview` (post-PR-B shape), change

```python
    activation_funnel: dict[str, int]
    top_failure_categories: list[FailureCategoryCount]
    users_needing_attention: list[AnalyticsUserListItem]
```

to

```python
    activation_funnel: dict[str, int]
    billing_lane_mix: list[BillingLaneDay] = Field(default_factory=list)
    top_failure_categories: list[FailureCategoryCount]
    users_needing_attention: list[AnalyticsUserListItem]
```

At the end of `AnalyticsQueryService.get_overview` (the `return AnalyticsOverview(` call), add the keyword argument, converting PR B's dict to rows sorted by day:

```python
            billing_lane_mix=[
                BillingLaneDay(
                    day=date.fromisoformat(day),
                    platform_credits=int(lanes.get("platform_credits", 0)),
                    byok=int(lanes.get("byok", 0)),
                )
                for day, lanes in sorted(self.billing_lane_mix(filters=filters).items())
            ],
```

Add `"BillingLaneDay"` to `query_service.py`'s `__all__` (the module exports its response models by name; place it alphabetically after `"AnalyticsUserProfile"`).

`dashboard/backend/domain/analytics/value_queries.py`. Extend the `.lifecycle` import (lines 10-17):

```python
from .lifecycle import (
    CommercialTier,
    LifecycleResult,
    LifecycleSegment,
    OperationalResult,
    OperationalState,
    is_lifecycle_activity,
    resolve_group_badge,
)
```

Before `CommercialAnalyticsResponse` (line 253) add:

```python
class LedgerDayTotal(BaseModel):
    """One UTC day of settled ledger movement, from `CreditsStore.sum_ledger_by_day`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    day: date
    amount_micro: int = Field(ge=0)


class OperationalReasonCount(BaseModel):
    """How many users yesterday's facts row put in a non-healthy state for one reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    state: OperationalState
    users: int = Field(ge=0)
```

Change `CommercialAnalyticsResponse` (lines 253-261) from

```python
class CommercialAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    tier_counts: dict[CommercialTier, int]
    lifetime_net_purchased_micro: int = Field(ge=0)
    selected_period: CommercialPeriodSummary
    current_balances: BalanceTotals
    availability: SectionAvailability
```

to

```python
class CommercialAnalyticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    tier_counts: dict[CommercialTier, int]
    lifetime_net_purchased_micro: int = Field(ge=0)
    selected_period: CommercialPeriodSummary
    current_balances: BalanceTotals
    purchased_by_day: Sequence[LedgerDayTotal] = Field(default_factory=tuple)
    consumed_by_day: Sequence[LedgerDayTotal] = Field(default_factory=tuple)
    availability: SectionAvailability
```

Change `OperationalAnalyticsResponse` (lines 264-276) so the line

```python
    top_failure_categories: Sequence[FailureCategoryCount]
```

becomes

```python
    top_failure_categories: Sequence[FailureCategoryCount]
    top_operational_reasons: Sequence[OperationalReasonCount] = Field(default_factory=tuple)
```

In `UserValueFilters` (line 305 onwards; PR B removed `legacy_status` and its validator branch), change

```python
    q: str | None = Field(default=None, max_length=100)
    lifecycle_segment: LifecycleSegment | None = None
```

to

```python
    q: str | None = Field(default=None, max_length=100)
    role: Literal["user", "admin"] | None = None
    lifecycle_segment: LifecycleSegment | None = None
```

Change `ValueUserListItem` (lines 365-377) from

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
    priority_group: PriorityGroup
    profile_path: str
```

to

```python
class ValueUserListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: int = Field(gt=0)
    display_name: str
    email: str
    joined_at: datetime
    user_group: UserGroup
    role: Literal["user", "admin"]
    group_badge: str = Field(min_length=1, max_length=32)
    last_meaningful_activity_at: datetime | None = None
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial_tier: CommercialTier
    lifetime_net_purchased_micro: int = Field(ge=0)
    priority_group: PriorityGroup
    profile_path: str
```

Change `ValueUserProfile` (lines 389-395) from

```python
class ValueUserProfile(AnalyticsUserProfile):
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial: CommercialValueFact
    selected_period_start: date
    selected_period_end: date
    recent_lifecycle_transitions: Sequence[LifecycleTransition]
```

to

```python
class ValueUserProfile(AnalyticsUserProfile):
    user_group: UserGroup
    role: Literal["user", "admin"]
    group_badge: str = Field(min_length=1, max_length=32)
    last_meaningful_activity_at: datetime | None = None
    lifecycle: LifecycleResult
    operational: OperationalResult
    commercial: CommercialValueFact
    selected_period_start: date
    selected_period_end: date
    recent_lifecycle_transitions: Sequence[LifecycleTransition]
```

Populate the list item. PR B moved `list_users` onto SQL, so the construction site has moved from `main`'s lines 1319-1331; find it with `rg -n "ValueUserListItem\(" dashboard/backend/domain/analytics/value_queries.py`. Wherever it is, the row that reaches it carries the `users` columns (`role`, `user_group`) and the user's `snapshot`/facts and `fact` (commercial). Replace

```python
            selected.append(
                ValueUserListItem(
                    user_id=user_id,
                    display_name=str(user.get("display_name") or ""),
                    email=str(user.get("email") or ""),
                    joined_at=_parse_timestamp(user["created_at"]),
                    lifecycle=_lifecycle(snapshot),
                    operational=_operational(snapshot),
                    commercial_tier=fact.commercial_tier,
                    lifetime_net_purchased_micro=fact.lifetime_net_purchased_micro,
                    priority_group=group,
                    profile_path=f"/admin/analytics/users/{user_id}",
                )
            )
```

with

```python
            role = "admin" if user.get("role") == "admin" else "user"
            selected.append(
                ValueUserListItem(
                    user_id=user_id,
                    display_name=str(user.get("display_name") or ""),
                    email=str(user.get("email") or ""),
                    joined_at=_parse_timestamp(user["created_at"]),
                    user_group=user_group,
                    role=role,
                    group_badge=resolve_group_badge(
                        role=role, user_group=user_group, tier=fact.commercial_tier
                    ),
                    last_meaningful_activity_at=snapshot.last_meaningful_activity_at,
                    lifecycle=_lifecycle(snapshot),
                    operational=_operational(snapshot),
                    commercial_tier=fact.commercial_tier,
                    lifetime_net_purchased_micro=fact.lifetime_net_purchased_micro,
                    priority_group=group,
                    profile_path=f"/admin/analytics/users/{user_id}",
                )
            )
```

and add the `role` predicate beside the existing `user_group` one (in the SQL `WHERE` PR B built, or — if the checkout still filters in Python — immediately after `if filters.user_group is not None and user_group != filters.user_group: continue`):

```python
            if filters.role is not None and role != filters.role:
                continue
```

(`user_group` is already `coerce_user_group(user.get("user_group"))` two lines above the construction on `main`; the SQL version selects the same column.)

Populate the profile. In `get_user_profile` (lines 1351-1390), after

```python
        snapshot = self.value_store.get_current_snapshot(subject_id)
        if snapshot is None:
            raise LookupError("Analytics value snapshot was not found")
```

(PR B renamed the store read onto `user_activity`; the variable that carries `last_meaningful_activity_at` is what matters) add

```python
        user = self.user_store.get_user_admin(subject_id)
        if user is None:
            raise LookupError("Analytics user was not found")
        user_group = coerce_user_group(user.get("user_group"))
        role = "admin" if user.get("role") == "admin" else "user"
```

and change the return from

```python
        return ValueUserProfile(
            **legacy.model_dump(),
            lifecycle=_lifecycle(snapshot),
            operational=_operational(snapshot),
            commercial=commercial,
            selected_period_start=start,
            selected_period_end=end,
            recent_lifecycle_transitions=transitions,
        )
```

to

```python
        return ValueUserProfile(
            **legacy.model_dump(),
            user_group=user_group,
            role=role,
            group_badge=resolve_group_badge(
                role=role, user_group=user_group, tier=commercial.commercial_tier
            ),
            last_meaningful_activity_at=snapshot.last_meaningful_activity_at,
            lifecycle=_lifecycle(snapshot),
            operational=_operational(snapshot),
            commercial=commercial,
            selected_period_start=start,
            selected_period_end=end,
            recent_lifecycle_transitions=transitions,
        )
```

Wire the two remaining §9 fields at the end of `get_operational` and `get_commercial` (their `return OperationalAnalyticsResponse(` / `return CommercialAnalyticsResponse(` calls):

```python
            top_operational_reasons=[
                OperationalReasonCount(reason_code=reason_code, state=state, users=users)
                for reason_code, state, users in self.top_operational_reasons(
                    include_internal=include_internal
                )
            ],
```

```python
            purchased_by_day=[
                LedgerDayTotal(day=date.fromisoformat(day), amount_micro=amount)
                for day, amount in sorted(
                    self.purchased_by_day(start=start, end=end, include_internal=include_internal).items()
                )
            ],
            consumed_by_day=[
                LedgerDayTotal(day=date.fromisoformat(day), amount_micro=amount)
                for day, amount in sorted(
                    self.consumed_by_day(start=start, end=end, include_internal=include_internal).items()
                )
            ],
```

Add `"LedgerDayTotal"` and `"OperationalReasonCount"` to `__all__` (alphabetical: after `"GroupAnalyticsResponse"` and after `"OperationalAnalyticsResponse"` respectively).

- [ ] **Step 4: Rewrite the router onto declared parameters**

Replace `dashboard/backend/api/routers/admin_analytics.py` in full with the module below. What it keeps byte-for-byte: every constant, every regex, every validation rule (`_parse_date`'s exact-round-trip check, the 20-character integer guard, the 180-day cap via `MAX_VALUE_RANGE_DAYS`, the `to < from` reversal check, `_exclusive_date_end`'s `OverflowError` → 422), every 422/404/503 detail string, `_record_access`, and the handlers' `try/except → _raise_service_error` shape. What changes: `Request`-reading helpers become dependency functions with `Query(...)` parameters (aliases for `from`; `str | None` for everything so FastAPI's own validation can never echo a value — see the note above Step 1); `_invalid_query()` becomes `raise InvalidAnalyticsQuery()`; `_query_values`, `_user_filters` and the `AnalyticsUserFilters` import are gone (PR A's D24 deleted the dead users-list stack; this rewrite must not re-import it). `_raise_service_error` is shown in its PR A (D24) form — a `print` before the 503 and no `from None`; if PR A's merged body differs in wording, keep PR A's body.

```python
"""Admin-only, display-safe Analytics query endpoints.

Every query parameter is declared (so it appears in ``app.openapi()``), typed
``str | None`` on purpose, and validated in Python: the global
``RequestValidationError`` handler echoes offending input back to the caller,
and these routes must never echo a value an admin typed (it may be a pasted
secret). Every validation failure is the one fixed body,
``{"detail": "Invalid Analytics query."}``.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Never

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from dashboard.backend.api.auth import require_admin
from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
    AnalyticsQueryService,
    get_analytics_query_service,
    get_value_analytics_query_service,
)
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)
from dashboard.backend.domain.analytics.value_queries import (
    CommercialAnalyticsResponse,
    GroupAnalyticsResponse,
    LifecycleAnalyticsResponse,
    MAX_VALUE_RANGE_DAYS,
    OperationalAnalyticsResponse,
    PaginatedValueUsers,
    RetentionAnalyticsResponse,
    UserValueFilters,
    ValueAnalyticsQueryService,
    ValueUserProfile,
)
from dashboard.backend.domain.user_groups import parse_user_group


router = APIRouter(
    prefix="/admin/analytics",
    tags=["admin-analytics"],
    dependencies=[Depends(require_admin)],
)

_INVALID_QUERY_DETAIL = "Invalid Analytics query."
_NOT_FOUND_DETAIL = "Analytics user was not found."
_UNAVAILABLE_DETAIL = "Analytics is temporarily unavailable."
_PROVIDER_ID_PATTERN = re.compile(r"^[a-z0-9_]{2,64}$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-:]{0,255}$")
_POSITIVE_INTEGER_PATTERN = re.compile(r"^[0-9]+$")
_ROLES = {"user", "admin"}
_ACTIVITY_SECTIONS = {"timeline", "runs", "usage", "sessions"}
_LIFECYCLE_SEGMENTS = {"new", "onboarding", "growing", "core", "at_risk", "dormant"}
_OPERATIONAL_STATES = {"blocked", "needs_attention", "healthy"}
_COMMERCIAL_TIERS = {"unpaid", "starter", "invested", "high_value"}
_LIFECYCLE_MOVEMENT_RANGES = {"5d", "1w", "1m", "1y"}
_BILLING_MODES = {"byok", "platform_credits"}


class InvalidAnalyticsQuery(HTTPException):
    """The one 422 these routes ever raise; carries no caller input."""

    def __init__(self) -> None:
        super().__init__(status_code=422, detail=_INVALID_QUERY_DETAIL)


def _parse_date(value: str) -> date:
    if len(value) != 10:
        raise InvalidAnalyticsQuery()
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise InvalidAnalyticsQuery() from None
    if parsed.isoformat() != value:
        raise InvalidAnalyticsQuery()
    return parsed


def _utc_midnight(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _exclusive_date_end(value: date) -> datetime:
    try:
        return _utc_midnight(value + timedelta(days=1))
    except OverflowError:
        raise InvalidAnalyticsQuery() from None


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise InvalidAnalyticsQuery()


def _optional_bool(value: str | None, default: bool) -> bool:
    return _parse_bool(value) if value is not None else default


def _parse_integer(
    value: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if len(value) > 20 or not _POSITIVE_INTEGER_PATTERN.fullmatch(value):
        raise InvalidAnalyticsQuery()
    parsed = int(value)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise InvalidAnalyticsQuery()
    return parsed


def _parse_user_id(value: str) -> int:
    return _parse_integer(value, minimum=1)


def _one_of(value: str | None, allowed: set[str]) -> str | None:
    if value is not None and value not in allowed:
        raise InvalidAnalyticsQuery()
    return value


def _provider_id(value: str | None) -> str | None:
    if value is not None and not _PROVIDER_ID_PATTERN.fullmatch(value):
        raise InvalidAnalyticsQuery()
    return value


def _model_id(value: str | None) -> str | None:
    if value is not None and not _MODEL_ID_PATTERN.fullmatch(value):
        raise InvalidAnalyticsQuery()
    return value


def _user_group(value: str | None):
    if value is None:
        return None
    try:
        return parse_user_group(value)
    except ValueError:
        raise InvalidAnalyticsQuery() from None


def _ordered_dates(from_: str | None, to: str | None) -> tuple[date | None, date | None]:
    from_date = _parse_date(from_) if from_ is not None else None
    to_date = _parse_date(to) if to is not None else None
    if from_date is not None and to_date is not None and to_date < from_date:
        raise InvalidAnalyticsQuery()
    return from_date, to_date


def _overview_filters(
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    billing_mode: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    include_internal: str | None = Query(default=None),
) -> AnalyticsMetricFilters:
    now = datetime.now(timezone.utc)
    from_date, to_date = _ordered_dates(from_, to)
    end = _exclusive_date_end(to_date) if to_date else now
    try:
        start = _utc_midnight(from_date) if from_date else end - timedelta(days=30)
    except OverflowError:
        raise InvalidAnalyticsQuery() from None
    try:
        return AnalyticsMetricFilters(
            start=start,
            end=end,
            billing_mode=_one_of(billing_mode, _BILLING_MODES),
            provider_id=_provider_id(provider),
            model_id=_model_id(model),
            include_internal=_optional_bool(include_internal, False),
        )
    except (ValidationError, ValueError):
        raise InvalidAnalyticsQuery() from None


def _value_dates(from_: str | None, to: str | None) -> tuple[date, date]:
    today = datetime.now(timezone.utc).date()
    from_date, to_date = _ordered_dates(from_, to)
    try:
        end = (to_date or today) + timedelta(days=1)
        start = from_date or end - timedelta(days=30)
    except OverflowError:
        raise InvalidAnalyticsQuery() from None
    if end <= start or (end - start).days > MAX_VALUE_RANGE_DAYS:
        raise InvalidAnalyticsQuery()
    return start, end


class ValueRange:
    """`from`/`to`/`include_internal` resolved to a validated `[start, end)` window."""

    __slots__ = ("start", "end", "include_internal")

    def __init__(self, start: date, end: date, include_internal: bool) -> None:
        self.start = start
        self.end = end
        self.include_internal = include_internal


def _value_range(
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    include_internal: str | None = Query(default=None),
) -> ValueRange:
    start, end = _value_dates(from_, to)
    return ValueRange(start, end, _optional_bool(include_internal, False))


def _profile_range(
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
) -> tuple[date, date]:
    return _value_dates(from_, to)


def _movement_range(movement_range: str | None = Query(default=None)) -> str:
    return _one_of(movement_range, _LIFECYCLE_MOVEMENT_RANGES) or "5d"


class OperationalDimensions:
    __slots__ = ("billing_mode", "provider_id", "model_id")

    def __init__(self, billing_mode, provider_id, model_id) -> None:
        self.billing_mode = billing_mode
        self.provider_id = provider_id
        self.model_id = model_id


def _operational_dimensions(
    billing_mode: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
) -> OperationalDimensions:
    return OperationalDimensions(
        _one_of(billing_mode, _BILLING_MODES),
        _provider_id(provider),
        _model_id(model),
    )


def _selected_group(user_group: str | None = Query(default=None)):
    return _user_group(user_group)


class UserListQuery:
    __slots__ = ("filters", "limit", "offset")

    def __init__(self, filters: UserValueFilters, limit: int, offset: int) -> None:
        self.filters = filters
        self.limit = limit
        self.offset = offset


def _value_user_filters(
    q: str | None = Query(default=None),
    role: str | None = Query(default=None),
    lifecycle_segment: str | None = Query(default=None),
    operational_state: str | None = Query(default=None),
    commercial_tier: str | None = Query(default=None),
    user_group: str | None = Query(default=None),
    activated: str | None = Query(default=None),
    last_meaningful_activity_from: str | None = Query(default=None),
    last_meaningful_activity_to: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    offset: str | None = Query(default=None),
    include_internal: str | None = Query(default=None),
) -> UserListQuery:
    if q is not None and len(q) > 100:
        raise InvalidAnalyticsQuery()
    from_date, to_date = _ordered_dates(
        last_meaningful_activity_from, last_meaningful_activity_to
    )
    try:
        filters = UserValueFilters(
            q=q,
            role=_one_of(role, _ROLES),
            lifecycle_segment=_one_of(lifecycle_segment, _LIFECYCLE_SEGMENTS),
            operational_state=_one_of(operational_state, _OPERATIONAL_STATES),
            commercial_tier=_one_of(commercial_tier, _COMMERCIAL_TIERS),
            user_group=_user_group(user_group),
            activated=_parse_bool(activated) if activated is not None else None,
            last_meaningful_activity_from=(
                _utc_midnight(from_date) if from_date is not None else None
            ),
            last_meaningful_activity_to=(
                _exclusive_date_end(to_date) - timedelta(microseconds=1)
                if to_date is not None
                else None
            ),
            priority=_optional_bool(priority, False),
            include_internal=_optional_bool(include_internal, False),
        )
    except (ValidationError, ValueError):
        raise InvalidAnalyticsQuery() from None
    return UserListQuery(
        filters,
        _parse_integer(limit if limit is not None else "50", minimum=1, maximum=100),
        _parse_integer(offset if offset is not None else "0", minimum=0),
    )


class ActivityQuery:
    __slots__ = ("section", "limit", "cursor")

    def __init__(self, section: str, limit: int, cursor: str | None) -> None:
        self.section = section
        self.limit = limit
        self.cursor = cursor


def _activity_query(
    section: str | None = Query(default=None),
    limit: str | None = Query(default=None),
    cursor: str | None = Query(default=None),
) -> ActivityQuery:
    if section not in _ACTIVITY_SECTIONS:
        raise InvalidAnalyticsQuery()
    if cursor is not None and (not cursor or len(cursor) > 256):
        raise InvalidAnalyticsQuery()
    return ActivityQuery(
        section,
        _parse_integer(limit if limit is not None else "50", minimum=1, maximum=100),
        cursor,
    )


def _raise_service_error(exc: Exception) -> Never:
    if isinstance(exc, LookupError):
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)
    if isinstance(exc, (ValidationError, ValueError)):
        raise HTTPException(status_code=422, detail=_INVALID_QUERY_DETAIL)
    print(
        f"ERROR: admin_analytics.service_failure category={type(exc).__name__}",
        flush=True,
    )
    raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL)


def _record_access(
    service: AnalyticsService,
    *,
    admin: dict,
    subject_user_id: int,
    section: str,
) -> None:
    try:
        service.record_admin_profile_access(
            actor=admin,
            subject_user_id=subject_user_id,
            section=section,
        )
    except Exception:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_DETAIL) from None


@router.get("/overview", response_model=AnalyticsOverview)
def get_overview(
    filters: AnalyticsMetricFilters = Depends(_overview_filters),
    service: AnalyticsQueryService = Depends(get_analytics_query_service),
):
    try:
        return service.get_overview(filters=filters)
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/lifecycle", response_model=LifecycleAnalyticsResponse)
def get_lifecycle(
    window: ValueRange = Depends(_value_range),
    movement_range: str = Depends(_movement_range),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.get_lifecycle(
            start=window.start,
            end=window.end,
            include_internal=window.include_internal,
            movement_range=movement_range,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/retention", response_model=RetentionAnalyticsResponse)
def get_retention(
    window: ValueRange = Depends(_value_range),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.get_retention(
            start=window.start,
            end=window.end,
            include_internal=window.include_internal,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/commercial", response_model=CommercialAnalyticsResponse)
def get_commercial(
    window: ValueRange = Depends(_value_range),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.get_commercial(
            start=window.start,
            end=window.end,
            include_internal=window.include_internal,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/operational", response_model=OperationalAnalyticsResponse)
def get_operational(
    window: ValueRange = Depends(_value_range),
    dimensions: OperationalDimensions = Depends(_operational_dimensions),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.get_operational(
            start=window.start,
            end=window.end,
            include_internal=window.include_internal,
            billing_mode=dimensions.billing_mode,
            provider_id=dimensions.provider_id,
            model_id=dimensions.model_id,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/groups", response_model=GroupAnalyticsResponse)
def get_groups(
    window: ValueRange = Depends(_value_range),
    selected_group=Depends(_selected_group),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.get_groups(
            start=window.start,
            end=window.end,
            include_internal=window.include_internal,
            user_group=selected_group,
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users", response_model=PaginatedValueUsers)
def list_users(
    query: UserListQuery = Depends(_value_user_filters),
    service: ValueAnalyticsQueryService = Depends(get_value_analytics_query_service),
):
    try:
        return service.list_users(
            filters=query.filters, limit=query.limit, offset=query.offset
        )
    except Exception as exc:
        _raise_service_error(exc)


@router.get("/users/{user_id}", response_model=ValueUserProfile)
def get_user_profile(
    user_id: str,
    window: tuple[date, date] = Depends(_profile_range),
    admin: dict = Depends(require_admin),
    query_service: ValueAnalyticsQueryService = Depends(
        get_value_analytics_query_service
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    start, end = window
    subject_user_id = _parse_user_id(user_id)
    try:
        profile = query_service.get_user_profile(
            user_id=subject_user_id,
            start=start,
            end=end,
        )
    except Exception as exc:
        _raise_service_error(exc)
    _record_access(
        analytics_service,
        admin=admin,
        subject_user_id=subject_user_id,
        section="overview",
    )
    return profile


@router.get("/users/{user_id}/activity", response_model=AnalyticsActivityPage)
def get_user_activity(
    user_id: str,
    activity_query: ActivityQuery = Depends(_activity_query),
    admin: dict = Depends(require_admin),
    query_service: AnalyticsQueryService = Depends(get_analytics_query_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    subject_user_id = _parse_user_id(user_id)
    try:
        activity = query_service.get_user_activity(
            user_id=subject_user_id,
            section=activity_query.section,
            limit=activity_query.limit,
            cursor=activity_query.cursor,
        )
    except Exception as exc:
        _raise_service_error(exc)
    _record_access(
        analytics_service,
        admin=admin,
        subject_user_id=subject_user_id,
        section=activity_query.section,
    )
    return activity


__all__ = ["router"]
```

Handler names are unchanged (`get_overview`, `get_lifecycle`, `get_retention`, `get_commercial`, `get_operational`, `get_groups`, `list_users`, `get_user_profile`, `get_user_activity`), so `EXPECTED_ADMIN_ANALYTICS_ROUTES` in `test_app_composition.py` (lines 70-84) does not change. Run `rg -n "_query_values|_invalid_query|AnalyticsUserFilters|request.query_params" dashboard/backend/api/routers/admin_analytics.py` and confirm zero hits.

- [ ] **Step 5: Update the fixtures**

`tests/fixtures/admin_analytics/overview.json` and `overview_partial_error.json`: after the `"activation_funnel": {...}` object add

```json
  "billing_lane_mix": [
    {"day": "2026-08-25", "platform_credits": 11, "byok": 7},
    {"day": "2026-08-26", "platform_credits": 14, "byok": 8}
  ],
```

(in `overview_partial_error.json` use `"billing_lane_mix": [],` — the growth panel is unavailable there).

`operational.json`: after the `"top_failure_categories": [...]` array add

```json
  "top_operational_reasons": [
    {"reason_code": "no_usable_billing_lane", "state": "blocked", "users": 2},
    {"reason_code": "credential_invalid", "state": "needs_attention", "users": 4}
  ],
```

`commercial.json`: after the `"current_balances": {...}` object add

```json
  "purchased_by_day": [
    {"day": "2026-09-01", "amount_micro": 5000000},
    {"day": "2026-09-02", "amount_micro": 7000000}
  ],
  "consumed_by_day": [
    {"day": "2026-09-01", "amount_micro": 1800000},
    {"day": "2026-09-02", "amount_micro": 3000000}
  ],
```

`users.json`: in item `101` after `"joined_at": "2026-07-01T09:00:00Z",` add

```json
      "user_group": "invited",
      "role": "user",
      "group_badge": "invited",
      "last_meaningful_activity_at": "2026-08-22T11:20:00Z",
```

and in item `102` after its `joined_at` add

```json
      "user_group": "unknown",
      "role": "user",
      "group_badge": "free",
      "last_meaningful_activity_at": "2026-09-01T08:00:00Z",
```

(`102` is `unpaid` with `unknown` group, so the server badge is `free` — the fixture records what the server would compute; nothing client-side derives it.)

`user_detail.json`: after `"last_meaningful_activity": "2026-08-22T11:20:00Z",` add

```json
  "user_group": "invited",
  "role": "user",
  "group_badge": "invited",
  "last_meaningful_activity_at": "2026-08-22T11:20:00Z",
```

- [ ] **Step 6: Run the suite for this surface and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models dashboard/backend/tests/test_app_composition.py::test_admin_analytics_router_contract dashboard/backend/tests/domain/analytics -v
```

Expected: **PASS** across the file. `test_admin_analytics_rejects_invalid_queries_without_echo` still passes (its `provider` canary is rejected by `_provider_id`, never by FastAPI). `test_fixtures_validate_against_committed_analytics_models` passes because every fixture now carries the required new fields. `test_admin_analytics_router_contract` passes because the nine `(method, path, name)` triples are unchanged.

- [ ] **Step 7: Commit**

```bash
git add dashboard/backend/domain/analytics/query_service.py dashboard/backend/domain/analytics/value_queries.py dashboard/backend/api/routers/admin_analytics.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/fixtures/admin_analytics/overview.json dashboard/backend/tests/fixtures/admin_analytics/overview_partial_error.json dashboard/backend/tests/fixtures/admin_analytics/operational.json dashboard/backend/tests/fixtures/admin_analytics/commercial.json dashboard/backend/tests/fixtures/admin_analytics/users.json dashboard/backend/tests/fixtures/admin_analytics/user_detail.json
git commit -m "$(cat <<'EOF'
feat: re-cut the admin analytics contract onto declared query parameters

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `GET /api/admin/stats` gains `max_active_dashboard_backtests`

**Files:**
- Modify: `dashboard/backend/api/routers/admin_users.py` — `admin_stats`, lines 176-192.
- Create: `dashboard/backend/tests/fixtures/admin_analytics/admin_stats.json`.
- Test: `dashboard/backend/tests/test_admin_users.py` — `test_admin_stats_endpoint`, lines 290-314 (extend).

**Interfaces:**
- Consumes: `dashboard.backend.api.routers.backtests.MAX_ACTIVE_DASHBOARD_BACKTESTS` (the module constant parsed once at import by `_max_active_dashboard_backtests()`, `backtests.py:556-590`; default 5).
- Produces: the stats dict gains `"max_active_dashboard_backtests": int`. The six existing keys are unchanged (§10.4).

- [ ] **Step 1: Write the failing test**

In `dashboard/backend/tests/test_admin_users.py`, change the tail of `test_admin_stats_endpoint` from

```python
    assert body["agents"] >= 1
    assert "active_dashboard_backtests" in body
```

to

```python
    assert body["agents"] >= 1
    assert "active_dashboard_backtests" in body
    # The live row's real slot ceiling (design D17): the parsed constant, not
    # the mock's "24 slots". Exposed here because this is the live-operations
    # route (D16); the analytics routes never carry sub-day numbers.
    from dashboard.backend.api.routers.backtests import MAX_ACTIVE_DASHBOARD_BACKTESTS

    assert body["max_active_dashboard_backtests"] == MAX_ACTIVE_DASHBOARD_BACKTESTS
    assert set(body) == {
        "users", "admins", "agents", "active_dashboard_backtests",
        "max_active_dashboard_backtests", "credits_metering_enabled", "default_credits",
    }
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_users.py::test_admin_stats_endpoint -v
```

Expected: **FAIL** with `KeyError: 'max_active_dashboard_backtests'`.

- [ ] **Step 3: Add the key**

In `dashboard/backend/api/routers/admin_users.py`, change

```python
    from dashboard.backend.api.routers.backtests import count_active_dashboard_backtests
```

to

```python
    from dashboard.backend.api.routers.backtests import (
        MAX_ACTIVE_DASHBOARD_BACKTESTS,
        count_active_dashboard_backtests,
    )
```

and

```python
        "active_dashboard_backtests": count_active_dashboard_backtests(),
```

to

```python
        "active_dashboard_backtests": count_active_dashboard_backtests(),
        # Per-process slot ceiling (`MAX_ACTIVE_DASHBOARD_BACKTESTS`, default 5).
        # The /admin live row prints "running / ceiling"; a second replica would
        # under-report both, which the row's caveat says (design D17).
        "max_active_dashboard_backtests": MAX_ACTIVE_DASHBOARD_BACKTESTS,
```

Create `dashboard/backend/tests/fixtures/admin_analytics/admin_stats.json` (the committed payload the live-row renderer test in Task 6 feeds; shaped exactly like the route's dict):

```json
{
  "users": 412,
  "admins": 3,
  "agents": 4380,
  "active_dashboard_backtests": 2,
  "max_active_dashboard_backtests": 5,
  "credits_metering_enabled": false,
  "default_credits": 100
}
```

- [ ] **Step 4: Run it and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_users.py::test_admin_stats_endpoint dashboard/backend/tests/test_credit_metering.py -v -k "stats or metering_enabled"
```

Expected: **PASS**. `test_credit_metering.py:447-463` reads `credits_metering_enabled` and asserts membership, not equality of the whole dict, so the added key does not disturb it. The old console's `loadAdminStats()` (`app.js:4253-4278`) reads only the `data-stat` keys present in `#adminStats`, so an extra key is ignored there.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/admin_users.py dashboard/backend/tests/test_admin_users.py dashboard/backend/tests/fixtures/admin_analytics/admin_stats.json
git commit -m "$(cat <<'EOF'
feat: report the dashboard backtest slot ceiling on /api/admin/stats

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Serving and topology — `/admin` on Render and Vercel, `/admin-analytics` → 308, middleware exemption

**Files:**
- Modify: `dashboard/backend/app.py` — `serve_admin_analytics`, lines 384-392.
- Modify: `dashboard/backend/middleware.py` — `EXEMPT_PATHS`, lines 13-31 (the `/admin-analytics` entry at line 30).
- Modify: `dashboard/frontend/vercel.json` — whole file (46 lines).
- Modify: `dashboard/backend/tests/test_app_composition.py` — `EXPECTED_FULL_CONTRACT`, the `("GET", "/strategy")` … `("GET", "/admin-analytics")` neighbourhood at lines 281-282.
- Modify: `dashboard/backend/tests/test_vercel_cache_headers.py` — extend (62 lines).
- Create: `dashboard/backend/tests/test_middleware_exemptions.py`.
- Test: the three test files above.

**Interfaces:**
- Consumes: `FileResponse`, `RedirectResponse`, `Request` (already imported in `app.py:10-14`).
- Produces: `GET /admin` → `admin.html`; `GET /admin.css` → `admin.css` (`text/css`); `GET /admin-analytics` → 308 to `/admin` preserving the query string; `is_exempt("/admin") is True`, `is_exempt("/admin-analytics") is True`; Vercel: `cleanUrls` serves `admin.html` at `/admin`, the `/admin/:path*` rewrite is gone, a permanent redirect sends `/admin-analytics` to `/admin`.

Why the Vercel header rules are ordered the way they are: Vercel applies every matching `headers` entry and the **last** match wins per key (`test_catch_all_precedes_the_cache_control_overrides` documents this). The existing API rule `"/(api|paper|backtest|runs|config|admin|ticker|health|compare)(/.*)?"` sets `no-store`, and because its trailing group is optional it matches the bare `/admin` too — an accident of the `/admin/runs/{run_id}` API prefix sharing the page's path. The two new rules for `/admin` and `/admin.html` therefore go **after** that rule so the shell gets the same `public, max-age=0, must-revalidate` policy every other HTML shell here has (`/`, `/app`, `/app.html`, and until now `/admin-analytics`): the HTML carries no data (D7), so a conditional revalidation is safe, and one policy for all shells is one fewer thing to explain. Design §7.1 notes the no-store accident as acceptable; this plan makes the policy explicit instead, and the order is what makes it take effect.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_middleware_exemptions.py`:

```python
"""The session middleware must let the admin shell and its redirect through.

`SessionMiddleware` demands `X-Session-Id` on every path that is not exempt, an
API route, or a paper-trading route. A page load sends no such header, so a
page path missing from `EXEMPT_PATHS` answers a 400 JSON body instead of HTML
(design §7.1). Nothing pinned `is_exempt('/admin-analytics')` before this file
(§4.6), which is how the live page depended on one string nobody tested.
"""

from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.middleware import EXEMPT_PATHS, is_exempt


def test_admin_shell_and_its_redirect_are_exempt():
    assert "/admin" in EXEMPT_PATHS
    assert is_exempt("/admin") is True
    # Kept until the redirect route itself is removed: an exact-match exemption
    # dropped early would 400 the redirect before it fires.
    assert "/admin-analytics" in EXEMPT_PATHS
    assert is_exempt("/admin-analytics") is True
    # The stylesheet rides the extension rule, not the set.
    assert is_exempt("/admin.css") is True
    # Exact match only: the API surface under /admin/ keeps its own rules.
    assert is_exempt("/admin/runs/1") is False


def test_admin_page_loads_without_a_session_header():
    with TestClient(app) as client:
        response = client.get("/admin")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>ATL Admin</title>" in response.text


def test_admin_css_is_served():
    with TestClient(app) as client:
        response = client.get("/admin.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_admin_analytics_redirects_to_admin_preserving_the_query():
    with TestClient(app) as client:
        bare = client.get("/admin-analytics", follow_redirects=False)
        with_query = client.get("/admin-analytics?range=1M&group=organic", follow_redirects=False)
    assert bare.status_code == 308
    assert bare.headers["location"] == "/admin"
    assert with_query.status_code == 308
    assert with_query.headers["location"] == "/admin?range=1M&group=organic"
```

(`test_admin_page_loads_without_a_session_header` and `test_admin_css_is_served` need Task 4's files to exist; they fail here with 404 and go green after Task 4 — that is the intended order, and the Task 4 pass step re-runs this file.)

Extend `dashboard/backend/tests/test_vercel_cache_headers.py` — append:

```python
def test_admin_shell_must_revalidate_like_every_other_shell():
    for source in ("/admin", "/admin.html"):
        assert _cache_control(source) == "public, max-age=0, must-revalidate", source


def test_admin_overrides_follow_the_api_no_store_rule():
    """The API no-store rule's optional group matches the bare /admin.

    `/(api|...|admin|...)(/.*)?` was written for `/admin/runs/{id}`; with the
    trailing group optional it also matches `/admin` itself. Last match wins per
    key, so the shell's override only overrides while it sits *after* that
    rule. Moving it earlier silently restores no-store with every
    `_cache_control` assertion above still green.
    """
    order = [entry["source"] for entry in VERCEL["headers"]]
    api_no_store = order.index(
        "/(api|paper|backtest|runs|config|admin|ticker|health|compare)(/.*)?"
    )
    assert order.index("/(.*)") < order.index("/admin")
    assert api_no_store < order.index("/admin")
    assert api_no_store < order.index("/admin.html")


def test_no_rewrite_claims_the_admin_page():
    """`/admin/:path*` → Render is gone (design §7.1).

    Whether `:path*` also matches the bare `/admin` differs between
    path-to-regexp versions; with the rewrite present, the page's reachability
    on Vercel depended on a router version nobody controls.
    """
    for entry in VERCEL["rewrites"]:
        assert not entry["source"].startswith("/admin"), entry


def test_admin_analytics_redirects_permanently_to_admin():
    redirects = VERCEL["redirects"]
    entry = next(e for e in redirects if e["source"] == "/admin-analytics")
    assert entry["destination"] == "/admin"
    assert entry["permanent"] is True
    assert not any(e["source"] == "/admin-analytics" for e in VERCEL["headers"])
```

In `dashboard/backend/tests/test_app_composition.py`, change

```python
    ("GET", "/strategy"),
    ("GET", "/admin-analytics"),
    ("GET", "/styles.css"),
```

to

```python
    ("GET", "/strategy"),
    ("GET", "/admin"),
    ("GET", "/admin.css"),
    ("GET", "/admin-analytics"),
    ("GET", "/styles.css"),
```

- [ ] **Step 2: Run them and confirm the expected failures**

```bash
python -m pytest dashboard/backend/tests/test_middleware_exemptions.py dashboard/backend/tests/test_vercel_cache_headers.py dashboard/backend/tests/test_app_composition.py -v
```

Expected: **FAIL**. `test_admin_shell_and_its_redirect_are_exempt` fails at `assert "/admin" in EXEMPT_PATHS`. `test_admin_analytics_redirects_to_admin_preserving_the_query` fails with `308 != 200` (today the route serves the mock). `test_admin_shell_must_revalidate_like_every_other_shell` fails with `None != ...`. `test_no_rewrite_claims_the_admin_page` fails on `{"source": "/admin/:path*", ...}`. `test_admin_analytics_redirects_permanently_to_admin` fails with `KeyError: 'redirects'`. `test_app_composition` fails on the full-contract set difference (`("GET", "/admin")`, `("GET", "/admin.css")` missing from the app).

- [ ] **Step 3: Rename the route, add the stylesheet route and the redirect**

In `dashboard/backend/app.py`, replace lines 384-392:

```python
@app.get("/admin-analytics", include_in_schema=False)
async def serve_admin_analytics():
    """Serve the standalone admin analytics page (preview: synthetic sample data).

    The admin console's Analytics tab redirects here; the page gates itself
    client-side on an admin session via /api/auth/me. Vercel serves the same
    file through ``cleanUrls``; this route is for the Render origin.
    """
    return FileResponse(frontend_path / "admin-analytics.html")
```

with

```python
@app.get("/admin", include_in_schema=False)
async def serve_admin():
    """Serve the admin console shell (dashboard/frontend/admin.html).

    The HTML carries no data: every number arrives from a require_admin-gated
    /api/admin/* route after js/admin-shell.js has probed /api/auth/me. That
    probe is a courtesy redirect, not the gate -- Vercel serves this same file
    as a static asset with no session access, so a server-side gate here would
    exist on one host only (design D7). This route is for the Render origin;
    Vercel serves the file through ``cleanUrls``.
    """
    return FileResponse(frontend_path / "admin.html")


@app.get("/admin.css", include_in_schema=False)
async def serve_admin_css():
    """Serve admin.css beside /styles.css (every static file is an explicit route)."""
    return FileResponse(frontend_path / "admin.css", media_type="text/css")


@app.get("/admin-analytics", include_in_schema=False)
async def redirect_admin_analytics(request: Request):
    """308 /admin-analytics → /admin for one release, then this route goes.

    Preserve the query string the way /app/ → /app does: the page keeps its
    range and filters in the query (``?range=1M&group=organic``), and a bare
    redirect would drop them. The hash (``#users/42``) never reaches the
    server; browsers carry it across a redirect whose Location has none.
    """
    target = "/admin"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    return RedirectResponse(url=target, status_code=308)
```

In `dashboard/backend/middleware.py`, change line 30

```python
    '/admin-analytics',  # Static admin analytics page; gates itself client-side on /api/auth/me
```

to

```python
    '/admin',  # Admin console shell; gates itself client-side on /api/auth/me (design D7)
    '/admin-analytics',  # 308 → /admin for one release; the exemption must outlive the redirect route
```

Replace `dashboard/frontend/vercel.json` in full:

```json
{
  "name": "AgenticTrading Dashboard",
  "cleanUrls": true,
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=3600" },
        { "key": "Content-Security-Policy", "value": "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; connect-src 'self' https://agentictrading.onrender.com; img-src 'self' data: https:;" }
      ]
    },
    { "source": "/app.js", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/styles.css", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/admin.css", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/assets/(.*)", "headers": [{ "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }] },
    { "source": "/", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/app", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/app.html", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/(api|paper|backtest|runs|config|admin|ticker|health|compare)(/.*)?", "headers": [{ "key": "Cache-Control", "value": "no-store" }] },
    { "source": "/admin", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] },
    { "source": "/admin.html", "headers": [{ "key": "Cache-Control", "value": "public, max-age=0, must-revalidate" }] }
  ],
  "redirects": [
    { "source": "/admin-analytics", "destination": "/admin", "permanent": true }
  ],
  "rewrites": [
    { "source": "/app", "destination": "/app.html" },
    { "source": "/api/:path*", "destination": "https://agentictrading.onrender.com/api/:path*" },
    { "source": "/paper/:path*", "destination": "https://agentictrading.onrender.com/paper/:path*" },
    { "source": "/backtest/:path*", "destination": "https://agentictrading.onrender.com/backtest/:path*" },
    { "source": "/runs/:path*", "destination": "https://agentictrading.onrender.com/runs/:path*" },
    { "source": "/runs", "destination": "https://agentictrading.onrender.com/runs" },
    { "source": "/config/:path*", "destination": "https://agentictrading.onrender.com/config/:path*" },
    { "source": "/ticker", "destination": "https://agentictrading.onrender.com/ticker" },
    { "source": "/health", "destination": "https://agentictrading.onrender.com/health" },
    { "source": "/compare", "destination": "https://agentictrading.onrender.com/compare" }
  ]
}
```

Three deliberate changes against `main`: the two `/admin-analytics` header rules are gone (the path now redirects; a header rule on it is dead), the `/admin/:path*` rewrite is gone (its only consumer, `DELETE /admin/runs/{run_id}`, has no frontend caller — §7.1), and `/admin.css` joins `/styles.css` in the revalidate list because the page's `<link>` carries a `?v=` the browser must be allowed to see change. `permanent: true` is Vercel's 308.

- [ ] **Step 4: Run and confirm the pass (partial until Task 4)**

```bash
python -m pytest dashboard/backend/tests/test_middleware_exemptions.py dashboard/backend/tests/test_vercel_cache_headers.py dashboard/backend/tests/test_app_composition.py -v
```

Expected: **PASS** for every test except `test_admin_page_loads_without_a_session_header` and `test_admin_css_is_served`, which fail with 404 until Task 4 creates `admin.html` and `admin.css` (`FileResponse` on a missing path raises at response time). `test_app_composition` passes: the two new routes and the retained redirect route make the app's `(method, path)` set equal to the contract, and each is registered once.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/app.py dashboard/backend/middleware.py dashboard/frontend/vercel.json dashboard/backend/tests/test_middleware_exemptions.py dashboard/backend/tests/test_vercel_cache_headers.py dashboard/backend/tests/test_app_composition.py
git commit -m "$(cat <<'EOF'
feat: serve the admin console at /admin and redirect /admin-analytics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `admin.html` and `admin.css` — the data-free shell built from the mock's final DOM

**Files:**
- Create: `dashboard/frontend/admin.html`.
- Create: `dashboard/frontend/admin.css`.
- Create: `dashboard/backend/tests/test_admin_page_shell.py`.
- Test: `dashboard/backend/tests/test_admin_page_shell.py`, `dashboard/backend/tests/test_middleware_exemptions.py` (Task 3's two 404s go green).

**Interfaces:**
- Consumes: nothing at runtime (the shell carries no data, D7); `images/atltransparent.png` (served by `GET /images/{file_name}`).
- Produces: the element ids and `data-*` hooks Tasks 5–8 target — `#filters`, `#filterGroup`, `#filterSegment`, `#filterTier`, `#filterInternal`, `.range button[data-range]`, `#analyticsSubnav a[data-route]`, `#analyticsParent`, `#liveTiles [data-live]`, `#liveUpdated`, `#panelAttention` … `#panelRevenue` (each with `[data-headline]`, `[data-body]`, `[data-status]`, `[data-error]`, `[data-retry]`), `#detail`, `#usersView` (`#usersSearch`, `#usersQuery`, `#usersPriority`, `#usersBody`, `#usersRange`, `#usersPrev`, `#usersNext`), `#profile`, `#freshnessLegend`, `#rulesDialog` (`#rulesList`), `#evidenceDialog` (`#evidenceBody`, `#evidenceAccount`, `#evidenceProfile`).

The element list is design §8.2 applied to the final rendered DOM (V8 §2/§9). Built: header + product nav, aside (Analytics subnav, plus Account management / Providers / Activity links to the old console — §7.5), range picker **1W / 1M / 1Y**, filters **Source / Lifecycle stage / Tier / Include internal accounts**, the live row with **Total users / Total agents / Backtests running · this instance (with the real slot ceiling)** and its *this instance* caveat, the nine panels in the mock's final five-row pairing (V8 block 9), the shared "Open Credits & revenue" link, `#detail`, the `#users` list view, `#profile`, the freshness legend, the rules dialog and the evidence dialog. Not built: `Sample data` label and footer notice, `1D`, the Cohort and Paid/Unpaid filters (Tier replaces the latter), Online now, Queued, the Blocked-users live tile, the `#live` detail route, the hidden `.inline-key` legend, the `#health` "Affected users" column, the profile's Region/Device/Browser/top-page facts, and the orphan `#usage`/`#revenue` routes.

- [ ] **Step 1: Write the failing source-shape guard**

Create `dashboard/backend/tests/test_admin_page_shell.py`:

```python
"""Source-shape guard for the /admin shell (design §7.6, D6, D7).

The page must carry no data: every number arrives from a require_admin-gated
endpoint after the gate has run. The mock this page replaces failed exactly
that property -- sixteen inline scripts wrote literals into every panel -- so
the guard checks the *markup*: no inline script, every script pinned, and no
numeric or percentage literal inside any panel region (placeholders are "--").
"""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
ADMIN_CSS = (FRONTEND / "admin.css").read_text(encoding="utf-8")

EXPECTED_SCRIPTS = [
    "js/admin-shell.js?v=1",
    "js/credit-format.js?v=1",
    "js/admin-live.js?v=1",
    "js/admin-overview.js?v=1",
    "js/admin-users.js?v=1",
]

SCRIPT_TAG = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S)
PANEL_REGION = re.compile(
    r'<section\b[^>]*\bdata-panel="([^"]+)"[^>]*>(.*?)</section>', re.S
)
TAG = re.compile(r"<[^>]+>")
# A standalone number: not glued to a letter, underscore, hash, dot or dash on
# the left (so "W1", "#users" and "8-29" never count) and not followed by a
# word character ("1W" is a range label, not a value). Optional decimals and %.
NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z0-9_#.\-])\d+(?:[.,]\d+)?%?(?![A-Za-z0-9_])")


def test_every_script_is_external_and_none_is_inline():
    tags = SCRIPT_TAG.findall(ADMIN_HTML)
    assert len(tags) == len(EXPECTED_SCRIPTS)
    for attrs, body in tags:
        assert 'src="' in attrs, attrs
        assert "defer" in attrs, attrs
        assert body.strip() == "", body


def test_gate_module_loads_first_and_every_script_is_pinned():
    srcs = re.findall(r'<script src="([^"]+)" defer></script>', ADMIN_HTML)
    assert srcs == EXPECTED_SCRIPTS
    assert srcs[0] == "js/admin-shell.js?v=1"
    for src in srcs:
        assert "?v=" in src, src


def test_stylesheet_is_admin_css_and_the_page_does_not_inherit_styles_css():
    assert '<link rel="stylesheet" href="admin.css?v=1">' in ADMIN_HTML
    assert "styles.css" not in ADMIN_HTML
    assert "@import" not in ADMIN_CSS
    assert "cdn.jsdelivr.net" not in ADMIN_HTML
    assert "chart.js" not in ADMIN_HTML.lower()


def test_title_is_the_admin_console():
    assert "<title>ATL Admin</title>" in ADMIN_HTML


def test_panel_regions_carry_no_numeric_or_percentage_literal():
    regions = PANEL_REGION.findall(ADMIN_HTML)
    names = [name for name, _body in regions]
    assert names == [
        "live", "attention", "active-users", "activation", "sources", "retention",
        "value", "lifecycle", "credits", "revenue", "detail", "users", "profile",
    ]
    for name, body in regions:
        text = TAG.sub(" ", body)
        match = NUMERIC_LITERAL.search(text)
        assert match is None, (name, match and match.group(0), text.strip()[:200])


def test_every_value_slot_is_a_dash_placeholder():
    for attr in ("data-headline", "data-live"):
        slots = re.findall(rf"<strong[^>]*\b{attr}[^>]*>([^<]*)</strong>", ADMIN_HTML)
        assert slots, attr
        assert set(slots) == {"—"}, (attr, slots)
    metas = re.findall(r'<small[^>]*\bdata-live="slots"[^>]*>([^<]*)</small>', ADMIN_HTML)
    assert metas == ["—"]


def test_filter_bar_and_range_match_the_survival_table():
    filters_start = ADMIN_HTML.index('<form class="filters"')
    filters_end = ADMIN_HTML.index("</form>", filters_start)
    filters = ADMIN_HTML[filters_start:filters_end]
    assert filters.count("<select") == 3
    for control in ("filterGroup", "filterSegment", "filterTier", "filterInternal"):
        assert f'id="{control}"' in filters, control
    assert "Include internal accounts" in filters
    assert "Cohort" not in filters and "intake" not in filters
    assert "All paid states" not in filters
    for option in ("internal", "invited", "organic", "competition", "partner", "unknown"):
        assert f'value="{option}"' in filters
    for option in ("new", "onboarding", "growing", "core", "at_risk", "dormant"):
        assert f'value="{option}"' in filters
    for option in ("unpaid", "starter", "invested", "high_value"):
        assert f'value="{option}"' in filters
    ranges = re.findall(r'data-range="([^"]+)"', ADMIN_HTML)
    assert ranges == ["1W", "1M", "1Y"]


def test_cut_elements_are_absent():
    for token in (
        "Sample data", "synthetic", "Sample snapshot", "Online now", "Queued",
        'href="#live"', "Layout proposal", "inline-key", "Affected users",
        "September intake", 'data-range="1D"', ">Region<", ">Device<", ">Browser<",
        "Top product page",
    ):
        assert token not in ADMIN_HTML, token


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


def test_freshness_legend_replaces_the_sample_notice():
    assert 'id="freshnessLegend"' in ADMIN_HTML
    assert "Daily figures complete through" in ADMIN_HTML
    assert "live tiles are this instance" in ADMIN_HTML


def test_pruned_css_carries_none_of_the_dead_mock_passes():
    for selector in (
        ".funnel", ".activation-river", ".activation-stages", ".live-cell",
        ".live-grid", ".live-composite", ".mini-viz", ".sparkline", ".run-lanes",
        ".blocked-reasons", ".source-bars", ".stack-column", ".inline-key",
        ".sample-label", ".queue-", ".operation-lane", ".blocker-item",
        ".value-donut", ".detail-live-grid", ".snapshot-tile.online",
        ".snapshot-tile.queued", ".snapshot-tile.blocked",
    ):
        assert selector not in ADMIN_CSS, selector
    for selector in (
        ".layered-route", ".source-donut", ".source-legend-six", ".value-steps",
        ".lifecycle-bar", ".retention-chart", ".credits-paired", ".revenue-viz",
        ".snapshot-grid", ".attention-row", ".group-badge", ".panel-error",
        ".freshness-legend", "@media(max-width:1040px)", "@media(max-width:680px)",
        "@media(prefers-reduced-motion:reduce)",
    ):
        assert selector in ADMIN_CSS, selector
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_page_shell.py -v
```

Expected: **ERROR at collection** — `FileNotFoundError: .../dashboard/frontend/admin.html` from the module-level `read_text`.

- [ ] **Step 3: Create `admin.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ATL Admin</title>
  <link rel="icon" href="favicon.svg">
  <link rel="stylesheet" href="admin.css?v=1">
  <!-- No inline scripts (design D6): pytest can only execute frontend code it
       can lift from an external file. admin-shell.js loads first because it
       owns the /api/auth/me gate and the router the other modules listen to;
       all five are deferred, so they run in this order before DOMContentLoaded. -->
  <script src="js/admin-shell.js?v=1" defer></script>
  <script src="js/credit-format.js?v=1" defer></script>
  <script src="js/admin-live.js?v=1" defer></script>
  <script src="js/admin-overview.js?v=1" defer></script>
  <script src="js/admin-users.js?v=1" defer></script>
</head>
<body>
  <header>
    <a class="brand" href="#overview"><img src="images/atltransparent.png" alt="ATL">Agentic Trading Lab</a>
    <nav class="product-nav" aria-label="Product"><a href="/app">Home</a><a href="/app?view=playground&amp;playgroundTab=agents">My Agents</a><a href="/app?view=competition">Competition</a><a href="/app?view=community">Community</a></nav>
  </header>
  <div class="workspace">
    <aside>
      <strong>Administration</strong>
      <a class="analytics-parent" id="analyticsParent" href="#overview" aria-expanded="true">Analytics <span aria-hidden="true">⌄</span></a>
      <nav class="analytics-subnav" id="analyticsSubnav" aria-label="Analytics modules">
        <a class="active" href="#overview" data-route="overview">Overview</a>
        <a href="#sources" data-route="sources">User sources</a>
        <a href="#retention" data-route="retention">Retention</a>
        <a href="#credits" data-route="credits">Credits &amp; revenue</a>
        <a href="#lifecycle" data-route="lifecycle">User lifecycle</a>
        <a href="#health" data-route="health">System health</a>
        <a href="#users" data-route="users">Users</a>
      </nav>
      <!-- Account management, Providers and Activity stay in the old console
           until the follow-up port (design D5, §7.5). -->
      <a href="/app?view=admin&amp;adminTab=users">Account management</a>
      <a href="/app?view=admin&amp;adminTab=providers">Providers</a>
      <a href="/app?view=admin&amp;adminTab=activity">Activity</a>
    </aside>
    <main>
      <section id="overview" aria-label="Analytics overview">
        <div class="page-head">
          <div><h1>Overview</h1><p class="muted">User activity, acquisition and attention</p></div>
          <div class="page-actions">
            <div class="range" role="group" aria-label="Date range">
              <button type="button" data-range="1W" aria-pressed="true">1W</button>
              <button type="button" data-range="1M" aria-pressed="false">1M</button>
              <button type="button" data-range="1Y" aria-pressed="false">1Y</button>
            </div>
            <small>UTC</small>
          </div>
        </div>
        <form class="filters" id="filters" aria-label="Analytics filters">
          <label class="filter-field"><span>Source</span><select id="filterGroup" name="group"><option value="">All sources</option><option value="internal">Internal</option><option value="invited">Invited</option><option value="organic">Organic</option><option value="competition">Competition</option><option value="partner">Partner</option><option value="unknown">Unknown</option></select></label>
          <label class="filter-field"><span>Lifecycle stage</span><select id="filterSegment" name="segment"><option value="">All lifecycle stages</option><option value="new">New</option><option value="onboarding">Onboarding</option><option value="growing">Growing</option><option value="core">Core</option><option value="at_risk">At risk</option><option value="dormant">Dormant</option></select></label>
          <label class="filter-field"><span>Tier</span><select id="filterTier" name="tier"><option value="">All tiers</option><option value="unpaid">Unpaid</option><option value="starter">Starter</option><option value="invested">Invested</option><option value="high_value">High value</option></select></label>
          <label class="check-field"><input type="checkbox" id="filterInternal" name="internal">Include internal accounts</label>
          <p class="filter-note muted">Lifecycle stage and Tier narrow the user lists; Source also narrows the sources panels.</p>
        </form>

        <div class="overview-row priority-row">
          <section class="panel live-row" data-panel="live" aria-labelledby="liveTitle" aria-busy="true">
            <div class="live-head"><div class="live-title"><i class="live-dot" aria-hidden="true"></i><div><h2 id="liveTitle">Live operations</h2><small>This instance</small></div></div><small id="liveUpdated">—</small></div>
            <div class="snapshot-grid" id="liveTiles">
              <article class="snapshot-tile"><span class="snapshot-label">Total users</span><strong class="snapshot-value" data-live="users">—</strong></article>
              <article class="snapshot-tile"><span class="snapshot-label">Total agents</span><strong class="snapshot-value" data-live="agents">—</strong></article>
              <article class="snapshot-tile runs"><span class="snapshot-label">Backtests running · this instance</span><strong class="snapshot-value" data-live="active_dashboard_backtests">—</strong><small class="snapshot-meta" data-live="slots">—</small></article>
            </div>
            <p class="live-caveat muted" id="liveCaveat">Counters are this instance's process state; a second replica would under-report them.</p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
          </section>
          <section class="panel" id="panelAttention" data-panel="attention" aria-labelledby="attentionTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="attentionTitle">Users needing attention</h2><small>As of yesterday UTC</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>affected users</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#users">Open users</a>
          </section>
        </div>

        <div class="overview-row priority-row">
          <section class="panel" id="panelActiveUsers" data-panel="active-users" aria-labelledby="activeUsersTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="activeUsersTitle">Active users</h2><small>Daily · selected range</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>active in the last seven days</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#users">Open active users</a>
          </section>
          <section class="panel" id="panelActivation" data-panel="activation" aria-labelledby="activationTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="activationTitle">Activation progress</h2><small>From account creation to first successful result</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>overall activation</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#sources">Open activation progress</a>
          </section>
        </div>

        <div class="overview-row priority-row">
          <section class="panel" id="panelSources" data-panel="sources" aria-labelledby="sourcesTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="sourcesTitle">Where users come from</h2><small>Account source · lifetime</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>identified users</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#sources">Open source analysis</a>
          </section>
          <section class="panel" id="panelRetention" data-panel="retention" aria-labelledby="retentionTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="retentionTitle">Users coming back</h2><small>First-success cohorts</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>week-one return</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#retention">Open retention</a>
          </section>
        </div>

        <div class="overview-row priority-row">
          <section class="panel" id="panelValue" data-panel="value" aria-labelledby="valueTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="valueTitle">Users reaching value</h2><small>Successful-result depth · selected range</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>users with a result</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#lifecycle">Open user lifecycle</a>
          </section>
          <section class="panel" id="panelLifecycle" data-panel="lifecycle" aria-labelledby="lifecycleTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="lifecycleTitle">User lifecycle</h2><small>Current distribution</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>classified users</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
            <a class="module-link" href="#lifecycle">Open lifecycle</a>
          </section>
        </div>

        <div class="overview-row priority-row">
          <section class="panel" id="panelCredits" data-panel="credits" aria-labelledby="creditsTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="creditsTitle">Credits usage</h2><small class="selected-range">Selected range</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>ATL Credits settled</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
          </section>
          <section class="panel" id="panelRevenue" data-panel="revenue" aria-labelledby="revenueTitle" aria-busy="true">
            <div class="section-head"><div><h2 id="revenueTitle">Revenue</h2><small class="selected-range">Selected range</small></div><div class="section-meta"><strong class="headline-number" data-headline>—</strong><small>purchased Credits</small></div></div>
            <div class="panel-body" data-body></div>
            <p class="revenue-note muted">Admin Grants excluded from revenue</p>
            <p class="panel-status muted" data-status hidden></p>
            <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
          </section>
        </div>
        <div class="shared-link-row"><a class="module-link" href="#credits">Open Credits &amp; revenue</a></div>
      </section>

      <section id="detail" class="detail-view" data-panel="detail" hidden aria-live="polite"></section>

      <section id="usersView" class="detail-view" data-panel="users" hidden aria-labelledby="usersTitle">
        <nav class="breadcrumb" aria-label="Breadcrumb"><a href="#overview">Analytics</a><span>/</span><span>Users</span></nav>
        <div class="page-head"><div><h1 id="usersTitle" tabindex="-1">Users</h1><p class="muted">Every account with its group, lifecycle stage and operational state</p></div></div>
        <div class="users-toolbar">
          <form id="usersSearch" role="search" aria-label="Search users"><input id="usersQuery" type="search" name="q" maxlength="100" placeholder="Search by email or display name" autocomplete="off"><button type="submit">Search</button></form>
          <label class="check-field"><input type="checkbox" id="usersPriority">Priority users only</label>
        </div>
        <div class="table-wrap">
          <table class="users-table">
            <thead><tr><th scope="col">User</th><th scope="col">Group</th><th scope="col">Lifecycle</th><th scope="col">Operational</th><th scope="col">Last active (UTC)</th><th scope="col"><span class="visually-hidden">Actions</span></th></tr></thead>
            <tbody id="usersBody"></tbody>
          </table>
        </div>
        <p class="panel-status muted" data-status hidden></p>
        <p class="panel-error" data-error hidden><span></span> <button type="button" class="retry" data-retry>Retry</button></p>
        <div class="pager"><span id="usersRange"></span><button type="button" id="usersPrev" disabled>Previous</button><button type="button" id="usersNext" disabled>Next</button></div>
      </section>

      <section id="profile" class="detail-view" data-panel="profile" hidden aria-live="polite"></section>

      <footer id="freshnessLegend" class="freshness-legend">Daily figures complete through — UTC; live tiles are this instance's process state</footer>
    </main>
  </div>

  <dialog id="rulesDialog" aria-labelledby="rulesTitle">
    <div class="dialog-head"><h2 id="rulesTitle">How states are determined</h2><button type="button" data-dialog-close aria-label="Close">×</button></div>
    <dl id="rulesList" class="rules-list"></dl>
    <div class="dialog-actions"><button type="button" data-dialog-close data-dialog-initial-focus>Close</button></div>
  </dialog>

  <dialog id="evidenceDialog" aria-labelledby="evidenceTitle">
    <div class="dialog-head"><h2 id="evidenceTitle">User evidence</h2><button type="button" data-dialog-close aria-label="Close">×</button></div>
    <div id="evidenceBody"></div>
    <div class="dialog-actions"><a id="evidenceAccount" class="module-link" href="/app?view=admin&amp;adminTab=users">Open account management</a><a id="evidenceProfile" class="module-link" href="#users" data-dialog-initial-focus>Open full profile</a></div>
  </dialog>
</body>
</html>
```

- [ ] **Step 4: Create `admin.css`**

The rules below are the mock's inline `<style>` blocks (`admin-analytics.html` lines 8-27 base + media queries, 342-361 final source/revenue, 466-476 priority rows, 512-525 operation snapshot, 574-583 attention polish, 641-652 layered route, 666-676 credits paired) externalised, minus everything V8 §1/§3 lists as inert or destroyed in the final DOM (the `.funnel`/`.activation-*` passes at 26-30, 440-465, 489-511, 526-573, 599-640; the live-detail widgets at 31-49 and 51-115; `.source-bars`, `.stack-columns`, `.value-donut`, `.inline-key`, `.sample-label`; the `#overview .overview-row.equal*` and `.overview-row.single` rules whose rows block 9 removed), plus the rules the new elements need (`.freshness-legend`, `.group-badge`, panel state classes, the users table, the profile facts, the rules list). `#overview` prefixes are dropped where the selector only ever matched inside `#overview`.

```css
/* admin.css — the /admin console shell (design §7.2). Externalised from the
   retired admin-analytics.html mock and pruned to what its final rendered
   DOM used (design §8.1). This file does not import styles.css: the admin
   page must not inherit a 15,000-line stylesheet to get a header. */
:root{color-scheme:dark;--bg:#0a0e27;--surface:#12172b;--line:#2a3041;--line-strong:#384158;--text:#e5e7eb;--muted:#9ca3af;--accent:#00bfff;--green:#42cc8b;--amber:#e6b75c;--red:#f08080;--violet:#8797ff;--steel:#6f86a8;--font-base:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",sans-serif;font-family:var(--font-base);color:var(--text);background:var(--bg)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 var(--font-base)}
button,select,input{font:inherit}
button,select,input[type=search]{min-height:40px;color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:6px}
button{padding:8px 12px;cursor:pointer}
button[disabled]{cursor:default;opacity:.5}
select{width:100%;padding:8px 32px 8px 11px}
input[type=search]{width:100%;padding:8px 11px}
a{color:inherit;text-decoration:none}
a:hover,button:hover{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
[hidden]{display:none!important}
h1,h2,h3,p{margin-top:0}
h1{margin-bottom:6px;font-size:29px;line-height:1.15}
h2{margin-bottom:0;font-size:18px;line-height:1.25}
h3{margin-bottom:8px;font-size:15px}
small,.muted{color:var(--muted)}
strong{font-variant-numeric:tabular-nums}
.visually-hidden{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* Header, aside, main (mock lines 10-12) */
header{display:flex;align-items:center;justify-content:space-between;gap:20px;min-height:80px;padding:14px 28px;background:var(--surface);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px;font-size:22px;font-weight:700}
.brand img{width:48px;height:48px;object-fit:contain}
.product-nav{display:flex;gap:25px;color:var(--muted)}
.workspace{display:grid;grid-template-columns:196px minmax(0,1fr);min-height:calc(100vh - 80px)}
aside{padding:24px 16px;border-right:1px solid var(--line)}
aside>strong{display:block;margin:0 12px 20px}
aside a{display:block;margin-bottom:4px;padding:10px 12px;border-radius:6px;color:var(--muted)}
aside a:hover{background:#101c31}
.analytics-parent{display:flex;align-items:center;justify-content:space-between;color:var(--accent);background:#132c41}
.analytics-subnav{margin:0 0 14px 15px;padding-left:8px;border-left:1px solid var(--line)}
.analytics-subnav a{padding:7px 10px;font-size:12px}
.analytics-subnav a.active{color:var(--accent);background:#102536}
main{width:100%;max-width:1420px;margin:0 auto;padding:30px 36px 50px}
.page-head,.section-head,.live-head,.breadcrumb{display:flex;align-items:center;justify-content:space-between;gap:16px}
.page-head p{margin:0}
.page-actions{display:flex;align-items:center;gap:10px}
.range{display:flex;gap:3px;padding:3px;border:1px solid var(--line);border-radius:7px}
.range button{min-width:45px;min-height:34px;padding:5px 10px;background:transparent;border-color:transparent}
.range button[aria-pressed=true]{color:#061521;background:var(--accent)}

/* Filters (mock line 13; three selects instead of four) */
.filters{display:grid;grid-template-columns:repeat(3,minmax(130px,1fr)) auto;gap:12px;align-items:end;margin:22px 0 0;padding:0 0 22px;border-bottom:1px solid var(--line)}
.filter-field{display:grid;gap:7px;min-width:0;color:var(--muted);font-size:12px}
.check-field{display:flex;align-items:center;gap:8px;min-height:40px;white-space:nowrap;color:var(--muted);font-size:12px}
.check-field input{accent-color:var(--accent)}
.filter-note{grid-column:1/-1;margin:0;font-size:11px}
.module-link{display:inline-flex;align-items:center;gap:7px;margin-top:18px;color:var(--accent);font-weight:600}
.module-link:after{content:"→";transition:transform .16s ease}
.module-link:hover:after{transform:translateX(3px)}

/* Panels and their states (mock lines 15, 466-476; §15.6 loading/empty/error/stale) */
.overview-row.priority-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:34px;max-width:none;margin:0}
.panel{display:block;min-width:0;padding:26px 0;border-bottom:1px solid var(--line)}
.section-head{align-items:flex-start;margin-bottom:20px}
.section-head small{display:block;margin-top:5px}
.section-meta{text-align:right}
.headline-number{display:block;margin-bottom:3px;color:var(--text);font-size:25px;line-height:1;font-weight:650}
.panel-body{min-height:180px}
.panel[aria-busy=true] .panel-body{opacity:.55}
.panel.is-stale .headline-number{color:var(--muted)}
.panel-status{margin:12px 0 0;font-size:11px}
.panel-error{display:flex;align-items:center;gap:12px;margin:12px 0 0;color:var(--amber);font-size:12px}
.panel-error .retry{min-height:32px;padding:4px 10px}
.panel-empty{margin:0;padding:24px 0;color:var(--muted);font-size:12px}
.shared-link-row{padding:0 0 28px;border-bottom:1px solid var(--line)}

/* Live operations row (mock lines 512-525, three tiles) */
.live-row .live-head{display:flex;align-items:flex-start;justify-content:space-between;margin:0 0 18px;padding:0}
.live-title{display:flex;align-items:center;gap:10px}
.live-dot{width:8px;height:8px;flex:none;border-radius:50%;background:var(--green);box-shadow:0 0 0 5px rgba(66,204,139,.12)}
.snapshot-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border:1px solid var(--line);border-radius:8px;overflow:hidden;background:rgba(255,255,255,.018)}
.snapshot-tile{position:relative;min-height:104px;padding:17px 18px;border:0;overflow:hidden}
.snapshot-tile+.snapshot-tile{border-left:1px solid var(--line)}
.snapshot-label{display:block;color:var(--muted);font-size:12px;font-weight:600}
.snapshot-value{display:block;margin-top:7px;color:var(--text);font-size:34px;line-height:1;font-variant-numeric:tabular-nums}
.snapshot-tile.runs .snapshot-value{color:var(--accent)}
.snapshot-meta{display:block;margin-top:6px;font-size:11px}
.live-caveat{margin:14px 0 0;font-size:11px}

/* Active users bar chart (mock lines 15, 104-108) */
.bar-chart{display:flex;align-items:end;gap:11px;height:180px;padding-top:8px;border-bottom:1px solid var(--line)}
.bar-column{display:flex;flex:1;flex-direction:column;justify-content:end;align-items:center;gap:7px;height:100%;min-width:18px;color:var(--muted);font-size:11px}
.bar-column b{font-weight:500}
.bar-column i{display:block;width:min(34px,70%);min-height:3px;height:var(--height);border-radius:3px 3px 0 0;background:var(--accent)}
.chart-with-axis{position:relative;background:repeating-linear-gradient(to bottom,transparent 0,transparent calc(25% - 1px),rgba(56,65,88,.42) calc(25% - 1px),rgba(56,65,88,.42) 25%)}
.bar-chart.with-axis{padding:12px 0 24px 32px;border-bottom:1px solid var(--line)}
.chart-y-axis{position:absolute;left:0;top:9px;bottom:22px;display:flex;flex-direction:column;justify-content:space-between;color:var(--muted);font-size:10px;line-height:1;pointer-events:none}
.chart-y-axis span{display:block;min-width:24px;text-align:left}
.bar-chart.with-axis::after{content:"";position:absolute;left:32px;right:0;bottom:23px;border-bottom:1px solid var(--line-strong);pointer-events:none}

/* Activation progress — layered route (mock lines 641-652, the one funnel pass that renders) */
.layered-route{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;height:218px;padding:16px 0 0;overflow:visible}
.layered-route::before{content:"";position:absolute;left:7%;right:7%;bottom:22px;height:1px;background:var(--line)}
.layered-route .layer{position:relative;display:flex;flex-direction:column;justify-content:space-between;min-width:0;padding:14px 15px 15px;background:linear-gradient(155deg,rgba(21,30,58,.96),rgba(13,20,42,.92));border:1px solid var(--line);border-radius:8px 8px 3px 3px;overflow:hidden}
.layered-route .layer::after{content:"";position:absolute;left:0;right:0;bottom:0;height:var(--depth);background:linear-gradient(90deg,rgba(10,185,242,.5),rgba(10,185,242,.08));border-top:1px solid rgba(10,185,242,.55);pointer-events:none}
.layered-route .layer:last-child{border-color:rgba(69,208,146,.55)}
.layered-route .layer:last-child::after{background:linear-gradient(90deg,rgba(69,208,146,.52),rgba(69,208,146,.08));border-color:rgba(69,208,146,.55)}
.layered-route .layer-name,.layered-route .layer-value,.layered-route .layer-meta{position:relative;z-index:1}
.layered-route .layer-name{color:var(--muted);font-size:12px;font-weight:600;line-height:1.25}
.layered-route .layer-value{font-size:34px;line-height:1;font-variant-numeric:tabular-nums}
.layered-route .layer-meta{color:var(--muted);font-size:11px}
.layered-route .layer-meta strong{color:var(--text)}

/* Sources donut (mock lines 342-352; the gradient is set per payload by JS) */
.source-wrap{display:grid;grid-template-columns:178px minmax(0,1fr);align-items:center;gap:24px;min-height:196px}
.source-donut{position:relative;width:168px;height:168px;flex:none;border-radius:50%;background:var(--line);box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)}
.source-donut:after{content:attr(data-total);position:absolute;inset:31px;display:grid;place-items:center;border-radius:50%;background:var(--bg);color:var(--text);font-size:17px;font-weight:650;text-align:center;white-space:pre-line}
.source-legend-six{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 18px}
.legend-row{display:grid;grid-template-columns:9px 1fr auto;align-items:center;gap:7px;color:var(--muted);font-size:12px}
.legend-row i{width:9px;height:9px;border-radius:2px;background:var(--accent)}
.legend-row i.source-green{background:var(--green)}
.legend-row i.source-violet{background:var(--violet)}
.legend-row i.source-amber{background:var(--amber)}
.legend-row i.source-red{background:var(--red)}
.legend-row i.source-steel{background:var(--steel)}
.legend-row b{color:var(--text);font-weight:600}

/* Users needing attention (mock lines 574-583) */
.attention-list{margin-top:14px;border-top:1px solid var(--line)}
.attention-row{display:grid;grid-template-columns:8px minmax(0,1fr) auto;grid-template-areas:"severity label count" "severity detail detail";column-gap:12px;row-gap:3px;align-items:center;padding:12px 0;border-bottom:1px solid var(--line)}
.attention-row .severity{grid-area:severity;width:7px;height:34px;border-radius:99px;background:var(--red)}
.attention-row .severity.warn{background:var(--amber)}
.attention-row .severity.muted{background:#718096}
.attention-row span{grid-area:label;color:var(--text);font-size:13px;font-weight:600}
.attention-row b{grid-area:count;color:var(--text);font-size:22px;font-variant-numeric:tabular-nums}
.attention-row small{grid-area:detail;color:var(--muted);font-size:11px}
.attention-note{display:flex;align-items:center;justify-content:space-between;gap:14px;padding-top:12px;color:var(--muted);font-size:11px}
.attention-note>div{display:grid;gap:2px}
.attention-note strong{display:block;color:var(--text);font-size:12px;margin-top:3px}
.attention-note a{color:var(--accent);font-weight:600;white-space:nowrap}

/* Retention heatmap (mock line 21) */
.retention-chart{display:grid;grid-template-columns:68px repeat(4,1fr);gap:6px;align-items:center}
.retention-chart>span{color:var(--muted);font-size:11px;text-align:center}
.retention-chart .row-label{text-align:left}
.retention-cell{display:grid;place-items:center;min-height:34px;border:1px solid var(--line);border-radius:4px;background:rgba(0,191,255,var(--alpha));color:var(--text);font-size:11px}
.retention-cell.empty{color:#596177;background:transparent}

/* Users reaching value — steps (mock lines 95-104) */
.value-steps{display:grid;gap:13px;min-height:180px;padding-top:10px}
.value-step{display:grid;grid-template-columns:132px 1fr auto;align-items:center;gap:10px;color:var(--muted);font-size:12px}
.value-step>span{white-space:nowrap}
.value-step b{color:var(--text);font-size:12px;white-space:nowrap}
.value-track{height:12px;overflow:hidden;border-radius:2px;background:#171d31}
.value-track i{display:block;width:var(--share);height:100%;background:var(--accent)}
.value-step:nth-child(2) .value-track i{background:var(--green)}
.value-step:nth-child(3) .value-track i{background:var(--amber)}

/* User lifecycle (mock line 23; segment widths come from the payload) */
.lifecycle-bar{display:flex;height:14px;margin:18px 0 20px;overflow:hidden;border-radius:3px}
.lifecycle-bar i{display:block;flex:var(--share) 1 0}
.lifecycle-bar i:nth-child(1){background:var(--accent)}
.lifecycle-bar i:nth-child(2){background:var(--violet)}
.lifecycle-bar i:nth-child(3){background:var(--green)}
.lifecycle-bar i:nth-child(4){background:var(--amber)}
.lifecycle-bar i:nth-child(5){background:var(--red)}
.lifecycle-bar i:nth-child(6){background:#697586}
.lifecycle-legend{display:grid;grid-template-columns:repeat(3,1fr);gap:12px 18px}
.lifecycle-legend div{display:grid;grid-template-columns:8px 1fr auto;align-items:center;gap:7px;color:var(--muted);font-size:12px}
.lifecycle-legend i{width:8px;height:8px;border-radius:2px;background:var(--accent)}
.lifecycle-legend div:nth-child(2) i{background:var(--violet)}
.lifecycle-legend div:nth-child(3) i{background:var(--green)}
.lifecycle-legend div:nth-child(4) i{background:var(--amber)}
.lifecycle-legend div:nth-child(5) i{background:var(--red)}
.lifecycle-legend div:nth-child(6) i{background:#697586}
.lifecycle-legend b{color:var(--text)}

/* Credits usage — paired stems (mock lines 666-676) */
.credits-paired{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px;height:164px;align-items:end;padding:0 4px;border-bottom:1px solid var(--line);position:relative}
.credits-paired .credit-day{position:relative;height:100%;display:flex;align-items:end;justify-content:center;gap:4px;padding-bottom:24px}
.credits-paired .credit-day::after{content:"";position:absolute;left:50%;top:16px;bottom:24px;width:1px;background:rgba(255,255,255,.06)}
.credits-paired .credit-stem{position:relative;z-index:1;width:11px;border-radius:7px 7px 2px 2px;background:var(--accent)}
.credits-paired .credit-stem.violet{background:#8998ff}
.credits-paired .credit-stem b{position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);font-size:10px;color:var(--text);font-variant-numeric:tabular-nums}
.credits-paired .credit-date{position:absolute;bottom:4px;color:var(--muted);font-size:10px;white-space:nowrap}
.credits-paired .credit-axis{position:absolute;left:0;right:0;top:2px;display:flex;justify-content:space-between;color:var(--muted);font-size:10px;pointer-events:none}
.credits-paired .credit-legend{grid-column:1/-1;display:flex;gap:16px;color:var(--muted);font-size:11px;margin-top:10px}
.credits-paired .credit-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--accent);margin-right:5px}
.credits-paired .credit-legend i.violet{background:#8998ff}

/* Revenue — generated point line (mock lines 353-361) */
.revenue-viz{height:190px;padding-top:8px;border-bottom:1px solid var(--line)}
.revenue-viz svg{display:block;width:100%;height:174px;overflow:visible}
.revenue-viz .revenue-grid{stroke:var(--line);stroke-width:1}
.revenue-viz .revenue-axis{stroke:var(--line-strong);stroke-width:1}
.revenue-viz .revenue-axis-label,.revenue-viz .revenue-x-label{fill:var(--muted);font-size:10px}
.revenue-viz .revenue-area{fill:rgba(66,204,139,.11)}
.revenue-viz .revenue-line{fill:none;stroke:var(--green);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round}
.revenue-viz .revenue-point{fill:var(--green);stroke:var(--bg);stroke-width:2}
.revenue-viz .revenue-point-label{fill:var(--text);font-size:10px;font-weight:600;text-anchor:middle}
.revenue-note{margin:10px 0 0;font-size:11px}

/* System health failure bars (mock line 24) */
.failure-bars{display:grid;gap:15px}
.failure-row{display:grid;grid-template-columns:145px 1fr 28px;align-items:center;gap:10px}
.failure-row span{color:var(--muted);font-size:12px}
.failure-track{height:10px;overflow:hidden;border-radius:2px;background:#171d31}
.failure-track i{display:block;width:var(--width);height:100%;background:var(--red);opacity:.85}

/* Detail views, tables, badges (mock line 25) */
.detail-view{padding-top:4px}
.breadcrumb{justify-content:flex-start;margin-bottom:22px;color:var(--muted)}
.breadcrumb a{color:var(--accent)}
.metric-strip{display:grid;grid-template-columns:repeat(4,1fr);margin:26px 0 4px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.detail-metric{padding:20px;border-left:1px solid var(--line)}
.detail-metric:first-child{padding-left:0;border-left:0}
.detail-metric span{color:var(--muted);font-size:12px}
.detail-metric strong{display:block;margin-top:7px;font-size:27px}
.detail-section{padding:26px 0;border-bottom:1px solid var(--line)}
.table-wrap{overflow:auto}
table{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}
th,td{padding:13px 11px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-size:12px;font-weight:500}
th:first-child,td:first-child{padding-left:0}
.badge{display:inline-block;padding:4px 7px;border-radius:4px;background:#252b36;font-size:12px}
.badge.bad{color:var(--red)}
.badge.warn{color:var(--amber)}
.badge.good{color:var(--green)}
.group-badge{display:inline-block;padding:3px 8px;border-radius:999px;background:#1c2a44;color:var(--accent);font-size:11px;font-weight:600;text-transform:capitalize}
.group-badge.is-admin{color:var(--amber)}
.group-badge.is-paid{color:var(--green)}
.group-badge.is-free{color:var(--muted)}
.group-badge.is-internal{color:var(--violet)}
.freshness-legend{padding-top:24px;color:var(--muted);font-size:11px}

/* Users list */
.users-toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin:20px 0 12px}
.users-toolbar form{display:flex;gap:8px;flex:1;min-width:260px;max-width:460px}
.users-table td .who{display:grid;gap:2px}
.users-table td .who small{font-size:11px}
.users-table td .row-action{min-height:32px;padding:4px 10px;font-size:12px}
.pager{display:flex;align-items:center;gap:12px;margin-top:14px;color:var(--muted);font-size:12px}
.pager span{flex:1}

/* Profile (mock line 25 + facts grid) */
.profile-head{align-items:flex-start}
.identity{display:flex;align-items:center;gap:15px}
.avatar{display:grid;place-items:center;width:52px;height:52px;border-radius:50%;background:#153c48;color:var(--accent);font-size:20px}
.identity-panel{position:relative;margin-top:26px;padding:22px 54px 22px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.help-button{position:absolute;top:18px;right:0;display:grid;place-items:center;width:36px;min-height:36px;padding:0;border-radius:50%}
.milestones{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}
.milestone{padding-top:13px;border-top:3px solid var(--green)}
.tabs{display:flex;gap:6px;margin:24px 0 0;overflow:auto}
.tabs button{background:transparent;border-color:transparent;border-bottom:2px solid transparent;border-radius:0}
.tabs button[aria-selected=true]{color:var(--accent);border-bottom-color:var(--accent)}
.profile-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px 24px;margin:0}
.profile-facts dt{color:var(--muted);font-size:12px}
.profile-facts dd{margin:4px 0 0}
.evidence-list{margin:8px 0 0;padding-left:18px;color:var(--muted);font-size:12px}
.timeline{list-style:none;margin:0;padding:0;display:grid;gap:12px}
.timeline li>div{display:flex;justify-content:space-between;gap:12px}
.timeline li p{margin:4px 0 0;color:var(--muted);font-size:12px}
.load-more{margin-top:14px}

/* Dialogs (mock line 25) */
dialog{width:min(540px,90vw);padding:24px;color:var(--text);background:var(--surface);border:1px solid var(--line);border-radius:8px}
dialog::backdrop{background:rgba(0,0,0,.64)}
.dialog-head{display:flex;align-items:center;justify-content:space-between}
.dialog-head button{width:36px;min-height:36px;padding:0}
.rules-list{display:grid;gap:10px;margin:18px 0 0}
.rules-list dt{font-weight:600}
.rules-list dd{margin:2px 0 0;color:var(--muted);font-size:12px}
.dialog-actions{display:flex;justify-content:flex-end;gap:14px;margin-top:18px}
.dialog-actions .module-link{margin-top:0}

/* Responsive (mock lines 26-27, pruned) */
@media(max-width:1040px){.product-nav{display:none}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}.overview-row.priority-row{grid-template-columns:1fr;gap:0}.metric-strip{grid-template-columns:repeat(2,1fr)}.detail-metric:nth-child(3){padding-left:0;border-left:0;border-top:1px solid var(--line)}.detail-metric:nth-child(4){border-top:1px solid var(--line)}}
@media(max-width:680px){header{padding:12px 15px}.brand{font-size:17px}.brand img{width:40px;height:40px}.workspace{grid-template-columns:66px minmax(0,1fr)}aside{padding:18px 6px}aside>strong{display:none}aside a{overflow:hidden;padding:10px 7px;font-size:0;text-align:center}aside a::first-letter{font-size:14px}.analytics-parent span{display:none}.analytics-subnav{margin-left:9px;padding-left:3px}main{padding:24px 16px 40px}.page-head{flex-direction:column;align-items:stretch}.page-actions{flex-direction:row;align-items:center;justify-content:space-between}.live-row .live-head{flex-direction:column;align-items:flex-start}.range button{min-width:38px;padding-inline:7px}.filters{grid-template-columns:1fr}.snapshot-grid{grid-template-columns:1fr}.snapshot-tile+.snapshot-tile{border-left:0;border-top:1px solid var(--line)}.source-wrap{grid-template-columns:1fr;justify-items:center;gap:17px}.source-legend-six{width:100%;gap:10px 14px}.attention-note{align-items:flex-start;flex-direction:column;gap:8px}.retention-chart{grid-template-columns:55px repeat(4,1fr);gap:4px}.lifecycle-legend{grid-template-columns:repeat(2,1fr)}.value-step{grid-template-columns:112px 1fr auto;gap:8px}.value-step>span{white-space:normal}.failure-row{grid-template-columns:120px 1fr 24px}.layered-route{min-width:600px;overflow:visible}.credits-paired{min-width:560px}.priority-row:has(.layered-route),.priority-row:has(.credits-paired){overflow-x:auto}.revenue-viz{height:184px}.revenue-viz svg{height:168px}.metric-strip{grid-template-columns:1fr}.detail-metric,.detail-metric:first-child,.detail-metric:nth-child(3){padding:17px 0;border-left:0;border-top:1px solid var(--line)}.milestones{grid-template-columns:repeat(2,1fr)}}
@media(prefers-reduced-motion:reduce){.module-link:after{transition:none}}
```

- [ ] **Step 5: Run the guards and Task 3's two deferred tests; confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_page_shell.py dashboard/backend/tests/test_middleware_exemptions.py -v
```

Expected: **PASS**. `test_panel_regions_carry_no_numeric_or_percentage_literal` finds exactly the thirteen `data-panel` regions and no standalone number in any of them (the copy deliberately says "active in the last seven days" and "week-one return" rather than `7` and `Week 1`), and `test_admin_page_loads_without_a_session_header` / `test_admin_css_is_served` now return 200.

- [ ] **Step 6: Commit**

```bash
git add dashboard/frontend/admin.html dashboard/frontend/admin.css dashboard/backend/tests/test_admin_page_shell.py
git commit -m "$(cat <<'EOF'
feat: add the data-free /admin shell and its externalised stylesheet

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `js/admin-shell.js` — gate, hash router, URL-backed state, guarded requests, panel states, dialogs

**Files:**
- Create: `dashboard/frontend/js/admin-shell.js`.
- Create: `dashboard/backend/tests/_admin_dom_stub.py` (the shared node harness the module tests in Tasks 5–8 use).
- Create: `dashboard/backend/tests/test_admin_shell_frontend.py`.
- Test: `dashboard/backend/tests/test_admin_shell_frontend.py`.

**Interfaces:**
- Consumes: `GET /api/auth/me` (`{"user": {"role": ...}}`, 401 when signed out — `api/auth.py:594-618`); `window.CreditFormat.formatCreditsMicro(value)` from `js/credit-format.js` (loaded by `admin.html` after this module; used lazily at render time, never at load).
- Produces `window.AdminShell` (frozen) with:
  - constants: `ROUTES`, `RANGE_DAYS`, `LIFECYCLE_LABELS`, `OPERATIONAL_LABELS`, `COMMERCIAL_LABELS`, `USER_GROUP_LABELS`, `LIFECYCLE_RULES`, `OPERATIONAL_RULES`, `SECTION_UNAVAILABLE`, `STALE_NOTICE`, `INCOMPLETE`, `DASH`;
  - `state` (`{ route, routeId, range, filters: { group, segment, tier, internal, q, priority }, admin }`);
  - pure: `parseHash(hash) -> { route, id }`, `rangeDates(range, today: Date) -> { from, to }`, `readUrlState(search) -> { range, filters }`, `buildSearch(range, filters) -> string`, `analyticsParams({ withGroup }) -> URLSearchParams`, `userListParams({ offset, limit }) -> URLSearchParams`, `formatNumber`, `formatPercent`, `formatCredits`, `formatDateOnly`, `formatShortDay`, `formatTimestamp`, `humanize`, `availabilityIncomplete`, `freshnessLegendText(now: Date)`, `rulesEntries() -> [label, rule][]`, `el(tag, className, text)`, `clear(node)`;
  - effectful: `request(path)`, `nextSeq(surface)`, `isCurrent(surface, seq)`, `invalidateAll()`, `handleAccessLost(error) -> Promise<boolean>`, `setPanelState(panel, { busy, status, error, stale, empty })`, `openDialog(dialog, opener)`, `closeDialog(dialog)`, `openRules(opener)`, `navigate(hash)`;
  - events on `document`: `admin:route` (`detail: { route, id, range, filters }`) on every hash/popstate/filter change; `admin:retry` (`detail: { panel }`) when a panel's Retry button is pressed.

The harvest (design §7.4): the stale-response guard is `nextSeq`/`isCurrent` (from `admin-analytics.js::loadOverview` lines 492-513, one counter per surface, bumped on auth loss); URL-backed state is `readUrlState`/`buildSearch` (from `readUrlFilters`/`replaceAnalyticsUrl`, `admin-analytics.js:119-200` and `admin-analytics-value.js:200-226`), but keyed on the short names `range`, `group`, `segment`, `tier`, `internal`, `q` because the page has its own URL and no `adminTab`; the profile opens through the hash (`#users/42`), which is itself a history entry, so `pushProfileUrl`'s "Back closes the profile" property comes free and is not re-implemented; the dialogs with a return-focus `Map`, Escape and backdrop close are `openDialog`/`closeDialog` (`admin-analytics-value.js:135-148`, `1054-1072`); `availabilityIncomplete` is `admin-analytics-value.js:323-327` extended to also read the overview's `{available: false}` shape (`admin-analytics.js:244-246`); `handleAccessLost` is both files' 401/403 exit, retargeted to `/app` because there is no in-app `navigateToPage` here; the rules copy is `LIFECYCLE_RULES`/`OPERATIONAL_RULES` (`admin-analytics-value.js:54-66`, which restate §15.2/§15.3) plus §15.4's commercial-value sentence.

- [ ] **Step 1: Write the shared node harness and the failing tests**

Create `dashboard/backend/tests/_admin_dom_stub.py`:

```python
"""Node harness for the /admin page modules.

Each module is an IIFE that only registers listeners at load, so the whole file
runs under `node -e` after this stub, and a test then calls the exported
renderers with a committed fixture. The stub is a minimal DOM: enough for
`createElement`/`textContent`/`append`/`setAttribute`/`classList`/`style`, and a
`toJSON()` that flattens a subtree so a test can assert on what a renderer
built. It has no `innerHTML` on purpose -- a renderer that reaches for it
throws here and fails the test before the source-shape pin in
test_admin_page_modules.py ever runs.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def source(name: str) -> str:
    return (FRONTEND / "js" / name).read_text(encoding="utf-8")


def fixture(name: str) -> str:
    """The fixture as a JSON literal ready to paste into a scenario."""
    return (FIXTURES / name).read_text(encoding="utf-8")


DOM_STUB = r"""
const nav = [];
const docListeners = {};
const winListeners = {};
class Text {
  constructor(text) { this.textContent = String(text); this.nodeType = 3; }
  toJSON() { return { text: this.textContent }; }
}
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.nodeType = 1;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.style = { setProperty(key, value) { this[key] = String(value); } };
    this._text = '';
    this._classes = new Set();
    this.hidden = false;
    this.disabled = false;
    this.listeners = {};
    this.classList = {
      add: (...names) => names.forEach((name) => this._classes.add(name)),
      remove: (...names) => names.forEach((name) => this._classes.delete(name)),
      toggle: (name, on) => { const next = on === undefined ? !this._classes.has(name) : Boolean(on); if (next) this._classes.add(name); else this._classes.delete(name); return next; },
      contains: (name) => this._classes.has(name),
    };
  }
  get className() { return [...this._classes].join(' '); }
  set className(value) { this._classes = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get textContent() { return this.children.length ? this.children.map((child) => child.textContent).join('') : this._text; }
  set textContent(value) { this.children = []; this._text = String(value); }
  get firstChild() { return this.children[0] || null; }
  get isConnected() { return true; }
  appendChild(child) {
    if (child.tagName === '#FRAGMENT') { child.children.slice().forEach((item) => this.appendChild(item)); child.children = []; return child; }
    this.children.push(child); child.parentNode = this; return child;
  }
  append(...items) { items.forEach((item) => this.appendChild(typeof item === 'string' ? new Text(item) : item)); }
  replaceChildren(...items) { this.children = []; this.append(...items); }
  removeChild(child) { this.children = this.children.filter((item) => item !== child); return child; }
  setAttribute(key, value) { this.attributes[key] = String(value); if (key === 'class') this.className = value; }
  getAttribute(key) { return key in this.attributes ? this.attributes[key] : null; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  dispatchEvent() { return true; }
  focus() { globalThis.focused = this; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  toJSON() {
    const style = Object.fromEntries(Object.entries(this.style).filter(([, value]) => typeof value !== 'function'));
    return {
      tag: this.tagName,
      class: this.className || undefined,
      attrs: Object.keys(this.attributes).length ? this.attributes : undefined,
      dataset: Object.keys(this.dataset).length ? this.dataset : undefined,
      style: Object.keys(style).length ? style : undefined,
      hidden: this.hidden || undefined,
      text: this.children.length ? undefined : this._text,
      children: this.children.length ? this.children.map((child) => child.toJSON()) : undefined,
    };
  }
}
const elements = {};
globalThis.CustomEvent = class { constructor(type, init) { this.type = type; this.detail = init && init.detail; } };
globalThis.Event = class { constructor(type) { this.type = type; } };
globalThis.document = {
  createElement: (tag) => new Node(tag),
  createElementNS: (_ns, tag) => new Node(tag),
  createTextNode: (text) => new Text(text),
  createDocumentFragment: () => new Node('#fragment'),
  getElementById: (id) => elements[id] || null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener(name, fn) { (docListeners[name] ||= []).push(fn); },
  dispatchEvent(event) { (docListeners[event.type] || []).forEach((fn) => fn(event)); return true; },
  activeElement: null,
  documentElement: new Node('html'),
};
globalThis.window = {
  location: { href: 'https://atl.example/admin', hostname: 'atl.example', pathname: '/admin', search: '', hash: '',
    replace(url) { nav.push(['replace', url]); }, assign(url) { nav.push(['assign', url]); } },
  history: { state: null, replaceState(state, _title, url) { nav.push(['replaceState', String(url)]); }, pushState(state, _title, url) { nav.push(['pushState', String(url)]); } },
  addEventListener(name, fn) { (winListeners[name] ||= []).push(fn); },
  scrollTo() {},
};
globalThis.fetchQueue = [];
globalThis.fetchCalls = [];
globalThis.fetch = (url, options) => {
  fetchCalls.push([url, options]);
  const next = fetchQueue.shift() || { ok: true, status: 200, body: {} };
  return Promise.resolve({ ok: next.ok, status: next.status, json: () => Promise.resolve(next.body) });
};
function register(id, node) { elements[id] = node; return node; }
function flatten(node, out = []) { out.push(node); (node.children || []).forEach((child) => flatten(child, out)); return out; }
function byClass(node, name) { return flatten(node).filter((item) => item._classes && item._classes.has(name)); }
function byTag(node, tag) { return flatten(node).filter((item) => item.tagName === tag.toUpperCase()); }
function texts(nodes) { return nodes.map((item) => item.textContent); }
function panelStub() {
  const panel = new Node('section');
  const parts = { headline: new Node('strong'), body: new Node('div'), status: new Node('p'), error: new Node('p'), errorText: new Node('span'), retry: new Node('button') };
  parts.headline.textContent = '—';
  parts.error.appendChild(parts.errorText); parts.error.appendChild(parts.retry);
  panel.querySelector = (selector) => ({ '[data-headline]': parts.headline, '[data-body]': parts.body, '[data-status]': parts.status, '[data-error]': parts.error, '[data-error] span': parts.errorText, '[data-retry]': parts.retry }[selector] || null);
  return { panel, parts };
}
"""


def run_node(*parts: str, timeout: int = 30) -> object:
    """Run the stub, the given sources/scenario in order, and parse the last line as JSON."""
    script = "\n".join([DOM_STUB, *parts])
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])
```

Create `dashboard/backend/tests/test_admin_shell_frontend.py`:

```python
"""js/admin-shell.js under node: router, URL state, formatters, guards, dialogs."""

import json

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")


def _eval(expression: str, *setup: str) -> object:
    return run_node(SHELL, CREDIT_FORMAT, *setup, f"console.log(JSON.stringify({expression}));")


def test_hash_router_knows_exactly_the_seven_routes_and_the_profile():
    assert _eval("window.AdminShell.ROUTES") == [
        "overview", "sources", "retention", "credits", "lifecycle", "health", "users",
    ]
    assert _eval("window.AdminShell.parseHash('')") == {"route": "overview", "id": None}
    assert _eval("window.AdminShell.parseHash('#health')") == {"route": "health", "id": None}
    assert _eval("window.AdminShell.parseHash('#users')") == {"route": "users", "id": None}
    assert _eval("window.AdminShell.parseHash('#users/42')") == {"route": "users", "id": "42"}
    assert _eval("window.AdminShell.parseHash('#users/abc')") == {"route": "users", "id": None}
    # No #live route (design §8.2, D16) and no orphan routes: unknown → overview.
    for unknown in ("#live", "#usage", "#revenue", "#profiles", "#funnel", "#nonsense"):
        assert _eval(f"window.AdminShell.parseHash('{unknown}')") == {"route": "overview", "id": None}, unknown


def test_range_maps_to_inclusive_utc_dates_within_the_180_day_cap():
    today = "new Date('2026-09-15T13:00:00Z')"
    assert _eval(f"window.AdminShell.rangeDates('1W', {today})") == {"from": "2026-09-09", "to": "2026-09-15"}
    assert _eval(f"window.AdminShell.rangeDates('1M', {today})") == {"from": "2026-08-17", "to": "2026-09-15"}
    # 180 inclusive days: `(end - start).days > MAX_VALUE_RANGE_DAYS` is the server's bound.
    assert _eval(f"window.AdminShell.rangeDates('1Y', {today})") == {"from": "2026-03-20", "to": "2026-09-15"}
    assert _eval(f"window.AdminShell.rangeDates('1D', {today})") == {"from": "2026-09-09", "to": "2026-09-15"}


def test_url_state_round_trips_and_drops_unknown_values():
    parsed = _eval(
        "window.AdminShell.readUrlState('?range=1M&group=organic&segment=bogus&tier=invested&internal=true&q=ada&priority=true')"
    )
    assert parsed == {
        "range": "1M",
        "filters": {"group": "organic", "segment": "", "tier": "invested", "internal": True, "q": "ada", "priority": True},
    }
    assert _eval(
        "window.AdminShell.buildSearch('1M', {group: 'organic', segment: '', tier: 'invested', internal: true, q: 'ada', priority: true})"
    ) == "?range=1M&group=organic&tier=invested&internal=true&q=ada&priority=true"
    assert _eval("window.AdminShell.buildSearch('1W', {group: '', segment: '', tier: '', internal: false, q: '', priority: false})") == ""


def test_request_parameters_use_the_declared_query_names():
    setup = (
        "window.AdminShell.state.range = '1W';"
        "window.AdminShell.state.filters = {group: 'partner', segment: 'core', tier: 'invested', internal: true, q: 'ada', priority: true};"
        "window.AdminShell.today = () => new Date('2026-09-15T00:00:00Z');"
    )
    assert _eval("window.AdminShell.analyticsParams().toString()", setup) == "from=2026-09-09&to=2026-09-15&include_internal=true"
    assert _eval("window.AdminShell.analyticsParams({withGroup: true}).toString()", setup) == "from=2026-09-09&to=2026-09-15&include_internal=true&user_group=partner"
    assert _eval("window.AdminShell.userListParams({offset: 50}).toString()", setup) == (
        "q=ada&user_group=partner&lifecycle_segment=core&commercial_tier=invested&priority=true&include_internal=true&limit=50&offset=50"
    )


def test_formatters_are_display_safe_and_fixed_locale():
    assert _eval("window.AdminShell.formatNumber(1234)") == "1,234"
    assert _eval("window.AdminShell.formatNumber(null)") == "—"
    assert _eval("window.AdminShell.formatPercent(0.6666)") == "66.7%"
    assert _eval("window.AdminShell.formatPercent(null)") == "—"
    assert _eval("window.AdminShell.formatPercent(1.5)") == "—"
    assert _eval("window.AdminShell.formatCredits(4800000)") == "4.800000 Credits"
    assert _eval("window.AdminShell.formatCredits(null)") == "—"
    assert _eval("window.AdminShell.formatDateOnly('2026-09-01')") == "Sep 1, 2026"
    assert _eval("window.AdminShell.formatShortDay('2026-09-04')") == "Sep 4"
    assert _eval("window.AdminShell.formatTimestamp('2026-08-22T11:20:00Z')") == "Aug 22, 2026, 11:20 UTC"
    assert _eval("window.AdminShell.formatTimestamp(null, 'No activity')") == "No activity"
    assert _eval("window.AdminShell.humanize('no_usable_billing_lane')") == "No Usable Billing Lane"


def test_freshness_legend_names_yesterday_utc():
    assert _eval("window.AdminShell.freshnessLegendText(new Date('2026-09-15T02:00:00Z'))") == (
        "Daily figures complete through 2026-09-14 UTC; live tiles are this instance's process state"
    )


def test_availability_incomplete_reads_both_payload_shapes():
    assert _eval("window.AdminShell.availabilityIncomplete({available: false, error_code: 'temporarily_unavailable'})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({available: true, stale: false, status: 'partial'})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({snapshot: {available: true}, growth: {available: false}})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({current: {status: 'ready'}, history: {status: 'partial'}})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({snapshot: {available: true}, growth: {available: true}})") is False
    assert _eval("window.AdminShell.availabilityIncomplete(null)") is False


def test_rules_dialog_carries_section_15_copy():
    entries = _eval("window.AdminShell.rulesEntries()")
    assert [label for label, _rule in entries] == [
        "New", "Onboarding", "Growing", "Core", "At risk", "Dormant",
        "Blocked", "Needs attention", "Healthy", "Commercial value",
    ]
    assert entries[4][1] == "Last meaningful activity was 8–29 UTC days ago."
    assert entries[9][1] == "Commercial value uses settled purchases minus refunds. Admin Grants do not count as purchases."


def test_request_seq_guard_and_access_loss():
    result = _eval(
        "(async () => {"
        "  const shell = window.AdminShell;"
        "  const a = shell.nextSeq('overview'); const b = shell.nextSeq('overview');"
        "  const stale = shell.isCurrent('overview', a); const fresh = shell.isCurrent('overview', b);"
        "  shell.invalidateAll();"
        "  const afterInvalidate = shell.isCurrent('overview', b);"
        "  const lost = await shell.handleAccessLost({status: 403});"
        "  const kept = await shell.handleAccessLost({status: 500});"
        "  return {stale, fresh, afterInvalidate, lost, kept, nav};"
        "})()"
    )
    assert result == {"stale": False, "fresh": True, "afterInvalidate": False, "lost": True, "kept": False, "nav": [["replace", "/app"]]}


def test_request_is_a_credentialed_get_that_throws_with_status():
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {users: 1}});"
        "  fetchQueue.push({ok: false, status: 503, body: {detail: 'x'}});"
        "  const ok = await window.AdminShell.request('/api/admin/stats');"
        "  let failed = null;"
        "  try { await window.AdminShell.request('/api/admin/analytics/overview'); } catch (error) { failed = error.status; }"
        "  return {ok, failed, calls: fetchCalls.map(([url, options]) => [url, options.method, options.credentials])};"
        "})()"
    )
    assert result == {
        "ok": {"users": 1},
        "failed": 503,
        "calls": [["/api/admin/stats", "GET", "include"], ["/api/admin/analytics/overview", "GET", "include"]],
    }


def test_gate_sends_non_admins_to_the_app():
    for body, ok, expected in (
        ({"user": {"role": "admin"}}, True, {"admin": True, "nav": []}),
        ({"user": {"role": "user"}}, True, {"admin": False, "nav": [["replace", "/app"]]}),
        ({"detail": "Not authenticated"}, False, {"admin": False, "nav": [["replace", "/app"]]}),
    ):
        result = _eval(
            "(async () => {"
            f"  fetchQueue.push({{ok: {str(ok).lower()}, status: {200 if ok else 401}, body: {json.dumps(body)}}});"
            "  const admin = await window.AdminShell.gate();"
            "  return {admin, nav};"
            "})()"
        )
        assert result == expected, body


def test_set_panel_state_covers_loading_empty_error_and_stale():
    result = _eval(
        "(() => {"
        "  const {panel, parts} = panelStub();"
        "  const shell = window.AdminShell;"
        "  shell.setPanelState(panel, {busy: true});"
        "  const loading = [panel.getAttribute('aria-busy'), parts.status.hidden];"
        "  shell.setPanelState(panel, {busy: false, error: shell.SECTION_UNAVAILABLE});"
        "  const errored = [panel.getAttribute('aria-busy'), parts.error.hidden, parts.errorText.textContent];"
        "  shell.setPanelState(panel, {busy: false, stale: true});"
        "  const stale = [panel.classList.contains('is-stale'), parts.status.textContent, parts.status.hidden];"
        "  shell.setPanelState(panel, {busy: false, status: shell.INCOMPLETE});"
        "  const incomplete = [panel.classList.contains('is-stale'), parts.status.textContent, parts.error.hidden];"
        "  return {loading, errored, stale, incomplete};"
        "})()"
    )
    assert result == {
        "loading": ["true", True],
        "errored": ["false", False, "This section is temporarily unavailable."],
        "stale": [True, "Showing the last successful response; refresh failed.", False],
        "incomplete": [False, "Incomplete data", True],
    }


def test_dialogs_return_focus_to_the_opener():
    result = _eval(
        "(() => {"
        "  const dialog = document.createElement('dialog'); dialog.id = 'evidenceDialog';"
        "  dialog.showModal = () => { dialog.open = true; }; dialog.close = () => { dialog.open = false; };"
        "  const opener = document.createElement('button'); globalThis.focused = null;"
        "  window.AdminShell.openDialog(dialog, opener);"
        "  const opened = dialog.open;"
        "  window.AdminShell.closeDialog(dialog);"
        "  return {opened, closed: !dialog.open, returned: globalThis.focused === opener};"
        "})()"
    )
    assert result == {"opened": True, "closed": True, "returned": True}
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_shell_frontend.py -v
```

Expected: **ERROR at collection** — `FileNotFoundError` for `dashboard/frontend/js/admin-shell.js` from `source("admin-shell.js")`. (With `node` absent the file skips instead; that is the repo convention, not a pass.)

- [ ] **Step 3: Create `js/admin-shell.js`**

```js
/** /admin console shell: courtesy gate, hash router, URL state, guarded GETs, dialogs. */
(function () {
  'use strict';

  // Design §7.2: one page, seven routes plus the profile (#users/{id}). There is
  // deliberately no live-operations route -- the live row has no detail page (§8.2, D16).
  const ROUTES = ['overview', 'sources', 'retention', 'credits', 'lifecycle', 'health', 'users'];
  const DETAIL_ROUTES = ['sources', 'retention', 'credits', 'lifecycle', 'health'];
  // 1D is cut (§8.2): no cross-user source finer than a day exists. 1Y is 180
  // inclusive days because the value routes reject a window wider than
  // MAX_VALUE_RANGE_DAYS (180) measured as (end - start).days with end = to + 1.
  const RANGE_DAYS = Object.freeze({ '1W': 7, '1M': 30, '1Y': 180 });
  const USER_GROUPS = ['internal', 'invited', 'organic', 'competition', 'partner', 'unknown'];
  const LIFECYCLE_SEGMENTS = ['new', 'onboarding', 'growing', 'core', 'at_risk', 'dormant'];
  const COMMERCIAL_TIERS = ['unpaid', 'starter', 'invested', 'high_value'];
  const LIFECYCLE_LABELS = Object.freeze({
    new: 'New', onboarding: 'Onboarding', growing: 'Growing',
    core: 'Core', at_risk: 'At risk', dormant: 'Dormant',
  });
  const OPERATIONAL_LABELS = Object.freeze({
    blocked: 'Blocked', needs_attention: 'Needs attention', healthy: 'Healthy',
  });
  const COMMERCIAL_LABELS = Object.freeze({
    unpaid: 'Unpaid', starter: 'Starter', invested: 'Invested', high_value: 'High value',
  });
  const USER_GROUP_LABELS = Object.freeze({
    internal: 'Internal', invited: 'Invited', organic: 'Organic',
    competition: 'Competition', partner: 'Partner', unknown: 'Unknown',
  });
  // Copy is design §15.2-§15.4; harvested from admin-analytics-value.js (§7.4).
  const LIFECYCLE_RULES = Object.freeze({
    new: 'Account is 0–6 UTC days old and has no successful backtest.',
    onboarding: 'No successful backtest yet; the account is no longer New and is not inactive.',
    growing: 'Activated and active in the last 7 UTC days, below the Core repeat-value threshold.',
    core: 'At least 3 active days and 3 successful backtests in 30 UTC days, active in the last 7 days.',
    at_risk: 'Last meaningful activity was 8–29 UTC days ago.',
    dormant: 'Last meaningful activity was at least 30 UTC days ago.',
  });
  const OPERATIONAL_RULES = Object.freeze({
    blocked: 'A current issue prevents a core action, such as an unavailable billing lane.',
    needs_attention: 'A supported issue needs operator review but may not block every action.',
    healthy: 'No supported current blocker or attention condition matched.',
  });
  const COMMERCIAL_RULE = 'Commercial value uses settled purchases minus refunds. Admin Grants do not count as purchases.';
  const SECTION_UNAVAILABLE = 'This section is temporarily unavailable.';
  const STALE_NOTICE = 'Showing the last successful response; refresh failed.';
  const INCOMPLETE = 'Incomplete data';
  const DASH = '—';
  const LOCALE = 'en-US';

  const state = {
    route: 'overview',
    routeId: null,
    range: '1W',
    filters: { group: '', segment: '', tier: '', internal: false, q: '', priority: false },
    admin: false,
    seq: {},
  };
  const returnFocus = new Map();

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function isoDay(date) {
    return date.toISOString().slice(0, 10);
  }

  function today() {
    return new Date();
  }

  function parseHash(hash) {
    const raw = String(hash || '').replace(/^#/, '');
    const [head, tail] = raw.split('/');
    if (head === 'users') {
      return { route: 'users', id: /^\d+$/.test(tail || '') ? tail : null };
    }
    return { route: ROUTES.includes(head) ? head : 'overview', id: null };
  }

  function rangeDates(range, now) {
    const days = RANGE_DAYS[range] || RANGE_DAYS['1W'];
    const base = now || api.today();
    const end = new Date(Date.UTC(base.getUTCFullYear(), base.getUTCMonth(), base.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - (days - 1));
    return { from: isoDay(start), to: isoDay(end) };
  }

  function readUrlState(search) {
    const params = new URLSearchParams(search || '');
    const range = params.get('range');
    const group = params.get('group') || '';
    const segment = params.get('segment') || '';
    const tier = params.get('tier') || '';
    return {
      range: Object.hasOwn(RANGE_DAYS, range) ? range : '1W',
      filters: {
        group: USER_GROUPS.includes(group) ? group : '',
        segment: LIFECYCLE_SEGMENTS.includes(segment) ? segment : '',
        tier: COMMERCIAL_TIERS.includes(tier) ? tier : '',
        internal: params.get('internal') === 'true',
        q: String(params.get('q') || '').slice(0, 100),
        priority: params.get('priority') === 'true',
      },
    };
  }

  function buildSearch(range, filters) {
    const params = new URLSearchParams();
    if (range && range !== '1W') params.set('range', range);
    if (filters.group) params.set('group', filters.group);
    if (filters.segment) params.set('segment', filters.segment);
    if (filters.tier) params.set('tier', filters.tier);
    if (filters.internal) params.set('internal', 'true');
    if (filters.q) params.set('q', filters.q);
    if (filters.priority) params.set('priority', 'true');
    const text = params.toString();
    return text ? `?${text}` : '';
  }

  function analyticsParams({ withGroup = false } = {}) {
    const { from, to } = rangeDates(state.range);
    const params = new URLSearchParams();
    params.set('from', from);
    params.set('to', to);
    params.set('include_internal', state.filters.internal ? 'true' : 'false');
    if (withGroup && state.filters.group) params.set('user_group', state.filters.group);
    return params;
  }

  function userListParams({ offset = 0, limit = 50 } = {}) {
    const params = new URLSearchParams();
    const filters = state.filters;
    if (filters.q) params.set('q', filters.q);
    if (filters.group) params.set('user_group', filters.group);
    if (filters.segment) params.set('lifecycle_segment', filters.segment);
    if (filters.tier) params.set('commercial_tier', filters.tier);
    if (filters.priority) params.set('priority', 'true');
    params.set('include_internal', filters.internal ? 'true' : 'false');
    params.set('limit', String(limit));
    params.set('offset', String(offset));
    return params;
  }

  function formatNumber(value) {
    if (value == null || value === '') return DASH;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? new Intl.NumberFormat(LOCALE).format(numeric) : DASH;
  }

  function formatPercent(value) {
    if (value == null || value === '') return DASH;
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) return DASH;
    return new Intl.NumberFormat(LOCALE, { style: 'percent', maximumFractionDigits: 1 }).format(numeric);
  }

  function formatCredits(value) {
    const formatted = window.CreditFormat.formatCreditsMicro(value);
    return formatted === DASH ? DASH : `${formatted} Credits`;
  }

  function formatDateOnly(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(date.getTime())
      ? new Intl.DateTimeFormat(LOCALE, { dateStyle: 'medium', timeZone: 'UTC' }).format(date)
      : fallback;
  }

  function formatShortDay(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(date.getTime())
      ? new Intl.DateTimeFormat(LOCALE, { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date)
      : fallback;
  }

  function formatTimestamp(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return fallback;
    const day = new Intl.DateTimeFormat(LOCALE, { dateStyle: 'medium', timeZone: 'UTC' }).format(date);
    const time = new Intl.DateTimeFormat(LOCALE, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: 'UTC' }).format(date);
    return `${day}, ${time} UTC`;
  }

  function humanize(value) {
    const key = String(value || 'unknown');
    return key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function incompleteItem(item) {
    if (!item || typeof item !== 'object') return false;
    if (item.available === false) return true;
    return Boolean(item.status && item.status !== 'ready');
  }

  function availabilityIncomplete(availability) {
    if (!availability || typeof availability !== 'object') return false;
    if (incompleteItem(availability)) return true;
    return Object.values(availability).some(incompleteItem);
  }

  function freshnessLegendText(now) {
    const base = now || api.today();
    const yesterday = new Date(Date.UTC(base.getUTCFullYear(), base.getUTCMonth(), base.getUTCDate() - 1));
    return `Daily figures complete through ${isoDay(yesterday)} UTC; live tiles are this instance's process state`;
  }

  function rulesEntries() {
    return [
      ...LIFECYCLE_SEGMENTS.map((segment) => [LIFECYCLE_LABELS[segment], LIFECYCLE_RULES[segment]]),
      ...Object.keys(OPERATIONAL_LABELS).map((key) => [OPERATIONAL_LABELS[key], OPERATIONAL_RULES[key]]),
      ['Commercial value', COMMERCIAL_RULE],
    ];
  }

  async function request(path) {
    const response = await fetch(path, {
      method: 'GET',
      credentials: 'include',
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) {
      const error = new Error(`Request failed with status ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function nextSeq(surface) {
    state.seq[surface] = (state.seq[surface] || 0) + 1;
    return state.seq[surface];
  }

  function isCurrent(surface, seq) {
    return state.seq[surface] === seq;
  }

  function invalidateAll() {
    Object.keys(state.seq).forEach((surface) => { state.seq[surface] += 1; });
  }

  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    invalidateAll();
    state.admin = false;
    window.location.replace('/app');
    return true;
  }

  // Courtesy redirect, not the gate (design D7): Vercel serves this HTML as a
  // static file with no session access, so the real gate is require_admin on
  // every /api/admin/* route. A non-admin who defeats this sees empty panels
  // and 403s. Until the probe resolves the shell shows placeholders, never
  // sample numbers.
  async function gate() {
    try {
      const response = await fetch('/api/auth/me', { method: 'GET', credentials: 'include', headers: { Accept: 'application/json' } });
      const body = response.ok ? await response.json() : null;
      const user = body && body.user;
      if (!user || user.role !== 'admin') {
        window.location.replace('/app');
        return false;
      }
      state.admin = true;
      return true;
    } catch (_error) {
      window.location.replace('/app');
      return false;
    }
  }

  function setPanelState(panel, { busy = false, status = '', error = '', stale = false, empty = false } = {}) {
    if (!panel) return;
    panel.setAttribute('aria-busy', busy ? 'true' : 'false');
    panel.classList.toggle('is-stale', Boolean(stale));
    const statusNode = panel.querySelector('[data-status]');
    if (statusNode) {
      const text = stale ? STALE_NOTICE : status;
      statusNode.textContent = text || '';
      statusNode.hidden = !text;
    }
    const errorNode = panel.querySelector('[data-error]');
    if (errorNode) {
      const textNode = panel.querySelector('[data-error] span');
      if (textNode) textNode.textContent = error || '';
      errorNode.hidden = !error;
    }
    const body = panel.querySelector('[data-body]');
    if (body && empty) {
      clear(body);
      body.appendChild(el('p', 'panel-empty', typeof empty === 'string' ? empty : 'Nothing to show for this range.'));
    }
  }

  function openDialog(dialog, opener) {
    if (!dialog || typeof dialog.showModal !== 'function') return;
    returnFocus.set(dialog.id, opener || document.activeElement);
    dialog.showModal();
    dialog.querySelector('[data-dialog-initial-focus]')?.focus();
  }

  function closeDialog(dialog) {
    if (!dialog?.open) return;
    dialog.close();
    const opener = returnFocus.get(dialog.id);
    returnFocus.delete(dialog.id);
    if (opener?.isConnected) opener.focus();
  }

  function fillRulesDialog() {
    const list = document.getElementById('rulesList');
    if (!list) return;
    clear(list);
    rulesEntries().forEach(([label, rule]) => {
      const wrapper = el('div');
      wrapper.appendChild(el('dt', '', label));
      wrapper.appendChild(el('dd', '', rule));
      list.appendChild(wrapper);
    });
  }

  function openRules(opener) {
    fillRulesDialog();
    openDialog(document.getElementById('rulesDialog'), opener);
  }

  function writeUrl() {
    if (!window.history?.replaceState) return;
    const search = buildSearch(state.range, state.filters);
    window.history.replaceState(window.history.state, '', `${window.location.pathname}${search}${window.location.hash}`);
  }

  function announce() {
    document.dispatchEvent(new CustomEvent('admin:route', {
      detail: { route: state.route, id: state.routeId, range: state.range, filters: { ...state.filters } },
    }));
  }

  function showView(id, visible) {
    const node = document.getElementById(id);
    if (node) node.hidden = !visible;
  }

  function route() {
    const parsed = parseHash(window.location.hash);
    if (window.location.hash && parsed.route === 'overview' && window.location.hash !== '#overview') {
      window.location.hash = 'overview';
      return;
    }
    state.route = parsed.route;
    state.routeId = parsed.id;
    showView('overview', parsed.route === 'overview');
    showView('detail', DETAIL_ROUTES.includes(parsed.route));
    showView('usersView', parsed.route === 'users' && !parsed.id);
    showView('profile', parsed.route === 'users' && Boolean(parsed.id));
    document.querySelectorAll('#analyticsSubnav a[data-route]').forEach((link) => {
      link.classList.toggle('active', link.dataset.route === parsed.route);
    });
    window.scrollTo(0, 0);
    announce();
  }

  function navigate(hash) {
    window.location.hash = String(hash).replace(/^#/, '');
  }

  function syncControls() {
    document.querySelectorAll('.range button[data-range]').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.range === state.range ? 'true' : 'false');
    });
    const group = document.getElementById('filterGroup');
    const segment = document.getElementById('filterSegment');
    const tier = document.getElementById('filterTier');
    const internal = document.getElementById('filterInternal');
    if (group) group.value = state.filters.group;
    if (segment) segment.value = state.filters.segment;
    if (tier) tier.value = state.filters.tier;
    if (internal) internal.checked = state.filters.internal;
    document.querySelectorAll('.selected-range').forEach((node) => {
      node.textContent = `Selected range · ${state.range}`;
    });
    const legend = document.getElementById('freshnessLegend');
    if (legend) legend.textContent = freshnessLegendText();
  }

  function readUrlIntoState() {
    const parsed = readUrlState(window.location.search);
    state.range = parsed.range;
    state.filters = parsed.filters;
  }

  function setRange(range) {
    if (!Object.hasOwn(RANGE_DAYS, range)) return;
    state.range = range;
    syncControls();
    writeUrl();
    announce();
  }

  function setFilters(patch) {
    state.filters = { ...state.filters, ...patch };
    syncControls();
    writeUrl();
    announce();
  }

  function bindControls() {
    document.querySelectorAll('.range button[data-range]').forEach((button) => {
      button.addEventListener('click', () => setRange(button.dataset.range));
    });
    document.getElementById('filterGroup')?.addEventListener('change', (event) => setFilters({ group: event.target.value }));
    document.getElementById('filterSegment')?.addEventListener('change', (event) => setFilters({ segment: event.target.value }));
    document.getElementById('filterTier')?.addEventListener('change', (event) => setFilters({ tier: event.target.value }));
    document.getElementById('filterInternal')?.addEventListener('change', (event) => setFilters({ internal: Boolean(event.target.checked) }));
    document.getElementById('filters')?.addEventListener('submit', (event) => event.preventDefault());
    document.getElementById('analyticsParent')?.addEventListener('click', (event) => {
      event.preventDefault();
      const subnav = document.getElementById('analyticsSubnav');
      if (!subnav) return;
      subnav.hidden = !subnav.hidden;
      event.currentTarget.setAttribute('aria-expanded', String(!subnav.hidden));
    });
    document.querySelectorAll('[data-retry]').forEach((button) => {
      button.addEventListener('click', () => {
        const panel = button.closest('[data-panel]');
        document.dispatchEvent(new CustomEvent('admin:retry', { detail: { panel: panel?.dataset.panel || '' } }));
      });
    });
    document.querySelectorAll('dialog').forEach((dialog) => {
      dialog.querySelectorAll('[data-dialog-close]').forEach((button) => {
        button.addEventListener('click', () => closeDialog(dialog));
      });
      dialog.addEventListener('cancel', (event) => { event.preventDefault(); closeDialog(dialog); });
      dialog.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        closeDialog(dialog);
      });
      dialog.addEventListener('click', (event) => { if (event.target === dialog) closeDialog(dialog); });
    });
  }

  async function boot() {
    bindControls();
    const admin = await gate();
    if (!admin) return;
    readUrlIntoState();
    syncControls();
    window.addEventListener('hashchange', route);
    window.addEventListener('popstate', () => { readUrlIntoState(); syncControls(); route(); });
    route();
  }

  const api = {
    ROUTES, RANGE_DAYS, LIFECYCLE_LABELS, OPERATIONAL_LABELS, COMMERCIAL_LABELS, USER_GROUP_LABELS,
    LIFECYCLE_RULES, OPERATIONAL_RULES, SECTION_UNAVAILABLE, STALE_NOTICE, INCOMPLETE, DASH,
    state, today,
    parseHash, rangeDates, readUrlState, buildSearch, analyticsParams, userListParams,
    formatNumber, formatPercent, formatCredits, formatDateOnly, formatShortDay, formatTimestamp, humanize,
    availabilityIncomplete, freshnessLegendText, rulesEntries, el, clear,
    request, nextSeq, isCurrent, invalidateAll, handleAccessLost, gate,
    setPanelState, openDialog, closeDialog, openRules, navigate, setFilters,
  };
  window.AdminShell = api;
  document.addEventListener('DOMContentLoaded', boot);
})();
```

`window.AdminShell` is a plain object rather than `Object.freeze`d so the tests can substitute `AdminShell.today` (`rangeDates`/`freshnessLegendText` read `api.today()` for exactly that reason); nothing else writes to it.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_shell_frontend.py -v
```

Expected: **PASS** on all twelve tests. `test_formatters_are_display_safe_and_fixed_locale` passes on any machine because every `Intl` call names `en-US` and `timeZone: 'UTC'` explicitly rather than inheriting the host's locale.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/admin-shell.js dashboard/backend/tests/_admin_dom_stub.py dashboard/backend/tests/test_admin_shell_frontend.py
git commit -m "$(cat <<'EOF'
feat: add the /admin shell module with gate, router and guarded requests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `js/admin-live.js` — the live-operations row from `GET /api/admin/stats` only

**Files:**
- Create: `dashboard/frontend/js/admin-live.js`.
- Create: `dashboard/backend/tests/test_admin_live_frontend.py`.
- Test: `dashboard/backend/tests/test_admin_live_frontend.py` (uses `tests/fixtures/admin_analytics/admin_stats.json` from Task 2).

**Interfaces:**
- Consumes: `window.AdminShell` (`request`, `nextSeq`, `isCurrent`, `handleAccessLost`, `setPanelState`, `formatNumber`, `formatTimestamp`, `el`, `clear`, `DASH`, `SECTION_UNAVAILABLE`); `GET /api/admin/stats` → `{ users, admins, agents, active_dashboard_backtests, max_active_dashboard_backtests, credits_metering_enabled, default_credits }` (Task 2).
- Produces `window.AdminLive = { renderTiles(stats) -> element, load() -> Promise, STATS_PATH, CAVEAT }`. `renderTiles` is pure: three `article.snapshot-tile` children (Total users, Total agents, Backtests running · this instance with an "of N slots" meta line) inside a `div`; missing values render `—`. No detail route, no `#live` link (§8.2, D16).

Design D16/D17: this module never touches `/api/admin/analytics/*`; the numbers are the component's own process state, and the row says so.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_admin_live_frontend.py`:

```python
"""js/admin-live.js under node: the live row reads /api/admin/stats and nothing else."""

import re

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
LIVE = source("admin-live.js")
STATS = fixture("admin_stats.json")


def _eval(expression: str, *setup: str) -> object:
    return run_node(SHELL, CREDIT_FORMAT, LIVE, *setup, f"console.log(JSON.stringify({expression}));")


def test_tiles_render_users_agents_and_running_with_the_real_ceiling():
    result = _eval(
        "(() => {"
        f"  const root = window.AdminLive.renderTiles({STATS});"
        "  return byClass(root, 'snapshot-tile').map((tile) => ["
        "    byClass(tile, 'snapshot-label')[0].textContent,"
        "    byClass(tile, 'snapshot-value')[0].textContent,"
        "    (byClass(tile, 'snapshot-meta')[0] || {textContent: null}).textContent,"
        "  ]);"
        "})()"
    )
    assert result == [
        ["Total users", "412", None],
        ["Total agents", "4,380", None],
        ["Backtests running · this instance", "2", "of 5 slots on this instance"],
    ]


def test_tiles_show_dashes_for_a_missing_payload():
    result = _eval(
        "(() => { const root = window.AdminLive.renderTiles({}); return texts(byClass(root, 'snapshot-value')).concat(texts(byClass(root, 'snapshot-meta'))); })()"
    )
    assert result == ["—", "—", "—", "—"]


def test_load_paints_the_row_and_marks_the_update_time():
    result = _eval(
        "(async () => {"
        "  register('liveTiles', document.createElement('div'));"
        "  register('liveUpdated', document.createElement('small'));"
        "  const {panel} = panelStub();"
        "  document.querySelector = (selector) => selector === '[data-panel=\"live\"]' ? panel : null;"
        f"  fetchQueue.push({{ok: true, status: 200, body: {STATS}}});"
        "  await window.AdminLive.load();"
        "  return {calls: fetchCalls.map(([url]) => url), values: texts(byClass(document.getElementById('liveTiles'), 'snapshot-value')), updated: document.getElementById('liveUpdated').textContent, busy: panel.getAttribute('aria-busy')};"
        "})()"
    )
    assert result["calls"] == ["/api/admin/stats"]
    assert result["values"] == ["412", "4,380", "2"]
    assert re.fullmatch(r"Updated [A-Z][a-z]{2} \d{1,2}, \d{4}, \d{2}:\d{2} UTC", result["updated"]), result["updated"]
    assert result["busy"] == "false"


def test_load_failure_keeps_the_last_good_row_and_says_so():
    result = _eval(
        "(async () => {"
        "  register('liveTiles', document.createElement('div'));"
        "  register('liveUpdated', document.createElement('small'));"
        "  const {panel, parts} = panelStub();"
        "  document.querySelector = (selector) => selector === '[data-panel=\"live\"]' ? panel : null;"
        f"  fetchQueue.push({{ok: true, status: 200, body: {STATS}}});"
        "  await window.AdminLive.load();"
        "  fetchQueue.push({ok: false, status: 503, body: {}});"
        "  await window.AdminLive.load();"
        "  const stale = {values: texts(byClass(document.getElementById('liveTiles'), 'snapshot-value')), error: parts.errorText.textContent, isStale: panel.classList.contains('is-stale')};"
        "  fetchQueue.push({ok: false, status: 403, body: {}});"
        "  await window.AdminLive.load();"
        "  return {stale, nav};"
        "})()"
    )
    assert result["stale"] == {"values": ["412", "4,380", "2"], "error": "This section is temporarily unavailable.", "isStale": True}
    assert result["nav"] == [["replace", "/app"]]


def test_module_reads_only_the_stats_route():
    assert LIVE.count("/api/admin/") == 1
    assert "/api/admin/stats" in LIVE
    assert "/api/admin/analytics" not in LIVE
    assert "#live" not in LIVE
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_live_frontend.py -v
```

Expected: **ERROR at collection** — `FileNotFoundError` for `js/admin-live.js`.

- [ ] **Step 3: Create `js/admin-live.js`**

```js
/** /admin live-operations row. Reads GET /api/admin/stats and nothing else (design D16, D17). */
(function () {
  'use strict';

  const STATS_PATH = '/api/admin/stats';
  // Two caveats the row must carry (design D17): the slot ledger is per-process,
  // and the ceiling is the parsed MAX_ACTIVE_DASHBOARD_BACKTESTS, not a fiction.
  const CAVEAT = "Counters are this instance's process state; a second replica would under-report them.";
  const state = { data: null };

  function shell() {
    return window.AdminShell;
  }

  function tile(label, value, meta, extraClass) {
    const s = shell();
    const article = s.el('article', extraClass ? `snapshot-tile ${extraClass}` : 'snapshot-tile');
    article.appendChild(s.el('span', 'snapshot-label', label));
    article.appendChild(s.el('strong', 'snapshot-value', value));
    if (meta !== undefined) article.appendChild(s.el('small', 'snapshot-meta', meta));
    return article;
  }

  function renderTiles(stats) {
    const s = shell();
    const ceiling = Number(stats?.max_active_dashboard_backtests);
    const root = s.el('div');
    root.appendChild(tile('Total users', s.formatNumber(stats?.users)));
    root.appendChild(tile('Total agents', s.formatNumber(stats?.agents)));
    root.appendChild(tile(
      'Backtests running · this instance',
      s.formatNumber(stats?.active_dashboard_backtests),
      Number.isFinite(ceiling) ? `of ${s.formatNumber(ceiling)} slots on this instance` : s.DASH,
      'runs'
    ));
    return root;
  }

  function paint(stats) {
    const s = shell();
    const tiles = document.getElementById('liveTiles');
    if (tiles) {
      s.clear(tiles);
      Array.from(renderTiles(stats).children).forEach((node) => tiles.appendChild(node));
    }
    const updated = document.getElementById('liveUpdated');
    if (updated) updated.textContent = `Updated ${s.formatTimestamp(new Date().toISOString())}`;
  }

  async function load() {
    const s = shell();
    const panel = document.querySelector('[data-panel="live"]');
    const seq = s.nextSeq('live');
    s.setPanelState(panel, { busy: true });
    try {
      const stats = await s.request(STATS_PATH);
      if (!s.isCurrent('live', seq)) return;
      state.data = stats;
      paint(stats);
      s.setPanelState(panel, { busy: false });
    } catch (error) {
      if (!s.isCurrent('live', seq)) return;
      if (await s.handleAccessLost(error)) return;
      s.setPanelState(panel, { busy: false, error: s.SECTION_UNAVAILABLE, stale: Boolean(state.data) });
    }
  }

  document.addEventListener('admin:route', (event) => {
    if (event.detail?.route === 'overview') load();
  });
  document.addEventListener('admin:retry', (event) => {
    if (event.detail?.panel === 'live') load();
  });

  window.AdminLive = { renderTiles, load, STATS_PATH, CAVEAT };
})();
```

`paint` copies the detached root's children with `Array.from(...)` before moving them, and that choice is load-bearing in both environments. In a browser `element.children` is a live `HTMLCollection`: it has no `.slice`, so `.children.slice()` would throw `TypeError` on the first paint — and `test_admin_live_frontend.py` would never see it, because the stub's `children` is a plain array (`_admin_dom_stub.py`, `this.children = []`) and is *more* permissive than the real DOM. Appending a node also removes it from the live collection, so a `.forEach` directly on `children` would skip every other tile; the static copy sidesteps that. The other idiom, `while (root.firstChild) tiles.appendChild(root.firstChild)`, is correct in a browser but loops forever under the stub, whose `appendChild` does not detach a node from its previous parent. `Array.from` over an array-like is the one form both a real `HTMLCollection` and the stub's array accept. Everything the row prints comes from the payload or a formatter; the labels are the only literals.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_live_frontend.py -v
```

Expected: **PASS** on all five.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/admin-live.js dashboard/backend/tests/test_admin_live_frontend.py
git commit -m "$(cat <<'EOF'
feat: render the /admin live row from the stats route

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `js/admin-overview.js` — nine panels and five detail routes

**Files:**
- Create: `dashboard/frontend/js/admin-overview.js`.
- Create: `dashboard/backend/tests/fixtures/admin_analytics/groups.json`.
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py` — `test_fixtures_validate_against_committed_analytics_models` (lines 87-102) gains the groups fixture.
- Create: `dashboard/backend/tests/test_admin_overview_frontend.py`.
- Test: `dashboard/backend/tests/test_admin_overview_frontend.py`, `dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models`.

**Interfaces:**
- Consumes: `window.AdminShell`; the six endpoints `GET /api/admin/analytics/{overview,lifecycle,retention,commercial,operational,groups}` with `from`/`to`/`include_internal` (+ `user_group` on `/groups`), payload shapes per Task 1 and `tests/fixtures/admin_analytics/*.json`.
- Produces `window.AdminOverview` with pure renderers `renderAttention(operational, lifecycle)`, `renderActiveUsers(overview)`, `renderActivation(overview)`, `renderSources(groups)`, `renderRetention(retention)`, `renderValue(groups)`, `renderLifecycle(lifecycle)`, `renderCredits(commercial, overview)`, `renderRevenue(commercial)` — each `payload → { headline: string, body: element }` — and `detailSources(groups)`, `detailRetention(retention)`, `detailCredits(commercial)`, `detailLifecycle(lifecycle)`, `detailHealth(operational)` — each `payload → element`; plus `PANELS`, `DETAIL_NEEDS`, `state`, `paint(def)`, `loadAll(names)`, `showDetail(route)`.

Panel → source mapping is design §8.2 and §7.2's ownership table: attention ← `/operational.operational_state_counts` + `/operational.top_operational_reasons` + `/lifecycle.segment_counts.at_risk`; active users ← `/overview.daily_active_users` + `active_users_7d`; activation ← `/overview.activation_funnel` + `first_success_conversion` ("% remain" is computed client-side from the counts); sources ← `/groups.groups[].users`; retention ← `/retention.cohorts` + `summary_week_1`; reaching value ← sums of `/groups.successful_run_users` / `repeat_users`; lifecycle ← `/lifecycle.segment_counts`; credits ← `/commercial.selected_period.consumed_micro` + `/overview.billing_lane_mix`; revenue ← `/commercial.selected_period.purchased_micro` + `/commercial.purchased_by_day`. Detail routes: `#sources` ← `/groups` (six-row table with activation per group), `#retention` ← `/retention`, `#credits` ← `/commercial`, `#lifecycle` ← `/lifecycle` + the §15.2 rules, `#health` ← `/operational` (failed runs, success rate, `top_failure_categories`, `top_operational_reasons`; no "Affected users" column). Charts are CSS bars and one generated SVG (the revenue line, harvested from the mock's block 6 `drawRevenue`, with its `\\.0$` regex bug fixed); no Chart.js.

- [ ] **Step 1: Add the groups fixture and write the failing tests**

Create `dashboard/backend/tests/fixtures/admin_analytics/groups.json` (six rows in taxonomy order; totals 100 users, 52 with a result, 20 repeat — the mock's own numbers, so the sources and reaching-value panels can be checked against V8 §2):

```json
{
  "as_of": "2026-09-03T12:00:00Z",
  "groups": [
    {"group": "internal", "label": "Internal", "users": 22, "successful_run_users": 14, "repeat_users": 8, "total_runs": 120, "atl_cost_micro_usd": 8400000, "paid_users": 3},
    {"group": "invited", "label": "Invited", "users": 18, "successful_run_users": 10, "repeat_users": 4, "total_runs": 61, "atl_cost_micro_usd": 3100000, "paid_users": 2},
    {"group": "organic", "label": "Organic", "users": 22, "successful_run_users": 9, "repeat_users": 3, "total_runs": 44, "atl_cost_micro_usd": 2600000, "paid_users": 1},
    {"group": "competition", "label": "Competition", "users": 12, "successful_run_users": 7, "repeat_users": 2, "total_runs": 30, "atl_cost_micro_usd": 1500000, "paid_users": 1},
    {"group": "partner", "label": "Partner", "users": 18, "successful_run_users": 8, "repeat_users": 3, "total_runs": 39, "atl_cost_micro_usd": 2200000, "paid_users": 0},
    {"group": "unknown", "label": "Unknown", "users": 8, "successful_run_users": 4, "repeat_users": 0, "total_runs": 11, "atl_cost_micro_usd": 600000, "paid_users": 0}
  ],
  "selected_user_group": null,
  "availability": {"available": true, "stale": false, "status": "ready", "coverage_start": "2026-08-01", "coverage_end": "2026-08-31"}
}
```

In `dashboard/backend/tests/test_admin_analytics_frontend.py`, extend the imports (lines 10-17) with `GroupAnalyticsResponse` and add to `test_fixtures_validate_against_committed_analytics_models`, after `OperationalAnalyticsResponse.model_validate(load_fixture("operational.json"))`:

```python
    GroupAnalyticsResponse.model_validate(load_fixture("groups.json"))
```

Create `dashboard/backend/tests/test_admin_overview_frontend.py`:

```python
"""js/admin-overview.js under node: every panel renderer against the committed fixtures."""

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
OVERVIEW = source("admin-overview.js")
F = {name: fixture(f"{name}.json") for name in (
    "overview", "overview_partial_error", "operational", "lifecycle", "retention", "commercial", "groups",
)}


def _eval(expression: str, *setup: str) -> object:
    return run_node(SHELL, CREDIT_FORMAT, OVERVIEW, *setup, f"console.log(JSON.stringify({expression}));")


def test_active_users_bars_and_axis_follow_the_daily_series():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderActiveUsers({F['overview']});"
        "  return {headline: r.headline, bars: texts(byTag(r.body, 'b')), labels: texts(byClass(r.body, 'bar-column').map((c) => byTag(c, 'span')[0])), axis: texts(byClass(r.body, 'chart-y-axis')[0].children), heights: byClass(r.body, 'bar-column').map((c) => byTag(c, 'i')[0].style['--height'])};"
        "})()"
    )
    assert result == {
        "headline": "42",
        "bars": ["31", "36"],
        "labels": ["Aug 25", "Aug 26"],
        "axis": ["36", "24", "12", "0"],
        "heights": ["65.44444444444444%", "76%"],
    }


def test_activation_progress_computes_percent_remaining_from_the_counts():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderActivation({F['overview']});"
        "  return {headline: r.headline, names: texts(byClass(r.body, 'layer-name')), values: texts(byClass(r.body, 'layer-value')), metas: texts(byClass(r.body, 'layer-meta'))};"
        "})()"
    )
    assert result == {
        "headline": "62.5%",
        "names": ["Account created", "Credential verified", "Agent created", "First successful result"],
        "values": ["80", "67", "59", "50"],
        "metas": ["100% of cohort", "83.8% remain", "73.8% remain", "62.5% remain"],
    }


def test_sources_donut_and_legend_cover_the_six_groups():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderSources({F['groups']});"
        "  const donut = byClass(r.body, 'source-donut')[0];"
        "  return {headline: r.headline, total: donut.getAttribute('data-total'), gradient: donut.style.background.startsWith('conic-gradient('), labels: texts(byClass(r.body, 'legend-row').map((row) => byTag(row, 'span')[0])), shares: texts(byClass(r.body, 'legend-row').map((row) => byTag(row, 'b')[0]))};"
        "})()"
    )
    assert result == {
        "headline": "100",
        "total": "100\nusers",
        "gradient": True,
        "labels": ["Internal", "Invited", "Organic", "Competition", "Partner", "Unknown"],
        "shares": ["22%", "18%", "22%", "12%", "18%", "8%"],
    }


def test_retention_heatmap_leaves_immature_cells_empty():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderRetention({F['retention']});"
        "  return {headline: r.headline, rows: texts(byClass(r.body, 'row-label')), cells: byClass(r.body, 'retention-cell').map((c) => [c.textContent, c.classList.contains('empty')])};"
        "})()"
    )
    assert result == {
        "headline": "66.7%",
        "rows": ["Jul 6", "Aug 24"],
        "cells": [["3", False], ["66.7%", False], ["33.3%", False], ["33.3%", False], ["2", False], ["—", True], ["—", True], ["—", True]],
    }


def test_reaching_value_sums_the_groups():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderValue({F['groups']});"
        "  return {headline: r.headline, steps: texts(byClass(r.body, 'value-step').map((s) => byTag(s, 'b')[0]))};"
        "})()"
    )
    assert result == {"headline": "52", "steps": ["48 · 48%", "32 · 32%", "20 · 20%"]}


def test_lifecycle_bar_and_legend_follow_segment_counts():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderLifecycle({F['lifecycle']});"
        "  return {headline: r.headline, shares: byClass(r.body, 'lifecycle-bar')[0].children.map((i) => i.style['--share']), legend: texts(byClass(r.body, 'lifecycle-legend')[0].children.map((d) => byTag(d, 'b')[0]))};"
        "})()"
    )
    assert result == {"headline": "33", "shares": ["3", "6", "7", "8", "5", "4"], "legend": ["3", "6", "7", "8", "5", "4"]}


def test_credits_pairs_platform_and_byok_per_day():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderCredits({F['commercial']}, {F['overview']});"
        "  return {headline: r.headline, stems: texts(byClass(r.body, 'credit-stem').map((s) => byTag(s, 'b')[0])), days: texts(byClass(r.body, 'credit-date')), legend: texts(byClass(r.body, 'credit-legend')[0].children)};"
        "})()"
    )
    assert result == {
        "headline": "4.800000 Credits",
        "stems": ["11", "7", "14", "8"],
        "days": ["Aug 25", "Aug 26"],
        "legend": ["Platform Credits", "BYOK runs"],
    }


def test_revenue_line_is_generated_svg_with_fixed_axis_labels():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderRevenue({F['commercial']});"
        "  const svg = byTag(r.body, 'svg')[0];"
        "  return {headline: r.headline, label: svg.getAttribute('aria-label'), texts: texts(byTag(svg, 'text')), titles: texts(byTag(svg, 'title')), points: byTag(svg, 'circle').length};"
        "})()"
    )
    assert result == {
        "headline": "12.000000 Credits",
        "label": "Purchased Credits revenue trend with 2 data points",
        "texts": ["7", "3.5", "0", "5", "7", "Sep 1", "Sep 2"],
        "titles": ["Sep 1: 5.0 purchased Credits", "Sep 2: 7.0 purchased Credits"],
        "points": 2,
    }


def test_attention_counts_and_top_reason():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderAttention({F['operational']}, {F['lifecycle']});"
        "  return {headline: r.headline, counts: texts(byClass(r.body, 'attention-row').map((row) => byTag(row, 'b')[0])), note: byClass(r.body, 'attention-note')[0].children[0].children[1].textContent, link: byTag(byClass(r.body, 'attention-note')[0], 'a')[0].getAttribute('href')};"
        "})()"
    )
    assert result == {"headline": "11", "counts": ["2", "4", "5"], "note": "No Usable Billing Lane · 2 users", "link": "#health"}


def test_health_detail_has_no_affected_users_column():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminOverview.detailHealth({F['operational']});"
        "  const tables = byTag(node, 'table');"
        "  return {h1: byTag(node, 'h1')[0].textContent, metrics: texts(byClass(node, 'detail-metric').map((m) => byTag(m, 'strong')[0])), headers: tables.map((t) => texts(byTag(t, 'th'))), rows: tables.map((t) => byTag(t, 'tbody')[0].children.map((tr) => texts(tr.children)))};"
        "})()"
    )
    assert result["h1"] == "System health"
    assert result["metrics"] == ["10", "80%", "40", "2"]
    assert result["headers"] == [["Failure category", "Count"], ["Reason", "State", "Users"]]
    assert result["rows"] == [
        [["Credential Invalid", "4"], ["Provider Timeout", "2"]],
        [["No Usable Billing Lane", "Blocked", "2"], ["Credential Invalid", "Needs attention", "4"]],
    ]
    assert "Affected users" not in OVERVIEW


def test_sources_detail_lists_six_groups_with_activation_per_group():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminOverview.detailSources({F['groups']});"
        "  const table = byTag(node, 'table')[0];"
        "  return {headers: texts(byTag(table, 'th')), rows: byTag(table, 'tbody')[0].children.map((tr) => texts(tr.children))};"
        "})()"
    )
    assert result["headers"] == ["Source", "Users", "Share", "Successful run users", "Repeat users", "Total runs", "ATL cost", "Paid users", "Activation"]
    assert result["rows"][0] == ["Internal", "22", "22%", "14", "8", "120", "$8.40", "3", "63.6%"]
    assert len(result["rows"]) == 6


def test_paint_marks_partial_availability_as_incomplete_and_never_blanks_a_sibling():
    result = _eval(
        "(async () => {"
        "  const stubs = {};"
        "  window.AdminOverview.PANELS.forEach((def) => { stubs[def.id] = panelStub(); register(def.id, stubs[def.id].panel); });"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['overview_partial_error']}}});"  # overview
        "  fetchQueue.push({ok: false, status: 503, body: {}});"  # lifecycle
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['retention']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['commercial']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['operational']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['groups']}}});"
        "  await window.AdminOverview.loadAll(['overview', 'lifecycle', 'retention', 'commercial', 'operational', 'groups']);"
        "  const read = (id) => ({headline: stubs[id].parts.headline.textContent, status: stubs[id].parts.status.textContent, error: stubs[id].parts.errorText.textContent, body: stubs[id].parts.body.children.length});"
        "  return {activeUsers: read('panelActiveUsers'), attention: read('panelAttention'), lifecycle: read('panelLifecycle'), sources: read('panelSources'), calls: fetchCalls.map(([url]) => url.split('?')[0])};"
        "})()"
    )
    assert result["calls"] == [
        "/api/admin/analytics/overview", "/api/admin/analytics/lifecycle", "/api/admin/analytics/retention",
        "/api/admin/analytics/commercial", "/api/admin/analytics/operational", "/api/admin/analytics/groups",
    ]
    # The partial overview still paints (its funnel and headline are present) and says "Incomplete data".
    assert result["activeUsers"]["headline"] == "42"
    assert result["activeUsers"]["status"] == "Incomplete data"
    # Lifecycle failed: the two panels that need it show the error; the others are untouched.
    assert result["lifecycle"] == {"headline": "—", "status": "", "error": "This section is temporarily unavailable.", "body": 0}
    assert result["attention"]["error"] == "This section is temporarily unavailable."
    assert result["sources"] == {"headline": "100", "status": "", "error": "", "body": 1}
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_overview_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models -v
```

Expected: **ERROR at collection** for the overview test (`js/admin-overview.js` missing); the fixture validation test **PASSES** already (the groups fixture is added in this step and validates against `GroupAnalyticsResponse`).

- [ ] **Step 3: Create `js/admin-overview.js`**

```js
/** /admin overview: nine panels and five detail routes from the analytics endpoints (design §7.2, §8.2). */
(function () {
  'use strict';

  const ENDPOINTS = Object.freeze({
    overview: '/api/admin/analytics/overview',
    lifecycle: '/api/admin/analytics/lifecycle',
    retention: '/api/admin/analytics/retention',
    commercial: '/api/admin/analytics/commercial',
    operational: '/api/admin/analytics/operational',
    groups: '/api/admin/analytics/groups',
  });
  const ALL = Object.keys(ENDPOINTS);
  const FUNNEL_LABELS = Object.freeze({
    account_signed_up: 'Account created',
    credential_verified: 'Credential verified',
    agent_created: 'Agent created',
    backtest_requested: 'Backtest attempted',
    backtest_completed: 'First successful result',
  });
  const GROUP_CLASSES = Object.freeze({
    internal: '', invited: 'source-green', organic: 'source-violet',
    competition: 'source-amber', partner: 'source-red', unknown: 'source-steel',
  });
  const GROUP_COLORS = Object.freeze({
    internal: 'var(--accent)', invited: 'var(--green)', organic: 'var(--violet)',
    competition: 'var(--amber)', partner: 'var(--red)', unknown: 'var(--steel)',
  });
  const SEGMENT_ORDER = ['new', 'onboarding', 'growing', 'core', 'at_risk', 'dormant'];
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const DETAIL_NEEDS = Object.freeze({
    sources: ['groups'], retention: ['retention'], credits: ['commercial'],
    lifecycle: ['lifecycle'], health: ['operational'],
  });

  const state = { data: {}, errors: {}, signature: '' };

  function shell() {
    return window.AdminShell;
  }

  function sum(rows, key) {
    return (rows || []).reduce((total, row) => total + (Number(row?.[key]) || 0), 0);
  }

  function ratio(part, whole) {
    return whole ? (Number(part) || 0) / whole : null;
  }

  function usdFromMicro(value) {
    if (value == null || value === '') return shell().DASH;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return shell().DASH;
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(numeric / 1000000);
  }

  function emptyBody(text) {
    const body = shell().el('div');
    body.appendChild(shell().el('p', 'panel-empty', text));
    return body;
  }

  // ---------------------------------------------------------------- panels

  function renderAttention(operational, lifecycle) {
    const s = shell();
    const counts = operational?.operational_state_counts || {};
    const blocked = Number(counts.blocked) || 0;
    const needs = Number(counts.needs_attention) || 0;
    const atRisk = Number(lifecycle?.segment_counts?.at_risk) || 0;
    const body = s.el('div');
    const list = s.el('div', 'attention-list');
    [
      ['severity', 'Blocked', blocked, 'A current issue prevents a core action'],
      ['severity warn', 'Needs attention', needs, 'A supported issue needs operator review'],
      ['severity muted', 'At risk', atRisk, 'Inactive, not yet dormant'],
    ].forEach(([tone, label, count, detail]) => {
      const row = s.el('div', 'attention-row');
      row.appendChild(s.el('i', tone));
      row.appendChild(s.el('span', '', label));
      row.appendChild(s.el('b', '', s.formatNumber(count)));
      row.appendChild(s.el('small', '', detail));
      list.appendChild(row);
    });
    body.appendChild(list);
    const reasons = Array.isArray(operational?.top_operational_reasons) ? operational.top_operational_reasons : [];
    const note = s.el('div', 'attention-note');
    const text = s.el('div');
    text.appendChild(s.el('span', '', 'Blocking signal'));
    text.appendChild(s.el('strong', '', reasons.length
      ? `${s.humanize(reasons[0].reason_code)} · ${s.formatNumber(reasons[0].users)} users`
      : 'No blocking signal recorded'));
    note.appendChild(text);
    const link = s.el('a', '', 'View reason details');
    link.setAttribute('href', '#health');
    note.appendChild(link);
    body.appendChild(note);
    return { headline: s.formatNumber(blocked + needs + atRisk), body };
  }

  function renderActiveUsers(overview) {
    const s = shell();
    const series = Object.entries(overview?.daily_active_users || {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    const headline = s.formatNumber(overview?.active_users_7d);
    if (!series.length) return { headline, body: emptyBody('No daily activity recorded for this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'bar-chart with-axis chart-with-axis');
    chart.setAttribute('role', 'img');
    const max = Math.max(1, ...series.map(([, value]) => Number(value) || 0));
    const axis = s.el('div', 'chart-y-axis');
    [max, Math.round(max * 2 / 3), Math.round(max / 3), 0].forEach((tick) => axis.appendChild(s.el('span', '', s.formatNumber(tick))));
    chart.appendChild(axis);
    series.forEach(([day, value]) => {
      const column = s.el('div', 'bar-column');
      column.appendChild(s.el('b', '', s.formatNumber(value)));
      const bar = s.el('i');
      bar.style.setProperty('--height', `${Math.max(4, (Number(value) || 0) * 76 / max)}%`);
      column.appendChild(bar);
      column.appendChild(s.el('span', '', s.formatShortDay(day)));
      chart.appendChild(column);
    });
    chart.setAttribute('aria-label', `Active users ${series.map(([day, value]) => `${s.formatShortDay(day)}: ${s.formatNumber(value)}`).join(', ')}`);
    body.appendChild(chart);
    return { headline, body };
  }

  function renderActivation(overview) {
    const s = shell();
    const stages = Object.entries(overview?.activation_funnel || {});
    const headline = s.formatPercent(overview?.first_success_conversion);
    if (!stages.length) return { headline, body: emptyBody('No activation events recorded for this range.') };
    const body = s.el('div');
    const route = s.el('div', 'layered-route');
    route.setAttribute('role', 'img');
    const first = Number(stages[0][1]) || 0;
    const depths = [72, 58, 46, 37, 30];
    stages.forEach(([key, value], index) => {
      const layer = s.el('div', 'layer');
      layer.style.setProperty('--depth', `${depths[Math.min(index, depths.length - 1)]}%`);
      layer.appendChild(s.el('span', 'layer-name', FUNNEL_LABELS[key] || s.humanize(key)));
      layer.appendChild(s.el('strong', 'layer-value', s.formatNumber(value)));
      const meta = s.el('small', 'layer-meta');
      if (index === 0) {
        meta.textContent = '100% of cohort';
      } else {
        meta.appendChild(s.el('strong', '', s.formatPercent(ratio(value, first))));
        meta.append(' remain');
      }
      layer.appendChild(meta);
      route.appendChild(layer);
    });
    route.setAttribute('aria-label', `Activation progress: ${stages.map(([key, value]) => `${s.formatNumber(value)} ${(FUNNEL_LABELS[key] || s.humanize(key)).toLowerCase()}`).join(', ')}`);
    body.appendChild(route);
    return { headline, body };
  }

  function renderSources(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const total = sum(rows, 'users');
    const headline = s.formatNumber(total);
    if (!rows.length) return { headline, body: emptyBody('No account sources recorded.') };
    const body = s.el('div');
    const wrap = s.el('div', 'source-wrap');
    const donut = s.el('div', 'source-donut');
    donut.setAttribute('role', 'img');
    donut.setAttribute('data-total', `${s.formatNumber(total)}\nusers`);
    let cursor = 0;
    const stops = rows.map((row) => {
      const share = total ? (Number(row.users) || 0) / total * 100 : 0;
      const start = cursor;
      cursor += share;
      return `${GROUP_COLORS[row.group] || 'var(--steel)'} ${start.toFixed(2)}% ${cursor.toFixed(2)}%`;
    });
    donut.style.background = total ? `conic-gradient(${stops.join(',')})` : 'var(--line)';
    donut.setAttribute('aria-label', `User sources: ${rows.map((row) => `${row.label || s.humanize(row.group)} ${s.formatPercent(ratio(row.users, total))}`).join(', ')}`);
    wrap.appendChild(donut);
    const legend = s.el('div', 'source-legend-six');
    rows.forEach((row) => {
      const item = s.el('div', 'legend-row');
      item.appendChild(s.el('i', GROUP_CLASSES[row.group] ?? 'source-steel'));
      item.appendChild(s.el('span', '', row.label || s.humanize(row.group)));
      item.appendChild(s.el('b', '', s.formatPercent(ratio(row.users, total))));
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
    body.appendChild(wrap);
    return { headline, body };
  }

  function renderRetention(retention) {
    const s = shell();
    const cohorts = Array.isArray(retention?.cohorts) ? retention.cohorts : [];
    const headline = s.formatPercent(retention?.summary_week_1?.rate);
    if (!cohorts.length) return { headline, body: emptyBody('No activation cohorts in this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'retention-chart');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', 'Retention by activation week');
    ['', 'W0', 'W1', 'W2', 'W4'].forEach((label) => chart.appendChild(s.el('span', '', label)));
    cohorts.forEach((cohort) => {
      chart.appendChild(s.el('span', 'row-label', s.formatShortDay(cohort.cohort_week)));
      const activated = s.el('div', 'retention-cell', s.formatNumber(cohort.activated_users));
      activated.style.setProperty('--alpha', '.55');
      chart.appendChild(activated);
      [cohort.week_1, cohort.week_2, cohort.week_4].forEach((cell) => {
        const mature = Boolean(cell?.mature) && cell.rate != null;
        const node = s.el('div', mature ? 'retention-cell' : 'retention-cell empty', mature ? s.formatPercent(cell.rate) : s.DASH);
        if (mature) node.style.setProperty('--alpha', (Number(cell.rate) * 0.55).toFixed(2));
        if (cell?.data_quality === 'partial') node.setAttribute('title', s.INCOMPLETE);
        chart.appendChild(node);
      });
    });
    body.appendChild(chart);
    return { headline, body };
  }

  function renderValue(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const users = sum(rows, 'users');
    const successful = sum(rows, 'successful_run_users');
    const repeat = sum(rows, 'repeat_users');
    const headline = s.formatNumber(successful);
    if (!rows.length) return { headline, body: emptyBody('No users recorded.') };
    const steps = [
      ['Not yet successful', Math.max(0, users - successful)],
      ['Successful once', Math.max(0, successful - repeat)],
      ['Repeat users', repeat],
    ];
    const body = s.el('div');
    const list = s.el('div', 'value-steps');
    list.setAttribute('role', 'img');
    list.setAttribute('aria-label', `Value progression: ${steps.map(([label, count]) => `${s.formatNumber(count)} ${label.toLowerCase()}`).join(', ')}`);
    steps.forEach(([label, count]) => {
      const step = s.el('div', 'value-step');
      step.appendChild(s.el('span', '', label));
      const track = s.el('div', 'value-track');
      const fill = s.el('i');
      fill.style.setProperty('--share', s.formatPercent(ratio(count, users)) === s.DASH ? '0%' : s.formatPercent(ratio(count, users)));
      track.appendChild(fill);
      step.appendChild(track);
      step.appendChild(s.el('b', '', `${s.formatNumber(count)} · ${s.formatPercent(ratio(count, users))}`));
      list.appendChild(step);
    });
    body.appendChild(list);
    return { headline, body };
  }

  function renderLifecycle(lifecycle) {
    const s = shell();
    const counts = lifecycle?.segment_counts || {};
    const total = SEGMENT_ORDER.reduce((acc, key) => acc + (Number(counts[key]) || 0), 0);
    const headline = s.formatNumber(total);
    if (!lifecycle?.segment_counts) return { headline, body: emptyBody('No lifecycle distribution recorded.') };
    const body = s.el('div');
    const bar = s.el('div', 'lifecycle-bar');
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', `Lifecycle distribution: ${SEGMENT_ORDER.map((key) => `${s.LIFECYCLE_LABELS[key]} ${s.formatNumber(counts[key] || 0)}`).join(', ')}`);
    SEGMENT_ORDER.forEach((key) => {
      const segment = s.el('i');
      segment.style.setProperty('--share', String(total ? (Number(counts[key]) || 0) : 1));
      bar.appendChild(segment);
    });
    body.appendChild(bar);
    const legend = s.el('div', 'lifecycle-legend');
    SEGMENT_ORDER.forEach((key) => {
      const item = s.el('div');
      item.appendChild(s.el('i'));
      item.appendChild(s.el('span', '', s.LIFECYCLE_LABELS[key]));
      item.appendChild(s.el('b', '', s.formatNumber(counts[key] || 0)));
      legend.appendChild(item);
    });
    body.appendChild(legend);
    return { headline, body };
  }

  function renderCredits(commercial, overview) {
    const s = shell();
    const headline = s.formatCredits(commercial?.selected_period?.consumed_micro);
    const days = Array.isArray(overview?.billing_lane_mix) ? overview.billing_lane_mix : [];
    if (!days.length) return { headline, body: emptyBody('No run activity by billing lane in this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'credits-paired');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `Platform Credits and BYOK runs by date for ${s.state.range}`);
    chart.style.gridTemplateColumns = `repeat(${days.length},minmax(0,1fr))`;
    chart.style.height = days.length > 8 ? '184px' : '164px';
    const axis = s.el('div', 'credit-axis');
    axis.appendChild(s.el('span', '', 'Higher usage'));
    axis.appendChild(s.el('span', '', `Selected range · ${s.state.range}`));
    chart.appendChild(axis);
    const max = Math.max(1, ...days.flatMap((day) => [Number(day.platform_credits) || 0, Number(day.byok) || 0]));
    days.forEach((day) => {
      const column = s.el('div', 'credit-day');
      [[day.platform_credits, 'credit-stem', 12], [day.byok, 'credit-stem violet', 10]].forEach(([value, className, floor]) => {
        const stem = s.el('span', className);
        stem.style.height = `${Math.max(floor, (Number(value) || 0) / max * 124)}px`;
        stem.appendChild(s.el('b', '', s.formatNumber(value)));
        column.appendChild(stem);
      });
      column.appendChild(s.el('span', 'credit-date', s.formatShortDay(day.day)));
      chart.appendChild(column);
    });
    const legend = s.el('div', 'credit-legend');
    [['Platform Credits', ''], ['BYOK runs', 'violet']].forEach(([label, className]) => {
      const item = s.el('span');
      item.appendChild(s.el('i', className));
      item.append(label);
      legend.appendChild(item);
    });
    chart.appendChild(legend);
    body.appendChild(chart);
    return { headline, body };
  }

  function svgNode(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text != null) node.textContent = String(text);
    return node;
  }

  function axisLabel(value) {
    return value >= 100 ? String(Math.round(value)) : value.toFixed(1).replace(/\.0$/, '');
  }

  function renderRevenue(commercial) {
    const s = shell();
    const headline = s.formatCredits(commercial?.selected_period?.purchased_micro);
    const series = Array.isArray(commercial?.purchased_by_day) ? commercial.purchased_by_day : [];
    if (!series.length) return { headline, body: emptyBody('No settled purchases in this range.') };
    const values = series.map((row) => (Number(row.amount_micro) || 0) / 1000000);
    const labels = series.map((row) => s.formatShortDay(row.day));
    const width = 520, height = 180, left = 42, right = 508, top = 19, bottom = 141;
    const plotWidth = right - left, plotHeight = bottom - top;
    const max = Math.max(1, ...values);
    const points = values.map((value, index) => ({
      x: left + (values.length === 1 ? 0 : index / (values.length - 1) * plotWidth),
      y: bottom - (value / max) * plotHeight,
      value,
      label: labels[index],
    }));
    const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `Purchased Credits revenue trend with ${values.length} data points` });
    [0, 0.5, 1].forEach((fraction) => svg.appendChild(svgNode('line', { class: 'revenue-grid', x1: left, y1: bottom - fraction * plotHeight, x2: right, y2: bottom - fraction * plotHeight })));
    svg.appendChild(svgNode('line', { class: 'revenue-axis', x1: left, y1: top, x2: left, y2: bottom }));
    svg.appendChild(svgNode('line', { class: 'revenue-axis', x1: left, y1: bottom, x2: right, y2: bottom }));
    [max, max / 2, 0].forEach((value, index) => svg.appendChild(svgNode('text', { class: 'revenue-axis-label', x: 5, y: [top + 3, (top + bottom) / 2 + 3, bottom + 3][index] }, axisLabel(value))));
    const path = points.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ');
    svg.appendChild(svgNode('path', { class: 'revenue-area', d: `${path} L ${right} ${bottom} L ${left} ${bottom} Z` }));
    svg.appendChild(svgNode('path', { class: 'revenue-line', d: path }));
    points.forEach((point) => {
      const circle = svgNode('circle', { class: 'revenue-point', cx: point.x, cy: point.y, r: 4, tabindex: '0' });
      circle.appendChild(svgNode('title', {}, `${point.label}: ${point.value.toFixed(1)} purchased Credits`));
      svg.appendChild(circle);
      svg.appendChild(svgNode('text', { class: 'revenue-point-label', x: point.x, y: Math.max(top + 11, point.y - 9) }, axisLabel(point.value)));
    });
    const shown = [0, Math.floor((labels.length - 1) / 2), labels.length - 1].filter((index, position, list) => list.indexOf(index) === position);
    shown.forEach((index) => svg.appendChild(svgNode('text', { class: 'revenue-x-label', x: points[index].x, y: 164, 'text-anchor': index === 0 ? 'start' : index === labels.length - 1 ? 'end' : 'middle' }, labels[index])));
    const body = s.el('div');
    const wrap = s.el('div', 'revenue-viz');
    wrap.appendChild(svg);
    body.appendChild(wrap);
    return { headline, body };
  }

  // --------------------------------------------------------- detail views

  function table(headers, rows, emptyText) {
    const s = shell();
    const wrap = s.el('div', 'table-wrap');
    const node = s.el('table');
    const head = s.el('thead');
    const headRow = s.el('tr');
    headers.forEach((label) => {
      const cell = s.el('th', '', label);
      cell.setAttribute('scope', 'col');
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    node.appendChild(head);
    const body = s.el('tbody');
    rows.forEach((cells) => {
      const row = s.el('tr');
      cells.forEach((cell) => row.appendChild(typeof cell === 'string' ? s.el('td', '', cell) : (() => { const td = s.el('td'); td.appendChild(cell); return td; })()));
      body.appendChild(row);
    });
    if (!rows.length) {
      const row = s.el('tr');
      const cell = s.el('td', 'panel-empty', emptyText);
      cell.setAttribute('colspan', String(headers.length));
      row.appendChild(cell);
      body.appendChild(row);
    }
    node.appendChild(body);
    wrap.appendChild(node);
    return wrap;
  }

  function detailShell(title, description, metrics) {
    const s = shell();
    const root = s.el('div');
    const crumb = s.el('nav', 'breadcrumb');
    crumb.setAttribute('aria-label', 'Breadcrumb');
    const parent = s.el('a', '', 'Analytics');
    parent.setAttribute('href', '#overview');
    crumb.appendChild(parent);
    crumb.appendChild(s.el('span', '', '/'));
    crumb.appendChild(s.el('span', '', title));
    root.appendChild(crumb);
    const head = s.el('div', 'page-head');
    const text = s.el('div');
    const heading = s.el('h1', '', title);
    heading.setAttribute('tabindex', '-1');
    text.appendChild(heading);
    text.appendChild(s.el('p', 'muted', description));
    head.appendChild(text);
    head.appendChild(s.el('small', 'muted', `Selected range · ${s.state.range} · UTC`));
    root.appendChild(head);
    const strip = s.el('div', 'metric-strip');
    metrics.forEach(([label, value]) => {
      const metric = s.el('div', 'detail-metric');
      metric.appendChild(s.el('span', '', label));
      metric.appendChild(s.el('strong', '', value));
      strip.appendChild(metric);
    });
    root.appendChild(strip);
    return root;
  }

  function section(root, title, meta, content) {
    const s = shell();
    const node = s.el('section', 'detail-section');
    const head = s.el('div', 'section-head');
    const text = s.el('div');
    text.appendChild(s.el('h2', '', title));
    if (meta) text.appendChild(s.el('small', '', meta));
    head.appendChild(text);
    node.appendChild(head);
    node.appendChild(content);
    root.appendChild(node);
    return node;
  }

  function detailSources(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const total = sum(rows, 'users');
    const successful = sum(rows, 'successful_run_users');
    const largest = rows.slice().sort((a, b) => (Number(b.users) || 0) - (Number(a.users) || 0))[0];
    const root = detailShell('User sources', 'Compare the six account-source groups and see where users progress toward a first successful result.', [
      ['Identified users', s.formatNumber(total)],
      ['Largest source', largest ? `${largest.label || s.humanize(largest.group)} · ${s.formatPercent(ratio(largest.users, total))}` : s.DASH],
      ['Users with a result', s.formatNumber(successful)],
      ['Activation', s.formatPercent(ratio(successful, total))],
    ]);
    section(root, 'Source groups', 'Users, results and cost by account source', table(
      ['Source', 'Users', 'Share', 'Successful run users', 'Repeat users', 'Total runs', 'ATL cost', 'Paid users', 'Activation'],
      rows.map((row) => [
        row.label || s.humanize(row.group), s.formatNumber(row.users), s.formatPercent(ratio(row.users, total)),
        s.formatNumber(row.successful_run_users), s.formatNumber(row.repeat_users), s.formatNumber(row.total_runs),
        usdFromMicro(row.atl_cost_micro_usd), s.formatNumber(row.paid_users), s.formatPercent(ratio(row.successful_run_users, row.users)),
      ]),
      'No account sources recorded.'
    ));
    return root;
  }

  function retentionCell(cell) {
    const s = shell();
    if (!cell?.mature || cell.rate == null) return 'Not mature';
    return `${s.formatNumber(cell.retained_users)} / ${s.formatNumber(cell.eligible_users)} · ${s.formatPercent(cell.rate)}`;
  }

  function detailRetention(retention) {
    const s = shell();
    const summary = (cell) => (cell?.mature ? s.formatPercent(cell.rate) : 'Not mature');
    const root = detailShell('Retention', 'See whether users return after reaching their first successful result.', [
      ['W1 return', summary(retention?.summary_week_1)],
      ['W2 return', summary(retention?.summary_week_2)],
      ['W4 return', summary(retention?.summary_week_4)],
      ['Eligible users', s.formatNumber(retention?.summary_week_1?.eligible_users)],
    ]);
    section(root, 'Activation cohorts', 'UTC Monday-to-Sunday week of the first successful backtest', table(
      ['Activation week', 'Activated', 'W1', 'W2', 'W4'],
      (retention?.cohorts || []).map((cohort) => [
        s.formatDateOnly(cohort.cohort_week), s.formatNumber(cohort.activated_users),
        retentionCell(cohort.week_1), retentionCell(cohort.week_2), retentionCell(cohort.week_4),
      ]),
      'No activation cohorts in this range.'
    ));
    return root;
  }

  function detailCredits(commercial) {
    const s = shell();
    const period = commercial?.selected_period || {};
    const balances = commercial?.current_balances || {};
    const tiers = commercial?.tier_counts || {};
    const root = detailShell('Credits & revenue', 'Review platform Credits usage and settled purchases together.', [
      ['ATL Credits settled', s.formatCredits(period.consumed_micro)],
      ['Purchased Credits', s.formatCredits(period.purchased_micro)],
      ['Refunds', s.formatCredits(period.refunded_micro)],
      ['Admin Grants', s.formatCredits(period.admin_grant_activity_micro)],
    ]);
    section(root, 'Selected period', 'Ledger movement in the selected range', table(
      ['Measure', 'Value', 'Scope'],
      [
        ['ATL Credits settled', s.formatCredits(period.consumed_micro), 'Selected period'],
        ['Purchased Credits', s.formatCredits(period.purchased_micro), 'Selected period'],
        ['Refunds', s.formatCredits(period.refunded_micro), 'Selected period'],
        ['Admin Grants', s.formatCredits(period.admin_grant_activity_micro), 'Excluded from revenue'],
        ['Platform model cost', usdFromMicro(period.platform_model_cost_micro_usd), 'Platform Credits lane'],
        ['Lifetime net purchased', s.formatCredits(commercial?.lifetime_net_purchased_micro), 'Lifetime'],
      ],
      'No ledger activity.'
    ));
    section(root, 'Commercial tiers', 'Lifetime net purchase per user', table(
      ['Tier', 'Users'],
      Object.keys(s.COMMERCIAL_LABELS).map((tier) => [s.COMMERCIAL_LABELS[tier], s.formatNumber(tiers[tier] || 0)]),
      'No users recorded.'
    ));
    section(root, 'Current balances', 'Spendable Credits right now', table(
      ['Balance', 'Value', 'Note'],
      [
        ['Grant balance', s.formatCredits(balances.grant_available_micro), 'Not revenue'],
        ['Purchased balance', s.formatCredits(balances.purchased_available_micro), 'Customer-funded'],
        ['Total available', s.formatCredits(balances.total_available_micro), 'Current spendable balance'],
      ],
      'No balances recorded.'
    ));
    return root;
  }

  function detailLifecycle(lifecycle) {
    const s = shell();
    const counts = lifecycle?.segment_counts || {};
    const root = detailShell('User lifecycle', 'See current user maturity and inactivity without mixing it with operational blockers.', [
      ['Growing', s.formatNumber(counts.growing || 0)],
      ['Core', s.formatNumber(counts.core || 0)],
      ['At risk', s.formatNumber(counts.at_risk || 0)],
      ['Dormant', s.formatNumber(counts.dormant || 0)],
    ]);
    section(root, 'Segments', 'Rules from the lifecycle definition', table(
      ['Stage', 'Users', 'Rule'],
      SEGMENT_ORDER.map((key) => [s.LIFECYCLE_LABELS[key], s.formatNumber(counts[key] || 0), s.LIFECYCLE_RULES[key]]),
      'No segments recorded.'
    ));
    section(root, 'Recent movement', 'Segment transitions in the selected range', table(
      ['From', 'To', 'Users', 'Period'],
      (lifecycle?.transitions || []).map((transition) => [
        s.LIFECYCLE_LABELS[transition.from_segment] || s.humanize(transition.from_segment),
        s.LIFECYCLE_LABELS[transition.to_segment] || s.humanize(transition.to_segment),
        s.formatNumber(transition.users),
        `${s.formatDateOnly(transition.period_start)} – ${s.formatDateOnly(transition.period_end)}${transition.data_quality === 'partial' ? ` · ${s.INCOMPLETE}` : ''}`,
      ]),
      'No lifecycle transitions in this range.'
    ));
    return root;
  }

  function detailHealth(operational) {
    const s = shell();
    const counts = operational?.operational_state_counts || {};
    const root = detailShell('System health', 'Track reliability and the failure reasons that prevent users from receiving results.', [
      ['Failed runs', s.formatNumber(operational?.failed_runs)],
      ['Success rate', s.formatPercent(operational?.backtest_success_rate)],
      ['Completed runs', s.formatNumber(operational?.completed_runs)],
      ['Blocked users', s.formatNumber(counts.blocked || 0)],
    ]);
    // Rollups are anonymous (design D13, §8.2): the failure table carries a count
    // per category and no "Affected users" column.
    section(root, 'Failure categories', 'Display-safe categories from the daily rollups', table(
      ['Failure category', 'Count'],
      (operational?.top_failure_categories || []).map((failure) => [s.humanize(failure.error_category), s.formatNumber(failure.affected_users)]),
      'No failure categories in this range.'
    ));
    section(root, 'What is blocking users?', 'Operational reasons as of yesterday UTC', table(
      ['Reason', 'State', 'Users'],
      (operational?.top_operational_reasons || []).map((reason) => [s.humanize(reason.reason_code), s.OPERATIONAL_LABELS[reason.state] || s.humanize(reason.state), s.formatNumber(reason.users)]),
      'No blocking reasons recorded.'
    ));
    return root;
  }

  // ------------------------------------------------------------ loading

  const PANELS = [
    { id: 'panelAttention', name: 'attention', needs: ['operational', 'lifecycle'], render: (d) => renderAttention(d.operational, d.lifecycle) },
    { id: 'panelActiveUsers', name: 'active-users', needs: ['overview'], render: (d) => renderActiveUsers(d.overview) },
    { id: 'panelActivation', name: 'activation', needs: ['overview'], render: (d) => renderActivation(d.overview) },
    { id: 'panelSources', name: 'sources', needs: ['groups'], render: (d) => renderSources(d.groups) },
    { id: 'panelRetention', name: 'retention', needs: ['retention'], render: (d) => renderRetention(d.retention) },
    { id: 'panelValue', name: 'value', needs: ['groups'], render: (d) => renderValue(d.groups) },
    { id: 'panelLifecycle', name: 'lifecycle', needs: ['lifecycle'], render: (d) => renderLifecycle(d.lifecycle) },
    { id: 'panelCredits', name: 'credits', needs: ['commercial', 'overview'], render: (d) => renderCredits(d.commercial, d.overview) },
    { id: 'panelRevenue', name: 'revenue', needs: ['commercial'], render: (d) => renderRevenue(d.commercial) },
  ];

  function pathFor(name) {
    const s = shell();
    return `${ENDPOINTS[name]}?${s.analyticsParams({ withGroup: name === 'groups' })}`;
  }

  function paint(def) {
    const s = shell();
    const panel = document.getElementById(def.id);
    if (!panel) return;
    const missing = def.needs.filter((name) => !state.data[name]);
    const failed = def.needs.some((name) => state.errors[name]);
    if (missing.length) {
      s.setPanelState(panel, { busy: false, error: s.SECTION_UNAVAILABLE });
      return;
    }
    const result = def.render(state.data);
    const headline = panel.querySelector('[data-headline]');
    if (headline) headline.textContent = result.headline;
    const body = panel.querySelector('[data-body]');
    if (body) {
      s.clear(body);
      body.appendChild(result.body);
    }
    const incomplete = def.needs.some((name) => s.availabilityIncomplete(state.data[name]?.availability));
    s.setPanelState(panel, { busy: false, status: incomplete ? s.INCOMPLETE : '', error: failed ? s.SECTION_UNAVAILABLE : '', stale: failed });
  }

  async function loadAll(names) {
    const s = shell();
    const seq = s.nextSeq('overview');
    PANELS.forEach((def) => s.setPanelState(document.getElementById(def.id), { busy: true }));
    const results = await Promise.allSettled(names.map((name) => s.request(pathFor(name))));
    if (!s.isCurrent('overview', seq)) return;
    for (const [index, name] of names.entries()) {
      const result = results[index];
      if (result.status === 'fulfilled') {
        state.data[name] = result.value;
        state.errors[name] = null;
      } else {
        if (await s.handleAccessLost(result.reason)) return;
        state.errors[name] = s.SECTION_UNAVAILABLE;
      }
    }
    PANELS.forEach((def) => paint(def));
  }

  const DETAILS = Object.freeze({
    sources: (d) => detailSources(d.groups),
    retention: (d) => detailRetention(d.retention),
    credits: (d) => detailCredits(d.commercial),
    lifecycle: (d) => detailLifecycle(d.lifecycle),
    health: (d) => detailHealth(d.operational),
  });

  async function showDetail(route) {
    const s = shell();
    const target = document.getElementById('detail');
    if (!target || !DETAILS[route]) return;
    const seq = s.nextSeq('detail');
    const needs = DETAIL_NEEDS[route];
    const missing = needs.filter((name) => !state.data[name]);
    if (missing.length) {
      const results = await Promise.allSettled(missing.map((name) => s.request(pathFor(name))));
      if (!s.isCurrent('detail', seq)) return;
      for (const [index, name] of missing.entries()) {
        const result = results[index];
        if (result.status === 'fulfilled') { state.data[name] = result.value; state.errors[name] = null; }
        else {
          if (await s.handleAccessLost(result.reason)) return;
          state.errors[name] = s.SECTION_UNAVAILABLE;
        }
      }
    }
    s.clear(target);
    if (needs.some((name) => !state.data[name])) {
      target.appendChild(s.el('p', 'panel-error', s.SECTION_UNAVAILABLE));
      return;
    }
    target.appendChild(DETAILS[route](state.data));
    if (needs.some((name) => s.availabilityIncomplete(state.data[name]?.availability))) {
      target.appendChild(s.el('p', 'panel-status muted', s.INCOMPLETE));
    }
    target.querySelector('h1')?.focus?.({ preventScroll: true });
  }

  function signatureOf(detail) {
    return JSON.stringify([detail.range, detail.filters.group, detail.filters.internal]);
  }

  document.addEventListener('admin:route', (event) => {
    const detail = event.detail || {};
    const signature = signatureOf(detail);
    if (signature !== state.signature) {
      state.signature = signature;
      state.data = {};
      state.errors = {};
    }
    if (detail.route === 'overview') loadAll(ALL);
    else if (DETAILS[detail.route]) showDetail(detail.route);
  });
  document.addEventListener('admin:retry', (event) => {
    const def = PANELS.find((panel) => panel.name === event.detail?.panel);
    if (def) loadAll(def.needs);
  });

  window.AdminOverview = {
    PANELS, DETAIL_NEEDS, state,
    renderAttention, renderActiveUsers, renderActivation, renderSources, renderRetention,
    renderValue, renderLifecycle, renderCredits, renderRevenue,
    detailSources, detailRetention, detailCredits, detailLifecycle, detailHealth,
    paint, loadAll, showDetail,
  };
})();
```

Lifecycle stage and Tier deliberately do not reach these six requests: no analytics route takes `lifecycle_segment` or `commercial_tier` (§8.2 sources them on `/users`), and sending them would 422. Source reaches only `/groups` (`user_group`), which is why the filter-bar note in `admin.html` says so.

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_overview_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_validate_against_committed_analytics_models -v
```

Expected: **PASS** on all twelve overview tests and the fixture validation. `test_paint_marks_partial_availability_as_incomplete_and_never_blanks_a_sibling` shows the two properties §15.6 asks for: `overview_partial_error.json` (growth panel unavailable) still paints the active-users headline `42` with the status "Incomplete data", and the 503 on `/lifecycle` errors exactly the two panels that need it while `panelSources` renders from `/groups`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/admin-overview.js dashboard/backend/tests/fixtures/admin_analytics/groups.json dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_overview_frontend.py
git commit -m "$(cat <<'EOF'
feat: render the /admin overview panels and detail routes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `js/admin-users.js` — the users list, the lazy cursor-paged profile, the evidence dialog

**Files:**
- Create: `dashboard/frontend/js/admin-users.js`.
- Create: `dashboard/backend/tests/test_admin_users_frontend.py`.
- Test: `dashboard/backend/tests/test_admin_users_frontend.py`.

**Interfaces:**
- Consumes: `window.AdminShell`; `GET /api/admin/analytics/users?{q,user_group,lifecycle_segment,commercial_tier,priority,include_internal,limit,offset}` (Task 1 — items carry `user_group`, `role`, `group_badge`, `last_meaningful_activity_at`); `GET /api/admin/analytics/users/{id}?from&to`; `GET /api/admin/analytics/users/{id}/activity?section&limit=50[&cursor]` (`AnalyticsActivityPage`).
- Produces `window.AdminUsers` with pure renderers `groupBadge(badge) -> element`, `signalBadge(kind, value) -> element`, `renderUserRows(payload) -> fragment of <tr>`, `renderPager(payload) -> string`, `renderEvidence(user) -> element`, `activationWeekLabel(activatedAt) -> string`, `renderProfileHeader(profile) -> element`, `renderProfileOverview(profile) -> element`, `renderActivityItems(section, items) -> element`, `accountManagementHref(profile) -> string`; plus `loadList({ offset })`, `openProfile(id)`, `selectSection(section)`, `loadSection(section, { append })`, `state`.

Rules this module carries: the group badge is rendered **verbatim** from `group_badge` (D11) — no precedence logic here, and Task 10 pins that `role === 'admin'` never appears in this file; the D15 fields (`country_code`, `device_category`, `browser_family`, `top_product_page`) stay in the payload and are never read here; the sessions table has three columns (Started · Events · Visible time), not the old six; "Open account management" deep-links to `/app?view=admin&adminTab=users&adminUserQuery=<email>` — the pre-fill `AdminTabs.openAccountManagement` used to perform in-process (Task 9 teaches `admin-tabs.js` to read that parameter). The profile opens at `#users/{id}`, so the browser's Back closes it (the harvested `pushProfileUrl` property, §7.4). Timeline/Runs/Usage/Sessions fetch once on first open and page with `next_cursor` (harvested `selectProfileSection`/`loadProfileSection`, `admin-analytics.js:900-1072`); the Overview tab needs no fetch because the profile payload carries it.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_admin_users_frontend.py`:

```python
"""js/admin-users.js under node: list rows, profile, activity tabs, evidence — against the fixtures."""

import json

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
USERS_JS = source("admin-users.js")
USERS = fixture("users.json")
PROFILE = fixture("user_detail.json")
SESSIONS = fixture("activity_sessions.json")
USAGE = fixture("activity_usage.json")
RUNS = fixture("activity_runs.json")
TIMELINE = fixture("activity_timeline.json")


def _eval(expression: str, *setup: str) -> object:
    return run_node(SHELL, CREDIT_FORMAT, USERS_JS, *setup, f"console.log(JSON.stringify({expression}));")


def test_user_rows_show_group_lifecycle_operational_and_last_active():
    result = _eval(
        "(() => {"
        f"  const rows = window.AdminUsers.renderUserRows({USERS}).children;"
        "  return rows.map((tr) => tr.children.map((td) => td.textContent));"
        "})()"
    )
    assert result == [
        ["Synthetic Adaada.synthetic@example.test", "invited", "At risk", "Healthy", "Aug 22, 2026, 11:20 UTC", "Review evidence"],
        ["<Synthetic & Grace>grace.synthetic@example.test", "free", "Onboarding", "Blocked", "Sep 1, 2026, 08:00 UTC", "Review evidence"],
    ]


def test_group_badge_is_rendered_verbatim_from_the_server():
    """D11: the server owns the precedence rule; the client prints the string it is given."""
    result = _eval(
        "(() => {"
        "  const badge = window.AdminUsers.groupBadge('zzz-not-a-real-badge');"
        "  const admin = window.AdminUsers.groupBadge('admin');"
        "  return [badge.textContent, badge.className, admin.textContent, admin.className];"
        "})()"
    )
    assert result == ["zzz-not-a-real-badge", "group-badge is-zzz-not-a-real-badge", "admin", "group-badge is-admin"]
    assert "role === 'admin'" not in USERS_JS
    assert "commercial_tier === 'unpaid'" not in USERS_JS


def test_user_rows_link_to_the_profile_hash_and_escape_nothing_by_hand():
    result = _eval(
        "(() => {"
        f"  const rows = window.AdminUsers.renderUserRows({USERS}).children;"
        "  return rows.map((tr) => byTag(tr, 'a')[0].getAttribute('href'));"
        "})()"
    )
    assert result == ["#users/101", "#users/102"]
    assert "innerHTML" not in USERS_JS


def test_empty_list_and_pager_copy():
    assert _eval("window.AdminUsers.renderUserRows({items: []}).children.map((tr) => tr.textContent)") == ["No users match these filters."]
    assert _eval("window.AdminUsers.renderPager({total: 120, offset: 50, items: new Array(50).fill({})})") == "Showing 51–100 of 120"
    assert _eval("window.AdminUsers.renderPager({total: 0, offset: 0, items: []})") == "0 users"


def test_evidence_dialog_content_is_display_safe_reasons_and_evidence():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderEvidence({USERS}.items[1]);"
        "  return {badges: texts(byClass(node, 'badge')), reasons: texts(byTag(node, 'p')), evidence: texts(byTag(node, 'li'))};"
        "})()"
    )
    assert result == {
        "badges": ["Onboarding", "Blocked", "Unpaid"],
        "reasons": ["grace.synthetic@example.test", "The account has not completed a successful backtest.", "The Credits account is restricted from model spending."],
        "evidence": ["no successful backtest recorded", "Credits account restriction is unresolved."],
    }


def test_profile_header_carries_badges_activation_week_and_milestones():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderProfileHeader({PROFILE});"
        "  return {h1: byTag(node, 'h1')[0].textContent, identity: byClass(node, 'identity')[0].children[1].children[1].textContent, badges: texts(byClass(node, 'badge')), group: texts(byClass(node, 'group-badge')), milestones: texts(byClass(node, 'milestone')), tabs: texts(byTag(node, 'button').filter((b) => b.getAttribute('role') === 'tab')), account: byTag(node, 'a').map((a) => a.getAttribute('href'))};"
        "})()"
    )
    assert result["h1"] == "Synthetic Ada"
    assert result["identity"] == "ada.synthetic@example.test · invited · Activation week of Jun 29, 2026"
    assert result["badges"] == ["At risk", "Healthy", "Invested"]
    assert result["group"] == ["invited"]
    assert result["milestones"] == [
        "Account signed upJul 1, 2026, 09:00 UTC", "Credential verifiedJul 1, 2026, 10:00 UTC",
        "Agent createdJul 1, 2026, 11:00 UTC", "Backtest completedJul 2, 2026, 10:00 UTC",
    ]
    assert result["tabs"] == ["Overview", "Timeline", "Runs", "Usage", "Sessions"]
    assert "/app?view=admin&adminTab=users&adminUserQuery=ada.synthetic%40example.test" in result["account"]


def test_activation_week_is_the_utc_monday():
    assert _eval("window.AdminUsers.activationWeekLabel('2026-07-02T10:00:00Z')") == "Activation week of Jun 29, 2026"
    assert _eval("window.AdminUsers.activationWeekLabel('2026-06-29T00:00:00Z')") == "Activation week of Jun 29, 2026"
    assert _eval("window.AdminUsers.activationWeekLabel(null)") == "Not yet activated"


def test_profile_overview_renders_value_facts_and_drops_the_d15_fields():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderProfileOverview({PROFILE});"
        "  const dts = texts(byTag(node, 'dt')); const dds = texts(byTag(node, 'dd'));"
        "  return {facts: dts.map((label, index) => [label, dds[index]]), all: flatten(node).map((n) => n.nodeType === 3 ? n.textContent : (n._text || ''))};"
        "})()"
    )
    facts = dict(result["facts"])
    assert facts["Activated"] == "Jul 2, 2026, 10:00 UTC"
    assert facts["Active days (30d)"] == "2"
    assert facts["Successful backtests (30d)"] == "2"
    assert facts["Inactive UTC days"] == "12"
    assert facts["Lifetime net purchased"] == "7.000000 Credits"
    assert facts["Consumed in period"] == "0.750000 Credits"
    assert facts["Available balance"] == "2.000000 Credits"
    assert facts["Completed"] == "8"
    assert facts["ATL Credits debited"] == "4.250000 Credits"
    assert "Top product page" not in facts
    for leaked in ("US", "desktop", "Chrome"):
        assert leaked not in result["all"], leaked
    for field in ("country_code", "device_category", "browser_family", "top_product_page"):
        assert field not in USERS_JS, field


def test_sessions_table_has_three_columns_and_no_region_device_browser():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('sessions', {SESSIONS}.items);"
        "  return {headers: texts(byTag(node, 'th')), rows: byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children))};"
        "})()"
    )
    assert result == {
        "headers": ["Started", "Events", "Visible time"],
        "rows": [["Aug 26, 2026, 10:30 UTC", "9", "35m 0s"], ["Aug 25, 2026, 08:00 UTC", "4", "8m 0s"]],
    }


def test_usage_table_never_prints_an_atl_charge_for_byok():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('usage', {USAGE}.items);"
        "  return byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children));"
        "})()"
    )
    assert result[0] == ["Aug 26, 2026, 11:02 UTC", "Model usage recorded", "openrouter · openai/gpt-5.5", "Platform Credits", "1,400", "350", "$0.43", "—"]
    assert result[1] == ["Aug 25, 2026, 09:18 UTC", "Model usage recorded", "openrouter · anthropic/claude-sonnet-4", "BYOK — no ATL charge", "900", "210", "—", "—"]
    assert result[2] == ["Aug 26, 2026, 11:03 UTC", "ATL Credits debited", "—", "Platform Credits", "—", "—", "—", "0.430000 Credits"]


def test_runs_and_timeline_renderers_label_events():
    runs = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('runs', {RUNS}.items);"
        "  return byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children));"
        "})()"
    )
    assert runs == [["Aug 26, 2026, 11:02 UTC", "Backtest completed", "Succeeded", "openrouter · openai/gpt-5.5", "Platform Credits", "—"]]
    timeline = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('timeline', {TIMELINE}.items);"
        "  return byTag(node, 'li').map((li) => [byTag(li, 'strong')[0].textContent, byTag(li, 'p')[0].textContent]);"
        "})()"
    )
    assert timeline == [["Backtest failed", "Failed · Openrouter · Openai/Gpt-5.5 · Platform Credits · Provider Timeout"]]
    assert _eval("window.AdminUsers.renderActivityItems('timeline', []).textContent") == "No activity in this section."


def test_list_load_uses_the_declared_query_names_and_paints_rows():
    result = _eval(
        "(async () => {"
        "  const body = register('usersBody', document.createElement('tbody'));"
        "  register('usersRange', document.createElement('span'));"
        "  register('usersPrev', document.createElement('button'));"
        "  register('usersNext', document.createElement('button'));"
        "  register('usersView', panelStub().panel);"
        "  window.AdminShell.state.filters = {group: 'invited', segment: '', tier: '', internal: false, q: 'ada', priority: false};"
        f"  fetchQueue.push({{ok: true, status: 200, body: {USERS}}});"
        "  await window.AdminUsers.loadList({offset: 0});"
        "  return {url: fetchCalls[0][0], rows: body.children.length, range: document.getElementById('usersRange').textContent, prev: document.getElementById('usersPrev').disabled, next: document.getElementById('usersNext').disabled};"
        "})()"
    )
    assert result == {
        "url": "/api/admin/analytics/users?q=ada&user_group=invited&include_internal=false&limit=50&offset=0",
        "rows": 2,
        "range": "Showing 1–2 of 2",
        "prev": True,
        "next": True,
    }


def test_profile_sections_fetch_once_and_page_with_the_cursor():
    result = _eval(
        "(async () => {"
        "  const profile = register('profile', document.createElement('section'));"
        f"  fetchQueue.push({{ok: true, status: 200, body: {PROFILE}}});"
        "  await window.AdminUsers.openProfile('101');"
        "  const afterOpen = fetchCalls.map(([url]) => url);"
        f"  fetchQueue.push({{ok: true, status: 200, body: {TIMELINE}}});"
        "  await window.AdminUsers.selectSection('timeline');"
        "  await window.AdminUsers.selectSection('timeline');"
        "  const afterTimeline = fetchCalls.map(([url]) => url);"
        f"  fetchQueue.push({{ok: true, status: 200, body: {{items: [], next_cursor: null}}}});"
        "  await window.AdminUsers.loadSection('timeline', {append: true});"
        "  return {afterOpen, afterTimeline, afterMore: fetchCalls.map(([url]) => url), items: window.AdminUsers.state.profile.sections.timeline.items.length, cursor: window.AdminUsers.state.profile.sections.timeline.nextCursor};"
        "})()"
    )
    assert len(result["afterOpen"]) == 1
    assert result["afterOpen"][0].startswith("/api/admin/analytics/users/101?from=")
    assert "&to=" in result["afterOpen"][0]
    assert result["afterTimeline"][1] == "/api/admin/analytics/users/101/activity?section=timeline&limit=50"
    assert len(result["afterTimeline"]) == 2
    assert result["afterMore"][2] == "/api/admin/analytics/users/101/activity?section=timeline&limit=50&cursor=synthetic-timeline-cursor"
    assert result["items"] == 1
    assert result["cursor"] is None
```

- [ ] **Step 2: Run it and confirm the expected failure**

```bash
python -m pytest dashboard/backend/tests/test_admin_users_frontend.py -v
```

Expected: **ERROR at collection** — `FileNotFoundError` for `js/admin-users.js`.

- [ ] **Step 3: Create `js/admin-users.js`**

```js
/** /admin users: the SQL-filtered list and the lazy, cursor-paged profile (design §7.2, D5, D11, D15). */
(function () {
  'use strict';

  const USERS_PATH = '/api/admin/analytics/users';
  const PAGE_SIZE = 50;
  const SECTIONS = ['overview', 'timeline', 'runs', 'usage', 'sessions'];
  const SECTION_LABELS = Object.freeze({ overview: 'Overview', timeline: 'Timeline', runs: 'Runs', usage: 'Usage', sessions: 'Sessions' });
  // Harvested from admin-analytics.js:35-50.
  const EVENT_LABELS = Object.freeze({
    account_signed_up: 'Account signed up',
    credential_verified: 'Credential verified',
    agent_created: 'Agent created',
    backtest_requested: 'Backtest requested',
    backtest_started: 'Backtest started',
    backtest_completed: 'Backtest completed',
    backtest_failed: 'Backtest failed',
    backtest_cancelled: 'Backtest cancelled',
    model_usage_recorded: 'Model usage recorded',
    credits_reserved: 'ATL Credits reserved',
    credits_settled: 'ATL Credits debited',
    credits_refunded: 'ATL Credits refunded',
    page_viewed: 'Product page viewed',
    session: 'Product session',
  });
  const BADGE_TONE = Object.freeze({
    lifecycle: { at_risk: 'warn', dormant: 'bad', core: 'good' },
    operational: { blocked: 'bad', needs_attention: 'warn', healthy: 'good' },
    commercial: { high_value: 'good', invested: 'good' },
  });

  function emptySection() {
    return { items: [], nextCursor: null, loading: false, loaded: false, error: null, requestSeq: 0 };
  }

  const state = {
    list: { offset: 0, total: 0, items: [], loaded: false },
    profile: { userId: null, detail: null, section: 'overview', sections: {} },
    evidenceUser: null,
  };

  function shell() {
    return window.AdminShell;
  }

  function eventLabel(value) {
    return EVENT_LABELS[value] || shell().humanize(value);
  }

  function labelFor(kind, value) {
    const s = shell();
    const labels = kind === 'lifecycle' ? s.LIFECYCLE_LABELS : kind === 'operational' ? s.OPERATIONAL_LABELS : s.COMMERCIAL_LABELS;
    return labels[value] || s.humanize(value);
  }

  // D11: the badge string is the server's `group_badge`; nothing here recomputes it.
  function groupBadge(badge) {
    const text = String(badge || 'unknown');
    return shell().el('span', `group-badge is-${text}`, text);
  }

  function signalBadge(kind, value) {
    const s = shell();
    const tone = BADGE_TONE[kind]?.[value];
    const node = s.el('span', tone ? `badge ${tone}` : 'badge', labelFor(kind, value));
    const rules = kind === 'lifecycle' ? s.LIFECYCLE_RULES : kind === 'operational' ? s.OPERATIONAL_RULES : null;
    if (rules?.[value]) {
      node.setAttribute('title', rules[value]);
      node.setAttribute('aria-label', `${labelFor(kind, value)}: ${rules[value]}`);
    }
    return node;
  }

  function usdFromMicro(value) {
    if (value == null || value === '') return shell().DASH;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return shell().DASH;
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(numeric / 1000000);
  }

  function formatVisibleTime(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return shell().DASH;
    const seconds = Math.round(numeric / 1000);
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }

  function td(content, className) {
    const s = shell();
    const cell = s.el('td', className);
    if (typeof content === 'string') cell.textContent = content;
    else cell.appendChild(content);
    return cell;
  }

  // ------------------------------------------------------------ list

  function renderUserRows(payload) {
    const s = shell();
    const fragment = document.createDocumentFragment();
    const items = Array.isArray(payload?.items) ? payload.items : [];
    if (!items.length) {
      const row = s.el('tr');
      const cell = s.el('td', 'panel-empty', 'No users match these filters.');
      cell.setAttribute('colspan', '6');
      row.appendChild(cell);
      fragment.appendChild(row);
      return fragment;
    }
    items.forEach((user) => {
      const row = s.el('tr');
      const who = s.el('div', 'who');
      const link = s.el('a', '', user.display_name || user.email || `User #${user.user_id}`);
      link.setAttribute('href', `#users/${encodeURIComponent(String(user.user_id))}`);
      who.appendChild(link);
      who.appendChild(s.el('small', '', user.email || ''));
      row.appendChild(td(who));
      row.appendChild(td(groupBadge(user.group_badge)));
      row.appendChild(td(signalBadge('lifecycle', user.lifecycle?.segment)));
      row.appendChild(td(signalBadge('operational', user.operational?.state)));
      row.appendChild(td(s.formatTimestamp(user.last_meaningful_activity_at, 'No activity')));
      const button = s.el('button', 'row-action', 'Review evidence');
      button.setAttribute('type', 'button');
      button.setAttribute('aria-haspopup', 'dialog');
      button.addEventListener('click', () => openEvidence(user, button));
      row.appendChild(td(button));
      fragment.appendChild(row);
    });
    return fragment;
  }

  function renderPager(payload) {
    const s = shell();
    const total = Number(payload?.total) || 0;
    const offset = Number(payload?.offset) || 0;
    const shown = Array.isArray(payload?.items) ? payload.items.length : 0;
    if (!total) return '0 users';
    return `Showing ${s.formatNumber(offset + 1)}–${s.formatNumber(offset + shown)} of ${s.formatNumber(total)}`;
  }

  async function loadList({ offset = 0 } = {}) {
    const s = shell();
    const view = document.getElementById('usersView');
    const body = document.getElementById('usersBody');
    const seq = s.nextSeq('users');
    s.setPanelState(view, { busy: true });
    try {
      const payload = await s.request(`${USERS_PATH}?${s.userListParams({ offset, limit: PAGE_SIZE })}`);
      if (!s.isCurrent('users', seq)) return;
      state.list = { offset, total: Number(payload.total) || 0, items: payload.items || [], loaded: true };
      if (body) {
        s.clear(body);
        body.appendChild(renderUserRows(payload));
      }
      const range = document.getElementById('usersRange');
      if (range) range.textContent = renderPager(payload);
      const prev = document.getElementById('usersPrev');
      const next = document.getElementById('usersNext');
      if (prev) prev.disabled = offset <= 0;
      if (next) next.disabled = offset + PAGE_SIZE >= state.list.total;
      s.setPanelState(view, { busy: false });
    } catch (error) {
      if (!s.isCurrent('users', seq)) return;
      if (await s.handleAccessLost(error)) return;
      s.setPanelState(view, { busy: false, error: s.SECTION_UNAVAILABLE, stale: state.list.loaded });
    }
  }

  // ------------------------------------------------------- evidence

  function evidenceList(values, fallback) {
    const s = shell();
    const list = s.el('ul', 'evidence-list');
    (values || []).forEach((fact) => list.appendChild(s.el('li', '', fact)));
    if (!list.children.length) list.appendChild(s.el('li', 'muted', fallback));
    return list;
  }

  function renderEvidence(user) {
    const s = shell();
    const root = s.el('div');
    root.appendChild(s.el('p', 'muted', user.email || `User #${user.user_id}`));
    const signals = s.el('div', 'signals');
    signals.appendChild(signalBadge('lifecycle', user.lifecycle?.segment));
    signals.appendChild(signalBadge('operational', user.operational?.state));
    signals.appendChild(signalBadge('commercial', user.commercial_tier ?? user.commercial?.commercial_tier));
    root.appendChild(signals);
    const lifecycle = s.el('section');
    lifecycle.appendChild(s.el('h3', '', 'Lifecycle evidence'));
    lifecycle.appendChild(s.el('p', '', user.lifecycle?.reason || 'No lifecycle reason is available.'));
    lifecycle.appendChild(evidenceList(user.lifecycle?.evidence, 'No lifecycle evidence is available.'));
    root.appendChild(lifecycle);
    const operational = s.el('section');
    operational.appendChild(s.el('h3', '', 'Operational evidence'));
    operational.appendChild(s.el('p', '', user.operational?.reason || 'No operational reason is available.'));
    operational.appendChild(evidenceList(user.operational?.evidence, 'No operational evidence is available.'));
    root.appendChild(operational);
    return root;
  }

  function accountManagementHref(user) {
    const query = user?.email || (user?.user_id != null ? String(user.user_id) : '');
    return `/app?view=admin&adminTab=users${query ? `&adminUserQuery=${encodeURIComponent(query)}` : ''}`;
  }

  function openEvidence(user, opener) {
    const s = shell();
    state.evidenceUser = user;
    const body = document.getElementById('evidenceBody');
    if (body) {
      s.clear(body);
      body.appendChild(renderEvidence(user));
    }
    const title = document.getElementById('evidenceTitle');
    if (title) title.textContent = user.display_name || user.email || `User #${user.user_id}`;
    document.getElementById('evidenceProfile')?.setAttribute('href', `#users/${encodeURIComponent(String(user.user_id))}`);
    document.getElementById('evidenceAccount')?.setAttribute('href', accountManagementHref(user));
    s.openDialog(document.getElementById('evidenceDialog'), opener);
  }

  // -------------------------------------------------------- profile

  function activationWeekLabel(activatedAt) {
    const s = shell();
    if (!activatedAt) return 'Not yet activated';
    const date = new Date(activatedAt);
    if (!Number.isFinite(date.getTime())) return 'Not yet activated';
    const monday = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
    monday.setUTCDate(monday.getUTCDate() - ((monday.getUTCDay() + 6) % 7));
    return `Activation week of ${s.formatDateOnly(monday.toISOString().slice(0, 10))}`;
  }

  function renderProfileHeader(profile) {
    const s = shell();
    const root = s.el('div');
    const crumb = s.el('nav', 'breadcrumb');
    crumb.setAttribute('aria-label', 'Breadcrumb');
    const parent = s.el('a', '', 'Users');
    parent.setAttribute('href', '#users');
    crumb.appendChild(parent);
    crumb.appendChild(s.el('span', '', '/'));
    crumb.appendChild(s.el('span', '', profile.display_name || profile.email || `User #${profile.user_id}`));
    root.appendChild(crumb);

    const head = s.el('div', 'page-head profile-head');
    const identity = s.el('div', 'identity');
    identity.appendChild(s.el('div', 'avatar', String(profile.display_name || profile.email || '?').trim().charAt(0).toUpperCase()));
    const text = s.el('div');
    const heading = s.el('h1', '', profile.display_name || profile.email || `User #${profile.user_id}`);
    heading.setAttribute('tabindex', '-1');
    text.appendChild(heading);
    const line = s.el('p', 'muted');
    line.append(`${profile.email || ''} · `);
    line.appendChild(groupBadge(profile.group_badge));
    line.append(` · ${activationWeekLabel(profile.lifecycle?.activated_at)}`);
    text.appendChild(line);
    identity.appendChild(text);
    head.appendChild(identity);
    const back = s.el('a', 'module-link', 'Back to users');
    back.setAttribute('href', '#users');
    head.appendChild(back);
    root.appendChild(head);

    const panel = s.el('section', 'identity-panel');
    panel.appendChild(s.el('h2', '', 'Current identity'));
    const help = s.el('button', 'help-button', '?');
    help.setAttribute('type', 'button');
    help.setAttribute('aria-label', 'How states are determined');
    help.setAttribute('title', 'How states are determined');
    help.addEventListener('click', () => s.openRules(help));
    panel.appendChild(help);
    const badges = s.el('p');
    badges.appendChild(signalBadge('lifecycle', profile.lifecycle?.segment));
    badges.append(' ');
    badges.appendChild(signalBadge('operational', profile.operational?.state));
    badges.append(' ');
    badges.appendChild(signalBadge('commercial', profile.commercial?.commercial_tier));
    panel.appendChild(badges);
    const reason = profile.operational?.state === 'healthy' ? profile.lifecycle?.reason : profile.operational?.reason;
    const evaluated = profile.operational?.calculated_at || profile.lifecycle?.calculated_at;
    panel.appendChild(s.el('p', '', `${reason || 'No reason recorded'} · last evaluated ${s.formatTimestamp(evaluated)}`));
    panel.appendChild(s.el('small', '', `Last meaningful activity ${s.formatTimestamp(profile.last_meaningful_activity_at, 'none')} · First successful result ${s.formatTimestamp(profile.lifecycle?.activated_at, 'none')}`));
    const account = s.el('a', 'module-link', 'Open account management');
    account.setAttribute('href', accountManagementHref(profile));
    panel.appendChild(account);
    root.appendChild(panel);

    const milestones = s.el('div', 'milestones');
    Object.entries(profile.activation_milestones || {})
      .sort((left, right) => new Date(left[1]) - new Date(right[1]))
      .forEach(([name, occurredAt]) => {
        const item = s.el('div', 'milestone');
        item.append(eventLabel(name));
        item.appendChild(s.el('br'));
        item.appendChild(s.el('small', '', s.formatTimestamp(occurredAt)));
        milestones.appendChild(item);
      });
    if (!milestones.children.length) milestones.appendChild(s.el('p', 'panel-empty', 'No activation milestones recorded.'));
    root.appendChild(milestones);

    const tabs = s.el('div', 'tabs');
    tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', 'User analytics sections');
    SECTIONS.forEach((section) => {
      const button = s.el('button', '', SECTION_LABELS[section]);
      button.setAttribute('type', 'button');
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', section === 'overview' ? 'true' : 'false');
      button.setAttribute('aria-controls', `profileSection-${section}`);
      button.dataset.section = section;
      button.addEventListener('click', () => selectSection(section));
      button.addEventListener('keydown', (event) => {
        const keys = { ArrowRight: 1, ArrowLeft: -1, Home: 0, End: SECTIONS.length - 1 };
        if (!(event.key in keys)) return;
        event.preventDefault();
        const index = SECTIONS.indexOf(section);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? SECTIONS.length - 1 : (index + keys[event.key] + SECTIONS.length) % SECTIONS.length;
        selectSection(SECTIONS[next]);
      });
      tabs.appendChild(button);
    });
    root.appendChild(tabs);
    return root;
  }

  function definitions(entries) {
    const s = shell();
    const list = s.el('dl', 'profile-facts');
    entries.forEach(([label, value]) => {
      list.appendChild(s.el('dt', '', label));
      list.appendChild(s.el('dd', '', value));
    });
    return list;
  }

  function renderProfileOverview(profile) {
    const s = shell();
    const lifecycle = profile.lifecycle || {};
    const commercial = profile.commercial || {};
    const root = s.el('div');
    const value = s.el('section', 'detail-section');
    value.appendChild(s.el('h3', '', 'User value summary'));
    value.appendChild(definitions([
      ['Selected period', `${s.formatDateOnly(profile.selected_period_start)} – ${s.formatDateOnly(profile.selected_period_end)}`],
      ['Activated', s.formatTimestamp(lifecycle.activated_at, 'Not activated')],
      ['Active days (30d)', s.formatNumber(lifecycle.active_days_30d)],
      ['Successful backtests (30d)', s.formatNumber(lifecycle.successful_backtests_30d)],
      ['Inactive UTC days', s.formatNumber(lifecycle.inactive_days)],
      ['Lifetime net purchased', s.formatCredits(commercial.lifetime_net_purchased_micro)],
      ['Consumed in period', s.formatCredits(commercial.consumed_micro)],
      ['Available balance', s.formatCredits(commercial.total_available_micro)],
    ]));
    root.appendChild(value);

    const evidence = s.el('section', 'detail-section');
    evidence.appendChild(s.el('h3', '', 'Lifecycle evidence'));
    evidence.appendChild(evidenceList(lifecycle.evidence, 'No lifecycle evidence is available.'));
    evidence.appendChild(s.el('h3', '', 'Operational evidence'));
    evidence.appendChild(evidenceList(profile.operational?.evidence, 'No operational evidence is available.'));
    root.appendChild(evidence);

    const runs = s.el('section', 'detail-section');
    runs.appendChild(s.el('h3', '', 'Run summary'));
    runs.appendChild(definitions(Object.entries(profile.run_summary || {}).map(([key, count]) => [s.humanize(key), s.formatNumber(count)])));
    root.appendChild(runs);

    const usage = s.el('section', 'detail-section');
    usage.appendChild(s.el('h3', '', 'Billing and usage'));
    const totalTokens = Number(profile.input_tokens) + Number(profile.output_tokens);
    usage.appendChild(definitions([
      ['Input tokens', s.formatNumber(profile.input_tokens)],
      ['Output tokens', s.formatNumber(profile.output_tokens)],
      ['Total tokens', Number.isFinite(totalTokens) ? s.formatNumber(totalTokens) : s.DASH],
      ['ATL platform model cost', usdFromMicro(Number(profile.platform_model_cost_usd) * 1000000)],
      ['ATL Credits debited', s.formatCredits(profile.credits_debited_micro)],
      ...Object.entries(profile.billing_lane_mix || {}).map(([lane, count]) => [
        lane === 'byok' ? 'BYOK usage — no ATL Credits debit' : s.humanize(lane), s.formatNumber(count),
      ]),
    ]));
    root.appendChild(usage);

    const movement = s.el('section', 'detail-section');
    movement.appendChild(s.el('h3', '', 'Recent lifecycle movement'));
    const transitions = s.el('ol', 'timeline');
    (profile.recent_lifecycle_transitions || []).forEach((transition) => {
      const item = s.el('li');
      const head = s.el('div');
      head.appendChild(s.el('strong', '', `${labelFor('lifecycle', transition.from_segment)} → ${labelFor('lifecycle', transition.to_segment)}`));
      head.appendChild(s.el('span', '', `${s.formatNumber(transition.users)} user${Number(transition.users) === 1 ? '' : 's'}`));
      item.appendChild(head);
      item.appendChild(s.el('p', '', `${s.formatDateOnly(transition.period_start)} – ${s.formatDateOnly(transition.period_end)}${transition.data_quality === 'partial' ? ` · ${s.INCOMPLETE}` : ''}`));
      transitions.appendChild(item);
    });
    if (!transitions.children.length) transitions.appendChild(s.el('li', 'panel-empty', 'No lifecycle transitions in this range.'));
    movement.appendChild(transitions);
    root.appendChild(movement);

    const footprint = s.el('section', 'detail-section');
    footprint.appendChild(s.el('h3', '', 'Recent footprint'));
    const list = s.el('ol', 'timeline');
    (profile.recent_footprint || []).forEach((item) => {
      const row = s.el('li');
      const head = s.el('div');
      head.appendChild(s.el('strong', '', eventLabel(item.event_name)));
      head.appendChild(s.el('span', '', s.formatTimestamp(item.occurred_at)));
      row.appendChild(head);
      const details = [item.page_view, item.provider_id, item.model_id, item.billing_mode, item.outcome, item.error_category].filter(Boolean).map(s.humanize).join(' · ');
      if (details) row.appendChild(s.el('p', '', details));
      list.appendChild(row);
    });
    if (!list.children.length) list.appendChild(s.el('li', 'panel-empty', 'No recent footprint events.'));
    footprint.appendChild(list);
    root.appendChild(footprint);
    return root;
  }

  function activityTable(headers, rows) {
    const s = shell();
    const wrap = s.el('div', 'table-wrap');
    const table = s.el('table');
    const head = s.el('thead');
    const headRow = s.el('tr');
    headers.forEach((label) => {
      const cell = s.el('th', '', label);
      cell.setAttribute('scope', 'col');
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = s.el('tbody');
    rows.forEach((cells) => {
      const row = s.el('tr');
      cells.forEach((cell) => row.appendChild(td(cell)));
      body.appendChild(row);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function renderActivityItems(section, items) {
    const s = shell();
    const rows = Array.isArray(items) ? items : [];
    if (!rows.length) return s.el('p', 'panel-empty', 'No activity in this section.');
    if (section === 'timeline') {
      const list = s.el('ol', 'timeline');
      rows.forEach((item) => {
        const row = s.el('li');
        const head = s.el('div');
        head.appendChild(s.el('strong', '', eventLabel(item.event_name)));
        head.appendChild(s.el('span', '', s.formatTimestamp(item.occurred_at)));
        row.appendChild(head);
        const details = [item.outcome, item.provider_id, item.model_id, item.billing_mode, item.error_category].filter(Boolean).map(s.humanize).join(' · ');
        row.appendChild(s.el('p', '', details || 'No additional display-safe details.'));
        list.appendChild(row);
      });
      return list;
    }
    if (section === 'runs') {
      return activityTable(['Time', 'Run event', 'Outcome', 'Provider / model', 'Billing lane', 'Error category'], rows.map((item) => [
        s.formatTimestamp(item.occurred_at), eventLabel(item.event_name),
        item.outcome ? s.humanize(item.outcome) : s.DASH,
        [item.provider_id, item.model_id].filter(Boolean).join(' · ') || s.DASH,
        item.billing_mode ? s.humanize(item.billing_mode) : s.DASH,
        item.error_category ? s.humanize(item.error_category) : s.DASH,
      ]));
    }
    if (section === 'usage') {
      return activityTable(['Time', 'Usage event', 'Provider / model', 'Billing lane', 'Input', 'Output', 'ATL cost', 'ATL Credits debited'], rows.map((item) => {
        const byok = item.billing_mode === 'byok';
        return [
          s.formatTimestamp(item.occurred_at), eventLabel(item.event_name),
          [item.provider_id, item.model_id].filter(Boolean).join(' · ') || s.DASH,
          byok ? 'BYOK — no ATL charge' : (item.billing_mode ? s.humanize(item.billing_mode) : s.DASH),
          s.formatNumber(item.input_tokens), s.formatNumber(item.output_tokens),
          byok || item.cost_micro_usd == null ? s.DASH : usdFromMicro(item.cost_micro_usd),
          byok || item.amount_micro == null ? s.DASH : s.formatCredits(item.amount_micro),
        ];
      }));
    }
    // Sessions: Started · Events · Visible time. Region / Device / Browser are
    // collected but not displayed (design D15); the columns are gone, not hidden.
    return activityTable(['Started', 'Events', 'Visible time'], rows.map((item) => [
      s.formatTimestamp(item.occurred_at), s.formatNumber(item.session_event_count), formatVisibleTime(item.visible_ms),
    ]));
  }

  function sectionPanel(section) {
    return document.getElementById(`profileSection-${section}`);
  }

  function renderSectionPanel(section) {
    const s = shell();
    const panel = sectionPanel(section);
    if (!panel) return;
    const sectionState = state.profile.sections[section];
    s.clear(panel);
    if (section === 'overview') {
      panel.appendChild(renderProfileOverview(state.profile.detail));
      return;
    }
    if (sectionState.error) {
      panel.appendChild(s.el('p', 'panel-error', sectionState.error));
    }
    if (sectionState.loaded) panel.appendChild(renderActivityItems(section, sectionState.items));
    else if (sectionState.loading) panel.appendChild(s.el('p', 'panel-status muted', 'Loading activity…'));
    if (sectionState.nextCursor) {
      const more = s.el('button', 'load-more', `Load more ${SECTION_LABELS[section].toLowerCase()}`);
      more.setAttribute('type', 'button');
      more.disabled = sectionState.loading;
      more.addEventListener('click', () => loadSection(section, { append: true }));
      panel.appendChild(more);
    }
  }

  async function loadSection(section, { append = false } = {}) {
    const s = shell();
    const sectionState = state.profile.sections[section];
    const userId = state.profile.userId;
    if (!sectionState || !userId || sectionState.loading) return;
    if (append && !sectionState.nextCursor) return;
    sectionState.loading = true;
    sectionState.error = null;
    const seq = ++sectionState.requestSeq;
    renderSectionPanel(section);
    const params = new URLSearchParams({ section, limit: String(PAGE_SIZE) });
    if (append) params.set('cursor', sectionState.nextCursor);
    try {
      const payload = await s.request(`${USERS_PATH}/${encodeURIComponent(String(userId))}/activity?${params}`);
      if (String(state.profile.userId) !== String(userId) || seq !== sectionState.requestSeq) return;
      const next = Array.isArray(payload.items) ? payload.items : [];
      sectionState.items = append ? sectionState.items.concat(next) : next;
      sectionState.nextCursor = payload.next_cursor || null;
      sectionState.loaded = true;
    } catch (error) {
      if (seq !== sectionState.requestSeq) return;
      if (await s.handleAccessLost(error)) return;
      sectionState.error = append ? 'More activity is temporarily unavailable.' : s.SECTION_UNAVAILABLE;
    } finally {
      if (seq === sectionState.requestSeq) {
        sectionState.loading = false;
        renderSectionPanel(section);
      }
    }
  }

  async function selectSection(value) {
    const section = SECTIONS.includes(value) ? value : 'overview';
    state.profile.section = section;
    const root = document.getElementById('profile');
    root?.querySelectorAll('[role="tab"]').forEach((button) => {
      const selected = button.dataset.section === section;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    SECTIONS.forEach((name) => {
      const panel = sectionPanel(name);
      if (panel) panel.hidden = name !== section;
    });
    if (section !== 'overview' && !state.profile.sections[section]?.loaded) {
      await loadSection(section, { append: false });
    }
  }

  function paintProfile(profile) {
    const s = shell();
    const root = document.getElementById('profile');
    if (!root) return;
    s.clear(root);
    root.appendChild(renderProfileHeader(profile));
    SECTIONS.forEach((section) => {
      const panel = s.el('section', 'detail-section');
      panel.id = `profileSection-${section}`;
      panel.setAttribute('role', 'tabpanel');
      panel.hidden = section !== 'overview';
      root.appendChild(panel);
    });
    renderSectionPanel('overview');
    root.querySelector('h1')?.focus?.({ preventScroll: true });
  }

  async function openProfile(userId) {
    const s = shell();
    if (!/^\d+$/.test(String(userId || ''))) return;
    state.profile = {
      userId: String(userId),
      detail: null,
      section: 'overview',
      sections: Object.fromEntries(SECTIONS.filter((name) => name !== 'overview').map((name) => [name, emptySection()])),
    };
    const seq = s.nextSeq('profile');
    const root = document.getElementById('profile');
    if (root) {
      s.clear(root);
      root.appendChild(s.el('p', 'panel-status muted', 'Loading user analytics…'));
    }
    try {
      const { from, to } = s.rangeDates(s.state.range);
      const params = new URLSearchParams({ from, to });
      const profile = await s.request(`${USERS_PATH}/${encodeURIComponent(String(userId))}?${params}`);
      if (!s.isCurrent('profile', seq) || String(state.profile.userId) !== String(userId)) return;
      state.profile.detail = profile;
      paintProfile(profile);
    } catch (error) {
      if (!s.isCurrent('profile', seq)) return;
      if (await s.handleAccessLost(error)) return;
      if (root) {
        s.clear(root);
        root.appendChild(s.el('p', 'panel-error', error?.status === 404 ? 'Analytics user was not found.' : 'User analytics are temporarily unavailable.'));
      }
    }
  }

  function bind() {
    const s = shell();
    document.getElementById('usersSearch')?.addEventListener('submit', (event) => {
      event.preventDefault();
      const input = document.getElementById('usersQuery');
      s.setFilters({ q: String(input?.value || '').trim().slice(0, 100) });
    });
    document.getElementById('usersPriority')?.addEventListener('change', (event) => {
      s.setFilters({ priority: Boolean(event.target.checked) });
    });
    document.getElementById('usersPrev')?.addEventListener('click', () => loadList({ offset: Math.max(0, state.list.offset - PAGE_SIZE) }));
    document.getElementById('usersNext')?.addEventListener('click', () => loadList({ offset: state.list.offset + PAGE_SIZE }));
  }

  document.addEventListener('DOMContentLoaded', bind);
  document.addEventListener('admin:route', (event) => {
    const detail = event.detail || {};
    if (detail.route !== 'users') return;
    const query = document.getElementById('usersQuery');
    if (query) query.value = detail.filters?.q || '';
    const priority = document.getElementById('usersPriority');
    if (priority) priority.checked = Boolean(detail.filters?.priority);
    if (detail.id) openProfile(detail.id);
    else loadList({ offset: 0 });
  });
  document.addEventListener('admin:retry', (event) => {
    if (event.detail?.panel === 'users') loadList({ offset: state.list.offset });
  });

  window.AdminUsers = {
    state,
    groupBadge, signalBadge, renderUserRows, renderPager, renderEvidence, activationWeekLabel,
    renderProfileHeader, renderProfileOverview, renderActivityItems, accountManagementHref,
    loadList, openProfile, selectSection, loadSection,
  };
})();
```

- [ ] **Step 4: Run and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_users_frontend.py -v
```

Expected: **PASS** on all thirteen tests. Two are the D11/D15 pins in code form: `test_group_badge_is_rendered_verbatim_from_the_server` feeds a badge string no server would emit and gets it back unchanged, and `test_profile_overview_renders_value_facts_and_drops_the_d15_fields` walks every text node of the rendered profile for `US`, `desktop` and `Chrome` (the fixture's region/device/browser values) and finds none.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/js/admin-users.js dashboard/backend/tests/test_admin_users_frontend.py
git commit -m "$(cat <<'EOF'
feat: render the /admin users list and lazy user profile

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: The old console — drop the Analytics tab and the redirect, delete the mock and the two wired modules, bump and re-pin

**Files:**
- Modify: `dashboard/frontend/js/admin-tabs.js` — whole file (109 lines).
- Modify: `dashboard/frontend/app.html` — line 16 (`styles.css?v=`), the rail at lines 2140-2145, `#adminPanelAnalytics` at lines 2149-2426 plus the `hidden` on `#adminPanelUsers` at line 2428, the two dialogs `#adminAnalyticsRulesDialog` / `#adminAnalyticsEvidenceDialog` (lines 2618-2645, between `#adminView`'s closing `</div>` and `#adminGrantReasonDialog`), the script tags at lines 2701 and 2708-2710.
- Modify: `dashboard/frontend/app.js` — the auth-sync block at lines 5037-5043, the Profile → Admin handler at lines 5405-5408, the refresh handler at lines 5412-5414, the `page === 'admin'` branch at lines 10839-10860.
- Modify: `dashboard/frontend/styles.css` — the `/* —— Admin user-value analytics —— */` block (line 2785 through the last `.admin-value-dialog-actions` media block closing at line 3660) and the `/* —— Admin Analytics: dense, read-only operator intelligence —— */` block (line 14759 through the `prefers-reduced-motion` block closing at line 15617).
- Delete: `dashboard/frontend/admin-analytics.html`, `dashboard/frontend/js/admin-analytics.js`, `dashboard/frontend/js/admin-analytics-value.js`.
- Modify: `dashboard/backend/tests/test_admin_tabs_redirect.py` (rewritten), `dashboard/backend/tests/test_admin_analytics_frontend.py` (rewritten), `dashboard/backend/tests/test_admin_credits_frontend.py` (lines 48-49, 132-135, 137-148, 152-154), `dashboard/backend/tests/test_frontend_fast_boot.py` (lines 194, 196, 202), `dashboard/backend/tests/test_credit_format_frontend.py` (line 79), `dashboard/backend/tests/test_analytics_frontend.py` (line 44), `dashboard/backend/tests/test_backtest_comparison_frontend.py` (lines 194-195).
- Delete: `dashboard/backend/tests/test_admin_analytics_value_frontend.py`.

**Interfaces:**
- Consumes: nothing new.
- Produces: `window.AdminTabs = { onEnter, openAccountManagement, setTab }` with `DEFAULT_TAB = 'users'`, `ALLOWED_TABS = {users, providers, activity}`, no navigation anywhere, and `onEnter()` honouring `?adminUserQuery=<email>` by calling `openAccountManagement({ email })` — the pre-fill Task 8's "Open account management" link carries. `app.html`: `app.js?v=133`, `styles.css?v=141`, `js/admin-tabs.js?v=9`; no `admin-analytics*.js` tag. `app.js`: Profile → Admin navigates to `/admin`.

Cache-buster numbers, and where they come from. At `c3bbf2ed` — the checkout this plan was written against, where PR 0 has **not** merged — `app.html` reads `app.js?v=131` (line 2701), `js/admin-tabs.js?v=7` (line 2710) and `styles.css?v=140` (line 16), and every test that pins them agrees (`test_admin_credits_frontend.py:153` → `?v=7`; `test_frontend_fast_boot.py:194`, `test_analytics_frontend.py:44`, `test_backtest_comparison_frontend.py:194` → `?v=131`). PR 0's Task 4 bumps `app.js` → `?v=132` and `js/admin-tabs.js` → `?v=8` (its plan, lines 463 and 678-680) and leaves `styles.css` alone, and §13 merges PR 0 before this plan starts (Global Constraints). The execution baseline is therefore **132 / 8 / 140**, and this task bumps each by one to **133 / 9 / 141** — the literals every test below and Step 4 use.

- [ ] **Step 0: Verify the baseline before writing a single literal**

```bash
rg -n 'app\.js\?v=|admin-tabs\.js\?v=|styles\.css\?v=' dashboard/frontend/app.html
```

Expected: `styles.css?v=140`, `app.js?v=132`, `js/admin-tabs.js?v=8`. Then proceed with the numbers as written. Two other outcomes:

- `app.js?v=131` / `js/admin-tabs.js?v=7`: PR 0 has not merged into this branch. Stop — this plan is not yet runnable (§13 ordering), exactly as Task 1 stops when `legacy_status` is still present.
- Any other values (an unrelated PR bumped a tag in between): the target is **current + 1** for that tag. Substitute it for `133` / `9` / `141` in **every** occurrence this task writes — the Interfaces line above, `test_app_lifecycle_and_cache_versions_are_wired` in Step 1, the four per-file pin edits at the end of Step 1, and the Step 4 `app.html` edits — before running anything. (Task 10's `test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded` matches `?v=\d+` and Task 4's guard pins only the page's own `?v=1` tags; neither carries the console's numbers.) The tags and the pins move together or `test_app_lifecycle_and_cache_versions_are_wired` fails, which is the point of that test; a pin written from this plan's numbers against a tag written from the checkout's is the one way to make it fail for a reason that is not a bug.

Also record what the four other pin files currently say (`rg -n 'app\.js\?v=|admin-tabs\.js\?v=|styles\.css\?v=' dashboard/backend/tests/`). PR 0's plan re-pins only `test_admin_analytics_frontend.py`; if its run left the other four on `?v=131` / `?v=7`, they are red on the baseline before this task touches them — a pre-existing PR 0 gap, not a regression to chase. The per-file edits at the end of Step 1 name their `c3bbf2ed` text and replace whatever value stands there with this task's.

- [ ] **Step 1: Write the failing tests**

Replace `dashboard/backend/tests/test_admin_tabs_redirect.py` in full:

```python
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
```

Replace `dashboard/backend/tests/test_admin_analytics_frontend.py` in full (the fixture tests stay; every assertion that pinned the removed in-app surface goes; the cache-version test becomes the lockstep owner for the console's three bumped tags and the new page's five pins; the style test now asserts the deleted families are gone):

```python
"""Fixture contracts for the admin analytics API, plus the console's cache-buster pins."""

import json
from pathlib import Path

from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
)
from dashboard.backend.domain.analytics.value_queries import (
    CommercialAnalyticsResponse,
    GroupAnalyticsResponse,
    LifecycleAnalyticsResponse,
    OperationalAnalyticsResponse,
    PaginatedValueUsers,
    RetentionAnalyticsResponse,
    ValueUserProfile,
)
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, STYLES


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")


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


def test_fixtures_match_committed_analytics_shapes():
    overview = load_fixture("overview.json")
    partial = load_fixture("overview_partial_error.json")
    lifecycle = load_fixture("lifecycle.json")
    retention = load_fixture("retention.json")
    commercial = load_fixture("commercial.json")
    operational = load_fixture("operational.json")
    users = load_fixture("users.json")
    profile = load_fixture("user_detail.json")
    assert {"daily_active_users", "billing_lane_mix", "availability", "last_updated"} <= overview.keys()
    assert partial["availability"]["growth"] == {
        "available": False,
        "error_code": "temporarily_unavailable",
    }
    assert partial["availability"]["snapshot"]["available"] is True
    assert {"headline", "segment_counts", "weekly_segments", "transitions"} <= lifecycle.keys()
    assert {"cohorts", "summary_week_1", "summary_week_2", "summary_week_4"} <= retention.keys()
    assert {"tier_counts", "selected_period", "current_balances", "purchased_by_day", "consumed_by_day"} <= commercial.keys()
    assert {"operational_state_counts", "top_failure_categories", "top_operational_reasons"} <= operational.keys()
    assert {"items", "total", "limit", "offset"} == users.keys()
    assert {"user_group", "role", "group_badge", "last_meaningful_activity_at"} <= users["items"][0].keys()
    assert {"state", "activation_milestones", "lifecycle", "operational", "commercial", "group_badge"} <= profile.keys()
    assert "next_cursor" in load_fixture("activity_timeline.json")
    assert {"users", "agents", "active_dashboard_backtests", "max_active_dashboard_backtests"} <= load_fixture("admin_stats.json").keys()


def test_fixtures_validate_against_committed_analytics_models():
    AnalyticsOverview.model_validate(load_fixture("overview.json"))
    AnalyticsOverview.model_validate(load_fixture("overview_partial_error.json"))
    LifecycleAnalyticsResponse.model_validate(load_fixture("lifecycle.json"))
    RetentionAnalyticsResponse.model_validate(load_fixture("retention.json"))
    CommercialAnalyticsResponse.model_validate(load_fixture("commercial.json"))
    OperationalAnalyticsResponse.model_validate(load_fixture("operational.json"))
    GroupAnalyticsResponse.model_validate(load_fixture("groups.json"))
    PaginatedValueUsers.model_validate(load_fixture("users.json"))
    ValueUserProfile.model_validate(load_fixture("user_detail.json"))
    for name in (
        "activity_timeline.json",
        "activity_runs.json",
        "activity_usage.json",
        "activity_sessions.json",
    ):
        AnalyticsActivityPage.model_validate(load_fixture(name))


def test_byok_fixture_never_reports_atl_cost():
    payload = load_fixture("activity_usage.json")
    byok = next(item for item in payload["items"] if item["billing_mode"] == "byok")
    assert byok["cost_micro_usd"] == 0
    assert byok["amount_micro"] is None


def test_in_app_analytics_surface_is_gone():
    """PR C (design §7.5): one analytics surface, at /admin."""
    for marker in (
        'id="adminTabAnalytics"', 'id="adminPanelAnalytics"', 'id="adminAnalyticsOverview"',
        'id="adminAnalyticsProfile"', 'id="adminAnalyticsRulesDialog"', 'id="adminAnalyticsEvidenceDialog"',
        "admin-analytics-legacy-overview", "js/admin-analytics.js", "js/admin-analytics-value.js",
    ):
        assert marker not in APP_HTML, marker
    assert not (FRONTEND / "admin-analytics.html").exists()
    assert not (FRONTEND / "js" / "admin-analytics.js").exists()
    assert not (FRONTEND / "js" / "admin-analytics-value.js").exists()
    for call in (
        "window.AdminAnalytics.syncAuth(user)", "window.AdminAnalytics.onEnter()",
        "window.AdminAnalytics.refresh()", "window.AdminAnalyticsValue.syncAuth(user)",
        "window.AdminAnalyticsValue.onEnter()",
    ):
        assert call not in APP_JS, call


def test_admin_rail_has_three_tabs_defaulting_to_users():
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index("</nav>", nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    expected = ["users", "providers", "activity"]
    assert nav_markup.count("data-admin-tab=") == 3
    assert [nav_markup.index(f'data-admin-tab="{value}"') for value in expected] == sorted(
        nav_markup.index(f'data-admin-tab="{value}"') for value in expected
    )
    assert 'aria-orientation="vertical"' in nav_markup
    assert 'id="adminTabUsers" class="admin-tab is-active"' in nav_markup
    assert '<section id="adminPanelUsers" class="admin-tab-panel" role="tabpanel" aria-labelledby="adminTabUsers" data-admin-panel="users">' in APP_HTML


def test_profile_menu_admin_entry_opens_the_admin_page():
    start = APP_JS.index("document.getElementById('accountMenuAdminBtn')?.addEventListener('click'")
    handler = APP_JS[start:start + 600]
    assert "window.location.assign('/admin')" in handler
    assert "navigateToPage('admin')" not in handler


def test_app_lifecycle_and_cache_versions_are_wired():
    # Lockstep owner for the console's bumped tags and the /admin page's pins:
    # every bump edits this test in the same change (Global Constraints).
    assert 'styles.css?v=141' in APP_HTML
    assert 'app.js?v=133' in APP_HTML
    assert 'js/admin-tabs.js?v=9' in APP_HTML
    for tag in (
        'href="admin.css?v=1"',
        'src="js/admin-shell.js?v=1"',
        'src="js/credit-format.js?v=1"',
        'src="js/admin-live.js?v=1"',
        'src="js/admin-overview.js?v=1"',
        'src="js/admin-users.js?v=1"',
    ):
        assert tag in ADMIN_HTML, tag


def test_analytics_style_families_left_styles_css():
    for family in (
        ".admin-analytics-", ".admin-value-", ".admin-priority-", ".admin-lifecycle-",
        ".admin-group-", ".admin-help-btn", ".admin-profile-evidence-grid", ".admin-commercial-tier-grid",
    ):
        assert family not in STYLES, family
    for kept in (".admin-workspace", ".admin-rail", ".admin-tab:focus-visible", ".admin-rail button"):
        assert kept in STYLES, kept
```

`dashboard/backend/tests/test_admin_credits_frontend.py`: change lines 48-49

```python
    assert nav_markup.count('data-admin-tab=') == 4
    assert nav_markup.index('data-admin-tab="analytics"') < nav_markup.index('data-admin-tab="users"')
```

to

```python
    assert nav_markup.count('data-admin-tab=') == 3
```

lines 132-135

```python
def test_admin_tabs_default_to_analytics_and_are_url_backed():
    assert "DEFAULT_TAB = 'analytics'" in ADMIN_TABS_JS
```

to

```python
def test_admin_tabs_default_to_users_and_are_url_backed():
    assert "DEFAULT_TAB = 'users'" in ADMIN_TABS_JS
```

lines 138 and 143-144 (inside `test_admin_tabs_have_four_tabs_in_usage_order_and_legacy_alias`, renamed `test_admin_tabs_have_three_tabs_in_usage_order_and_legacy_alias`)

```python
    assert nav_markup.count('data-admin-tab=') == 4
    assert nav_markup.index('data-admin-tab="analytics"') < nav_markup.index('data-admin-tab="users"')
```

to

```python
    assert nav_markup.count('data-admin-tab=') == 3
```

and line 153 `assert 'js/admin-tabs.js?v=7' in APP_HTML` → `assert 'js/admin-tabs.js?v=9' in APP_HTML`.

The four pins below are quoted as they read at `c3bbf2ed`. If PR 0's run re-pinned any of them to `?v=132` / `?v=8`, the left-hand side differs but the edit is the same: the `?v=` in that assertion becomes this task's number (Step 0).

`dashboard/backend/tests/test_frontend_fast_boot.py`: line 194 `"app.js?v=131"` → `"app.js?v=133"`; line 196 `"styles.css?v=140"` → `"styles.css?v=141"`; delete line 202 (`assert "js/admin-analytics.js?v=6" in APP_HTML`).

`dashboard/backend/tests/test_credit_format_frontend.py`: delete line 79 (`'src="js/admin-analytics.js?v=6"',`) from the `for asset in (...)` tuple.

`dashboard/backend/tests/test_analytics_frontend.py` line 44: `app.js?v=131` → `app.js?v=133`.

`dashboard/backend/tests/test_backtest_comparison_frontend.py` lines 194-195: `app.js?v=131` → `app.js?v=133`, `styles.css?v=140` → `styles.css?v=141`.

Confirm no pin was missed: `rg -n 'app\.js\?v=|admin-tabs\.js\?v=|styles\.css\?v=' dashboard/backend/tests/` must show only `133` / `9` / `141` (plus Task 4's and Task 10's `?v=1` / `?v=\d+` on the new page).

Delete `dashboard/backend/tests/test_admin_analytics_value_frontend.py` (`git rm`): every assertion in it pins `js/admin-analytics-value.js` or the `#adminAnalyticsValueOverview` markup; its two transferable pins (sensitive fields absent from the client; `Intl` formatting) are re-pinned against the new modules in Task 10.

- [ ] **Step 2: Run them and confirm the expected failures**

```bash
python -m pytest dashboard/backend/tests/test_admin_tabs_redirect.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_backtest_comparison_frontend.py -v
```

Expected: **FAIL**. `test_entering_the_admin_view_never_navigates` fails with `nav == [["replace", "/admin-analytics"]]`; `test_default_tab_is_users_and_analytics_is_not_a_tab` fails at `DEFAULT_TAB = 'users'`; `test_admin_user_query_prefills_account_management` fails with `submitted == []`; `test_in_app_analytics_surface_is_gone` fails at `id="adminTabAnalytics"`; `test_admin_rail_has_three_tabs_defaulting_to_users` fails with count 4; `test_profile_menu_admin_entry_opens_the_admin_page` fails (`navigateToPage('admin')` present); every cache-version assertion fails on the old numbers; `test_analytics_style_families_left_styles_css` fails at `.admin-analytics-`.

- [ ] **Step 3: Rewrite `js/admin-tabs.js`**

Replace the file in full:

```js
/** Admin console tab state. No permissions or data access live here. */
(function () {
  'use strict';

  // Analytics lives on the standalone /admin page (design §7.5). This console
  // keeps Users (account management + grant pool), Providers and Activity until
  // the follow-up port (D5). Nothing here navigates away from /app any more.
  const DEFAULT_TAB = 'users';
  const ALLOWED_TABS = new Set(['users', 'providers', 'activity']);
  let initialized = false;

  function normalizeTab(value) {
    const normalizedValue = value === 'grant-pool' ? 'users' : value;
    return ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
  }

  function setTab(value, { updateUrl = true } = {}) {
    const tab = normalizeTab(value);
    const tablist = document.getElementById('adminTabs');
    tablist?.querySelectorAll('[data-admin-tab]').forEach((button) => {
      const selected = button.dataset.adminTab === tab;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-admin-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.adminPanel !== tab;
    });
    if (updateUrl && window.history?.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.set('adminTab', tab);
      window.history.replaceState(window.history.state, '', url);
    }
    document.dispatchEvent(new CustomEvent('admin:tabchange', { detail: { tab } }));
    return tab;
  }

  function bind() {
    if (initialized) return;
    initialized = true;
    const tablist = document.getElementById('adminTabs');
    tablist?.querySelectorAll('[data-admin-tab]').forEach((button) => {
      button.addEventListener('click', () => setTab(button.dataset.adminTab));
      button.addEventListener('keydown', (event) => {
        const keys = new Set(['ArrowDown', 'ArrowUp', 'Home', 'End']);
        if (!keys.has(event.key)) return;
        event.preventDefault();
        const buttons = [...tablist.querySelectorAll('[data-admin-tab]')];
        const index = buttons.indexOf(button);
        const next = event.key === 'Home'
          ? buttons[0]
          : event.key === 'End'
            ? buttons[buttons.length - 1]
            : event.key === 'ArrowDown'
              ? buttons[(index + 1) % buttons.length]
              : buttons[(index - 1 + buttons.length) % buttons.length];
        next.focus();
        setTab(next.dataset.adminTab);
      });
    });
    window.addEventListener('popstate', () => {
      const requested = new URL(window.location.href).searchParams.get('adminTab');
      setTab(requested || DEFAULT_TAB, { updateUrl: false });
    });
  }

  function openAccountManagement({ userId, email } = {}) {
    setTab('users');
    const url = new URL(window.location.href);
    url.searchParams.delete('adminUserQuery');
    window.history.replaceState(window.history.state, '', url);
    const input = document.getElementById('adminCreditsUserQuery');
    const form = document.getElementById('adminCreditsUserSearch');
    if (input) input.value = String(email || userId || '');
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    input?.focus();
  }

  function onEnter() {
    bind();
    const params = new URL(window.location.href).searchParams;
    setTab(params.get('adminTab') || DEFAULT_TAB);
    // The /admin profile deep-links here with the account to look up, the
    // hand-off its in-process predecessor (the evidence dialog) used to make.
    const query = params.get('adminUserQuery');
    if (query) openAccountManagement({ email: query });
  }

  // Page load only initialises panel state; app.js routes `?view=admin` to
  // navigateToPage('admin') → onEnter.
  function init() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    setTab(requested || DEFAULT_TAB);
  }

  window.AdminTabs = { onEnter, openAccountManagement, setTab };
  document.addEventListener('DOMContentLoaded', init);
})();
```

The `leave` option, the `analyticsUser`/`analyticsProfile`/`analyticsSection` deletions in `openAccountManagement`, and PR 0's boolean return from `onEnter` all go: each existed for the in-app analytics panel or its redirect, neither of which survives this task.

- [ ] **Step 4: Edit `app.html`**

Line 16: `<link rel="stylesheet" href="styles.css?v=140">` → `<link rel="stylesheet" href="styles.css?v=141">`.

Lines 2140-2145 — the rail. Delete the `adminTabAnalytics` button (line 2141) and make Users the selected default; the block becomes:

```html
        <nav id="adminTabs" class="admin-tabs admin-rail" aria-label="Admin sections" role="tablist" aria-orientation="vertical">
            <button id="adminTabUsers" class="admin-tab is-active" type="button" role="tab" aria-label="Users" title="Users" aria-selected="true" aria-controls="adminPanelUsers" tabindex="0" data-admin-tab="users"><svg aria-hidden="true"><use href="#icon-users"/></svg><span>Users</span></button>
            <button id="adminTabProviders" class="admin-tab" type="button" role="tab" aria-label="Providers" title="Providers" aria-selected="false" aria-controls="adminPanelProviders" tabindex="-1" data-admin-tab="providers"><svg aria-hidden="true"><use href="#icon-network"/></svg><span>Providers</span></button>
            <button id="adminTabActivity" class="admin-tab" type="button" role="tab" aria-label="Activity" title="Activity" aria-selected="false" aria-controls="adminPanelActivity" tabindex="-1" data-admin-tab="activity"><svg aria-hidden="true"><use href="#icon-activity"/></svg><span>Activity</span></button>
        </nav>
```

Lines 2149-2426: delete from `<section id="adminPanelAnalytics" class="admin-tab-panel" role="tabpanel" aria-labelledby="adminTabAnalytics" data-admin-panel="analytics">` through the `</section>` that closes it on line 2426 (the one immediately after `</article>` closing `#adminAnalyticsProfile`). Confirm the span first with `awk 'NR==2149 || NR==2426' dashboard/frontend/app.html` — the two lines must be the opening `<section id="adminPanelAnalytics"` and a bare `        </section>`. Then on the next remaining line (`<section id="adminPanelUsers" ... data-admin-panel="users" hidden>`) remove the trailing ` hidden` so the default tab's panel is visible before `admin-tabs.js` runs:

```html
        <section id="adminPanelUsers" class="admin-tab-panel" role="tabpanel" aria-labelledby="adminTabUsers" data-admin-panel="users">
```

Delete the two dialogs that follow `#adminView`'s closing `</div>`: from `<dialog id="adminAnalyticsRulesDialog" class="admin-value-dialog" aria-labelledby="adminAnalyticsRulesTitle">` through the `</dialog>` that closes `<dialog id="adminAnalyticsEvidenceDialog" ...>` (the line before the blank line preceding `<dialog id="adminGrantReasonDialog"`).

Script tags (the left-hand values are the post-PR-0 baseline Step 0 confirmed — `?v=131` / `?v=7` at `c3bbf2ed`, bumped once by PR 0): `<script src="app.js?v=132" defer></script>` → `<script src="app.js?v=133" defer></script>`; delete `<script src="js/admin-analytics.js?v=6" defer></script>` and `<script src="js/admin-analytics-value.js?v=5" defer></script>`; `<script src="js/admin-tabs.js?v=8" defer></script>` → `<script src="js/admin-tabs.js?v=9" defer></script>`. Edit the tag by its `src` path, not by a find-and-replace on the full string, so a baseline that differs from the expected one cannot leave the tag untouched.

Verify: `rg -n "adminAnalytics|admin-analytics|AdminAnalytics" dashboard/frontend/app.html` returns nothing, and `rg -n 'app\.js\?v=|admin-tabs\.js\?v=|styles\.css\?v=' dashboard/frontend/app.html` shows exactly `styles.css?v=141`, `app.js?v=133`, `js/admin-tabs.js?v=9`.

- [ ] **Step 5: Edit `app.js`**

Delete lines 5037-5043:

```js
  if (window.AdminAnalytics) {
    window.AdminAnalytics.syncAuth(user);
  }

  if (window.AdminAnalyticsValue) {
    window.AdminAnalyticsValue.syncAuth(user);
  }
```

Change lines 5405-5408 from

```js
  document.getElementById('accountMenuAdminBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    navigateToPage('admin');
  });
```

to

```js
  document.getElementById('accountMenuAdminBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    // Profile → Admin lands on the standalone console (design D3, D4). The old
    // in-app console stays reachable at ?view=admin for account management,
    // providers and the grant audit trail until the follow-up port (D5).
    window.location.assign('/admin');
  });
```

In the `adminRefreshBtn` handler (lines 5409-5420), delete

```js
    if (window.AdminAnalytics) {
      window.AdminAnalytics.refresh();
    }
```

In `navigateToPage`'s `page === 'admin'` branch (lines 10839-10860), delete the two blocks

```js
            if (window.AdminAnalytics) {
                window.AdminAnalytics.onEnter();
            }
            if (window.AdminAnalyticsValue) {
                window.AdminAnalyticsValue.onEnter();
            }
```

PR 0 (its Task 4) changed the `AdminTabs` block just above them to return early when a redirect was scheduled; whichever form the checkout shows, the branch must end up reading:

```js
        } else if (page === 'admin') {
            currentMode = 'admin';
            if (adminView) adminView.style.display = 'block';
            // Stats load on entry and on explicit refresh — not on every
            // pager click, which only changes the user page.
            loadAdminStats();
            loadAdminUsers();
            if (window.AdminTabs) {
                window.AdminTabs.onEnter();
            }
            if (window.AdminModelProviders) {
                window.AdminModelProviders.onEnter();
            }
            if (window.AdminCredits) {
                window.AdminCredits.onEnter();
            }
        }
```

Verify: `rg -n "AdminAnalytics" dashboard/frontend/app.js` returns nothing.

- [ ] **Step 6: Prune `styles.css`**

Delete the block that begins at line 2785 with `/* —— Admin user-value analytics —— */` and ends at line 3660 with the `}` closing the `@media (max-width: 600px)` block whose last rule is `.admin-value-dialog-actions { display: grid; }` — the line after it is the comment `/* Final Backtest feed overrides stay scoped ...`. One rule inside that block is not analytics and must survive; re-add it in place of the deleted block:

```css
.admin-rail button {
    touch-action: manipulation;
    -webkit-tap-highlight-color: rgba(103, 232, 249, 0.12);
}
```

Delete the block that begins at line 14759 with `/* —— Admin Analytics: dense, read-only operator intelligence —— */` and ends at line 15617 with the `}` closing the `@media (prefers-reduced-motion: reduce)` block whose rules are `.admin-analytics-funnel li::before, .admin-analytics-section, .admin-analytics-profile-tabs button { transition: none; }` — the line after it is the `/* --- First-loop onboarding checklist (My Agents)` comment. One selector inside it belongs to the surviving rail; re-add it in place of the deleted block:

```css
.admin-tab:focus-visible {
    outline: 2px solid #67e8f9;
    outline-offset: 2px;
}
```

Verify both edits with:

```bash
rg -c "admin-analytics|admin-value-|admin-priority-|admin-lifecycle-|admin-group-|admin-help-btn|admin-profile-evidence|admin-commercial-tier" dashboard/frontend/styles.css
rg -n "^\.admin-workspace|^\.admin-rail|^\.admin-tab:focus-visible" dashboard/frontend/styles.css
```

The first must print `0`; the second must list `.admin-rail button`, `.admin-workspace`, `.admin-rail` (inside the 900px media block at the old line 14235) and `.admin-tab:focus-visible`.

- [ ] **Step 7: Delete the mock and the two wired modules**

```bash
git rm dashboard/frontend/admin-analytics.html dashboard/frontend/js/admin-analytics.js dashboard/frontend/js/admin-analytics-value.js dashboard/backend/tests/test_admin_analytics_value_frontend.py
```

Then `rg -n "admin-analytics" dashboard/ --glob '!dashboard/backend/tests/test_middleware_exemptions.py' --glob '!dashboard/backend/tests/test_app_composition.py' --glob '!dashboard/backend/tests/test_vercel_cache_headers.py' --glob '!dashboard/backend/app.py' --glob '!dashboard/backend/middleware.py' --glob '!dashboard/backend/tests/test_admin_tabs_redirect.py' --glob '!dashboard/backend/tests/test_admin_analytics_frontend.py'` must return nothing: the only remaining mentions of the string are the 308 redirect route, its exemption, and the tests that pin them.

- [ ] **Step 8: Run the frontend suite and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_tabs_redirect.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_console_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_backtest_comparison_frontend.py dashboard/backend/tests/test_admin_page_shell.py -v
```

Expected: **PASS**. `test_admin_console_frontend.py` is included because it reads the admin rail markup too; it asserts on Users/Providers/Activity ids only (`rg -n "analytics" dashboard/backend/tests/test_admin_console_frontend.py` is empty), so it passes unchanged.

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/js/admin-tabs.js dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_tabs_redirect.py dashboard/backend/tests/test_admin_analytics_frontend.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_credit_format_frontend.py dashboard/backend/tests/test_analytics_frontend.py dashboard/backend/tests/test_backtest_comparison_frontend.py
git commit -m "$(cat <<'EOF'
refactor: retire the in-app analytics panel and the admin-analytics mock

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

(The `git rm` in Step 7 already staged the four deletions.)

---

### Task 10: Cross-module discipline pins for the four `/admin` modules

**Files:**
- Create: `dashboard/backend/tests/test_admin_page_modules.py`.
- Test: `dashboard/backend/tests/test_admin_page_modules.py`.

**Interfaces:**
- Consumes: the four module sources, `admin.html`, `dashboard/backend/tests/_frontend_source.py::fn_body`.
- Produces: the re-pinned §7.4 guards (exact endpoints and query names; read-only, `textContent`-only rendering; no prohibited field names; no `localStorage`) plus the §13 "Must not" pins in code form (no `#live` route, no Chart.js, no client-side group badge) and the D15 pin (no display field read). These are static text assertions — the behaviour is covered by Tasks 5–8's node tests; this file exists so a regression in any of them is a red test, not a review comment.

- [ ] **Step 1: Write the tests (they pass against Tasks 5–8's modules; each one fails if the property it pins is broken — Step 2 proves that)**

Create `dashboard/backend/tests/test_admin_page_modules.py`:

```python
"""Source-shape pins shared by the four /admin modules (design §7.4, §7.6, §13 row C).

Harvested from test_admin_analytics_frontend.py's `test_client_uses_exact_pr2_
endpoints_and_query_names` and `test_analytics_is_read_only_and_uses_safe_dom_
rendering`, re-targeted at js/admin-*.js, plus the pins the design's "Must not"
column for PR C asks for: no #live route, no Chart.js, no client-side group
badge, and none of the D15 display fields read.
"""

import re
from pathlib import Path

from dashboard.backend.tests._frontend_source import fn_body

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
NAMES = ("admin-shell.js", "admin-live.js", "admin-overview.js", "admin-users.js")
MODULES = {name: (FRONTEND / "js" / name).read_text(encoding="utf-8") for name in NAMES}
ALL = "\n".join(MODULES.values())
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
GLOBALS = {
    "admin-shell.js": "AdminShell",
    "admin-live.js": "AdminLive",
    "admin-overview.js": "AdminOverview",
    "admin-users.js": "AdminUsers",
}


def test_each_module_is_an_iife_exposing_exactly_one_global():
    for name, source in MODULES.items():
        assert source.lstrip().startswith("/**"), name
        assert "(function () {\n  'use strict';" in source, name
        assigned = set(re.findall(r"^\s*window\.(\w+) = ", source, re.M))
        assert assigned == {GLOBALS[name]}, (name, assigned)


def test_every_module_admin_html_loads_exists_and_nothing_else_is_loaded():
    srcs = re.findall(r'<script src="js/([^?"]+)\?v=\d+" defer></script>', ADMIN_HTML)
    assert set(srcs) == set(NAMES) | {"credit-format.js"}
    for name in srcs:
        assert (FRONTEND / "js" / name).exists(), name


def test_rendering_is_text_content_only():
    for name, source in MODULES.items():
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function("):
            assert forbidden not in source, (name, forbidden)
        assert "textContent" in source, name


def test_every_request_is_a_credentialed_get_made_by_the_shell():
    assert set(re.findall(r"method:\s*'(\w+)'", ALL)) == {"GET"}
    for name, source in MODULES.items():
        if name == "admin-shell.js":
            assert source.count("fetch(") == 2  # request() and gate()
            assert "credentials: 'include'" in source
        else:
            assert "fetch(" not in source, name
            assert "XMLHttpRequest" not in source, name
    for verb in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{verb}'" not in ALL


def test_exact_endpoints_and_query_names():
    assert "/api/auth/me" in MODULES["admin-shell.js"]
    assert "/api/admin/stats" in MODULES["admin-live.js"]
    for endpoint in (
        "/api/admin/analytics/overview", "/api/admin/analytics/lifecycle", "/api/admin/analytics/retention",
        "/api/admin/analytics/commercial", "/api/admin/analytics/operational", "/api/admin/analytics/groups",
    ):
        assert endpoint in MODULES["admin-overview.js"], endpoint
    assert "/api/admin/analytics/users" in MODULES["admin-users.js"]
    assert "/activity?" in MODULES["admin-users.js"]
    for query_name in (
        "'from'", "'to'", "'include_internal'", "'user_group'", "'lifecycle_segment'",
        "'commercial_tier'", "'q'", "'priority'", "'limit'", "'offset'", "cursor",
    ):
        assert query_name in ALL, query_name
    assert "section, limit:" in MODULES["admin-users.js"]
    for rejected in ("'status'", "start_date", "provider_id'", "model_id'", "'cohort'", "analyticsUser", "adminTab=analytics"):
        assert rejected not in ALL, rejected


def test_no_live_route_and_no_chartjs():
    assert "#live" not in ALL and "#live" not in ADMIN_HTML
    routes_line = re.search(r"const ROUTES = \[([^\]]+)\];", MODULES["admin-shell.js"]).group(1)
    assert "'live'" not in routes_line
    assert "'usage'" not in routes_line and "'revenue'" not in routes_line and "'profiles'" not in routes_line
    for forbidden in ("window.Chart", "new Chart(", "chart.js", "cdn.jsdelivr.net"):
        assert forbidden not in ALL, forbidden
        assert forbidden not in ADMIN_HTML, forbidden
    assert "createElementNS" in MODULES["admin-overview.js"]  # the revenue line is generated SVG


def test_group_badge_is_never_computed_client_side():
    """D11: `group_badge` is rendered verbatim; the precedence rule has one owner, the server."""
    assert "group_badge" in MODULES["admin-users.js"]
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        source = MODULES[name]
        for forbidden in ("role === 'admin'", "=== 'unpaid'", "'paid'", "'free'", "user_group ===", "!== 'unknown'"):
            assert forbidden not in source, (name, forbidden)
    # The shell's gate is the one place `role` is compared, and only against 'admin' for the redirect.
    assert MODULES["admin-shell.js"].count("user.role !== 'admin'") == 1


def test_d15_display_fields_are_not_read():
    for field in ("country_code", "device_category", "browser_family", "top_product_page"):
        assert field not in ALL, field


def test_prohibited_field_names_and_local_storage_are_absent():
    for prohibited in (
        "api_key", "session_id", "network_hash", "provider_response_body",
        "credential_ciphertext", "prompt", "strategy", "portfolio", "password", "raw_user_agent",
    ):
        assert prohibited not in ALL, prohibited
    assert "localStorage" not in ALL
    assert "sessionStorage" not in ALL


def test_formatting_goes_through_intl_and_the_shared_credit_formatter():
    shell = MODULES["admin-shell.js"]
    assert "window.CreditFormat.formatCreditsMicro(value)" in shell
    assert "Intl.NumberFormat" in shell and "Intl.DateTimeFormat" in shell
    for name in ("admin-live.js", "admin-overview.js", "admin-users.js"):
        assert "formatCreditsMicro" not in MODULES[name], name  # only via AdminShell.formatCredits
        assert ".toFixed(6)" not in MODULES[name], name


def test_renderers_can_be_lifted_with_fn_body():
    """§7.6: every renderer is a named function the shared harness can slice."""
    for signature in (
        "function renderAttention(", "function renderActiveUsers(", "function renderActivation(",
        "function renderSources(", "function renderRetention(", "function renderValue(",
        "function renderLifecycle(", "function renderCredits(", "function renderRevenue(",
        "function detailSources(", "function detailRetention(", "function detailCredits(",
        "function detailLifecycle(", "function detailHealth(",
    ):
        body = fn_body(signature, MODULES["admin-overview.js"])
        assert body.startswith(signature) and body.endswith("}"), signature
        assert "innerHTML" not in body
    for signature in (
        "function renderUserRows(", "function renderPager(", "function renderEvidence(",
        "function renderProfileHeader(", "function renderProfileOverview(", "function renderActivityItems(",
    ):
        body = fn_body(signature, MODULES["admin-users.js"])
        assert body.startswith(signature) and body.endswith("}"), signature
    body = fn_body("function renderTiles(", MODULES["admin-live.js"])
    assert "max_active_dashboard_backtests" in body
```

- [ ] **Step 2: Run it, then prove each pin bites**

```bash
python -m pytest dashboard/backend/tests/test_admin_page_modules.py -v
```

Expected: **PASS** on all eleven against Tasks 5–8's modules. Then, without committing, make each of these three one-line edits in turn, re-run, and revert (`git checkout -- dashboard/frontend/js/<file>`):

1. In `admin-users.js`, append the comment `// role === 'admin'` to the `groupBadge` return line. Expected: `test_group_badge_is_never_computed_client_side` **FAIL** at `("admin-users.js", "role === 'admin'")`.
2. In `admin-overview.js`, append `// window.Chart` anywhere. Expected: `test_no_live_route_and_no_chartjs` **FAIL**.
3. In `admin-live.js`, change `updated.textContent = ...` to `updated.innerHTML = ...`. Expected: `test_rendering_is_text_content_only` **FAIL**, and Task 6's `test_load_paints_the_row_and_marks_the_update_time` **FAIL** too (the node stub has no `innerHTML`, so the assignment leaves `textContent` empty).

Revert all three; re-run the file and confirm **PASS**.

- [ ] **Step 3: Run the whole suite once**

```bash
python -m pytest dashboard/backend/tests/ -q
```

Expected: **PASS** end to end (the suite is green on a fresh run per `CLAUDE.md`; every test this plan touched is listed in the tasks above, and no test outside them references the deleted files — confirmed in Task 9 Step 7's `rg`).

- [ ] **Step 4: Commit**

```bash
git add dashboard/backend/tests/test_admin_page_modules.py
git commit -m "$(cat <<'EOF'
test: pin the /admin modules' read-only, single-source discipline

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Not in this PR

Per design §13 (PR C's "Must not" column) and the decisions it cites:

- **Porting Users (account management and the grant pool), Providers or Activity into `/admin`** — D5: those three tabs have their own routers, stores, pagination and tests, none of them analytics, and would double this PR for no analytics benefit. The aside links to `/app?view=admin&adminTab=users|providers|activity` (§7.5); full absorption is the committed end state and a named follow-up (§14).
- **Computing the group badge client-side** — D11: the server owns the precedence rule (`resolve_group_badge`, PR B); the client renders `group_badge` verbatim. Two owners of one rule is how the season-number badge and its banner came to disagree. Task 10 pins it.
- **Inline `<script>` in `admin.html`** — D6: inline scripts are invisible to the only frontend test harness this repository has. Task 4's guard fails on any inline block.
- **A `#live` route or a live detail page** — §8.2 (the eight run lanes, queue-pressure chart and "Current blockers" are cut: the lanes carry no information beyond the count, there is no queue, and blockers are the daily operational reasons shown in `#health`), D16. Task 10 pins that `ROUTES` has no `live`.
- **Chart.js on the page** — §7.2: charts stay as the mock draws them (CSS bars and generated SVG) so every renderer is a pure function testable under `node`.
- **Deciding whether to keep collecting page, country and device** — D15/§14: those event properties stay in the payload (Task 1's test asserts they still arrive) and leave the *display*; the collection decision belongs to the 08-26 document's allowlist and is a follow-up.
- **A server-side gate for `admin.html`** — D7: Vercel serves the same file as a static asset with no session access; a server gate would exist on one host only. The real gate is `require_admin` on every `/api/admin/*` route, which this PR does not change.
- **Removing `GET /admin-analytics` or its middleware exemption** — §7.1: the 308 lives for one release; dropping the exact-match exemption early would 400 the redirect before it fires. Both go together in a later PR, with `("GET", "/admin-analytics")` leaving `EXPECTED_FULL_CONTRACT` then.
- **An "online now" tile, a per-run list for the live row, or multi-replica live counters** — §14 follow-ups; each needs a data source or a deployment change this PR does not make.
- **Any change to the nine routes' semantics beyond the §9 fields and the declared parameters** — D18/D19: the contract is re-cut once, here, and `test_app_composition.py`'s route triples are unchanged.
- **Rejecting undeclared or duplicated query keys** — the hand-rolled parser's behaviour, not the API's; with declared parameters FastAPI ignores unknown keys and reads the last of a repeated one (§6.14 row 4). Task 1's `test_undeclared_query_keys_are_ignored_not_rejected` pins the new behaviour deliberately.

## Acceptance

- [ ] `AnalyticsOverview.billing_lane_mix`, `OperationalAnalyticsResponse.top_operational_reasons`, `CommercialAnalyticsResponse.purchased_by_day` / `consumed_by_day`, and `user_group` / `role` / `group_badge` / `last_meaningful_activity_at` on `ValueUserListItem` and `ValueUserProfile` are on the response models, populated from PR B's service methods, and present in every committed fixture (Task 1).
- [ ] Every `/api/admin/analytics/*` query parameter is a declared `Query(...)` visible in `app.openapi()` under today's names (plus `role`), `_query_values` and `_invalid_query` are gone, and every validation failure is still the display-safe `{"detail": "Invalid Analytics query."}` with no echo (Task 1).
- [ ] `GET /api/admin/stats` carries `max_active_dashboard_backtests` equal to the parsed constant and keeps its six existing keys (Task 2).
- [ ] `GET /admin` serves `admin.html`, `GET /admin.css` serves the stylesheet, `GET /admin-analytics` answers 308 to `/admin` preserving the query string; `is_exempt("/admin")` and `is_exempt("/admin-analytics")` are both true and pinned; `vercel.json` has no `/admin`-prefixed rewrite, a permanent `/admin-analytics` → `/admin` redirect, and `/admin` + `/admin.html` must-revalidate rules ordered after the API no-store rule (Task 3).
- [ ] `admin.html` has five external, pinned, deferred scripts with `admin-shell.js` first, no inline script, no Chart.js, no `styles.css`, and no numeric or percentage literal inside any of its thirteen panel regions (Task 4).
- [ ] `admin.css` carries the mock's surviving styles and none of the dead funnel/live-detail passes (Task 4).
- [ ] `AdminShell` routes exactly `#overview #sources #retention #credits #lifecycle #health #users #users/{id}` (unknown and `#live` → `#overview`), keeps range and filters in the URL, guards every surface with `requestSeq`, exits to `/app` on 401/403 and on a non-admin `/api/auth/me`, and owns the rules and evidence dialogs with return-focus (Task 5).
- [ ] The live row reads `/api/admin/stats` only and shows Total users, Total agents, Backtests running · this instance with the real ceiling and the *this instance* caveat; a failed refresh keeps the last good row and says so (Task 6).
- [ ] The nine overview panels and five detail routes render from the six analytics endpoints as §8.2 sources them, "% remain" is computed from the counts, one endpoint's failure blanks only the panels that need it, and partial availability reads "Incomplete data" (Task 7).
- [ ] The users list is SQL-filtered through the declared parameters and shows user · group badge (verbatim) · lifecycle · operational · last active; the profile has Overview / Timeline / Runs / Usage / Sessions with lazy, cursor-paged tabs; Region / Device / Browser / top page are not rendered; "Open account management" deep-links with `adminUserQuery` (Task 8).
- [ ] `admin-tabs.js` has no redirect and no Analytics tab, `DEFAULT_TAB` is `users`, and `adminUserQuery` pre-fills the account search; `app.html` has no analytics panel, legacy block, profile article, analytics dialog or analytics script tag; `app.js` calls neither `AdminAnalytics` nor `AdminAnalyticsValue` and Profile → Admin opens `/admin`; `styles.css` has none of the analytics rule families; `admin-analytics.html`, `js/admin-analytics.js`, `js/admin-analytics-value.js` and `test_admin_analytics_value_frontend.py` are deleted; every `?v=` pin matches (Task 9).
- [ ] The four modules are read-only, `textContent`-only, single-global IIFEs that hit exactly the named endpoints with exactly the declared query names, never compute a badge, never read a D15 field, never touch storage, and every renderer lifts with `fn_body` (Task 10).
- [ ] No element §8.2 marks **cut** exists on the page; every element built appears there as keep, keep-relabelled or added.
- [ ] `python -m pytest dashboard/backend/tests/ -q` is green, and CI's Postgres tier is green before merge.

## Self-review notes

- **Spec coverage.** §7.1 → Task 3 (routes, redirect, exemption, `vercel.json`) and Task 4 (the page). §7.2 → Tasks 5–8 (the four modules, one global each, `defer` + `?v=1`, the ownership table's panel→module→route mapping reproduced in Task 7's `PANELS` and `DETAIL_NEEDS` and Task 8's `#users`/`#users/{id}`), plus `admin.css` in Task 4. §7.3 → Task 5 `gate()` with the D7 comment beside it. §7.4 → each row's destination is named in Task 5's preamble (requestSeq, URL state, dialogs, availability, access-lost, rules copy), Task 8's preamble (lazy disclosures, cursor paging), and Task 10 (the two re-pinned tests). §7.5 → Task 9 (every removed item listed, every kept item untouched, `DEFAULT_TAB = 'users'`, Profile → Admin). §7.6 → Task 4 (source-shape guard, including the numeric-literal regex and the `data-panel` region selector), Task 9 (`test_admin_tabs_redirect.py` rewrite, `test_app_composition.py` already in Task 3), Task 3 (`test_middleware_exemptions.py`, `test_vercel_cache_headers.py`), Tasks 5–8 (node-driven fixture tests). §8.1 → Task 4's element list is derived from the *final* DOM (V8 §2/§9), not the markup. §8.2 → the survival check below. §8.3 → Task 4 (Total users, Total agents, real ceiling, *this instance* label). §9 → Task 1 (added fields on models), Task 2 (`max_active_dashboard_backtests`), Task 8 (D15 fields dropped from display, kept in payload — pinned by Task 1's `test_recut_fields_are_on_every_response`). §10.3 → Task 1 (declared params, `_query_values` deleted, names kept, fixtures regenerated); §10.4 → Task 2. §13 row C → every item in its "Content" column maps to a task above; its "Must not" column is the "Not in this PR" list and Task 10's pins. D3/D4 → Task 3 + Task 9 (Profile → Admin → `/admin`). D5 → aside links (Task 4), nothing ported. D6 → Task 4 guard. D7 → Task 5 gate + Task 3 docstring. D8 → harvest recorded per task; deletion in Task 9. D12 → Task 4's row order (live row first, attention beside it, then activity, activation, sources, retention, value, lifecycle, credits, revenue). D15 → Task 8 + Task 10. D16/D17 → Task 6 (stats route only, caveat), Task 2. D18/D19 → Task 1 is the single re-cut. §15 → Task 5's `LIFECYCLE_RULES`/`OPERATIONAL_RULES`/`COMMERCIAL_RULE` and Task 8's badge tooltips.
- **Survival-table check (§8.2), element by element.** Built as **keep**: header, product nav, aside with Analytics subnav and the three console links; range 1W/1M/1Y; Source, Lifecycle stage, Tier (relabelled from Paid/Unpaid, four values), Include internal accounts (relabelled); live row Backtests running · this instance with the real ceiling; Users needing attention with its blocking-reason note from `top_operational_reasons`; Active users; Activation progress with client-side "% remain"; Where users come from (six groups); Users coming back; Users reaching value; User lifecycle; Credits usage (ledger total + `billing_lane_mix` lanes); Revenue (ledger total + `purchased_by_day` line) with the "Admin Grants excluded from revenue" copy; `#sources` six-row table; `#retention`, `#lifecycle` (with the §15 rules table), `#credits` details; `#health` with failed runs, success rate, failure-category table and the operational-reasons table; `#users` list (user · group badge · lifecycle · operational · last active) and `#users/{id}` profile with badges, blocker, milestones and tabs; profile identity line as group badge + activation week (**replaced** row); rules dialog. Built as **added**: Total users, Total agents, the real slot ceiling, the *this instance* caveat, the freshness legend. **Not built (cut):** Sample-data label and footer notice, 1D, Cohort filter, Online now, Queued, Blocked-users live tile, the `#live` detail (lanes, queue chart, blockers), the hidden `.inline-key` legend, the `#health` "Affected users" column, Region/Device/Browser/top page on the profile, the `#usage`/`#revenue` orphan routes. Task 4's `test_cut_elements_are_absent` and Task 10's pins enforce the cut list; nothing on the page lacks a §8.2 row. One labelling choice outside the table: the aside's link to the old console's Users tab reads "Account management" so the page has one "Users" (the analytics list at `#users`) rather than two entries with one name — V8 §7 item 11 recorded exactly that duplication in the mock.
- **Placeholder scan.** No `TBD`, `TODO`, "similar to Task N", "as in the old plan", or elided body: every code block is the complete file or the complete before/after hunk, written against the worktree at `c3bbf2ed` (`git log -1` while writing). The three places this plan cannot quote the merged text — PR 0's exact `app.js` admin-branch shape, PR B's `list_users` construction site after the SQL rewrite, and the four `?v=` test pins PR 0's plan does not re-pin — show the current `main` text, name the post-merge target text in full, and give the `rg` that finds the site. Nothing points at the superseded 09-12 plan as a live file; its one reused artefact (the `resolve_group_badge` signature) is cited as `git show c3bbf2ed:docs/superpowers/plans/2026-09-12-user-analytics-architecture.md` line 2336 in Task 1's Interfaces via design D9/D11.
- **Name consistency across tasks.** `BillingLaneDay`, `OperationalReasonCount`, `LedgerDayTotal`, `InvalidAnalyticsQuery` (Task 1) are referenced only there. `max_active_dashboard_backtests` (Task 2) is read by Task 6's `renderTiles` and pinned by Task 10. The element ids in Task 4's Interfaces are the ones Tasks 5–8 target (`filterGroup`, `liveTiles`, `panelAttention` … `panelRevenue`, `usersBody`, `usersRange`, `usersPrev`, `usersNext`, `profile`, `detail`, `rulesList`, `evidenceBody`, `evidenceProfile`, `evidenceAccount`, `freshnessLegend`). `AdminShell`'s exported names in Task 5 are the ones Tasks 6–8 call (`el`, `clear`, `request`, `nextSeq`, `isCurrent`, `handleAccessLost`, `setPanelState`, `analyticsParams`, `userListParams`, `rangeDates`, `formatNumber`, `formatPercent`, `formatCredits`, `formatDateOnly`, `formatShortDay`, `formatTimestamp`, `humanize`, `availabilityIncomplete`, `openDialog`, `openRules`, `setFilters`, `state`, `INCOMPLETE`, `SECTION_UNAVAILABLE`, `DASH`, `LIFECYCLE_LABELS`, `OPERATIONAL_LABELS`, `COMMERCIAL_LABELS`, `LIFECYCLE_RULES`, `OPERATIONAL_RULES`). The `admin:route` / `admin:retry` events dispatched in Task 5 are the ones Tasks 6–8 listen for, with the same `detail` keys. `adminUserQuery` is written by Task 8's `accountManagementHref` and read by Task 9's `onEnter`. The `?v=` numbers (133 / 9 / 141 / 1) are identical in Task 9's tags and every pin Task 9 lists, and in Task 4's guard; they are `c3bbf2ed`'s 131 / 7 / 140 plus PR 0's one bump on the first two plus this task's one bump on all three, and Task 9 Step 0 checks that arithmetic against the checkout before any of them is written.
- **Verified, not assumed.** The global `RequestValidationError` handler echoes input for non-v2 paths (`api/v2/errors.py:67-85`, registered at `app.py:62`) — the reason Task 1 types every parameter `str | None`. FastAPI/Starlette reads the last value of a repeated query key (`ImmutableMultiDict` construction), the basis of dropping the duplicate-key 422. The API no-store header rule's optional group matches bare `/admin` (regex `(/.*)?`), the basis of Task 3's ordering. `test_admin_console_frontend.py` has no `analytics` assertion (`rg`), so Task 9 leaves it alone. `activated_at = 2026-07-02` is a Thursday and its UTC Monday is 2026-06-29 (Task 8's expectation). `2026-09-15` minus 179 days is `2026-03-20` (Task 5's 1Y expectation).
- **Open items I could not close from code alone** (also returned as open questions): the exact names/signatures of PR B's service methods (Task 1 assumes `billing_lane_mix`, `top_operational_reasons`, `purchased_by_day`, `consumed_by_day`, `resolve_group_badge(*, role, user_group, tier)` and names the one call site for each); whether PR B kept `FailureCategoryCount.affected_users` as the field name once the count comes from rollups (Task 7 labels the column "Count" and reads `affected_users`); whether PR 0's run re-pinned the four `?v=` tests outside `test_admin_analytics_frontend.py` that its plan does not name (Task 9 Step 0 reads them off the checkout; the tags themselves are settled — 131 / 7 / 140 at `c3bbf2ed`, 132 / 8 / 140 after PR 0); and the Vercel header policy for `/admin` (this plan follows the orchestrator's instruction — must-revalidate, ordered after the no-store rule — where design §7.1 had accepted the accidental no-store).
