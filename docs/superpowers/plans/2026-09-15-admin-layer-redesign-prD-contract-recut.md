# Admin Layer PR D: The Contract Re-cut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-cut the nine `/api/admin/analytics/*` routes once (D19): the §9 fields land on the response models, populated from the service methods PR B shipped; the hand-rolled `_query_values()` parser is replaced by declared FastAPI `Query(...)` parameters under today's names plus `role`; the committed fixtures gain the §9 fields and the `/admin` page's target-shape copies are retired — and nothing visible changes on the page except that every "Awaiting data source" slot fills in.

**Architecture:** Backend only, on the existing routers and response models: new Pydantic fields wired to `AnalyticsQueryService.billing_lane_mix`, `ValueAnalyticsQueryService.top_operational_reasons` / `purchased_by_day` / `consumed_by_day` and `lifecycle.resolve_group_badge` (PR B), hand-parsed query strings replaced by declared parameters that keep today's names and today's display-safe 422s. The `/admin` page (PR C) was built against exactly this shape from `tests/fixtures/admin_analytics/target/`; Task 2 folds those copies into the committed fixtures, deletes the directory and the module that pinned the two sets apart, and points the two renderer test modules at the committed files. No JavaScript changes, so no `?v=` bump.

**Tech Stack:** FastAPI + Pydantic v2 (backend), pytest run from the repo root, Node.js via `subprocess` for the two renderer test modules whose fixture calls change (`skipif(shutil.which("node") is None)`).

**Spec:** `docs/superpowers/specs/2026-09-15-admin-layer-redesign-design.md` — §9 (the added fields, this PR's field list), §10.3 (the re-cut rule), §13 row **D** including its "Must not" column, decisions D11, D15, D18, D19 (as amended 2026-09-16). The PR C plan's "Interim contract" section names every slot this PR fills.

## Global Constraints

- Never commit `dashboard/storage/data/backtest.db` — a bare backend import rewrites it (`CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH`); stage files by name (`git add <path> <path>`), never `-A`, never `.`, never a bare `-u`.
- Run every `pytest` invocation from the repo root.
- Ordering (§13, re-ordered 2026-09-16): PR 0, PR C, PR T, PR A and PR B are merged before this branch is cut. Task 1 Step 0 is the hard stop: `user_state_counts` / `legacy_status` still in the domain means PR B has not merged; no `tests/fixtures/admin_analytics/target/` means PR C has not merged.
- This is the **one** re-cut (D19). Fields listed in §9 land; D15 fields stay in the payload (the page already does not display them); parameter names stay exactly `role` (new), `commercial_tier`, `user_group`, `lifecycle_segment`, `operational_state`, `include_internal`, `from`, `to`, `movement_range`, `limit`, `offset`, `cursor`, `section`, `q`, `priority`, `activated`, `last_meaningful_activity_from`, `last_meaningful_activity_to`, `billing_mode`, `provider`, `model`. No other route changes shape; `test_app_composition.py`'s route triples are unchanged.
- The display-safe 422 contract is load-bearing: every validation failure stays `{"detail": "Invalid Analytics query."}` with no echo of the input (`test_admin_analytics_rejects_invalid_queries_without_echo`). Every declared parameter is `str | None = Query(default=None, ...)`; the value rules stay in Python. See the note above Task 1 Step 1.
- No store method, no table, no DDL: this PR reads what PR B shipped. CI's Postgres tier (`ci.yml`, `TEST_POSTGRES_URL`, `@pg_only`) must be green before merge — this is the first HTTP exercise of `CreditsStore.sum_ledger_by_day` / `resolve_group_badge` / `top_operational_reasons`, so a twin PR B left one-sided surfaces here.
- Frontend: **no JavaScript file changes** and no `?v=` bump. The only frontend-side edits are in three test modules (`_admin_dom_stub.py`, `test_admin_overview_frontend.py`, `test_admin_users_frontend.py`) and the fixtures they read.
- The group badge stays server-computed (D11): `resolve_group_badge` is called at the two construction sites named in Task 1; nothing else derives it.

## File Map

Backend:
- `dashboard/backend/domain/analytics/query_service.py`: `BillingLaneDay` model; `AnalyticsOverview.billing_lane_mix`.
- `dashboard/backend/domain/analytics/value_queries.py`: `OperationalReasonCount` (or PR B's, see Task 1 Step 0), `LedgerDayTotal`; `OperationalAnalyticsResponse.top_operational_reasons`; `CommercialAnalyticsResponse.purchased_by_day` / `consumed_by_day`; `UserValueFilters.role`; `ValueUserListItem` and `ValueUserProfile` gain `user_group`, `role`, `group_badge`, `last_meaningful_activity_at`; the two construction sites populate them.
- `dashboard/backend/api/routers/admin_analytics.py`: declared `Query(...)` dependencies replace `_query_values()`/`_invalid_query()`; every validation rule and every 422 body stays byte-identical.

Fixtures and tests:
- Modified: `dashboard/backend/tests/fixtures/admin_analytics/{overview,overview_partial_error,operational,commercial,users,user_detail}.json` (gain the §9 fields — the same hunks PR C applied to the `target/` copies).
- Deleted: `dashboard/backend/tests/fixtures/admin_analytics/target/` (six files), `dashboard/backend/tests/test_admin_target_fixtures.py`.
- Modified: `test_admin_analytics_api.py` (Task 1), `_admin_dom_stub.py`, `test_admin_overview_frontend.py`, `test_admin_users_frontend.py`, `test_admin_analytics_frontend.py::test_fixtures_match_committed_analytics_shapes` (Task 2).

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
- Consumes (PR B service methods, named in design §9/§13 row B; PR B's plan (`2026-09-15-admin-layer-redesign-prB-read-paths-delete.md`, Tasks 1, 4, 5, 8) declares `billing_lane_mix(*, start, end)`, `purchased_by_day(*, start, end)` / `consumed_by_day(*, start, end)`, `top_operational_reasons(*, day, limit=10) -> list[OperationalReasonCount]` and `resolve_group_badge(*, role, user_group, tier)`, which differ from the keyword shapes this task was drafted against below; Step 0 reconciles them — verify each with `rg -n "def billing_lane_mix|def top_operational_reasons|def purchased_by_day|def consumed_by_day|def resolve_group_badge" dashboard/backend/domain` before Step 3 and substitute at the single call site named for each):
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

- [ ] **Step 0: Confirm the checkout and reconcile PR B's names**

```bash
rg -n "user_state_counts|legacy_status" dashboard/backend/domain/analytics dashboard/backend/api/routers/admin_analytics.py
ls dashboard/backend/tests/fixtures/admin_analytics/target/
rg -n "class OperationalReasonCount|class BillingLaneDay|class LedgerDayTotal|def billing_lane_mix|def top_operational_reasons|def purchased_by_day|def consumed_by_day|def resolve_group_badge" dashboard/backend/domain
```

Expected: the first `rg` prints nothing (PR B has merged — if it prints, stop: §13 ordering); `ls` lists six files (PR C has merged — if it errors, stop); the third `rg` prints one `def` per method and, depending on how PR B's Task 8 landed, `class OperationalReasonCount` in `value_queries.py`. Reconcile against what it prints:

| PR B shipped | This task's text assumes | Do |
|---|---|---|
| `billing_lane_mix(*, start: date, end: date)` | `billing_lane_mix(*, filters=...)` | at the one call site in Step 3, pass `start=`/`end=` from the parsed range instead of `filters=` |
| `purchased_by_day(*, start, end)` / `consumed_by_day(*, start, end)` without `include_internal` | with `include_internal` | drop the keyword at the call site |
| `top_operational_reasons(*, day: date, limit=10) -> list[OperationalReasonCount]` | `(*, include_internal, limit=5) -> list[tuple]` | pass `day=yesterday`; **do not** redefine `OperationalReasonCount` — import PR B's and delete the class from Step 3; skip the tuple unpacking |
| `resolve_group_badge(*, role, user_group, tier)` | the same | nothing |

Line numbers quoted below are `main@c3bbf2ed`; PR B moved them. Each step names the `rg` that finds the site.

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
    # only their display leaves the page (PR C's `admin-users.js` renders none of them).
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

### Task 2: Fold the target fixtures into the committed set and retire the interim seam

**Files:**
- Delete: `dashboard/backend/tests/fixtures/admin_analytics/target/overview.json`, `target/overview_partial_error.json`, `target/operational.json`, `target/commercial.json`, `target/users.json`, `target/user_detail.json`; `dashboard/backend/tests/test_admin_target_fixtures.py`.
- Modify: `dashboard/backend/tests/_admin_dom_stub.py` (delete `target_fixture`), `dashboard/backend/tests/test_admin_overview_frontend.py` (the `F` dict), `dashboard/backend/tests/test_admin_users_frontend.py` (`USERS`, `PROFILE`), `dashboard/backend/tests/test_admin_analytics_frontend.py` (`test_fixtures_match_committed_analytics_shapes`).
- Test: `dashboard/backend/tests/test_admin_overview_frontend.py`, `test_admin_users_frontend.py`, `test_admin_analytics_frontend.py`, `test_admin_page_modules.py`.

**Interfaces:**
- Consumes: the committed fixtures as Task 1 Step 5 left them (every §9 field present); PR C's `target_fixture` helper and the two fixture constants that call it.
- Produces: one fixture set. The renderers' pending branches (`AdminShell.fieldPending`, `PENDING`) stay — after this PR an absent §9 field is a contract bug and must read "Awaiting data source", never as an empty chart (fail-visible) — and their tests keep exercising them by deleting keys from the committed payloads, so no expectation in the two renderer modules changes.

- [ ] **Step 1: Confirm the seam has fired**

```bash
python -m pytest dashboard/backend/tests/test_admin_target_fixtures.py -v
```

Expected: **FAIL** — `test_target_fixtures_add_exactly_the_section_9_fields` with `a committed fixture carries a §9 field: PR D has landed, delete target/ and this module`. That assertion exists to fire exactly now; if it **passes**, Task 1 Step 5 did not reach every committed fixture — go back before deleting anything.

- [ ] **Step 2: Delete the target copies and the module, re-point the tests**

```bash
git rm -r dashboard/backend/tests/fixtures/admin_analytics/target
git rm dashboard/backend/tests/test_admin_target_fixtures.py
```

In `dashboard/backend/tests/_admin_dom_stub.py` delete the helper PR C added after `fixture`:

```python
def target_fixture(name: str) -> str:
    """The target-shape copy (today's payload plus the §9 fields; see the PR C plan's
    "Interim contract"). PR D folds these into the committed fixtures and deletes both
    the directory and this helper."""
    return (FIXTURES / "target" / name).read_text(encoding="utf-8")
```

In `dashboard/backend/tests/test_admin_overview_frontend.py` change

```python
from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source, target_fixture
```

to

```python
from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source
```

and

```python
# The four payloads that gain §9 fields come from the target-shape copies (Task 1);
# the absent-field test below deletes those keys again. PR D switches these to `fixture`.
TARGET = ("overview", "overview_partial_error", "operational", "commercial")
F = {name: (target_fixture if name in TARGET else fixture)(f"{name}.json") for name in (
    "overview", "overview_partial_error", "operational", "lifecycle", "retention", "commercial", "groups",
)}
```

to

```python
F = {name: fixture(f"{name}.json") for name in (
    "overview", "overview_partial_error", "operational", "lifecycle", "retention", "commercial", "groups",
)}
```

In `dashboard/backend/tests/test_admin_users_frontend.py` change

```python
from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source, target_fixture
```

to

```python
from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source
```

and

```python
# Target-shape copies (Task 1): today's payloads plus the §9 fields. PR D switches these to `fixture`.
USERS = target_fixture("users.json")
PROFILE = target_fixture("user_detail.json")
```

to

```python
USERS = fixture("users.json")
PROFILE = fixture("user_detail.json")
```

In `dashboard/backend/tests/test_admin_analytics_frontend.py::test_fixtures_match_committed_analytics_shapes` restore the §9 keys PR C's Task 9 left out. Change

```python
    # Today's shapes (D18): the §9 fields live in fixtures/admin_analytics/target/ until PR D
    # folds them in and restores them to these assertions.
    assert {"daily_active_users", "availability", "last_updated"} <= overview.keys()
```

to

```python
    assert {"daily_active_users", "billing_lane_mix", "availability", "last_updated"} <= overview.keys()
```

and

```python
    assert {"tier_counts", "selected_period", "current_balances"} <= commercial.keys()
    assert {"operational_state_counts", "top_failure_categories"} <= operational.keys()
    assert {"items", "total", "limit", "offset"} == users.keys()
    assert {"lifecycle", "operational", "commercial_tier", "priority_group"} <= users["items"][0].keys()
    assert {"state", "activation_milestones", "lifecycle", "operational", "commercial"} <= profile.keys()
```

to

```python
    assert {"tier_counts", "selected_period", "current_balances", "purchased_by_day", "consumed_by_day"} <= commercial.keys()
    assert {"operational_state_counts", "top_failure_categories", "top_operational_reasons"} <= operational.keys()
    assert {"items", "total", "limit", "offset"} == users.keys()
    assert {"user_group", "role", "group_badge", "last_meaningful_activity_at"} <= users["items"][0].keys()
    assert {"state", "activation_milestones", "lifecycle", "operational", "commercial", "group_badge"} <= profile.keys()
```

The `rglob` loop in `test_safe_fixtures_have_no_prohibited_response_fields` stays as PR C wrote it: with `target/` gone it walks the committed files only.

- [ ] **Step 3: Run the page's tests and confirm the pass**

```bash
python -m pytest dashboard/backend/tests/test_admin_overview_frontend.py dashboard/backend/tests/test_admin_users_frontend.py dashboard/backend/tests/test_admin_shell_frontend.py dashboard/backend/tests/test_admin_live_frontend.py dashboard/backend/tests/test_admin_page_modules.py dashboard/backend/tests/test_admin_analytics_frontend.py -v
rg -n "target_fixture|admin_analytics/target|test_admin_target_fixtures" dashboard/ docs/superpowers/plans/2026-09-15-admin-layer-redesign-prD-contract-recut.md --glob '!docs/**'
```

Expected: **PASS** on every test, including `test_recut_fields_absent_render_awaiting_data_source_not_an_empty_chart` and `test_rows_and_profile_without_recut_fields_render_dashes_not_guesses` (they delete the keys from the committed payloads, so the pending branches are still exercised); the `rg` prints nothing under `dashboard/`.

- [ ] **Step 4: Full suite**

```bash
python -m pytest dashboard/backend/tests/ -q
git status --short
```

Expected: **PASS**; `git status` shows no `backtest.db` change (if it does, `git checkout -- dashboard/storage/data/backtest.db` before staging).

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/tests/_admin_dom_stub.py dashboard/backend/tests/test_admin_overview_frontend.py dashboard/backend/tests/test_admin_users_frontend.py dashboard/backend/tests/test_admin_analytics_frontend.py
git status --short
git commit -m "$(cat <<'EOF'
test: fold the target-shape analytics fixtures into the committed set

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

(The `git rm` calls in Step 2 already staged the deletions.)

---

## Not in this PR

- **Any JavaScript change.** The page renders every field this PR adds and already carries the absent-field rule; if a slot still reads "Awaiting data source" after this PR deploys, the bug is in Task 1's population of that field, not in the module.
- **A store method, a table or DDL** — PR A and PR B's surface; this PR calls what they shipped.
- **Removing the pending branches from the renderers** — they are the fail-visible guard for the contract this PR creates (`CLAUDE.md`, "Fail-closed is not fail-visible").
- **Dropping D15's `country_code`, `device_category`, `browser_family`, `top_product_page` from the payload** — the collection decision is a §14 follow-up; the page does not display them (PR C Task 8 and Task 10 pin that).
- **Snapshotting response schemas from `app.openapi()`** — §14 "route-contract guard depth", worth doing now that every parameter is declared, but a separate PR.
- **`GET /admin-analytics` and its middleware exemption** — the 308 lives for one release (PR C); both go together later.

## Acceptance

- [ ] `AnalyticsOverview.billing_lane_mix`, `OperationalAnalyticsResponse.top_operational_reasons`, `CommercialAnalyticsResponse.purchased_by_day` / `consumed_by_day`, and `user_group` / `role` / `group_badge` / `last_meaningful_activity_at` on `ValueUserListItem` and `ValueUserProfile` are on the response models, populated from PR B's service methods, and present in every committed fixture (Task 1).
- [ ] Every `/api/admin/analytics/*` query parameter is a declared `Query(...)` visible in `app.openapi()` under today's names (plus `role`), `_query_values` and `_invalid_query` are gone, and every validation failure is still the display-safe `{"detail": "Invalid Analytics query."}` with no echo (Task 1).
- [ ] `tests/fixtures/admin_analytics/target/` and `test_admin_target_fixtures.py` are deleted; `_admin_dom_stub.target_fixture` is gone; the two renderer test modules read the committed fixtures and still pass unchanged expectations, including the two absent-field tests (Task 2).
- [ ] No file under `dashboard/frontend/` changes; no `?v=` bump.
- [ ] `python -m pytest dashboard/backend/tests/ -q` is green, and CI's Postgres tier is green before merge.

## Self-review notes

- **Spec coverage.** §9 "Added" table → Task 1 (every row except `max_active_dashboard_backtests`, which PR C shipped). §10.3 → Task 1 (declared params, `_query_values` deleted, names kept, fixtures regenerated). §13 row D → Tasks 1–2; its "Must not" column is the "Not in this PR" list. D11 → the two `resolve_group_badge` call sites in Task 1 and no other. D15 → fields kept in the payload (Task 1's `test_recut_fields_are_on_every_response`). D18/D19 → this is the single re-cut, after B and after C.
- **Provenance.** Task 1 is PR C's original Task 1, moved here verbatim on 2026-09-16 when the delivery order changed (design §13); Step 0 was added to reconcile the method signatures PR B's plan declares with the keyword shapes the task was drafted against, and the one cross-reference to PR C's Task 8 was reworded. Task 2 undoes exactly what PR C's Task 1, Task 5 (`target_fixture`), Task 7/8 (fixture constants) and Task 9 (the shape assertions) put in place, quoting each hunk in full.
- **Placeholder scan.** No `TBD`, `TODO`, "similar to Task N", or elided body. The one thing this plan cannot quote is the merged text of PR B's service signatures; Step 0 reads them off the checkout and gives the substitution per method.
- **Name consistency.** `BillingLaneDay`, `OperationalReasonCount`, `LedgerDayTotal`, `InvalidAnalyticsQuery` (Task 1) are referenced only there. The test names Task 2 relies on (`test_recut_fields_absent_render_awaiting_data_source_not_an_empty_chart`, `test_rows_and_profile_without_recut_fields_render_dashes_not_guesses`, `test_fixtures_match_committed_analytics_shapes`, `test_target_fixtures_add_exactly_the_section_9_fields`) are the ones the PR C plan defines.
