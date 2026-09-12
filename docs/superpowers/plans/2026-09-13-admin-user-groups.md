# Admin User Groups and Source Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add one canonical, admin-editable user-source group to every account and expose a six-row, filterable source analysis in Admin Analytics without changing billing, lifecycle, operational-state, or legacy acquisition semantics.

**Architecture:** Keep the source classification on the canonical users account row as a validated user_group value. Both SQLite and PostgreSQL stores expose the same admin projection and atomic patch method; the existing Admin Credits Account Management table is the one editing surface. Analytics reuses the existing batched user, event, run, and Credits readers to build a fixed-order group summary, while the vanilla frontend keeps the selected group in URL-backed state and renders a compact accessible table with proportional bars.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLite, PostgreSQL/psycopg, pytest, vanilla JavaScript, semantic HTML/CSS, and the existing Chart.js-loaded ATL admin shell.

**Spec:** docs/superpowers/specs/2026-09-13-admin-user-groups-design.md

## Global Constraints

- The only stored values are internal, invited, organic, competition, partner, and unknown, in that order; all labels and UI copy remain English.
- unknown is the default for new and historical accounts. Do not infer a new group from old acquisition source or cohort values.
- user_group is independent from Paid status, lifecycle segment, operational state, and legacy acquisition fields.
- include_internal keeps its current Analytics exclusion meaning; selecting Internal must not silently toggle it.
- Only the admin projection exposes the group. Public signup, login, /api/auth/me, and ordinary session payloads must not expose this admin-only field.
- PATCH /api/admin/users/{user_id} remains the single atomic mutation for role, entitlements, and group. Omitted fields remain unchanged; explicit null and unsupported values are rejected.
- Group Summary always returns six rows, including zero rows, in fixed taxonomy order. Each row contains Users, Successful run users, Repeat users, Total runs, ATL cost, and Paid users.
- Successful run users means at least one backtest_completed event in the selected UTC period. Repeat users means successful completions on at least two different UTC dates. Total runs counts one backtest_requested event per run attempt. Paid users use lifetime settled purchases minus settled refunds from the Credits ledger. ATL cost includes settled platform model cost only and excludes BYOK usage.
- Analytics endpoints remain admin-only, display-safe, batched, and independently available. Never return prompts, provider bodies, secrets, raw User-Agent values, full IPs, or credential material.
- Additive lazy migrations must work on existing SQLite and PostgreSQL deployments. Do not edit or commit any database file or production fixture.
- The canonical Account Management table gets the only Group editor; the hidden legacy Users table does not receive a second editor.
- Use native controls, proportional bars, and accessible text-table fallbacks. Do not use emoji as icons and do not introduce a second analytics architecture.
- Tests use synthetic stores and temporary databases only. Never print or stage API keys, database URLs with credentials, dashboard/storage/data/backtest.db, dashboard/storage/backtest.db, .superpowers/, work/, or generated mockups.
- Before deployment, local API/browser smoke and CI are required. A Vercel Preview cannot prove API persistence because its /api rewrite targets the current production backend. Do not report production completion until the real Render API and a real admin session have been verified.

## File Map

- dashboard/backend/domain/user_groups.py: single taxonomy, labels, strict write parsing, and tolerant read coercion.
- dashboard/backend/users.py: SQLite users.user_group schema/migration, account creation default, admin projection, and atomic store patch.
- dashboard/backend/users_postgres.py: PostgreSQL schema/migration, creation/default projection, and twin atomic patch.
- dashboard/backend/api/routers/admin_users.py: Pydantic request contract, HTTP validation, atomic patch wiring, and audit line.
- dashboard/backend/api/routers/admin_credits.py: pass the canonical group through the Account Management list response.
- dashboard/backend/domain/analytics/value_queries.py: group-aware filter model, fixed-order summary model, and batched aggregation.
- dashboard/backend/api/routers/admin_analytics.py: query-string validation and GET /api/admin/analytics/groups.
- dashboard/frontend/app.html: Account Management Group column/control and Analytics Group Summary/filter markup.
- dashboard/frontend/js/admin-credits.js: Account Management select rendering, save flow, and inline error recovery.
- dashboard/frontend/js/admin-analytics-value.js: URL state, group filter requests, summary rendering, and partial-availability handling.
- dashboard/frontend/styles.css: table, select, proportional-bar, overflow, and ATL-theme states.
- Focused tests: dashboard/backend/tests/test_user_groups.py, test_user_group_store.py, test_admin_users.py, test_admin_credits_api.py, test_admin_credits_frontend.py, domain/analytics/test_value_queries.py, test_admin_analytics_api.py, test_admin_analytics_value_frontend.py, test_users_postgres.py, and test_store_twin_parity.py.

---

### Task 1: Define the six-value group contract

**Files:**
- Create: dashboard/backend/domain/user_groups.py
- Create: dashboard/backend/tests/test_user_groups.py

**Interfaces:**
- Produce UserGroup = Literal["internal", "invited", "organic", "competition", "partner", "unknown"].
- Produce USER_GROUPS: tuple[UserGroup, ...], USER_GROUP_LABELS, parse_user_group(value: object) -> UserGroup, coerce_user_group(value: object) -> UserGroup, and user_group_label(value: UserGroup) -> str.
- parse_user_group strips and lowercases a string and raises ValueError("invalid_user_group") for None, blank, non-string, or unsupported values. coerce_user_group returns unknown for missing or malformed stored data.

- [ ] **Step 1: Write the failing taxonomy and parser tests.**

    import pytest
    from dashboard.backend.domain.user_groups import (
        USER_GROUPS, USER_GROUP_LABELS, coerce_user_group, parse_user_group
    )

    def test_groups_have_fixed_product_order_and_labels():
        assert USER_GROUPS == (
            "internal", "invited", "organic", "competition", "partner", "unknown"
        )
        assert [USER_GROUP_LABELS[group] for group in USER_GROUPS] == [
            "Internal", "Invited", "Organic", "Competition", "Partner", "Unknown"
        ]

    @pytest.mark.parametrize("value", ["internal", " ORGANIC ", "competition"])
    def test_parse_user_group_normalizes_supported_values(value):
        assert parse_user_group(value) in USER_GROUPS

    @pytest.mark.parametrize("value", [None, "", "not-a-group", 3, True])
    def test_parse_user_group_rejects_invalid_writes(value):
        with pytest.raises(ValueError, match="invalid_user_group"):
            parse_user_group(value)

    @pytest.mark.parametrize("value", [None, "", "old-acquisition-value", object()])
    def test_coerce_user_group_defaults_bad_stored_values_to_unknown(value):
        assert coerce_user_group(value) == "unknown"

- [ ] **Step 2: Run the focused test and verify the new module is missing.**

Run: pytest -q dashboard/backend/tests/test_user_groups.py

Expected: collection fails because dashboard.backend.domain.user_groups does not exist.

- [ ] **Step 3: Implement the constants and two explicit read/write paths.**

    from typing import Literal, Mapping, cast

    UserGroup = Literal[
        "internal", "invited", "organic", "competition", "partner", "unknown"
    ]
    USER_GROUPS = (
        "internal", "invited", "organic", "competition", "partner", "unknown"
    )
    USER_GROUP_LABELS = {
        "internal": "Internal", "invited": "Invited", "organic": "Organic",
        "competition": "Competition", "partner": "Partner", "unknown": "Unknown",
    }

    def parse_user_group(value: object) -> UserGroup:
        if not isinstance(value, str):
            raise ValueError("invalid_user_group")
        normalized = value.strip().lower()
        if normalized not in USER_GROUPS:
            raise ValueError("invalid_user_group")
        return cast(UserGroup, normalized)

    def coerce_user_group(value: object) -> UserGroup:
        try:
            return parse_user_group(value)
        except ValueError:
            return "unknown"

    def user_group_label(value: UserGroup) -> str:
        return USER_GROUP_LABELS[value]

- [ ] **Step 4: Run the tests and commit the standalone contract.**

Run: pytest -q dashboard/backend/tests/test_user_groups.py

Expected: PASS.

    git add dashboard/backend/domain/user_groups.py dashboard/backend/tests/test_user_groups.py
    git commit -m "feat: define admin user group taxonomy"

### Task 2: Add SQLite persistence and admin projections

**Files:**
- Modify: dashboard/backend/users.py
- Create: dashboard/backend/tests/test_user_group_store.py

**Interfaces:**
- UserStore._init_schema() declares users.user_group TEXT NOT NULL DEFAULT 'unknown' and lazily adds the same column to old databases.
- UserStore.create_user() creates unknown explicitly or through the schema default.
- UserStore.apply_admin_patch(..., user_group: str | None = None, ...) validates a supplied group, leaves an omitted group unchanged, updates it inside the existing BEGIN IMMEDIATE transaction, and returns it in the admin projection.
- public_user() remains the public projection without user_group; public_user_with_entitlements() and admin_user_rows_to_payloads() include a coerced group.

- [ ] **Step 1: Write migration, default, projection, and atomic-update tests.**

Create a store fixture in the new test file before the cases:

    @pytest.fixture
    def store(tmp_path):
        return UserStore(db_path=tmp_path / "users.db")

    import sqlite3
    from pathlib import Path
    import pytest
    from dashboard.backend.users import UserStore

    def test_existing_users_table_gets_unknown_group_column(tmp_path: Path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, display_name TEXT, "
            "password_hash TEXT, role TEXT, created_at TEXT)"
        )
        conn.execute(
            "INSERT INTO users VALUES (1, 'legacy@example.test', 'Legacy', 'hash', "
            "'user', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
        conn.close()
        store = UserStore(db_path=path)
        assert store.get_user_admin(1)["user_group"] == "unknown"

    def test_new_users_default_to_unknown_and_store_patch_is_atomic(store):
        user = store.create_user("group@example.test", "Group", "SecurePass1!")
        assert store.get_user_admin(user["id"])["user_group"] == "unknown"
        updated = store.apply_admin_patch(
            user["id"], user_group="organic", credits=7, max_concurrent_backtests=2
        )
        assert updated["user_group"] == "organic"
        assert updated["entitlements"]["credits"] == 7
        assert store.get_user_admin(user["id"])["user_group"] == "organic"

    def test_store_rejects_invalid_group_and_omitted_group_preserves_value(store):
        user = store.create_user("preserve@example.test", "Preserve", "SecurePass1!")
        store.apply_admin_patch(user["id"], user_group="partner")
        with pytest.raises(ValueError, match="invalid_user_group"):
            store.apply_admin_patch(user["id"], user_group="friends")
        assert store.apply_admin_patch(user["id"], credits=9)["user_group"] == "partner"

- [ ] **Step 2: Run the tests and verify the column/projection are absent.**

Run: pytest -q dashboard/backend/tests/test_user_group_store.py

Expected: FAIL because users.user_group, the patch parameter, and admin projection field do not yet exist.

- [ ] **Step 3: Add the additive SQLite migration and shared projection handling.**

Keep the existing PRAGMA table_info(users) pattern and add:

    if "user_group" not in columns:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN user_group TEXT NOT NULL DEFAULT 'unknown'"
        )

Import parse_user_group and coerce_user_group. Add user_group to the admin projection only:

    payload = public_user(row)
    payload.pop("avatar", None)
    payload["user_group"] = coerce_user_group(dict(row).get("user_group"))
    payload["entitlements"] = entitlements

Extend apply_admin_patch with user_group: Optional[str] = None, call parse_user_group only when supplied, execute UPDATE users SET user_group = ? WHERE id = ? inside the existing transaction, then read the row back before commit. Do not add the field to public_user().

- [ ] **Step 4: Run SQLite store and existing admin-store tests.**

Run: pytest -q dashboard/backend/tests/test_user_group_store.py dashboard/backend/tests/test_admin_users.py::test_apply_role_and_list_admin dashboard/backend/tests/test_store_twin_parity.py

Expected: the new SQLite tests pass; twin-parity may remain red until Task 3 adds the PostgreSQL twin.

- [ ] **Step 5: Commit the SQLite implementation.**

    git add dashboard/backend/users.py dashboard/backend/tests/test_user_group_store.py
    git commit -m "feat: persist admin user groups in sqlite"

### Task 3: Mirror the store contract in PostgreSQL

**Files:**
- Modify: dashboard/backend/users_postgres.py
- Modify: dashboard/backend/tests/test_users_postgres.py
- Modify: dashboard/backend/tests/test_store_twin_parity.py

**Interfaces:**
- PostgresUserStore exposes the same public methods and signatures as UserStore, including apply_admin_patch(..., user_group: str | None = None, ...).
- Fresh and existing PostgreSQL users tables contain user_group TEXT NOT NULL DEFAULT 'unknown'; the existing-table path uses ALTER TABLE users ADD COLUMN IF NOT EXISTS user_group TEXT NOT NULL DEFAULT 'unknown'.
- PostgreSQL admin list/get/create/patch payloads match SQLite exactly for user_group and continue omitting it from public session payloads.

- [ ] **Step 1: Add twin tests before changing PostgreSQL code.**

Extend the existing pg_only cases in test_users_postgres.py:

    @pg_only
    def test_user_group_default_and_atomic_patch_postgres(temp_postgres_store):
        user = temp_postgres_store.create_user(
            "pg-group@example.test", "PG Group", "securepass1"
        )
        assert temp_postgres_store.get_user_admin(user["id"])["user_group"] == "unknown"
        updated = temp_postgres_store.apply_admin_patch(
            user["id"], user_group="competition", credits=11
        )
        assert updated["user_group"] == "competition"
        assert updated["entitlements"]["credits"] == 11

Add a source-level migration assertion next to the existing twin checks:

    def test_users_postgres_repeats_sqlite_user_group_migration():
        source = Path("dashboard/backend/users_postgres.py").read_text(
            encoding="utf-8"
        )
        assert "ADD COLUMN IF NOT EXISTS user_group" in source

- [ ] **Step 2: Run the parity and PostgreSQL-focused tests.**

Run: pytest -q dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_users_postgres.py -k 'user_group or store_twins'

Expected: FAIL because the PostgreSQL DDL, signature, and projection do not yet match SQLite.

- [ ] **Step 3: Add PostgreSQL DDL, lazy migration, and atomic update.**

Add user_group TEXT NOT NULL DEFAULT 'unknown' to the CREATE TABLE users literal and an ALTER TABLE ... ADD COLUMN IF NOT EXISTS user_group TEXT NOT NULL DEFAULT 'unknown' beside the existing Discord/avatar migrations. Import and reuse the strict parser and tolerant projection from users.py.

In the existing transaction in PostgresUserStore.apply_admin_patch, normalize the optional group once and add a parameterized UPDATE users SET user_group = %s WHERE id = %s before the existing entitlements upsert. Keep the advisory lock and last-admin guard unchanged; read the row back with the existing RETURNING/readback path so the returned admin payload is identical to SQLite.

- [ ] **Step 4: Run both twin suites.**

Run: pytest -q dashboard/backend/tests/test_store_twin_parity.py dashboard/backend/tests/test_user_group_store.py dashboard/backend/tests/test_users_postgres.py -k 'user_group or store_twins'

Expected: PASS when TEST_POSTGRES_URL is configured; without it, live cases are skipped and source-level parity still passes.

- [ ] **Step 5: Commit the PostgreSQL twin.**

    git add dashboard/backend/users_postgres.py dashboard/backend/tests/test_users_postgres.py dashboard/backend/tests/test_store_twin_parity.py
    git commit -m "feat: mirror user groups in postgres store"

### Task 4: Expose and audit the admin mutation

**Files:**
- Modify: dashboard/backend/api/routers/admin_users.py
- Modify: dashboard/backend/api/routers/admin_credits.py
- Modify: dashboard/backend/tests/test_admin_users.py
- Modify: dashboard/backend/tests/test_admin_credits_api.py

**Interfaces:**
- AdminUserPatch.user_group uses the shared UserGroup type and remains optional.
- PATCH /api/admin/users/{user_id} accepts { "user_group": "organic" }, rejects explicit null with the existing no-null contract, returns 422 for an unsupported value, and returns the saved admin user projection.
- The audit line includes actor, target, old_group, and new_group without logging email, secrets, or arbitrary request text.
- GET /api/admin/credits/users includes user_group in each canonical Account Management row.

- [ ] **Step 1: Write HTTP contract tests.**

Add cases to test_admin_users.py:

    def test_admin_can_patch_group_and_public_me_does_not_expose_it(
        isolated_auth, capsys
    ):
        client, store = isolated_auth
        admin = _signup(client, "group-admin@example.com")
        target = _signup(client, "group-target@example.com")
        _promote(store, admin["id"])
        _login(client, "group-admin@example.com")
        response = client.patch(
            "/api/admin/users/" + str(target["id"]),
            json={"user_group": "organic"},
        )
        assert response.status_code == 200
        assert response.json()["user"]["user_group"] == "organic"
        audit = capsys.readouterr().out
        assert "old_group=unknown" in audit
        assert "new_group=organic" in audit
        assert "user_group" not in client.get("/api/auth/me").json()["user"]

    def test_admin_group_patch_rejects_null_and_unknown_value(isolated_auth):
        client, store = isolated_auth
        admin = _signup(client, "group-validation@example.com")
        _promote(store, admin["id"])
        _login(client, "group-validation@example.com")
        target = _signup(client, "group-invalid@example.com")
        assert client.patch(
            "/api/admin/users/" + str(target["id"]),
            json={"user_group": None},
        ).status_code == 400
        assert client.patch(
            "/api/admin/users/" + str(target["id"]),
            json={"user_group": "friends"},
        ).status_code == 422

Update the exact-response assertion in test_admin_credits_api.py::test_admin_user_search_composes_identity_with_bucket_projection to include user_group: unknown.

- [ ] **Step 2: Run the API tests and verify the route still drops the new field.**

Run: pytest -q dashboard/backend/tests/test_admin_users.py dashboard/backend/tests/test_admin_credits_api.py -k 'group or search_composes'

Expected: FAIL because the request model, audit payload, and credits-user projection do not yet contain the group.

- [ ] **Step 3: Wire the shared type, old/new audit values, and credits list projection.**

Add user_group: Optional[UserGroup] = None to AdminUserPatch; include it in explicit-null/no-fields checks and pass it to apply_admin_patch. Read the existing admin row before the atomic call so the audit line can record its coerced old value, then emit:

    _audit(
        "user_patched",
        actor=int(admin["id"]),
        target=int(user_id),
        old_group=previous_group,
        new_group=updated["user_group"],
        role=payload.role,
        max_concurrent_backtests=payload.max_concurrent_backtests,
        credits=payload.credits,
    )

In list_grant_users, copy identity["user_group"] into the response object. Do not add a second editor to the hidden legacy Users table.

- [ ] **Step 4: Run the complete focused admin/API suite.**

Run: pytest -q dashboard/backend/tests/test_admin_users.py dashboard/backend/tests/test_admin_credits_api.py dashboard/backend/tests/test_auth.py -k 'admin or group or me'

Expected: PASS.

- [ ] **Step 5: Commit the admin API contract.**

    git add dashboard/backend/api/routers/admin_users.py dashboard/backend/api/routers/admin_credits.py dashboard/backend/tests/test_admin_users.py dashboard/backend/tests/test_admin_credits_api.py
    git commit -m "feat: expose editable user groups to admins"

### Task 5: Add group-aware Analytics filters and fixed six-row summary

**Files:**
- Modify: dashboard/backend/domain/analytics/value_queries.py
- Modify: dashboard/backend/api/routers/admin_analytics.py
- Modify: dashboard/backend/tests/domain/analytics/test_value_queries.py
- Modify: dashboard/backend/tests/test_admin_analytics_api.py

**Interfaces:**
- UserValueFilters gains user_group: UserGroup | None = None; ValueAnalyticsQueryService.list_users() filters on the canonical account group without changing lifecycle/operational/commercial filters.
- Add UserGroupSummary with fields group, label, users, successful_run_users, repeat_users, total_runs, atl_cost_micro_usd, and paid_users.
- Add GroupAnalyticsResponse with as_of, groups: Sequence[UserGroupSummary], selected_user_group: UserGroup | None, and availability: SectionAvailability.
- Add ValueAnalyticsQueryService.get_groups(start: date, end: date, include_internal: bool = False, user_group: UserGroup | None = None, now: datetime | None = None) -> GroupAnalyticsResponse.
- Add GET /api/admin/analytics/groups?from=YYYY-MM-DD&to=YYYY-MM-DD&user_group=...&include_internal=.... It always serializes six rows in USER_GROUPS order.

- [ ] **Step 1: Add failing domain/API tests for definitions and query state.**

Extend the synthetic user helper in test_value_queries.py so it supplies user_group, and add:

    def test_group_summary_has_six_rows_zeroes_and_fixed_order():
        service = _service(
            snapshots={1: _snapshot(1)},
            user_groups={1: "organic"},
            group_events=[
                event(1, "backtest_requested", "2026-09-01T01:00:00+00:00"),
                event(1, "backtest_completed", "2026-09-01T02:00:00+00:00"),
                event(1, "backtest_completed", "2026-09-03T02:00:00+00:00"),
            ],
        )
        result = service.get_groups(
            date(2026, 9, 1), date(2026, 9, 4), now=NOW
        )
        assert [row.group for row in result.groups] == [
            "internal", "invited", "organic", "competition", "partner", "unknown"
        ]
        organic = result.groups[2]
        assert organic.users == 1
        assert organic.successful_run_users == 1
        assert organic.repeat_users == 1
        assert organic.total_runs == 1
        assert result.groups[0].users == 0

    def test_user_group_filter_applies_to_priority_users():
        filters = UserValueFilters(user_group="partner", priority=True)
        assert filters.user_group == "partner"

The synthetic event helper must include UTC timestamps and safe properties so the test proves repeat-day counting and cost filtering without production data.

- [ ] **Step 2: Add failing HTTP validation and endpoint tests.**

In test_admin_analytics_api.py, add cases that assert a valid user_group=organic reaches the service and an unsupported value returns 422; assert the response has six rows even when the selected group has no members.

    def test_group_endpoint_rejects_unknown_group(admin_analytics_api):
        response = admin_analytics_api.client.get(
            "/api/admin/analytics/groups",
            params={
                "from": "2026-09-01",
                "to": "2026-09-03",
                "user_group": "friends",
            },
        )
        assert response.status_code == 422

- [ ] **Step 3: Implement the filter and safe aggregation.**

Import the shared taxonomy. In _value_user_filters, allow and validate user_group, pass it into UserValueFilters, and add the same key to the router allowed query set. In list_users, skip rows whose user.get("user_group") does not match the filter.

Implement get_groups() by:
1. Loading all eligible admin-projection users in pages of 500 and coercing each group to unknown.
2. Applying the optional selected-group filter only to the eligible account set.
3. Loading the selected UTC period metric events through the existing display-safe query store, retaining only backtest_requested, backtest_completed, and model_usage_recorded facts for those user ids.
4. Counting one backtest_requested as one run attempt, unique completed users, and users whose completed-event UTC date set has at least two entries.
5. Reusing the existing safe cost_micro_usd extraction and billing-mode checks for model_usage_recorded; sum only platform_credits cost and never BYOK.
6. Loading batched commercial facts for the same users and counting lifetime_net_purchased_micro > 0 as paid.
7. Constructing a row for every value in USER_GROUPS, with zeros for absent groups and the display label from USER_GROUP_LABELS. Return availability status partial when an underlying section is incomplete, but never omit rows.

Keep aggregation in the service/query layer; do not add a new event stream or expose raw event properties.

- [ ] **Step 4: Add the /groups router and URL validation.**

Add user_group to _value_range(..., additional=...), parse it with the strict shared parser, and call service.get_groups(...). Map ValidationError/ValueError to the existing 422 contract and preserve the router-wide admin dependency.

- [ ] **Step 5: Run the domain and API tests.**

Run: pytest -q dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_admin_analytics_api.py -k 'group or user_group or value_user'

Expected: PASS, including fixed row order, zero rows, UTC repeat-day semantics, BYOK exclusion, filter propagation, and invalid-query handling.

- [ ] **Step 6: Commit the Analytics contract.**

    git add dashboard/backend/domain/analytics/value_queries.py dashboard/backend/api/routers/admin_analytics.py dashboard/backend/tests/domain/analytics/test_value_queries.py dashboard/backend/tests/test_admin_analytics_api.py
    git commit -m "feat: add user group analytics summary"

### Task 6: Add the Group editor to canonical Account Management

**Files:**
- Modify: dashboard/frontend/app.html
- Modify: dashboard/frontend/js/admin-credits.js
- Modify: dashboard/frontend/styles.css
- Modify: dashboard/backend/tests/test_admin_credits_frontend.py
- Modify: dashboard/backend/tests/test_admin_credits_api.py

**Interfaces:**
- The canonical adminCreditsUsersTable receives a Group column and one native select.admin-credits-group-select per row with all six labels.
- The select sends PATCH /api/admin/users/{id} with { user_group: nextValue }, stays disabled while saving, restores the prior value on failure, and replaces the in-memory value with the server response on success.
- Empty-state colSpan, table accessibility text, and narrow-screen overflow remain correct after adding the ninth column.

- [ ] **Step 1: Add static frontend contract tests.**

Extend test_admin_credits_frontend.py:

    def test_account_management_has_one_group_editor():
        assert '<th scope="col">Group</th>' in APP_HTML
        source = Path(
            "dashboard/frontend/js/admin-credits.js"
        ).read_text(encoding="utf-8")
        assert "admin-credits-group-select" in source
        assert "PATCH" in source and "/api/admin/users/" in source
        for label in (
            "Internal", "Invited", "Organic", "Competition", "Partner", "Unknown"
        ):
            assert label in source

- [ ] **Step 2: Run the frontend contract test and verify the markup is missing.**

Run: pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py -k 'group or account_management'

Expected: FAIL because the table and renderer have no Group column/control.

- [ ] **Step 3: Add the semantic table column and renderer.**

Insert Group between Account and Role in app.html; change the empty row to colspan="9". In admin-credits.js, define the shared ordered option list and render a native select:

    const USER_GROUP_OPTIONS = [
      ["internal", "Internal"], ["invited", "Invited"], ["organic", "Organic"],
      ["competition", "Competition"], ["partner", "Partner"], ["unknown", "Unknown"],
    ];

    function renderUserGroup(user) {
      const select = document.createElement("select");
      select.className = "admin-credits-group-select";
      select.setAttribute(
        "aria-label",
        "Group for " + userName(user)
      );
      USER_GROUP_OPTIONS.forEach(([value, label]) => {
        const option = new Option(
          label, value, false, value === (user.user_group || "unknown")
        );
        select.appendChild(option);
      });
      select.addEventListener("change", () => mutateUserGroup(user, select));
      return select;
    }

mutateUserGroup captures the previous value, disables the select, sends the single PATCH, applies data.user.user_group on success, and restores the previous value plus an inline status error on failure. Keep role confirmation and Grant mutations unchanged.

- [ ] **Step 4: Add ATL-consistent CSS and responsive behavior.**

Style .admin-credits-group-select with existing input tokens, a bounded minimum width, readable focus ring, and the same dark color scheme as the role select. Keep .admin-credits-users-table horizontally scrollable on narrow screens rather than squeezing labels into unreadable widths.

- [ ] **Step 5: Run the Account Management contracts.**

Run: pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py

Expected: PASS.

- [ ] **Step 6: Commit the Account Management UI.**

    git add dashboard/frontend/app.html dashboard/frontend/js/admin-credits.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py
    git commit -m "feat: add account management group editor"

### Task 7: Render Group Summary and URL-backed Analytics filtering

**Files:**
- Modify: dashboard/frontend/app.html
- Modify: dashboard/frontend/js/admin-analytics-value.js
- Modify: dashboard/frontend/styles.css
- Modify: dashboard/backend/tests/test_admin_analytics_value_frontend.py

**Interfaces:**
- admin-analytics-value.js adds groups: /api/admin/analytics/groups, ordered group labels/colors, state.userFilters.group, and URL key analyticsGroup.
- readUrlState(), writeUrlState(), setControls(), rangeParams(), and userParams() preserve and send the selected group without altering date-range or include_internal behavior.
- fetchGroups() loads the summary; renderGroups(payload) renders six rows, horizontal proportional bars scaled to the largest users count, aligned numeric columns, and a text-table fallback with an accessible caption.
- Group-summary errors follow existing section-level unavailable/stale behavior and do not blank lifecycle or priority-user panels.

- [ ] **Step 1: Add frontend contract tests before markup/code changes.**

Extend test_admin_analytics_value_frontend.py:

    def test_analytics_has_group_filter_and_summary_contract():
        assert 'id="adminValueGroup"' in APP_HTML
        source = Path(
            "dashboard/frontend/js/admin-analytics-value.js"
        ).read_text(encoding="utf-8")
        assert "/api/admin/analytics/groups" in source
        assert "analyticsGroup" in source
        assert "renderGroups" in source
        for label in (
            "Internal", "Invited", "Organic", "Competition", "Partner", "Unknown"
        ):
            assert label in APP_HTML or label in source

- [ ] **Step 2: Run the frontend contract test and verify the new surface is absent.**

Run: pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py -k group

Expected: FAIL because the Analytics filter, endpoint, and summary table do not yet exist.

- [ ] **Step 3: Add semantic filter and summary markup.**

Place a native User group select with id adminValueGroup in the existing value-filter form, with an All user groups option and the six taxonomy options. Add a Group summary section immediately below the primary headline and before deeper disclosures:

    <section id="adminAnalyticsGroupSummary" class="admin-value-card"
             aria-labelledby="adminAnalyticsGroupSummaryTitle">
      <div class="admin-value-section-head">
        <div>
          <p class="credits-section-kicker">Where users come from</p>
          <h4 id="adminAnalyticsGroupSummaryTitle">Group summary</h4>
        </div>
        <span id="adminAnalyticsGroupSummaryStatus"
              class="admin-value-coverage" role="status" aria-live="polite"></span>
      </div>
      <div class="admin-value-table-wrap">
        <table id="adminAnalyticsGroupTable" class="admin-value-group-table">
          <caption class="sr-only">Users and value signals by user group</caption>
          <thead>
            <tr>
              <th scope="col">Group</th>
              <th scope="col">Users</th>
              <th scope="col">Successful run users</th>
              <th scope="col">Repeat users</th>
              <th scope="col">Total runs</th>
              <th scope="col">ATL cost</th>
              <th scope="col">Paid users</th>
            </tr>
          </thead>
          <tbody id="adminAnalyticsGroupBody"></tbody>
        </table>
      </div>
      <p id="adminAnalyticsGroupFallback" class="admin-value-empty" hidden></p>
    </section>

- [ ] **Step 4: Implement URL state, request fan-out, and proportional bars.**

Add group to the existing userFilters object and use analyticsGroup in readUrlState() and writeUrlState(). Include user_group in rangeParams() (for /groups) and userParams() (for priority users). On filter change, refresh group summary and priority users while leaving the lifecycle range request independent.

renderGroups() clears and rebuilds the body from the server six rows, computes maxUsers = Math.max(1, ...rows.map(row => row.users)), and sets each bar width to the rounded users/maxUsers percentage in a DOM element with an accessible text value. If the endpoint is unavailable, show the existing unavailable message in the summary status and keep the last successful rows when available.

- [ ] **Step 5: Add compact ATL-theme CSS and narrow-screen overflow.**

Give each row a consistent height, a muted track plus group-colored fill, tabular numeric alignment, and enough minimum table width for all seven columns. Reuse --font-mono, var(--bg-card), var(--border-color), and existing focus tokens; do not introduce emoji or a new card grid of six detached numbers.

- [ ] **Step 6: Run frontend and static route contracts.**

Run: pytest -q dashboard/backend/tests/test_admin_analytics_value_frontend.py dashboard/backend/tests/test_static_routes.py

Expected: PASS.

- [ ] **Step 7: Commit the Analytics UI.**

    git add dashboard/frontend/app.html dashboard/frontend/js/admin-analytics-value.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_analytics_value_frontend.py
    git commit -m "feat: show user group summary in analytics"

### Task 8: Local browser smoke, review gates, and deployment acceptance

**Files:**
- Modify only if a focused test exposes a contract mismatch; otherwise no product files.
- Evidence: temporary local database and screenshot files outside the repository or in an ignored path.

**Interfaces:**
- Local verification proves the full path: account creation defaults to Unknown, an admin changes a group, the value persists after refresh, and Analytics returns/renders all six rows with the selected filter.
- CI/review/deployment reporting distinguishes a received Render Deploy Hook from a completed Render build and persistent production behavior.

- [ ] **Step 1: Run the full focused Python suite and syntax checks.**

Run:

    pytest -q \
      dashboard/backend/tests/test_user_groups.py \
      dashboard/backend/tests/test_user_group_store.py \
      dashboard/backend/tests/test_admin_users.py \
      dashboard/backend/tests/test_admin_credits_api.py \
      dashboard/backend/tests/test_admin_credits_frontend.py \
      dashboard/backend/tests/domain/analytics/test_value_queries.py \
      dashboard/backend/tests/test_admin_analytics_api.py \
      dashboard/backend/tests/test_admin_analytics_value_frontend.py \
      dashboard/backend/tests/test_store_twin_parity.py
    python -m compileall -q dashboard/backend
    git diff --check

Expected: PASS with no whitespace errors. If TEST_POSTGRES_URL is unset, report PostgreSQL behavioral cases as skipped rather than claiming live parity.

- [ ] **Step 2: Start the app against a temporary database and perform the browser smoke.**

Use a temporary DATABASE_PATH/equivalent local configuration and a synthetic admin account. In the browser:
1. Open Users -> Account Management and confirm a new row displays Unknown.
2. Change that row to Organic, wait for the save status, reload, and confirm it remains Organic.
3. Open Analytics, confirm the User group select is URL-backed, choose Organic, and confirm the summary still shows all six rows with non-zero values only where data exists.
4. Clear the filter and confirm lifecycle, date range, and include_internal state are unchanged.
5. Capture screenshots of Account Management and Analytics at desktop and a narrow viewport; inspect the rendered page, not just HTTP 200.

- [ ] **Step 3: Inspect the diff for scope and secrets.**

Run:

    git status --short
    git diff --stat origin/main...HEAD
    git diff --name-only --cached

Expected: only planned source/tests/docs are staged; no database files, secrets, .superpowers/, work/, or generated mockups are present.

- [ ] **Step 4: Push the branch and create a reviewable PR without claiming deployment.**

Run the repository CI workflow on feat/admin-user-groups, open or update the Draft PR, and record CI plus visual-smoke results. Explain that Vercel Preview is static-only for this change because its API rewrite targets current production Render.

- [ ] **Step 5: Complete production acceptance only after merge.**

After review and merge to main, wait for the Render build (not merely the Deploy Hook job). Verify /health, OpenAPI's /api/admin/analytics/groups route, the deployed Admin UI, and one real administrator round trip: read Unknown, save Organic, refresh, and filter Analytics by Organic. Report the PR and deployment complete only after those persistence checks pass.

## Plan self-review

- Spec coverage: taxonomy/defaults and boundaries are Task 1; SQLite/PostgreSQL persistence and lazy migration are Tasks 2-3; atomic API/audit/public projection rules are Task 4; filter and six-row definitions are Task 5; Account Management and Analytics UI behavior are Tasks 6-7; local/CI/Render acceptance is Task 8.
- Placeholder scan: no step relies on TBD, TODO, or an unspecified "add appropriate handling" instruction; each implementation step names a file, interface, and concrete test/command.
- Type consistency: UserGroup, USER_GROUPS, parse_user_group, and coerce_user_group originate in Task 1 and are reused by store, API, query, and frontend tasks; UserGroupSummary and GroupAnalyticsResponse originate in Task 5 and are the exact response consumed by Task 7.
