# Admin Users and Grant Pool Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Grant Pool management into the Admin Users page and remove the standalone Grant Pool tab without changing backend Credits behavior.

**Architecture:** Keep `admin-credits.js` as the single owner of Grant Pool loading and mutations. Move the existing Grant Pool markup into `adminPanelUsers`, remove only the Grant Pool tab/panel, and normalize legacy `adminTab=grant-pool` URLs to Users in `admin-tabs.js`. Update static frontend contracts to describe the three-tab layout.

**Tech Stack:** HTML, vanilla JavaScript, CSS, pytest static contracts.

**Spec:** `docs/superpowers/specs/2026-08-24-admin-users-grant-pool-integration-design.md`

## Global Constraints

- Do not change backend routes, database schema, ledger calculations, or Credits accounting.
- Preserve existing Grant Pool element IDs, API payloads, idempotency fields, fixed `admin-console` source, and reason fields.
- Keep the legacy user-quota table hidden and preserve its backend logic.
- Do not run browser automation or visual QA; the user performs browser acceptance.
- Do not expose or commit real API keys, local databases, `.superpowers/`, or `work/`.

---

### Task 1: Remove the standalone Grant Pool tab and relocate its existing markup

**Files:**
- Modify: `dashboard/frontend/app.html`
- Test contract: `dashboard/backend/tests/test_admin_credits_frontend.py`

**Interfaces:**
- Consumes: Existing `adminCreditsSection` markup and stable element IDs.
- Produces: One Users panel containing stats, Grant Pool controls, and the Credits user table; three navigation tabs: Users, Providers, Activity.

- [ ] **Step 1: Remove the Grant Pool navigation button**

Delete the `adminTabGrantPool` button from `adminTabs`, leaving Users first, followed by Providers and Activity.

- [ ] **Step 2: Move the existing Grant Pool section**

Remove the outer `adminPanelGrantPool` wrapper and place its existing `adminCreditsSection` immediately after `adminStats` inside `adminPanelUsers`. Keep every existing Grant Pool element ID unchanged.

- [ ] **Step 3: Update the static markup contract**

Change the frontend test to assert exactly three tabs and assert that `adminStats` appears before `adminCreditsSection`, which appears before `adminCreditsUsers`. Assert that the standalone Grant Pool panel and tab are absent.

- [ ] **Step 4: Commit the markup contract**

```bash
git add dashboard/frontend/app.html dashboard/backend/tests/test_admin_credits_frontend.py
git commit -m "feat: merge grant pool into admin users"
```

---

### Task 2: Normalize legacy tab URLs and keep tab accessibility correct

**Files:**
- Modify: `dashboard/frontend/js/admin-tabs.js`
- Test contract: `dashboard/backend/tests/test_admin_credits_frontend.py`

**Interfaces:**
- Consumes: Existing `AdminTabs.setTab(value, options)` and `onEnter()`.
- Produces: `setTab('grant-pool')` resolves to Users; left/right navigation cycles Users, Providers, Activity.

- [ ] **Step 1: Add legacy-tab normalization**

Map the legacy value `grant-pool` to `users` before validating `ALLOWED_TABS`. Keep the allowed set limited to `users`, `providers`, and `activity`.

- [ ] **Step 2: Normalize the URL on Admin entry**

Call `setTab` from `onEnter` with URL updates enabled so `adminTab=grant-pool` becomes `adminTab=users`.

- [ ] **Step 3: Update tab static assertions**

Replace the four-tab order assertion with a three-tab order assertion and add a source assertion for the legacy mapping.

- [ ] **Step 4: Commit tab compatibility**

```bash
git add dashboard/frontend/js/admin-tabs.js dashboard/backend/tests/test_admin_credits_frontend.py
git commit -m "fix: normalize removed admin grant pool tab"
```

---

### Task 3: Preserve styling and perform static handoff checks

**Files:**
- Modify if needed: `dashboard/frontend/styles.css`
- Modify if needed: `dashboard/frontend/app.html`
- Test/verification: changed frontend files and static contract file

**Interfaces:**
- Consumes: The moved DOM structure and existing Grant Pool CSS.
- Produces: A clean handoff with no duplicate Grant Pool IDs, no visible legacy quota table, and fresh asset cache versions if markup or tab JavaScript changed.

- [ ] **Step 1: Check CSS assumptions**

Confirm the existing `.admin-credits-console`, summary, ring, and action styles work when the section is inside Users. Add only a scoped spacing adjustment if the move creates a duplicated top border or excessive gap.

- [ ] **Step 2: Verify asset cache versions**

Increment the stylesheet or `admin-credits.js` query version only if the browser would otherwise reuse stale assets.

- [ ] **Step 3: Perform non-browser static checks**

The user will run the focused pytest and Node commands. In this session only inspect the final diff and ensure no real secrets, local database, `.superpowers/`, or `work/` files are staged.

- [ ] **Step 4: Commit final styling/cache adjustments if any**

```bash
git add dashboard/frontend/app.html dashboard/frontend/styles.css dashboard/frontend/js/admin-tabs.js
git commit -m "chore: finalize admin users grant pool layout"
```

