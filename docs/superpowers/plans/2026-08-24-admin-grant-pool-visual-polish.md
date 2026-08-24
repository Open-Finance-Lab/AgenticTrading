# Admin Grant Pool Visual Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Grant Pool's two summary metrics with a data-backed donut card, compact the Admin Provider save controls, and place `Users` before `Grant Pool` in the Admin tabs.

**Architecture:** Keep the existing Admin API and auth flow unchanged. Extend the existing `renderPool(pool)` client renderer to update the donut's total, legend values, and SVG stroke segments; use scoped CSS for the provider form buttons; reorder only the tab buttons so URL state, panel IDs, and keyboard behavior remain intact.

**Tech Stack:** Static HTML, CSS, vanilla JavaScript, pytest static frontend contracts, Node syntax checks.

**Spec:** `docs/superpowers/specs/2026-08-24-admin-provider-controls-and-tab-order-design.md`, plus the approved Grant Pool donut mockup from the conversation.

## Global Constraints

- Do not change Admin API payloads, ledger behavior, credential handling, authorization, or URL parameter names.
- The donut uses green for `Available`, red for `Allocated`, and shows their sum as `Total Pool`.
- Provider action SVG icons must have explicit 15px dimensions; no real API keys may be added to fixtures, docs, or output.
- Browser login, screenshots, and visual interaction checks remain manual for the user.

---

### Task 1: Lock the new Admin visual contracts in static tests

**Files:**
- Modify: `dashboard/backend/tests/test_admin_credits_frontend.py`
- Modify: `dashboard/backend/tests/test_admin_model_providers_frontend.py`

**Interfaces:**
- Consumes: Existing `APP_HTML`, `ADMIN_JS`, and `STYLES` source strings loaded by both test modules.
- Produces: Regression guards for the donut IDs, tab order, provider button styles, and icon dimensions.

- [ ] **Step 1: Add Grant Pool donut assertions**

In `test_admin_grant_console_is_separate_from_user_credits_page`, assert that
the Admin markup contains `adminCreditsPoolTotal`,
`adminCreditsPoolRingAvailable`, `adminCreditsPoolRingAllocated`, and the
`Available`/`Allocated` legend labels. Assert that the old
`admin-credits-metric` markup is absent from the Admin summary. In the client
contract test, assert that `display_pool_available_credits`,
`display_allocated_to_users_credits`, and `strokeDasharray` are used.

- [ ] **Step 2: Add tab-order and style assertions**

Add a test that extracts the `adminTabs` nav and asserts the index order is
`users` before `grant-pool`, followed by `providers` and `activity`. Extend
the responsive-style test with `.admin-credits-pool-ring` and
`.admin-credits-pool-legend`.

In `test_admin_model_providers_frontend.py`, assert the styles contain
`.admin-provider-form > .auth-btn`,
`.admin-platform-key-form > .auth-btn`, and
`.admin-provider-form .auth-btn svg` with explicit `width: 15px` and
`height: 15px` declarations.

- [ ] **Step 3: Run only the affected static tests and confirm they fail**

Run:

```bash
pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_model_providers_frontend.py
```

Expected: the new contract assertions fail because the markup and styles are
not implemented yet.

### Task 2: Implement the Grant Pool donut and Admin tab order

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/admin-credits.js`
- Modify: `dashboard/frontend/js/admin-tabs.js` only if the existing default/order contract needs a focused comment or constant update
- Modify: `dashboard/frontend/styles.css`

**Interfaces:**
- Consumes: `pool.display_pool_available_credits` and
  `pool.display_allocated_to_users_credits` from the existing Grant Pool API.
- Produces: `renderPool(pool)` updates `adminCreditsPoolTotal`,
  `adminCreditsPoolAvailable`, `adminCreditsAllocated`,
  `adminCreditsPoolRingAvailable`, and `adminCreditsPoolRingAllocated`.

- [ ] **Step 1: Replace the two summary metrics with the donut markup**

In `app.html`, keep the `admin-credits-summary` container but replace its two
`admin-credits-metric` children with one donut figure and one legend. Use a
220px viewBox with a base track circle and two circles carrying the IDs
`adminCreditsPoolRingAvailable` and `adminCreditsPoolRingAllocated`. Put
`Total Pool` and `adminCreditsPoolTotal` in the ring center. Keep the existing
available and allocated element IDs in the legend so the API renderer has
stable targets.

- [ ] **Step 2: Update `renderPool(pool)` with safe segment math**

Add a local circumference constant for the ring radius and a helper that
clamps a ratio to `[0, 1]`, calculates visible/remaining stroke lengths, and
sets `style.strokeDasharray` and `style.strokeDashoffset`. Parse the two
display strings as finite numbers, fall back to zero for invalid values, sum
them for the total, and format the total to six decimal places. Render the
existing formatted values in the legend and offset the allocated segment by
the available segment length. Do not use `innerHTML`, local storage, or any
secret-bearing field.

- [ ] **Step 3: Add scoped donut and compact provider button CSS**

Replace the old two-column metric rules with a single responsive summary card:
the ring sits beside the legend on desktop and stacks above it below 600px.
Define explicit SVG dimensions, green/red segment colors, a muted base track,
center typography, and legend rows with small square swatches.

Add scoped rules for
`.admin-provider-form > .auth-btn` and
`.admin-platform-key-form > .auth-btn` that use `inline-flex`, `gap: 8px`,
`justify-self: start`, compact padding, and `width: max-content`. Add a
scoped child SVG rule with `width: 15px`, `height: 15px`, and fixed flex basis.
Keep the existing mobile full-width rule for the platform credential form.

- [ ] **Step 4: Swap the tab button order**

In the `adminTabs` nav, move the `Users` button before `Grant Pool` while
leaving all IDs, `aria-controls`, and panel order unchanged. The existing
`DEFAULT_TAB = 'users'` and arrow-key logic will then follow the requested
order without API or URL changes.

### Task 3: Verify the implemented layer and commit

**Files:**
- Test: `dashboard/backend/tests/test_admin_credits_frontend.py`
- Test: `dashboard/backend/tests/test_admin_model_providers_frontend.py`
- Check: `dashboard/frontend/js/admin-credits.js`
- Check: `dashboard/frontend/js/admin-tabs.js`
- Check: `dashboard/frontend/app.html`
- Check: `dashboard/frontend/styles.css`

**Interfaces:**
- Consumes: The completed markup, renderer, styles, and static contracts from Tasks 1-2.
- Produces: A clean Admin frontend patch ready for the user's manual browser check.

- [ ] **Step 1: Re-run the affected static tests**

Run:

```bash
pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_model_providers_frontend.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```bash
node --check dashboard/frontend/js/admin-credits.js
node --check dashboard/frontend/js/admin-tabs.js
git diff --check
```

Expected: all commands exit successfully.

- [ ] **Step 3: Inspect the final diff for scope and secrets**

Run:

```bash
git diff --stat
git diff -- dashboard/frontend/app.html dashboard/frontend/js/admin-credits.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_model_providers_frontend.py
```

Confirm that no API keys, local database files, `.superpowers/`, or `work/`
paths are staged.

- [ ] **Step 4: Commit the implementation**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/admin-credits.js dashboard/frontend/styles.css dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_model_providers_frontend.py
git commit -m "feat: polish admin grant pool visuals"
```
