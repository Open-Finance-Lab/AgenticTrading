# Admin Console Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group Admin controls into Grant Pool, Users, Providers, and Activity tabs and replace separate pool Fund/Reduce forms with one signed adjustment input.

**Architecture:** Keep the existing Admin API routes and ledger unchanged. Add a small client-side tab controller in the existing Admin page, move the existing sections into tab panels, and route a signed number to the existing fund/reduce endpoints.

**Tech Stack:** Static HTML, CSS, browser JavaScript, pytest static contracts, Node syntax checks.

**Spec:** `docs/superpowers/specs/2026-08-24-admin-console-simplification-design.md`

## Global Constraints

- Never modify or display a real API key.
- Never commit `.superpowers/`, `work/`, or generated local databases.
- Keep commits and user-facing copy in English.
- Do not change Admin API routes or ledger semantics.

### Task 1: Add tab structure and simplify Grant Pool markup

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_admin_credits_frontend.py`

- [ ] Write static assertions for four tabs, Users default, no Overview/monthly/source/Fund/Reduce markup, and the signed input attributes.
- [ ] Wrap the existing stats/users, Grant Pool, Providers, and Activity sections in associated tab panels.
- [ ] Replace the two pool forms with one `adminGrantPoolForm` and a signed `adminGrantPoolAmount` number input.
- [ ] Add tab styling that matches the existing Credits tab treatment and responsive layout.
- [ ] Run the focused frontend tests.

### Task 2: Update Admin client behavior

**Files:**
- Modify: `dashboard/frontend/js/admin-credits.js`
- Create: `dashboard/frontend/js/admin-tabs.js`
- Test: `dashboard/backend/tests/test_admin_credits_frontend.py`

- [ ] Add strict signed micro-Credit parsing and dispatch positive values to `fund` and negative values to `reduce`.
- [ ] Send `source: 'admin-console'` without reading a source field.
- [ ] Keep reason validation, idempotency, access-loss handling, refresh, and user grant operations intact.
- [ ] Add URL-backed tab state with Users as the default and keyboard-safe ARIA state.
- [ ] Run Node syntax checks and focused tests.

### Task 3: Browser and regression verification

**Files:**
- Modify: none unless a test exposes a defect.

- [ ] Run the complete Admin frontend test slice and existing Admin API tests.
- [ ] Run the full backend suite and record any unrelated flaky test separately.
- [ ] Verify in the local browser that Users opens by default, tabs switch without stacking content, and signed pool adjustments use the existing routes.
- [ ] Run `git diff --check` and confirm only intended files are tracked.
