# Admin Credits Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Admin Credits Account Management page size from 100 to 25 accounts while preserving existing pagination and search behavior.

**Architecture:** Keep offset pagination in the existing Admin Credits browser state. Change the shared frontend request default and the FastAPI endpoint default together, while retaining response metadata as the source of truth for rendering the range and navigation state.

**Tech Stack:** Vanilla JavaScript, FastAPI/Pydantic, pytest, static frontend contract tests.

**Spec:** `docs/superpowers/specs/2026-08-28-admin-credits-pagination-design.md`

## Global Constraints

- Only the Admin Credits `Account Management` list changes; the legacy Admin Users table remains untouched.
- The page size is exactly 25 accounts by default.
- Existing `Previous`, `Next`, search reset, and out-of-range fallback behavior remains intact.
- No new dependencies or API fields are introduced.

### Task 1: Align the pagination defaults

**Files:**
- Modify: `dashboard/frontend/js/admin-credits.js:5-16`
- Modify: `dashboard/backend/api/routers/admin_credits.py:181-182`
- Test: `dashboard/backend/tests/test_admin_credits_frontend.py`
- Test: `dashboard/backend/tests/test_admin_credits_api.py`

**Interfaces:**
- Consumes: existing `state.usersLimit`, `loadUsers()`, and `list_grant_users()` response metadata.
- Produces: a default request/response limit of `25` with unchanged `users`, `total`, `limit`, and `offset` fields.

- [ ] **Step 1: Add failing contract assertions**

  Assert the frontend state initializes `usersLimit: 25`, and exercise the API endpoint without a `limit` query parameter to assert `limit == 25` and that the first page contains at most 25 users.

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run:

  ```bash
  pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py
  ```

  Expected: the new assertions fail because both current defaults are 100.

- [ ] **Step 3: Change both defaults to 25**

  In `admin-credits.js`, change `usersLimit: 100` to `usersLimit: 25`. In `admin_credits.py`, change `limit: int = Query(default=100, ge=1, le=500)` to `limit: int = Query(default=25, ge=1, le=500)`. Do not change the maximum or offset logic.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Run:

  ```bash
  pytest -q dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py
  ```

  Expected: all Admin Credits tests pass.

- [ ] **Step 5: Verify the patch and commit**

  Run:

  ```bash
  git diff --check
  git add dashboard/frontend/js/admin-credits.js dashboard/backend/api/routers/admin_credits.py dashboard/backend/tests/test_admin_credits_frontend.py dashboard/backend/tests/test_admin_credits_api.py docs/superpowers/specs/2026-08-28-admin-credits-pagination-design.md docs/superpowers/plans/2026-08-28-admin-credits-pagination.md
  git commit -m "fix: shorten admin credits user pages"
  ```
