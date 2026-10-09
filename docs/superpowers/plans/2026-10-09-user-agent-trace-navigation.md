# User Agent Trace Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an Admin navigation path from User Analytics to a user's agents, runs, and live reasoning trace.

**Architecture:** Reuse the existing admin user profile and trace API. Add a server-owned user-to-agent/run summary endpoint, then add profile-level links that navigate to the existing trace list/detail views. Keep trace storage and event contracts unchanged.

**Tech Stack:** FastAPI, existing Agent/Run repositories, vanilla JavaScript Admin modules, pytest and the repository DOM harness.

**Spec:** `docs/superpowers/specs/2026-10-07-agent-reasoning-trace-design.md`

## Global Constraints

- Keep trace data in the existing Agent Postgres/Neon boundary.
- Do not expose hidden chain-of-thought; display bounded reasoning summaries only.
- Reuse `require_admin`, existing AdminShell request/routing, and textContent-based DOM rendering.
- Preserve existing user analytics pagination and profile section behavior.
- Every loop ends with focused tests, a commit, and a push to PR #622.
- Do not add a second database or duplicate trace storage.

---

### Task 1: Define the Admin user-agent-run contract

**Files:**
- Create: `dashboard/backend/api/routers/admin_agent_activity.py`
- Modify: `dashboard/backend/api/router.py`
- Test: `dashboard/backend/tests/test_admin_agent_activity_api.py`
- Modify: `dashboard/backend/tests/test_app_composition.py`

**Interfaces:**
- `GET /api/admin/users/{user_id}/agent-activity`
- Response: `{user_id, agents: [{agent_id, name, agent_type, created_at, runs: [{run_id, status, created_at, trace_id}]}]}`
- Endpoint is admin-only and returns empty arrays for a valid user with no agents.

- [ ] Add authorization, empty-user, owned-agent, run-link, and missing-trace tests.
- [ ] Run the new API test and confirm it fails because the route is absent.
- [ ] Implement repository-backed aggregation using existing agent and run stores; resolve trace IDs through `trace_store.get_trace_for_run`.
- [ ] Register the router and route contract.
- [ ] Run focused API and route-composition tests.
- [ ] Commit `feat: expose admin user agent activity`.
- [ ] Push the branch and verify PR checks start.

### Task 2: Render agent and run links in User Analytics profiles

**Files:**
- Modify: `dashboard/frontend/js/admin-users.js`
- Modify: `dashboard/frontend/admin.css`
- Test: `dashboard/backend/tests/test_admin_users_frontend.py`

**Interfaces:**
- The user profile gets an “Agent activity” section.
- Each agent displays name/id and its runs.
- A run with a trace links to `#traces/<trace_id>`.
- Running traces use copy such as “View live trace”; terminal traces use “View trace”.
- Loading, empty, error, and stale states remain explicit.

- [ ] Add DOM tests for empty activity, agent rows, trace links, and safe special-character rendering.
- [ ] Run focused frontend tests and confirm failure before implementation.
- [ ] Add lazy loading on profile open/section selection via AdminShell.
- [ ] Render all labels using DOM nodes/textContent.
- [ ] Run focused frontend and existing Admin shell tests.
- [ ] Commit `feat: link user analytics to agent traces`.
- [ ] Push and verify PR checks.

### Task 3: Add trace context and back navigation

**Files:**
- Modify: `dashboard/frontend/js/admin-traces.js`
- Modify: `dashboard/backend/tests/test_admin_traces_frontend.py`

**Interfaces:**
- Trace detail accepts optional `user_id`, `agent_id`, and `run_id` context in the hash query.
- Detail breadcrumb includes `Back to user analytics` when `user_id` is present.
- Existing direct `#traces` and `#traces/<trace_id>` behavior remains unchanged.

- [ ] Add DOM tests for contextual breadcrumb and query-preserving navigation.
- [ ] Implement context-safe links and breadcrumb.
- [ ] Run frontend trace tests and route tests.
- [ ] Commit `feat: add user context to trace navigation`.
- [ ] Push and verify PR checks.

### Task 4: Verify the complete operator flow

**Files:**
- Modify: `dashboard/backend/tests/test_admin_user_trace_flow.py`
- Modify: `docs/superpowers/specs/2026-10-07-agent-reasoning-trace-design.md`

**Interfaces:**
- Contract test covers User -> Agent -> Run -> Trace.
- Documentation records the Admin entry point and near-real-time polling behavior.

- [ ] Add an API contract test with a user, agent, run, trace, and terminal event.
- [ ] Run the focused flow, trace, and Admin suites.
- [ ] Run `git diff --check` and frontend syntax checks.
- [ ] Update documentation with the navigation path.
- [ ] Commit `docs: document admin trace navigation`.
- [ ] Push and report local and remote CI status.

