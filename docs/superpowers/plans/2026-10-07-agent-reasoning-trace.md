# Agent Reasoning Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable, ordered reasoning trace for ATL Agent runs and expose it to the Admin console, starting with deterministic rule-based/SDK runs and expanding to tool, data, and child-Agent events.

**Architecture:** Add a trace repository and service beside the existing run repository. A trace is a run-level envelope; ordered events reference existing Run, Step, Decision, and artifact identifiers rather than copying business records. Admin reads the trace through require-admin API routes, while the frontend initially polls an incremental event endpoint.

**Tech Stack:** Python 3, FastAPI, SQLite development/test stores, Postgres/Neon production store conventions, pytest, vanilla JavaScript Admin frontend.

**Spec:** `docs/superpowers/specs/2026-10-07-agent-reasoning-trace-design.md`

## Global Constraints

- Keep trace data in the existing Agent Postgres/Neon boundary; tests must work with the repository's SQLite test setup.
- Do not replace existing Run, Step, Decision, ExecutionResult, or Artifact APIs.
- Store bounded rationale summaries and redacted payloads; never persist credentials or hidden chain-of-thought.
- Every event is ordered by `(trace_id, sequence_no)` and safe to retry by idempotency key.
- New Admin routes must be registered in the repository's route-contract tests.
- Every loop ends with focused tests, relevant regression tests, and a small commit.

### Task 1: Trace repository primitives (L1)

**Files:**
- Create: `dashboard/backend/domain/traces/__init__.py`
- Create: `dashboard/backend/domain/traces/repository.py`
- Create: `dashboard/backend/tests/test_trace_repository.py`
- Modify: `dashboard/backend/domain/runs/repository.py` only if a shared timestamp/schema helper is required

**Interfaces:**
- `TraceStore.create_trace(...) -> dict`
- `TraceStore.append_event(...) -> dict`
- `TraceStore.get_trace(trace_id) -> dict | None`
- `TraceStore.list_events(trace_id, *, after_sequence=0, limit=100) -> dict`
- `trace_store` module singleton following existing store conventions

- [ ] Write failing tests for table creation, trace creation, ordered event append, duplicate idempotency key, and bounded limit.
- [ ] Run `pytest -q dashboard/backend/tests/test_trace_repository.py` and confirm the new tests fail for missing repository symbols.
- [ ] Implement SQLite schema migration, safe JSON serialization, redaction/size validation, sequence allocation under a transaction, and idempotent retry behavior.
- [ ] Run the focused repository tests and then `pytest -q dashboard/backend/tests/test_run_lifecycle_unification.py`.
- [ ] Commit `feat: add ordered agent trace repository`.

### Task 2: Postgres twin and store selection (L1)

**Files:**
- Create: `dashboard/backend/domain/traces/repository_postgres.py`
- Create: `dashboard/backend/tests/test_trace_store_postgres.py`
- Modify: `dashboard/backend/domain/traces/repository.py`
- Modify: `dashboard/backend/tests/test_store_twin_parity.py` only for the declared trace twin

**Interfaces:**
- `PostgresTraceStore` implements the same public methods as `TraceStore`.
- `build_trace_store(database_url=None)` chooses SQLite or Postgres from the resolved Agent/content database configuration.

- [ ] Add parity tests for create, append, duplicate retry, and incremental event reads.
- [ ] Run the focused tests and confirm the Postgres tests skip only when no test database is configured.
- [ ] Implement the twin using the repository's existing Postgres connection-pool and dict-row conventions.
- [ ] Run parity tests and architecture-boundary tests.
- [ ] Commit `feat: add postgres persistence for agent traces`.

### Task 3: Trace lifecycle service and deterministic run instrumentation (L2)

**Files:**
- Create: `dashboard/backend/domain/traces/service.py`
- Modify: `dashboard/backend/domain/runs/service.py`
- Modify: `dashboard/backend/api/v2/runs.py` only at the existing lifecycle boundaries
- Create: `dashboard/backend/tests/test_trace_run_instrumentation.py`

**Interfaces:**
- `start_trace_for_run(run, initial_input) -> dict`
- `record_decision_event(trace_id, step_id, decision_id, decision) -> dict`
- `record_execution_event(trace_id, step_id, result) -> dict`
- `complete_trace(trace_id, result_summary) -> dict`
- `fail_trace(trace_id, error_code) -> dict`

- [ ] Add a deterministic end-to-end test that creates a rule-based/SDK run and asserts `run_started`, `decision_recorded`, `execution_result`, and `run_completed` order.
- [ ] Run the test and confirm no trace is produced before instrumentation exists.
- [ ] Add lifecycle calls at run creation, accepted decision finalization, execution result persistence, and terminal completion; keep existing run behavior unchanged.
- [ ] Add failure instrumentation without persisting exception text that may contain secrets.
- [ ] Run the trace test plus `test_v2_http_runs.py`, `test_v2_runs.py`, and `test_run_lifecycle_unification.py`.
- [ ] Commit `feat: instrument deterministic runs with traces`.

### Task 4: Admin trace API (L3)

**Files:**
- Create: `dashboard/backend/api/routers/admin_traces.py`
- Modify: `dashboard/backend/api/router.py`
- Create: `dashboard/backend/tests/test_admin_traces_api.py`
- Modify: `dashboard/backend/tests/test_app_composition.py` for route contract entries

**Interfaces:**
- `GET /api/admin/traces?agent_id=&run_id=&status=&limit=&cursor=`
- `GET /api/admin/traces/{trace_id}`
- `GET /api/admin/traces/{trace_id}/events?after_sequence=0&limit=100`

- [ ] Add tests for admin authorization, list filters, missing trace, pagination, incremental events, and status consistency.
- [ ] Run the focused API tests and confirm route and auth failures before implementation.
- [ ] Implement require-admin routes with stable JSON envelopes and bounded query parameters.
- [ ] Register every path in route-contract tests and run the Admin API suite.
- [ ] Commit `feat: expose admin agent trace api`.

### Task 5: Admin trace timeline (L4)

**Files:**
- Modify: `dashboard/frontend/admin.html`
- Create: `dashboard/frontend/js/admin-traces.js`
- Modify: `dashboard/frontend/admin-shell.js` or the current Admin tab router
- Modify: `dashboard/frontend/admin.css`
- Create: `dashboard/backend/tests/test_admin_traces_frontend.py`

**Interfaces:**
- The page consumes the three Admin trace endpoints and renders running, completed, failed, empty, and permission-denied states.

- [ ] Add DOM contract tests for the trace list, detail shell, event timeline, and explicit empty/error states.
- [ ] Run the frontend test and confirm the required DOM is absent.
- [ ] Add the Admin navigation entry and timeline renderer with bounded text rendering and artifact links.
- [ ] Run the focused frontend tests and existing Admin shell/frontend tests.
- [ ] Verify one completed trace in a browser and commit `feat: add admin agent trace timeline`.

### Task 6: Tool and data retrieval events (L5)

**Files:**
- Modify: `dashboard/backend/domain/traces/service.py`
- Modify: `dashboard/backend/infrastructure/market_data/provider.py` to emit retrieval boundaries
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/openai.py` to emit provider-call boundaries
- Create: `dashboard/backend/tests/test_trace_tool_events.py`

**Interfaces:**
- `record_tool_call(...) -> dict`
- `record_tool_result(...) -> dict`
- `record_data_retrieval(...) -> dict`

- [ ] Add tests for redacted arguments, provider/source metadata, duration, and failure events.
- [ ] Run the focused tests and confirm the new events are absent.
- [ ] Implement wrappers that store names, bounded summaries, source identifiers, and timings while excluding secrets and raw oversized data.
- [ ] Run tool-event tests and relevant market-data/LLM adapter tests.
- [ ] Commit `feat: record trace tool and data events`.

### Task 7: Incremental Admin updates (L6)

**Files:**
- Modify: `dashboard/frontend/js/admin-traces.js`
- Modify: `dashboard/backend/tests/test_admin_traces_frontend.py`
- Create or modify: `dashboard/backend/tests/test_admin_trace_polling.py` if the frontend test harness supports route sequencing

- [ ] Add a test fixture where the event endpoint returns sequence 1, then sequence 2, and assert the UI appends only the new event.
- [ ] Run the test and confirm the current page does not update incrementally.
- [ ] Implement 2-5 second polling with `after_sequence`, stop polling on terminal status, and show a stale/error state without deleting existing events.
- [ ] Run frontend and Admin API regression tests, then visually verify a running trace.
- [ ] Commit `feat: stream agent traces through incremental polling`.

### Task 8: Parent-child traces and production hardening (L7)

**Files:**
- Modify: `dashboard/backend/domain/traces/repository.py` for parent trace columns and indexes
- Modify: `dashboard/backend/domain/traces/repository_postgres.py` for the matching Postgres migration
- Modify: `dashboard/backend/domain/traces/service.py` for child creation and terminal-link validation
- Modify: `dashboard/backend/api/routers/admin_traces.py` to return parent/child links
- Modify: `dashboard/frontend/js/admin-traces.js` to render expandable child traces
- Modify: `dashboard/frontend/admin.html` only if the existing timeline shell lacks a child-trace mount point
- Create: `dashboard/backend/tests/test_trace_hardening.py`
- Modify: `docs/integrations/` with the public trace contract and retention notes

- [ ] Add failing tests for parent-child trace links, permission denial, payload size limits, duplicate event retries, terminal status consistency, and retention-safe listing.
- [ ] Run the focused tests and capture each failure.
- [ ] Add `parent_trace_id`/`parent_event_id` only where the tests prove they are needed, enforce indexes, and document retention and redaction rules.
- [ ] Run the complete backend suite, Admin frontend suite, and Postgres parity suite.
- [ ] Commit `feat: harden agent reasoning traces for production`.

## Final verification and PR

- [ ] Run `git diff --check` and inspect the full branch diff.
- [ ] Run the focused trace tests, the complete backend test suite, and packaging tests.
- [ ] Run a local Admin browser smoke test with one deterministic run and one live polling run.
- [ ] Push `feat/agent-reasoning-trace` and open a pull request with the design, loop commits, test evidence, and known deferred decisions.
