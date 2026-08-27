# Admin Footprint and User Analytics — PR 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add server-authoritative instrumentation, metric computation, daily rollups, explainable user-state snapshots, an idempotent 180-day authoritative backfill, and read-only Admin Analytics query APIs without shipping the Admin Analytics UI.

**Architecture:** PR 2 consumes the SQLite/PostgreSQL event repository, validation, authenticated ingestion, subject-exclusion, access-log, and retention contracts delivered by PR 1. A small domain instrumentation facade emits allowlisted events after source commits and never changes the source operation's result. Metrics and state calculation read authoritative source tables plus analytics events, write bounded daily rollups and current snapshots, and expose a query service that merges completed-day rollups with the current day's raw events. The API router is an Admin-gated adapter only; it does not contain metric or lifecycle logic.

**Tech Stack:** Python 3.12, FastAPI, Pydantic 2, SQLite, psycopg 3 PostgreSQL twin repositories, pytest, existing `dashboard.backend` domain/API layering, UTC-aware `datetime`, and deterministic HMAC/error-category helpers from PR 1.

**Spec:** `docs/superpowers/specs/2026-08-26-admin-user-analytics-design.md`

## Global Constraints

- PR 2 scope is limited to Account, Credential, Agent, Run, Resource instrumentation; metric calculations; daily rollups; five explainable user states; authoritative backfill; and Admin query services/APIs. Do not implement the Admin Analytics tab or User Analytics Profile UI; those belong to PR 3.
- Server-authoritative events are emitted only after the source operation reaches its authoritative committed outcome. Analytics append failures must never fail login, credential operations, Agent mutations, backtests, model execution, or Credits settlement.
- Raw user-level event detail is retained for exactly 180 days through the PR 1 retention contract. Daily non-identifying rollups remain available long term. Admin profile-access audit records remain under the PR 1 365-day policy.
- Analytics must never persist, return, or log full API keys/tokens, passwords/codes, prompts, strategy or portfolio text, raw provider response bodies, full IP addresses, raw User-Agent headers, or encrypted credential ciphertext.
- The request identity is the authenticated `user_id` supplied by the server. Network data may use only PR 1's monthly HMAC pseudonymization and allowlisted browser/device reduction; missing or invalid `ANALYTICS_PSEUDONYMIZATION_KEY` omits `network_hash` and never stores plaintext.
- SQLite and PostgreSQL implementations expose the same repository contract and produce equivalent event, rollup, snapshot, cursor, and API behavior.
- Default metrics exclude Admin accounts and `analytics_excluded` subjects. `include_internal=true` is an explicit Admin-only diagnostic override.
- Active users use meaningful accepted page visits or server-authoritative business events; token refresh, background polling, and unattended heartbeat traffic are excluded.
- First-success conversion uses only mature signup cohorts whose seven-day observation window has elapsed. Backtest success rate is `completed / (completed + failed)` and excludes user-cancelled, queued, and running runs. Repeat run rate requires a second success at least 24 hours and at most 30 days after the first success. Platform model cost sums actual `platform_credits` evidence only; BYOK never contributes to ATL cost or ATL Credits debits.
- User-state precedence is exactly `Blocked`, `Needs Attention`, `Dormant`, `Onboarding`, `Active`; each snapshot stores `status`, `reason_code`, `human_readable_reason`, `evidence_event_ids`, and `calculated_at`.
- All `/api/admin/analytics/*` routes use the centralized `require_admin` dependency. Profile reads append an Admin analytics access record without storing response bodies.
- Raw accepted events must be queryable within one minute. Completed historical days read primarily from `analytics_daily_rollups`; the current incomplete UTC day reads raw events and is merged by `AnalyticsQueryService`.
- Backfill is deterministic and idempotent, uses `event_source='backfill'`, never fabricates page views/sessions/device/region/network data, and uses stable source-event IDs for every reconstructed record.
- No real API keys, database URLs, `.superpowers/`, or `work/` paths may appear in tests, logs, plan commits, or API responses.

## PR 1 Dependency Contract

PR 2 must start only after PR 1 is merged or its equivalent foundation is present. The following names and signatures are the contract PR 2 consumes. If the merged PR 1 uses different internal names, stop implementation and ask the PR 1 owner to publish equivalent aliases in a separate PR 1 change; this PR 2 branch must not modify or reimplement the foundation tables, ingestion validation, retention, pseudonymization, subject settings, or access-log schema.

```python
# dashboard/backend/domain/analytics/models.py
class AnalyticsEvent(BaseModel):
    event_id: str
    schema_version: int
    event_name: str
    event_group: Literal["account", "credential", "agent", "run", "resource", "experience"]
    user_id: int
    session_id: str | None
    occurred_at: datetime
    received_at: datetime
    event_source: Literal["server", "frontend", "backfill"]
    source_event_id: str | None
    source_record_type: str | None
    source_record_id: str | None
    correlation_id: str | None
    page_view: str | None
    provider_id: str | None
    model_id: str | None
    billing_mode: str | None
    outcome: str | None
    error_category: str | None
    country_code: str | None
    device_category: str | None
    browser_family: str | None
    network_hash: str | None
    properties_json: str


# dashboard/backend/domain/analytics/repository.py and repository_postgres.py
class AnalyticsRepository(Protocol):
    def append_event(self, event: AnalyticsEvent) -> bool: ...
    def list_events(self, filters: AnalyticsEventFilters, *, limit: int, cursor: str | None) -> Page[AnalyticsEvent]: ...
    def upsert_daily_rollup(self, key: DailyRollupKey, values: DailyRollupValues) -> None: ...
    def list_daily_rollups(self, filters: RollupFilters) -> list[DailyRollup]: ...
    def get_user_snapshot(self, user_id: int) -> UserAnalyticsSnapshot | None: ...
    def upsert_user_snapshot(self, snapshot: UserAnalyticsSnapshot) -> None: ...
    def list_stale_snapshot_user_ids(self, before: datetime, *, limit: int) -> list[int]: ...
    def record_admin_access(self, *, admin_user_id: int, subject_user_id: int, section: str, accessed_at: datetime) -> None: ...
    def list_subject_settings(self) -> dict[int, SubjectSettings]: ...


# dashboard/backend/domain/analytics/service.py
def append_event_best_effort(event: AnalyticsEvent) -> bool: ...
def record_admin_profile_access(*, admin_user_id: int, subject_user_id: int, section: str) -> None: ...


# dashboard/backend/domain/analytics/privacy.py
def classify_safe_error(exc: BaseException) -> str: ...
def build_network_hash(*, client_ip: str | None, occurred_at: datetime) -> str | None: ...
def reduce_user_agent(raw_user_agent: str | None) -> tuple[str | None, str | None]: ...
```

The PR 1 repository must expose the same methods through its SQLite and PostgreSQL factories. `append_event` returns `False` for a duplicate `source_event_id`/event identity and `True` for a newly stored event; callers must treat both as successful observation outcomes. `append_event_best_effort` returns `True` for inserted or already-present events and `False` only when observation failed, while swallowing storage exceptions after emitting a safe operational warning.

## File Map

| File | Responsibility in PR 2 |
| --- | --- |
| `dashboard/backend/domain/analytics/instrumentation.py` | Event names, server-event builders, correlation/source-ID rules, and best-effort lifecycle emitters. |
| `dashboard/backend/domain/analytics/metrics.py` | Deterministic UTC metric formulas and filter normalization. |
| `dashboard/backend/domain/analytics/rollups.py` | Bounded daily aggregation and raw/current-day merge helpers. |
| `dashboard/backend/domain/analytics/states.py` | Five-state precedence, reason codes, evidence selection, and snapshot repair. |
| `dashboard/backend/domain/analytics/backfill.py` | Deterministic 180-day reconstruction from authoritative stores. |
| `dashboard/backend/domain/analytics/query_service.py` | Overview, user list, profile header, and independently pageable activity queries. |
| `dashboard/backend/domain/analytics/maintenance.py` | Idempotent rollup and stale-snapshot repair pass registered from the composition root. |
| `dashboard/backend/api/routers/admin_analytics.py` | Admin-only query endpoints and safe response models. |
| `dashboard/backend/api/router.py` | Register the PR 2 Admin Analytics router exactly once. |
| `dashboard/backend/app.py` | Register the analytics maintenance sweep from the existing reaper composition hook; no route bodies. |
| `dashboard/backend/api/auth.py` | Emit signup and authenticated session-start events after successful source writes. |
| `dashboard/backend/domain/model_providers/service.py` | Emit user credential lifecycle events after credential store commits. |
| `dashboard/backend/domain/agents/service.py` | Emit Agent create/update/delete events after repository commits. |
| `dashboard/backend/domain/runs/service.py` | Emit protocol run requested/queued/started/completed/failed/cancelled events after persisted transitions. |
| `dashboard/backend/domain/backtesting/external_run_service.py` | Emit legacy `/api/v1/backtest/*` lifecycle events after its authoritative run persistence. |
| `dashboard/backend/api/v2/runs.py` | Emit typed v2 run lifecycle and cancellation events after its direct `RunStore` writes. |
| `dashboard/backend/api/routers/backtests.py` | Emit dashboard backtest lifecycle events at accepted, terminal, and cancellation outcomes. |
| `dashboard/backend/infrastructure/llm/execution/service.py` | Emit model usage and safe execution-error events after usage/billing evidence is authoritative. |
| `dashboard/backend/domain/credits/service.py` | Emit Credits reserved/settled/refunded events after ledger/reservation commits. |
| `dashboard/scripts/backfill_analytics.py` | Operator-invoked backfill CLI using the domain backfill service; no secrets in arguments or output. |
| `dashboard/backend/tests/domain/analytics/test_instrumentation.py` | Event contract, idempotency, and failure-isolation tests. |
| `dashboard/backend/tests/domain/analytics/test_metrics.py` | UTC-boundary formulas and exclusion tests. |
| `dashboard/backend/tests/domain/analytics/test_rollups.py` | Daily rollup dimensions, merge behavior, and idempotency. |
| `dashboard/backend/tests/domain/analytics/test_states.py` | State precedence and deterministic evidence tests. |
| `dashboard/backend/tests/domain/analytics/test_backfill.py` | 180-day cutoff, deterministic IDs, and replay tests. |
| `dashboard/backend/tests/domain/analytics/test_repository_contract.py` | PR 1 repository consumption and SQLite/PostgreSQL parity contract. |
| `dashboard/backend/tests/test_admin_analytics_api.py` | Admin authorization, filters, cursors, profile access logging, and safe responses. |
| `dashboard/backend/tests/test_analytics_integration.py` | End-to-end synthetic lifecycle and source-operation failure isolation. |
| `dashboard/backend/tests/test_analytics_maintenance.py` | Maintenance registration and bounded repair behavior. |
| `dashboard/scripts/benchmark_analytics_queries.py` | Disposable synthetic overview/profile response-time verification. |

### Task 1: Build the PR 2 instrumentation facade

**Files:**
- Create: `dashboard/backend/domain/analytics/instrumentation.py`
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py`

**Interfaces:**
- Consumes: PR 1 `AnalyticsEvent`, `append_event_best_effort`, `classify_safe_error`, and the repository's allowlist validation.
- Produces: `emit_account_event`, `emit_credential_event`, `emit_agent_event`, `emit_run_event`, `emit_resource_event`, and `emit_safe_error_event`, each returning `None` and swallowing append failures after writing a safe operational warning.

- [ ] **Step 1: Write failing tests for event construction and isolation**

```python
def test_server_event_contains_no_sensitive_properties(monkeypatch):
    captured = []
    monkeypatch.setattr(instrumentation, "append_event_best_effort", lambda event: captured.append(event) or True)

    emit_credential_event(
        name="credential_saved",
        user_id=7,
        credential_id="cred-1",
        provider_id="openrouter",
        key_last_four="1234",
    )

    event = captured[0]
    assert event.event_group == "credential"
    assert event.source_event_id == "credential:credential_saved:cred-1"
    assert event.properties_json == '{"key_last_four":"1234"}'
    assert "api_key" not in event.properties_json


def test_append_failure_does_not_escape(monkeypatch, capsys):
    def fail(_event):
        raise RuntimeError("database contains a secret-looking detail")

    monkeypatch.setattr(instrumentation, "append_event_best_effort", fail)
    emit_run_event(name="run_requested", user_id=7, run_id="run-1")
    assert "database contains" not in capsys.readouterr().out


def test_safe_error_event_uses_stable_category(monkeypatch):
    captured = []
    monkeypatch.setattr(instrumentation, "append_event_best_effort", lambda event: captured.append(event) or True)
    emit_safe_error_event(user_id=7, source_record_type="run", source_record_id="run-1", exc=TimeoutError())
    assert captured[0].error_category == "provider_timeout"
    assert captured[0].properties_json == "{}"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py -q`

Expected: FAIL because `dashboard.backend.domain.analytics.instrumentation` and its emitters do not exist.

- [ ] **Step 3: Implement the minimal facade**

```python
def emit_run_event(*, name: str, user_id: int, run_id: str, status: str | None = None, correlation_id: str | None = None) -> None:
    event = _server_event(
        name=name,
        group="run",
        user_id=user_id,
        source_record_type="run",
        source_record_id=run_id,
        source_event_id=f"run:{name}:{run_id}",
        correlation_id=correlation_id or run_id,
        outcome=status,
    )
    _append_after_commit(event)


def _append_after_commit(event: AnalyticsEvent) -> None:
    try:
        observed = append_event_best_effort(event)
    except Exception as exc:  # noqa: BLE001 - analytics is observational
        print(f"WARNING: analytics append failed category={classify_safe_error(exc)}", flush=True)
        return
    if not observed:
        print("WARNING: analytics append failed category=internal_error", flush=True)
```

Use fixed event-name/group maps and reject names not in the PR 2 allowlist before constructing an event. Build properties only from explicit keyword arguments; never serialize arbitrary source records, exception messages, prompts, credential ciphertext, or request headers.

Use canonical source-event IDs shared by live instrumentation and backfill. Immutable outcomes use `{group}:{event_name}:{source_record_id}`. Repeatable mutations append an authoritative version: Agent/credential `updated_at`, Credits `operation_id`/reservation ID, or run call index. For example: `agent:agent_updated:a-1:2026-08-26T10:00:00+00:00`, `credential:credential_reverified:cred-1:2026-08-26T10:05:00+00:00`, and `resource:model_usage_recorded:run-1:3`. Never derive an ID from raw request JSON, prompt text, credential material, IP address, or User-Agent. Events without a stable non-secret source version set `source_event_id=None` rather than collapsing distinct operations.

Source modules must import the module (`from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation`) and call `analytics_instrumentation.emit_*`. Do not bind emitters with direct function imports; module calls keep failure-isolation tests and runtime dependency replacement reliable.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py -q`

Expected: PASS with event source IDs stable across repeated calls and append failures isolated.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/instrumentation.py dashboard/backend/tests/domain/analytics/test_instrumentation.py
git commit -m "feat: add analytics server instrumentation facade"
```

### Task 2: Instrument Account, Credential, and Agent outcomes

**Files:**
- Modify: `dashboard/backend/api/auth.py:388-575`
- Modify: `dashboard/backend/domain/model_providers/service.py:662-770`
- Modify: `dashboard/backend/domain/agents/service.py:301-735`
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py`
- Test: `dashboard/backend/tests/test_analytics_integration.py`

**Interfaces:**
- Consumes: Task 1 emitters; existing `user_store`, model-provider store, and Agent repository commit results.
- Produces: exactly one post-commit event per authoritative signup, authenticated session start, credential lifecycle transition, and Agent lifecycle mutation. Failed source operations emit only a safe error event when a stable user/record identity exists; they do not emit a success event.

- [ ] **Step 1: Pin failing source-operation tests**

```python
def test_signup_and_login_emit_account_events(client, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_account_event", lambda **kwargs: events.append(kwargs))
    signup = client.post("/api/auth/signup", json={"email": "u@example.com", "display_name": "U", "password": "SecurePass1!"})
    assert signup.status_code == 200
    login = client.post("/api/auth/login", json={"email": "u@example.com", "password": "SecurePass1!"})
    assert login.status_code == 200
    assert [event["name"] for event in events] == ["account_signed_up", "session_started"]


def test_credential_event_is_emitted_only_after_store_commit(credential_api, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_credential_event", lambda **kwargs: events.append(kwargs))
    response = credential_api.create_verified_credential()
    assert response.status_code == 201
    assert events[0]["name"] == "credential_saved"


def test_agent_delete_failure_does_not_emit_success(agent_context, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_agent_event", lambda **kwargs: events.append(kwargs))
    agent_context.store.delete_agent = lambda _agent_id: False
    agent_context.delete_agent()
    assert events == []
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: FAIL because lifecycle hooks are not wired to the source commits.

- [ ] **Step 3: Add post-commit hooks at authoritative boundaries**

Use these exact mappings:

```python
ACCOUNT_EVENTS = {
    "signup": "account_signed_up",
    "login": "session_started",
}

CREDENTIAL_EVENTS = {
    "create": "credential_saved",
    "verify_success": "credential_verified",
    "verify_failure": "credential_invalid",
    "set_default": "credential_defaulted",
    "reverify_success": "credential_reverified",
    "revoke": "credential_revoked",
}

AGENT_EVENTS = {
    "create": "agent_created",
    "update": "agent_updated",
    "delete": "agent_deleted",
}
```

In `auth.signup`, call `emit_account_event(name="account_signed_up", user_id=created_user["id"], source_record_type="user", source_record_id=str(created_user["id"]), occurred_at=created_user["created_at"])` after the user/session response can be formed. In `auth.login`, call `emit_account_event(name="session_started", ...)` only after the session token has been persisted. `session_started` uses a fresh Analytics event ID with `source_event_id=None`; it must not persist the raw authentication token, its database hash, its cookie value, or an authentication-session identifier as an Analytics session ID. It may attach only PR 1's monthly network HMAC, trusted country header, and reduced browser/device values calculated transiently from the request.

In `ModelProviderService`, emit after each successful store method returns its public credential projection; include only provider ID, credential ID, lifecycle outcome, and permitted last-four value. Creating a credential always emits `credential_saved`; when the same authoritative operation finishes verification it additionally emits `credential_verified` or `credential_invalid` with its own stable source event ID. A later explicit verify emits `credential_reverified` on success or `credential_invalid` on failure. In `AgentService`, emit after `create_agent`, `update_agent`, and `delete_agent` return/confirm the repository result; never include pipeline prompts or runtime config contents.

Only authenticated user-owned Agent operations are Analytics subjects. If an Agent has no `owner_user_id`, preserve the existing guest/API-only behavior and skip the user Analytics event instead of inventing an identity. For delete, capture the authenticated `owner_user_id` and safe Agent ID before the repository deletes the row, then emit only after `delete_agent` returns `True`.

Wrap each emitter with the Task 1 best-effort boundary. A credential verification error maps to `credential_invalid` or `provider_unavailable`; it must not serialize the adapter exception or verification message.

- [ ] **Step 4: Run the focused tests and full relevant auth/Agent/provider suites**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py dashboard/backend/tests/test_auth.py dashboard/backend/tests/test_agents_api.py dashboard/backend/tests/test_model_credentials_api.py -q`

Expected: PASS; source-operation status codes and persisted rows remain unchanged when analytics append is forced to raise.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/auth.py dashboard/backend/domain/model_providers/service.py dashboard/backend/domain/agents/service.py dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py
git commit -m "feat: instrument account credential and agent lifecycles"
```

### Task 3: Instrument protocol and dashboard backtest lifecycles

**Files:**
- Modify: `dashboard/backend/domain/runs/service.py:340-430,600-735,900-1010`
- Modify: `dashboard/backend/domain/backtesting/external_run_service.py:260-735`
- Modify: `dashboard/backend/api/v2/runs.py:330-490`
- Modify: `dashboard/backend/api/routers/backtests.py:620-910,1800-1940`
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py`
- Test: `dashboard/backend/tests/test_analytics_integration.py`

**Interfaces:**
- Consumes: Task 1 `emit_run_event`; `RunStore`/`BacktestDatabase` terminal state writes; existing run IDs, owner IDs, and correlation IDs.
- Produces: one idempotent lifecycle event for `run_requested`, `run_queued`, `run_started`, `run_completed`, `run_failed`, and `run_cancelled` across legacy `/api/v1/backtest/*`, protocol `/api/v1/runs`, typed `/api/v2/runs`, and dashboard `/backtest/run` surfaces.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_protocol_run_emits_requested_started_completed_in_order(protocol_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_run_event", lambda **kwargs: events.append(kwargs))
    run_id = protocol_fixture.create_run()
    protocol_fixture.mark_started(run_id)
    protocol_fixture.mark_completed(run_id)
    assert [item["name"] for item in events] == ["run_requested", "run_started", "run_completed"]
    assert all(item["correlation_id"] == run_id for item in events)


def test_dashboard_failed_run_emits_failed_not_completed(backtest_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_run_event", lambda **kwargs: events.append(kwargs))
    backtest_fixture.finish(error="provider unavailable")
    assert [item["name"] for item in events] == ["run_failed"]
    assert events[0]["error_category"] == "provider_unavailable"


def test_cancelled_run_is_excluded_from_success_denominator(backtest_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_run_event", lambda **kwargs: events.append(kwargs))
    backtest_fixture.cancel()
    assert events[0]["name"] == "run_cancelled"


@pytest.mark.parametrize("surface", ["legacy_v1", "protocol_v1", "typed_v2", "dashboard"])
def test_every_run_surface_emits_authoritative_lifecycle(surface, run_surface_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_run_event", lambda **kwargs: events.append(kwargs))
    run_surface_fixture(surface).complete_successfully()
    assert events[0]["name"] in {"run_requested", "run_queued"}
    assert events[-1]["name"] == "run_completed"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: FAIL because run lifecycle transitions do not yet call the analytics facade.

- [ ] **Step 3: Hook only committed transitions**

In `domain/runs/service.py`, resolve `user_id` from the persisted run's `agent_id` through the Agent repository; skip user Analytics when the Agent is API-only and has no account owner. Emit `run_requested` immediately after `run_store.create_run(..., status="running")` succeeds, emit `run_started` after the engine/session has been registered, and emit terminal events only after `run_store.update_run` commits the terminal status. In `_sync_status`, preserve the current terminal mapping and add the corresponding event after the update.

Apply the same rule to `external_run_service.py` and `api/v2/runs.py`, because both write run state outside `domain/runs/service.py`. The typed v2 cancellation endpoint emits `run_cancelled` only after its `RunStore` update succeeds. The legacy v1 external service emits from its persisted `agent_runs` insertion/finalization, not from read-only status polling. Reaper/recovery transitions use the same stable source event IDs, so a terminal status observed twice remains one event.

In `api/routers/backtests.py`, emit `run_queued` after the dashboard slot/run record is accepted, `run_started` after the worker has begun the authoritative run, and terminal events from `_finalize_slot` after the persisted `agent_runs` result is written. Use the existing live run ID as `source_record_id` and correlation ID. Pass only safe outcome/error-category values; retain detailed provider text in the existing source path, never in Analytics. When an authenticated run request is authoritatively refused before a run row exists, emit `safe_error_classified` with a fresh event ID and one stable category such as `credential_missing`, `provider_unavailable`, `credits_unavailable`, `model_not_allowed`, or `account_restricted`; this event is the evidence that can make the user's state `Blocked`.

Guard each call with a stable source event ID generated by Task 1 so retries from reaper/recovery or duplicate finalizer calls cannot create duplicate events.

- [ ] **Step 4: Run lifecycle and regression suites**

Run: `pytest dashboard/backend/tests/domain/runs dashboard/backend/tests/test_protocol_api.py dashboard/backend/tests/test_v2_http_runs.py dashboard/backend/tests/test_external_backtest_api.py dashboard/backend/tests/test_run_lifecycle_unification.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_launch_visibility.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: PASS; run state transitions and existing API responses remain byte-compatible apart from no new fields on existing run APIs.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/runs/service.py dashboard/backend/domain/backtesting/external_run_service.py dashboard/backend/api/v2/runs.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py
git commit -m "feat: instrument protocol and dashboard run lifecycles"
```

### Task 4: Instrument model usage, Credits reservations, settlements, refunds, and safe errors

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py:80-330`
- Modify: `dashboard/backend/domain/credits/service.py:219-315`
- Test: `dashboard/backend/tests/domain/analytics/test_instrumentation.py`
- Test: `dashboard/backend/tests/test_analytics_integration.py`

**Interfaces:**
- Consumes: `LLMExecutionRequest`, `LLMUsage`, `BillingEvidence`, reservation/settlement result models, and Task 1 resource emitters.
- Produces: one `model_usage_recorded` event per authoritative usage evidence, one event for each successful `credits_reserved`, `credits_settled`, or `credits_refunded` operation, and safe error events for denied/unavailable execution paths. BYOK events include usage but never a Credits debit.

- [ ] **Step 1: Write failing resource tests**

```python
def test_byok_usage_is_recorded_without_platform_cost_or_debit(execution_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_resource_event", lambda **kwargs: events.append(kwargs))
    result = execution_fixture.execute_byok()
    usage = next(item for item in events if item["name"] == "model_usage_recorded")
    assert usage["billing_mode"] == "byok"
    assert usage["platform_cost_usd"] == 0
    assert usage["debited_credits_micro"] == 0


def test_platform_settlement_emits_after_commit(execution_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_resource_event", lambda **kwargs: events.append(kwargs))
    execution_fixture.execute_platform()
    assert [item["name"] for item in events] == ["credits_reserved", "model_usage_recorded", "credits_settled"]


def test_provider_error_is_safe_and_does_not_echo_body(execution_fixture, monkeypatch):
    events = []
    monkeypatch.setattr(instrumentation, "emit_safe_error_event", lambda **kwargs: events.append(kwargs))
    execution_fixture.fail_with_provider_body("secret upstream response")
    assert events[0]["error_category"] == "provider_unavailable"
    assert "secret upstream response" not in repr(events[0])
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: FAIL because execution and Credits boundaries do not emit resource events.

- [ ] **Step 3: Add post-commit resource hooks**

In `LLMExecutionService._execute_platform`, emit `credits_reserved` only after `reserve_llm_credits` returns an open reservation, emit `model_usage_recorded` after the provider usage and cost evidence has passed validation, and emit `credits_settled` only after `settle_llm_credits` returns. In `_release_after_failure`/`finalize_run`, emit `credits_refunded` only after release commits. For BYOK, emit `model_usage_recorded` with `billing_mode="byok"`, actual token counts, provider/model IDs, and zero ATL debit/cost.

In `CreditsService`, instrument purchased/admin-grant refund operations after their ledger commits. Do not treat a purchased Credits ledger entry as a model-spend event; the analytics resource event must carry `resource_kind` so query code can keep purchased Credits separate from `user_entitlements.credits` model metering.

Map `LLMExecutionError` and repository failures through the fixed safe categories: `credential_missing`, `credential_invalid`, `provider_timeout`, `provider_unavailable`, `credits_unavailable`, `model_not_allowed`, `internal_error`, or `usage_unavailable`. Never include exception strings, raw provider bodies, or serialized request messages.

For execution failures that occur before a run exists, `emit_safe_error_event` uses `source_event_id=None` and the authenticated `user_id`; do not derive an idempotency key from prompt text, raw request JSON, or a credential. For failures tied to a persisted run/call, use `resource:error:{run_id}:{call_index}:{category}`.

- [ ] **Step 4: Run execution, Credits, and integration suites**

Run: `pytest dashboard/backend/tests/infrastructure/llm dashboard/backend/tests/domain/credits dashboard/backend/tests/test_credit_metering.py dashboard/backend/tests/integration/test_credits_checkout_flow.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: PASS; BYOK never changes ATL Credits state and analytics failures never change reservation/settlement outcomes.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/execution/service.py dashboard/backend/domain/credits/service.py dashboard/backend/tests/domain/analytics/test_instrumentation.py dashboard/backend/tests/test_analytics_integration.py
git commit -m "feat: instrument model usage and credits outcomes"
```

### Task 5: Implement deterministic metrics and daily rollups

**Files:**
- Create: `dashboard/backend/domain/analytics/metrics.py`
- Create: `dashboard/backend/domain/analytics/rollups.py`
- Test: `dashboard/backend/tests/domain/analytics/test_metrics.py`
- Test: `dashboard/backend/tests/domain/analytics/test_rollups.py`

**Interfaces:**
- Consumes: PR 1 event query/repository contract, authoritative users/runs/usage/Credits stores, subject settings, and Task 1 event names.
- Produces: `AnalyticsMetricFilters`, `calculate_overview_metrics(...)`, `rollup_day(...)`, `rollup_current_day(...)`, and bounded dimension keys used by Tasks 6 and 8.

- [ ] **Step 1: Write formula tests with fixed UTC fixtures**

```python
def test_active_users_uses_meaningful_events_only(event_repo, fixed_now):
    event_repo.seed_page("home", user_id=1, at=fixed_now - timedelta(days=1))
    event_repo.seed_event("session_heartbeat", user_id=2, at=fixed_now - timedelta(days=1))
    event_repo.seed_event("token_refreshed", user_id=3, at=fixed_now - timedelta(days=1))
    result = calculate_active_users(event_repo, now=fixed_now, include_internal=False)
    assert result == 1


def test_first_success_conversion_excludes_immature_cohorts(authoritative, fixed_now):
    authoritative.signup(user_id=1, at=fixed_now - timedelta(days=8))
    authoritative.signup(user_id=2, at=fixed_now - timedelta(days=2))
    authoritative.successful_run(user_id=1, at=fixed_now - timedelta(days=7, hours=1))
    assert calculate_first_success_conversion(authoritative, now=fixed_now) == 1.0


def test_success_rate_excludes_cancelled_and_non_terminal(authoritative, fixed_now):
    authoritative.run("completed", at=fixed_now - timedelta(days=1))
    authoritative.run("failed", at=fixed_now - timedelta(days=1))
    authoritative.run("cancelled", at=fixed_now - timedelta(days=1))
    authoritative.run("running", at=fixed_now - timedelta(hours=1))
    assert calculate_backtest_success_rate(authoritative, now=fixed_now) == 0.5


def test_platform_cost_excludes_byok(authoritative, fixed_now):
    authoritative.usage(billing_mode="platform_credits", cost_usd=1.25, at=fixed_now - timedelta(days=1))
    authoritative.usage(billing_mode="byok", cost_usd=99.0, at=fixed_now - timedelta(days=1))
    assert calculate_platform_model_cost(authoritative, now=fixed_now) == 1.25
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_metrics.py dashboard/backend/tests/domain/analytics/test_rollups.py -q`

Expected: FAIL because metric functions and rollup writers do not exist.

- [ ] **Step 3: Implement formulas and bounded rollups**

Use UTC-aware half-open intervals `[start, end)` for every date range. Normalize filters to a maximum range of 180 raw days while allowing historical rollup queries beyond that range. Implement the exact formulas:

```python
def backtest_success_rate(completed: int, failed: int) -> float | None:
    denominator = completed + failed
    return None if denominator == 0 else completed / denominator


def repeat_run_rate(users_with_first_success: int, users_with_repeat_success: int) -> float | None:
    return None if users_with_first_success == 0 else users_with_repeat_success / users_with_first_success
```

`rollup_day(day)` must aggregate only allowlisted dimensions: event name, billing mode, provider, model, outcome, error category, and user-state count. Never group by email, raw user ID, session ID, network hash, prompt, strategy, or arbitrary properties. Upsert on `(day, dimension_name, dimension_value, metric_name)` so rerunning a day is idempotent. `rollup_current_day` reads raw accepted events and overlays them on the previous completed-day rollups without mutating historical rows.

Persist these bounded aggregate metric names: `daily_active_users`, `rolling_active_users_7d`, `completed_runs`, `terminal_completed`, `terminal_failed`, `mature_signup_cohort_users`, `first_success_within_7d_users`, `users_with_first_success`, `users_with_repeat_success_24h_30d`, `platform_model_cost_usd`, `input_tokens`, `output_tokens`, `affected_users`, and `user_state_count`. Cohort numerators/denominators are stored as counts so long-range conversion can be recomputed without retaining identifying cohort rows.

Do not sum non-additive distinct-user counts across days. Store `rolling_active_users_7d` as the exact seven-day distinct count calculated at each completed day's UTC boundary; calculate the current incomplete day's value from raw events in the preceding seven days. `daily_active_users` is for the daily trend only. Likewise, compute an exact date-range `affected_users` count from raw events only while the requested range is inside the 180-day detail window; historical charts may use daily affected-user rollups but must not present their sum as a distinct range count.

For `user_state_count`, calculate each non-excluded user's state as of the day's closing boundary from events and authoritative records available at that time. Do not copy today's `user_analytics_snapshots` backward into historical rollups.

- [ ] **Step 4: Run metric, rollup, and regression tests**

Run: `pytest dashboard/backend/tests/domain/analytics/test_metrics.py dashboard/backend/tests/domain/analytics/test_rollups.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: PASS; fixed-boundary cases, admin/exclusion filters, and repeat execution produce identical values.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/metrics.py dashboard/backend/domain/analytics/rollups.py dashboard/backend/tests/domain/analytics/test_metrics.py dashboard/backend/tests/domain/analytics/test_rollups.py
git commit -m "feat: calculate analytics metrics and daily rollups"
```

### Task 6: Add explainable five-state snapshots and repair logic

**Files:**
- Create: `dashboard/backend/domain/analytics/states.py`
- Modify: `dashboard/backend/domain/analytics/instrumentation.py`
- Test: `dashboard/backend/tests/domain/analytics/test_states.py`
- Test: `dashboard/backend/tests/test_analytics_integration.py`

**Interfaces:**
- Consumes: Task 1 lifecycle/error events, authoritative user/run/credential/provider state, PR 1 snapshot repository methods, and subject exclusions.
- Produces: `calculate_user_state(user_id, now=...) -> UserAnalyticsSnapshot`, `recalculate_user_snapshot(user_id, ...)`, and `repair_stale_snapshots(now=..., limit=...) -> int`.

- [ ] **Step 1: Write precedence and evidence tests**

```python
def test_blocked_wins_over_needs_attention_and_onboarding(state_fixture, fixed_now):
    state_fixture.attempted_run_without_billing(at=fixed_now - timedelta(hours=1))
    state_fixture.failed_runs(count=3, within=timedelta(hours=24))
    snapshot = calculate_user_state(state_fixture.user_id, now=fixed_now)
    assert snapshot.status == "Blocked"
    assert snapshot.reason_code == "billing_lane_unavailable"
    assert snapshot.evidence_event_ids == [state_fixture.blocking_event_id]


def test_new_user_without_run_is_onboarding_not_blocked(state_fixture, fixed_now):
    snapshot = calculate_user_state(state_fixture.user_id, now=fixed_now)
    assert snapshot.status == "Onboarding"
    assert snapshot.reason_code == "no_successful_run"


def test_needs_attention_requires_three_failures_or_invalid_default(state_fixture, fixed_now):
    state_fixture.failed_runs(count=3, within=timedelta(hours=24))
    snapshot = calculate_user_state(state_fixture.user_id, now=fixed_now)
    assert snapshot.status == "Needs Attention"
    assert snapshot.reason_code == "three_consecutive_failed_runs"


def test_dormant_and_active_are_30_day_activity_states(state_fixture, fixed_now):
    state_fixture.first_success(at=fixed_now - timedelta(days=45))
    state_fixture.activity(at=fixed_now - timedelta(days=31))
    assert calculate_user_state(state_fixture.user_id, now=fixed_now).status == "Dormant"
    state_fixture.activity(at=fixed_now - timedelta(days=1))
    assert calculate_user_state(state_fixture.user_id, now=fixed_now).status == "Active"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_states.py -q`

Expected: FAIL because state calculation and snapshot persistence do not exist.

- [ ] **Step 3: Implement exact precedence and stable reason codes**

Evaluate rules in this order and stop at the first match:

```python
STATE_RULES = (
    ("Blocked", "billing_lane_unavailable", _is_blocked),
    ("Needs Attention", "three_consecutive_failed_runs", _needs_attention),
    ("Dormant", "no_meaningful_activity_30d", _is_dormant),
    ("Onboarding", "no_successful_run", _is_onboarding),
    ("Active", "recent_successful_run_and_activity", _is_active),
)
```

Use additional stable reason codes `provider_disabled`, `account_restricted`, `invalid_default_credential`, `run_deadline_exceeded`, and `successful_run_missing` when those branches provide the strongest evidence. A blocking condition is unresolved only when its latest evidence is newer than the corresponding resolving event: verified/defaulted credential, provider re-enabled, Credits made available, restriction removed, or a later successful run. Three failed runs are "consecutive" only when the three newest terminal runs inside the preceding 24 hours are all failed; a completed or cancelled run breaks the sequence. A run is beyond its safe deadline only when the authoritative run remains non-terminal after its stored execution/decision deadline.

A new user with no attempted core action is always `Onboarding`, even when no billing lane is configured. Evidence IDs must be sorted by occurrence time descending, then event ID ascending, and capped at five IDs. Upsert the snapshot after calculation; do not append immutable state history.

After the state service exists, extend Task 1's `_append_after_commit` to call a registered `recalculate_user_snapshot(user_id)` callback for event names in `SNAPSHOT_RELEVANT_EVENTS`. The callback is best-effort and runs only when `append_event_best_effort` reports that the event is stored or already present; its failure produces a safe operational warning and never changes the source operation. Keep `repair_stale_snapshots` as the periodic safety net for failed callbacks, deploy restarts, and backfilled rows.

- [ ] **Step 4: Run state and integration tests**

Run: `pytest dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_analytics_integration.py -q`

Expected: PASS; every user has exactly one state, reasons are deterministic, and snapshot failures do not alter source operations.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/states.py dashboard/backend/domain/analytics/instrumentation.py dashboard/backend/tests/domain/analytics/test_states.py dashboard/backend/tests/test_analytics_integration.py
git commit -m "feat: add explainable analytics user states"
```

### Task 7: Implement deterministic 180-day authoritative backfill and CLI

**Files:**
- Create: `dashboard/backend/domain/analytics/backfill.py`
- Create: `dashboard/scripts/backfill_analytics.py`
- Test: `dashboard/backend/tests/domain/analytics/test_backfill.py`

**Interfaces:**
- Consumes: users, Agent repository, `agent_runs`/protocol run stores, model usage evidence, Credits reservation/ledger stores, Task 1 emitters, and PR 1 idempotent append contract.
- Produces: `backfill_analytics(*, now: datetime | None = None, days: int = 180) -> BackfillReport` and an operator CLI with `--days`, `--before`, and `--dry-run`.

- [ ] **Step 1: Write failing cutoff, source-ID, and replay tests**

```python
def test_backfill_uses_only_authoritative_sources(backfill_fixture, fixed_now):
    backfill_fixture.user(user_id=1, created_at=fixed_now - timedelta(days=10))
    backfill_fixture.agent(agent_id="a-1", user_id=1, created_at=fixed_now - timedelta(days=9))
    backfill_fixture.run(run_id="r-1", user_id=1, status="completed", created_at=fixed_now - timedelta(days=8))
    backfill_fixture.page_view(user_id=1, at=fixed_now - timedelta(days=8))
    report = backfill_analytics(now=fixed_now)
    assert report.inserted == 3
    assert all(event.event_source == "backfill" for event in backfill_fixture.events())
    assert not any(event.event_name == "page_viewed" for event in backfill_fixture.events())


def test_backfill_ids_are_deterministic_and_idempotent(backfill_fixture, fixed_now):
    backfill_fixture.run(run_id="r-1", user_id=1, status="failed", created_at=fixed_now - timedelta(days=2))
    first = backfill_analytics(now=fixed_now)
    second = backfill_analytics(now=fixed_now)
    assert first.inserted == 1
    assert second.inserted == 0
    assert first.source_event_ids == ["run:run_failed:r-1"]


def test_backfill_excludes_records_older_than_180_days(backfill_fixture, fixed_now):
    backfill_fixture.run(run_id="old", user_id=1, status="completed", created_at=fixed_now - timedelta(days=181))
    assert backfill_analytics(now=fixed_now).inserted == 0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/domain/analytics/test_backfill.py -q`

Expected: FAIL because the backfill service and CLI do not exist.

- [ ] **Step 3: Implement source adapters and deterministic IDs**

Reconstruct only these records when their authoritative timestamp is within `[now - 180 days, now]`:

```python
BACKFILL_ID_TEMPLATES = {
    "signup": "account:account_signed_up:{user_id}",
    "agent": "agent:agent_created:{agent_id}",
    "run": "run:run_{terminal_status}:{run_id}",
    "usage": "resource:model_usage_recorded:{run_id}:{call_index}",
    "credits": "resource:credits_{outcome}:{reservation_id}",
}
```

Emit signup, Agent-created, terminal run, model-usage, and Credits reservation/settlement/refund events from source rows. Use `session_id=None`, `page_view=None`, `country_code=None`, `device_category=None`, `browser_family=None`, and `network_hash=None` for all backfilled events. Never inspect or copy prompt, strategy, credential ciphertext, raw provider error, browser, IP, or arbitrary metadata fields. Sort source rows by `(occurred_at, stable source ID)` before appending so reports are repeatable.

Resolve a historical run to a user only through authoritative ownership: `protocol_runs.agent_id -> external_agents.owner_user_id`, or `agent_runs.session_id -> external_agents.session_id -> owner_user_id`. If no authenticated owner can be proven, increment `skipped_unmapped_owner` and do not invent a user association. Read model usage and Credits outcomes from `credit_llm_reservations`, `credit_llm_usage_entries`, and their safe evidence fields; use `agent_runs.metadata.llm_execution` only when the typed `LLMRunEvidence` projection validates successfully.

Use the same canonical source IDs as live instrumentation so running backfill after deployment does not duplicate already-observed events. Suppress per-event snapshot callbacks during bulk load, then recalculate one snapshot per affected user and rebuild every affected completed UTC day's rollup exactly once. `--dry-run` performs source enumeration and validation without appending events, snapshots, or rollups.

The CLI must print counts and safe categories only:

```bash
python dashboard/scripts/backfill_analytics.py --days 180 --dry-run
python dashboard/scripts/backfill_analytics.py --days 180
```

Reject `--days` outside `1..180` and never accept secrets or database URLs as command-line arguments.

- [ ] **Step 4: Run backfill and CLI tests**

Run: `pytest dashboard/backend/tests/domain/analytics/test_backfill.py dashboard/backend/tests/test_analytics_integration.py -q && python dashboard/scripts/backfill_analytics.py --help`

Expected: PASS; help lists only safe options, the second backfill inserts zero duplicates, and no page/session events are fabricated.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/backfill.py dashboard/scripts/backfill_analytics.py dashboard/backend/tests/domain/analytics/test_backfill.py
git commit -m "feat: add idempotent analytics historical backfill"
```

### Task 8: Build the Admin Analytics query service

**Files:**
- Create: `dashboard/backend/domain/analytics/query_service.py`
- Create: `dashboard/backend/domain/analytics/maintenance.py`
- Modify: `dashboard/backend/app.py:210-240`
- Test: `dashboard/backend/tests/test_analytics_maintenance.py`
- Test: `dashboard/backend/tests/test_admin_analytics_api.py`

**Interfaces:**
- Consumes: Tasks 5–7, PR 1 repository/settings/access-log contracts, `user_store`, Agent/run/provider/Credits stores, and centralized Admin identity.
- Produces: `AnalyticsQueryService.get_overview`, `.list_users`, `.get_user_profile`, `.get_user_activity`, and `run_analytics_maintenance` for router and startup wiring.

- [ ] **Step 1: Write failing query and maintenance tests**

```python
def test_query_service_merges_completed_rollups_with_current_raw_day(query_fixture, fixed_now):
    query_fixture.rollup_completed_day(fixed_now.date() - timedelta(days=1), completed_runs=4)
    query_fixture.raw_current_day_completed_run(user_id=7, at=fixed_now - timedelta(minutes=2))
    overview = query_fixture.service.get_overview(now=fixed_now, filters=AnalyticsMetricFilters())
    assert overview.completed_runs == 5
    assert overview.last_updated == fixed_now


def test_maintenance_is_bounded_and_idempotent(maintenance_fixture, fixed_now):
    first = run_analytics_maintenance(now=fixed_now, snapshot_limit=25)
    second = run_analytics_maintenance(now=fixed_now, snapshot_limit=25)
    assert first.rollup_days == second.rollup_days
    assert first.repaired_snapshots <= 25
    assert second.repaired_snapshots == 0
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/test_admin_analytics_api.py -q`

Expected: FAIL because the query service and maintenance pass do not exist.

- [ ] **Step 3: Implement safe query DTOs and source merge**

Define these exact service methods:

```python
class AnalyticsQueryService:
    def get_overview(self, *, filters: AnalyticsMetricFilters, now: datetime | None = None) -> AnalyticsOverview: ...
    def list_users(self, *, filters: AnalyticsUserFilters, limit: int, offset: int) -> PaginatedUsers: ...
    def get_user_profile(self, *, user_id: int, now: datetime | None = None) -> AnalyticsUserProfile: ...
    def get_user_activity(self, *, user_id: int, section: ActivitySection, limit: int, cursor: str | None) -> Page[AnalyticsActivityItem]: ...
```

The overview DTO must contain `active_users_7d`, `first_success_conversion`, `backtest_success_rate`, `repeat_run_rate`, `platform_model_cost_usd`, `daily_active_users`, `daily_completed_runs`, `activation_funnel`, `user_state_counts`, `top_failure_categories`, `users_needing_attention`, `last_updated`, and the normalized filter echo. Each panel value carries `available: bool`; a failed component uses a fixed `error_code="temporarily_unavailable"` without exception text while other components remain populated.

The user-list DTO contains display name, email, user ID, state/reason, last meaningful activity, recent run/failure counts, and a stable profile path. The profile DTO contains join date, last meaningful activity, state/evidence, primary billing lane, default provider, coarse country, reduced device/browser, activation milestones, recent footprint, run summary, billing-lane mix, input/output tokens, ATL platform cost, ATL Credits debited, and top product page. It never returns arbitrary `properties_json`.

Normalize `limit` to `1..100`, `offset` to `>=0`, and cursors through the PR 1 signed/opaque cursor helper. For overview, query additive completed-day counts/costs from rollups, query the current UTC day from raw events, and merge by metric key. Read historical daily active-user points and completed-day rolling-seven counts directly from their rollup rows; never add distinct counts. Calculate the live rolling-seven active-user value from raw events because the entire window is inside 180-day retention. For user lists, join only display-safe fields from `user_store` with snapshot and bounded summary counts; default filter excludes Admin and `analytics_excluded`. For profile activity, allow only `timeline`, `runs`, `usage`, and `sessions`; sessions expose aggregate duration/count and reduced browser/device fields, never raw session IDs or headers. Query failures return a panel-level unavailable result with a fixed error code; they do not raise an all-or-nothing exception.

`run_analytics_maintenance` must process one completed UTC day, recalculate stale snapshots in batches of at most 100, and return counts. Keep a process-local last-run guard so the existing run reaper can invoke it frequently without reprocessing the same day on every pass.

- [ ] **Step 4: Wire maintenance through the composition root and run tests**

Register `run_analytics_maintenance` using `dashboard.backend.domain.runs.service.register_reaper_sweep` inside `startup_event`, guarded by its own `try/except` and safe warning. Do not add Analytics route bodies to `app.py`.

Run: `pytest dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_app_composition.py -q`

Expected: PASS; startup remains non-blocking, maintenance errors do not prevent the server or reaper from starting, and responses contain only display-safe Analytics fields.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/analytics/query_service.py dashboard/backend/domain/analytics/maintenance.py dashboard/backend/app.py dashboard/backend/tests/test_analytics_maintenance.py dashboard/backend/tests/test_admin_analytics_api.py
git commit -m "feat: add analytics query service and maintenance pass"
```

### Task 9: Expose Admin Analytics query APIs

**Files:**
- Create: `dashboard/backend/api/routers/admin_analytics.py`
- Modify: `dashboard/backend/api/router.py:1-60`
- Test: `dashboard/backend/tests/test_admin_analytics_api.py`

**Interfaces:**
- Consumes: Task 8 `AnalyticsQueryService`, PR 1 `require_admin` and `record_admin_profile_access`, and existing request/query parsing conventions.
- Produces: `GET /api/admin/analytics/overview`, `GET /api/admin/analytics/users`, `GET /api/admin/analytics/users/{user_id}`, and `GET /api/admin/analytics/users/{user_id}/activity`.

- [ ] **Step 1: Write failing API and authorization tests**

```python
def test_non_admin_cannot_query_analytics(client, signed_in_user):
    response = client.get("/api/admin/analytics/overview")
    assert response.status_code == 403


def test_admin_overview_accepts_documented_filters(admin_client, analytics_fixture):
    response = admin_client.get(
        "/api/admin/analytics/overview",
        params={"from": "2026-08-01", "to": "2026-08-26", "billing_mode": "byok", "provider": "openrouter", "model": "gpt-4.1", "include_internal": "false"},
    )
    assert response.status_code == 200
    assert response.json()["filters"]["billing_mode"] == "byok"


def test_profile_read_records_access_without_response_body(admin_client, analytics_fixture):
    response = admin_client.get(f"/api/admin/analytics/users/{analytics_fixture.user_id}")
    assert response.status_code == 200
    access = analytics_fixture.latest_admin_access()
    assert access["subject_user_id"] == analytics_fixture.user_id
    assert "response" not in access


def test_activity_sections_are_independently_pageable(admin_client, analytics_fixture):
    response = admin_client.get(f"/api/admin/analytics/users/{analytics_fixture.user_id}/activity", params={"section": "runs", "limit": 2})
    assert response.status_code == 200
    assert "next_cursor" in response.json()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest dashboard/backend/tests/test_admin_analytics_api.py -q`

Expected: FAIL because the router is not registered.

- [ ] **Step 3: Implement the gated router and safe response models**

```python
router = APIRouter(
    prefix="/admin/analytics",
    tags=["admin-analytics"],
    dependencies=[Depends(require_admin)],
)


@router.get("/overview", response_model=AnalyticsOverviewResponse)
def get_overview(..., service: AnalyticsQueryService = Depends(get_analytics_query_service)):
    return service.get_overview(filters=_filters_from_query(...))


@router.get("/users", response_model=AnalyticsUserListResponse)
def list_users(...):
    return service.list_users(filters=_user_filters_from_query(...), limit=limit, offset=offset)


@router.get("/users/{user_id}", response_model=AnalyticsUserProfileResponse)
def get_user_profile(user_id: int, request: Request, current_admin: dict = Depends(require_admin), ...):
    profile = service.get_user_profile(user_id=user_id)
    record_admin_profile_access(admin_user_id=current_admin["id"], subject_user_id=user_id, section="overview")
    return profile


@router.get("/users/{user_id}/activity", response_model=AnalyticsActivityPageResponse)
def get_user_activity(user_id: int, section: Literal["timeline", "runs", "usage", "sessions"], cursor: str | None = None, ...):
    record_admin_profile_access(admin_user_id=current_admin["id"], subject_user_id=user_id, section=section)
    return service.get_user_activity(user_id=user_id, section=section, limit=limit, cursor=cursor)
```

Use date-only query strings parsed to UTC midnight boundaries; reject `to < from`, unknown billing/provider/model filters, invalid status values, limits over 100, and malformed cursors with the existing safe 422/400 conventions. The router must never echo unknown query values in error details if they could contain sensitive text. Register it once in `api/router.py` and add route-pair coverage to the existing composition tests.

`GET /users` accepts `q`, `status`, `last_activity_from`, `last_activity_to`, `sort`, `order`, `limit`, and `offset`. Allow `sort` only for `last_activity`, `joined_at`, `recent_runs`, or `recent_failures`; allow `order` only for `asc` or `desc`. `GET /users/{user_id}/activity` accepts exactly one section and its own cursor, so the PR 3 page can paginate Timeline, Runs, Usage, and Sessions independently.

- [ ] **Step 4: Run API, route, and security tests**

Run: `pytest dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_app_composition.py dashboard/backend/tests/test_architecture_boundaries.py -q`

Expected: PASS; every Analytics route is Admin-gated, profile access is logged, cursors paginate independently, and response JSON has no prohibited fields.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/admin_analytics.py dashboard/backend/api/router.py dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_app_composition.py
git commit -m "feat: expose admin analytics query APIs"
```

### Task 10: Complete SQLite/PostgreSQL parity and end-to-end acceptance coverage

**Files:**
- Create: `dashboard/scripts/benchmark_analytics_queries.py`
- Test: `dashboard/backend/tests/domain/analytics/test_repository_contract.py`
- Test: `dashboard/backend/tests/test_analytics_integration.py`
- Test: `dashboard/backend/tests/test_admin_analytics_api.py`

**Interfaces:**
- Consumes: all prior tasks and a disposable PostgreSQL database when the `POSTGRES_TEST_DATABASE_URL` test environment is available.
- Produces: evidence that the complete PR 2 surface behaves equivalently on SQLite and PostgreSQL, that analytics failures are isolated, that the documented synthetic acceptance scenario passes without real credentials, and that the query targets can be measured on repeatable synthetic data.

- [ ] **Step 1: Add repository contract and end-to-end failing tests**

```python
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_analytics_repository_contract(backend, analytics_repository_factory):
    repo = analytics_repository_factory(backend)
    event = make_event(user_id=7, source_event_id="source-1")
    assert repo.append_event(event) is True
    assert repo.append_event(event) is False
    assert repo.list_events(AnalyticsEventFilters(user_id=7), limit=10, cursor=None).items[0].event_id == event.event_id


def test_synthetic_acceptance_scenario(analytics_app_fixture):
    analytics_app_fixture.create_user()
    analytics_app_fixture.create_and_verify_fake_byok_credential()
    analytics_app_fixture.record_failed_then_successful_run()
    analytics_app_fixture.record_platform_usage_and_settlement()
    overview = analytics_app_fixture.admin_overview()
    profile = analytics_app_fixture.admin_profile()
    assert overview["platform_model_cost_usd"] > 0
    assert profile["billing_lane_mix"]["byok"] == 1
    assert profile["billing_lane_mix"]["platform_credits"] == 1
    assert profile["credits_debited_micro"] > 0
    assert profile["state"]["evidence_event_ids"]
```

- [ ] **Step 2: Run the tests and verify the parity/acceptance cases fail or expose gaps**

Run: `pytest dashboard/backend/tests/domain/analytics/test_repository_contract.py dashboard/backend/tests/test_analytics_integration.py dashboard/backend/tests/test_admin_analytics_api.py -q`

Expected: Any failure identifies a concrete parity, privacy, or lifecycle gap; do not weaken assertions to accommodate backend divergence.

- [ ] **Step 3: Fix only contract-level parity defects**

Do not change PR 1 repository code on this branch. If parity tests expose a foundation defect, stop and report the exact failing contract to the PR 1 owner; resume PR 2 only after the foundation fix is merged or cherry-picked. Within PR 2 query/service code, normalize timestamps to the same UTC ISO representation, preserve duplicate source-event behavior, use the same cursor ordering `(occurred_at DESC, event_id DESC)`, and return identical null/empty conventions. Do not add joins across users/content/run databases; query services may compose separate stores in Python, as existing domain services do.

Create `benchmark_analytics_queries.py` with deterministic synthetic generation flags `--users`, `--events-per-user`, `--days`, and `--iterations`. It must use a disposable database path, print p50/p95 for overview and initial profile, and exit non-zero when overview p95 exceeds 1.0 second or profile p95 exceeds 0.5 seconds. Default to `--users 10000 --events-per-user 40 --days 180 --iterations 20`; never connect to a database URL passed on the command line.

- [ ] **Step 4: Run the full proportional verification set**

Run: `pytest dashboard/backend/tests/domain/analytics -q && pytest dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_analytics_integration.py dashboard/backend/tests/test_architecture_boundaries.py dashboard/backend/tests/test_app_composition.py -q && python dashboard/scripts/benchmark_analytics_queries.py --users 10000 --events-per-user 40 --days 180 --iterations 20 && pytest dashboard/backend/tests/ -q`

Expected: PASS. If PostgreSQL integration is unavailable, the contract test must report a clean skip using the repository's existing `_postgres_testing.py` convention; SQLite coverage remains mandatory.

- [ ] **Step 5: Commit the final verification-only changes**

```bash
git add dashboard/scripts/benchmark_analytics_queries.py dashboard/backend/tests/domain/analytics dashboard/backend/tests/test_admin_analytics_api.py dashboard/backend/tests/test_analytics_integration.py dashboard/backend/tests/test_architecture_boundaries.py dashboard/backend/tests/test_app_composition.py
git commit -m "test: verify analytics parity and acceptance flow"
```

## Self-Review Checklist

1. **Spec coverage:** Collection scope is covered by Tasks 1–4; metric formulas and rollups by Task 5; all five states and evidence by Task 6; authoritative 180-day backfill by Task 7; freshness/merge, filters, pagination, access logs, and partial failures by Tasks 8–9; SQLite/PostgreSQL parity and the synthetic acceptance scenario by Task 10. PR 3 UI is explicitly excluded.
2. **Privacy coverage:** Every event builder uses fixed allowlists; credential and execution hooks pass IDs/provider/model/usage evidence only; backfill sets all client-derived fields to null; API response models are display-safe; tests assert secret/body absence.
3. **Failure isolation:** Every source hook is post-commit and best-effort; startup maintenance is independently guarded; query panels return unavailable results rather than blanking the whole response.
4. **Idempotency:** Server retries use stable source-event IDs; rollups upsert bounded keys; snapshots overwrite current state; backfill replay inserts zero duplicates.
5. **Type consistency:** PR 1 types `AnalyticsEvent`, `AnalyticsRepository`, `append_event_best_effort`, and `record_admin_profile_access`, plus PR 2 types `AnalyticsMetricFilters`, `AnalyticsUserFilters`, `AnalyticsOverview`, `AnalyticsUserProfile`, and `AnalyticsActivityPageResponse`, are used consistently across all tasks.
6. **Placeholder scan:** Every task has concrete files, interfaces, test code, commands, and an English commit message; no deferred implementation marker or unspecified generic testing/error-handling step remains.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-admin-user-analytics-metrics.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, and keep each commit independently testable.
2. **Inline Execution** — execute the tasks in this session with executing-plans checkpoints.

Do not start implementation until the user confirms the plan and selects an execution option.
