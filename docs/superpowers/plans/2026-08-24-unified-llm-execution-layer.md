# Unified LLM Execution Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run dashboard backtests through real OpenAI, OpenRouter, Anthropic, Gemini, or approved OpenAI-compatible models with one server-side execution contract, correct Platform Credits settlement, and zero ATL Credits usage for BYOK.

**Architecture:** A typed execution service owns credential resolution, provider adapter selection, usage normalization, and billing. The API performs a credential preflight, then passes a signed, secret-free execution handoff to the worker; the worker resolves the encrypted secret only inside the child process immediately before constructing the provider client. Platform mode reserves and settles the existing Grant/Purchased ledger, while BYOK records usage evidence without touching the ATL Credits ledger. Explicit LLM runs fail visibly and never fall back to rule-based decisions.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLite, PostgreSQL/psycopg, `openai==1.101.0`, `anthropic==0.95.0`, `httpx==0.28.1`, existing encrypted credential vault, existing DNS/IP-pinning HTTP safety helpers, vanilla dashboard JavaScript.

**Spec:** `docs/superpowers/specs/2026-08-24-unified-llm-execution-layer-design.md`

## Global Constraints

- `platform_credits`: ATL uses a verified platform credential and charges ATL Credits after the provider reports usage.
- Platform billing is usage-based: final debit equals the provider/model pricing snapshot applied to the provider-reported input and output tokens, converted at the existing `$1 = 1 Credit` rate. There is no fixed per-run or per-call Credit charge.
- `byok`: ATL uses the caller's verified default credential and charges **zero** ATL Credits. The provider bill belongs to the caller.
- BYOK and Platform Credits are separate payment lanes. A BYOK run must never silently consume Grant Credits or Purchased Credits, and a Platform Credits run must never silently use a user's BYOK key.
- The request never accepts a raw API key from the browser.
- Full API keys stay inside the encrypted vault and short-lived server memory.
- No raw key appears in subprocess arguments, environment dumps, logs, URLs, frontend state, database rows, or error responses.
- Approved OpenAI-compatible origins retain the existing DNS/IP pinning rules; arbitrary custom origins are rejected.
- Missing usage on a Platform call fails closed for billing, releases the reservation, and does not apply the model result as a successful billable run.
- Missing usage on a BYOK call keeps the run result available and records usage as unavailable.
- User cancellation or worker crash finalizes outstanding Platform reservations idempotently.
- No automatic provider fallback, no live trading orders, no new provider onboarding UI, no Stripe semantic changes, and no charging BYOK users ATL Credits.
- Tests, logs, API responses, commits, and PR descriptions must contain only fake sentinel credentials such as `test-key-not-real`.
- The implementer writes and documents tests but does not run pytest, browser checks, or screenshots in this work session; the user runs those checks.

---

## File Map

- `dashboard/backend/infrastructure/llm/execution/models.py`: typed request, result, usage, pricing, and billing evidence objects.
- `dashboard/backend/infrastructure/llm/execution/errors.py`: fixed safe execution error categories.
- `dashboard/backend/infrastructure/llm/execution/handoff.py`: signed, one-time, secret-free parent/worker envelope.
- `dashboard/backend/infrastructure/llm/execution/adapters/base.py`: provider execution protocol and normalized response contract.
- `dashboard/backend/infrastructure/llm/execution/adapters/openai.py`: OpenAI wire adapter.
- `dashboard/backend/infrastructure/llm/execution/adapters/openrouter.py`: OpenRouter wire adapter.
- `dashboard/backend/infrastructure/llm/execution/adapters/anthropic.py`: Anthropic Messages adapter.
- `dashboard/backend/infrastructure/llm/execution/adapters/gemini.py`: Gemini REST adapter using `httpx`.
- `dashboard/backend/infrastructure/llm/execution/adapters/openai_compatible.py`: allowlisted custom-origin adapter.
- `dashboard/backend/infrastructure/llm/execution/adapters/registry.py`: provider-record to adapter mapping.
- `dashboard/backend/infrastructure/llm/execution/service.py`: credential resolution, provider call, usage evidence, and billing orchestration.
- `dashboard/backend/infrastructure/llm/token_cost.py`: immutable pricing snapshots and normalized token/cost helpers.
- `dashboard/backend/domain/model_providers/repository.py`: SQLite verified-default and verified-platform credential queries.
- `dashboard/backend/domain/model_providers/repository_postgres.py`: PostgreSQL twin queries and constraints.
- `dashboard/backend/domain/model_providers/service.py`: server-side credential resolution methods.
- `dashboard/backend/domain/credits/models.py`: LLM reservation and settlement result models.
- `dashboard/backend/domain/credits/repository.py`: SQLite reservation table, projection accounting, and idempotent settlement/release.
- `dashboard/backend/domain/credits/repository_postgres.py`: PostgreSQL reservation parity.
- `dashboard/backend/domain/credits/service.py`: public reservation/settlement/release orchestration.
- `dashboard/backend/api/routers/backtests.py`: explicit execution fields, authentication/preflight, handoff creation, and removal of old one-credit metering.
- `dashboard/scripts/backtest_hourly_agent.py`: worker argument parsing and execution-service construction.
- `dashboard/backend/domain/backtesting/engine.py`: per-call executor invocation and evidence accumulation.
- `dashboard/backend/domain/backtesting/portfolio_manager.py`: strict execution result parsing with no implicit fallback.
- `dashboard/backend/database.py`: SQLite non-secret run evidence upsert.
- `dashboard/backend/database_postgres.py`: PostgreSQL non-secret run evidence upsert.
- `dashboard/frontend/app.js`: billing mode/provider/model payload and visible failure rendering.
- `dashboard/frontend/app.html`: LLM execution controls and evidence labels.
- `dashboard/backend/tests/llm/test_execution_models.py`: typed contract tests.
- `dashboard/backend/tests/llm/test_credential_resolution.py`: credential lane and default-key tests.
- `dashboard/backend/tests/llm/test_execution_adapters.py`: adapter normalization and safe-error tests.
- `dashboard/backend/tests/llm/test_execution_service.py`: billing and failure-policy tests.
- `dashboard/backend/tests/domain/credits/test_llm_reservations.py`: SQLite reservation tests.
- `dashboard/backend/tests/domain/credits/test_llm_reservations_postgres.py`: PostgreSQL parity tests.
- `dashboard/backend/tests/test_backtest_execution_api.py`: API validation and handoff tests.
- `dashboard/backend/tests/llm/test_backtest_execution_wiring.py`: worker/engine/strict-mode tests.
- `dashboard/backend/tests/test_run_llm_evidence.py`: SQLite/PostgreSQL evidence persistence tests.
- `dashboard/backend/tests/test_frontend_llm_execution.py`: static frontend contract tests.

## Task 1: Add the typed execution contract

**Files:**
- Create: `dashboard/backend/infrastructure/llm/execution/__init__.py`
- Create: `dashboard/backend/infrastructure/llm/execution/models.py`
- Create: `dashboard/backend/infrastructure/llm/execution/errors.py`
- Test: `dashboard/backend/tests/llm/test_execution_models.py`

**Interfaces:**
- Consumes: existing provider ids and `dashboard.backend.infrastructure.llm.token_cost` model naming.
- Produces: `BillingMode`, `LLMMessage`, `UsagePolicy`, `LLMExecutionRequest`, `LLMUsage`, `PricingSnapshot`, `LLMExecutionResult`, `BillingEvidence`, `LLMExecutionError`.

- [ ] **Step 1: Write the failing test**

```python
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingMode, LLMExecutionRequest, LLMMessage,
)


def test_request_contains_lane_and_never_accepts_raw_secret():
    request = LLMExecutionRequest(
        user_id=7,
        run_id="run-1",
        call_index=0,
        billing_mode=BillingMode.BYOK,
        provider_id="openai",
        model_id="gpt-4o-mini",
        messages=(LLMMessage(role="user", content="hello"),),
        max_output_tokens=128,
    )
    assert request.billing_mode is BillingMode.BYOK
    assert "api_key" not in request.model_dump()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_execution_models.py::test_request_contains_lane_and_never_accepts_raw_secret -v`

Expected: FAIL with `ModuleNotFoundError` because the execution package does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement immutable Pydantic models. Restrict `billing_mode` to `platform_credits|byok`, validate non-empty provider/model/run ids, bound `max_output_tokens` to the configured worker maximum, and omit any secret-bearing field from every public model. Define fixed error categories: `credential_missing`, `credential_invalid`, `provider_unavailable`, `provider_timeout`, `response_invalid`, `usage_unavailable`, `billing_failed`, and `worker_failed`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_execution_models.py -v`

Expected: PASS for model validation, serialization, and safe error-category tests.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/execution dashboard/backend/tests/llm/test_execution_models.py
git commit -m "feat: add typed llm execution contract"
```

## Task 2: Resolve verified credentials by billing lane

**Files:**
- Modify: `dashboard/backend/domain/model_providers/repository.py`
- Modify: `dashboard/backend/domain/model_providers/repository_postgres.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Create: `dashboard/backend/tests/llm/test_credential_resolution.py`

**Interfaces:**
- Consumes: existing encrypted `get_user_credential_secret`, `get_platform_credential_secret`, provider enable flags, and `UserCredentialPublic` records.
- Produces: `ModelProviderService.resolve_user_default_credential(user_id: int, provider_id: str) -> ResolvedCredential` and `ModelProviderService.resolve_platform_credential(provider_id: str) -> ResolvedCredential`.

- [ ] **Step 1: Write the failing test**

```python
def test_byok_resolves_only_verified_default_for_same_user(provider_store):
    service = ModelProviderService(store=provider_store)
    resolved = service.resolve_user_default_credential(7, "openai")
    assert resolved.credential_id == "cred-openai-default"
    assert resolved.secret == "test-key-not-real"


def test_platform_resolution_rejects_unverified_platform_credential(provider_store):
    service = ModelProviderService(store=provider_store)
    with pytest.raises(CredentialResolutionError) as exc:
        service.resolve_platform_credential("openai")
    assert exc.value.category == "credential_invalid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_credential_resolution.py -v`

Expected: FAIL because the repository queries and service methods are not defined.

- [ ] **Step 3: Write minimal implementation**

Add SQLite and PostgreSQL queries that select a credential only when provider status is `enabled`, the lane flag is enabled, credential status is `verified`, and `is_default=1` for BYOK. Return no row when the user owns zero or multiple verified defaults; enforce the existing database uniqueness rule so exactly one default can exist. Add a verified-status check to platform resolution. Decrypt only after the row has passed ownership, status, and provider checks. Return a `ResolvedCredential` containing `credential_id`, `provider_id`, `label`, `key_last_four`, and the transient secret; never return it from an API route.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_credential_resolution.py -v`

Expected: PASS for user ownership, default selection, provider flags, revoked keys, and platform credential status.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/model_providers dashboard/backend/tests/llm/test_credential_resolution.py
git commit -m "feat: resolve verified llm credentials by lane"
```

## Task 3: Build provider execution adapters

**Files:**
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/base.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/openai.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/openrouter.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/anthropic.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/gemini.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/openai_compatible.py`
- Create: `dashboard/backend/infrastructure/llm/execution/adapters/registry.py`
- Test: `dashboard/backend/tests/llm/test_execution_adapters.py`

**Interfaces:**
- Consumes: `LLMExecutionRequest`, `ResolvedCredential`, existing provider registry records, and `safe_http` DNS/IP-pinning helpers.
- Produces: `ProviderExecutionAdapter.complete(request, credential, provider) -> AdapterResponse`, `get_execution_adapter(provider_record)`, and fixed safe adapter errors.

- [ ] **Step 1: Write the failing test**

```python
def test_openai_adapter_normalizes_text_and_usage(fake_openai_client):
    adapter = OpenAIExecutionAdapter(client_factory=lambda **_: fake_openai_client)
    response = adapter.complete(_request("openai", "gpt-4o-mini"), _credential(), _provider("openai"))
    assert response.text == '{"actions": []}'
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 8


def test_custom_origin_is_rejected_before_network_call():
    with pytest.raises(ProviderExecutionError) as exc:
        get_execution_adapter(_provider("custom", approved_base_url="http://127.0.0.1:9"))
    assert exc.value.category == "provider_unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_execution_adapters.py -v`

Expected: FAIL because execution adapters and registry are not present.

- [ ] **Step 3: Write minimal implementation**

Implement one adapter protocol that accepts normalized messages and returns text plus usage. Use `openai.OpenAI` for OpenAI and OpenRouter, passing only the transient secret into client construction. Use `anthropic.Anthropic` for Anthropic Messages. Use `httpx.Client` for Gemini's REST `generateContent` endpoint and extract `usageMetadata`. Route approved OpenAI-compatible providers through the OpenAI client with the provider's allowlisted HTTPS origin. Reuse `safe_http` origin validation and DNS/IP pinning; reject arbitrary schemes, redirects, loopback, link-local, and private destinations. Map SDK/network/HTTP errors to the fixed categories without copying response bodies. The adapter never rounds usage to a run-level flat fee; it returns the provider's token counts unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_execution_adapters.py -v`

Expected: PASS for all four official providers, approved custom origins, text extraction, usage extraction, timeout mapping, and secret-free errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/execution/adapters dashboard/backend/tests/llm/test_execution_adapters.py
git commit -m "feat: add normalized provider execution adapters"
```

## Task 4: Normalize usage and pricing evidence

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/token_cost.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/models.py`
- Create: `dashboard/backend/tests/llm/test_token_cost_evidence.py`

**Interfaces:**
- Consumes: existing `price_for_model`, `estimate_cost_usd`, and provider usage payloads.
- Produces: `PricingSnapshot.from_model(model_id)`, `normalize_usage(payload)`, `estimate_cost_from_snapshot(snapshot, usage)`, and `build_cost_evidence(...)`.

- [ ] **Step 1: Write the failing test**

```python
def test_cost_evidence_keeps_snapshot_and_authority():
    snapshot = PricingSnapshot.from_model("gpt-4o-mini")
    evidence = build_cost_evidence(
        provider="openai",
        model="gpt-4o-mini",
        usage=LLMUsage(input_tokens=100, output_tokens=50),
        provider_cost_usd=None,
        pricing_snapshot=snapshot,
    )
    assert evidence.usage_authority == "provider_usage_pricing_snapshot"
    assert evidence.provider_cost_usd is None
    assert evidence.estimated_cost_usd > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_token_cost_evidence.py -v`

Expected: FAIL because snapshot and authority helpers are not defined.

- [ ] **Step 3: Write minimal implementation**

Make pricing immutable and serializable with model id, input/output USD per million tokens, currency, and source version. Normalize `prompt_tokens|completion_tokens`, `input_tokens|output_tokens`, and Gemini `usageMetadata` into one `LLMUsage`; reject negative or non-integer values. Prefer provider-reported cost when present, otherwise estimate from the captured snapshot. Mark unavailable usage explicitly instead of treating it as zero.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_token_cost_evidence.py -v`

Expected: PASS for OpenAI, Anthropic, Gemini, OpenRouter-style payloads, unknown models, and unavailable usage.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/token_cost.py dashboard/backend/infrastructure/llm/execution/models.py dashboard/backend/tests/llm/test_token_cost_evidence.py
git commit -m "feat: normalize llm usage and pricing evidence"
```

## Task 5: Add Platform Credits reservations and settlement

**Files:**
- Modify: `dashboard/backend/domain/credits/models.py`
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Modify: `dashboard/backend/domain/credits/service.py`
- Create: `dashboard/backend/tests/domain/credits/test_llm_reservations.py`
- Create: `dashboard/backend/tests/domain/credits/test_llm_reservations_postgres.py`

**Interfaces:**
- Consumes: existing Grant/Purchased bucket projection, append-only `credit_ledger_entries`, and store idempotency conventions.
- Produces: `LLMReservation`, `CreditsService.reserve_llm_credits(...)`, `CreditsService.settle_llm_credits(...)`, `CreditsService.release_llm_credits(...)`, and `CreditsService.release_run_llm_reservations(run_id)`.

- [ ] **Step 1: Write the failing test**

```python
def test_platform_reservation_reduces_available_then_settles(store):
    service = CreditsService(store=store)
    reservation = service.reserve_llm_credits(
        user_id=7, run_id="run-1", call_index=0, amount_micro=2_000_000,
    )
    assert service.get_balance(7).balance_micro == 8_000_000
    settled = service.settle_llm_credits(
        reservation.reservation_id, actual_micro=1_250_000,
        evidence={"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert settled.debited_micro == 1_250_000
    assert service.get_balance(7).balance_micro == 8_750_000


def test_release_is_idempotent_and_creates_no_debit(store):
    service = CreditsService(store=store)
    reservation = service.reserve_llm_credits(
        user_id=7, run_id="run-2", call_index=0, amount_micro=500_000,
    )
    service.release_llm_credits(reservation.reservation_id, reason="provider_timeout")
    service.release_llm_credits(reservation.reservation_id, reason="provider_timeout")
    assert service.get_balance(7).balance_micro == 10_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/domain/credits/test_llm_reservations.py -v`

Expected: FAIL because the reservation table and service methods do not exist.

- [ ] **Step 3: Write minimal implementation**

Create `credit_llm_reservations` and an append-only `credit_llm_usage_entries` sub-ledger in both stores. Keep the historical Purchase/Refund/Grant ledger table unchanged so enabling model spending does not require rebuilding a foreign-keyed SQLite ledger. A reservation is only a temporary ceiling calculated from the captured model pricing snapshot and the request's bounded maximum output; it is not a charge. Settlement atomically computes `actual_micro = provider_cost_usd` when authoritative, otherwise the exact input/output-token price calculation, converts that USD amount at `$1 = 1 Credit`, applies Grant-first/Purchased-second bucket ordering, appends negative usage rows to the LLM sub-ledger, marks the reservation settled, and releases the unused ceiling. Balance and Grant Pool projections combine the historical ledger with the LLM usage sub-ledger. Release marks a reservation released without a usage row. Enforce `actual_micro <= reserved_micro`, positive reservation amounts, account restrictions, row locks/`BEGIN IMMEDIATE`, and idempotent replay for both SQLite and PostgreSQL. Tests must prove a 10-token call costs less than a 10,000-token call and that two calls with the same run length can debit different Credit amounts.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/domain/credits/test_llm_reservations.py dashboard/backend/tests/domain/credits/test_llm_reservations_postgres.py -v`

Expected: PASS for bucket ordering, insufficient balance, duplicate operation keys, settle/release replay, rollback, and SQLite/PostgreSQL parity.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/credits dashboard/backend/tests/domain/credits/test_llm_reservations.py dashboard/backend/tests/domain/credits/test_llm_reservations_postgres.py
git commit -m "feat: add idempotent llm credit reservations"
```

## Task 6: Implement the unified execution service

**Files:**
- Create: `dashboard/backend/infrastructure/llm/execution/service.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/__init__.py`
- Create: `dashboard/backend/tests/llm/test_execution_service.py`

**Interfaces:**
- Consumes: `LLMExecutionRequest`, `ModelProviderService.resolve_*_credential`, adapter registry, pricing evidence, and CreditsService reservation methods.
- Produces: `LLMExecutionService.execute(request) -> LLMExecutionResult`, `LLMExecutionService.finalize_run(run_id)`, and one sanitized exception type for callers.

- [ ] **Step 1: Write the failing test**

```python
def test_byok_executes_without_credit_store_calls(fake_provider, credential_service, credit_service):
    service = LLMExecutionService(
        providers=credential_service, credits=credit_service,
        adapter_resolver=lambda _: fake_provider,
    )
    result = service.execute(_request(billing_mode="byok"))
    assert result.billing_source == "byok"
    credit_service.assert_no_mutations()


def test_platform_missing_usage_releases_reservation(fake_provider, credential_service, credit_service):
    fake_provider.response.usage = None
    service = LLMExecutionService(
        providers=credential_service, credits=credit_service,
        adapter_resolver=lambda _: fake_provider,
    )
    with pytest.raises(LLMExecutionError) as exc:
        service.execute(_request(billing_mode="platform_credits"))
    assert exc.value.category == "usage_unavailable"
    credit_service.assert_released_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_execution_service.py -v`

Expected: FAIL because `LLMExecutionService` is not defined.

- [ ] **Step 3: Write minimal implementation**

Resolve the selected lane before constructing a client. For Platform mode, capture a pricing snapshot, reserve only a temporary upper bound before the provider call, execute once, require authoritative usage, calculate the exact model cost from provider-reported usage, settle that exact amount at `$1 = 1 Credit`, and release unused reservation capacity. No execution path may debit a fixed amount merely because a run or call started. For BYOK mode, execute with the verified default key and never call a Credits mutation. Normalize adapter responses into `LLMExecutionResult`, preserve provider/model/credential-safe identity, and map all failures to fixed categories. `finalize_run` must release every open reservation for the run and be safe to call repeatedly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_execution_service.py -v`

Expected: PASS for both lanes, reservation lifecycle, provider failure, timeout, invalid response, no-call release, cancellation finalization, and no-secret result serialization.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/llm/execution dashboard/backend/tests/llm/test_execution_service.py
git commit -m "feat: orchestrate llm execution and billing"
```

## Task 7: Add the signed, secret-free worker handoff

**Files:**
- Create: `dashboard/backend/infrastructure/llm/execution/handoff.py`
- Modify: `dashboard/backend/api/routers/backtests.py`
- Modify: `dashboard/scripts/backtest_hourly_agent.py`
- Create: `dashboard/backend/tests/test_backtest_execution_api.py`

**Interfaces:**
- Consumes: authenticated user id, explicit billing/provider/model selection, and the existing background subprocess launcher.
- Produces: `create_execution_handoff(...) -> str`, `consume_execution_handoff(payload) -> ExecutionHandoff`, and worker arguments containing no credential secret.

- [ ] **Step 1: Write the failing test**

```python
def test_handoff_contains_identity_but_not_secret(monkeypatch):
    payload = create_execution_handoff(
        user_id=7, run_id="run-1", billing_mode="byok",
        provider_id="openai", model_id="gpt-4o-mini",
    )
    assert "test-key-not-real" not in payload
    handoff = consume_execution_handoff(payload)
    assert handoff.user_id == 7
    assert handoff.provider_id == "openai"


def test_llm_request_rejects_missing_authenticated_user(api_client):
    response = api_client.post("/backtest/run", json={
        "decision_source": "llm", "billing_mode": "byok",
        "provider_id": "openai", "model": "gpt-4o-mini",
    })
    assert response.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtest_execution_api.py -v`

Expected: FAIL because the handoff functions and new request fields are absent.

- [ ] **Step 3: Write minimal implementation**

Extend `BacktestRunRequest` and the route merge logic with `billing_mode: Literal['platform_credits','byok']` and `provider_id`. Require an authenticated user for explicit LLM runs, validate provider/model syntax, and preflight the requested lane before acquiring a slot or starting a thread. Replace `authorize_llm_run` and `refund_llm_run` calls with the execution service lifecycle. Create a short-lived HMAC-signed JSON envelope containing user id, run id, lane, provider, model, nonce, expiry, and a digest of prompt/pipeline metadata; send it through the child process stdin, never command arguments or environment variables. Consume once, reject replay/expiry/signature mismatch, and zero the in-memory payload after credential construction.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/test_backtest_execution_api.py -v`

Expected: PASS for request precedence, authenticated BYOK/platform preflight, no old quota calls, safe subprocess argv/env, one-time handoff consumption, and visible 4xx/5xx errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/infrastructure/llm/execution/handoff.py dashboard/scripts/backtest_hourly_agent.py dashboard/backend/tests/test_backtest_execution_api.py
git commit -m "feat: add secure llm worker handoff"
```

## Task 8: Wire the execution service into the backtest engine

**Files:**
- Modify: `dashboard/backend/domain/backtesting/engine.py`
- Modify: `dashboard/backend/domain/backtesting/portfolio_manager.py`
- Modify: `dashboard/scripts/backtest_hourly_agent.py`
- Create: `dashboard/backend/tests/llm/test_backtest_execution_wiring.py`

**Interfaces:**
- Consumes: `LLMExecutionService.execute`, `ExecutionHandoff`, existing portfolio prompt construction, and action parsing.
- Produces: strict per-call model decisions, accumulated normalized usage, and a finalizer call in worker `finally` blocks.

- [ ] **Step 1: Write the failing test**

```python
def test_explicit_llm_failure_is_not_rule_based_success(fake_executor):
    fake_executor.execute.side_effect = LLMExecutionError("provider_timeout", "Provider timed out")
    manager = _manager_with_executor(fake_executor)
    with pytest.raises(LLMDecisionError, match="Provider timed out"):
        manager.make_trading_decision_with_llm(_portfolio_state(), mode="safe_trading")


def test_engine_passes_provider_and_lane_to_every_call(fake_executor):
    engine = _engine_with_executor(fake_executor)
    engine.run()
    assert all(call.billing_mode == "byok" for call in fake_executor.requests)
    assert all(call.provider_id == "openai" for call in fake_executor.requests)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_backtest_execution_wiring.py -v`

Expected: FAIL because the engine still builds an environment-only client and the portfolio manager still permits fallback.

- [ ] **Step 3: Write minimal implementation**

Add an execution context to the engine containing user id, run id, lane, provider, model, and service. Replace the direct `make_llm_client()` path for dashboard LLM runs with an executor callback that receives the generated system/user messages and call index. Preserve rule-based behavior only when `decision_source='rule_based'`; when `decision_source='llm'`, missing client, provider error, parse error, and unavailable usage raise `LLMDecisionError`. Accumulate `LLMExecutionResult` evidence and invoke `finalize_run(run_id)` in every worker `finally` path, including subprocess failure and cancellation.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_backtest_execution_wiring.py -v`

Expected: PASS for real executor delegation, strict failure, call indexes, evidence accumulation, no silent fallback, and idempotent finalization.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py dashboard/backend/domain/backtesting/portfolio_manager.py dashboard/scripts/backtest_hourly_agent.py dashboard/backend/tests/llm/test_backtest_execution_wiring.py
git commit -m "feat: wire unified execution into backtests"
```

## Task 9: Persist non-secret execution evidence and failure state

**Files:**
- Modify: `dashboard/backend/database.py`
- Modify: `dashboard/backend/database_postgres.py`
- Modify: `dashboard/backend/domain/backtesting/engine.py`
- Modify: `dashboard/backend/api/routers/backtests.py`
- Create: `dashboard/backend/tests/test_run_llm_evidence.py`

**Interfaces:**
- Consumes: accumulated `LLMExecutionResult` values and sanitized execution errors.
- Produces: `BacktestDatabase.update_run_llm_evidence(run_id, evidence)` and equivalent PostgreSQL behavior, storing lane, provider, model, credential id/key-last-four, token totals, provider cost, pricing snapshot, billing status, usage status, and failure category without secrets.

- [ ] **Step 1: Write the failing test**

```python
def test_run_evidence_round_trips_without_secret(sqlite_db):
    sqlite_db.update_run_llm_evidence("run-1", {
        "billing_mode": "byok", "provider": "openai", "model": "gpt-4o-mini",
        "credential_id": "cred-1", "key_last_four": "1234",
        "input_tokens": 12, "output_tokens": 8,
    })
    row = sqlite_db.get_run("run-1")
    assert row["metadata"]["llm_execution"]["billing_mode"] == "byok"
    assert "test-key-not-real" not in json.dumps(row)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_run_llm_evidence.py -v`

Expected: FAIL because neither database twin exposes the evidence upsert.

- [ ] **Step 3: Write minimal implementation**

Add an idempotent metadata merge keyed by `metadata.llm_execution`, preserving existing run metadata and baseline links. Store only safe identity and usage fields; store pricing as a numeric snapshot, not a provider response. Add a sanitized failed-run update so preflight, provider, parse, usage, worker, and billing failures remain visible in status and run history. Ensure SQLite and PostgreSQL use the same JSON shape and transaction semantics.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/test_run_llm_evidence.py -v`

Expected: PASS for success, partial usage, failed runs, repeated updates, and SQLite/PostgreSQL JSON parity.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/database.py dashboard/backend/database_postgres.py dashboard/backend/domain/backtesting/engine.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_run_llm_evidence.py
git commit -m "feat: persist non-secret llm run evidence"
```

## Task 10: Remove the old fixed-fee quota path from dashboard backtests

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py`
- Modify: `dashboard/backend/domain/entitlements/credits.py` only where dashboard callers remain exposed
- Modify: `dashboard/backend/tests/test_credit_metering.py` or add a focused regression test under `dashboard/backend/tests/llm/`

**Interfaces:**
- Consumes: the new execution service reservation API.
- Produces: no dashboard LLM call to `authorize_llm_run` or `refund_llm_run`; legacy entitlement helpers remain isolated for callers that still explicitly use them.

- [ ] **Step 1: Write the failing test**

```python
def test_dashboard_platform_run_does_not_use_fixed_run_fee(monkeypatch, api_client):
    monkeypatch.setattr(credits, "authorize_llm_run", lambda *_: pytest.fail("legacy meter called"))
    response = api_client.post("/backtest/run", json={
        "decision_source": "llm", "billing_mode": "platform_credits",
        "provider_id": "openai", "model": "gpt-4o-mini",
    })
    assert response.status_code in {200, 202}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_dashboard_metering_regression.py -v`

Expected: FAIL because the route still invokes the legacy fixed-fee authorization path.

- [ ] **Step 3: Write minimal implementation**

Delete the dashboard route's fixed-fee authorization/refund block and its `charged_credit_user_id` thread argument. Keep the old entitlement module only for unrelated legacy surfaces until each caller is migrated, and add a comment at the route boundary stating that all dashboard LLM billing belongs to `LLMExecutionService`, where actual provider usage determines the final debit.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_dashboard_metering_regression.py -v`

Expected: PASS with no legacy meter invocation, no accidental Credit mutation for BYOK, and Platform Credits debited only after the execution service has exact usage-based cost evidence.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/domain/entitlements/credits.py dashboard/backend/tests/llm/test_dashboard_metering_regression.py
git commit -m "refactor: retire fixed dashboard llm quota"
```

## Task 11: Add frontend lane and provider/model selection

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Create: `dashboard/frontend/js/llm-execution.js` if the current backtest controls are too large for `app.js`
- Test: `dashboard/backend/tests/test_frontend_llm_execution.py`

**Interfaces:**
- Consumes: existing provider/credential list endpoints and the extended `/backtest/run` request.
- Produces: explicit `billing_mode`, `provider_id`, and model payload fields, provider availability labels, and safe visible errors.

- [ ] **Step 1: Write the failing test**

```python
def test_llm_payload_contains_selected_lane_and_provider():
    source = Path("dashboard/frontend/app.js").read_text()
    assert "billing_mode" in source
    assert "provider_id" in source
    assert "decision_source" in source


def test_frontend_does_not_store_raw_api_keys_in_run_payload():
    source = Path("dashboard/frontend/app.js").read_text()
    assert "api_key" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_frontend_llm_execution.py -v`

Expected: FAIL because the current run form sends only the legacy decision/model fields.

- [ ] **Step 3: Write minimal implementation**

Add a compact billing selector with `Platform Credits` and `BYOK`, provider select filtered by enabled capabilities, and model input constrained to the server model-id grammar. In BYOK mode show only provider/key-last-four/default status; never copy a raw key into form state or payload. In Platform mode show the provider's platform availability. Render preflight/provider/usage/billing errors as explicit status text and keep the existing rule-based form behavior unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/test_frontend_llm_execution.py -v`

Expected: PASS for payload shape, lane switching, disabled-provider handling, and raw-secret exclusion.

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/js/llm-execution.js dashboard/backend/tests/test_frontend_llm_execution.py
git commit -m "feat: add dashboard llm billing selection"
```

## Task 12: Complete parity and security regression coverage

**Files:**
- Modify: `dashboard/backend/tests/test_backtest_db_postgres.py` only for shared fixtures/helpers when needed
- Create: `dashboard/backend/tests/llm/test_execution_security.py`
- Create: `dashboard/backend/tests/llm/test_execution_integration.py`
- Modify: `docs/superpowers/specs/2026-08-24-unified-llm-execution-layer-design.md` to mark implemented acceptance evidence after code review

**Interfaces:**
- Consumes: every execution, credential, billing, worker, persistence, and frontend contract from Tasks 1–11.
- Produces: a reviewable acceptance matrix and deterministic secret-safety checks.

- [ ] **Step 1: Write the failing test**

```python
def test_secret_safety_scan_rejects_only_real_secret_sources():
    forbidden = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"}
    for path in tracked_runtime_files():
        text = path.read_text()
        assert not any(value in text for value in forbidden)


def test_byok_integration_leaves_grant_and_purchased_balances_unchanged(fake_adapters):
    before = balance_snapshot(user_id=7)
    run = run_backtest_with_lane("byok", fake_adapters)
    assert run.evidence["billing_mode"] == "byok"
    assert balance_snapshot(user_id=7) == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/llm/test_execution_security.py dashboard/backend/tests/llm/test_execution_integration.py -v`

Expected: FAIL until all provider lanes, worker handoff, and secret-redaction paths are wired.

- [ ] **Step 3: Write minimal implementation**

Add deterministic fake-provider integration fixtures for OpenAI BYOK, OpenAI Platform Credits, OpenRouter, Anthropic, Gemini, and approved OpenAI-compatible providers. Assert normalized result shape, exact balance deltas, reservation release on every failure category, no fallback decision, safe errors, replay-safe finalization, and absence of raw credential strings in subprocess captures, logs, responses, metadata, commits, and PR text. Keep all fixture credentials as `test-key-not-real`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest dashboard/backend/tests/llm/test_execution_security.py dashboard/backend/tests/llm/test_execution_integration.py -v`

Expected: PASS for every acceptance criterion in the design document, with SQLite and PostgreSQL test matrices producing the same evidence and balance behavior.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/tests/llm dashboard/backend/tests/test_backtest_db_postgres.py docs/superpowers/specs/2026-08-24-unified-llm-execution-layer-design.md
git commit -m "test: cover unified llm execution acceptance"
```

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-24-unified-llm-execution-layer.md`. Two execution options:

1. **Subagent-Driven** — dispatch one fresh implementation agent per task with review gates.
2. **Inline Execution** — execute tasks in this session in small batches with checkpoints.

The user requested to run tests personally, so either execution option must stop after code/static checks and hand over the listed pytest/browser commands for user validation.
