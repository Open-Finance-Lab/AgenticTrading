# Platform Credits Environment Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Platform Credits use the existing server `OPENROUTER_API_KEY` when no verified stored Platform Credential exists, while preserving BYOK isolation and Grant-first usage billing.

**Architecture:** Keep credential resolution inside `ModelProviderService`. Add one explicit provider-to-environment mapping for OpenRouter, prefer the encrypted verified Platform Credential, and use the environment value only as a transient fallback after `platform_enabled=true` has been checked. Reuse the existing execution service, provider adapter, reservation, settlement, release, and frontend contracts; no frontend behavior or database schema changes are needed.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, SQLite/PostgreSQL repositories, pytest, existing OpenAI-compatible execution adapter.

**Spec:** `docs/superpowers/specs/2026-08-25-platform-credits-environment-fallback-design.md`

## Global Constraints

- BYOK uses only the user's verified default credential and never debits ATL Credits.
- Platform Credits requires an enabled provider with `platform_enabled=true`.
- Stored verified Platform Credential wins over the environment fallback.
- Only `openrouter` maps to `OPENROUTER_API_KEY` in this change.
- Environment secrets are transient: never persist, return in API responses, log, or put in browser storage/source.
- Platform Credits must reserve a ceiling, settle actual usage, and release on failure.
- Ledger debit order is Grant Credits first, then Purchased Credits.
- Do not add dependencies, change pricing, or modify the database schema.
- Do not touch or commit `dashboard/storage/data/backtest.db`, `.superpowers/`, or `work/`.
- The user runs pytest and browser checks; the implementation session only provides commands and inspects user-reported results.

---

### Task 1: Add the explicit environment-backed platform credential resolver

**Files:**
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py`

**Interfaces:**
- Add a private constant mapping `{"openrouter": "OPENROUTER_API_KEY"}`.
- Add a private helper `_environment_platform_secret(provider_id: str) -> str | None` that returns a stripped non-empty value only for the mapped provider, without logging.
- Update `ModelProviderService.resolve_platform_credential(provider_id: str) -> ResolvedCredential` to keep the existing provider/platform flag checks, prefer `store.get_verified_platform_credential(provider_id)`, and then resolve the environment fallback.
- Update `ModelProviderService.preflight_platform_credential(provider_id: str) -> None` with the same availability rule without decrypting a stored secret.

- [ ] **Step 1: Write failing direct resolver tests**

Add these tests to `dashboard/backend/tests/domain/model_providers/test_service.py`:

```python
def test_platform_resolver_uses_openrouter_environment_key_when_store_is_empty(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    resolved = service.resolve_platform_credential("openrouter")

    assert resolved.credential_id is None
    assert resolved.key_last_four == "abcd"
    assert resolved.secret == "sk-or-env-test-abcd"


def test_verified_stored_platform_credential_precedes_environment_key(
    tmp_path, monkeypatch
):
    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=True)
    store.upsert_platform_credential(
        provider_id="openrouter",
        secret="sk-or-stored-test-wxyz",
        status="verified",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    resolved = service.resolve_platform_credential("openrouter")

    assert resolved.key_last_four == "wxyz"
    assert resolved.secret == "sk-or-stored-test-wxyz"


def test_environment_fallback_is_provider_specific_and_fails_closed(
    tmp_path, monkeypatch
):
    from dashboard.backend.domain.model_providers.service import CredentialResolutionError
    from dashboard.backend.infrastructure.llm.execution.errors import ExecutionErrorCategory

    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openai", byok_enabled=True, platform_enabled=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    with pytest.raises(CredentialResolutionError) as exc_info:
        service.resolve_platform_credential("openai")
    assert exc_info.value.category is ExecutionErrorCategory.CREDENTIAL_MISSING


def test_environment_fallback_requires_platform_enabled(tmp_path, monkeypatch):
    from dashboard.backend.domain.model_providers.service import CredentialResolutionError
    from dashboard.backend.infrastructure.llm.execution.errors import ExecutionErrorCategory

    service, store = _service(tmp_path, FakeAdapter())
    _set_provider_modes(store, "openrouter", byok_enabled=True, platform_enabled=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-test-abcd")

    with pytest.raises(CredentialResolutionError) as exc_info:
        service.preflight_platform_credential("openrouter")
    assert exc_info.value.category is ExecutionErrorCategory.CREDENTIAL_MISSING
```

The tests import `CredentialResolutionError` and `ExecutionErrorCategory` and
assert the fixed `credential_missing` category without inspecting
secret-bearing exception text.

- [ ] **Step 2: Run the focused tests and verify they fail**

User-run command:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  -k "environment_fallback or stored_platform_credential_precedes"
```

Expected: the new tests fail because the service currently requires a verified stored credential and does not read `OPENROUTER_API_KEY`.

- [ ] **Step 3: Implement the minimal resolver change**

In `service.py`, keep provider validation unchanged, then use this order:

```python
stored = self.store.get_verified_platform_credential(provider_id)
if stored:
    return ResolvedCredential(
        credential_id=None,
        provider_id=str(stored["provider_id"]),
        key_last_four=str(stored["key_last_four"])[-4:],
        secret=str(stored["secret"]),
    )

secret = _environment_platform_secret(provider_id)
if secret:
    return ResolvedCredential(
        credential_id=None,
        provider_id=provider_id,
        key_last_four=secret[-4:],
        secret=secret,
    )
raise CredentialResolutionError(ExecutionErrorCategory.CREDENTIAL_MISSING)
```

The helper must return `None` for every provider except `openrouter`, strip whitespace, and treat an empty value as missing. `preflight_platform_credential` must check the public stored status first and then the helper, without calling a decrypting repository method.

- [ ] **Step 4: Give the user the focused test command**

Do not run it in the implementation session. Ask the user to run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  -k "environment_fallback or stored_platform_credential_precedes"
```

Expected: PASS for all new resolver tests. Stop immediately if any test fails.

- [ ] **Step 5: Commit the isolated resolver layer**

```bash
git add dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/tests/domain/model_providers/test_service.py
git commit -m "feat: allow platform credits to use env provider key"
```

---

### Task 2: Expose safe Platform Credits availability through execution options

**Files:**
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py`
- Test: `dashboard/backend/tests/test_model_credentials_api.py`

**Interfaces:**
- `ModelProviderService.list_execution_options(user_id: int) -> list[ExecutionProviderOption]` must set `platform_credits_available` for OpenRouter when `platform_enabled=true` and either a verified stored credential or a non-empty `OPENROUTER_API_KEY` exists.
- The existing `GET /api/credits/execution-options` response shape remains unchanged.

- [ ] **Step 1: Write failing availability and secrecy tests**

Add service-level cases covering environment-only availability, missing environment unavailability, and stored-credential precedence. Add an API-level case that sets a fake sentinel with `monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-api-test-abcd")`, enables OpenRouter Platform Credits in the fixture store, calls `GET /api/credits/execution-options`, and asserts:

```python
assert response.status_code == 200
item = next(item for item in response.json()["providers"] if item["provider_id"] == "openrouter")
assert item["platform_credits_available"] is True
assert "sk-or-api-test-abcd" not in response.text
assert "api_key" not in response.text.lower()
```

Also add the inverse case with the environment variable removed and no stored credential; `platform_credits_available` must be false.

- [ ] **Step 2: Run the focused API tests and verify they fail**

User-run command:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  -k "execution_options or environment"
```

Expected: the environment-only availability assertion fails while the existing stored-credential behavior remains green.

- [ ] **Step 3: Implement availability using the same helper**

In `list_execution_options`, keep the existing public-credential lookup and compute:

```python
platform_credential_available = bool(
    platform
    and platform["status"] == "verified"
)
environment_available = bool(_environment_platform_secret(provider.provider_id))
platform_credits_available = (
    provider.platform_enabled
    and (platform_credential_available or environment_available)
)
```

Do not include the environment value, last four, or variable name in the response. Keep model routes unchanged.

- [ ] **Step 4: Give the user the API test command**

User-run command:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  -k "execution_options or environment"
```

Expected: PASS, with no complete sentinel key in the response. Stop on failure.

- [ ] **Step 5: Commit the availability layer**

```bash
git add dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py
git commit -m "feat: expose env-backed platform credit availability"
```

---

### Task 3: Verify the real Platform Credits execution seam and Grant-first settlement

**Files:**
- Modify: `dashboard/backend/tests/test_credit_metering.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py` (create)

**Interfaces:**
- `LLMExecutionService.execute()` continues to call `ModelProviderService.resolve_platform_credential()` for `BillingMode.PLATFORM_CREDITS`.
- `ModelProviderService.preflight_platform_credential()` accepts the same environment-backed OpenRouter lane as the worker.
- No request payload or frontend contract changes are introduced.

- [ ] **Step 1: Write a fake-adapter execution test before changing integration code**

Create a focused test module with a fake OpenRouter adapter response containing deterministic usage, a fake provider service with an enabled OpenRouter record, and a real credits service/store fixture. The test must set only a sentinel environment value (`sk-or-execution-test-abcd`) and assert:

```python
result = execution_service.execute(request_with_platform_credits)
assert result.credential_id is None
assert result.credential_key_last_four == "abcd"
assert result.billing.billing_source.value == "platform_credits"
assert result.billing.debited_credits_micro > 0
```

The ledger assertions must show the reservation settled and Grant Credits reduced before Purchased Credits. Add a failure test where the fake provider returns no usage and assert the reservation is released without a debit.

- [ ] **Step 2: Run the focused execution tests and verify the environment path**

User-run command:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/test_credit_metering.py \
  -k "platform or grant or purchased"
```

Expected before implementation: the new execution test fails with credential missing or preflight rejection. Existing BYOK tests must remain green.

- [ ] **Step 3: Keep execution and preflight on the shared resolver**

Do not add a second credential path to `LLMExecutionService`. Confirm that the Task 1 resolver is used by both worker execution and API preflight. If a route fixture currently hard-codes a fake preflight service that rejects environment fallback, update only that fixture to model the accepted contract; do not change production route payloads.

- [ ] **Step 4: Give the user the complete focused verification command**

User-run command:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_credit_metering.py \
  dashboard/backend/tests/test_byok_backtest_frontend.py
```

Expected: all focused tests pass; warnings from existing FastAPI lifecycle deprecations are non-blocking. The user then manually verifies: enable OpenRouter Platform Credits, open Run Backtest, select `Use ATL Credits`, choose a model, and confirm a real run debits Grant Credits by actual usage.

- [ ] **Step 5: Commit the execution verification layer**

```bash
git add dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/test_credit_metering.py
git commit -m "test: verify env-backed platform credit execution"
```

## Final handoff

After the user confirms the focused tests and browser flow, inspect `git diff --check`, verify that only intended source/test files are staged, and report the commits. Do not stage the local database or the user's unrelated `dashboard/frontend/app.js` change.
