# Platform Provider Failover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry an OpenRouter Platform Credits model call once through CommonStack only after an explicit upstream balance or quota exhaustion error, without changing the model request, reasoning behavior, BYOK isolation, or single-debit billing guarantee.

**Architecture:** Register CommonStack as an allowlisted OpenAI-compatible platform provider, classify only bounded explicit quota failures into a new safe category, and route that category through a second provider attempt. Give each attempt an idempotent reservation identity while preserving the logical call index, then record the actual provider in call, run, Credits activity, and analytics evidence.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2, OpenAI Python SDK compatibility layer, SQLite, PostgreSQL/psycopg, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-platform-provider-failover-design.md`

## Global Constraints

- OpenRouter remains the preferred Platform Credits provider.
- CommonStack uses `https://api.commonstack.ai/v1` and `COMMONSTACK_API_KEY`.
- CommonStack is platform-enabled, BYOK-disabled, and allowlists only `openai/gpt-5.5`, `google/gemini-3.1-pro-preview`, `anthropic/claude-sonnet-4-6`, `deepseek/deepseek-v4-pro`, and `qwen/qwen3.7-plus`.
- `anthropic/claude-haiku-4-5` must not fail over until that exact CommonStack route is separately verified.
- Fail over only for HTTP `402` or an approved structured balance/quota code or phrase; a plain `429`, timeout, `5xx`, invalid response, invalid credential, unsupported model, local Credits error, and worker failure must not fail over.
- Retry CommonStack at most once and never retry a provider call that returned a usable completion.
- Preserve model, system message, messages, maximum output tokens, temperature, and reasoning effort exactly; never replace reasoning effort with `none`.
- BYOK uses only the user's verified default credential and never uses an Admin or environment Platform Credential.
- Release the primary reservation before creating the fallback reservation; one logical call may settle at most one debit.
- Results, run evidence, Credits activity, and analytics must name the provider that actually completed the call.
- SQLite and PostgreSQL behavior and migrations must remain equivalent.
- Use only fake credentials and provider responses in automated tests; do not call real OpenRouter, CommonStack, Render, or production databases.
- Do not mutate Render configuration before merge and separate deployment authorization.
- Do not touch or commit `dashboard/storage/data/backtest.db`, real secrets, database URLs, `.superpowers/`, or `work/`.

---

### Task 1: Register the CommonStack platform provider

**Files:**
- Modify: `dashboard/backend/domain/model_providers/repository_common.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_repository_contract.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_service.py`
- Test: `dashboard/backend/tests/test_model_provider_store_postgres.py`

**Interfaces:**
- Produces: seeded provider `commonstack` with adapter type `openai_compatible`, fixed approved origin, exact model allowlist, `byok_enabled=False`, and `platform_enabled=True`.
- Produces: `_ENVIRONMENT_PLATFORM_KEY_NAMES["commonstack"] == "COMMONSTACK_API_KEY"`.
- Preserves: stored verified Platform Credential precedence over the environment key.
- Preserves: all existing provider order while stably placing CommonStack after existing providers, so OpenRouter remains ahead of CommonStack.
- Consumed by Task 4: `ModelProviderService.resolve_platform_credential("commonstack")`, `preflight_platform_credential("commonstack")`, and `preflight_execution_model("commonstack", model_id)`.

- [ ] **Step 1: Add failing seed and catalog contract tests**

Add these assertions to the SQLite repository contract test, importing the
catalog routing functions there:

```python
def test_seeded_commonstack_is_platform_only_with_verified_model_allowlist(store):
    provider = store.get_provider("commonstack")

    assert provider["adapter_type"] == "openai_compatible"
    assert provider["approved_base_url"] == "https://api.commonstack.ai/v1"
    assert provider["byok_enabled"] is False
    assert provider["platform_enabled"] is True
    assert provider["capabilities"].model_allowlist == (
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-sonnet-4-6",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    )


def test_commonstack_routes_only_the_verified_catalog_models(store):
    provider = ProviderRecord.model_validate(store.get_provider("commonstack"))

    assert [route.catalog_id for route in list_execution_model_routes(provider)] == [
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    ]
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(provider, "anthropic/claude-haiku-4-5")
```

Import `ProviderRecord`, `UnsupportedExecutionModel`,
`list_execution_model_routes`, and `resolve_execution_model_route` from their
existing domain modules. Add a PostgreSQL seed assertion with the same fixed
provider fields under the existing `@pg_only` marker.

- [ ] **Step 2: Run the seed tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py \
  -k "commonstack"
```

Expected: FAIL because `commonstack` is not present in `SEEDED_PROVIDERS`.
The PostgreSQL case may report SKIPPED when `TEST_POSTGRES_URL` is not set.

- [ ] **Step 3: Add the fixed CommonStack seed**

Define one reusable allowlist in `repository_common.py` and add the seed:

```python
COMMONSTACK_MODEL_ALLOWLIST = (
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.7-plus",
)

{
    "provider_id": "commonstack",
    "display_name": "CommonStack",
    "adapter_type": "openai_compatible",
    "approved_base_url": "https://api.commonstack.ai/v1",
    "byok_enabled": False,
    "platform_enabled": True,
    "capabilities": ProviderCapabilities(
        model_discovery=True,
        system_messages=True,
        reasoning=True,
        supported_parameters=(
            "temperature",
            "max_output_tokens",
            "reasoning_effort",
        ),
        model_allowlist=COMMONSTACK_MODEL_ALLOWLIST,
    ),
},
```

Keep the repositories' current insert-if-missing seed behavior so reopening a
store never overwrites an Admin-disabled or edited CommonStack row.

- [ ] **Step 4: Add failing CommonStack credential and ordering tests**

Add service tests using fake key values only:

```python
def test_commonstack_platform_credential_uses_explicit_environment_mapping(
    tmp_path, monkeypatch
):
    service, _store = _service(tmp_path, FakeAdapter())
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-environment-abcd")

    resolved = service.resolve_platform_credential("commonstack")

    assert resolved.provider_id == "commonstack"
    assert resolved.key_last_four == "abcd"
    assert resolved.secret == "cs-fake-environment-abcd"


def test_execution_options_keep_openrouter_ahead_of_commonstack(
    tmp_path, monkeypatch
):
    service, _store = _service(tmp_path, FakeAdapter())
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake-options-abcd")
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-options-wxyz")

    provider_ids = [
        option.provider_id
        for option in service.list_execution_options(7)
        if option.platform_credits_available
    ]

    assert provider_ids.index("openrouter") < provider_ids.index("commonstack")
    assert "cs-fake-options-wxyz" not in repr(service.list_execution_options(7))
```

Also test that a verified stored CommonStack Platform Credential wins over
`COMMONSTACK_API_KEY`, matching the existing OpenRouter precedence contract.

- [ ] **Step 5: Run the service tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  -k "commonstack or openrouter_ahead"
```

Expected: FAIL because CommonStack has no environment mapping and sorts before
OpenRouter by display name.

- [ ] **Step 6: Implement credential resolution and stable ordering**

Extend the explicit mapping and stably move only CommonStack to the end:

```python
_ENVIRONMENT_PLATFORM_KEY_NAMES = {
    "openrouter": "OPENROUTER_API_KEY",
    "commonstack": "COMMONSTACK_API_KEY",
}

# After building all options. Python's sort is stable, so existing providers
# keep their current order and CommonStack follows them.
options.sort(key=lambda option: option.provider_id == "commonstack")
return options
```

Do not special-case CommonStack inside BYOK resolution. Its seeded
`byok_enabled=False` flag remains the authority for that lane.

- [ ] **Step 7: Run all provider contracts**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py
```

Expected: PASS, with only the configured PostgreSQL skips allowed.

- [ ] **Step 8: Commit the provider layer**

```bash
git add dashboard/backend/domain/model_providers/repository_common.py \
  dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/tests/domain/model_providers/test_repository_contract.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py
git commit -m "feat: register CommonStack platform provider"
```

---

### Task 2: Classify explicit provider quota exhaustion safely

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/errors.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/base.py`
- Modify: `dashboard/backend/domain/analytics/models.py`
- Create: `dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py`
- Test: `dashboard/backend/tests/domain/analytics/test_models.py`

**Interfaces:**
- Produces: `ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"` with a fixed safe message.
- Produces: `map_provider_error(exc: Exception) -> ProviderExecutionError` that returns the new category only for the spec's explicit bounded signals.
- Preserves: timeout, credential, unsafe-address, and generic unavailable mappings.
- Consumed by Task 4: the execution service uses only the typed category, never an upstream body, to decide whether to fail over.

- [ ] **Step 1: Write the failing positive and negative classifier table**

Create `test_provider_error_mapping.py` with a bounded fake response and this
parameter table:

```python
import json
from types import SimpleNamespace

import pytest

from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    map_provider_error,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
)


class _ProviderError(Exception):
    def __init__(self, status_code: int, payload: dict[str, object]):
        super().__init__("synthetic provider failure")
        content = json.dumps(payload).encode("utf-8")
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, content=content)


def _error(status_code: int, payload: dict[str, object]) -> _ProviderError:
    return _ProviderError(status_code, payload)


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (402, {"error": {"message": "payment required"}}),
        (429, {"error": {"code": "in_flight_budget_exhausted"}}),
        (400, {"error": {"type": "insufficient_quota"}}),
        (400, {"error": {"message": "Insufficient balance for this request"}}),
        (429, {"error": {"message": "You exceeded your current quota"}}),
    ],
)
def test_explicit_balance_or_quota_errors_are_typed(status_code, payload):
    mapped = map_provider_error(_error(status_code, payload))
    assert mapped.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (429, {"error": {"message": "rate limit exceeded"}}, "provider_unavailable"),
        (500, {"error": {"message": "insufficient capacity"}}, "provider_unavailable"),
        (401, {"error": {"message": "invalid key"}}, "credential_invalid"),
        (503, {"error": {"message": "service unavailable"}}, "provider_unavailable"),
    ],
)
def test_non_quota_errors_do_not_become_failover_signals(
    status_code, payload, expected
):
    mapped = map_provider_error(_error(status_code, payload))
    assert mapped.category.value == expected
```

Add cases for malformed JSON, a payload larger than the fixed limit, and a
quota-looking string outside the structured error fields; all must map to
`provider_unavailable` unless the status is `402`.

- [ ] **Step 2: Add the failing safe-message and analytics allowlist tests**

Add assertions that the new public exception text is fixed and that the safe
analytics model accepts the category:

```python
def test_quota_exhausted_error_message_is_fixed():
    error = LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)
    assert str(error) == "The selected model provider has insufficient balance or quota."


def test_safe_error_accepts_provider_quota_exhausted():
    record = AnalyticsEventRecord.model_validate(_record_payload(
        event_name="safe_error_recorded",
        event_group="resource",
        event_source="server",
        session_id=None,
        page_view=None,
        source_event_id="safe-error:quota-test",
        error_category="provider_quota_exhausted",
    ))
    assert record.error_category == "provider_quota_exhausted"
```

Use the existing `_record_payload` helper in `test_models.py`; import
`LLMExecutionError` and `ExecutionErrorCategory` for the fixed-message test.

- [ ] **Step 3: Run the new classifier tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py \
  dashboard/backend/tests/domain/analytics/test_models.py \
  -k "quota or balance"
```

Expected: FAIL because the category, message, bounded classifier, and analytics
allowlist do not exist.

- [ ] **Step 4: Add the category and fixed safe message**

Update `errors.py` and the analytics allowlist:

```python
class ExecutionErrorCategory(StrEnum):
    PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"


_SAFE_MESSAGES[ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED] = (
    "The selected model provider has insufficient balance or quota."
)
```

Add `"provider_quota_exhausted"` to `ALLOWED_ERROR_CATEGORIES` without widening
any other analytics property allowlist.

- [ ] **Step 5: Implement the bounded structured classifier**

In `adapters/base.py`, use fixed identifiers, phrases, and a 4 KiB payload cap:

```python
_PROVIDER_ERROR_PAYLOAD_MAX_BYTES = 4096
_QUOTA_ERROR_IDENTIFIERS = frozenset({
    "in_flight_budget_exhausted",
    "insufficient_quota",
    "quota_exceeded",
    "quota_exhausted",
    "insufficient_balance",
    "credit_balance_exhausted",
})
_QUOTA_ERROR_PHRASES = (
    "insufficient balance",
    "insufficient credits",
    "quota exceeded",
    "quota exhausted",
    "exceeded your current quota",
    "not enough credits",
)


def _bounded_error_payload(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    content = getattr(response, "content", b"")
    if not isinstance(content, bytes) or len(content) > _PROVIDER_ERROR_PAYLOAD_MAX_BYTES:
        return {}
    try:
        parsed = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
```

Read only `error.code`, `error.type`, `error.message`, and equivalent top-level
fields. Lowercase and trim bounded strings. In `map_provider_error`, preserve
timeout precedence, then classify status `402` or an approved payload signal,
then apply the existing credential/address/unavailable mappings. Never include
the parsed payload in the returned exception.

- [ ] **Step 6: Run the complete error and analytics tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/domain/analytics/test_models.py
```

Expected: PASS.

- [ ] **Step 7: Commit the error-classification layer**

```bash
git add dashboard/backend/infrastructure/llm/execution/errors.py \
  dashboard/backend/infrastructure/llm/execution/adapters/base.py \
  dashboard/backend/domain/analytics/models.py \
  dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py \
  dashboard/backend/tests/domain/analytics/test_models.py
git commit -m "feat: classify provider quota exhaustion"
```

---

### Task 3: Give each provider attempt an independent reservation identity

**Files:**
- Modify: `dashboard/backend/domain/credits/models.py`
- Modify: `dashboard/backend/domain/credits/service.py`
- Modify: `dashboard/backend/domain/credits/repository.py`
- Modify: `dashboard/backend/domain/credits/repository_postgres.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository_postgres.py`
- Test: `dashboard/backend/tests/domain/credits/test_service.py`
- Test: `dashboard/backend/tests/domain/credits/test_promotion_grants.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`
- Test: `dashboard/backend/tests/test_credits_api.py`

**Interfaces:**
- Produces: `CreditsService.reserve_llm_credits(..., provider_id: str, attempt_index: int = 0) -> LLMReservation`.
- Produces: matching SQLite and PostgreSQL store parameters `provider_id: str` and `attempt_index: int`.
- Produces: `LLMReservation.provider_id: str | None`, `LLMReservation.attempt_index: StrictInt`, `LLMSettlementResult.provider_id: str | None`, and `LLMSettlementResult.attempt_index: StrictInt`.
- Storage uniqueness: `(user_id, run_id, call_index, attempt_index)`.
- Preserves: `(user_id, run_id, call_index)` as the logical model-call identity used for activity call counts.
- Consumed by Task 4: attempt `0` for OpenRouter and attempt `1` for CommonStack.

- [ ] **Step 1: Write the failing SQLite two-attempt contract**

Add a repository test that releases attempt zero before creating attempt one:

```python
def test_provider_attempts_have_independent_reservations_for_one_logical_call(tmp_path):
    store = _store(tmp_path)
    _pending_order(store, amount_usd_cents=100, credits_micro=1_000_000)
    _pay_order(store, amount_usd_cents=100)

    primary = store.reserve_llm_credits(
        reservation_id="attempt-primary",
        user_id=1,
        run_id="attempt-run",
        call_index=3,
        attempt_index=0,
        provider_id="openrouter",
        reserved_micro=100_000,
        operation_key="attempt-primary-operation",
        request_digest="a" * 64,
    )
    store.release_llm_credits(primary["reservation_id"], reason="provider_quota_exhausted")
    fallback = store.reserve_llm_credits(
        reservation_id="attempt-fallback",
        user_id=1,
        run_id="attempt-run",
        call_index=3,
        attempt_index=1,
        provider_id="commonstack",
        reserved_micro=100_000,
        operation_key="attempt-fallback-operation",
        request_digest="b" * 64,
    )

    assert primary["attempt_index"] == 0
    assert primary["provider_id"] == "openrouter"
    assert fallback["attempt_index"] == 1
    assert fallback["provider_id"] == "commonstack"
```

Add negative cases for a negative attempt index, a blank provider identifier,
and idempotent replay with a different provider or attempt.

- [ ] **Step 2: Write the failing migration and PostgreSQL parity tests**

Extend the existing SQLite legacy-table fixture so it lacks both new columns,
open the store, and assert the legacy row reads as `attempt_index=0` and
`provider_id is None`. Assert a new attempt-one row can then share its logical
call index after the legacy row is released.

Add the PostgreSQL equivalent under `@pg_only`:

```python
fallback = store.reserve_llm_credits(
    reservation_id="pg-attempt-fallback",
    user_id=1,
    run_id="pg-attempt-run",
    call_index=4,
    attempt_index=1,
    provider_id="commonstack",
    reserved_micro=100_000,
    operation_key="pg-attempt-fallback-operation",
    request_digest="c" * 64,
)
assert fallback["attempt_index"] == 1
assert fallback["provider_id"] == "commonstack"
```

The PostgreSQL migration test must inspect `pg_constraint` and assert that the
named replacement unique constraint covers `user_id`, `run_id`, `call_index`,
and `attempt_index`.

- [ ] **Step 3: Run the repository tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  -k "attempt or migration"
```

Expected: FAIL because the reservation schema and method signatures do not
accept provider attempt identity. PostgreSQL may be SKIPPED when unconfigured.

- [ ] **Step 4: Extend the domain models and service signature**

Add validated fields and include both values in operation identity:

```python
class LLMReservation(BaseModel):
    provider_id: str | None = None
    attempt_index: StrictInt = Field(default=0, ge=0)


class LLMSettlementResult(BaseModel):
    provider_id: str | None = None
    attempt_index: StrictInt = Field(default=0, ge=0)
```

Update `CreditsService.reserve_llm_credits` to require `provider_id`, validate
it by calling
`dashboard.backend.domain.model_providers.repository_common.validate_provider_id`,
reject a boolean or negative `attempt_index`, and calculate defaults as:

```python
operation_key = operation_key or _operation_id(
    "llm_reserve", user_id, run_id, call_index, attempt_index, provider_id
)
request_digest = request_digest or _canonical_digest({
    "user_id": int(user_id),
    "run_id": run_id,
    "call_index": call_index,
    "attempt_index": attempt_index,
    "provider_id": provider_id,
    "amount_micro": amount_micro,
})
reservation_id = _operation_id(
    "llm_res", user_id, run_id, call_index, attempt_index, provider_id, operation_key
)
```

Populate the two new fields in `_llm_reservation_model` and
`_llm_settlement_model`, using `None` and `0` only when reading legacy rows.

- [ ] **Step 5: Migrate the SQLite reservation table safely**

Update `_LLM_RESERVATION_DDL` with nullable `provider_id`, non-negative
`attempt_index DEFAULT 0`, and the four-column unique constraint. Replace the
current narrow rebuild check with a rebuild condition covering missing new
columns, the legacy three-column uniqueness, or the obsolete settlement
ceiling:

```python
needs_rebuild = (
    "provider_id" not in reservation_columns
    or "attempt_index" not in reservation_columns
    or "unique(user_id,run_id,call_index)" in table_sql
    or "settled_micro<=reserved_micro" in table_sql
)
```

Back up `credit_llm_usage_entries`, rebuild reservations, and copy old rows with
`NULL AS provider_id` and `0 AS attempt_index` when those columns are absent.
Restore usage rows unchanged so reservation foreign keys and historical
evidence survive. Order run-level reservation reads by
`call_index, attempt_index, reservation_id`.

- [ ] **Step 6: Migrate PostgreSQL without losing rows**

In the PostgreSQL schema initialization SQL:

```sql
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS provider_id TEXT;
ALTER TABLE credit_llm_reservations
ADD COLUMN IF NOT EXISTS attempt_index INTEGER NOT NULL DEFAULT 0;
ALTER TABLE credit_llm_reservations
DROP CONSTRAINT IF EXISTS credit_llm_reservations_attempt_index_check;
ALTER TABLE credit_llm_reservations
ADD CONSTRAINT credit_llm_reservations_attempt_index_check
CHECK (attempt_index >= 0);
```

Use a `pg_constraint` block joined to `pg_attribute` through `conkey` ordinality
to drop only unique constraints whose ordered column array is exactly
`ARRAY['user_id', 'run_id', 'call_index']`. Then create the named replacement
unique constraint after first running
`DROP CONSTRAINT IF EXISTS credit_llm_reservations_logical_attempt_key`. The
new constraint covers `(user_id, run_id, call_index, attempt_index)`. Update
fresh-install DDL, reservation selects, inserts, idempotency comparisons, and
run-level ordering to include the new fields.

- [ ] **Step 7: Update repository reservation logic and existing callers**

Both repositories must validate `provider_id` and `attempt_index`, query an
existing reservation with this logical key:

```sql
user_id = ? AND run_id = ? AND call_index = ? AND attempt_index = ?
```

Use `%s` placeholders in the PostgreSQL version. Compare `provider_id` during
idempotent replay and reject different input with
`LLMReservationConflictError`.

Update every existing `reserve_llm_credits` caller reported by this command:

```bash
rg -l 'reserve_llm_credits\(' dashboard/backend --glob '*.py'
```

Existing single-attempt tests and application calls use
`provider_id="openrouter"` and the default `attempt_index=0`; tests whose
billing evidence names another provider use that provider identifier instead.

- [ ] **Step 8: Run Credits model, service, and repository tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/domain/credits/test_service.py \
  dashboard/backend/tests/domain/credits/test_promotion_grants.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py
```

Expected: PASS, with only configured PostgreSQL skips allowed. Confirm the
activity tests still count distinct `call_index` values rather than attempt
rows.

- [ ] **Step 9: Commit the reservation layer**

```bash
git add dashboard/backend/domain/credits/models.py \
  dashboard/backend/domain/credits/service.py \
  dashboard/backend/domain/credits/repository.py \
  dashboard/backend/domain/credits/repository_postgres.py \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/domain/credits/test_service.py \
  dashboard/backend/tests/domain/credits/test_promotion_grants.py \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/test_credits_api.py
git commit -m "feat: track provider attempt reservations"
```

---

### Task 4: Execute the strict OpenRouter-to-CommonStack fallback

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/models.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`

**Interfaces:**
- Produces: optional `LLMExecutionResult.requested_provider_id` populated by the production service.
- Produces: `LLMExecutionService._execute_once(request, *, attempt_index, requested_provider_id) -> LLMExecutionResult` for one credential, reservation, provider call, and settlement lifecycle.
- Produces: `LLMExecutionService._execute_with_platform_failover(request) -> LLMExecutionResult` for the one-way policy.
- Consumes: `ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED` from Task 2 and attempt-aware reservations from Task 3.
- Preserves: one outer safe-error emission for the final outcome only.

- [ ] **Step 1: Write the failing successful-fallback test**

Extend `test_platform_credits_env_fallback.py` with a scripted adapter and make
its existing `_execution_service` helper accept a provider-to-adapter mapping.
Keep the existing two-value return contract and replace its fixed pricing
snapshot lambda with one that uses the supplied model and provider identifiers:

```python
class ScriptedExecutionAdapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, request, credential, provider):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def snapshot_for(model_id: str, provider_id: str) -> PricingSnapshot:
    return PricingSnapshot(
        provider_id=provider_id,
        model_id=model_id,
        input_usd_per_million_tokens=1000.0,
        output_usd_per_million_tokens=1000.0,
        source_version="test-pricing",
    )
```

The adapter resolver inside `_execution_service` must use
`adapters[provider.provider_id]` when the optional mapping is supplied and keep
the current single-adapter behavior otherwise. Then add this test using the
real temporary provider and Credits stores:

```python
def test_platform_quota_error_retries_once_through_commonstack(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-failover-abcd")
    primary_adapter = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback_adapter = ScriptedExecutionAdapter([AdapterResponse(
        text="BUY",
        model_id="qwen/qwen3.7-plus",
        usage=LLMUsage(input_tokens=40, output_tokens=20),
        finish_reason="stop",
    )])
    service, credits_store = _execution_service(
        tmp_path,
        monkeypatch,
        primary_adapter,
    adapters={
            "openrouter": primary_adapter,
            "commonstack": fallback_adapter,
        },
    )
    assert service.providers.store.get_provider("commonstack")["platform_enabled"] is True
    request = _request("failover-run").model_copy(update={
        "model_id": "qwen/qwen3.7-plus",
        "reasoning_effort": "high",
        "temperature": 0.2,
    })

    result = service.execute(request)

    assert result.provider_id == "commonstack"
    assert result.requested_provider_id == "openrouter"
    assert [call.provider_id for call in primary_adapter.calls] == ["openrouter"]
    assert [call.provider_id for call in fallback_adapter.calls] == ["commonstack"]
    assert fallback_adapter.calls[0].reasoning_effort == "high"
    assert fallback_adapter.calls[0].temperature == 0.2
    with credits_store._get_connection() as connection:
        rows = connection.execute(
            "SELECT attempt_index, provider_id, status "
            "FROM credit_llm_reservations WHERE run_id = ? "
            "ORDER BY attempt_index",
            ("failover-run",),
        ).fetchall()
    assert [(row["attempt_index"], row["provider_id"], row["status"]) for row in rows] == [
        (0, "openrouter", "released"),
        (1, "commonstack", "settled"),
    ]
```

Compare `system_message`, `messages`, `model_id`, `usage_policy`, temperature,
and reasoning effort between the two captured requests; only `provider_id` may
differ.

- [ ] **Step 2: Write failing non-fallback and dual-failure tests**

Parameterize primary failures for `PROVIDER_TIMEOUT`, `PROVIDER_UNAVAILABLE`,
`RESPONSE_INVALID`, `USAGE_UNAVAILABLE`, `CREDENTIAL_INVALID`, and
`BILLING_FAILED`. Each case must assert zero CommonStack calls and the original
category.

```python
@pytest.mark.parametrize("category", [
    ExecutionErrorCategory.PROVIDER_TIMEOUT,
    ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
    ExecutionErrorCategory.RESPONSE_INVALID,
    ExecutionErrorCategory.USAGE_UNAVAILABLE,
    ExecutionErrorCategory.CREDENTIAL_INVALID,
    ExecutionErrorCategory.BILLING_FAILED,
])
def test_non_quota_platform_failures_do_not_fail_over(
    tmp_path, monkeypatch, category
):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-unused-abcd")
    primary = ScriptedExecutionAdapter([ProviderExecutionError(category)])
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request(f"no-failover-{category.value}"))

    assert exc_info.value.category is category
    assert fallback.calls == []
```

Add these separate cases using `ScriptedExecutionAdapter`, `_execution_service`,
and `_request` from the same file:

```python
def test_byok_quota_error_never_uses_platform_fallback(
    tmp_path, monkeypatch
):
    primary = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    calls = []

    def fail_once(request, **_kwargs):
        calls.append(request.provider_id)
        raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED)

    monkeypatch.setattr(service, "_execute_once", fail_once)
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("byok-no-fallback").model_copy(update={
            "billing_mode": BillingMode.BYOK,
        }))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert calls == ["openrouter"]
    assert fallback.calls == []


def test_fallback_failure_returns_commonstack_safe_category(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-timeout-abcd")
    primary = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_TIMEOUT),
    ])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("dual-failure"))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_TIMEOUT
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


def test_missing_commonstack_route_preserves_primary_quota_error(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("COMMONSTACK_API_KEY", raising=False)
    primary = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("missing-fallback"))
    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert fallback.calls == []
```

Add the release-failure and double-quota guards explicitly:

```python
def test_primary_release_failure_aborts_before_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-unused-abcd")
    primary = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback = ScriptedExecutionAdapter([])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    def fail_release(*_args, **_kwargs):
        raise RuntimeError("synthetic release failure")

    monkeypatch.setattr(service.credits, "release_llm_credits", fail_release)
    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("release-failure"))

    assert exc_info.value.category is ExecutionErrorCategory.BILLING_FAILED
    assert fallback.calls == []


def test_two_quota_failures_stop_after_commonstack(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMONSTACK_API_KEY", "cs-fake-double-quota-abcd")
    primary = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    fallback = ScriptedExecutionAdapter([
        ProviderExecutionError(ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED),
    ])
    service, _store = _execution_service(
        tmp_path,
        monkeypatch,
        primary,
        adapters={"openrouter": primary, "commonstack": fallback},
    )

    with pytest.raises(LLMExecutionError) as exc_info:
        service.execute(_request("double-quota"))

    assert exc_info.value.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1
```

- [ ] **Step 3: Run the failover tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  -k "failover or quota_error"
```

Expected: FAIL because execution currently performs exactly the requested
provider attempt and has no requested-versus-actual attribution.

- [ ] **Step 4: Add additive result attribution**

Extend the result model without breaking older test doubles:

```python
class LLMExecutionResult(BaseModel):
    requested_provider_id: str | None = Field(default=None, min_length=2, max_length=64)
```

Update `_result` to accept `requested_provider_id: str`, set the actual
`provider_id` from the attempt request, and always populate the requested value
in production results.

- [ ] **Step 5: Refactor one provider attempt behind a private method**

Move the current provider, credential, pricing, adapter, BYOK, and Platform
Credits logic into `_execute_once`. Pass attempt identity to the existing
platform execution method:

```python
reservation = self.credits.reserve_llm_credits(
    user_id=request.user_id,
    run_id=request.run_id,
    call_index=request.call_index,
    attempt_index=attempt_index,
    provider_id=request.provider_id,
    amount_micro=reserved_micro,
)
```

Keep `_execute_platform` responsible for releasing its own reservation before
its error escapes. A release error must replace the provider category with
`billing_failed`, which prevents routing into fallback.

- [ ] **Step 6: Implement the one-way policy around `_execute_once`**

Use the typed category and immutable request copy only:

```python
def _execute_with_platform_failover(
    self,
    request: LLMExecutionRequest,
) -> LLMExecutionResult:
    try:
        return self._execute_once(
            request,
            attempt_index=0,
            requested_provider_id=request.provider_id,
        )
    except LLMExecutionError as primary_error:
        if (
            request.billing_mode is not BillingMode.PLATFORM_CREDITS
            or request.provider_id != "openrouter"
            or primary_error.category
            is not ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED
        ):
            raise

        try:
            self.providers.preflight_execution_model("commonstack", request.model_id)
            self.providers.preflight_platform_credential("commonstack")
        except (ProviderNotFoundError, CredentialResolutionError, UnsupportedExecutionModel):
            raise primary_error

        fallback_request = request.model_copy(update={"provider_id": "commonstack"})
        return self._execute_once(
            fallback_request,
            attempt_index=1,
            requested_provider_id=request.provider_id,
        )
```

Import the three explicit preflight exceptions from their existing domain
modules. Do not catch an unexpected repository exception as route ineligibility.
The public `execute` method must wrap this policy in its existing single outer
analytics try/except so a recovered primary quota error does not emit a failed
safe-error event.

- [ ] **Step 7: Attribute success and final errors correctly**

Update `_emit_model_usage` to use `result.provider_id` while retaining the
original request's user, run, call index, model, and billing mode. Extend
`_analytics_error_category` with:

```python
ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED: "provider_quota_exhausted",
```

Use the result's actual provider for pricing evidence by building the fallback
snapshot from the copied request. Do not emit a separate failure event for the
released primary attempt.

Add an OpenAI-compatible adapter contract using a CommonStack provider record
with `model_allowlist=("qwen/qwen3.7-plus",)`. Capture the SDK kwargs and assert:

```python
assert captured["model"] == "qwen/qwen3.7-plus"
assert captured["max_tokens"] == 4096
assert captured["temperature"] == 0.2
assert captured["extra_body"] == {"reasoning": {"effort": "high"}}
```

The test request must set `reasoning_effort="high"`; it must not contain a
second call with `reasoning_effort="none"`.

- [ ] **Step 8: Run failover and existing platform execution tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py
```

Expected: PASS. Inspect captured requests to confirm no test changes reasoning
effort to `none`.

- [ ] **Step 9: Commit the execution layer**

```bash
git add dashboard/backend/infrastructure/llm/execution/models.py \
  dashboard/backend/infrastructure/llm/execution/service.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py
git commit -m "feat: fail over platform quota errors to CommonStack"
```

---

### Task 5: Preserve actual and mixed provider evidence

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/models.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/client.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_client.py`
- Test: `dashboard/backend/tests/test_backtests_router.py`
- Test: `dashboard/backend/tests/domain/analytics/test_backfill.py`
- Test: `dashboard/backend/tests/domain/credits/test_repository.py`

**Interfaces:**
- Produces: additive `LLMRunEvidence.requested_provider_id`, `provider_ids`, and `provider_mixed` fields with historical defaults.
- Uniform run: `provider_id` is the one actual provider, `provider_ids` contains one value, and `provider_mixed=False`.
- Mixed run: `provider_id="mixed"`, ordered unique `provider_ids`, and `provider_mixed=True`.
- Preserves: credential identity and pricing snapshot only when every completed call has one equal value.
- Consumes: actual and requested provider fields from Task 4.

- [ ] **Step 1: Write the failing mixed-run summary test**

Allow the existing `_result` test helper to accept `provider_id`,
`requested_provider_id`, credential last four, and a matching pricing snapshot.
When `provider_id` is overridden, set the helper's snapshot provider to the same
value; use distinct snapshot values for the two providers so the mixed summary
correctly returns `pricing_snapshot=None`.
Add:

```python
def test_execution_summary_records_mixed_actual_providers():
    service = _FakeExecutionService([
        _result(
            provider_id="openrouter",
            requested_provider_id="openrouter",
            credential_key_last_four="1111",
        ),
        _result(
            provider_id="commonstack",
            requested_provider_id="openrouter",
            credential_key_last_four="2222",
        ),
    ])
    client = AnthropicCompatibleExecutionClient(
        execution_service=service,
        handoff=_handoff(),
    )

    _create(client)
    _create(client)
    summary = client.execution_summary()

    assert summary.requested_provider_id == "openrouter"
    assert summary.provider_id == "mixed"
    assert summary.provider_ids == ("openrouter", "commonstack")
    assert summary.provider_mixed is True
    assert summary.credential_id is None
    assert summary.credential_key_last_four is None
    assert summary.pricing_snapshot is None
```

Keep the existing uniform-provider test and add assertions for the one-element
`provider_ids` tuple and `provider_mixed=False`.

- [ ] **Step 2: Write failing historical and API projection tests**

Add a model test that validates an old `LLMRunEvidence` dictionary without the
three new fields and asserts inferred values. Extend the sanitized backtest
metadata test with:

```python
assert serialized["llm_execution"]["requested_provider_id"] == "openrouter"
assert serialized["llm_execution"]["provider_id"] == "mixed"
assert serialized["llm_execution"]["provider_ids"] == ["openrouter", "commonstack"]
assert serialized["llm_execution"]["provider_mixed"] is True
```

Retain the raw-secret rejection assertion. Add a one-call analytics backfill
fixture whose actual provider is CommonStack and requested provider is
OpenRouter; assert the backfilled event uses `provider_id="commonstack"`.

- [ ] **Step 3: Run the evidence tests and verify they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/domain/analytics/test_backfill.py \
  -k "provider or llm_execution"
```

Expected: FAIL because the run model rejects mixed provider results and lacks
the additive attribution fields.

- [ ] **Step 4: Add backward-compatible run evidence fields**

Extend `LLMRunEvidence` and infer missing historical values before validation:

```python
requested_provider_id: str | None = Field(default=None, min_length=2, max_length=64)
provider_ids: tuple[str, ...] = ()
provider_mixed: bool = False

@model_validator(mode="before")
@classmethod
def populate_provider_attribution(cls, value: object) -> object:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    provider_id = payload.get("provider_id")
    if "requested_provider_id" not in payload:
        payload["requested_provider_id"] = provider_id
    if "provider_ids" not in payload:
        payload["provider_ids"] = () if provider_id == "mixed" else (provider_id,)
    if "provider_mixed" not in payload:
        payload["provider_mixed"] = provider_id == "mixed"
    return payload
```

Validate each provider identifier with the existing provider-id rules and
enforce consistent mixed state:

```python
@field_validator("provider_ids")
@classmethod
def validate_provider_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = tuple(validate_provider_id(value) for value in values)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("provider_ids must be ordered and unique")
    return cleaned

@model_validator(mode="after")
def validate_provider_mix(self) -> "LLMRunEvidence":
    if self.provider_mixed:
        if self.provider_id != "mixed" or len(self.provider_ids) < 2:
            raise ValueError("mixed provider evidence requires multiple providers")
    elif self.provider_ids != (self.provider_id,):
        raise ValueError("uniform provider evidence requires one matching provider")
    return self
```

Import `validate_provider_id` from the existing model-provider validation
module. Historical uniform records therefore become explicit without modifying
stored rows.

- [ ] **Step 5: Aggregate actual providers without weakening identity checks**

In `execution_summary`, continue requiring one billing mode and canonical model.
Replace the old actual-provider-equals-handoff check with a requested-provider
check:

```python
requested_provider_ids = {
    result.requested_provider_id or self.handoff.provider_id
    for result in self._completed_results
}
if requested_provider_ids != {self.handoff.provider_id}:
    raise RuntimeError("inconsistent requested LLM provider identity")

provider_ids = tuple(dict.fromkeys(
    result.provider_id for result in self._completed_results
))
provider_mixed = len(provider_ids) > 1
provider_id = "mixed" if provider_mixed else provider_ids[0]
```

Populate `requested_provider_id`, `provider_ids`, and `provider_mixed` in the
summary. Keep the existing set-based credential and snapshot uniformity logic,
which correctly returns `None` for mixed values.

- [ ] **Step 6: Verify Credits activity uses settled actual-provider evidence**

Extend the existing mixed activity repository test so two logical call indexes
settle evidence with OpenRouter and CommonStack while both use the same model.
Change that test's second call from `provider_id="openai", model_id="model-b"`
to `provider_id="commonstack", model_id="model-a"`, then assert:

```python
assert activity["provider_id"] is None
assert activity["provider_mixed"] is True
assert activity["model_mixed"] is False
```

No activity implementation change is expected because it already derives
provider identity from each settlement's pricing snapshot. If the test fails,
fix only `summarize_activity_evidence`; do not add raw reservation evidence to
the public API.

- [ ] **Step 7: Run the evidence and activity suites**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/domain/analytics/test_backfill.py \
  dashboard/backend/tests/domain/credits/test_repository.py
```

Expected: PASS.

- [ ] **Step 8: Run the complete focused regression set**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers \
  dashboard/backend/tests/domain/credits \
  dashboard/backend/tests/domain/analytics/test_models.py \
  dashboard/backend/tests/domain/analytics/test_backfill.py \
  dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py \
  dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py \
  dashboard/backend/tests/test_model_provider_store_postgres.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_backtests_router.py
```

Expected: PASS, with only PostgreSQL tests skipped when
`TEST_POSTGRES_URL` is absent.

- [ ] **Step 9: Run formatting, diff, and secret checks**

Run:

```bash
git diff --check
git status --short
git diff --name-only
rg -n --hidden --glob '!dashboard/storage/data/**' \
  '(sk-[A-Za-z0-9_-]{16,}|OPENROUTER_API_KEY=.{8,}|COMMONSTACK_API_KEY=.{8,})' \
  dashboard docs .env.example
```

Expected: no whitespace errors; only intended source, test, spec, and plan files
appear. Inspect every secret-scan match: only strings containing explicit
`fake`, `test`, or `your_..._here` markers are allowed. Confirm
`dashboard/storage/data/backtest.db` remains unstaged and absent from every
commit.

- [ ] **Step 10: Commit the evidence layer**

```bash
git add dashboard/backend/infrastructure/llm/execution/models.py \
  dashboard/backend/infrastructure/llm/execution/client.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/domain/analytics/test_backfill.py \
  dashboard/backend/tests/domain/credits/test_repository.py
git commit -m "feat: record platform failover attribution"
```

- [ ] **Step 11: Review the final branch without deploying**

Run:

```bash
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git status --short --branch
```

Expected: the design and plan commits plus five implementation commits, no
staged files, and only the pre-existing local
`dashboard/storage/data/backtest.db` change. Do not push, open a PR, copy Render
secrets, or deploy until the user explicitly authorizes those actions.
