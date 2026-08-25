# BYOK Backtest Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add provider-compatible model selection to verified default API keys and carry a safe BYOK selection into the existing Run Backtest flow, while keeping Platform Credits available as a separate metered lane.

**Architecture:** A backend-owned ATL model catalog validates provider/model compatibility and exposes safe execution options. The signed worker handoff and pricing evidence retain the canonical ATL model id; each execution adapter derives the provider-native request model only immediately before the network call. The API Keys page stores a one-use, 10-minute BYOK selection containing no credential material, and the existing Run Backtest modal consumes it.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL provider stores, vanilla JavaScript, HTML/CSS, pytest static frontend contracts

**Spec:** `docs/superpowers/specs/2026-08-24-byok-backtest-entry-design.md`

## Global Constraints

- BYOK calls deduct zero ATL Credits; Platform Credits calls reserve and settle against real provider usage and the pricing snapshot.
- Keep canonical ATL model ids, such as `openai/gpt-5.5`, in the signed handoff, pricing snapshot, evidence, and public run metadata.
- Derive provider-native request ids, such as `gpt-5.5`, only at the provider adapter boundary.
- Never place a full API key in HTML, JavaScript state, URL parameters, browser storage, logs, tests, commits, or API responses.
- `sessionStorage` may contain only `billing_mode`, `provider_id`, `model_id`, and `expires_at` under `atlPendingByokBacktest`.
- Rule-based and hosted runtime flows must remain unchanged.
- Do not stage or commit `.superpowers/`, `work/`, or `dashboard/storage/data/backtest.db`.
- The user runs all pytest commands, browser acceptance, and real-provider calls. The implementing agent may run only static syntax checks, focused source inspection, and `git diff --check`.
- Stop after the first failed check or user-reported failed test; do not continue to the next task.

---

### Task 1: Add the backend-owned execution model catalog

**Files:**
- Create: `dashboard/backend/domain/model_providers/execution_catalog.py`
- Modify: `dashboard/backend/domain/model_providers/models.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/openai.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/anthropic.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/adapters/gemini.py`
- Create: `dashboard/backend/tests/domain/model_providers/test_execution_catalog.py`
- Create: `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`
- Modify: `dashboard/backend/tests/test_frontend_model_facets.py`

**Interfaces:**
- Produces: `ExecutionModelRoute(catalog_id: str, label: str, provider_model_id: str)`.
- Produces: `list_execution_model_routes(provider: ProviderRecord) -> tuple[ExecutionModelRoute, ...]`.
- Produces: `resolve_execution_model_route(provider: ProviderRecord, catalog_id: str) -> ExecutionModelRoute`.
- Consumed by: Tasks 2 and 3 use the catalog for execution options and request preflight.

- [ ] **Step 1: Write the failing catalog tests**

Create `dashboard/backend/tests/domain/model_providers/test_execution_catalog.py` with these contracts:

```python
import pytest

from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    list_execution_model_routes,
    resolve_execution_model_route,
)
from dashboard.backend.domain.model_providers.models import ProviderRecord


def _provider(adapter_type: str, *, allowlist: tuple[str, ...] = ()) -> ProviderRecord:
    return ProviderRecord(
        provider_id=(
            "approved_compatible"
            if adapter_type == "openai_compatible"
            else adapter_type
        ),
        display_name="Provider",
        adapter_type=adapter_type,
        approved_base_url="https://provider.example/v1",
        capabilities={"model_allowlist": allowlist},
    )


def test_openrouter_can_run_the_full_atl_catalog():
    routes = list_execution_model_routes(_provider("openrouter"))
    assert [route.catalog_id for route in routes] == [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.7-plus",
    ]
    assert all(route.provider_model_id == route.catalog_id for route in routes)


@pytest.mark.parametrize(
    ("adapter_type", "catalog_id", "provider_model_id"),
    [
        ("openai", "openai/gpt-5.5", "gpt-5.5"),
        (
            "anthropic",
            "anthropic/claude-haiku-4-5",
            "claude-haiku-4-5",
        ),
        (
            "anthropic",
            "anthropic/claude-sonnet-4-6",
            "claude-sonnet-4-6",
        ),
        (
            "gemini",
            "google/gemini-3.1-pro-preview",
            "gemini-3.1-pro-preview",
        ),
    ],
)
def test_native_provider_routes_strip_only_the_expected_vendor_prefix(
    adapter_type,
    catalog_id,
    provider_model_id,
):
    route = resolve_execution_model_route(_provider(adapter_type), catalog_id)
    assert route.catalog_id == catalog_id
    assert route.provider_model_id == provider_model_id


def test_native_provider_rejects_an_incompatible_model():
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(
            _provider("openai"),
            "anthropic/claude-sonnet-4-6",
        )


def test_custom_provider_requires_an_explicit_allowlist():
    provider = _provider(
        "openai_compatible",
        allowlist=("openai/gpt-5.5",),
    )
    assert [
        route.catalog_id
        for route in list_execution_model_routes(provider)
    ] == ["openai/gpt-5.5"]
    with pytest.raises(UnsupportedExecutionModel):
        resolve_execution_model_route(
            provider,
            "deepseek/deepseek-v4-pro",
        )
```

Add a drift guard to `test_frontend_model_facets.py` that imports
`ATL_EXECUTION_MODELS`, slices the existing `SUPPORTED_MODELS` declaration from
`app.js`, asserts every catalog id appears as a `slug`, and asserts the number
of `slug:` entries equals `len(ATL_EXECUTION_MODELS)`.

- [ ] **Step 2: Ask the user to run the catalog test and stop until it fails for the expected missing module**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/test_frontend_model_facets.py
```

Expected: collection fails because `execution_catalog.py` does not exist yet.

- [ ] **Step 3: Add the catalog types and compatibility rules**

Add `model_allowlist` to `ProviderCapabilities` in `models.py`:

```python
model_allowlist: tuple[str, ...] = ()

@field_validator("model_allowlist")
@classmethod
def validate_model_allowlist(
    cls,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or len(item) > 64:
            raise ValueError(
                "model_allowlist contains an invalid model id"
            )
        if item not in cleaned:
            cleaned.append(item)
    return tuple(cleaned)
```

Create `execution_catalog.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from .models import ProviderRecord


class UnsupportedExecutionModel(ValueError):
    pass


@dataclass(frozen=True)
class CatalogModel:
    catalog_id: str
    label: str
    vendor: str


@dataclass(frozen=True)
class ExecutionModelRoute:
    catalog_id: str
    label: str
    provider_model_id: str


ATL_EXECUTION_MODELS = (
    CatalogModel(
        "anthropic/claude-haiku-4-5",
        "Claude Haiku 4.5",
        "anthropic",
    ),
    CatalogModel(
        "anthropic/claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        "anthropic",
    ),
    CatalogModel("openai/gpt-5.5", "GPT-5.5", "openai"),
    CatalogModel(
        "google/gemini-3.1-pro-preview",
        "Gemini 3.1 Pro Preview",
        "google",
    ),
    CatalogModel(
        "deepseek/deepseek-v4-pro",
        "DeepSeek V4 Pro",
        "deepseek",
    ),
    CatalogModel("qwen/qwen3.7-plus", "Qwen3.7 Plus", "qwen"),
)

_NATIVE_VENDOR = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
}


def _provider_model_id(
    provider: ProviderRecord,
    model: CatalogModel,
) -> str | None:
    if provider.adapter_type == "openrouter":
        return model.catalog_id
    native_vendor = _NATIVE_VENDOR.get(provider.adapter_type)
    if native_vendor:
        if model.vendor != native_vendor:
            return None
        return model.catalog_id.split("/", 1)[1]
    if provider.adapter_type == "openai_compatible":
        return (
            model.catalog_id
            if model.catalog_id in provider.capabilities.model_allowlist
            else None
        )
    return None


def list_execution_model_routes(
    provider: ProviderRecord,
) -> tuple[ExecutionModelRoute, ...]:
    routes: list[ExecutionModelRoute] = []
    for model in ATL_EXECUTION_MODELS:
        provider_model_id = _provider_model_id(provider, model)
        if provider_model_id:
            routes.append(
                ExecutionModelRoute(
                    catalog_id=model.catalog_id,
                    label=model.label,
                    provider_model_id=provider_model_id,
                )
            )
    return tuple(routes)


def resolve_execution_model_route(
    provider: ProviderRecord,
    catalog_id: str,
) -> ExecutionModelRoute:
    requested = str(catalog_id or "").strip()
    for route in list_execution_model_routes(provider):
        if route.catalog_id == requested:
            return route
    raise UnsupportedExecutionModel(
        "model is not available from this provider"
    )
```

- [ ] **Step 4: Write adapter boundary tests**

Create `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`. Use fake SDK clients and assert that canonical ATL ids remain on `LLMExecutionRequest`, while outbound provider calls receive `gpt-5.5`, `claude-sonnet-4-6`, and `gemini-3.1-pro-preview`.

The OpenAI assertion must be:

```python
assert captured["model"] == "gpt-5.5"
assert request.model_id == "openai/gpt-5.5"
```

The OpenRouter assertion must prove the outbound model remains `openai/gpt-5.5`.

- [ ] **Step 5: Route model ids at the adapter boundary**

In each adapter, resolve the route before constructing the provider request:

```python
from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
    resolve_execution_model_route,
)

try:
    provider_model_id = resolve_execution_model_route(
        provider,
        request.model_id,
    ).provider_model_id
except UnsupportedExecutionModel as exc:
    raise ProviderExecutionError("provider_unavailable") from exc
```

Use `provider_model_id` in `chat.completions.create`, `messages.create`, or the Gemini endpoint. Keep `request.model_id` in normalized execution results and billing evidence so pricing continues to use the canonical ATL id.

- [ ] **Step 6: Ask the user to run the catalog and adapter tests**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/test_frontend_model_facets.py
```

Expected: all tests pass.

- [ ] **Step 7: Run permitted static checks and commit**

Agent runs:

```bash
/opt/anaconda3/bin/python3 -m py_compile \
  dashboard/backend/domain/model_providers/execution_catalog.py \
  dashboard/backend/domain/model_providers/models.py \
  dashboard/backend/infrastructure/llm/execution/adapters/openai.py \
  dashboard/backend/infrastructure/llm/execution/adapters/anthropic.py \
  dashboard/backend/infrastructure/llm/execution/adapters/gemini.py
git diff --check
```

Commit only Task 1 files:

```bash
git add \
  dashboard/backend/domain/model_providers/execution_catalog.py \
  dashboard/backend/domain/model_providers/models.py \
  dashboard/backend/infrastructure/llm/execution/adapters/openai.py \
  dashboard/backend/infrastructure/llm/execution/adapters/anthropic.py \
  dashboard/backend/infrastructure/llm/execution/adapters/gemini.py \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "feat: add provider-compatible model routes"
```

---

### Task 2: Expose safe execution options for the authenticated user

**Files:**
- Modify: `dashboard/backend/domain/model_providers/models.py`
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Modify: `dashboard/backend/api/routers/model_credentials.py`
- Modify: `dashboard/backend/tests/domain/model_providers/test_service.py`
- Modify: `dashboard/backend/tests/test_model_credentials_api.py`

**Interfaces:**
- Consumes: `list_execution_model_routes(provider)` from Task 1.
- Produces: `ModelProviderService.list_execution_options(user_id: int) -> list[ExecutionProviderOption]`.
- Produces: authenticated `GET /api/credits/execution-options`.
- Consumed by: API Keys quick start and Run Backtest modal.

- [ ] **Step 1: Add failing service and API tests**

Tests must assert this OpenRouter shape after creating one verified default credential:

```python
{
    "provider_id": "openrouter",
    "display_name": "OpenRouter",
    "adapter_type": "openrouter",
    "byok_available": True,
    "platform_credits_available": False,
    "models": [
        {
            "model_id": "anthropic/claude-haiku-4-5",
            "label": "Claude Haiku 4.5",
        },
        {
            "model_id": "anthropic/claude-sonnet-4-6",
            "label": "Claude Sonnet 4.6",
        },
        {"model_id": "openai/gpt-5.5", "label": "GPT-5.5"},
        {
            "model_id": "google/gemini-3.1-pro-preview",
            "label": "Gemini 3.1 Pro Preview",
        },
        {
            "model_id": "deepseek/deepseek-v4-pro",
            "label": "DeepSeek V4 Pro",
        },
        {
            "model_id": "qwen/qwen3.7-plus",
            "label": "Qwen3.7 Plus",
        },
    ],
}
```

Also prove:

- BYOK is available only for exactly one verified default credential.
- Platform Credits is available only when `platform_enabled` is true and the public platform credential status is `verified`.
- No secret, encrypted blob, fingerprint, credential id, proxy URL, or upstream body appears in endpoint JSON.
- A custom provider with an empty `model_allowlist` returns an empty `models` list.

- [ ] **Step 2: Ask the user to run the focused tests and stop until they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py
```

Expected: new assertions fail because the service method and route do not exist.

- [ ] **Step 3: Add typed safe response models**

Add to `models.py`:

```python
class ExecutionModelOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=100)


class ExecutionProviderOption(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    adapter_type: AdapterType
    byok_available: bool
    platform_credits_available: bool
    models: tuple[ExecutionModelOption, ...] = ()
```

- [ ] **Step 4: Implement availability without decrypting credentials**

Add this method to `ModelProviderService`:

```python
def list_execution_options(
    self,
    user_id: int,
) -> list[ExecutionProviderOption]:
    credentials = self.store.list_user_credentials(int(user_id))
    defaults = {
        item["provider_id"]
        for item in credentials
        if item["status"] == "verified" and item["is_default"]
    }
    options: list[ExecutionProviderOption] = []
    for raw_provider in self.store.list_all_providers():
        provider = ProviderRecord.model_validate(raw_provider)
        if provider.status != "enabled":
            continue
        platform = self.store.get_platform_credential_public(
            provider.provider_id
        )
        routes = list_execution_model_routes(provider)
        options.append(
            ExecutionProviderOption(
                provider_id=provider.provider_id,
                display_name=provider.display_name,
                adapter_type=provider.adapter_type,
                byok_available=(
                    provider.byok_enabled
                    and provider.provider_id in defaults
                ),
                platform_credits_available=(
                    provider.platform_enabled
                    and bool(platform)
                    and platform["status"] == "verified"
                ),
                models=tuple(
                    ExecutionModelOption(
                        model_id=route.catalog_id,
                        label=route.label,
                    )
                    for route in routes
                ),
            )
        )
    return options
```

- [ ] **Step 5: Add the authenticated endpoint**

Add to `model_credentials.py`:

```python
@router.get("/credits/execution-options")
async def list_execution_options(
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(
        get_model_provider_service
    ),
):
    providers = await run_in_threadpool(
        service.list_execution_options,
        int(current_user["id"]),
    )
    return {
        "providers": [
            provider.model_dump(mode="json")
            for provider in providers
        ]
    }
```

- [ ] **Step 6: Ask the user to rerun the focused tests**

Run the Task 2 command again. Expected: all tests pass.

- [ ] **Step 7: Run permitted static checks and commit**

Agent runs `py_compile` on the modified Python files and `git diff --check`.

Commit only Task 2 files:

```bash
git add \
  dashboard/backend/domain/model_providers/models.py \
  dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/api/routers/model_credentials.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/test_model_credentials_api.py
git commit -m "feat: expose LLM execution options"
```

---

### Task 3: Validate provider/model compatibility before worker launch

**Files:**
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Modify: `dashboard/backend/api/routers/backtests.py`
- Modify: `dashboard/backend/tests/test_backtests_router.py`
- Modify: `dashboard/backend/tests/infrastructure/llm/test_token_cost.py`

**Interfaces:**
- Consumes: `resolve_execution_model_route` from Task 1.
- Produces: `ModelProviderService.preflight_execution_model(provider_id: str, catalog_model_id: str) -> ExecutionModelRoute`.
- Preserves: the signed handoff carries the canonical catalog id for pricing and evidence consistency.

- [ ] **Step 1: Add failing router tests**

Add tests with these assertions:

```python
def test_openai_byok_accepts_gpt_catalog_id_and_keeps_it_in_handoff():
    assert captured_handoff.model_id == "openai/gpt-5.5"


def test_openai_byok_rejects_claude_before_worker_start():
    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "The selected model is not available "
            "from this provider."
        )
    }
    assert spy.calls == 0
```

Also assert:

```python
snapshot = PricingSnapshot.from_model(
    "openai/gpt-5.5",
    "openai",
)
assert snapshot.input_usd_per_million_tokens == 5.0
assert snapshot.output_usd_per_million_tokens == 30.0
```

- [ ] **Step 2: Ask the user to run the focused tests and stop until they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/infrastructure/llm/test_token_cost.py
```

Expected: the incompatible-pair test fails because the router currently validates only the provider id and non-empty model string.

- [ ] **Step 3: Add service-level model preflight**

Add to `ModelProviderService`:

```python
def preflight_execution_model(
    self,
    provider_id: str,
    catalog_model_id: str,
) -> ExecutionModelRoute:
    provider = self.store.get_provider(provider_id)
    if not provider or provider["status"] != "enabled":
        raise ProviderNotFoundError("provider not found")
    return resolve_execution_model_route(
        ProviderRecord.model_validate(provider),
        catalog_model_id,
    )
```

- [ ] **Step 4: Validate the model before creating the signed handoff**

In the pipeline LLM block in `backtests.py`, call `preflight_execution_model` after validating `provider_id` and before credential preflight. Map `UnsupportedExecutionModel` to:

```python
raise HTTPException(
    status_code=422,
    detail=(
        "The selected model is not available "
        "from this provider."
    ),
)
```

Pass `route.catalog_id` to `create_execution_handoff`. Do not pass `route.provider_model_id`; the canonical id is required by pricing snapshots and evidence.

- [ ] **Step 5: Ask the user to rerun the focused tests**

Run the Task 3 command again. Expected: all tests pass.

- [ ] **Step 6: Run permitted static checks and commit**

Agent runs `py_compile` for the modified Python files and `git diff --check`.

Commit only Task 3 files:

```bash
git add \
  dashboard/backend/domain/model_providers/service.py \
  dashboard/backend/api/routers/backtests.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/infrastructure/llm/test_token_cost.py
git commit -m "feat: validate backtest model routes"
```

---

### Task 4: Add model selection and Run Backtest to verified default key rows

**Files:**
- Modify: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/frontend/styles.css`
- Modify: `dashboard/backend/tests/test_credits_frontend.py`

**Interfaces:**
- Consumes: `GET /api/credits/execution-options` from Task 2.
- Produces: one-use `sessionStorage.atlPendingByokBacktest` selection.
- Consumed by: Task 5 reads and clears the pending selection.

- [ ] **Step 1: Replace the obsolete frontend assertions with failing quick-start contracts**

Update `test_credits_frontend.py` to assert:

```python
def test_verified_default_key_can_prepare_a_safe_byok_backtest():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "/api/credits/execution-options" in source
    assert "atlPendingByokBacktest" in source
    assert "sessionStorage.setItem" in source
    assert "billing_mode: 'byok'" in source
    assert "provider_id: credential.provider_id" in source
    assert "model_id: modelId" in source
    assert "expires_at:" in source
    assert "'Run Backtest'" in source
    assert "localStorage" not in source
    assert ".innerHTML" not in source


def test_quick_start_state_never_contains_a_secret():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    start = source.index("function beginByokBacktest")
    end = source.index("\n  function ", start + 1)
    body = source[start:end]
    assert "api_key" not in body
    assert "key_last_four" not in body
    assert "credential_id" not in body
```

Retain the existing assertions that API values never enter `innerHTML`.

- [ ] **Step 2: Ask the user to run the frontend contract and stop until it fails**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_credits_frontend.py
```

Expected: new quick-start assertions fail.

- [ ] **Step 3: Load execution options with credentials**

Extend the Credits page state:

```javascript
const PENDING_BYOK_STORAGE_KEY = 'atlPendingByokBacktest';
const PENDING_BYOK_TTL_MS = 10 * 60 * 1000;

const state = {
  user: null,
  providers: [],
  credentials: [],
  executionOptions: [],
  selection: { kind: 'package', value: 'usd_20' },
  pendingPurchase: null,
  pendingRefund: null,
};
```

Change `loadApiKeys()` to request all three safe resources:

```javascript
const [
  providersResult,
  credentialsResult,
  executionOptionsResult,
] = await Promise.allSettled([
  apiRequest('/api/credits/model-providers'),
  apiRequest('/api/credits/api-keys'),
  apiRequest('/api/credits/execution-options'),
]);
```

On success, store `executionOptionsResult.value.providers || []`. On failure, set `state.executionOptions = []` and write `Backtest options could not be loaded.` to `creditsApiKeyStatus`.

- [ ] **Step 4: Render provider-compatible models only on verified default rows**

Add the lookup:

```javascript
function executionOption(providerId) {
  return state.executionOptions.find(
    (item) => item.provider_id === providerId,
  ) || null;
}
```

For `credential.status === 'verified' && credential.is_default`, create:

```javascript
const launch = document.createElement('div');
launch.className = 'credits-key-launch';

const modelLabel = textNode(
  'label',
  'credits-key-model-label',
  'Model',
);
const modelSelect = document.createElement('select');
modelSelect.className = 'credits-key-model';
modelSelect.setAttribute(
  'aria-label',
  `Model for ${credential.label}`,
);

const option = executionOption(credential.provider_id);
(option?.models || []).forEach((model) => {
  const modelOption = document.createElement('option');
  modelOption.value = model.model_id;
  modelOption.textContent = model.label;
  modelSelect.appendChild(modelOption);
});

const run = textNode(
  'button',
  'credits-key-action credits-key-run',
  'Run Backtest',
);
run.type = 'button';
run.disabled = (
  !option?.byok_available
  || modelSelect.options.length === 0
);
run.addEventListener(
  'click',
  () => beginByokBacktest(credential, modelSelect),
);

launch.append(modelLabel, modelSelect, run);
```

Append `launch` before the existing Reverify/Revoke `actions`. Verified non-default credentials keep `Set default` and do not receive a launch group.

- [ ] **Step 5: Store the safe pending launch and navigate**

Add:

```javascript
function beginByokBacktest(credential, modelSelect) {
  const modelId = String(modelSelect?.value || '').trim();
  if (
    credential.status !== 'verified'
    || !credential.is_default
    || !modelId
  ) {
    setStatus(
      element('creditsApiKeyStatus'),
      (
        'Choose a verified default key and model '
        + 'before starting a backtest.'
      ),
      'error',
    );
    return;
  }
  sessionStorage.setItem(
    PENDING_BYOK_STORAGE_KEY,
    JSON.stringify({
      billing_mode: 'byok',
      provider_id: credential.provider_id,
      model_id: modelId,
      expires_at: Date.now() + PENDING_BYOK_TTL_MS,
    }),
  );
  const target = new URL(window.location.href);
  target.searchParams.set('view', 'agents');
  window.location.assign(target.href);
}
```

Do not include the key, credential id, last four characters, label, or provider URL in the stored value.

- [ ] **Step 6: Add responsive key-row styling**

Add:

```css
.credits-key-launch {
    display: grid;
    grid-template-columns: minmax(130px, 1fr) auto;
    gap: 6px;
    align-items: end;
}

.credits-key-model-label {
    grid-column: 1 / -1;
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
}

.credits-key-model {
    min-height: 30px;
    min-width: 150px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
}

.credits-key-run {
    border-color: rgba(34, 211, 238, 0.45);
    color: #67e8f9;
}
```

Change the wide `.credits-key-row` columns to include the launch group:

```css
grid-template-columns:
    minmax(120px, 0.8fr)
    minmax(170px, 1fr)
    minmax(230px, 1fr)
    auto;
```

Inside the existing `@media (max-width: 600px)` block, set `.credits-key-launch { grid-template-columns: 1fr; }` and `.credits-key-model-label { grid-column: 1; }`.

- [ ] **Step 7: Ask the user to rerun the frontend contract**

Run the Task 4 pytest command again. Expected: pass.

- [ ] **Step 8: Run permitted static checks and commit**

Agent runs:

```bash
node --check dashboard/frontend/js/credits.js
git diff --check
```

Commit only Task 4 files:

```bash
git add \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py
git commit -m "feat: add BYOK backtest quick start"
```

---

### Task 5: Add billing, provider, and model controls to Run Backtest

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Modify: `dashboard/frontend/styles.css`
- Create: `dashboard/backend/tests/test_byok_backtest_frontend.py`
- Modify: `dashboard/backend/tests/test_my_agents_capital_ui.py`

**Interfaces:**
- Consumes: execution options from Task 2 and pending selection from Task 4.
- Produces: explicit `billing_mode`, `provider_id`, and canonical `model` in `POST /backtest/run` for pipeline LLM runs.
- Preserves: hosted runtime and rule-based requests omit the new fields.

- [ ] **Step 1: Add failing modal and payload contracts**

Create `dashboard/backend/tests/test_byok_backtest_frontend.py`:

```python
from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    start = APP_JS.index(f"function {name}(")
    next_function = APP_JS.find("\nfunction ", start + 1)
    return APP_JS[
        start:
        next_function if next_function >= 0 else len(APP_JS)
    ]


def test_run_backtest_modal_has_execution_controls():
    assert 'id="runBacktestBillingGroup"' in APP_HTML
    assert 'data-billing-mode="byok"' in APP_HTML
    assert 'data-billing-mode="platform_credits"' in APP_HTML
    assert 'id="runBacktestProviderSelect"' in APP_HTML
    assert 'id="modelSelect"' in APP_HTML
    assert "Model for this run" in APP_HTML


def test_pending_byok_selection_is_validated_and_consumed():
    assert "atlPendingByokBacktest" in APP_JS
    assert "sessionStorage.getItem" in APP_JS
    assert "sessionStorage.removeItem" in APP_JS
    assert "expires_at" in APP_JS


def test_pipeline_llm_payload_sends_explicit_execution_lane():
    body = _function_body("runBacktest")
    assert "payload.billing_mode" in body
    assert "payload.provider_id" in body
    assert "payload.model" in body
    assert "Choose an AI billing method, provider, and model." in body
```

Extend `test_my_agents_capital_ui.py` only to prove that the modal still contains no editable capital input after the new controls are inserted.

- [ ] **Step 2: Ask the user to run the focused frontend tests and stop until they fail**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_my_agents_capital_ui.py
```

Expected: the new file fails because the billing controls and payload fields do not exist.

- [ ] **Step 3: Add accessible modal controls**

Insert this control group before the existing Model group in `app.html`:

```html
<div class="control-group" id="runBacktestBillingGroup">
    <label>AI billing</label>
    <div
        class="run-backtest-billing-toggle"
        role="radiogroup"
        aria-label="AI billing"
    >
        <button
            type="button"
            data-billing-mode="byok"
            role="radio"
            aria-checked="false"
        >Use my API key</button>
        <button
            type="button"
            data-billing-mode="platform_credits"
            role="radio"
            aria-checked="false"
        >Use ATL Credits</button>
    </div>
    <label for="runBacktestProviderSelect">Provider</label>
    <select
        class="control-select"
        id="runBacktestProviderSelect"
    ></select>
    <p
        id="runBacktestBillingHint"
        class="control-helper"
    ></p>
</div>
```

Change the existing Model label to `Model for this run` and use the existing `#modelSelect` as the canonical model selector for pipeline LLM runs.

- [ ] **Step 4: Add safe execution-option and pending-selection helpers**

Add near the existing modal state:

```javascript
const PENDING_BYOK_STORAGE_KEY = 'atlPendingByokBacktest';
let runBacktestExecutionOptions = [];
let runBacktestBillingMode = null;

function readPendingByokBacktest() {
  let parsed = null;
  try {
    parsed = JSON.parse(
      sessionStorage.getItem(PENDING_BYOK_STORAGE_KEY)
      || 'null',
    );
  } catch (_) {
    parsed = null;
  }
  const valid = (
    parsed
    && parsed.billing_mode === 'byok'
    && /^[a-z0-9_]{2,64}$/.test(
      String(parsed.provider_id || ''),
    )
    && /^[A-Za-z0-9][A-Za-z0-9._/-]{0,63}$/.test(
      String(parsed.model_id || ''),
    )
    && Number.isFinite(Number(parsed.expires_at))
    && Number(parsed.expires_at) > Date.now()
  );
  if (!valid) {
    sessionStorage.removeItem(PENDING_BYOK_STORAGE_KEY);
    return null;
  }
  return parsed;
}

function clearPendingByokBacktest() {
  sessionStorage.removeItem(PENDING_BYOK_STORAGE_KEY);
}
```

Load `/api/credits/execution-options` through `API.request` when the modal opens. Fail closed by disabling `#runBacktestModalSubmit` if options cannot be loaded for a pipeline LLM run.

- [ ] **Step 5: Apply the accepted defaulting rules**

When the modal opens:

1. Use a valid pending BYOK selection first, then clear it.
2. Otherwise prefer BYOK when a compatible provider has `byok_available=true` for the agent's saved canonical model.
3. Otherwise use Platform Credits when a compatible provider has `platform_credits_available=true`.
4. If neither exists, disable submit and show `Add and verify a default API key, or ask an administrator to enable a platform provider.`

Switching billing mode rebuilds the provider list. Switching provider rebuilds the model list. Insert every API-derived option label with `textContent`.

- [ ] **Step 6: Bind the segmented control**

Add event listeners in the existing modal setup block:

```javascript
document
  .querySelectorAll(
    '#runBacktestBillingGroup [data-billing-mode]',
  )
  .forEach((button) => {
    button.addEventListener('click', () => {
      setRunBacktestBillingMode(button.dataset.billingMode);
    });
  });

document
  .getElementById('runBacktestProviderSelect')
  ?.addEventListener(
    'change',
    syncRunBacktestModelOptions,
  );
```

`setRunBacktestBillingMode` must update `runBacktestBillingMode`, `aria-checked`, selected styling, available providers, model options, hint copy, and submit availability in one place.

- [ ] **Step 7: Keep rule-based and hosted runtime flows unchanged**

Hide `#runBacktestBillingGroup` when:

- the resolved decision source is `rule_based`; or
- `(agent.runtime_type || 'pipeline') !== 'pipeline'`.

Do not add billing fields to either request. Re-run the visibility sync whenever `syncMarketDataSourceUI` changes the decision source.

- [ ] **Step 8: Send explicit LLM execution fields**

Inside `runBacktest()`, after `activeAgent`, `isHostedRuntime`, and `model` are resolved, validate the controls before disabling the submit button:

```javascript
let selectedProviderId = '';
if (
  decisionSource === LLM_DECISION_SOURCE
  && !isHostedRuntime
) {
  selectedProviderId = (
    document
      .getElementById('runBacktestProviderSelect')
      ?.value
    || ''
  );
  if (
    !runBacktestBillingMode
    || !selectedProviderId
    || !model
  ) {
    showModalError(
      'Choose an AI billing method, provider, and model.',
    );
    return;
  }
}
```

When constructing `payload`, add:

```javascript
if (
  decisionSource === LLM_DECISION_SOURCE
  && !isHostedRuntime
) {
  payload.billing_mode = runBacktestBillingMode;
  payload.provider_id = selectedProviderId;
  payload.model = model;
}
```

Keep the canonical model in `params` and `payload`. Do not store or send a credential id.

- [ ] **Step 9: Add modal styling**

Add:

```css
.run-backtest-billing-toggle {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    padding: 4px;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.55);
}

.run-backtest-billing-toggle button {
    min-height: 34px;
    border: 1px solid transparent;
    border-radius: 6px;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 12px;
    font-weight: 700;
}

.run-backtest-billing-toggle button[aria-checked="true"] {
    border-color: rgba(34, 211, 238, 0.42);
    background: rgba(34, 211, 238, 0.12);
    color: #67e8f9;
}
```

Do not increase the existing `560px` modal maximum width.

- [ ] **Step 10: Ask the user to rerun the focused frontend tests**

Run the Task 5 command again. Expected: pass.

- [ ] **Step 11: Run permitted static checks and commit**

Agent runs:

```bash
node --check dashboard/frontend/app.js
git diff --check
```

Commit only Task 5 files:

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_my_agents_capital_ui.py
git commit -m "feat: add backtest billing controls"
```

---

### Task 6: Remove obsolete copy and complete user-owned verification

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/js/credits.js`
- Modify: `dashboard/backend/tests/test_credits_frontend.py`
- Modify: `docs/superpowers/specs/2026-08-24-byok-backtest-entry-design.md` only if an implemented public identifier or fixed error message differs from the approved spec

**Interfaces:**
- Consumes: all previous tasks.
- Produces: accurate customer-facing billing wording and final acceptance evidence.

- [ ] **Step 1: Replace stale copy contracts**

Remove:

```text
Spending Credits on model runs is not enabled yet.
Held on your account. Spending Credits on model runs is not enabled yet.
```

Use:

```text
Available for metered ATL model runs. BYOK runs use your provider account and do not deduct ATL Credits.
```

Update `test_credits_frontend.py` to ban `not enabled yet` and require the new wording.

- [ ] **Step 2: Ask the user to run the complete focused automated suite**

Run:

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  dashboard/backend/tests/domain/model_providers/test_execution_catalog.py \
  dashboard/backend/tests/domain/model_providers/test_service.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py \
  dashboard/backend/tests/infrastructure/llm/test_token_cost.py \
  dashboard/backend/tests/test_model_credentials_api.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_my_agents_capital_ui.py
```

Expected: all tests pass. Stop immediately on the first reported failure.

- [ ] **Step 3: Ask the user to perform browser acceptance**

The user verifies:

1. A verified default OpenRouter key shows all six models and a `Run Backtest` button.
2. Clicking it lands on My Agents; opening an agent's modal preselects `Use my API key`, OpenRouter, and the chosen model.
3. OpenAI shows only GPT-5.5; Anthropic shows only Claude; Gemini shows only Gemini.
4. Closing the modal clears the pending quick-start selection.
5. Rule-based and hosted runs do not show or submit the billing controls.
6. Platform Credits remains a distinct selectable lane only when a verified platform credential is available.

- [ ] **Step 4: Ask the user to perform one real BYOK run**

The user runs one short OpenRouter backtest and verifies:

- the provider call succeeds;
- the run records provider, canonical model, and real usage;
- ATL Credits do not change; and
- no full API key appears in the UI, logs shared in chat, or API responses.

- [ ] **Step 5: Ask the user to perform one Platform Credits run when configured**

The user verifies that the run reserves and settles based on real token usage and pricing evidence, not a fixed per-run charge.

- [ ] **Step 6: Run final permitted static checks and commit**

Agent runs:

```bash
node --check dashboard/frontend/js/credits.js
node --check dashboard/frontend/app.js
git diff --check
git status --short
```

Verify that `dashboard/storage/data/backtest.db`, `.superpowers/`, and `work/` are not staged.

Commit only the copy and test files:

```bash
git add \
  dashboard/frontend/app.html \
  dashboard/frontend/js/credits.js \
  dashboard/backend/tests/test_credits_frontend.py
git commit -m "fix: update model billing copy"
```

- [ ] **Step 7: Prepare the PR description**

Use:

```markdown
## Summary
- add provider-compatible BYOK model quick start from saved API keys
- add explicit BYOK and Platform Credits controls to Run Backtest
- preserve canonical model pricing while routing native provider request ids

## Security
- keep full API keys in the encrypted server-side vault
- store only provider/model/billing identifiers in one-use session state
- fail closed on unavailable credentials, provider routes, usage, or billing

## Verification
- automated commands were run by the user
- browser and real-provider acceptance were performed by the user
```
