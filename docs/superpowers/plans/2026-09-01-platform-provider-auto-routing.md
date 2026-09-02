# Platform Provider Auto-Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let ATL Credits backtests choose a model without exposing provider selection, automatically attempting OpenRouter before CommonStack with safe failover.

**Architecture:** Extend the signed execution handoff and in-process execution request with an ordered provider candidate tuple. The route resolves compatible, credentialed platform providers in OpenRouter-first order; the execution service attempts each candidate and creates/settles an independent reservation per attempt. The frontend treats all platform providers as one deduplicated ATL Credits model inventory while preserving explicit BYOK controls.

**Tech Stack:** FastAPI, Pydantic v2, Python `pytest`, vanilla browser JavaScript, static frontend contract tests.

**Spec:** `docs/superpowers/specs/2026-09-01-platform-provider-auto-routing-design.md`

## Global Constraints

- Automatic ATL Credits candidates are limited to `openrouter` then `commonstack`.
- Only `credential_missing`, `credential_invalid`, `provider_unavailable`, `provider_timeout`, and `provider_quota_exhausted` may trigger provider failover.
- Never fail over ATL account restrictions, response validation, usage-unavailable, or billing failures.
- BYOK remains explicit-provider and never uses platform fallback.
- Do not stage or modify `dashboard/storage/data/backtest.db`; do not commit secrets, `.superpowers/`, or `work/`.

### Task 1: Add Ordered Provider Candidates To Handoffs And Requests

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/handoff.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/models.py`
- Modify: `dashboard/backend/infrastructure/llm/execution/client.py`
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_handoff.py` (create if absent)
- Test: `dashboard/backend/tests/infrastructure/llm/test_execution_client.py`

**Interfaces:**
- `ExecutionHandoff.provider_ids: tuple[str, ...]` is ordered, unique, and contains `provider_id` first.
- `create_execution_handoff(..., provider_ids: tuple[str, ...] | None = None)` signs the normalized candidate tuple.
- `LLMExecutionRequest.provider_ids: tuple[str, ...] = ()` defaults legacy requests to `(provider_id,)`.
- `AnthropicCompatibleExecutionClient` copies `handoff.provider_ids` into each request.

- [ ] **Step 1: Write failing handoff and request tests**

```python
def test_handoff_round_trips_ordered_provider_candidates():
    payload = create_execution_handoff(
        user_id=1,
        run_id="run-1",
        billing_mode="platform_credits",
        provider_id="openrouter",
        provider_ids=("openrouter", "commonstack"),
        model_id="qwen/qwen3.7-plus",
        now=1_000,
    )
    handoff = consume_execution_handoff(payload, now=1_001)
    assert handoff.provider_id == "openrouter"
    assert handoff.provider_ids == ("openrouter", "commonstack")


def test_handoff_rejects_duplicate_or_misordered_candidates():
    with pytest.raises(ValueError):
        create_execution_handoff(
            user_id=1,
            run_id="run-1",
            billing_mode="platform_credits",
            provider_id="openrouter",
            provider_ids=("commonstack", "openrouter"),
            model_id="qwen/qwen3.7-plus",
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_execution_handoff.py dashboard/backend/tests/infrastructure/llm/test_execution_client.py -q`

Expected: FAIL because the handoff and request types do not yet carry `provider_ids`.

- [ ] **Step 3: Implement the typed candidate tuple**

Add a tuple field to both Pydantic models. Normalize legacy payloads to a one-item tuple, validate every identifier with the existing provider-id validator, reject duplicates, and require the first tuple item to equal `provider_id`. Include `provider_ids` in the signed JSON payload and pass it from the client when constructing `LLMExecutionRequest`.

- [ ] **Step 4: Run the focused tests and the existing client suite**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_execution_handoff.py dashboard/backend/tests/infrastructure/llm/test_execution_client.py -q`

Expected: PASS, including all existing single-provider callers.

- [ ] **Step 5: Commit the handoff contract**

```bash
git add dashboard/backend/infrastructure/llm/execution/handoff.py dashboard/backend/infrastructure/llm/execution/models.py dashboard/backend/infrastructure/llm/execution/client.py dashboard/backend/tests/infrastructure/llm/test_execution_handoff.py dashboard/backend/tests/infrastructure/llm/test_execution_client.py
git commit -m "feat: carry ordered platform provider candidates"
```

### Task 2: Resolve Platform Provider Candidates At The API Boundary

**Files:**
- Modify: `dashboard/backend/domain/model_providers/service.py`
- Modify: `dashboard/backend/api/routers/backtests.py`
- Test: `dashboard/backend/tests/test_backtests_router.py`
- Test: `dashboard/backend/tests/domain/model_providers/test_execution_catalog.py`

**Interfaces:**
- `ModelProviderService.resolve_platform_execution_candidates(model_id: str, preferred_provider_id: str | None = None) -> tuple[str, ...]` returns only enabled, platform-enabled, model-compatible providers with a verified stored or environment credential.
- The route accepts omitted `provider_id` for `platform_credits`, sets the first candidate as `provider_id`, and signs the complete tuple.

- [ ] **Step 1: Add failing router/service contract tests**

Cover OpenRouter-first ordering, CommonStack-only availability, model incompatibility, and a `422` response when no platform candidate exists. Add a request assertion that `billing_mode=platform_credits` without `provider_id` reaches handoff creation.

- [ ] **Step 2: Run the focused router tests and verify failure**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/domain/model_providers/test_execution_catalog.py -q`

Expected: FAIL because the service method and omitted-provider path do not exist.

- [ ] **Step 3: Implement candidate resolution**

Use the existing provider registry, `list_execution_model_routes`, `platform_enabled`, and `preflight_platform_credential`. Sort the default candidates with the fixed preference `("openrouter", "commonstack")`; when a legacy provider is supplied, validate it and place it first only if it is compatible and available. Keep the returned model ID as the canonical catalog ID.

- [ ] **Step 4: Update `/backtest/run` validation and handoff creation**

Require `provider_id` only for BYOK. For ATL Credits, call candidate resolution when it is absent, reject an empty result with a safe `422`, and call `create_execution_handoff` with both `provider_id=candidates[0]` and `provider_ids=candidates`. Keep the response's `provider_id` as the first internal candidate for backwards compatibility.

- [ ] **Step 5: Run focused tests and existing backtest router coverage**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/domain/model_providers/test_execution_catalog.py -q`

Expected: PASS with no raw provider credential or upstream error in responses.

- [ ] **Step 6: Commit the API routing contract**

```bash
git add dashboard/backend/domain/model_providers/service.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/domain/model_providers/test_execution_catalog.py
git commit -m "feat: resolve platform providers automatically"
```

### Task 3: Execute Ordered Failover With Safe Categories

**Files:**
- Modify: `dashboard/backend/infrastructure/llm/execution/service.py`
- Modify: `dashboard/backend/domain/analytics/instrumentation.py` only if required to preserve attempt evidence
- Test: `dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py`

**Interfaces:**
- `LLMExecutionService._execute_with_platform_failover(request)` iterates `request.provider_ids or (request.provider_id,)`.
- `requested_provider_id` remains the first candidate; `provider_id` is the successful candidate.
- Each attempt increments `attempt_index` and uses the candidate-specific reservation and pricing snapshot.

- [ ] **Step 1: Expand failing execution tests**

Parameterize fallback tests over `credential_missing`, `credential_invalid`, `provider_unavailable`, `provider_timeout`, and `provider_quota_exhausted`. Add assertions that `response_invalid`, `usage_unavailable`, `billing_failed`, and `account_restricted` do not call the fallback. Add a two-candidate request assertion for reservation attempt indexes `0` and `1`.

- [ ] **Step 2: Run the focused fallback tests and verify the new cases fail**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py -q`

Expected: FAIL for the newly allowed categories and candidate-list requests.

- [ ] **Step 3: Implement bounded candidate iteration**

Define a module-level immutable failover category set containing exactly the five allowed categories. For each candidate, clone the request with that provider ID, preserve the full candidate tuple, and call `_execute_once` with the current attempt index and original requested provider. Stop at the first success; if all candidates fail, raise the last safe `LLMExecutionError`. Never retry BYOK.

- [ ] **Step 4: Run fallback, adapter mapping, and credit ledger tests**

Run: `pytest dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py dashboard/backend/tests/infrastructure/llm/test_provider_error_mapping.py dashboard/backend/tests/domain/credits -q`

Expected: PASS; failed reservations are released and successful fallback reservations settle exactly once.

- [ ] **Step 5: Commit execution failover**

```bash
git add dashboard/backend/infrastructure/llm/execution/service.py dashboard/backend/tests/infrastructure/llm/test_platform_credits_env_fallback.py
git commit -m "feat: fail over platform execution safely"
```

### Task 4: Hide Provider Selection In The ATL Credits UI

**Files:**
- Modify: `dashboard/frontend/app.html`
- Modify: `dashboard/frontend/app.js`
- Test: `dashboard/backend/tests/test_byok_backtest_frontend.py`

**Interfaces:**
- `syncRunBacktestModelOptions()` returns the selected provider's models for BYOK and a deduplicated union of available platform models for ATL Credits.
- `syncRunBacktestProviderVisibility()` hides `#runBacktestProviderControl` when `runBacktestBillingMode === 'platform_credits'`.
- `runBacktest()` sends `provider_id` only for BYOK; ATL Credits sends billing mode and model only.

- [ ] **Step 1: Add failing static frontend contracts**

Assert a provider control wrapper exists, the ATL hint explains automatic provider routing, provider visibility is tied to `platform_credits`, model options are deduplicated, and the ATL payload does not set `provider_id` while the BYOK payload still does.

- [ ] **Step 2: Run the focused frontend contracts and verify failure**

Run: `pytest dashboard/backend/tests/test_byok_backtest_frontend.py -q`

Expected: FAIL because the wrapper, visibility helper, deduplicated platform model logic, and lane-specific payload are not present.

- [ ] **Step 3: Update the modal markup and lane state**

Wrap the existing Provider label/select in `#runBacktestProviderControl`. Keep it in the DOM for BYOK and set `hidden` for ATL Credits. Change the ATL hint to say that ATL Credits automatically chooses an available provider, with OpenRouter preferred.

- [ ] **Step 4: Implement deduplicated model inventory and submit validation**

For platform credits, collect models from `availableRunBacktestProviders('platform_credits')`, deduplicate by normalized `model_id`, and preserve the first provider's label. For BYOK, keep the current provider-specific list. Require only billing mode plus model for ATL Credits; require provider plus model for BYOK.

- [ ] **Step 5: Omit ATL provider input from the request**

Set `selectedProviderId` only in the BYOK branch. Keep `launchConfigBase.providerId` null for the user-facing ATL launch; completed run rendering can continue to use backend execution evidence.

- [ ] **Step 6: Run frontend contracts and static syntax checks**

Run: `pytest dashboard/backend/tests/test_byok_backtest_frontend.py dashboard/backend/tests/test_credits_frontend.py -q`

Expected: PASS. Run: `node --check dashboard/frontend/app.js` and expect exit code `0`.

- [ ] **Step 7: Commit the UI behavior**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/backend/tests/test_byok_backtest_frontend.py
git commit -m "feat: simplify ATL credits model selection"
```

### Task 5: Full Verification And PR Handoff

**Files:**
- No new production files.
- Test: all existing backend test files under `dashboard/backend/tests`.

- [ ] **Step 1: Run the full backend suite**

Run: `pytest dashboard/backend/tests -q`

Expected: PASS with the existing database file still unstaged.

- [ ] **Step 2: Inspect the diff and secret boundary**

Run: `git diff origin/main...HEAD --stat`, `git diff origin/main...HEAD -- dashboard/frontend dashboard/backend`, and `git status --short`.

Confirm no API key, database file, `.superpowers/`, or `work/` path is staged.

- [ ] **Step 3: Push the feature branch**

```bash
git push -u origin feature/platform-provider-failover
```

- [ ] **Step 4: Report the PR-ready commit set**

Provide the branch, commits, tests, and the expected latency behavior: quota/credential failures fail fast; only provider timeouts add one timeout window.
