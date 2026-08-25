# Unified BYOK Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Make Platform Credits overage settlement lossless, persist real non-secret execution evidence, expose LLM debits in Credits Activity, and allow legacy agents to select a valid one-run model.

**Architecture:** Extend the existing reservation sub-ledger instead of introducing a second debt system. Carry typed execution evidence through the compatibility client into run metadata, merge both Credit ledgers behind one stable activity cursor, and keep the frontend selection change isolated to the Run Backtest modal.

**Tech Stack:** Python 3, Pydantic, SQLite, PostgreSQL/psycopg, FastAPI, vanilla JavaScript, pytest.

**Spec:** docs/superpowers/specs/2026-08-25-unified-byok-review-fixes-design.md

## Global Constraints

- Do not commit or push changes.
- Never persist or return a raw provider credential.
- Keep BYOK and Platform Credits credential resolution strictly separate.
- SQLite and PostgreSQL must expose equivalent settlement and activity behavior.
- Preserve historical run, reservation, and numeric ledger-cursor compatibility.
- Do not modify Stripe or Render configuration.
- Run only targeted tests for changed components.

---

### Task 1: Atomically settle Platform Credits overages

**Files:**
- Modify: dashboard/backend/domain/credits/models.py
- Modify: dashboard/backend/domain/credits/repository.py
- Modify: dashboard/backend/domain/credits/repository_postgres.py
- Modify: dashboard/backend/domain/credits/service.py
- Modify: dashboard/backend/infrastructure/llm/execution/models.py
- Modify: dashboard/backend/infrastructure/llm/token_cost.py
- Modify: dashboard/backend/infrastructure/llm/execution/service.py
- Modify: dashboard/backend/tests/test_credit_metering.py
- Modify: dashboard/backend/tests/domain/credits/test_repository_postgres.py

**Interfaces:**
- Consumes: CreditsService reservation methods, Grant-first bucket allocation, and credit_accounts.status.
- Produces: settlement results with actual_micro and outstanding_micro, BillingEvidence.provider_cost_credits_micro, and atomic account restriction.

- [ ] **Step 1: Add failing SQLite overage tests**

Add a test with this accounting contract:

~~~python
def test_platform_overage_debits_reservation_and_restricts_account(credits_service):
    reservation = credits_service.reserve_llm_credits(
        user_id=7,
        run_id="run-overage",
        call_index=0,
        amount_micro=1_000_000,
    )
    settled = credits_service.settle_llm_credits(
        reservation.reservation_id,
        actual_micro=1_250_000,
        evidence={"provider_id": "openrouter", "model_id": "openai/gpt-5.5"},
    )
    assert settled.status == "settled"
    assert settled.actual_micro == 1_250_000
    assert settled.settled_micro == 1_000_000
    assert settled.outstanding_micro == 250_000
    assert credits_service.get_balance(7).account_status == "restricted"
~~~

Add replay assertions: identical input returns the same result; a replay with a different actual amount or evidence raises LLMReservationConflictError. Add a normal settlement assertion with zero outstanding amount.

- [ ] **Step 2: Verify the tests fail**

Run:

~~~bash
pytest dashboard/backend/tests/test_credit_metering.py -k 'overage or settlement_replay' -v
~~~

Expected: failure because the models/schema lack actual_micro and outstanding_micro and the store rejects costs above the reservation.

- [ ] **Step 3: Extend typed models and pricing evidence**

Add non-negative actual_micro and outstanding_micro fields to LLMReservation and LLMSettlementResult, defaulting to zero for historical rows.

Add these BillingEvidence fields:

~~~python
provider_cost_credits_micro: int = Field(default=0, ge=0)
outstanding_credits_micro: int = Field(default=0, ge=0)
~~~

Keep debited_credits_micro as the amount actually deducted. In build_cost_evidence calculate cost_micro once, store it in provider_cost_credits_micro, debit it only for Platform Credits, and leave BYOK debit at zero.

- [ ] **Step 4: Add backward-compatible database migrations**

Add actual_micro and outstanding_micro columns to both create-table definitions.

For SQLite, inspect PRAGMA table_info(credit_llm_reservations) during store initialization and execute each missing ALTER TABLE independently:

~~~sql
ALTER TABLE credit_llm_reservations
ADD COLUMN actual_micro INTEGER NOT NULL DEFAULT 0;

ALTER TABLE credit_llm_reservations
ADD COLUMN outstanding_micro INTEGER NOT NULL DEFAULT 0;
~~~

For PostgreSQL, add equivalent BIGINT NOT NULL DEFAULT 0 statements with ADD COLUMN IF NOT EXISTS to the existing migration batch.

- [ ] **Step 5: Implement atomic settlement in both stores**

Inside the existing settlement transaction calculate:

~~~python
reserved_micro = int(reservation["reserved_micro"])
debit_micro = min(actual_micro, reserved_micro)
outstanding_micro = actual_micro - debit_micro
~~~

Allocate debit_micro over the held Grant/Purchased buckets, insert usage rows, update all settlement amounts and evidence, mark the reservation settled, and set the account status to restricted when outstanding_micro is positive before committing.

Idempotent replay compares actual_micro, settled_micro, outstanding_micro, and canonical evidence. Update both reservation-result helpers to return the new fields.

- [ ] **Step 6: Return successful overage execution evidence**

In LLMExecutionService derive the final evidence before settlement:

~~~python
actual_micro = billing.provider_cost_credits_micro
debited_micro = min(actual_micro, reserved_micro)
outstanding_micro = actual_micro - debited_micro
final_billing = billing.model_copy(
    update={
        "debited_credits_micro": debited_micro,
        "outstanding_credits_micro": outstanding_micro,
    }
)
~~~

Pass actual_micro plus final evidence to settle_llm_credits. Remove the pre-settlement overage exception. Validate the returned amounts; repository failures still map to billing_failed.

- [ ] **Step 7: Add PostgreSQL parity tests and run Task 1 tests**

Mirror the normal, overage, replay, and restriction assertions using the established PostgreSQL fixture.

Run:

~~~bash
pytest dashboard/backend/tests/test_credit_metering.py dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'llm or overage or settlement' -v
~~~

Expected: selected tests pass; established environment-based PostgreSQL skips are acceptable.

- [ ] **Step 8: Review Task 1 diff without committing**

Inspect only the Credits and execution files. Confirm no checkout, webhook, Stripe gateway, Render, or deployment configuration changed.

---

### Task 2: Persist real execution evidence with completed backtests

**Files:**
- Modify: dashboard/backend/infrastructure/llm/execution/models.py
- Modify: dashboard/backend/infrastructure/llm/execution/client.py
- Modify: dashboard/backend/domain/backtesting/engine.py
- Modify: dashboard/backend/api/routers/backtests.py
- Modify: dashboard/frontend/app.js
- Create: dashboard/backend/tests/infrastructure/llm/test_execution_client.py
- Modify: dashboard/backend/tests/test_backtests_router.py
- Modify: dashboard/backend/tests/test_byok_backtest_frontend.py

**Interfaces:**
- Consumes: complete LLMExecutionResult values.
- Produces: LLMRunEvidence, AnthropicCompatibleExecutionClient.execution_summary(), metadata.llm_execution, and RunMetadata.llm_execution.

- [ ] **Step 1: Add failing client aggregation tests**

Use a fake service returning two complete results. Assert the compatibility response remains unchanged and execution_summary returns consistent identity, two calls, summed tokens/costs/debits, and settled_overage when outstanding Credits are non-zero.

Add a BYOK result with usage_available=False and assert the summary also reports false rather than authoritative zero usage.

- [ ] **Step 2: Verify aggregation tests fail**

Run:

~~~bash
pytest dashboard/backend/tests/infrastructure/llm/test_execution_client.py -v
~~~

Expected: failure because the client currently discards complete execution results.

- [ ] **Step 3: Add LLMRunEvidence and accumulate complete results**

Define an immutable Pydantic model containing safe identity, call and token totals, usage availability, provider and estimated cost totals, pricing snapshot, debit/outstanding totals, and billing outcome.

Append each successful LLMExecutionResult immediately after execute returns. execution_summary must:

- reject inconsistent provider/model/lane identity;
- sum token, estimated-cost, debit, and outstanding fields;
- mark usage available only if every call has usage;
- expose provider cost only when every call reports it;
- retain only safe credential id and key last four; and
- report settled_overage when any outstanding amount exists.

- [ ] **Step 4: Persist evidence and stop static repricing**

In _agent_run_metadata attach summary.model_dump(mode="json") under llm_execution.

For unified execution runs, set est_cost_usd from the complete provider-cost total when available, otherwise from the accumulated captured-snapshot estimate. Use legacy token_cost.estimate_cost_usd only when no unified summary exists.

- [ ] **Step 5: Expose evidence through run APIs**

Add this backward-compatible field to RunMetadata:

~~~python
llm_execution: Optional[Dict[str, Any]] = None
~~~

Lift metadata["llm_execution"] in _run_metadata_response. Add tests for a new evidence-bearing run and a historical run without evidence. Assert a fixture raw secret is absent from serialized responses.

- [ ] **Step 6: Make completed frontend results backend-authoritative**

For completed runs read run.llm_execution before launchConfig. Use launchConfig only while running or for historical rows without backend evidence. Display usage as unavailable when the evidence flag is false and use backend provider/model/lane for the Billing label.

- [ ] **Step 7: Run Task 2 tests**

Run:

~~~bash
pytest dashboard/backend/tests/infrastructure/llm/test_execution_client.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_byok_backtest_frontend.py -v
~~~

Expected: all selected tests pass.

- [ ] **Step 8: Review Task 2 diff without committing**

Confirm persisted and returned credential data is limited to identifier and key last four.

---

### Task 3: Merge LLM usage into Credits Activity

**Files:**
- Modify: dashboard/backend/domain/credits/repository_common.py
- Modify: dashboard/backend/domain/credits/repository.py
- Modify: dashboard/backend/domain/credits/repository_postgres.py
- Modify: dashboard/backend/domain/credits/service.py
- Modify: dashboard/backend/api/routers/credits.py
- Modify: dashboard/frontend/js/credits.js
- Modify: dashboard/backend/tests/domain/credits/test_repository.py
- Modify: dashboard/backend/tests/domain/credits/test_repository_postgres.py
- Modify: dashboard/backend/tests/test_credits_api.py
- Modify: dashboard/backend/tests/test_credits_frontend.py

**Interfaces:**
- Consumes: historical credit_ledger_entries and per-bucket credit_llm_usage_entries.
- Produces: normalized Activity pages and opaque cursor helpers encode_activity_cursor/decode_activity_cursor.

- [ ] **Step 1: Add failing activity and pagination tests**

Create purchase/Grant plus a two-bucket LLM settlement. Assert the two usage rows become one llm_usage item with the summed negative amount, run id, provider, model, and an opaque string next_cursor.

Add equal-timestamp pagination coverage proving no duplicate/missing items. Add a request using an existing numeric historical-ledger cursor.

- [ ] **Step 2: Verify activity tests fail**

Run:

~~~bash
pytest dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py -k 'ledger or activity' -v
~~~

Expected: failure because only the historical ledger is queried and cursors are integers.

- [ ] **Step 3: Implement shared cursor helpers**

Encode the ordering tuple [created_at, source_kind, source_id] as compact JSON using URL-safe base64 without padding. Decode with strict type/length checks and raise ValueError("invalid activity cursor") for malformed input.

For decimal legacy cursors, look up the user's historical ledger row and convert its timestamp, source kind ledger, and id to the same tuple.

- [ ] **Step 4: Implement equivalent UNION queries**

Create historical_activity and llm_activity CTEs. Aggregate LLM buckets by reservation/run/call/evidence/timestamp with SUM(amount_micro) and MAX(id). UNION ALL both sources, apply the lexicographic cursor boundary, then order by:

~~~sql
ORDER BY created_at DESC, source_kind DESC, source_id DESC
LIMIT page_size + 1
~~~

Parse evidence JSON defensively after the query. Return provider_id, model_id, and billing_source when valid; malformed evidence leaves them null without hiding the debit.

- [ ] **Step 5: Update API cursor and response shape**

Accept cursor as str | None. Extend _public_ledger_entry with safe optional LLM fields while retaining historical fields. Never return raw evidence_json.

Add API pagination tests using the returned opaque cursor.

- [ ] **Step 6: Render negative Model usage entries**

In renderLedger add an llm_usage branch with title Model usage, a negative amount, provider/model text, and shortened run id. Keep purchase, refund, and Grant rendering compatible.

- [ ] **Step 7: Run Task 3 tests**

Run:

~~~bash
pytest dashboard/backend/tests/domain/credits/test_repository.py dashboard/backend/tests/domain/credits/test_repository_postgres.py dashboard/backend/tests/test_credits_api.py dashboard/backend/tests/test_credits_frontend.py -k 'ledger or activity or llm_usage' -v
~~~

Expected: selected tests pass with established PostgreSQL skips allowed.

- [ ] **Step 8: Review Task 3 diff without committing**

Confirm Activity responses contain no raw evidence JSON or credential values.

---

### Task 4: Fall back from legacy models to a valid one-run model

**Files:**
- Modify: dashboard/frontend/app.js
- Modify: dashboard/backend/tests/test_byok_backtest_frontend.py

**Interfaces:**
- Consumes: execution options and setRunBacktestBillingMode.
- Produces: exact-match-first fallback selection and a one-run override hint.

- [ ] **Step 1: Add failing selection-order tests**

Assert selection order is pending BYOK, exact BYOK, exact Platform Credits, first BYOK fallback, then first Platform Credits fallback. Assert unavailable is reached only when neither lane contains a provider/model.

- [ ] **Step 2: Verify the test fails**

Run:

~~~bash
pytest dashboard/backend/tests/test_byok_backtest_frontend.py -k 'legacy or fallback or execution_options' -v
~~~

Expected: failure because the existing loader requires the saved model to match.

- [ ] **Step 3: Implement one-run fallback**

After exact-model searches fail, choose the first BYOK provider with a model, otherwise the first Platform Credits provider with a model. Pass its first model to setRunBacktestBillingMode and show:

~~~text
Saved model is unavailable; this run will use <model label> instead.
~~~

Do not mutate agent.model_name or send an agent update.

- [ ] **Step 4: Run Task 4 tests**

Run:

~~~bash
pytest dashboard/backend/tests/test_byok_backtest_frontend.py -v
~~~

Expected: all tests in the file pass.

- [ ] **Step 5: Review Task 4 diff without committing**

Confirm pending deep links and exact saved-model matches retain priority.

---

### Task 5: Targeted integration and safety verification

**Files:**
- Verify all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: completed accounting, persistence, API, and frontend changes.
- Produces: targeted test evidence and a final uncommitted diff.

- [ ] **Step 1: Run the focused combined tests**

Run:

~~~bash
pytest \
  dashboard/backend/tests/test_credit_metering.py \
  dashboard/backend/tests/domain/credits/test_repository.py \
  dashboard/backend/tests/domain/credits/test_repository_postgres.py \
  dashboard/backend/tests/infrastructure/llm/test_execution_client.py \
  dashboard/backend/tests/test_backtests_router.py \
  dashboard/backend/tests/test_credits_api.py \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_byok_backtest_frontend.py \
  -v
~~~

Expected: runnable targeted tests pass; only established environment-based PostgreSQL skips are accepted.

- [ ] **Step 2: Run secret and deployment-config checks**

Run the existing review patterns against git diff for Stripe/OpenRouter/database keys, private-key headers, Render, .env, YAML/TOML, and Docker files. Expected: no secret-value or deployment-config match.

- [ ] **Step 3: Inspect final changes**

List the diff stat and changed filenames. Confirm every changed file belongs to the approved design or targeted tests/documentation. Do not commit.

