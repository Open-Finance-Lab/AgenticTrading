# Unified BYOK Review Fixes Design

## Scope

Repair the four issues found while reviewing PR #399 without changing the
fundamental BYOK and Platform Credits lane separation:

1. A provider-reported cost above the reservation ceiling must not release the
   entire hold after the provider has already charged ATL.
2. Non-secret execution, usage, pricing, and billing evidence must survive the
   Anthropic-compatible bridge and be stored with the completed backtest run.
3. LLM Credit debits must appear in the user-facing Credits activity stream.
4. A legacy agent model must not prevent the user from selecting another model
   for one backtest run.

Raw credentials remain excluded from command arguments, URLs, API responses,
run metadata, logs, and browser storage. SQLite and PostgreSQL must implement
the same accounting and response contracts.

## 1. Platform Credits overage settlement

### Reservation data

Extend `credit_llm_reservations` in both stores with two non-negative amounts:

- `actual_micro`: authoritative provider cost converted to Credit micro-units.
- `outstanding_micro`: provider cost not covered by the reservation.

Existing rows migrate with both values set to zero. The existing `status`
values remain `open`, `settled`, and `released`; an overage is a settled
reservation with `outstanding_micro > 0`. `settled_micro` remains the amount
actually debited and may never exceed `reserved_micro`.

The accounting invariants are:

- normal settlement: `actual_micro = settled_micro` and
  `outstanding_micro = 0`;
- overage settlement: `settled_micro = reserved_micro` and
  `outstanding_micro = actual_micro - settled_micro`;
- released reservation: all three settlement amounts are zero.

### Atomic transaction

`settle_llm_credits` accepts the authoritative `actual_micro` even when it is
greater than the reservation. Inside the same SQLite transaction or PostgreSQL
transaction it:

1. locks and validates the open reservation;
2. calculates `debit_micro = min(actual_micro, reserved_micro)`;
3. writes Grant-first/Purchased-second usage rows for `debit_micro`;
4. stores `actual_micro`, `settled_micro`, `outstanding_micro`, and evidence;
5. marks the reservation settled; and
6. changes the Credit account to `restricted` when `outstanding_micro > 0`.

This ordering has no crash window in which the debit is committed but the
account restriction is lost. Replaying the same settlement is idempotent;
replaying it with different actual cost or evidence remains a conflict.

### Execution evidence

`BillingEvidence` distinguishes three amounts:

- provider/billable Credit micro-units derived from authoritative cost;
- Credit micro-units actually debited;
- outstanding Credit micro-units.

`LLMExecutionService` no longer raises and releases the reservation merely
because authoritative cost exceeds the ceiling. It prepares the final evidence
using the known reservation ceiling, settles atomically, validates the returned
amounts, and returns the provider response. A restricted account cannot create
new Platform Credits reservations, while BYOK execution remains independent.

Administrators continue to use the existing account reinstatement operation
after external reconciliation of the outstanding amount.

## 2. Backtest execution evidence

### Bridge accumulation

`AnthropicCompatibleExecutionClient` continues returning the SDK-compatible
text/model/usage object expected by the backtest pipeline. In parallel, it
retains each complete `LLMExecutionResult` and exposes a deterministic summary
for the engine. The summary contains no raw credential material.

The run summary includes:

- billing mode, provider, canonical model;
- credential identifier and key last four when safe and available;
- call count, input/output token totals, and usage availability;
- provider-reported and snapshot-estimated cost totals;
- the immutable pricing snapshot used by the calls;
- debited and outstanding Credit micro-units; and
- billing outcome (`settled`, `settled_overage`, `byok`, or `unavailable`).

If any call lacks authoritative usage, the summary records usage as unavailable
instead of representing the run as authoritative zero-token usage.

### Run persistence and API

The engine writes the summary under `metadata.llm_execution` when it inserts the
agent run. For unified execution runs, `est_cost_usd` uses accumulated
provider-reported cost when available and otherwise the captured pricing
snapshot estimate; it does not reprice the run against the current global table.

`RunMetadata` and `_run_metadata_response` expose `llm_execution`. Existing run
rows without this field remain valid. The frontend reads completed-run billing,
provider, model, usage availability, and cost from this backend evidence.
Browser launch configuration remains a running-state placeholder only and may
not override completed backend evidence.

## 3. Unified Credits activity

### Repository query

`list_ledger_entries` returns a normalized activity stream composed of the
historical Credit ledger and aggregated LLM usage rows. Grant and Purchased
usage rows belonging to the same reservation/call are combined into one
`llm_usage` item with their signed amount summed.

Activity ordering is deterministic by `(created_at, source_kind, source_id)` in
descending order. Pagination uses an opaque URL-safe cursor containing that
tuple. A numeric cursor is still accepted as a legacy historical-ledger cursor;
new responses always return the opaque form.

LLM activity safely exposes `run_id`, provider, model, billing source, and the
stored non-secret cost evidence. Malformed legacy evidence is ignored rather
than failing the entire page.

### API and frontend

`GET /credits/ledger` returns both historical and `llm_usage` entries. The
Credits Activity renderer displays model usage as a negative amount and labels
it with model/provider plus a short run identifier. Purchase, refund, and Grant
rendering remain backward compatible.

## 4. Legacy model execution fallback

The Run Backtest modal keeps this selection order:

1. a valid pending BYOK deep link;
2. a BYOK provider supporting the agent's saved model;
3. a Platform Credits provider supporting the saved model;
4. the first available BYOK provider and its first model; or
5. the first available Platform Credits provider and its first model.

Steps 4 and 5 are one-run overrides and do not mutate the agent. The modal
shows a concise hint when an override occurs. The unavailable state is used
only when no provider/model exists in either lane.

## 5. Error handling and compatibility

- Provider, credential, and usage errors retain the existing sanitized error
  categories.
- A successful provider response with an overage remains a successful model
  call whose billing outcome is `settled_overage`.
- Database settlement failures still fail closed and do not return a successful
  model result.
- Existing reservations, ledger callers, and historical run rows remain
  readable after migration.
- No Stripe or Render configuration changes are required.

## 6. Verification

Targeted tests cover:

- normal and overage settlement, idempotent replay, account restriction, and
  SQLite/PostgreSQL parity;
- execution-result accumulation and run metadata/API round trips without
  secrets;
- provider cost taking precedence over static repricing;
- combined Activity ordering, aggregation, and opaque/legacy cursor behavior;
- frontend rendering of negative model usage and backend-first completed-run
  evidence; and
- legacy-model fallback to a valid one-run provider/model selection.

Only tests relevant to these changed components are required for this repair.
