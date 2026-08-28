# Backtest Credit Activity Precision and Aggregation Design

**Date:** 2026-08-28

**Status:** Approved
**Scope:** Exact ATL Credit display and run-level Credits activity

## Goal

Make ATL Credit balances and debits auditable at the platform's authoritative
micro-Credit precision, and show one settled debit per backtest instead of one
row per model call.

The Credits ledger already stores integer micro-Credits, where one Credit is
`1,000,000` micro-Credits. This change preserves that accounting model. It
fixes presentation and activity projection only; it does not recalculate,
round, or migrate stored charges.

## Current problem

The Credits & Billing frontend truncates `display_credits` to two decimal
places. A valid debit smaller than `0.01` Credits therefore appears as
`-0.00`, even though the API and ledger contain an exact six-decimal value.
The balance has the same two-decimal limitation.

The Activity query currently groups settled usage by reservation and
`call_index`. A single backtest can make many model calls, so one backtest
produces many nearly identical `Model usage` rows. This makes it difficult to
answer the user-level question: how many ATL Credits did this backtest cost?

## Product decisions

1. Every displayed ATL Credit amount uses exactly six decimal places.
2. Activity groups settled model usage by account and `run_id`.
3. A run appears once when it has at least one settled debit, whether the
   backtest ultimately succeeded or failed.
4. A failed or cancelled run with no settled debit does not create a zero-cost
   Activity row.
5. Per-call usage remains in the append-only accounting tables and analytics
   evidence. Activity is a read projection, not a replacement ledger.

## Exact amount display

All ATL Credit amounts rendered in the dashboard's billing and analytics
surfaces use a fixed six-decimal representation:

- `4.79` becomes `4.790000 Credits`;
- `-0.000137` remains `-0.000137 Credits`;
- `10` becomes `10.000000 Credits`.

Formatting must start from the authoritative integer `amount_micro` or the
server-provided six-decimal string. It must not round through a binary floating
point value. Thousands separators may be added to the integer portion, but the
fractional portion must contain exactly six digits. Invalid or missing amounts
render as the existing unavailable marker instead of a fabricated zero.

This fixed-precision contract applies to:

- the Credits & Billing balance;
- Credits & Billing Activity amounts;
- Admin Grant Credits balances and activity; and
- Admin Analytics ATL Credit amounts.

USD checkout and refund amounts remain currency values with two decimal
places. Token counts, ratios, and non-Credit quotas keep their existing
formatting.

## Run-level activity projection

### Aggregation identity

SQLite and PostgreSQL must produce the same projection. Settled rows from
`credit_llm_usage_entries` are grouped by `user_id` and `run_id`.

For each run-level group:

- `amount_micro` is the exact integer sum of all Grant and Purchased bucket
  usage rows for every settled model call in the run;
- `model_call_count` is the number of distinct settled `call_index` values;
- `created_at` is the latest settlement timestamp in the run;
- the stable activity identifier is derived from the greatest usage-entry id
  in the group; and
- the public `entry_type` is `backtest_usage`.

The amount remains negative because Activity represents a debit. Grant and
Purchased rows for the same call are combined before the user sees the result.
The underlying bucket allocation is not exposed by this summary.

### Provider and model context

Safe pricing evidence is used only to summarize provider and model identity.
The API returns a provider or model id only when all settled calls in the run
agree on that value.

- One provider and one model: return both ids.
- Multiple providers: return no single provider id and mark the provider
  summary as mixed.
- Multiple models: return no single model id and mark the model summary as
  mixed.
- Missing or malformed historical evidence: return unknown context without
  failing or dropping the exact debit.

Raw `evidence_json`, provider response bodies, credential identifiers, and API
keys never enter the public response.

### Ordering and pagination

Run summaries and purchase, refund, and Admin Grant ledger entries continue to
share one reverse-chronological Activity feed. A run summary is ordered by its
latest settlement timestamp, with its greatest underlying usage-entry id as
the deterministic tie-breaker.

The opaque cursor continues to encode the projected row's timestamp, source
kind, and stable identifier. Aggregation happens before the page limit is
applied, so a backtest cannot be split across pages and a page cannot show a
partial run total. Existing decimal legacy cursors for historical ledger rows
remain accepted.

A running backtest can acquire additional settled calls after an Activity
response. A later refresh shows the new exact total and latest settlement
time. The endpoint does not promise snapshot isolation across separate page
requests.

## API response

`GET /api/credits/ledger` remains the authenticated endpoint. Purchase,
refund, and Admin Grant rows retain their current public shape. A run-level
usage item contains the existing safe fields plus:

```json
{
  "source_kind": "llm_usage",
  "entry_type": "backtest_usage",
  "amount_micro": -1284,
  "display_credits": "-0.001284",
  "run_id": "run_example",
  "model_call_count": 12,
  "provider_id": "openrouter",
  "model_id": "anthropic/claude-haiku-4-5",
  "provider_mixed": false,
  "model_mixed": false,
  "created_at": "2026-08-27T15:44:00+00:00"
}
```

`source_kind` stays `llm_usage` so the existing opaque cursor domain remains
compatible. `call_index` and `reservation_id` are omitted from run-level public
items because they describe an individual model call rather than a backtest.

## Frontend behavior

Credits & Billing renders one `Backtest usage` row per run. Its secondary text
contains:

- the safe provider/model summary when available;
- `1 model call` or `<N> model calls`;
- the existing shortened run id; and
- the latest settlement time.

Mixed provider/model runs use explicit `Multiple providers` or `Multiple
models` copy. Unknown historical context is omitted rather than guessed. The
amount uses the server's exact signed six-decimal Credit string.

No disclosure control for per-call details is added in this change. Per-call
inspection remains an Admin Analytics or backend audit concern.

## Failure and compatibility behavior

- A malformed amount from the API renders as unavailable and does not crash
  the Activity list.
- Malformed historical evidence cannot prevent the exact run debit from being
  returned.
- A run with settled debits is visible even if its backtest status is failed.
- Released reservations and failed calls with no usage entry contribute zero
  rows and zero debit.
- Existing ledger data needs no schema migration or backfill.
- SQLite and PostgreSQL repository contracts must remain behaviorally
  equivalent.

## Security and privacy

The existing append-only ledger and authenticated account boundary remain
unchanged. Aggregation is always scoped by `user_id`; equal `run_id` values
from different accounts must never be combined. The public serializer keeps
its allowlist and never returns raw billing evidence, credential material, or
provider payloads.

Tests use synthetic micro-Credit amounts, safe mock pricing evidence, and fake
run ids. Real API keys, local databases, `.superpowers/`, and `work/` are not
committed.

## Test coverage

Focused regression tests cover:

- fixed six-decimal formatting for whole, two-decimal, sub-cent, negative,
  large, invalid, and missing Credit values without floating point rounding;
- one run-level row for multiple calls and two bucket rows per call;
- exact summed `amount_micro` and distinct `model_call_count`;
- failed-run settled usage remaining visible and zero-debit failures remaining
  absent by construction;
- single, mixed, unknown, and malformed provider/model evidence;
- aggregation before pagination, deterministic ordering, and legacy cursor
  compatibility;
- account isolation for repeated `run_id` values;
- identical SQLite and PostgreSQL public behavior;
- the API allowlist omitting call-level and secret evidence; and
- frontend title, metadata, pluralization, and exact balance/Activity display.

## Non-goals

- Changing reservation, settlement, Grant-first allocation, pricing, or
  provider cost calculation.
- Adding a new database table or rewriting historical ledger rows.
- Showing unreserved BYOK provider charges as ATL Credit debits.
- Adding per-call expansion to the Credits & Billing page.
- Changing backtest execution or failure-state rules.
- Changing Stripe Test Mode purchase or refund behavior.

## Acceptance criteria

1. A balance of `4,790,000` micro-Credits displays as `4.790000 Credits`.
2. A settled debit of `137` micro-Credits displays as `-0.000137`, never
   `-0.00`.
3. Twelve settled model calls for one `run_id` appear as one `Backtest usage`
   row whose amount equals the exact sum of all twelve calls.
4. The row states `12 model calls` and contains safe provider/model context or
   an explicit mixed summary.
5. A charged failed backtest remains in Activity; an uncharged failure does
   not create a usage row.
6. One run is never split across Activity pages.
7. Purchase, refund, Admin Grant, balance accounting, and BYOK behavior remain
   unchanged apart from fixed six-decimal ATL Credit presentation.
8. No public response, test fixture, commit, or PR exposes a real secret or
   local database.
