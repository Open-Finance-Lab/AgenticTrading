# Unified LLM Execution Layer Design

**Date:** 2026-08-24  
**Status:** Proposed  
**Scope:** Real model-backed dashboard backtests for Platform Credits and BYOK

## Goal

Replace the current environment-only or rule-based fallback behavior with one
server-side execution layer that can run a backtest through OpenAI, OpenRouter,
Anthropic, Gemini, or an approved OpenAI-compatible provider.

The caller chooses an explicit billing mode:

- `platform_credits`: ATL uses a verified platform credential and charges ATL
  Credits after the provider reports usage.
- `byok`: ATL uses the caller's verified default credential and charges **zero**
  ATL Credits. The provider bill belongs to the caller.

BYOK and Platform Credits are separate payment lanes. A BYOK run must never
silently consume Grant Credits or Purchased Credits, and a Platform Credits run
must never silently use a user's BYOK key.

## Current gap

The repository already has provider registration, encrypted credential storage,
verification, model-list discovery, token-cost helpers, and Grant/Purchased
Credit ledgers. The dashboard backtest worker still creates its LLM client from
server environment variables and the purchased Credits ledger is not connected
to that worker. A verified user key can therefore exist without affecting a
backtest.

## Architecture

### 1. Execution request and result

Add a small execution service with explicit typed inputs and outputs. The
request contains `user_id`, `run_id`, `billing_mode`, `provider_id`,
`model_id`, system/user messages, and a usage policy. It never accepts a raw
API key from the browser.

The result contains normalized response text, provider/model identity,
`input_tokens`, `output_tokens`, `total_tokens`, provider-reported cost when
available, the pricing snapshot used for estimation, and the billing source.
The result also reports whether usage is authoritative or unavailable.

### 2. Credential resolution

The service resolves credentials server-side:

- Platform mode requires an enabled provider with `platform_enabled=true` and
  a verified platform credential.
- BYOK mode requires an enabled provider with `byok_enabled=true`, a verified
  user credential belonging to `user_id`, and exactly one verified default for
  that provider.

Secrets are decrypted only immediately before client construction. They are
  never placed in command arguments, URLs, API responses, persistent run rows,
  or logs. The subprocess handoff uses a short-lived protected channel and is
  destroyed after client construction.

### 3. Provider adapters

All providers implement one execution contract while keeping provider-specific
wire details inside adapters. The adapters normalize:

- request messages and model identifiers;
- response text extraction;
- usage extraction;
- provider-reported cost when present; and
- safe, non-secret error categories.

The existing discovery adapters remain reusable for verification. Execution
adapters must not permit arbitrary custom origins; approved OpenAI-compatible
origins retain the existing DNS/IP pinning rules.

### 4. Billing and settlement

Platform mode reserves Credits before the first model call. After every model
call, the service settles the reservation using the provider/model pricing
snapshot and authoritative usage. Unused reserved Credits are released. A
failed call, timeout, parse failure, or run with no model call releases the
reservation and does not create a successful billable run.

BYOK mode records usage and estimated/provider-reported cost evidence but does
not create any ATL Credit reservation, debit, refund, or bucket movement.

Grant/Purchased bucket policy remains the existing ledger policy and is applied
only by Platform mode. The execution layer does not invent a second quota.

### 5. Backtest behavior

The backtest request and UI carry an explicit execution mode and provider/model
selection. If the selected credential is unavailable, the request fails before
the worker starts. If the provider call fails, the run is visibly failed. There
is no silent rule-based fallback for an explicitly requested LLM run.

Each run stores only non-secret evidence: execution mode, provider, model,
credential identifier or safe fingerprint, token usage, cost snapshot, billing
outcome, and failure category.

## Failure handling

- Invalid, revoked, or missing credentials: reject before starting the run.
- Provider/network/timeout error: mark the run failed; release any Platform
  reservation; never charge BYOK Credits.
- Missing usage on a Platform call: fail closed for billing, release the
  reservation, and do not apply the model result as a successful billable run.
- Missing usage on a BYOK call: keep the run result available, record usage as
  unavailable, and show that cost evidence is incomplete.
- User cancellation or worker crash: finalizer releases any outstanding
  Platform reservation idempotently.

## Security and privacy

- Full API keys stay inside the encrypted vault and short-lived server memory.
- No raw key appears in subprocess arguments, environment dumps, logs, URLs,
  frontend state, database rows, or error responses.
- Provider errors are mapped to fixed safe categories; upstream response bodies
  are not copied into user-visible errors.
- A BYOK run cannot be converted into a Platform Credits run by fallback.

## Out of scope

- Live trading orders.
- Automatic provider fallback after a failed call.
- Charging BYOK users ATL Credits.
- New provider onboarding UI beyond the existing registry and key vault.
- Changing Stripe purchase, refund, or webhook semantics.

## Acceptance criteria

1. A verified OpenAI BYOK key can run a real GPT backtest and the run records
   provider/model/usage without changing any ATL Credit balance.
2. A Platform Credits run uses only a verified platform credential, reserves
   and settles Credits, and records a pricing/usage snapshot.
3. OpenRouter, Anthropic, Gemini, and approved OpenAI-compatible providers use
   the same execution interface and normalized result shape.
4. Missing credentials, provider failures, and usage failures are visible and
   never silently become rule-based success.
5. No test, log, API response, commit, or PR contains a real API key.
