# Platform Credits Environment-Key Fallback Design

Date: 2026-08-25

## Goal

Allow a user to run a model backtest with Admin-assigned Grant Credits when
the deployment already has the legacy shared `OPENROUTER_API_KEY`, without
requiring the same key to be copied into the encrypted Platform Credential
table.

The existing billing priority remains:

1. BYOK — use the user's verified default key and do not debit ATL Credits.
2. Grant Credits — reserve and settle ATL Credits using provider-reported usage
   and the pricing snapshot.
3. Purchased Credits — cover any remaining ATL Credits debit after Grant
   Credits.

## Scope and non-goals

In scope:

- Add an environment-backed fallback for the `openrouter` Platform Credits
  lane.
- Keep an Admin-managed, verified Platform Credential as the first choice.
- Keep the existing provider/model catalog, execution adapter, reservation,
  settlement, release, and audit behavior.
- Expose only safe availability metadata and the last four key characters.

Out of scope for this change:

- Copying environment secrets into SQLite or PostgreSQL.
- Supporting the legacy CommonStack auto-detection path in the unified
  execution catalog.
- Changing BYOK behavior or adding silent BYOK-to-Platform fallback.
- Adding new providers or changing model pricing.

## Credential resolution

For `billing_mode=platform_credits` and provider `openrouter`:

1. Require the provider to be enabled and `platform_enabled=true`.
2. Resolve a verified Admin Platform Credential from the encrypted credential
   store, if one exists.
3. If no verified stored credential exists, read `OPENROUTER_API_KEY` from the
   process environment at execution time.
4. If neither source is available, fail closed with the existing
   `credential_missing` category.

The environment value is transient and is never written to the credential
store, API response, browser storage, logs, or billing evidence. Its last four
characters may be returned in the internal execution result only, matching the
existing safe credential contract. The credential identifier remains `None`
for an environment-backed credential.

The fallback is provider-specific and explicit. It must not make an arbitrary
environment variable activate a provider, and it must not make BYOK use a
server secret.

## Execution-options API and frontend

`GET /api/credits/execution-options` will mark OpenRouter's
`platform_credits_available` as true when:

- the provider is enabled;
- `platform_enabled=true`; and
- either a verified stored Platform Credential exists or a non-empty
  `OPENROUTER_API_KEY` is present.

The frontend needs no new billing mode. Existing provider/model selection will
therefore enable `Use ATL Credits` when the fallback is available. The helper
copy should continue to state that ATL Credits are settled from actual model
usage. No secret-derived value is sent to the browser.

## Billing and failure behavior

The unified `LLMExecutionService` remains the only execution path:

- reserve the conservative usage ceiling before the provider call;
- call OpenRouter with the resolved credential;
- require provider usage for Platform Credits;
- settle actual cost using the existing price snapshot;
- allocate the debit Grant-first, then Purchased;
- release the reservation on provider, usage, or billing failure.

If a BYOK request fails, it remains a BYOK failure. It must never retry with
the environment-backed Platform Credits credential. A user must explicitly
select `Use ATL Credits` to incur an ATL Credits debit.

## Test coverage

Add focused tests for:

- environment-backed OpenRouter availability in execution options;
- stored verified Platform Credential taking precedence over the environment;
- missing environment and stored credential failing closed;
- environment secrets never appearing in API responses, logs, or browser
  source;
- Platform Credits still reserving, settling, and releasing through the
  existing Grant-first ledger behavior;
- BYOK never resolving the environment-backed credential.

The user will run the pytest and browser verification commands after each
implementation layer. Local databases and real API keys must not be committed.

## Acceptance criteria

- Admin enables OpenRouter's Platform Credits lane.
- With only the server's `OPENROUTER_API_KEY` configured, the user sees an
  enabled `Use ATL Credits` option and compatible models.
- A selected model can run a real backtest through OpenRouter.
- The resulting ledger records actual usage-based debit with Grant Credits
  consumed before Purchased Credits.
- Removing the environment variable disables the lane unless a verified stored
  Platform Credential exists.
- No complete API key is returned, persisted in the database, or exposed in
  frontend code.
