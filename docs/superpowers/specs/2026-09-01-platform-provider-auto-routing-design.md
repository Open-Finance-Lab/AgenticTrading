# Platform Provider Auto-Routing Design

## Goal

When a user runs an LLM backtest with `Use ATL Credits`, the UI exposes only the approved model list. The backend automatically tries OpenRouter first and falls back to CommonStack when the first provider cannot serve the request because of quota, balance, credential availability, timeout, or temporary provider unavailability. BYOK keeps its explicit provider selection.

## Scope

- Hide the Provider selector only for the `platform_credits` billing lane.
- Keep the Provider selector and default-key validation for BYOK.
- Merge platform-available model options into one model picker without duplicate models.
- Accept an omitted `provider_id` for platform-credit LLM backtests.
- Resolve an ordered, model-compatible provider candidate list with OpenRouter before CommonStack.
- Preserve the requested model and record the provider that actually completed each call.
- Fail over only for provider-selection failures where another provider may succeed:
  `credential_missing`, `credential_invalid`, `provider_unavailable`, `provider_timeout`, and `provider_quota_exhausted`.
- Do not fail over `response_invalid`, `usage_unavailable`, `billing_failed`, or account-level ATL credit restrictions.
- Never expose provider secrets or raw upstream errors.

## Architecture

The browser continues to load `/api/credits/execution-options`, but treats OpenRouter and CommonStack platform models as one ATL Credits inventory. It sends `billing_mode=platform_credits`, `model`, and no user-selected provider. The route resolves the first compatible enabled platform credential in preferred order (`openrouter`, then `commonstack`) and signs that choice into the handoff. The execution service retries the same catalog model through the next candidate when the first attempt returns a failover-safe category; each attempt owns an independent reservation and settlement lifecycle.

The signed handoff carries the ordered provider candidates so the worker can make the same decision without trusting client input. Existing BYOK handoffs remain single-provider. A fallback result uses the original requested provider in `requested_provider_id` and the successful provider in `provider_id`, preserving analytics and run evidence.

## API Contract

`POST /backtest/run`:

- BYOK LLM runs: `billing_mode=byok`, `provider_id` required, `model` required.
- ATL Credits LLM runs: `billing_mode=platform_credits`, `provider_id` optional, `model` required. When omitted, the server resolves the ordered candidate list. A supplied provider is accepted only for backwards compatibility and is treated as the first candidate, not as a UI-controlled preference.
- If no compatible platform provider is configured, return a safe `422` explaining that ATL Credits model execution is unavailable.

`ExecutionHandoff`:

- `provider_id` remains the first candidate for backwards compatibility.
- `provider_ids` is an ordered unique tuple with one or more provider IDs for platform-credit handoffs; BYOK contains one ID.
- The prompt digest covers the candidate list through the signed payload.

## UI Behavior

- `Use my API key`: show Provider and Model controls and retain current copy and validation.
- `Use ATL Credits`: hide the Provider label/select, show one deduplicated Model select, and explain that ATL Credits automatically chooses the best available provider.
- The submit button requires a selected model and an available ATL Credits model, but never a visible provider value.
- The launch config may retain the backend-selected provider for diagnostics, but user-facing billing copy says `ATL Credits`.

## Error Handling and Latency

Provider quota/balance failures are detected from the existing fixed error categories and trigger one fallback attempt. Missing or invalid platform credentials are filtered before the worker starts. A provider timeout may add the configured adapter timeout once; quota and credential failures normally fail quickly. No balance API is assumed, because the current provider contracts do not expose reliable cross-provider balance data.

## Testing

- Backend router contract tests cover omitted platform `provider_id`, ordered compatible candidates, and safe no-provider responses.
- Handoff tests cover signed candidate lists and replay/validation behavior.
- Execution tests cover fallback for each allowed category, no fallback for response/usage/billing/account errors, independent reservations, and final provider attribution.
- Frontend contract tests cover hidden platform Provider control, deduplicated models, payload omission of `provider_id` for ATL Credits, and unchanged BYOK behavior.
