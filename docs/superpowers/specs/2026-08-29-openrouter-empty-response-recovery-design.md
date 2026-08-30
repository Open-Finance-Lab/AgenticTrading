# OpenRouter Empty Response Recovery Design

**Date:** 2026-08-29
**Status:** Shipped (PR #420), amended by PRs #421, #422, the #421 review
follow-ups (#423, #425) and the non-pipeline recovery follow-up — see
*Amendments* below; the sections after it describe the original design.
**Scope:** Recover platform-credits backtests when OpenRouter returns no text

## Amendments (what actually shipped)

- **The recovery lever is a larger output budget, not disabled reasoning.**
  PR #421's follow-up commit and PR #422 replaced `reasoning_effort="none"`
  with re-sending the identical request at
  `max_tokens=RECOVERY_MAX_OUTPUT_TOKENS` (`max(LLM_MAX_OUTPUT_TOKENS, 4096)`)
  with reasoning preserved. Every adapter honours `max_tokens`; the native
  Gemini/Anthropic/OpenAI adapters had silently dropped `reasoning_effort`, so
  on those routes the earlier lever was a byte-identical second request.
- **A second trigger: a truncated first reply.** Besides `response_invalid`,
  a first reply that parses to nothing *and* was cut at the output ceiling
  earns the same single recovery attempt. Truncation is decided by the
  provider's own signal first — the normalised `stop_reason`/`finish_reason`
  (`max_tokens`, threaded from every adapter through `LLMExecutionResult`
  to the compatibility client's response) or `output_tokens` reaching the
  request's ceiling — and only then by the structural scan
  (`_looks_like_truncated_json`: an unclosed decision envelope of ≥64 chars,
  fences and preamble tolerated).
- **One recovery attempt per step, whichever trigger fires.** Both triggers
  send the same request, so a still-unusable recovery reply has nothing
  different left to ask for.
- **A failed truncation retry degrades; it does not raise.** The first reply
  was a real, billed call, so a retry the provider rejects as
  `response_invalid`, or that returns no text block, leaves the step at the
  first attempt's `None` and returns the usage already recorded — the caller
  side-effects those deltas into `llm_calls` (the billing counter) only from
  a normal return. Other error categories still propagate. The
  `response_invalid`-first path keeps its original contract: a second
  `response_invalid` re-raises.
- **The non-pipeline path recovers too.** The single-call route in
  `PortfolioManager.make_trading_decision_with_llm` — the one the leaderboard
  `llm_agent` drives, which never runs a pipeline — originally had no
  truncation recovery at all: a reply cut at the ceiling parsed to `None`
  and the step ended as a billed call with no H6 decision. It now applies the
  same two signals through the shared `pipeline_runner.truncation_reason`
  and spends the same single budget; that budget is also what the final
  rescue call for a run of no-text replies uses, so a truncated rescue reply
  does not buy a sixth call.

## Incident

The latest Qwen/OpenRouter platform-credits backtest reached the model call but
ended with the safe error `ProviderExecutionError: The model returned an
invalid response.` There was no Render out-of-memory signal. The OpenAI-
compatible adapter currently extracts only `choices[0].message.content`.
Reasoning models can spend the response budget on reasoning blocks, return an
empty `content`, or otherwise provide no final text. The adapter correctly
classifies that response as `response_invalid`, but the strict pipeline path
does not recover from it, so the worker raises `LLMDecisionError` and the UI
returns to its initial state without a result.

The existing `{"orders": []}` normalization fix is separate: it handles valid
JSON after text extraction and must remain unchanged.

## Goal

Give an OpenRouter reasoning model one bounded opportunity to return the same
pipeline response with reasoning disabled. A successful retry must continue
through the existing JSON parser and portfolio decision path. A second invalid
response must fail exactly as it does today.

## Request and provider contract

1. Preserve the selected OpenRouter provider, model, prompt, max-output-token
   limit, billing mode, and run identity for both attempts.
2. Keep the current OpenRouter reasoning configuration as the default first
   attempt. The configured `reasoning_effort` is passed through the existing
   execution request and adapter contract.
3. The recovery attempt explicitly passes `reasoning_effort="none"`. The
   existing OpenRouter adapter maps that override to the provider's
   reasoning-disabled payload, taking precedence over environment defaults.
4. The retry is available only after an execution error whose category is
   `response_invalid` (including an empty or reasoning-only response). It must
   not be triggered by malformed request data, credential failures, provider
   availability errors, timeouts, billing/usage failures, or invalid business
   JSON returned as non-empty text.
5. Retry at most once per pipeline step. There is no recursive retry, provider
   fallback, model substitution, or frontend polling change.

## Pipeline behavior

`run_pipeline_decision()` remains sequential. For each step it performs the
normal call first. If that call raises `LLMExecutionError` with category
`response_invalid`, it logs a safe recovery message and repeats the exact
request with `reasoning_effort="none"`. Each attempt receives its own
monotonic `call_index`; only completed responses contribute token usage and
the existing billable `llm_calls`/execution-summary aggregates.

If the retry returns text, parsing and `pipeline_output_to_decision()` proceed
normally. In particular, a valid `{"orders": []}` response remains the
model-driven HOLD represented by `{"actions": []}`. If the retry is also
empty/invalid, the original safe `response_invalid` error is re-raised so
strict LLM backtests fail closed. Non-strict callers retain their existing
fallback behavior.

The retry helper must work with the existing Anthropic-shaped execution client
and lightweight test doubles. It may add the override only to the retry
request; callers that do not accept `reasoning_effort` must not be silently
converted into a different error category.

## Billing and settlement

Each attempt is an independent call through the existing execution service.
The client continues to allocate a unique `call_index` and to run the normal
platform-credits reservation and settlement lifecycle for every attempt.

- A failed attempt releases its reservation according to the current service
  behavior.
- A successful retry settles only the usage reported for that retry; aggregate
  execution evidence continues to include completed calls using existing
  semantics. The failed empty-response attempt is not treated as a billable
  successful call.
- BYOK calls never reserve, debit, or refund ATL Credits; this change does not
  alter that lane.
- No new ledger, refund, quota, or pricing policy is introduced.

## Error and safety boundaries

- Keep all public errors in the existing fixed `ExecutionErrorCategory` set.
- Do not log upstream response bodies, reasoning text, API keys, or provider
  request headers.
- Do not treat an empty response as a HOLD. Only parseable empty envelopes
  such as `{"orders": []}` are HOLD decisions.
- Do not change Render memory, worker concurrency, deployment settings,
  database schema, frontend state, or UI copy.

## Implementation units

- `dashboard/backend/infrastructure/llm/execution/adapters/openai.py`: retain
  the safe empty-text classification and ensure the OpenRouter reasoning
  override reaches the wire without changing other providers.
- `dashboard/backend/infrastructure/llm/pipeline_runner.py`: add the bounded
  `response_invalid` recovery around each decision-step model call.
- `dashboard/backend/tests/infrastructure/llm/test_execution_adapter_model_routes.py`:
  cover OpenRouter reasoning payloads and empty-response classification.
- `dashboard/backend/tests/infrastructure/llm/test_pipeline_runner.py`: cover
  one retry, retry success, retry exhaustion, non-retryable errors, and empty
  `orders` HOLD handling.

## Testing strategy

Use deterministic fake clients and responses; no network calls or real
credentials.

1. Verify the first OpenRouter request preserves the configured reasoning
   effort and the recovery request uses `none`.
2. Verify an empty/ reasoning-only first response is retried once and a valid
   second response is returned with the expected token totals and existing
   billable call count.
3. Verify a second `response_invalid` raises and does not make a third call.
4. Verify credential, timeout, provider, billing, usage, and parse/business
   JSON errors are not retried.
5. Verify a retried `{"orders": []}` response is normalized to
   `{"actions": []}` and reaches the existing HOLD behavior.

Run the focused adapter and pipeline-runner tests, then the related backend
LLM/backtesting test modules. Existing tests for non-empty decisions,
OpenRouter environment defaults, and platform/BYOK settlement must remain
green.

## Acceptance criteria

- The reproduced Qwen/OpenRouter empty-text failure either recovers on one
  reasoning-disabled retry or reports the same safe failure promptly.
- A successful retry produces a persisted backtest result rather than silently
  returning to the initial page.
- Only `response_invalid` receives one retry; all other failures keep current
  fail-closed behavior.
- Every attempt uses the existing billing and secret-handling paths.
- No frontend, deployment, database, concurrency, or memory changes are part
  of this pull request.
