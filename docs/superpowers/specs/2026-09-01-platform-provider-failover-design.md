# Platform Provider Failover Design

Date: 2026-09-01

## Goal

Keep Platform Credits backtests available when OpenRouter explicitly rejects a
request because its shared account has no usable balance or quota. OpenRouter
remains the preferred platform provider. For an eligible OpenRouter failure,
the same logical model call may retry once through CommonStack.

The failover is deliberately narrow. It is not a general retry system, it does
not hide model or infrastructure failures, and it never moves a BYOK request
onto a platform-owned credential.

## Decisions

- OpenRouter remains the primary provider for Platform Credits requests that
  select OpenRouter.
- CommonStack is the only fallback provider, using its OpenAI-compatible API at
  `https://api.commonstack.ai/v1`.
- Failover requires an explicit upstream balance or quota exhaustion signal.
- A logical model call gets at most one CommonStack attempt.
- The retry preserves the canonical model, messages, system message, output
  token limit, temperature, and reasoning effort. Reasoning is never disabled
  or changed to `none` by failover.
- BYOK never uses this fallback, even when the user's OpenRouter key has no
  balance.
- Each provider attempt has an independent Credits reservation lifecycle, while
  at most one attempt can settle a debit for the logical call.
- Results, run evidence, Credits activity, and analytics identify the provider
  that actually completed each call.

## CommonStack provider registration

Add a seeded provider with this fixed configuration:

- `provider_id`: `commonstack`
- `display_name`: `CommonStack`
- `adapter_type`: `openai_compatible`
- `approved_base_url`: `https://api.commonstack.ai/v1`
- `byok_enabled`: `false`
- `platform_enabled`: `true`
- system messages and reasoning enabled in its declared capabilities
- `temperature`, `max_output_tokens`, and `reasoning_effort` declared as its
  supported parameters

The platform credential resolver maps `commonstack` explicitly to
`COMMONSTACK_API_KEY`, with the same stored-credential-first and transient
environment-key behavior already used by OpenRouter. The complete key is never
persisted from the environment or returned through an API, log, event, or run
record.

The initial CommonStack model allowlist contains only catalog identifiers that
the repository's CommonStack integration report verified:

- `openai/gpt-5.5`
- `google/gemini-3.1-pro-preview`
- `anthropic/claude-sonnet-4-6`
- `deepseek/deepseek-v4-pro`
- `qwen/qwen3.7-plus`

`anthropic/claude-haiku-4-5` is excluded until that exact route is verified.
An OpenRouter request for a model outside the CommonStack allowlist fails with
the original safe OpenRouter error and does not substitute another model.

Because CommonStack is a registered, platform-enabled provider, it may also be
selected directly when its credential is available. There is no reverse
CommonStack-to-OpenRouter failover. Platform execution options explicitly rank
OpenRouter ahead of CommonStack so adding the alphabetically earlier
CommonStack display name cannot change the existing default platform choice.

Seed initialization must add the provider to both new and existing SQLite and
PostgreSQL installations without overwriting later administrator changes.

## Eligible failure classification

Introduce a provider-neutral `provider_quota_exhausted` execution category.
The OpenAI-compatible adapter assigns it only when an upstream error contains
one of these explicit signals:

1. HTTP status `402`; or
2. a structured error code or type from a bounded allowlist, including
   `in_flight_budget_exhausted`, `insufficient_quota`, `quota_exceeded`,
   `quota_exhausted`, `insufficient_balance`, and
   `credit_balance_exhausted`; or
3. a bounded provider error-message field containing an approved balance or
   quota phrase such as `insufficient balance`, `insufficient credits`,
   `quota exceeded`, `quota exhausted`, `exceeded your current quota`, or
   `not enough credits`.

Classification inspects only an exception's structured status and error
payload. It never inspects successful model output. Provider payloads are
bounded before parsing and are never copied into logs, analytics, persisted
evidence, or user-visible errors.

HTTP `429` by itself is not a quota-exhaustion signal because it can represent
a temporary request-rate limit. Generic rate limits, timeouts, malformed or
empty responses, unsupported models, invalid credentials, network errors, and
provider `5xx` responses retain their existing categories and do not trigger
failover.

The safe user message for `provider_quota_exhausted` states that the selected
provider has insufficient balance or quota. It does not include upstream text,
account identifiers, or secret-derived data. Unrecovered errors map to the
same fixed `provider_quota_exhausted` analytics category rather than the
generic internal-error bucket.

## Routing flow

`LLMExecutionService` remains the only provider execution and Credits billing
entry point.

For `billing_mode=platform_credits` with requested provider `openrouter`:

1. Resolve OpenRouter, validate the requested model, resolve its platform
   credential, and execute the normal reserved call.
2. If the attempt succeeds, return it without contacting CommonStack.
3. If it fails with `provider_quota_exhausted`, first complete the failed
   attempt's reservation release.
4. Confirm that CommonStack is enabled, has a platform credential, and
   allowlists the same canonical model.
5. Create an immutable request copy whose only changed routing field is
   `provider_id=commonstack`, then execute one normal reserved call.
6. Return the CommonStack result or its final safe error. Never attempt a third
   provider call.

All other requested providers execute once with no automatic alternate route.
A missing or ineligible CommonStack route leaves the original OpenRouter quota
error as the final error.

No failover occurs after OpenRouter returned a usable completion. In
particular, missing usage, local settlement failure, response parsing failure,
or downstream worker failure must not issue a second provider request, because
the first request may already be billable.

For `billing_mode=byok`, execution remains exactly one call through the user's
selected provider and verified default credential. A BYOK quota error is
returned safely to the user and never resolves `OPENROUTER_API_KEY`,
`COMMONSTACK_API_KEY`, or an Admin-managed Platform Credential.

## Credits reservations and storage

A provider attempt must be independently idempotent and auditable. Extend LLM
reservation identity with a non-negative `attempt_index` and the attempted
`provider_id`:

- primary attempt: `attempt_index=0`, `provider_id=openrouter`;
- fallback attempt: `attempt_index=1`, `provider_id=commonstack`.

The reservation operation key, reservation identifier, request digest, and
uniqueness constraint include `attempt_index`. The logical call identity
remains `(user_id, run_id, call_index)` so aggregate call counts do not count a
fallback as a second model decision.

SQLite and PostgreSQL migrate existing reservation rows with
`attempt_index=0`. A nullable legacy `provider_id` is allowed only for rows
created before this migration; every new reservation requires a validated
provider identifier. Repository reads remain backward compatible.

The primary quota failure releases its complete reservation before the
fallback may reserve. If release fails, fail closed with `billing_failed` and
do not contact CommonStack. If fallback reservation fails, no fallback request
is sent. If CommonStack succeeds, only its reservation settles against actual
usage using the CommonStack pricing snapshot. Run finalization releases every
still-open attempt reservation.

These rules ensure that an eligible two-attempt call never holds both ceilings
at once and never produces two Credits debits.

## Result and analytics attribution

`LLMExecutionResult.provider_id`, credential last four, pricing snapshot, and
billing evidence describe the provider and credential that produced the
successful response. Add `requested_provider_id` so a recovered call still
records that the user selected OpenRouter. The new field is additive and
optional at the model boundary for compatibility with existing test doubles;
the production execution service always populates it.

`model_usage_recorded` uses the result's actual provider rather than the
original request provider. A recovered OpenRouter error does not emit a failed
run event; only an unrecovered final error emits the existing sanitized error
event.

Run-level `LLMRunEvidence` adds ordered unique `provider_ids` and a
`provider_mixed` flag. Its compatibility field behaves as follows:

- one actual provider across all completed calls: `provider_id` is that
  provider;
- both OpenRouter and CommonStack completed calls: `provider_id` is `mixed`,
  `provider_mixed=true`, and `provider_ids` contains both actual identifiers.

For a recovered call, the run's `requested_provider_id` remains OpenRouter. A
direct CommonStack run records CommonStack as both requested and actual.
Existing single-provider runs retain their existing field values, and
historical evidence without the new additive fields remains readable by
inferring `provider_ids=(provider_id,)` and `provider_mixed=false`. Credential
identity and pricing snapshot remain populated only when uniform across the
completed calls; mixed values stay `None` rather than reporting misleading
evidence.

Credits activity continues deriving provider attribution from each settled
reservation's billing evidence. A run containing settlements from both
providers renders as `Multiple providers` through the existing mixed-provider
contract.

## Dual-failure behavior

When CommonStack is actually attempted and fails, its safe category is the
final execution error. If both providers report explicit quota exhaustion, the
final category remains `provider_quota_exhausted`. If CommonStack times out or
is unavailable, the final category reflects that condition.

Only fixed categories and the ordered provider identifiers may be retained for
internal diagnostics. Neither upstream response body, full API key, prompt,
model response, nor account detail may enter exception messages, logs,
analytics, or persisted failure evidence.

## Out of scope

- Polling either provider's account balance before a model call.
- Round-robin, load-based, latency-based, or generic outage failover.
- Retrying timeouts, invalid responses, model errors, or generic rate limits.
- Switching the requested model or reducing its reasoning behavior.
- Falling back from BYOK to Platform Credits.
- Adding more than one CommonStack attempt per logical call.
- Mutating Render environment variables or copying a key between Render
  services as part of the code change.

## Verification

Focused unit and contract tests must cover:

- CommonStack seed parity and non-overwrite behavior in SQLite and PostgreSQL;
- stored and environment-backed `COMMONSTACK_API_KEY` resolution without secret
  disclosure;
- the explicit CommonStack model allowlist and rejected models;
- exact quota classification for `402`, approved structured codes, and approved
  phrases, plus negative cases for plain `429`, timeout, `5xx`, invalid
  credential, and invalid response;
- one OpenRouter success with no fallback and one eligible OpenRouter failure
  followed by exactly one CommonStack call;
- value-equivalent model input and generation-control fields across the retry,
  with only `provider_id` changed and non-`none` reasoning effort preserved;
- no failover for BYOK or any non-quota category;
- primary release before fallback reserve, one final settlement, release
  failure fail-closed behavior, and finalizer cleanup;
- two-attempt reservation idempotency, logical call counting, and migration
  compatibility in both storage backends;
- actual and mixed provider attribution in result evidence, run metadata,
  Credits activity, and analytics; and
- dual-failure category selection with no secret or upstream-body leakage.

All automated tests use fake provider responses and fake credentials. Real
OpenRouter, CommonStack, Render, database, and API credentials must not appear
in fixtures, snapshots, logs, documentation, or commits.

## Acceptance criteria

- A Platform Credits call requested through OpenRouter succeeds through
  CommonStack after one explicit OpenRouter balance/quota rejection.
- The same request does not fail over for a timeout, generic `429`, `5xx`, empty
  response, unsupported model, invalid credential, local Credits failure, or
  BYOK execution.
- The fallback preserves the model input and all generation controls,
  including reasoning effort.
- One logical call produces at most one settled Credits debit, with every
  failed or abandoned reservation released.
- Completed-call evidence and analytics name the actual provider; mixed runs
  are represented explicitly.
- Removing or disabling the CommonStack platform credential restores the safe
  OpenRouter quota failure without exposing any secret.
- The change requires no production Render mutation until after merge and a
  separate deployment authorization.
