# BYOK Backtest Entry Design

**Date:** 2026-08-24
**Status:** Approved
**Scope:** Provider-compatible model selection on saved API keys and a direct BYOK entry into the existing backtest flow

## Goal

Let a user move directly from a verified default API key to a real BYOK
backtest. The user chooses a model that the selected provider can actually run,
clicks `Run Backtest`, and lands on My Agents with BYOK, provider, and model
already selected for the next run.

The full API key remains in the encrypted server-side vault. The browser carries
only non-secret identifiers.

## User experience

### Saved key row

Each saved key row keeps its existing identity, verification state, last four
characters, default badge, `Reverify`, `Set default`, and `Revoke` controls.

A key that is both `Verified` and `Default` also shows:

- a `Model` select;
- provider-compatible model options; and
- a `Run Backtest` button.

A verified non-default key does not become executable implicitly. It continues
to show `Set default`; the model selector and backtest button remain unavailable
until that key becomes the provider's verified default. Invalid,
verification-unavailable, and revoked keys cannot start a BYOK backtest.

### Provider-compatible models

The ATL supported model catalog remains the product boundary. A provider may
offer more models than ATL supports, but the quick-start selector only shows
models that both ATL and that provider can execute.

The initial official-provider matrix is:

| Provider adapter | Models shown in the saved-key selector |
| --- | --- |
| `openrouter` | Claude Haiku 4.5, Claude Sonnet 4.6, GPT-5.5, Gemini 3.1 Pro Preview, DeepSeek V4 Pro, Qwen3.7 Plus |
| `openai` | GPT-5.5 |
| `anthropic` | Claude Haiku 4.5, Claude Sonnet 4.6 |
| `gemini` | Gemini 3.1 Pro Preview |
| `openai_compatible` | Only ATL models present in an explicit admin-approved provider model allowlist; providers without that allowlist do not expose the quick-start action |

OpenRouter keeps the catalog's provider-qualified model id, such as
`openai/gpt-5.5`. Native adapters receive their native id:

- OpenAI: `gpt-5.5`
- Anthropic: `claude-haiku-4-5` or `claude-sonnet-4-6`
- Gemini: `gemini-3.1-pro-preview`

The backend owns this normalization. The frontend submits only the ATL catalog
model id and never invents or persists a provider request model id. Adding an
admin editor for custom-provider model allowlists is outside this slice; an
existing custom provider remains hidden from quick start until its registry
record has an explicit allowlist.

### Direct backtest entry

Clicking `Run Backtest` on a saved key:

1. stores a short-lived, non-secret pending launch selection in
   `sessionStorage` under `atlPendingByokBacktest`, containing
   `billing_mode=byok`, `provider_id`, the ATL catalog model id, and an expiry
   timestamp no more than 10 minutes in the future;
2. navigates to `view=agents`; and
3. leaves the user on My Agents to choose the agent that will run.

The action does not start a run without an agent and does not change an agent's
saved model.

When the user opens an agent's Run Backtest modal, the pending selection is
consumed and displayed as:

- `Use my API key`;
- the selected provider; and
- `Model for this run`.

The selected model is a one-run override. It is not written back to the agent's
configuration. Closing the modal without launching clears the pending quick
start so it cannot unexpectedly affect a later run.

## Run Backtest modal

The existing modal gains an `AI billing` section for pipeline LLM runs:

- a segmented control with `Use my API key` and `Use ATL Credits`;
- a provider select filtered to providers available for the selected billing
  lane; and
- a `Model for this run` select filtered to the provider/model compatibility
  matrix.

Default behavior is:

1. use the quick-start selection when the user arrived from a saved key;
2. otherwise prefer `Use my API key` when at least one verified default BYOK
   credential is compatible with the agent's saved model; and
3. otherwise select `Use ATL Credits` when a compatible platform provider is
   available.

The modal does not show this billing section for rule-based simulations or
hosted runtimes whose billing/provider route is fixed by the server.

If neither lane has a compatible provider, the submit button is disabled and
the modal gives a direct next action: add and verify a default API key, or ask an
administrator to enable a platform provider.

## Safe execution-options API

Add one authenticated read endpoint for the UI, for example
`GET /api/credits/execution-options`. It returns only safe availability data:

- provider id and display name;
- adapter type;
- whether the current user has a verified default BYOK key;
- whether a verified platform credential is available; and
- ATL catalog model ids supported in each lane.

It never returns an API key, encrypted credential blob, credential fingerprint,
proxy information, upstream response, or platform secret metadata.

The endpoint derives availability from the same provider and credential service
used by execution preflight. The frontend must not guess platform readiness or
treat a merely saved credential as executable.

## Data flow

```text
Verified default key row
  -> choose ATL catalog model
  -> store { billing_mode, provider_id, model_id }
  -> navigate to My Agents
  -> choose agent
  -> Run Backtest modal consumes pending selection
  -> POST /backtest/run with billing_mode + provider_id + ATL catalog model id
  -> backend validates compatibility and signs the ATL catalog model id
  -> backend resolves the current verified default credential
  -> signed worker handoff
  -> provider adapter derives the provider-native request model id
  -> provider call
  -> usage/cost evidence
```

No credential id is required in the launch payload. BYOK execution continues to
resolve the current verified default credential for the authenticated user and
provider. This preserves the existing one-default-per-provider rule and avoids
turning a stale browser selection into authority to use an old key.

## Billing behavior

- BYOK records provider, model, usage, and cost evidence but deducts zero ATL
  Credits.
- Platform Credits reserves and settles ATL Credits from authoritative token
  usage and the pricing snapshot.
- Neither lane silently falls back to the other.
- No run is charged a fixed number of Credits merely because it started.

## Failure handling

- A key loses verified/default status before launch: reject before the worker
  starts and direct the user back to API Keys.
- A provider/model pairing is no longer available: refresh execution options,
  clear the stale selection, and require a new choice.
- Execution options cannot be loaded: disable BYOK/platform launch controls;
  do not guess.
- Provider or usage failure: preserve the existing fail-closed execution and
  reservation cleanup behavior.
- Pending quick-start data is malformed or expired: discard it and open the
  modal with normal defaults. Consuming it or closing that modal also removes
  `atlPendingByokBacktest` from `sessionStorage`.

## Accessibility and interaction

- The model select has an explicit `Model` label.
- The billing segmented control exposes pressed/selected state through ARIA.
- Disabled actions explain why they are disabled in adjacent text, not only in
  a tooltip.
- Keyboard users can reach the selector and `Run Backtest` action in the same
  logical order as the visual layout.
- Status changes use the existing visible status region and do not reveal
  upstream response bodies.

## Copy updates

Remove the obsolete statements:

- `Spending Credits on model runs is not enabled yet.`
- `Held on your account. Spending Credits on model runs is not enabled yet.`

Replace them with accurate lane-specific wording. Purchased and Grant Credits
can fund Platform Credits runs; BYOK runs use the user's provider account and do
not deduct ATL Credits.

## Security constraints

- Never place a full API key in HTML, JavaScript state, URL parameters,
  session/local storage, logs, tests, screenshots, commits, or API responses.
- Pending quick-start state contains only billing mode, provider id, and model
  id, and is consumed once.
- The backend remains authoritative for credential ownership, default status,
  provider enablement, model compatibility, and request-model normalization.
- Custom OpenAI-compatible origins retain DNS/IP pinning and cannot use the
  official-provider proxy exception.

## Acceptance criteria

1. A verified default OpenRouter key offers all six ATL-supported models and
   can enter My Agents with a BYOK model selection.
2. A verified default OpenAI key offers only GPT-5.5; Anthropic and Gemini keys
   are similarly restricted to their compatible models.
3. A non-default, invalid, unavailable, or revoked key cannot be used by the
   quick-start action.
4. The Run Backtest modal submits explicit `billing_mode`, `provider_id`, and
   ATL catalog model id for pipeline LLM runs. The backend validates the pairing
   before creating the worker handoff, retains the catalog id for pricing and
   evidence, and derives the provider-native request id at the adapter boundary.
5. A BYOK run never changes ATL Credit balances.
6. A Platform Credits run is metered from real usage and pricing evidence.
7. Rule-based and hosted runtime flows retain their existing behavior.
8. No full API key is exposed or persisted outside the encrypted vault.

## Verification ownership

Implementation may add or update automated tests, but the user will execute
pytest, browser acceptance, and real-provider model calls. During development,
the agent is limited to static syntax checks, focused source inspection, and
`git diff --check`, stopping immediately if any permitted check fails.

## Out of scope

- Automatically launching a run before the user chooses an agent.
- Persistently changing an agent's configured model from the API Keys page.
- Selecting a specific non-default credential for one run.
- Provider fallback after a failed model call.
- Free-text model ids that bypass the ATL catalog and provider compatibility
  rules.
