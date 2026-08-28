# BYOK API Key Guidance Design

**Date:** 2026-08-28
**Status:** Approved
**Target base:** latest `origin/main`
**Scope:** frontend guidance for creating, copying, saving, and verifying BYOK credentials

## Goal

Make the existing BYOK flow understandable to a first-time user who does not
already know where model-provider API keys are created.

The successful path is:

1. choose an approved provider in ATL;
2. open that provider's official API key page in a new tab;
3. create and copy a key on the provider site;
4. return to ATL and paste the key into the existing secure form;
5. select `Save and verify`; and
6. use the verified default key for a backtest.

ATL does not retrieve the key from the provider. The user copies and pastes it
manually. This is the smallest reliable flow because ATL has no OAuth or
delegated credential exchange with these providers. It also preserves the
existing vault contract: the full secret enters ATL only in the password input
and credential-create request.

## Product Decisions

- The guidance is inline in the existing `API Keys` tab, not a separate wizard
  or modal.
- Every official provider uses the same three primary instructions:
  `Open official API key page`, `Create and copy the key`, and
  `Return here and paste it below`.
- Provider billing is not a primary step. Billing or account credits appear
  only in troubleshooting copy after verification or execution fails.
- The visual guide uses ATL-owned diagrammatic illustrations and arrows, not
  screenshots or copied provider-site UI. Provider pages change independently;
  a screenshot-based tutorial would become misleading without a code change.
- The Run Backtest unavailable state links directly back to the `API Keys` tab
  when the execution-options request succeeds but reports no usable lane.
- This work does not change credential storage, verification, provider
  execution, Credits accounting, or any backend API contract.

## Existing Surface

The implementation extends these existing frontend boundaries:

- `dashboard/frontend/app.html`: the `creditsApiKeysPanel` and Run Backtest
  modal markup;
- `dashboard/frontend/js/credits.js`: provider loading, API key form state,
  save/verify status, and Credits tab selection;
- `dashboard/frontend/app.js`: Run Backtest execution-option states and
  navigation to Credits; and
- `dashboard/frontend/styles.css`: responsive guide, illustration, connector,
  focus, and error-help styling.

The existing authenticated endpoints remain authoritative:

- `GET /api/credits/model-providers` supplies the approved provider list;
- `POST /api/credits/api-keys` saves and verifies the submitted secret; and
- `GET /api/credits/execution-options` decides whether a verified BYOK or
  Platform Credits execution lane is actually available.

## API Keys Guidance

### Provider selection

The existing Provider select remains the first form control. Selecting a
provider updates the guide immediately. The guide uses the exact seeded
`provider_id`, not a display-name substring or the provider's approved API base
URL, to select an official destination.

| Provider ID | Display name | Official API key page |
| --- | --- | --- |
| `openai` | OpenAI | `https://platform.openai.com/api-keys` |
| `openrouter` | OpenRouter | `https://openrouter.ai/keys` |
| `anthropic` | Anthropic | `https://platform.claude.com/settings/keys` |
| `gemini` | Google Gemini | `https://aistudio.google.com/apikey` |

The mapping is a frontend constant containing only these four exact HTTPS
URLs. A custom or newly approved provider does not inherit a link from its
adapter type. Its guide instead says that ATL does not have an official setup
link for that provider and directs the user to contact the administrator who
enabled it.

While approved providers are loading, the guide is inert. If the provider list
cannot be loaded or is empty, the form keeps its existing disabled-provider
state and does not show an external setup action.

### Three-step visual flow

The guide sits inside the key form after the Provider select and before the
credential fields. It is an ordered three-step sequence:

1. **Open official API key page.** Show the selected provider name and one
   external-link action. The link opens a new tab with `target="_blank"` and
   `rel="noopener noreferrer"`.
2. **Create and copy the key.** Show a compact ATL-owned illustration of a
   generic provider page with a `Create API key` action followed by a masked
   key and copy icon. This is a visual landmark, not an interactive replica and
   not a claim about exact provider wording.
3. **Return here and paste it below.** Lead directly into the real Name, API
   key, default-key checkbox, and `Save and verify` controls. Do not draw a
   second fake ATL form.

Numbered steps carry the sequence semantically. Arrow connectors are decorative
and have `aria-hidden="true"`, so the instructions still make sense without
CSS, color, icons, or images.

On wider layouts the sequence remains compact within the existing form column.
At narrow widths it stacks vertically, uses downward connectors, and keeps each
control at least as large as the current mobile form controls. No horizontal
scroll is introduced. The guide follows the existing light and dark themes and
does not animate continuously. Any connector transition is disabled under
`prefers-reduced-motion`.

### Save and verification states

The existing `Save and verify` request remains unchanged. During submission,
the primary action remains disabled and the existing polite status region says
`Saving and verifying...`. The password input is cleared in every success and
failure path, exactly as it is today.

After a verified response, the status says `Key saved and verified.` and the
saved-key list refreshes. A non-verified response or request error keeps the
sanitized server message and adds concise contextual help near the status:

`Check that the full key was copied and is active. Some providers also require
billing or account credits before API calls can run.`

This help is secondary and appears only after a relevant failure. It does not
add provider billing buttons to the normal three-step path and does not expose
an upstream response body.

If the approved-provider request, saved-key request, or execution-options
request fails independently, the other successful regions remain usable. The
guide depends only on the approved-provider result; the saved-key list and
quick-start controls retain their existing partial-error behavior.

## Run Backtest Recovery

The Run Backtest modal distinguishes two unavailable states:

1. If `GET /api/credits/execution-options` fails, keep the existing load-error
   message and disabled execution controls. Do not claim that the user lacks a
   key, because availability is unknown.
2. If the request succeeds but no provider/model is available in either lane,
   show the existing explanation plus a visible `Go to API Keys` button.

Selecting `Go to API Keys` closes the modal, navigates to the Credits page,
activates its `API Keys` tab through the Credits module's public tab-selection
entry point, and moves focus to the enabled Provider select. If no approved
provider is available, focus moves to the programmatically focusable API Keys
heading instead. It does not start a run, create a credential, or retain a
stale pending launch.

The recovery action is a real button because it changes application view state.
It remains reachable in normal keyboard order and is hidden whenever an
execution lane becomes available or the modal closes.

## Data Flow

```text
Approved provider response
  -> exact frontend provider-id lookup
  -> official HTTPS setup link or custom-provider fallback
  -> user opens provider site in a new tab
  -> user creates and copies a key there
  -> user returns and pastes into the existing password input
  -> existing POST /api/credits/api-keys
  -> existing server-side verification and encrypted vault
  -> existing execution-options response
  -> BYOK backtest availability
```

No API key, credential id, user id, email address, or access token is appended
to an external URL. ATL never reads the provider tab or clipboard and never
attempts to automate provider login, key creation, or copying.

## Security and Privacy

- Only allow the four literal HTTPS destinations in the official-link map.
- Open external destinations with `noopener noreferrer` in a new tab.
- Do not construct setup links from backend-provided URLs or display names.
- Do not put a full API key in HTML, rendered help, JavaScript state, a URL,
  browser storage, analytics, logs, tests, screenshots, or commits.
- Keep the secret in the existing password input for one submit lifecycle and
  clear it in `finally`.
- Do not inspect the clipboard or attempt cross-tab communication.
- Illustrations use obviously fake masked data only, such as `sk-...7K2`, and
  never resemble a real full credential.
- Custom OpenAI-compatible providers receive no guessed third-party link.

## Accessibility

- Use an ordered list or equivalent semantic step structure with visible step
  numbers and headings.
- The external action names the selected provider in accessible text, even if
  its visible label remains `Open official API key page`.
- Decorative arrows and mock-interface details are hidden from assistive
  technology; useful instructions remain real text.
- The dynamic guide description uses `aria-live="polite"` without announcing
  every decorative change.
- Existing form labels remain explicit. Error help is associated with the API
  key field and the existing `role="status"`/`aria-live="polite"` status
  region; this feature does not create a competing announcement region.
- Provider selection, external link, fields, submit action, saved-key controls,
  and Run Backtest recovery follow visual order in the native tab sequence.
- Focus is visible in both themes, and the interface does not rely on hover,
  color, or arrow direction alone.

## Testing Strategy

The no-build frontend continues to use focused Python source-contract tests and
safe fixtures. Extend the nearest contracts in
`dashboard/backend/tests/test_credits_frontend.py` and
`dashboard/backend/tests/test_byok_backtest_frontend.py`, or add one narrowly
scoped BYOK guidance contract file if isolation is clearer.

Automated checks cover:

- all four exact provider IDs and official HTTPS destinations;
- `target="_blank"` plus `rel="noopener noreferrer"` on the external action;
- custom-provider fallback without a guessed URL;
- provider changes updating the guide while loading/empty/error states stay
  inert;
- the existing secret lifecycle: no persistence, no rendering, and clearing
  after submit;
- failure-only troubleshooting copy and sanitized status handling;
- successful-empty execution options showing `Go to API Keys`, while a failed
  options request does not make the same diagnosis;
- Credits tab activation and focus movement from the recovery action; and
- mobile, dark-theme, keyboard, and reduced-motion style contracts.

Browser acceptance uses mocked provider and credential responses. It verifies
the four provider variants at desktop and mobile widths, keyboard traversal,
visible focus, partial request failures, save success/failure states, and the
Run Backtest recovery route. Automated tests do not open provider sites or use
a real API key. A human may separately confirm that each official link still
lands on the intended provider page.

## Acceptance Criteria

1. A signed-in user can select OpenAI, OpenRouter, Anthropic, or Gemini and open
   the matching official API key page from ATL.
2. The page explains the same create-copy-return-paste flow for every official
   provider, with numbered visual steps and clear arrows.
3. The user pastes into the existing secure input and can complete the existing
   `Save and verify` flow without a new credential API.
4. Billing does not distract from the primary flow; it appears only as
   troubleshooting after a relevant failure.
5. A custom provider never receives a guessed official link.
6. A Run Backtest modal with a confirmed empty execution inventory offers a
   keyboard-accessible route to the `API Keys` tab.
7. Provider-loading failures, credential-list failures, verification failures,
   and execution-options failures do not leak secrets or misstate their cause.
8. The guide remains readable and operable at mobile and desktop widths, in
   light and dark themes, with keyboard-only input and reduced motion.

## Out of Scope

- OAuth, delegated authorization, or automatic provider-key import.
- Reading from or writing to the clipboard.
- Creating provider accounts, projects, billing profiles, or API keys on the
  user's behalf.
- Provider-specific billing buttons in the primary setup flow.
- Shipping or maintaining screenshots of external provider pages.
- Changing the encrypted credential vault, verification adapters, execution
  catalog, model compatibility, Credits accounting, or backend API responses.
- Adding official links for arbitrary custom OpenAI-compatible providers.
