# BYOK API Key Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-aware, three-step BYOK setup guide and a direct Run Backtest recovery path to the existing API Keys tab without changing credential or execution APIs.

**Architecture:** Keep the official provider destinations in an exact frontend allowlist keyed by seeded `provider_id`. Render one semantic guide inside the existing key form, reuse the existing credential submit lifecycle, and expose a small Credits-page navigation method so the Run Backtest modal can return the user to the correct tab. The backend remains authoritative for approved providers, verification, and executable provider/model availability.

**Tech Stack:** Static HTML, vanilla JavaScript, CSS, existing SVG icon sprite, Python `pytest` source-contract tests, browser network interception with safe fixtures

**Spec:** `docs/superpowers/specs/2026-08-28-byok-api-key-guidance-design.md`

## Global Constraints

- Do not change any backend endpoint, response model, credential vault, verification adapter, execution catalog, or Credits accounting behavior.
- The only official setup destinations are `https://platform.openai.com/api-keys`, `https://openrouter.ai/keys`, `https://platform.claude.com/settings/keys`, and `https://aistudio.google.com/apikey`.
- Select an official setup destination only by exact seeded `provider_id`; never construct a URL from `display_name`, `approved_base_url`, or adapter type.
- Every external setup action must use `target="_blank"` and `rel="noopener noreferrer"`.
- Never place a full API key in HTML, rendered help, JavaScript state, a URL, browser storage, analytics, logs, tests, screenshots, or commits.
- Keep the existing password-input submit lifecycle and clear the secret in every completion path.
- Use ATL-owned generic illustrations; do not add external provider screenshots or image assets.
- Billing and provider credits appear only in failure troubleshooting, not in the primary three-step guide.
- Use safe mock fixtures for browser acceptance. Do not open provider sites automatically and do not submit a real API key.
- Preserve the existing light/dark theme, native keyboard order, visible focus, mobile layout, and `prefers-reduced-motion` behavior.
- Do not add dependencies or a frontend build step.

## File Structure

- Modify `dashboard/frontend/app.html` for the semantic guide, troubleshooting text, Run Backtest recovery button, focus target, and static asset revisions.
- Modify `dashboard/frontend/js/credits.js` for the official-link allowlist, provider-driven guide state, failure help, tab activation, and focus handoff.
- Modify `dashboard/frontend/app.js` for Run Backtest recovery visibility and navigation.
- Modify `dashboard/frontend/styles.css` for the compact guide, generic illustrations, arrow connectors, responsive stacking, and recovery action spacing.
- Modify `dashboard/backend/tests/test_credits_frontend.py` for API Keys markup, link, secret-lifecycle, troubleshooting, and Credits focus contracts.
- Modify `dashboard/backend/tests/test_byok_backtest_frontend.py` for successful-empty versus request-failure recovery contracts.
- Modify `dashboard/backend/tests/test_frontend_fast_boot.py`, `dashboard/backend/tests/test_analytics_frontend.py`, and `dashboard/backend/tests/test_admin_analytics_frontend.py` for the final static asset revisions.

No new production module is warranted. The guide is state owned by the existing Credits IIFE, and the recovery decision is state owned by the existing Run Backtest functions.

---

### Task 1: Provider-Aware Three-Step Setup Guide

**Files:**
- Modify: `dashboard/backend/tests/test_credits_frontend.py:1-147`
- Modify: `dashboard/frontend/app.html:1917-1952`
- Modify: `dashboard/frontend/js/credits.js:1-137, 303-330, 739-758`
- Modify: `dashboard/frontend/styles.css:10928-11014, 11694-11840`

**Interfaces:**
- Consumes: `state.providers`, `creditsApiKeyProvider`, `renderProviderOptions()`, and the existing icon symbols `icon-external-link`, `icon-arrow-right`, and `icon-check-circle`.
- Produces: `OFFICIAL_API_KEY_PAGES: Readonly<Record<string, { displayName: string, url: string }>>` and `renderProviderGuide(): void` inside `credits.js`; DOM ids `creditsApiKeyGuide`, `creditsApiKeyGuideSteps`, `creditsApiKeyGuideProvider`, `creditsApiKeyOfficialLink`, and `creditsApiKeyGuideFallback`.

- [ ] **Step 1: Write failing source-contract tests for semantic steps and the exact URL allowlist**

Add one module-level source value and these tests to `test_credits_frontend.py`:

```python
CREDITS_JS = CREDITS_JS_PATH.read_text(encoding="utf-8")


def _credits_function(name: str) -> str:
    start = CREDITS_JS.index(f"  function {name}(")
    boundaries = (
        CREDITS_JS.find("\n  function ", start + 1),
        CREDITS_JS.find("\n  async function ", start + 1),
    )
    end = min(boundary for boundary in boundaries if boundary >= 0)
    return CREDITS_JS[start:end]


def test_api_key_guide_has_three_semantic_steps_and_safe_external_link():
    start = APP_HTML.index('id="creditsApiKeyGuide"')
    end = APP_HTML.index('id="creditsApiKeyGuideFallback"', start)
    guide = APP_HTML[start:end]
    assert 'id="creditsApiKeyGuideSteps"' in guide
    assert '<ol' in guide
    assert guide.count('class="credits-key-guide-step"') == 3
    assert 'id="creditsApiKeyOfficialLink"' in guide
    assert 'target="_blank"' in guide
    assert 'rel="noopener noreferrer"' in guide
    assert 'Open official API key page' in guide
    assert 'Create and copy the key' in guide
    assert 'Return here and paste it below' in guide


def test_api_key_guide_uses_only_the_exact_official_provider_allowlist():
    expected = {
        "openai": "https://platform.openai.com/api-keys",
        "openrouter": "https://openrouter.ai/keys",
        "anthropic": "https://platform.claude.com/settings/keys",
        "gemini": "https://aistudio.google.com/apikey",
    }
    for provider_id, url in expected.items():
        assert f"{provider_id}: Object.freeze({{" in CREDITS_JS
        assert f"url: '{url}'" in CREDITS_JS
    render = _credits_function("renderProviderGuide")
    assert "approved_base_url" not in render
    assert "adapter_type" not in render
    assert "creditsApiKeyProvider')?.addEventListener('change', renderProviderGuide)" in CREDITS_JS


def test_custom_provider_guide_never_guesses_an_external_destination():
    assert 'id="creditsApiKeyGuideFallback"' in APP_HTML
    assert 'ATL does not have an official setup link for this provider.' in APP_HTML
    assert "officialPage = OFFICIAL_API_KEY_PAGES[providerId] || null" in CREDITS_JS
    assert "officialLink.removeAttribute('href')" in CREDITS_JS
    assert "steps.hidden = !officialPage" in CREDITS_JS
    assert "fallback.hidden = Boolean(officialPage)" in CREDITS_JS
```

- [ ] **Step 2: Run the focused tests and confirm the guide contract fails**

Run:

```bash
pytest dashboard/backend/tests/test_credits_frontend.py \
  -k 'api_key_guide or custom_provider_guide' -v
```

Expected: FAIL because the guide ids and `OFFICIAL_API_KEY_PAGES` do not exist.

- [ ] **Step 3: Add the semantic guide markup inside the existing form**

In `app.html`, make the API Keys heading a deterministic fallback focus target:

```html
<h3 id="creditsApiKeysHeading" tabindex="-1">Connect your model providers</h3>
```

Immediately after the Provider field, add:

```html
<div id="creditsApiKeyGuide" class="credits-key-guide" hidden>
    <ol id="creditsApiKeyGuideSteps" class="credits-key-guide-steps" aria-label="API key setup steps">
        <li class="credits-key-guide-step">
            <div class="credits-key-guide-head">
                <span class="credits-key-guide-number" aria-hidden="true">1</span>
                <div>
                    <h4>Open official API key page</h4>
                    <p id="creditsApiKeyGuideProvider" aria-live="polite">Open the selected provider's official key page.</p>
                </div>
            </div>
            <a
                id="creditsApiKeyOfficialLink"
                class="auth-btn auth-btn-secondary credits-key-guide-link"
                target="_blank"
                rel="noopener noreferrer"
            >
                <span>Open official API key page</span>
                <svg aria-hidden="true"><use href="#icon-external-link"/></svg>
            </a>
            <svg class="credits-key-guide-arrow" aria-hidden="true"><use href="#icon-arrow-right"/></svg>
        </li>
        <li class="credits-key-guide-step">
            <div class="credits-key-guide-head">
                <span class="credits-key-guide-number" aria-hidden="true">2</span>
                <div>
                    <h4>Create and copy the key</h4>
                    <p>Sign in there, create a key, then copy it once.</p>
                </div>
            </div>
            <div class="credits-key-guide-illustration" aria-hidden="true">
                <span class="credits-key-guide-window-bar"></span>
                <span class="credits-key-guide-create">Create API key</span>
                <span class="credits-key-guide-masked">sk-...7K2</span>
                <span class="credits-key-guide-copy">Copy</span>
            </div>
            <svg class="credits-key-guide-arrow" aria-hidden="true"><use href="#icon-arrow-right"/></svg>
        </li>
        <li class="credits-key-guide-step">
            <div class="credits-key-guide-head">
                <span class="credits-key-guide-number" aria-hidden="true">3</span>
                <div>
                    <h4>Return here and paste it below</h4>
                    <p>Use the real ATL fields below, then save and verify.</p>
                </div>
            </div>
            <div class="credits-key-guide-paste" aria-hidden="true">
                <span>API key</span>
                <span class="credits-key-guide-input">Paste key here</span>
                <span class="credits-key-guide-verify">Save and verify</span>
            </div>
        </li>
    </ol>
    <p id="creditsApiKeyGuideFallback" class="credits-key-guide-fallback" hidden>
        ATL does not have an official setup link for this provider. Contact the administrator who enabled it.
    </p>
</div>
```

The illustration text is non-interactive and its container is `aria-hidden`; the real instructions remain in each heading and paragraph.

- [ ] **Step 4: Add the literal allowlist and provider-driven renderer**

Near the existing Credits constants in `credits.js`, add:

```javascript
  const OFFICIAL_API_KEY_PAGES = Object.freeze({
    openai: Object.freeze({
      displayName: 'OpenAI',
      url: 'https://platform.openai.com/api-keys',
    }),
    openrouter: Object.freeze({
      displayName: 'OpenRouter',
      url: 'https://openrouter.ai/keys',
    }),
    anthropic: Object.freeze({
      displayName: 'Anthropic',
      url: 'https://platform.claude.com/settings/keys',
    }),
    gemini: Object.freeze({
      displayName: 'Google Gemini',
      url: 'https://aistudio.google.com/apikey',
    }),
  });
```

After `providerDisplayName()`, add:

```javascript
  function renderProviderGuide() {
    const guide = element('creditsApiKeyGuide');
    const steps = element('creditsApiKeyGuideSteps');
    const fallback = element('creditsApiKeyGuideFallback');
    const officialLink = element('creditsApiKeyOfficialLink');
    const providerCopy = element('creditsApiKeyGuideProvider');
    const providerId = element('creditsApiKeyProvider')?.value || '';
    const selectedProvider = state.providers.find(
      (provider) => provider.provider_id === providerId,
    ) || null;
    const officialPage = OFFICIAL_API_KEY_PAGES[providerId] || null;

    if (!guide || !steps || !fallback || !officialLink || !providerCopy) return;
    guide.hidden = !selectedProvider;
    steps.hidden = !officialPage;
    fallback.hidden = Boolean(officialPage);
    officialLink.removeAttribute('href');
    officialLink.removeAttribute('aria-label');

    if (!selectedProvider || !officialPage) return;
    officialLink.href = officialPage.url;
    officialLink.setAttribute(
      'aria-label',
      `Open ${officialPage.displayName} official API key page in a new tab`,
    );
    providerCopy.textContent = (
      `Continue on ${officialPage.displayName}, then return to ATL.`
    );
  }
```

Call `renderProviderGuide()` before every return from `renderProviderOptions()` and once after a successful provider select is populated. Wire the select in `wireControls()`:

```javascript
    element('creditsApiKeyProvider')?.addEventListener('change', renderProviderGuide);
```

On sign-out, call `renderProviderGuide()` after clearing `state.providers`, so a stale external destination cannot survive a user transition.

- [ ] **Step 5: Add compact, theme-aware guide styling and responsive arrow direction**

Add these rules beside the current API Keys styles:

```css
.credits-key-guide,
.credits-key-guide-steps {
    min-width: 0;
}

.credits-key-guide-steps {
    display: grid;
    gap: 10px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.credits-key-guide-step {
    position: relative;
    display: grid;
    gap: 9px;
    min-width: 0;
    padding-bottom: 12px;
}

.credits-key-guide-step:last-child {
    padding-bottom: 0;
}

.credits-key-guide-head {
    display: flex;
    align-items: flex-start;
    gap: 9px;
}

.credits-key-guide-head h4,
.credits-key-guide-head p {
    margin: 0;
}

.credits-key-guide-head h4 {
    color: var(--text-primary);
    font-size: 12px;
}

.credits-key-guide-head p {
    margin-top: 3px;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
}

.credits-key-guide-number {
    display: inline-flex;
    flex: 0 0 24px;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-primary);
    color: var(--info-color);
    font-size: 11px;
    font-weight: 700;
}

.credits-key-guide-link {
    width: 100%;
    min-height: 40px;
    box-sizing: border-box;
}

.credits-key-guide-link svg,
.credits-key-guide-arrow {
    width: 16px;
    height: 16px;
    fill: none;
    stroke: currentColor;
}

.credits-key-guide-arrow {
    justify-self: center;
    transform: rotate(90deg);
    color: var(--text-muted);
}

.credits-key-guide-illustration,
.credits-key-guide-paste {
    display: grid;
    grid-template-columns: minmax(0, 1fr) max-content;
    gap: 7px;
    align-items: center;
    min-width: 0;
    padding: 10px;
    background: var(--bg-primary);
    color: var(--text-secondary);
    font-size: 10px;
}

.credits-key-guide-window-bar {
    grid-column: 1 / -1;
    height: 3px;
    background: var(--border-color);
}

.credits-key-guide-create,
.credits-key-guide-verify {
    color: var(--info-color);
    font-weight: 700;
}

.credits-key-guide-masked,
.credits-key-guide-input {
    min-width: 0;
    overflow: hidden;
    font-family: var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
}

.credits-key-guide-paste > span:first-child {
    grid-column: 1 / -1;
}

.credits-key-guide-fallback {
    margin: 0;
    padding: 10px 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.5;
}

@media (max-width: 600px) {
    .credits-key-guide-link {
        min-height: 44px;
    }
}

```

The guide introduces no animation or transition, so reduced-motion users see
the same static arrows and illustrations without a separate override.

- [ ] **Step 6: Run the focused guide and existing secret-safety tests**

Run:

```bash
pytest dashboard/backend/tests/test_credits_frontend.py \
  -k 'api_key_guide or custom_provider_guide or secret' -v
node --check dashboard/frontend/js/credits.js
```

Expected: PASS. The HTML contains three semantic steps, JavaScript contains only the literal official destinations, and existing secret-safety guards remain green.

- [ ] **Step 7: Commit the provider-aware guide**

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py
git commit -m "feat: guide users through BYOK key setup"
```

---

### Task 2: Failure-Only Verification Help

**Files:**
- Modify: `dashboard/backend/tests/test_credits_frontend.py`
- Modify: `dashboard/frontend/app.html:1939-1952`
- Modify: `dashboard/frontend/js/credits.js:333-380, 739-758`
- Modify: `dashboard/frontend/styles.css` beside the Task 1 guide rules

**Interfaces:**
- Consumes: `creditsApiKeySecret`, `creditsApiKeyStatus`, `saveApiKey(event)`, `setStatus()`, and `renderProviderGuide()` from Task 1.
- Produces: `setApiKeyTroubleshooting(visible: boolean): void` and DOM id `creditsApiKeyTroubleshooting`.

- [ ] **Step 1: Write failing tests for hidden-by-default, failure-only troubleshooting**

Add to `test_credits_frontend.py`:

```python
def test_api_key_troubleshooting_is_failure_only_and_field_associated():
    assert 'id="creditsApiKeyTroubleshooting"' in APP_HTML
    assert 'aria-describedby="creditsApiKeyTroubleshooting"' in APP_HTML
    assert 'id="creditsApiKeyTroubleshooting" class="credits-key-troubleshooting" hidden' in APP_HTML
    assert 'Some providers also require billing or account credits' in APP_HTML
    assert "function setApiKeyTroubleshooting(visible)" in CREDITS_JS
    assert "setApiKeyTroubleshooting(status !== 'verified')" in CREDITS_JS
    assert "setApiKeyTroubleshooting(true)" in CREDITS_JS
    assert "setApiKeyTroubleshooting(false)" in CREDITS_JS


def test_api_key_troubleshooting_does_not_add_billing_to_the_primary_steps():
    guide_start = APP_HTML.index('id="creditsApiKeyGuide"')
    help_start = APP_HTML.index('id="creditsApiKeyTroubleshooting"')
    guide = APP_HTML[guide_start:help_start]
    assert "billing" not in guide.lower()
    assert "account credits" not in guide.lower()
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest dashboard/backend/tests/test_credits_frontend.py \
  -k 'troubleshooting' -v
```

Expected: FAIL because the troubleshooting element and state helper do not exist.

- [ ] **Step 3: Add associated troubleshooting markup after the key input**

Update the real password input and place the help immediately after its label:

```html
<input
    id="creditsApiKeySecret"
    type="password"
    maxlength="4096"
    autocomplete="new-password"
    spellcheck="false"
    aria-describedby="creditsApiKeyTroubleshooting"
    required
>
</label>
<p id="creditsApiKeyTroubleshooting" class="credits-key-troubleshooting" hidden>
    Check that the full key was copied and is active. Some providers also require billing or account credits before API calls can run.
</p>
```

Keep `creditsApiKeyStatus` as the single `role="status" aria-live="polite"` announcement region.

- [ ] **Step 4: Add deterministic troubleshooting visibility to the submit lifecycle**

Add beside `setStatus()`:

```javascript
  function setApiKeyTroubleshooting(visible) {
    const help = element('creditsApiKeyTroubleshooting');
    if (help) help.hidden = !visible;
  }
```

In `saveApiKey()`:

1. Call `setApiKeyTroubleshooting(false)` before local required-field validation.
2. After the response status is read, call:

```javascript
      setApiKeyTroubleshooting(status !== 'verified');
```

3. In `catch`, call `setApiKeyTroubleshooting(true)` before `setStatus(...)`.
4. Preserve the existing `finally` block that clears `secretInput.value` and re-enables the button.

Also hide stale troubleshooting when the provider changes and when the user signs out:

```javascript
  function onApiKeyProviderChange() {
    setApiKeyTroubleshooting(false);
    renderProviderGuide();
  }
```

In the `!signedIn` branch of `syncAuth()`, add the same reset beside the
existing password-input clear:

```javascript
      setApiKeyTroubleshooting(false);
```

Wire `onApiKeyProviderChange` instead of wiring `renderProviderGuide` directly, and update the Task 1 contract assertion to expect:

```python
assert "creditsApiKeyProvider')?.addEventListener('change', onApiKeyProviderChange)" in CREDITS_JS
```

- [ ] **Step 5: Add restrained failure-help styling**

```css
.credits-key-troubleshooting {
    margin: -4px 0 0;
    color: #fbbf24;
    font-size: 11px;
    line-height: 1.45;
}
```

Do not add a billing button, provider-specific billing URL, disclosure widget, or second live region.

- [ ] **Step 6: Run troubleshooting, partial-error, and secret-lifecycle tests**

Run:

```bash
pytest dashboard/backend/tests/test_credits_frontend.py -v
node --check dashboard/frontend/js/credits.js
```

Expected: PASS. Existing tests still prove the full secret is never stored or rendered, and the new help is absent from the primary guide.

- [ ] **Step 7: Commit failure-only help**

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py
git commit -m "feat: explain BYOK verification failures"
```

---

### Task 3: Run Backtest Recovery to API Keys

**Files:**
- Modify: `dashboard/backend/tests/test_byok_backtest_frontend.py:1-67`
- Modify: `dashboard/backend/tests/test_credits_frontend.py`
- Modify: `dashboard/frontend/app.html:410-439, 1917-1952`
- Modify: `dashboard/frontend/app.js:7416-7559, 7889-7996, 4937-4950`
- Modify: `dashboard/frontend/js/credits.js:14-29, 98-137, 739-771`
- Modify: `dashboard/frontend/styles.css:4982-5035`

**Interfaces:**
- Consumes: `setRunBacktestBillingMode()`, `setRunBacktestExecutionUnavailable()`, `clearPendingByokBacktest()`, `closeRunBacktestModal()`, `navigateToPage()`, `setCreditsTab()`, and `renderProviderOptions()`.
- Produces: `setRunBacktestApiKeysRecovery(visible: boolean): void`, `goToApiKeys(): void`, and `window.CreditsPage.openApiKeys({ focus: boolean }): void`; DOM id `runBacktestApiKeysBtn`; state field `focusApiKeysOnReady: boolean`.

- [ ] **Step 1: Write failing contracts for the two unavailable causes**

Add to `test_byok_backtest_frontend.py`:

```python
def test_confirmed_empty_execution_inventory_offers_api_key_recovery():
    _assert_contains(APP_HTML, 'id="runBacktestApiKeysBtn"')
    _assert_contains(APP_HTML, '>Go to API Keys</button>')
    body = _function_body("loadRunBacktestExecutionOptions")
    _assert_contains(body, "showApiKeysRecovery: true")
    assert body.index("catch (_error)") < body.index("showApiKeysRecovery: true")


def test_execution_options_request_failure_does_not_diagnose_a_missing_key():
    body = _function_body("loadRunBacktestExecutionOptions")
    catch_start = body.index("catch (_error)")
    catch_end = body.index("if (pending)", catch_start)
    catch_body = body[catch_start:catch_end]
    _assert_contains(catch_body, "Backtest execution options could not be loaded.")
    assert "showApiKeysRecovery: true" not in catch_body


def test_api_key_recovery_clears_pending_state_and_navigates_to_credits():
    body = _function_body("goToApiKeys")
    _assert_contains(body, "clearPendingByokBacktest()")
    _assert_contains(body, "closeRunBacktestModal()")
    _assert_contains(body, "navigateToPage('credits')")
    _assert_contains(body, "window.CreditsPage?.openApiKeys({ focus: true })")
```

Add to `test_credits_frontend.py`:

```python
def test_credits_module_exposes_api_keys_focus_handoff():
    assert "focusApiKeysOnReady: false" in CREDITS_JS
    assert "function openApiKeys({ focus = false } = {})" in CREDITS_JS
    assert "setCreditsTab('api-keys', { reload: false })" in CREDITS_JS
    assert "window.CreditsPage = { onEnter, syncAuth, openApiKeys }" in CREDITS_JS
    assert "state.focusApiKeysOnReady" in CREDITS_JS
    assert "creditsApiKeysHeading" in CREDITS_JS
```

- [ ] **Step 2: Run the recovery contracts and confirm they fail**

Run:

```bash
pytest dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_credits_frontend.py \
  -k 'recovery or focus_handoff' -v
```

Expected: FAIL because the button, recovery flag, navigation function, and Credits public method do not exist.

- [ ] **Step 3: Add the hidden recovery action to the billing group**

After `runBacktestBillingHint` in `app.html`, add:

```html
<button
    id="runBacktestApiKeysBtn"
    class="auth-btn auth-btn-secondary run-backtest-api-keys"
    type="button"
    hidden
>Go to API Keys</button>
```

The action remains hidden for loading, normal availability, fixed runtimes, rule-based runs, modal close, and execution-options request failure.

- [ ] **Step 4: Distinguish unknown availability from confirmed empty availability**

In `app.js`, add:

```javascript
function setRunBacktestApiKeysRecovery(visible) {
    const button = document.getElementById('runBacktestApiKeysBtn');
    if (button) button.hidden = !visible;
}
```

Change the unavailable helper signature and set recovery from its explicit option:

```javascript
function setRunBacktestExecutionUnavailable(
    message,
    { showApiKeysRecovery = false } = {},
) {
    runBacktestBillingMode = null;
    setRunBacktestApiKeysRecovery(showApiKeysRecovery);
    clearSelectOptions(document.getElementById('runBacktestProviderSelect'));
    const modelSelect = document.getElementById('modelSelect');
    clearSelectOptions(modelSelect);
    if (
        modelSelect
        && document.getElementById('marketDataSourceSelect')?.value
            === IFIND_ASHARE_SOURCE
    ) {
        const ruleOption = document.createElement('option');
        ruleOption.value = RULE_BASED_DECISION_SOURCE;
        ruleOption.textContent = 'Rule-based';
        modelSelect.appendChild(ruleOption);
        modelSelect.value = RULE_BASED_DECISION_SOURCE;
    }
    document
        .querySelectorAll('#runBacktestBillingGroup [data-billing-mode]')
        .forEach((button) => {
            button.setAttribute('aria-checked', 'false');
            button.classList.remove('is-selected');
        });
    const hint = document.getElementById('runBacktestBillingHint');
    if (hint) hint.textContent = message;
    syncBacktestModelFieldMode();
    syncRunBacktestSubmitAvailability();
}
```

At the start of `setRunBacktestBillingMode()`, call:

```javascript
    setRunBacktestApiKeysRecovery(false);
```

Leave the request-error call unchanged so its default remains false. Change only the final successful-empty call to:

```javascript
    setRunBacktestExecutionUnavailable(
        'Add and verify a default API key, or ask an administrator to enable a platform provider.',
        { showApiKeysRecovery: true },
    );
```

Also call `setRunBacktestApiKeysRecovery(false)` from `closeRunBacktestModal()` and before a newly opened modal starts loading options.

- [ ] **Step 5: Add a Credits tab focus handoff without duplicate API loads**

Add `focusApiKeysOnReady: false` to the Credits state. Change the tab helper to accept a reload option:

```javascript
  function setCreditsTab(tab, { reload = true } = {}) {
    const allowed = new Set(['overview', 'api-keys', 'activity']);
    const next = tab === 'top-up' ? 'overview' : (allowed.has(tab) ? tab : 'overview');
    state.activeTab = next;
    document.querySelectorAll('[data-credits-tab]').forEach((button) => {
      const selected = button.dataset.creditsTab === next;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-credits-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.creditsPanel !== next;
    });
    if (reload && next === 'api-keys' && state.user) loadApiKeys();
  }
```

Add:

```javascript
  function focusApiKeysEntry() {
    const provider = element('creditsApiKeyProvider');
    const heading = element('creditsApiKeysHeading');
    const target = provider && !provider.disabled ? provider : heading;
    state.focusApiKeysOnReady = false;
    window.requestAnimationFrame(() => target?.focus());
  }

  function openApiKeys({ focus = false } = {}) {
    setCreditsTab('api-keys', { reload: false });
    if (!focus) return;
    state.focusApiKeysOnReady = true;
    if (state.providers.length || element('creditsApiKeyProvider')?.disabled) {
      focusApiKeysEntry();
    }
  }
```

At the end of both success and empty/error paths in `renderProviderOptions()`, run:

```javascript
    if (state.focusApiKeysOnReady) focusApiKeysEntry();
```

Export the entry point:

```javascript
  window.CreditsPage = { onEnter, syncAuth, openApiKeys };
```

The `reload: false` path is important: `navigateToPage('credits')` already invokes `onEnter()` and begins the page loads.

- [ ] **Step 6: Wire the modal action and keep its state ephemeral**

Add in `app.js`:

```javascript
function goToApiKeys() {
    clearPendingByokBacktest();
    closeRunBacktestModal();
    navigateToPage('credits');
    window.CreditsPage?.openApiKeys({ focus: true });
}
```

Wire it beside the existing modal close and submit controls:

```javascript
    document.getElementById('runBacktestApiKeysBtn')?.addEventListener('click', goToApiKeys);
```

Add minimal alignment without introducing another nested panel:

```css
.run-backtest-api-keys {
    margin-top: 8px;
}
```

- [ ] **Step 7: Run the recovery, Credits, and JavaScript syntax suites**

Run:

```bash
pytest dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_credits_frontend.py -v
node --check dashboard/frontend/app.js
node --check dashboard/frontend/js/credits.js
```

Expected: PASS. The failed-request branch has no `showApiKeysRecovery: true`; the confirmed-empty branch does; the action clears pending state, closes the modal, navigates once, selects API Keys without a duplicate reload, and focuses a deterministic target.

- [ ] **Step 8: Commit the recovery path**

```bash
git add dashboard/frontend/app.html \
  dashboard/frontend/app.js \
  dashboard/frontend/js/credits.js \
  dashboard/frontend/styles.css \
  dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_byok_backtest_frontend.py
git commit -m "feat: recover blocked backtests through API Keys"
```

---

### Task 4: Static Asset Revisions and Browser Acceptance

**Files:**
- Modify: `dashboard/backend/tests/test_frontend_fast_boot.py:176-198`
- Modify: `dashboard/backend/tests/test_analytics_frontend.py:43-51`
- Modify: `dashboard/backend/tests/test_admin_analytics_frontend.py:189-196`
- Modify: `dashboard/frontend/app.html:16, 2402, 2405`

**Interfaces:**
- Consumes: final changed `styles.css`, `app.js`, and `js/credits.js` from Tasks 1-3.
- Produces: shipped references `styles.css?v=126`, `app.js?v=117`, and `js/credits.js?v=5`.

- [ ] **Step 1: Advance the single-owner cache-buster tests first**

Update `test_frontend_fast_boot.py` to require:

```python
assert "app.js?v=117" in APP_HTML
assert "styles.css?v=126" in APP_HTML
assert "js/credits.js?v=5" in APP_HTML
```

Update the app reference in `test_analytics_frontend.py`:

```python
app_at = APP_HTML.index('<script src="app.js?v=117" defer></script>')
```

Update the two shared-asset assertions in `test_admin_analytics_frontend.py`:

```python
assert 'styles.css?v=126' in APP_HTML
assert 'app.js?v=117' in APP_HTML
```

- [ ] **Step 2: Run the cache-buster contracts and verify they fail**

Run:

```bash
pytest dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py \
  -k 'cache or script_loads_between or lifecycle_and_cache' -v
```

Expected: FAIL because `app.html` still references revisions 116, 125, and 4.

- [ ] **Step 3: Update the three shipped asset references**

Change only these references in `app.html`:

```html
<link rel="stylesheet" href="styles.css?v=126">
<script src="app.js?v=117" defer></script>
<script src="js/credits.js?v=5" defer></script>
```

- [ ] **Step 4: Run the complete focused regression gate**

Run:

```bash
pytest dashboard/backend/tests/test_credits_frontend.py \
  dashboard/backend/tests/test_byok_backtest_frontend.py \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py -v
node --check dashboard/frontend/app.js
node --check dashboard/frontend/js/credits.js
git diff --check
```

Expected: all tests PASS, both JavaScript syntax checks exit 0, and `git diff --check` prints nothing.

- [ ] **Step 5: Start the local static frontend for browser acceptance**

Run in a persistent terminal:

```bash
python3 -m http.server 4173 --directory dashboard/frontend
```

Open `http://127.0.0.1:4173/app.html?view=credits`. Use browser network interception for the following safe responses; do not create fixture files in the repository:

```json
{
  "/api/auth/me": {"user":{"id":9001,"email":"fixture@example.test","role":"user"}},
  "/api/credits/model-providers": {
    "providers": [
      {"provider_id":"openai","display_name":"OpenAI"},
      {"provider_id":"openrouter","display_name":"OpenRouter"},
      {"provider_id":"anthropic","display_name":"Anthropic"},
      {"provider_id":"gemini","display_name":"Google Gemini"}
    ]
  },
  "/api/credits/api-keys": {"items":[]},
  "/api/credits/execution-options": {"providers":[]},
  "/api/credits/balance": {"balance":{"balance_micro":0,"display_credits":"0.000000","account_status":"active","billing_available":false}},
  "/api/credits/ledger?limit=50": {"items":[]}
}
```

Set only this safe local user state before reload:

```javascript
localStorage.setItem('auth-user', JSON.stringify({id: 9001, email: 'fixture@example.test', role: 'user'}));
```

- [ ] **Step 6: Verify the UI at desktop and mobile widths**

At 1280x900 and 390x844, verify:

1. OpenAI, OpenRouter, Anthropic, and Gemini each update the external action to the exact allowlisted destination.
2. The three steps, generic illustration, arrows, real fields, and saved-key empty state do not overlap or create horizontal scroll.
3. Tab order is Provider, official link, Name, API key, default checkbox, `Save and verify`, then saved-key controls; focus remains visible.
4. Dark and light themes keep text and controls readable; reduced motion has no continuing animation.
5. A custom safe provider fixture shows only the administrator fallback and has no focusable external link.

Inspect the anchor `href`, `target`, and `rel` in the browser; do not activate the link during automated acceptance.

- [ ] **Step 7: Verify partial errors, verification failure, and Backtest recovery**

Use interception variants one at a time:

1. Reject only `/api/credits/model-providers`: the guide is inert and saved-key/execution results remain independently rendered.
2. Reject only `/api/credits/api-keys`: the official guide remains usable and the saved-key region reports its own failure.
3. Return a sanitized failed credential-create response: the password input clears, the existing status reports the error, and failure-only troubleshooting appears.
4. Reject `/api/credits/execution-options` in Run Backtest: controls disable and `Go to API Keys` stays hidden.
5. Return `{ "providers": [] }` from execution options: `Go to API Keys` appears, closes the modal, opens Credits/API Keys, and focuses Provider after it loads.

Confirm the browser console has no uncaught exception and no request or URL contains the entered fake secret.

- [ ] **Step 8: Commit cache revisions after browser acceptance**

```bash
git add dashboard/frontend/app.html \
  dashboard/backend/tests/test_frontend_fast_boot.py \
  dashboard/backend/tests/test_analytics_frontend.py \
  dashboard/backend/tests/test_admin_analytics_frontend.py
git commit -m "chore: refresh BYOK guidance assets"
```

- [ ] **Step 9: Verify the final branch contains only intended source, tests, spec, and plan**

Run:

```bash
git status --short
git diff --check origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Expected changed paths are limited to:

```text
dashboard/backend/tests/test_admin_analytics_frontend.py
dashboard/backend/tests/test_analytics_frontend.py
dashboard/backend/tests/test_byok_backtest_frontend.py
dashboard/backend/tests/test_credits_frontend.py
dashboard/backend/tests/test_frontend_fast_boot.py
dashboard/frontend/app.html
dashboard/frontend/app.js
dashboard/frontend/js/credits.js
dashboard/frontend/styles.css
docs/superpowers/plans/2026-08-28-byok-api-key-guidance.md
docs/superpowers/specs/2026-08-28-byok-api-key-guidance-design.md
```

`git status --short` must be empty. No database, real credential, `.superpowers/`, `work/`, screenshot, generated browser file, or unrelated source may be committed.
