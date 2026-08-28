# Account Page Layout Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize only the signed-in Account page into a balanced identity-summary and settings workspace while preserving all existing account behavior and leaving every other route unchanged.

**Architecture:** Keep the existing single-page HTML shell and all current account form IDs. Add a presentational `.account-workspace` wrapper with an identity column and a settings grid, then apply Account-scoped CSS in the existing account style section. A small `updateAccountPage()` presentation update may populate the identity role/status label; no API or workflow changes are needed.

**Tech Stack:** Static HTML, vanilla JavaScript, existing `styles.css`, pytest source-contract tests, and the current local frontend server.

**Spec:** `docs/superpowers/specs/2026-08-28-account-page-layout-redesign.md`

## Global Constraints

- Redesign only the signed-in Account page content rendered by `#accountView`.
- Do not change the global header, brand bar, market ticker, primary navigation, shared page shell, or other page views.
- Preserve existing account form IDs, event listeners, API calls, email-code stages, password policy hints, avatar behavior, and status/error semantics.
- Add or adjust CSS only with `.account-view`-scoped selectors; do not alter shared `.page-view`, `.page-header`, `.auth-form`, or global button behavior.
- Do not change backend files, API contracts, or unrelated frontend assets.
- Do not add `.superpowers/`, real secrets, database connections, or `work/` files to the commit.

### Task 1: Lock the Account DOM and CSS Contracts

**Files:**
- Modify: `dashboard/backend/tests/test_frontend_account_page.py`
- Test: `dashboard/backend/tests/test_frontend_account_page.py`

**Interfaces:**
- Consumes: the existing `#accountView`, `#accountSignedIn`, account form IDs, and shipped `styles.css` source.
- Produces: source contracts that require the new identity/settings wrappers and prevent unscoped layout changes.

- [ ] **Step 1: Add a failing wrapper contract**

Add a test that extracts the signed-in card and asserts the new wrappers and required IDs:

```python
def test_account_layout_has_identity_and_settings_regions():
    card = _account_card()
    assert 'class="account-workspace"' in card
    assert 'class="account-identity"' in card
    assert 'class="account-settings-grid"' in card
    identity_start = card.index('class="account-identity"')
    settings_start = card.index('class="account-settings-grid"')
    assert identity_start < settings_start
    assert card.index('id="accountDisplayName"') < settings_start
    assert card.index('id="accountEmail"') < settings_start
    for marker in (
        'id="accountDisplayNameForm"',
        'id="accountEmailForm"',
        'id="avatarUploadBtn"',
        'id="changePasswordForm"',
    ):
        assert settings_start < card.index(marker)
```

- [ ] **Step 2: Add a failing CSS scope contract**

Add a test that requires the new layout selectors to be rooted under `.account-view` and rejects bare global layout selectors:

```python
def test_account_redesign_selectors_are_scoped_to_account_view():
    css = _STYLES_CSS.read_text(encoding="utf-8")
    for selector in (
        ".account-view {",
        ".account-view .account-workspace",
        ".account-view .account-identity",
        ".account-view .account-settings-grid",
    ):
        assert selector in css
    assert "\n.account-workspace" not in css
    assert "\n.account-identity" not in css
    assert "\n.account-settings-grid" not in css
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
pytest -q dashboard/backend/tests/test_frontend_account_page.py
```

Expected: the two new tests fail because the wrappers and scoped selectors do not exist yet; all existing account contracts should continue to pass.

### Task 2: Restructure Only the Account Markup

**Files:**
- Modify: `dashboard/frontend/app.html:1746-1845`
- Modify: `dashboard/frontend/app.js:3234-3258` (presentation-only role/status text, if needed)
- Test: `dashboard/backend/tests/test_frontend_account_page.py`

**Interfaces:**
- Consumes: all current account element IDs and `updateAccountPage()` behavior.
- Produces: a `.account-workspace` containing `.account-identity` and `.account-settings-grid` while preserving every existing form and control contract.

- [ ] **Step 1: Move the existing summary rows into the identity column**

Inside `#accountSignedIn`, remove the two top-level `.account-row` elements and replace them with:

```html
<div class="account-workspace">
    <aside class="account-identity" aria-labelledby="accountIdentityHeading">
        <div class="account-identity-avatar-wrap">
            <span id="accountAvatarPreview" class="auth-avatar auth-avatar--large" aria-hidden="true"></span>
            <span id="accountProfileStatus" class="account-profile-status">Signed in</span>
        </div>
        <p class="account-section-kicker">Account identity</p>
        <h3 id="accountIdentityHeading" class="account-identity-name">—</h3>
        <p id="accountIdentityEmail" class="account-identity-email">—</p>
        <dl class="account-identity-facts">
            <div><dt>Display name</dt><dd id="accountDisplayName">—</dd></div>
            <div><dt>Email</dt><dd id="accountEmail">—</dd></div>
            <div><dt>Role</dt><dd id="accountRole">Member</dd></div>
        </dl>
        <div class="account-identity-actions">
            <button id="authLogoutBtn" class="auth-btn auth-btn-danger" type="button">Log out</button>
        </div>
    </aside>

    <div class="account-settings-grid">
        <!-- existing Display name, email, avatar, and password sections stay here unchanged -->
    </div>
</div>
```

Keep the existing `#accountDisplayName` and `#accountEmail` IDs in the identity facts; do not duplicate those IDs elsewhere. The existing `#accountAvatarPreview` moves into the identity column, while the Profile photo section keeps its upload/remove controls and no longer renders a second avatar preview.

- [ ] **Step 2: Keep all existing editable sections and controls intact**

Place the current `accountDisplayNameForm`, `accountEmailForm`, profile photo controls, and `changePasswordForm` inside `.account-settings-grid` in this order:

1. Display name
2. Profile photo
3. Change email address
4. Change password

Keep `emailChangeIdle`, `emailChangeCodeStep`, `emailChangeStepCopy`, `emailChangeSubmitBtn`, `emailChangeCancelBtn`, password policy hints, success/error elements, and avatar file input IDs unchanged.

- [ ] **Step 3: Update only presentation fields in `updateAccountPage()`**

After the existing `nameEl` and `emailEl` updates, populate the new identity presentation nodes without changing any API calls:

```javascript
const identityName = document.getElementById('accountIdentityHeading');
const identityEmail = document.getElementById('accountIdentityEmail');
const roleEl = document.getElementById('accountRole');
if (identityName) identityName.textContent = user.display_name || '—';
if (identityEmail) identityEmail.textContent = user.email || '—';
if (roleEl) roleEl.textContent = user.role === 'admin' ? 'Administrator' : 'Member';
```

Do not change the existing focused-input guard, avatar rendering, remove-button state, signed-out state, or form event handlers.

- [ ] **Step 4: Run the structural tests**

Run:

```bash
pytest -q dashboard/backend/tests/test_frontend_account_page.py
```

Expected: all Account source contracts pass.

### Task 3: Add the Account-Scoped Two-Column Visual System

**Files:**
- Modify: `dashboard/frontend/styles.css` in the existing Account section near `.account-section`
- Test: `dashboard/backend/tests/test_frontend_account_page.py`

**Interfaces:**
- Consumes: `.account-workspace`, `.account-identity`, `.account-settings-grid`, and existing account form classes.
- Produces: desktop two-column layout and mobile single-column fallback without changing shared styles.

- [ ] **Step 1: Add scoped desktop layout rules**

Add rules under the existing account section:

```css
.account-view {
    max-width: 1180px;
}

.account-view .account-workspace {
    display: grid;
    grid-template-columns: minmax(220px, 0.72fr) minmax(0, 1.7fr);
    gap: 20px;
    align-items: start;
}

.account-view .account-identity {
    position: sticky;
    top: 88px;
    min-width: 0;
    padding: 24px;
    border: 1px solid rgba(103, 232, 249, 0.22);
    border-radius: 12px;
    background: linear-gradient(145deg, rgba(18, 33, 62, 0.98), rgba(11, 20, 40, 0.98));
}

.account-view .account-settings-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    min-width: 0;
}
```

Use Account-only selectors for identity typography, fact rows, panel surfaces, form spacing, focus-visible states, and the logout placement. Do not rewrite shared `.auth-btn`, `.auth-field`, `.page-header`, or `.page-view` declarations.

- [ ] **Step 2: Normalize the four setting panels**

Style `.account-view .account-settings-grid > .account-section` as equal visual panels with consistent padding, border, radius, and `min-width: 0`. Keep the email and password forms readable by allowing them to span both columns when required:

```css
.account-view .account-settings-grid > .account-section {
    margin-top: 0;
    min-width: 0;
    padding: 18px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px;
    background: rgba(17, 29, 54, 0.82);
}

.account-view .account-settings-grid > .account-section:nth-child(3),
.account-view .account-settings-grid > .account-section:nth-child(4) {
    grid-column: span 2;
}
```

Ensure inputs and button rows use `width: 100%` only inside `.account-view`; preserve the existing email `1fr auto` action layout and password policy hint behavior.

- [ ] **Step 3: Add responsive and reduced-motion-safe behavior**

At `max-width: 860px`, collapse both grids to one column, remove sticky positioning, and make the identity panel full width. At `max-width: 540px`, stack avatar actions and let email action buttons wrap without horizontal overflow. Use existing transition conventions and do not add animation that affects other routes.

- [ ] **Step 4: Run source and lint checks**

Run:

```bash
git diff --check
pytest -q dashboard/backend/tests/test_frontend_account_page.py dashboard/backend/tests/test_frontend_fast_boot.py
```

Expected: no whitespace errors and all Account/fast-boot contracts pass.

### Task 4: Visual QA and Regression Verification

**Files:**
- Modify: none unless QA finds an Account-only issue
- Test: `dashboard/backend/tests/test_frontend_account_page.py`, `dashboard/backend/tests/test_frontend_fast_boot.py`

**Interfaces:**
- Consumes: the completed Account markup and scoped CSS.
- Produces: verified desktop/mobile rendering and a clean, focused diff.

- [ ] **Step 1: Start the local frontend server**

Use the repository's existing frontend start command or static server from the repository root. Open the Account route with a signed-in test session and verify the page title, identity panel, four setting panels, logout placement, and all focus states.

- [ ] **Step 2: Check desktop and mobile widths**

Inspect at approximately 1440px and 390px wide. Confirm the desktop view fills the content area without the screenshot's large unused right side, while the mobile view is a single readable column with no horizontal scroll.

- [ ] **Step 3: Check non-Account routes for selector leakage**

Open Home, Community, Competition, Credits, and Admin. Confirm their layout is unchanged; the final CSS diff must contain only `.account-view`-scoped redesign selectors plus the existing account rules that are intentionally adjusted.

- [ ] **Step 4: Run the focused regression suite**

Run:

```bash
pytest -q dashboard/backend/tests/test_frontend_account_page.py dashboard/backend/tests/test_frontend_fast_boot.py dashboard/backend/tests/test_credits_frontend.py
git diff --check
git status --short
```

Expected: all focused tests pass, no diff-check errors, and no `.superpowers/` files are staged.

- [ ] **Step 5: Commit the implementation**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_frontend_account_page.py
git commit -m "feat: redesign account settings layout"
```
