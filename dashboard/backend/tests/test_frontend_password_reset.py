"""Password-reset UI source-shape guards (#187).

/app has no JS harness, so the reset mode's structural contracts are asserted
against the shipped source (the test_frontend_account_page.py convention, via
_frontend_source's brace-matched slicing).
"""

import re

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, FRONTEND, fn_body


def test_forgot_password_link_exists_and_is_login_mode_only():
    assert 'id="authForgotPasswordBtn"' in APP_HTML
    set_mode = fn_body("function setAuthMode")
    assert "forgotBtn.hidden = mode !== 'login'" in set_mode


def test_reset_mode_drops_the_password_requirement():
    # A hidden required input fails native form validation silently, so the
    # required flag must travel with the field's visibility.
    set_mode = fn_body("function setAuthMode")
    assert "passwordInput.required = mode !== 'reset'" in set_mode
    assert "passwordField.hidden = mode === 'reset'" in set_mode


def test_reset_branch_precedes_the_shared_email_password_guard():
    # Reset mode has no password value, so the shared guard would silently
    # no-op stage 1; the reset branch must come first in the submit handler.
    init = fn_body("function initAuthUI")
    submit_at = init.index("form?.addEventListener('submit'")
    reset_branch = init.index("authMode === 'reset'", submit_at)
    shared_guard = init.index("if (!email || !password)", submit_at)
    assert reset_branch < shared_guard
    # ...and the reset branch never runs the login/signup success path.
    reset_block = init[reset_branch:shared_guard]
    assert "setAuthState" not in reset_block
    assert "navigateToPage('agents')" not in reset_block
    assert "claimAgentsForUser" not in reset_block


def test_logging_out_resets_the_password_reset_form():
    # clearAuthState is the choke point every sign-out path funnels through;
    # without the reset, user B resumes user A's half-finished reset stage.
    clear_fn = fn_body("function clearAuthState")
    assert "resetPasswordResetForm()" in clear_fn
    assert "let resetPasswordResetForm = () => {};" in APP_JS
    # ...and the hook is actually rebound to the closure's reset.
    assert "resetPasswordResetForm = () => {" in fn_body("function initAuthUI")


def test_any_mode_switch_resets_the_reset_stage():
    assert "resetPasswordResetForm()" in fn_body("function setAuthMode")


def test_stage_two_copy_mentions_expiry_and_the_spam_folder():
    init = fn_body("function initAuthUI")
    assert "Check your spam folder too." in init
    assert "expires in 15 minutes" in init
    # The masked address is the user's own typed input, never stored data.
    assert "maskEmailForDisplay(email)" in init


def test_deep_link_accepts_auth_reset_and_the_landing_page_uses_it():
    open_fn = fn_body("function openAuthFromUrl")
    assert "'reset'" in open_fn
    # The landing page's hand-inlined modal links here rather than growing a
    # duplicate reset UI.
    landing = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "/app?auth=reset" in landing
    assert 'id="landingAuthForgot"' in landing


def test_cache_bust_version_was_bumped():
    # Parsed >= rather than ==, per the convention in
    # test_frontend_account_page.py::test_cache_bust_versions_were_bumped.
    app_version = int(re.search(r"app\.js\?v=(\d+)", APP_HTML).group(1))
    assert app_version >= 125
