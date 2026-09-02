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
    assert app_version >= 126


# --- Resend code -------------------------------------------------------------


def _reset_step_markup():
    start = APP_HTML.index('id="resetCodeStep"')
    return APP_HTML[start : APP_HTML.index('id="authError"', start)]


def test_resend_button_lives_inside_the_code_step():
    step = _reset_step_markup()
    assert 'id="resetResendBtn"' in step
    # A submit button here would fire the stage-2 handler on Enter.
    assert re.search(r'<button[^>]*type="button"[^>]*id="resetResendBtn"', step)


def test_stage_one_success_starts_the_resend_countdown():
    init = fn_body("function initAuthUI")
    advance = init.index("resetStage = 2")
    assert "startResendCountdown(" in init[advance : advance + 900]


def test_resend_and_stage_two_use_the_address_stage_one_submitted():
    # The code went to the address stage 1 sent, not whatever is in the input
    # now; both the resend and the final submit must key on that same value.
    init = fn_body("function initAuthUI")
    assert "resetEmail = email" in init
    assert "AuthAPI.requestPasswordReset(resetEmail)" in init
    assert "AuthAPI.resetPassword(resetEmail," in init
    assert "AuthAPI.resetPassword(email," not in init
    # ...and the locked field shows that address, not an edit made in flight.
    assert "emailInput.value = resetEmail" in init
    assert "emailInput.readOnly = true" in init


def test_resend_state_is_cleared_with_the_form():
    init = fn_body("function initAuthUI")
    start = init.index("resetPasswordResetForm = () => {")
    body = init[start : init.index("};", start)]
    assert "stopResendCountdown()" in body
    assert "resetEmail = ''" in body
    assert "readOnly = false" in body


def test_resend_reads_retry_after_from_a_429():
    init = fn_body("function initAuthUI")
    click = init.index("resetResendBtn?.addEventListener('click'")
    handler = init[click : click + 2500]
    assert "error.retryAfter" in handler
    # A stale rate-limit banner must not sit beside fresh "sent" copy.
    assert "errorEl.hidden = true" in handler


def test_stage_one_429_still_opens_the_code_step():
    # A resubmit inside the minute (reload, second tab, the deep link) is
    # refused by the cooldown -- but the code already mailed is still valid,
    # so the user must get the code input, not a dead stage 1.
    init = fn_body("function initAuthUI")
    stage1 = init.index("if (resetStage === 1)")
    catch = init.index("} catch (error) {", stage1)
    shared_guard = init.index("if (!email || !password)", catch)
    catch_block = init[catch:shared_guard]
    assert "error.status === 429" in catch_block
    assert "enterCodeStep(" in catch_block
    assert "error.retryAfter" in catch_block
    # The helper is the single owner of "we are on stage 2 now".
    assert "resetStage = 2" in fn_body("function initAuthUI")
    assert init.count("resetStage = 2") == 1


def test_stale_reset_responses_are_dropped_after_a_mode_switch():
    # Every await in the reset flow re-checks a generation counter that
    # resetPasswordResetForm bumps, so a response landing after "Back to sign
    # in" cannot lock the login email field or repaint the login error.
    init = fn_body("function initAuthUI")
    start = init.index("resetPasswordResetForm = () => {")
    body = init[start : init.index("};", start)]
    assert "resetGeneration += 1" in body
    # stage-1 success + catch, resend success + catch: four checks at least.
    assert init.count("gen !== resetGeneration") >= 4


def test_auth_api_exposes_retry_after():
    request = fn_body("async request(path, options = {})")
    assert "retry-after" in request
    assert "error.retryAfter" in request
