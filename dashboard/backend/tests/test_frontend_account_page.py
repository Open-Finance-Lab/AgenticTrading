"""Account-page markup and cascade guards.

The frontend has no JS test harness, and these two contracts are structural
rather than behavioural -- an ordering and a CSS source-order requirement -- so
they are asserted against the shipped source directly.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_STYLES_CSS = _FRONTEND / "styles.css"


def _account_card() -> str:
    html = _APP_HTML.read_text(encoding="utf-8")
    start = html.index('<div id="accountSignedIn"')
    end = html.index('<div id="accountSignedOut"')
    return html[start:end]


def test_logout_button_is_last_in_the_account_identity_region():
    card = _account_card()
    identity_start = card.index('class="account-identity"')
    identity_end = card.index("</aside>", identity_start)
    identity = card[identity_start:identity_end]
    logout_at = identity.index('id="authLogoutBtn"')

    for marker in ('id="accountDisplayName"', 'id="accountEmail"', 'id="accountRole"'):
        assert identity.index(marker) < logout_at, f"{marker} must come before Log out"

    # Logout is the final interactive item in the identity panel. The editable
    # settings intentionally follow in their own sibling region.
    tail = identity[logout_at + len('id="authLogoutBtn"'):]
    assert 'id="' not in tail, f"something with an id follows Log out: {tail!r}"
    assert card.index('class="account-settings-grid"') > identity_end


def test_logout_button_carries_the_danger_class():
    card = _account_card()
    match = re.search(r'<button[^>]*id="authLogoutBtn"[^>]*>', card)
    assert match, "logout button not found in the account card"
    tag = match.group(0)
    # Assert on the button's OWN tag. A substring search over the whole card
    # would pass if "auth-btn-danger" appeared anywhere else, and a fixed-width
    # window before the id cannot see the class at all -- this file's markup
    # puts id= before class=.
    assert "auth-btn-danger" in tag
    assert "auth-btn-secondary" not in tag


def test_header_dropdown_logout_is_untouched():
    # The brief targeted the account-page button only. Removing the dropdown
    # item would also make docs/source/lab/accounts.rst factually wrong.
    html = _APP_HTML.read_text(encoding="utf-8")
    assert 'id="accountMenuLogoutBtn"' in html


def test_auth_btn_danger_is_declared_after_the_generic_hover():
    """.auth-btn:hover and .auth-btn-danger:hover both score (0,2,0).

    With identical specificity, source order alone decides. Declared earlier,
    the logout button reverts to info-blue on hover -- red at rest, wrong on
    mouseover, which a screenshot taken at rest would not catch.
    """
    css = _STYLES_CSS.read_text(encoding="utf-8")
    assert css.index(".auth-btn-danger:hover") > css.index(".auth-btn:hover")
    assert css.index(".auth-btn-danger {") > css.index(".auth-btn {")


def test_account_card_section_order():
    card = _account_card()
    order = [
        'id="accountDisplayName"',
        'id="accountEmail"',
        'id="authLogoutBtn"',
        'id="accountDisplayNameForm"',
        'id="avatarUploadBtn"',
        'id="accountEmailForm"',
        'id="changePasswordForm"',
    ]
    positions = [card.index(marker) for marker in order]
    assert positions == sorted(positions), "account card sections are out of order"


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


def test_email_change_copy_mentions_the_spam_folder():
    """An unauthenticated single sender has materially degraded inbox placement,
    and a code silently in spam is indistinguishable from one never sent. BOTH
    stages must say so -- a file-wide count could be satisfied by two mentions
    in one branch, or by unrelated text elsewhere in the file."""
    js = (_FRONTEND / "app.js").read_text(encoding="utf-8")
    mentions = [line.lower() for line in js.splitlines() if "spam folder" in line.lower()]

    assert len(mentions) >= 2
    # stage 'new' -- code went to the new address
    assert any("code sent to" in line for line in mentions)
    # stage 'old' -- code went to the current address
    assert any("we sent a 6-character code" in line for line in mentions)


def test_email_form_states_the_change_interval_before_the_user_submits():
    """The 7-day limit has to be visible up front, not discovered as a 429.

    Reads the number out of the backend constant rather than hardcoding it, so
    changing the policy without changing the copy fails here instead of quietly
    shipping a screen that lies.
    """
    from dashboard.backend.users import EMAIL_CHANGE_MIN_INTERVAL_DAYS

    html = _APP_HTML.read_text(encoding="utf-8")
    idle = html[html.index('<div id="emailChangeIdle">'):html.index('<div id="emailChangeCodeStep"')]

    assert f"once every {EMAIL_CHANGE_MIN_INTERVAL_DAYS} days" in idle


def test_logging_out_resets_the_email_change_form():
    """initEmailChangeForm keeps its `stage` in a closure, so without a reset
    on sign-out, user B logging in on the same tab resumes user A's
    half-finished change: a code box for a request B cannot complete.
    clearAuthState is the choke point every sign-out path funnels through
    (explicit logout, missing token, expired session) -- the reset must live
    there, not just on the logout button."""
    js = (_FRONTEND / "app.js").read_text(encoding="utf-8")

    clear_fn = js[js.index("function clearAuthState") : js.index("function updateAccountPage")]
    assert "resetEmailChangeForm()" in clear_fn
    # ...and the hook must actually be bound to the closure's reset, or the
    # call above is a no-op forever.
    assert "resetEmailChangeForm = reset" in js


def test_active_agent_is_only_cleared_on_a_real_sign_out():
    """A transient /api/auth/me failure must not cost the agent selection.

    refreshAuthUser() funnels *every* failure through clearAuthState() from a
    bare catch -- a Render free-tier cold start or a first-request-after-idle
    500 included -- and it runs immediately after restoreActiveAgentSession()
    on boot. Clearing the active agent inside clearAuthState() therefore
    silently undoes the restore for a signed-in user whose boot probe merely
    timed out. The wipe belongs on the paths that prove the session is gone:
    the logout button, and a 401.
    """
    from dashboard.backend.tests._frontend_source import fn_body

    clear_auth = fn_body("function clearAuthState")
    assert "ACTIVE_AGENT_KEY" not in clear_auth
    assert "trading-session-id" not in clear_auth

    clear_agent = fn_body("function clearActiveAgentSession")
    assert "localStorage.removeItem(ACTIVE_AGENT_KEY)" in clear_agent
    assert "trading-session-id" in clear_agent

    assert "clearActiveAgentSession()" in fn_body("async function logoutUser")

    refresh = fn_body("async function refreshAuthUser")
    assert "error?.status === 401" in refresh
    assert "clearActiveAgentSession()" in refresh
    # The status has to actually reach the caller -- AuthAPI.request throws a
    # bare Error, so without this the 401 gate is dead code that never fires.
    assert "error.status = response.status" in fn_body("  async request(path")


def test_cache_bust_versions_were_bumped():
    """Parsed >= rather than a literal ==: these counters are bumped by unrelated
    work too, and an equality assert turns CI red on every open PR the moment
    anyone else bumps one (which is what blocked #88-#91 here before)."""
    html = _APP_HTML.read_text(encoding="utf-8")
    styles_version = int(re.search(r"styles\.css\?v=(\d+)", html).group(1))
    app_version = int(re.search(r"app\.js\?v=(\d+)", html).group(1))

    assert styles_version >= 69
    assert app_version >= 51
