"""Static-source guards for the Admin console UI.

/app has no build step and no JS test toolchain, so these contracts are
asserted against the shipped source as text (see ``_frontend_source``).
"""

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, STYLES, fn_body


def test_admin_list_request_carries_a_page_window():
    """Without limit/offset the console silently shows only the first page."""
    body = fn_body("  listUsers({ limit = ADMIN_USERS_PAGE_SIZE, offset = 0 } = {})")
    assert "URLSearchParams" in body
    assert "limit" in body and "offset" in body
    assert "/api/admin/users?" in body


def test_admin_pager_controls_are_wired():
    for element_id in ("adminPrevBtn", "adminNextBtn", "adminUsersRange"):
        assert f'id="{element_id}"' in APP_HTML, element_id
    assert ".admin-pager" in STYLES

    pager = fn_body("function _renderAdminPager()")
    # The count is the whole point: 100 rows with no total reads as "that is
    # everyone" when it may be the first 100 of 400.
    assert "Showing" in pager and "of ${total}" in pager
    assert "prevBtn.disabled" in pager and "nextBtn.disabled" in pager

    load = fn_body("async function loadAdminUsers({ offset } = {})")
    assert "adminUsersPage.total" in load
    assert "_renderAdminPager()" in load


def test_blank_quota_input_is_refused_not_silently_dropped():
    """NaN -> null -> Pydantic "omitted" made a no-op save flash success."""
    reader = fn_body(
        "function _readAdminQuota(rowEl, field, label, { min, max })"
    )
    assert "cannot be blank" in reader
    assert "Number.isInteger" in reader

    save = fn_body("async function saveAdminUserRow(rowEl)")
    assert "_readAdminQuota(" in save
    # The guard has to return before the request, not merely annotate it.
    assert "if (invalid) {" in save
    assert "maxField.value" in save and "creditsField.value" in save


def test_admin_403_refreshes_the_cached_role():
    """A demoted admin keeps the menu until the client re-reads /me."""
    handler = fn_body("async function _handleAdminAccessLost()")
    assert "AuthAPI.me()" in handler
    assert "applyUpdatedUser" in handler
    assert "navigateToPage('home')" in handler

    load = fn_body("async function loadAdminUsers({ offset } = {})")
    assert "error?.status === 403" in load
    assert "_handleAdminAccessLost()" in load


def test_request_errors_carry_their_status_code():
    """The 403 handling above is only reachable if the status survives."""
    assert "error.status = response.status;" in APP_JS


def test_a_zero_quota_survives_the_console_round_trip():
    """0 is the suspend value, and 0 is what a truthiness check drops.

    The console has three places a quota can silently vanish: the input's own
    ``min``, the diff that decides what to PATCH, and the repaint that writes
    the server's answer back. Each has to be zero-safe, and each is written in
    a style where the unsafe version reads perfectly natural (``if (value)``,
    ``value || fallback``).
    """
    assert "max_concurrent_backtests: { min: 0, max: 20 }" in APP_JS

    save = fn_body("async function saveAdminUserRow(rowEl)")
    # String comparison, not a truthy diff: `if (maxField.value !== ...)` would
    # still work, but `if (maxField.value && ...)` would drop the suspend.
    assert "String(maxField.value) !== rowEl.getAttribute('data-server-max')" in save

    repaint = fn_body("function _applyAdminRowFromUser(rowEl, userPayload)")
    # `== null`, not `!value` — the server answering 0 must repaint as 0.
    assert "if (value == null) return;" in repaint


def test_credits_column_states_what_it_binds():
    """Credits are metered now (issue #351), but only when the deployment arms
    ``CREDITS_METERING_ENABLED`` — so the note is filled from the stats
    response rather than written into the markup. The static label it replaced
    said "not enforced yet", which would have gone on saying that forever.
    Without an accurate note an operator zeroes an abusive account, sees a 200,
    and believes they acted."""
    assert 'id="adminCreditsNote"' in APP_HTML
    assert "not enforced yet" not in APP_HTML
    assert ".admin-th-note" in STYLES


def test_role_confirm_dialog_cannot_be_forged_by_an_address():
    """confirm() is plain text with no markup to escape, and the prompts are
    multi-line — so an address carrying newlines writes its own sentences into
    the box an admin reads before granting admin."""
    assert "function _adminConfirmEmail(value)" in APP_JS
    helper = fn_body("function _adminConfirmEmail(value)")
    assert "replace(/\\s+/g, ' ')" in helper

    role_save = fn_body("async function saveAdminUserRole(rowEl, nextRole)")
    assert "_adminConfirmEmail(" in role_save
    assert "window.confirm(" in role_save
