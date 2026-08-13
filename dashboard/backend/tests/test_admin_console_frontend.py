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
