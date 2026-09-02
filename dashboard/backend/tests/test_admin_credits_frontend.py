"""Static contracts for the Admin Grant Credits console."""

from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
ADMIN_JS = (FRONTEND / "js" / "admin-credits.js").read_text(encoding="utf-8")
ADMIN_TABS_JS = (FRONTEND / "js" / "admin-tabs.js").read_text(encoding="utf-8")
STYLES = (FRONTEND / "styles.css").read_text(encoding="utf-8")


def test_admin_grant_console_is_integrated_into_admin_users_panel():
    admin_start = APP_HTML.index('id="adminView"')
    admin_end = APP_HTML.index('id="creditsRefundDialog"')
    admin_markup = APP_HTML[admin_start:admin_end]
    summary_start = admin_markup.index('class="admin-credits-summary"')
    summary_end = admin_markup.index('<div class="admin-credits-actions"', summary_start)
    summary_markup = admin_markup[summary_start:summary_end]
    assert 'id="adminCreditsSection"' in admin_markup
    assert 'id="adminGrantPoolForm"' in admin_markup
    assert 'id="adminCreditsUsersBody"' in admin_markup
    assert 'id="adminCreditsUsersHeading">Account Management</h4>' in admin_markup
    assert 'id="adminCreditsUserCount"' in admin_markup
    assert 'id="adminCreditsActivityBody"' in admin_markup
    assert 'id="adminGrantReasonDialog"' in admin_markup
    assert 'id="adminGrantReason" type="text" maxlength="500" value="Approved Grant allocation."' in admin_markup
    assert 'id="adminCreditsMutationReason"' not in admin_markup
    assert 'Account allocation' not in admin_markup
    assert 'Assign Grant Credits' not in admin_markup
    assert 'id="adminCreditsPoolTotal"' in summary_markup
    assert 'id="adminCreditsPoolRingAvailable"' in summary_markup
    assert 'id="adminCreditsPoolRingAllocated"' in summary_markup
    assert '>Available</' in summary_markup
    assert '>Allocated</' in summary_markup
    assert 'class="admin-credits-metric"' not in summary_markup
    assert 'id="creditsApiKeySecret"' not in admin_markup
    assert 'id="adminPanelGrantPool"' not in admin_markup
    assert 'id="adminTabGrantPool"' not in admin_markup
    assert 'data-admin-tab="grant-pool"' not in admin_markup
    assert 'data-admin-tab="users"' in admin_markup
    assert 'data-admin-tab="providers"' in admin_markup
    assert 'data-admin-tab="activity"' in admin_markup
    nav_start = admin_markup.index('<nav id="adminTabs"')
    nav_end = admin_markup.index('</nav>', nav_start)
    nav_markup = admin_markup[nav_start:nav_end]
    assert nav_markup.count('data-admin-tab=') == 4
    assert nav_markup.index('data-admin-tab="analytics"') < nav_markup.index('data-admin-tab="users"')
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="providers"')
    assert nav_markup.index('data-admin-tab="providers"') < nav_markup.index('data-admin-tab="activity"')
    assert admin_markup.index('id="adminStats"') < admin_markup.index('id="adminCreditsSection"')
    assert admin_markup.index('id="adminCreditsSection"') < admin_markup.index('class="admin-credits-users"')
    assert 'id="adminGrantPoolAmount" type="number" step="0.000001"' in admin_markup
    assert 'Assigned this month' not in admin_markup
    assert 'Reclaimed this month' not in admin_markup
    assert 'id="adminGrantFundForm"' not in admin_markup
    assert 'id="adminGrantReduceForm"' not in admin_markup
    assert 'id="adminGrantFundSource"' not in admin_markup
    assert 'id="adminGrantReduceSource"' not in admin_markup


def test_admin_grant_client_uses_server_api_and_never_persists_secrets():
    assert "/api/admin/credits" in ADMIN_JS
    assert "client_request_id" in ADMIN_JS
    assert "amount_micro" in ADMIN_JS
    assert "parseSignedCreditsMicro" in ADMIN_JS
    assert "source: 'admin-console'" in ADMIN_JS
    assert "Math.abs(signedAmountMicro)" in ADMIN_JS
    assert "pool.pool_available_micro" in ADMIN_JS
    assert "pool.allocated_to_users_micro" in ADMIN_JS
    assert "formatCreditsMicro" in ADMIN_JS
    assert "strokeDasharray" in ADMIN_JS
    assert "localStorage" not in ADMIN_JS
    assert "api_key" not in ADMIN_JS
    assert "innerHTML" not in ADMIN_JS


def test_admin_grant_console_surfaces_restrictions_and_refund_reinstate_only():
    assert ">Status</th>" in APP_HTML
    assert "restriction_reason" in ADMIN_JS
    assert "outstanding_credits_micro" in ADMIN_JS
    assert "reason !== 'llm_overage'" in ADMIN_JS
    assert "/api/admin/credits/accounts/${Number(user.id)}/reinstate" in ADMIN_JS


def test_admin_grant_client_defaults_to_short_user_pages():
    assert "ADMIN_CREDITS_USERS_PAGE_SIZE = 25" in ADMIN_JS
    assert "usersLimit: ADMIN_CREDITS_USERS_PAGE_SIZE" in ADMIN_JS


def test_admin_grant_console_is_wired_to_auth_navigation_and_refresh():
    assert "window.AdminCredits.syncAuth(user)" in APP_JS
    assert APP_JS.count("window.AdminCredits.onEnter()") >= 2


def test_admin_grant_console_has_responsive_operation_and_audit_styles():
    assert ".admin-credits-console" in STYLES
    assert ".admin-credits-summary" in STYLES
    assert ".admin-credits-users-table" in STYLES
    assert ".admin-credits-activity-table" in STYLES
    assert ".admin-tabs" in STYLES
    assert ".admin-credits-pool-ring" in STYLES
    assert ".admin-credits-pool-legend" in STYLES
    assert "@media (max-width: 600px)" in STYLES


def test_admin_tabs_default_to_analytics_and_are_url_backed():
    assert "DEFAULT_TAB = 'analytics'" in ADMIN_TABS_JS
    assert "adminTab" in ADMIN_TABS_JS
    assert "aria-selected" in ADMIN_TABS_JS


def test_admin_tabs_have_four_tabs_in_usage_order_and_legacy_alias():
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index('</nav>', nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    assert nav_markup.count('data-admin-tab=') == 4
    assert nav_markup.index('data-admin-tab="analytics"') < nav_markup.index('data-admin-tab="users"')
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="providers"')
    assert nav_markup.index('data-admin-tab="providers"') < nav_markup.index('data-admin-tab="activity"')
    assert "value === 'grant-pool' ? 'users' : value" in ADMIN_TABS_JS
    assert "setTab(requested || DEFAULT_TAB);" in ADMIN_TABS_JS


def test_admin_visual_assets_use_fresh_cache_versions():
    assert 'js/admin-credits.js?v=6' in APP_HTML
    assert 'js/admin-tabs.js?v=3' in APP_HTML
