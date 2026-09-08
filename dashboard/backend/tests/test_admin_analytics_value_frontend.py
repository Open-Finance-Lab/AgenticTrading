"""Contracts for the demo-aligned Admin user-value Analytics frontend."""

from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, STYLES


ROOT = Path(__file__).resolve().parents[2]
VALUE_JS_PATH = ROOT / "frontend" / "js" / "admin-analytics-value.js"
PROFILE_JS_PATH = ROOT / "frontend" / "js" / "admin-analytics.js"


def value_source() -> str:
    return VALUE_JS_PATH.read_text(encoding="utf-8")


def profile_source() -> str:
    return PROFILE_JS_PATH.read_text(encoding="utf-8")


def test_demo_aligned_controller_uses_acquisition_as_the_pulse_source():
    source = value_source()
    for contract in (
        "summarizeAcquisition",
        "renderPulse",
        "renderReturnPanel",
        "renderValueExchange",
        "renderActionQueue",
        "/api/admin/analytics/overview",
        "accepted_runs_in_range",
    ):
        assert contract in source
    assert "analyticsRange: '1w'" in source
    assert "Promise.allSettled" in source
    assert "innerHTML" not in source


def test_overview_has_the_approved_story_order_and_metrics():
    start = APP_HTML.index('id="adminAnalyticsOverview"')
    end = APP_HTML.index('id="adminAnalyticsDeepSections"', start)
    markup = APP_HTML[start:end]
    ordered = [
        'id="adminAnalyticsPulse"',
        'id="adminAnalyticsAcquisition"',
        'id="adminAnalyticsReturn"',
        'id="adminAnalyticsValueExchange"',
        'id="adminAnalyticsActionQueue"',
    ]
    assert [markup.index(item) for item in ordered] == sorted(markup.index(item) for item in ordered)
    for label in (
        "Active users",
        "Task completed",
        "Repeat users",
        "ATL Credits settled",
    ):
        assert label in markup
    for removed in ("Core users", "Lifecycle distribution", "Recent 5-day movement"):
        assert removed not in markup
    for removed_id in ("adminAcquisitionBlocked", "adminAcquisitionGroupBy", "adminPriorityFilters"):
        assert removed_id not in markup


def test_filters_apply_without_a_visible_apply_button():
    source = value_source()
    assert "scheduleFilterRefresh" in source
    assert "adminAnalyticsValueFilters" in source
    assert "change" in source
    start = APP_HTML.index('id="adminAnalyticsOverview"')
    end = APP_HTML.index('id="adminAnalyticsDeepSections"', start)
    assert "Apply filters" not in APP_HTML[start:end]


def test_acquisition_drilldown_opens_a_url_backed_users_directory():
    source = value_source()
    for contract in (
        "openAnalyticsUsers",
        "adminAnalyticsUsersDirectory",
        "analyticsUsersView",
        "AdminTabs?.setTab('users')",
    ):
        assert contract in source or contract in APP_HTML
    for element_id in (
        "adminAnalyticsUsersDirectoryBody",
        "adminAnalyticsUsersDirectoryPrev",
        "adminAnalyticsUsersDirectoryNext",
    ):
        assert f'id="{element_id}"' in APP_HTML


def test_value_client_handles_access_loss_partial_errors_and_stale_data():
    source = value_source()
    assert "handleAccessLost" in source
    assert "error?.status !== 401" in source
    assert "error?.status !== 403" in source
    assert "keepStaleData" in source
    assert "Promise.allSettled" in source
    assert "renderPrimaryErrors" in source
    assert "Retry section" in APP_HTML


def test_value_rendering_is_safe_and_accessible():
    source = value_source()
    assert "innerHTML" not in source
    assert "textContent" in source
    assert "method: 'GET'" in source
    assert "aria-pressed" in source
    assert "admin-analytics-overview" in STYLES
    assert ".admin-value-disclosure" in STYLES


def test_profile_is_progress_first_and_keeps_five_sections():
    start = APP_HTML.index('id="adminAnalyticsProfile"')
    end = APP_HTML.index('id="adminPanelUsers"', start)
    markup = APP_HTML[start:end]
    assert markup.index('id="adminAnalyticsMilestones"') < markup.index('id="adminAnalyticsProfileTabs"')
    for section in ("Overview", "Timeline", "Runs", "Usage", "Sessions"):
        assert f">{section}<" in markup
    assert "Default provider" not in markup
    assert "Legacy status" not in markup
    source = profile_source()
    assert "account_signed_up" in source
    assert "backtest_completed" in source
    assert "Provider / model" not in source
    assert "provider_id" not in source


def test_profile_navigation_keeps_safe_dom_and_account_link():
    source = profile_source()
    assert "openAccountManagement" in source
    assert "openProfile" in source
    assert "closeProfile" in source
    assert "innerHTML" not in source
    assert "method: 'GET'" in source
    assert "data-analytics-section-tab" in APP_HTML
    assert "aria-selected" in APP_HTML


def test_acquisition_surface_excludes_token_and_provider_details():
    combined = value_source() + profile_source()
    for prohibited in (
        "Input tokens",
        "Output tokens",
        "Total tokens",
        "Provider / model",
        "provider_response_body",
        "prompt_text",
        "api_key",
    ):
        assert prohibited not in combined


def test_static_asset_versions_are_bumped_for_the_new_surface():
    assert "styles.css?v=138" in APP_HTML
    assert "js/admin-analytics.js?v=9" in APP_HTML
    assert "js/admin-analytics-value.js?v=7" in APP_HTML


def test_demo_aligned_visual_system_and_responsive_breakpoints_exist():
    for selector in (
        ".admin-rail-shell",
        ".admin-analytics-page-head",
        ".admin-analytics-filter-grid",
        ".admin-analytics-metrics",
        ".admin-analytics-story-section",
        ".admin-analytics-two-col",
        ".admin-analytics-acquisition-table",
        ".admin-analytics-action-table",
        ".admin-analytics-progress",
    ):
        assert selector in STYLES
    assert "@media (max-width: 760px)" in STYLES
    assert "@media (max-width: 470px)" in STYLES
    assert "prefers-reduced-motion: reduce" in STYLES
