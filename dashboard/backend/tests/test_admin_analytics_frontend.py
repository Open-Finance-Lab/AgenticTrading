"""Frontend and PR 2 response contracts for the read-only Admin Analytics UI."""

import json
from pathlib import Path

from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
    AnalyticsUserProfile,
    PaginatedUsers,
)
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, STYLES


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"
ANALYTICS_JS_PATH = FRONTEND / "js" / "admin-analytics.js"
ADMIN_TABS_JS_PATH = FRONTEND / "js" / "admin-tabs.js"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def analytics_source() -> str:
    return ANALYTICS_JS_PATH.read_text(encoding="utf-8")


def test_safe_fixtures_have_no_prohibited_response_fields():
    prohibited = {
        "api_key", "auth_token", "password", "verification_code",
        "prompt", "instruction", "strategy", "portfolio", "form_value",
        "provider_response_body", "ip_address", "user_agent",
        "credential_ciphertext", "network_hash", "session_id",
    }
    for path in sorted(FIXTURES.glob("*.json")):
        payload = load_fixture(path.name)
        assert prohibited.isdisjoint(set(walk_keys(payload))), path.name


def test_fixtures_match_committed_pr2_shapes():
    overview = load_fixture("overview.json")
    partial = load_fixture("overview_partial_error.json")
    users = load_fixture("users.json")
    profile = load_fixture("user_detail.json")
    assert {"daily_active_users", "availability", "last_updated"} <= overview.keys()
    assert partial["availability"]["growth"] == {
        "available": False,
        "error_code": "temporarily_unavailable",
    }
    assert partial["availability"]["snapshot"]["available"] is True
    assert {"items", "total", "limit", "offset"} == users.keys()
    assert "state" in profile and "activation_milestones" in profile
    assert "next_cursor" in load_fixture("activity_timeline.json")


def test_fixtures_validate_against_committed_pr2_models():
    AnalyticsOverview.model_validate(load_fixture("overview.json"))
    AnalyticsOverview.model_validate(load_fixture("overview_partial_error.json"))
    PaginatedUsers.model_validate(load_fixture("users.json"))
    AnalyticsUserProfile.model_validate(load_fixture("user_detail.json"))
    for name in (
        "activity_timeline.json",
        "activity_runs.json",
        "activity_usage.json",
        "activity_sessions.json",
    ):
        AnalyticsActivityPage.model_validate(load_fixture(name))


def test_byok_fixture_never_reports_atl_cost():
    payload = load_fixture("activity_usage.json")
    byok = next(item for item in payload["items"] if item["billing_mode"] == "byok")
    assert byok["cost_micro_usd"] == 0
    assert byok["amount_micro"] is None


def test_admin_analytics_surface_and_module_exist():
    assert 'id="adminTabAnalytics"' in APP_HTML
    assert 'id="adminPanelAnalytics"' in APP_HTML
    assert 'id="adminAnalyticsOverview"' in APP_HTML
    assert 'id="adminAnalyticsProfile"' in APP_HTML
    assert 'js/admin-analytics.js?v=2' in APP_HTML
    assert ANALYTICS_JS_PATH.exists()
    assert ".admin-analytics-overview" in STYLES
    assert ".admin-analytics-profile" in STYLES


def test_admin_tabs_are_accessible_default_and_url_backed():
    tabs = ADMIN_TABS_JS_PATH.read_text(encoding="utf-8")
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index("</nav>", nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    expected = ["analytics", "users", "providers", "activity"]
    assert nav_markup.count("data-admin-tab=") == 4
    assert [nav_markup.index(f'data-admin-tab="{value}"') for value in expected] == sorted(
        nav_markup.index(f'data-admin-tab="{value}"') for value in expected
    )
    assert "DEFAULT_TAB = 'analytics'" in tabs
    assert "value === 'grant-pool' ? 'users' : value" in tabs
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in tabs
    assert "admin:tabchange" in tabs
    assert "openAccountManagement" in tabs


def test_client_uses_exact_pr2_endpoints_and_query_names():
    source = analytics_source()
    for endpoint in (
        "/api/admin/analytics/overview",
        "/api/admin/analytics/users",
        "/activity",
    ):
        assert endpoint in source
    for query_name in (
        "from", "to", "billing_mode", "provider", "model",
        "include_internal", "q", "status", "last_activity_from",
        "last_activity_to", "sort", "order", "limit", "offset",
        "section", "cursor",
    ):
        assert query_name in source
    assert "start_date" not in source
    assert "params.set('provider_id'" not in source
    assert "params.set('model_id'" not in source


def test_client_owns_url_state_partial_errors_and_independent_sections():
    source = analytics_source()
    for key in (
        "analyticsStart", "analyticsEnd", "analyticsBilling",
        "analyticsProvider", "analyticsModel", "analyticsInternal",
        "analyticsUser", "analyticsSection",
    ):
        assert key in source
    assert "This metric is temporarily unavailable." in source
    assert "This section is temporarily unavailable." in source
    assert "More activity is temporarily unavailable." in source
    assert "Promise.allSettled" in source
    assert "nextCursor" in source
    assert "requestSeq" in source
    assert "URLSearchParams" in source
    assert "history.replaceState" in source
    assert "localStorage" not in source


def test_profile_markup_and_keyboard_contracts_are_present():
    for element_id in (
        "adminAnalyticsProfileBack", "adminAnalyticsProfileTitle",
        "adminAnalyticsOpenAccount", "adminAnalyticsProfileTabs",
        "adminAnalyticsSectionOverview", "adminAnalyticsSectionTimeline",
        "adminAnalyticsSectionRuns", "adminAnalyticsSectionUsage",
        "adminAnalyticsSectionSessions",
    ):
        assert f'id="{element_id}"' in APP_HTML
    source = analytics_source()
    assert "data-analytics-section-tab" in APP_HTML
    assert "aria-selected" in APP_HTML
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in source
    assert "preventDefault()" in source


def test_analytics_is_read_only_and_uses_safe_dom_rendering():
    source = analytics_source()
    assert "innerHTML" not in source
    assert "textContent" in source
    assert "method: 'GET'" in source
    for method in ("POST", "PATCH", "PUT", "DELETE"):
        assert f"method: '{method}'" not in source
    for prohibited in (
        "api_key", "session_id", "network_hash", "provider_response_body",
        "credential_ciphertext", "prompt", "strategy", "portfolio",
    ):
        assert prohibited not in source


def test_app_lifecycle_and_cache_versions_are_wired():
    assert "window.AdminAnalytics.syncAuth(user)" in APP_JS
    assert "window.AdminAnalytics.onEnter()" in APP_JS
    assert "window.AdminAnalytics.refresh()" in APP_JS
    assert 'styles.css?v=130' in APP_HTML
    assert 'app.js?v=125' in APP_HTML
    assert 'js/admin-analytics.js?v=2' in APP_HTML
    assert 'js/admin-tabs.js?v=3' in APP_HTML


def test_credit_costs_use_the_shared_exact_formatter():
    source = analytics_source()
    assert "window.CreditFormat.formatCreditsMicro(value)" in source
    assert "numeric / 1000000" not in source


def test_scoped_responsive_accessible_styles_exist():
    for selector in (
        ".admin-analytics-overview", ".admin-analytics-filters",
        ".admin-analytics-snapshot-grid", ".admin-analytics-trend",
        ".admin-analytics-state-badge", ".admin-analytics-attention-table",
        ".admin-analytics-profile-layout", ".admin-analytics-profile-tabs",
        ".admin-analytics-activity-table", ":focus-visible",
    ):
        assert selector in STYLES
    assert "@media (max-width: 900px)" in STYLES
    assert "@media (max-width: 600px)" in STYLES
    assert "@media (prefers-reduced-motion: reduce)" in STYLES
