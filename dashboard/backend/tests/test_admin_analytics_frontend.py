"""Fixture contracts for the admin analytics API, plus the console's cache-buster pins."""

import json
from pathlib import Path

from dashboard.backend.domain.analytics.query_service import (
    AnalyticsActivityPage,
    AnalyticsOverview,
)
from dashboard.backend.domain.analytics.value_queries import (
    CommercialAnalyticsResponse,
    GroupAnalyticsResponse,
    LifecycleAnalyticsResponse,
    OperationalAnalyticsResponse,
    PaginatedValueUsers,
    RetentionAnalyticsResponse,
    ValueUserProfile,
)
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, STYLES


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")


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


def test_safe_fixtures_have_no_prohibited_response_fields():
    prohibited = {
        "api_key", "auth_token", "password", "verification_code",
        "prompt", "instruction", "strategy", "portfolio", "form_value",
        "provider_response_body", "ip_address", "user_agent",
        "credential_ciphertext", "network_hash", "session_id",
    }
    for path in sorted(FIXTURES.rglob("*.json")):  # committed and target/ alike
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert prohibited.isdisjoint(set(walk_keys(payload))), str(path.relative_to(FIXTURES))


def test_fixtures_match_committed_analytics_shapes():
    overview = load_fixture("overview.json")
    partial = load_fixture("overview_partial_error.json")
    lifecycle = load_fixture("lifecycle.json")
    retention = load_fixture("retention.json")
    commercial = load_fixture("commercial.json")
    operational = load_fixture("operational.json")
    users = load_fixture("users.json")
    profile = load_fixture("user_detail.json")
    # Today's shapes (D18): the §9 fields live in fixtures/admin_analytics/target/ until PR D
    # folds them in and restores them to these assertions.
    assert {"daily_active_users", "availability", "last_updated"} <= overview.keys()
    assert partial["availability"]["growth"] == {
        "available": False,
        "error_code": "temporarily_unavailable",
    }
    assert partial["availability"]["snapshot"]["available"] is True
    assert {"headline", "segment_counts", "weekly_segments", "transitions"} <= lifecycle.keys()
    assert {"cohorts", "summary_week_1", "summary_week_2", "summary_week_4"} <= retention.keys()
    assert {"tier_counts", "selected_period", "current_balances"} <= commercial.keys()
    assert {"operational_state_counts", "top_failure_categories"} <= operational.keys()
    assert {"items", "total", "limit", "offset"} == users.keys()
    assert {"lifecycle", "operational", "commercial_tier", "priority_group"} <= users["items"][0].keys()
    assert {"state", "activation_milestones", "lifecycle", "operational", "commercial"} <= profile.keys()
    assert "next_cursor" in load_fixture("activity_timeline.json")
    assert {"users", "agents", "active_dashboard_backtests", "max_active_dashboard_backtests"} <= load_fixture("admin_stats.json").keys()


def test_fixtures_validate_against_committed_analytics_models():
    AnalyticsOverview.model_validate(load_fixture("overview.json"))
    AnalyticsOverview.model_validate(load_fixture("overview_partial_error.json"))
    LifecycleAnalyticsResponse.model_validate(load_fixture("lifecycle.json"))
    RetentionAnalyticsResponse.model_validate(load_fixture("retention.json"))
    CommercialAnalyticsResponse.model_validate(load_fixture("commercial.json"))
    OperationalAnalyticsResponse.model_validate(load_fixture("operational.json"))
    GroupAnalyticsResponse.model_validate(load_fixture("groups.json"))
    PaginatedValueUsers.model_validate(load_fixture("users.json"))
    ValueUserProfile.model_validate(load_fixture("user_detail.json"))
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


def test_in_app_analytics_surface_is_gone():
    """PR C (design §7.5): one analytics surface, at /admin."""
    for marker in (
        'id="adminTabAnalytics"', 'id="adminPanelAnalytics"', 'id="adminAnalyticsOverview"',
        'id="adminAnalyticsProfile"', 'id="adminAnalyticsRulesDialog"', 'id="adminAnalyticsEvidenceDialog"',
        "admin-analytics-legacy-overview", "js/admin-analytics.js", "js/admin-analytics-value.js",
    ):
        assert marker not in APP_HTML, marker
    assert not (FRONTEND / "admin-analytics.html").exists()
    assert not (FRONTEND / "js" / "admin-analytics.js").exists()
    assert not (FRONTEND / "js" / "admin-analytics-value.js").exists()
    for call in (
        "window.AdminAnalytics.syncAuth(user)", "window.AdminAnalytics.onEnter()",
        "window.AdminAnalytics.refresh()", "window.AdminAnalyticsValue.syncAuth(user)",
        "window.AdminAnalyticsValue.onEnter()",
    ):
        assert call not in APP_JS, call


def test_admin_rail_has_two_tabs_defaulting_to_account_management():
    """Providers moved to /admin in the 09-17 consolidation (design N2); Account
    Management and Activity are one module and port together in PR2."""
    admin_start = APP_HTML.index('id="adminView"')
    nav_start = APP_HTML.index('<nav id="adminTabs"', admin_start)
    nav_end = APP_HTML.index("</nav>", nav_start)
    nav_markup = APP_HTML[nav_start:nav_end]
    expected = ["users", "activity"]
    # Assert the set *and* its order in one go. This was a bare assignment that
    # nothing read (CodeQL py/unused-local-variable), so the tab names it names
    # were pinned by nothing -- the count below would have passed just as
    # happily on two tabs called something else entirely.
    assert [chunk.split('"')[0] for chunk in nav_markup.split('data-admin-tab="')[1:]] == expected
    assert nav_markup.count("data-admin-tab=") == 2
    assert nav_markup.index('data-admin-tab="users"') < nav_markup.index('data-admin-tab="activity"')
    assert 'data-admin-tab="providers"' not in nav_markup
    assert 'aria-orientation="vertical"' in nav_markup
    assert 'id="adminTabUsers" class="admin-tab is-active"' in nav_markup
    # N6: label-only. The tab id, the data attribute, the panel id and the
    # query value are all still "users", so no URL breaks and admin-tabs.js
    # needs no new normalisation.
    assert "<span>Account Management</span>" in nav_markup
    assert "<span>Users</span>" not in nav_markup
    assert 'aria-label="Account Management"' in nav_markup
    assert '<section id="adminPanelUsers" class="admin-tab-panel" role="tabpanel" aria-labelledby="adminTabUsers" data-admin-panel="users">' in APP_HTML


def test_profile_menu_admin_entry_opens_the_admin_page():
    start = APP_JS.index("document.getElementById('accountMenuAdminBtn')?.addEventListener('click'")
    handler = APP_JS[start:start + 600]
    assert "window.location.assign('/admin')" in handler
    assert "navigateToPage('admin')" not in handler


def test_app_lifecycle_and_cache_versions_are_wired():
    # Lockstep owner for the console's bumped tags and the /admin page's pins:
    # every bump edits this test in the same change (Global Constraints).
    assert 'styles.css?v=143' in APP_HTML
    assert 'app.js?v=135' in APP_HTML
    assert 'js/admin-tabs.js?v=12' in APP_HTML
    for tag in (
        'href="admin.css?v=4"',
        'src="js/admin-shell.js?v=5"',
        'src="js/credit-format.js?v=1"',
        'src="js/admin-live.js?v=1"',
        'src="js/admin-overview.js?v=1"',
        'src="js/admin-users.js?v=1"',
        'src="js/admin-providers.js?v=1"',
    ):
        assert tag in ADMIN_HTML, tag


def test_analytics_style_families_left_styles_css():
    for family in (
        ".admin-analytics-", ".admin-value-", ".admin-priority-", ".admin-lifecycle-",
        ".admin-group-", ".admin-help-btn", ".admin-profile-evidence-grid", ".admin-commercial-tier-grid",
    ):
        assert family not in STYLES, family
    for kept in (".admin-workspace", ".admin-rail", ".admin-tab:focus-visible", ".admin-rail button"):
        assert kept in STYLES, kept


def test_the_provider_families_left_styles_css_with_their_markup():
    """Dead CSS for markup that no longer exists is how a 14,000-line stylesheet
    is grown. The families move to admin.css in the same PR as the panel."""
    for family in (".admin-provider-", ".admin-platform-key-"):
        assert family not in STYLES, family
    for kept in (".admin-workspace", ".admin-rail", ".admin-tab:focus-visible", ".admin-rail button"):
        assert kept in STYLES, kept


def test_the_old_provider_module_is_gone_from_the_page_and_the_tree():
    assert "admin-model-providers" not in APP_HTML
    assert "AdminModelProviders" not in APP_JS
    assert not (FRONTEND / "js" / "admin-model-providers.js").exists()
