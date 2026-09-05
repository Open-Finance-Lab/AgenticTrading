"""Contracts for the Admin user-value Analytics frontend."""

from pathlib import Path

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    STYLES,
    fn_body,
    strip_comments,
)


ROOT = Path(__file__).resolve().parents[2]
VALUE_JS_PATH = ROOT / "frontend" / "js" / "admin-analytics-value.js"
PROFILE_JS_PATH = ROOT / "frontend" / "js" / "admin-analytics.js"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"


def value_source() -> str:
    return VALUE_JS_PATH.read_text(encoding="utf-8")


def profile_source() -> str:
    return PROFILE_JS_PATH.read_text(encoding="utf-8")


def test_value_client_uses_independent_endpoints():
    source = value_source()
    for endpoint in (
        "/lifecycle",
        "/retention",
        "/commercial",
        "/operational",
        "/users",
    ):
        assert endpoint in source
    assert "Promise.allSettled" in source


def test_deep_sections_fetch_only_on_first_open():
    source = value_source()
    assert "loaded: false" in source
    assert "aria-expanded" in source
    assert "ensureDisclosureLoaded" in source
    assert "This section is temporarily unavailable." in source
    assert 'data-admin-value-disclosure="retention"' in APP_HTML
    assert 'data-admin-value-disclosure="commercial"' in APP_HTML
    assert 'data-admin-value-disclosure="operational"' in APP_HTML


def test_value_filters_are_deep_linkable():
    source = value_source()
    for key in (
        "analyticsLifecycle",
        "analyticsOperational",
        "analyticsCommercial",
        "analyticsUser",
        "analyticsProfile",
    ):
        assert key in source
    assert "analyticsPanel" in source
    assert "history.replaceState" in source


def test_value_overview_has_stable_semantic_regions():
    for element_id in (
        "adminAnalyticsValueOverview",
        "adminAnalyticsValueTitle",
        "adminAnalyticsHeadline",
        "adminLifecycleDistribution",
        "adminLifecycleMovementChart",
        "adminLifecycleMovementTable",
        "adminPriorityUsers",
        "adminRetentionPanel",
        "adminCommercialPanel",
        "adminOperationalPanel",
    ):
        assert f'id="{element_id}"' in APP_HTML
    assert 'aria-describedby="adminLifecycleMovementTable"' in APP_HTML


def test_value_client_handles_access_loss_partial_errors_and_stale_data():
    source = value_source()
    assert "handleAccessLost" in source
    assert "error?.status !== 401" in source
    assert "error?.status !== 403" in source
    assert "keepStaleData" in source
    assert "Incomplete data" in source
    assert "Retry section" in APP_HTML


def test_value_rendering_is_safe_and_accessible():
    source = value_source()
    assert "innerHTML" not in source
    assert "textContent" in source
    assert "method: 'GET'" in source
    assert "aria-pressed" in source
    assert "window.Chart" in source
    assert ".admin-value-overview" in STYLES
    assert ".admin-value-disclosure" in STYLES


def test_rules_and_evidence_dialogs_are_named_and_focus_safe():
    assert 'id="adminAnalyticsRulesDialog"' in APP_HTML
    assert 'id="adminAnalyticsEvidenceDialog"' in APP_HTML
    assert 'aria-labelledby="adminAnalyticsRulesTitle"' in APP_HTML
    assert 'aria-labelledby="adminAnalyticsEvidenceTitle"' in APP_HTML
    source = value_source()
    for contract in (
        "showModal()",
        "Escape",
        "event.target === dialog",
        "returnFocus",
        "focus()",
    ):
        assert contract in source


def test_priority_signals_use_fixed_rules_and_display_safe_evidence():
    source = value_source()
    assert "LIFECYCLE_RULES" in source
    for segment in ("new", "onboarding", "growing", "core", "at_risk", "dormant"):
        assert f"{segment}:" in source
    assert "lifecycle?.evidence" in source
    assert "operational?.evidence" in source
    assert "Open full analytics profile" in APP_HTML
    assert "openAccountManagement" in source


def test_profile_keeps_full_sections_and_renders_value_axes():
    source = profile_source()
    for section in ("overview", "timeline", "runs", "usage", "sessions"):
        assert section in source
    assert "openAccountManagement" in source
    assert "profile.lifecycle" in source
    assert "profile.operational" in source
    assert "profile.commercial" in source
    assert "recent_lifecycle_transitions" in source
    assert "activeFilters" in source
    assert "formatExclusiveDateOnly" in source
    assert "getRange" in value_source()
    for element_id in (
        "adminAnalyticsProfileLifecycle",
        "adminAnalyticsProfileOperational",
        "adminAnalyticsProfileCommercial",
        "adminAnalyticsProfileValueFacts",
        "adminAnalyticsLifecycleEvidence",
        "adminAnalyticsOperationalEvidence",
        "adminAnalyticsLifecycleTransitions",
    ):
        assert f'id="{element_id}"' in APP_HTML


def test_value_fixtures_and_client_exclude_sensitive_fields():
    fixtures = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FIXTURE_DIR.glob("*.json"))
    )
    combined = f"{value_source()}\n{fixtures}"
    for prohibited in (
        "api_key",
        "password",
        "network_hash",
        "raw_user_agent",
        "provider_response_body",
        "credential_ciphertext",
        "strategy_content",
        "prompt_text",
    ):
        assert prohibited not in combined


def test_one_failed_section_does_not_blank_other_sections():
    source = value_source()
    assert "Promise.allSettled" in source
    assert "applySettledSection" in source
    assert "keepStaleData: true" in source
    assert "section.stale" in source


def test_charts_disclosures_and_controls_have_semantic_state():
    assert 'aria-describedby="adminLifecycleMovementTable"' in APP_HTML
    table_start = APP_HTML.index('id="adminLifecycleMovementTable"')
    assert 'class="sr-only"' in APP_HTML[table_start - 80 : table_start + 120]
    assert 'aria-expanded="false"' in APP_HTML
    for control_id in (
        "adminValueStart",
        "adminValueEnd",
        "adminPriorityQuery",
        "adminOperationalProvider",
        "adminOperationalModel",
    ):
        start = APP_HTML.index(f'id="{control_id}"')
        fragment = APP_HTML[start : start + 220]
        assert "name=" in fragment
        assert "autocomplete=" in fragment


def test_movement_ranges_and_profile_navigation_are_discoverable():
    source = value_source()
    for movement_range in ("5d", "1w", "1m", "1y"):
        assert f'data-movement-range="{movement_range}"' in APP_HTML
    assert "analyticsMovementRange" in source
    assert "movement_granularity" in source
    assert "admin-priority-profile-link" in source
    assert "admin-help-btn" in APP_HTML
    assert 'aria-label="How segments work"' in APP_HTML
    assert 'id="adminAnalyticsProfileBreadcrumbParent"' in APP_HTML
    header_start = APP_HTML.index('class="admin-value-header"')
    identity_start = APP_HTML.index('id="adminLifecycleDistributionTitle"')
    assert 'id="adminAnalyticsRulesOpen"' not in APP_HTML[header_start:identity_start]
    assert 'id="adminAnalyticsRulesOpen"' in APP_HTML[identity_start:identity_start + 700]


def test_value_formatting_uses_intl_and_dialogs_bound_scroll():
    source = value_source()
    assert "Intl.NumberFormat" in source
    assert "Intl.DateTimeFormat" in source
    assert "value == null" in source
    assert "Not mature" in source
    assert "toFixed(" not in source
    assert "overscroll-behavior: contain" in STYLES
    assert "touch-action: manipulation" in STYLES
    assert "table.sr-only" in STYLES


def test_profile_url_param_has_a_single_owner():
    """`analyticsProfile` is written by the profile controller, nowhere else.

    The value module used to keep its own copy in `state.userFilters` and write
    it back from `writeUrlState`, so the two disagreed the moment a profile was
    closed: the id survived in memory, and the next filter edit put the closed
    profile back into the URL.
    """
    value = strip_comments(value_source())
    assert "state.userFilters.profile" not in value
    assert "URL_KEYS.profile, state.userFilters.profile" not in value
    assert "URL_KEYS.profile" in value, "the deep-link guard still reads the key"
    assert "'analyticsProfile'" in strip_comments(profile_source())


def test_overview_deep_link_guard_reads_the_live_url():
    """The guard that suppresses the overview fetch must not read a cached id.

    Anchored on the URL because that is the state `closeProfile` clears; a
    cached copy left the overview permanently blank behind a closed profile.
    """
    body = strip_comments(fn_body("function onEnter(", value_source()))
    assert "searchParams" in body
    assert "state.userFilters" not in body


def test_movement_range_change_leaves_priority_users_alone():
    """The range switch drives the movement chart, not the priority table."""
    body = strip_comments(fn_body("function setMovementRange(", value_source()))
    assert "refreshLifecycle()" in body
    assert "refreshPrimary()" not in body


def test_movement_range_switch_carries_radio_semantics():
    """Roving tabindex needs a role that gives arrow keys a meaning.

    `role="group"` does not, so only the selected button was reachable by Tab
    and nothing announced that arrows moved between them.
    """
    start = APP_HTML.index('id="adminLifecycleMovementRanges"')
    end = APP_HTML.index("</div>", start)
    switch = APP_HTML[start:end]
    assert 'role="radiogroup"' in switch
    assert switch.count('role="radio"') == 4
    assert switch.count("aria-checked=") == 4
    assert "aria-pressed" not in switch
    assert "aria-checked" in strip_comments(value_source())
