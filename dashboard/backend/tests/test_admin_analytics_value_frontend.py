"""Contracts for the Admin user-value Analytics frontend."""

from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, STYLES


ROOT = Path(__file__).resolve().parents[2]
VALUE_JS_PATH = ROOT / "frontend" / "js" / "admin-analytics-value.js"


def value_source() -> str:
    return VALUE_JS_PATH.read_text(encoding="utf-8")


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
