"""Static contracts for the demo-aligned acquisition analytics UI."""

from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, STYLES


ROOT = Path(__file__).resolve().parents[2]
VALUE_SOURCE = (ROOT / "frontend/js/admin-analytics-value.js").read_text(encoding="utf-8")


def test_acquisition_markup_is_expanded_with_exact_demo_columns():
    start = APP_HTML.index('id="adminAnalyticsOverview"')
    end = APP_HTML.index('id="adminAnalyticsDeepSections"', start)
    markup = APP_HTML[start:end]
    assert 'aria-expanded="true"' in markup
    for removed_id in ("adminAcquisitionBlocked", "adminAcquisitionGroupBy", "adminPriorityFilters"):
        assert removed_id not in markup
    table_start = markup.index('class="admin-analytics-acquisition-table"')
    table_markup = markup[table_start:table_start + 1800]
    headers = [
        "Group", "Users", "Active", "Task completed", "Repeat",
        "Runs / user", "ATL Credits", "Paid intent", "Paid",
    ]
    assert [table_markup.index(f">{label}<") for label in headers] == sorted(
        table_markup.index(f">{label}<") for label in headers
    )


def test_acquisition_client_has_safe_filters_and_drilldown():
    for key in (
        "analyticsAcquisitionSource",
        "analyticsAcquisitionCohort",
        "analyticsAcquisitionOpen",
        "openAnalyticsUsers",
        "acquisition_task_completed",
        "acquisition_paid_intent",
    ):
        assert key in VALUE_SOURCE
    assert "/api/admin/analytics/acquisition" in VALUE_SOURCE
    assert "date_range" in VALUE_SOURCE
    assert "Promise.allSettled" in VALUE_SOURCE
    assert ".admin-analytics-acquisition-table" in STYLES


def test_acquisition_ui_does_not_render_prohibited_details():
    for prohibited in (
        "invite_token",
        "referral_url",
        "payment_payload",
        "provider_response_body",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        assert prohibited not in VALUE_SOURCE
