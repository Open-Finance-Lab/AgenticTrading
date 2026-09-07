"""Static contracts for Admin acquisition analytics UI."""

import json
from pathlib import Path

from dashboard.backend.domain.analytics.value_queries import (
    AcquisitionAnalyticsResponse,
)
from dashboard.backend.tests._frontend_source import APP_HTML, STYLES


ROOT = Path(__file__).resolve().parents[2]
VALUE_SOURCE = (ROOT / "frontend/js/admin-analytics-value.js").read_text(
    encoding="utf-8"
)
FIXTURES = Path(__file__).resolve().parent / "fixtures/admin_analytics"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def test_acquisition_fixtures_validate_and_exclude_token_metrics():
    for name in ("acquisition.json", "acquisition_partial_error.json"):
        payload = _fixture(name)
        AcquisitionAnalyticsResponse.model_validate(payload)
        assert {"input_tokens", "output_tokens", "total_tokens"}.isdisjoint(
            set(_walk_keys(payload))
        )


def test_acquisition_markup_has_filters_disclosure_and_semantic_table_target():
    for element_id in (
        "adminAcquisitionSource",
        "adminAcquisitionCohort",
        "adminAcquisitionLifecycle",
        "adminAcquisitionBlocked",
        "adminAcquisitionPaid",
        "adminAcquisitionGroupBy",
        "adminAcquisitionToggle",
        "adminAcquisitionGroups",
    ):
        assert f'id="{element_id}"' in APP_HTML
    assert 'aria-expanded="false"' in APP_HTML
    assert 'aria-controls="adminAcquisitionPanel"' in APP_HTML
    assert "Acquisition groups by source or operating cohort" in APP_HTML


def test_acquisition_client_has_url_state_independent_request_and_drilldown():
    for key in (
        "analyticsAcquisitionSource",
        "analyticsAcquisitionCohort",
        "analyticsAcquisitionGroupBy",
        "analyticsAcquisitionOpen",
    ):
        assert key in VALUE_SOURCE
    assert "/api/admin/analytics/acquisition" in VALUE_SOURCE
    assert "analyticsRange" in VALUE_SOURCE
    assert "date_range" in VALUE_SOURCE
    assert "purchase_window_complete" in VALUE_SOURCE
    assert "acquisition_task_completed" in VALUE_SOURCE
    assert "acquisition_paid_intent" in VALUE_SOURCE
    assert "Promise.allSettled" in VALUE_SOURCE
    assert ".admin-acquisition-table" in STYLES


def test_acquisition_ui_does_not_render_prohibited_details():
    combined = VALUE_SOURCE + json.dumps(_fixture("acquisition.json"))
    for prohibited in (
        "invite_token",
        "referral_url",
        "payment_payload",
        "provider_response_body",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        assert prohibited not in combined
