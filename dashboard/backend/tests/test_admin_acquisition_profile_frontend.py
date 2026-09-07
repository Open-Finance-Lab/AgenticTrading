"""Static contracts for acquisition attribution in User Analytics Profile."""

from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, STYLES


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "frontend/js/admin-analytics.js").read_text(encoding="utf-8")


def test_profile_has_attribution_projection_and_editor():
    for element_id in (
        "adminAnalyticsProfileSource",
        "adminAnalyticsProfileCohort",
        "adminAnalyticsProfileAttributionMethod",
        "adminAnalyticsProfileOriginalSource",
        "adminAnalyticsProfileAttributionSourceInput",
        "adminAnalyticsProfileAttributionCohortInput",
        "adminAnalyticsProfileAttributionSave",
        "adminAnalyticsProfileAttributionStatus",
    ):
        assert f'id="{element_id}"' in APP_HTML
    assert 'aria-live="polite"' in APP_HTML
    assert ".admin-profile-attribution" in STYLES


def test_profile_patch_sends_only_source_and_cohort():
    assert "updateAttribution" in SOURCE
    assert "method: 'PATCH'" in SOURCE
    assert "JSON.stringify({ source, cohort })" in SOURCE
    assert "correction reason" not in APP_HTML.lower()
    assert "invite_token" not in SOURCE
    assert "referral_url" not in SOURCE
