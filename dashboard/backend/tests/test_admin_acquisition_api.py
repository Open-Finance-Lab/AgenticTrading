"""Admin acquisition API query and correction contracts."""

from dashboard.backend.tests.test_admin_analytics_api import admin_analytics_api


def test_acquisition_route_parses_group_and_shared_filters(admin_analytics_api):
    api = admin_analytics_api
    response = api["client"].get(
        "/api/admin/analytics/acquisition",
        params={
            "date_range": "1w",
            "group_by": "cohort",
            "acquisition_source": "community",
            "acquisition_cohort": "fall-course",
            "lifecycle": "active",
            "blocked": "false",
            "paid": "true",
        },
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    assert response.json()["groups"][0]["group"]["value"] == "community"
    name, call = api["value_query_service"].calls[-1]
    assert name == "acquisition"
    assert call["group_by"] == "cohort"
    assert call["filters"].source == "community"
    assert call["filters"].cohort == "fall-course"
    assert call["filters"].lifecycle == "active"
    assert call["filters"].blocked is False
    assert call["filters"].paid is True


def test_acquisition_route_rejects_invalid_or_duplicate_filters(admin_analytics_api):
    api = admin_analytics_api
    calls = (
        {"group_by": "campaign"},
        {"acquisition_source": "advertisement"},
        {"acquisition_cohort": "Bad/Slug"},
        [("paid", "true"), ("paid", "false")],
    )
    for params in calls:
        response = api["client"].get(
            "/api/admin/analytics/acquisition",
            params=params,
            headers=api["admin_headers"],
        )
        assert response.status_code == 422, (params, response.text)
        assert response.json() == {"detail": "Invalid Analytics query."}


def test_admin_can_correct_attribution_without_a_reason(admin_analytics_api):
    api = admin_analytics_api
    subject_id = api["subject"]["id"]
    response = api["client"].patch(
        f"/api/admin/analytics/users/{subject_id}/attribution",
        json={"source": "competition", "cohort": "competition-2026"},
        headers=api["admin_headers"],
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "competition"
    assert body["cohort"] == "competition-2026"
    assert body["last_corrected_by_admin_id"] == api["admin"]["id"]
    assert "reason" not in body


def test_attribution_correction_rejects_extra_fields_and_non_admin(admin_analytics_api):
    api = admin_analytics_api
    path = f"/api/admin/analytics/users/{api['subject']['id']}/attribution"
    extra = api["client"].patch(
        path,
        json={"source": "friend", "cohort": None, "reason": "not collected"},
        headers=api["admin_headers"],
    )
    outsider = api["client"].patch(
        path,
        json={"source": "friend", "cohort": None},
        headers=api["outsider_headers"],
    )

    assert extra.status_code == 422
    assert outsider.status_code == 403
