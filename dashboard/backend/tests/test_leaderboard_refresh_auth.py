"""GET /api/v1/leaderboard?refresh=true is an operator action, not a public one.

A forced refresh recomputes every baseline and refetches market data; honoured
for anyone, it let an anonymous caller burn the platform's data quota and
append agent_runs rows on every request.
"""

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.routers.leaderboard as leaderboard_router
from dashboard.backend.app import app


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_get_leaderboard(*, force_refresh, period):
        seen.append(force_refresh)
        return {"period": period, "entries": []}

    monkeypatch.setattr(leaderboard_router, "get_leaderboard", fake_get_leaderboard)
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "s" * 40)
    return seen


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_anonymous_refresh_is_refused_before_any_recompute(client, calls):
    resp = client.get("/api/v1/leaderboard", params={"refresh": "true"})
    assert resp.status_code == 403
    assert calls == []


def test_wrong_secret_is_refused(client, calls):
    resp = client.get(
        "/api/v1/leaderboard",
        params={"refresh": "true"},
        headers={"X-Leaderboard-Refresh-Secret": "nope"},
    )
    assert resp.status_code == 403
    assert calls == []


def test_refresh_secret_forces_a_refresh(client, calls):
    resp = client.get(
        "/api/v1/leaderboard",
        params={"refresh": "true"},
        headers={"X-Leaderboard-Refresh-Secret": "s" * 40},
    )
    assert resp.status_code == 200
    assert calls == [True]


@pytest.mark.parametrize("role, expected", [("admin", 200), ("user", 403)])
def test_only_an_admin_session_forces_a_refresh(client, calls, monkeypatch, role, expected):
    import dashboard.backend.api.auth as auth

    monkeypatch.setattr(auth, "get_current_user", lambda request, authorization=None: {"id": 1, "role": role})
    resp = client.get("/api/v1/leaderboard", params={"refresh": "true"})
    assert resp.status_code == expected
    assert calls == ([True] if expected == 200 else [])


def test_plain_read_stays_public(client, calls):
    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 200
    assert calls == [False]
