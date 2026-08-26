"""Authenticated, non-echoing Analytics ingestion API tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from dashboard.backend import users as users_module
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.api.routers import analytics as analytics_router
from dashboard.backend.app import app
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)
from dashboard.backend.users import UserStore


def frontend_payload(**overrides):
    value = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "event_name": "page_viewed",
        "session_id": str(uuid4()),
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "page_view": "home",
        "properties": {},
    }
    value.update(overrides)
    return value


@pytest.fixture
def analytics_api(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "analytics_api.db"
        user_store = UserStore(db_path=db_path)
        event_store = AnalyticsStore(db_path=db_path)
        service = AnalyticsService(store=event_store)
        monkeypatch.setattr(users_module, "user_store", user_store)
        app.dependency_overrides[get_analytics_service] = lambda: service
        analytics_router.reset_analytics_ingestion_limiter()
        with TestClient(app) as client:
            yield client, user_store, event_store
        analytics_router.reset_analytics_ingestion_limiter()
        app.dependency_overrides.pop(get_analytics_service, None)


def _signup(client: TestClient, email="analytics@example.test") -> dict:
    response = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Analytics User",
            "password": "SecurePass1!",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["user"]


def test_analytics_ingestion_requires_authentication(analytics_api):
    client, _users, _events = analytics_api
    response = client.post("/api/analytics/events", json=frontend_payload())
    assert response.status_code == 401


def test_analytics_ingestion_returns_generic_acceptance(analytics_api):
    client, _users, event_store = analytics_api
    user = _signup(client)
    response = client.post("/api/analytics/events", json=frontend_payload())
    assert response.status_code == 202
    assert response.json() == {"accepted": True}
    saved = event_store.list_user_events(user["id"])["items"]
    assert len(saved) == 1
    assert saved[0].user_id == user["id"]


def test_analytics_validation_never_echoes_secret_canary(
    analytics_api,
    capsys,
):
    client, _users, _events = analytics_api
    _signup(client)
    payload = frontend_payload()
    payload["api_key"] = "synthetic-secret-canary"
    response = client.post("/api/analytics/events", json=payload)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid analytics event."}
    assert "synthetic-secret-canary" not in response.text
    assert "synthetic-secret-canary" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("body", "expected_status"),
    [
        (b"not-json", 422),
        (json.dumps(["not", "an", "object"]).encode(), 422),
        (json.dumps({"x": "y" * 9000}).encode(), 413),
    ],
)
def test_analytics_body_is_bounded_and_generic(
    analytics_api,
    body,
    expected_status,
):
    client, _users, _events = analytics_api
    _signup(client)
    response = client.post(
        "/api/analytics/events",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == expected_status
    assert "y" * 100 not in response.text


def test_analytics_declared_body_size_is_bounded(analytics_api):
    client, _users, _events = analytics_api
    _signup(client)
    response = client.post(
        "/api/analytics/events",
        content=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": "9000",
        },
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Analytics event is too large."}


def test_analytics_ingestion_is_rate_limited_per_authenticated_user(
    analytics_api,
    monkeypatch,
):
    client, _users, _events = analytics_api
    monkeypatch.setattr(
        analytics_router,
        "_ANALYTICS_INGESTION_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=300),
    )
    _signup(client, "first@example.test")
    assert client.post("/api/analytics/events", json=frontend_payload()).status_code == 202
    assert client.post("/api/analytics/events", json=frontend_payload()).status_code == 202
    blocked = client.post("/api/analytics/events", json=frontend_payload())
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1

    _signup(client, "second@example.test")
    allowed = client.post("/api/analytics/events", json=frontend_payload())
    assert allowed.status_code == 202


def test_analytics_store_failure_returns_generic_503(analytics_api, capsys):
    client, _users, _events = analytics_api
    _signup(client)

    class FailingService:
        def accept_frontend_event(self, **kwargs):
            raise RuntimeError("synthetic-secret-canary upstream body")

    app.dependency_overrides[get_analytics_service] = lambda: FailingService()
    response = client.post("/api/analytics/events", json=frontend_payload())
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Analytics is temporarily unavailable."
    }
    assert "synthetic-secret-canary" not in response.text
    assert "synthetic-secret-canary" not in capsys.readouterr().out


def test_analytics_idempotency_conflict_is_generic(analytics_api):
    client, _users, _events = analytics_api
    _signup(client)
    payload = frontend_payload()
    assert client.post("/api/analytics/events", json=payload).status_code == 202
    payload["page_view"] = "credits"
    conflict = client.post("/api/analytics/events", json=payload)
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": "Analytics event conflicts with a replay."}


def test_analytics_route_is_registered_once():
    matches = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/analytics/events"
        and "POST" in getattr(route, "methods", set())
    ]
    assert len(matches) == 1
