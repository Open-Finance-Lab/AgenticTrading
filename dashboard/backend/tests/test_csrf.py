"""CSRF gate for cookie-authenticated mutating requests."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.csrf import csrf_cookie_name
from dashboard.backend.users import UserStore


@pytest.fixture
def csrf_client(monkeypatch):
    monkeypatch.setenv("ATL_CSRF", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "csrf_auth.db")
        from dashboard.backend import users

        monkeypatch.setattr(users, "user_store", store)
        yield TestClient(app)


def _signup(client: TestClient, email: str = "csrf@example.com") -> None:
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Csrf",
            "password": "securepass1",
        },
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200, resp.text
    assert csrf_cookie_name() in client.cookies


def _csrf_headers(client: TestClient, origin: str = "http://testserver") -> dict:
    token = client.cookies.get(csrf_cookie_name())
    assert token
    return {"Origin": origin, "X-CSRF-Token": token}


def test_cookie_mutating_request_requires_csrf_header(csrf_client):
    _signup(csrf_client)
    blocked = csrf_client.post(
        "/api/auth/logout",
        headers={"Origin": "http://testserver"},
    )
    assert blocked.status_code == 403
    assert "CSRF" in blocked.json()["detail"]

    ok = csrf_client.post("/api/auth/logout", headers=_csrf_headers(csrf_client))
    assert ok.status_code == 200


def test_disallowed_origin_is_rejected(csrf_client):
    _signup(csrf_client)
    headers = _csrf_headers(csrf_client, origin="https://evil.example")
    blocked = csrf_client.post("/api/auth/logout", headers=headers)
    assert blocked.status_code == 403
    assert "Cross-origin" in blocked.json()["detail"]


def test_x_api_key_alone_skips_csrf(csrf_client):
    # No session cookie: a bare X-API-Key must not be CSRF-gated.
    resp = csrf_client.post(
        "/api/v1/agents",
        headers={
            "Origin": "http://testserver",
            "X-API-Key": "not-a-real-agent-key",
            "Content-Type": "application/json",
        },
        json={"name": "nope"},
    )
    # Auth may 401/403 the key; CSRF middleware must not be what fails.
    assert resp.status_code != 403 or "CSRF" not in str(resp.json().get("detail", ""))


def test_forged_api_key_cannot_bypass_csrf_with_session_cookie(csrf_client):
    _signup(csrf_client)
    resp = csrf_client.post(
        "/api/auth/logout",
        headers={
            "Origin": "http://testserver",
            "X-API-Key": "forged-key",
        },
    )
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["detail"]
