"""api/auth.py maps a user-store connectivity failure to 503, not a bare 500.

The September 2026 outage's failure mode: the users/content Postgres project
returned errors for ~5 hours (Neon free-tier egress exhausted by analytics
maintenance -- see the burner-kill plan's Tasks 1-3), and every auth call --
signup, login, the /me boot probe, and get_current_user's own dependency
lookup -- propagated a raw psycopg.OperationalError (or, on the SQLite twin,
sqlite3.OperationalError) as an unmapped 500 with no server-side log line.
This pins the fix at all four call sites: each maps the failure to a 503 the
caller can retry, and prints one line naming the route and the exception
category.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import PoolTimeout

from dashboard.backend.app import app
from dashboard.backend.users import UserStore


@pytest.fixture
def temp_user_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "auth_unavailable_test.db")
        yield store


@pytest.fixture
def client(temp_user_store, monkeypatch):
    from dashboard.backend import users
    from dashboard.backend.api import auth
    from dashboard.backend.domain.credits.repository import CreditsStore
    from dashboard.backend.domain.credits.service import CreditsService

    monkeypatch.setattr(users, "user_store", temp_user_store)
    monkeypatch.setattr(
        auth,
        "credits_service",
        CreditsService(store=CreditsStore(temp_user_store.db_path)),
    )
    return TestClient(app)


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


def test_pool_timeout_is_covered_by_the_operational_error_arm():
    """One arm, because PoolTimeout is an OperationalError subclass."""
    from dashboard.backend.api import auth

    assert issubclass(PoolTimeout, psycopg.OperationalError)
    assert psycopg.OperationalError in auth._USER_STORE_OUTAGE
    assert sqlite3.OperationalError in auth._USER_STORE_OUTAGE


def test_get_current_user_maps_a_sqlite_operational_error(client, temp_user_store, monkeypatch, capsys):
    signup = client.post(
        "/api/auth/signup",
        json={"email": "gcu@example.com", "display_name": "GCU", "password": "securepass1"},
    )
    assert signup.status_code == 200

    monkeypatch.setattr(
        temp_user_store,
        "get_user_for_token",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/auth/me")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert (
        "ERROR: auth.user_store_unavailable route=get_current_user category=OperationalError"
        in output
    )


def test_login_maps_a_psycopg_operational_error(client, temp_user_store, monkeypatch, capsys):
    monkeypatch.setattr(
        temp_user_store,
        "authenticate",
        _raiser(psycopg.OperationalError("connection refused")),
    )

    response = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "whatever1"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=login category=OperationalError" in output


def test_signup_maps_a_pool_timeout(client, temp_user_store, monkeypatch, capsys):
    monkeypatch.setattr(
        temp_user_store,
        "create_user",
        _raiser(PoolTimeout("couldn't get a connection")),
    )

    response = client.post(
        "/api/auth/signup",
        json={
            "email": "signup-pool@example.com",
            "display_name": "Signup Pool",
            "password": "securepass1",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=signup category=PoolTimeout" in output


def test_me_maps_a_sqlite_operational_error_from_get_entitlements(client, temp_user_store, monkeypatch, capsys):
    from dashboard.backend import users as users_module

    signup = client.post(
        "/api/auth/signup",
        json={
            "email": "me-entitlements@example.com",
            "display_name": "Me Entitlements",
            "password": "securepass1",
        },
    )
    assert signup.status_code == 200

    monkeypatch.setattr(users_module, "entitlements_from_session_row", lambda *a, **k: None)
    monkeypatch.setattr(
        temp_user_store,
        "get_entitlements",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/auth/me")

    assert response.status_code == 503
    assert response.json()["detail"] == "Service temporarily unavailable"
    output = capsys.readouterr().out
    assert "ERROR: auth.user_store_unavailable route=me category=OperationalError" in output


def test_health_is_unaffected_by_a_user_store_outage(client, temp_user_store, monkeypatch):
    monkeypatch.setattr(
        temp_user_store,
        "get_user_for_token",
        _raiser(sqlite3.OperationalError("database is locked")),
    )

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
