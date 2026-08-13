"""Admin users API + entitlements store behaviour."""

import tempfile
from pathlib import Path

import pytest

from dashboard.backend.users import UserStore, user_store


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield UserStore(db_path=Path(tmpdir) / "users.db")


def test_entitlements_default_then_upsert(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    defaults = store.get_entitlements(user["id"])
    assert defaults["max_concurrent_backtests"] == 1
    assert defaults["credits"] == 0

    updated = store.set_entitlements(
        user["id"],
        max_concurrent_backtests=5,
        credits=100,
        updated_by_admin_id=user["id"],
    )
    assert updated["max_concurrent_backtests"] == 5
    assert updated["credits"] == 100
    assert updated["updated_by_admin_id"] == user["id"]


def test_set_user_role_and_list_admin(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    store.set_user_role(user["id"], "admin")
    store.set_entitlements(user["id"], max_concurrent_backtests=5, credits=10)

    listed = store.list_users_admin()
    assert len(listed) == 1
    assert listed[0]["role"] == "admin"
    assert listed[0]["entitlements"]["max_concurrent_backtests"] == 5


def test_cannot_demote_last_admin(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    store.set_user_role(user["id"], "admin")
    with pytest.raises(ValueError, match="last_admin"):
        store.set_user_role(user["id"], "user")


def _signup(client, email="admin@example.com"):
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Admin",
            "password": "SecurePass1!",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["user"]


def test_admin_users_requires_admin():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    user = _signup(client, "plain@example.com")
    resp = client.get("/api/admin/users")
    assert resp.status_code == 403

    user_store.set_user_role(user["id"], "admin")
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["role"] == "admin"
    assert "entitlements" in me.json()["user"]

    listed = client.get("/api/admin/users")
    assert listed.status_code == 200
    emails = {row["email"] for row in listed.json()["users"]}
    assert "plain@example.com" in emails


def test_admin_cannot_demote_self():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    admin = _signup(client, "selfadmin@example.com")
    user_store.set_user_role(admin["id"], "admin")
    # Second admin so last_admin is not the reason for refusal.
    other = _signup(client, "otheradmin@example.com")
    user_store.set_user_role(other["id"], "admin")

    client.post(
        "/api/auth/login",
        json={"email": "selfadmin@example.com", "password": "SecurePass1!"},
    )
    resp = client.patch(
        f"/api/admin/users/{admin['id']}",
        json={"role": "user"},
    )
    assert resp.status_code == 400, resp.text
    assert "yourself" in resp.json()["detail"].lower()
    assert user_store.get_user_by_id(admin["id"])["role"] == "admin"


def test_admin_stats_endpoint():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    from dashboard.backend.domain.agents.repository import agent_store

    client = TestClient(app)
    admin = _signup(client, "stats-admin@example.com")
    user_store.set_user_role(admin["id"], "admin")
    _signup(client, "stats-user@example.com")
    agent_store.create_agent(
        name="stats-agent",
        description="for admin stats",
        owner_user_id=admin["id"],
    )

    client.post(
        "/api/auth/login",
        json={"email": "stats-admin@example.com", "password": "SecurePass1!"},
    )
    resp = client.get("/api/admin/stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["users"] >= 2
    assert body["admins"] >= 1
    assert body["agents"] >= 1
    assert "active_dashboard_backtests" in body


def test_admin_patch_entitlements_and_role():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    admin = _signup(client, "boss@example.com")
    user_store.set_user_role(admin["id"], "admin")
    target = _signup(client, "member@example.com")

    client.post(
        "/api/auth/login",
        json={"email": "boss@example.com", "password": "SecurePass1!"},
    )

    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"role": "user", "max_concurrent_backtests": 3, "credits": 50},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()["user"]
    assert body["entitlements"]["max_concurrent_backtests"] == 3
    assert body["entitlements"]["credits"] == 50


def test_promote_first_admin_then_refuse(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    other = store.create_user("b@example.com", "B", "securepass1")
    promoted = store.promote_first_admin(user["id"])
    assert promoted["role"] == "admin"
    with pytest.raises(ValueError, match="admin_exists"):
        store.promote_first_admin(other["id"])
    with pytest.raises(ValueError, match="admin_exists"):
        store.promote_first_admin(user["id"])


def test_secrets_equal_accepts_match_and_rejects_length_mismatch():
    from dashboard.backend.api.routers.admin_users import secrets_equal

    assert secrets_equal("same-secret", "same-secret") is True
    assert secrets_equal("short-ok", "a-much-longer-secret") is False
    assert secrets_equal("wrong-secret", "right-secret") is False


def test_admin_users_unauthenticated_is_401():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    resp = client.get("/api/admin/users")
    assert resp.status_code == 401


def test_admin_patch_rejects_out_of_range_quotas():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    admin = _signup(client, "quota-admin@example.com")
    user_store.set_user_role(admin["id"], "admin")
    target = _signup(client, "quota-member@example.com")
    client.post(
        "/api/auth/login",
        json={"email": "quota-admin@example.com", "password": "SecurePass1!"},
    )
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"max_concurrent_backtests": 0},
    )
    assert resp.status_code == 422


@pytest.fixture
def isolated_auth(monkeypatch):
    """Fresh UserStore so bootstrap tests do not see admins from other cases."""
    import dashboard.backend.users as users_module
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    from dashboard.backend.api.routers import admin_users as admin_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        admin_mod._BOOTSTRAP_LIMITER.reset()
        yield TestClient(app), store
        admin_mod._BOOTSTRAP_LIMITER.reset()


def test_bootstrap_unset_is_503(isolated_auth, monkeypatch):
    monkeypatch.delenv("ADMIN_BOOTSTRAP_SECRET", raising=False)
    client, _store = isolated_auth
    _signup(client, "boot-unset@example.com")
    resp = client.post("/api/admin/bootstrap", json={"secret": "atleast8chars"})
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()


def test_bootstrap_wrong_secret_is_403_even_on_length_mismatch(isolated_auth, monkeypatch):
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value-32chars!!")
    client, store = isolated_auth
    user = _signup(client, "boot-wrong@example.com")
    resp = client.post("/api/admin/bootstrap", json={"secret": "short-ok"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Invalid bootstrap secret"
    assert store.get_user_by_id(user["id"])["role"] == "user"


def test_bootstrap_first_caller_succeeds_second_refused(isolated_auth, monkeypatch):
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value")
    client, store = isolated_auth
    first = _signup(client, "boot-first@example.com")
    ok = client.post(
        "/api/admin/bootstrap", json={"secret": "correct-secret-value"}
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()["user"]
    assert body["role"] == "admin"
    assert body["entitlements"]["max_concurrent_backtests"] >= 5
    assert store.get_user_by_id(first["id"])["role"] == "admin"

    client.post("/api/auth/logout")
    second = _signup(client, "boot-second@example.com")
    denied = client.post(
        "/api/admin/bootstrap", json={"secret": "correct-secret-value"}
    )
    assert denied.status_code == 403, denied.text
    assert "no admin exists" in denied.json()["detail"].lower()
    assert store.get_user_by_id(second["id"])["role"] == "user"


def test_bootstrap_rate_limits_wrong_secret(isolated_auth, monkeypatch):
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value")
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=900),
    )
    client, store = isolated_auth
    user = _signup(client, "boot-limit@example.com")
    payload = {"secret": "wrong-secret-value"}
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    limited = client.post("/api/admin/bootstrap", json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    assert store.get_user_by_id(user["id"])["role"] == "user"
