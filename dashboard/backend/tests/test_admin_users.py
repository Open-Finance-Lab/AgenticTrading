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

def test_backtest_slot_respects_entitlement():
    import dashboard.backend.api.routers.backtests as bt

    with bt._backtest_slots_lock:
        bt._active_slots.clear()
        bt._recent_slots.clear()
    bt.backtest_status.update({
        "running": False,
        "error": None,
        "runs_count": 0,
        "started_at": None,
        "progress_file": None,
        "live_run_id": None,
    })

    assert bt._try_acquire_backtest_slot(
        live_run_id="r1", session_id="s1", user_id=None
    ) is None
    assert bt._try_acquire_backtest_slot(
        live_run_id="r2", session_id="s1", user_id=None
    ) == "Backtest already running. Please wait for it to complete."

    with bt._backtest_slots_lock:
        bt._active_slots.clear()

    user = user_store.create_user("slot@example.com", "Slot", "SecurePass1!")
    user_store.set_entitlements(user["id"], max_concurrent_backtests=2)

    assert bt._try_acquire_backtest_slot(
        live_run_id="a1", session_id="sx", user_id=user["id"]
    ) is None
    assert bt._try_acquire_backtest_slot(
        live_run_id="a2", session_id="sx", user_id=user["id"]
    ) is None
    refused = bt._try_acquire_backtest_slot(
        live_run_id="a3", session_id="sx", user_id=user["id"]
    )
    assert refused and "2 backtests" in refused
