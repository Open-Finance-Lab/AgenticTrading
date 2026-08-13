"""Admin users API + entitlements store behaviour."""

import tempfile
from pathlib import Path

import pytest

import dashboard.backend.users as users_module


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield users_module.UserStore(db_path=Path(tmpdir) / "users.db")


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

    users_module.user_store.set_user_role(user["id"], "admin")
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
    users_module.user_store.set_user_role(admin["id"], "admin")
    # Second admin so last_admin is not the reason for refusal.
    other = _signup(client, "otheradmin@example.com")
    users_module.user_store.set_user_role(other["id"], "admin")

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
    assert users_module.user_store.get_user_by_id(admin["id"])["role"] == "admin"


def test_admin_stats_endpoint():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    from dashboard.backend.domain.agents.repository import agent_store

    client = TestClient(app)
    admin = _signup(client, "stats-admin@example.com")
    users_module.user_store.set_user_role(admin["id"], "admin")
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
    users_module.user_store.set_user_role(admin["id"], "admin")
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


def test_secrets_equal_matches_and_survives_hostile_input():
    from dashboard.backend.api.routers.admin_users import secrets_equal

    assert secrets_equal("same-secret", "same-secret") is True
    assert secrets_equal("short-ok", "a-much-longer-secret") is False
    assert secrets_equal("wrong-secret", "right-secret") is False
    # The actual hazard the SHA-256 wrapper exists for: compare_digest raises
    # TypeError on a non-ASCII str, and a JSON body can carry any character.
    # (A length mismatch never raised — it just compares false, as above.)
    assert secrets_equal("pässwörd-ünicode", "correct-secret") is False
    assert secrets_equal("naïve-secret-value", "naïve-secret-value") is True


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
    users_module.user_store.set_user_role(admin["id"], "admin")
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
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    from dashboard.backend.api.routers import admin_users as admin_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        store = users_module.UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        admin_mod.reset_bootstrap_limiters()
        yield TestClient(app), store
        admin_mod.reset_bootstrap_limiters()


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


def test_partial_entitlement_patch_leaves_the_other_field_alone(store):
    """A field the caller omitted must not be rewritten with a stale read.

    The upsert used to read both values, then write both back, across two
    connections. Two admins patching different fields concurrently therefore
    lost one edit; the COALESCE upsert only touches what was supplied.
    """
    user = store.create_user("split@example.com", "S", "securepass1")
    store.set_entitlements(user["id"], max_concurrent_backtests=7, credits=250)

    only_credits = store.set_entitlements(user["id"], credits=999)
    assert only_credits["credits"] == 999
    assert only_credits["max_concurrent_backtests"] == 7

    only_max = store.set_entitlements(user["id"], max_concurrent_backtests=3)
    assert only_max["max_concurrent_backtests"] == 3
    assert only_max["credits"] == 999


def test_apply_admin_patch_is_all_or_nothing(store):
    """Role and entitlements land together, or neither does."""
    admin = store.create_user("keeper@example.com", "K", "securepass1")
    store.set_user_role(admin["id"], "admin")
    target = store.create_user("target@example.com", "T", "securepass1")

    updated = store.apply_admin_patch(
        target["id"],
        role="admin",
        max_concurrent_backtests=4,
        credits=12,
        updated_by_admin_id=admin["id"],
    )
    assert updated["role"] == "admin"
    assert updated["entitlements"]["max_concurrent_backtests"] == 4
    assert updated["entitlements"]["credits"] == 12

    # A rejected role change must not smuggle the quota half through.
    with pytest.raises(ValueError, match="invalid_role"):
        store.apply_admin_patch(target["id"], role="superuser", credits=77)
    assert store.get_entitlements(target["id"])["credits"] == 12


def test_last_admin_demotion_rolls_back_the_quota_half(store):
    user = store.create_user("solo@example.com", "S", "securepass1")
    store.set_user_role(user["id"], "admin")
    store.set_entitlements(user["id"], credits=5)

    with pytest.raises(ValueError, match="last_admin"):
        store.apply_admin_patch(user["id"], role="user", credits=4242)

    assert store.get_user_by_id(user["id"])["role"] == "admin"
    assert store.get_entitlements(user["id"])["credits"] == 5


def test_admin_list_omits_avatars_and_reports_total(store):
    """The console renders text and two numbers; avatars are pure payload.

    Each one is a data: URI bounded at 200_000 chars, so a 100-row page would
    otherwise be tens of megabytes off a free-tier box.
    """
    for i in range(3):
        user = store.create_user(f"av{i}@example.com", f"A{i}", "securepass1")
        store.set_avatar(user["id"], "data:image/png;base64,AAAA")

    listed = store.list_users_admin(limit=2, offset=0)
    assert len(listed) == 2
    assert all("avatar" not in row for row in listed)
    assert store.get_user_admin(listed[0]["id"]).get("avatar") is None

    page_two = store.list_users_admin(limit=2, offset=2)
    assert len(page_two) == 1
    assert store.count_users() == 3


def test_admin_users_endpoint_paginates():
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    admin = _signup(client, "pager-admin@example.com")
    users_module.user_store.set_user_role(admin["id"], "admin")
    for i in range(3):
        _signup(client, f"pager-member{i}@example.com")
    client.post(
        "/api/auth/login",
        json={"email": "pager-admin@example.com", "password": "SecurePass1!"},
    )

    first = client.get("/api/admin/users?limit=2&offset=0")
    assert first.status_code == 200, first.text
    body = first.json()
    assert len(body["users"]) == 2
    assert body["limit"] == 2 and body["offset"] == 0
    # Without total the console cannot tell a full list from a first page.
    assert body["total"] >= 4
    assert all("avatar" not in row for row in body["users"])

    second = client.get("/api/admin/users?limit=2&offset=2")
    assert second.status_code == 200
    first_ids = {row["id"] for row in body["users"]}
    second_ids = {row["id"] for row in second.json()["users"]}
    assert not (first_ids & second_ids)


def test_me_reports_entitlements_without_a_second_query(monkeypatch):
    """/me is on the boot path — entitlements ride the session join."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    user = _signup(client, "boot-ent@example.com")
    users_module.user_store.set_entitlements(
        user["id"], max_concurrent_backtests=6, credits=42
    )

    calls = []
    real = users_module.user_store.get_entitlements
    monkeypatch.setattr(
        users_module.user_store,
        "get_entitlements",
        lambda uid: (calls.append(uid), real(uid))[1],
    )
    resp = client.get("/api/auth/me")
    assert resp.status_code == 200, resp.text
    entitlements = resp.json()["user"]["entitlements"]
    assert entitlements["max_concurrent_backtests"] == 6
    assert entitlements["credits"] == 42
    assert calls == []


def test_bootstrap_survives_a_failed_entitlement_grant(isolated_auth, monkeypatch):
    """Promotion is committed and one-shot: the quota seed must not 500.

    A 500 here tells the operator bootstrap failed, and their retry then hits
    ``admin_exists`` -> 403 — while they have in fact been an admin all along.
    """
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value")
    client, store = isolated_auth
    user = _signup(client, "boot-grant@example.com")

    def _boom(*args, **kwargs):
        raise RuntimeError("entitlements table is on fire")

    monkeypatch.setattr(store, "set_entitlements", _boom)
    resp = client.post("/api/admin/bootstrap", json={"secret": "correct-secret-value"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["role"] == "admin"
    assert store.get_user_by_id(user["id"])["role"] == "admin"


def test_bootstrap_budget_is_not_reset_by_a_fresh_account(isolated_auth, monkeypatch):
    """Signup is open, so a per-user counter is not a bound on guessing."""
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value")
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=900),
    )
    client, _store = isolated_auth
    payload = {"secret": "wrong-secret-value"}

    _signup(client, "burner-one@example.com")
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403

    # Same client, brand-new account: the per-client key is already spent.
    client.post("/api/auth/logout")
    _signup(client, "burner-two@example.com")
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 429


def test_bootstrap_global_ceiling_applies(isolated_auth, monkeypatch):
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value")
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_GLOBAL_LIMITER",
        FixedWindowRateLimiter(max_events=1, window_seconds=900),
    )
    client, _store = isolated_auth
    _signup(client, "global-cap@example.com")
    payload = {"secret": "wrong-secret-value"}
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    limited = client.post("/api/admin/bootstrap", json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


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
