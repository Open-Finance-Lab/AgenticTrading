"""Admin users API + entitlements store behaviour."""

import tempfile
from pathlib import Path

import pytest

import dashboard.backend.users as users_module

# >= admin_users._BOOTSTRAP_MIN_LENGTH (32). A shorter value is now
# refused as if unset, so every bootstrap test has to use a realistic one.
_SECRET = "correct-secret-value-long-enough-to-pass"
_WRONG = "wrong-secret-value-long-enough-to-pass!!"


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield users_module.UserStore(db_path=Path(tmpdir) / "users.db")


@pytest.fixture
def isolated_auth(monkeypatch):
    """Fresh UserStore swapped in for the module singleton.

    Every test that signs up accounts or grants the admin role uses this —
    promoting an admin in the shared conftest store would leak the row into
    every later test in the session, making admin counts (and the last-admin
    guard) order-dependent.
    """
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app
    from dashboard.backend.api.routers import admin_users as admin_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        store = users_module.UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        admin_mod.reset_bootstrap_limiters()
        yield TestClient(app), store
        admin_mod.reset_bootstrap_limiters()


def _promote(store, user_id):
    """Grant admin through the one production write path for role changes."""
    return store.apply_admin_patch(user_id, role="admin")


def test_entitlements_default_then_upsert(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    defaults = store.get_entitlements(user["id"])
    # Read from the constant, not a literal: this value is every existing
    # account's live quota on the deploy that ships the entitlement plane
    # (nothing seeds a row at signup and nothing backfills), so pinning a
    # number here would let a regression to 1 pass as "the test says 1".
    assert (
        defaults["max_concurrent_backtests"]
        == users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS
    )
    # Same reasoning as above, and the same trap: this is the balance every
    # existing account holds the moment an operator arms metering.
    assert defaults["credits"] == users_module.DEFAULT_CREDITS

    updated = store.set_entitlements(
        user["id"],
        max_concurrent_backtests=5,
        credits=100,
        updated_by_admin_id=user["id"],
    )
    assert updated["max_concurrent_backtests"] == 5
    assert updated["credits"] == 100
    assert updated["updated_by_admin_id"] == user["id"]


def test_default_quota_does_not_throttle_existing_accounts():
    """The default has to cover what a user could already do before this PR.

    Before the entitlement plane, one account's concurrency was bounded only by
    MAX_ACTIVE_RUNS_PER_AGENT per agent it owned. A default below that number
    silently demotes every account that exists on the day this ships — with no
    backfill, and no way back that does not require an admin.
    """
    from dashboard.backend.domain.runs.service import MAX_ACTIVE_RUNS_PER_AGENT

    assert users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS >= MAX_ACTIVE_RUNS_PER_AGENT


@pytest.mark.parametrize(
    "env_var, builtin, cap",
    [
        (
            "DEFAULT_MAX_CONCURRENT_BACKTESTS",
            users_module._BUILTIN_DEFAULT_CONCURRENT_BACKTESTS,
            users_module.MAX_CONCURRENT_BACKTESTS_CAP,
        ),
        (
            "DEFAULT_CREDITS",
            users_module._BUILTIN_DEFAULT_CREDITS,
            users_module.MAX_CREDITS_CAP,
        ),
    ],
)
def test_entitlement_defaults_are_env_overridable(monkeypatch, env_var, builtin, cap):
    monkeypatch.setenv(env_var, "3")
    assert users_module._default_entitlement(env_var, builtin, cap) == 3
    # A typo in a Render var must not take the process down, and must not
    # silently become a quota of 0 either.
    for bad in ("not-a-number", str(cap + 1), "-1"):
        monkeypatch.setenv(env_var, bad)
        assert users_module._default_entitlement(env_var, builtin, cap) == builtin, bad
    # 0 is a legal value, not a typo: it reads as "suspended" for both fields.
    monkeypatch.setenv(env_var, "0")
    assert users_module._default_entitlement(env_var, builtin, cap) == 0


def test_apply_role_and_list_admin(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    _promote(store, user["id"])
    store.set_entitlements(user["id"], max_concurrent_backtests=5, credits=10)

    listed = store.list_users_admin()
    assert len(listed) == 1
    assert listed[0]["role"] == "admin"
    assert listed[0]["entitlements"]["max_concurrent_backtests"] == 5


def test_admin_user_search_is_trimmed_case_insensitive_and_escaped(store):
    store.create_user("alice@example.com", "Research Alice", "securepass1")
    store.create_user("bob@example.com", "Operations", "securepass1")
    store.create_user("percent%name@example.com", "Percent Name", "securepass1")

    assert [row["email"] for row in store.list_users_admin(query="  ALICE  ")] == [
        "alice@example.com"
    ]
    assert store.count_users(query="research") == 1
    assert store.list_users_admin(query="%")[0]["email"] == (
        "percent%name@example.com"
    )


def test_cannot_demote_last_admin(store):
    user = store.create_user("a@example.com", "A", "securepass1")
    _promote(store, user["id"])
    with pytest.raises(ValueError, match="last_admin"):
        store.apply_admin_patch(user["id"], role="user")


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


def test_admin_users_requires_admin(isolated_auth):
    client, store = isolated_auth
    user = _signup(client, "plain@example.com")
    resp = client.get("/api/admin/users")
    assert resp.status_code == 403

    _promote(store, user["id"])
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["role"] == "admin"
    assert "entitlements" in me.json()["user"]

    listed = client.get("/api/admin/users")
    assert listed.status_code == 200
    emails = {row["email"] for row in listed.json()["users"]}
    assert "plain@example.com" in emails


def test_every_admin_route_refuses_a_non_admin(isolated_auth):
    """All four, not just the one that happened to get a test.

    ``GET /users`` was the only route with a non-admin case, which is exactly
    the coverage shape that lets a missing guard ship: the console is small
    enough that whoever adds route five will copy route four, and only route
    one is pinned. Enumerated from the router itself so a new route without a
    case here fails loudly rather than silently going ungated.
    """
    from dashboard.backend.api.routers.admin_users import router

    client, store = isolated_auth
    user = _signup(client, "outsider@example.com")

    calls = {
        ("GET", "/api/admin/stats"): lambda: client.get("/api/admin/stats"),
        ("GET", "/api/admin/users"): lambda: client.get("/api/admin/users"),
        ("GET", "/api/admin/users/{user_id}"): lambda: client.get(
            f"/api/admin/users/{user['id']}"
        ),
        ("PATCH", "/api/admin/users/{user_id}"): lambda: client.patch(
            f"/api/admin/users/{user['id']}", json={"role": "admin"}
        ),
    }
    registered = {
        (method, f"/api{route.path}")
        for route in router.routes
        for method in route.methods
        if not route.path.endswith("/bootstrap")  # deliberately not admin-gated
    }
    assert registered == set(calls), (
        "an admin route was added or renamed without a non-admin case here"
    )

    for (method, path), call in calls.items():
        resp = call()
        assert resp.status_code == 403, f"{method} {path} -> {resp.status_code}"
    assert store.get_user_by_id(user["id"])["role"] == "user"


def test_admin_routes_are_gated_at_the_router_not_per_handler():
    """Defence in depth for the route nobody has written yet.

    Before, the guard was a per-route opt-in spelled ``_admin:`` — a
    leading-underscore parameter that reads as an unused argument to anyone
    tidying the file, and whose absence on a new route is invisible.
    """
    from dashboard.backend.api.routers.admin_users import admin_router, router

    assert admin_router.dependencies, "admin_router must carry require_admin"
    assert admin_router.routes, "admin routes must live on the gated sub-router"
    # /bootstrap is the deliberate exception: it is the one route that has to
    # work when no admin exists yet.
    bootstrap = [r for r in router.routes if r.path.endswith("/bootstrap")]
    assert len(bootstrap) == 1
    assert not bootstrap[0].dependencies


def test_role_and_quota_changes_leave_an_audit_line(isolated_auth, capsys):
    """Both twins write a bare UPDATE: no actor, no timestamp, no history.

    A privilege system where "who promoted whom" is recorded nowhere is a real
    gap, and an audit table is a schema change this PR should not grow — but
    the log is better than nothing, and it is the operator's only real channel
    in prod (logger output is invisible under deployed uvicorn).
    """
    client, store = isolated_auth
    admin = _signup(client, "audit-admin@example.com")
    _promote(store, admin["id"])
    target = _signup(client, "audit-target@example.com")
    client.post(
        "/api/auth/login",
        json={"email": "audit-admin@example.com", "password": "SecurePass1!"},
    )
    capsys.readouterr()
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"role": "admin", "max_concurrent_backtests": 9},
    )
    assert resp.status_code == 200, resp.text
    out = capsys.readouterr().out
    assert "admin.user_patched" in out
    assert f"actor={admin['id']}" in out
    assert f"target={target['id']}" in out
    assert "role=admin" in out
    assert "max_concurrent_backtests=9" in out
    # Never the email: attacker-influenced strings do not go to a print sink
    # (see api/auth.py::_email_domain for the log-injection that taught this).
    assert "audit-target@example.com" not in out


def test_admin_cannot_demote_self(isolated_auth):
    client, store = isolated_auth
    admin = _signup(client, "selfadmin@example.com")
    _promote(store, admin["id"])
    # Second admin so last_admin is not the reason for refusal.
    other = _signup(client, "otheradmin@example.com")
    _promote(store, other["id"])

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
    assert store.get_user_by_id(admin["id"])["role"] == "admin"


def test_admin_stats_endpoint(isolated_auth):
    from dashboard.backend.domain.agents.repository import agent_store

    client, store = isolated_auth
    admin = _signup(client, "stats-admin@example.com")
    _promote(store, admin["id"])
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
    # Exact on the isolated store: two signups, one admin.
    assert body["users"] == 2
    assert body["admins"] == 1
    assert body["agents"] >= 1
    assert "active_dashboard_backtests" in body


def test_admin_patch_entitlements_and_role(isolated_auth):
    client, store = isolated_auth
    admin = _signup(client, "boss@example.com")
    _promote(store, admin["id"])
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


def test_backtest_slot_respects_entitlement(monkeypatch, tmp_path):
    """The dashboard runner's per-owner slots read the same entitlement.

    Swaps the module singleton rather than using the shared conftest store:
    ``_max_concurrent_for_user`` resolves ``users.user_store`` at call time, so
    the patched attribute is what it reads, and an account created here must
    not leak into later tests' admin counts.
    """
    import dashboard.backend.api.routers.backtests as bt

    store = users_module.UserStore(db_path=tmp_path / "users.db")
    monkeypatch.setattr(users_module, "user_store", store)

    bt._reset_slots_for_tests()
    try:
        assert bt._try_acquire_backtest_slot(
            live_run_id="r1", session_id="s1", user_id=None
        ) is None
        assert bt._try_acquire_backtest_slot(
            live_run_id="r2", session_id="s1", user_id=None
        ) == "Backtest already running. Please wait for it to complete."

        bt._reset_slots_for_tests()

        user = store.create_user("slot@example.com", "Slot", "SecurePass1!")
        store.set_entitlements(user["id"], max_concurrent_backtests=2)

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
    finally:
        bt._reset_slots_for_tests()


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
    # Canonical home: session_tokens, shared by any shared-secret gate.
    from dashboard.backend.session_tokens import secrets_equal

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


def test_admin_patch_rejects_out_of_range_quotas(isolated_auth):
    client, store = isolated_auth
    admin = _signup(client, "quota-admin@example.com")
    _promote(store, admin["id"])
    target = _signup(client, "quota-member@example.com")
    client.post(
        "/api/auth/login",
        json={"email": "quota-admin@example.com", "password": "SecurePass1!"},
    )
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"max_concurrent_backtests": -1},
    )
    assert resp.status_code == 422
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"max_concurrent_backtests": 21},
    )
    assert resp.status_code == 422


def test_admin_can_suspend_an_account_with_a_zero_quota(isolated_auth):
    """0 is the only value that actually *stops* an account.

    The floor used to be 1 — the same number a fresh signup already has — so
    lowering a quota did nothing to the case an admin would reach for it in:
    an account abusing the platform right now. 0 needs no new column to mean
    "suspended": check_owner_active_run_cap refuses at ``active >= limit``.
    """
    from dashboard.backend.domain.runs.service import check_owner_active_run_cap

    client, store = isolated_auth
    admin = _signup(client, "suspend-admin@example.com")
    _promote(store, admin["id"])
    target = _signup(client, "suspend-member@example.com")
    client.post(
        "/api/auth/login",
        json={"email": "suspend-admin@example.com", "password": "SecurePass1!"},
    )
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"max_concurrent_backtests": 0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["entitlements"]["max_concurrent_backtests"] == 0
    assert store.get_entitlements(target["id"])["max_concurrent_backtests"] == 0
    # A zero budget refuses the *first* run, not just the second.
    violation = check_owner_active_run_cap(
        {"limit": 0, "agent_ids": [], "scope": "account"}
    )
    assert violation == {
        "active_runs_for_account": 0,
        "limit": 0,
        "scope": "account",
    }


def test_admin_patch_rejects_explicit_null(isolated_auth):
    """``{"credits": null}`` used to be indistinguishable from omitting the
    key — a 200 that changed nothing. The API now refuses the null."""
    client, store = isolated_auth
    admin = _signup(client, "null-admin@example.com")
    _promote(store, admin["id"])
    target = _signup(client, "null-member@example.com")
    store.set_entitlements(target["id"], credits=7)
    client.post(
        "/api/auth/login",
        json={"email": "null-admin@example.com", "password": "SecurePass1!"},
    )
    resp = client.patch(
        f"/api/admin/users/{target['id']}",
        json={"credits": None},
    )
    assert resp.status_code == 400, resp.text
    assert "null" in resp.json()["detail"].lower()
    assert store.get_entitlements(target["id"])["credits"] == 7


def test_bootstrap_unset_answers_exactly_like_a_wrong_secret(isolated_auth, monkeypatch):
    """Unconfigured must not be distinguishable from wrong.

    A 503 "not configured" told any caller who can sign up whether this
    deployment is bootstrappable at all — i.e. whether guessing the secret is
    worth their time, and whether an admin exists yet. The repo already makes
    the opposite trade nowhere: LEADERBOARD_DAILY_REFRESH_SECRET 401s whether
    or not it is armed and sends the operator's signal to the log. Same here.
    """
    monkeypatch.delenv("ADMIN_BOOTSTRAP_SECRET", raising=False)
    client, _store = isolated_auth
    _signup(client, "boot-unset@example.com")
    unset = client.post("/api/admin/bootstrap", json={"secret": "atleast8chars"})

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    wrong = client.post("/api/admin/bootstrap", json={"secret": _WRONG})

    assert unset.status_code == 403
    assert unset.json() == wrong.json()


def test_bootstrap_refuses_a_weak_secret(isolated_auth, monkeypatch, capsys):
    """A short ADMIN_BOOTSTRAP_SECRET is refused as if unset, and says so.

    This value promotes its bearer to admin with no account behind it and no
    lockout to hide behind, so "the operator picked something strong" cannot be
    an assumption the route rests on. The operator's signal goes to the log —
    which already implies server access — never to the caller.
    """
    from dashboard.backend.api.routers import admin_users as admin_mod

    weak = "hunter2!"
    assert len(weak) < admin_mod._BOOTSTRAP_MIN_LENGTH
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", weak)
    client, store = isolated_auth
    user = _signup(client, "boot-weak@example.com")

    resp = client.post("/api/admin/bootstrap", json={"secret": weak})
    assert resp.status_code == 403, resp.text
    assert store.get_user_by_id(user["id"])["role"] == "user"
    out = capsys.readouterr().out
    assert "shorter than" in out
    assert weak not in out  # never the value, not even a slice of it


def test_bootstrap_wrong_secret_is_403_even_on_length_mismatch(isolated_auth, monkeypatch):
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", "correct-secret-value-32chars!!")
    client, store = isolated_auth
    user = _signup(client, "boot-wrong@example.com")
    resp = client.post("/api/admin/bootstrap", json={"secret": "short-ok"})
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == "Invalid bootstrap secret"
    assert store.get_user_by_id(user["id"])["role"] == "user"


def test_bootstrap_first_caller_succeeds_second_refused(isolated_auth, monkeypatch):
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    client, store = isolated_auth
    first = _signup(client, "boot-first@example.com")
    ok = client.post(
        "/api/admin/bootstrap", json={"secret": _SECRET}
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()["user"]
    assert body["role"] == "admin"
    assert body["entitlements"]["max_concurrent_backtests"] >= 5
    assert store.get_user_by_id(first["id"])["role"] == "admin"

    client.post("/api/auth/logout")
    second = _signup(client, "boot-second@example.com")
    denied = client.post(
        "/api/admin/bootstrap", json={"secret": _SECRET}
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
    _promote(store, admin["id"])
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
    _promote(store, user["id"])
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


def test_admin_users_endpoint_paginates(isolated_auth):
    client, store = isolated_auth
    admin = _signup(client, "pager-admin@example.com")
    _promote(store, admin["id"])
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
    assert body["total"] == 4
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
    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    client, store = isolated_auth
    user = _signup(client, "boot-grant@example.com")

    def _boom(*args, **kwargs):
        raise RuntimeError("entitlements table is on fire")

    monkeypatch.setattr(store, "set_entitlements", _boom)
    resp = client.post("/api/admin/bootstrap", json={"secret": _SECRET})
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["role"] == "admin"
    assert store.get_user_by_id(user["id"])["role"] == "admin"


def test_bootstrap_budget_is_not_reset_by_a_fresh_account(isolated_auth, monkeypatch):
    """Signup is open, so a per-user counter is not a bound on guessing."""
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=900),
    )
    client, _store = isolated_auth
    payload = {"secret": _WRONG}

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

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_GLOBAL_LIMITER",
        FixedWindowRateLimiter(max_events=1, window_seconds=900),
    )
    client, _store = isolated_auth
    _signup(client, "global-cap@example.com")
    payload = {"secret": _WRONG}
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    limited = client.post("/api/admin/bootstrap", json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers


def test_a_spent_global_budget_still_lets_the_right_secret_through(
    isolated_auth, monkeypatch
):
    """Wrong guesses must not be able to lock the real operator out.

    The global ceiling is the one budget a header-rotating caller cannot dodge,
    which made it the one they could aim at the operator: it was checked before
    the secret was even read, so 20 wrong guesses per window — re-spent every
    window, from any account — refused the correct secret indefinitely. And the
    window it blocked is exactly the fresh-deploy window bootstrap exists for.
    Comparing first leaks nothing: secrets_equal is constant-time and a wrong
    guess still leaves with 403 or 429.
    """
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_GLOBAL_LIMITER",
        FixedWindowRateLimiter(max_events=1, window_seconds=900),
    )
    client, store = isolated_auth

    # An attacker burns the whole server-wide budget from their own account.
    _signup(client, "attacker@example.com")
    assert client.post("/api/admin/bootstrap", json={"secret": _WRONG}).status_code == 403
    assert client.post("/api/admin/bootstrap", json={"secret": _WRONG}).status_code == 429

    # The operator, holding the real secret, is still let through.
    client.post("/api/auth/logout")
    operator = _signup(client, "operator@example.com")
    ok = client.post("/api/admin/bootstrap", json={"secret": _SECRET})
    assert ok.status_code == 200, ok.text
    assert store.get_user_by_id(operator["id"])["role"] == "admin"


def test_bootstrap_rate_limits_wrong_secret(isolated_auth, monkeypatch):
    from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
    from dashboard.backend.api.routers import admin_users as admin_mod

    monkeypatch.setenv("ADMIN_BOOTSTRAP_SECRET", _SECRET)
    monkeypatch.setattr(
        admin_mod,
        "_BOOTSTRAP_LIMITER",
        FixedWindowRateLimiter(max_events=2, window_seconds=900),
    )
    client, store = isolated_auth
    user = _signup(client, "boot-limit@example.com")
    payload = {"secret": _WRONG}
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    assert client.post("/api/admin/bootstrap", json=payload).status_code == 403
    limited = client.post("/api/admin/bootstrap", json=payload)
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers
    assert store.get_user_by_id(user["id"])["role"] == "user"
