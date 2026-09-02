"""
PostgresUserStore tests.

Two tiers:
1. Dispatch-logic tests (no live Postgres needed) - verify users.py picks
   the right store class based on USERS_DATABASE_URL.
2. Behavioral tests against a real Postgres - skipped unless
   TEST_POSTGRES_URL is set. Point it at a throwaway database, e.g.:
     docker run --rm -e POSTGRES_PASSWORD=test -e POSTGRES_DB=atl_test \
       -p 5433:5432 postgres:18-alpine
     export TEST_POSTGRES_URL=postgresql://postgres:test@localhost:5433/atl_test

Both modules under test are imported in module form (``import x as x_module``)
throughout, never ``from x import Name``. The monkeypatch fixtures below need
the module object to rebind ``PostgresUserStore`` on it, so the two forms cannot
both be used -- mixing them is a real inconsistency, not a style preference, and
CodeQL's py/import-and-import-from flags it.
"""

import os
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token
from dashboard.backend.tests._postgres_testing import require_local_postgres_url

TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")

pg_only = pytest.mark.skipif(
    not TEST_POSTGRES_URL,
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)


def test_build_user_store_defaults_to_sqlite(monkeypatch):
    import dashboard.backend.users as users_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    store = users_module._build_user_store()
    assert isinstance(store, users_module.UserStore)


@pytest.mark.parametrize("marker", ["RENDER", "ATL_ENV"])
def test_build_user_store_refuses_to_boot_without_a_session_secret(monkeypatch, marker):
    """A missing SESSION_HASH_SECRET has to kill the boot, not the traffic.

    session_hash_secret() raises at *call* time. With nothing resolving it
    during startup the service comes up healthy, /health returns 200 and the CI
    deploy hook reports success -- then every login and every authenticated
    request 500s with no CORS headers, which reaches the browser as a CORS
    error rather than a server error. Render keeps the previous version live
    when a boot fails, so this is the safer half of the trade.
    """
    import dashboard.backend.users as users_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.delenv("SESSION_HASH_SECRET", raising=False)
    monkeypatch.setenv(marker, "true" if marker == "RENDER" else "production")

    with pytest.raises(RuntimeError, match="SESSION_HASH_SECRET"):
        users_module._build_user_store()


def test_build_user_store_picks_postgres_when_url_set(monkeypatch):
    import dashboard.backend.users as users_module
    import dashboard.backend.users_postgres as users_postgres_module

    created = {}

    class FakePostgresUserStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(users_postgres_module, "PostgresUserStore", FakePostgresUserStore)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://fake/db")

    store = users_module._build_user_store()

    assert isinstance(store, FakePostgresUserStore)
    assert created["database_url"] == "postgresql://fake/db"


@pytest.fixture
def temp_postgres_store():
    require_local_postgres_url(TEST_POSTGRES_URL)
    import dashboard.backend.users_postgres as users_postgres_module

    store = users_postgres_module.PostgresUserStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM password_reset_requests")
            cur.execute("DELETE FROM email_change_requests")
            cur.execute("DELETE FROM auth_sessions")
            # Redundant on a schema this repo created (every child table FKs
            # users with ON DELETE CASCADE), like the three above -- the
            # explicit wipe is what keeps a long-lived dev database honest
            # after a review branch's divergent DDL or manual tampering (#437).
            cur.execute("DELETE FROM user_entitlements")
            cur.execute("DELETE FROM users")
    yield store


@pytest.fixture
def pg_client(temp_postgres_store, monkeypatch):
    import dashboard.backend.users as users_module

    # api/auth.py resolves users_module.user_store at call time (issue #185),
    # so this single patch redirects every auth route. Before that fix it also
    # needed dashboard.backend.api.auth patched, and without it this "postgres"
    # test silently exercised SQLite -- caught only when CI first ran the live
    # tier and it collided with test_auth.py's alice@example.com.
    monkeypatch.setattr(users_module, "user_store", temp_postgres_store)
    return TestClient(app)


@pg_only
def test_signup_login_me_logout_flow_postgres(pg_client, temp_postgres_store):
    signup = pg_client.post(
        "/api/auth/signup",
        json={"email": "alice@example.com", "display_name": "Alice", "password": "securepass1"},
    )
    assert signup.status_code == 200
    signup_data = signup.json()
    assert signup_data["user"]["email"] == "alice@example.com"
    assert signup_data["user"]["display_name"] == "Alice"
    assert signup_data["user"]["role"] == "user"
    assert "password_hash" not in signup_data["user"]
    assert "token" not in signup_data
    assert _cookie_session_token(pg_client)  # signup set the session cookie

    # Prove the route's write actually landed in Postgres. Without this, a
    # regression that re-detaches the routes from the patched store would
    # leave this test green while testing SQLite -- which is exactly the
    # state it shipped in.
    assert temp_postgres_store.get_user_by_email("alice@example.com") is not None

    duplicate = pg_client.post(
        "/api/auth/signup",
        json={"email": "alice@example.com", "display_name": "Alice 2", "password": "securepass1"},
    )
    assert duplicate.status_code == 409

    login = pg_client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "securepass1"},
    )
    assert login.status_code == 200
    assert "token" not in login.json()
    token = _cookie_session_token(pg_client)

    me = pg_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "alice@example.com"

    logout = pg_client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout.status_code == 200

    me_after = pg_client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_after.status_code == 401


@pg_only
def test_login_invalid_password_postgres(pg_client):
    pg_client.post(
        "/api/auth/signup",
        json={"email": "bob@example.com", "display_name": "Bob", "password": "securepass1"},
    )
    response = pg_client.post(
        "/api/auth/login",
        json={"email": "bob@example.com", "password": "wrong-password"},
    )
    assert response.status_code == 401


def test_build_user_store_ignores_content_database_url(monkeypatch, capsys):
    """The two URLs are scoped per store (spec, Decision 2), and that separation
    is only a claim until something asserts it.

    This is the inverse of the precedence test the fallback design would have
    needed: CONTENT_DATABASE_URL must not reach the users store at all, not
    merely lose to USERS_DATABASE_URL. Without this, re-adding the fallback --
    a one-line "convenience" a future contributor could plausibly think is an
    improvement -- keeps the suite green while silently binding accounts to the
    content database.
    """
    import dashboard.backend.users as users_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://fake/content")

    store = users_module._build_user_store()

    assert isinstance(store, users_module.UserStore)
    # capsys, not caplog: the factory print()s. A caplog test would pass even if
    # the line were invisible in prod -- see the plan's Global Constraints.
    assert "user_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out


def test_build_user_store_announces_sqlite_backend(monkeypatch, capsys):
    import dashboard.backend.users as users_module

    monkeypatch.delenv("USERS_DATABASE_URL", raising=False)
    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = users_module._build_user_store()
    assert isinstance(store, users_module.UserStore)
    assert "user_store backend: sqlite (ephemeral on Render)" in capsys.readouterr().out


def test_build_user_store_never_prints_the_credentials(monkeypatch, capsys):
    import dashboard.backend.users as users_module
    import dashboard.backend.users_postgres as users_postgres_module

    class FakePostgresUserStore:
        def __init__(self, database_url):
            pass

    monkeypatch.setattr(users_postgres_module, "PostgresUserStore", FakePostgresUserStore)
    monkeypatch.setenv("USERS_DATABASE_URL", "postgresql://admin:sup3r-s3cret@host/db")

    users_module._build_user_store()

    out = capsys.readouterr().out
    assert "sup3r-s3cret" not in out
    assert "user_store backend: postgres (host/db)" in out


def test_unreachable_postgres_raises_instead_of_falling_back():
    """Fail loud: a set-but-unreachable URL must not silently degrade to SQLite.

    This is the tier that exercises PostgresUserStore.__init__ for real -- the
    dispatch tests above monkeypatch the class away, so nothing else does. Needs
    no live Postgres: a closed port refuses instantly. connect_timeout keeps a
    firewall that DROPs rather than REJECTs from hanging the suite.
    """
    import psycopg

    import dashboard.backend.users_postgres as users_postgres_module

    with pytest.raises(psycopg.OperationalError):
        users_postgres_module.PostgresUserStore("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """See the agent-store twin of this test (test_agent_store_postgres.py).

    USERS_DATABASE_URL has held a live Neon credential in prod since the account
    persistence fix shipped, so this store had the longest exposure to the leak.
    """
    import dashboard.backend.users_postgres as users_postgres_module

    with pytest.raises(ValueError) as excinfo:
        users_postgres_module.PostgresUserStore('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)


@pg_only
def test_change_password_and_avatar_postgres(pg_client, temp_postgres_store):
    signup = pg_client.post(
        "/api/auth/signup",
        json={"email": "nina@example.com", "display_name": "Nina", "password": "orig-sturdy-pw-1"},
    )
    assert signup.status_code == 200
    assert "token" not in signup.json()
    token_a = _cookie_session_token(pg_client)
    pg_client.post(
        "/api/auth/login",
        json={"email": "nina@example.com", "password": "orig-sturdy-pw-1"},
    )
    token_b = _cookie_session_token(pg_client)

    change = pg_client.post(
        "/api/auth/change-password",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"current_password": "orig-sturdy-pw-1", "new_password": "new-sturdy-pw-2"},
    )
    assert change.status_code == 200

    # Prove the write landed in Postgres and sessions were pruned there.
    user = temp_postgres_store.get_user_by_email("nina@example.com")
    import dashboard.backend.users as users_module

    assert users_module.verify_password("new-sturdy-pw-2", user["password_hash"])
    assert pg_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_a}"}
    ).status_code == 200
    assert pg_client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {token_b}"}
    ).status_code == 401

    # Avatar round-trip against the live Postgres store.
    import base64 as _b64

    tiny_jpeg = _b64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")
    uri = f"data:image/jpeg;base64,{tiny_jpeg}"
    put = pg_client.put(
        "/api/auth/avatar",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"avatar": uri},
    )
    assert put.status_code == 200
    assert temp_postgres_store.get_user_by_email("nina@example.com")["avatar"] == uri

    delete = pg_client.delete(
        "/api/auth/avatar", headers={"Authorization": f"Bearer {token_a}"}
    )
    assert delete.status_code == 200
    assert temp_postgres_store.get_user_by_email("nina@example.com")["avatar"] is None


@pg_only
def test_legacy_plaintext_session_table_is_migrated_postgres(temp_postgres_store):
    """The destructive half of hashed sessions, on the store that keeps data.

    USERS_DATABASE_URL points at Neon in prod, so this DROP takes every live
    login with it -- and unlike the SQLite path it cannot be replayed by
    redeploying. The SQLite twin of this test lives in test_auth.py; both exist
    because "it worked when I ran it once" is not coverage for a DROP.
    """
    import dashboard.backend.users_postgres as users_postgres_module

    user = temp_postgres_store.create_user("legacy@example.com", "Legacy", "securepass1")
    with temp_postgres_store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS auth_sessions")
            cur.execute(
                """
                CREATE TABLE auth_sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                "INSERT INTO auth_sessions (token, user_id, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s)",
                ("legacy-plaintext-token", user["id"], "2026-01-01", "2099-01-01"),
            )

    migrated = users_postgres_module.PostgresUserStore(TEST_POSTGRES_URL)

    with migrated._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'auth_sessions'"
            )
            columns = {row["column_name"] for row in cur.fetchall()}
            cur.execute("SELECT COUNT(*) AS n FROM auth_sessions")
            sessions_left = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM users")
            users_left = cur.fetchone()["n"]

    assert "token_hash" in columns and "token" not in columns
    assert sessions_left == 0, "legacy sessions cannot be re-hashed; they must go"
    assert users_left == 1, "the accounts must survive the session table's DROP"
    assert migrated.get_user_for_token("legacy-plaintext-token") is None


@pg_only
def test_expired_sessions_are_reclaimed_postgres(temp_postgres_store):
    """Twin of the SQLite sweep tests -- see test_auth.py for the rationale."""
    from dashboard.backend.session_tokens import hash_session_token
    import dashboard.backend.users as users_module

    user = temp_postgres_store.create_user("sweep@example.com", "Sweep", "securepass1")
    dead = temp_postgres_store.create_session(user["id"])
    live = temp_postgres_store.create_session(user["id"])
    with temp_postgres_store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE auth_sessions SET expires_at = %s WHERE token_hash = %s",
                (
                    users_module.format_stored_timestamp(
                        users_module._utcnow() - timedelta(days=1)
                    ),
                    hash_session_token(dead),
                ),
            )

    assert temp_postgres_store.purge_expired_sessions() == 1
    assert temp_postgres_store.get_user_for_token(live) is not None


@pg_only
def test_avatar_column_lazy_migration_postgres():
    """A pre-avatar users table gains the column on next store init."""
    require_local_postgres_url(TEST_POSTGRES_URL)
    import dashboard.backend.users_postgres as users_postgres_module

    store = users_postgres_module.PostgresUserStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE users DROP COLUMN IF EXISTS avatar")

    migrated = users_postgres_module.PostgresUserStore(TEST_POSTGRES_URL)  # re-init runs the lazy ALTER
    with migrated._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'users' AND column_name = 'avatar'"
            )
            assert cur.fetchone() is not None


@pg_only
def test_update_display_name_postgres(temp_postgres_store):
    user = temp_postgres_store.create_user("pgname@example.com", "PG Name", "securepass1")

    updated = temp_postgres_store.update_display_name(user["id"], "  PG Renamed  ")

    assert updated["display_name"] == "PG Renamed"
    assert temp_postgres_store.get_user_by_id(user["id"])["display_name"] == "PG Renamed"


@pg_only
def test_update_display_name_missing_user_postgres(temp_postgres_store):
    with pytest.raises(ValueError, match="user_not_found"):
        temp_postgres_store.update_display_name(999_999, "Ghost")


@pg_only
def test_email_change_request_lifecycle_postgres(temp_postgres_store):
    from dashboard.backend.verification_codes import hash_code

    store = temp_postgres_store
    user = store.create_user("pgmail@example.com", "PG Mail", "securepass1")

    row = store.create_email_change_request(user["id"], "  NEXT@Example.COM ", hash_code("A"))
    assert row["stage"] == "old"
    assert row["new_email"] == "next@example.com"
    assert row["attempts"] == 0

    # Hoisted: record_... increments a counter, and -O would strip the call.
    attempts = store.record_email_change_attempt(row["id"])
    assert attempts == 1

    advanced = store.advance_email_change(row["id"], hash_code("Z9Y8X7"))
    assert advanced["stage"] == "new"
    assert advanced["attempts"] == 0
    assert advanced["code_hash"] == hash_code("Z9Y8X7")

    store.mark_email_change_used(row["id"])
    assert store.get_active_email_change(user["id"]) is None
    assert store.last_email_change_request_at(user["id"]) is not None
    assert store.last_email_change_completed_at(user["id"]) is not None

    store.cancel_email_change(user["id"])
    assert store.get_active_email_change(user["id"]) is None
    # Deactivated, not deleted -- the cooldown clock must survive a cancel.
    assert store.last_email_change_request_at(user["id"]) is not None
    # And cancel must not relabel the completed change as a cancelled one.
    assert store.last_email_change_completed_at(user["id"]) is not None


@pg_only
def test_email_change_request_log_is_append_only_postgres(temp_postgres_store):
    """The daily cap and the 7-day interval both read history off this table.

    Mirrors the SQLite twin's supersede-but-retain and windowing cases. Worth
    running against real Postgres rather than trusting parity: the created_at
    window is a lexicographic TEXT comparison, which is a property of the column
    type in each engine, not of the shared Python above it.
    """
    import dashboard.backend.users as users_module
    from dashboard.backend.verification_codes import hash_code

    def stored_time(**delta):
        return users_module.format_stored_timestamp(users_module._utcnow() - timedelta(**delta))

    store = temp_postgres_store
    user = store.create_user("pgwindow@example.com", "PG Window", "securepass1")

    first = store.create_email_change_request(user["id"], "one@example.com", hash_code("A"))
    store.create_email_change_request(user["id"], "two@example.com", hash_code("B"))

    # Superseded, not deleted.
    day = stored_time(days=1)
    assert len(store.email_change_request_times_since(user["id"], day)) == 2

    # ...and the older one is no longer active.
    assert store.get_active_email_change(user["id"])["new_email"] == "two@example.com"

    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE email_change_requests SET created_at = %s WHERE id = %s",
                (stored_time(days=2), first["id"]),
            )

    within_a_day = store.email_change_request_times_since(user["id"], day)
    assert len(within_a_day) == 1
    assert within_a_day == sorted(within_a_day)


@pg_only
def test_update_email_postgres(temp_postgres_store):
    store = temp_postgres_store
    user = store.create_user("pgold@example.com", "PG Old", "securepass1")

    updated = store.update_email(user["id"], "  PGNEW@Example.COM  ")

    # Lowercasing is mandatory here: this twin's UNIQUE is NOT case-insensitive,
    # so an un-normalized write would let two casings of one address coexist in
    # prod while SQLite rejects them locally.
    assert updated["email"] == "pgnew@example.com"
    assert store.get_user_by_email("pgnew@example.com") is not None


@pg_only
def test_update_email_conflict_postgres(temp_postgres_store):
    store = temp_postgres_store
    user = store.create_user("pgmine@example.com", "PG Mine", "securepass1")
    store.create_user("pgtaken@example.com", "PG Taken", "securepass1")

    with pytest.raises(ValueError, match="email_already_registered"):
        store.update_email(user["id"], "pgtaken@example.com")


@pg_only
def test_entitlements_upsert_partial_patch_postgres(temp_postgres_store):
    """The ``::integer`` casts + COALESCE merge, against real Postgres.

    The whole local suite runs on SQLite, so psycopg's None -> OID-0 trap
    (Postgres refuses an untyped NULL inside COALESCE) is invisible anywhere
    but this tier — removing a cast passes every other test and 500s in prod.
    """
    store = temp_postgres_store
    user = store.create_user("pgent@example.com", "PG Ent", "securepass1")

    import dashboard.backend.users as users_module

    assert (
        store.get_entitlements(user["id"])["max_concurrent_backtests"]
        == users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS
    )

    first = store.set_entitlements(
        user["id"], max_concurrent_backtests=7, credits=250
    )
    assert first["max_concurrent_backtests"] == 7
    assert first["credits"] == 250

    # Partial patch: the omitted field must ride COALESCE, not a stale read.
    only_credits = store.set_entitlements(user["id"], credits=999)
    assert only_credits["credits"] == 999
    assert only_credits["max_concurrent_backtests"] == 7


@pg_only
def test_entitlements_missing_user_is_user_not_found_postgres(temp_postgres_store):
    """The enforced FK is the existence check on this twin; its violation
    must surface as user_not_found (the SQLite twin's answer), not a 500."""
    store = temp_postgres_store
    with pytest.raises(ValueError, match="user_not_found"):
        store.set_entitlements(999_999, credits=5)


@pg_only
def test_credit_spend_and_refund_postgres(temp_postgres_store):
    """The credit ledger against real Postgres.

    Every parameter in these three statements is cast ``::integer``, and the
    seed is the reason: ``INSERT ... SELECT $1`` resolves parameter types from
    the SELECT alone, without the insert target's columns, so an uncast
    parameter is a hard 42P08 on the very first spend. LEAST()'s polymorphism
    is the same trap in the refund. The SQLite twin runs all three clean
    either way, so dropping a cast reddens nothing outside this tier and 500s
    every metered backtest in prod.
    """
    import dashboard.backend.users as users_module

    store = temp_postgres_store
    user = store.create_user("pgcredit@example.com", "PG Credit", "securepass1")

    # Seeds the defaults, then debits — the INSERT ... SELECT path.
    assert store.try_spend_credits(user["id"]) == users_module.DEFAULT_CREDITS - 1
    # Second spend takes the ON CONFLICT DO NOTHING path instead.
    assert store.try_spend_credits(user["id"]) == users_module.DEFAULT_CREDITS - 2

    # A metered spend is not an admin edit.
    ent = store.get_entitlements(user["id"])
    assert ent["updated_at"] is None
    assert ent["updated_by_admin_id"] is None

    store.set_entitlements(user["id"], credits=1, updated_by_admin_id=user["id"])
    assert store.try_spend_credits(user["id"]) == 0
    assert store.try_spend_credits(user["id"]) is None

    assert store.refund_credits(user["id"]) == 1
    store.set_entitlements(user["id"], credits=users_module.MAX_CREDITS_CAP)
    assert store.refund_credits(user["id"]) == users_module.MAX_CREDITS_CAP


@pg_only
def test_credit_spend_for_missing_user_postgres(temp_postgres_store):
    """The seed selects from ``users``, so it inserts nothing rather than
    tripping the FK — a refusal, not the user_not_found that ``set_entitlements``
    raises. Both twins answer the same way for the same reason."""
    store = temp_postgres_store
    assert store.try_spend_credits(999_999) is None
    assert store.refund_credits(999_999) is None


@pg_only
def test_apply_admin_patch_atomic_postgres(temp_postgres_store):
    store = temp_postgres_store
    admin = store.create_user("pgboss@example.com", "PG Boss", "securepass1")
    store.apply_admin_patch(admin["id"], role="admin")
    target = store.create_user("pgmember@example.com", "PG Member", "securepass1")

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

    demoted = store.apply_admin_patch(target["id"], role="user")
    assert demoted["role"] == "user"
    # Last-admin guard, serialized by the advisory lock.
    with pytest.raises(ValueError, match="last_admin"):
        store.apply_admin_patch(admin["id"], role="user")


@pg_only
def test_admin_counts_and_listing_postgres(temp_postgres_store):
    store = temp_postgres_store
    first = store.create_user("pgc1@example.com", "C1", "securepass1")
    store.create_user("pgc2@example.com", "C2", "securepass1")
    store.apply_admin_patch(first["id"], role="admin")

    assert store.count_users_and_admins() == {"users": 2, "admins": 1}

    listed = store.list_users_admin()
    by_email = {row["email"]: row for row in listed}
    assert set(by_email) == {"pgc1@example.com", "pgc2@example.com"}
    # No entitlements row yet -> defaults from the LEFT JOIN miss.
    import dashboard.backend.users as users_module

    assert (
        by_email["pgc2@example.com"]["entitlements"]["max_concurrent_backtests"]
        == users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS
    )
    assert "avatar" not in by_email["pgc1@example.com"]


@pg_only
def test_admin_search_is_trimmed_case_insensitive_and_escaped_postgres(
    temp_postgres_store,
):
    store = temp_postgres_store
    store.create_user("pgalice@example.com", "Research Alice", "securepass1")
    store.create_user("pgbob@example.com", "Operations", "securepass1")
    store.create_user("pgpercent%name@example.com", "Percent Name", "securepass1")

    assert [row["email"] for row in store.list_users_admin(query="  ALICE  ")] == [
        "pgalice@example.com"
    ]
    assert store.count_users(query="research") == 1
    assert store.list_users_admin(query="%")[0]["email"] == (
        "pgpercent%name@example.com"
    )


@pg_only
def test_promote_first_admin_is_one_shot_postgres(temp_postgres_store):
    """Prod's copy of the bootstrap predicate, which had no test at all.

    ``promote_first_admin`` appeared zero times in this file: the one-shot
    guard — the thing standing between "no admin yet" and "anyone with the
    secret is admin" — was only ever exercised on SQLite, while every
    deployment with USERS_DATABASE_URL set runs this twin. The twin-parity
    guard cannot cover it either: it compares method names and signatures, and
    never reads a WHERE clause.
    """
    store = temp_postgres_store
    first = store.create_user("pgboot1@example.com", "Boot One", "securepass1")
    second = store.create_user("pgboot2@example.com", "Boot Two", "securepass1")

    promoted = store.promote_first_admin(first["id"])
    assert promoted["role"] == "admin"
    assert store.get_user_by_id(first["id"])["role"] == "admin"

    # Inert the moment ANY admin exists — including for the caller who already
    # used it, whose retry must not silently succeed a second time.
    with pytest.raises(ValueError, match="admin_exists"):
        store.promote_first_admin(second["id"])
    with pytest.raises(ValueError, match="admin_exists"):
        store.promote_first_admin(first["id"])
    assert store.get_user_by_id(second["id"])["role"] == "user"
    assert store.count_users_and_admins()["admins"] == 1


@pg_only
def test_promote_first_admin_unknown_user_postgres(temp_postgres_store):
    """No admin exists, so the one-shot check passes and the UPDATE matches
    nothing — that has to be user_not_found, not a silent success."""
    store = temp_postgres_store
    with pytest.raises(ValueError, match="user_not_found"):
        store.promote_first_admin(999_999)
    assert store.count_users_and_admins()["admins"] == 0


@pg_only
def test_zero_quota_round_trips_postgres(temp_postgres_store):
    """0 is the suspend value; ``COALESCE(%s::integer, …)`` must not read it
    as "field omitted" and restore the previous quota."""
    store = temp_postgres_store
    user = store.create_user("pgzero@example.com", "PG Zero", "securepass1")
    store.set_entitlements(user["id"], max_concurrent_backtests=7)
    zeroed = store.set_entitlements(user["id"], max_concurrent_backtests=0)
    assert zeroed["max_concurrent_backtests"] == 0
    assert store.get_entitlements(user["id"])["max_concurrent_backtests"] == 0


def test_store_twins_expose_the_same_public_surface():
    """A method in one twin and not the other is a prod-only crash.

    This compares classes, not instances, so it needs no live Postgres -- which
    matters because every @pg_only case above fails open when TEST_POSTGRES_URL
    is unset.
    """
    import dashboard.backend.users as users_module
    import dashboard.backend.users_postgres as users_postgres_module

    def public_methods(cls):
        return {
            name
            for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name))
        }

    sqlite_methods = public_methods(users_module.UserStore)
    postgres_methods = public_methods(users_postgres_module.PostgresUserStore)
    assert sqlite_methods == postgres_methods


@pg_only
def test_password_reset_request_lifecycle_postgres(temp_postgres_store):
    """Mirrors the SQLite twin's password-reset store cases (#187): supersede
    leaves one active row, the used CAS has a single winner, and the
    status-blind reads still see the whole log."""
    import dashboard.backend.users as users_module
    from dashboard.backend.verification_codes import hash_code

    store = temp_postgres_store
    user = store.create_user("reset-pg@example.com", "Reset PG", "securepass1")

    row = store.create_password_reset_request(user["id"], hash_code("A"))
    assert row["attempts"] == 0
    assert store.get_active_password_reset(user["id"])["id"] == row["id"]
    assert store.record_password_reset_attempt(row["id"]) == 1

    newer = store.create_password_reset_request(user["id"], hash_code("B"))
    active = store.get_active_password_reset(user["id"])
    assert active["id"] == newer["id"]
    assert active["code_hash"] == hash_code("B")

    assert store.mark_password_reset_used(newer["id"]) is True
    assert store.mark_password_reset_used(newer["id"]) is False
    assert store.get_active_password_reset(user["id"]) is None

    assert store.last_password_reset_request_at(user["id"]) is not None
    day = users_module.format_stored_timestamp(
        users_module._utcnow() - timedelta(days=1)
    )
    assert len(store.password_reset_request_times_since(user["id"], day)) == 2

    store.cancel_password_reset(user["id"])
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT used_at, cancelled_at FROM password_reset_requests WHERE id = %s",
                (newer["id"],),
            )
            stored = cur.fetchone()
    # cancel is scoped to still-active rows: the used row keeps its used_at.
    assert stored["used_at"] is not None
    assert stored["cancelled_at"] is None


@pg_only
def test_password_reset_attempt_cap_cancels_postgres(temp_postgres_store):
    import dashboard.backend.users as users_module
    from dashboard.backend.verification_codes import hash_code

    store = temp_postgres_store
    user = store.create_user("cap-pg@example.com", "Cap PG", "securepass1")
    row = store.create_password_reset_request(user["id"], hash_code("A"))

    cap = users_module.PASSWORD_RESET_MAX_ATTEMPTS
    for expected in range(1, cap + 1):
        assert store.record_password_reset_attempt(row["id"]) == expected
    assert store.get_active_password_reset(user["id"]) is None
    # The conditional UPDATE refuses to push past the cap.
    assert store.record_password_reset_attempt(row["id"]) == cap
