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
"""

import os

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
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
            cur.execute("DELETE FROM auth_sessions")
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
    assert signup_data["token"]

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
    token = login.json()["token"]

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

    from dashboard.backend.users_postgres import PostgresUserStore

    with pytest.raises(psycopg.OperationalError):
        PostgresUserStore("postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2")


def test_malformed_url_is_rejected_before_psycopg_can_echo_it():
    """See the agent-store twin of this test (test_agent_store_postgres.py).

    USERS_DATABASE_URL has held a live Neon credential in prod since the account
    persistence fix shipped, so this store had the longest exposure to the leak.
    """
    from dashboard.backend.users_postgres import PostgresUserStore

    with pytest.raises(ValueError) as excinfo:
        PostgresUserStore('"postgresql://u:sup3r-s3cret@ep-x.neon.tech/atl"')
    assert "sup3r-s3cret" not in str(excinfo.value)


@pg_only
def test_change_password_and_avatar_postgres(pg_client, temp_postgres_store):
    signup = pg_client.post(
        "/api/auth/signup",
        json={"email": "nina@example.com", "display_name": "Nina", "password": "orig-sturdy-pw-1"},
    )
    assert signup.status_code == 200
    token_a = signup.json()["token"]
    token_b = pg_client.post(
        "/api/auth/login",
        json={"email": "nina@example.com", "password": "orig-sturdy-pw-1"},
    ).json()["token"]

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
def test_avatar_column_lazy_migration_postgres():
    """A pre-avatar users table gains the column on next store init."""
    require_local_postgres_url(TEST_POSTGRES_URL)
    from dashboard.backend.users_postgres import PostgresUserStore

    store = PostgresUserStore(TEST_POSTGRES_URL)
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE users DROP COLUMN IF EXISTS avatar")

    migrated = PostgresUserStore(TEST_POSTGRES_URL)  # re-init runs the lazy ALTER
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
