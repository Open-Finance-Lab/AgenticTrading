"""SQLite persistence contract for the admin-only user-group dimension."""

import sqlite3
from pathlib import Path

import pytest

from dashboard.backend.domain.user_groups import DEFAULT_USER_GROUP
from dashboard.backend.users import MAX_CREDITS_CAP, UserStore


@pytest.fixture
def store(tmp_path: Path) -> UserStore:
    return UserStore(db_path=tmp_path / "users.db")


def _column(store: UserStore, name: str) -> dict:
    conn = store._get_connection()
    try:
        rows = conn.execute("PRAGMA table_info(users)").fetchall()
    finally:
        conn.close()
    return dict(next(row for row in rows if row["name"] == name))


def test_fresh_users_table_declares_required_unknown_group(store: UserStore):
    column = _column(store, "user_group")

    assert column["notnull"] == 1
    assert column["dflt_value"] == "'unknown'"


def test_existing_users_table_gets_unknown_group_column(tmp_path: Path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, display_name TEXT, "
        "password_hash TEXT, role TEXT, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO users VALUES (1, 'legacy@example.test', 'Legacy', 'hash', "
        "'user', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    migrated = UserStore(db_path=path)

    assert migrated.get_user_admin(1)["user_group"] == "unknown"
    column = _column(migrated, "user_group")
    assert column["notnull"] == 1
    assert column["dflt_value"] == "'unknown'"


@pytest.mark.parametrize("junk", ["organic ", "PARTNER", "friends", ""])
def test_rows_holding_an_unparseable_group_are_repaired_on_the_next_boot(
    tmp_path: Path, junk: str
):
    """The data half of making the Python constant the owner of this column.

    Without this repair the column keeps a value parse_user_group() rejects, and
    the only thing that would ever fix it is an admin happening to re-save that
    one account. Reads coerce, so the bad row is invisible on screen -- which is
    exactly why it has to be repaired in the store rather than left to the read
    path.
    """
    path = tmp_path / "junk.db"
    UserStore(db_path=path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO users (id, email, display_name, password_hash, role, user_group) "
        "VALUES (1, 'junk@example.test', 'Junk', 'hash', 'user', ?)",
        (junk,),
    )
    conn.commit()
    conn.close()

    reopened = UserStore(db_path=path)

    stored = reopened._get_connection()
    try:
        row = stored.execute("SELECT user_group FROM users WHERE id = 1").fetchone()
    finally:
        stored.close()
    assert row["user_group"] == DEFAULT_USER_GROUP


def test_a_canonical_group_survives_the_repair(tmp_path: Path):
    path = tmp_path / "kept.db"
    store = UserStore(db_path=path)
    user = store.create_user("kept@example.test", "Kept", "SecurePass1!")
    store.apply_admin_patch(user["id"], user_group="competition")

    reopened = UserStore(db_path=path)
    assert reopened.get_user_admin(user["id"])["user_group"] == "competition"


def test_new_users_are_written_into_the_default_group_not_left_to_the_column(
    store: UserStore,
):
    """create_user names the group explicitly (see users.py for why).

    Reading the stored row rather than the admin projection is the point: the
    projection coerces, so it would report the default even if the INSERT had
    written nothing and a stale column DEFAULT had supplied something else.
    """
    created = store.create_user("group@example.test", "Group", "SecurePass1!")

    assert "user_group" not in created
    conn = store._get_connection()
    try:
        row = conn.execute(
            "SELECT user_group FROM users WHERE id = ?", (created["id"],)
        ).fetchone()
    finally:
        conn.close()
    assert row["user_group"] == DEFAULT_USER_GROUP
    assert store.get_user_admin(created["id"])["user_group"] == DEFAULT_USER_GROUP
    assert store.list_users_admin()[0]["user_group"] == DEFAULT_USER_GROUP


def test_store_patch_updates_group_and_entitlements_atomically(store: UserStore):
    user = store.create_user("atomic@example.test", "Atomic", "SecurePass1!")

    updated = store.apply_admin_patch(
        user["id"],
        role="admin",
        user_group=" ORGANIC ",
        credits=7,
        max_concurrent_backtests=2,
    )

    assert updated["role"] == "admin"
    assert updated["user_group"] == "organic"
    assert updated["entitlements"]["credits"] == 7
    assert updated["entitlements"]["max_concurrent_backtests"] == 2
    assert store.get_user_admin(user["id"])["user_group"] == "organic"


def test_invalid_or_omitted_group_cannot_overwrite_stored_group(store: UserStore):
    user = store.create_user("preserve@example.test", "Preserve", "SecurePass1!")
    store.apply_admin_patch(user["id"], user_group="partner")

    with pytest.raises(ValueError, match="invalid_user_group"):
        store.apply_admin_patch(user["id"], user_group="friends")

    unchanged = store.apply_admin_patch(user["id"], credits=9)
    assert unchanged["user_group"] == "partner"


def test_failed_multi_field_patch_rolls_group_back(store: UserStore):
    user = store.create_user("rollback@example.test", "Rollback", "SecurePass1!")
    store.apply_admin_patch(user["id"], user_group="invited")

    with pytest.raises(ValueError, match="invalid_credits"):
        store.apply_admin_patch(
            user["id"],
            user_group="competition",
            credits=MAX_CREDITS_CAP + 1,
        )

    assert store.get_user_admin(user["id"])["user_group"] == "invited"


def test_malformed_stored_group_is_coerced_in_admin_projection(store: UserStore):
    user = store.create_user("malformed@example.test", "Malformed", "SecurePass1!")
    conn = store._get_connection()
    conn.execute(
        "UPDATE users SET user_group = ? WHERE id = ?",
        ("legacy-source", user["id"]),
    )
    conn.commit()
    conn.close()

    assert store.get_user_admin(user["id"])["user_group"] == "unknown"


def test_auth_me_does_not_expose_admin_only_group(store: UserStore, monkeypatch):
    from fastapi.testclient import TestClient

    from dashboard.backend import users as users_module
    from dashboard.backend.app import app

    user = store.create_user("private@example.test", "Private", "SecurePass1!")
    store.apply_admin_patch(user["id"], user_group="internal")
    token = store.create_session(user["id"])
    monkeypatch.setattr(users_module, "user_store", store)

    with TestClient(app) as client:
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200, response.text
    assert "user_group" not in response.json()["user"]
