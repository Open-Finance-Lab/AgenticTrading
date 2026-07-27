"""UserStore (SQLite twin) behaviour.

The Postgres twin mirrors every case here under @pg_only in
test_users_postgres.py -- a method that exists in one twin and not the other is
a prod-only crash.
"""

import tempfile
from pathlib import Path

import pytest

from dashboard.backend.users import UserStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield UserStore(db_path=Path(tmpdir) / "users.db")


@pytest.fixture
def user(store):
    return store.create_user("owner@example.com", "Owner", "securepass1")


def test_update_display_name_persists_and_returns_public_user(store, user):
    updated = store.update_display_name(user["id"], "Renamed")

    assert updated["display_name"] == "Renamed"
    assert "password_hash" not in updated
    assert store.get_user_by_id(user["id"])["display_name"] == "Renamed"


def test_update_display_name_strips_surrounding_whitespace(store, user):
    updated = store.update_display_name(user["id"], "  Padded  ")
    assert updated["display_name"] == "Padded"


def test_update_display_name_rejects_a_missing_user(store):
    with pytest.raises(ValueError, match="user_not_found"):
        store.update_display_name(999_999, "Ghost")
