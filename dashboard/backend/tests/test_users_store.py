"""UserStore (SQLite twin) behaviour.

The Postgres twin mirrors every case here under @pg_only in
test_users_postgres.py -- a method that exists in one twin and not the other is
a prod-only crash.
"""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from dashboard.backend.users import (
    EMAIL_CHANGE_TTL_MINUTES,
    UserStore,
    _utcnow,
    parse_stored_timestamp,
)
from dashboard.backend.verification_codes import hash_code


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


def test_create_email_change_request_starts_at_stage_old(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("ABC234"))

    assert row["stage"] == "old"
    assert row["new_email"] == "next@example.com"
    assert row["attempts"] == 0
    assert row["used_at"] is None
    expires = parse_stored_timestamp(row["expires_at"])
    assert timedelta(minutes=EMAIL_CHANGE_TTL_MINUTES - 1) < expires - _utcnow()


def test_create_email_change_request_normalizes_the_new_email(store, user):
    row = store.create_email_change_request(user["id"], "  MiXeD@Example.COM ", hash_code("A"))
    assert row["new_email"] == "mixed@example.com"


def test_create_email_change_request_replaces_any_prior_request(store, user):
    store.create_email_change_request(user["id"], "first@example.com", hash_code("A"))
    store.create_email_change_request(user["id"], "second@example.com", hash_code("B"))

    active = store.get_active_email_change(user["id"])
    assert active["new_email"] == "second@example.com"


def test_get_active_email_change_is_none_without_a_request(store, user):
    assert store.get_active_email_change(user["id"]) is None


def test_get_active_email_change_ignores_an_expired_request(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    stale = (_utcnow() - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    conn = store._get_connection()
    conn.execute(
        "UPDATE email_change_requests SET expires_at = ? WHERE id = ?", (stale, row["id"])
    )
    conn.commit()
    conn.close()

    assert store.get_active_email_change(user["id"]) is None


def test_advance_email_change_moves_to_stage_new_and_resets_attempts(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    store.record_email_change_attempt(row["id"])

    advanced = store.advance_email_change(row["id"], hash_code("Z9Y8X7"))

    assert advanced["stage"] == "new"
    assert advanced["code_hash"] == hash_code("Z9Y8X7")
    assert advanced["attempts"] == 0
    assert advanced["new_email"] == "next@example.com"


def test_record_email_change_attempt_increments_and_returns_the_count(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    assert store.record_email_change_attempt(row["id"]) == 1
    assert store.record_email_change_attempt(row["id"]) == 2


def test_mark_email_change_used_deactivates_but_keeps_the_row(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    store.mark_email_change_used(row["id"])

    assert store.get_active_email_change(user["id"]) is None
    # Still visible to the cooldown, so a completed change cannot be immediately
    # followed by another.
    assert store.last_email_change_request_at(user["id"]) is not None


def test_cancel_email_change_deactivates_but_preserves_the_cooldown(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))

    store.cancel_email_change(user["id"])

    assert store.get_active_email_change(user["id"]) is None
    # The cooldown clock must survive a cancel: otherwise an authenticated caller
    # who knows the password could loop request/cancel/request with the cooldown
    # never enforced, mail-bombing the account and burning the shared quota.
    assert store.last_email_change_request_at(user["id"]) == row["created_at"]


def test_last_email_change_request_at_is_none_without_a_request(store, user):
    assert store.last_email_change_request_at(user["id"]) is None


def test_update_email_persists_lowercased(store, user):
    updated = store.update_email(user["id"], "  NEW@Example.COM  ")

    assert updated["email"] == "new@example.com"
    assert store.get_user_by_email("new@example.com") is not None


def test_update_email_rejects_an_address_another_account_owns(store, user):
    store.create_user("taken@example.com", "Taken", "securepass1")

    with pytest.raises(ValueError, match="email_already_registered"):
        store.update_email(user["id"], "taken@example.com")

    # The original address is untouched.
    assert store.get_user_by_id(user["id"])["email"] == "owner@example.com"


def test_update_email_rejects_a_missing_user(store):
    with pytest.raises(ValueError, match="user_not_found"):
        store.update_email(999_999, "ghost@example.com")
