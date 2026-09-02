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
    format_stored_timestamp,
    parse_stored_timestamp,
)
from dashboard.backend.verification_codes import hash_code


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield UserStore(db_path=Path(tmpdir) / "users.db")


def _ago(**delta) -> str:
    """A stored-format timestamp `delta` in the past."""
    return format_stored_timestamp(_utcnow() - timedelta(**delta))


def _backdate_request(store, table, request_id, **columns) -> None:
    """Rewrite timestamp columns on one request row.

    The windows under test are hours and days wide, so the alternative is
    freezing the clock. Reaching into the table keeps these tests about the
    queries rather than about a time-mocking layer.
    """
    assignments = ", ".join(f"{name} = ?" for name in columns)
    conn = store._get_connection()
    conn.execute(
        f"UPDATE {table} SET {assignments} WHERE id = ?",  # noqa: S608
        (*columns.values(), request_id),
    )
    conn.commit()
    conn.close()


def _backdate_email_change(store, request_id, **columns) -> None:
    _backdate_request(store, "email_change_requests", request_id, **columns)


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


def test_create_email_change_request_supersedes_but_retains_the_prior_request(store, user):
    store.create_email_change_request(user["id"], "first@example.com", hash_code("A"))
    store.create_email_change_request(user["id"], "second@example.com", hash_code("B"))

    active = store.get_active_email_change(user["id"])
    assert active["new_email"] == "second@example.com"

    # ORDER BY id DESC alone satisfies the assertion above, so check the state of
    # the superseded row directly. It must survive -- deleting it would erase the
    # created_at that the rolling daily cap counts and the used_at that the
    # 7-day interval reads, so making a request would clear both limits -- and it
    # must be inactive, or "how many are open" stops meaning anything.
    conn = store._get_connection()
    rows = conn.execute(
        """
        SELECT new_email, cancelled_at FROM email_change_requests
        WHERE user_id = ? ORDER BY id ASC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()
    assert [row["new_email"] for row in rows] == ["first@example.com", "second@example.com"]
    assert rows[0]["cancelled_at"] is not None
    assert rows[1]["cancelled_at"] is None


def test_cancel_email_change_leaves_an_already_used_row_alone(store, user):
    """A completed change must not be relabelled as a cancelled one.

    last_email_change_completed_at reads used_at, so this is not merely cosmetic
    bookkeeping in a log now kept for audit.
    """
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    store.mark_email_change_used(row["id"])

    store.cancel_email_change(user["id"])

    conn = store._get_connection()
    stored = conn.execute(
        "SELECT used_at, cancelled_at FROM email_change_requests WHERE id = ?",
        (row["id"],),
    ).fetchone()
    conn.close()
    assert stored["used_at"] is not None
    assert stored["cancelled_at"] is None


def test_last_email_change_completed_at_is_none_until_one_completes(store, user):
    row = store.create_email_change_request(user["id"], "next@example.com", hash_code("A"))
    assert store.last_email_change_completed_at(user["id"]) is None

    store.cancel_email_change(user["id"])
    # A cancelled request is not a completed change.
    assert store.last_email_change_completed_at(user["id"]) is None

    store.mark_email_change_used(row["id"])
    assert store.last_email_change_completed_at(user["id"]) is not None


def test_last_email_change_completed_at_orders_by_completion_not_by_id(store, user):
    first = store.create_email_change_request(user["id"], "one@example.com", hash_code("A"))
    second = store.create_email_change_request(user["id"], "two@example.com", hash_code("B"))
    store.mark_email_change_used(first["id"])
    store.mark_email_change_used(second["id"])

    # The higher-id row completed LONG ago; the lower-id row completed recently.
    # ORDER BY id DESC would return the stale one and hand the caller a 7-day
    # clock that expired weeks back.
    recent = _ago(hours=1)
    _backdate_email_change(store, second["id"], used_at=_ago(days=30))
    _backdate_email_change(store, first["id"], used_at=recent)

    assert store.last_email_change_completed_at(user["id"]) == recent


def test_email_change_request_times_since_windows_by_created_at(store, user):
    old = store.create_email_change_request(user["id"], "old@example.com", hash_code("A"))
    _backdate_email_change(store, old["id"], created_at=_ago(days=2))
    store.create_email_change_request(user["id"], "new@example.com", hash_code("B"))

    within_a_day = store.email_change_request_times_since(user["id"], _ago(hours=24))
    assert len(within_a_day) == 1

    # Cancelled and used rows still count: the cap exists to bound messages
    # already sent, and cancelling does not un-send them.
    everything = store.email_change_request_times_since(user["id"], _ago(days=7))
    assert len(everything) == 2
    assert everything == sorted(everything)  # oldest first, so [0] is when the window frees


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

    first_attempt = store.record_email_change_attempt(row["id"])
    second_attempt = store.record_email_change_attempt(row["id"])

    assert first_attempt == 1
    assert second_attempt == 2


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


# --- password_reset_requests (#187) -----------------------------------------


def _backdate_password_reset(store, request_id, **columns) -> None:
    _backdate_request(store, "password_reset_requests", request_id, **columns)


def test_create_password_reset_request_leaves_exactly_one_active_row(store, user):
    store.create_password_reset_request(user["id"], hash_code("A"))
    newer = store.create_password_reset_request(user["id"], hash_code("B"))

    active = store.get_active_password_reset(user["id"])
    assert active["id"] == newer["id"]
    assert active["code_hash"] == hash_code("B")
    assert active["attempts"] == 0

    # The superseded row survives (the cooldown reads it) but is inactive.
    conn = store._get_connection()
    rows = conn.execute(
        """
        SELECT code_hash, cancelled_at FROM password_reset_requests
        WHERE user_id = ? ORDER BY id ASC
        """,
        (user["id"],),
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert rows[0]["cancelled_at"] is not None
    assert rows[1]["cancelled_at"] is None


def test_get_active_password_reset_folds_expiry_into_no_active_row(store, user):
    row = store.create_password_reset_request(user["id"], hash_code("A"))
    _backdate_password_reset(store, row["id"], expires_at=_ago(minutes=1))

    assert store.get_active_password_reset(user["id"]) is None
    # ...while the status-blind cooldown read still sees the row.
    assert store.last_password_reset_request_at(user["id"]) == row["created_at"]


def test_record_password_reset_attempt_cancels_at_the_cap(store, user):
    from dashboard.backend.users import PASSWORD_RESET_MAX_ATTEMPTS

    row = store.create_password_reset_request(user["id"], hash_code("A"))

    for expected in range(1, PASSWORD_RESET_MAX_ATTEMPTS):
        assert store.record_password_reset_attempt(row["id"]) == expected
        assert store.get_active_password_reset(user["id"]) is not None

    assert store.record_password_reset_attempt(row["id"]) == PASSWORD_RESET_MAX_ATTEMPTS
    assert store.get_active_password_reset(user["id"]) is None

    # Further attempts never push past the cap.
    assert store.record_password_reset_attempt(row["id"]) == PASSWORD_RESET_MAX_ATTEMPTS
    conn = store._get_connection()
    attempts = conn.execute(
        "SELECT attempts FROM password_reset_requests WHERE id = ?", (row["id"],)
    ).fetchone()["attempts"]
    conn.close()
    assert attempts == PASSWORD_RESET_MAX_ATTEMPTS


def test_record_password_reset_attempt_cap_holds_under_concurrency(store, user):
    import threading

    from dashboard.backend.users import PASSWORD_RESET_MAX_ATTEMPTS

    row = store.create_password_reset_request(user["id"], hash_code("A"))

    def spin():
        for _ in range(3):
            store.record_password_reset_attempt(row["id"])

    threads = [threading.Thread(target=spin) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    conn = store._get_connection()
    attempts = conn.execute(
        "SELECT attempts FROM password_reset_requests WHERE id = ?", (row["id"],)
    ).fetchone()["attempts"]
    conn.close()
    # 12 racing attempts; the SQL predicate is what bounds the count.
    assert attempts == PASSWORD_RESET_MAX_ATTEMPTS


def test_mark_password_reset_used_is_a_single_winner_cas(store, user):
    row = store.create_password_reset_request(user["id"], hash_code("A"))

    assert store.mark_password_reset_used(row["id"]) is True
    assert store.mark_password_reset_used(row["id"]) is False

    assert store.get_active_password_reset(user["id"]) is None
    assert store.last_password_reset_request_at(user["id"]) is not None


def test_cancel_password_reset_leaves_a_used_row_alone(store, user):
    row = store.create_password_reset_request(user["id"], hash_code("A"))
    store.mark_password_reset_used(row["id"])

    store.cancel_password_reset(user["id"])

    conn = store._get_connection()
    stored = conn.execute(
        "SELECT used_at, cancelled_at FROM password_reset_requests WHERE id = ?",
        (row["id"],),
    ).fetchone()
    conn.close()
    assert stored["used_at"] is not None
    assert stored["cancelled_at"] is None


def test_password_reset_request_times_since_is_status_blind(store, user):
    old = store.create_password_reset_request(user["id"], hash_code("A"))
    _backdate_password_reset(store, old["id"], created_at=_ago(days=2))
    store.create_password_reset_request(user["id"], hash_code("B"))

    # The first row was superseded (cancelled); both still count.
    assert len(store.password_reset_request_times_since(user["id"], _ago(days=7))) == 2
    within_a_day = store.password_reset_request_times_since(user["id"], _ago(hours=24))
    assert len(within_a_day) == 1
    assert within_a_day == sorted(within_a_day)
