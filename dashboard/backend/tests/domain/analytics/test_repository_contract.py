"""Shared behavioral contract for Analytics persistence backends."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dashboard.backend.domain.analytics.models import AnalyticsEventRecord
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.repository_common import (
    AnalyticsIdempotencyConflictError,
    AnalyticsStoreError,
    decode_event_cursor,
)
from dashboard.backend.users import UserStore


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def event_record(user_id: int, **overrides) -> AnalyticsEventRecord:
    value = {
        "event_id": str(uuid4()),
        "schema_version": 1,
        "event_name": "page_viewed",
        "event_group": "experience",
        "user_id": user_id,
        "session_id": str(uuid4()),
        "occurred_at": NOW,
        "received_at": NOW + timedelta(seconds=1),
        "event_source": "frontend",
        "page_view": "home",
        "device_category": "desktop",
        "browser_family": "Chrome",
        "properties": {},
    }
    value.update(overrides)
    return AnalyticsEventRecord.model_validate(value)


@pytest.fixture
def sqlite_contract(tmp_path):
    db_path = tmp_path / "analytics.db"
    users = UserStore(db_path=db_path)
    admin = users.create_user(
        "analytics-admin@example.test",
        "Analytics Admin",
        "SecurePass1!",
    )
    target = users.create_user(
        "analytics-user@example.test",
        "Analytics User",
        "SecurePass1!",
    )
    users.apply_admin_patch(admin["id"], role="admin")
    store = AnalyticsStore(db_path=db_path)
    return store, int(admin["id"]), int(target["id"])


def assert_event_idempotency_contract(store, user_id):
    event = event_record(
        user_id,
        event_id="10000000-0000-4000-8000-000000000001",
    )
    first = store.append_event(event)
    replay = store.append_event(
        event.model_copy(update={"received_at": event.received_at + timedelta(seconds=5)})
    )
    assert first.created is True
    assert replay.created is False
    assert replay.event == first.event

    changed = event.model_copy(update={"page_view": "credits"})
    with pytest.raises(AnalyticsIdempotencyConflictError):
        store.append_event(changed)


def assert_source_event_idempotency_contract(store, user_id):
    event = event_record(
        user_id,
        event_id="10000000-0000-4000-8000-000000000002",
        event_source="server",
        source_event_id="run:run_123:completed",
        source_record_type="run",
        source_record_id="run_123",
        event_name="backtest_completed",
        event_group="run",
        page_view=None,
        session_id=None,
        outcome="succeeded",
    )
    assert store.append_event(event).created is True
    replay = event.model_copy(
        update={
            "event_id": "10000000-0000-4000-8000-000000000003",
            "received_at": event.received_at + timedelta(seconds=10),
        }
    )
    result = store.append_event(replay)
    assert result.created is False
    assert result.event.event_id == event.event_id

    changed = replay.model_copy(update={"outcome": "failed"})
    with pytest.raises(AnalyticsIdempotencyConflictError):
        store.append_event(changed)


def assert_cursor_contract(store, user_id):
    for index in range(3):
        store.append_event(
            event_record(
                user_id,
                event_id=f"20000000-0000-4000-8000-00000000000{index}",
                occurred_at=NOW + timedelta(minutes=index),
            )
        )
    first = store.list_user_events(user_id, limit=2)
    assert len(first["items"]) == 2
    assert first["next_cursor"]
    decode_event_cursor(first["next_cursor"])

    second = store.list_user_events(
        user_id,
        limit=2,
        cursor=first["next_cursor"],
    )
    assert len(second["items"]) == 1
    assert second["next_cursor"] is None
    first_ids = {item.event_id for item in first["items"]}
    second_ids = {item.event_id for item in second["items"]}
    assert not (first_ids & second_ids)


def assert_subject_and_access_contract(store, admin_id, user_id):
    setting = store.set_subject_exclusion(
        user_id,
        excluded=True,
        actor_user_id=admin_id,
        reason="Synthetic QA account.",
    )
    assert setting["excluded"] is True
    assert user_id in store.list_excluded_user_ids()
    assert admin_id in store.list_excluded_user_ids(include_admin_accounts=True)

    access = store.record_admin_access(admin_id, user_id, "overview")
    assert access["admin_user_id"] == admin_id
    assert access["subject_user_id"] == user_id
    assert "response" not in access
    assert store.list_admin_access(user_id)[0]["section"] == "overview"

    cleared = store.set_subject_exclusion(
        user_id,
        excluded=False,
        actor_user_id=admin_id,
        reason="QA account is now included.",
    )
    assert cleared["excluded"] is False
    assert user_id not in store.list_excluded_user_ids()


def test_sqlite_schema_contains_all_foundation_tables(sqlite_contract):
    store, _admin_id, _user_id = sqlite_contract
    with store._get_connection() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "analytics_events",
        "analytics_daily_rollups",
        "user_analytics_snapshots",
        "analytics_subject_settings",
        "admin_analytics_access_log",
    } <= names


def test_sqlite_runs_shared_event_contracts(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    assert_event_idempotency_contract(store, user_id)
    assert_source_event_idempotency_contract(store, user_id)


def test_sqlite_runs_shared_cursor_contract(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    assert_cursor_contract(store, user_id)


def test_sqlite_runs_shared_subject_and_access_contract(sqlite_contract):
    store, admin_id, user_id = sqlite_contract
    assert_subject_and_access_contract(store, admin_id, user_id)


@pytest.mark.parametrize("cursor", ["", "not-base64!", "WzEsMl0", "W10"])
def test_invalid_cursor_is_rejected(sqlite_contract, cursor):
    store, _admin_id, user_id = sqlite_contract
    with pytest.raises(ValueError, match="invalid analytics cursor"):
        store.list_user_events(user_id, cursor=cursor)


@pytest.mark.parametrize("limit", [0, 101, True])
def test_invalid_limits_are_rejected(sqlite_contract, limit):
    store, _admin_id, user_id = sqlite_contract
    with pytest.raises(ValueError, match="limit"):
        store.list_user_events(user_id, limit=limit)


def test_subject_reason_and_access_section_are_closed(sqlite_contract):
    store, admin_id, user_id = sqlite_contract
    for reason in ("", " padded ", "x" * 501):
        with pytest.raises(ValueError, match="reason"):
            store.set_subject_exclusion(
                user_id,
                excluded=True,
                actor_user_id=admin_id,
                reason=reason,
            )
    with pytest.raises(ValueError, match="section"):
        store.record_admin_access(admin_id, user_id, "raw_response")


def test_foreign_keys_reject_missing_users(sqlite_contract):
    store, _admin_id, _user_id = sqlite_contract
    with pytest.raises(AnalyticsStoreError):
        store.append_event(event_record(999_999))
    with pytest.raises((AnalyticsStoreError, sqlite3.IntegrityError)):
        store.record_admin_access(999_998, 999_999, "overview")
