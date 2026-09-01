"""Shared behavioral contract for Analytics persistence backends."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.models import AnalyticsEventRecord
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsQueryService,
    AnalyticsUserFilters,
)
from dashboard.backend.domain.analytics.repository import (
    ANALYTICS_SQLITE_DDL,
    AnalyticsStore,
)
from dashboard.backend.domain.analytics.repository_common import (
    AnalyticsIdempotencyConflictError,
    AnalyticsStoreError,
    decode_event_cursor,
)
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    UserAnalyticsSnapshot,
    recalculate_user_snapshot,
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


def assert_error_category_contract(store, user_id):
    event = event_record(
        user_id,
        event_name="safe_error_recorded",
        event_group="resource",
        event_source="server",
        source_event_id="safe-error:quota-contract",
        session_id=None,
        page_view=None,
        error_category="provider_quota_exhausted",
    )
    assert store.append_event(event).event.error_category == "provider_quota_exhausted"


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


class ContractUsers:
    def __init__(self, user_id: int):
        self.user = {
            "id": user_id,
            "email": "analytics-user@example.test",
            "display_name": "Analytics User",
            "role": "user",
            "created_at": (NOW - timedelta(days=20)).isoformat(),
        }

    def list_users_admin(self, *, limit=100, offset=0, query=None):
        rows = [self.user]
        if query:
            needle = query.lower()
            rows = [
                row
                for row in rows
                if needle in row["email"].lower()
                or needle in row["display_name"].lower()
            ]
        return rows[offset : offset + limit]

    def get_user_admin(self, user_id):
        return self.user if user_id == self.user["id"] else None


def assert_pr2_query_contract(store, user_id):
    events = AnalyticsService(store)
    common = {
        "user_id": user_id,
        "source_record_type": "run",
        "received_at": NOW,
    }
    events.record_server_event(
        event_name="account_signed_up",
        source_event_id=f"account:account_signed_up:{user_id}",
        source_record_type="user",
        source_record_id=str(user_id),
        occurred_at=NOW - timedelta(days=20),
        **{key: value for key, value in common.items() if key != "source_record_type"},
    )
    events.record_server_event(
        event_name="backtest_failed",
        source_event_id="run:backtest_failed:contract-failed",
        source_record_id="contract-failed",
        correlation_id="contract-failed",
        outcome="failed",
        error_category="provider_timeout",
        occurred_at=NOW - timedelta(hours=3),
        **common,
    )
    completed = events.record_server_event(
        event_name="backtest_completed",
        source_event_id="run:backtest_completed:contract-success",
        source_record_id="contract-success",
        correlation_id="contract-success",
        outcome="succeeded",
        occurred_at=NOW - timedelta(hours=2),
        **common,
    ).event
    events.record_server_event(
        event_name="model_usage_recorded",
        source_event_id="resource:model_usage_recorded:contract-success:0",
        source_record_id="contract-success",
        correlation_id="contract-success",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode="platform_credits",
        outcome="succeeded",
        properties={
            "input_tokens": 120,
            "output_tokens": 30,
            "cost_micro_usd": 250_000,
        },
        occurred_at=NOW - timedelta(hours=1),
        **common,
    )
    events.record_server_event(
        event_name="credits_settled",
        source_event_id="resource:credits_settled:contract-success:0",
        source_record_type="credit_reservation",
        source_record_id="contract-reservation",
        correlation_id="contract-success",
        billing_mode="platform_credits",
        properties={"amount_micro": 100, "bucket": "grant"},
        occurred_at=NOW - timedelta(minutes=59),
        **{key: value for key, value in common.items() if key != "source_record_type"},
    )
    state_store = AnalyticsStateStore(store)
    recalculate_user_snapshot(user_id, now=NOW, store=state_store)
    service = AnalyticsQueryService(
        store=store,
        user_store=ContractUsers(user_id),
    )

    overview = service.get_overview(
        filters=AnalyticsMetricFilters(
            start=NOW - timedelta(days=30),
            end=NOW + timedelta(seconds=1),
        ),
        now=NOW,
    )
    users = service.list_users(
        filters=AnalyticsUserFilters(),
        limit=10,
        offset=0,
        now=NOW,
    )
    profile = service.get_user_profile(user_id=user_id, now=NOW)
    activity = service.get_user_activity(
        user_id=user_id,
        section="runs",
        limit=10,
        cursor=None,
    )
    state_store.upsert_snapshot(
        UserAnalyticsSnapshot(
            user_id=user_id,
            status="needs_attention",
            reason_code="invalid_default_credential",
            human_readable_reason="The default model credential is invalid.",
            evidence_event_ids=[completed.event_id],
            calculated_at=NOW,
        )
    )
    attention_overview = service.get_overview(
        filters=AnalyticsMetricFilters(
            start=NOW - timedelta(days=30),
            end=NOW + timedelta(seconds=1),
        ),
        now=NOW,
    )

    assert overview.completed_runs == 1
    assert overview.failed_runs == 1
    assert overview.platform_model_cost_usd == 0.25
    assert users.total == 1
    assert users.items[0].status == "active"
    assert profile.state.status == "active"
    assert completed.event_id in profile.state.evidence_event_ids
    assert profile.input_tokens == 120
    assert profile.output_tokens == 30
    assert profile.credits_debited_micro == 100
    assert [item.event_name for item in activity.items] == [
        "backtest_completed",
        "backtest_failed",
    ]
    assert activity.next_cursor is None
    assert attention_overview.users_needing_attention[0].user_id == user_id
    assert attention_overview.users_needing_attention[0].recent_failures == 1


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


def test_sqlite_accepts_provider_quota_exhausted_category(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    assert_error_category_contract(store, user_id)


def test_sqlite_migrates_legacy_error_category_constraint(tmp_path):
    db_path = tmp_path / "analytics-legacy.db"
    users = UserStore(db_path=db_path)
    user = users.create_user(
        "legacy-analytics-user@example.test",
        "Legacy Analytics User",
        "SecurePass1!",
    )
    legacy_ddl = ANALYTICS_SQLITE_DDL.replace(
        "'provider_unavailable', 'provider_quota_exhausted',\n            'credits_unavailable',",
        "'provider_unavailable', 'credits_unavailable',",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(legacy_ddl)

    store = AnalyticsStore(db_path=db_path)
    event = event_record(
        int(user["id"]),
        event_id="10000000-0000-4000-8000-000000000005",
        event_name="safe_error_recorded",
        event_group="resource",
        event_source="server",
        source_event_id="safe-error:quota-migration",
        session_id=None,
        page_view=None,
        error_category="provider_quota_exhausted",
    )
    assert store.append_event(event).created is True


def test_sqlite_defers_error_category_migration_until_users_table_exists(tmp_path):
    db_path = tmp_path / "analytics-before-users.db"
    legacy_ddl = ANALYTICS_SQLITE_DDL.replace(
        "'provider_unavailable', 'provider_quota_exhausted',\n            'credits_unavailable',",
        "'provider_unavailable', 'credits_unavailable',",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(legacy_ddl)

    store = AnalyticsStore(db_path=db_path)
    user = UserStore(db_path=db_path).create_user(
        "deferred-analytics-user@example.test",
        "Deferred Analytics User",
        "SecurePass1!",
    )
    event = event_record(
        int(user["id"]),
        event_id="10000000-0000-4000-8000-000000000006",
        event_name="safe_error_recorded",
        event_group="resource",
        event_source="server",
        source_event_id="safe-error:quota-deferred-migration",
        session_id=None,
        page_view=None,
        error_category="provider_quota_exhausted",
    )

    assert store.append_event(event).created is True


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


def test_sqlite_runs_pr2_query_contract(sqlite_contract):
    store, _admin_id, user_id = sqlite_contract
    assert_pr2_query_contract(store, user_id)


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
