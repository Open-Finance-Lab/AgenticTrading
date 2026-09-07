"""Authoritative acquisition facts and group aggregation contracts."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

from dashboard.backend.domain.analytics.acquisition import (
    AcquisitionAttribution,
    AcquisitionFilters,
    AcquisitionGroupFacts,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.value_queries import (
    AcquisitionGroup,
    ValueAnalyticsQueryService,
)
from dashboard.backend.domain.analytics.value_repository import (
    UserValueSnapshot,
    ValueAnalyticsStore,
)
from dashboard.backend.tests.domain.analytics.test_repository_contract import (
    event_record,
)
from dashboard.backend.users import UserStore


UTC = timezone.utc
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 9, 8, tzinfo=UTC)


def _snapshot(user_id: int, *, lifecycle: str = "core", operational: str = "healthy"):
    return UserValueSnapshot(
        user_id=user_id,
        lifecycle_segment=lifecycle,
        lifecycle_reason_code=f"{lifecycle}_synthetic",
        lifecycle_reason=f"Synthetic {lifecycle} state.",
        lifecycle_evidence=("Synthetic evidence.",),
        operational_state=operational,
        operational_reason_code=f"{operational}_synthetic",
        operational_reason=f"Synthetic {operational} state.",
        operational_evidence=("Synthetic evidence.",),
        activated_at=NOW - timedelta(days=30),
        last_meaningful_activity_at=NOW - timedelta(days=1),
        inactive_days=1,
        active_days_30d=3,
        successful_backtests_30d=2,
        calculated_at=NOW,
    )


class FakeUsers:
    def __init__(self, user_ids):
        self.rows = [
            {
                "id": user_id,
                "email": f"user-{user_id}@example.test",
                "display_name": f"User {user_id}",
                "created_at": (NOW - timedelta(days=30)).isoformat(),
            }
            for user_id in user_ids
        ]

    def list_users_admin(self, *, limit, offset):
        return self.rows[offset : offset + limit]


class FakeStore:
    def __init__(self, attributions, excluded=()):
        self.attributions = attributions
        self.excluded = set(excluded)

    def list_excluded_user_ids(self, *, include_admin_accounts):
        assert include_admin_accounts is True
        return self.excluded

    def list_user_attributions(self, user_ids):
        return {
            user_id: self.attributions[user_id]
            for user_id in user_ids
            if user_id in self.attributions
        }


class FakeValues:
    def __init__(self, snapshots, facts):
        self.snapshots = snapshots
        self.facts = facts

    def list_current_snapshots(self, user_ids):
        return {user_id: self.snapshots[user_id] for user_id in user_ids}

    def list_acquisition_facts(self, user_ids, *, start, end):
        assert start == START
        assert end == END
        return {user_id: self.facts[user_id] for user_id in user_ids}


def test_group_metrics_and_filters_use_confirmed_business_definitions():
    user_ids = [1, 2, 3]
    service = ValueAnalyticsQueryService(
        store=FakeStore(
            {
                1: AcquisitionAttribution(user_id=1, source="community"),
                2: AcquisitionAttribution(user_id=2, source="community"),
                3: AcquisitionAttribution(user_id=3, source="student"),
            },
            excluded={3},
        ),
        user_store=FakeUsers(user_ids),
        value_store=FakeValues(
            {user_id: _snapshot(user_id) for user_id in user_ids},
            {
                1: AcquisitionGroupFacts(
                    active=True,
                    task_completed=True,
                    repeat=True,
                    runs=3,
                    atl_credits_settled_micro=420_000,
                    paid_intent=True,
                    paid=True,
                ),
                2: AcquisitionGroupFacts(task_completed=True),
                3: AcquisitionGroupFacts(active=True, runs=5),
            },
        ),
    )

    response = service.get_acquisition_groups(
        start=START.date(),
        end=END.date(),
        group_by="source",
        filters=AcquisitionFilters(source="community"),
        now=NOW,
    )

    assert len(response.groups) == 1
    community = response.groups[0]
    assert community.users == 2
    assert community.active == 1
    assert community.task_completed == 2
    assert community.repeat == 1
    assert community.runs == 3
    assert community.runs_per_active_user == 3.0
    assert community.atl_credits_settled_micro == 420_000
    assert community.paid_intent == 1
    assert community.paid == 1
    assert response.purchase_window_complete is True


def test_group_response_has_no_token_fields_and_recent_window_is_partial():
    assert {"input_tokens", "output_tokens", "total_tokens"}.isdisjoint(
        AcquisitionGroup.model_fields
    )

    service = ValueAnalyticsQueryService(
        store=FakeStore({1: AcquisitionAttribution(user_id=1)}),
        user_store=FakeUsers([1]),
        value_store=FakeValues(
            {1: _snapshot(1)},
            {1: AcquisitionGroupFacts()},
        ),
    )
    response = service.get_acquisition_groups(
        start=START.date(),
        end=END.date(),
        now=END + timedelta(days=7, seconds=-1),
    )

    assert response.purchase_window_complete is False
    assert response.availability.available is True


class SyntheticCredits:
    def __init__(self, db_path):
        self.db_path = db_path
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE credit_llm_usage_entries (
                    user_id INTEGER NOT NULL,
                    amount_micro INTEGER NOT NULL,
                    operation_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE credit_ledger_entries (
                    user_id INTEGER NOT NULL,
                    entry_type TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _append_run_event(store, user_id, event_name, occurred_at, run_id):
    store.append_event(
        event_record(
            user_id,
            event_id=f"00000000-0000-4000-8000-{run_id:012d}",
            event_name=event_name,
            event_group="run",
            event_source="server",
            source_event_id=f"run:{event_name}:{run_id}",
            source_record_type="run",
            source_record_id=str(run_id),
            session_id=None,
            page_view=None,
            occurred_at=occurred_at,
            outcome="succeeded" if event_name == "backtest_completed" else None,
        )
    )


def test_batched_facts_use_range_activity_and_lifetime_success(tmp_path):
    analytics_path = tmp_path / "analytics.db"
    user = UserStore(db_path=analytics_path).create_user(
        "facts@example.test", "Facts User", "SecurePass1!"
    )
    user_id = int(user["id"])
    analytics = AnalyticsStore(db_path=analytics_path)
    credits = SyntheticCredits(tmp_path / "credits.db")

    _append_run_event(
        analytics, user_id, "backtest_completed", START - timedelta(days=2), 1
    )
    _append_run_event(
        analytics, user_id, "backtest_requested", START + timedelta(days=1), 2
    )
    _append_run_event(
        analytics, user_id, "backtest_requested", START + timedelta(days=2), 3
    )
    _append_run_event(
        analytics, user_id, "backtest_completed", START + timedelta(days=2), 4
    )
    analytics.append_event(
        event_record(
            user_id,
            event_id="00000000-0000-4000-8000-000000000005",
            event_name="checkout_started",
            event_group="resource",
            event_source="server",
            source_event_id="resource:checkout_started:ord-5",
            source_record_type="credit_checkout",
            source_record_id="ord-5",
            session_id=None,
            page_view=None,
            occurred_at=START + timedelta(days=3),
        )
    )
    with credits._get_connection() as conn:
        conn.executemany(
            "INSERT INTO credit_llm_usage_entries VALUES (?, ?, ?, ?)",
            [
                (
                    user_id,
                    -300_000,
                    "run:2:settle",
                    (START + timedelta(days=1)).isoformat(),
                ),
                (
                    user_id,
                    -90_000,
                    "run:2:recovery:1",
                    (START + timedelta(days=1)).isoformat(),
                ),
                (user_id, -500_000, "outside", (END + timedelta(days=1)).isoformat()),
            ],
        )
        conn.execute(
            "INSERT INTO credit_ledger_entries VALUES (?, 'purchase', 'purchased', ?)",
            (user_id, (START + timedelta(days=4)).isoformat()),
        )

    value_store = ValueAnalyticsStore(analytics, credits)
    fact = value_store.list_acquisition_facts([user_id], start=START, end=END)[user_id]

    assert fact == AcquisitionGroupFacts(
        active=True,
        task_completed=True,
        repeat=True,
        runs=2,
        atl_credits_settled_micro=300_000,
        paid_intent=True,
        paid=True,
    )
