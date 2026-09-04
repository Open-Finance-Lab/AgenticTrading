"""Explainable Analytics user-state precedence and snapshot persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    UserAnalyticsSnapshot,
    calculate_user_value_snapshot,
    calculate_user_state,
    recalculate_user_snapshot,
    repair_stale_value_snapshots,
)
from dashboard.backend.domain.analytics.value_repository import (
    CurrentOperationalFacts,
    UserValueSnapshot,
    ValueAnalyticsStore,
)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class RecordingValueStore:
    def __init__(self):
        self.activities = ()
        self.snapshots = {}
        self.daily = []
        self.operational = CurrentOperationalFacts(user_id=1)
        self.upsert_count = 0

    @property
    def current(self):
        return self.snapshots.get(1)

    @current.setter
    def current(self, value):
        if value is None:
            self.snapshots.pop(1, None)
        else:
            self.snapshots[1] = value

    def list_credit_activity(self, user_ids, *, start, end):
        return {user_id: self.activities for user_id in user_ids}

    def get_current_snapshot(self, user_id):
        return self.snapshots.get(user_id)

    def get_operational_facts(self, user_id, *, now):
        return self.operational.model_copy(update={"user_id": user_id})

    def upsert_current_snapshot(self, snapshot):
        self.upsert_count += 1
        self.snapshots[snapshot.user_id] = snapshot
        return snapshot

    def upsert_daily_snapshot(self, snapshot):
        self.daily = [
            row
            for row in self.daily
            if (row.snapshot_date, row.user_id)
            != (snapshot.snapshot_date, snapshot.user_id)
        ]
        self.daily.append(snapshot)
        return snapshot


class FailingValueStore(RecordingValueStore):
    def list_credit_activity(self, user_ids, *, start, end):
        raise RuntimeError("private projection failure")


class SelectiveFailingValueStore(RecordingValueStore):
    def list_credit_activity(self, user_ids, *, start, end):
        if user_ids == [1]:
            raise RuntimeError("private first-user failure")
        return super().list_credit_activity(user_ids, start=start, end=end)


def _fixture(tmp_path, *, created_at=NOW - timedelta(days=1)):
    path = tmp_path / "states.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO users VALUES (1, 'user@example.test', 'User', 'x', 'user', ?)",
            (created_at.isoformat(),),
        )
    analytics = AnalyticsStore(path)
    state_store = AnalyticsStateStore(analytics)
    return (
        AnalyticsService(
            analytics,
            state_store=state_store,
            value_store=RecordingValueStore(),
            project_snapshots=True,
        ),
        state_store,
    )


def _run(service, name, index, at, *, error_category=None):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=f"run:{name}:run-{index}",
        source_record_type="run",
        source_record_id=f"run-{index}",
        correlation_id=f"run-{index}",
        error_category=error_category,
        occurred_at=at,
    ).event


def test_blocked_wins_over_needs_attention(tmp_path):
    service, store = _fixture(tmp_path)
    for index in range(3):
        _run(
            service,
            "backtest_failed",
            index,
            NOW - timedelta(hours=index + 2),
            error_category="internal_error",
        )
    blocked = service.record_server_event(
        event_name="safe_error_recorded",
        user_id=1,
        source_event_id="resource:safe_error_recorded:run-blocked:credits_unavailable",
        source_record_type="run",
        source_record_id="run-blocked",
        error_category="credits_unavailable",
        occurred_at=NOW - timedelta(hours=1),
    ).event

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "blocked"
    assert snapshot.reason_code == "billing_lane_unavailable"
    assert snapshot.evidence_event_ids == [blocked.event_id]


def test_new_user_without_run_is_onboarding(tmp_path):
    _service, store = _fixture(tmp_path)

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "onboarding"
    assert snapshot.reason_code == "no_successful_run"


def test_three_newest_terminal_failures_need_attention(tmp_path):
    service, store = _fixture(tmp_path)
    failures = [
        _run(
            service,
            "backtest_failed",
            index,
            NOW - timedelta(hours=index + 1),
            error_category="internal_error",
        )
        for index in range(3)
    ]

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "needs_attention"
    assert snapshot.reason_code == "three_consecutive_failed_runs"
    assert set(snapshot.evidence_event_ids) == {event.event_id for event in failures}


def test_completed_or_cancelled_run_breaks_failure_sequence(tmp_path):
    service, store = _fixture(tmp_path)
    _run(service, "backtest_failed", 1, NOW - timedelta(hours=1))
    _run(service, "backtest_cancelled", 2, NOW - timedelta(hours=2))
    _run(service, "backtest_failed", 3, NOW - timedelta(hours=3))
    _run(service, "backtest_failed", 4, NOW - timedelta(hours=4))

    snapshot = calculate_user_state(1, now=NOW, store=store)

    assert snapshot.status == "onboarding"


def test_dormant_and_active_use_thirty_day_activity(tmp_path):
    service, store = _fixture(tmp_path, created_at=NOW - timedelta(days=60))
    _run(service, "backtest_completed", 1, NOW - timedelta(days=45))

    dormant = calculate_user_state(1, now=NOW, store=store)
    _run(service, "backtest_completed", 2, NOW - timedelta(days=1))
    active = calculate_user_state(1, now=NOW, store=store)

    assert dormant.status == "dormant"
    assert dormant.reason_code == "no_meaningful_activity_30d"
    assert active.status == "active"


def test_recalculate_upserts_one_current_snapshot(tmp_path):
    service, store = _fixture(tmp_path)
    first = recalculate_user_snapshot(1, now=NOW, store=store)
    _run(service, "backtest_completed", 1, NOW - timedelta(hours=1))
    second = recalculate_user_snapshot(1, now=NOW, store=store)

    assert first.status == "onboarding"
    assert second.status == "active"
    assert store.get_snapshot(1) == second


def test_value_snapshot_keeps_lifecycle_and_operational_axes_separate(tmp_path):
    service, state_store = _fixture(
        tmp_path,
        created_at=NOW - timedelta(days=20),
    )
    _run(service, "backtest_completed", 1, NOW - timedelta(days=9))
    value_store = RecordingValueStore()
    value_store.operational = CurrentOperationalFacts(
        user_id=1,
        account_restricted=True,
    )

    snapshot = calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=value_store,
    )

    assert snapshot.lifecycle_segment == "at_risk"
    assert snapshot.operational_state == "blocked"
    assert snapshot.lifecycle_reason_code == "at_risk_previously_activated"
    assert snapshot.operational_reason_code == "account_restricted"


def test_value_snapshot_counts_distinct_utc_activity_days(tmp_path):
    service, state_store = _fixture(
        tmp_path,
        created_at=NOW - timedelta(days=20),
    )
    for index, days_ago in enumerate((4, 3, 2), start=1):
        _run(service, "backtest_completed", index, NOW - timedelta(days=days_ago))
    _run(service, "agent_updated", 4, NOW - timedelta(days=2, hours=1))
    value_store = RecordingValueStore()
    value_store.activities = (
        NOW - timedelta(days=1),
        NOW - timedelta(days=1, hours=1),
    )

    snapshot = calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=value_store,
    )

    assert snapshot.lifecycle_segment == "core"
    assert snapshot.active_days_30d == 4
    assert snapshot.successful_backtests_30d == 3
    assert snapshot.last_meaningful_activity_at == NOW - timedelta(days=1)


def test_legacy_snapshot_fields_remain_unchanged(tmp_path):
    service, state_store = _fixture(tmp_path)
    recalculate_user_snapshot(1, now=NOW, store=state_store)
    _run(service, "backtest_completed", 1, NOW - timedelta(hours=1))
    legacy = state_store.get_snapshot(1)

    calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=RecordingValueStore(),
    )

    assert state_store.get_snapshot(1) == legacy


def test_relevant_event_recalculates_value_projection(tmp_path):
    _unused_service, state_store = _fixture(tmp_path)
    value_store = RecordingValueStore()
    value_store.analytics_base = state_store.base_store
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=value_store,
        project_snapshots=True,
    )

    result = service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:completed:synthetic-1",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    persisted = ValueAnalyticsStore(
        state_store.base_store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    ).get_current_snapshot(1)
    assert persisted is not None
    assert persisted.lifecycle_segment == "growing"
    assert persisted.operational_state == "healthy"
    assert persisted.calculated_at == NOW
    legacy = state_store.get_snapshot(1)
    assert legacy.status == "active"
    assert legacy.calculated_at == persisted.calculated_at
    assert value_store.upsert_count == 0


def test_projection_failure_does_not_reject_an_accepted_event(tmp_path, capsys):
    _unused_service, state_store = _fixture(tmp_path)
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=FailingValueStore(),
        project_snapshots=True,
    )

    result = service.record_server_event(
        event_name="backtest_requested",
        user_id=1,
        source_event_id="run:requested:synthetic-2",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    output = capsys.readouterr().out
    assert "analytics.value_projection_failed" in output
    assert "event=backtest_requested" in output
    assert "category=RuntimeError" in output
    assert "private projection failure" not in output
    assert state_store.get_snapshot(1) is None


def test_projection_is_disabled_by_default(tmp_path):
    _unused_service, state_store = _fixture(tmp_path)
    value_store = RecordingValueStore()
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=value_store,
    )

    result = service.record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:completed:projection-disabled",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    assert value_store.current is None
    assert state_store.get_snapshot(1) is None


@pytest.mark.parametrize(
    "event_name",
    ["authenticated_session_started", "credential_revoked", "agent_deleted"],
)
def test_irrelevant_server_event_does_not_recalculate_value_projection(
    tmp_path,
    event_name,
):
    _unused_service, state_store = _fixture(tmp_path)
    value_store = RecordingValueStore()
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=value_store,
        project_snapshots=True,
    )

    result = service.record_server_event(
        event_name=event_name,
        user_id=1,
        source_event_id=f"event:{event_name}:synthetic-1",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    assert value_store.current is None
    assert state_store.get_snapshot(1) is None


@pytest.mark.parametrize(
    "event_name",
    ["credential_saved", "account_signed_up", "safe_error_recorded"],
)
def test_only_supported_relevant_event_groups_trigger_projection(tmp_path, event_name):
    _unused_service, state_store = _fixture(tmp_path)
    value_store = RecordingValueStore()
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=value_store,
        project_snapshots=True,
    )

    result = service.record_server_event(
        event_name=event_name,
        user_id=1,
        source_event_id=f"event:{event_name}:synthetic",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert result.created is True
    assert value_store.upsert_count == 1


def test_idempotent_event_replay_does_not_recalculate_projection(tmp_path):
    _unused_service, state_store = _fixture(tmp_path)
    value_store = RecordingValueStore()
    service = AnalyticsService(
        state_store.base_store,
        state_store=state_store,
        value_store=value_store,
        project_snapshots=True,
    )
    kwargs = {
        "event_name": "agent_created",
        "user_id": 1,
        "source_event_id": "agent:created:synthetic",
        "occurred_at": NOW,
        "received_at": NOW,
    }

    first = service.record_server_event(**kwargs)
    replay = service.record_server_event(**kwargs)

    assert first.created is True
    assert replay.created is False
    assert value_store.upsert_count == 1


def test_cleared_current_account_is_not_blocked_by_an_old_error(tmp_path):
    service, state_store = _fixture(tmp_path)
    service.record_server_event(
        event_name="safe_error_recorded",
        user_id=1,
        source_event_id="run:safe-error:credits",
        source_record_type="run",
        source_record_id="run-credits",
        error_category="credits_unavailable",
        occurred_at=NOW - timedelta(hours=2),
        received_at=NOW - timedelta(hours=2),
    )
    value_store = RecordingValueStore()

    snapshot = calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=value_store,
    )

    assert snapshot.operational_state == "healthy"
    assert snapshot.operational_reason_code == "no_supported_issue"
    assert state_store.get_snapshot(1).status == "blocked"


def test_activation_timestamp_survives_raw_event_retention(tmp_path):
    _service, state_store = _fixture(
        tmp_path,
        created_at=NOW - timedelta(days=300),
    )
    first_success = NOW - timedelta(days=200)
    value_store = RecordingValueStore()
    value_store.current = UserValueSnapshot(
        user_id=1,
        lifecycle_segment="dormant",
        lifecycle_reason_code="dormant_previously_activated",
        lifecycle_reason="Previously activated and inactive.",
        operational_state="healthy",
        operational_reason_code="no_supported_issue",
        operational_reason="No supported current operational issue was detected.",
        activated_at=first_success,
        last_meaningful_activity_at=first_success,
        inactive_days=199,
        active_days_30d=0,
        successful_backtests_30d=0,
        calculated_at=NOW - timedelta(days=1),
    )

    snapshot = calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=value_store,
    )

    assert snapshot.activated_at == first_success
    assert snapshot.last_meaningful_activity_at == first_success
    assert snapshot.lifecycle_reason_code == "dormant_previously_activated"


def test_future_previous_projection_timestamps_are_ignored(tmp_path):
    _service, state_store = _fixture(
        tmp_path,
        created_at=NOW - timedelta(days=10),
    )
    value_store = RecordingValueStore()
    value_store.current = UserValueSnapshot(
        user_id=1,
        lifecycle_segment="growing",
        lifecycle_reason_code="growing_activated_below_core_threshold",
        lifecycle_reason="Activated.",
        operational_state="healthy",
        operational_reason_code="no_supported_issue",
        operational_reason="No supported current operational issue was detected.",
        activated_at=NOW + timedelta(days=1),
        last_meaningful_activity_at=NOW + timedelta(days=2),
        inactive_days=0,
        active_days_30d=1,
        successful_backtests_30d=1,
        calculated_at=NOW + timedelta(days=2),
    )

    snapshot = calculate_user_value_snapshot(
        1,
        now=NOW,
        state_store=state_store,
        value_store=value_store,
    )

    assert snapshot.activated_at is None
    assert snapshot.last_meaningful_activity_at is None
    assert snapshot.lifecycle_reason_code == "at_risk_never_activated"


def test_utc_day_transition_repairs_current_and_daily_value_snapshots(tmp_path):
    transition_now = datetime(2026, 8, 27, 0, 5, tzinfo=timezone.utc)
    _service, state_store = _fixture(
        tmp_path,
        created_at=transition_now - timedelta(days=1),
    )
    state_store.upsert_snapshot(
        UserAnalyticsSnapshot(
            user_id=1,
            status="onboarding",
            reason_code="no_successful_run",
            human_readable_reason="The user has not completed a successful backtest yet.",
            calculated_at=transition_now - timedelta(minutes=6),
        )
    )
    value_store = RecordingValueStore()

    repaired = repair_stale_value_snapshots(
        now=transition_now,
        limit=10,
        state_store=state_store,
        value_store=value_store,
    )

    assert repaired == 1
    assert value_store.current is not None
    assert len(value_store.daily) == 1
    assert value_store.daily[0].snapshot_date == transition_now.date()
    assert value_store.daily[0].data_quality == "complete"

    repair_stale_value_snapshots(
        now=transition_now,
        limit=10,
        state_store=state_store,
        value_store=value_store,
    )
    assert len(value_store.daily) == 1


def test_value_snapshot_repair_isolates_one_user_failure(tmp_path, capsys):
    _service, state_store = _fixture(
        tmp_path,
        created_at=NOW - timedelta(days=1),
    )
    with state_store.base_store._get_connection() as conn:
        conn.execute(
            "INSERT INTO users VALUES (2, 'second@example.test', 'Second', 'x', 'user', ?)",
            ((NOW - timedelta(days=1)).isoformat(),),
        )
    value_store = SelectiveFailingValueStore()

    repaired = repair_stale_value_snapshots(
        now=NOW,
        limit=10,
        state_store=state_store,
        value_store=value_store,
    )

    assert repaired == 1
    assert state_store.get_snapshot(1) is None
    assert state_store.get_snapshot(2) is not None
    assert value_store.get_current_snapshot(2) is not None
    output = capsys.readouterr().out
    assert "analytics.value_snapshot_repair_failed" in output
    assert "category=RuntimeError" in output
    assert "private first-user failure" not in output
