"""Bounded Analytics retention and failure-isolated scheduling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dashboard.backend.domain.analytics.models import (
    AnalyticsEventRecord,
    RetentionResult,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.retention import (
    ADMIN_ACCESS_RETENTION_DAYS,
    RAW_EVENT_RETENTION_DAYS,
    RETENTION_BACKLOG_RETRY_SECONDS,
    RETENTION_INTERVAL_SECONDS,
    AnalyticsRetentionCoordinator,
    AnalyticsRetentionService,
)
from dashboard.backend.domain.analytics.repository_common import utc_iso
from dashboard.backend.users import UserStore


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class RecordingStore:
    def __init__(self):
        self.calls = []

    def delete_expired(self, **kwargs):
        self.calls.append(kwargs)
        return RetentionResult()


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RecordingRetentionService:
    def __init__(self):
        self.calls = 0

    def run_once(self):
        self.calls += 1
        return RetentionResult()


class AlwaysBackloggedStore:
    def __init__(self):
        self.calls = 0

    def delete_expired(self, **_kwargs):
        self.calls += 1
        return RetentionResult(raw_events_deleted=1, has_more_raw_events=True)


class FailingRetentionService:
    def __init__(self, message):
        self.message = message

    def run_once(self):
        raise RuntimeError(self.message)


def _event(user_id: int, received_at: datetime) -> AnalyticsEventRecord:
    return AnalyticsEventRecord(
        event_id=str(uuid4()),
        schema_version=1,
        event_name="page_viewed",
        event_group="experience",
        user_id=user_id,
        session_id=str(uuid4()),
        occurred_at=received_at,
        received_at=received_at,
        event_source="frontend",
        page_view="home",
        properties={},
    )


def test_run_once_uses_180_and_365_day_cutoffs():
    store = RecordingStore()
    service = AnalyticsRetentionService(store=store, batch_size=500)

    result = service.run_once(NOW)

    assert store.calls == [
        {
            "raw_before": NOW - timedelta(days=RAW_EVENT_RETENTION_DAYS),
            "access_before": NOW - timedelta(days=ADMIN_ACCESS_RETENTION_DAYS),
            "batch_size": 500,
        }
    ]
    assert result == RetentionResult()


def test_coordinator_runs_at_most_once_per_24_hours():
    clock = FakeClock()
    service = RecordingRetentionService()
    coordinator = AnalyticsRetentionCoordinator(service=service, clock=clock)

    coordinator.run_if_due()
    coordinator.run_if_due()
    assert service.calls == 1

    clock.advance(RETENTION_INTERVAL_SECONDS)
    coordinator.run_if_due()
    assert service.calls == 2


def test_coordinator_retries_soon_when_retention_batch_limit_leaves_backlog():
    clock = FakeClock()
    store = AlwaysBackloggedStore()
    service = AnalyticsRetentionService(
        store=store,
        batch_size=1,
        max_batches=2,
    )
    coordinator = AnalyticsRetentionCoordinator(service=service, clock=clock)

    assert coordinator.run_if_due() == RetentionResult(
        raw_events_deleted=2,
        has_more_raw_events=True,
    )
    assert store.calls == 2

    clock.advance(RETENTION_BACKLOG_RETRY_SECONDS - 1)
    assert coordinator.run_if_due() is None

    clock.advance(1)
    assert coordinator.run_if_due() == RetentionResult(
        raw_events_deleted=2,
        has_more_raw_events=True,
    )
    assert store.calls == 4


def test_retention_failure_is_swallowed_and_reports_only_safe_metadata(capsys):
    coordinator = AnalyticsRetentionCoordinator(
        service=FailingRetentionService("synthetic-secret-canary"),
        clock=lambda: 0,
    )

    assert coordinator.run_if_due() is None

    output = capsys.readouterr().out
    assert "analytics.retention_failed" in output
    assert "consecutive_failures=1" in output
    assert "category=RuntimeError" in output
    assert "synthetic-secret-canary" not in output


def test_success_resets_consecutive_failure_count():
    clock = FakeClock()
    coordinator = AnalyticsRetentionCoordinator(
        service=FailingRetentionService("synthetic failure"),
        clock=clock,
    )
    coordinator.run_if_due()
    assert coordinator.consecutive_failures == 1

    clock.advance(RETENTION_INTERVAL_SECONDS)
    coordinator.service = RecordingRetentionService()
    coordinator.run_if_due()
    assert coordinator.consecutive_failures == 0


def test_sqlite_retention_is_bounded_and_preserves_aggregates(tmp_path):
    db_path = tmp_path / "analytics_retention.db"
    users = UserStore(db_path=db_path)
    admin = users.create_user(
        "retention-admin@example.test",
        "Retention Admin",
        "SecurePass1!",
    )
    subject = users.create_user(
        "retention-user@example.test",
        "Retention User",
        "SecurePass1!",
    )
    users.apply_admin_patch(admin["id"], role="admin")
    store = AnalyticsStore(db_path=db_path)

    raw_cutoff = NOW - timedelta(days=RAW_EVENT_RETENTION_DAYS)
    access_cutoff = NOW - timedelta(days=ADMIN_ACCESS_RETENTION_DAYS)
    for age in (timedelta(days=181), timedelta(days=200)):
        store.append_event(_event(int(subject["id"]), NOW - age))
    recent_event = store.append_event(
        _event(int(subject["id"]), raw_cutoff + timedelta(seconds=1))
    ).event

    with store._get_connection() as conn:
        for accessed_at in (
            access_cutoff - timedelta(seconds=1),
            access_cutoff - timedelta(days=5),
            access_cutoff + timedelta(seconds=1),
        ):
            conn.execute(
                """
                INSERT INTO admin_analytics_access_log (
                    admin_user_id, subject_user_id, section, accessed_at
                ) VALUES (?, ?, 'overview', ?)
                """,
                (int(admin["id"]), int(subject["id"]), utc_iso(accessed_at)),
            )
        conn.execute(
            """
            INSERT INTO analytics_daily_rollups (
                rollup_date, metric_name, value_count, updated_at
            ) VALUES ('2025-01-01', 'active_users', 7, ?)
            """,
            (utc_iso(NOW),),
        )
        conn.execute(
            """
            INSERT INTO user_analytics_snapshots (
                user_id, status, reason_code, human_readable_reason,
                evidence_event_ids_json, calculated_at
            ) VALUES (?, 'active', 'recent_activity', 'Recent activity.', '[]', ?)
            """,
            (int(subject["id"]), utc_iso(NOW)),
        )

    class RecordingDeleteStore:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.results = []

        def delete_expired(self, **kwargs):
            result = self.wrapped.delete_expired(**kwargs)
            self.results.append(result)
            return result

    recording_store = RecordingDeleteStore(store)
    result = AnalyticsRetentionService(
        store=recording_store,
        batch_size=1,
    ).run_once(NOW)

    assert result.raw_events_deleted == 2
    assert result.access_rows_deleted == 2
    assert result.has_more_raw_events is False
    assert result.has_more_access_rows is False
    assert all(item.raw_events_deleted <= 1 for item in recording_store.results)
    assert all(item.access_rows_deleted <= 1 for item in recording_store.results)

    remaining_events = store.list_user_events(int(subject["id"]))["items"]
    assert [item.event_id for item in remaining_events] == [recent_event.event_id]
    remaining_access = store.list_admin_access(int(subject["id"]))
    assert len(remaining_access) == 1
    assert remaining_access[0]["accessed_at"] == utc_iso(
        access_cutoff + timedelta(seconds=1)
    )
    with store._get_connection() as conn:
        assert conn.execute(
            "SELECT value_count FROM analytics_daily_rollups"
        ).fetchone()[0] == 7
        assert conn.execute(
            "SELECT status FROM user_analytics_snapshots"
        ).fetchone()[0] == "active"
