"""Lifecycle history reconstruction is bounded, resumable, and point-in-time."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.lifecycle_backfill import (
    BACKFILL_JOB_NAME,
    LifecycleBackfillEvidence,
    LifecycleBackfillSource,
    LifecycleBackfillUser,
    backfill_lifecycle_history,
    run_lifecycle_backfill_batch,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    UserLifecycleDailySnapshot,
    ValueAnalyticsStore,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


class FakeSource:
    def __init__(self):
        self.users = [
            LifecycleBackfillUser(
                user_id=1,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            )
        ]
        self.evidence = {1: []}
        self.complete_from = date.min

    def list_users(self, *, after_user_id, limit):
        return tuple(user for user in self.users if user.user_id > after_user_id)[
            :limit
        ]

    def list_evidence(self, user_id, *, start, end):
        return tuple(
            item
            for item in self.evidence.get(user_id, ())
            if start <= item.occurred_at < end
        )

    def quality_for(self, snapshot_date, *, required_from):
        del required_from
        return "complete" if snapshot_date >= self.complete_from else "partial"


def _store(tmp_path, user_ids=(1, 2)):
    path = tmp_path / "lifecycle-backfill.db"
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
        conn.executemany(
            "INSERT INTO users VALUES (?, ?, ?, 'x', 'user', ?)",
            [
                (
                    user_id,
                    f"user-{user_id}@example.test",
                    f"User {user_id}",
                    "2026-07-01T00:00:00+00:00",
                )
                for user_id in user_ids
            ],
        )
    return ValueAnalyticsStore(
        AnalyticsStore(path),
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )


def test_backfill_does_not_use_a_later_success_for_an_earlier_day(tmp_path):
    source = FakeSource()
    source.evidence[1].append(
        LifecycleBackfillEvidence(
            event_name="backtest_completed",
            occurred_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
        )
    )
    store = _store(tmp_path, user_ids=(1,))

    backfill_lifecycle_history(
        start=date(2026, 8, 18),
        end=date(2026, 8, 22),
        batch_size=50,
        now=NOW,
        source=source,
        store=store,
    )

    snapshots = {
        row.snapshot_date: row
        for row in store.list_daily_snapshots(
            start=date(2026, 8, 18),
            end=date(2026, 8, 22),
            user_ids=[1],
        )
    }
    assert snapshots[date(2026, 8, 19)].lifecycle_segment == "onboarding"
    assert snapshots[date(2026, 8, 20)].lifecycle_segment == "growing"


def test_backfill_resumes_without_duplicate_daily_rows(tmp_path):
    source = FakeSource()
    source.users.append(
        LifecycleBackfillUser(
            user_id=2,
            created_at=datetime(2026, 7, 2, tzinfo=UTC),
        )
    )
    source.evidence[2] = []
    store = _store(tmp_path)
    start = date(2026, 8, 18)
    end = date(2026, 8, 20)

    first = backfill_lifecycle_history(
        start=start,
        end=end,
        batch_size=1,
        now=NOW,
        source=source,
        store=store,
    )
    second = backfill_lifecycle_history(
        start=start,
        end=end,
        batch_size=50,
        cursor=first.next_cursor,
        now=NOW,
        source=source,
        store=store,
    )

    rows = store.list_daily_snapshots(start=start, end=end)
    assert first.complete is False
    assert second.complete is True
    assert len({(row.snapshot_date, row.user_id) for row in rows}) == len(rows) == 4


def test_untrustworthy_source_horizon_stays_partial(tmp_path):
    source = FakeSource()
    source.users[0] = LifecycleBackfillUser(
        user_id=1,
        created_at=datetime(2026, 7, 1, tzinfo=UTC),
    )
    source.complete_from = date(2026, 8, 10)
    store = _store(tmp_path, user_ids=(1,))

    report = backfill_lifecycle_history(
        start=date(2026, 8, 1),
        end=date(2026, 8, 12),
        batch_size=100,
        now=NOW,
        source=source,
        store=store,
    )

    assert report.partial_days == 9
    rows = store.list_daily_snapshots(
        start=date(2026, 8, 1),
        end=date(2026, 8, 12),
    )
    assert {row.data_quality for row in rows[:9]} == {"partial"}
    assert {row.data_quality for row in rows[9:]} == {"complete"}


def test_backfill_rejects_more_than_eight_weeks_and_wrong_window_cursor(tmp_path):
    source = FakeSource()
    source.users.append(
        LifecycleBackfillUser(
            user_id=2,
            created_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    )
    source.evidence[2] = []
    store = _store(tmp_path, user_ids=(1,))
    start = date(2026, 8, 1)
    end = date(2026, 8, 3)
    first = backfill_lifecycle_history(
        start=start,
        end=end,
        batch_size=1,
        now=NOW,
        source=source,
        store=store,
    )

    with pytest.raises(ValueError, match="cursor window"):
        backfill_lifecycle_history(
            start=start,
            end=end.replace(day=4),
            cursor=first.next_cursor,
            now=NOW,
            source=source,
            store=store,
        )
    with pytest.raises(ValueError, match="at most 56 days"):
        backfill_lifecycle_history(
            start=date(2026, 6, 1),
            end=date(2026, 8, 1),
            now=NOW,
            source=source,
            store=store,
        )


def test_deployment_backfill_persists_cursor_and_becomes_a_noop(tmp_path):
    source = FakeSource()
    source.users.append(
        LifecycleBackfillUser(
            user_id=2,
            created_at=datetime(2026, 8, 13, tzinfo=UTC),
        )
    )
    source.evidence[2] = []
    store = _store(tmp_path)

    first = run_lifecycle_backfill_batch(
        now=NOW,
        batch_size=1,
        source=source,
        store=store,
    )
    running_job = store.get_projection_job(BACKFILL_JOB_NAME)
    second = run_lifecycle_backfill_batch(
        now=NOW,
        batch_size=50,
        source=source,
        store=store,
    )
    complete_job = store.get_projection_job(BACKFILL_JOB_NAME)
    third = run_lifecycle_backfill_batch(
        now=NOW,
        batch_size=50,
        source=source,
        store=store,
    )

    assert first.complete is False
    assert running_job.status == "running"
    assert running_job.window_end - running_job.window_start == timedelta(days=56)
    assert second.complete is True
    assert complete_job.status == "complete"
    assert third.processed_users == 0
    assert third.written_rows == 0


def test_partial_backfill_does_not_downgrade_a_complete_daily_snapshot(tmp_path):
    source = FakeSource()
    source.complete_from = date.max
    store = _store(tmp_path, user_ids=(1,))
    snapshot_date = date(2026, 8, 20)
    complete = UserLifecycleDailySnapshot(
        snapshot_date=snapshot_date,
        user_id=1,
        lifecycle_segment="growing",
        lifecycle_reason_code="growing_activated_below_core_threshold",
        data_quality="complete",
        calculated_at=NOW,
    )
    store.upsert_daily_snapshot(complete)

    report = backfill_lifecycle_history(
        start=snapshot_date,
        end=snapshot_date + timedelta(days=1),
        now=NOW,
        source=source,
        store=store,
    )

    assert report.written_rows == 0
    assert store.list_daily_snapshots(
        start=snapshot_date,
        end=snapshot_date + timedelta(days=1),
        user_ids=[1],
    ) == [complete]


def test_sqlite_source_lists_only_eligible_users_and_safe_evidence(tmp_path):
    store = _store(tmp_path)
    analytics = store.analytics_base
    with analytics._get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE id = 2")
    AnalyticsService(analytics).record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:completed:source-test",
        occurred_at=datetime(2026, 8, 20, tzinfo=UTC),
        received_at=datetime(2026, 8, 20, tzinfo=UTC),
    )
    source = LifecycleBackfillSource(
        analytics_base=analytics,
        value_store=store,
        complete_from=date(2026, 8, 1),
    )

    users = source.list_users(after_user_id=0, limit=10)
    evidence = source.list_evidence(
        1,
        start=datetime(2026, 8, 19, tzinfo=UTC),
        end=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert [user.user_id for user in users] == [1]
    assert [(item.event_name, item.occurred_at) for item in evidence] == [
        ("backtest_completed", datetime(2026, 8, 20, tzinfo=UTC))
    ]
    assert (
        source.quality_for(
            date(2026, 8, 1),
            required_from=date(2026, 7, 3),
        )
        == "complete"
    )
