"""Ingestion maintains one current-state row per user and never reads history."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


from dashboard.backend.domain.analytics import rollups as rollups_module
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.value_repository import (
    build_value_analytics_store,
)
from dashboard.backend.tests.domain.analytics.test_value_repository import (
    _value_snapshot,
)


NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)


class CountingAnalyticsStore(AnalyticsStore):
    """The real SQLite store, counting every per-user history read."""

    def __init__(self, path):
        super().__init__(path)
        self.event_reads = 0

    def list_user_events(self, *args, **kwargs):
        self.event_reads += 1
        return super().list_user_events(*args, **kwargs)


def _fixture(tmp_path, monkeypatch):
    path = tmp_path / "activity.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                user_group TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO users VALUES (1, 'user@example.test', 'User', 'x', 'user', "
            "'unknown', ?)",
            ((NOW - timedelta(days=60)).isoformat(),),
        )
    store = CountingAnalyticsStore(path)

    def _no_history_scan(*_args, **_kwargs):
        raise AssertionError("ingestion must not scan analytics_events")

    monkeypatch.setattr(
        rollups_module.AnalyticsRollupStore, "list_events", _no_history_scan
    )
    value_store = build_value_analytics_store(
        store,
        credits_base=object(),
        provider_base=object(),
        agent_base=object(),
        run_base=object(),
    )
    service = AnalyticsService(
        store,
        value_store=value_store,
        maintain_activity=True,
    )
    return service, store, value_store


def _emit(service, index, occurred_at, *, name="backtest_requested"):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=f"run:{name}:run-{index}",
        source_record_type="run",
        source_record_id=f"run-{index}",
        occurred_at=occurred_at,
        received_at=NOW,
    )


def test_ingestion_never_reads_a_users_history(tmp_path, monkeypatch):
    """Accepting an event costs one upsert, not a scan.

    The 2026-09-11 outage was this path: every accepted lifecycle event
    recomputed a label from a 180-day read of the log it had just appended to.
    """
    service, store, _value_store = _fixture(tmp_path, monkeypatch)
    for index in range(20):
        _emit(service, index, NOW - timedelta(seconds=index))

    assert store.event_reads == 0


def test_activated_at_holds_the_earliest_success(tmp_path, monkeypatch):
    """Activation is the *first* success, in occurred_at order."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    _emit(service, "a", NOW - timedelta(days=3), name="backtest_completed")
    first = value_store.get_activity(1).activated_at

    _emit(service, "b", NOW, name="backtest_completed")
    activity = value_store.get_activity(1)

    assert first == NOW - timedelta(days=3)
    assert activity.activated_at == first
    assert activity.last_meaningful_activity_at == NOW


def test_an_out_of_order_success_moves_activated_at_earlier(tmp_path, monkeypatch):
    """The newest write does not own the value; the earliest instant does.

    Ingestion sees events in arrival order, not occurred_at order: a
    `backtest_completed` can be appended late, replayed from the queue, or
    backdated by the 24-hour acceptance window in `service.py:96-97`. If
    activation were pinned by whichever row landed first, this user's
    activation cohort week -- and therefore their whole column of the
    retention grid -- would be permanently wrong, with nothing able to
    correct it.
    """
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    for suffix, occurred in (("late", NOW), ("early", NOW - timedelta(days=5))):
        _emit(service, suffix, occurred, name="backtest_completed")

    activity = value_store.get_activity(1)

    assert activity.activated_at == NOW - timedelta(days=5)
    # The activity clock still only advances.
    assert activity.last_meaningful_activity_at == NOW


def test_page_views_and_signups_do_not_advance_the_activity_clock(
    tmp_path, monkeypatch
):
    """Only the meaningful-activity set counts. A visit is not activity."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    service.record_server_event(
        event_name="account_signed_up",
        user_id=1,
        source_event_id="account:account_signed_up:1",
        source_record_type="user",
        source_record_id="1",
        occurred_at=NOW,
        received_at=NOW,
    )

    assert value_store.get_activity(1) is None


def test_a_replayed_event_does_not_touch_the_row(tmp_path, monkeypatch):
    """`created=False` means the log already had it; the clock must not move."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)

    _emit(service, "once", NOW - timedelta(days=1))
    before = value_store.get_activity(1)
    replay = _emit(service, "once", NOW - timedelta(days=1))

    assert replay.created is False
    assert value_store.get_activity(1) == before


def test_list_activity_returns_every_row_when_no_ids_are_given(tmp_path, monkeypatch):
    service, _store, value_store = _fixture(tmp_path, monkeypatch)
    _emit(service, "x", NOW - timedelta(hours=1))

    everyone = value_store.list_activity()
    some = value_store.list_activity([1])
    none = value_store.list_activity([])

    assert set(everyone) == {1}
    assert some == everyone
    assert none == {}


def test_seed_copies_the_legacy_timestamps_and_is_idempotent(tmp_path, monkeypatch):
    """Design doc SS11: the migration must carry ``activated_at`` across.

    Ingestion only writes ``user_activity`` for events that arrive after the
    deploy, and the daily job's one-day scan cannot see an activation from
    last year. Without this seed every existing user's activation date would
    be lost on the night PR B drops ``user_analytics_snapshots``.
    """
    _service, _store, value_store = _fixture(tmp_path, monkeypatch)
    legacy = _value_snapshot(1)  # activated NOW-20d, last activity NOW-2d
    value_store.upsert_current_snapshot(legacy)

    first = value_store.seed_activity_from_snapshots(now=NOW)
    seeded = value_store.get_activity(1)
    second = value_store.seed_activity_from_snapshots(now=NOW + timedelta(hours=1))
    reseeded = value_store.get_activity(1)

    assert first == 1
    assert seeded.activated_at == legacy.activated_at
    assert seeded.last_meaningful_activity_at == legacy.last_meaningful_activity_at
    assert second == 1  # the upsert touches the row again ...
    assert reseeded.activated_at == seeded.activated_at  # ... and changes nothing
    assert reseeded.last_meaningful_activity_at == seeded.last_meaningful_activity_at


def test_seed_never_regresses_a_row_ingestion_already_advanced(tmp_path, monkeypatch):
    """MIN on activation, MAX on activity: the seed can only fill gaps."""
    service, _store, value_store = _fixture(tmp_path, monkeypatch)
    value_store.upsert_current_snapshot(_value_snapshot(1))  # NOW-20d / NOW-2d
    _emit(service, "earlier", NOW - timedelta(days=30), name="backtest_completed")
    _emit(service, "today", NOW)

    value_store.seed_activity_from_snapshots(now=NOW)
    activity = value_store.get_activity(1)

    assert activity.activated_at == NOW - timedelta(days=30)
    assert activity.last_meaningful_activity_at == NOW
