"""Admin Analytics query-service and API contract coverage."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from dashboard.backend.domain.analytics.metrics import AnalyticsMetricFilters
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsQueryService,
    AnalyticsUserFilters,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.rollups import (
    AnalyticsRollupStore,
    DailyRollup,
)
from dashboard.backend.domain.analytics.service import AnalyticsService
from dashboard.backend.domain.analytics.models import (
    FrontendAnalyticsEvent,
    RequestAnalyticsContext,
)
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    recalculate_user_snapshot,
)


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


class QueryUsers:
    def __init__(self, rows):
        self.rows = list(rows)

    def list_users_admin(self, *, limit=100, offset=0, query=None):
        rows = self.rows
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
        return next((row for row in self.rows if row["id"] == user_id), None)


def _fixture(tmp_path):
    path = tmp_path / "query.db"
    users = [
        {
            "id": 1,
            "email": "one@example.test",
            "display_name": "One",
            "role": "user",
            "created_at": (NOW - timedelta(days=20)).isoformat(),
        },
        {
            "id": 2,
            "email": "admin@example.test",
            "display_name": "Admin",
            "role": "admin",
            "created_at": (NOW - timedelta(days=30)).isoformat(),
        },
    ]
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
            "INSERT INTO users VALUES (?, ?, ?, 'x', ?, ?)",
            [
                (
                    row["id"],
                    row["email"],
                    row["display_name"],
                    row["role"],
                    row["created_at"],
                )
                for row in users
            ],
        )
    analytics = AnalyticsStore(path)
    return (
        analytics,
        AnalyticsService(analytics),
        AnalyticsRollupStore(analytics),
        AnalyticsStateStore(analytics),
        QueryUsers(users),
    )


def _event(service, name, at, source_id, **kwargs):
    return service.record_server_event(
        event_name=name,
        user_id=1,
        source_event_id=source_id,
        source_record_type=kwargs.pop("source_record_type", "run"),
        source_record_id=kwargs.pop("source_record_id", source_id.rsplit(":", 1)[-1]),
        occurred_at=at,
        **kwargs,
    ).event


def test_query_service_merges_completed_rollups_with_current_raw_day(tmp_path):
    analytics, events, rollups, _states, users = _fixture(tmp_path)
    yesterday = NOW.date() - timedelta(days=1)
    rollups.replace_day(
        yesterday,
        [
            DailyRollup(
                rollup_date=yesterday,
                metric_name="terminal_completed",
                value_count=4,
                updated_at=datetime.combine(
                    NOW.date(), datetime.min.time(), tzinfo=timezone.utc
                ),
            ),
            DailyRollup(
                rollup_date=yesterday,
                metric_name="daily_active_users",
                value_count=3,
                updated_at=datetime.combine(
                    NOW.date(), datetime.min.time(), tzinfo=timezone.utc
                ),
            ),
        ],
    )
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=2),
        "run:backtest_completed:today",
        outcome="succeeded",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=datetime.combine(
                yesterday, datetime.min.time(), tzinfo=timezone.utc
            ),
            end=NOW,
        ),
    )

    assert overview.completed_runs == 5
    assert overview.daily_completed_runs[yesterday.isoformat()] == 4
    assert overview.daily_completed_runs[NOW.date().isoformat()] == 1
    assert overview.last_updated == NOW
    assert overview.availability["growth"].available is True


def test_user_list_and_profile_are_display_safe(tmp_path):
    analytics, events, _rollups, states, users = _fixture(tmp_path)
    _event(
        events,
        "account_signed_up",
        NOW - timedelta(days=20),
        "account:account_signed_up:1",
        source_record_type="user",
        source_record_id="1",
    )
    success = _event(
        events,
        "backtest_completed",
        NOW - timedelta(hours=1),
        "run:backtest_completed:run-1",
        correlation_id="run-1",
        outcome="succeeded",
    )
    _event(
        events,
        "model_usage_recorded",
        NOW - timedelta(minutes=50),
        "resource:model_usage_recorded:run-1:0",
        correlation_id="run-1",
        provider_id="openrouter",
        model_id="openai/gpt-5.5",
        billing_mode="platform_credits",
        outcome="succeeded",
        properties={
            "input_tokens": 120,
            "output_tokens": 30,
            "cost_micro_usd": 250_000,
        },
    )
    _event(
        events,
        "credits_settled",
        NOW - timedelta(minutes=49),
        "resource:credits_settled:reservation-1:grant",
        source_record_type="credit_reservation",
        source_record_id="reservation-1",
        correlation_id="run-1",
        billing_mode="platform_credits",
        properties={"amount_micro": 100, "bucket": "grant"},
    )
    recalculate_user_snapshot(1, now=NOW, store=states)
    service = AnalyticsQueryService(store=analytics, user_store=users)

    listing = service.list_users(
        filters=AnalyticsUserFilters(),
        limit=25,
        offset=0,
        now=NOW,
    )
    profile = service.get_user_profile(user_id=1, now=NOW)
    serialized = profile.model_dump(mode="json")

    assert listing.total == 1
    assert listing.items[0].user_id == 1
    assert profile.state.status == "active"
    assert success.event_id in profile.state.evidence_event_ids
    assert profile.input_tokens == 120
    assert profile.platform_model_cost_usd == 0.25
    assert profile.credits_debited_micro == 100
    assert "properties" not in str(serialized)
    assert "session_id" not in str(serialized)


def test_activity_sections_page_independently_and_hide_session_ids(tmp_path):
    analytics, events, _rollups, _states, users = _fixture(tmp_path)
    for index in range(3):
        _event(
            events,
            "backtest_completed",
            NOW - timedelta(minutes=index + 1),
            f"run:backtest_completed:run-{index}",
            source_record_id=f"run-{index}",
            outcome="succeeded",
        )
    context = RequestAnalyticsContext(
        country_code="US",
        device_category="desktop",
        browser_family="Chrome",
    )
    session_id = str(uuid4())
    for index, name in enumerate(("page_viewed", "session_heartbeat")):
        events.accept_frontend_event(
            user={"id": 1},
            payload=FrontendAnalyticsEvent(
                event_id=str(uuid4()),
                schema_version=1,
                event_name=name,
                session_id=session_id,
                occurred_at=NOW - timedelta(minutes=10 - index),
                page_view="agents",
                properties=({} if name == "page_viewed" else {"visible_ms": 500}),
            ),
            context=context,
            received_at=NOW,
        )
    service = AnalyticsQueryService(store=analytics, user_store=users)

    first_runs = service.get_user_activity(
        user_id=1,
        section="runs",
        limit=2,
        cursor=None,
    )
    second_runs = service.get_user_activity(
        user_id=1,
        section="runs",
        limit=2,
        cursor=first_runs.next_cursor,
    )
    sessions = service.get_user_activity(
        user_id=1,
        section="sessions",
        limit=2,
        cursor=None,
    )

    assert len(first_runs.items) == 2
    assert len(second_runs.items) == 1
    assert first_runs.next_cursor is not None
    assert sessions.items[0].session_event_count == 2
    assert sessions.items[0].visible_ms == 500
    assert "session_id" not in str(sessions.model_dump(mode="json"))


def test_overview_marks_only_failed_rollup_panel_unavailable(tmp_path, monkeypatch):
    analytics, events, _rollups, _states, users = _fixture(tmp_path)
    _event(
        events,
        "backtest_completed",
        NOW - timedelta(minutes=1),
        "run:backtest_completed:current",
        outcome="succeeded",
    )
    service = AnalyticsQueryService(store=analytics, user_store=users)
    monkeypatch.setattr(
        service.query_store.rollups,
        "list_rollups",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private detail")),
    )

    overview = service.get_overview(
        now=NOW,
        filters=AnalyticsMetricFilters(
            start=NOW - timedelta(days=1),
            end=NOW,
        ),
    )

    assert overview.active_users_7d == 1
    assert overview.completed_runs is None
    assert overview.availability["snapshot"].available is True
    assert overview.availability["growth"].available is False
    assert overview.availability["growth"].error_code == "temporarily_unavailable"
