"""Authoritative Analytics backfill is bounded, safe, and replayable."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.domain.analytics.backfill import (
    AuthoritativeBackfillSource,
    BackfillCandidate,
    BackfillCollection,
    _usage_candidate,
    backfill_analytics,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import AnalyticsService


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc)


def test_one_call_backfill_uses_actual_provider_after_failover():
    candidate = _usage_candidate(
        {
            "run_id": "fallback-one-call",
            "created_at": NOW.isoformat(),
            "metadata": {
                "llm_execution": {
                    "billing_mode": "platform_credits",
                    "requested_provider_id": "openrouter",
                    "provider_id": "commonstack",
                    "provider_ids": ["commonstack"],
                    "provider_mixed": False,
                    "model_id": "qwen/qwen3.7-plus",
                    "call_count": 1,
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "usage_available": True,
                    "provider_cost_usd": 0.001,
                    "estimated_cost_usd": 0.001,
                    "pricing_snapshot": None,
                    "debited_credits_micro": 1_000,
                    "outstanding_credits_micro": 0,
                    "outcome": "settled",
                }
            },
        },
        user_id=1,
    )

    assert candidate is not None
    assert candidate.provider_id == "commonstack"
    assert candidate.model_id == "qwen/qwen3.7-plus"


class StaticSource:
    def __init__(self, candidates, *, skipped_unmapped_owner=0, skipped_invalid=0):
        self.collection = BackfillCollection(
            candidates=list(candidates),
            skipped_unmapped_owner=skipped_unmapped_owner,
            skipped_invalid=skipped_invalid,
        )

    def collect(self, *, start, end):
        return self.collection


def _analytics_store(tmp_path):
    path = tmp_path / "backfill.db"
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
                (1, "one@example.test", "One", (NOW - timedelta(days=10)).isoformat()),
                (2, "two@example.test", "Two", (NOW - timedelta(days=5)).isoformat()),
            ],
        )
    return AnalyticsStore(path)


def _candidate(
    event_name,
    source_event_id,
    occurred_at,
    *,
    user_id=1,
    source_record_type="run",
    source_record_id="run-1",
    **kwargs,
):
    return BackfillCandidate(
        event_name=event_name,
        user_id=user_id,
        occurred_at=occurred_at,
        source_event_id=source_event_id,
        source_record_type=source_record_type,
        source_record_id=source_record_id,
        **kwargs,
    )


def test_backfill_uses_only_authoritative_candidates(tmp_path):
    store = _analytics_store(tmp_path)
    source = StaticSource(
        [
            _candidate(
                "account_signed_up",
                "account:account_signed_up:1",
                NOW - timedelta(days=10),
                source_record_type="user",
                source_record_id="1",
            ),
            _candidate(
                "agent_created",
                "agent:agent_created:agent-1",
                NOW - timedelta(days=9),
                source_record_type="agent",
                source_record_id="agent-1",
            ),
            _candidate(
                "backtest_completed",
                "run:backtest_completed:run-1",
                NOW - timedelta(days=8),
                outcome="succeeded",
            ),
        ]
    )

    report = backfill_analytics(
        now=NOW,
        source=source,
        store=store,
        recalculate_snapshot=lambda *_args, **_kwargs: None,
        rebuild_rollup=lambda *_args, **_kwargs: None,
    )
    events = store.list_user_events(1, limit=50)["items"]

    assert report.inserted == 3
    assert all(event.event_source == "backfill" for event in events)
    assert all(event.session_id is None and event.page_view is None for event in events)
    assert all(event.network_hash is None for event in events)
    assert not any(event.event_name == "page_viewed" for event in events)


def test_backfill_ids_are_deterministic_and_idempotent(tmp_path):
    store = _analytics_store(tmp_path)
    candidate = _candidate(
        "backtest_failed",
        "run:backtest_failed:run-1",
        NOW - timedelta(days=2),
        outcome="failed",
        error_category="internal_error",
    )
    source = StaticSource([candidate])

    first = backfill_analytics(
        now=NOW,
        source=source,
        store=store,
        recalculate_snapshot=lambda *_args, **_kwargs: None,
        rebuild_rollup=lambda *_args, **_kwargs: None,
    )
    first_event_id = store.list_user_events(1, limit=10)["items"][0].event_id
    second = backfill_analytics(
        now=NOW,
        source=source,
        store=store,
        recalculate_snapshot=lambda *_args, **_kwargs: None,
        rebuild_rollup=lambda *_args, **_kwargs: None,
    )
    second_event_id = store.list_user_events(1, limit=10)["items"][0].event_id

    assert first.inserted == 1
    assert second.inserted == 0
    assert second.duplicates == 1
    assert first.source_event_ids == ["run:backtest_failed:run-1"]
    assert first_event_id == second_event_id


def test_backfill_treats_existing_live_source_event_as_duplicate(tmp_path):
    store = _analytics_store(tmp_path)
    at = NOW - timedelta(days=1)
    AnalyticsService(store).record_server_event(
        event_name="backtest_completed",
        user_id=1,
        source_event_id="run:backtest_completed:run-live",
        source_record_type="run",
        source_record_id="run-live",
        correlation_id="run-live",
        outcome="succeeded",
        occurred_at=at,
    )

    report = backfill_analytics(
        now=NOW,
        source=StaticSource(
            [
                _candidate(
                    "backtest_completed",
                    "run:backtest_completed:run-live",
                    at,
                    source_record_id="run-live",
                    correlation_id="run-live",
                    outcome="succeeded",
                )
            ]
        ),
        store=store,
        recalculate_snapshot=lambda *_args, **_kwargs: None,
        rebuild_rollup=lambda *_args, **_kwargs: None,
    )

    assert report.inserted == 0
    assert report.duplicates == 1
    assert len(store.list_user_events(1, limit=10)["items"]) == 1


def test_backfill_enforces_window_dry_run_and_bulk_repairs(tmp_path):
    store = _analytics_store(tmp_path)
    snapshots = []
    rollup_days = []
    source = StaticSource(
        [
            _candidate(
                "backtest_completed",
                "run:backtest_completed:old",
                NOW - timedelta(days=181),
                source_record_id="old",
                outcome="succeeded",
            ),
            _candidate(
                "backtest_completed",
                "run:backtest_completed:yesterday",
                NOW - timedelta(days=1),
                source_record_id="yesterday",
                outcome="succeeded",
            ),
            _candidate(
                "agent_created",
                "agent:agent_created:today",
                NOW - timedelta(hours=1),
                user_id=2,
                source_record_type="agent",
                source_record_id="today",
            ),
        ],
        skipped_unmapped_owner=2,
    )

    dry = backfill_analytics(
        now=NOW,
        source=source,
        store=store,
        dry_run=True,
        recalculate_snapshot=lambda user_id, **_kwargs: snapshots.append(user_id),
        rebuild_rollup=lambda day, **_kwargs: rollup_days.append(day),
    )
    written = backfill_analytics(
        now=NOW,
        source=source,
        store=store,
        recalculate_snapshot=lambda user_id, **_kwargs: snapshots.append(user_id),
        rebuild_rollup=lambda day, **_kwargs: rollup_days.append(day),
    )

    assert dry.would_insert == 2
    assert dry.inserted == 0
    assert written.inserted == 2
    assert written.skipped_outside_window == 1
    assert written.skipped_unmapped_owner == 2
    assert snapshots == [1, 2]
    assert rollup_days == [(NOW - timedelta(days=1)).date()]


def test_dry_run_validates_event_properties_without_writing(tmp_path):
    store = _analytics_store(tmp_path)
    report = backfill_analytics(
        now=NOW,
        dry_run=True,
        source=StaticSource(
            [
                _candidate(
                    "credits_reserved",
                    "resource:credits_reserved:invalid:grant",
                    NOW - timedelta(days=1),
                    source_record_type="credit_reservation",
                    source_record_id="invalid",
                    billing_mode="platform_credits",
                    properties={"amount_micro": 0, "bucket": "grant"},
                )
            ]
        ),
        store=store,
    )

    assert report.would_insert == 0
    assert report.skipped_invalid == 1
    assert store.list_user_events(1, limit=10)["items"] == []


@pytest.mark.parametrize("days", [0, 181, True])
def test_backfill_rejects_invalid_day_bounds(tmp_path, days):
    with pytest.raises(ValueError, match="days"):
        backfill_analytics(
            now=NOW,
            days=days,
            source=StaticSource([]),
            store=_analytics_store(tmp_path),
        )


class SqliteRows:
    def __init__(self, path):
        self.path = path

    def _get_connection(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn


class FakeUsers:
    def list_users_admin(self, *, limit, offset, query=None):
        del limit, query
        rows = [
            {
                "id": 1,
                "email": "one@example.test",
                "display_name": "One",
                "role": "user",
                "created_at": (NOW - timedelta(days=10)).isoformat(),
            }
        ]
        return rows[offset:]


class FakeProtocolRuns:
    def list_runs(self, agent_id):
        assert agent_id == "agent-1"
        return [
            {
                "run_id": "protocol-1",
                "status": "failed",
                "created_at": (NOW - timedelta(days=4)).isoformat(),
                "updated_at": (NOW - timedelta(days=3)).isoformat(),
                "result_run_id": None,
            }
        ]


class FakeRunHistory:
    def get_runs_by_sessions(self, session_ids):
        assert session_ids == ["session-1"]
        return {
            "session-1": [
                {
                    "run_id": "legacy-1",
                    "mode": "backtest",
                    "created_at": (NOW - timedelta(days=2)).isoformat(),
                    "updated_at": (NOW - timedelta(days=2)).isoformat(),
                    "metadata": {
                        "llm_execution": {
                            "billing_mode": "byok",
                            "provider_id": "openrouter",
                            "model_id": "openai/gpt-5.5",
                            "credential_id": "credential-reference",
                            "credential_key_last_four": "1234",
                            "call_count": 1,
                            "input_tokens": 100,
                            "output_tokens": 25,
                            "usage_available": True,
                            "provider_cost_usd": None,
                            "estimated_cost_usd": None,
                            "pricing_snapshot": None,
                            "debited_credits_micro": 0,
                            "outstanding_credits_micro": 0,
                            "outcome": "byok",
                        }
                    },
                }
            ]
        }


def test_authoritative_source_combines_safe_independent_store_evidence(tmp_path):
    content_path = tmp_path / "content.db"
    with sqlite3.connect(content_path) as conn:
        conn.execute(
            """
            CREATE TABLE external_agents (
                agent_id TEXT, session_id TEXT, owner_user_id INTEGER, created_at TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO external_agents VALUES (?, ?, ?, ?)",
            [
                (
                    "agent-1",
                    "session-1",
                    1,
                    (NOW - timedelta(days=9)).isoformat(),
                ),
                (
                    "guest-agent",
                    "guest-session",
                    None,
                    (NOW - timedelta(days=8)).isoformat(),
                ),
            ],
        )
    credits_path = tmp_path / "credits.db"
    with sqlite3.connect(credits_path) as conn:
        conn.execute(
            """
            CREATE TABLE credit_llm_reservations (
                reservation_id TEXT, user_id INTEGER, run_id TEXT, call_index INTEGER,
                reserved_grant_micro INTEGER, reserved_purchased_micro INTEGER,
                status TEXT, created_at TEXT, updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE credit_llm_usage_entries (
                id INTEGER, user_id INTEGER, reservation_id TEXT, run_id TEXT,
                call_index INTEGER, bucket TEXT, amount_micro INTEGER, created_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO credit_llm_reservations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "reservation-1",
                1,
                "legacy-1",
                0,
                90,
                10,
                "settled",
                (NOW - timedelta(days=2, hours=1)).isoformat(),
                (NOW - timedelta(days=2)).isoformat(),
            ),
        )
        conn.execute(
            "INSERT INTO credit_llm_usage_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                1,
                "reservation-1",
                "legacy-1",
                0,
                "grant",
                -80,
                (NOW - timedelta(days=2)).isoformat(),
            ),
        )

    collection = AuthoritativeBackfillSource(
        user_store=FakeUsers(),
        agent_store=SqliteRows(content_path),
        protocol_run_store=FakeProtocolRuns(),
        run_history_store=FakeRunHistory(),
        credits_store=SqliteRows(credits_path),
    ).collect(start=NOW - timedelta(days=180), end=NOW)
    by_source_id = {item.source_event_id: item for item in collection.candidates}

    assert collection.skipped_unmapped_owner == 1
    assert "account:account_signed_up:1" in by_source_id
    assert "agent:agent_created:agent-1" in by_source_id
    assert "run:backtest_failed:protocol-1" in by_source_id
    assert "run:backtest_completed:legacy-1" in by_source_id
    assert "resource:model_usage_recorded:legacy-1:0" in by_source_id
    assert "resource:credits_reserved:reservation-1:grant" in by_source_id
    assert "resource:credits_settled:reservation-1:grant" in by_source_id
    assert "resource:credits_refunded:reservation-1:grant" in by_source_id
    usage = by_source_id["resource:model_usage_recorded:legacy-1:0"]
    assert usage.billing_mode == "byok"
    assert usage.properties["cost_micro_usd"] == 0
    assert "credential" not in str(usage.model_dump()).lower()
