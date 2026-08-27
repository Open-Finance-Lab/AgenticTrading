"""Cross-domain Analytics lifecycle wiring with synthetic source records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dashboard.backend import users as users_module
from dashboard.backend.api.routers import backtests as backtests_router
from dashboard.backend.api.v2 import runs as v2_runs
from dashboard.backend.app import app
from dashboard.backend.domain.analytics import instrumentation
from dashboard.backend.domain.analytics.query_service import (
    AnalyticsQueryService,
    get_analytics_query_service,
)
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.domain.analytics.service import (
    AnalyticsService,
    get_analytics_service,
)
from dashboard.backend.domain.analytics.states import (
    AnalyticsStateStore,
    recalculate_user_snapshot,
)
from dashboard.backend.domain.runs import service as run_service
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.tests._v2_fakes import FakeBackend
from dashboard.backend.users import UserStore


def test_protocol_owned_run_emits_requested_and_started_after_create(
    tmp_path,
    monkeypatch,
):
    store = RunStore(tmp_path / "analytics-protocol.db")
    events = []
    monkeypatch.setattr(run_service, "run_store", store)
    monkeypatch.setattr(
        run_service,
        "get_environment",
        lambda _environment_id: {
            "type": "backtest",
            "universe": ["AAPL"],
            "constraints": {},
        },
    )
    monkeypatch.setattr(
        run_service.ebs,
        "start_backtest",
        lambda **_kwargs: {"backtest_id": "bt-owned"},
    )
    monkeypatch.setattr(
        run_service.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        run_service,
        "run_view",
        lambda run_id: {"run_id": run_id, "status": "running"},
    )

    result = run_service.create_run(
        agent={
            "agent_id": "agent-owned",
            "session_id": "session-owned",
            "owner_user_id": 7,
            "name": "Owned",
            "model_name": "model",
        },
        agent_version={"agent_version_id": "version-1"},
        environment_id="backtest",
        config={
            "start_date": "2026-08-01",
            "end_date": "2026-08-02",
            "symbols": ["AAPL"],
        },
    )

    assert result["status"] == "running"
    assert [event["event_name"] for event in events] == [
        "backtest_requested",
        "backtest_started",
    ]
    assert all(event["run_id"] == result["run_id"] for event in events)


def test_v2_cancel_emits_cancelled_only_after_ledger_update(monkeypatch):
    events = []
    updates = []
    backend = FakeBackend(
        run_id="run-analytics-cancel",
        total_steps=2,
        session_id="session-owned",
    )
    v2_runs.register_run(
        "run-analytics-cancel",
        backend,
        "session-owned",
        "agent-owned",
    )
    monkeypatch.setattr(
        v2_runs.run_repo,
        "run_store",
        SimpleNamespace(
            update_run=lambda run_id, **kwargs: updates.append((run_id, kwargs))
        ),
    )
    monkeypatch.setattr(
        v2_runs.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: events.append(kwargs),
    )

    result = v2_runs.cancel_run(
        "run-analytics-cancel",
        agent={"session_id": "session-owned", "owner_user_id": 7},
    )

    assert result["status"] == "closed"
    assert updates[0][1]["status"] == "closed"
    assert [event["event_name"] for event in events] == [
        "backtest_cancelled"
    ]


def test_dashboard_finalizer_emits_terminal_event_for_authenticated_slot(
    monkeypatch,
):
    events = []
    monkeypatch.setattr(
        backtests_router.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: events.append(kwargs),
    )
    with backtests_router._backtest_slots_lock:
        backtests_router._active_slots["dashboard-run"] = {
            "live_run_id": "dashboard-run",
            "user_id": 7,
            "owner_session": "browser",
            "session_id": "agent-session",
            "running": True,
            "error": None,
            "runs_count": 0,
            "started_at": 1.0,
            "progress_file": None,
        }

    backtests_router._finalize_slot(
        "dashboard-run",
        error=None,
        runs_count=1,
    )

    assert [event["event_name"] for event in events] == [
        "backtest_completed"
    ]
    assert events[0]["user_id"] == 7


def test_dashboard_guest_slot_does_not_invent_analytics_subject(monkeypatch):
    events = []
    monkeypatch.setattr(
        backtests_router.analytics_instrumentation,
        "emit_run_event",
        lambda **kwargs: events.append(kwargs),
    )
    with backtests_router._backtest_slots_lock:
        backtests_router._active_slots["guest-run"] = {
            "live_run_id": "guest-run",
            "user_id": None,
            "owner_session": "browser",
            "session_id": "browser",
            "running": True,
            "error": None,
            "runs_count": 0,
            "started_at": 1.0,
            "progress_file": None,
        }

    backtests_router._finalize_slot("guest-run", error=None, runs_count=1)

    assert events == []


def test_synthetic_acceptance_scenario_has_no_real_credentials(monkeypatch):
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "analytics-acceptance.db"
        users = UserStore(path)
        admin = users.create_user(
            "admin@example.test",
            "Analytics Admin",
            "SecurePass1!",
        )
        users.apply_admin_patch(admin["id"], role="admin")
        subject = users.create_user(
            "subject@example.test",
            "Synthetic Subject",
            "SecurePass1!",
        )
        store = AnalyticsStore(path)
        event_service = AnalyticsService(store)
        query_service = AnalyticsQueryService(store=store, user_store=users)
        state_store = AnalyticsStateStore(store)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        monkeypatch.setattr(users_module, "user_store", users)
        monkeypatch.setattr(
            instrumentation,
            "get_analytics_service",
            lambda: event_service,
        )
        monkeypatch.setattr(
            instrumentation,
            "_snapshot_recalculator",
            lambda user_id: recalculate_user_snapshot(
                user_id,
                now=now,
                store=state_store,
            ),
        )
        app.dependency_overrides[get_analytics_query_service] = lambda: query_service
        app.dependency_overrides[get_analytics_service] = lambda: event_service

        instrumentation.emit_account_event(
            event_name="account_signed_up",
            user_id=subject["id"],
            source_record_id=subject["id"],
            occurred_at=now - timedelta(days=10),
        )
        instrumentation.emit_credential_event(
            event_name="credential_saved",
            user_id=subject["id"],
            credential_id="synthetic-credential",
            provider_id="openrouter",
            occurred_at=now - timedelta(hours=3),
        )
        instrumentation.emit_credential_event(
            event_name="credential_verified",
            user_id=subject["id"],
            credential_id="synthetic-credential",
            provider_id="openrouter",
            occurred_at=now - timedelta(hours=2, minutes=59),
        )
        instrumentation.emit_agent_event(
            event_name="agent_created",
            user_id=subject["id"],
            agent_id="synthetic-agent",
            occurred_at=now - timedelta(hours=2, minutes=50),
        )
        instrumentation.emit_run_event(
            event_name="backtest_failed",
            user_id=subject["id"],
            run_id="synthetic-byok-run",
            error_category="provider_timeout",
            occurred_at=now - timedelta(hours=2),
        )
        instrumentation.emit_resource_event(
            event_name="model_usage_recorded",
            user_id=subject["id"],
            source_record_type="run",
            source_record_id="synthetic-byok-run",
            correlation_id="synthetic-byok-run",
            provider_id="openrouter",
            model_id="openai/gpt-5.5",
            billing_mode="byok",
            outcome="failed",
            properties={
                "input_tokens": 100,
                "output_tokens": 20,
                "cost_micro_usd": 0,
            },
            occurred_at=now - timedelta(hours=1, minutes=59),
        )
        instrumentation.emit_run_event(
            event_name="backtest_completed",
            user_id=subject["id"],
            run_id="synthetic-platform-run",
            occurred_at=now - timedelta(hours=1),
        )
        instrumentation.emit_resource_event(
            event_name="model_usage_recorded",
            user_id=subject["id"],
            source_record_type="run",
            source_record_id="synthetic-platform-run",
            correlation_id="synthetic-platform-run",
            provider_id="openrouter",
            model_id="openai/gpt-5.5",
            billing_mode="platform_credits",
            outcome="succeeded",
            properties={
                "input_tokens": 200,
                "output_tokens": 40,
                "cost_micro_usd": 420_000,
            },
            occurred_at=now - timedelta(minutes=59),
        )
        instrumentation.emit_resource_event(
            event_name="credits_settled",
            user_id=subject["id"],
            source_record_type="credit_reservation",
            source_record_id="synthetic-reservation",
            correlation_id="synthetic-platform-run",
            billing_mode="platform_credits",
            properties={"amount_micro": 420, "bucket": "grant"},
            occurred_at=now - timedelta(minutes=58),
        )

        token = users.create_session(admin["id"])
        headers = {"Authorization": f"Bearer {token}"}
        try:
            with TestClient(app) as client:
                overview_response = client.get(
                    "/api/admin/analytics/overview",
                    headers=headers,
                )
                profile_response = client.get(
                    f"/api/admin/analytics/users/{subject['id']}",
                    headers=headers,
                )
        finally:
            app.dependency_overrides.pop(get_analytics_query_service, None)
            app.dependency_overrides.pop(get_analytics_service, None)

        assert overview_response.status_code == 200, overview_response.text
        assert profile_response.status_code == 200, profile_response.text
        overview = overview_response.json()
        profile = profile_response.json()
        assert overview["platform_model_cost_usd"] == 0.42
        assert profile["billing_lane_mix"] == {
            "byok": 1,
            "platform_credits": 1,
        }
        assert profile["credits_debited_micro"] == 420
        assert profile["state"]["status"] == "active"
        assert profile["state"]["evidence_event_ids"]
        assert "api_key" not in str(profile)
