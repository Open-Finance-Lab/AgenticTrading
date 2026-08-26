"""Cross-domain Analytics lifecycle wiring with synthetic source records."""

from __future__ import annotations

from types import SimpleNamespace

from dashboard.backend.api.routers import backtests as backtests_router
from dashboard.backend.api.v2 import runs as v2_runs
from dashboard.backend.domain.runs import service as run_service
from dashboard.backend.domain.runs.repository import RunStore
from dashboard.backend.tests._v2_fakes import FakeBackend


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
