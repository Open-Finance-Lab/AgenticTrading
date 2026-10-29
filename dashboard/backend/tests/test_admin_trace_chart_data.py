"""The admin trace chart reuses the Backtest panel's payload for the exact run."""

import pytest
from fastapi import HTTPException

import dashboard.backend.api.routers.backtests as bt
from dashboard.backend.api.routers import admin_traces as api
from dashboard.backend.database import db


def _curve():
    return [
        {"timestamp": "2026-05-04T14:30:00", "equity": 1_000.0},
        {"timestamp": "2026-05-04T15:30:00", "equity": 1_010.0},
    ]


def _install(monkeypatch, *, run, curves):
    monkeypatch.setattr(api.trace_store, "get_trace", lambda _: {"run_id": "agent-1", "status": "completed"})
    calls = []

    def get_run(run_id):
        calls.append(run_id)
        return run if run_id == "agent-1" else {"run_id": run_id, "agent_name": "buy-and-hold"}

    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_equity_curve", lambda run_id: curves.get(run_id, []))
    monkeypatch.setattr(bt, "_filter_equity_for_run", lambda _run, curve: curve)

    def fake_indexes(timestamps, *_args, **_kwargs):
        return ([
            ("DJIA index", "index:^DJI", [1_000.0, 1_005.0]),
            ("Nasdaq-100", "index:^NDX", [1_000.0, 1_015.0]),
        ], True)

    monkeypatch.setattr("dashboard.backend.equity_plot.market_index_baselines_with_status", fake_indexes)
    return calls


def test_chart_data_has_the_backtest_series_for_the_exact_run(monkeypatch):
    run = {
        "run_id": "agent-1", "agent_name": "Agent", "start_date": "2026-05-04", "end_date": "2026-05-04",
        "initial_equity": 1_000.0, "baseline_buyhold_run_id": "buyhold-1",
        "metadata": {"data_source": "alpaca"},
    }
    calls = _install(monkeypatch, run=run, curves={"agent-1": _curve(), "buyhold-1": _curve()})
    result = api.get_trace_chart_data("trace")
    assert calls[0] == "agent-1"
    assert result["available"] is True
    assert result["agent_run_id"] == "agent-1"
    assert [s["run_id"] for s in result["series"]] == ["agent-1", "buyhold-1", "index:^DJI", "index:^NDX"]
    assert result["series"][0]["dashed"] is False
    assert all(s["dashed"] for s in result["series"][1:])
    assert result["run"] == {"data_source": "alpaca", "baseline_buyhold_run_id": "buyhold-1", "reporting_currency": None}


def test_a_run_without_stored_equity_is_unavailable_not_an_error(monkeypatch):
    _install(monkeypatch, run={"run_id": "agent-1", "metadata": {}}, curves={})
    assert api.get_trace_chart_data("trace") == {"available": False, "run_id": "agent-1"}


def test_a_trace_whose_run_is_gone_is_unavailable(monkeypatch):
    monkeypatch.setattr(api.trace_store, "get_trace", lambda _: {"run_id": "gone", "status": "failed"})
    monkeypatch.setattr(db, "get_run", lambda _: None)
    assert api.get_trace_chart_data("trace") == {"available": False, "run_id": "gone"}


def test_missing_trace_is_404(monkeypatch):
    monkeypatch.setattr(api.trace_store, "get_trace", lambda _: None)
    with pytest.raises(HTTPException) as exc:
        api.get_trace_chart_data("missing")
    assert exc.value.status_code == 404


def test_chart_data_route_keeps_admin_dependency():
    from dashboard.backend.api.auth import require_admin
    route = next(r for r in api.router.routes if r.path.endswith("/chart-data"))
    assert require_admin in [dependency.call for dependency in route.dependant.dependencies]
