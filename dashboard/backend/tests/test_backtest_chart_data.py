"""Focused contracts for the run-scoped Backtest chart-data route."""

import dashboard.backend.api.routers.backtests as bt
import pytest
from fastapi import HTTPException


def _curve():
    return [
        {"timestamp": "2026-05-04T14:30:00", "equity": 1_000.0},
        {"timestamp": "2026-05-04T15:30:00", "equity": 1_010.0},
    ]


def _install_route_fakes(monkeypatch, *, buyhold_curve, index_ok=True):
    run = {
        "run_id": "agent-1",
        "agent_name": "Agent",
        "start_date": "2026-05-04",
        "end_date": "2026-05-04",
        "initial_equity": 1_000.0,
        "baseline_buyhold_run_id": "buyhold-1",
        "metadata": {"data_source": "alpaca"},
    }
    monkeypatch.setattr(
        bt, "get_session_id_from_request", lambda _request: "session-1"
    )
    monkeypatch.setattr(
        bt.db,
        "get_run_with_session",
        lambda run_id, session_id: run
        if (run_id, session_id) == ("agent-1", "session-1")
        else None,
    )
    monkeypatch.setattr(
        bt.db,
        "get_run",
        lambda run_id: {"run_id": run_id, "agent_name": "buy-and-hold"}
        if run_id == "buyhold-1"
        else None,
    )
    monkeypatch.setattr(
        bt.db,
        "get_equity_curve",
        lambda run_id: _curve() if run_id == "agent-1" else buyhold_curve,
    )
    monkeypatch.setattr(bt, "_filter_equity_for_run", lambda _run, curve: curve)
    monkeypatch.setattr(
        bt.agent_service.agents,
        "get_agent_by_session",
        lambda _session_id: None,
    )

    def fake_indexes(timestamps, *_args, **_kwargs):
        indexes = [
            ("DJIA index", "index:^DJI", [1_000.0, 1_005.0]),
            ("Nasdaq-100", "index:^NDX", [1_000.0, 1_015.0]),
        ]
        return (indexes if index_ok else [], index_ok)

    monkeypatch.setattr(
        "dashboard.backend.equity_plot.market_index_baselines_with_status",
        fake_indexes,
    )


def test_us_chart_data_includes_buyhold_and_market_indexes(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve())

    response = bt.get_backtest_chart_data("agent-1", object())

    assert [series.run_id for series in response.series] == [
        "agent-1",
        "buyhold-1",
        "index:^DJI",
        "index:^NDX",
    ]


def test_missing_buyhold_curve_does_not_substitute_another_run(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=[])

    response = bt.get_backtest_chart_data("agent-1", object())

    assert "buyhold-1" not in [series.run_id for series in response.series]
    assert response.index_baselines_ok is True


def test_index_failure_retains_agent_and_buyhold(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve(), index_ok=False)

    response = bt.get_backtest_chart_data("agent-1", object())

    assert [series.run_id for series in response.series] == [
        "agent-1",
        "buyhold-1",
    ]
    assert response.index_baselines_ok is False


def test_chart_data_keeps_selected_run_session_scoped(monkeypatch):
    _install_route_fakes(monkeypatch, buyhold_curve=_curve())
    monkeypatch.setattr(
        bt, "get_session_id_from_request", lambda _request: "other-session"
    )

    with pytest.raises(HTTPException) as exc_info:
        bt.get_backtest_chart_data("agent-1", object())

    assert exc_info.value.status_code == 404
