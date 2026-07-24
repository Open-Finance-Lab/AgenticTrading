"""T3: 60 s deadline default + the timeout_holds integrity counter."""

import dashboard.backend.domain.backtesting.external_run_service as ebs


def test_default_decision_timeout_is_60():
    # conftest strips EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS at import, so
    # the module constant IS the default.
    assert ebs.DECISION_TIMEOUT_SECONDS == 60


import numpy as np
import pandas as pd
import pytest

import dashboard.backend.database as db_module
from dashboard.backend.domain.backtesting import baseline_worker as bw


def _synth_bars(symbols, start, end):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si + np.linspace(0, 1.0, n)
        data[sym] = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0}, index=idx)
    return data


class _Loader:
    def fetch_bars(self, symbols, start, end):
        return _synth_bars(symbols, start, end)


@pytest.fixture
def session(monkeypatch, tmp_path):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "holds.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(bw.HourlyBacktester, "run_buyhold_baseline",
                        lambda self: (None, None))
    monkeypatch.setattr(bw.HourlyBacktester, "run_djia_baseline",
                        lambda self: (None, None))
    s = ebs.ExternalBacktestSession(
        backtest_id="bt_holds", session_id="sess_h", agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    s.load_market_data()
    return s, test_db


def test_expired_poll_increments_timeout_holds(session, monkeypatch):
    s, test_db = session
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 0.0)
    s.drain_expired()  # every step auto-holds
    assert s.status == "completed"
    assert s.timeout_holds == s.total_steps
    assert s.get_status()["timeout_holds"] == s.total_steps
    step_view = s.get_current_step()
    assert step_view["status"] == "completed"
    assert step_view["timeout_holds"] == s.total_steps
    row = test_db.get_run(s.run_id)
    assert row["metadata"]["timeout_holds"] == s.total_steps
    assert row["metadata"]["decision_timeout_seconds"] == 0.0
    metrics = ebs.build_final_metrics(row)
    assert metrics["timeout_holds"] == s.total_steps


def test_late_submit_increments_timeout_holds(session, monkeypatch):
    s, _ = session
    assert s.timeout_holds == 0
    monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 0.0)  # deadline now past
    res = s.submit_decisions({"actions": []})
    assert res["accepted"] is False and res["outcome"] == "timeout_hold"
    assert s.timeout_holds == 1


def test_clean_run_reports_zero_holds(session):
    s, test_db = session
    for _ in range(s.total_steps):
        s.submit_decisions({"actions": []})
    assert s.status == "completed"
    assert s.timeout_holds == 0
    assert test_db.get_run(s.run_id)["metadata"]["timeout_holds"] == 0
    assert ebs.get_run_result(s.run_id, "sess_h")["metrics"]["timeout_holds"] == 0
