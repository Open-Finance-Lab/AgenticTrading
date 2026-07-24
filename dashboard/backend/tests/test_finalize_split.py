"""Finalize split (T2): the final submit persists + completes in-request;
baselines arrive asynchronously via the worker; polled surfaces self-heal."""

import threading

import numpy as np
import pandas as pd
import pytest

import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as ebs
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


class _FakeBacktester:
    instances = 0
    gate = None  # when set (threading.Event), baselines wait on it

    def __init__(self, start, end, session_id, use_llm=False, mode="safe_trading"):
        type(self).instances += 1
        self.all_data = None

    def run_buyhold_baseline(self):
        if type(self).gate is not None:
            type(self).gate.wait(10)
        return "buyhold_shared", []

    def run_djia_baseline(self):
        return "djia_shared", []


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    test_db = db_module.BacktestDatabase(db_path=tmp_path / "finalize_split.db")
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(ebs, "db", test_db)
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _Loader)
    monkeypatch.setattr(ebs, "_sessions", {})
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    bw._reset_for_tests()
    _FakeBacktester.instances = 0
    _FakeBacktester.gate = None
    yield test_db
    bw._reset_for_tests()


def _run_to_completion(i):
    s = ebs.ExternalBacktestSession(
        backtest_id=f"bt_fin_{i}", session_id=f"sess_{i}", agent_name=f"a{i}",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    s.load_market_data()
    last = None
    for _ in range(s.total_steps):
        last = s.submit_decisions({"actions": []})
    return s, last


def test_final_submit_completes_before_baselines(_isolate):
    _FakeBacktester.gate = threading.Event()  # hold the worker mid-baseline
    s, last = _run_to_completion(0)
    # The one-shot decision response: completed + persisted, baselines pending.
    assert last["status"] == "completed"
    assert last["run_id"] == s.run_id
    assert last["metrics"]["final_equity"] is not None       # results persisted
    assert last["compare_url"] == f"/compare?run_ids={s.run_id}"  # no baseline ids
    assert s.baseline_run_ids == {}
    assert _isolate.get_run(s.run_id) is not None
    _FakeBacktester.gate.set()
    assert bw.wait_idle(10)
    # Polled surfaces self-heal once the worker lands.
    status = s.get_status()
    assert status["baseline_run_ids"] == {"buy_and_hold": "buyhold_shared",
                                          "djia": "djia_shared"}
    assert "buyhold_shared" in status["compare_url"]
    row = _isolate.get_run(s.run_id)
    assert row["baseline_buyhold_run_id"] == "buyhold_shared"
    assert row["baseline_djia_run_id"] == "djia_shared"


def test_same_config_finalizes_share_one_baseline_pair(_isolate):
    s1, _ = _run_to_completion(1)
    s2, _ = _run_to_completion(2)
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 1  # 2 runs, 1 baseline backtester
    assert s1.baseline_run_ids == s2.baseline_run_ids
    assert s1.baseline_run_ids is not s2.baseline_run_ids  # swapped-in copies
    assert _isolate.get_run(s1.run_id)["baseline_djia_run_id"] == "djia_shared"
    assert _isolate.get_run(s2.run_id)["baseline_djia_run_id"] == "djia_shared"


def test_evicted_session_still_gets_db_baselines(_isolate):
    s, _ = _run_to_completion(3)
    ebs.evict_session(s.backtest_id)  # reaper freed the session before the worker ran
    assert bw.wait_idle(10)
    row = _isolate.get_run(s.run_id)
    assert row["baseline_buyhold_run_id"] == "buyhold_shared"  # DB is source of truth
