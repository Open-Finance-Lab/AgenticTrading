"""Two same-config sessions share one dataset object; the loader runs once."""

import numpy as np
import pandas as pd
import pytest

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds


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


class _CountingLoader:
    calls = 0

    def fetch_bars(self, symbols, start, end):
        type(self).calls += 1
        return _synth_bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _CountingLoader)
    monkeypatch.setattr(ebs, "_sessions", {})  # keep the global registry test-local
    _CountingLoader.calls = 0
    yield


def _session(i):
    return ebs.ExternalBacktestSession(
        backtest_id=f"bt_share_{i}", session_id=f"sess_{i}", agent_name=f"a{i}",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )


def test_same_config_sessions_share_one_dataset_and_one_fetch():
    s1, s2 = _session(1), _session(2)
    s1.load_market_data()
    s2.load_market_data()
    assert _CountingLoader.calls == 1
    assert s1.all_data is s2.all_data          # shared object identity
    assert s1.price_cache is s2.price_cache
    assert s1.timestamps is s2.timestamps
    assert s1.status == s2.status == "waiting_decision"
    assert s1.total_steps == s2.total_steps > 0


def test_adopt_dataset_respects_terminal_status():
    ds = mds.get_dataset(["AAPL", "MSFT"], "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    s = _session(3)
    s.status = "closed"  # cancelled while loading
    s.adopt_dataset(ds)
    assert s.status == "closed"  # never resurrected
    assert s.total_steps > 0     # data attached is fine; status is not touched


def test_v1_start_backtest_fast_path_skips_loader_thread():
    # Warm the cache through a normal session load.
    _session(0).load_market_data()
    assert _CountingLoader.calls == 1

    res = ebs.start_backtest(
        session_id="sess_fast", agent_name="fast", model_name="m",
        start_date="2026-04-15", end_date="2026-04-16",
    )
    assert res["status"] == "loading"  # wire literal is frozen (Decision 5)
    session = ebs.get_session(res["backtest_id"])
    # Resident hit: step 0 opened synchronously — no loader thread, no fetch.
    assert session.status == "waiting_decision"
    assert _CountingLoader.calls == 1


def test_v2_start_background_load_fast_path(monkeypatch):
    import dashboard.backend.execution.backtest_backend as bb_mod

    _session(0).load_market_data()  # warm
    assert _CountingLoader.calls == 1

    row_updates = []

    class _FakeRunStore:
        def update_run(self, run_id, **kw):
            row_updates.append((run_id, kw))

    monkeypatch.setattr(bb_mod.run_repo, "run_store", _FakeRunStore())
    backend = bb_mod.BacktestBackend(
        run_id="run_fast", session_id="sess_v2", agent_name="a", model_name="m",
        start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.start_background_load()
    assert backend.session.status == "waiting_decision"  # no thread needed
    assert _CountingLoader.calls == 1
    assert ("run_fast", {"status": "running"}) in row_updates
