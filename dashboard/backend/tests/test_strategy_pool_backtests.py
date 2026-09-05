"""Selected pools survive API, worker, CLI and trading-engine boundaries."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.domain.backtesting import engine
from dashboard.backend.infrastructure.market_data.strategy_universe import resolve_strategy_universe
from dashboard.backend.infrastructure.llm.validator import DJIA_30, create_prompt
from dashboard.backend.infrastructure.llm.backtest_harness import system_prompt_for_market
from dashboard.backend.tests.infrastructure.market_data.test_strategy_universe import catalog


class WorkerSpy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass


@pytest.mark.parametrize("pool", ["ordinary", "fund", "large_cap", "small_mid_cap", "all"])
@pytest.mark.parametrize("mode", ["top30", "all"])
def test_api_resolves_and_freezes_all_options(catalog, monkeypatch, pool, mode):
    threads = []

    def thread(**kwargs):
        threads.append(kwargs)
        return WorkerSpy(**kwargs)

    monkeypatch.setattr(backtests, "_BackgroundThread", thread)
    response = TestClient(app).post("/backtest/run", json={
        "start_date": "2026-05-01", "end_date": "2026-05-02",
        "decision_source": "rule_based", "stock_pool": pool, "pool_mode": mode,
    }, headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 200, response.text
    expected = resolve_strategy_universe(pool, mode)
    assert response.json()["assets"] == expected["symbols"]
    assert response.json()["universe_selection"] == expected
    assert threads[0]["kwargs"]["universe_selection"] == expected


@pytest.mark.parametrize("fields,status", [
    ({"pool_mode": "all"}, 422), ({"stock_pool": "typo"}, 422),
    ({"stock_pool": "fund", "pool_mode": "typo"}, 422),
    ({"stock_pool": "fund", "assets": ["AAPL"]}, 422),
    ({"stock_pool": "ordinary", "data_source": "ifind_ashare"}, 422),
    ({"assets": [f"S{i:03}" for i in range(31)]}, 422),
])
def test_invalid_requests_never_schedule(catalog, monkeypatch, fields, status):
    def unexpected(**kwargs):
        pytest.fail("invalid request scheduled a worker")
    monkeypatch.setattr(backtests, "_BackgroundThread", unexpected)
    response = TestClient(app).post("/backtest/run", json={
        "decision_source": "rule_based", **fields,
    }, headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == status, response.text


def test_query_switch_and_body_override(catalog, monkeypatch):
    monkeypatch.setattr(backtests, "_BackgroundThread", WorkerSpy)
    client = TestClient(app)
    response = client.post("/backtest/run?stock_pool=ordinary&pool_mode=top30", json={
        "decision_source": "rule_based", "stock_pool": "fund", "pool_mode": "all",
    }, headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 40
    assert response.json()["universe_selection"]["stock_pool"] == "fund"
    response = client.post("/backtest/run?stock_pool=fund&pool_mode=all&decision_source=rule_based",
                           headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 40


def test_unconfigured_reference_data_returns_actionable_503(catalog, monkeypatch):
    catalog[0].unlink()
    monkeypatch.setattr(backtests, "_BackgroundThread", lambda **_: pytest.fail("scheduled without reference data"))
    response = TestClient(app).post("/backtest/run", json={
        "stock_pool": "large_cap", "pool_mode": "all", "decision_source": "rule_based",
    }, headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 503
    assert "US_EQUITY_CATALOG" in response.json()["detail"]


def test_default_and_options_endpoint(monkeypatch):
    monkeypatch.setattr(backtests, "_BackgroundThread", WorkerSpy)
    client = TestClient(app)
    response = client.post("/backtest/run?decision_source=rule_based",
                           headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 200
    assert response.json()["assets"] == list(DJIA_30)
    assert "universe_selection" not in response.json()
    assert client.get("/config/stock-pools").json()["pool_modes"] == ["top30", "all", "representative30"]


def test_worker_transfers_snapshot_by_file_and_cleans_up(catalog, monkeypatch):
    selection = resolve_strategy_universe("all", "all")
    captured = {}

    def fake_run(command, **kwargs):
        assert "--assets" not in command
        path = Path(command[command.index("--universe-selection-file") + 1])
        captured["path"] = path
        assert json.loads(path.read_text()) == selection
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda *_: [])
    backtests.run_backtest_background("2026-05-01", "2026-05-02", "test-pool",
                                     decision_source="rule_based", universe_selection=selection)
    assert not captured["path"].exists()


def test_engine_uses_full_pool_for_data_orders_and_metadata(catalog, monkeypatch):
    selection = resolve_strategy_universe("fund", "all")
    fetched = []
    timestamps = pd.date_range("2026-05-01 10:00", periods=2, freq="h", tz="US/Eastern")
    frame = pd.DataFrame({"open": [10., 10.], "high": [11., 11.], "low": [9., 9.],
                          "close": [10., 10.], "volume": [100., 100.], "rsi_14": [50., 50.],
                          "sma20": [9., 9.], "sma50": [12., 12.]}, index=timestamps)

    class Provider:
        def fetch_bars(self, symbols, start, end):
            fetched.extend(symbols)
            data = {symbol: frame.copy() for symbol in symbols}
            data[symbols[-1]]["rsi_14"] = 20.
            data[symbols[-1]]["sma20"] = 11.
            return data

    monkeypatch.setattr(engine, "create_market_data_provider", lambda *_: Provider())
    bt = engine.HourlyBacktester("2026-05-01", "2026-05-02", use_llm=False,
                                 universe_selection=selection, initial_capital=100_000)
    catalog[0].unlink()  # Worker must never re-resolve a refreshed/deleted catalog.
    bt.load_data()
    run_id, equity = bt.run_agent_backtest()
    assert fetched == selection["symbols"]
    assert len(equity) == 2
    stored = backtests.db.get_run(run_id)
    assert stored["metadata"]["universe_selection"] == selection
    assert bt._llm_market_context()["symbols"] == selection["symbols"]
    assert bt._llm_market_context()["stock_pool"] == "fund"
    assert {t["symbol"] for t in backtests.db.get_trades(run_id)} == {"F039"}


def test_cli_switch_reaches_engine(catalog, monkeypatch):
    from dashboard.scripts import backtest_hourly_agent as cli
    captured = {}

    class StopAtEngine(Exception):
        pass

    def factory(*args, **kwargs):
        captured.update(kwargs)
        raise StopAtEngine

    monkeypatch.setattr(cli, "HourlyBacktester", factory)
    monkeypatch.setattr(sys, "argv", ["backtest_hourly_agent", "--no-llm", "--stock-pool", "fund", "--pool-mode", "all"])
    with pytest.raises(StopAtEngine):
        cli.main()
    assert captured["universe_selection"]["symbols"] == resolve_strategy_universe("fund", "all")["symbols"]


@pytest.mark.parametrize("mode", ["safe_trading", "buy_and_hold"])
def test_pool_prompts_do_not_instruct_funds_to_buy_dow_stocks(mode):
    context = {"market": "US", "stock_pool": "fund", "symbols": ["SPY", "QQQ"]}
    prompt = create_prompt({"market": context}, mode=mode, allowed_symbols=context["symbols"])
    assert "DJIA" not in prompt
    assert "AAPL" not in prompt
    assert "SPY, QQQ" in prompt
    assert "DJIA" not in system_prompt_for_market(context)
