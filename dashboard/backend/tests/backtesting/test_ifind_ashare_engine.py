"""iFinD A-share profile behavior at the HourlyBacktester boundary."""

from __future__ import annotations

from datetime import datetime, time, timedelta
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.llm import backtest_harness as llm_harness
from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6,
    A_SHARE_DEMO_6_SYMBOLS,
    CSI300_SAMPLE_20_2026H2,
    CSI300_SAMPLE_20_2026H2_SYMBOLS,
    IFIND_ASHARE,
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
)
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)


CN = ZoneInfo("Asia/Shanghai")
START = "2026-04-01"
END = "2026-05-01"


class RecordingProvider:
    def __init__(self, bars):
        self.bars = bars
        self.calls = []

    def fetch_bars(self, symbols, start, end):
        self.calls.append((symbols, start, end))
        return {symbol: self.bars[symbol] for symbol in symbols}


class RecordingDB:
    def __init__(self):
        self.runs = []
        self.equity_points = []
        self.trades = []

    def insert_run(self, **kwargs):
        self.runs.append(kwargs)

    def insert_equity_points(self, run_id, points):
        self.equity_points.append((run_id, list(points)))

    def insert_trades(self, run_id, trades):
        self.trades.append((run_id, list(trades)))


def make_cn_bars(symbols=A_SHARE_DEMO_6_SYMBOLS, count=60):
    timestamps = []
    current = datetime(2026, 4, 1).date()
    sessions = (time(10, 30), time(11, 30), time(14), time(15))
    while len(timestamps) < count:
        if current.weekday() < 5:
            timestamps.extend(
                datetime.combine(current, session, tzinfo=CN)
                for session in sessions
            )
        current += timedelta(days=1)
    index = pd.DatetimeIndex(timestamps[:count], name="timestamp")
    bars = {}
    for offset, symbol in enumerate(symbols):
        prices = [100.0 + offset * 10 + row * 0.05 for row in range(count)]
        bars[symbol] = pd.DataFrame(
            {
                "open": prices,
                "high": [price + 1 for price in prices],
                "low": [price - 1 for price in prices],
                "close": prices,
                "volume": [10_000] * count,
            },
            index=index,
        )
    return bars


def test_ifind_engine_uses_profile_symbols_and_forces_rule_mode(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(engine_module, "create_market_data_provider", lambda _source, universe=None: provider)
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)

    def fail_llm_client():
        raise AssertionError("iFinD A-share mode must not initialize an LLM")

    monkeypatch.setattr(engine_module, "make_llm_client", fail_llm_client)
    recording_db = RecordingDB()
    monkeypatch.setattr(engine_module, "db", recording_db)

    backtester = HourlyBacktester(
        START,
        END,
        session_id="ifind-engine-test",
        use_llm=True,
        data_source=IFIND_ASHARE,
    )

    assert backtester.use_llm is False
    assert backtester.llm_client is None
    assert backtester.symbols == A_SHARE_DEMO_6_SYMBOLS

    backtester.load_data()
    assert provider.calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    backtester.calculate_indicators()

    agent_id, agent_curve = backtester.run_agent_backtest()
    buyhold_id, buyhold_curve = backtester.run_buyhold_baseline()
    djia_id, djia_curve = backtester.run_djia_baseline()

    assert agent_id and agent_curve
    assert buyhold_id and buyhold_curve
    assert djia_id is None
    assert djia_curve == []
    assert len(recording_db.runs) == 2

    agent_metadata = recording_db.runs[0]["metadata"]
    assert agent_metadata == {
        "data_source": IFIND_ASHARE,
        "market": "CN",
        "universe": "a_share_demo_6",
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
        "symbols": list(A_SHARE_DEMO_6_SYMBOLS),
    }
    assert {run["metadata"]["data_source"] for run in recording_db.runs} == {
        IFIND_ASHARE
    }
    assert all("DJIA" not in run["agent_name"] for run in recording_db.runs)


def test_ifind_engine_resolves_csi300_sample20_and_records_provenance(
    monkeypatch,
):
    provider = RecordingProvider(make_cn_bars(CSI300_SAMPLE_20_2026H2_SYMBOLS))
    factory_calls = []

    def factory(data_source, universe=None):
        factory_calls.append((data_source, universe))
        return provider

    monkeypatch.setattr(engine_module, "create_market_data_provider", factory)

    backtester = HourlyBacktester(
        START,
        END,
        use_llm=True,
        data_source=IFIND_ASHARE,
        universe=CSI300_SAMPLE_20_2026H2,
    )
    backtester.load_data()

    assert factory_calls == [(IFIND_ASHARE, CSI300_SAMPLE_20_2026H2)]
    assert backtester.symbols == CSI300_SAMPLE_20_2026H2_SYMBOLS
    assert provider.calls == [(CSI300_SAMPLE_20_2026H2_SYMBOLS, START, END)]
    assert backtester.use_llm is False
    assert backtester._run_metadata() == {
        "data_source": IFIND_ASHARE,
        "market": "CN",
        "universe": CSI300_SAMPLE_20_2026H2,
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
        "symbols": list(CSI300_SAMPLE_20_2026H2_SYMBOLS),
    }


@pytest.mark.parametrize(
    ("universe", "symbols"),
    [
        (A_SHARE_DEMO_6, A_SHARE_DEMO_6_SYMBOLS),
        (CSI300_SAMPLE_20_2026H2, CSI300_SAMPLE_20_2026H2_SYMBOLS),
    ],
)
def test_ifind_registered_universe_runs_explicit_llm_with_strict_market_context(
    monkeypatch,
    universe,
    symbols,
):
    provider = RecordingProvider(make_cn_bars(symbols))
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    llm_client = object()
    monkeypatch.setattr(engine_module, "make_llm_client", lambda: llm_client)
    recording_db = RecordingDB()
    monkeypatch.setattr(engine_module, "db", recording_db)
    decision_calls = []

    def fake_llm_decision(
        manager,
        _state,
        received_client,
        **kwargs,
    ):
        decision_calls.append((received_client, kwargs))
        manager.llm_calls += 1
        manager.llm_decisions += 1
        manager.input_tokens += 10
        manager.output_tokens += 2
        return {"actions": []}

    monkeypatch.setattr(
        engine_module.PortfolioManager,
        "make_trading_decision_with_llm",
        fake_llm_decision,
    )

    backtester = HourlyBacktester(
        START,
        END,
        session_id="ifind-llm-engine-test",
        use_llm=False,
        model="test-a-share-model",
        data_source=IFIND_ASHARE,
        universe=universe,
        decision_source=LLM_DECISION_SOURCE,
    )

    assert backtester.use_llm is True
    assert backtester.decision_source == LLM_DECISION_SOURCE
    assert backtester.strict_llm is True
    assert backtester.llm_client is llm_client

    backtester.load_data()
    backtester.calculate_indicators()
    run_id, curve = backtester.run_agent_backtest()

    assert run_id and curve
    assert decision_calls
    assert all(call[0] is llm_client for call in decision_calls)
    assert all(call[1]["strict_llm"] is True for call in decision_calls)
    assert all(
        call[1]["market_context"]
        == {
            "market": "CN",
            "timezone": "Asia/Shanghai",
            "timeframe": "60m",
            "symbols": list(symbols),
            "paper_backtest": True,
        }
        for call in decision_calls
    )
    saved = recording_db.runs[0]
    assert saved["llm_model"] == "test-a-share-model"
    assert saved["llm_calls"] == len(decision_calls)
    assert saved["metadata"]["decision_source"] == LLM_DECISION_SOURCE


def test_ifind_explicit_llm_requires_a_client(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(engine_module, "make_llm_client", lambda: None)

    with pytest.raises(llm_harness.LLMConfigurationError, match="client"):
        HourlyBacktester(
            START,
            END,
            data_source=IFIND_ASHARE,
            decision_source=LLM_DECISION_SOURCE,
        )


def test_ifind_default_decision_source_remains_rule_based(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )

    backtester = HourlyBacktester(
        START,
        END,
        data_source=IFIND_ASHARE,
        use_llm=True,
    )

    assert backtester.decision_source == RULE_BASED_DECISION_SOURCE
    assert backtester.use_llm is False
    assert backtester.strict_llm is False


def test_cli_exposes_explicit_decision_source(tmp_path):
    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 0, result.stderr
    assert "--decision-source {rule_based,llm}" in result.stdout


def test_cli_rejects_conflicting_legacy_llm_flag(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "dashboard/scripts/backtest_hourly_agent.py",
            "--data-source",
            IFIND_ASHARE,
            "--decision-source",
            LLM_DECISION_SOURCE,
            "--no-llm",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )

    assert result.returncode == 2
    assert "--decision-source llm conflicts with --no-llm" in result.stderr


def test_ifind_engine_rejects_incomplete_provider_result_before_trading(monkeypatch):
    bars = make_cn_bars(CSI300_SAMPLE_20_2026H2_SYMBOLS)
    bars.pop(CSI300_SAMPLE_20_2026H2_SYMBOLS[-1])

    class IncompleteProvider(RecordingProvider):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((symbols, start, end))
            return self.bars

    provider = IncompleteProvider(bars)
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    backtester = HourlyBacktester(
        START,
        END,
        use_llm=False,
        data_source=IFIND_ASHARE,
        universe=CSI300_SAMPLE_20_2026H2,
    )

    with pytest.raises(MarketDataUnavailableError, match="missing"):
        backtester.load_data()


def test_ifind_engine_rejects_frames_without_a_common_start(monkeypatch):
    bars = make_cn_bars(CSI300_SAMPLE_20_2026H2_SYMBOLS)
    for offset, symbol in enumerate(CSI300_SAMPLE_20_2026H2_SYMBOLS):
        bars[symbol] = bars[symbol].copy()
        bars[symbol].index = bars[symbol].index + pd.Timedelta(minutes=offset)

    provider = RecordingProvider(bars)
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    backtester = HourlyBacktester(
        START,
        END,
        use_llm=False,
        data_source=IFIND_ASHARE,
        universe=CSI300_SAMPLE_20_2026H2,
    )

    with pytest.raises(MarketDataUnavailableError, match="common timestamp"):
        backtester.load_data()


def test_ifind_buyhold_passes_fixed_symbols_and_timezone_to_baselines(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(engine_module, "create_market_data_provider", lambda _source, universe=None: provider)
    backtester = HourlyBacktester(START, END, use_llm=False, data_source=IFIND_ASHARE)
    backtester.all_data = make_cn_bars()
    captured = {}

    def fake_generate_baselines(**kwargs):
        captured.update(kwargs)
        return (
            [
                {
                    "timestamp": "2026-04-01T10:30:00+08:00",
                    "equity": 100_000,
                    "cash": 0,
                    "positions_value": 100_000,
                }
            ],
            [],
        )

    monkeypatch.setattr(engine_module, "generate_baselines", fake_generate_baselines)

    run_id, curve = backtester.run_buyhold_baseline()

    assert run_id
    assert curve
    assert captured["symbols_list"] == list(A_SHARE_DEMO_6_SYMBOLS)
    assert captured["market_timezone"] == "Asia/Shanghai"


def test_ifind_djia_baseline_is_a_noop(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(engine_module, "create_market_data_provider", lambda _source, universe=None: provider)
    backtester = HourlyBacktester(START, END, use_llm=False, data_source=IFIND_ASHARE)
    backtester.all_data = make_cn_bars()

    def fail_baseline_call(**_kwargs):
        raise AssertionError("iFinD mode must not generate a DJIA baseline")

    monkeypatch.setattr(engine_module, "generate_baselines", fail_baseline_call)

    assert backtester.run_djia_baseline() == (None, [])
