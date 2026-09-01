"""iFinD A-share profile behavior at the HourlyBacktester boundary."""

from __future__ import annotations

from datetime import datetime, time, timedelta
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.currency import CurrencyContext
from dashboard.backend.domain.backtesting.market_rules import (
    DailyMarketRule,
    MarketRuleCalendar,
)
from dashboard.backend.infrastructure.llm import backtest_harness as llm_harness
from dashboard.backend.infrastructure.market_data.profiles import (
    ALPACA,
    A_SHARE_DEMO_6,
    A_SHARE_DEMO_6_SYMBOLS,
    ASHARE_TRANSACTION_COST_PROFILE,
    CSI300_SAMPLE_20_2026H2,
    CSI300_SAMPLE_20_2026H2_SYMBOLS,
    IFIND_ASHARE,
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
    get_market_profile,
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
        self.fx_calls = []
        self.rule_calls = []

    def fetch_bars(self, symbols, start, end):
        self.calls.append((symbols, start, end))
        return {symbol: self.bars[symbol] for symbol in symbols}

    def fetch_usd_cny(self, symbols, start, end):
        self.fx_calls.append((symbols, start, end))
        return {
            datetime(2026, 3, 31).date(): 7.0,
            datetime(2026, 4, 15).date(): 7.1,
        }

    def fetch_market_rules(self, symbols, start, end, *, bars_by_symbol):
        self.rule_calls.append((symbols, start, end, bars_by_symbol))
        rules = []
        required_dates = sorted({
            timestamp.date()
            for frame in bars_by_symbol.values()
            for timestamp in frame.index
        })
        for symbol in symbols:
            frame = bars_by_symbol[symbol]
            for trading_date in required_dates:
                daily = frame[frame.index.date == trading_date]
                if daily.empty:
                    rules.append(DailyMarketRule(
                        symbol=symbol,
                        trading_date=trading_date,
                        suspended=True,
                    ))
                    continue
                final_bar = daily.index[-1].to_pydatetime()
                rules.append(DailyMarketRule(
                    symbol=symbol,
                    trading_date=trading_date,
                    suspended=False,
                    official_close_price=daily.iloc[-1]["close"],
                    final_bar_timestamp=final_bar,
                ))
        return MarketRuleCalendar(rules)


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


def test_rejected_order_serializer_normalizes_timestamp_and_numpy_numbers():
    serialized = HourlyBacktester._serialize_rejected_orders([{
        "timestamp": pd.Timestamp("2026-04-01T10:00:00", tz=CN),
        "requested_shares": pd.Series([100], dtype="int64").iloc[0],
        "executed_shares": pd.Series([40], dtype="int64").iloc[0],
        "unfilled_shares": pd.Series([60], dtype="int64").iloc[0],
        "reason": "t1_frozen",
    }])

    assert serialized == [{
        "timestamp": "2026-04-01T10:00:00+08:00",
        "requested_shares": 100,
        "executed_shares": 40,
        "unfilled_shares": 60,
        "reason": "t1_frozen",
    }]


def test_order_event_serializer_converts_currency_and_preserves_fractional_request():
    backtester = object.__new__(HourlyBacktester)
    backtester.currency_context = CurrencyContext(
        native_currency="CNY",
        reporting_currency="USD",
        timezone="Asia/Shanghai",
        rates={datetime(2026, 4, 1).date(): 7.0},
    )
    timestamp = pd.Timestamp("2026-04-01T10:00:00", tz=CN)

    serialized = backtester._serialize_order_events([{
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": pd.Series([100.5], dtype="float64").iloc[0],
        "executed_shares": pd.Series([0], dtype="int64").iloc[0],
        "unfilled_shares": pd.Series([100.5], dtype="float64").iloc[0],
        "price": pd.Series([700], dtype="float64").iloc[0],
        "executed_value": pd.Series([0], dtype="float64").iloc[0],
        "status": "rejected",
        "reason": "invalid_lot_size",
        "strategy_reason": "Model request",
    }])

    assert serialized == [{
        "timestamp": "2026-04-01T10:00:00+08:00",
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": 100.5,
        "executed_shares": 0,
        "unfilled_shares": 100.5,
        "price": 100.0,
        "executed_value": 0.0,
        "status": "rejected",
        "reason": "invalid_lot_size",
            "strategy_reason": "Model request",
            "native_price": 700.0,
            "native_executed_value": 0.0,
            "native_value": 0.0,
        "fx_rate": 7.0,
    }]
    json.dumps(serialized)


def test_serializers_preserve_market_rule_audit_in_native_cny():
    backtester = object.__new__(HourlyBacktester)
    backtester.currency_context = CurrencyContext(
        native_currency="CNY",
        reporting_currency="USD",
        timezone="Asia/Shanghai",
        rates={datetime(2026, 4, 1).date(): 7.0},
    )
    timestamp = pd.Timestamp("2026-04-01T15:00:00", tz=CN)
    audit = {
        "market_rule_date": "2026-04-01",
        "market_rule_suspended": False,
        "market_rule_closing_limit_state": "upper",
        "market_rule_official_close": 1_400.0,
        "market_rule_closing_gate_effective": True,
    }

    trade = backtester._serialize_trades([{
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "SELL",
        "shares": 100,
        "price": 1_400.0,
        "proceeds": 140_000.0,
        **audit,
    }])[0]
    event = backtester._serialize_order_events([{
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": 100,
        "executed_shares": 0,
        "unfilled_shares": 100,
        "price": 1_400.0,
        "executed_value": 0,
        "status": "rejected",
        "reason": "limit_up_buy_blocked",
        **audit,
    }])[0]

    for record in (trade, event):
        assert record["market_rule_date"] == "2026-04-01"
        assert record["market_rule_closing_limit_state"] == "upper"
        assert record["market_rule_official_close"] == 1_400.0
        assert record["market_rule_closing_gate_effective"] is True
    assert trade["price"] == 200.0
    assert event["price"] == 200.0
    json.dumps([trade, event])


def test_live_progress_keeps_bounded_order_event_tail(tmp_path):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_order_events"
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")
    def _event(seq, status):
        return {
            "timestamp": datetime(2026, 4, 1, 10),
            "symbol": "AAPL",
            "side": "BUY",
            "requested_shares": 1,
            "executed_shares": 1 if status == "filled" else 0,
            "unfilled_shares": 0 if status == "filled" else 1,
            "price": 100,
            "executed_value": 100 if status == "filled" else 0,
            "status": status,
            "reason": "" if status == "filled" else "insufficient_cash",
            "strategy_reason": str(seq),
        }

    manager = SimpleNamespace(
        get_equity_curve=lambda: [],
        trades=[],
        rejected_orders=[],
        # Fills are interleaved to prove they are dropped by content, not by
        # position: the live payload's own `trades` list already carries them.
        order_events=(
            [_event(seq, "rejected") for seq in range(60)]
            + [_event(900 + seq, "filled") for seq in range(25)]
        ),
    )

    backtester._publish_live_progress(3, 10, manager)

    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["order_events_count"] == 60
    assert len(payload["order_events"]) == 50
    assert payload["order_events"][0]["strategy_reason"] == "10"
    assert all(
        event["status"] != "filled" for event in payload["order_events"]
    )


@pytest.mark.parametrize(
    ("decision_source", "wants_llm"),
    [
        (RULE_BASED_DECISION_SOURCE, False),
        (LLM_DECISION_SOURCE, True),
    ],
)
def test_ifind_rule_and_llm_paths_share_t1_execution(
    monkeypatch,
    decision_source,
    wants_llm,
):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(engine_module, "make_llm_client", object)
    recording_db = RecordingDB()
    monkeypatch.setattr(engine_module, "db", recording_db)

    created_managers = []
    base_manager = engine_module.PortfolioManager

    class ScriptedPortfolioManager(base_manager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.script_step = 0
            created_managers.append(self)

        def _scripted_decision(self):
            self.script_step += 1
            if self.script_step == 1:
                return {"actions": [{
                    "symbol": A_SHARE_DEMO_6_SYMBOLS[0],
                    "action": "buy",
                    "shares": 100,
                }]}
            if self.script_step == 2:
                return {"actions": [{
                    "symbol": A_SHARE_DEMO_6_SYMBOLS[0],
                    "action": "sell",
                    "shares": 100,
                }]}
            return {"actions": []}

        def make_trading_decision(self, _portfolio_state):
            return self._scripted_decision()

        def make_trading_decision_with_llm(self, *_args, **_kwargs):
            return self._scripted_decision()

    monkeypatch.setattr(engine_module, "PortfolioManager", ScriptedPortfolioManager)

    backtester = HourlyBacktester(
        START,
        END,
        session_id=f"ifind-t1-{decision_source}",
        use_llm=wants_llm,
        data_source=IFIND_ASHARE,
        initial_capital=2000,
        decision_source=decision_source,
    )
    backtester.load_data()
    backtester.calculate_indicators()
    backtester.run_agent_backtest()

    manager = created_managers[0]
    assert manager.t_plus_one_enabled is True
    assert manager.lot_size == 100
    assert [trade["side"] for trade in manager.trades] == ["BUY"]
    assert manager.rejected_orders[0]["reason"] == "t1_frozen"
    assert [event["status"] for event in manager.order_events] == [
        "filled",
        "rejected",
    ]
    assert recording_db.runs[0]["num_trades"] == 1
    # The in-memory ledger above holds both outcomes; only the non-filled one
    # is persisted, because the filled one is already a row in `trades` and
    # copying it here is what would push real rejections out of the sample.
    assert recording_db.runs[0]["metadata"]["order_events_count"] == 1
    assert [
        event["status"]
        for event in recording_db.runs[0]["metadata"]["order_events"]
    ] == ["rejected"]
    assert recording_db.runs[0]["metadata"]["rejected_orders"][0][
        "reason"
    ] == "t1_frozen"


def test_ifind_engine_uses_profile_symbols_in_explicit_rule_mode(monkeypatch):
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
        use_llm=False,
        data_source=IFIND_ASHARE,
        decision_source=RULE_BASED_DECISION_SOURCE,
    )

    assert backtester.use_llm is False
    assert backtester.llm_client is None
    assert backtester.symbols == A_SHARE_DEMO_6_SYMBOLS

    backtester.load_data()
    assert provider.calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    assert provider.fx_calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    assert backtester.native_initial_capital == pytest.approx(7_000)
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
        "t_plus_one_enabled": True,
        "symbols": list(A_SHARE_DEMO_6_SYMBOLS),
        "native_currency": "CNY",
        "reporting_currency": "USD",
        "lot_size": 100,
        "fx_pair": "USD/CNY",
        "fx_source": "ifind_history_currency_conversion",
        "fx_policy": "daily_implied_median_forward_fill",
        "fx_symbols": list(A_SHARE_DEMO_6_SYMBOLS),
        "fx_max_relative_deviation": 0.0025,
        "fx_start_rate": 7.0,
        "fx_end_rate": 7.1,
        "fx_market_start_date": "2026-04-01",
        "fx_market_end_date": "2026-04-21",
        "fx_observation_start_date": "2026-03-31",
        "fx_observation_end_date": "2026-04-15",
        "native_initial_capital": 7_000.0,
        "transaction_cost_profile": ASHARE_TRANSACTION_COST_PROFILE.to_metadata(),
        # An agent run executes through the cost path, so it did pay. The flag
        # separates the market's rule from this run's ledger — see the index
        # baseline, which carries the same profile with this set False.
            "transaction_costs_applied": True,
            "market_rule_profile": {
                "enabled": True,
                "source": "ifind_http",
                "version": "ifind-ashare-closing-rules-v1",
                "observations": 90,
                "scope": "full_day_suspension_and_closing_limits",
            },
        # No rejected_orders* keys at all: a clean run writes nothing rather
        # than an empty array on every A-share row.
    }
    first_equity = recording_db.equity_points[0][1][0]
    assert first_equity["equity"] == pytest.approx(1_000)
    assert first_equity["native_equity"] == pytest.approx(7_000)
    assert first_equity["fx_rate"] == pytest.approx(7.0)
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

    # The use_llm downgrade asserted below only fires with no LLM key in the
    # environment; a developer's dashboard/.env leaks ANTHROPIC_API_KEY into
    # os.environ whenever an earlier-collected test imports dashboard.backend.app.
    for key in ("COMMONSTACK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)

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
    assert provider.fx_calls == [(CSI300_SAMPLE_20_2026H2_SYMBOLS, START, END)]
    assert backtester.use_llm is False
    metadata = backtester._run_metadata()
    assert metadata == {
        "data_source": IFIND_ASHARE,
        "market": "CN",
        "universe": CSI300_SAMPLE_20_2026H2,
        "timeframe": "60m",
        "timezone": "Asia/Shanghai",
        "decision_source": "rule_based",
        "benchmark": "equal_weight_buyhold",
        "t_plus_one_enabled": True,
        "symbols": list(CSI300_SAMPLE_20_2026H2_SYMBOLS),
        "native_currency": "CNY",
        "reporting_currency": "USD",
        "lot_size": 100,
        "fx_pair": "USD/CNY",
        "fx_source": "ifind_history_currency_conversion",
        "fx_policy": "daily_implied_median_forward_fill",
        "fx_symbols": list(CSI300_SAMPLE_20_2026H2_SYMBOLS),
        "fx_max_relative_deviation": 0.0025,
        "fx_start_rate": 7.0,
        "fx_end_rate": 7.1,
        "fx_market_start_date": "2026-04-01",
        "fx_market_end_date": "2026-04-21",
        "fx_observation_start_date": "2026-03-31",
        "fx_observation_end_date": "2026-04-15",
        "native_initial_capital": 7_000.0,
        "transaction_cost_profile": ASHARE_TRANSACTION_COST_PROFILE.to_metadata(),
        # Bare _run_metadata() is the provenance-only call the index baseline
        # makes; it advertises the market's cost rules without charging them.
        "transaction_costs_applied": False,
        "market_rule_profile": {
            "enabled": True,
            "source": "ifind_http",
            "version": "ifind-ashare-closing-rules-v1",
            "observations": 300,
            "scope": "full_day_suspension_and_closing_limits",
        },
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
            "native_currency": "CNY",
            "reporting_currency": "USD",
            "lot_size": 100,
            # Same reasoning as settlement_note below: a bare `100` does not
            # tell the model that an off-lot size is rejected outright rather
            # than rounded down.
            "lot_size_note": (
                "Order quantities must be positive whole multiples of 100 "
                "shares; any other size is rejected in full rather than "
                "rounded."
            ),
            # A_share profiles settle T+1, and the model is told so in words —
            # a bare sellable_shares integer in each holding is not
            # self-describing.
            "settlement": "T+1",
            "settlement_note": (
                "Shares bought today cannot be sold until the next trading "
                "day. Each holding reports sellable_shares; a sell above that "
                "amount is truncated to it."
            ),
            "fx_pair": "USD/CNY",
            "fx_source": "iFinD Historical Conversion Rate",
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


def test_ifind_default_decision_source_matches_alpaca_llm_default(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    llm_client = object()
    monkeypatch.setattr(engine_module, "make_llm_client", lambda: llm_client)

    backtester = HourlyBacktester(
        START,
        END,
        data_source=IFIND_ASHARE,
        use_llm=True,
    )

    assert backtester.decision_source == LLM_DECISION_SOURCE
    assert backtester.use_llm is True
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
    backtester.load_data()
    monkeypatch.setattr(engine_module, "db", RecordingDB())
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
    assert captured["currency_context"] is backtester.currency_context


def test_ifind_djia_baseline_is_a_noop(monkeypatch):
    provider = RecordingProvider(make_cn_bars())
    monkeypatch.setattr(engine_module, "create_market_data_provider", lambda _source, universe=None: provider)
    backtester = HourlyBacktester(START, END, use_llm=False, data_source=IFIND_ASHARE)
    backtester.all_data = make_cn_bars()

    def fail_baseline_call(**_kwargs):
        raise AssertionError("iFinD mode must not generate a DJIA baseline")

    monkeypatch.setattr(engine_module, "generate_baselines", fail_baseline_call)

    assert backtester.run_djia_baseline() == (None, [])


# ---------------------------------------------------------------------------
# FX bootstrap error reporting
#
# A gap in the rate series and a bad token are different problems; sending the
# operator to "check your credentials" for a data gap wastes the investigation.
# ---------------------------------------------------------------------------

def _ifind_backtester(monkeypatch, provider):
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda _source, universe=None: provider,
    )
    monkeypatch.setattr(engine_module, "HAS_ANTHROPIC", True)
    monkeypatch.setattr(engine_module, "make_llm_client", lambda: None)
    monkeypatch.setattr(engine_module, "db", RecordingDB())
    return HourlyBacktester(
        START,
        END,
        session_id="ifind-fx-error-test",
        use_llm=False,
        data_source=IFIND_ASHARE,
        decision_source=RULE_BASED_DECISION_SOURCE,
    )


def test_provider_without_fx_capability_is_named_not_blamed_on_credentials(monkeypatch):
    class NoFxProvider:
        """A bars-only provider — no fetch_usd_cny at all."""

        def __init__(self, bars):
            self.bars = bars
            self.rule_calls = []

        def fetch_bars(self, symbols, start, end):
            return {symbol: self.bars[symbol] for symbol in symbols}

        def fetch_market_rules(self, symbols, start, end, *, bars_by_symbol):
            return RecordingProvider.fetch_market_rules(
                self,
                symbols,
                start,
                end,
                bars_by_symbol=bars_by_symbol,
            )

    backtester = _ifind_backtester(monkeypatch, NoFxProvider(make_cn_bars()))

    with pytest.raises(MarketDataUnavailableError) as excinfo:
        backtester.load_data()

    message = str(excinfo.value)
    assert "conversion rate" in message
    assert "token" not in message


def test_fx_validation_failure_surfaces_its_reason_instead_of_credentials(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
    )

    class GappyProvider(RecordingProvider):
        def fetch_usd_cny(self, symbols, start, end):
            raise IFindFxValidationError(
                "iFinD historical FX returned no usable daily rates "
                "(resolved=0/20 dates; blank_closes=120)"
            )

    backtester = _ifind_backtester(monkeypatch, GappyProvider(make_cn_bars()))

    with pytest.raises(MarketDataUnavailableError) as excinfo:
        backtester.load_data()

    message = str(excinfo.value)
    assert "no usable daily rates" in message
    assert "blank_closes=120" in message
    assert "permission" not in message


def test_transport_failure_still_points_at_credentials(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindHttpError,
    )

    class ForbiddenProvider(RecordingProvider):
        def fetch_usd_cny(self, symbols, start, end):
            raise IFindHttpError("iFinD request failed status=403", 403)

    backtester = _ifind_backtester(monkeypatch, ForbiddenProvider(make_cn_bars()))

    with pytest.raises(MarketDataUnavailableError, match="permission"):
        backtester.load_data()


# ---------------------------------------------------------------------------
# Rejected-order persistence is bounded, and says so
# ---------------------------------------------------------------------------

def _metadata_stub(
    rejected,
    monkeypatch,
    data_source=IFIND_ASHARE,
    deferrals=(),
    order_events=(),
):
    """Run _agent_run_metadata over just the T+1 audit branches."""
    backtester = object.__new__(HourlyBacktester)
    backtester.data_source = data_source
    backtester.rejected_orders = rejected
    backtester.t1_deferrals = list(deferrals)
    backtester.order_events = list(order_events)
    backtester.use_llm = False
    backtester.prompt_adaptations = None
    backtester.initial_pipeline = None
    backtester.pipeline = None
    monkeypatch.setattr(
        HourlyBacktester, "_run_metadata", lambda self, *args, **kwargs: {}
    )
    return backtester._agent_run_metadata()


def test_agent_metadata_omits_rejected_orders_entirely_when_there_are_none(monkeypatch):
    """A clean run must not write an empty array onto every A-share row."""
    meta = _metadata_stub([], monkeypatch)
    assert "rejected_orders" not in meta
    assert "rejected_orders_count" not in meta
    assert "rejected_orders_truncated" not in meta


def test_agent_metadata_keeps_a_small_rejection_list_whole(monkeypatch):
    records = [{"reason": "t1_frozen", "seq": i} for i in range(3)]
    meta = _metadata_stub(records, monkeypatch)
    assert meta["rejected_orders"] == records
    assert meta["rejected_orders_count"] == 3
    assert "rejected_orders_truncated" not in meta


def test_agent_metadata_counts_market_rule_rejections(monkeypatch):
    records = [
        {"reason": "suspended"},
        {"reason": "suspended"},
        {"reason": "limit_up_buy_blocked"},
        {"reason": "limit_down_sell_blocked"},
        {"reason": "t1_frozen"},
    ]

    meta = _metadata_stub(records, monkeypatch)

    assert meta["market_rule_rejections"] == {
        "suspended": 2,
        "limit_up_buy_blocked": 1,
        "limit_down_sell_blocked": 1,
    }


def test_agent_metadata_caps_the_sample_and_reports_the_true_total(monkeypatch):
    """The cap must never be silent.

    A 20-symbol A-share run can emit thousands of records at ~198 bytes each,
    which would otherwise land whole in one agent_runs.metadata JSON cell.
    """
    limit = engine_module.REJECTED_ORDER_SAMPLE_LIMIT
    records = [{"reason": "t1_frozen", "seq": i} for i in range(limit + 500)]
    meta = _metadata_stub(records, monkeypatch)

    assert len(meta["rejected_orders"]) == limit
    assert meta["rejected_orders_count"] == limit + 500
    assert meta["rejected_orders_truncated"] == 500
    # Head sample: the first rejections are the diagnostic ones, and the count
    # above is what tells a reader the list is partial.
    assert meta["rejected_orders"][0]["seq"] == 0


def test_agent_metadata_skips_rejected_orders_for_non_ashare_sources(monkeypatch):
    meta = _metadata_stub(
        [{"reason": "t1_frozen"}], monkeypatch, data_source="alpaca"
    )
    assert "rejected_orders" not in meta
    assert "rejected_orders_count" not in meta


def test_djia_market_context_carries_no_lot_size_key(monkeypatch):
    """A single-share market's prompt must be byte-identical to before.

    `_llm_market_context` is serialized straight into the LLM prompt, so any
    unconditional key changes every DJIA prompt and makes new runs
    non-comparable with the historical ones already on the leaderboard. The
    A-share side asserts the key IS present; this is the other half.
    """
    backtester = object.__new__(HourlyBacktester)
    backtester.profile = get_market_profile(ALPACA)
    backtester.symbols = ("AAPL", "MSFT")
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")

    context = backtester._llm_market_context()

    assert backtester.profile.lot_size == 1
    assert "lot_size" not in context
    assert "lot_size_note" not in context
    # The keys that must still be there, so this cannot pass by returning {}.
    assert context["market"] == "US"
    assert context["paper_backtest"] is True


def test_fills_are_not_persisted_because_trades_already_holds_them(monkeypatch):
    """Copying fills into the capped sample is what made the cap lossy."""
    events = [
        {"status": "filled", "seq": 0},
        {"status": "rejected", "seq": 1},
        {"status": "partial", "seq": 2},
        {"status": "filled", "seq": 3},
    ]

    kept = engine_module._unfilled_order_events(events)

    assert [event["seq"] for event in kept] == [1, 2]
    assert engine_module._unfilled_order_events([]) == []
    assert engine_module._unfilled_order_events(None) == []


def test_agent_metadata_caps_order_events_for_every_market(monkeypatch):
    limit = engine_module.REJECTED_ORDER_SAMPLE_LIMIT
    records = [{"status": "rejected", "seq": i} for i in range(limit + 25)]

    meta = _metadata_stub(
        [],
        monkeypatch,
        data_source="alpaca",
        order_events=records,
    )

    assert len(meta["order_events"]) == limit
    assert meta["order_events_count"] == limit + 25
    assert meta["order_events_truncated"] == 25
    assert meta["order_events"][0]["seq"] == 0


# ---------------------------------------------------------------------------
# T+1 deferrals: the "did the rule actually bind?" metric
# ---------------------------------------------------------------------------

def _deferral(seq, deferred=10):
    return {
        "date": f"2026-04-{(seq % 28) + 1:02d}",
        "symbol": f"60000{seq % 5}.SH",
        "requested_shares": deferred + 5,
        "sellable_shares": 5,
        "deferred_shares": deferred,
    }


def test_agent_metadata_omits_deferrals_when_t1_never_bound(monkeypatch):
    meta = _metadata_stub([], monkeypatch, deferrals=[])
    assert "t1_deferrals" not in meta
    assert "t1_deferred_events" not in meta
    assert "t1_deferred_shares" not in meta


def test_agent_metadata_totals_the_deferred_shares(monkeypatch):
    records = [_deferral(i, deferred=10) for i in range(3)]
    meta = _metadata_stub([], monkeypatch, deferrals=records)

    assert meta["t1_deferred_events"] == 3
    assert meta["t1_deferred_shares"] == 30
    assert meta["t1_deferrals"] == records
    assert "t1_deferrals_truncated" not in meta


def test_agent_metadata_caps_the_deferral_sample_too(monkeypatch):
    limit = engine_module.REJECTED_ORDER_SAMPLE_LIMIT
    records = [_deferral(i) for i in range(limit + 7)]
    meta = _metadata_stub([], monkeypatch, deferrals=records)

    assert len(meta["t1_deferrals"]) == limit
    assert meta["t1_deferred_events"] == limit + 7
    assert meta["t1_deferrals_truncated"] == 7
    # The total counts every event, not just the persisted sample.
    assert meta["t1_deferred_shares"] == 10 * (limit + 7)


def test_deferral_serializer_sorts_and_isoformats_dates():
    from datetime import date as _date

    serialized = HourlyBacktester._serialize_t1_deferrals({
        ("601318.SH", _date(2026, 4, 2)): {
            "date": _date(2026, 4, 2), "symbol": "601318.SH",
            "requested_shares": 10, "sellable_shares": 0, "deferred_shares": 10,
        },
        ("600519.SH", _date(2026, 4, 1)): {
            "date": _date(2026, 4, 1), "symbol": "600519.SH",
            "requested_shares": pd.Series([7], dtype="int64").iloc[0],
            "sellable_shares": 2, "deferred_shares": 5,
        },
    })

    assert [item["date"] for item in serialized] == ["2026-04-01", "2026-04-02"]
    # numpy scalar unwrapped to a plain int, or json.dumps would reject it.
    assert serialized[0]["requested_shares"] == 7
    assert type(serialized[0]["requested_shares"]) is int
    json.dumps(serialized)
