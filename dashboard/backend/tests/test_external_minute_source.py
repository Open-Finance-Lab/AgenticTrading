from datetime import datetime

import pandas as pd
import pytest

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.domain.backtesting.bar_aggregation import ExecutionFill
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
)
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
)


def _minute_bars(symbols, start, end):
    timestamps = pd.date_range(
        "2026-04-15 13:30:00+00:00",
        "2026-04-15 19:55:00+00:00",
        freq="5min",
    )
    prices = [100 + index * 0.01 for index in range(len(timestamps))]
    frame = pd.DataFrame(
        {
            "open": [price + 0.25 for price in prices],
            "high": [price + 0.5 for price in prices],
            "low": [price - 0.5 for price in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        },
        index=timestamps,
    )
    frames = {symbol: frame.copy() for symbol in symbols}
    for symbol_frame in frames.values():
        symbol_frame.attrs[FRAME_ATTR_FEED] = "iex"
        symbol_frame.attrs[FRAME_ATTR_SIP_FALLBACK] = True
        symbol_frame.attrs[FRAME_ATTR_END_CLAMPED] = False
        symbol_frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 5
    return frames


class _MinuteLoader:
    source_timeframe = "60m"

    def configure_source_timeframe(self, value):
        self.source_timeframe = value

    def fetch_bars(self, symbols, start, end):
        return _minute_bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _isolate_store(monkeypatch):
    mds._reset_for_tests()
    monkeypatch.setattr(ebs, "AlpacaDataLoader", _MinuteLoader)
    monkeypatch.setattr(ebs, "DJIA_30", ["AAPL"])
    yield
    mds._reset_for_tests()


def test_external_session_serves_hourly_bars_but_fills_and_values_on_5m():
    session = ebs.ExternalBacktestSession(
        backtest_id="bt-minute",
        session_id="sess-minute",
        agent_name="agent-minute",
        model_name="test-model",
        start_date="2026-04-15",
        end_date="2026-04-15",
        symbols=["AAPL"],
    )
    session.load_market_data()

    assert session.source_timeframe == "5m"
    assert session.intraday_mode is True
    # Seven: the 16:00 bucket fills at the close of the 15:55 bar rather than
    # needing an after-hours bar to open at 16:00.
    assert session.total_steps == 7
    assert session.data_quality["total_decision_bars"] == 7
    assert session.data_quality["usable_decision_bars"] == 7
    assert session.data_quality["dropped_decision_bars"] == 0
    assert session.frequency_contract["verification_status"] == "verified"
    assert session.market_data_provenance == {
        "market_data_feed": "iex",
        "sip_fallback_to_iex": True,
        "end_clamped": False,
    }
    assert session.get_current_step()["timestamp"] == "2026-04-15T14:30:00+00:00"
    assert session.protocol_bars(session.timestamps[0])["AAPL"]["close"] == pytest.approx(100.11)

    result = session.submit_decisions(
        {
            "actions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "confidence": 1.0,
                    "reasoning": "test buy",
                    "position_size": 1,
                }
            ]
        }
    )

    assert result["accepted"] is True
    assert session.manager.trades[0]["timestamp"] == pd.Timestamp(
        datetime(2026, 4, 15, 14, 30), tz="UTC"
    )
    assert session.manager.trades[0]["price"] == pytest.approx(100.37)
    decision_audit = session.get_decisions()[0]
    assert decision_audit["timestamp"] == "2026-04-15T14:30:00+00:00"
    assert decision_audit["execution_timestamp"] == "2026-04-15T14:30:00+00:00"
    assert decision_audit["actions_executed"] == 1
    # 09:30 through 10:30 ET inclusive: 13 five-minute valuation points.
    assert len(session.manager.equity_history) == 13


def test_the_closing_decision_fills_at_the_last_regular_hours_close(monkeypatch):
    """Alpaca stamps bars at their open and serves extended hours, so the bar
    stamped 16:00 ET is 16:00-16:05 after hours. It must neither fill the
    closing decision nor mark the curve; whether it exists is the tape's call.
    """
    after_hours_open = 999.0

    class _ExtendedHoursLoader(_MinuteLoader):
        def fetch_bars(self, symbols, start, end):
            bars = _minute_bars(symbols, start, end)
            extra = pd.Timestamp("2026-04-15 20:00:00+00:00")
            for frame in bars.values():
                frame.loc[extra] = {
                    "open": after_hours_open,
                    "high": after_hours_open,
                    "low": after_hours_open,
                    "close": after_hours_open,
                    "volume": 10,
                }
            return bars

    monkeypatch.setattr(ebs, "AlpacaDataLoader", _ExtendedHoursLoader)
    session = ebs.ExternalBacktestSession(
        backtest_id="bt-close",
        session_id="sess-close",
        agent_name="agent-close",
        model_name="test-model",
        start_date="2026-04-15",
        end_date="2026-04-15",
        symbols=["AAPL"],
    )
    session.load_market_data()

    after_hours = pd.Timestamp("2026-04-15 20:00:00+00:00")
    last_regular = pd.Timestamp("2026-04-15 19:55:00+00:00")
    assert after_hours not in session.source_timestamps
    assert session.total_steps == 7
    # Priced off the 15:55 bar's close, and stamped at that close (16:00), so
    # the fill never predates the decision it answers.
    assert session.execution_fills[-1] == ExecutionFill(
        last_regular, "close", after_hours
    )
    assert [fill.price_field for fill in session.execution_fills] == (
        ["open"] * 6 + ["close"]
    )

    for _ in range(session.total_steps - 1):
        session.submit_decisions({"actions": []})
    session.submit_decisions(
        {
            "actions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "confidence": 1.0,
                    "reasoning": "closing buy",
                    "position_size": 1,
                }
            ]
        }
    )

    trade = session.manager.trades[-1]
    assert trade["timestamp"] == after_hours
    assert trade["timestamp"] >= session.timestamps[-1]
    assert session.get_decisions()[-1]["execution_timestamp"] == (
        "2026-04-15T20:00:00+00:00"
    )
    last_close = session.source_data["AAPL"].loc[last_regular, "close"]
    assert trade["price"] == pytest.approx(last_close)
    assert trade["price"] != pytest.approx(after_hours_open)
    assert all(
        pd.Timestamp(point["timestamp"]) != after_hours
        for point in session.manager.equity_history
    )


def test_an_unaggregated_run_fills_at_the_decision_bar_close(monkeypatch):
    """A run whose source bars already are its decision bars has no execution
    plan. Each bar is stamped at its close, so its open is an hour before the
    decision and filling there is look-ahead: the contract says
    ``decision_bar_close``, and both the recorded fill and the price agree."""

    class _LegacyHourlyLoader:
        # No ``source_timeframe``: the dataset takes it as 60m, unaggregated.
        def fetch_bars(self, symbols, start, end):
            timestamps = pd.date_range(
                "2026-04-15 14:00:00+00:00",
                "2026-04-15 20:00:00+00:00",
                freq="60min",
            )
            closes = [200.0 + index for index in range(len(timestamps))]
            frame = pd.DataFrame(
                {
                    "open": [close - 50.0 for close in closes],
                    "high": [close + 1.0 for close in closes],
                    "low": [close - 51.0 for close in closes],
                    "close": closes,
                    "volume": [1000] * len(closes),
                },
                index=timestamps,
            )
            return {symbol: frame.copy() for symbol in symbols}

    monkeypatch.setattr(ebs, "AlpacaDataLoader", _LegacyHourlyLoader)
    session = ebs.ExternalBacktestSession(
        backtest_id="bt-hourly",
        session_id="sess-hourly",
        agent_name="agent-hourly",
        model_name="test-model",
        start_date="2026-04-15",
        end_date="2026-04-15",
        symbols=["AAPL"],
    )
    session.load_market_data()

    assert session.intraday_mode is False
    assert session.execution_fills == [
        ExecutionFill(timestamp, "close", timestamp) for timestamp in session.timestamps
    ]
    assert session.frequency_contract["fill_policy"] == "decision_bar_close"

    session.submit_decisions(
        {
            "actions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "confidence": 1.0,
                    "reasoning": "first bar buy",
                    "position_size": 1,
                }
            ]
        }
    )

    decision_bar = session.timestamps[0]
    trade = session.manager.trades[0]
    assert trade["timestamp"] == decision_bar
    assert trade["price"] == pytest.approx(
        session.all_data["AAPL"].loc[decision_bar, "close"]
    )
    assert trade["price"] != pytest.approx(
        session.all_data["AAPL"].loc[decision_bar, "open"]
    )


def test_external_session_does_not_report_fill_without_next_symbol_bar(monkeypatch):
    class _MissingExecutionBarLoader(_MinuteLoader):
        def fetch_bars(self, symbols, start, end):
            bars = _minute_bars(symbols, start, end)
            execution_timestamp = pd.Timestamp("2026-04-15 14:30:00+00:00")
            bars["AAPL"] = bars["AAPL"].drop(execution_timestamp)
            return bars

    monkeypatch.setattr(ebs, "AlpacaDataLoader", _MissingExecutionBarLoader)
    monkeypatch.setattr(ebs, "DJIA_30", ["AAPL", "MSFT"])
    session = ebs.ExternalBacktestSession(
        backtest_id="bt-missing-fill",
        session_id="sess-missing-fill",
        agent_name="agent-missing-fill",
        model_name="test-model",
        start_date="2026-04-15",
        end_date="2026-04-15",
        symbols=["AAPL", "MSFT"],
    )
    session.load_market_data()
    assert session.timestamps[0] == pd.Timestamp("2026-04-15 14:30:00+00:00")
    assert session.execution_fills[0].bar == pd.Timestamp(
        "2026-04-15 14:30:00+00:00"
    )
    assert "AAPL" not in session._source_market_data_at(
        session.execution_fills[0].bar
    )

    result = session.submit_decisions(
        {
            "actions": [
                {
                    "symbol": "AAPL",
                    "action": "buy",
                    "confidence": 1.0,
                    "reasoning": "missing execution bar",
                    "position_size": 1,
                }
            ]
        }
    )

    assert result["accepted"] is True
    assert result["executed_count"] == 0
    assert result["executed"] == []
    assert session.manager.trades == []
    audit = session.get_decisions()[0]
    assert audit["actions_executed"] == 0
    assert audit["execution_timestamp"] == "2026-04-15T14:30:00+00:00"


def test_final_metrics_expose_minute_contract_without_symbol_quality_details():
    metrics = ebs.build_final_metrics(
        {
            "total_return": 0.1,
            "metadata": {
                "frequency_contract": {
                    "source_timeframe": "5m",
                    "decision_timeframe": "60m",
                    "decision_frequency": "1h",
                    "execution_timeframe": "5m",
                    "valuation_frequency": "5m",
                    "aggregation": "session_anchored_completed_bars",
                    "fill_policy": "next_source_bar_open",
                    "verification_status": "verified",
                },
                "market_data_quality": {
                    "policy": "drop_incomplete_decision_bars",
                    "total_decision_bars": 70,
                    "usable_decision_bars": 69,
                    "dropped_decision_bars": 1,
                    "missing_source_bars": 1,
                    "duplicate_source_bars": 0,
                    "off_grid_source_bars": 0,
                    "invalid_source_bars": 0,
                    "symbols": {"AAPL": {"dropped_decision_bars": 1}},
                },
                "market_data_feed": "iex",
                "sip_fallback_to_iex": True,
                "end_clamped": False,
            },
        }
    )

    assert metrics["frequency_contract"]["source_timeframe"] == "5m"
    assert metrics["frequency_contract"]["verification_status"] == "verified"
    assert metrics["market_data_quality"]["dropped_decision_bars"] == 1
    assert "symbols" not in metrics["market_data_quality"]
    assert metrics["market_data_feed"] == "iex"
    assert metrics["sip_fallback_to_iex"] is True
    assert metrics["end_clamped"] is False


def test_the_engine_plans_the_same_closing_fill():
    """The dashboard engine's twin of the protocol test above: the plan both
    paths build through ``plan_execution_fills`` from the same source bars."""
    from pathlib import Path

    from dashboard.backend.domain.backtesting import engine as engine_module
    from dashboard.backend.infrastructure.market_data import profiles

    bars = _minute_bars(["AAPL"], None, None)
    after_hours = pd.Timestamp("2026-04-15 20:00:00+00:00")
    bars["AAPL"].loc[after_hours] = [999.0] * 5
    backtester = engine_module.HourlyBacktester.__new__(
        engine_module.HourlyBacktester
    )
    backtester.data_source = profiles.ALPACA
    backtester.profile = profiles.get_market_profile(profiles.ALPACA)
    backtester.intraday_mode = True
    backtester.source_timeframe = "5m"
    backtester.source_data = bars
    decisions = list(
        pd.date_range("2026-04-15 14:30", "2026-04-15 19:30", freq="h", tz="UTC")
    ) + [after_hours]  # 10:30 ... 15:30 and the 16:00 ET closing bucket

    kept, valuation, plan = backtester._plan_executions(decisions)

    last_regular = pd.Timestamp("2026-04-15 19:55:00+00:00")
    assert kept == decisions
    assert after_hours not in valuation and valuation[-1] == last_regular
    assert plan[after_hours] == ExecutionFill(last_regular, "close", after_hours)
    assert all(plan[ts] == ExecutionFill(ts, "open", ts) for ts in decisions[:-1])
    # ...and the run loop prices by the fill's field and stamps the trade at
    # its fill instant, not a literal "open" on the bar's own stamp.
    source = Path(engine_module.__file__).read_text(encoding="utf-8")
    assert "symbol: row[fill.price_field]" in source
    assert "                fill.filled_at,\n" in source


def test_protocol_baselines_keep_each_session_close(monkeypatch):
    """``baseline_worker._run_job`` builds an ``HourlyBacktester`` it never
    loads -- ``intraday_mode`` stays False -- and hands it the dataset's
    aggregated bars. Those are stamped at their close, so the 16:00 bar is the
    day's closing mark. Deciding the convention from ``intraday_mode`` read
    them as raw 5m bars and dropped it; the frames now say which they are."""
    from dashboard.backend.domain.backtesting import engine as engine_module

    session = ebs.ExternalBacktestSession(
        backtest_id="bt-baseline",
        session_id="sess-baseline",
        agent_name="agent-baseline",
        model_name="test-model",
        start_date="2026-04-15",
        end_date="2026-04-15",
        symbols=["AAPL"],
    )
    session.load_market_data()

    class _DB:
        def insert_run(self, **kwargs):
            pass

        def insert_equity_points(self, run_id, points):
            pass

    monkeypatch.setattr(engine_module, "db", _DB())
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda *args, **kwargs: _MinuteLoader(),
    )
    # Exactly what `_run_job` does with a finalized run's dataset.
    backtester = engine_module.HourlyBacktester(
        session.start_date, session.end_date, session.session_id, use_llm=False
    )
    backtester.all_data = session.all_data
    assert backtester.intraday_mode is False
    _run_id, history = backtester.run_buyhold_baseline()

    assert history, "the baseline must produce a curve"
    closing = pd.Timestamp("2026-04-15 20:00:00+00:00")  # 16:00 ET
    assert pd.Timestamp(history[-1]["timestamp"]) == closing
