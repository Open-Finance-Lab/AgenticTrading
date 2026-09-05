from datetime import datetime

import pandas as pd
import pytest

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
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
    assert session.total_steps == 6
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
    assert session.execution_timestamps[0] == pd.Timestamp(
        "2026-04-15 14:30:00+00:00"
    )
    assert "AAPL" not in session._source_market_data_at(
        session.execution_timestamps[0]
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
