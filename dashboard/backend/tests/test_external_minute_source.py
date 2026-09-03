from datetime import datetime

import pandas as pd
import pytest

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.domain.backtesting import market_data_store as mds


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
    return {symbol: frame.copy() for symbol in symbols}


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
    # 09:30 through 10:30 ET inclusive: 13 five-minute valuation points.
    assert len(session.manager.equity_history) == 13
