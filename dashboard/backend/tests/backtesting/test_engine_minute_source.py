from datetime import datetime, timedelta

import pandas as pd
import pytz

from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting import engine as engine_mod


class _MinuteLoader:
    def __init__(self, bars):
        self.bars = bars
        self.source_timeframe = None

    def configure_source_timeframe(self, value):
        self.source_timeframe = value

    def fetch_bars(self, symbols, start_date, end_date):
        return {symbol: self.bars[symbol] for symbol in symbols}


class _DB:
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

    def insert_decisions(self, run_id, decisions):
        pass


def _make_minute_bars():
    eastern = pytz.timezone("US/Eastern")
    timestamps = []
    day = datetime(2026, 3, 2)
    while len(timestamps) < 10 * 78:
        if day.weekday() < 5:
            timestamps.extend(
                pd.date_range(
                    eastern.localize(datetime(day.year, day.month, day.day, 9, 30)),
                    eastern.localize(datetime(day.year, day.month, day.day, 15, 55)),
                    freq="5min",
                )
            )
        day += timedelta(days=1)
    timestamps = timestamps[: 10 * 78]
    prices = [100 + index * 0.01 for index in range(len(timestamps))]
    return {
        "AAPL": pd.DataFrame(
            {
                "open": [price + 0.25 for price in prices],
                "high": [price + 0.5 for price in prices],
                "low": [price - 0.5 for price in prices],
                "close": prices,
                "volume": [1000] * len(prices),
            },
            index=pd.DatetimeIndex(timestamps),
        )
    }


def test_minute_source_keeps_hourly_decisions_and_5m_execution(monkeypatch):
    loader = _MinuteLoader(_make_minute_bars())

    def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
        loader.configure_source_timeframe(source_timeframe)
        return loader

    fake_db = _DB()
    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    monkeypatch.setattr(engine_mod, "db", fake_db)
    decisions = []

    def buy_once(self, state):
        decisions.append(state["timestamp"])
        if not self.positions:
            return {"actions": [{"symbol": "AAPL", "action": "buy", "shares": 1}]}
        return {"actions": []}

    monkeypatch.setattr(
        "dashboard.backend.domain.backtesting.portfolio_manager.PortfolioManager.make_trading_decision",
        buy_once,
    )

    backtester = HourlyBacktester(
        "2026-03-02",
        "2026-03-13",
        use_llm=False,
        symbols=["AAPL"],
    )
    backtester.load_data()
    assert backtester.source_timeframe == "5m"
    assert backtester.intraday_mode is True
    assert len(backtester.source_data["AAPL"]) == 780
    assert len(backtester.all_data["AAPL"]) == 70

    backtester.calculate_indicators()
    run_id, equity_curve = backtester.run_agent_backtest()

    # Seven completed hourly buckets exist per day, but the 16:00 bucket has
    # no next source bar and is intentionally not an executable decision.
    assert len(decisions) == 60
    assert len(equity_curve) == 780
    assert run_id.startswith("agent_")

    trade = fake_db.trades[0][1][0]
    assert trade["timestamp"] == "2026-03-02T10:30:00-05:00"
    assert trade["price"] == 100.37

    frequency = fake_db.runs[0]["metadata"]["frequency_contract"]
    assert frequency["source_timeframe"] == "5m"
    assert frequency["decision_frequency"] == "1h"
    assert frequency["fill_policy"] == "next_source_bar_open"
    quality = fake_db.runs[0]["metadata"]["market_data_quality"]
    assert quality["policy"] == "drop_incomplete_decision_bars"
    assert quality["total_decision_bars"] == 70
    assert quality["usable_decision_bars"] == 70
    assert quality["dropped_decision_bars"] == 0
