"""A dashboard backtest's indicators arrive warm (issue #540).

``load_data`` fetches ``INDICATOR_WARMUP_CALENDAR_DAYS`` of history before
``start_date`` so sma20/sma50, Bollinger and MACD are real figures on the first
decision bar instead of the warm-up fallbacks (0 for MACD, 50 for RSI). Those
pre-window bars are indicator input only: no decision, trade, equity point or
baseline point may carry a timestamp before ``start_date``.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from math import sin
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import pytz

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.market_rules import (
    CorporateActionGap,
    CorporateActionGapError,
    DailyMarketRule,
    MarketRuleCalendar,
)
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_FEED,
    MarketDataUnavailableError,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6_SYMBOLS,
    IFIND_ASHARE,
    RULE_BASED_DECISION_SOURCE,
)
from dashboard.backend.infrastructure.market_data.provider import (
    INDICATOR_WARMUP_CALENDAR_DAYS,
    warmup_fetch_start,
)
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
)

START, END, PROVIDER_END = "2026-09-08", "2026-09-11", "2026-09-12"
WARMUP_START = "2026-08-09"
SYMBOLS = ["AAPL", "MSFT"]
_ET = pytz.timezone("US/Eastern")
_CN = ZoneInfo("Asia/Shanghai")


def _weekdays(start, end):
    """Weekdays in the half-open ``[start, end)``."""
    day = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while day < stop:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def _close(row):
    # Trending and oscillating, so MACD and RSI move off their neutral values.
    return 100.0 + 0.15 * row + 3.0 * sin(row / 3.0)


def _hourly_bars(symbols, start, end):
    idx = pd.DatetimeIndex(
        [
            _ET.localize(datetime(day.year, day.month, day.day, hour))
            for day in _weekdays(start, end)
            for hour in range(10, 17)
        ]
    )
    frames = {}
    for offset, symbol in enumerate(symbols):
        closes = [_close(row) + offset for row in range(len(idx))]
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": 1000.0,
            },
            index=idx,
        )
        frame.attrs[FRAME_ATTR_FEED] = "sip"
        frames[symbol] = frame
    return frames


class _RangeLoader:
    """Answers exactly the half-open window it is asked for."""

    def __init__(self):
        self.calls = []

    def fetch_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        return _hourly_bars(symbols, start, end)


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

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def loader(monkeypatch):
    loader = _RangeLoader()
    monkeypatch.setattr(engine_mod, "create_market_data_provider", lambda *a, **k: loader)
    monkeypatch.setattr(engine_mod, "db", _DB())
    return loader


def _market_date(timestamp, zone=_ET):
    return pd.Timestamp(timestamp).tz_convert(zone).date()


def test_the_pad_is_thirty_calendar_days():
    assert INDICATOR_WARMUP_CALENDAR_DAYS == 30
    assert warmup_fetch_start(START) == WARMUP_START
    # Zero-padded like `exclusive_end`, so an unpadded route date keys the
    # same bar-cache entry as its padded spelling.
    assert warmup_fetch_start("2026-9-8") == WARMUP_START
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        warmup_fetch_start("2026/09/08")


def test_load_data_fetches_the_warmup_pad(loader):
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    assert loader.calls == [(tuple(SYMBOLS), WARMUP_START, PROVIDER_END)]
    assert bt.start_date == START  # the recorded window is unchanged


def test_the_first_decision_bar_carries_warm_indicators(loader):
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    bt.calculate_indicators()

    first = bt.all_data["AAPL"].iloc[0]
    assert _market_date(first.name) == date.fromisoformat(START)
    # Real figures, not the warm-up fallbacks.
    assert first["macd"] != 0.0
    assert first["macd_signal"] != 0.0
    assert first["rsi_14"] != 50.0

    # Equal to a full-history computation, and causal: the padded frame cut at
    # this bar gives the same row, so no bar after it leaked in.
    padded = _hourly_bars(SYMBOLS, WARMUP_START, PROVIDER_END)["AAPL"]
    causal = TechnicalIndicators.calculate_indicators(padded.loc[: first.name])
    expected = causal.iloc[-1]
    for column in ("sma20", "sma50", "bb_upper", "bb_lower", "macd", "macd_signal", "rsi_14"):
        assert first[column] == pytest.approx(expected[column]), column
    assert first["sma50"] == pytest.approx(padded["close"].loc[: first.name].iloc[-50:].mean())


def test_no_pre_window_bar_reaches_the_run(loader):
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    bt.calculate_indicators()
    start = date.fromisoformat(START)

    for frames in (bt.all_data, bt.source_data):
        for frame in frames.values():
            assert _market_date(frame.index[0]) == start

    _, agent_curve = bt.run_agent_backtest()
    _, buyhold_curve = bt.run_buyhold_baseline()
    fake_db = engine_mod.db

    window_bars = len(bt.all_data["AAPL"])
    assert window_bars == 4 * 7  # Tue-Fri, seven hourly bars each
    assert agent_curve and buyhold_curve
    for point in agent_curve + buyhold_curve:
        assert _market_date(point["timestamp"]) >= start
    for _, points in fake_db.equity_points:
        assert all(_market_date(p["timestamp"]) >= start for p in points)
    for _, trades in fake_db.trades:
        assert all(_market_date(t["timestamp"]) >= start for t in trades)
    # The initial-capital point is the first in-window bar, not a pad bar.
    assert _market_date(agent_curve[0]["timestamp"]) == start


def test_run_metadata_records_the_warmup_start(loader):
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    metadata = bt._run_metadata()
    assert metadata["warmup_start_date"] == WARMUP_START
    assert metadata["provider_end_date"] == PROVIDER_END


def test_a_window_with_only_pad_bars_is_still_no_data(monkeypatch):
    class _PadOnly(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((tuple(symbols), start, end))
            return _hourly_bars(symbols, start, START)

    monkeypatch.setattr(engine_mod, "create_market_data_provider", lambda *a, **k: _PadOnly())
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    with pytest.raises(MarketDataUnavailableError, match="No alpaca market data"):
        bt.load_data()


# ---------------------------------------------------------------------------
# Minute source: the pad is aggregated with the window, then dropped
# ---------------------------------------------------------------------------

class _MinuteLoader:
    def __init__(self):
        self.source_timeframe = None
        self.calls = []

    def configure_source_timeframe(self, value):
        self.source_timeframe = value

    def fetch_bars(self, symbols, start, end):
        self.calls.append((tuple(symbols), start, end))
        timestamps = []
        for day in _weekdays(start, end):
            timestamps.extend(
                pd.date_range(
                    _ET.localize(datetime(day.year, day.month, day.day, 9, 30)),
                    _ET.localize(datetime(day.year, day.month, day.day, 15, 55)),
                    freq="5min",
                )
            )
        closes = [_close(row / 12) for row in range(len(timestamps))]
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [1000] * len(closes),
            },
            index=pd.DatetimeIndex(timestamps),
        )
        frame.attrs[FRAME_ATTR_FEED] = "sip"
        frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 5
        return {symbol: frame.copy() for symbol in symbols}


def test_minute_source_trims_the_pad_after_aggregation(monkeypatch):
    loader = _MinuteLoader()

    def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
        loader.configure_source_timeframe(source_timeframe)
        return loader

    fake_db = _DB()
    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    monkeypatch.setattr(engine_mod, "db", fake_db)

    bt = HourlyBacktester(START, END, use_llm=False, symbols=["AAPL"])
    bt.load_data()
    assert loader.calls[0][1] == WARMUP_START
    assert bt.intraday_mode is True
    assert len(bt.source_data["AAPL"]) == 4 * 78
    assert len(bt.all_data["AAPL"]) == 4 * 7
    # The open-stamp convention survives the split: the session filter reads it.
    assert bt.source_data["AAPL"].attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] == 5
    # Quality describes the traded window, not the pad.
    assert bt.data_quality["total_decision_bars"] == 4 * 7

    bt.calculate_indicators()
    assert bt.all_data["AAPL"].iloc[0]["macd"] != 0.0
    _, curve = bt.run_agent_backtest()
    assert len(curve) == 4 * 78
    assert _market_date(curve[0]["timestamp"]) == date.fromisoformat(START)


# ---------------------------------------------------------------------------
# A-share: the corporate-action gap check sees the trade window only
# ---------------------------------------------------------------------------

CN_START, CN_END, CN_PROVIDER_END = "2026-04-01", "2026-04-14", "2026-04-15"
CN_WARMUP_START = "2026-03-02"
EX_RIGHTS = date(2026, 3, 16)  # inside the pad


def _cn_bars(symbols, start, end):
    sessions = (time(10, 30), time(11, 30), time(14), time(15))
    idx = pd.DatetimeIndex(
        [
            datetime.combine(day, session, tzinfo=_CN)
            for day in _weekdays(start, end)
            for session in sessions
        ],
        name="timestamp",
    )
    frames = {}
    for offset, symbol in enumerate(symbols):
        closes = [round(_close(row) + offset * 10, 2) for row in range(len(idx))]
        frames[symbol] = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [10_000] * len(closes),
            },
            index=idx,
        )
    return frames


class _AshareProvider:
    """Raises the real refusal for a gap on any date it is asked to gate,
    which is how ``response_to_market_rules`` behaves: it checks every date
    in ``bars_by_symbol``."""

    def __init__(self):
        self.calls = []
        self.depth_starts = []
        self.rule_calls = []

    def fetch_bars(self, symbols, start, end, *, depth_start=None):
        self.calls.append((tuple(symbols), start, end))
        self.depth_starts.append(depth_start)
        return _cn_bars(symbols, start, end)

    def fetch_usd_cny(self, symbols, start, end):
        return {date(2026, 3, 31): 7.0}

    def fetch_market_rules(self, symbols, start, end, *, bars_by_symbol):
        self.rule_calls.append((start, end, bars_by_symbol))
        dates = sorted({ts.date() for frame in bars_by_symbol.values() for ts in frame.index})
        if EX_RIGHTS in dates:
            raise CorporateActionGapError(
                [CorporateActionGap(symbols[0], EX_RIGHTS, Decimal("-0.23"))]
            )
        rules = []
        for symbol in symbols:
            frame = bars_by_symbol[symbol]
            for trading_date in dates:
                daily = frame[frame.index.date == trading_date]
                rules.append(DailyMarketRule(
                    symbol=symbol,
                    trading_date=trading_date,
                    suspended=False,
                    official_close_price=daily.iloc[-1]["close"],
                    final_bar_timestamp=daily.index[-1].to_pydatetime(),
                ))
        return MarketRuleCalendar(rules)


def test_an_ex_rights_date_in_the_pad_does_not_refuse_the_run(monkeypatch):
    provider = _AshareProvider()
    monkeypatch.setattr(
        engine_mod, "create_market_data_provider", lambda _source, universe=None: provider
    )
    monkeypatch.setattr(engine_mod, "db", _DB())
    bt = HourlyBacktester(
        CN_START,
        CN_END,
        use_llm=False,
        data_source=IFIND_ASHARE,
        decision_source=RULE_BASED_DECISION_SOURCE,
    )
    bt.load_data()

    assert provider.calls == [(A_SHARE_DEMO_6_SYMBOLS, CN_WARMUP_START, CN_PROVIDER_END)]
    # The provider's depth floor judges the traded window, not the pad.
    assert provider.depth_starts == [CN_START]
    rule_start, _, rule_bars = provider.rule_calls[0]
    assert rule_start == CN_START
    assert min(
        ts.date() for frame in rule_bars.values() for ts in frame.index
    ) == date.fromisoformat(CN_START)
    assert bt._ifind_common_start.date() == date.fromisoformat(CN_START)

    bt.calculate_indicators()
    first = bt.all_data[A_SHARE_DEMO_6_SYMBOLS[0]].iloc[0]
    assert first.name.date() == date.fromisoformat(CN_START)
    assert first["macd"] != 0.0
    _, curve = bt.run_agent_backtest()
    assert curve[0]["timestamp"].startswith(CN_START)
