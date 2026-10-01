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
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
)
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.indicator_warmup import (
    PAD_BARS_FOR_WARM_FIRST_BAR,
)
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
    VNPY_SIMULATION,
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


def _market_dates(index, zone=_ET):
    return index.tz_convert(zone).date


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


def test_only_the_agent_row_records_the_warmup_evidence(loader):
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    metadata = bt._agent_run_metadata()
    evidence = metadata["indicator_warmup"]
    assert evidence["fetch_start"] == WARMUP_START
    # Weekdays 2026-08-10..09-07 (the fake has no holidays), seven bars each.
    assert evidence["min_pad_bars"] == 21 * 7
    assert evidence["bars_for_warm_start"] == PAD_BARS_FOR_WARM_FIRST_BAR == 49
    assert evidence["short_symbols"] == {}
    assert evidence["unadjusted_gap_trims"] == {}
    assert metadata["provider_end_date"] == PROVIDER_END
    # Baseline rows compute no indicators, and the Dow row is fetched
    # unpadded, so the key would be false on them.
    assert "indicator_warmup" not in bt._run_metadata()

    bt.calculate_indicators()
    bt.run_agent_backtest()
    bt.run_buyhold_baseline()
    rows = {run["agent_name"]: run["metadata"] for run in engine_mod.db.runs}
    assert rows["buy-and-hold"].get("indicator_warmup") is None
    assert sum("indicator_warmup" in m for m in rows.values()) == 1


def test_a_thin_pad_is_recorded_not_reported_as_warm(monkeypatch, capsys):
    """A pad that came back short -- a suspension, a recent listing, a
    provider answering part of the range -- leaves the first bars on cold
    fallbacks. The record must say so rather than name the requested start
    as if the history had arrived."""

    class _ThinPad(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((tuple(symbols), start, end))
            frames = _hourly_bars(symbols, start, end)
            # MSFT's history begins three trading days before the window.
            frames["MSFT"] = frames["MSFT"].loc["2026-09-03":]
            return frames

    monkeypatch.setattr(engine_mod, "create_market_data_provider", lambda *a, **k: _ThinPad())
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()

    evidence = bt._agent_run_metadata()["indicator_warmup"]
    assert evidence["short_symbols"] == {"MSFT": 21}
    assert evidence["min_pad_bars"] == 21
    assert "1 symbol(s) have under 49 pad bars" in capsys.readouterr().out


def test_an_unadjusted_split_in_the_pad_trims_the_pad(monkeypatch):
    """Prices are unadjusted on both feeds. A 4-for-1 split in the pad would put
    4x closes into the window's sma50 -- every early bar reads as a crash -- and
    the window's own corporate-action check never sees pad dates. The pad is
    cut at the break instead, and the cut is recorded."""
    split_day = date(2026, 8, 24)

    class _Split(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((tuple(symbols), start, end))
            frames = _hourly_bars(symbols, start, end)
            aapl = frames["AAPL"]
            before = _market_dates(aapl.index) < split_day
            aapl.loc[before, ["open", "high", "low", "close"]] *= 4
            return frames

    monkeypatch.setattr(engine_mod, "create_market_data_provider", lambda *a, **k: _Split())
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()

    assert min(_market_dates(bt.warmup_data["AAPL"].index)) == split_day
    # MSFT has no break, so its pad is whole.
    assert min(_market_dates(bt.warmup_data["MSFT"].index)) == date(2026, 8, 10)
    evidence = bt._agent_run_metadata()["indicator_warmup"]
    assert evidence["unadjusted_gap_trims"] == {"AAPL": "2026-08-24"}

    bt.calculate_indicators()
    first = bt.all_data["AAPL"].iloc[0]
    # On the post-split scale: no pre-split close reached the average.
    assert first["sma50"] == pytest.approx(
        bt.all_data["AAPL"]["close"].iloc[0], rel=0.1
    )


def test_a_break_on_start_date_drops_the_whole_pad(monkeypatch):
    class _SplitOnStart(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((tuple(symbols), start, end))
            frames = _hourly_bars(symbols, start, end)
            aapl = frames["AAPL"]
            before = _market_dates(aapl.index) < date.fromisoformat(START)
            aapl.loc[before, ["open", "high", "low", "close"]] *= 2
            return frames

    monkeypatch.setattr(
        engine_mod, "create_market_data_provider", lambda *a, **k: _SplitOnStart()
    )
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    bt.load_data()
    assert "AAPL" not in bt.warmup_data
    evidence = bt._agent_run_metadata()["indicator_warmup"]
    assert evidence["unadjusted_gap_trims"] == {"AAPL": START}
    assert evidence["short_symbols"] == {"AAPL": 0}


def test_vnpy_simulation_is_not_padded(loader):
    """The simulator scripts its prices by bar index from the fetch start and
    seeds its base price on (symbol, start, end): a padded fetch would move the
    scripted path into the pad and price the agent's AAPL off a different
    series than the index baseline's."""
    bt = HourlyBacktester(
        START, END, use_llm=False, data_source=VNPY_SIMULATION, symbols=SYMBOLS
    )
    bt.load_data()
    assert loader.calls == [(tuple(SYMBOLS), START, PROVIDER_END)]
    assert bt.warmup_data == {}
    # No pad was fetched, so there is nothing to record -- not a pad that
    # "began" at the window start.
    assert "indicator_warmup" not in bt._agent_run_metadata()


def test_a_hosted_runtime_is_not_padded():
    """The AI Hedge Fund runtime decides off closes and loads its own lookback
    upstream; it never reads the engine's indicators, so a pad is a ~4-5x
    larger fetch bought for nothing on the memory-sensitive runtime (#308)."""
    bt = HourlyBacktester.__new__(HourlyBacktester)
    bt.data_source = "alpaca"
    bt.runtime_type = AI_HEDGE_FUND_RUNTIME_TYPE
    assert bt._wants_indicator_warmup() is False
    bt.runtime_type = PIPELINE_RUNTIME_TYPE
    assert bt._wants_indicator_warmup() is True


def test_a_window_with_only_pad_bars_is_still_no_data(monkeypatch):
    class _PadOnly(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            self.calls.append((tuple(symbols), start, end))
            return _hourly_bars(symbols, start, START)

    monkeypatch.setattr(engine_mod, "create_market_data_provider", lambda *a, **k: _PadOnly())
    bt = HourlyBacktester(START, END, use_llm=False, symbols=SYMBOLS)
    with pytest.raises(MarketDataUnavailableError, match="No alpaca market data"):
        bt.load_data()


def test_a_minute_source_with_only_pad_bars_is_still_no_data(monkeypatch):
    """The intraday branch -- the default US path -- must report a window with
    no session of its own (a weekend, a holiday) as no data, not as the
    data-quality fault "no completed decision bars could be built"."""

    class _PadOnlyMinutes(_MinuteLoader):
        def fetch_bars(self, symbols, start, end):
            return super().fetch_bars(symbols, start, START)

    loader = _PadOnlyMinutes()

    def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
        loader.configure_source_timeframe(source_timeframe)
        return loader

    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    bt = HourlyBacktester(START, END, use_llm=False, symbols=["AAPL"])
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


def test_a_forwarding_wrapper_is_not_offered_depth_start(monkeypatch):
    """A caching/logging wrapper declaring only ``**kwargs`` says nothing about
    whether the provider it forwards to takes ``depth_start``. Offering it
    anyway turned every A-share run into an unexpected-keyword TypeError; not
    offering it degrades to the floor counting the pad, as before #540."""
    inner = _AshareProvider()

    class _LegacyInner:
        def fetch_bars(self, symbols, start, end):
            return inner.fetch_bars(symbols, start, end)

    legacy = _LegacyInner()

    class _Wrapper:
        def fetch_bars(self, symbols, start, end, **kwargs):
            return legacy.fetch_bars(symbols, start, end, **kwargs)

        def __getattr__(self, name):
            return getattr(inner, name)

    monkeypatch.setattr(
        engine_mod, "create_market_data_provider", lambda _source, universe=None: _Wrapper()
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
    assert inner.calls == [(A_SHARE_DEMO_6_SYMBOLS, CN_WARMUP_START, CN_PROVIDER_END)]
    assert inner.depth_starts == [None]
    assert engine_mod._accepts_keyword(_Wrapper().fetch_bars, "depth_start") is True
    assert (
        engine_mod._accepts_keyword(
            _Wrapper().fetch_bars, "depth_start", via_var_keyword=False
        )
        is False
    )


def test_an_ex_rights_break_in_an_a_share_pad_trims_it(monkeypatch):
    """The window's corporate-action audit sees window dates only, so a
    10-for-3 bonus issue inside the pad would warp the first ~50 bars' sma50
    with no label anywhere. The pad is cut at the break and the cut recorded."""
    symbol = A_SHARE_DEMO_6_SYMBOLS[0]

    class _ExRights(_AshareProvider):
        def fetch_bars(self, symbols, start, end, *, depth_start=None):
            frames = super().fetch_bars(symbols, start, end, depth_start=depth_start)
            frame = frames[symbol]
            before = frame.index.date < EX_RIGHTS
            frame.loc[before, ["open", "high", "low", "close"]] *= 1.3
            return frames

    provider = _ExRights()
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
    assert min(bt.warmup_data[symbol].index.date) == EX_RIGHTS
    evidence = bt._agent_run_metadata()["indicator_warmup"]
    assert evidence["unadjusted_gap_trims"] == {symbol: EX_RIGHTS.isoformat()}
    # Every other symbol keeps its whole pad.
    other = A_SHARE_DEMO_6_SYMBOLS[1]
    assert min(bt.warmup_data[other].index.date) == date.fromisoformat(CN_WARMUP_START)
