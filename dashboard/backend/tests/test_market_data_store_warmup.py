"""The shared protocol / ``/api/v2`` dataset gets the same indicator warm-up
pad as a dashboard backtest (#540).

Before, only ``HourlyBacktester.load_data`` fetched one, so an external agent
saw neutral RSI 50, MACD 0 and short-window SMA fallbacks for its first ~50
bars while a dashboard run over the same window saw warm figures: two surfaces
over one engine disagreeing on the features of one window.
"""

from datetime import date
from math import sin

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.infrastructure.market_data.provider import warmup_fetch_start

START, END = "2026-04-15", "2026-04-17"


def _bars(symbols, start, end):
    idx = pd.date_range(start=start, end=end, freq="1h", tz="US/Eastern", inclusive="left")
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 10) & (idx.hour <= 16)]
    out = {}
    for offset, symbol in enumerate(sorted(symbols)):
        close = [100.0 + offset + 0.1 * i + 3 * sin(i / 3) for i in range(len(idx))]
        out[symbol] = pd.DataFrame(
            {"open": close, "high": [c + 0.5 for c in close],
             "low": [c - 0.5 for c in close], "close": close, "volume": 1000.0},
            index=idx,
        )
    return out


class _RangeLoader:
    calls = []

    def fetch_bars(self, symbols, start, end):
        type(self).calls.append((tuple(symbols), start, end))
        return _bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _fresh_store():
    mds._reset_for_tests()
    _RangeLoader.calls = []
    yield
    mds._reset_for_tests()


def test_the_dataset_fetches_the_pad_and_trades_only_the_window():
    ds = mds.get_dataset(["AAPL", "MSFT"], START, END, loader_factory=_RangeLoader)

    assert _RangeLoader.calls[0][1] == warmup_fetch_start(START)
    # The key, and every caller that looks the dataset up by it, still says
    # START: the pad is a function of it.
    assert ds.key[1] == START
    start = date.fromisoformat(START)
    for frames in (ds.all_data, ds.source_data):
        for frame in frames.values():
            assert frame.index[0].date() == start
    assert all(ts.date() >= start for ts in ds.timestamps)
    assert ds.indicator_warmup["fetch_start"] == warmup_fetch_start(START)
    assert ds.indicator_warmup["short_symbols"] == {}


def test_the_first_bar_matches_a_dashboard_runs_warm_figures():
    ds = mds.get_dataset(["AAPL"], START, END, loader_factory=_RangeLoader)
    first = ds.all_data["AAPL"].iloc[0]
    assert first["macd"] != 0.0
    assert first["rsi_14"] != 50.0

    padded = _bars(["AAPL"], warmup_fetch_start(START), "2026-04-18")["AAPL"]
    expected = TechnicalIndicators.calculate_indicators(padded.loc[: first.name]).iloc[-1]
    for column in ("sma20", "sma50", "macd", "macd_signal", "rsi_14", "bb_upper"):
        assert first[column] == pytest.approx(expected[column]), column


def test_a_pad_only_reply_is_still_no_data():
    class _PadOnly(_RangeLoader):
        def fetch_bars(self, symbols, start, end):
            return _bars(symbols, start, START)

    with pytest.raises(RuntimeError, match="No market data returned"):
        mds.get_dataset(["AAPL"], START, END, loader_factory=_PadOnly)
