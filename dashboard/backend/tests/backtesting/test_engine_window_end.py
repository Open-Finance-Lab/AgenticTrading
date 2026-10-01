"""A backtest's ``end_date`` is a day it trades.

Every provider reads ``end`` as half-open, so both engines -- the dashboard's
``HourlyBacktester`` and the protocol runs' shared ``market_data_store`` --
must hand them the day after, and must hand the SAME bound to every fetch one
run makes. These pin the Alpaca/US paths (the prod default, and where the
Sep 7-11 run lost its Friday); the iFinD paths are pinned in
``test_ifind_ashare_engine.py``.
"""

from datetime import date, datetime

import pandas as pd
import pytz
import pytest

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.infrastructure.market_data import provider as provider_mod

START, END, PROVIDER_END = "2026-09-07", "2026-09-11", "2026-09-12"
_ET = pytz.timezone("US/Eastern")


def _bars(symbols):
    idx = pd.DatetimeIndex(
        [_ET.localize(datetime(2026, 9, day, hour)) for day in range(7, 12) for hour in range(10, 16)]
    )
    return {
        symbol: pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
            index=idx,
        )
        for symbol in symbols
    }


class _RecordingLoader:
    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def fetch_bars(self, symbols, start, end):
        type(self).calls.append((tuple(symbols), start, end))
        return _bars(symbols)


class _FakeDB:
    def __init__(self):
        self.runs = []

    def insert_run(self, **kwargs):
        self.runs.append(kwargs)

    def insert_equity_points(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.fixture
def engine(monkeypatch):
    _RecordingLoader.calls = []
    monkeypatch.setattr(
        engine_mod, "create_market_data_provider", lambda *a, **k: _RecordingLoader()
    )
    monkeypatch.setattr(engine_mod, "db", _FakeDB())
    return engine_mod


def test_load_data_fetches_through_the_selected_end_date(engine):
    bt = HourlyBacktester(START, END, use_llm=False)
    bt.load_data()
    # The start is padded for indicator warm-up (#540); the end is not.
    assert [(start, end) for _, start, end in _RecordingLoader.calls] == [
        (provider_mod.warmup_fetch_start(START), PROVIDER_END)
    ]
    assert bt.end_date == END  # the recorded window stays inclusive


def test_djia_refetch_uses_the_bars_bound_not_a_fresh_one(engine, monkeypatch):
    """REGRESSION. The index baseline re-fetches the Dow when the loaded
    universe lacks names. It must cover exactly the agent's window."""
    monkeypatch.setattr(engine_mod, "generate_baselines", lambda **kw: ([], [
        {"timestamp": "2026-09-08T10:00:00", "equity": 1.0, "cash": 0.0, "positions_value": 1.0}
    ]))
    bt = HourlyBacktester(START, END, use_llm=False)
    bt.load_data()
    bt.all_data = {"AAPL": bt.all_data["AAPL"]}  # a universe short of the Dow
    bt.run_djia_baseline()
    assert {end for _, _, end in _RecordingLoader.calls} == {PROVIDER_END}
    assert len(_RecordingLoader.calls) == 2  # the load, then the Dow refetch


def test_the_bound_is_pinned_across_midnight(engine, monkeypatch):
    """Resolved once: a run whose window reaches today must not fetch its
    baseline over a longer window after the date rolls over."""
    today = [date(2026, 9, 11)]
    monkeypatch.setattr(provider_mod, "market_today", lambda market=None: today[0])
    bt = HourlyBacktester(START, END, use_llm=False)
    assert bt.provider_end_date == "2026-09-11"  # today's open session excluded
    today[0] = date(2026, 9, 12)
    assert bt.provider_end_date == "2026-09-11"


def test_a_window_ending_today_does_not_trade_the_open_session(engine, monkeypatch, capsys):
    monkeypatch.setattr(provider_mod, "market_today", lambda market=None: date(2026, 9, 11))
    bt = HourlyBacktester(START, END, use_llm=False)
    bt.load_data()
    assert _RecordingLoader.calls[0][2] == "2026-09-11"
    assert "still open" in capsys.readouterr().out
    metadata = bt._run_metadata()
    assert metadata["open_session_excluded"] is True
    assert metadata["end_date_inclusive"] is True


def test_a_window_entirely_in_the_open_session_says_so(engine, monkeypatch):
    monkeypatch.setattr(provider_mod, "market_today", lambda market=None: date(2026, 9, 11))
    bt = HourlyBacktester(END, END, use_llm=False)
    with pytest.raises(ValueError, match="No completed session"):
        bt.load_data()
    assert _RecordingLoader.calls == []


def test_run_metadata_records_the_window_semantics(engine):
    bt = HourlyBacktester(START, END, use_llm=False)
    bt.load_data()
    metadata = bt._run_metadata()
    assert metadata["end_date_inclusive"] is True
    assert metadata["provider_end_date"] == PROVIDER_END
    assert metadata["open_session_excluded"] is False


# ---------------------------------------------------------------------------
# Protocol runs: the shared dataset store
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    mds._reset_for_tests()
    _RecordingLoader.calls = []
    yield
    mds._reset_for_tests()


def test_protocol_dataset_fetches_through_the_selected_end_date(store):
    dataset = mds.get_dataset(["AAPL"], START, END, loader_factory=_RecordingLoader)
    assert _RecordingLoader.calls == [(("AAPL",), START, PROVIDER_END)]
    assert dataset.provider_end == PROVIDER_END


def test_a_one_day_protocol_run_gets_its_day(store):
    mds.get_dataset(["AAPL"], END, END, loader_factory=_RecordingLoader)
    assert _RecordingLoader.calls == [(("AAPL",), END, PROVIDER_END)]


def test_peek_finds_what_get_dataset_built(store):
    built = mds.get_dataset(["AAPL"], START, END, loader_factory=_RecordingLoader)
    assert mds.peek(["AAPL"], START, END) is built


def test_unpadded_and_padded_ends_are_one_dataset(store):
    built = mds.get_dataset(["AAPL"], START, END, loader_factory=_RecordingLoader)
    assert mds.peek(["AAPL"], START, "2026-9-11") is built


def test_a_malformed_end_is_refused_not_passed_through(store):
    """REGRESSION. The helper used to return an unparseable end unchanged, and
    the provider then read it as the exclusive bound -- the last day dropped
    again, with no signal."""
    assert mds.peek(["AAPL"], START, "2026/09/11") is None  # a miss, not a 500
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        mds.get_dataset(["AAPL"], START, "2026/09/11", loader_factory=_RecordingLoader)
    assert _RecordingLoader.calls == []
