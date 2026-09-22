"""Shared market-data store: blocking single-flight, key isolation, negative
cache, LRU eviction (T1 of the 2026-07-24 agent-scale spec)."""

import threading
import time
from datetime import datetime, time as clock_time

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import market_data_store as mds
from dashboard.backend.infrastructure.market_data.frequency import FrequencyConfigError


def _synth_bars(symbols=("AAPL", "MSFT"), start="2026-04-15", end="2026-04-16"):
    idx = pd.date_range(start=start, end=str(end) + " 23:59", freq="1h", tz="UTC")
    et = idx.tz_convert("US/Eastern")
    mask = (et.dayofweek < 5) & (
        ((et.hour > 9) & (et.hour < 16)) | ((et.hour == 16) & (et.minute == 0))
    )
    idx = idx[mask]
    data = {}
    for si, sym in enumerate(sorted(symbols)):
        n = len(idx)
        close = 100.0 + si * 5 + np.linspace(0, 1.0, n)
        df = pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 1000.0},
            index=idx,
        )
        data[sym] = df
    return data


class _CountingLoader:
    calls = 0

    def fetch_bars(self, symbols, start, end):
        type(self).calls += 1
        return _synth_bars(symbols, start, end)


@pytest.fixture(autouse=True)
def _fresh_store():
    mds._reset_for_tests()
    _CountingLoader.calls = 0
    yield
    mds._reset_for_tests()


SYMS = ["AAPL", "MSFT"]


def test_build_returns_complete_bundle():
    ds = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    assert set(ds.all_data) == {"AAPL", "MSFT"}
    assert ds.total_steps == len(ds.timestamps) > 0
    assert "AAPL" in ds.price_cache
    assert _CountingLoader.calls == 1


def test_single_flight_one_build_for_concurrent_requesters():
    n = 8
    barrier = threading.Barrier(n)
    out = [None] * n

    def go(i):
        barrier.wait()
        out[i] = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                                 loader_factory=_CountingLoader)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert _CountingLoader.calls == 1
    assert all(o is out[0] for o in out)  # identical object, not copies


def test_key_isolation_by_date_range():
    a = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_CountingLoader)
    b = mds.get_dataset(SYMS, "2026-04-13", "2026-04-14", loader_factory=_CountingLoader)
    assert a is not b
    assert _CountingLoader.calls == 2


def test_symbol_order_is_not_part_of_the_key():
    """REGRESSION (#512). The key held `tuple(symbols)`, so the same universe
    in a different order missed the single-flight cache and the dataset was
    built and held TWICE -- two loaded bar windows for one universe, against
    the memory ceiling MAX_ACTIVE_DASHBOARD_BACKTESTS is sized for. Never a
    wrong number, just a duplicate, which is why nothing caught it.
    """
    a = mds.get_dataset(["AAPL", "MSFT"], "2026-04-15", "2026-04-16",
                        loader_factory=_CountingLoader)
    b = mds.get_dataset(["MSFT", "AAPL"], "2026-04-15", "2026-04-16",
                        loader_factory=_CountingLoader)
    assert a is b
    assert _CountingLoader.calls == 1
    # peek reaches the same entry, so the v2 fast path is not order-sensitive
    # either -- it is the caller most likely to build its list differently.
    assert mds.peek(["MSFT", "AAPL"], "2026-04-15", "2026-04-16") is a


def test_two_markets_do_not_collide_on_one_entry():
    """REGRESSION (#511). The key had no market dimension while
    `_build_dataset` hardcoded US rules, so the same symbols over the same
    window on two markets were one entry -- computed under whichever market
    built first, and then served to the other.
    """
    us = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader, market="US",
                         timezone="US/Eastern")
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16", market="US") is us
    # The resident US dataset is NOT handed to a CN caller, which is the whole
    # defect: it would have been served bars bucketed on ET sessions.
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16", market="CN") is None
    assert mds._dataset_key(SYMS, "2026-04-15", "2026-04-16", "60m", "60m",
                            "US") != mds._dataset_key(
        SYMS, "2026-04-15", "2026-04-16", "60m", "60m", "CN")


def test_a_cn_build_of_us_bars_refuses_instead_of_filtering_them_as_et():
    """The behavioural consequence of threading the market through, and the
    reason the key change alone would not have been enough.

    These synthetic bars are US regular trading hours. Asked for them as a CN
    market, the build now finds no in-session timestamps and raises. Before, it
    filtered them against 09:30-16:00 ET regardless of the market asked for and
    returned a perfectly ordinary-looking dataset -- US bars, US sessions,
    published under a CN run.
    """
    with pytest.raises(RuntimeError, match="No trading hours"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                        loader_factory=_CountingLoader, market="CN",
                        timezone="Asia/Shanghai")


def test_trading_timestamps_follow_the_market_sessions():
    """The fourth US assumption, and the one that mattered most: the in-session
    filter carried its own copy of 09:30-16:00 ET as a literal. For a CN
    profile that window is 21:30-04:00 CST, so every A-share bar was dropped --
    after the aggregation had correctly bucketed them on CN sessions. The
    bounds now come from `bar_aggregation.session_windows`, one owner.
    """
    bars = _synth_bars()
    us = mds._build_trading_timestamps(bars, market="US", timezone="US/Eastern")
    assert us, "US bars must survive the US session filter"

    # Same UTC bars read as CN: US RTH lands overnight in Shanghai, outside
    # both CN sessions, so none of them is in-session.
    cn = mds._build_trading_timestamps(bars, market="CN", timezone="Asia/Shanghai")
    assert cn == []

    # And a genuinely CN-session bar passes the CN filter but not the US one.
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-04-15 10:30", tz="Asia/Shanghai")], name="timestamp"
    )
    frame = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
         "volume": [1.0]},
        index=idx,
    )
    cn_bars = {"600000.SH": frame}
    assert mds._build_trading_timestamps(
        cn_bars, market="CN", timezone="Asia/Shanghai"
    ) == list(idx)
    assert mds._build_trading_timestamps(
        cn_bars, market="US", timezone="US/Eastern"
    ) == []


def test_the_session_filter_and_the_aggregation_share_one_owner():
    """Guard against the copy coming back. Two owners of the session bounds is
    what made the CN case drop every bar while each half looked correct.
    """
    from dashboard.backend.domain.backtesting import bar_aggregation

    # `clock_time`, not `time`: this module imports the time MODULE, and
    # `from datetime import time` would rebind it -- the exact shadowing
    # engine.py:23-31 has a comment block about.
    assert mds.session_windows is bar_aggregation.session_windows
    assert bar_aggregation.session_windows("CN") == (
        (clock_time(9, 30), clock_time(11, 30)),
        (clock_time(13, 0), clock_time(15, 0)),
    )
    assert bar_aggregation.session_windows("US") == (
        (clock_time(9, 30), clock_time(16, 0)),
    )


def test_peek_is_nonblocking_and_only_returns_resident():
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is None  # cold miss

    started, release = threading.Event(), threading.Event()

    class _SlowLoader:
        def fetch_bars(self, symbols, start, end):
            started.set()
            release.wait(5)
            return _synth_bars(symbols, start, end)

    t = threading.Thread(
        target=lambda: mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                                       loader_factory=_SlowLoader))
    t.start()
    assert started.wait(5)
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is None  # in-flight: still None
    release.set()
    t.join(10)
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is not None  # resident hit


def test_build_failure_propagates_and_negative_caches(monkeypatch):
    class _Boom:
        calls = 0

        def fetch_bars(self, symbols, start, end):
            type(self).calls += 1
            raise RuntimeError("alpaca down")

    with pytest.raises(RuntimeError, match="alpaca down"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Boom)
    # Within the negative TTL: same error, NO second fetch (no retry stampede).
    with pytest.raises(RuntimeError, match="alpaca down"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Boom)
    assert _Boom.calls == 1
    # After the negative TTL the build is retried (and can now succeed).
    real_now = time.monotonic()
    monkeypatch.setattr(mds, "_now", lambda: real_now + 31.0)
    ds = mds.get_dataset(SYMS, "2026-04-15", "2026-04-16",
                         loader_factory=_CountingLoader)
    assert ds.total_steps > 0
    assert _CountingLoader.calls == 1


def test_failure_propagates_to_concurrent_waiters():
    started, release = threading.Event(), threading.Event()
    errors = []

    class _SlowBoom:
        def fetch_bars(self, symbols, start, end):
            started.set()
            release.wait(5)
            raise RuntimeError("alpaca down")

    def leader():
        try:
            mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_SlowBoom)
        except RuntimeError as e:
            errors.append(("leader", str(e)))

    def waiter():
        started.wait(5)
        try:
            mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_SlowBoom)
        except RuntimeError as e:
            errors.append(("waiter", str(e)))

    tl, tw = threading.Thread(target=leader), threading.Thread(target=waiter)
    tl.start(); tw.start()
    started.wait(5)
    time.sleep(0.05)  # let the waiter reach event.wait()
    release.set()
    tl.join(10); tw.join(10)
    assert sorted(who for who, _ in errors) == ["leader", "waiter"]
    assert all("alpaca down" in msg for _, msg in errors)


def test_lru_eviction_bounds_entries_and_never_breaks_holders(monkeypatch):
    monkeypatch.setattr(mds, "MARKET_DATA_CACHE_MAX_ENTRIES", 2)
    d1 = mds.get_dataset(SYMS, "2026-04-13", "2026-04-14", loader_factory=_CountingLoader)
    mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_CountingLoader)
    mds.get_dataset(SYMS, "2026-04-17", "2026-04-18", loader_factory=_CountingLoader)
    assert mds.peek(SYMS, "2026-04-13", "2026-04-14") is None      # LRU-evicted
    assert mds.peek(SYMS, "2026-04-15", "2026-04-16") is not None
    assert mds.peek(SYMS, "2026-04-17", "2026-04-18") is not None
    # The evicted dataset stays fully usable via the held reference (GC contract).
    assert d1.total_steps > 0 and "AAPL" in d1.all_data


def test_empty_fetch_raises_runtime_error():
    class _Empty:
        def fetch_bars(self, symbols, start, end):
            return {}

    with pytest.raises(RuntimeError, match="No market data"):
        mds.get_dataset(SYMS, "2026-04-15", "2026-04-16", loader_factory=_Empty)


def test_completed_leader_is_not_self_evicted_under_cap(monkeypatch):
    """A slow-built dataset must survive its own completion. It is the hottest
    entry (its requester is about to use it), so evicting it here would drop the
    hot dataset AND let a same-key request racing into the pre-signal window
    become a second leader (redundant build == single-flight violation).
    Regression: the leader path must refresh LRU recency before eviction, like
    every other access path already does."""
    monkeypatch.setattr(mds, "MARKET_DATA_CACHE_MAX_ENTRIES", 1)

    started, release = threading.Event(), threading.Event()

    class _SlowLoader:
        def fetch_bars(self, symbols, start, end):
            started.set()
            release.wait(5)
            return _synth_bars(symbols, start, end)

    slow = ("2026-04-13", "2026-04-14")
    fast = ("2026-04-15", "2026-04-16")

    t = threading.Thread(
        target=lambda: mds.get_dataset(SYMS, *slow, loader_factory=_SlowLoader))
    t.start()
    assert started.wait(5)  # slow build is in flight (entry not yet 'done')

    # This completes while the slow build is still running, so when the slow
    # build finishes both entries are 'done' and cap=1 forces one eviction.
    mds.get_dataset(SYMS, *fast, loader_factory=_CountingLoader)

    release.set()
    t.join(10)

    # The just-completed (most-recently-used) slow entry is retained; the older
    # fast entry is the eviction victim. The pre-fix behavior evicted the slow
    # entry it had just built.
    assert mds.peek(SYMS, *slow) is not None
    assert mds.peek(SYMS, *fast) is None


def test_timestamp_quorum_rounds_up_to_a_real_eighty_percent():
    timestamp = pd.Timestamp("2026-04-15 14:30:00+00:00")
    frame = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex([timestamp]),
    )
    bars = {
        "A": frame.copy(),
        "B": frame.copy(),
        "C": frame.iloc[0:0].copy(),
    }

    assert mds._build_trading_timestamps(bars) == []


def test_minute_dataset_exposes_dropped_bucket_quality():
    eastern = "US/Eastern"
    timestamps = pd.date_range(
        pd.Timestamp(datetime(2026, 4, 15, 9, 30), tz=eastern),
        pd.Timestamp(datetime(2026, 4, 15, 11, 25), tz=eastern),
        freq="5min",
    ).delete(15)
    prices = np.arange(len(timestamps), dtype=float) + 100
    bars = {
        "AAPL": pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": 1000.0,
            },
            index=timestamps,
        )
    }

    class _MinuteLoader:
        source_timeframe = "60m"

        def configure_source_timeframe(self, value):
            self.source_timeframe = value

        def fetch_bars(self, symbols, start, end):
            return bars

    dataset = mds.get_dataset(
        ["AAPL"],
        "2026-04-15",
        "2026-04-15",
        loader_factory=_MinuteLoader,
        source_timeframe="5m",
        decision_timeframe="60m",
    )

    assert dataset.total_steps == 1
    assert dataset.data_quality["total_decision_bars"] == 2
    assert dataset.data_quality["usable_decision_bars"] == 1
    assert dataset.data_quality["dropped_decision_bars"] == 1
    assert dataset.data_quality["missing_source_bars"] == 1


def test_minute_dataset_rejects_loader_that_ignores_requested_timeframe():
    class _IgnoringLoader:
        source_timeframe = "60m"

        def configure_source_timeframe(self, value):
            pass

        def fetch_bars(self, symbols, start, end):
            pytest.fail("frequency drift must fail before fetching")

    with pytest.raises(
        FrequencyConfigError,
        match="configured source timeframe mismatch: requested 5m, reported 60m",
    ):
        mds.get_dataset(
            ["AAPL"],
            "2026-04-15",
            "2026-04-15",
            loader_factory=_IgnoringLoader,
            source_timeframe="5m",
            decision_timeframe="60m",
        )


def test_minute_dataset_rejects_fetch_evidence_with_wrong_timeframe():
    class _MisreportingLoader:
        source_timeframe = "60m"

        def configure_source_timeframe(self, value):
            self.source_timeframe = value

        def fetch_bars(self, symbols, start, end):
            self.last_fetch = {"source_timeframe": "60m"}
            return _synth_bars(symbols, start, end)

    with pytest.raises(
        FrequencyConfigError,
        match="fetch source timeframe mismatch: requested 5m, reported 60m",
    ):
        mds.get_dataset(
            ["AAPL"],
            "2026-04-15",
            "2026-04-15",
            loader_factory=_MisreportingLoader,
            source_timeframe="5m",
            decision_timeframe="60m",
        )
