"""Shared market-data store: blocking single-flight, key isolation, negative
cache, LRU eviction (T1 of the 2026-07-24 agent-scale spec)."""

import threading
import time

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import market_data_store as mds


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
