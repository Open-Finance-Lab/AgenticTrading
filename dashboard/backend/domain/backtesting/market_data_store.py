"""Shared, immutable market-data datasets for backtest sessions (T1).

One dataset (indicator-enriched bars + trading timestamps + price cache) per
``(symbols, start_date, end_date)`` key, shared by every session with that
config. READ-ONLY CONTRACT: every consumer treats ``all_data`` frames,
``timestamps`` and ``price_cache`` as immutable — verified convention across
the engine, baselines, and PortfolioManager. Never mutate a dataset.

Concurrency model (deliberately NOT cache.py's coordinator, whose followers
never block): the first requester for a key builds; concurrent requesters
block on a ``threading.Event`` and receive the same object. A build failure
propagates to every waiter and is negative-cached for ``NEGATIVE_TTL_SECONDS``
so a dead upstream doesn't trigger a retry stampede.

LOCK RULE: ``get_dataset`` may block for a full Alpaca fetch — it must only be
called from loader threads, NEVER while holding the run-creation lock.
``peek`` is non-blocking and is the only entry point allowed under that lock.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader

# Read once at import (tests monkeypatch the module constant). Entry count, not
# bytes: measured ~1.7 MB for a month-long dataset (was cited as ~50 MB), but
# that is a floor — the size print below counts only the all_data frames (not
# timestamps or price_cache) and was taken on synthetic harness bars, not real
# Alpaca DJIA-30 data. It no longer supports the old ~200 MB worst-case claim
# against the 512 MB free tier; there is no settled byte budget, so the
# 4-entry cap rests on entry count alone. Byte-aware accounting is a
# 1000-tier refinement; the size print below keeps a pathological mix visible.
MARKET_DATA_CACHE_MAX_ENTRIES = int(os.getenv("MARKET_DATA_CACHE_MAX_ENTRIES", "4"))
NEGATIVE_TTL_SECONDS = 30.0

_ET_TZ = pytz.timezone("US/Eastern")

_now = time.monotonic  # indirection so tests can advance the clock


class MarketDataset:
    """Immutable bundle of everything a session needs from market data."""

    __slots__ = ("key", "all_data", "timestamps", "price_cache", "total_steps")

    def __init__(self, key: Tuple, all_data: Dict[str, pd.DataFrame],
                 timestamps: List[Any], price_cache: Dict[str, Dict[Any, float]]):
        self.key = key
        self.all_data = all_data
        self.timestamps = timestamps
        self.price_cache = price_cache
        self.total_steps = len(timestamps)


class _Entry:
    __slots__ = ("event", "dataset", "error", "negative_until")

    def __init__(self):
        self.event = threading.Event()
        self.dataset: Optional[MarketDataset] = None
        self.error: Optional[BaseException] = None
        self.negative_until: float = 0.0


_cache_lock = threading.Lock()
_cache: "OrderedDict[Tuple, _Entry]" = OrderedDict()


def _dataset_key(symbols, start_date, end_date) -> Tuple:
    return (tuple(symbols), str(start_date), str(end_date))


def peek(symbols, start_date, end_date) -> Optional[MarketDataset]:
    """Non-blocking: the resident dataset, or None (miss / build in flight /
    negative-cached failure). The only store call allowed under _create_lock."""
    with _cache_lock:
        entry = _cache.get(_dataset_key(symbols, start_date, end_date))
        if entry is None or entry.dataset is None:
            return None
        _cache.move_to_end(entry.dataset.key)
        return entry.dataset


def get_dataset(symbols, start_date, end_date,
                loader_factory: Optional[Callable[[], Any]] = None) -> MarketDataset:
    """Blocking single-flight build-or-wait. NEVER call under _create_lock."""
    key = _dataset_key(symbols, start_date, end_date)
    factory = loader_factory or AlpacaDataLoader
    while True:
        with _cache_lock:
            entry = _cache.get(key)
            if (entry is not None and entry.error is not None
                    and _now() >= entry.negative_until):
                del _cache[key]  # negative entry expired: retry the build
                entry = None
            if entry is None:
                entry = _Entry()
                _cache[key] = entry
                is_leader = True
            else:
                _cache.move_to_end(key)
                is_leader = False

        if is_leader:
            try:
                dataset = _build_dataset(key, symbols, start_date, end_date, factory)
            except BaseException as exc:
                with _cache_lock:
                    entry.error = exc
                    entry.negative_until = _now() + NEGATIVE_TTL_SECONDS
                entry.event.set()
                raise
            with _cache_lock:
                entry.dataset = dataset
                # Mark the just-built entry most-recently-used BEFORE evicting.
                # Every other access path (waiter, peek, non-leader) refreshes
                # recency; a leader's entry otherwise keeps its stale
                # insertion-time position, so after a slow build it can be the
                # LRU victim and get evicted here — before event.set() below.
                # That both drops the hottest dataset and lets a same-key
                # request racing into the pre-signal window become a second
                # leader (redundant build == single-flight violation). Refreshing
                # keeps the fresh entry at the back, safe for any cap >= 1.
                _cache.move_to_end(key)
                _evict_lru_locked()
            entry.event.set()
            return dataset

        entry.event.wait()
        if entry.error is not None:
            raise entry.error
        if entry.dataset is not None:
            with _cache_lock:
                if _cache.get(key) is entry:
                    _cache.move_to_end(key)
            return entry.dataset
        # Entry was reset underneath us (tests); retry from scratch.


def _build_dataset(key, symbols, start_date, end_date, factory) -> MarketDataset:
    loader = factory()
    all_data = loader.fetch_bars(list(symbols), start_date, end_date)
    if not all_data:
        raise RuntimeError("No market data returned from Alpaca")
    for symbol, df in all_data.items():
        all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
    timestamps = _build_trading_timestamps(all_data)
    if not timestamps:
        raise RuntimeError("No trading hours in the selected date range")
    price_cache = _build_price_cache(all_data, timestamps)
    dataset = MarketDataset(key, all_data, timestamps, price_cache)
    mb = sum(float(df.memory_usage(deep=True).sum()) for df in all_data.values()) / 1e6
    print(f"📊 market-data dataset built: {key[1]}→{key[2]} "
          f"({len(key[0])} syms, {dataset.total_steps} steps, ~{mb:.1f} MB)")
    return dataset


def _build_trading_timestamps(all_data: Dict[str, pd.DataFrame]) -> List[Any]:
    """Moved verbatim from ExternalBacktestSession._build_trading_timestamps."""
    all_timestamps: set = set()
    for df in all_data.values():
        all_timestamps.update(df.index)
    ordered = sorted(all_timestamps)

    min_required = int(len(all_data) * 0.8)
    filtered = []
    for ts in ordered:
        real_count = sum(1 for df in all_data.values() if ts in df.index)
        if real_count >= min_required:
            filtered.append(ts)
    ordered = filtered if filtered else ordered

    market_hours = []
    for ts in ordered:
        ts_et = ts.astimezone(_ET_TZ)
        hour, minute = ts_et.hour, ts_et.minute
        is_market = (
            (hour > 9 and hour < 16)
            or (hour == 9 and minute >= 30)
            or (hour == 16 and minute == 0)
        )
        if is_market:
            market_hours.append(ts)
    return market_hours


def _build_price_cache(all_data: Dict[str, pd.DataFrame],
                       timestamps: List[Any]) -> Dict[str, Dict[Any, float]]:
    """Moved verbatim from ExternalBacktestSession._build_price_cache."""
    cache: Dict[str, Dict[Any, float]] = {}
    for symbol, df in all_data.items():
        cache[symbol] = {}
        last_price = None
        for timestamp in timestamps:
            if timestamp in df.index:
                last_price = df.loc[timestamp, "close"]
                cache[symbol][timestamp] = float(last_price)
            elif last_price is not None:
                cache[symbol][timestamp] = float(last_price)
    return cache


def _evict_lru_locked() -> None:
    """Drop least-recently-used COMPLETED entries beyond the cap. In-flight
    builds are never evicted. Sessions hold direct references, so eviction
    only stops future sharing — it cannot break a live run."""
    done = [k for k, e in _cache.items() if e.dataset is not None or e.error is not None]
    excess = len(done) - MARKET_DATA_CACHE_MAX_ENTRIES
    for k in done[:max(0, excess)]:
        del _cache[k]


def _reset_for_tests() -> None:
    with _cache_lock:
        for entry in _cache.values():
            entry.event.set()  # release any stranded waiter
        _cache.clear()
