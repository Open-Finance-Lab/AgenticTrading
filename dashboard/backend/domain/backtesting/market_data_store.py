"""Shared, immutable market-data datasets for backtest sessions (T1).

One dataset (indicator-enriched decision bars + source bars + trading
timestamps + price caches) per ``(symbols, start_date, end_date,
source_timeframe, decision_timeframe, equity_metadata_source, market)`` key,
shared by every session with that config. ``symbols`` is sorted, so the same
universe in a different order is one entry and not two. READ-ONLY CONTRACT: every consumer treats the dataset frames,
timestamps and caches as immutable — verified convention across the engine,
baselines, and PortfolioManager. Never mutate a dataset.

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
from bisect import bisect_left
from collections import OrderedDict
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pytz

from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.bar_aggregation import (
    aggregate_bars_by_symbol,
    session_windows,
    summarize_aggregation_quality,
)
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
from dashboard.backend.infrastructure.market_data.equity_metadata import (
    configured_dataset_path,
    load_and_enrich_us_equity_bars,
)
from dashboard.backend.infrastructure.market_data.frequency import (
    normalize_bar_timeframe,
    timeframe_minutes,
    verify_source_timeframe,
)

# Read once at import (tests monkeypatch the module constant). Entry count, not
# bytes: measured ~1.7 MB for a month-long dataset (was cited as ~50 MB), but
# that is a floor — the size print below counts only the all_data frames (not
# timestamps or price_cache) and was taken on synthetic harness bars, not real
# Alpaca DJIA-30 data. It no longer supports the old ~200 MB worst-case claim
# against what was then a 512 MB free tier; there is no settled byte budget, so
# the 4-entry cap rests on entry count alone. (Prod has been Render Standard /
# 2GB since 2026-09-11, which moves a ceiling this was never actually sized
# against.) Byte-aware accounting is a
# 1000-tier refinement; the size print below keeps a pathological mix visible.
MARKET_DATA_CACHE_MAX_ENTRIES = int(os.getenv("MARKET_DATA_CACHE_MAX_ENTRIES", "4"))
NEGATIVE_TTL_SECONDS = 30.0

_now = time.monotonic  # indirection so tests can advance the clock


class MarketDataset:
    """Immutable bundle of everything a session needs from market data."""

    __slots__ = (
        "key", "all_data", "timestamps", "price_cache", "total_steps",
        "source_data", "source_timestamps", "source_price_cache",
        "execution_timestamps", "source_timeframe", "decision_timeframe",
        "data_quality",
        "equity_metadata",
    )

    def __init__(self, key: Tuple, all_data: Dict[str, pd.DataFrame],
                 timestamps: List[Any], price_cache: Dict[str, Dict[Any, float]],
                 *, source_data: Optional[Dict[str, pd.DataFrame]] = None,
                 source_timestamps: Optional[List[Any]] = None,
                 source_price_cache: Optional[Dict[str, Dict[Any, float]]] = None,
                 execution_timestamps: Optional[List[Any]] = None,
                 source_timeframe: str = "60m",
                 decision_timeframe: str = "60m",
                 data_quality: Optional[Dict[str, Any]] = None,
                 equity_metadata: Optional[Dict[str, Any]] = None):
        self.key = key
        self.all_data = all_data
        self.timestamps = timestamps
        self.price_cache = price_cache
        self.total_steps = len(timestamps)
        self.source_data = source_data if source_data is not None else all_data
        self.source_timestamps = (
            source_timestamps if source_timestamps is not None else timestamps
        )
        self.source_price_cache = (
            source_price_cache
            if source_price_cache is not None
            else price_cache
        )
        self.execution_timestamps = (
            execution_timestamps
            if execution_timestamps is not None
            else list(timestamps)
        )
        self.source_timeframe = source_timeframe
        self.decision_timeframe = decision_timeframe
        self.data_quality = data_quality or {}
        self.equity_metadata = equity_metadata or {}


class _Entry:
    __slots__ = ("event", "dataset", "error", "negative_until")

    def __init__(self):
        self.event = threading.Event()
        self.dataset: Optional[MarketDataset] = None
        self.error: Optional[BaseException] = None
        self.negative_until: float = 0.0


_cache_lock = threading.Lock()
_cache: "OrderedDict[Tuple, _Entry]" = OrderedDict()


#: What this store assumed unconditionally before the market became a
#: parameter, and therefore what a caller that passes nothing still gets. Kept
#: as the default rather than made required so the in-process test doubles and
#: the legacy callers that predate the market dimension are unaffected: the
#: three shipped call sites all hold a `MarketProfile` and pass it.
DEFAULT_MARKET = "US"
DEFAULT_TIMEZONE = "US/Eastern"


def _dataset_key(
    symbols,
    start_date,
    end_date,
    source_timeframe: str = "60m",
    decision_timeframe: str = "60m",
    market: str = DEFAULT_MARKET,
) -> Tuple:
    return (
        # SORTED, not as passed. The same universe in a different order is the
        # same dataset, and keying on the order meant the single-flight cache
        # missed: the dataset was built and held TWICE, two loaded bar windows
        # for one universe, in the process whose memory ceiling
        # MAX_ACTIVE_DASHBOARD_BACKTESTS is sized against. Never a wrong
        # number, just a duplicate. Not reachable while every caller passes a
        # stable config order; it becomes reachable the moment one builds the
        # list from a set, a dict's keys or user input. The key is
        # process-local, so there is no stored key to migrate.
        tuple(sorted(symbols)),
        str(start_date),
        str(end_date),
        normalize_bar_timeframe(source_timeframe),
        normalize_bar_timeframe(decision_timeframe),
        str(configured_dataset_path() or ""),
        # The market selects the session bounds everything downstream is
        # bucketed against -- 09:30-16:00 ET versus 09:30-11:30 + 13:00-15:00
        # CST. Without it the same symbols over the same window on two markets
        # collide on one entry, and the survivor is whichever market happened
        # to build first. Added together with the threading below, never alone:
        # a key that separates two markets while both are computed under US
        # rules just stores the same wrong answer twice.
        str(market),
    )


def peek(
    symbols,
    start_date,
    end_date,
    *,
    source_timeframe: str = "60m",
    decision_timeframe: str = "60m",
    market: str = DEFAULT_MARKET,
) -> Optional[MarketDataset]:
    """Non-blocking: the resident dataset, or None (miss / build in flight /
    negative-cached failure). The only store call allowed under _create_lock."""
    with _cache_lock:
        entry = _cache.get(
            _dataset_key(
                symbols,
                start_date,
                end_date,
                source_timeframe,
                decision_timeframe,
                market,
            )
        )
        if entry is None or entry.dataset is None:
            return None
        _cache.move_to_end(entry.dataset.key)
        return entry.dataset


def get_dataset(symbols, start_date, end_date,
                loader_factory: Optional[Callable[[], Any]] = None,
                *, source_timeframe: str = "60m",
                decision_timeframe: str = "60m",
                market: str = DEFAULT_MARKET,
                timezone: str = DEFAULT_TIMEZONE) -> MarketDataset:
    """Blocking single-flight build-or-wait. NEVER call under _create_lock.

    ``market``/``timezone`` come from the session's ``MarketProfile``. Only
    ``market`` is in the key: the timezone is a function of it (a profile
    pairing "CN" with US/Eastern is a malformed profile, not a second dataset),
    and adding a derived field to a cache key buys misses rather than safety.
    """
    key = _dataset_key(
        symbols,
        start_date,
        end_date,
        source_timeframe,
        decision_timeframe,
        market,
    )
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
                dataset = _build_dataset(
                    key,
                    symbols,
                    start_date,
                    end_date,
                    factory,
                    source_timeframe=source_timeframe,
                    decision_timeframe=decision_timeframe,
                    market=market,
                    timezone=timezone,
                )
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


def _build_dataset(
    key,
    symbols,
    start_date,
    end_date,
    factory,
    *,
    source_timeframe: str,
    decision_timeframe: str,
    market: str = DEFAULT_MARKET,
    timezone: str = DEFAULT_TIMEZONE,
) -> MarketDataset:
    loader = factory()
    requested_source = normalize_bar_timeframe(source_timeframe)
    requested_decision = normalize_bar_timeframe(decision_timeframe)
    configure = getattr(loader, "configure_source_timeframe", None)
    if callable(configure):
        configure(requested_source)
    configured_source = getattr(loader, "source_timeframe", None)
    if configured_source is None:
        # Legacy test doubles and old hourly loaders have no runtime evidence;
        # preserve their historical 60m behavior without attesting it as 5m.
        actual_source = "60m"
    else:
        actual_source = verify_source_timeframe(
            requested_source,
            configured_source,
            evidence="configured",
        )
    source_data = loader.fetch_bars(list(symbols), start_date, end_date)
    if not source_data:
        raise RuntimeError("No market data returned from Alpaca")
    fetch_evidence = getattr(loader, "last_fetch", None)
    if isinstance(fetch_evidence, dict) and fetch_evidence.get("source_timeframe"):
        actual_source = verify_source_timeframe(
            requested_source,
            fetch_evidence["source_timeframe"],
            evidence="fetch",
        )
    data_quality: Dict[str, Any] = {}
    if timeframe_minutes(actual_source) < timeframe_minutes(requested_decision):
        aggregated_data = aggregate_bars_by_symbol(
            source_data,
            source_timeframe=actual_source,
            decision_timeframe=requested_decision,
            market=market,
            timezone=timezone,
        )
        data_quality = summarize_aggregation_quality(aggregated_data)
        all_data = {
            symbol: frame.loc[frame["is_complete"]].copy()
            for symbol, frame in aggregated_data.items()
            if not frame.empty
        }
    else:
        all_data = source_data
    if not all_data:
        raise RuntimeError("No completed decision bars returned from Alpaca")
    for symbol, df in all_data.items():
        all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
    # Passed the profile's timezone for the same reason `engine.py` does: the
    # enrichment reads each bar's LOCAL date to pick which yearly metadata
    # partition to load, so a CST session bucketed as ET straddles a day
    # boundary. It is a no-op unless a US equity metadata dataset is
    # configured, which is what keeps it harmless on a non-US market.
    all_data, equity_metadata = load_and_enrich_us_equity_bars(
        all_data,
        timezone=timezone,
    )
    timestamps = _build_trading_timestamps(
        all_data, market=market, timezone=timezone
    )
    if not timestamps:
        raise RuntimeError("No trading hours in the selected date range")
    price_cache = _build_price_cache(all_data, timestamps)
    source_timestamps = _build_trading_timestamps(
        source_data,
        min_symbol_coverage=0.0,
        market=market,
        timezone=timezone,
    )
    source_price_cache = _build_price_cache(source_data, source_timestamps)
    execution_timestamps = _build_execution_timestamps(
        timestamps,
        source_timestamps,
        timezone=timezone,
    )
    if any(execution_timestamp is None for execution_timestamp in execution_timestamps):
        timestamps = [
            timestamp
            for timestamp, execution_timestamp in zip(
                timestamps, execution_timestamps
            )
            if execution_timestamp is not None
        ]
        execution_timestamps = [
            timestamp for timestamp in execution_timestamps if timestamp is not None
        ]
        price_cache = _build_price_cache(all_data, timestamps)
    dataset = MarketDataset(
        key,
        all_data,
        timestamps,
        price_cache,
        source_data=source_data,
        source_timestamps=source_timestamps,
        source_price_cache=source_price_cache,
        execution_timestamps=execution_timestamps,
        source_timeframe=actual_source,
        decision_timeframe=requested_decision,
        data_quality=data_quality,
        equity_metadata=equity_metadata,
    )
    mb = sum(float(df.memory_usage(deep=True).sum()) for df in all_data.values()) / 1e6
    print(f"📊 market-data dataset built: {key[1]}→{key[2]} "
          f"({len(key[0])} syms, {dataset.total_steps} steps, ~{mb:.1f} MB)")
    return dataset


def _market_day_key(timestamp, timezone: str) -> str:
    if timestamp.tzinfo is None:
        local = pytz.timezone(timezone).localize(timestamp)
    else:
        local = timestamp.astimezone(pytz.timezone(timezone))
    return local.date().isoformat()


def _build_execution_timestamps(
    decision_timestamps: List[Any],
    source_timestamps: List[Any],
    *,
    timezone: str,
) -> List[Any]:
    """Map a decision close to the source bar opening at that exact boundary."""
    source_by_day: Dict[str, List[Any]] = {}
    for timestamp in source_timestamps:
        source_by_day.setdefault(_market_day_key(timestamp, timezone), []).append(
            timestamp
        )
    result = []
    for timestamp in decision_timestamps:
        same_day = source_by_day.get(_market_day_key(timestamp, timezone), [])
        index = bisect_left(same_day, timestamp)
        exact_match = (
            same_day[index]
            if index < len(same_day) and same_day[index] == timestamp
            else None
        )
        result.append(exact_match)
    return result


def _build_trading_timestamps(
    all_data: Dict[str, pd.DataFrame],
    *,
    min_symbol_coverage: float = 0.8,
    market: str = DEFAULT_MARKET,
    timezone: str = DEFAULT_TIMEZONE,
) -> List[Any]:
    """Return in-session timestamps meeting the requested symbol coverage.

    The session bounds come from ``bar_aggregation.session_windows`` rather
    than a literal here, so this filter and the aggregation that produced the
    bars cannot disagree about when the market is open.
    """
    all_timestamps: set = set()
    for df in all_data.values():
        all_timestamps.update(df.index)
    ordered = sorted(all_timestamps)

    min_required = max(1, ceil(len(all_data) * min_symbol_coverage))
    filtered = []
    for ts in ordered:
        real_count = sum(1 for df in all_data.values() if ts in df.index)
        if real_count >= min_required:
            filtered.append(ts)
    ordered = filtered

    windows = session_windows(market)
    local_tz = pytz.timezone(timezone)
    market_hours = []
    for ts in ordered:
        local_time = ts.astimezone(local_tz).time()
        # Inclusive at both ends, matching the literal this replaced: it
        # admitted 16:00 exactly (the close), because a decision bar is stamped
        # at the END of its bucket and the last US bucket ends at the close.
        if any(start <= local_time <= end for start, end in windows):
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
