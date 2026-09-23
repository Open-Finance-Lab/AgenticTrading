"""Shared, immutable market-data datasets for backtest sessions (T1).

One dataset (indicator-enriched decision bars + source bars + trading
timestamps + price caches) per ``(symbols, start_date, end_date,
source_timeframe, decision_timeframe, equity_metadata_source, market)`` key,
shared by every session with that config. ``symbols`` is a sorted set and
``market`` is canonical, so the same universe in a different order, with a
repeat, or with the market spelled ``"us"`` is one entry and not two -- and the
dataset is built in that same sorted order, so which caller happened to build
first cannot change what a later one receives. READ-ONLY CONTRACT: every
consumer treats the dataset frames, timestamps and caches as immutable —
verified convention across the engine, baselines, and PortfolioManager. Never
mutate a dataset.

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
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.bar_aggregation import (
    ExecutionFill,
    aggregate_bars_by_symbol,
    plan_execution_fills,
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
from dashboard.backend.infrastructure.market_data.sessions import (
    DEFAULT_MARKET,
    canonical_market,
    frames_open_stamped_minutes,
    is_in_session,
    timezone_for_market,
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
        "execution_fills",
        "source_timeframe", "decision_timeframe",
        "data_quality",
        "equity_metadata",
    )

    def __init__(self, key: Tuple, all_data: Dict[str, pd.DataFrame],
                 timestamps: List[Any], price_cache: Dict[str, Dict[Any, float]],
                 *, source_data: Optional[Dict[str, pd.DataFrame]] = None,
                 source_timestamps: Optional[List[Any]] = None,
                 source_price_cache: Optional[Dict[str, Dict[Any, float]]] = None,
                 execution_fills: Optional[List[ExecutionFill]] = None,
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
        # One ExecutionFill per step. Without a plan -- a dataset whose
        # decision bars ARE its source bars -- each step fills at its own bar's
        # close (``decision_bar_close``), as ``engine._plan_executions`` does.
        # Its open is an hour before the decision, and a field left naming it
        # is one unconditional ``execution_prices`` away from look-ahead.
        self.execution_fills = (
            execution_fills
            if execution_fills is not None
            else [ExecutionFill(timestamp, "close", timestamp) for timestamp in timestamps]
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


# `DEFAULT_MARKET` (imported from `sessions`, which owns it) is what this store
# assumed unconditionally before the market became a parameter, and therefore
# what a caller that passes nothing still gets. Kept as the default rather than
# made required so the in-process test doubles and the legacy callers that
# predate the market dimension are unaffected: the three shipped call sites all
# hold a `MarketProfile` and pass it.
DEFAULT_TIMEZONE = timezone_for_market(DEFAULT_MARKET)


def _resolve_timezone(market: str, timezone: Optional[str]) -> str:
    """The market's timezone; a caller-supplied one must agree with it.

    Only the market is in the key, so the timezone has to be a function of it.
    It was a second, independent argument with its own US default: passing
    ``market="CN"`` alone checked CN sessions against US/Eastern clocks and
    cached the result under the CN key, where every later CN caller -- including
    the ones passing the right zone -- was served it. Refusing a disagreeing
    zone keeps the key honest without adding a derived field to it.
    """
    expected = timezone_for_market(market)
    if timezone is not None and timezone != expected:
        raise ValueError(
            f"timezone {timezone!r} does not match market {market!r} "
            f"(expected {expected!r})"
        )
    return expected


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
        # A set as well as sorted: a list assembled from two sources can repeat
        # a symbol, and the repeat fetches nothing extra while still being a
        # second key for the same dataset.
        tuple(sorted(set(symbols))),
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
        # rules just stores the same wrong answer twice. Canonical, not as
        # passed: "us", "US " and None select the same sessions.
        canonical_market(market),
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
                timezone: Optional[str] = None) -> MarketDataset:
    """Blocking single-flight build-or-wait. NEVER call under _create_lock.

    ``market``/``timezone`` come from the session's ``MarketProfile``. Only
    ``market`` is in the key: the timezone is a function of it (a profile
    pairing "CN" with US/Eastern is a malformed profile, not a second dataset,
    and is refused by ``_resolve_timezone``), and adding a derived field to a
    cache key buys misses rather than safety. Omit ``timezone`` to take the
    market's own.
    """
    market = canonical_market(market)
    timezone = _resolve_timezone(market, timezone)
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
    # Build from the key's symbols -- sorted, deduplicated -- not the caller's
    # list. The key makes [MSFT, AAPL] and [AAPL, MSFT] one entry, so building
    # in the caller's order made the dataset's order depend on which session
    # got there first; a session without explicit symbols iterates
    # `all_data` directly, and two identical runs could prompt differently.
    symbols = list(key[0])
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
    source_data = loader.fetch_bars(symbols, start_date, end_date)
    if not source_data:
        raise RuntimeError("No market data returned from Alpaca")
    # And pin the order of what came back, which is the loader's to choose.
    source_data = {symbol: source_data[symbol] for symbol in sorted(source_data)}
    fetch_evidence = getattr(loader, "last_fetch", None)
    if isinstance(fetch_evidence, dict) and fetch_evidence.get("source_timeframe"):
        actual_source = verify_source_timeframe(
            requested_source,
            fetch_evidence["source_timeframe"],
            evidence="fetch",
        )
    data_quality: Dict[str, Any] = {}
    aggregated = timeframe_minutes(actual_source) < timeframe_minutes(requested_decision)
    if aggregated:
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
    # US only, as in `engine.calculate_indicators`, which gates it on the
    # Alpaca source. It is not a no-op on another market once a US metadata
    # dataset is configured: A-share symbols get looked up in US market-cap and
    # SIC partitions, and a configured-but-missing path raises
    # EquityMetadataUnavailableError for a build that never needed it.
    equity_metadata: Dict[str, Any] = {}
    if market == "US":
        all_data, equity_metadata = load_and_enrich_us_equity_bars(
            all_data,
            timezone=timezone,
        )
    timestamps = _build_trading_timestamps(
        all_data,
        market=market,
        timezone=timezone,
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
    execution_fills = None
    if aggregated:
        fills = plan_execution_fills(
            timestamps,
            source_timestamps,
            source_minutes=timeframe_minutes(actual_source),
            market=market,
            timezone=timezone,
        )
        if len(fills) < len(timestamps):
            timestamps = [timestamp for timestamp in timestamps if timestamp in fills]
            price_cache = _build_price_cache(all_data, timestamps)
        execution_fills = [fills[timestamp] for timestamp in timestamps]
    dataset = MarketDataset(
        key,
        all_data,
        timestamps,
        price_cache,
        source_data=source_data,
        source_timestamps=source_timestamps,
        source_price_cache=source_price_cache,
        execution_fills=execution_fills,
        source_timeframe=actual_source,
        decision_timeframe=requested_decision,
        data_quality=data_quality,
        equity_metadata=equity_metadata,
    )
    mb = sum(float(df.memory_usage(deep=True).sum()) for df in all_data.values()) / 1e6
    print(f"📊 market-data dataset built: {key[1]}→{key[2]} "
          f"({len(key[0])} syms, {dataset.total_steps} steps, ~{mb:.1f} MB)")
    return dataset


def _build_trading_timestamps(
    all_data: Dict[str, pd.DataFrame],
    *,
    min_symbol_coverage: float = 0.8,
    market: str = DEFAULT_MARKET,
    timezone: str = DEFAULT_TIMEZONE,
) -> List[Any]:
    """Return in-session timestamps meeting the requested symbol coverage.

    Session membership comes from ``market_data.sessions`` rather than a
    literal here, so this filter and the aggregation that produced the bars
    agree about when the market is open -- including for tz-naive bars, which
    both read as market-local time. The stamp convention is the frames' own
    (``sessions.frames_open_stamped_minutes``): a raw Alpaca bar is stamped at
    its open, an aggregated decision bar or a legacy loader's at its close.
    """
    open_stamped_minutes = frames_open_stamped_minutes(all_data)
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

    return [
        ts for ts in ordered
        if is_in_session(
            ts,
            market=market,
            timezone=timezone,
            open_stamped_minutes=open_stamped_minutes,
        )
    ]


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
