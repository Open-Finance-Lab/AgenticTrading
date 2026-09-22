"""Pre-fetch the default backtest windows into the on-disk bar cache.

A cold instance otherwise charges the first visitor the full bar fetch, which
is precisely the user this cache exists for. Runs on a daemon thread from
``app.py``'s startup hook -- the PARENT web process, never a backtest child
(a child never runs ``app.py``, so there is nothing to suppress there).

Cost, named rather than discovered later: three batched Alpaca calls per
deploy, and merging to ``main`` auto-deploys prod via the CI hook. Negligible
quota, but it is a new recurring outbound call -- which is why
``ATL_BAR_CACHE_WARM`` is **strict opt-in** and this module does nothing until
an operator sets it. Default-on billed the Docker image, every fork and
self-host, and every ``uvicorn --reload`` save on a developer machine holding
keys. A failure is logged and swallowed -- a cold cache is the status quo,
not an outage.

This module is separate from ``bar_cache`` for one reason: it imports
``AlpacaDataLoader``, which imports ``bar_cache``. Keeping the loader out of
``bar_cache`` is what keeps that dependency one-directional.
"""

from __future__ import annotations

import json
from typing import List, Set, Tuple

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.infrastructure.market_data import bar_cache
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    AlpacaDataLoader,
    MarketDataUnavailableError,
    configured_feed_name,
)
from dashboard.backend.paths import CONFIG_DIR

#: The bare ``POST /backtest/run`` defaults. Deliberately a local copy rather
#: than an import: reaching into ``api/routers/backtests.py`` from
#: ``infrastructure/`` inverts the layering. The copies are pinned equal by
#: ``test_route_defaults_match_the_route_signature``.
ROUTE_DEFAULT_START = "2026-05-01"
ROUTE_DEFAULT_END = "2026-05-07"

#: The default US profile fetches 5m source bars and aggregates to 60m
#: decisions (``profiles.py``, the ``(ALPACA, "djia_30")`` entry). The cache
#: key includes the source timeframe, so warming at any other resolution warms
#: nothing a real run can use.
WARM_SOURCE_TIMEFRAME = "5m"


def _defaults_window():
    """``(symbols, start, end)`` from ``config/defaults.json``, or None."""
    try:
        payload = json.loads(
            (CONFIG_DIR / "defaults.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    settings = (payload or {}).get("defaultSettings") or {}
    symbols = [
        str(symbol).strip().upper()
        for symbol in (settings.get("assetList") or [])
        if str(symbol).strip()
    ]
    start = str(settings.get("startDate") or "").strip()
    end = str(settings.get("endDate") or "").strip()
    if not symbols or not start or not end:
        return None
    return symbols, start, end


def warm_windows() -> List[Tuple[List[str], str, str]]:
    """The ``(symbols, start, end)`` triples worth holding warm, in order."""
    windows: List[Tuple[List[str], str, str]] = []
    defaults = _defaults_window()
    if defaults is not None:
        symbols, start, end = defaults
        windows.append((symbols, start, end))
        # Every default run ALSO fetches the full Dow over the same window for
        # the index baseline (`engine.py`'s index-baseline block passes
        # `self.start_date`/`self.end_date` verbatim). Same key, so the five
        # Mag7 names warmed above are hits and only twenty-five are requested.
        windows.append((list(DJIA_30), start, end))
    # A bare `POST /backtest/run` resolves to the djia_30 profile, so its
    # universe is the full Dow, not the modal's Mag7.
    windows.append((list(DJIA_30), ROUTE_DEFAULT_START, ROUTE_DEFAULT_END))
    return windows


def _entries_on_disk(symbols, *, start: str, end: str, feed: str) -> Set[str]:
    """Which of these symbols a later run would actually hit, as a set.

    A ``stat`` per file rather than a parquet read: the question is whether
    the entry exists, and an entry is both files or neither.

    A SET, not a count, because both callers below need identity rather than
    quantity -- one to tell this window's own writes from an earlier window's,
    the other to avoid counting a shared symbol twice.
    """
    stored: Set[str] = set()
    for symbol in symbols:
        try:
            paths = bar_cache.entry_paths(
                symbol,
                start=start,
                end=end,
                source_timeframe=WARM_SOURCE_TIMEFRAME,
                feed=feed,
            )
            if all(path.exists() for path in paths):
                stored.add(str(symbol))
        except OSError:  # a cold cache is the status quo, never an outage
            continue
    return stored


def warm_bar_cache() -> int:
    """Fetch each warm window once. Returns how many symbol-windows are ready."""
    if not bar_cache.enabled() or not bar_cache.warm_enabled():
        return 0
    try:
        loader = AlpacaDataLoader()
    except MarketDataUnavailableError as exc:
        print(f"📦 bar cache warm: skipped ({exc})", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - a cold cache is the status quo
        print(f"📦 bar cache warm: skipped ({exc})", flush=True)
        return 0
    loader.configure_source_timeframe(WARM_SOURCE_TIMEFRAME)
    try:
        # Part of the cache key, so the count below has to ask about the same
        # entries a later run would. Read once: an unreadable value here fails
        # every window's fetch anyway, and this is the cheaper place to say so.
        feed = configured_feed_name()
    except Exception as exc:  # noqa: BLE001 - a cold cache is the status quo
        print(f"📦 bar cache warm: skipped ({exc})", flush=True)
        return 0
    # The (symbol, window) keys confirmed on disk. A SET because the first two
    # windows share a `start`/`end` and therefore share keys for every symbol
    # in both -- `warm_windows`' own comment says the Mag7 names are hits when
    # the Dow window runs. Adding the per-window counts reported those symbols
    # twice: 67 "ready" for ~62 entries, a number that overstates by an amount
    # depending on the defaults file.
    ready: Set[Tuple[str, str, str]] = set()
    for symbols, start, end in warm_windows():
        wanted = {str(symbol) for symbol in symbols}
        # Sampled BEFORE the fetch so this window is judged on its own writes.
        # Counting the directory afterwards asked the wrong question: window
        # two found the overlap window one had stored and reported a non-zero
        # `stored`, so the alarm below could not fire for it even when all
        # twenty-five of its own writes were refused -- the alarm silenced by
        # exactly the case it was added to catch.
        before = _entries_on_disk(symbols, start=start, end=end, feed=feed)
        try:
            frames = loader.fetch_bars(list(symbols), start, end)
        except Exception as exc:  # noqa: BLE001
            print(
                f"📦 bar cache warm: {start}..{end} failed: {exc}",
                flush=True,
            )
            continue
        # What is STORED, not what came back. `fetch_bars` returns its frames
        # whether or not `write_many` accepted a byte of them -- the int it
        # returns is discarded at the call site -- so on an account with no
        # SIP entitlement every window falls back to IEX, every write is
        # refused, three billable calls are made per deploy forever, and the
        # line below used to report a full warm. "Ran and cached nothing"
        # must not print the sentence that means it worked.
        after = _entries_on_disk(symbols, start=start, end=end, feed=feed)
        if frames and before != wanted and after == before:
            # Something was missing, a fetch answered for it, and disk did not
            # move. `before != wanted` is what keeps an already-warm window
            # (nothing to store, nothing stored) from tripping the alarm.
            print(
                f"📦 bar cache warm: {start}..{end} fetched {len(frames)} "
                "symbols and stored none -- see the refusal above; this "
                "window will be re-fetched on every deploy until it is fixed",
                flush=True,
            )
        ready.update((symbol, start, end) for symbol in after)
    print(f"📦 bar cache warm: {len(ready)} symbol-windows ready", flush=True)
    return len(ready)
