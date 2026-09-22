"""Pre-fetch the default backtest windows into the on-disk bar cache.

A cold instance otherwise charges the first visitor the full bar fetch, which
is precisely the user this cache exists for. Runs on a daemon thread from
``app.py``'s startup hook -- the PARENT web process, never a backtest child
(a child never runs ``app.py``, so there is nothing to suppress there).

Cost, named rather than discovered later: three batched Alpaca calls per
deploy, and merging to ``main`` auto-deploys prod via the CI hook. Negligible
quota, but it is a new recurring outbound call. A failure is logged and
swallowed -- a cold cache is the status quo, not an outage.

This module is separate from ``bar_cache`` for one reason: it imports
``AlpacaDataLoader``, which imports ``bar_cache``. Keeping the loader out of
``bar_cache`` is what keeps that dependency one-directional.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.infrastructure.market_data import bar_cache
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    AlpacaDataLoader,
    MarketDataUnavailableError,
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
    warmed = 0
    for symbols, start, end in warm_windows():
        try:
            frames = loader.fetch_bars(list(symbols), start, end)
        except Exception as exc:  # noqa: BLE001
            print(
                f"📦 bar cache warm: {start}..{end} failed: {exc}",
                flush=True,
            )
            continue
        warmed += len(frames)
    print(f"📦 bar cache warm: {warmed} symbol-windows ready", flush=True)
    return warmed
