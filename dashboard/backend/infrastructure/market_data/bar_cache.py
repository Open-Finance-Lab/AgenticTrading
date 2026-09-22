"""Cross-process on-disk cache for Alpaca source bars.

Why a file and not a dict: a dashboard backtest is a real OS subprocess
(``Popen``, a fresh interpreter), so the in-process ``market_data_store``
``OrderedDict`` is empty on every run and can never serve a backtest child.
A file on the instance's disk can. It does **not** need to survive a redeploy
-- Render's live service has ``disk: null`` -- it only needs to outlive a
``Popen``.

This module deliberately imports nothing from ``alpaca_bars``: that module
imports this one, and the refusal rules take their provenance flags as
arguments rather than reading them back off ``DataFrame.attrs``. One direction
of dependency, one owner per flag.

Design notes that are load-bearing:

* An entry is TWO files -- ``<name>.parquet`` and ``<name>.json``. A read
  requires both; either one alone is a miss. The parquet is written second,
  so its appearance is the commit point.
* For any given key the sidecar's contents are a pure function of that key:
  everything in ``last_fetch`` is either a key component or a flag
  :func:`write_many` refuses to store. Two processes racing one key therefore
  write byte-identical sidecars, which is why two files need no cross-file
  atomicity.
* A half-entry younger than ``_STRAY_GRACE_SECONDS`` is left alone: a
  concurrent writer is legitimately between its two ``os.replace`` calls, and
  up to ``MAX_ACTIVE_DASHBOARD_BACKTESTS`` children race one key. A reader
  that unlinked the survivor would destroy in-flight writes and could keep a
  contended key cold indefinitely. Only a *stale* half-entry -- a writer that
  died -- is cleared, by the reader that finds it or by the eviction pass.
* ``fetched_at`` in the sidecar, not the file mtime, drives the TTL. Reads
  touch mtime for LRU, which would otherwise keep a hot entry alive forever
  and turn the TTL into a no-op.
* A window is stored only once it has *settled*: its ``end`` is at least
  ``_SETTLE_MARGIN_SECONDS`` in the past. The clamp and fallback flags cover
  the SIP-on-Basic path only; under ``iex``, ``ALPACA_ALLOW_RECENT_SIP=1`` or
  a zero delay, a window whose end has not arrived yet comes back partial with
  ``end_clamped=False`` and would otherwise be stored as complete.
* Every failure path degrades to "no cache" and never raises. A cache that can
  fail a backtest is worse than no cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from dashboard.backend.paths import BAR_CACHE_DIR

#: Bumped whenever the stored layout changes, so a format change invalidates
#: every entry at once instead of producing unreadable ones. It is part of the
#: key, so old entries simply stop being found and age out through the cap.
SCHEMA_VERSION = 1

_DEFAULT_MAX_MB = 256
_DEFAULT_TTL_DAYS = 7
_MIN_MAX_MB, _MAX_MAX_MB = 1, 16384
_MIN_TTL_DAYS, _MAX_TTL_DAYS = 1, 365

#: A file younger than this may belong to a live writer: a ``*.tmp`` mid-write,
#: or a sidecar whose parquet is about to land. Older, it is a crashed writer's
#: leftover and is reclaimed -- by the reader that trips over it or by the
#: eviction pass. Generous on purpose: a stray costs bytes, a wrong guess costs
#: another process its fetch.
_STRAY_GRACE_SECONDS = 3600.0

#: A window is stored only once its ``end`` is this far in the past. Alpaca's
#: ``end`` is exclusive and filters on each bar's *opening* timestamp (see
#: ``AlpacaDataLoader._effective_end``), so a date-only end of ``D`` covers bars
#: opening before ``D 00:00 UTC`` -- through ``D-1``'s session. A day covers
#: the longest supported source bar (60m, ``SUPPORTED_BAR_TIMEFRAMES``) plus
#: every feed delay with no timeframe arithmetic in here, and stays right if a
#: daily source ever lands. The price: a window ending yesterday is served
#: cold until tomorrow -- a shape none of the default windows takes.
_SETTLE_MARGIN_SECONDS = 24 * 3600.0

#: How long a process trusts its last full directory scan before rescanning
#: regardless of what it has written since. Other processes' writes are
#: invisible to the running estimate, so the cap is enforced within this many
#: seconds of being crossed, not on the byte -- it is a runaway bound, not a
#: quota.
_SWEEP_INTERVAL_SECONDS = 60.0

#: The same vocabulary ``allow_recent_sip`` uses in alpaca_bars.py. Anything
#: outside it reads as OFF: for a default-on kill switch, "junk keeps it on"
#: would defeat the switch at exactly the moment someone reached for it.
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def _flag(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUTHY:
        return True
    if raw not in _FALSEY:
        print(f"WARNING: {name}={raw!r} is not a boolean; reading it as off", flush=True)
    return False


def _bounded_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer. Never raises -- this module is on the boot path."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(
            f"WARNING: {name}={raw!r} is not an integer; using {default}",
            flush=True,
        )
        return default
    if not minimum <= value <= maximum:
        print(
            f"WARNING: {name}={value} is outside {minimum}..{maximum}; "
            f"using {default}",
            flush=True,
        )
        return default
    return value


def enabled() -> bool:
    """Whether reads and writes are served. Default ON.

    Default-on is deliberate: an opt-in cache that is off in production
    delivers nothing, and the refusal rules in :func:`write_many` make it fail
    safe. The blast radius is one deploy, because the store is ephemeral.
    """
    return _flag("ATL_BAR_CACHE", default=True)


def warm_enabled() -> bool:
    """Whether ``bar_cache_warm`` pre-fetches on boot. Default ON."""
    return _flag("ATL_BAR_CACHE_WARM", default=True)


def cache_dir() -> Path:
    """Where entries live. ``ATL_BAR_CACHE_DIR`` overrides, for an operator
    pointing at a mounted volume and for tests pointing at ``tmp_path``."""
    override = (os.getenv("ATL_BAR_CACHE_DIR") or "").strip()
    return Path(override) if override else BAR_CACHE_DIR


def max_bytes() -> int:
    return (
        _bounded_int(
            "ATL_BAR_CACHE_MAX_MB",
            default=_DEFAULT_MAX_MB,
            minimum=_MIN_MAX_MB,
            maximum=_MAX_MAX_MB,
        )
        * 1024
        * 1024
    )


def ttl_seconds() -> float:
    return (
        float(
            _bounded_int(
                "ATL_BAR_CACHE_TTL_DAYS",
                default=_DEFAULT_TTL_DAYS,
                minimum=_MIN_TTL_DAYS,
                maximum=_MAX_TTL_DAYS,
            )
        )
        * 86400.0
    )


def describe() -> str:
    """One line for the boot log, matching the ``<store> backend: …`` idiom."""
    if not enabled():
        return "bar cache: disabled"
    return (
        f"bar cache: enabled ({cache_dir()}, "
        f"cap {max_bytes() // (1024 * 1024)}MB, "
        f"ttl {ttl_seconds() / 86400:.0f}d)"
    )
