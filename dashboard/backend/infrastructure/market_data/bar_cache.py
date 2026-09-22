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


# --- key derivation --------------------------------------------------------


def _digest(
    symbol: str, *, start: str, end: str, source_timeframe: str, feed: str
) -> str:
    # \x1f (unit separator) cannot appear in a symbol, date or feed name, so
    # no two distinct keys can join to the same string.
    raw = "\x1f".join(
        (
            str(SCHEMA_VERSION),
            str(symbol),
            str(start),
            str(end),
            str(source_timeframe),
            str(feed),
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def entry_paths(
    symbol: str, *, start: str, end: str, source_timeframe: str, feed: str
) -> Tuple[Path, Path]:
    """``(parquet_path, meta_path)`` for one key.

    The filename carries a readable symbol slug so the directory can be
    eyeballed, plus the digest so that slug -- which drops the dot in
    ``BRK.B`` -- can never collide with a different key.
    """
    slug = re.sub(r"[^A-Za-z0-9]", "", str(symbol).upper())[:12] or "SYM"
    base = cache_dir() / "{}-{}".format(
        slug,
        _digest(
            symbol,
            start=start,
            end=end,
            source_timeframe=source_timeframe,
            feed=feed,
        ),
    )
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def _discard(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


def _discard_if_stale(now: float, *paths: Path) -> bool:
    """Remove ``paths`` only if every one that exists is older than the grace.

    A younger file may belong to a live writer. Returns True if removed.
    """
    for path in paths:
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue  # vanished: the writer finished or another reader cleared it
        if age < _STRAY_GRACE_SECONDS:
            return False
    _discard(*paths)
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    handle, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp, path)
    except BaseException:
        _discard(Path(tmp))
        raise


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    handle, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    os.close(handle)
    try:
        frame.to_parquet(tmp, engine="pyarrow", compression="snappy")
        os.replace(tmp, path)
    except BaseException:
        _discard(Path(tmp))
        raise


def _meta_matches(
    meta: Any, *, symbol: str, start: str, end: str, source_timeframe: str, feed: str
) -> bool:
    """Guard the truncated digest: a collision becomes a miss, not wrong bars."""
    return (
        isinstance(meta, dict)
        and meta.get("schema_version") == SCHEMA_VERSION
        and meta.get("symbol") == str(symbol)
        and meta.get("start") == str(start)
        and meta.get("end") == str(end)
        and meta.get("source_timeframe") == str(source_timeframe)
        and meta.get("feed") == str(feed)
    )


# --- settlement --------------------------------------------------------------


def window_is_settled(end: Any, *, now: Optional[float] = None) -> bool:
    """True once every bar the window can contain closed at least a day ago.

    ``end`` is what the caller handed ``fetch_bars``: a ``YYYY-MM-DD`` string
    on every shipped path, an ISO datetime on none of them but accepted. A
    naive value is read as UTC, matching ``parse_alpaca_end`` in alpaca_bars
    (re-implemented here rather than imported: the dependency runs one way).
    Anything unparseable is NOT settled -- the cache cannot prove a window it
    cannot read is closed, and a miss is the safe answer.
    """
    try:
        text = str(end).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    clock = time.time() if now is None else float(now)
    return parsed.timestamp() + _SETTLE_MARGIN_SECONDS <= clock


# --- read / write ----------------------------------------------------------


def read_many(
    symbols: Iterable[str],
    *,
    start: str,
    end: str,
    source_timeframe: str,
    feed: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """Serve whatever is on disk. Returns ``(hits, metas)``.

    A symbol appears in both mappings or in neither: the parquet holds the
    bars, the sidecar holds the ``last_fetch`` its caller must restore, and
    half an entry is not an entry.
    """
    hits: Dict[str, pd.DataFrame] = {}
    metas: Dict[str, Dict[str, Any]] = {}
    if not enabled():
        return hits, metas
    ttl = ttl_seconds()
    now = time.time()
    for symbol in symbols:
        parquet_path, meta_path = entry_paths(
            symbol,
            start=start,
            end=end,
            source_timeframe=source_timeframe,
            feed=feed,
        )
        parquet_exists = parquet_path.exists()
        meta_exists = meta_path.exists()
        if not (parquet_exists and meta_exists):
            # Half an entry is a miss either way. Whether to CLEAR it depends
            # on its age: a concurrent writer is legitimately between its two
            # os.replace calls (sidecar landed, parquet still being written),
            # and unlinking its sidecar here would destroy that write -- under
            # contention every child would pay the fetch and the key could
            # stay cold indefinitely. Only a stale survivor is a dead writer's.
            if parquet_exists or meta_exists:
                _discard_if_stale(now, parquet_path, meta_path)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _discard(parquet_path, meta_path)
            continue
        if not _meta_matches(
            meta,
            symbol=symbol,
            start=start,
            end=end,
            source_timeframe=source_timeframe,
            feed=feed,
        ):
            _discard(parquet_path, meta_path)
            continue
        fetched_at = meta.get("fetched_at")
        if not isinstance(fetched_at, (int, float)) or now - float(fetched_at) > ttl:
            _discard(parquet_path, meta_path)
            continue
        try:
            frame = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001 - never fail a backtest
            print(
                f"📦 bar cache: discarding unreadable entry for {symbol}: {exc}",
                flush=True,
            )
            _discard(parquet_path, meta_path)
            continue
        # mtime is the LRU clock; the TTL reads `fetched_at` above so touching
        # here cannot keep a stale entry alive forever.
        try:
            os.utime(parquet_path, None)
            os.utime(meta_path, None)
        except OSError:
            pass
        hits[symbol] = frame
        stored = meta.get("last_fetch")
        metas[symbol] = dict(stored) if isinstance(stored, dict) else {}
    return hits, metas


def write_many(
    frames: Dict[str, pd.DataFrame],
    *,
    start: str,
    end: str,
    source_timeframe: str,
    feed: str,
    last_fetch: Optional[Dict[str, Any]],
    sip_fallback_to_iex: bool = False,
    end_clamped: bool = False,
) -> int:
    """Store frames under one key window. Returns how many entries were written.

    The refusals here are the reason this cache is safe. They are whole-batch,
    not per-frame: a clamped or IEX-fallback response is wrong for **every**
    symbol in it, not some of them.
    """
    if not enabled() or not frames:
        return 0
    if end_clamped:
        # A SIP request reaching into the last ALPACA_SIP_DELAY_MINUTES has its
        # end capped to now-15m, so the SAME requested window returns a shorter
        # frame depending on when you ask. Storing it under the full window's
        # key would make that truncation permanent for the life of the instance.
        print("📦 bar cache: not storing a clamped SIP window", flush=True)
        return 0
    if sip_fallback_to_iex:
        # The IEX-on-refusal retry re-requests with the original unclamped end
        # and never sets end_clamped, so the result looks pristine while being
        # a different tape at ~2.5% of volume.
        print("📦 bar cache: not storing an IEX-fallback window", flush=True)
        return 0
    if not window_is_settled(end):
        # The two flags above only fire on the SIP-on-Basic path. Under iex,
        # ALPACA_ALLOW_RECENT_SIP=1 or a zero delay, a window whose end has not
        # arrived yet returns a partial frame with end_clamped=False -- and
        # baselines.py passes end_date+1 while backtests.py has no future-date
        # check, so such requests do reach here. Stored under the full window's
        # key, that half-day would be served as complete for the TTL.
        print(f"📦 bar cache: not storing a window that has not settled (end={end})", flush=True)
        return 0
    if not isinstance(last_fetch, dict):
        # Only the failure paths leave `last_fetch` unset, and those return no
        # frames -- so this is a caller contract violation, not a data state.
        return 0
    directory = cache_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"📦 bar cache: cannot create {directory}: {exc}", flush=True)
        return 0
    stored_last_fetch = _jsonable(last_fetch)
    now = time.time()
    written = 0
    # What this call put on disk, for the eviction pass: it must never evict
    # the batch that triggered it, and it sizes the pass from these bytes.
    written_paths: List[Path] = []
    written_bytes = 0
    for symbol, frame in frames.items():
        # An absent or empty symbol is not a negative fact worth persisting:
        # "Alpaca had nothing for AAPL today" must not become "AAPL has no
        # data" for the next seven days.
        if frame is None or getattr(frame, "empty", True):
            continue
        parquet_path, meta_path = entry_paths(
            symbol,
            start=start,
            end=end,
            source_timeframe=source_timeframe,
            feed=feed,
        )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "fetched_at": now,
            "symbol": str(symbol),
            "start": str(start),
            "end": str(end),
            "source_timeframe": str(source_timeframe),
            "feed": str(feed),
            "last_fetch": stored_last_fetch,
        }
        try:
            _atomic_write_text(meta_path, json.dumps(meta))
            # Parquet second: its appearance is the commit point.
            _atomic_write_parquet(parquet_path, frame)
        except Exception as exc:  # noqa: BLE001 - never fail a backtest
            print(f"📦 bar cache: write failed for {symbol}: {exc}", flush=True)
            _discard(parquet_path, meta_path)
            continue
        written += 1
        written_paths.extend((parquet_path, meta_path))
        for path in (parquet_path, meta_path):
            try:
                written_bytes += path.stat().st_size
            except OSError:
                pass
    return written
