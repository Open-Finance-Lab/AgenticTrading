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

Known altitude limit, so nobody mistakes the measured win for a general one:
the key folds ``start`` and ``end`` in VERBATIM, so an entry serves exactly
one window. A user who runs 05-04..05-12 and then nudges the end to 05-13
shares nothing -- every symbol is re-fetched, and the second window is stored
as a second full copy. What this cache speeds up is therefore a window that
has already been run byte-for-byte: the warmed defaults, a re-run, and the
DJIA index-baseline fetch inside a single run (same window, so the run's own
universe is already on disk). It does NOT speed up an arbitrary window a user
types, which is most of the "loading_bars is ~86% of the dark window" problem.
Making it general means partitioning on ``(symbol, source_timeframe, feed,
day)`` so any sub- or super-range composes from the same entries -- a
different read/write/settlement contract, tracked as a follow-up rather than
smuggled in here. Do not read the ``fetch_seconds`` numbers this feature
publishes as evidence that arbitrary windows got faster.
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

from dashboard.backend.paths import BAR_CACHE_DIR, REPO_ROOT

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


#: ``(name, raw value)`` pairs this process has already complained about.
#: These readers are called PER FETCH -- ``max_bytes`` twice per write batch --
#: because ``ATL_BAR_CACHE_DIR`` and the rest must stay live for tests that
#: point them at ``tmp_path`` after import, so the values cannot be frozen at
#: import the way ``ALPACA_HTTP_TIMEOUT_SECONDS`` is. Warning every time turned
#: one typo into a line per fetch forever, into the same child stdout the
#: parent truncates to ``SUBPROCESS_LOG_HEAD_CHARS`` -- crowding out the
#: ``⏱ phase …`` lines this feature is measured by. Keyed on the value as well
#: as the name so a corrected variable still announces itself if it regresses.
_warned_env: Set[Tuple[str, str]] = set()


def _warn_once(name: str, raw: str, message: str) -> None:
    key = (name, raw)
    if key in _warned_env:
        return
    _warned_env.add(key)
    print(message, flush=True)


def _flag(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUTHY:
        return True
    if raw not in _FALSEY:
        _warn_once(
            name, raw, f"WARNING: {name}={raw!r} is not a boolean; reading it as off"
        )
    return False


def _bounded_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer. Never raises -- this module is on the boot path."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn_once(
            name, raw, f"WARNING: {name}={raw!r} is not an integer; using {default}"
        )
        return default
    if not minimum <= value <= maximum:
        _warn_once(
            name,
            raw,
            f"WARNING: {name}={value} is outside {minimum}..{maximum}; "
            f"using {default}",
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
    """Whether ``bar_cache_warm`` pre-fetches on boot. **Strict opt-in.**

    Unlike :func:`enabled` above, this one SPENDS: it makes three batched
    Alpaca calls from ``app.py``'s startup hook, and that hook runs on every
    boot of every deployment -- the Docker image, each self-host and fork, and
    every ``uvicorn --reload`` restart a developer with keys in
    ``dashboard/.env`` triggers by saving a file. Defaulting it on armed a
    billable outbound call on machines whose operator never asked for one, for
    windows they may never run; CLAUDE.md's ``LEADERBOARD_DAILY_AUTO_DEPLOY``
    bullet forbids exactly that shape, and the evidence it leaks is that both
    ``tests/conftest.py`` and ``scripts/loadtest/stress_serve.py`` had to force
    it off to get a quiet boot.

    Being off costs a cold instance the first visitor's full bar fetch -- the
    status quo before this feature, and recoverable by setting the flag. Being
    on by mistake costs money nobody authorised. Set it in the Render
    dashboard, where the cost is a deployment's decision to make.
    """
    return _flag("ATL_BAR_CACHE_WARM", default=False)


def cache_dir() -> Path:
    """Where entries live. ``ATL_BAR_CACHE_DIR`` overrides, for an operator
    pointing at a mounted volume and for tests pointing at ``tmp_path``.

    A relative override is anchored at ``REPO_ROOT``, matching
    ``ai_hedge_fund/adapter.py`` and ``strategy_universe.py``. The parent
    uvicorn runs from the repo root and a backtest child is spawned with
    ``cwd=DASHBOARD_DIR``, so an unanchored relative value names two
    different directories and the cross-process cache -- the entire point of
    the feature -- misses every time, silently, with both processes logging
    the same text. ``.resolve()`` is NOT the fix: it resolves per process and
    reproduces the split exactly.
    """
    override = (os.getenv("ATL_BAR_CACHE_DIR") or "").strip()
    if not override:
        return BAR_CACHE_DIR
    path = Path(override).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


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


#: Exactly what :func:`entry_paths` produces -- ``<SLUG>-<16 hex>`` with a
#: ``.parquet`` or ``.json`` suffix -- plus the ``*.tmp`` that ``mkstemp``
#: derives from it by appending random characters to that full name.
_OWNED_NAME = re.compile(r"[A-Z0-9]{1,12}-[0-9a-f]{16}\.(?:parquet|json)")


def _is_owned(name: str) -> bool:
    """Whether this directory entry is one this module wrote.

    The eviction sweep classifies by SUFFIX and then unlinks, and
    :func:`cache_dir`'s docstring invites an operator to point
    ``ATL_BAR_CACHE_DIR`` at a directory of their choosing. Point it at
    ``dashboard/storage/data`` -- the parent of the default, and populated
    today -- and an unfiltered sweep deletes ``algo_submissions.json`` and
    the leaderboard state files as "orphan sidecars" the hour they age out.
    Checked once on the way in rather than at each unlink, so the stray
    sweep, the LRU pass and the byte total all inherit it: the cap measures
    this cache's own footprint, never a neighbour's.
    """
    if name.endswith(".tmp"):
        return _OWNED_NAME.match(name) is not None
    return _OWNED_NAME.fullmatch(name) is not None


def _discard(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            # The caller has already decided not to trust this entry (stale,
            # unreadable, or shape-mismatched); a failed unlink just leaves it
            # on disk as an orphan the next stray sweep (below) will reclaim,
            # not a reason to fail the backtest that triggered this discard.
            pass


def _discard_judged(
    parquet_path: Path, meta_path: Path, judged: Optional[str]
) -> None:
    """Unlink an entry only while its sidecar still holds what we judged.

    :func:`_discard` unlinks by PATH, but every caller of this helper decided
    to distrust the entry from CONTENT read earlier, and between the two
    another process can replace both files with a good entry for the same key.
    The TTL branch is the one that actually happens: the child that expired
    this entry a moment ago is refreshing exactly it, and up to
    ``MAX_ACTIVE_DASHBOARD_BACKTESTS`` of them race one key. Unlinking then
    throws away a fetch someone paid for and leaves the key cold for whoever
    asks next -- the same failure the half-entry branch in :func:`read_many`
    spends a grace period to avoid.

    Re-reading first does not make this atomic; it narrows the window from
    "read, parse, judge, unlink" to "read, unlink", and turns the case that
    actually costs something -- a refresh that has already landed -- into a
    no-op. An age guard cannot do this job here: reads touch mtime for LRU,
    so a hot TTL-expired entry always looks young and would never be cleared
    at all.
    """
    if judged is not None:
        try:
            if meta_path.read_text(encoding="utf-8") != judged:
                return  # refreshed since we read it; no longer ours to judge
        except OSError:
            return  # vanished or unreadable: another process is already on it
    _discard(parquet_path, meta_path)


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
            raw_meta = meta_path.read_text(encoding="utf-8")
        except OSError:
            # Could not read it, so there is nothing to judge it on -- and the
            # likely cause is another process already discarding or replacing
            # it. Leave it: a miss is the answer either way, and the stray
            # sweep reclaims a survivor nobody owns.
            continue
        try:
            meta = json.loads(raw_meta)
        except ValueError:
            _discard_judged(parquet_path, meta_path, raw_meta)
            continue
        if not _meta_matches(
            meta,
            symbol=symbol,
            start=start,
            end=end,
            source_timeframe=source_timeframe,
            feed=feed,
        ):
            _discard_judged(parquet_path, meta_path, raw_meta)
            continue
        fetched_at = meta.get("fetched_at")
        if not isinstance(fetched_at, (int, float)) or now - float(fetched_at) > ttl:
            _discard_judged(parquet_path, meta_path, raw_meta)
            continue
        try:
            frame = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception as exc:  # noqa: BLE001 - never fail a backtest
            print(
                f"📦 bar cache: discarding unreadable entry for {symbol}: {exc}",
                flush=True,
            )
            _discard_judged(parquet_path, meta_path, raw_meta)
            continue
        # mtime is the LRU clock; the TTL reads `fetched_at` above so touching
        # here cannot keep a stale entry alive forever.
        try:
            os.utime(parquet_path, None)
            os.utime(meta_path, None)
        except OSError:
            # A failed touch just means this hit won't push out its LRU
            # eviction turn as far as it should have -- it can be evicted
            # sooner than a "true" access time would justify, which costs a
            # future cache miss, never a wrong or missing hit right now.
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
            # Deliberately does NOT discard. These paths may already hold a
            # perfectly good entry from an earlier run -- this key is being
            # rewritten, not created -- and unlinking it turned a failed
            # refresh into a permanent miss. Under the failure most likely to
            # get here, ENOSPC on the ephemeral disk, every write in the batch
            # fails and every failure deleted a working entry: the cache
            # emptied itself exactly when the size cap mattered most.
            #
            # Both surviving states are ones this module already serves.
            # `_atomic_write_text` failing leaves the old pair untouched (it
            # removes only its own tmp). It succeeding and the parquet failing
            # leaves the old parquet under a new sidecar -- still the right
            # bars, since for one key the sidecar varies only in `fetched_at`
            # and the window is settled -- or, with no old parquet, a half
            # entry, which `read_many` reads as a miss and the stray sweep
            # reclaims once no live writer could own it.
            print(f"📦 bar cache: write failed for {symbol}: {exc}", flush=True)
            continue
        written += 1
        written_paths.extend((parquet_path, meta_path))
        for path in (parquet_path, meta_path):
            try:
                written_bytes += path.stat().st_size
            except OSError as exc:
                print(f"📦 bar cache: could not stat {path}: {exc}", flush=True)
    if written:
        try:
            _maybe_enforce_size_cap(directory, written_bytes, protect=written_paths)
        except Exception as exc:  # noqa: BLE001 - eviction must not fail a write
            print(f"📦 bar cache: eviction failed: {exc}", flush=True)
    return written


# --- eviction --------------------------------------------------------------


#: Per directory: ``[last_full_scan_at, bytes_on_disk_at_that_scan,
#: bytes_this_process_wrote_since]``. Process-local on purpose -- a shared
#: index file would be a third thing to keep atomic -- which is why the time
#: bound in ``_maybe_enforce_size_cap`` exists.
#:
#: Bounded because nothing else bounds it. In every deployed configuration
#: ``cache_dir()`` resolves once for the life of the process and this holds
#: exactly one key; it grows only where ``ATL_BAR_CACHE_DIR`` moves at runtime,
#: which today means a test suite handing each case its own ``tmp_path``. The
#: bound is cheap insurance rather than a fix for a live leak, and dropping an
#: entry is safe in a way dropping a cache entry is not: this state only
#: decides whether the NEXT write can skip a directory scan, so a missing entry
#: costs one scan -- the same scan it would have taken with no state at all --
#: and can never produce a wrong answer or a missed eviction.
_MAX_SCAN_STATE_DIRS = 8
_scan_state: Dict[str, List[float]] = {}


def _remember_scan(directory: Path, scanned_at: float, total: float) -> List[float]:
    """Record a completed scan, evicting the stalest directories over the bound.

    Returns the state list so the caller can keep amending it: the entry it
    just wrote is the newest and so the last thing eviction would reach, but
    holding the reference means an eviction that did reach it degrades to a
    detached write instead of a ``KeyError`` on the next line.
    """
    state = [scanned_at, float(total), 0.0]
    _scan_state[str(directory)] = state
    if len(_scan_state) > _MAX_SCAN_STATE_DIRS:
        stalest = sorted(_scan_state, key=lambda key: _scan_state[key][0])
        for key in stalest[: len(_scan_state) - _MAX_SCAN_STATE_DIRS]:
            _scan_state.pop(key, None)
    return state


def enforce_size_cap(*, protect: Iterable[Path] = ()) -> int:
    """One directory scan: sweep stale strays, then evict LRU until under cap.

    The cap is a runaway bound, not a working-set estimate: one symbol over a
    7-weekday window at 5m bars is tens of kilobytes of parquet, so the 256MB
    default holds thousands of symbol-windows. It exists so arbitrary user
    windows cannot grow the cache without bound on an ephemeral disk.

    The byte total counts every file this module owns, strays included; only
    whole entries can be evicted. See the comment on ``total`` below.

    ``protect`` is the batch the caller just wrote. It is never evicted, even
    if that leaves the directory over cap until the next pass: an eviction
    that discards the symbols just fetched turns a paid fetch into a miss, and
    with coarse mtimes (entries written within one tick tie) it did exactly
    that. Ordering among the rest is ``(mtime, name)`` so a tie still evicts
    deterministically.
    """
    directory = cache_dir()
    if not directory.exists():
        # Only reachable before any write (write_many creates it first), so
        # saying nothing here makes "nothing to evict" and "swept clean"
        # the same empty stdout.
        print(f"📦 bar cache: nothing to evict, {directory} does not exist", flush=True)
        return 0
    now = time.time()
    keep: Set[Path] = {Path(path) for path in protect}
    cap = max_bytes()
    # Everything from ONE listing: the parquets, their sidecars, and the
    # strays. A second glob per kind would double the directory walks on a
    # pass that already stats every entry.
    sizes: Dict[Path, Tuple[float, int]] = {}
    try:
        with os.scandir(directory) as listing:
            for item in listing:
                if not _is_owned(item.name):
                    continue  # not ours: never swept, never evicted, never counted
                try:
                    stat = item.stat()
                except OSError:
                    continue
                sizes[Path(item.path)] = (stat.st_mtime, stat.st_size)
    except OSError as exc:
        # Every other failure in this module logs; a silent eviction failure
        # and a clean pass are otherwise indistinguishable.
        print(f"📦 bar cache: cannot scan {directory}: {exc}", flush=True)
        return 0
    # A crashed writer's leftovers -- a *.tmp mid-write, or a sidecar whose
    # parquet never landed -- match no entry and would never be reclaimed by
    # the LRU pass. A young one belongs to a live writer and is left alone.
    for path, (mtime, _size) in list(sizes.items()):
        is_tmp = path.suffix == ".tmp"
        is_orphan_meta = (
            path.suffix == ".json" and path.with_suffix(".parquet") not in sizes
        )
        # The mirror image, which the write ordering makes rarer but does not
        # rule out: `_discard` swallows an OSError per path, so a discard whose
        # sidecar unlink succeeded and whose parquet unlink failed leaves a
        # parquet nothing will ever read. `read_many` clears one only if that
        # exact key is requested again; otherwise it sat here counting against
        # the cap until LRU eviction happened to reach it.
        is_orphan_parquet = (
            path.suffix == ".parquet" and path.with_suffix(".json") not in sizes
        )
        if (is_tmp or is_orphan_meta or is_orphan_parquet) and (
            now - mtime > _STRAY_GRACE_SECONDS
        ):
            _discard(path)
            del sizes[path]
    entries: List[Tuple[float, str, int, Path, Path]] = []
    # The cap bounds BYTES ON DISK, so every file this module owns counts --
    # including a `*.tmp` or an orphan sidecar still inside the grace above.
    # Totalling whole entries only meant a stray contributed nothing: a run of
    # writers killed mid-write (by `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS`, or
    # by the cancel route) piled up half-written parquets against a cap that
    # believed the directory was empty. A stray cannot be EVICTED, only aged
    # out by the sweep above, so counting it buys the room back by evicting
    # entries instead -- which is what a disk-usage bound should do.
    total = sum(size for _mtime, size in sizes.values())
    for path, (mtime, size) in sizes.items():
        if path.suffix != ".parquet":
            continue
        meta_path = path.with_suffix(".json")
        entries.append(
            (
                mtime,
                path.name,
                size + sizes.get(meta_path, (0.0, 0))[1],
                path,
                meta_path,
            )
        )
    state = _remember_scan(directory, now, total)
    if total <= cap:
        return 0
    entries.sort(key=lambda item: item[:2])
    removed = 0
    for _mtime, _name, size, parquet_path, meta_path in entries:
        if total <= cap:
            break
        if parquet_path in keep or meta_path in keep:
            continue
        _discard(parquet_path, meta_path)
        total -= size
        removed += 1
    state[1] = float(total)
    print(
        f"📦 bar cache: evicted {removed} entries to stay under "
        f"{cap // (1024 * 1024)}MB",
        flush=True,
    )
    return removed


def _maybe_enforce_size_cap(
    directory: Path, written_bytes: int, *, protect: Iterable[Path]
) -> None:
    """Run the full scan only when it could matter.

    This process's running estimate (bytes at the last scan plus what it has
    written since) says whether *its* writes could have crossed the cap. Other
    processes' writes are invisible to it, so the scan also runs once the last
    one is older than ``_SWEEP_INTERVAL_SECONDS``. Net effect: a fetch far
    under cap costs one ``stat`` of state rather than one per entry, and the
    cap is enforced within a minute of being crossed rather than on the byte
    -- which is what a runaway bound needs.
    """
    state = _scan_state.get(str(directory))
    now = time.time()
    if state is not None:
        state[2] += float(written_bytes)
        scanned_at, total_at_scan, written_since = state
        if (
            now - scanned_at < _SWEEP_INTERVAL_SECONDS
            and total_at_scan + written_since <= max_bytes()
        ):
            return
    enforce_size_cap(protect=protect)
