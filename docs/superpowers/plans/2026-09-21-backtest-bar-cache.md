# Backtest Bar Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** serve a repeated `(symbol, window, timeframe, feed)` bar request from disk instead of from Alpaca, across process boundaries, without ever serving a truncated or wrong-tape window.

**Architecture:** a new `bar_cache` module stores one parquet entry per symbol under `dashboard/storage/data/bar_cache/`, and `AlpacaDataLoader.fetch_bars` becomes a thin wrapper that splits the requested symbols into disk hits and misses, fetches only the misses through the existing (renamed) fetch body, and merges. The cache resolves *above* the >100-symbol batch recursion. A boot-time warm thread in the parent web process pre-fetches the default windows, and `load_data` records a fetch/post-fetch split so the residual is the aggregation cost.

**Tech Stack:** Python 3, pandas 2.3.2, pyarrow 23.0.1 (both already pinned in `requirements.txt` — this change adds **no** new dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-backtest-bar-cache-design.md`

## Global Constraints

- **Nothing in this plan may make a live network call from a test.** The whole suite is offline; `tests/conftest.py` must keep it that way.
- **The cache is ON by default in production and OFF in the test suite.** `tests/conftest.py` sets `ATL_BAR_CACHE=0` at import time. Cache tests opt back in explicitly with `monkeypatch` plus a `tmp_path` directory. The default-on behaviour is pinned by one dedicated test that reads the flag function with no env set, never by turning the cache on suite-wide.
- **Never raise at import.** This module is on the app boot path. Junk or out-of-range values for the two integer variables log a `WARNING` and fall back to the default; a junk value for either **flag** logs a `WARNING` and reads as **off**, because the only reason to set a kill switch is to turn it off, and a typo'd kill switch that stays on is the one outcome the switch exists to prevent. Both readers share `allow_recent_sip`'s truthy vocabulary (`1`/`true`/`yes`/`on`) and, like it, treat anything else as off — the warning is the only addition. A bare `int()` at module scope has killed app boot in this repo before.
- **A cache must never be able to fail a backtest.** Every read failure, write failure, eviction failure and warm failure is caught, logged, and degrades to "no cache".
- **`bar_cache.py` imports nothing from `alpaca_bars.py`.** The dependency runs one way only (`alpaca_bars` → `bar_cache`). Provenance flags are passed as arguments, never read back off `DataFrame.attrs` inside the cache.
- **Exact constants**, copied from the spec: `SCHEMA_VERSION = 1`; size cap default **256 MB** (`ATL_BAR_CACHE_MAX_MB`, range 1–16384); TTL default **7 days** (`ATL_BAR_CACHE_TTL_DAYS`, range 1–365); cache directory `DATA_DIR / "bar_cache"` (`ATL_BAR_CACHE_DIR` overrides); flags `ATL_BAR_CACHE` and `ATL_BAR_CACHE_WARM`, both default **on** when unset, on for `1`/`true`/`yes`/`on`, off for anything else (a recognised `0`/`false`/`no`/`off` silently, junk with a `WARNING`); settle margin **24 hours** (`_SETTLE_MARGIN_SECONDS`, spec §5); stray-file grace **1 hour** (`_STRAY_GRACE_SECONDS`, spec §6); full-scan interval **60 seconds** (`_SWEEP_INTERVAL_SECONDS`, spec §6).
- **Refusals are whole-batch and derived from the frames, never from `last_fetch`.** `last_fetch` describes the *last* request the loader made, which for a >100-symbol call is the last 100-symbol chunk; the `.attrs` stamps are per frame and cover every chunk. The wrapper in Task 4 folds them with `any()`.
- **Never write into `dashboard/storage/data/cache/`.** That directory holds nine git-tracked orphan CSVs. `bar_cache/` is a fresh sibling; `.gitignore:225` (`dashboard/storage/data`) already ignores it wholesale.
- **Do not commit seed-DB mutations.** `dashboard/storage/data/backtest.db` must stay at 688,128 bytes. Stage by explicit path; never `git add -A`. Any ad-hoc `python -c` that imports a backend module must run with `DATABASE_PATH` pointed at a throwaway file.
- **Line numbers in the spec and in this plan are advisory.** Grep for the quoted symbol or signature; never jump to a number.
- Run the suite from the repo root: `pytest dashboard/backend/tests/ -q`.

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/backend/paths.py` (modify) | Add `BAR_CACHE_DIR`. Single source of truth for on-disk locations, as the module's docstring says. |
| `dashboard/backend/infrastructure/market_data/bar_cache.py` (create) | The whole cache: config readers, key derivation, read, write, refusal rules, TTL, eviction. Imports nothing from `alpaca_bars`. |
| `dashboard/backend/infrastructure/market_data/bar_cache_warm.py` (create) | Boot-time warm. Separate module precisely because it *does* import `AlpacaDataLoader`, which `bar_cache.py` must not. |
| `dashboard/backend/infrastructure/market_data/alpaca_bars.py` (modify) | `fetch_bars` splits into a cache-aware wrapper plus `_fetch_bars_uncached` (today's body, including the >100 recursion). |
| `dashboard/backend/domain/backtesting/engine.py` (modify) | Generalise the phase-extras mechanism from `starting`-only to any phase; record `fetch_seconds` inside `loading_bars`. |
| `dashboard/backend/app.py` (modify) | Startup log line + the warm daemon thread. |
| `dashboard/backend/tests/conftest.py` (modify) | Disable the cache and the warm step for the suite. |
| `CLAUDE.md` (modify) | Document the four environment variables and the cache's place in the market-data path. |
| `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py` (create) | Unit tests for the cache module in isolation. |
| `dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py` (create) | Integration tests driving the real `fetch_bars` against the fake Alpaca client. |
| `dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py` (create) | Warm-window construction, the conftest guard, and the route-default source guard. |
| `dashboard/backend/tests/backtesting/test_engine_progress_phases.py` (modify) | Phase-extras generalisation and the `fetch_seconds` split. |

---

## Task 1: The switch and its safe default

Nothing is wired yet. This task lands the directory constant, the four environment readers, and — critically — the conftest change, **before** anything can read the flag. That ordering is deliberate: once Task 4 wires the cache in, a suite that has not already disabled it would start writing into the real `dashboard/storage/data/` tree on every market-data test.

**Files:**
- Modify: `dashboard/backend/paths.py`
- Create: `dashboard/backend/infrastructure/market_data/bar_cache.py`
- Modify: `dashboard/backend/tests/conftest.py`
- Test: `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `paths.BAR_CACHE_DIR: Path`; and from `bar_cache`: `SCHEMA_VERSION: int`, `enabled() -> bool`, `warm_enabled() -> bool`, `cache_dir() -> Path`, `max_bytes() -> int`, `ttl_seconds() -> float`, `describe() -> str`.

- [ ] **Step 1: Add the directory constant**

In `dashboard/backend/paths.py`, directly after the `DATA_DIR` line, add:

```python
# On-disk bar cache (see infrastructure/market_data/bar_cache.py). A fresh
# sibling of `cache/`, never that directory: `cache/` still holds nine
# git-tracked orphan CSVs, and gitignore does not untrack what is already
# tracked. `.gitignore` ignores `dashboard/storage/data` wholesale, so nothing
# written here can be staged by accident.
BAR_CACHE_DIR = DATA_DIR / "bar_cache"
```

- [ ] **Step 2: Write the failing tests for the config surface**

Create `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`:

```python
"""Unit tests for the cross-process on-disk bar cache.

No network: this module never touches Alpaca. Every test points
ATL_BAR_CACHE_DIR at tmp_path, so nothing is written under
dashboard/storage/data.
"""

import json
import os
import time

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data import bar_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Enable the cache against an isolated directory."""
    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "bar_cache"))
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    return tmp_path / "bar_cache"


def _frame(rows=3, start="2026-05-04T13:30:00Z"):
    index = pd.date_range(start, periods=rows, freq="5min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": [10.0 + i for i in range(rows)],
            "high": [11.0 + i for i in range(rows)],
            "low": [9.0 + i for i in range(rows)],
            "close": [10.5 + i for i in range(rows)],
            "volume": [100 * (i + 1) for i in range(rows)],
        },
        index=index,
    )


KEY = dict(start="2026-05-04", end="2026-05-12", source_timeframe="5m", feed="sip")
LAST_FETCH = {
    "feed": "sip",
    "source_timeframe": "5m",
    "requested_end": "2026-05-12",
    "effective_end": "2026-05-12",
    "sip_fallback_to_iex": False,
    "end_clamped": False,
}


# --- configuration surface -------------------------------------------------


def test_cache_is_enabled_by_default(monkeypatch):
    """Production default is ON. Pinned here rather than by enabling the
    cache suite-wide, which would change every existing market-data test."""
    monkeypatch.delenv("ATL_BAR_CACHE", raising=False)
    assert bar_cache.enabled() is True


def test_warm_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE_WARM", raising=False)
    assert bar_cache.warm_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " 0 "])
def test_falsey_values_disable_the_cache(monkeypatch, value):
    monkeypatch.setenv("ATL_BAR_CACHE", value)
    assert bar_cache.enabled() is False


def test_a_junk_flag_value_reads_as_off_and_warns(monkeypatch, capsys):
    """The only reason to set a kill switch is to turn it off. A typo'd value
    that kept the cache ON would defeat the switch at exactly the moment
    someone reached for it; `allow_recent_sip` in alpaca_bars.py already reads
    anything outside its truthy set as off, and this matches it -- adding only
    the warning, so the typo is visible."""
    monkeypatch.setenv("ATL_BAR_CACHE", "maybe")
    assert bar_cache.enabled() is False
    assert "ATL_BAR_CACHE" in capsys.readouterr().out


def test_a_junk_warm_flag_reads_as_off_without_touching_the_cache_flag(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE", raising=False)
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "fasle")
    assert bar_cache.warm_enabled() is False
    assert bar_cache.enabled() is True


def test_defaults_match_the_spec(monkeypatch):
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    assert bar_cache.max_bytes() == 256 * 1024 * 1024
    assert bar_cache.ttl_seconds() == 7 * 86400.0


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5", "99999"])
def test_out_of_range_size_cap_falls_back_and_logs(monkeypatch, capsys, value):
    monkeypatch.setenv("ATL_BAR_CACHE_MAX_MB", value)
    assert bar_cache.max_bytes() == 256 * 1024 * 1024
    assert "ATL_BAR_CACHE_MAX_MB" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["nope", "0", "-1", "400"])
def test_out_of_range_ttl_falls_back_and_logs(monkeypatch, capsys, value):
    monkeypatch.setenv("ATL_BAR_CACHE_TTL_DAYS", value)
    assert bar_cache.ttl_seconds() == 7 * 86400.0
    assert "ATL_BAR_CACHE_TTL_DAYS" in capsys.readouterr().out


def test_cache_dir_defaults_to_the_paths_constant(monkeypatch):
    from dashboard.backend.paths import BAR_CACHE_DIR

    monkeypatch.delenv("ATL_BAR_CACHE_DIR", raising=False)
    assert bar_cache.cache_dir() == BAR_CACHE_DIR
    assert BAR_CACHE_DIR.name == "bar_cache"
    assert BAR_CACHE_DIR.parent.name == "data"


def test_describe_names_the_state(cache_dir, monkeypatch):
    assert bar_cache.describe().startswith("bar cache: enabled (")
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    assert bar_cache.describe() == "bar cache: disabled"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'dashboard.backend.infrastructure.market_data.bar_cache'`.

- [ ] **Step 4: Write the config surface**

Create `dashboard/backend/infrastructure/market_data/bar_cache.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q`
Expected: PASS.

- [ ] **Step 6: Disable the cache and the warm step for the suite**

In `dashboard/backend/tests/conftest.py`, alongside the other import-time `os.environ` lines (near the block that pops `ATL_BACKTEST_WORKER`), add:

```python
# The on-disk bar cache (infrastructure/market_data/bar_cache.py) is ON by
# default in production. The suite runs with it OFF so every existing
# market-data test keeps asserting the exact request shapes it always has --
# test_alpaca_bars.py asserts batching is literally [100, 100, 35], which a
# warm cache would shorten. Cache tests opt back in with monkeypatch plus a
# tmp_path directory; the production default is pinned by
# test_bar_cache.py::test_cache_is_enabled_by_default.
os.environ["ATL_BAR_CACHE"] = "0"

# Warm-on-boot makes LIVE Alpaca calls from app.py's startup hook. Importing
# the app anywhere in the suite must never do that: it is both a network
# dependency in an offline suite and real money. Same reason RENDER and the
# IFIND_* credentials are stripped above.
os.environ["ATL_BAR_CACHE_WARM"] = "0"

# A developer's own cache tuning must not reach the suite.
os.environ.pop("ATL_BAR_CACHE_DIR", None)
os.environ.pop("ATL_BAR_CACHE_MAX_MB", None)
os.environ.pop("ATL_BAR_CACHE_TTL_DAYS", None)
```

- [ ] **Step 7: Run the whole suite to prove nothing moved**

Run: `pytest dashboard/backend/tests/ -q`
Expected: the same pass/skip counts as before this task (the suite was green at `1167ae97`). Nothing is wired yet, so any change here is a real regression.

- [ ] **Step 8: Verify the seed DB is untouched, then commit**

```bash
stat -c '%s' dashboard/storage/data/backtest.db   # must print 688128
git status --short dashboard/storage/data/backtest.db   # must print nothing
git add dashboard/backend/paths.py \
        dashboard/backend/infrastructure/market_data/bar_cache.py \
        dashboard/backend/tests/conftest.py \
        dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py
git commit -m "feat: add the bar cache switch and its safe defaults"
```

---

## Task 2: Store and restore one entry

**Files:**
- Modify: `dashboard/backend/infrastructure/market_data/bar_cache.py`
- Test: `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`

**Interfaces:**
- Consumes: `enabled()`, `cache_dir()`, `ttl_seconds()`, `SCHEMA_VERSION` from Task 1.
- Produces:
  - `read_many(symbols: Iterable[str], *, start: str, end: str, source_timeframe: str, feed: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]` — returns `(hits, metas)`; `metas[symbol]` is the stored `last_fetch` dict. A symbol appears in both mappings or in neither. A half-entry is a miss; it is *cleared* only when older than `_STRAY_GRACE_SECONDS`, so a concurrent writer's in-flight pair is never destroyed by a reader.
  - `write_many(frames: Dict[str, pd.DataFrame], *, start: str, end: str, source_timeframe: str, feed: str, last_fetch: Optional[Dict[str, Any]], sip_fallback_to_iex: bool = False, end_clamped: bool = False) -> int` — returns the number of entries written.
  - `entry_paths(symbol, *, start, end, source_timeframe, feed) -> Tuple[Path, Path]` — `(parquet_path, meta_path)`, exposed for tests.
  - `window_is_settled(end, *, now: Optional[float] = None) -> bool` — the fourth refusal rule (spec §5): a window is stored only once its `end` is `_SETTLE_MARGIN_SECONDS` in the past. `now` is injectable for tests.

- [ ] **Step 1: Write the failing round-trip tests**

Append to `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`:

```python
# --- store and restore -----------------------------------------------------


def test_write_then_read_returns_an_equal_frame(cache_dir):
    frame = _frame()
    assert bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY) == 1
    hits, metas = bar_cache.read_many(["AAPL"], **KEY)
    assert set(hits) == {"AAPL"}
    # check_freq=False: parquet has no slot for DatetimeIndex.freq, so the
    # fixture's freq="5min" comes back as None. Nothing downstream reads freq
    # (aggregate_bars_by_symbol resamples from the timestamps), so the values,
    # dtypes, index name and tz are the contract -- and those are all checked.
    pd.testing.assert_frame_equal(hits["AAPL"], frame, check_freq=False)
    assert metas["AAPL"] == LAST_FETCH


def test_read_restores_the_three_attrs_stamps(cache_dir):
    """feed_provenance() reads these back and the engine persists the result
    into agent_runs.metadata. A hit that loses them is a silent behaviour
    change, not a speed-up."""
    frame = _frame()
    frame.attrs["alpaca_feed"] = "sip"
    frame.attrs["alpaca_sip_fallback"] = False
    frame.attrs["alpaca_end_clamped"] = False
    bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits["AAPL"].attrs == {
        "alpaca_feed": "sip",
        "alpaca_sip_fallback": False,
        "alpaca_end_clamped": False,
    }


def test_read_preserves_the_index_name_and_timezone(cache_dir):
    frame = _frame()
    bar_cache.write_many({"AAPL": frame}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits["AAPL"].index.name == "timestamp"
    assert str(hits["AAPL"].index.dtype) == "datetime64[ns, UTC]"


def test_a_miss_returns_nothing_for_that_symbol(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    hits, metas = bar_cache.read_many(["AAPL", "MSFT"], **KEY)
    assert set(hits) == {"AAPL"}
    assert set(metas) == {"AAPL"}


@pytest.mark.parametrize(
    "override",
    [
        {"feed": "iex"},
        {"source_timeframe": "60m"},
        {"start": "2026-05-05"},
        {"end": "2026-05-13"},
    ],
)
def test_every_key_dimension_is_load_bearing(cache_dir, override):
    """Changing any one of them must miss. The feed especially: curves priced
    off different tapes are not comparable, and omitting it is the exact
    defect market_data_store's own key has today."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["AAPL"], **{**KEY, **override})
    assert hits == {}


def test_symbols_are_keyed_individually_so_order_cannot_matter(cache_dir):
    """Sidesteps market_data_store._dataset_key's order-sensitive
    tuple(symbols): the same set in a different order is a hit here."""
    bar_cache.write_many(
        {"AAPL": _frame(), "MSFT": _frame(rows=4)}, last_fetch=LAST_FETCH, **KEY
    )
    hits, _ = bar_cache.read_many(["MSFT", "AAPL"], **KEY)
    assert set(hits) == {"AAPL", "MSFT"}


def test_a_schema_version_bump_invalidates_every_entry(cache_dir, monkeypatch):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    monkeypatch.setattr(bar_cache, "SCHEMA_VERSION", bar_cache.SCHEMA_VERSION + 1)
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert hits == {}


def test_a_datetime_in_last_fetch_is_stored_as_an_iso_string(cache_dir):
    """`_effective_end` returns a datetime on the SIP path. The sidecar is
    JSON, so it comes back as a string -- and that is fine: the only readers
    are `baselines.py`, which prints it, and the two `source_timeframe`
    checks."""
    from datetime import datetime, timezone

    last_fetch = dict(
        LAST_FETCH, effective_end=datetime(2026, 5, 12, tzinfo=timezone.utc)
    )
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=last_fetch, **KEY)
    _, metas = bar_cache.read_many(["AAPL"], **KEY)
    assert metas["AAPL"]["effective_end"] == "2026-05-12T00:00:00+00:00"


def test_writes_are_atomic_and_leave_no_temp_files(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    assert sorted(p.suffix for p in cache_dir.iterdir()) == [".json", ".parquet"]


def test_a_disabled_cache_writes_and_reads_nothing(cache_dir, monkeypatch):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    assert bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY) == 0
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q -k "write_then_read or attrs_stamps"`
Expected: FAIL with `AttributeError: module 'dashboard.backend.infrastructure.market_data.bar_cache' has no attribute 'write_many'`.

- [ ] **Step 3: Implement key derivation, atomic writes, read and write**

Append to `dashboard/backend/infrastructure/market_data/bar_cache.py`:

```python
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
```

Also add, between the key-derivation helpers and the `# --- read / write` section:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/infrastructure/market_data/bar_cache.py \
        dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py
git commit -m "feat: store and restore one bar cache entry"
```

---

## Task 3: The refusal rules, expiry, corruption and eviction

This is the safety core. Every refusal test must be shown to fail when its guard is removed — a guard never seen to fail is a comment.

**Files:**
- Modify: `dashboard/backend/infrastructure/market_data/bar_cache.py`
- Test: `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `enforce_size_cap(*, protect: Iterable[Path] = ()) -> int` — one directory scan; deletes stale strays (temp files, orphan sidecars) and then least-recently-used entries until under cap, never touching a path in `protect`; returns how many entries were removed. And `_maybe_enforce_size_cap(directory, written_bytes, *, protect)` — what `write_many` actually calls: it runs the full scan only when this process's running estimate says the cap could have been crossed, or when the last scan is older than `_SWEEP_INTERVAL_SECONDS`.

- [ ] **Step 1: Write the failing safety tests**

Append to `dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py`:

```python
# --- the refusal rules (design section 5) ----------------------------------


def test_a_clamped_window_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if end_clamped:` guard in write_many and
    this must fail. A clamped SIP window returns a SHORTER frame for the same
    requested key depending on the wall clock; caching it pins the truncation
    for the life of the instance."""
    written = bar_cache.write_many(
        {"AAPL": _frame()},
        last_fetch=dict(LAST_FETCH, end_clamped=True),
        end_clamped=True,
        **KEY,
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "clamped" in capsys.readouterr().out


def test_an_iex_fallback_window_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if sip_fallback_to_iex:` guard and this must
    fail. The fallback retry re-requests with the original unclamped end and
    never sets end_clamped, so the frame looks pristine while carrying ~2.5%
    of the volume the key claims."""
    written = bar_cache.write_many(
        {"AAPL": _frame()},
        last_fetch=dict(LAST_FETCH, sip_fallback_to_iex=True),
        sip_fallback_to_iex=True,
        **KEY,
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "IEX-fallback" in capsys.readouterr().out


def test_a_window_that_has_not_settled_is_never_stored(cache_dir, capsys):
    """MUTATION TEST: delete the `if not window_is_settled(end):` guard and
    this must fail. The clamp and fallback flags only cover SIP-on-Basic;
    under iex, ALPACA_ALLOW_RECENT_SIP=1 or a zero delay a window whose end is
    still ahead of the clock comes back partial with end_clamped=False."""
    from datetime import date, timedelta

    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    written = bar_cache.write_many(
        {"AAPL": _frame()}, last_fetch=LAST_FETCH, **{**KEY, "end": tomorrow}
    )
    assert written == 0
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []
    assert "not settled" in capsys.readouterr().out


@pytest.mark.parametrize(
    "end, now, expected",
    [
        # `now` is 2026-05-14T00:00:00Z throughout. A date-only end of D covers
        # bars opening before D 00:00 UTC, so D=05-13 settles at 05-14 00:00.
        ("2026-05-13", 1778716800.0, True),
        ("2026-05-13", 1778716799.0, False),
        ("2026-05-14", 1778716800.0, False),
        ("2026-05-13T00:00:00Z", 1778716800.0, True),
        ("2026-05-13T00:00:00+00:00", 1778716800.0, True),
        ("2026-05-12T20:00:00-04:00", 1778716800.0, True),
        ("not-a-date", 1778716800.0, False),
        (None, 1778716800.0, False),
    ],
)
def test_window_is_settled_needs_a_full_day_past_the_exclusive_end(end, now, expected):
    assert bar_cache.window_is_settled(end, now=now) is expected


def test_an_absent_symbol_is_not_cached_as_an_empty_frame(cache_dir):
    """A missing symbol is not a negative fact worth persisting."""
    empty = _frame().iloc[0:0]
    assert bar_cache.write_many({"AAPL": empty}, last_fetch=LAST_FETCH, **KEY) == 0
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_a_missing_last_fetch_stores_nothing(cache_dir):
    assert bar_cache.write_many({"AAPL": _frame()}, last_fetch=None, **KEY) == 0


# --- expiry, corruption, half-entries --------------------------------------


def test_an_entry_past_its_ttl_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - (8 * 86400)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_read_touch_cannot_keep_a_stale_entry_alive(cache_dir):
    """mtime is the LRU clock; the TTL reads `fetched_at`. If the TTL were
    computed from mtime, touching on read would make it unreachable."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["fetched_at"] = time.time() - (8 * 86400)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    os.utime(parquet_path, None)  # fresh mtime, stale fetch
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_a_corrupt_parquet_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.write_bytes(b"not a parquet file")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_corrupt_sidecar_is_a_miss_and_is_removed(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta_path.write_text("{not json", encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists() and not meta_path.exists()


def test_a_fresh_half_entry_is_a_miss_but_is_left_for_its_writer(cache_dir):
    """MUTATION TEST: replace `_discard_if_stale` in read_many with `_discard`
    and this must fail. A sidecar without its parquet is what a concurrent
    writer looks like between its two os.replace calls; a reader that unlinks
    it destroys that write, and under contention the key stays cold while
    every child pays the fetch."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert meta_path.exists()


def test_a_stale_half_entry_is_a_miss_and_is_removed(cache_dir):
    """Older than the grace, the survivor belongs to a writer that died."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta_path.unlink()
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(parquet_path, (stamp, stamp))
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})
    assert not parquet_path.exists()


def test_a_sidecar_whose_key_fields_disagree_is_a_miss(cache_dir):
    """Turns a truncated-digest collision into a miss rather than wrong bars."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    _, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["feed"] = "iex"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert bar_cache.read_many(["AAPL"], **KEY) == ({}, {})


def test_two_writers_racing_one_key_leave_a_readable_entry(cache_dir):
    """os.replace is atomic on POSIX, so a reader sees the old entry or the
    complete new one -- never a partial parquet. Up to
    MAX_ACTIVE_DASHBOARD_BACKTESTS children race the same key on one instance."""
    import threading

    barrier = threading.Barrier(2)

    def writer(rows):
        barrier.wait()
        for _ in range(20):
            bar_cache.write_many(
                {"AAPL": _frame(rows=rows)}, last_fetch=LAST_FETCH, **KEY
            )

    threads = [threading.Thread(target=writer, args=(n,)) for n in (3, 5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    hits, _ = bar_cache.read_many(["AAPL"], **KEY)
    assert len(hits["AAPL"]) in (3, 5)


# --- eviction --------------------------------------------------------------


def _tiny_cap(monkeypatch, cache_dir, keep_entries=2):
    """Shrink the cap to `keep_entries` times the size of one entry on disk.

    Monkeypatching `max_bytes` rather than setting ATL_BAR_CACHE_MAX_MB, whose
    floor is 1MB: a parquet of a few hundred OHLCV rows is tens of kilobytes,
    so four of them never reach 1MB and an env-var version of this test would
    pass while evicting nothing. The env-var path is covered separately by
    test_defaults_match_the_spec and test_out_of_range_size_cap_falls_back.
    Sized from the LARGEST entry present, not the mean: every entry holds the
    same 200 rows, but the sidecars differ by a byte or two (the repr of
    `fetched_at` varies in length), and a mean-based cap can sit one byte
    under the two survivors it means to keep.
    """
    per_entry = max(
        path.stat().st_size + path.with_suffix(".json").stat().st_size
        for path in cache_dir.glob("*.parquet")
    )
    cap = per_entry * keep_entries
    monkeypatch.setattr(bar_cache, "max_bytes", lambda: cap)
    return cap


def _stamp(cache_dir, symbol, offset_seconds):
    """Give one entry a definite mtime, `offset_seconds` from now."""
    stamp = time.time() + offset_seconds
    for path in bar_cache.entry_paths(symbol, **KEY):
        os.utime(path, (stamp, stamp))


def test_the_size_cap_evicts_the_least_recently_used_entries(cache_dir, monkeypatch):
    """Write everything FIRST, then order the mtimes, then shrink the cap.

    Filesystem mtimes are coarse (jiffy granularity on Linux), so entries
    written within one tick tie and "least recently used" is undefined among
    them. Setting the stamps after all the writes, and only then running the
    pass, is what makes the expected survivors deterministic.
    """
    for index in range(4):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    for index in range(4):
        _stamp(cache_dir, f"S{index}", -(10 - index))  # S0 oldest, S3 newest
    cap = _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    assert bar_cache.enforce_size_cap() == 2
    total = sum(path.stat().st_size for path in cache_dir.iterdir())
    assert total <= cap
    hits, _ = bar_cache.read_many(["S0", "S1", "S2", "S3"], **KEY)
    assert set(hits) == {"S2", "S3"}


def test_a_write_never_evicts_its_own_batch(cache_dir, monkeypatch):
    """MUTATION TEST: drop the `protect=` argument from write_many's eviction
    call and this must fail. Under cap pressure the trailing pass would
    otherwise discard the symbols just fetched -- a paid fetch that never
    becomes a hit -- and with tying mtimes it did exactly that."""
    bar_cache.write_many({"OLD": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    # Make the resident entry look NEWER than anything written next, so pure
    # LRU order would pick the fresh batch as the victim.
    _stamp(cache_dir, "OLD", +100)
    _tiny_cap(monkeypatch, cache_dir, keep_entries=1)
    bar_cache.write_many({"NEW": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    hits, _ = bar_cache.read_many(["OLD", "NEW"], **KEY)
    assert set(hits) == {"NEW"}


def test_eviction_leaves_no_half_entries(cache_dir, monkeypatch):
    for index in range(6):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    bar_cache.enforce_size_cap()
    parquets = list(cache_dir.glob("*.parquet"))
    assert parquets, "eviction removed everything"
    for path in parquets:
        assert path.with_suffix(".json").exists()
    for path in cache_dir.glob("*.json"):
        assert path.with_suffix(".parquet").exists()


def test_stale_temp_files_are_swept(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    orphan = cache_dir / "AAPL-deadbeef.parquet7xk.tmp"
    orphan.write_bytes(b"crashed writer")
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(orphan, (stamp, stamp))
    bar_cache.enforce_size_cap()
    assert not orphan.exists()


def test_a_stale_orphan_sidecar_is_swept(cache_dir):
    """A crashed writer's sidecar matches no *.parquet glob, so the LRU pass
    would never reclaim it; the stray sweep does, once it is past the grace."""
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    stamp = time.time() - 2 * bar_cache._STRAY_GRACE_SECONDS
    os.utime(meta_path, (stamp, stamp))
    bar_cache.enforce_size_cap()
    assert not meta_path.exists()


def test_a_fresh_orphan_sidecar_survives_the_sweep(cache_dir):
    bar_cache.write_many({"AAPL": _frame()}, last_fetch=LAST_FETCH, **KEY)
    parquet_path, meta_path = bar_cache.entry_paths("AAPL", **KEY)
    parquet_path.unlink()
    bar_cache.enforce_size_cap()
    assert meta_path.exists()


def test_a_fresh_temp_file_is_left_alone(cache_dir):
    """A concurrent writer's in-flight temp file must survive another
    process's eviction pass."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    in_flight = cache_dir / "AAPL-deadbeef.parquetab1.tmp"
    in_flight.write_bytes(b"in flight")
    bar_cache.enforce_size_cap()
    assert in_flight.exists()


def test_write_many_enforces_the_cap(cache_dir, monkeypatch):
    """Eviction is not a separate chore someone has to remember to run."""
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    cap = _tiny_cap(monkeypatch, cache_dir, keep_entries=2)
    for index in range(1, 6):
        bar_cache.write_many(
            {f"S{index}": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY
        )
    total = sum(path.stat().st_size for path in cache_dir.iterdir())
    assert total <= cap


def test_write_many_skips_the_scan_when_it_cannot_have_crossed_the_cap(
    cache_dir, monkeypatch
):
    """The full directory scan is O(entries) and the cap holds thousands of
    them; paying it on every fetch far under cap is the wrong trade. The first
    write scans (nothing is known yet); the second, with the running estimate
    well under cap and the scan fresh, must not."""
    scans = []
    real = bar_cache.enforce_size_cap
    monkeypatch.setattr(
        bar_cache, "enforce_size_cap", lambda **kw: scans.append(1) or real(**kw)
    )
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 1
    bar_cache.write_many({"S1": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 1


def test_write_many_rescans_once_the_interval_has_elapsed(cache_dir, monkeypatch):
    """Other processes' writes are invisible to the estimate, so the scan
    also runs on a clock, not only on this process's bytes."""
    scans = []
    real = bar_cache.enforce_size_cap
    monkeypatch.setattr(
        bar_cache, "enforce_size_cap", lambda **kw: scans.append(1) or real(**kw)
    )
    bar_cache.write_many({"S0": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    state = bar_cache._scan_state[str(cache_dir)]
    state[0] -= 2 * bar_cache._SWEEP_INTERVAL_SECONDS
    bar_cache.write_many({"S1": _frame(rows=200)}, last_fetch=LAST_FETCH, **KEY)
    assert len(scans) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q -k "clamped or settled or enforce or evict or temp or orphan or scan or own_batch"`
Expected: FAIL — the eviction, sweep, scan and own-batch tests fail with `AttributeError: … has no attribute 'enforce_size_cap'` (or `_scan_state`). The refusal tests written in Task 2's body already pass; they are listed here because this is the task that mutation-tests them (Step 5).

- [ ] **Step 3: Implement eviction and call it from `write_many`**

Append to `dashboard/backend/infrastructure/market_data/bar_cache.py`:

```python
# --- eviction --------------------------------------------------------------


#: Per directory: ``[last_full_scan_at, bytes_on_disk_at_that_scan,
#: bytes_this_process_wrote_since]``. Process-local on purpose -- a shared
#: index file would be a third thing to keep atomic -- which is why the time
#: bound in ``_maybe_enforce_size_cap`` exists.
_scan_state: Dict[str, List[float]] = {}


def enforce_size_cap(*, protect: Iterable[Path] = ()) -> int:
    """One directory scan: sweep stale strays, then evict LRU until under cap.

    The cap is a runaway bound, not a working-set estimate: one symbol over a
    7-weekday window at 5m bars is tens of kilobytes of parquet, so the 256MB
    default holds thousands of symbol-windows. It exists so arbitrary user
    windows cannot grow the cache without bound on an ephemeral disk.

    ``protect`` is the batch the caller just wrote. It is never evicted, even
    if that leaves the directory over cap until the next pass: an eviction
    that discards the symbols just fetched turns a paid fetch into a miss, and
    with coarse mtimes (entries written within one tick tie) it did exactly
    that. Ordering among the rest is ``(mtime, name)`` so a tie still evicts
    deterministically.
    """
    directory = cache_dir()
    if not directory.exists():
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
                try:
                    stat = item.stat()
                except OSError:
                    continue
                sizes[Path(item.path)] = (stat.st_mtime, stat.st_size)
    except OSError:
        return 0
    # A crashed writer's leftovers -- a *.tmp mid-write, or a sidecar whose
    # parquet never landed -- match no entry and would never be reclaimed by
    # the LRU pass. A young one belongs to a live writer and is left alone.
    for path, (mtime, _size) in list(sizes.items()):
        is_tmp = path.suffix == ".tmp"
        is_orphan_meta = (
            path.suffix == ".json" and path.with_suffix(".parquet") not in sizes
        )
        if (is_tmp or is_orphan_meta) and now - mtime > _STRAY_GRACE_SECONDS:
            _discard(path)
            del sizes[path]
    entries: List[Tuple[float, str, int, Path, Path]] = []
    total = 0
    for path, (mtime, size) in sizes.items():
        if path.suffix != ".parquet":
            continue
        meta_path = path.with_suffix(".json")
        size += sizes.get(meta_path, (0.0, 0))[1]
        total += size
        entries.append((mtime, path.name, size, path, meta_path))
    _scan_state[str(directory)] = [now, float(total), 0.0]
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
    _scan_state[str(directory)][1] = float(total)
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
```

Then, at the end of `write_many`, replace `return written` with:

```python
    if written:
        try:
            _maybe_enforce_size_cap(directory, written_bytes, protect=written_paths)
        except Exception as exc:  # noqa: BLE001 - eviction must not fail a write
            print(f"📦 bar cache: eviction failed: {exc}", flush=True)
    return written
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation-test the three refusal guards and the two race guards**

Do this by hand and record the result; do not skip it. For each guard:

```bash
# 1. Comment out the `if end_clamped:` block in write_many.
pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py \
  -q -k "clamped_window_is_never_stored"
# Expected: FAIL. Restore the guard, re-run, expect PASS.

# 2. Comment out the `if sip_fallback_to_iex:` block in write_many.
pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py \
  -q -k "iex_fallback_window_is_never_stored"
# Expected: FAIL. Restore the guard, re-run, expect PASS.

# 3. Comment out the `if not window_is_settled(end):` block in write_many.
pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py \
  -q -k "has_not_settled_is_never_stored"
# Expected: FAIL. Restore the guard, re-run, expect PASS.

# 4. In read_many, replace `_discard_if_stale(now, parquet_path, meta_path)`
#    with `_discard(parquet_path, meta_path)`.
pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py \
  -q -k "fresh_half_entry"
# Expected: FAIL. Restore, re-run, expect PASS.

# 5. In write_many's eviction call, drop `protect=written_paths`.
pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py \
  -q -k "never_evicts_its_own_batch"
# Expected: FAIL. Restore, re-run, expect PASS.
```

Confirm `git diff` is clean of the mutations before committing (`git diff --stat` should show only the intended additions).

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/infrastructure/market_data/bar_cache.py \
        dashboard/backend/tests/infrastructure/market_data/test_bar_cache.py
git commit -m "feat: refuse to cache clamped or fallback bar windows"
```

---

## Task 4: Wire the cache into `fetch_bars`

**Files:**
- Modify: `dashboard/backend/infrastructure/market_data/alpaca_bars.py`
- Test: `dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py` (create)

**Interfaces:**
- Consumes: `bar_cache.enabled()`, `bar_cache.read_many()`, `bar_cache.write_many()`.
- Produces: `AlpacaDataLoader._fetch_bars_uncached(symbols, start, end)` — today's `fetch_bars` body verbatim, including the >100-symbol recursion, which now recurses into **itself** and not into `fetch_bars`. `fetch_bars` keeps its exact public signature and return type.

- [ ] **Step 1: Write the failing integration tests**

Create `dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py`:

```python
"""The on-disk bar cache, driven through the real AlpacaDataLoader.fetch_bars.

No network: the Alpaca SDK client is a fake, exactly as in test_alpaca_bars.py.
Every test points ATL_BAR_CACHE_DIR at tmp_path.
"""

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data import bar_cache
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
    AlpacaDataLoader,
)

CLIENT_TARGET = "alpaca.data.historical.StockHistoricalDataClient"
CLAMP_TARGET = (
    "dashboard.backend.infrastructure.market_data.alpaca_bars.clamp_end_for_sip"
)


def _bars_df(symbols, rows=2):
    frames = []
    for symbol in symbols:
        index = pd.MultiIndex.from_tuples(
            [
                (symbol, pd.Timestamp("2026-05-04T13:30:00Z") + pd.Timedelta(minutes=5 * i))
                for i in range(rows)
            ],
            names=["symbol", "timestamp"],
        )
        frames.append(
            pd.DataFrame(
                [
                    {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100}
                    for _ in range(rows)
                ],
                index=index,
            )
        )
    if not frames:
        # pd.concat([]) raises "No objects to concatenate", and the fixture's
        # initial state is exactly this.
        return pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]),
        )
    return pd.concat(frames)


@pytest.fixture
def fake_alpaca(monkeypatch):
    state = {"df": _bars_df([]), "exc": None, "requests": []}

    class _FakeBars:
        def __init__(self, df):
            self.df = df

    class _FakeSession:
        def request(self, *args, **kwargs):
            raise NotImplementedError("not exercised by these tests")

    class _FakeClient:
        def __init__(self, api_key, secret_key):
            self._session = _FakeSession()

        def get_stock_bars(self, request):
            state["requests"].append(request)
            if state["exc"] is not None:
                raise state["exc"]
            return _FakeBars(state["df"])

    monkeypatch.setattr(CLIENT_TARGET, _FakeClient)
    return state


@pytest.fixture
def cached_loader(fake_alpaca, tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "bar_cache"))
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    loader = AlpacaDataLoader()
    loader.configure_source_timeframe("5m")
    return loader


def _requested(state):
    return [list(request.symbol_or_symbols) for request in state["requests"]]


def test_a_second_identical_request_makes_no_alpaca_call(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL", "MSFT"])
    first = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 1
    second = cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 1  # served entirely from disk
    assert set(second) == set(first) == {"AAPL", "MSFT"}
    pd.testing.assert_frame_equal(second["AAPL"], first["AAPL"])


def test_a_mixed_request_fetches_only_the_missing_symbols(cached_loader, fake_alpaca):
    """This is what makes the DJIA_30 index baseline cheap after a Mag7 run:
    five of its thirty names are already on disk under the same window."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["df"] = _bars_df(["MSFT", "NVDA"])
    result = cached_loader.fetch_bars(
        ["AAPL", "MSFT", "NVDA"], "2026-05-04", "2026-05-12"
    )
    assert _requested(fake_alpaca) == [["AAPL"], ["MSFT", "NVDA"]]
    assert set(result) == {"AAPL", "MSFT", "NVDA"}


def test_a_cache_hit_restores_last_fetch(cached_loader, fake_alpaca):
    """market_data_store._build_dataset and engine.load_data read last_fetch to
    verify the source timeframe with evidence="fetch". A hit that leaves it
    stale silently downgrades that to the weaker evidence="configured" path."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.last_fetch = {"source_timeframe": "60m", "feed": "iex"}
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert cached_loader.last_fetch["source_timeframe"] == "5m"
    assert cached_loader.last_fetch["feed"] == "sip"
    assert cached_loader.last_fetch["sip_fallback_to_iex"] is False
    assert cached_loader.last_fetch["end_clamped"] is False


def test_a_cache_hit_restores_the_attrs_stamps(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    hit = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert hit["AAPL"].attrs[FRAME_ATTR_FEED] == "sip"
    assert hit["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is False
    assert hit["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is False


def test_changing_the_feed_misses(cached_loader, fake_alpaca, monkeypatch):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2


def test_changing_the_source_timeframe_misses(cached_loader, fake_alpaca):
    """source_timeframe is a mutable instance attribute set by
    configure_source_timeframe, not an argument -- it must be read at call
    time, and it materially changes the bars for an identical window."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.configure_source_timeframe("60m")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2


def test_a_clamped_response_is_never_written(cached_loader, fake_alpaca, monkeypatch):
    """MUTATION TEST, through the real path: remove the end_clamped guard in
    bar_cache.write_many and this must fail."""
    import datetime as _dt

    monkeypatch.setattr(
        CLAMP_TARGET,
        lambda end, **kwargs: _dt.datetime(2026, 5, 11, tzinfo=_dt.timezone.utc),
    )
    fake_alpaca["df"] = _bars_df(["AAPL"])
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert result["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is True
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_an_iex_fallback_response_is_never_written(cached_loader, fake_alpaca):
    """MUTATION TEST, through the real path: remove the sip_fallback_to_iex
    guard in bar_cache.write_many and this must fail."""
    calls = {"n": 0}
    real_df = _bars_df(["AAPL"])

    class _FallbackClient:
        def __init__(self, *args, **kwargs):
            self._session = None

        def get_stock_bars(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("your subscription does not permit this")

            class _Bars:
                df = real_df

            return _Bars()

    cached_loader.client = _FallbackClient()
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert result["AAPL"].attrs[FRAME_ATTR_SIP_FALLBACK] is True
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_a_window_ending_in_the_future_is_never_written_on_iex(
    cached_loader, fake_alpaca, monkeypatch
):
    """MUTATION TEST, through the real path: remove the window_is_settled
    guard in bar_cache.write_many and this must fail. `_effective_end` returns
    (end, False) for IEX -- no clamp, no flag -- so with the window still
    open nothing else stops a half-day frame being stored as complete."""
    from datetime import date, timedelta

    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    fake_alpaca["df"] = _bars_df(["AAPL"])
    result = cached_loader.fetch_bars(["AAPL"], "2026-05-04", tomorrow)
    assert result["AAPL"].attrs[FRAME_ATTR_END_CLAMPED] is False
    hits, _ = bar_cache.read_many(
        ["AAPL"],
        start="2026-05-04",
        end=tomorrow,
        source_timeframe="5m",
        feed="iex",
    )
    assert hits == {}


def test_a_failed_fetch_for_the_misses_fails_the_whole_request(
    cached_loader, fake_alpaca
):
    """Before the cache a request returned what Alpaca had or {}, and
    engine.load_data raises on {}. Five Dow names on disk plus Alpaca down for
    the other twenty-five must still be {} -- not a 5-symbol "Dow"."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["exc"] = RuntimeError("alpaca is down")
    assert cached_loader.fetch_bars(["AAPL", "MSFT"], "2026-05-04", "2026-05-12") == {}
    assert cached_loader.last_fetch is None


def test_a_symbol_alpaca_had_no_bars_for_still_returns_the_hits(
    cached_loader, fake_alpaca
):
    """The failure test keys on last_fetch being None, not on the fetch being
    empty: a request that succeeded but had nothing for the missing symbol
    answers with what is on disk, exactly as the old code answered with what
    Alpaca had."""
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    fake_alpaca["df"] = _bars_df([])
    result = cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    assert set(result) == {"AAPL"}
    assert cached_loader.last_fetch is not None


def test_an_iex_fallback_in_an_earlier_chunk_refuses_the_whole_batch(
    cached_loader, fake_alpaca
):
    """MUTATION TEST: derive the write_many flags from `self.last_fetch`
    instead of the frames and this must fail. `last_fetch` describes only the
    LAST 100-symbol chunk; here chunk one falls back to IEX and chunk two
    succeeds on SIP, so it reads sip_fallback_to_iex=False while 100 of the
    150 frames are IEX."""
    symbols = [f"S{i:03}" for i in range(150)]
    calls = {"n": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            self._session = None

        def get_stock_bars(self, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("your subscription does not permit this")

            class _Bars:
                df = _bars_df(list(request.symbol_or_symbols), rows=1)

            return _Bars()

    cached_loader.client = _Client()
    result = cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert len(result) == 150
    assert result["S000"].attrs[FRAME_ATTR_SIP_FALLBACK] is True
    assert result["S149"].attrs[FRAME_ATTR_SIP_FALLBACK] is False
    assert cached_loader.last_fetch["sip_fallback_to_iex"] is False  # the trap
    hits, _ = bar_cache.read_many(
        symbols,
        start="2026-05-04",
        end="2026-05-12",
        source_timeframe="5m",
        feed="sip",
    )
    assert hits == {}


def test_an_unconfigured_client_caches_nothing(cached_loader, tmp_path):
    cached_loader.client = None
    assert cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12") == {}
    assert cached_loader.last_fetch is None
    cache_dir = tmp_path / "bar_cache"
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


def test_a_symbol_alpaca_did_not_return_is_not_cached(cached_loader, fake_alpaca):
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    fake_alpaca["df"] = _bars_df(["NOPE"])
    result = cached_loader.fetch_bars(["AAPL", "NOPE"], "2026-05-04", "2026-05-12")
    # AAPL served from disk, NOPE re-requested because nothing was stored.
    assert _requested(fake_alpaca) == [["AAPL", "NOPE"], ["NOPE"]]
    assert set(result) == {"AAPL", "NOPE"}


def test_the_cache_resolves_above_the_hundred_symbol_recursion(
    cached_loader, fake_alpaca
):
    """Below the recursion, the cache would run once per 100-symbol chunk and
    the chunking -- not the cache -- would decide what is fetched. Above it,
    the recursion simply sees a shorter list."""
    symbols = [f"S{i:03}" for i in range(235)]
    fake_alpaca["df"] = _bars_df(symbols, rows=1)
    cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert [len(batch) for batch in _requested(fake_alpaca)] == [100, 100, 35]
    fake_alpaca["requests"].clear()
    fake_alpaca["df"] = _bars_df(symbols[:5], rows=1)
    cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert _requested(fake_alpaca) == []  # all 235 served from disk


def test_a_partially_warm_large_request_rebatches_only_the_misses(
    cached_loader, fake_alpaca
):
    symbols = [f"S{i:03}" for i in range(235)]
    fake_alpaca["df"] = _bars_df(symbols[:120], rows=1)
    cached_loader.fetch_bars(symbols[:120], "2026-05-04", "2026-05-12")
    fake_alpaca["requests"].clear()
    fake_alpaca["df"] = _bars_df(symbols[120:], rows=1)
    result = cached_loader.fetch_bars(symbols, "2026-05-04", "2026-05-12")
    assert [len(batch) for batch in _requested(fake_alpaca)] == [100, 15]
    assert len(result) == 235


def test_an_empty_symbol_list_does_not_take_the_all_hit_shortcut(
    cached_loader, fake_alpaca, tmp_path
):
    """With no symbols there are no misses either, so a shortcut keyed only on
    `not misses` would fire and `next(symbol for symbol in symbols ...)` would
    raise StopIteration straight out of fetch_bars. The `not symbols` guard
    ahead of it is what prevents that."""
    assert cached_loader.fetch_bars([], "2026-05-04", "2026-05-12") == {}
    cache_dir = tmp_path / "bar_cache"
    assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


def test_a_disabled_cache_restores_the_old_behaviour(
    cached_loader, fake_alpaca, monkeypatch
):
    monkeypatch.setenv("ATL_BAR_CACHE", "0")
    fake_alpaca["df"] = _bars_df(["AAPL"])
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    cached_loader.fetch_bars(["AAPL"], "2026-05-04", "2026-05-12")
    assert len(fake_alpaca["requests"]) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py -q`
Expected: FAIL — every caching test fails with two Alpaca requests where one was expected, because `fetch_bars` has no cache.

- [ ] **Step 3: Split `fetch_bars` into a wrapper and `_fetch_bars_uncached`**

In `dashboard/backend/infrastructure/market_data/alpaca_bars.py`:

1. Add the import, after the existing `from dashboard.backend.infrastructure.market_data.frequency import (...)` block:

```python
from dashboard.backend.infrastructure.market_data import bar_cache
```

2. Rename the existing `def fetch_bars(self, symbols, start, end)` to `def _fetch_bars_uncached(self, symbols, start, end)`, keeping its body **verbatim** except for two edits:
   - the recursive call inside the `len(symbols) > 100` branch becomes `self._fetch_bars_uncached(...)`, not `self.fetch_bars(...)`;
   - its docstring gains the note below.

```python
    def _fetch_bars_uncached(
        self, symbols: List[str], start: str, end: str
    ) -> Dict[str, pd.DataFrame]:
        """Today's fetch, unchanged: batch, request, stamp, record.

        Called only by :meth:`fetch_bars`, which has already removed every
        symbol the on-disk cache could serve. The >100 recursion therefore
        recurses into THIS method, never back into the wrapper -- otherwise the
        cache would resolve once per chunk.
        """
        # A full catalog can contain thousands of tickers. Bound URL length and
        # response size per request while preserving every selected symbol.
        if len(symbols) > 100:
            data = {}
            for offset in range(0, len(symbols), 100):
                data.update(
                    self._fetch_bars_uncached(symbols[offset:offset + 100], start, end)
                )
            return data
        if not self.client:
            print("⚠️ Alpaca not configured — skipping bar fetch")
            self.last_fetch = None
            return {}
        # ... rest of the existing body unchanged ...
```

3. Add the new wrapper immediately above `_fetch_bars_uncached`:

```python
    def fetch_bars(
        self, symbols: List[str], start: str, end: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data at ``source_timeframe``, serving what the on-disk bar
        cache already holds and requesting only the rest.

        Args:
            symbols: List of stock symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            {symbol: DataFrame with timestamp, open, high, low, close, volume}

        The cache is resolved HERE, above the >100-symbol batch recursion in
        :meth:`_fetch_bars_uncached`. Below it, the cache would run once per
        100-symbol chunk and the chunking -- not the cache -- would decide what
        gets fetched. Above it, the recursion just sees a shorter list.

        Keying per symbol rather than per request is what makes the index
        baseline cheap: after a Mag7 run the DJIA_30 fetch (same window, same
        timeframe, same feed) finds five of its thirty names on disk. A
        per-request key would miss that entirely, because the symbol lists
        differ.
        """
        symbols = list(symbols)
        # Hoisted above the batch recursion. With no client every chunk
        # returned {} anyway, so the result is identical; the warning now
        # prints once instead of once per chunk. Kept ahead of the cache so an
        # unconfigured loader never reads or writes an entry.
        if not self.client:
            print("⚠️ Alpaca not configured — skipping bar fetch")
            self.last_fetch = None
            return {}
        if not symbols or not bar_cache.enabled():
            return self._fetch_bars_uncached(symbols, start, end)

        # `configured_feed_name`, not `_resolve_data_feed`: the key needs the
        # name, not the SDK enum, and both raise AlpacaFeedConfigError on a
        # typo'd feed at the same point in the call as before.
        key = {
            "start": str(start),
            "end": str(end),
            # A mutable instance attribute set by `configure_source_timeframe`,
            # so it must be read at call time.
            "source_timeframe": self.source_timeframe,
            "feed": configured_feed_name(),
        }
        hits, metas = bar_cache.read_many(symbols, **key)
        misses = [symbol for symbol in symbols if symbol not in hits]
        if hits:
            print(
                f"📦 bar cache: {len(hits)}/{len(symbols)} symbols on disk, "
                f"fetching {len(misses)}"
            )
        if not misses:
            # No live fetch happened, so `last_fetch` would still describe some
            # earlier request. `load_data` and `market_data_store._build_dataset`
            # read it to verify the source timeframe with evidence="fetch";
            # leaving it stale silently downgrades that to evidence="configured".
            # Any hit's sidecar will do -- for one key they are identical, since
            # every field is either a key component or a flag the cache refuses
            # to store.
            first = next(symbol for symbol in symbols if symbol in metas)
            self.last_fetch = dict(metas[first])
            return {symbol: hits[symbol] for symbol in symbols if symbol in hits}

        fetched = self._fetch_bars_uncached(misses, start, end)
        if not fetched and self.last_fetch is None:
            # Every failure exit of `_fetch_bars_uncached` clears `last_fetch`
            # and returns {}; a request that merely had no bars for a symbol
            # leaves `last_fetch` set. Before the cache a call either returned
            # what Alpaca had or {} -- and `engine.load_data` raises on {}.
            # Returning the hits alone here would let a DJIA_30 run after a
            # Mag7 run proceed on the five names already on disk: a 5-symbol
            # "Dow", an index baseline priced off five names, and frequency
            # verification quietly downgraded to evidence="configured".
            # (For a >100-symbol call `last_fetch` is the last chunk's, so a
            # failed final chunk with earlier chunks intact still merges --
            # exactly the pre-cache behaviour.)
            return {}
        if fetched:
            # The refusal flags come from the FRAMES, not from `last_fetch`.
            # `last_fetch` describes the last request the loader made, which
            # for a >100-symbol call is only the last 100-symbol chunk: with
            # chunk one on IEX fallback and chunk two on SIP it reads
            # sip_fallback_to_iex=False, and the IEX frames would be stored
            # under the SIP key for the TTL. The stamps are per frame and
            # cover every chunk. Whole-batch (`any`) because the cache's rule
            # is whole-batch: a tape mix is wrong for the batch, not a subset.
            bar_cache.write_many(
                fetched,
                last_fetch=self.last_fetch,
                sip_fallback_to_iex=any(
                    bool(frame.attrs.get(FRAME_ATTR_SIP_FALLBACK))
                    for frame in fetched.values()
                ),
                end_clamped=any(
                    bool(frame.attrs.get(FRAME_ATTR_END_CLAMPED))
                    for frame in fetched.values()
                ),
                **key,
            )
        merged: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = fetched.get(symbol)
            if frame is None:
                frame = hits.get(symbol)
            if frame is not None:
                merged[symbol] = frame
        return merged
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing market-data suite to prove nothing moved**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/ dashboard/backend/tests/test_market_data_store.py dashboard/backend/tests/test_market_data_sharing.py -q`
Expected: PASS, same counts as before. In particular `test_full_catalog_batches_every_symbol_without_a_thirty_name_limit` must still see `[100, 100, 35]` — it runs with the cache off via conftest.

- [ ] **Step 6: Mutation-test the guards through the real path**

```bash
# Comment out the `if end_clamped:` block in bar_cache.write_many:
pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py \
  -q -k "clamped_response_is_never_written"   # expect FAIL, then restore and expect PASS

# Comment out the `if sip_fallback_to_iex:` block:
pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py \
  -q -k "iex_fallback_response_is_never_written"   # expect FAIL, then restore and expect PASS

# Comment out the `if not window_is_settled(end):` block:
pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py \
  -q -k "ending_in_the_future_is_never_written"   # expect FAIL, then restore and expect PASS

# In the fetch_bars wrapper, replace both `any(... frame.attrs ...)` arguments
# with `bool((self.last_fetch or {}).get(<flag>))`:
pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py \
  -q -k "earlier_chunk_refuses_the_whole_batch"   # expect FAIL, then restore and expect PASS

# Delete the `if not fetched and self.last_fetch is None: return {}` block:
pytest dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py \
  -q -k "fails_the_whole_request"   # expect FAIL, then restore and expect PASS
```

- [ ] **Step 7: Run the whole suite, verify the seed DB, and commit**

```bash
pytest dashboard/backend/tests/ -q
stat -c '%s' dashboard/storage/data/backtest.db   # must print 688128
git add dashboard/backend/infrastructure/market_data/alpaca_bars.py \
        dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars_cache.py
git commit -m "feat: serve repeated bar windows from the on-disk cache"
```

---

## Task 5: Split `loading_bars` into fetch and post-fetch

This is the measurement the whole design is a bet on: with the cache warm, the residual is the aggregation half, directly, with no synthetic benchmark and no subtraction.

**Files:**
- Modify: `dashboard/backend/domain/backtesting/engine.py`
- Test: `dashboard/backend/tests/backtesting/test_engine_progress_phases.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `HourlyBacktester.record_phase_metric(key: str, value: float) -> None` — attaches a number to the phase that is currently open; it lands in that phase's `phases[]` record when the phase closes.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/backtesting/test_engine_progress_phases.py`. That file already imports `HourlyBacktester` and builds instances with `object.__new__(HourlyBacktester)`; these cases need no progress file, because `publish_phase` reads it with `getattr(self, "progress_file", None)` and returns early when it is absent. Add no imports:

```python
def test_record_phase_metric_lands_on_the_phase_that_was_open():
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=100.0)
    engine.record_phase_metric("fetch_seconds", 12.5)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 3.25)
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert by_name["starting"]["fetch_seconds"] == 12.5
    assert by_name["loading_bars"]["fetch_seconds"] == 3.25


def test_phase_extras_do_not_leak_into_the_next_phase():
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 1.0)
    engine.publish_phase("indicators")
    engine.publish_phase("first_decision")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert "fetch_seconds" in by_name["loading_bars"]
    assert "fetch_seconds" not in by_name["indicators"]


def test_record_phase_metric_tolerates_an_uninitialised_instance():
    """Every accessor in this block tolerates an instance built with __new__
    that never ran _init_progress_phases."""
    engine = object.__new__(HourlyBacktester)
    engine.record_phase_metric("fetch_seconds", 2.0)
    engine.publish_phase("loading_bars")  # must not raise


def test_a_metric_recorded_with_no_phase_open_lands_nowhere():
    """MUTATION TEST: drop the `_progress_phase is None` gate in
    record_phase_metric and this must fail. With no launch time nothing is
    open before the first publish_phase; a number recorded then has no owner,
    and generalising the extras merge would otherwise hand it to the first
    phase that closes -- `loading_bars`, which did not incur it."""
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=None)
    engine.record_phase_metric("fetch_seconds", 9.0)
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert "starting" not in by_name
    assert "fetch_seconds" not in by_name["loading_bars"]


def test_a_startup_clock_without_a_launch_time_does_not_leak_onto_loading_bars():
    """MUTATION TEST: seed `_progress_phase_extra` from `startup_clock`
    unconditionally in _init_progress_phases and this must fail.
    backtest_hourly_agent.py always passes startup_clock but launched_at only
    from --launched-at, which a bare CLI run omits. Today those three keys are
    simply dropped (`starting` never closes); once extras belong to whichever
    phase closes, they would surface on `loading_bars` as if it had a spawn
    time and a schema DDL cost."""
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(
        launched_at=None,
        startup_clock={
            "child_entered_at": 1.0,
            "imports_done_at": 3.0,
            "schema_init_seconds": 0.0,
        },
    )
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert set(by_name) == {"loading_bars"}
    assert not {"child_entered_at", "imports_done_at", "schema_init_seconds"} & set(
        by_name["loading_bars"]
    )


def test_the_starting_breakdown_line_is_unchanged(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(
        launched_at=0.0,
        startup_clock={
            "child_entered_at": 1.0,
            "imports_done_at": 3.0,
            "schema_init_seconds": 0.0,
        },
    )
    engine.publish_phase("loading_bars")
    out = capsys.readouterr().out
    assert "spawn+interpreter 1.00s" in out
    assert "imports+stores 2.00s" in out
    assert "schema DDL 0.00s" in out


def test_closing_loading_bars_prints_the_fetch_split(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 0.0)
    engine.publish_phase("indicators")
    out = capsys.readouterr().out
    assert "fetch 0.00s" in out
    assert "aggregate+verify" in out


def test_no_split_line_without_the_metric(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    assert "aggregate+verify" not in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py -q`
Expected: FAIL with `AttributeError: 'HourlyBacktester' object has no attribute 'record_phase_metric'`.

- [ ] **Step 3: Generalise the extras mechanism**

In `dashboard/backend/domain/backtesting/engine.py`, inside `_set_progress_phase`, replace:

```python
            if self._progress_phase == "starting":
                finished.update(getattr(self, "_progress_phase_extra", {}))
            self._progress_phases.append(finished)
```

with:

```python
            # Extras belong to whichever phase was open, not to `starting`
            # alone. `_init_progress_phases` seeds them for `starting` -- and
            # ONLY when it actually opens that phase, see below;
            # `record_phase_metric` adds them for any later phase, and only
            # while one is open. Cleared on every transition so a number can
            # never be reported against the wrong phase.
            finished.update(getattr(self, "_progress_phase_extra", {}))
            self._progress_phase_extra = {}
            self._progress_phases.append(finished)
```

Then in `_init_progress_phases`, replace:

```python
        extra: Dict = {}
        for key in ("child_entered_at", "imports_done_at", "schema_init_seconds"):
            value = (startup_clock or {}).get(key)
            if value is not None:
                extra[key] = float(value)
        self._progress_phase_extra: Dict = extra
```

with:

```python
        extra: Dict = {}
        # Seed the startup clock only when `starting` is actually open.
        # backtest_hourly_agent.py always passes `startup_clock` but
        # `launched_at` only from --launched-at, which a bare CLI run omits;
        # today the orphaned keys are dropped because `starting` never closes,
        # and once extras merge into whichever phase closes they would
        # otherwise land on `loading_bars` -- a spawn time and a DDL cost on
        # the phase that did neither.
        if launched_at is not None:
            for key in ("child_entered_at", "imports_done_at", "schema_init_seconds"):
                value = (startup_clock or {}).get(key)
                if value is not None:
                    extra[key] = float(value)
        self._progress_phase_extra: Dict = extra
```

Then, immediately after the existing `if finished["name"] == "starting" and {...}` print block, add:

```python
            elif finished["name"] == "loading_bars" and "fetch_seconds" in finished:
                # The design's measurement gate: with a warm cache the residual
                # here IS the aggregation cost, directly -- no synthetic
                # benchmark, no subtraction. It decides whether caching the
                # AGGREGATED output is worth a second change or whether the
                # fetch was the whole story. `elapsed` is the phase total,
                # already computed above with the `is None` guard that keeps a
                # 0.0 start time from reading as a phase that cost nothing.
                fetch_seconds = float(finished["fetch_seconds"])
                print(
                    f"     fetch {fetch_seconds:.2f}s"
                    f" | aggregate+verify {elapsed - fetch_seconds:.2f}s",
                    flush=True,
                )
```

- [ ] **Step 4: Add `record_phase_metric`**

Add it directly after `_progress_phase_fields` in the same "Progress phases" block:

```python
    def record_phase_metric(self, key: str, value: float) -> None:
        """Attach a number to the phase that is currently open.

        The phase *name* stays one word -- the card has nothing useful to say
        about fetch versus aggregate, and a name the status route must
        translate is a name the frontend must learn. But one undifferentiated
        number cannot justify an optimisation either, which is the same
        argument `_init_progress_phases` makes for splitting `starting` into
        four. So the record is split even though the phase is not.

        Tolerates an instance built with `__new__` that never ran
        `_init_progress_phases`, like every other accessor here.

        With no phase open the number has no owner and is dropped: the
        alternative is handing it to whichever phase closes first, which is
        the same misattribution `_init_progress_phases` guards against for
        the startup clock.
        """
        if not hasattr(self, "_progress_phase_extra"):
            self._init_progress_phases()
        if self._progress_phase is None:
            return
        self._progress_phase_extra[str(key)] = float(value)
```

- [ ] **Step 5: Record the fetch time in `load_data`**

In `load_data`, replace:

```python
        self.source_data = self.data_loader.fetch_bars(
            symbols, self.start_date, self.end_date
        )
```

with:

```python
        fetch_started_at = wall_clock()
        self.source_data = self.data_loader.fetch_bars(
            symbols, self.start_date, self.end_date
        )
        # Everything after this point in `loading_bars` -- the frequency
        # verification and, in intraday mode, `aggregate_bars_by_symbol` -- is
        # the half the phase name hides. Recorded here so the number survives
        # in `phases[]` as well as on stdout.
        self.record_phase_metric("fetch_seconds", wall_clock() - fetch_started_at)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/backtesting/test_engine_progress_phases.py -q`
Expected: PASS.

- [ ] **Step 7: Run every phase-aware test**

Run: `pytest dashboard/backend/tests/backtesting/ dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_backtest_progress_status.py dashboard/backend/tests/test_backtest_progress_format.py dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_backtest_attach_surface.py -q`
Expected: PASS. The `phases[]` records now carry an extra key on `loading_bars`; `starting` already carried three, so extra keys are tolerated by the status route.

- [ ] **Step 8: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py \
        dashboard/backend/tests/backtesting/test_engine_progress_phases.py
git commit -m "feat: split loading_bars into fetch and aggregate"
```

---

## Task 6: Warm the default windows on boot

Without this a cold instance charges the first visitor full price, which defeats the stated audience — prod users on the live dashboard.

**Files:**
- Create: `dashboard/backend/infrastructure/market_data/bar_cache_warm.py`
- Modify: `dashboard/backend/app.py`
- Test: `dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py` (create)

**Interfaces:**
- Consumes: `bar_cache.enabled()`, `bar_cache.warm_enabled()`, `AlpacaDataLoader`.
- Produces: `warm_windows() -> List[Tuple[List[str], str, str]]`, `warm_bar_cache() -> int`, and the constants `ROUTE_DEFAULT_START = "2026-05-01"`, `ROUTE_DEFAULT_END = "2026-05-07"`, `WARM_SOURCE_TIMEFRAME = "5m"`.

> **Three windows, per spec §8 (resolved 2026-09-21).** The middle one is `DJIA_30` over the `defaults.json` window: spec §1-A establishes that every default Mag7 run *also* fetches the full Dow over the same window for the index baseline, on the ordinary path, after `publish_phase("saving")`, and `engine.py`'s index-baseline block passes `self.start_date`/`self.end_date` verbatim, so it is the same key. It costs one extra batched call per deploy (25 symbols, since the five Mag7 overlaps are already hits) and removes the whole uncounted tail from the default run. The count is asserted as `len(warm_windows())`, never as a literal, so adding or removing a window is a one-place change.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py`:

```python
"""Boot-time bar cache warm: which windows, and never from a test."""

import json
import os
import re

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.infrastructure.market_data import bar_cache, bar_cache_warm
from dashboard.backend.paths import BACKEND_DIR, CONFIG_DIR


def _defaults():
    return json.loads((CONFIG_DIR / "defaults.json").read_text(encoding="utf-8"))


def test_the_suite_never_warms():
    """conftest sets ATL_BAR_CACHE_WARM=0 at import time. Without it,
    importing the app would make live Alpaca calls -- a network dependency in
    an offline suite, and real money."""
    assert os.environ.get("ATL_BAR_CACHE_WARM") == "0"
    assert bar_cache.warm_enabled() is False


def test_warm_bar_cache_is_a_no_op_when_disabled(monkeypatch):
    def _explode():
        raise AssertionError("must not construct a loader when warm is off")

    monkeypatch.setattr(bar_cache_warm, "AlpacaDataLoader", _explode)
    assert bar_cache_warm.warm_bar_cache() == 0


def test_the_first_window_is_the_onboarding_modal():
    settings = _defaults()["defaultSettings"]
    symbols, start, end = bar_cache_warm.warm_windows()[0]
    assert symbols == [s.upper() for s in settings["assetList"]]
    assert (start, end) == (settings["startDate"], settings["endDate"])


def test_the_second_window_is_the_index_baseline_over_the_same_dates():
    """Every default run also fetches the full Dow for the index baseline,
    over the SAME window (engine.py passes start_date/end_date verbatim)."""
    settings = _defaults()["defaultSettings"]
    symbols, start, end = bar_cache_warm.warm_windows()[1]
    assert symbols == list(DJIA_30)
    assert (start, end) == (settings["startDate"], settings["endDate"])


def test_the_third_window_is_the_bare_post_default():
    symbols, start, end = bar_cache_warm.warm_windows()[2]
    assert symbols == list(DJIA_30)
    assert (start, end) == (
        bar_cache_warm.ROUTE_DEFAULT_START,
        bar_cache_warm.ROUTE_DEFAULT_END,
    )


def test_route_defaults_match_the_route_signature():
    """SOURCE-SHAPE GUARD. The two dates are inline literals in
    `run_backtest_endpoint`'s signature; importing them here would point
    infrastructure at api. This asserts the copies agree so the warm cannot
    silently warm a window nobody requests."""
    source = (BACKEND_DIR / "api" / "routers" / "backtests.py").read_text(
        encoding="utf-8"
    )
    start = re.search(r'start_date:\s*str\s*=\s*"([\d-]+)"', source)
    end = re.search(r'end_date:\s*str\s*=\s*"([\d-]+)"', source)
    assert start and end, "run_backtest_endpoint's date defaults moved"
    assert start.group(1) == bar_cache_warm.ROUTE_DEFAULT_START
    assert end.group(1) == bar_cache_warm.ROUTE_DEFAULT_END


def test_warm_fetches_every_window_at_the_intraday_source_timeframe(monkeypatch):
    """The key includes source_timeframe, so warming at the wrong resolution
    warms nothing a real run can use. The default US profile is 5m -> 60m."""
    calls = []

    class _FakeLoader:
        def __init__(self):
            self.source_timeframe = "60m"

        def configure_source_timeframe(self, value):
            self.source_timeframe = value

        def fetch_bars(self, symbols, start, end):
            calls.append((self.source_timeframe, tuple(symbols), start, end))
            return {symbol: object() for symbol in symbols}

    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "1")
    monkeypatch.setattr(bar_cache_warm, "AlpacaDataLoader", _FakeLoader)
    warmed = bar_cache_warm.warm_bar_cache()
    # Derived, not a literal 3: the window list is the one owner of the count.
    assert len(calls) == len(bar_cache_warm.warm_windows())
    assert {timeframe for timeframe, *_ in calls} == {"5m"}
    assert warmed == sum(len(symbols) for _, symbols, _, _ in calls)


def test_a_failing_window_does_not_stop_the_others(monkeypatch, capsys):
    class _FlakyLoader:
        def __init__(self):
            self.calls = 0

        def configure_source_timeframe(self, value):
            pass

        def fetch_bars(self, symbols, start, end):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("alpaca is down")
            return {symbol: object() for symbol in symbols}

    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "1")
    monkeypatch.setattr(bar_cache_warm, "AlpacaDataLoader", _FlakyLoader)
    assert bar_cache_warm.warm_bar_cache() > 0
    assert "failed" in capsys.readouterr().out


def test_unconfigured_credentials_skip_the_warm_without_raising(monkeypatch, capsys):
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        MarketDataUnavailableError,
    )

    def _no_credentials():
        raise MarketDataUnavailableError("Alpaca credentials not found")

    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "1")
    monkeypatch.setattr(bar_cache_warm, "AlpacaDataLoader", _no_credentials)
    assert bar_cache_warm.warm_bar_cache() == 0
    assert "skipped" in capsys.readouterr().out


def test_app_starts_the_warm_on_a_daemon_thread():
    """SOURCE-SHAPE GUARD: a warm on the request path, or a blocking one,
    would delay boot and could fail the health check."""
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    # The whole call, not `daemon=True` on its own: app.py already starts
    # three other daemon threads, so a bare substring check would pass with
    # the warm thread missing entirely.
    assert re.search(
        r"Thread\(\s*target=warm_bar_cache_background,\s*daemon=True\s*\)", source
    )
    assert "bar_cache.describe()" in source
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py -q`
Expected: collection error — `ModuleNotFoundError: No module named '…bar_cache_warm'`.

- [ ] **Step 3: Write the warm module**

Create `dashboard/backend/infrastructure/market_data/bar_cache_warm.py`:

```python
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
```

- [ ] **Step 4: Wire the startup hook**

In `dashboard/backend/app.py`'s `startup_event`, directly after the `threading.Thread(target=init_daily_leaderboard, daemon=True).start()` line, add:

```python
    # On-disk bar cache: name the state at boot, matching the
    # `<store> backend: …` convention, then warm the default windows on a
    # daemon thread so a cold instance does not charge the first visitor the
    # full bar fetch. Non-blocking by construction: it must never delay boot
    # or fail the health check.
    from dashboard.backend.infrastructure.market_data import bar_cache

    print(bar_cache.describe())

    def warm_bar_cache_background():
        """Background: pre-fetch the default backtest windows."""
        try:
            from dashboard.backend.infrastructure.market_data.bar_cache_warm import (
                warm_bar_cache,
            )

            warm_bar_cache()
        except Exception as e:  # noqa: BLE001 - a cold cache is the status quo
            print(f"⚠️ Bar cache warm error: {e}")

    threading.Thread(target=warm_bar_cache_background, daemon=True).start()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py -q`
Expected: PASS.

- [ ] **Step 6: Prove the suite makes no live call**

`test_the_suite_never_warms` is the guard that holds in CI. Confirm it by eye as well — with **`-s`**, because pytest's capture hides stdout on a passing test, so a grep over a captured run prints 0 whether or not a warm ran. The only legitimate `📦 bar cache warm:` lines come from `test_bar_cache_warm.py`'s fake-loader cases, so exclude that file:

```bash
pytest dashboard/backend/tests/ -q -s \
  --ignore=dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py \
  2>&1 | grep -c "bar cache warm:"   # must print 0
```

- [ ] **Step 7: Commit**

```bash
stat -c '%s' dashboard/storage/data/backtest.db   # must print 688128
git add dashboard/backend/infrastructure/market_data/bar_cache_warm.py \
        dashboard/backend/app.py \
        dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py
git commit -m "feat: warm the default bar windows on boot"
```

---

## Task 7: Document the four environment variables

Every environment variable in this backend is documented in `CLAUDE.md`, with the reasoning, not just the name. A cache that is on by default in production and off in the suite is exactly the kind of asymmetry that has to be written down.

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the finished behaviour of Tasks 1–6.
- Produces: nothing code depends on.

- [ ] **Step 1: Add the environment bullet**

In `CLAUDE.md`'s "Environment & credentials" list, immediately after the `ALPACA_SIP_DELAY_MINUTES` / `ALPACA_ALLOW_RECENT_SIP` bullet, insert:

```markdown
- `ATL_BAR_CACHE` (optional, **default ON** when unset; on for `1`/`true`/`yes`/`on`, off for anything else — a recognised `0`/`false`/`no`/`off` silently, junk with a `WARNING`, because the only reason to set a kill switch is to turn it off and a typo'd one must not stay on) plus `ATL_BAR_CACHE_DIR`, `ATL_BAR_CACHE_MAX_MB` (default **256**, range 1–16384) and `ATL_BAR_CACHE_TTL_DAYS` (default **7**, range 1–365): the cross-process on-disk bar cache in `infrastructure/market_data/bar_cache.py`, applied inside `AlpacaDataLoader.fetch_bars`. One parquet entry per symbol under `dashboard/storage/data/bar_cache/`, keyed on `(symbol, start, end, source_timeframe, resolved_feed, SCHEMA_VERSION)`. It exists because a dashboard backtest is a **subprocess** — the in-process `market_data_store` `OrderedDict` is empty on every run, so a file on the instance's disk is the only cache a child and its parent can share. It does not need to survive a redeploy (`disk: null`), only a `Popen`. **The cache is resolved above the `len(symbols) > 100` batch recursion**: below it, the chunking rather than the cache would decide what is fetched. Keying per symbol rather than per request is what makes the index baseline cheap — after a Mag7 run the `DJIA_30` baseline fetch finds five of its thirty names already on disk under the same window. ⚠ **What it refuses to store is the whole safety argument**: a frame from a clamped SIP window (`ALPACA_SIP_DELAY_MINUTES` — the *same* requested window returns a shorter frame depending on when you ask, so caching it makes the truncation permanent), a frame from the IEX-on-refusal retry (which re-requests with the original unclamped `end` and never sets `end_clamped`, so it looks pristine while being ~2.5% of the volume the key claims), a window whose `end` is less than **24 hours** old (`window_is_settled` — the two flags only cover SIP-on-Basic; under `iex`, `ALPACA_ALLOW_RECENT_SIP=1` or a zero delay a still-open window returns a partial frame with `end_clamped=False`, and `baselines.py` passes `end_date+1` while `backtests.py` has no future-date check, so the shape does reach the loader), an unconfigured client, and a symbol absent from the response. The clamp and fallback flags are read off the **frames**, not `loader.last_fetch`, which for a >100-symbol call describes only the last chunk; and a failed miss-fetch still answers `{}` rather than the cached hits, so `engine.load_data` keeps raising instead of running a partial universe. A hit restores both `loader.last_fetch` — which `load_data` and `market_data_store._build_dataset` read to verify the source timeframe with `evidence="fetch"` — and the three `.attrs` stamps `feed_provenance()` persists into `agent_runs.metadata`. Read failures are misses: a corrupt or truncated entry is deleted and re-fetched, never raised; a half-written entry is a miss too but is *cleared* only once older than an hour, since a younger one is a concurrent child between its two atomic writes. Eviction is LRU by mtime, never touches the batch that triggered it, and the full directory scan runs only when the writing process's running estimate could have crossed the cap or the last scan is over a minute old — the cap is a runaway bound enforced within a minute, not a quota enforced on the byte. Junk or out-of-range values log and fall back rather than raising at import; this module is on the boot path. `tests/conftest.py` sets `ATL_BAR_CACHE=0` so the suite keeps asserting exact request shapes (`test_alpaca_bars.py` pins batching at literally `[100, 100, 35]`, which a warm cache would shorten) — the production default is pinned instead by `test_bar_cache.py::test_cache_is_enabled_by_default`. Design: `docs/superpowers/specs/2026-09-21-backtest-bar-cache-design.md`.
- `ATL_BAR_CACHE_WARM` (optional, **default ON**): whether `app.py`'s startup hook pre-fetches the default backtest windows into the bar cache on a daemon thread (`infrastructure/market_data/bar_cache_warm.py`). Three windows — the onboarding modal's Mag7 sleeve from `config/defaults.json`, the full Dow over that same window (every default run fetches it for the index baseline, after `publish_phase("saving")`, which is why the wait users notice does not end when the last bar does), and the bare `POST /backtest/run` default window. Without it a cold instance charges the first visitor full price, which defeats the point. **It costs three batched Alpaca calls per deploy, and merging to `main` auto-deploys prod** — negligible quota, but a new recurring outbound call. Failures are logged and swallowed; a cold cache is the status quo, not an outage. ⚠ `tests/conftest.py` strips it to `0`: importing the app in the suite would otherwise make **live Alpaca calls**, which is both a network dependency in an offline suite and real spend. The route-default dates are a deliberate local copy (importing `api/` from `infrastructure/` inverts the layering) pinned equal to the route signature by a source-shape guard in `test_bar_cache_warm.py`.
```

- [ ] **Step 2: Note the phase split**

In the `ATL_BACKTEST_WORKER` bullet, find the sentence listing what `phases[]` carries (`The `starting` entry additionally carries `child_entered_at`, `imports_done_at` and `schema_init_seconds`…`) and append:

```markdown
 `loading_bars` carries `fetch_seconds` for the same reason and prints `fetch <n>s | aggregate+verify <n>s` on the transition out: the phase is one name, but with the bar cache warm the residual **is** the aggregation cost, measured rather than inferred — which is what decides whether caching the aggregated output is worth a second change. `record_phase_metric` attaches a number to whichever phase is open — and drops it when none is — and extras are cleared on every transition, so a figure can never be reported against the wrong phase. The startup clock is seeded only when `--launched-at` actually opens `starting`; a bare CLI run passes the clock without a launch time, and those keys must not surface on `loading_bars`.
```

- [ ] **Step 3: Verify the claims before committing**

Every factual claim in the two bullets must be re-checked at source — a documentation pass that repeats a stale number is worse than none:

```bash
grep -n "BAR_CACHE_DIR" dashboard/backend/paths.py
grep -n "_DEFAULT_MAX_MB\|_DEFAULT_TTL_DAYS\|SCHEMA_VERSION =" \
  dashboard/backend/infrastructure/market_data/bar_cache.py
grep -n "ATL_BAR_CACHE" dashboard/backend/tests/conftest.py
grep -n "100, 100, 35" dashboard/backend/tests/infrastructure/market_data/test_alpaca_bars.py
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the bar cache environment variables"
```

---

## Final verification

Run before handing the branch over:

```bash
pytest dashboard/backend/tests/ -q                       # full suite green
pytest dashboard/backend/tests/ -q -s \
  --ignore=dashboard/backend/tests/infrastructure/market_data/test_bar_cache_warm.py \
  2>&1 | grep -c "bar cache warm:"                       # 0 (see Task 6 Step 6 for why -s)
stat -c '%s' dashboard/storage/data/backtest.db          # 688128
git status --short                                       # clean
git diff --stat origin/main -- dashboard/storage/data/backtest.db    # empty
python3 -c "import ast,sys; ast.parse(open('dashboard/backend/infrastructure/market_data/bar_cache.py').read())"
```

Then check the import direction holds — `bar_cache` must not have grown a dependency on `alpaca_bars`. Match import statements, not the bare name: the module's comments name `alpaca_bars` on purpose (they say *why* it is not imported), so a bare grep prints those and reads as a violation:

```bash
grep -nE "^\s*(from|import)\s.*alpaca_bars" \
  dashboard/backend/infrastructure/market_data/bar_cache.py   # must print nothing
```

Finally re-run the twelve mutations named in Task 3 Step 5, Task 4 Step 6 and the two Task 5 tests' docstrings; every one must turn its named test red. A guard never seen to fail is a comment.

## Follow-ups to file when this lands (spec §12)

Ask before filing — an issue on a shared repo assigns work to someone.

1. `market_data_store._build_dataset` hardcodes `market="US", timezone="US/Eastern"` and has no market dimension in its key — a latent A-share defect on the protocol/v2 path.
2. `_find_cached_run` (`domain/leaderboard/service.py`) keys a persisted `agent_runs` row without the feed.
3. `market_data_store._dataset_key` uses order-sensitive `tuple(symbols)`.
4. `baseline_generator._fetch_bars_for_symbol` is dead code — referenced only by `tests/test_baseline_generator_offline.py`.
5. Read `agent_runs.runtime_type` / `decision_source` in prod to settle the LLM-vs-rule-based mix, which is the ceiling on every latency change of this kind (spec §1-C).
6. Nine stale git-tracked CSVs under `dashboard/storage/data/cache/` with no reader.
