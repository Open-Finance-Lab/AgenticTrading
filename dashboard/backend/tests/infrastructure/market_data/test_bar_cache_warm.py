"""Boot-time bar cache warm: which windows, and never from a test."""

import json
import os
import re

import pandas as pd
import pytest

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.infrastructure.market_data import bar_cache, bar_cache_warm
from dashboard.backend.paths import BACKEND_DIR, CONFIG_DIR, REPO_ROOT


def _defaults():
    return json.loads((CONFIG_DIR / "defaults.json").read_text(encoding="utf-8"))


def _frame(rows=3):
    index = pd.date_range("2026-05-04T13:30:00Z", periods=rows, freq="5min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": [10.0] * rows,
            "high": [11.0] * rows,
            "low": [9.0] * rows,
            "close": [10.5] * rows,
            "volume": [100] * rows,
        },
        index=index,
    )


@pytest.fixture
def warm_cache_dir(tmp_path, monkeypatch):
    """Arm the warm against an isolated cache directory. No network: every
    loader in this module is a fake."""
    monkeypatch.setenv("ATL_BAR_CACHE", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_WARM", "1")
    monkeypatch.setenv("ATL_BAR_CACHE_DIR", str(tmp_path / "bar_cache"))
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    monkeypatch.delenv("ATL_BAR_CACHE_MAX_MB", raising=False)
    monkeypatch.delenv("ATL_BAR_CACHE_TTL_DAYS", raising=False)
    return tmp_path / "bar_cache"


def _writing_loader(calls, written, *, store=True):
    """A loader that writes through `bar_cache` exactly as the real one does.

    The warm now counts entries on disk, so a fake that only returns frames
    would measure nothing. `store=False` reproduces the case this whole
    counting change exists for: every window fetched, every write refused.
    """

    class _Loader:
        def __init__(self):
            self.source_timeframe = "60m"

        def configure_source_timeframe(self, value):
            self.source_timeframe = value

        def fetch_bars(self, symbols, start, end):
            calls.append((self.source_timeframe, tuple(symbols), start, end))
            frames = {symbol: _frame() for symbol in symbols}
            written.append(
                bar_cache.write_many(
                    frames,
                    start=start,
                    end=end,
                    source_timeframe=self.source_timeframe,
                    feed="sip",
                    last_fetch={
                        "feed": "sip",
                        "source_timeframe": self.source_timeframe,
                    },
                    # The real refusal, not a skipped call: an account without
                    # SIP entitlement answers every window on IEX.
                    sip_fallback_to_iex=not store,
                )
            )
            return frames

    return _Loader


def test_the_suite_never_warms():
    """conftest sets ATL_BAR_CACHE_WARM=0 at import time. Without it,
    importing the app would make live Alpaca calls -- a network dependency in
    an offline suite, and real money."""
    assert os.environ.get("ATL_BAR_CACHE_WARM") == "0"
    assert bar_cache.warm_enabled() is False


def test_the_load_test_harness_never_warms_and_never_writes_into_the_repo():
    """SOURCE-SHAPE GUARD. `stress_serve.py` patches the Alpaca loader and
    then runs the real app in-process -- but the startup hook warms through
    the REAL `AlpacaDataLoader` bound inside `bar_cache_warm`, so neither
    patched name is consulted, and the warm defaults ON outside the suite
    (whose conftest a standalone script never loads). Unset, the script makes
    three billable Alpaca calls at boot and writes entries under
    `dashboard/storage/data/`, against a docstring promising it measures "OUR
    stack ... not Alpaca's API" with "all artifacts ... never the repo tree"."""
    script = REPO_ROOT / "dashboard" / "scripts" / "loadtest" / "stress_serve.py"
    source = script.read_text(encoding="utf-8")
    assert re.search(r'environ\["ATL_BAR_CACHE_WARM"\]\s*=\s*"0"', source)
    assert re.search(
        r'environ\["ATL_BAR_CACHE_DIR"\]\s*=\s*os\.path\.join\(ARTIFACTS', source
    )
    # Before the first backend import, like every other env line up there:
    # after it, modules that read these at import time have already read them.
    first_import = re.search(r"^import dashboard\.backend", source, re.M)
    assert first_import, "stress_serve.py stopped importing the backend"
    assert source.index("ATL_BAR_CACHE_WARM") < first_import.start()


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


def test_warm_fetches_every_window_at_the_intraday_source_timeframe(
    warm_cache_dir, monkeypatch
):
    """The key includes source_timeframe, so warming at the wrong resolution
    warms nothing a real run can use. The default US profile is 5m -> 60m."""
    calls, written = [], []
    monkeypatch.setattr(
        bar_cache_warm, "AlpacaDataLoader", _writing_loader(calls, written)
    )
    warmed = bar_cache_warm.warm_bar_cache()
    # Derived, not a literal 3: the window list is the one owner of the count.
    assert len(calls) == len(bar_cache_warm.warm_windows())
    assert {timeframe for timeframe, *_ in calls} == {"5m"}
    # DISTINCT symbol-windows, derived from the window list rather than
    # summed per window: windows one and two share a start/end, so every
    # symbol in both is one entry counted twice by a running total. The fake
    # loader writes each window whole (it does not consult the cache), which
    # is what makes `sum(written)` the overstated figure here.
    expected = {
        (symbol, start, end)
        for symbols, start, end in bar_cache_warm.warm_windows()
        for symbol in symbols
    }
    assert warmed == len(expected) > 0
    assert warmed < sum(written), "the windows no longer overlap; pick another pair"


def test_an_overlapping_window_cannot_silence_the_stored_none_alarm(
    warm_cache_dir, monkeypatch, capsys
):
    """REGRESSION. The alarm used to read the directory AFTER the fetch and
    ask only whether anything was there. Window two shares its start/end with
    window one, so it found window one's entries and reported a non-zero
    `stored` -- the alarm could not fire for it even with all twenty-five of
    its own writes refused. It is now judged on the entries IT added."""
    calls, written = [], []
    stores = iter([True, False, False])

    class _Selective:
        """Stores window one, refuses every window after it."""

        def __init__(self):
            self.source_timeframe = "60m"
            self._store = True

        def configure_source_timeframe(self, value):
            self.source_timeframe = value

        def fetch_bars(self, symbols, start, end):
            store = next(stores)
            calls.append((tuple(symbols), start, end))
            frames = {symbol: _frame() for symbol in symbols}
            written.append(
                bar_cache.write_many(
                    frames,
                    start=start,
                    end=end,
                    source_timeframe=self.source_timeframe,
                    feed="sip",
                    last_fetch={"feed": "sip", "source_timeframe": self.source_timeframe},
                    sip_fallback_to_iex=not store,
                )
            )
            return frames

    monkeypatch.setattr(bar_cache_warm, "AlpacaDataLoader", _Selective)
    bar_cache_warm.warm_bar_cache()
    out = capsys.readouterr().out
    first_start, first_end = bar_cache_warm.warm_windows()[1][1:]
    assert f"{first_start}..{first_end} fetched" in out and "stored none" in out


def test_the_warm_reports_what_it_stored_not_what_it_fetched(
    warm_cache_dir, monkeypatch, capsys
):
    """MUTATION TEST: restore `warmed += len(frames)` and this fails.

    `fetch_bars` returns its frames whether or not `write_many` accepted a
    byte of them, and discards the int that would have said so. On an account
    with no SIP entitlement every window falls back to IEX, every write is
    refused, three billable Alpaca calls are made per deploy forever -- and
    the summary line an operator greps used to report a full warm."""
    calls, written = [], []
    monkeypatch.setattr(
        bar_cache_warm,
        "AlpacaDataLoader",
        _writing_loader(calls, written, store=False),
    )
    warmed = bar_cache_warm.warm_bar_cache()
    assert calls, "the windows were never fetched"
    assert written == [0] * len(calls), "the refusal did not fire"
    assert warmed == 0
    out = capsys.readouterr().out
    assert "stored none" in out
    assert "0 symbol-windows ready" in out


def test_a_failing_window_does_not_stop_the_others(
    warm_cache_dir, monkeypatch, capsys
):
    calls, written = [], []
    base = _writing_loader(calls, written)

    class _FlakyLoader(base):
        def fetch_bars(self, symbols, start, end):
            if not calls:
                calls.append(("raised", tuple(symbols), start, end))
                raise RuntimeError("alpaca is down")
            return super().fetch_bars(symbols, start, end)

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


def test_the_warm_timeframe_matches_the_profile_it_warms():
    """WARM_SOURCE_TIMEFRAME is a copy of the (ALPACA, "djia_30") profile's
    source_timeframe, and the source timeframe is part of the cache key.
    Change the profile without this and the warm keeps fetching 5m windows
    every deploy, writing entries no run can key into. Unlike the route
    dates one bullet above, there is no layering excuse: profiles.py is a
    sibling in this very package."""
    from dashboard.backend.infrastructure.market_data.profiles import (
        ALPACA,
        get_market_profile,
    )

    assert (
        bar_cache_warm.WARM_SOURCE_TIMEFRAME
        == get_market_profile(ALPACA, "djia_30").source_timeframe
    )


def test_the_warm_error_line_matches_the_live_call_detector():
    """SOURCE-SHAPE GUARD. The proof that this suite makes zero billable
    Alpaca calls is a `-s` run grepped for `bar cache warm:`. app.py's own
    handler printed `Bar cache warm error:`, which that grep does not match,
    so a raise escaping warm_bar_cache would have been invisible to it.
    Unreachable today is an accident of the current code; a safety proof a
    future edit can blind for free should not stay blindable."""
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    start = source.index("def bar_cache_background")
    body = source[start : source.index("threading.Thread", start)]
    logged = re.findall(r'print\(\s*f?"([^"]*)"', body)
    assert logged, "the warm's background wrapper stopped logging"
    # "bar cache:" is the prefix both halves share -- the sweep logs under it
    # too, and the detector greps the shorter string.
    assert all("bar cache" in line for line in logged), logged
    assert any("bar cache warm:" in line for line in logged), logged


def test_app_starts_the_warm_on_a_daemon_thread():
    """SOURCE-SHAPE GUARD: a warm on the request path, or a blocking one,
    would delay boot and could fail the health check."""
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    # The whole call, not `daemon=True` on its own: app.py already starts
    # three other daemon threads, so a bare substring check would pass with
    # the warm thread missing entirely.
    assert re.search(
        r"Thread\(\s*target=bar_cache_background,\s*daemon=True\s*\)", source
    )
    assert "bar_cache.describe()" in source


def test_the_startup_bar_cache_block_cannot_abort_the_rest_of_the_hook():
    """SOURCE-SHAPE GUARD. `startup_event` has no handler of its own, so an
    unguarded statement skips everything below it -- `recover_orphaned_runs`
    and `register_reaper_sweep(reap_v2_runs)`, which between them leave
    orphaned protocol runs `running` forever and abandoned v2 runs holding
    their concurrency slots. The import is not inert: it pulls in pandas, and
    `alpaca_bars` imports `bar_cache`, so the cycle edge is live."""
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    block = source[
        source.index("# Wrapped like every other block in this hook") : source.index(
            "def bar_cache_background"
        )
    ]
    assert "bar_cache.describe()" in block
    assert "try:" in block and "except Exception" in block
    describe_at = block.index("bar_cache.describe()")
    assert block.index("try:") < describe_at < block.index("except Exception")


def test_the_boot_sweep_runs_whether_or_not_the_warm_is_armed():
    """SOURCE-SHAPE GUARD. The stray sweep and the LRU pass otherwise fire
    only from inside `write_many`, so a deployment whose writes all fail
    stops reclaiming the `*.tmp` files its killed writers leave behind --
    exactly when reclaiming matters. Boot is the one moment guaranteed to
    arrive without a successful write in front of it, so the sweep must not
    sit behind the opt-in warm flag."""
    source = (BACKEND_DIR / "app.py").read_text(encoding="utf-8")
    body = source[
        source.index("def bar_cache_background") : source.index(
            "Thread(target=bar_cache_background"
        )
    ]
    assert "enforce_size_cap()" in body
    assert body.index("enforce_size_cap()") < body.index("warm_bar_cache")
    assert "warm_enabled" not in body, "the sweep must not be gated on the warm flag"
