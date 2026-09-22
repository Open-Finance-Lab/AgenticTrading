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
