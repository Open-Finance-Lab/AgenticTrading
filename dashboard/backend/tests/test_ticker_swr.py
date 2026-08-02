"""Stale-while-revalidate behavior of the /ticker quote cache.

The frontend polls /ticker every 30s and the fresh-TTL is 30s, so before SWR
nearly every poll paid the full multi-second provider fetch in-request. With
SWR a request that finds a stale-but-recent entry returns it immediately and
refreshes in a background thread; only a genuinely cold cache fetches inline.
"""

import threading
import time

import pytest

from dashboard.backend.infrastructure.market_data import quotes


def _wait_until(condition, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


@pytest.fixture(autouse=True)
def reset_ticker_cache_state():
    quotes._ticker_cache.clear()
    quotes._reset_ticker_refresh_state_for_tests()
    yield
    quotes._ticker_cache.clear()
    quotes._reset_ticker_refresh_state_for_tests()


OLD_QUOTES = [{"symbol": "AAPL", "price": 1.0, "changePercent": 0.0, "timestamp": "t0"}]
NEW_QUOTES = [{"symbol": "AAPL", "price": 2.0, "changePercent": 1.0, "timestamp": "t1"}]


def _seed_cache(age_seconds, payload):
    quotes._ticker_cache["AAPL"] = (time.time() - age_seconds, payload)


def test_fresh_cache_hit_never_fetches(monkeypatch):
    _seed_cache(0, OLD_QUOTES)

    def must_not_fetch(symbols):
        raise AssertionError("fresh cache hit must not reach the provider")

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", must_not_fetch)
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES


def test_cold_cache_fetches_inline_and_caches(monkeypatch):
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)
    assert quotes.get_market_quotes(["AAPL"]) == NEW_QUOTES
    assert quotes._ticker_cache["AAPL"][1] == NEW_QUOTES


def test_stale_cache_serves_stale_immediately_then_refreshes(monkeypatch):
    _seed_cache(quotes.TICKER_CACHE_TTL_SECONDS + 5, OLD_QUOTES)
    calls = []

    def fetch(symbols):
        calls.append(list(symbols))
        return NEW_QUOTES

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", fetch)

    started = time.perf_counter()
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES
    assert time.perf_counter() - started < 0.2, "stale entry must be served without waiting"

    assert _wait_until(lambda: quotes._ticker_cache["AAPL"][1] == NEW_QUOTES), (
        "background refresh never landed"
    )
    assert calls == [["AAPL"]]
    assert quotes.get_market_quotes(["AAPL"]) == NEW_QUOTES


def test_failed_background_refresh_keeps_stale_data(monkeypatch):
    _seed_cache(quotes.TICKER_CACHE_TTL_SECONDS + 5, OLD_QUOTES)
    calls = []

    def fetch(symbols):
        calls.append(1)
        return []  # provider outage: empty result

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", fetch)

    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES
    assert _wait_until(lambda: calls and not quotes._ticker_refresh_inflight)
    # The good-but-stale payload must survive a failed refresh.
    assert quotes._ticker_cache["AAPL"][1] == OLD_QUOTES
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES


def test_stale_beyond_serve_window_fetches_inline(monkeypatch):
    _seed_cache(quotes.TICKER_STALE_SERVE_SECONDS + 10, OLD_QUOTES)
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)
    assert quotes.get_market_quotes(["AAPL"]) == NEW_QUOTES


def test_empty_stale_entry_is_treated_as_cold(monkeypatch):
    # An empty cached payload has nothing worth serving stale.
    _seed_cache(quotes.TICKER_CACHE_TTL_SECONDS + 5, [])
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)
    assert quotes.get_market_quotes(["AAPL"]) == NEW_QUOTES


def test_concurrent_cold_requests_fetch_once(monkeypatch):
    calls = []

    def slow_fetch(symbols):
        calls.append(1)
        time.sleep(0.3)
        return NEW_QUOTES

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", slow_fetch)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(quotes.get_market_quotes(["AAPL"])))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(calls) == 1, "concurrent cold requests must share one provider fetch"
    assert results == [NEW_QUOTES] * 4
