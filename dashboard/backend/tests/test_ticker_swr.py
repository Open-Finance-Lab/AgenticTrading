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


def test_outage_past_serve_window_prefers_stale_over_empty(monkeypatch):
    """A provider outage must never be worse than having no cache at all.

    Preserving the stale payload protects the *cache*; without this it did not
    protect the *response*. Past the serve window every request re-paid the
    multi-second fetch and still handed back an empty ticker, while perfectly
    good data sat in the cache -- strictly worse than the pre-SWR behaviour.
    """
    _seed_cache(quotes.TICKER_STALE_SERVE_SECONDS + 10, OLD_QUOTES)
    calls = []

    def outage(symbols):
        calls.append(1)
        return []

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", outage)

    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES, (
        "stale data beats an empty ticker"
    )
    assert len(calls) == 1

    # Inside the backoff the slow provider must not be retried per request.
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES
    assert len(calls) == 1, "a failed fetch must back off, not retry every request"


def test_failure_backoff_expires_and_refetches(monkeypatch):
    _seed_cache(quotes.TICKER_STALE_SERVE_SECONDS + 10, OLD_QUOTES)
    quotes._ticker_failed_fetch_at["AAPL"] = (
        time.time() - quotes.TICKER_FAILED_FETCH_BACKOFF_SECONDS - 1
    )
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)

    assert quotes.get_market_quotes(["AAPL"]) == NEW_QUOTES
    assert "AAPL" not in quotes._ticker_failed_fetch_at, (
        "a successful fetch must clear the failure marker"
    )


class _ThreadStartFails:
    """Stands in for threading.Thread when the process cannot spawn one."""

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        raise RuntimeError("can't start new thread")


def test_refresh_thread_that_cannot_start_does_not_wedge_the_key(monkeypatch):
    _seed_cache(quotes.TICKER_CACHE_TTL_SECONDS + 5, OLD_QUOTES)
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)
    # Patches the real threading module for the duration of this test; nothing
    # else in-process spawns threads here and monkeypatch restores it.
    monkeypatch.setattr(quotes.threading, "Thread", _ThreadStartFails)

    # Must not raise: letting it propagate reaches the route's except branch and
    # returns an empty ticker instead of the stale payload we already hold.
    assert quotes.get_market_quotes(["AAPL"]) == OLD_QUOTES
    assert quotes._ticker_refresh_inflight == set(), (
        "a thread that never started must not leave its key marked inflight "
        "forever -- that permanently disables background refresh for it"
    )

    monkeypatch.undo()
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)
    quotes.get_market_quotes(["AAPL"])
    assert _wait_until(lambda: quotes._ticker_cache["AAPL"][1] == NEW_QUOTES), (
        "background refresh must recover once threads can start again"
    )


def test_refresh_failure_log_escapes_injected_newlines(monkeypatch, capsys):
    """cache_key is built from unauthenticated symbols=; it must not forge a log line."""
    evil = "AAPL\nFAKE LOG LINE"
    quotes._ticker_cache[evil] = (
        time.time() - quotes.TICKER_CACHE_TTL_SECONDS - 5,
        OLD_QUOTES,
    )

    def boom(symbols):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", boom)
    quotes.get_market_quotes([evil])
    assert _wait_until(lambda: not quotes._ticker_refresh_inflight)

    out = capsys.readouterr().out
    assert "Ticker background refresh failed" in out
    assert "\nFAKE LOG LINE" not in out, "a raw newline reached the log unescaped"


def test_cache_keys_are_bounded(monkeypatch):
    """symbols= is unauthenticated input, so the keyed maps must not grow forever."""
    monkeypatch.setattr(quotes, "_fetch_quotes_uncached", lambda symbols: NEW_QUOTES)

    for i in range(quotes.TICKER_MAX_CACHE_KEYS + 40):
        quotes.get_market_quotes([f"SYM{i}"])

    assert len(quotes._ticker_cache) <= quotes.TICKER_MAX_CACHE_KEYS
    assert len(quotes._ticker_fetch_locks) <= quotes.TICKER_MAX_CACHE_KEYS


def test_ticker_route_caps_symbol_fan_out(monkeypatch):
    from fastapi.testclient import TestClient

    from dashboard.backend.api.routers import market
    from dashboard.backend.app import app

    def must_not_fetch(symbols):
        raise AssertionError("an over-long symbol list must be rejected first")

    monkeypatch.setattr(market, "get_market_quotes", must_not_fetch)
    client = TestClient(app)

    too_many = ",".join(f"SYM{i}" for i in range(market.MAX_TICKER_SYMBOLS + 1))
    body = client.get(f"/ticker?symbols={too_many}").json()
    assert body["quotes"] == []
    assert "Too many symbols" in body["error"]


def test_ticker_route_dedupes_symbols(monkeypatch):
    from fastapi.testclient import TestClient

    from dashboard.backend.api.routers import market
    from dashboard.backend.app import app

    seen = []

    def record(symbols):
        seen.append(list(symbols))
        return NEW_QUOTES

    monkeypatch.setattr(market, "get_market_quotes", record)
    TestClient(app).get("/ticker?symbols=AAPL,aapl,AAPL")
    assert seen == [["AAPL"]], "duplicate symbols must collapse to one fetch"
