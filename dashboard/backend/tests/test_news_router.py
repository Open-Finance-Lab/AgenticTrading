import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend import cache as cache_mod
from dashboard.backend.api.routers import news as news_router
from dashboard.backend.integrations import news_sentiment as ns


@pytest.fixture
def live_feed(monkeypatch):
    """The live-proxy path, which NEWS_FEED_FROZEN bypasses while FinSearch is paused."""
    monkeypatch.setattr(news_router, "NEWS_FEED_FROZEN", False)


def test_news_signals_route_returns_payload(monkeypatch, live_feed):
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    fake = {"status": "ok", "status_reason": None, "generated_at": "2026-07-13T11:20:00+00:00",
            "staleness_hours": 1.2, "news_overview": "calm", "signals": {}, "feed": []}
    monkeypatch.setattr(ns, "get_latest_panel_payload", lambda tickers: fake)
    client = TestClient(app)
    resp = client.get("/api/news/signals")
    assert resp.status_code == 200
    assert resp.json()["news_overview"] == "calm"


def test_news_signals_route_fail_closed(monkeypatch, live_feed):
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    def _boom(tickers):
        raise RuntimeError("adapter exploded")
    monkeypatch.setattr(ns, "get_latest_panel_payload", _boom)
    client = TestClient(app)
    resp = client.get("/api/news/signals")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unavailable"


def test_news_signals_route_caches(monkeypatch, live_feed):
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    calls = []
    def _once(tickers):
        calls.append(1)
        return {"status": "ok", "status_reason": None, "generated_at": None,
                "staleness_hours": None, "news_overview": None, "signals": {}, "feed": []}
    monkeypatch.setattr(ns, "get_latest_panel_payload", _once)
    client = TestClient(app)
    client.get("/api/news/signals")
    client.get("/api/news/signals")
    assert len(calls) == 1


def test_news_signals_route_negative_caches_failures(monkeypatch, live_feed):
    """The whole point of the short-TTL negative cache: a second failing request
    within TTL_NEWS_UNAVAILABLE is served from cache, NOT re-run through the ~10s
    adapter behind _fetch_lock (the outage-serialization fix). Without the negative
    cache this asserts len(calls) == 2 — the regression the fix prevents."""
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    calls = []
    def _boom(tickers):
        calls.append(1)
        raise RuntimeError("adapter exploded")
    monkeypatch.setattr(ns, "get_latest_panel_payload", _boom)
    client = TestClient(app)
    assert client.get("/api/news/signals").json()["status"] == "unavailable"
    assert client.get("/api/news/signals").json()["status"] == "unavailable"
    assert len(calls) == 1  # second request served from the negative cache


def test_frozen_feed_serves_demo_snapshot_without_calling_finsearch(monkeypatch):
    def _never(tickers):
        raise AssertionError("frozen feed must not reach the producer")
    monkeypatch.setattr(ns, "get_latest_panel_payload", _never)
    body = TestClient(app).get("/api/news/signals").json()
    assert news_router.NEWS_FEED_FROZEN is True
    assert body["demo"] is True
    assert body["status"] == "ok"
    # A frozen "Updated Nh ago" would be false; the panel shows "Demo news" instead.
    assert body["staleness_hours"] is None
    assert body["feed"] and body["signals"]


def test_demo_snapshot_matches_the_panel_contract():
    """The snapshot stands in for get_latest_panel_payload, so it must carry
    every key the panel reads, in the shape the live projection produces."""
    body = news_router._demo_snapshot()
    for key in ("status", "status_reason", "generated_at", "staleness_hours",
                "news_overview", "signals", "feed"):
        assert key in body
    for item in body["feed"]:
        assert {"headline", "url", "source", "published", "ticker"} <= item.keys()
    for sig in body["signals"].values():
        assert set(ns._PANEL_SIGNAL_KEYS) <= sig.keys()


def test_unreadable_snapshot_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(news_router, "NEWS_DEMO_SNAPSHOT_PATH", tmp_path / "missing.json")
    news_router._demo_snapshot.cache_clear()
    try:
        body = TestClient(app).get("/api/news/signals").json()
    finally:
        news_router._demo_snapshot.cache_clear()
    assert body["status"] == "unavailable"
