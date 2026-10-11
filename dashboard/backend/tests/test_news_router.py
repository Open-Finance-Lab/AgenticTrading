import json

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend import cache as cache_mod
from dashboard.backend.api.routers import news as news_router
from dashboard.backend.integrations import news_sentiment as ns


@pytest.fixture
def live_feed(monkeypatch):
    """The live-proxy path, which the frozen snapshot bypasses while FinSearch is paused."""
    monkeypatch.setenv("NEWS_FEED_FROZEN", "0")


@pytest.fixture
def frozen_feed(monkeypatch):
    """Explicitly frozen, so these cases do not depend on the code default and
    keep passing once the default flips back to live."""
    monkeypatch.setenv("NEWS_FEED_FROZEN", "1")


@pytest.fixture
def fresh_snapshot_cache():
    news_router._reset_demo_snapshot_cache()
    yield
    news_router._reset_demo_snapshot_cache()


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
                "staleness_hours": 0.5, "news_overview": "", "signals": {}, "feed": []}
    monkeypatch.setattr(ns, "get_latest_panel_payload", _once)
    client = TestClient(app)
    client.get("/api/news/signals")
    client.get("/api/news/signals")
    assert len(calls) == 1


def test_news_signals_route_negative_caches_failures(monkeypatch, live_feed):
    """The whole point of the short-TTL negative cache: a second failing request
    within TTL_NEWS_UNAVAILABLE is served from cache, NOT re-run through the ~10s
    adapter behind _fetch_lock (the outage-serialization fix). Without the negative
    cache, every Home poll during an outage would block a worker for the full timeout."""
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    calls = []
    def _fail(tickers):
        calls.append(1)
        return dict(ns.UNAVAILABLE_PAYLOAD)
    monkeypatch.setattr(ns, "get_latest_panel_payload", _fail)
    client = TestClient(app)
    assert client.get("/api/news/signals").json()["status"] == "unavailable"
    assert client.get("/api/news/signals").json()["status"] == "unavailable"
    assert len(calls) == 1  # second request served from the negative cache


def test_frozen_feed_serves_demo_snapshot_without_calling_finsearch(monkeypatch, frozen_feed):
    def _never(tickers):
        raise AssertionError("frozen feed must not reach the producer")
    monkeypatch.setattr(ns, "get_latest_panel_payload", _never)
    body = TestClient(app).get("/api/news/signals").json()
    assert body["demo"] is True
    assert body["status"] == "ok"
    # A frozen "Updated Nh ago" would be false; the panel shows "Demo news" instead.
    assert body["staleness_hours"] is None
    assert body["feed"] and body["signals"]


@pytest.mark.parametrize("raw, frozen", [
    ("1", True), ("true", True), (" ON ", True), ("yes", True),
    ("0", False), ("false", False), ("Off", False), ("no", False),
])
def test_freeze_flag_vocabulary(monkeypatch, raw, frozen):
    monkeypatch.setenv("NEWS_FEED_FROZEN", raw)
    assert news_router.news_feed_frozen() is frozen


def test_freeze_flag_unset_uses_the_code_default(monkeypatch):
    monkeypatch.delenv("NEWS_FEED_FROZEN", raising=False)
    assert news_router.news_feed_frozen() is news_router.NEWS_FEED_FROZEN_DEFAULT


def test_freeze_flag_junk_reads_frozen_and_warns_once(monkeypatch, caplog):
    """Junk must not reach a possibly-dead upstream, and must not log per request."""
    monkeypatch.setenv("NEWS_FEED_FROZEN", "frozne")
    monkeypatch.setattr(news_router, "_warned_values", set())
    with caplog.at_level("WARNING", logger=news_router.__name__):
        results = [news_router.news_feed_frozen() for _ in range(3)]
    assert results == [True, True, True]
    assert sum("frozne" in r.getMessage() for r in caplog.records) == 1


def test_freeze_flag_is_read_per_request(monkeypatch):
    """Resuming from the Render dashboard needs no deploy: the env var is live."""
    calls = []
    def _live(tickers):
        calls.append(1)
        return {"status": "ok", "status_reason": None, "generated_at": None,
                "staleness_hours": 0.1, "news_overview": "", "signals": {}, "feed": []}
    monkeypatch.setattr(ns, "get_latest_panel_payload", _live)
    cache_mod.shared_ttl_cache.invalidate(cache_mod.CACHE_KEY_NEWS_SIGNALS)
    client = TestClient(app)
    monkeypatch.setenv("NEWS_FEED_FROZEN", "1")
    assert client.get("/api/news/signals").json().get("demo") is True
    monkeypatch.setenv("NEWS_FEED_FROZEN", "0")
    assert "demo" not in client.get("/api/news/signals").json()
    assert calls == [1]


def test_demo_snapshot_matches_the_panel_contract(fresh_snapshot_cache):
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


def test_loader_owns_the_frozen_markers(monkeypatch, tmp_path, fresh_snapshot_cache):
    """`demo` and the null staleness come from the loader, not the file, so an
    edited or regenerated snapshot cannot drop the label or claim freshness."""
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"status": "ok", "staleness_hours": 0.2,
                                "demo": False, "feed": [], "signals": {}}))
    monkeypatch.setattr(news_router, "NEWS_DEMO_SNAPSHOT_PATH", path)
    body = news_router._demo_snapshot()
    assert body["demo"] is True
    assert body["staleness_hours"] is None


def test_committed_snapshot_does_not_restate_the_demo_flag():
    raw = json.loads(news_router.NEWS_DEMO_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert "demo" not in raw, "the loader owns `demo`; keep one source of truth"


def test_each_response_is_a_private_copy(fresh_snapshot_cache):
    first = news_router._demo_snapshot()
    first["feed"].clear()
    first["signals"].clear()
    second = news_router._demo_snapshot()
    assert second["feed"] and second["signals"]


def test_unreadable_snapshot_fails_closed(monkeypatch, tmp_path, frozen_feed, fresh_snapshot_cache):
    monkeypatch.setattr(news_router, "NEWS_DEMO_SNAPSHOT_PATH", tmp_path / "missing.json")
    body = TestClient(app).get("/api/news/signals").json()
    assert body["status"] == "unavailable"


@pytest.mark.parametrize("content", ["[]", "null", '"text"', "42", '{"status": "ok"}'])
def test_non_payload_snapshot_fails_closed(monkeypatch, tmp_path, frozen_feed,
                                           fresh_snapshot_cache, content):
    """Valid JSON that is not a panel object must fail closed, not 500."""
    path = tmp_path / "snap.json"
    path.write_text(content)
    monkeypatch.setattr(news_router, "NEWS_DEMO_SNAPSHOT_PATH", path)
    resp = TestClient(app).get("/api/news/signals")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unavailable"


def test_a_failed_load_is_not_cached(monkeypatch, tmp_path, fresh_snapshot_cache):
    """A transient read error at boot must not pin `unavailable` until restart."""
    path = tmp_path / "snap.json"
    monkeypatch.setattr(news_router, "NEWS_DEMO_SNAPSHOT_PATH", path)
    assert news_router._demo_snapshot()["status"] == "unavailable"
    path.write_text(json.dumps({"status": "ok", "feed": [{"headline": "h"}], "signals": {}}))
    recovered = news_router._demo_snapshot()
    assert recovered["status"] == "ok" and recovered["demo"] is True
