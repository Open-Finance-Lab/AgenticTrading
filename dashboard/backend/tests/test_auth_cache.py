"""Auth TTL cache + last_used_at debounce + invalidation (T4)."""

import pytest

import dashboard.backend.domain.agents.repository as repo_module
from dashboard.backend.domain.agents import auth_cache


class _RecordingStore:
    def __init__(self):
        self.resolves = []          # (api_key, touch)
        self.agents = {}            # key -> agent dict

    def add(self, api_key, agent_id):
        self.agents[api_key] = {"agent_id": agent_id, "session_id": f"s_{agent_id}",
                                "scopes": ["runs:read"]}

    def resolve_api_key(self, api_key, touch=True):
        self.resolves.append((api_key, touch))
        return dict(self.agents[api_key]) if api_key in self.agents else None


@pytest.fixture
def store(monkeypatch):
    s = _RecordingStore()
    s.add("key-A", "agent-A")
    monkeypatch.setattr(repo_module, "agent_store", s)
    auth_cache._reset_for_tests()
    monkeypatch.setattr(auth_cache, "_jitter", lambda: 1.0)  # deterministic TTL
    yield s
    auth_cache._reset_for_tests()


def test_hit_within_ttl_skips_the_db(store, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(auth_cache, "_now", lambda: t[0])
    a1 = auth_cache.resolve_api_key("key-A")
    a2 = auth_cache.resolve_api_key("key-A")
    assert a1["agent_id"] == a2["agent_id"] == "agent-A"
    assert len(store.resolves) == 1                    # second call: cache hit
    t[0] += auth_cache.AGENT_AUTH_CACHE_TTL_SECONDS + 1
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2                    # expired: DB again


def test_last_used_write_debounced_to_60s(store, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(auth_cache, "_now", lambda: t[0])
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", True)       # first resolve touches
    t[0] += auth_cache.AGENT_AUTH_CACHE_TTL_SECONDS + 1
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", False)      # <60s since last write
    t[0] += 61.0
    auth_cache.resolve_api_key("key-A")
    assert store.resolves[-1] == ("key-A", True)       # debounce window passed


def test_invalidate_agent_evicts_by_reverse_index(store, monkeypatch):
    monkeypatch.setattr(auth_cache, "_now", lambda: 1000.0)
    auth_cache.resolve_api_key("key-A")
    auth_cache.invalidate_agent("agent-A")
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2                    # cache was evicted


def test_zero_ttl_disables_caching(store, monkeypatch):
    monkeypatch.setattr(auth_cache, "AGENT_AUTH_CACHE_TTL_SECONDS", 0.0)
    auth_cache.resolve_api_key("key-A")
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2
    assert all(touch for _, touch in store.resolves)   # passthrough always touches


def test_misses_are_not_cached(store):
    assert auth_cache.resolve_api_key("nope") is None
    assert auth_cache.resolve_api_key("nope") is None
    assert len(store.resolves) == 2


def test_inflight_resolve_not_recached_after_racing_invalidation(store, monkeypatch):
    """A rotate/delete landing WHILE a cache-miss resolve is mid-DB-read must not
    let that resolve write its stale pre-rotation snapshot into the cache — else
    the revoked key would authenticate from cache until the TTL."""
    monkeypatch.setattr(auth_cache, "_now", lambda: 1000.0)
    real_resolve = store.resolve_api_key

    def resolve_then_invalidate(api_key, touch=True):
        agent = real_resolve(api_key, touch=touch)
        auth_cache.invalidate_agent("agent-A")  # rotate/delete lands mid-flight
        return agent

    monkeypatch.setattr(store, "resolve_api_key", resolve_then_invalidate)
    assert auth_cache.resolve_api_key("key-A")["agent_id"] == "agent-A"  # in-flight ok
    assert len(store.resolves) == 1
    # Not resurrected: the next resolve must re-read the DB, not hit a poisoned cache.
    monkeypatch.setattr(store, "resolve_api_key", real_resolve)
    auth_cache.resolve_api_key("key-A")
    assert len(store.resolves) == 2


def test_mutating_returned_scopes_does_not_corrupt_cache(store, monkeypatch):
    """The cached agent's scopes list must be independent of what callers get,
    so an in-place edit by one caller can't leak into every other holder."""
    monkeypatch.setattr(auth_cache, "_now", lambda: 1000.0)
    first = auth_cache.resolve_api_key("key-A")
    first["scopes"].append("runs:write")   # a caller mutates its own view
    second = auth_cache.resolve_api_key("key-A")  # served from cache
    assert second["scopes"] == ["runs:read"]      # cache untouched
    assert len(store.resolves) == 1               # confirm it was a cache hit


def test_mutating_returned_pipeline_does_not_corrupt_cache(store, monkeypatch):
    """pipeline (a JSON-decoded list of step dicts) is the second mutable nested
    field on a real agent record; like scopes, the cache's copy must be
    independent of the caller's, down to the nested step dicts."""
    monkeypatch.setattr(auth_cache, "_now", lambda: 1000.0)
    store.agents["key-A"]["pipeline"] = [{"role": "researcher"}]
    first = auth_cache.resolve_api_key("key-A")     # miss: fills cache
    first["pipeline"].append({"role": "trader"})    # a caller mutates its view
    first["pipeline"][0]["role"] = "hijacked"       # ...including a nested step
    second = auth_cache.resolve_api_key("key-A")    # served from cache
    assert second["pipeline"] == [{"role": "researcher"}]  # cache untouched
    assert len(store.resolves) == 1                        # confirm cache hit


def test_rotate_endpoint_invalidates_old_key():
    """End-to-end: after key rotation the OLD key must fail immediately, not
    after the TTL — exactly the behavior the cache would break without
    invalidate_agent wired into the rotate paths. Route verified:
    POST /api/v2/agents/{agent_id}/rotate-key (api/v2/agents.py:69)."""
    from fastapi.testclient import TestClient
    from dashboard.backend.app import app

    client = TestClient(app)
    r = client.post("/api/v2/agents", json={"name": "rotate-me"}).json()
    old_key, agent_id = r["api_key"], r["agent_id"]
    # Warm the auth cache with the old key: an authenticated read of a missing
    # run answers 404; an unauthenticated one answers 401. No run is created.
    probe = client.get("/api/v2/runs/does-not-exist", headers={"X-API-Key": old_key})
    assert probe.status_code == 404
    rot = client.post(f"/api/v2/agents/{agent_id}/rotate-key",
                      headers={"X-API-Key": old_key})
    assert rot.status_code == 200, rot.text
    denied = client.get("/api/v2/runs/does-not-exist",
                        headers={"X-API-Key": old_key})
    assert denied.status_code == 401                   # immediately, not <=TTL
