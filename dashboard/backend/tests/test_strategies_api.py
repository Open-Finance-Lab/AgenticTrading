"""MEDIUM #4 — POST /api/strategies must bound prompt size and write rate.

The endpoint is public by design (shared links work without a session), but it
had no prompt size cap and no write rate limit — an anonymous client could write
unbounded, megabyte-sized prompts without any throttle. These tests pin the size
cap (422) and a per-client write rate limit (429). ``owner`` remains a
display-only attribution label (e.g. ``discord:<id>``), never an auth control.
"""

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.api.rate_limit import (
    _MAX_IP_KEY_LEN,
    FixedWindowRateLimiter,
    client_ip,
    client_key,
)
import dashboard.backend.api.routers.strategies as strategies_router


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # Guarded so the pure-limiter unit tests below pass even before the endpoint
    # is wired to a module-level limiter (clean red-green).
    rl = getattr(strategies_router, "_create_rate_limiter", None)
    if rl is not None:
        rl.reset()
    yield
    rl = getattr(strategies_router, "_create_rate_limiter", None)
    if rl is not None:
        rl.reset()


# --- FixedWindowRateLimiter unit tests (deterministic fake clock) -----------

def test_rate_limiter_allows_up_to_max_then_rejects():
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=2, window_seconds=10, clock=lambda: now[0])
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is False  # 3rd within window


def test_rate_limiter_window_expiry_allows_again():
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=1, window_seconds=10, clock=lambda: now[0])
    assert rl.allow("k") is True
    assert rl.allow("k") is False
    now[0] = 11.0  # past the window
    assert rl.allow("k") is True


def test_rate_limiter_keys_are_independent():
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=1, window_seconds=10, clock=lambda: now[0])
    assert rl.allow("a") is True
    assert rl.allow("b") is True  # different key, own budget
    assert rl.allow("a") is False


def test_rate_limiter_reclaims_expired_keys_when_over_max_keys():
    """Memory must not grow without bound as distinct keys are seen: once the key
    count hits max_keys, a new key sweeps fully-expired buckets."""
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=1, window_seconds=10, clock=lambda: now[0], max_keys=3)
    for i in range(3):
        assert rl.allow(f"k{i}") is True
    assert len(rl._events) == 3
    now[0] = 100.0  # every existing key's window has fully expired
    assert rl.allow("k_new") is True
    assert len(rl._events) <= 3  # expired buckets reclaimed, not unbounded


def test_rate_limiter_check_does_not_consume_budget():
    """check() must be a pure read, or the split is pointless: its whole purpose
    is guarding an expensive call without charging for a successful outcome."""
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=2, window_seconds=10, clock=lambda: now[0])
    for _ in range(50):
        assert rl.check("k") is True
    assert rl.allow("k") is True
    assert rl.allow("k") is True
    assert rl.check("k") is False


def test_rate_limiter_record_never_grows_past_max_events():
    """A caller that records without checking must not defeat the per-key cap --
    that cap is what bounds memory for a key under sustained abuse."""
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=2, window_seconds=10, clock=lambda: now[0])
    for _ in range(100):
        rl.record("k")
    assert len(rl._events["k"]) == 2


def test_rate_limiter_zero_max_events_disables_it():
    """0 means "switch this budget off" so an operator can disable a limit from
    config without a deploy."""
    rl = FixedWindowRateLimiter(max_events=0, window_seconds=10)
    assert rl.enabled is False
    for _ in range(1000):
        assert rl.allow("k") is True
    assert rl._events == {}  # disabled means no bookkeeping at all
    assert rl.retry_after_seconds("k") == 1


def test_rate_limiter_negative_max_events_still_rejected():
    with pytest.raises(ValueError):
        FixedWindowRateLimiter(max_events=-1, window_seconds=10)


def test_retry_after_seconds_counts_down_and_never_exceeds_the_window():
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=1, window_seconds=10, clock=lambda: now[0])
    assert rl.allow("k") is True
    assert rl.retry_after_seconds("k") == 10
    now[0] = 6.0
    assert rl.retry_after_seconds("k") == 4  # oldest event ages out at t=10
    assert rl.retry_after_seconds("never-seen") == 10  # full window, not more


def test_retry_after_seconds_prunes_expired_events():
    """Correct when called on its own, not only straight after a rejecting
    allow() that pruned as a side effect."""
    now = [0.0]
    rl = FixedWindowRateLimiter(max_events=1, window_seconds=10, clock=lambda: now[0])
    assert rl.allow("k") is True
    now[0] = 50.0  # the recorded event is long expired
    # Reading a stale q[0] here would return max(1, 10 + 0 - 50) == 1 by luck;
    # the real tell is that the bucket is empty, so the answer is a full window.
    assert rl.retry_after_seconds("k") == 10
    assert rl.check("k") is True


# --- client key derivation --------------------------------------------------

def _fake_request(*, headers=None, peer="10.0.0.1"):
    from starlette.requests import Request

    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "client": (peer, 1234) if peer else None})


def test_client_ip_prefers_the_leftmost_forwarded_entry():
    """Behind a PaaS router the socket peer is the router for every visitor, so
    keying on it alone collapses per-client budgets into one shared bucket."""
    req = _fake_request(headers={"X-Forwarded-For": "203.0.113.7, 70.41.3.18, 10.0.0.1"})
    assert client_ip(req) == "203.0.113.7"


def test_client_ip_falls_back_to_the_peer():
    assert client_ip(_fake_request()) == "10.0.0.1"
    assert client_ip(_fake_request(headers={"X-Forwarded-For": "  ,  "})) == "10.0.0.1"
    assert client_ip(_fake_request(peer=None)) == "unknown"


def test_client_ip_bounds_the_key_length():
    """The header is attacker-controlled; an untruncated value would let one
    client mint arbitrarily large keys in the limiter's dict."""
    req = _fake_request(headers={"X-Forwarded-For": "9" * 5000})
    assert len(client_ip(req)) == _MAX_IP_KEY_LEN


def test_client_key_prefers_browser_id_then_ip():
    assert client_key(_fake_request(headers={"X-Browser-Id": " abc "})) == "id:abc"
    # A blank id must not win: it would key every such caller to one bucket.
    assert client_key(_fake_request(headers={"X-Browser-Id": "   "})) == "ip:10.0.0.1"
    assert client_key(
        _fake_request(headers={"X-Forwarded-For": "203.0.113.7"})
    ) == "ip:203.0.113.7"


# --- endpoint tests ---------------------------------------------------------

def test_create_strategy_ok_returns_share_url():
    client = TestClient(app)
    resp = client.post("/api/strategies", json={"prompt": "buy the dip"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"]
    assert body["share_url"].endswith(f"/strategy?code={body['code']}")


def test_create_strategy_rejects_oversized_prompt():
    client = TestClient(app)
    resp = client.post("/api/strategies", json={"prompt": "x" * 6000})
    assert resp.status_code == 422


def test_create_strategy_rate_limited_per_client(monkeypatch):
    # Swap in a tiny limiter so we don't need 30 real DB writes to prove wiring.
    now = [0.0]
    monkeypatch.setattr(
        strategies_router,
        "_create_rate_limiter",
        FixedWindowRateLimiter(max_events=2, window_seconds=3600, clock=lambda: now[0]),
    )
    client = TestClient(app)
    headers = {"X-Session-Id": "rate-key-1"}
    assert client.post("/api/strategies", json={"prompt": "a"}, headers=headers).status_code == 200
    assert client.post("/api/strategies", json={"prompt": "b"}, headers=headers).status_code == 200
    third = client.post("/api/strategies", json={"prompt": "c"}, headers=headers)
    assert third.status_code == 429
