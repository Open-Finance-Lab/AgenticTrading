"""/paper/account must use the shared TTL cache like its sibling routes.

CACHE_KEY_ACCOUNT / TTL_ACCOUNT have existed in cache.py since the cache was
introduced, but the account route never consulted them -- every dashboard
poll paid a fresh Alpaca HTTP round-trip (up to its 10s timeout).
"""

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.api.routers import paper_trading
from dashboard.backend.cache import paper_trading_cache


@pytest.fixture(autouse=True)
def clear_cache():
    paper_trading_cache.clear_all()
    yield
    paper_trading_cache.clear_all()


class FakeAlpacaClient:
    instantiations = 0
    account_calls = 0

    def __init__(self):
        type(self).instantiations += 1

    def get_account(self):
        type(self).account_calls += 1
        return {"cash": "100000", "equity": "100000", "status": "ACTIVE"}


@pytest.fixture
def fake_alpaca(monkeypatch):
    FakeAlpacaClient.instantiations = 0
    FakeAlpacaClient.account_calls = 0
    monkeypatch.setattr(paper_trading, "AlpacaPaperTradingClient", FakeAlpacaClient)
    return FakeAlpacaClient


def test_account_served_from_cache_within_ttl(fake_alpaca):
    client = TestClient(app)

    first = client.get("/paper/account")
    assert first.status_code == 200
    assert first.json()["success"] is True
    assert fake_alpaca.account_calls == 1

    second = client.get("/paper/account")
    assert second.status_code == 200
    assert second.json()["success"] is True
    assert second.json()["account"] == first.json()["account"]
    assert fake_alpaca.account_calls == 1, (
        "second /paper/account within the TTL must be served from cache"
    )


def test_account_fetch_failure_is_not_cached(fake_alpaca):
    # No monkeypatch here on purpose: the fixture and the test share one
    # function-scoped monkeypatch instance, so undo() would also revert the
    # fixture's class patch and the recovery request would hit the REAL
    # Alpaca client -- green only on machines with live credentials.
    client = TestClient(app)

    working_get_account = FakeAlpacaClient.get_account

    def broken(self):
        raise RuntimeError("alpaca down")

    FakeAlpacaClient.get_account = broken
    try:
        failed = client.get("/paper/account")
        assert failed.json()["success"] is False
    finally:
        FakeAlpacaClient.get_account = working_get_account

    # Provider recovers: the failure must not have poisoned the cache.
    recovered = client.get("/paper/account")
    assert recovered.json()["success"] is True
    assert fake_alpaca.account_calls == 1, (
        "recovery must be served by the fake provider, not a cached failure "
        "or the real client"
    )
