"""Mission Control (``/api/v1/mission-control/overview``).

Covers the fix-pass findings for this router:

* C17 -- the route must be admin-gated (``require_admin``): 401 signed-out,
  403 signed-in-but-not-admin, 200 for an admin. Real-money account data has
  no per-caller scoping (there is exactly one operator-owned live account),
  unlike everything else behind ``/api``.
* C20 -- a broker-constructor failure other than "not configured" (e.g. a
  malformed ``credentials/alpaca_live.json``) must not leak raw exception text
  and must not 500 the whole response (which would also take the paper wallet
  half down with it).
* C21 -- both snapshots are cached for 30s in the shared ``paper_trading_cache``
  so a burst of admin page-loads doesn't hit the broker twice.

Client construction is always monkeypatched rather than relying on Alpaca
credentials being absent from the test environment -- the same convention
``test_error_detail_sanitization.py`` uses, since an ambient
``ALPACA_API_KEY``/``ALPACA_LIVE_API_KEY`` in the developer's shell is not
stripped by conftest.py the way the database/session env vars are.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.api.routers import mission_control as mc_mod
from dashboard.backend.paths import FRONTEND_DIR


@pytest.fixture
def isolated_auth(monkeypatch):
    """Fresh UserStore swapped in for the module singleton.

    Mirrors ``test_admin_users.py``'s fixture of the same name: promoting an
    admin in the shared conftest store would leak the row into every later
    test in the session.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = users_module.UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        yield TestClient(app), store


@pytest.fixture(autouse=True)
def _clear_mission_control_cache():
    """Every test starts and ends with a cold cache for these two keys.

    Narrower than ``paper_trading_cache.clear_all()`` on purpose: this cache is
    a process-wide singleton shared with the paper-trading router's own tests,
    and clearing the whole thing on every test in this file would be an
    unrelated blast radius.
    """
    mc_mod.paper_trading_cache.invalidate(mc_mod._CACHE_KEY_PAPER)
    mc_mod.paper_trading_cache.invalidate(mc_mod._CACHE_KEY_LIVE)
    yield
    mc_mod.paper_trading_cache.invalidate(mc_mod._CACHE_KEY_PAPER)
    mc_mod.paper_trading_cache.invalidate(mc_mod._CACHE_KEY_LIVE)


def _signup(client, email="mc@example.com"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "display_name": "MC", "password": "SecurePass1!"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["user"]


def _promote(store, user_id):
    return store.apply_admin_patch(user_id, role="admin")


def _make_admin(client, store, email="mc-admin@example.com"):
    user = _signup(client, email)
    _promote(store, user["id"])
    return user


class _FakePosition:
    """Stand-in for ``alpaca_paper.Position`` -- ``_paper_snapshot`` reads it
    by attribute, not by dict key."""

    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.qty = 10
        self.avg_fill_price = 100.0
        self.current_price = 110.0
        self.market_value = 1100.0
        self.unrealized_pl = 100.0
        self.unrealized_plpc = 0.1
        self.side = "long"


class _FakePaperClient:
    def __init__(self):
        pass

    def get_account(self):
        return {"cash": 5000.0, "equity": 10000.0, "buying_power": 5000.0, "portfolio_value": 10000.0}

    def get_positions(self):
        return [_FakePosition("AAPL")]


class _FakeLiveClient:
    def __init__(self):
        pass

    def get_account(self):
        return {"cash": 1000.0, "equity": 2000.0, "buying_power": 1000.0, "portfolio_value": 2000.0}

    def get_positions_detailed(self):
        return [
            {
                "symbol": "MSFT",
                "qty": 5,
                "avg_entry_price": 200.0,
                "current_price": 210.0,
                "market_value": 1050.0,
                "unrealized_pl": 50.0,
                "unrealized_plpc": 0.05,
                "side": "long",
            }
        ]


class _ExplodingLiveClient:
    """A plain ``RuntimeError`` -- distinct from ``AlpacaLiveCredentialsError``
    (which subclasses it), so this exercises the *other* except branch: e.g. a
    malformed ``credentials/alpaca_live.json`` raising mid-``__init__``."""

    def __init__(self):
        raise RuntimeError("TRACE-MARKER /opt/render/project/secret_config.py line 42")


def test_overview_unauthenticated_is_401():
    client = TestClient(app)
    resp = client.get("/api/v1/mission-control/overview")
    assert resp.status_code == 401


def test_overview_signed_in_non_admin_is_403(isolated_auth):
    client, _store = isolated_auth
    _signup(client, "outsider@example.com")
    resp = client.get("/api/v1/mission-control/overview")
    assert resp.status_code == 403


def test_overview_admin_returns_paper_and_live(isolated_auth, monkeypatch):
    client, store = isolated_auth
    _make_admin(client, store)

    monkeypatch.setattr(mc_mod, "AlpacaPaperTradingClient", _FakePaperClient)
    monkeypatch.setattr(mc_mod, "AlpacaLiveTradingClient", _FakeLiveClient)

    resp = client.get("/api/v1/mission-control/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "paper" in data and "live" in data
    assert data["paper"]["configured"] is True
    assert data["paper"]["positions"][0]["symbol"] == "AAPL"
    assert data["live"]["configured"] is True
    assert data["live"]["positions"][0]["symbol"] == "MSFT"


def test_live_generic_exception_is_sanitized_and_stays_200(isolated_auth, monkeypatch):
    """C20: anything other than AlpacaLiveCredentialsError must not 500 the
    whole response or leak the raw exception text -- it degrades to
    configured=False with the same sanitized message the frontend already
    knows how to render."""
    client, store = isolated_auth
    _make_admin(client, store, "mc-admin2@example.com")

    monkeypatch.setattr(mc_mod, "AlpacaPaperTradingClient", _FakePaperClient)
    monkeypatch.setattr(mc_mod, "AlpacaLiveTradingClient", _ExplodingLiveClient)

    resp = client.get("/api/v1/mission-control/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["live"]["configured"] is False
    assert data["live"]["error"] == "Failed to connect to live account"
    assert "TRACE-MARKER" not in str(data)
    # The paper half survives even though live blew up.
    assert data["paper"]["configured"] is True


def test_admin_overview_caches_within_ttl(isolated_auth, monkeypatch):
    """C21: two admin GETs inside the 30s TTL construct each broker client
    only once."""
    client, store = isolated_auth
    _make_admin(client, store, "mc-admin3@example.com")

    paper_calls = {"n": 0}
    live_calls = {"n": 0}

    class _CountingPaperClient(_FakePaperClient):
        def __init__(self):
            paper_calls["n"] += 1
            super().__init__()

    class _CountingLiveClient(_FakeLiveClient):
        def __init__(self):
            live_calls["n"] += 1
            super().__init__()

    monkeypatch.setattr(mc_mod, "AlpacaPaperTradingClient", _CountingPaperClient)
    monkeypatch.setattr(mc_mod, "AlpacaLiveTradingClient", _CountingLiveClient)

    first = client.get("/api/v1/mission-control/overview")
    second = client.get("/api/v1/mission-control/overview")

    assert first.status_code == 200
    assert second.status_code == 200
    assert paper_calls["n"] == 1
    assert live_calls["n"] == 1
    # And the two responses actually carry the (identical, cached) data.
    assert first.json()["paper"] == second.json()["paper"]
    assert first.json()["live"] == second.json()["live"]


def test_not_configured_snapshots_are_never_cached(isolated_auth, monkeypatch):
    """The error/not-configured branch must not be cached -- an operator
    dropping in credentials mid-session must see the fix on the very next
    request, not up to 30s later."""
    client, store = isolated_auth
    _make_admin(client, store, "mc-admin4@example.com")

    calls = {"n": 0}

    class _NotConfiguredThenFakeClient:
        def __new__(cls):
            calls["n"] += 1
            if calls["n"] == 1:
                raise mc_mod.AlpacaLiveCredentialsError("not configured")
            return _FakeLiveClient()

    monkeypatch.setattr(mc_mod, "AlpacaPaperTradingClient", _FakePaperClient)
    monkeypatch.setattr(mc_mod, "AlpacaLiveTradingClient", _NotConfiguredThenFakeClient)

    first = client.get("/api/v1/mission-control/overview")
    assert first.json()["live"]["configured"] is False

    second = client.get("/api/v1/mission-control/overview")
    assert second.json()["live"]["configured"] is True
    assert calls["n"] == 2


def test_mission_control_html_escapes_every_interpolated_value():
    """Source guard for C18: the three dynamic values reaching innerHTML
    (``snapshot.error``, ``p.symbol``, and the caught fetch error) must go
    through ``escapeHtml()`` rather than straight into string concatenation."""
    html = (FRONTEND_DIR / "mission-control.html").read_text(encoding="utf-8")
    assert "function escapeHtml(" in html
    for needle in ("escapeHtml(snapshot.error)", "escapeHtml(p.symbol)", "escapeHtml(e)"):
        assert needle in html, f"{needle} missing -- an interpolation into innerHTML is unescaped"
