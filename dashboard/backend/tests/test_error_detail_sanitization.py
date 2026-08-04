"""Exception text must not reach API response bodies (CodeQL py/stack-trace-exposure).

The ``{"error": str(e)}`` envelope in the paper-trading/market routers hands raw
exception text (filesystem paths, library internals, credential fragments) to
any unauthenticated caller — and #1098 already demonstrated those strings feed
frontend ``innerHTML`` sinks. Convention: ``print()`` the detail server-side
(the repo's real logging channel — ``logger.*`` is invisible under the deployed
config) and return a generic, route-specific message.
"""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token
from dashboard.backend.api.routers import market as market_mod
from dashboard.backend.api.routers import paper_trading as paper_mod
from dashboard.backend.domain.backtesting import algo_service

client = TestClient(app)

_MARKER = "TRACE-MARKER /opt/render/project/secret_config.py line 42"


class _ExplodingClient:
    def __init__(self):
        raise RuntimeError(_MARKER)


def test_ticker_error_body_hides_exception_detail(monkeypatch):
    def boom(symbols):
        raise RuntimeError(_MARKER)

    monkeypatch.setattr(market_mod, "get_market_quotes", boom)
    data = client.get("/ticker?symbols=AAPL").json()
    assert data["success"] is False
    assert "TRACE-MARKER" not in str(data)
    assert data["error"]  # still tells the client *something* went wrong


def test_paper_account_error_body_hides_exception_detail(monkeypatch):
    monkeypatch.setattr(paper_mod, "AlpacaPaperTradingClient", _ExplodingClient)
    # An account cached by an earlier test would short-circuit the handler
    # before the exploding client is ever constructed.
    paper_mod.paper_trading_cache.clear_all()
    data = client.get("/paper/account").json()
    assert data["success"] is False
    assert "TRACE-MARKER" not in str(data)


def test_paper_start_session_error_body_hides_exception_detail(monkeypatch):
    import tempfile
    from pathlib import Path

    import dashboard.backend.users as users_module

    monkeypatch.setattr(paper_mod, "AlpacaPaperTradingClient", _ExplodingClient)
    with tempfile.TemporaryDirectory() as tmpdir:
        store = users_module.UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        local = TestClient(app)
        signup = local.post(
            "/api/auth/signup",
            json={
                "email": "paper-err@example.com",
                "display_name": "Paper",
                "password": "securepass1",
            },
        )
        assert signup.status_code == 200
        token = _cookie_session_token(local)
        data = local.post(
            "/paper/start-session",
            headers={"Authorization": f"Bearer {token}"},
        ).json()
    assert data["success"] is False
    assert "TRACE-MARKER" not in str(data)


def test_llm_chat_fallback_reply_hides_exception_detail(monkeypatch):
    """``process_chat`` appended ``str(exc)`` to the user-visible reply."""

    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError(_MARKER)

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(algo_service, "_get_anthropic_client", lambda: _Client())
    result = algo_service.process_chat("use momentum", None)
    assert "TRACE-MARKER" not in result["reply"]
    # The degraded mode must still be communicated, just without internals.
    assert "fallback" in result["reply"].lower()


def test_routers_never_return_raw_exception_text():
    """Source guard: the ``"error": str(e)`` envelope pattern must stay gone."""
    routers = Path(market_mod.__file__).resolve().parent
    offenders = []
    for path in sorted(routers.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if re.search(r'"error":\s*str\(', src):
            offenders.append(path.name)
    assert offenders == []
