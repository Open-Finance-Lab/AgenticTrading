"""Characterization tests for the Alpaca paper-trading broker adapter (Phase 3C5A).

No real network/API calls: the ``requests`` module used by the adapter is
replaced with a fake that records calls and returns canned responses. Imports
use the canonical package path. No test contacts Alpaca or submits an order.
"""

import ast
import json
from pathlib import Path

import pytest

from dashboard.backend.infrastructure.brokers import alpaca_paper
from dashboard.backend.infrastructure.brokers.alpaca_paper import (
    AlpacaPaperTradingClient,
    Position,
)

_BACKEND = Path(__file__).resolve().parents[3]


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeRequests:
    """Records GET calls and returns queued responses."""

    def __init__(self):
        self.calls = []
        self.next_response = _FakeResponse(200, {})
        self.raise_exc = None

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "params": params, "timeout": timeout}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.next_response


@pytest.fixture
def fake_requests(monkeypatch):
    fake = _FakeRequests()
    monkeypatch.setattr(alpaca_paper, "requests", fake)
    return fake


@pytest.fixture
def client(monkeypatch):
    """A client with explicit credentials (no env/file lookup)."""
    return AlpacaPaperTradingClient(api_key="KEY123", secret_key="SECRET456")


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------

def test_explicit_credentials_used_verbatim():
    c = AlpacaPaperTradingClient(api_key="abc", secret_key="def")
    assert c.api_key == "abc"
    assert c.secret_key == "def"


def test_base_url_and_headers_unchanged():
    c = AlpacaPaperTradingClient(api_key="abc", secret_key="def")
    assert c.base_url == "https://paper-api.alpaca.markets"
    assert c.headers == {
        "APCA-API-KEY-ID": "abc",
        "APCA-API-SECRET-KEY": "def",
    }


def test_credentials_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "env-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "env-secret")
    c = AlpacaPaperTradingClient()
    assert c.api_key == "env-key"
    assert c.secret_key == "env-secret"


def test_credentials_fall_back_to_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_paper, "CREDENTIALS_DIR", tmp_path)
    (tmp_path / "alpaca.json").write_text(
        json.dumps({"api_key": "file-key", "secret_key": "file-secret"})
    )
    c = AlpacaPaperTradingClient()
    assert c.api_key == "file-key"
    assert c.secret_key == "file-secret"


def test_credentials_file_camelcase_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_paper, "CREDENTIALS_DIR", tmp_path)
    (tmp_path / "alpaca.json").write_text(
        json.dumps({"apiKey": "file-key", "secretKey": "file-secret"})
    )
    c = AlpacaPaperTradingClient()
    assert c.api_key == "file-key"
    assert c.secret_key == "file-secret"


def test_missing_credentials_raises_filenotfound(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_paper, "CREDENTIALS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        AlpacaPaperTradingClient()


# ---------------------------------------------------------------------------
# Account retrieval
# ---------------------------------------------------------------------------

def test_get_account_request_and_schema(client, fake_requests):
    fake_requests.next_response = _FakeResponse(
        200,
        {
            "cash": "1000.5",
            "equity": "2000.25",
            "buying_power": "4000",
            "portfolio_value": "2000.25",
            "multiplier": "4",
            "account_number": "PA123",
            "status": "ACTIVE",
            "created_at": "2024-01-01",
        },
    )
    result = client.get_account()

    call = fake_requests.calls[0]
    assert call["url"] == "https://paper-api.alpaca.markets/v2/account"
    assert call["headers"] == client.headers
    assert call["timeout"] == 10
    assert call["params"] is None

    assert result == {
        "cash": 1000.5,
        "equity": 2000.25,
        "buying_power": 4000.0,
        "portfolio_value": 2000.25,
        "multiplier": 4,
        "account_number": "PA123",
        "account_status": "ACTIVE",
        "created_at": "2024-01-01",
    }


def test_get_account_non_200_returns_none(client, fake_requests):
    fake_requests.next_response = _FakeResponse(401, None)
    assert client.get_account() is None


def test_get_account_exception_returns_none(client, fake_requests):
    fake_requests.raise_exc = RuntimeError("boom")
    assert client.get_account() is None


# ---------------------------------------------------------------------------
# Positions retrieval
# ---------------------------------------------------------------------------

def test_get_positions_returns_position_objects(client, fake_requests):
    fake_requests.next_response = _FakeResponse(
        200,
        [
            {
                "symbol": "AAPL",
                "qty": "10",
                "avg_fill_price": "150.0",
                "current_price": "155.0",
                "unrealized_pl": "50.0",
                "unrealized_plpc": "0.033",
                "side": "long",
                "market_value": "1550.0",
            }
        ],
    )
    positions = client.get_positions()

    call = fake_requests.calls[0]
    assert call["url"] == "https://paper-api.alpaca.markets/v2/positions"
    assert call["timeout"] == 10

    assert len(positions) == 1
    pos = positions[0]
    assert isinstance(pos, Position)
    assert pos.symbol == "AAPL"
    assert pos.qty == 10.0
    assert pos.avg_fill_price == 150.0
    assert pos.current_price == 155.0
    assert pos.unrealized_pl == 50.0
    assert pos.unrealized_plpc == 0.033
    assert pos.side == "long"
    assert pos.market_value == 1550.0


def test_get_positions_non_200_returns_empty(client, fake_requests):
    fake_requests.next_response = _FakeResponse(500, None)
    assert client.get_positions() == []


def test_get_positions_exception_returns_empty(client, fake_requests):
    fake_requests.raise_exc = ValueError("nope")
    assert client.get_positions() == []


# ---------------------------------------------------------------------------
# Orders / activities / portfolio history retrieval
# ---------------------------------------------------------------------------

def test_get_orders_params_and_passthrough(client, fake_requests):
    payload = [{"id": "o1"}]
    fake_requests.next_response = _FakeResponse(200, payload)
    result = client.get_orders(limit=25, status="open")

    call = fake_requests.calls[0]
    assert call["url"] == "https://paper-api.alpaca.markets/v2/orders"
    assert call["params"] == {"limit": 25, "status": "open"}
    assert call["timeout"] == 10
    assert result == payload


def test_get_orders_defaults(client, fake_requests):
    fake_requests.next_response = _FakeResponse(200, [])
    client.get_orders()
    assert fake_requests.calls[0]["params"] == {"limit": 50, "status": "all"}


def test_get_orders_non_200_returns_empty(client, fake_requests):
    fake_requests.next_response = _FakeResponse(403, None)
    assert client.get_orders() == []


def test_get_activities_params_and_passthrough(client, fake_requests):
    payload = [{"id": "a1", "symbol": "AAPL"}]
    fake_requests.next_response = _FakeResponse(200, payload)
    result = client.get_activities(activity_type="FILL", limit=5)

    call = fake_requests.calls[0]
    assert call["url"] == "https://paper-api.alpaca.markets/v2/account/activities"
    assert call["params"] == {"activity_type": "FILL", "limit": 5}
    assert call["timeout"] == 10
    assert result == payload


def test_get_activities_defaults(client, fake_requests):
    fake_requests.next_response = _FakeResponse(200, [])
    client.get_activities()
    assert fake_requests.calls[0]["params"] == {"activity_type": "FILL", "limit": 100}


def test_get_activities_non_200_returns_empty(client, fake_requests):
    fake_requests.next_response = _FakeResponse(500, None)
    assert client.get_activities() == []


def test_get_portfolio_history_params_and_passthrough(client, fake_requests):
    payload = {"equity": [1, 2], "timestamp": [10, 20]}
    fake_requests.next_response = _FakeResponse(200, payload)
    result = client.get_portfolio_history(timeframe="1W")

    call = fake_requests.calls[0]
    assert call["url"] == (
        "https://paper-api.alpaca.markets/v2/account/portfolio/history"
    )
    assert call["params"] == {"timeframe": "1W"}
    assert call["timeout"] == 10
    assert result == payload


def test_get_portfolio_history_default_timeframe(client, fake_requests):
    fake_requests.next_response = _FakeResponse(200, {})
    client.get_portfolio_history()
    assert fake_requests.calls[0]["params"] == {"timeframe": "1D"}


def test_get_portfolio_history_non_200_returns_none(client, fake_requests):
    fake_requests.next_response = _FakeResponse(500, None)
    assert client.get_portfolio_history() is None


# ---------------------------------------------------------------------------
# Runtime consumers + import boundaries
# ---------------------------------------------------------------------------

def _imported_modules(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_app_uses_canonical_broker_adapter():
    # Phase 3C5C: the paper routes moved to the canonical router, which is now
    # the runtime consumer of the broker adapter.
    mods = _imported_modules(
        _BACKEND / "api" / "routers" / "paper_trading.py"
    )
    assert "dashboard.backend.infrastructure.brokers.alpaca_paper" in mods


def test_paper_baselines_uses_canonical_broker_adapter():
    # Phase 3C5B: the calculator moved to the canonical baselines package, which
    # is the real runtime consumer of the broker adapter.
    mods = _imported_modules(
        _BACKEND / "domain" / "backtesting" / "baselines" / "paper.py"
    )
    assert "dashboard.backend.infrastructure.brokers.alpaca_paper" in mods


def test_broker_module_has_no_api_or_scripts_imports():
    mods = _imported_modules(
        _BACKEND / "infrastructure" / "brokers" / "alpaca_paper.py"
    )
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
