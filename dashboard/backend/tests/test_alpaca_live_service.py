"""Risk-gate tests for the Alpaca live-money path.

Mirrors ``test_robinhood_live_service.py``'s risk-gate coverage: this is the
only other module in the repo whose orders can reach a real brokerage
account, so the same defect classes are pinned down here as pure-function
tests before anything ever talks to a broker.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from dashboard.backend.execution import alpaca_live_service as live_service
from dashboard.backend.infrastructure.brokers import alpaca_live as live_broker
from dashboard.backend.infrastructure.llm.validator import MAX_ORDER_SHARES

CAP_USD = 25.0


def test_missing_quote_is_rejected_not_passed_through():
    orders = [{"symbol": "ZZZZ", "side": "buy", "quantity": 10000}]
    accepted, rejected = live_service.risk_gate_orders(orders, prices={}, holdings={}, max_usd=CAP_USD)
    assert accepted == []
    assert rejected[0]["reason"] == "no_quote"


def test_buy_is_clamped_to_usd_cap():
    orders = [{"symbol": "AAPL", "side": "buy", "quantity": 1000}]
    accepted, rejected = live_service.risk_gate_orders(
        orders, prices={"AAPL": 200.0}, holdings={}, max_usd=CAP_USD
    )
    assert rejected == []
    assert accepted[0]["quantity"] == 0.125  # 25 / 200
    assert accepted[0]["notional_usd"] <= CAP_USD


def test_sell_with_no_position_is_rejected_never_shorts():
    orders = [{"symbol": "AAPL", "side": "sell", "quantity": 5}]
    accepted, rejected = live_service.risk_gate_orders(
        orders, prices={"AAPL": 200.0}, holdings={}, max_usd=CAP_USD
    )
    assert accepted == []
    assert rejected[0]["reason"] == "no_position"


def test_sell_is_clamped_to_held_quantity():
    orders = [{"symbol": "AAPL", "side": "sell", "quantity": 5}]
    accepted, _ = live_service.risk_gate_orders(
        orders, prices={"AAPL": 1.0}, holdings={"AAPL": 2.0}, max_usd=1000.0
    )
    # Both the USD cap (1000 shares) and the held quantity (2) apply; held wins.
    assert accepted[0]["quantity"] == 2.0


def test_share_cap_applies_even_under_a_huge_usd_cap():
    orders = [{"symbol": "AAPL", "side": "buy", "quantity": 999999}]
    accepted, _ = live_service.risk_gate_orders(
        orders, prices={"AAPL": 0.01}, holdings={}, max_usd=1_000_000.0
    )
    assert accepted[0]["quantity"] <= MAX_ORDER_SHARES


def test_residual_below_minimum_is_rejected():
    orders = [{"symbol": "BRK.A", "side": "buy", "quantity": 1}]
    accepted, rejected = live_service.risk_gate_orders(
        orders, prices={"BRK.A": 500_000.0}, holdings={}, max_usd=CAP_USD
    )
    assert accepted == []
    assert rejected[0]["reason"] == "below_min_quantity"


def test_execute_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("ALPACA_LIVE_EXECUTE", raising=False)
    assert live_service.execute_enabled() is False


def test_execute_enabled_requires_explicit_true(monkeypatch):
    monkeypatch.setenv("ALPACA_LIVE_EXECUTE", "true")
    assert live_service.execute_enabled() is True
    monkeypatch.setenv("ALPACA_LIVE_EXECUTE", "nope")
    assert live_service.execute_enabled() is False


def test_max_order_usd_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("ALPACA_MAX_ORDER_USD", "not-a-number")
    assert live_service.max_order_usd() == live_service.DEFAULT_MAX_ORDER_USD
    monkeypatch.setenv("ALPACA_MAX_ORDER_USD", "-5")
    assert live_service.max_order_usd() == live_service.DEFAULT_MAX_ORDER_USD
    monkeypatch.setenv("ALPACA_MAX_ORDER_USD", "50")
    assert live_service.max_order_usd() == 50.0


# ===========================================================================
# End-to-end: _execute_live_run with a fake AlpacaLiveTradingClient
# ===========================================================================


class _FakeLiveClient:
    """Stand-in for ``AlpacaLiveTradingClient``. Sync surface only (the real
    client is sync; ``_execute_live_run`` is what wraps it in ``to_thread``)."""

    def __init__(self, *, price: float = 100.0, prices: Optional[Dict[str, float]] = None, positions: Optional[Dict[str, float]] = None):
        self.price = price
        self.prices = prices or {}
        self.positions = positions or {}
        self.quote_calls: List[List[str]] = []
        self.order_calls: List[Dict[str, Any]] = []

    def get_account(self) -> Dict[str, Any]:
        return {"cash": 100000.0, "buying_power": 100000.0, "equity": 100000.0, "portfolio_value": 100000.0}

    def get_positions(self) -> Dict[str, float]:
        return dict(self.positions)

    def get_quotes(self, symbols: List[str]) -> Dict[str, float]:
        self.quote_calls.append(list(symbols))
        return {s: self.prices.get(s, self.price) for s in symbols}

    def submit_market_order(self, symbol: str, qty: float, side: str) -> "live_broker.LiveOrderResult":
        self.order_calls.append({"symbol": symbol, "qty": qty, "side": side})
        return live_broker.LiveOrderResult(
            order_id="ord_1",
            status="accepted",
            symbol=symbol,
            side=side,
            qty=qty,
            raw={
                "id": "ord_1",
                "status": "accepted",
                "symbol": symbol,
                "qty": str(qty),
                "side": side,
                "submitted_at": "2026-01-01T00:00:00Z",
                "filled_qty": None,
                "filled_avg_price": None,
                "filled_at": None,
            },
        )


def _action(symbol: str, side: str = "buy", size: int = 3) -> Dict[str, Any]:
    """A schema-valid ActionItem payload (DJIA-30 symbol required, 5+ char reasoning)."""
    return {
        "action": side,
        "symbol": symbol,
        "confidence": 0.8,
        "reasoning": f"unit test {side} {symbol}",
        "position_size": size,
    }


def _run_live(
    monkeypatch,
    tmp_path,
    client: "_FakeLiveClient",
    *,
    actions: List[Dict[str, Any]],
    symbols: Optional[List[str]] = None,
    dry_run: bool = True,
    execute: str = "false",
    max_usd: str = "1000",
) -> Dict[str, Any]:
    monkeypatch.setattr(live_service, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setenv("ALPACA_LIVE_EXECUTE", execute)
    monkeypatch.setenv("ALPACA_MAX_ORDER_USD", max_usd)
    monkeypatch.setattr(live_service, "AlpacaLiveTradingClient", lambda: client)

    async def _fake_llm_decision(*, model_name, instruction, portfolio, prices):
        return {"actions": [dict(a) for a in actions]}

    monkeypatch.setattr(live_service, "_llm_decision", _fake_llm_decision)
    return asyncio.run(
        live_service._execute_live_run(
            instruction="unit test", model_name=None, symbols=symbols, dry_run=dry_run
        )
    )


def test_non_djia_symbol_never_reaches_order_building(monkeypatch, tmp_path):
    """A symbol outside the DJIA-30 universe must be rejected by
    ``validate_actions`` before it ever reaches ``_actions_to_orders`` /
    ``risk_gate_orders`` -- even when it has a usable quote, which would have
    let it slip through the risk gate on its own."""
    client = _FakeLiveClient(prices={"ZZZZ": 100.0})
    result = _run_live(
        monkeypatch, tmp_path, client, actions=[_action("ZZZZ", size=5)], symbols=["ZZZZ"], dry_run=True
    )
    assert result["orders_reviewed"] == []
    assert client.order_calls == []
    assert any(r["reason"] == "universe_violation" for r in result["rejected_actions"])


def test_execute_flag_is_read_once_per_run(monkeypatch, tmp_path):
    """``execute_enabled()`` must be read exactly once per run: the ``dry_run``/
    ``execute_enabled`` fields in the result must describe the run that actually
    happened, not a value re-read (and possibly changed) after the fact."""
    client = _FakeLiveClient(prices={"AAPL": 200.0})
    calls = {"n": 0}

    def _flip_execute_enabled() -> bool:
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(live_service, "execute_enabled", _flip_execute_enabled)
    result = _run_live(
        monkeypatch, tmp_path, client, actions=[_action("AAPL", size=1)], symbols=["AAPL"], dry_run=False
    )
    assert calls["n"] == 1
    assert result["execute_enabled"] is True
    assert result["dry_run"] is False
    assert len(client.order_calls) == 1


# ===========================================================================
# Broker adapter: dashboard.backend.infrastructure.brokers.alpaca_live
# ===========================================================================


class _FakeOrder:
    def __init__(
        self,
        *,
        symbol: str,
        qty: float,
        side: Any,
        order_id: str = "ord_1",
        status: str = "accepted",
        filled_qty: Any = None,
        filled_avg_price: Any = None,
        filled_at: Any = None,
    ):
        self.id = order_id
        self.status = status
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.submitted_at = "2026-01-01T00:00:00Z"
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.filled_at = filled_at


class _FakeTradingClient:
    def __init__(self, api_key, secret_key, paper):
        self.submitted: List[Any] = []

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return _FakeOrder(symbol=order_data.symbol, qty=order_data.qty, side=order_data.side)


class _FakeDataClient:
    def __init__(self, api_key, secret_key):
        self.last_request = None

    def get_stock_latest_quote(self, request):
        self.last_request = request
        return {}


def _make_broker_client(monkeypatch) -> "live_broker.AlpacaLiveTradingClient":
    monkeypatch.setattr(live_broker, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(live_broker, "StockHistoricalDataClient", _FakeDataClient)
    return live_broker.AlpacaLiveTradingClient(api_key="key123", secret_key="secret456")


def test_submit_market_order_rejects_unknown_side(monkeypatch):
    client = _make_broker_client(monkeypatch)
    with pytest.raises(ValueError):
        client.submit_market_order("AAPL", 1.0, "short")


def test_submit_market_order_raw_carries_fill_fields_none_safely(monkeypatch):
    """A market order is typically 'accepted'/'pending_new' at submit time, so the
    fill fields legitimately come back None -- but the keys must be present and
    populated whenever the broker does return them synchronously."""
    client = _make_broker_client(monkeypatch)

    result = client.submit_market_order("AAPL", 1.0, "buy")
    assert result.raw["filled_qty"] is None
    assert result.raw["filled_avg_price"] is None
    assert result.raw["filled_at"] is None

    monkeypatch.setattr(live_broker, "TradingClient", _FakeTradingClient)
    monkeypatch.setattr(live_broker, "StockHistoricalDataClient", _FakeDataClient)
    filled_client = live_broker.AlpacaLiveTradingClient(api_key="key123", secret_key="secret456")
    filled_client._trading.submit_order = lambda order_data: _FakeOrder(
        symbol=order_data.symbol,
        qty=order_data.qty,
        side=order_data.side,
        filled_qty=1.0,
        filled_avg_price=101.5,
        filled_at="2026-01-01T00:00:05Z",
    )
    filled_result = filled_client.submit_market_order("AAPL", 1.0, "buy")
    assert filled_result.raw["filled_qty"] == "1.0"
    assert filled_result.raw["filled_avg_price"] == "101.5"
    assert filled_result.raw["filled_at"] == "2026-01-01T00:00:05Z"


def test_get_quotes_passes_the_configured_feed(monkeypatch):
    from alpaca.data.enums import DataFeed

    monkeypatch.delenv("ALPACA_DATA_FEED", raising=False)
    client = _make_broker_client(monkeypatch)
    client.get_quotes(["AAPL"])
    assert client._data.last_request.feed == DataFeed.SIP

    monkeypatch.setenv("ALPACA_DATA_FEED", "iex")
    client.get_quotes(["AAPL"])
    assert client._data.last_request.feed == DataFeed.IEX


def test_get_quotes_propagates_a_bad_feed_config(monkeypatch):
    from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaFeedConfigError

    monkeypatch.setenv("ALPACA_DATA_FEED", "not-a-real-feed")
    client = _make_broker_client(monkeypatch)
    with pytest.raises(AlpacaFeedConfigError):
        client.get_quotes(["AAPL"])
