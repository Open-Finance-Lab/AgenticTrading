"""Risk-gate tests for the Alpaca live-money path.

Mirrors ``test_robinhood_live_service.py``'s risk-gate coverage: this is the
only other module in the repo whose orders can reach a real brokerage
account, so the same defect classes are pinned down here as pure-function
tests before anything ever talks to a broker.
"""

from __future__ import annotations

from dashboard.backend.execution import alpaca_live_service as live_service
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
