"""Risk-gate, quote-universe, review/execution and concurrency tests for the
Robinhood live path.

This is the only module in the repo whose orders can reach a real brokerage
account, so every gate it relies on is tested as a pure function first and then
once end-to-end through :func:`_execute_live_run` with a fake MCP client. There
is no network, no broker and no database access anywhere in this file.

The defects these tests pin down, in the order they appear below:

* a symbol with no usable quote used to be passed through the risk gate
  *unmodified* -- a 10,000-share order shipping under a nominal $25 cap;
* only ``buy`` orders were capped, so every sell was unbounded, and a sell for
  a symbol with no position opened a naked short;
* the quote call truncates at 20 symbols, so 10 of the 30 DJIA names reached
  the gate priceless -- which fed straight into the bug above;
* the pre-trade review was logged and ignored, and every placement was recorded
  as ``submitted`` regardless of what the broker answered;
* the notional cap was a module constant frozen at import, so tightening it
  needed a restart;
* concurrent runs for one user could interleave, and a retried request placed
  its orders twice;
* a missing LLM client produced fabricated hold actions instead of an error.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Callable, Dict, List, Optional

import pytest

from dashboard.backend.execution import robinhood_live_service as live_service
from dashboard.backend.infrastructure.llm.validator import DJIA_30, MAX_ORDER_SHARES

AGENT = {"agent_id": "agent_live_1", "model_name": "test-model", "live_trading_enabled": True}

#: Ceiling for any await that must not block. Everything in this file is
#: in-memory, so exceeding it means a lock was taken that should not have been.
_BLOCK_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _reset_live_state(monkeypatch, tmp_path):
    """Module-global locks / idempotency cache are process-wide; reset them, and
    keep audit writes inside the test's tmp dir rather than the repo."""
    live_service._user_locks.clear()
    live_service._idempotency_cache.clear()
    monkeypatch.setattr(live_service, "AUDIT_DIR", tmp_path / "audit")
    yield
    live_service._user_locks.clear()
    live_service._idempotency_cache.clear()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMCPClient:
    """Stand-in for ``RobinhoodMCPClient``. Async surface only, no transport.

    ``get_equity_quotes`` truncates at 20 exactly like the real client so the
    batching tests measure the real constraint rather than a friendlier fake.
    """

    def __init__(
        self,
        *,
        price: float = 100.0,
        prices: Optional[Dict[str, Any]] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        reviews: Optional[Callable[[Dict[str, Any]], Any]] = None,
        places: Optional[Callable[[Dict[str, Any]], Any]] = None,
        quotes_raise_for: Optional[str] = None,
    ):
        self.price = price
        self.prices = prices or {}
        self.positions = positions or []
        self._reviews = reviews
        self._places = places
        self._quotes_raise_for = quotes_raise_for
        self.quote_batches: List[List[str]] = []
        self.review_calls: List[Dict[str, Any]] = []
        self.place_calls: List[Dict[str, Any]] = []

    async def get_portfolio(self) -> Dict[str, Any]:
        return {"buying_power": 100000.0, "portfolio_value": 250000.0}

    async def get_equity_positions(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self.positions]

    async def get_equity_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        batch = list(symbols)
        self.quote_batches.append(batch)
        if self._quotes_raise_for and self._quotes_raise_for in batch:
            raise RuntimeError("quote tool unavailable")
        return [
            {"symbol": symbol, "last_trade_price": self.prices.get(symbol, self.price)}
            for symbol in batch[:20]  # mirrors RobinhoodMCPClient's symbols[:20]
        ]

    async def review_equity_order(self, order: Dict[str, Any]) -> Any:
        self.review_calls.append(dict(order))
        if self._reviews is None:
            return {"approved": True, "status": "ok"}
        return self._reviews(order)

    async def place_equity_order(self, order: Dict[str, Any]) -> Any:
        self.place_calls.append(dict(order))
        if self._places is None:
            return {"id": "ord_ok", "status": "queued"}
        return self._places(order)


def _action(symbol: str, side: str = "buy", size: int = 3) -> Dict[str, Any]:
    """A schema-valid ActionItem payload (DJIA-30 symbol, 5+ char reasoning)."""
    return {
        "action": side,
        "symbol": symbol,
        "confidence": 0.8,
        "reasoning": f"unit test {side} {symbol}",
        "position_size": size,
    }


def _install_fakes(
    monkeypatch,
    client: FakeMCPClient,
    *,
    actions: List[Dict[str, Any]],
    execute: str = "true",
    max_usd: str = "1000",
) -> None:
    monkeypatch.setenv("ROBINHOOD_EXECUTE", execute)
    monkeypatch.setenv("ROBINHOOD_MAX_ORDER_USD", max_usd)
    monkeypatch.setattr(live_service, "RobinhoodMCPClient", lambda access_token: client)
    monkeypatch.setattr(live_service, "_ensure_access_token", lambda uid: {"access_token": "tok"})

    async def _fake_refresh(user_id, tokens):
        return "tok"

    monkeypatch.setattr(live_service, "_refresh_if_needed", _fake_refresh)

    async def _fake_decision(*, agent, market_snapshot, portfolio_state):
        return {"actions": [dict(a) for a in actions]}

    monkeypatch.setattr(live_service, "_llm_decision", _fake_decision)


def _run_live(monkeypatch, client, *, actions, dry_run=False, execute="true", max_usd="1000", user_id=1):
    _install_fakes(monkeypatch, client, actions=actions, execute=execute, max_usd=max_usd)
    return asyncio.run(
        live_service._execute_live_run(user_id=user_id, agent=dict(AGENT), dry_run=dry_run)
    )


def _snapshot(**prices: Any) -> Dict[str, Any]:
    return {"symbols": {sym: {"price": price} for sym, price in prices.items()}}


def _order(symbol: str, side: str, quantity: float) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": "market",
        "time_in_force": "gfd",
    }


# ===========================================================================
# A. Risk gate -- pure function, no network
# ===========================================================================


def test_buy_is_clamped_to_the_notional_cap():
    accepted, rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 100.0)],
        _snapshot(AAPL=100.0),
        {"cash": 50000.0, "holdings": {}},
        25.0,
    )
    assert rejections == []
    assert len(accepted) == 1
    assert accepted[0]["quantity"] == 0.25  # $25 cap / $100 price
    assert accepted[0]["notional_usd"] == 25.0


def test_symbol_without_a_quote_is_rejected_not_passed_through():
    """The multi-million-dollar bypass: no price used to mean no clamping."""
    orders = [_order("AAPL", "buy", 10000.0), _order("MSFT", "buy", 1.0)]
    accepted, rejections = live_service._risk_gate_orders(
        orders,
        {"symbols": {"AAPL": {"price": None}, "MSFT": {"price": 100.0}}},
        {"cash": 50000.0, "holdings": {}},
        25.0,
    )
    assert [o["symbol"] for o in accepted] == ["MSFT"]
    assert all(o["symbol"] != "AAPL" for o in accepted)
    assert [r["reason"] for r in rejections] == ["no_quote"]
    assert rejections[0]["order"]["symbol"] == "AAPL"


def test_symbol_absent_from_the_snapshot_is_rejected():
    accepted, rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 10000.0)],
        {"symbols": {}},
        {"cash": 50000.0, "holdings": {}},
        25.0,
    )
    assert accepted == []
    assert rejections[0]["reason"] == "no_quote"


@pytest.mark.parametrize("bad_price", ["abc", "", None, 0, 0.0, -5.0, float("nan"), float("inf"), [1]])
def test_non_numeric_or_non_positive_price_is_rejected(bad_price):
    accepted, rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 10000.0)],
        {"symbols": {"AAPL": {"price": bad_price}}},
        {"cash": 50000.0, "holdings": {}},
        25.0,
    )
    assert accepted == []
    assert rejections[0]["reason"] == "no_quote"


def test_sell_is_capped_by_notional_exactly_like_a_buy():
    """The old gate only clamped ``buy``; every sell went out unbounded."""
    accepted, rejections = live_service._risk_gate_orders(
        [_order("MSFT", "sell", 40.0)],
        _snapshot(MSFT=100.0),
        {"cash": 0.0, "holdings": {"MSFT": 50.0}},
        25.0,
    )
    assert rejections == []
    assert accepted[0]["quantity"] == 0.25
    assert accepted[0]["notional_usd"] == 25.0


def test_sell_without_a_position_is_rejected():
    accepted, rejections = live_service._risk_gate_orders(
        [_order("MSFT", "sell", 10.0)],
        _snapshot(MSFT=100.0),
        {"cash": 0.0, "holdings": {"AAPL": 5.0}},
        100000.0,
    )
    assert accepted == []
    assert rejections[0]["reason"] == "no_position"


@pytest.mark.parametrize("held", [0, 0.0, "0", None, "not-a-number"])
def test_sell_with_an_unusable_holding_is_rejected(held):
    accepted, rejections = live_service._risk_gate_orders(
        [_order("MSFT", "sell", 10.0)],
        _snapshot(MSFT=100.0),
        {"cash": 0.0, "holdings": {"MSFT": held}},
        100000.0,
    )
    assert accepted == []
    assert rejections[0]["reason"] == "no_position"


def test_sell_larger_than_the_position_is_clamped_to_the_position():
    accepted, rejections = live_service._risk_gate_orders(
        [_order("MSFT", "sell", 10.0)],
        _snapshot(MSFT=1.0),
        {"cash": 0.0, "holdings": {"MSFT": 0.1}},
        100000.0,
    )
    assert rejections == []
    assert accepted[0]["quantity"] == 0.1


def test_max_order_shares_still_bounds_a_cheap_symbol():
    """A cent-priced symbol makes the USD cap permissive; the share ceiling holds."""
    accepted, rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 50000.0)],
        _snapshot(AAPL=0.001),
        {"cash": 50000.0, "holdings": {}},
        25.0,  # 25 / 0.001 == 25,000 shares, above MAX_ORDER_SHARES
    )
    assert rejections == []
    assert accepted[0]["quantity"] == float(MAX_ORDER_SHARES)
    assert accepted[0]["quantity"] * 0.001 <= 25.0


def test_sub_minimum_quantity_is_rejected():
    accepted, rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 5.0)],
        _snapshot(AAPL=1_000_000.0),
        {"cash": 50000.0, "holdings": {}},
        25.0,
    )
    assert accepted == []
    assert rejections[0]["reason"] == "below_min_quantity"


@pytest.mark.parametrize(
    "price", [0.007, 0.33, 1.0, 3.0, 7.77, 13.0, 99.99, 123.456, 1000.0, 33333.33]
)
@pytest.mark.parametrize("side", ["buy", "sell"])
def test_accepted_notional_never_exceeds_the_cap(price, side):
    """Guard against rounding slop: the quantity is floored, never rounded up."""
    cap = 25.0
    accepted, _rejections = live_service._risk_gate_orders(
        [_order("AAPL", side, 1_000_000.0)],
        _snapshot(AAPL=price),
        {"cash": 0.0, "holdings": {"AAPL": 1_000_000.0}},
        cap,
    )
    for order in accepted:
        assert order["quantity"] * price <= cap
        assert order["notional_usd"] <= cap
        assert order["quantity"] >= live_service.MIN_ORDER_QUANTITY


@pytest.mark.parametrize("bad_cap", [None, "abc", 0, -1.0, float("nan"), float("inf")])
def test_unusable_cap_falls_back_to_the_default(bad_cap):
    """An unparseable cap must not disable the cap."""
    accepted, _rejections = live_service._risk_gate_orders(
        [_order("AAPL", "buy", 10000.0)],
        _snapshot(AAPL=100.0),
        {"cash": 50000.0, "holdings": {}},
        bad_cap,
    )
    assert accepted[0]["quantity"] * 100.0 <= live_service.DEFAULT_MAX_ORDER_USD


def test_risk_gate_does_not_mutate_the_input_order():
    original = _order("AAPL", "buy", 100.0)
    snapshot = _snapshot(AAPL=100.0)
    accepted, _ = live_service._risk_gate_orders(
        [original], snapshot, {"cash": 1.0, "holdings": {}}, 25.0
    )
    assert original["quantity"] == 100.0  # the rejection record must quote the request
    assert accepted[0] is not original


# ===========================================================================
# B. Quote universe -- every offered symbol must be priced
# ===========================================================================


def test_chunk_symbols_never_exceeds_the_tool_limit():
    symbols = [f"S{i}" for i in range(47)]
    batches = live_service._chunk_symbols(symbols)
    assert all(len(batch) <= 20 for batch in batches)
    assert all(len(batch) <= live_service.QUOTE_BATCH_SIZE for batch in batches)
    assert [s for batch in batches for s in batch] == symbols  # order + membership preserved
    assert live_service.QUOTE_BATCH_SIZE <= 20


def test_chunk_symbols_handles_empty_and_degenerate_sizes():
    assert live_service._chunk_symbols([]) == []
    batches = live_service._chunk_symbols(["A", "B", "C"], size=0)
    assert batches == [["A", "B", "C"]]  # a non-positive size falls back, never loops


def test_fetch_market_snapshot_prices_the_whole_djia_universe():
    """A single quote call drops everything past the 20th symbol."""
    client = FakeMCPClient(price=42.0)
    universe = sorted(DJIA_30)
    snapshot = asyncio.run(live_service._fetch_market_snapshot(client, universe))

    assert len(universe) == 30
    assert all(len(batch) <= 20 for batch in client.quote_batches)
    assert sorted(s for batch in client.quote_batches for s in batch) == universe
    assert set(snapshot["symbols"]) == set(universe)
    assert all(snapshot["symbols"][sym]["price"] == 42.0 for sym in universe)
    # WMT sorts 30th: under the old single-call behaviour it came back priceless.
    assert snapshot["symbols"]["WMT"]["price"] == 42.0
    assert "quote_errors" not in snapshot


def test_fetch_market_snapshot_survives_a_failing_batch():
    """One dead batch must not blank the others -- and its symbols stay
    price-less so the risk gate rejects them rather than passing them through."""
    universe = sorted(DJIA_30)
    client = FakeMCPClient(price=42.0, quotes_raise_for=universe[25])
    snapshot = asyncio.run(live_service._fetch_market_snapshot(client, universe))

    assert set(snapshot["symbols"]) == set(universe)
    assert snapshot["symbols"][universe[0]]["price"] == 42.0
    assert snapshot["symbols"][universe[25]]["price"] is None
    assert snapshot["quote_errors"]

    accepted, rejections = live_service._risk_gate_orders(
        [_order(universe[25], "buy", 9999.0)], snapshot, {"holdings": {}}, 25.0
    )
    assert accepted == []
    assert rejections[0]["reason"] == "no_quote"


def test_live_run_quotes_every_symbol_it_offers_the_model(monkeypatch):
    """End-to-end: an order for the alphabetically-last DJIA name must price."""
    client = FakeMCPClient(price=100.0)
    result = _run_live(monkeypatch, client, actions=[_action("WMT")])

    assert all(len(batch) <= 20 for batch in client.quote_batches)
    quoted = {s for batch in client.quote_batches for s in batch}
    assert set(DJIA_30) <= quoted
    assert result["rejected_orders"] == []
    assert [e["status"] for e in result["executions"]] == ["submitted"]


# ===========================================================================
# C. Pre-trade review + execution status
# ===========================================================================


@pytest.mark.parametrize(
    "review",
    [
        {"approved": False},
        {"allowed": False},
        {"can_place": False},
        {"is_valid": False},
        {"error": "insufficient buying power"},
        {"errors": ["insufficient buying power"]},
        {"rejection_reason": "restricted symbol"},
        {"status": "rejected"},
        {"state": "denied"},
        {"status": "failed"},
    ],
)
def test_negative_review_prevents_placement_entirely(monkeypatch, review):
    client = FakeMCPClient(reviews=lambda order: review)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")])

    assert client.review_calls, "the review must actually be requested"
    assert client.place_calls == [], "a refused review must not reach place_equity_order"
    assert [e["status"] for e in result["executions"]] == ["skipped"]
    assert result["executions"][0]["reason"] == "review_rejected"


def test_empty_review_prevents_placement(monkeypatch):
    client = FakeMCPClient(reviews=lambda order: None)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")])
    assert client.place_calls == []
    assert result["executions"][0]["reason"] == "review_empty"


def test_review_that_raises_prevents_placement(monkeypatch):
    def _boom(order):
        raise RuntimeError("mcp review tool unavailable")

    client = FakeMCPClient(reviews=_boom)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")])

    assert client.place_calls == []
    assert result["executions"][0]["status"] == "skipped"
    assert result["executions"][0]["reason"] == "review_failed"
    assert result["orders_reviewed"][0]["review"] is None


def test_approving_review_allows_placement(monkeypatch):
    client = FakeMCPClient(reviews=lambda order: {"approved": True})
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")])
    assert len(client.place_calls) == 1
    assert result["executions"][0]["status"] == "submitted"


@pytest.mark.parametrize(
    "response",
    [
        {"status": "rejected", "rejection_reason": "market closed"},
        {"state": "failed"},
        {"id": "ord_1", "errors": ["insufficient funds"]},
        {"approved": False, "id": "ord_1"},
    ],
)
def test_rejected_placement_is_recorded_as_rejected(monkeypatch, response):
    """Recording every placement as ``submitted`` made the audit trail a lie."""
    client = FakeMCPClient(places=lambda order: response)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")])
    assert len(client.place_calls) == 1
    assert result["executions"][0]["status"] == "rejected"


def test_failed_placement_does_not_abort_the_remaining_orders(monkeypatch):
    def _places(order):
        if order["symbol"] == "AAPL":
            raise RuntimeError("connection reset")
        return {"id": "ord_2", "status": "queued"}

    client = FakeMCPClient(places=_places)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL"), _action("MSFT")])

    assert [o["symbol"] for o in client.place_calls] == ["AAPL", "MSFT"]
    statuses = [(e["order"]["symbol"], e["status"]) for e in result["executions"]]
    assert statuses == [("AAPL", "failed"), ("MSFT", "submitted")]
    assert result["executions"][0]["reason"] == "place_order_failed"


def test_dry_run_never_places(monkeypatch):
    client = FakeMCPClient()
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")], dry_run=True)
    assert client.place_calls == []
    assert result["dry_run"] is True
    assert result["executions"][0]["reason"] == "dry_run_or_execute_disabled"


def test_kill_switch_off_never_places(monkeypatch):
    client = FakeMCPClient()
    result = _run_live(monkeypatch, client, actions=[_action("AAPL")], execute="false")
    assert client.place_calls == []
    assert result["execute_enabled"] is False


@pytest.mark.parametrize(
    "review, blocked, reason",
    [
        (None, True, "review_empty"),
        ({"approved": False}, True, "review_rejected"),
        ({"status": "rejected"}, True, "review_rejected"),
        ({"approved": True}, False, ""),
        ({"status": "ok"}, False, ""),
        ({}, False, ""),
        ("unexpected string", False, ""),
    ],
)
def test_review_blocks_order_matrix(review, blocked, reason):
    assert live_service._review_blocks_order(review) == (blocked, reason)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"id": "ord_1"}, "submitted"),
        ({"order_id": "ord_1"}, "submitted"),
        ({"status": "queued"}, "submitted"),
        ({"approved": True}, "submitted"),
        ({"status": "rejected"}, "rejected"),
        ({"error": "nope"}, "rejected"),
        ({"order": {"status": "rejected"}}, "rejected"),
        ({"result": {"id": "ord_1"}}, "submitted"),
        ({}, "unknown"),
        (None, "unknown"),
        ("not a dict", "unknown"),
    ],
)
def test_execution_status_matrix(payload, expected):
    assert live_service._execution_status(payload) == expected


# ===========================================================================
# D. max_order_usd() is read per call, not frozen at import
# ===========================================================================


def test_max_order_usd_is_read_from_the_environment_per_call(monkeypatch):
    monkeypatch.delenv("ROBINHOOD_MAX_ORDER_USD", raising=False)
    assert live_service.max_order_usd() == live_service.DEFAULT_MAX_ORDER_USD

    monkeypatch.setenv("ROBINHOOD_MAX_ORDER_USD", "5")
    assert live_service.max_order_usd() == 5.0

    # Tightening the cap must take effect without a restart or re-import.
    monkeypatch.setenv("ROBINHOOD_MAX_ORDER_USD", "1.5")
    assert live_service.max_order_usd() == 1.5


@pytest.mark.parametrize("raw", ["", "   ", "abc", "0", "-10", "nan", "inf", "-inf"])
def test_unusable_max_order_usd_clamps_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("ROBINHOOD_MAX_ORDER_USD", raw)
    assert live_service.max_order_usd() == live_service.DEFAULT_MAX_ORDER_USD


def test_no_frozen_module_level_cap_constant():
    """The old module read the env var once at import into ``MAX_ORDER_USD``."""
    assert not hasattr(live_service, "MAX_ORDER_USD")


def test_execute_enabled_is_read_per_call(monkeypatch):
    monkeypatch.setenv("ROBINHOOD_EXECUTE", "false")
    assert live_service.execute_enabled() is False
    monkeypatch.setenv("ROBINHOOD_EXECUTE", "true")
    assert live_service.execute_enabled() is True
    monkeypatch.delenv("ROBINHOOD_EXECUTE", raising=False)
    assert live_service.execute_enabled() is False


def test_live_run_applies_the_current_cap(monkeypatch):
    """The cap that reaches the risk gate is the one set *now*."""
    client = FakeMCPClient(price=100.0)
    result = _run_live(monkeypatch, client, actions=[_action("AAPL", size=9)], max_usd="200")
    assert result["max_order_usd"] == 200.0
    assert client.place_calls[0]["quantity"] == 2.0  # $200 cap / $100 price

    client2 = FakeMCPClient(price=100.0)
    result2 = _run_live(monkeypatch, client2, actions=[_action("AAPL", size=9)], max_usd="500")
    assert result2["max_order_usd"] == 500.0
    assert client2.place_calls[0]["quantity"] == 5.0


# ===========================================================================
# E. Concurrency + idempotency
# ===========================================================================


def test_second_concurrent_run_for_the_same_user_is_rejected(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_run(*, user_id, agent, dry_run):
            started.set()
            await release.wait()
            return {"run_id": "run_first", "status": "completed"}

        monkeypatch.setattr(live_service, "_execute_live_run", _slow_run)
        task = asyncio.create_task(
            live_service.run_live_for_agent(user_id=21, agent=dict(AGENT), dry_run=False)
        )
        await started.wait()
        # wait_for, not a bare await: without the busy check the second call
        # would *block* on the lock instead of raising, and a test that hangs
        # forever reports nothing. Fail fast so the regression is visible.
        with pytest.raises(ValueError) as exc:
            await asyncio.wait_for(
                live_service.run_live_for_agent(user_id=21, agent=dict(AGENT), dry_run=False),
                timeout=_BLOCK_TIMEOUT,
            )
        assert str(exc.value) == "live_run_in_progress"
        release.set()
        first = await asyncio.wait_for(task, timeout=_BLOCK_TIMEOUT)
        assert first["run_id"] == "run_first"

    asyncio.run(scenario())


def test_a_different_user_is_not_blocked_by_another_users_run(monkeypatch):
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_run(*, user_id, agent, dry_run):
            if user_id == 31:
                started.set()
                await release.wait()
            return {"run_id": f"run_{user_id}", "status": "completed"}

        monkeypatch.setattr(live_service, "_execute_live_run", _slow_run)
        task = asyncio.create_task(
            live_service.run_live_for_agent(user_id=31, agent=dict(AGENT), dry_run=False)
        )
        await started.wait()
        # A single process-wide lock (rather than one per user) would block here.
        other = await asyncio.wait_for(
            live_service.run_live_for_agent(user_id=32, agent=dict(AGENT), dry_run=False),
            timeout=_BLOCK_TIMEOUT,
        )
        assert other["run_id"] == "run_32"
        release.set()
        await asyncio.wait_for(task, timeout=_BLOCK_TIMEOUT)

    asyncio.run(scenario())


def test_lock_is_released_when_a_run_raises(monkeypatch):
    async def scenario():
        async def _boom(*, user_id, agent, dry_run):
            raise RuntimeError("broker exploded")

        monkeypatch.setattr(live_service, "_execute_live_run", _boom)
        with pytest.raises(RuntimeError):
            await live_service.run_live_for_agent(user_id=41, agent=dict(AGENT), dry_run=False)
        # A leaked lock would wedge the account until the process restarts.
        assert live_service._user_lock(41).locked() is False

    asyncio.run(scenario())


def test_repeated_idempotency_key_replays_without_placing_twice(monkeypatch):
    client = FakeMCPClient()
    _install_fakes(monkeypatch, client, actions=[_action("AAPL")])
    agent = dict(AGENT)

    async def scenario():
        first = await live_service.run_live_for_agent(
            user_id=51, agent=agent, dry_run=False, idempotency_key="key-1"
        )
        second = await live_service.run_live_for_agent(
            user_id=51, agent=agent, dry_run=False, idempotency_key="key-1"
        )
        third = await live_service.run_live_for_agent(
            user_id=51, agent=agent, dry_run=False, idempotency_key="key-2"
        )
        return first, second, third

    first, second, third = asyncio.run(scenario())

    assert len(client.place_calls) == 2, "the replay must not re-place the order"
    assert second["idempotent_replay"] is True
    assert second["run_id"] == first["run_id"]
    assert first.get("idempotent_replay") is None
    assert third.get("idempotent_replay") is None
    assert third["run_id"] != first["run_id"]


def test_idempotency_is_scoped_per_user(monkeypatch):
    client = FakeMCPClient()
    _install_fakes(monkeypatch, client, actions=[_action("AAPL")])

    async def scenario():
        a = await live_service.run_live_for_agent(
            user_id=61, agent=dict(AGENT), dry_run=False, idempotency_key="shared"
        )
        b = await live_service.run_live_for_agent(
            user_id=62, agent=dict(AGENT), dry_run=False, idempotency_key="shared"
        )
        return a, b

    a, b = asyncio.run(scenario())
    assert b.get("idempotent_replay") is None
    assert a["run_id"] != b["run_id"]
    assert len(client.place_calls) == 2


def test_idempotency_cache_is_bounded(monkeypatch):
    async def scenario():
        async def _fast(*, user_id, agent, dry_run):
            return {"run_id": "r", "status": "completed"}

        monkeypatch.setattr(live_service, "_execute_live_run", _fast)
        for i in range(live_service.IDEMPOTENCY_MAX_ENTRIES + 25):
            await live_service.run_live_for_agent(
                user_id=71, agent=dict(AGENT), dry_run=False, idempotency_key=f"k{i}"
            )

    asyncio.run(scenario())
    assert len(live_service._idempotency_cache) <= live_service.IDEMPOTENCY_MAX_ENTRIES


def test_expired_idempotency_entries_are_dropped(monkeypatch):
    async def scenario():
        calls = []

        async def _count(*, user_id, agent, dry_run):
            calls.append(user_id)
            return {"run_id": f"r{len(calls)}", "status": "completed"}

        monkeypatch.setattr(live_service, "_execute_live_run", _count)
        await live_service.run_live_for_agent(
            user_id=81, agent=dict(AGENT), dry_run=False, idempotency_key="k"
        )
        monkeypatch.setattr(live_service, "IDEMPOTENCY_TTL_SECONDS", -1.0)
        second = await live_service.run_live_for_agent(
            user_id=81, agent=dict(AGENT), dry_run=False, idempotency_key="k"
        )
        assert second.get("idempotent_replay") is None
        assert len(calls) == 2

    asyncio.run(scenario())


def test_run_requires_live_trading_enabled():
    with pytest.raises(ValueError) as exc:
        asyncio.run(
            live_service.run_live_for_agent(
                user_id=91, agent={"agent_id": "a", "live_trading_enabled": False}, dry_run=True
            )
        )
    assert str(exc.value) == "live_trading_not_enabled"


# ===========================================================================
# F. LLM availability
# ===========================================================================


def test_llm_decision_without_a_client_raises(monkeypatch):
    """No provider configured must be an error, not a silent all-hold run."""
    monkeypatch.setattr(live_service, "make_llm_client", lambda *a, **k: None)
    with pytest.raises(ValueError) as exc:
        asyncio.run(
            live_service._llm_decision(
                agent=dict(AGENT), market_snapshot={"symbols": {}}, portfolio_state={}
            )
        )
    assert str(exc.value) == "llm_unavailable"


def test_llm_decision_with_an_unparseable_response_raises(monkeypatch):
    monkeypatch.setattr(live_service, "make_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(live_service, "request_trading_decision", lambda *a, **k: "raw")
    monkeypatch.setattr(live_service, "extract_response_text", lambda response: "not json")
    monkeypatch.setattr(live_service, "parse_llm_response", lambda text: None)
    with pytest.raises(ValueError) as exc:
        asyncio.run(
            live_service._llm_decision(
                agent=dict(AGENT), market_snapshot={"symbols": {}}, portfolio_state={}
            )
        )
    assert str(exc.value) == "llm_unavailable"


def test_llm_decision_without_an_actions_list_raises(monkeypatch):
    monkeypatch.setattr(live_service, "make_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(live_service, "request_trading_decision", lambda *a, **k: "raw")
    monkeypatch.setattr(live_service, "extract_response_text", lambda response: "{}")
    monkeypatch.setattr(live_service, "parse_llm_response", lambda text: {"reasoning": "hi"})
    with pytest.raises(ValueError) as exc:
        asyncio.run(
            live_service._llm_decision(
                agent=dict(AGENT), market_snapshot={"symbols": {}}, portfolio_state={}
            )
        )
    assert str(exc.value) == "llm_unavailable"


def test_llm_decision_returns_the_parsed_decision(monkeypatch):
    parsed = {"actions": [_action("AAPL")], "reasoning": "ok"}
    monkeypatch.setattr(live_service, "make_llm_client", lambda *a, **k: object())
    monkeypatch.setattr(live_service, "request_trading_decision", lambda *a, **k: "raw")
    monkeypatch.setattr(live_service, "extract_response_text", lambda response: "{}")
    monkeypatch.setattr(live_service, "parse_llm_response", lambda text: parsed)
    result = asyncio.run(
        live_service._llm_decision(
            agent=dict(AGENT), market_snapshot={"symbols": {}}, portfolio_state={}
        )
    )
    assert result == parsed


def test_live_run_propagates_llm_unavailable(monkeypatch):
    """The route maps this to 503; swallowing it would ship an empty run as success."""
    client = FakeMCPClient()
    _install_fakes(monkeypatch, client, actions=[])

    async def _no_llm(*, agent, market_snapshot, portfolio_state):
        raise ValueError("llm_unavailable")

    monkeypatch.setattr(live_service, "_llm_decision", _no_llm)
    with pytest.raises(ValueError) as exc:
        asyncio.run(live_service._execute_live_run(user_id=1, agent=dict(AGENT), dry_run=True))
    assert str(exc.value) == "llm_unavailable"
    assert client.place_calls == []


# ===========================================================================
# Supporting pure helpers
# ===========================================================================


def test_portfolio_state_reads_holdings_from_either_shape():
    rows = [{"symbol": "aapl", "quantity": "3"}, {"symbol": "MSFT", "qty": 2}]
    state = live_service._portfolio_state(rows, 500.0)
    assert state["cash"] == 500.0
    assert state["holdings"] == {"AAPL": 3.0, "MSFT": 2.0}

    wrapped = live_service._portfolio_state({"results": rows}, None)
    assert wrapped["holdings"] == {"AAPL": 3.0, "MSFT": 2.0}
    assert wrapped["cash"] == 0.0


def test_actions_to_orders_drops_holds_and_bad_sizes():
    orders = live_service._actions_to_robinhood_orders(
        [
            {"action": "hold", "symbol": "AAPL", "position_size": 5},
            {"action": "buy", "symbol": "", "position_size": 5},
            {"action": "buy", "symbol": "AAPL", "position_size": 0},
            {"action": "buy", "symbol": "AAPL", "position_size": "x"},
            {"action": "short", "symbol": "AAPL", "position_size": 5},
            {"action": "buy", "symbol": "aapl", "position_size": 5},
        ]
    )
    assert orders == [_order("AAPL", "buy", 5.0)]


def test_floor_quantity_rounds_down():
    assert live_service._floor_quantity(0.123456) == 0.1234
    assert live_service._floor_quantity(0.99999) == 0.9999
    assert live_service._floor_quantity(-1.0) == 0.0
    assert live_service._floor_quantity("nope") == 0.0
    assert live_service._floor_quantity(math.inf) == 0.0
