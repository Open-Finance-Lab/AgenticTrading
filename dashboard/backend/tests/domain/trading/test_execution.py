"""Characterization and A-share T+1 tests for order execution.

Locks in the exact behavior of
``dashboard.backend.domain.trading.execution.execute_actions`` and the legacy
``PortfolioManager.execute_actions`` that delegates to it. Imports use the
canonical package path; no external services are touched.
"""

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.domain.trading import execution as execution_module
from dashboard.backend.domain.trading.execution import (
    calculate_transaction_costs,
    execute_actions,
)
from dashboard.backend.domain.backtesting.market_rules import (
    ClosingLimitState,
    DailyMarketRule,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    ASHARE_TRANSACTION_COST_PROFILE,
)
from dashboard.scripts import backtest_hourly_agent as bha


def _row(close, **kwargs):
    data = {"close": close}
    data.update(kwargs)
    return pd.Series(data)


def _state(cash=100000, positions=None, entry_prices=None, trades=None):
    return {
        "cash": cash,
        "positions": dict(positions or {}),
        "entry_prices": dict(entry_prices or {}),
        "trades": list(trades if trades is not None else []),
    }


def _run(actions, market_data, timestamp="t0", **state):
    st = _state(**state)
    st["cash"] = execute_actions(
        actions=actions,
        market_data=market_data,
        timestamp=timestamp,
        cash=st["cash"],
        positions=st["positions"],
        entry_prices=st["entry_prices"],
        trades=st["trades"],
    )
    return st


def _run_ashare(actions, *, cash=100000, positions=None, available_positions=None,
                frozen_lots=None, timestamp=None, market_data=None,
                market_rule=None, fallback_prices=None):
    timestamp = timestamp or datetime(2026, 4, 1, 10)
    positions = dict(positions or {})
    entry_prices = {
        symbol: 100.0 for symbol in positions
    }
    trades = []
    order_events = []
    rejected_orders = []
    new_cash = execute_actions(
        actions=actions,
        market_data=(
            {"600519.SH": _row(100.0)} if market_data is None else market_data
        ),
        timestamp=timestamp,
        cash=cash,
        positions=positions,
        entry_prices=entry_prices,
        trades=trades,
        t_plus_one_enabled=True,
        available_positions=dict(available_positions or {}),
        frozen_lots=dict(frozen_lots or {}),
        rejected_orders=rejected_orders,
        lot_size=100,
        order_events=order_events,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
        market_rules=(
            {"600519.SH": market_rule} if market_rule is not None else None
        ),
        fallback_prices=fallback_prices,
    )
    return {
        "cash": new_cash,
        "positions": positions,
        "trades": trades,
        "order_events": order_events,
        "rejected_orders": rejected_orders,
    }


CN = ZoneInfo("Asia/Shanghai")


def _market_rule(*, suspended=False, state=ClosingLimitState.NONE):
    if suspended:
        return DailyMarketRule(
            symbol="600519.SH",
            trading_date=date(2026, 4, 1),
            suspended=True,
        )
    return DailyMarketRule(
        symbol="600519.SH",
        trading_date=date(2026, 4, 1),
        suspended=False,
        closing_limit_state=state,
        official_close_price=Decimal("100.00"),
        final_bar_timestamp=datetime(2026, 4, 1, 15, tzinfo=CN),
    )


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_suspension_rejects_both_sides_before_price_lot_or_t1_checks(side):
    result = _run_ashare(
        [{"symbol": "600519.SH", "action": side, "shares": 50}],
        cash=100000,
        positions={"600519.SH": 100},
        available_positions={"600519.SH": 100},
        timestamp=datetime(2026, 4, 1, 10, 30, tzinfo=CN),
        market_data={},
        market_rule=_market_rule(suspended=True),
        fallback_prices={"600519.SH": 99.5},
    )

    assert result["cash"] == 100000
    assert result["positions"] == {"600519.SH": 100}
    assert result["trades"] == []
    assert result["rejected_orders"][0]["reason"] == "suspended"
    event = result["order_events"][0]
    assert event["reason"] == "suspended"
    assert event["price"] == 99.5
    assert event["total_fees"] == 0
    assert event["net_cash_impact"] == 0
    assert event["market_rule_suspended"] is True


def test_closing_upper_limit_blocks_buy_only_on_final_bar():
    rule = _market_rule(state=ClosingLimitState.UPPER)
    earlier = _run_ashare(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        timestamp=datetime(2026, 4, 1, 14, tzinfo=CN),
        market_rule=rule,
    )
    closing = _run_ashare(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        timestamp=datetime(2026, 4, 1, 15, tzinfo=CN),
        market_rule=rule,
    )

    assert earlier["positions"] == {"600519.SH": 100}
    assert len(earlier["trades"]) == 1
    assert closing["positions"] == {}
    assert closing["cash"] == 100000
    assert closing["trades"] == []
    assert closing["order_events"][0]["reason"] == "limit_up_buy_blocked"
    assert closing["order_events"][0]["market_rule_closing_gate_effective"] is True


def test_closing_upper_limit_still_allows_sell():
    result = _run_ashare(
        [{"symbol": "600519.SH", "action": "sell", "shares": 100}],
        positions={"600519.SH": 100},
        available_positions={"600519.SH": 100},
        timestamp=datetime(2026, 4, 1, 15, tzinfo=CN),
        market_rule=_market_rule(state=ClosingLimitState.UPPER),
    )

    assert result["positions"] == {}
    assert result["trades"][0]["side"] == "SELL"


def test_closing_lower_limit_blocks_sell_but_allows_buy():
    rule = _market_rule(state=ClosingLimitState.LOWER)
    sell = _run_ashare(
        [{"symbol": "600519.SH", "action": "sell", "shares": 100}],
        positions={"600519.SH": 100},
        available_positions={"600519.SH": 100},
        timestamp=datetime(2026, 4, 1, 15, tzinfo=CN),
        market_rule=rule,
    )
    buy = _run_ashare(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        timestamp=datetime(2026, 4, 1, 15, tzinfo=CN),
        market_rule=rule,
    )

    assert sell["positions"] == {"600519.SH": 100}
    assert sell["cash"] == 100000
    assert sell["order_events"][0]["reason"] == "limit_down_sell_blocked"
    assert buy["positions"] == {"600519.SH": 100}
    assert buy["trades"][0]["side"] == "BUY"


# ---------------------------------------------------------------------------
# Deterministic A-share transaction-cost calculation
# ---------------------------------------------------------------------------

def test_ashare_buy_costs_use_adverse_tick_rounding_and_minimum_commission():
    costs = calculate_transaction_costs(
        side="buy",
        reference_price=100.0,
        shares=100,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
    )

    assert costs == {
        "reference_price": 100.0,
        "price": 100.05,
        "gross_value": 10005.0,
        "slippage_amount": 5.0,
        "commission": 5.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.1,
        "total_fees": 5.1,
        "net_cash_impact": -10010.1,
    }


def test_ashare_sell_costs_include_stamp_duty_and_two_sided_transfer_fee():
    costs = calculate_transaction_costs(
        side="sell",
        reference_price=100.0,
        shares=100,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
    )

    assert costs == {
        "reference_price": 100.0,
        "price": 99.95,
        "gross_value": 9995.0,
        "slippage_amount": 5.0,
        "commission": 5.0,
        "stamp_duty": 5.0,
        "transfer_fee": 0.1,
        "total_fees": 10.1,
        "net_cash_impact": 9984.9,
    }


def test_ashare_costed_buy_checks_fees_before_filling():
    rejected = _run_ashare(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        cash=10005.0,
    )
    assert rejected["cash"] == 10005.0
    assert rejected["positions"] == {}
    assert rejected["trades"] == []
    assert rejected["order_events"][-1]["reason"] == "insufficient_cash_for_lot"
    assert rejected["order_events"][-1]["commission"] == 0.0

    filled = _run_ashare(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        cash=10010.1,
    )
    assert filled["cash"] == pytest.approx(0.0)
    assert filled["positions"] == {"600519.SH": 100}
    assert filled["trades"][0]["net_cash_impact"] == -10010.1


def test_ashare_t1_partial_fill_charges_only_filled_quantity():
    result = _run_ashare(
        [{"symbol": "600519.SH", "action": "sell", "shares": 200}],
        cash=0.0,
        positions={"600519.SH": 200},
        available_positions={"600519.SH": 100},
        frozen_lots={
            "600519.SH": [{"quantity": 100, "buy_date": datetime(2026, 4, 1).date()}]
        },
    )

    assert result["trades"][0]["shares"] == 100
    assert result["trades"][0]["total_fees"] == 10.1
    assert result["cash"] == pytest.approx(9984.9)
    assert result["order_events"][-1]["status"] == "partial"
    assert result["order_events"][-1]["price"] == 99.95
    assert result["order_events"][-1]["gross_value"] == 9995.0
    assert result["order_events"][-1]["total_fees"] == 10.1


def test_portfolio_manager_accumulates_native_transaction_cost_totals():
    pm = bha.PortfolioManager(
        20000,
        allowed_symbols=["600519.SH"],
        t_plus_one_enabled=True,
        lot_size=100,
        transaction_cost_profile=ASHARE_TRANSACTION_COST_PROFILE,
    )
    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.transaction_cost_totals == {
        "gross_value": 10005.0,
        "slippage_amount": 5.0,
        "commission": 5.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.1,
        "total_fees": 5.1,
    }


# ---------------------------------------------------------------------------
# HOLD / no-op
# ---------------------------------------------------------------------------

def test_empty_action_list_noop():
    md = {"AAPL": _row(200.0)}
    st = _run([], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_hold_action_noop():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "hold", "shares": 10}], md,
              positions={"AAPL": 5}, entry_prices={"AAPL": 100.0})
    assert st["cash"] == 100000
    assert st["positions"] == {"AAPL": 5}
    assert st["trades"] == []


def test_unknown_action_type_noop():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "rebalance", "shares": 10}], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


# ---------------------------------------------------------------------------
# BUY
# ---------------------------------------------------------------------------

def test_valid_buy():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 10, "reason": "r"}], md)
    assert st["cash"] == 98000.0
    assert st["positions"] == {"AAPL": 10}
    assert st["entry_prices"] == {"AAPL": 200.0}
    assert st["trades"] == [{
        "timestamp": "t0",
        "symbol": "AAPL",
        "side": "BUY",
        "shares": 10,
        "price": 200.0,
        "cost": 2000.0,
        "reason": "r",
    }]


def test_buy_default_reason_empty():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 1}], md)
    assert st["trades"][0]["reason"] == ""


def test_multiple_buys_accumulate_position():
    md = {"AAPL": _row(200.0)}
    st = _run([
        {"symbol": "AAPL", "action": "buy", "shares": 10},
        {"symbol": "AAPL", "action": "buy", "shares": 5},
    ], md)
    assert st["positions"] == {"AAPL": 15}
    # entry price overwritten with last buy price
    assert st["entry_prices"] == {"AAPL": 200.0}
    assert st["cash"] == 100000 - 3000.0
    assert len(st["trades"]) == 2


def test_buy_exact_available_cash():
    md = {"AAPL": _row(100.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 1000}], md, cash=100000)
    assert st["cash"] == 0
    assert st["positions"] == {"AAPL": 1000}


def test_insufficient_cash_skips_buy():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 1000}], md, cash=1000)
    assert st["cash"] == 1000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_buy_missing_symbol_skipped():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "TSLA", "action": "buy", "shares": 10}], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_buy_zero_shares_skipped():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 0}], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_buy_missing_shares_defaults_zero_skipped():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy"}], md)
    assert st["cash"] == 100000
    assert st["trades"] == []


def test_buy_negative_shares_skipped():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": -10}], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_buy_fractional_shares():
    md = {"AAPL": _row(200.0)}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 2.5}], md)
    assert st["positions"] == {"AAPL": 2.5}
    assert st["cash"] == 100000 - 500.0
    assert st["trades"][0]["shares"] == 2.5


# ---------------------------------------------------------------------------
# SELL
# ---------------------------------------------------------------------------

def test_valid_full_sell_removes_position():
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell", "shares": 10, "reason": "x"}], md,
              positions={"AAPL": 10}, entry_prices={"AAPL": 200.0})
    assert st["cash"] == 100000 + 2500.0
    assert st["positions"] == {}
    assert st["entry_prices"] == {}
    assert st["trades"] == [{
        "timestamp": "t0",
        "symbol": "AAPL",
        "side": "SELL",
        "shares": 10,
        "price": 250.0,
        "proceeds": 2500.0,
        "reason": "x",
    }]


def test_partial_sell_keeps_position():
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell", "shares": 4}], md,
              positions={"AAPL": 10}, entry_prices={"AAPL": 200.0})
    assert st["positions"] == {"AAPL": 6}
    assert st["entry_prices"] == {"AAPL": 200.0}
    assert st["cash"] == 100000 + 1000.0
    assert st["trades"][0]["shares"] == 4


def test_sell_more_than_held_caps_at_holding():
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell", "shares": 999}], md,
              positions={"AAPL": 10}, entry_prices={"AAPL": 200.0})
    assert st["positions"] == {}
    assert st["cash"] == 100000 + 2500.0
    assert st["trades"][0]["shares"] == 10


def test_sell_missing_position_skipped():
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell", "shares": 5}], md)
    assert st["cash"] == 100000
    assert st["positions"] == {}
    assert st["trades"] == []


def test_sell_zero_shares_appends_trade_no_change():
    # min(0, 10) == 0 -> proceeds 0, position unchanged, but a trade IS appended.
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell", "shares": 0}], md,
              positions={"AAPL": 10}, entry_prices={"AAPL": 200.0})
    assert st["cash"] == 100000
    assert st["positions"] == {"AAPL": 10}
    assert len(st["trades"]) == 1
    assert st["trades"][0]["shares"] == 0
    assert st["trades"][0]["proceeds"] == 0


def test_sell_missing_shares_defaults_zero_appends_trade():
    md = {"AAPL": _row(250.0)}
    st = _run([{"symbol": "AAPL", "action": "sell"}], md,
              positions={"AAPL": 10})
    assert st["positions"] == {"AAPL": 10}
    assert len(st["trades"]) == 1
    assert st["trades"][0]["shares"] == 0


def test_multiple_sells():
    md = {"AAPL": _row(250.0)}
    st = _run([
        {"symbol": "AAPL", "action": "sell", "shares": 3},
        {"symbol": "AAPL", "action": "sell", "shares": 3},
    ], md, positions={"AAPL": 10}, entry_prices={"AAPL": 200.0})
    assert st["positions"] == {"AAPL": 4}
    assert len(st["trades"]) == 2


# ---------------------------------------------------------------------------
# Mixed / ordering / partial execution
# ---------------------------------------------------------------------------

def test_buy_then_sell_order_preserved():
    md = {"AAPL": _row(200.0)}
    st = _run([
        {"symbol": "AAPL", "action": "buy", "shares": 10},
        {"symbol": "AAPL", "action": "sell", "shares": 4},
    ], md)
    assert st["positions"] == {"AAPL": 6}
    assert [t["side"] for t in st["trades"]] == ["BUY", "SELL"]


def test_invalid_action_does_not_block_later_actions():
    md = {"AAPL": _row(200.0), "MSFT": _row(400.0)}
    st = _run([
        {"symbol": "TSLA", "action": "buy", "shares": 10},   # missing symbol -> skip
        {"symbol": "AAPL", "action": "buy", "shares": 10},   # valid
    ], md)
    assert st["positions"] == {"AAPL": 10}
    assert len(st["trades"]) == 1


def test_multiple_symbols():
    md = {"AAPL": _row(200.0), "MSFT": _row(400.0)}
    st = _run([
        {"symbol": "AAPL", "action": "buy", "shares": 10},
        {"symbol": "MSFT", "action": "buy", "shares": 5},
    ], md)
    assert st["positions"] == {"AAPL": 10, "MSFT": 5}
    assert st["cash"] == 100000 - 2000.0 - 2000.0


# ---------------------------------------------------------------------------
# Optional A-share T+1 execution
# ---------------------------------------------------------------------------

def _t1_manager(cash=100000):
    return bha.PortfolioManager(cash, t_plus_one_enabled=True)


def _ashare_manager(cash=100000):
    return bha.PortfolioManager(
        cash,
        t_plus_one_enabled=True,
        lot_size=100,
    )


@pytest.mark.parametrize("shares", [50, 150, 100.5])
def test_ashare_buy_rejects_non_lot_quantity_without_mutation(shares):
    pm = _ashare_manager()
    timestamp = datetime(2026, 4, 1, 10)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": shares}],
        {"600519.SH": _row(100.0)},
        timestamp,
    )

    assert pm.cash == 100000
    assert pm.positions == {}
    assert pm.trades == []
    assert pm.rejected_orders[-1]["reason"] == "invalid_lot_size"
    assert pm.order_events[-1] == {
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": shares,
        "executed_shares": 0,
        "unfilled_shares": shares,
        "price": 100.0,
        "executed_value": 0.0,
        "status": "rejected",
        "reason": "invalid_lot_size",
        "strategy_reason": "",
    }


@pytest.mark.parametrize("shares", [50, 150, 100.5])
def test_ashare_sell_rejects_non_lot_quantity_before_t1(shares):
    pm = _ashare_manager()
    pm.positions = {"600519.SH": 200}
    pm.available_positions = {"600519.SH": 200}

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": shares}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 2, 10),
    )

    assert pm.positions == {"600519.SH": 200}
    assert pm.trades == []
    assert pm.rejected_orders[-1]["reason"] == "invalid_lot_size"
    assert pm.order_events[-1]["status"] == "rejected"


def test_ashare_buy_one_lot_fills_and_records_order_event():
    pm = _ashare_manager(cash=20000)
    timestamp = datetime(2026, 4, 1, 10)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        {"600519.SH": _row(100.0)},
        timestamp,
    )

    assert pm.cash == 10000
    assert pm.positions == {"600519.SH": 100}
    assert pm.order_events[-1]["status"] == "filled"
    assert pm.order_events[-1]["executed_shares"] == 100
    assert pm.order_events[-1]["executed_value"] == 10000


def test_ashare_buy_rejects_when_cash_cannot_cover_one_lot():
    pm = _ashare_manager(cash=1000)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.cash == 1000
    assert pm.positions == {}
    assert pm.trades == []
    assert pm.rejected_orders[-1]["reason"] == "insufficient_cash_for_lot"
    assert pm.order_events[-1]["status"] == "rejected"


def test_ashare_invalid_lot_takes_priority_over_insufficient_cash():
    pm = _ashare_manager(cash=0)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 50}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert [item["reason"] for item in pm.rejected_orders] == [
        "invalid_lot_size"
    ]
    assert pm.order_events[0]["reason"] == "invalid_lot_size"


def test_ashare_buy_does_not_partially_fill_affordable_lots():
    pm = _ashare_manager(cash=15000)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 200}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.cash == 15000
    assert pm.positions == {}
    assert pm.trades == []
    assert pm.order_events[-1]["requested_shares"] == 200
    assert pm.order_events[-1]["executed_shares"] == 0
    assert pm.order_events[-1]["reason"] == "insufficient_cash_for_lot"


def test_ashare_t1_partial_sell_has_one_partial_order_event():
    pm = _ashare_manager()
    pm.positions = {"600519.SH": 200}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 100}
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 100, "buy_date": datetime(2026, 4, 1).date()}]
    }

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 200}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 14),
    )

    assert pm.trades[-1]["shares"] == 100
    assert pm.order_events[-1]["status"] == "partial"
    assert pm.order_events[-1]["requested_shares"] == 200
    assert pm.order_events[-1]["executed_shares"] == 100
    assert pm.order_events[-1]["unfilled_shares"] == 100
    assert pm.order_events[-1]["reason"] == "t1_frozen"


def test_ashare_order_event_prefers_t1_when_sell_has_two_rejection_causes():
    pm = _ashare_manager()
    pm.positions = {"600519.SH": 200}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 100}
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 100, "buy_date": datetime(2026, 4, 1).date()}]
    }

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 300}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 14),
    )

    assert [item["reason"] for item in pm.rejected_orders] == [
        "t1_frozen",
        "insufficient_position",
    ]
    assert len(pm.order_events) == 1
    assert pm.order_events[0]["reason"] == "t1_frozen"


def test_ashare_full_sell_records_one_filled_order_event():
    pm = _ashare_manager()
    pm.positions = {"600519.SH": 100}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 100}

    pm.execute_actions(
        [{
            "symbol": "600519.SH",
            "action": "sell",
            "shares": 100,
            "reason": "Exit signal",
        }],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 2, 10),
    )

    assert len(pm.trades) == 1
    assert pm.order_events == [{
        "timestamp": datetime(2026, 4, 2, 10),
        "symbol": "600519.SH",
        "side": "SELL",
        "requested_shares": 100,
        "executed_shares": 100,
        "unfilled_shares": 0,
        "price": 100.0,
        "executed_value": 10000.0,
        "status": "filled",
        "reason": "",
        "strategy_reason": "Exit signal",
    }]


def test_ashare_buy_then_same_day_sell_records_two_order_events():
    pm = _ashare_manager(cash=20000)
    timestamp = datetime(2026, 4, 1, 10)

    pm.execute_actions([
        {"symbol": "600519.SH", "action": "buy", "shares": 100},
        {"symbol": "600519.SH", "action": "sell", "shares": 100},
    ], {"600519.SH": _row(100.0)}, timestamp)

    assert [(item["side"], item["status"]) for item in pm.order_events] == [
        ("BUY", "filled"),
        ("SELL", "rejected"),
    ]
    assert pm.order_events[1]["reason"] == "t1_frozen"


def test_ashare_hold_does_not_record_order_event():
    pm = _ashare_manager()

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "hold", "shares": 100}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.order_events == []


# ---------------------------------------------------------------------------
# Order outcomes are recorded for every market, and repeats are collapsed
# ---------------------------------------------------------------------------

def test_single_share_market_records_an_unaffordable_buy():
    """DJIA must not drop an unfillable order the "All Orders" log promises.

    Execution behaviour is unchanged -- the buy still does not happen -- but it
    now leaves a trace, which is what the log needs to be honest.
    """
    pm = bha.PortfolioManager(500)

    pm.execute_actions(
        [{"symbol": "AAPL", "action": "buy", "shares": 10}],
        {"AAPL": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.cash == 500
    assert pm.positions == {}
    assert pm.trades == []
    # `rejected_orders` is the T+1 audit trail and stays A-share-only.
    assert pm.rejected_orders == []
    assert pm.order_events[-1]["status"] == "rejected"
    assert pm.order_events[-1]["reason"] == "insufficient_cash"
    assert pm.order_events[-1]["side"] == "BUY"


def test_single_share_market_records_a_sell_of_an_unheld_symbol():
    pm = bha.PortfolioManager(1000)

    pm.execute_actions(
        [{"symbol": "AAPL", "action": "sell", "shares": 5}],
        {"AAPL": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.cash == 1000
    assert pm.trades == []
    assert pm.rejected_orders == []
    assert pm.order_events[-1]["reason"] == "insufficient_position"


def test_a_zero_share_order_records_nothing():
    """The new every-market branch must not mint events for non-orders."""
    pm = bha.PortfolioManager(1000)

    pm.execute_actions(
        [
            {"symbol": "AAPL", "action": "buy", "shares": 0},
            {"symbol": "AAPL", "action": "sell", "shares": 0},
        ],
        {"AAPL": _row(100.0)},
        datetime(2026, 4, 1, 10),
    )

    assert pm.order_events == []


def test_repeated_rejection_collapses_per_symbol_trading_day():
    """A signal the agent cannot act on re-fires every bar.

    Without collapsing, those duplicates fill the persisted head sample end to
    end, so the audit is least useful exactly when the constraint bound
    hardest -- the failure `t1_deferrals` already had to solve.
    """
    pm = _ashare_manager(cash=1000)
    md = {"600519.SH": _row(100.0)}

    for hour in (10, 11, 13, 14):
        pm.execute_actions(
            [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
            md,
            datetime(2026, 4, 1, hour),
        )

    assert len(pm.order_events) == 1
    assert pm.order_events[0]["reason"] == "insufficient_cash_for_lot"
    assert pm.order_events[0]["repeat_count"] == 4
    # The T+1 audit list keeps its own per-bar records; only the UI-facing
    # order ledger collapses.
    assert len(pm.rejected_orders) == 4


def test_repeat_collapse_separates_days_symbols_and_reasons():
    pm = _ashare_manager(cash=1000)
    md = {"600519.SH": _row(100.0), "601318.SH": _row(100.0)}

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        md,
        datetime(2026, 4, 1, 10),
    )
    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
        md,
        datetime(2026, 4, 2, 10),
    )
    pm.execute_actions(
        [{"symbol": "601318.SH", "action": "buy", "shares": 100}],
        md,
        datetime(2026, 4, 2, 10),
    )
    pm.execute_actions(
        [{"symbol": "601318.SH", "action": "buy", "shares": 50}],
        md,
        datetime(2026, 4, 2, 10),
    )

    keys = [
        (event["symbol"], event["reason"]) for event in pm.order_events
    ]
    assert keys == [
        ("600519.SH", "insufficient_cash_for_lot"),
        ("600519.SH", "insufficient_cash_for_lot"),
        ("601318.SH", "insufficient_cash_for_lot"),
        ("601318.SH", "invalid_lot_size"),
    ]
    assert all("repeat_count" not in event for event in pm.order_events)


def test_fills_never_collapse_however_alike_they_look():
    """Two identical fills are two ledger movements, not one repeated refusal."""
    pm = _ashare_manager(cash=100000)
    md = {"600519.SH": _row(100.0)}

    for hour in (10, 11):
        pm.execute_actions(
            [{"symbol": "600519.SH", "action": "buy", "shares": 100}],
            md,
            datetime(2026, 4, 1, hour),
        )

    assert [event["status"] for event in pm.order_events] == [
        "filled",
        "filled",
    ]
    assert pm.positions == {"600519.SH": 200}


@pytest.mark.parametrize("shares", ["100", "abc", None, True, False])
def test_lot_validation_rejects_non_numeric_quantities(shares):
    """`float("100")` would pass every lot check, then crash at shares * price."""
    assert execution_module._is_valid_lot_quantity(shares, 100) is False


def test_t1_same_day_buy_then_sell_records_rejection_without_zero_trade():
    pm = _t1_manager()
    md = {"600519.SH": _row(100.0)}
    timestamp = datetime(2026, 4, 1, 10)

    pm.execute_actions([
        {"symbol": "600519.SH", "action": "buy", "shares": 10},
        {"symbol": "600519.SH", "action": "sell", "shares": 10},
    ], md, timestamp)

    assert pm.cash == 99000.0
    assert pm.positions == {"600519.SH": 10}
    assert pm.available_positions == {}
    assert pm.frozen_lots == {
        "600519.SH": [{"quantity": 10, "buy_date": timestamp.date()}]
    }
    assert [(trade["side"], trade["shares"]) for trade in pm.trades] == [
        ("BUY", 10)
    ]
    assert pm.rejected_orders == [{
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "action": "sell",
        "requested_shares": 10,
        "executed_shares": 0,
        "unfilled_shares": 10,
        "status": "rejected",
        "reason": "t1_frozen",
    }]


def test_t1_prior_buy_unlocks_on_next_data_trading_date_across_weekend():
    pm = _t1_manager()
    md = {"600519.SH": _row(100.0)}

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 10}],
        md,
        datetime(2026, 4, 3, 14),
    )
    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 10}],
        md,
        datetime(2026, 4, 6, 10),
    )

    assert pm.cash == 100000
    assert pm.positions == {}
    assert pm.available_positions == {}
    assert pm.frozen_lots == {}
    assert [trade["side"] for trade in pm.trades] == ["BUY", "SELL"]
    assert pm.rejected_orders == []


def test_t1_sell_above_available_partially_fills_and_audits_frozen_remainder():
    pm = _t1_manager()
    pm.positions = {"600519.SH": 100}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 40}
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 60, "buy_date": datetime(2026, 4, 1).date()}]
    }
    timestamp = datetime(2026, 4, 1, 14)

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 100}],
        {"600519.SH": _row(100.0)},
        timestamp,
    )

    assert pm.cash == 104000.0
    assert pm.positions == {"600519.SH": 60}
    assert pm.available_positions == {}
    assert pm.trades[-1]["shares"] == 40
    assert pm.rejected_orders[-1] == {
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "action": "sell",
        "requested_shares": 100,
        "executed_shares": 40,
        "unfilled_shares": 60,
        "status": "partial",
        "reason": "t1_frozen",
    }


def test_t1_multiple_buy_dates_release_only_prior_batches():
    pm = _t1_manager()
    md = {"600519.SH": _row(100.0)}

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 20}],
        md,
        datetime(2026, 4, 1, 10),
    )
    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "buy", "shares": 30}],
        md,
        datetime(2026, 4, 2, 10),
    )

    assert pm.positions == {"600519.SH": 50}
    assert pm.available_positions == {"600519.SH": 20}
    assert pm.frozen_lots == {
        "600519.SH": [
            {"quantity": 30, "buy_date": datetime(2026, 4, 2).date()}
        ]
    }


def test_t1_request_above_total_splits_frozen_and_insufficient_reasons():
    pm = _t1_manager()
    pm.positions = {"600519.SH": 100}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 40}
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 60, "buy_date": datetime(2026, 4, 1).date()}]
    }

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 150}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 1, 14),
    )

    assert [item["reason"] for item in pm.rejected_orders] == [
        "t1_frozen",
        "insufficient_position",
    ]
    assert [item["unfilled_shares"] for item in pm.rejected_orders] == [60, 50]


def test_t1_float_residue_does_not_mint_a_phantom_rejection():
    """A fully-filled fractional sell must not audit ~1e-17 unfilled shares.

    0.3 - 0.1 - 0.2 is -2.8e-17 in binary floating point, so an exact fill
    leaves negative-zero-ish residue that a bare ``> 0`` test reads as a real
    unfilled quantity — an ``insufficient_position`` record for a constraint
    that was never violated.
    """
    pm = _t1_manager()
    pm.positions = {"600519.SH": 0.3}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 0.3}

    for size in (0.1, 0.2):
        pm.execute_actions(
            [{"symbol": "600519.SH", "action": "sell", "shares": size}],
            {"600519.SH": _row(100.0)},
            datetime(2026, 4, 2, 10),
        )

    # Both sells fill (the second for the residual balance, itself inexact)…
    assert len(pm.trades) == 2
    assert sum(trade["shares"] for trade in pm.trades) == pytest.approx(0.3)
    # …and neither leaves an audit record behind.
    assert pm.rejected_orders == []


def test_t1_genuine_shortfall_still_audits_above_the_epsilon():
    """The tolerance must not swallow a real one-share shortfall."""
    pm = _t1_manager()
    pm.positions = {"600519.SH": 5}
    pm.entry_prices = {"600519.SH": 90.0}
    pm.available_positions = {"600519.SH": 5}

    pm.execute_actions(
        [{"symbol": "600519.SH", "action": "sell", "shares": 6}],
        {"600519.SH": _row(100.0)},
        datetime(2026, 4, 2, 10),
    )

    assert [item["reason"] for item in pm.rejected_orders] == ["insufficient_position"]
    assert pm.rejected_orders[0]["unfilled_shares"] == 1


# ---------------------------------------------------------------------------
# Trade records appended in place, earlier records unchanged
# ---------------------------------------------------------------------------

def test_existing_trades_preserved_and_appended_in_place():
    md = {"AAPL": _row(200.0)}
    prior = {"timestamp": "old", "symbol": "X", "side": "BUY"}
    trades = [prior]
    st = _state(trades=trades)
    # use the same list object to confirm in-place append
    st["cash"] = execute_actions(
        actions=[{"symbol": "AAPL", "action": "buy", "shares": 1}],
        market_data=md,
        timestamp="t0",
        cash=st["cash"],
        positions=st["positions"],
        entry_prices=st["entry_prices"],
        trades=trades,
    )
    assert trades[0] is prior
    assert len(trades) == 2
    assert trades[1]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# No price_cache fallback (distinct from portfolio valuation helpers)
# ---------------------------------------------------------------------------

def test_execution_ignores_price_cache_semantics():
    # Symbol not in market_data is always skipped; execution has no cache param.
    md = {}
    st = _run([{"symbol": "AAPL", "action": "buy", "shares": 10}], md)
    assert st["positions"] == {}
    assert st["trades"] == []


# ---------------------------------------------------------------------------
# Legacy equivalence: PortfolioManager.execute_actions delegates identically
# ---------------------------------------------------------------------------

def _golden_actions():
    return [
        {"symbol": "AAPL", "action": "buy", "shares": 10, "reason": "a"},
        {"symbol": "MSFT", "action": "buy", "shares": 5, "reason": "b"},
        {"symbol": "AAPL", "action": "sell", "shares": 4, "reason": "c"},
        {"symbol": "TSLA", "action": "buy", "shares": 1},      # missing symbol -> skip
        {"symbol": "MSFT", "action": "hold", "shares": 99},    # no-op
    ]


def _golden_md():
    return {"AAPL": _row(200.0), "MSFT": _row(400.0)}


def test_legacy_method_matches_canonical_helper():
    actions = _golden_actions()
    md = _golden_md()

    # Legacy path
    pm = bha.PortfolioManager(100000)
    assert pm.execute_actions(actions, md, "t0") is None  # returns None
    legacy = {
        "cash": pm.cash,
        "positions": pm.positions,
        "entry_prices": pm.entry_prices,
        "trades": pm.trades,
    }

    # Canonical path with identical inputs
    canon = _run(actions, md, timestamp="t0")

    assert legacy["cash"] == canon["cash"]
    assert legacy["positions"] == canon["positions"]
    assert legacy["entry_prices"] == canon["entry_prices"]
    assert legacy["trades"] == canon["trades"]


def test_legacy_golden_exact_values():
    pm = bha.PortfolioManager(100000)
    pm.execute_actions(_golden_actions(), _golden_md(), "t0")
    # AAPL: buy 10 @200 (-2000), sell 4 @200 (+800) -> 6 shares
    # MSFT: buy 5 @400 (-2000) -> 5 shares
    assert pm.cash == 100000 - 2000 + 800 - 2000
    assert pm.positions == {"AAPL": 6, "MSFT": 5}
    assert pm.entry_prices == {"AAPL": 200.0, "MSFT": 400.0}
    assert [(t["side"], t["symbol"], t["shares"]) for t in pm.trades] == [
        ("BUY", "AAPL", 10),
        ("BUY", "MSFT", 5),
        ("SELL", "AAPL", 4),
    ]


def test_subclass_inherits_execute_actions():
    class MyPM(bha.PortfolioManager):
        def custom_method(self):
            return "ok"

    pm = MyPM(100000)
    pm.execute_actions(
        [{"symbol": "AAPL", "action": "buy", "shares": 10}],
        {"AAPL": _row(200.0)},
        "t0",
    )
    assert pm.cash == 98000.0
    assert pm.positions == {"AAPL": 10}
    assert pm.custom_method() == "ok"
    # execute_actions resolves through the subclass MRO to the script-defined method
    assert MyPM.execute_actions is bha.PortfolioManager.execute_actions


# ---------------------------------------------------------------------------
# T+1 deferral ledger — the metric a capped order would otherwise erase
# ---------------------------------------------------------------------------

def test_t1_deferral_is_recorded_once_per_symbol_trading_day():
    """Four bars of one day that all want out must not be four records."""
    pm = _t1_manager()
    day = datetime(2026, 4, 1).date()
    pm.positions = {"600519.SH": 100}
    pm.frozen_lots = {"600519.SH": [{"quantity": 100, "buy_date": day}]}

    for _ in range(4):
        pm.record_t1_deferral("600519.SH", 100, 0)

    assert list(pm.t1_deferrals) == [("600519.SH", day)]
    assert pm.t1_deferrals[("600519.SH", day)]["deferred_shares"] == 100


def test_t1_deferral_keeps_the_worst_of_the_day():
    pm = _t1_manager()
    day = datetime(2026, 4, 1).date()
    pm.frozen_lots = {"600519.SH": [{"quantity": 100, "buy_date": day}]}

    pm.record_t1_deferral("600519.SH", 100, 60)   # 40 deferred
    pm.record_t1_deferral("600519.SH", 100, 10)   # 90 deferred — worse
    pm.record_t1_deferral("600519.SH", 100, 80)   # 20 deferred — not worse

    assert pm.t1_deferrals[("600519.SH", day)]["deferred_shares"] == 90
    assert pm.t1_deferrals[("600519.SH", day)]["sellable_shares"] == 10


def test_t1_deferral_separates_symbols_and_days():
    pm = _t1_manager()
    d1, d2 = datetime(2026, 4, 1).date(), datetime(2026, 4, 2).date()
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 10, "buy_date": d1}],
        "601318.SH": [{"quantity": 10, "buy_date": d1}],
    }
    pm.record_t1_deferral("600519.SH", 10, 0)
    pm.record_t1_deferral("601318.SH", 10, 0)
    # Next day: the same symbol blocking again is a distinct event.
    pm.frozen_lots["600519.SH"] = [{"quantity": 10, "buy_date": d2}]
    pm.record_t1_deferral("600519.SH", 10, 0)

    assert sorted(pm.t1_deferrals) == [
        ("600519.SH", d1), ("600519.SH", d2), ("601318.SH", d1),
    ]


def test_t1_deferral_ignores_a_sell_that_was_not_actually_capped():
    pm = _t1_manager()
    pm.frozen_lots = {
        "600519.SH": [{"quantity": 10, "buy_date": datetime(2026, 4, 1).date()}]
    }
    pm.record_t1_deferral("600519.SH", 10, 10)
    assert pm.t1_deferrals == {}


def test_t1_deferral_needs_a_frozen_lot_to_date_itself():
    """No frozen lot means nothing was blocking, so there is no event."""
    pm = _t1_manager()
    pm.record_t1_deferral("600519.SH", 10, 0)
    assert pm.t1_deferrals == {}


def test_sellable_positions_is_a_read_only_view():
    pm = _t1_manager()
    pm.available_positions = {"600519.SH": 5}
    view = pm.sellable_positions

    assert view["600519.SH"] == 5
    with pytest.raises(TypeError):
        view["600519.SH"] = 999
    # Still a live view, not a snapshot.
    pm.available_positions["600519.SH"] = 7
    assert view["600519.SH"] == 7


def test_sellable_positions_is_none_without_t_plus_one():
    pm = bha.PortfolioManager(100000)
    pm.available_positions = {"AAPL": 0}
    assert pm.sellable_positions is None


def test_symbol_outside_the_rule_calendar_is_rejected_not_raised():
    """A stray symbol must not trade ungated, and must not kill the run either.

    Every producer filters to the allowed universe before it gets here, so this
    is unreachable today. If one ever stops, raising would discard every bar
    already simulated for a single action — and the rejection record is what
    makes the gap visible instead of silent.
    """
    result = _run_ashare(
        [{"symbol": "600520.SH", "action": "buy", "shares": 100}],
        cash=100000,
        timestamp=datetime(2026, 4, 1, 10, 30, tzinfo=CN),
        market_data={"600519.SH": _row(100.0), "600520.SH": _row(50.0)},
        market_rule=_market_rule(),
    )

    assert result["cash"] == 100000
    assert result["positions"] == {}
    assert result["trades"] == []
    assert result["rejected_orders"][0]["reason"] == "market_rule_unavailable"
    assert result["rejected_orders"][0]["symbol"] == "600520.SH"
    event = result["order_events"][0]
    assert event["status"] == "rejected"
    assert event["reason"] == "market_rule_unavailable"
    # No rule means no audit to attach; the row must not claim one.
    assert "market_rule_date" not in event
