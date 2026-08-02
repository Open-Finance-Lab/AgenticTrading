"""Characterization and A-share T+1 tests for order execution.

Locks in the exact behavior of
``dashboard.backend.domain.trading.execution.execute_actions`` and the legacy
``PortfolioManager.execute_actions`` that delegates to it. Imports use the
canonical package path; no external services are touched.
"""

from datetime import datetime

import pandas as pd
import pytest

from dashboard.backend.domain.trading.execution import execute_actions
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
