"""Characterization tests for the extracted rule-based agent (Phase 2C1).

Locks in the exact behavior of
``dashboard.backend.domain.backtesting.reference_agent.make_rule_based_decision``
and the legacy ``PortfolioManager.make_trading_decision`` that delegates to it.
Imports use the canonical package path; no external services are touched.
"""

import copy

import numpy as np

from dashboard.backend.domain.backtesting.reference_agent import (
    make_rule_based_decision,
)
from dashboard.scripts import backtest_hourly_agent as bha


def _signal(price=None, rsi=None, sma20=None, sma50=None):
    return {"price": price, "rsi": rsi, "sma20": sma20, "sma50": sma50}


def _state(total_equity=100000, signals=None):
    return {
        "total_equity": total_equity,
        "market_signals": signals or {},
    }


def _decide(state, positions=None, cash=100000, lot_size=1):
    return make_rule_based_decision(
        portfolio_state=state,
        positions=dict(positions or {}),
        cash=cash,
        lot_size=lot_size,
    )


# ---------------------------------------------------------------------------
# Empty / incomplete input
# ---------------------------------------------------------------------------

def test_empty_market_data():
    assert _decide(_state(signals={})) == {"actions": []}


def test_none_indicators_skipped():
    state = _state(signals={"AAPL": _signal(price=100, rsi=None, sma20=None)})
    assert _decide(state) == {"actions": []}


def test_nan_indicators_skipped():
    state = _state(signals={"AAPL": _signal(price=100, rsi=np.nan, sma20=120)})
    assert _decide(state) == {"actions": []}


def test_missing_sma20_skipped():
    # sma20 None -> pd.isna true -> skipped
    state = _state(signals={"AAPL": _signal(price=100, rsi=20, sma20=None)})
    assert _decide(state) == {"actions": []}


def test_rsi_present_sma20_present_but_no_signal_holds():
    # neutral indicators -> no action (HOLD == empty action list)
    state = _state(signals={"AAPL": _signal(price=100, rsi=50, sma20=100, sma50=100)})
    assert _decide(state) == {"actions": []}


# ---------------------------------------------------------------------------
# BUY behavior
# ---------------------------------------------------------------------------

def test_buy_triggered_exact_action():
    # rsi < 30 and price < sma20, no position
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),
    })
    out = _decide(state, positions={}, cash=100000)
    # shares = int(100000 * 0.02 / 100) = int(20) = 20 ; 20*100=2000 <= cash
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "buy",
        "shares": 20,
        "reason": "RSI oversold (25), price below MA",
    }]}


def test_buy_fails_when_price_not_below_sma20():
    state = _state(signals={"AAPL": _signal(price=120, rsi=25, sma20=110)})
    assert _decide(state) == {"actions": []}


def test_buy_fails_when_rsi_not_oversold():
    state = _state(signals={"AAPL": _signal(price=100, rsi=35, sma20=110)})
    assert _decide(state) == {"actions": []}


def test_buy_skipped_when_existing_position():
    state = _state(signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    # has_position True -> BUY branch not taken; SELL branch needs rsi>70 / price>sma50*1.02
    assert _decide(state, positions={"AAPL": 5}) == {"actions": []}


def test_buy_skipped_when_zero_shares_to_buy():
    # tiny equity -> int(risk/price) == 0 -> no action
    state = _state(total_equity=100, signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    assert _decide(state, cash=100000) == {"actions": []}


def test_buy_skipped_when_insufficient_cash():
    state = _state(total_equity=100000, signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    # shares=20, cost=2000 > cash=1000 -> skipped
    assert _decide(state, cash=1000) == {"actions": []}


def test_buy_share_quantity_uses_int_truncation():
    # total_equity*0.02/price = 100000*0.02/150 = 13.33 -> int -> 13
    state = _state(total_equity=100000, signals={"AAPL": _signal(price=150, rsi=10, sma20=200)})
    out = _decide(state, cash=100000)
    assert out["actions"][0]["shares"] == 13
    assert isinstance(out["actions"][0]["shares"], int)


def test_ashare_buy_signal_requests_exactly_one_lot():
    state = _state(total_equity=100000, signals={
        "600519.SH": _signal(price=150, rsi=10, sma20=200),
    })

    out = _decide(state, cash=100000, lot_size=100)

    assert out["actions"][0]["shares"] == 100


def test_ashare_buy_signal_reaches_executor_when_cash_is_insufficient():
    state = _state(total_equity=1000, signals={
        "600519.SH": _signal(price=150, rsi=10, sma20=200),
    })

    out = _decide(state, cash=1000, lot_size=100)

    assert out == {"actions": [{
        "symbol": "600519.SH",
        "action": "buy",
        "shares": 100,
        "reason": "RSI oversold (10), price below MA",
    }]}


# ---------------------------------------------------------------------------
# SELL behavior
# ---------------------------------------------------------------------------

def test_sell_triggered_by_overbought_rsi():
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    out = _decide(state, positions={"AAPL": 7})
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "sell",
        "shares": 7,
        "reason": "RSI overbought (75)",
    }]}


def test_sell_triggered_by_price_above_sma50():
    # rsi not > 70, but price > sma50 * 1.02
    state = _state(signals={"AAPL": _signal(price=105, rsi=50, sma20=90, sma50=100)})
    out = _decide(state, positions={"AAPL": 3})
    assert out == {"actions": [{
        "symbol": "AAPL",
        "action": "sell",
        "shares": 3,
        "reason": "Price above MA50",
    }]}


def test_sell_not_triggered_when_price_just_below_threshold():
    # price == sma50 * 1.02 exactly -> not strictly greater -> no sell
    state = _state(signals={"AAPL": _signal(price=102, rsi=50, sma20=90, sma50=100)})
    assert _decide(state, positions={"AAPL": 3}) == {"actions": []}


def test_sell_requires_position():
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    assert _decide(state, positions={}) == {"actions": []}


def test_sell_zero_position_treated_as_no_position():
    # positions[symbol] == 0 -> has_position False -> not a sell; BUY needs rsi<30
    state = _state(signals={"AAPL": _signal(price=100, rsi=75, sma20=90, sma50=80)})
    assert _decide(state, positions={"AAPL": 0}) == {"actions": []}


def test_sell_sma50_falsy_only_rsi_path():
    # sma50 == 0 (falsy) -> price>sma50*1.02 branch short-circuits; needs rsi>70
    state = _state(signals={"AAPL": _signal(price=100, rsi=50, sma20=90, sma50=0)})
    assert _decide(state, positions={"AAPL": 3}) == {"actions": []}


# ---------------------------------------------------------------------------
# Multiple symbols / ordering / determinism
# ---------------------------------------------------------------------------

def test_multiple_symbols_buy_and_sell_order_preserved():
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),  # buy
        "MSFT": _signal(price=100, rsi=80, sma20=90, sma50=80),    # sell (held)
    })
    out = _decide(state, positions={"MSFT": 4}, cash=100000)
    assert [a["symbol"] for a in out["actions"]] == ["AAPL", "MSFT"]
    assert out["actions"][0]["action"] == "buy"
    assert out["actions"][1]["action"] == "sell"


def test_one_valid_one_incomplete_symbol():
    state = _state(signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110),     # buy
        "MSFT": _signal(price=100, rsi=None, sma20=None),  # skipped
    })
    out = _decide(state, cash=100000)
    assert [a["symbol"] for a in out["actions"]] == ["AAPL"]


def test_deterministic_repeated_calls():
    state = _state(signals={"AAPL": _signal(price=100, rsi=25, sma20=110)})
    a = _decide(state, cash=100000)
    b = _decide(state, cash=100000)
    assert a == b


# ---------------------------------------------------------------------------
# Inputs not mutated
# ---------------------------------------------------------------------------

def test_inputs_not_mutated():
    state = _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),
    })
    positions = {"MSFT": 5}
    state_copy = copy.deepcopy(state)
    positions_copy = copy.deepcopy(positions)
    make_rule_based_decision(portfolio_state=state, positions=positions, cash=100000)
    assert state == state_copy
    assert positions == positions_copy


# ---------------------------------------------------------------------------
# Legacy equivalence + subclass compatibility
# ---------------------------------------------------------------------------

def _golden_state():
    return _state(total_equity=100000, signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110, sma50=120),  # buy
        "MSFT": _signal(price=105, rsi=50, sma20=90, sma50=100),   # sell if held
        "JPM": _signal(price=100, rsi=None, sma20=None),           # skipped
    })


def test_legacy_matches_canonical():
    state = _golden_state()
    pm = bha.PortfolioManager(100000)
    pm.positions = {"MSFT": 4}
    pm.cash = 100000

    legacy = pm.make_trading_decision(copy.deepcopy(state))
    canon = make_rule_based_decision(
        portfolio_state=copy.deepcopy(state),
        positions={"MSFT": 4},
        cash=100000,
    )
    assert legacy == canon


def test_legacy_golden_exact():
    pm = bha.PortfolioManager(100000)
    pm.positions = {"MSFT": 4}
    out = pm.make_trading_decision(_golden_state())
    assert out == {"actions": [
        {"symbol": "AAPL", "action": "buy", "shares": 20,
         "reason": "RSI oversold (25), price below MA"},
        {"symbol": "MSFT", "action": "sell", "shares": 4,
         "reason": "Price above MA50"},
    ]}


def test_subclass_inherits_make_trading_decision():
    class MyPM(bha.PortfolioManager):
        def custom_method(self):
            return "ok"

    pm = MyPM(100000)
    out = pm.make_trading_decision(_state(signals={
        "AAPL": _signal(price=100, rsi=25, sma20=110),
    }))
    assert out["actions"][0]["action"] == "buy"
    assert pm.custom_method() == "ok"
    assert MyPM.make_trading_decision is bha.PortfolioManager.make_trading_decision


# ---------------------------------------------------------------------------
# T+1 sellable cap (available_positions)
# ---------------------------------------------------------------------------

def _sell_state():
    """A state whose only signal triggers the SELL rule for MSFT."""
    return _state(signals={"MSFT": _signal(price=110, rsi=80, sma20=100, sma50=100)})


def test_sell_uses_held_quantity_when_settlement_is_immediate():
    """The default (available_positions=None) must stay byte-identical."""
    out = _decide(_sell_state(), positions={"MSFT": 10})
    assert out["actions"] == [
        {"symbol": "MSFT", "action": "sell", "shares": 10,
         "reason": "RSI overbought (80)"},
    ]


def test_sell_is_capped_at_the_sellable_quantity():
    out = make_rule_based_decision(
        portfolio_state=_sell_state(),
        positions={"MSFT": 10},
        cash=100000,
        available_positions={"MSFT": 4},
    )
    assert out["actions"] == [
        {"symbol": "MSFT", "action": "sell", "shares": 4,
         "reason": "RSI overbought (80)",
         # The intent the cap discarded, so the decision log does not read as
         # though the agent only ever wanted 4.
         "requested_shares": 10},
    ]
    assert out["t1_deferrals"] == [
        {"symbol": "MSFT", "requested_shares": 10, "sellable_shares": 4},
    ]


def test_uncapped_sell_carries_no_deferral_or_requested_shares():
    """Nothing is reported when T+1 did not actually bind."""
    out = make_rule_based_decision(
        portfolio_state=_sell_state(),
        positions={"MSFT": 10},
        cash=100000,
        available_positions={"MSFT": 10},
    )
    assert out == {"actions": [
        {"symbol": "MSFT", "action": "sell", "shares": 10,
         "reason": "RSI overbought (80)"},
    ]}


def test_fully_frozen_holding_still_reports_the_deferral():
    """The case that matters most: no action, so nothing else would record it."""
    out = make_rule_based_decision(
        portfolio_state=_sell_state(),
        positions={"MSFT": 10},
        cash=100000,
        available_positions={"MSFT": 0},
    )
    assert out["actions"] == []
    assert out["t1_deferrals"] == [
        {"symbol": "MSFT", "requested_shares": 10, "sellable_shares": 0},
    ]


def test_fully_frozen_holding_emits_no_action():
    """The whole point: an unfillable order is not proposed at all.

    Without this the agent re-proposes the full holding on every bar of the buy
    date, and each one becomes a t1_frozen audit record for a rule it was never
    shown.
    """
    for available in ({"MSFT": 0}, {}):
        out = make_rule_based_decision(
            portfolio_state=_sell_state(),
            positions={"MSFT": 10},
            cash=100000,
            available_positions=available,
        )
        assert out["actions"] == []


def test_empty_available_positions_does_not_disable_selling_outright():
    """``{}`` means 'nothing settled yet'; ``None`` means 'no T+1 here'.

    Conflating them would silently stop every DJIA run from ever selling.
    """
    frozen = make_rule_based_decision(
        portfolio_state=_sell_state(), positions={"MSFT": 10},
        cash=100000, available_positions={},
    )
    immediate = make_rule_based_decision(
        portfolio_state=_sell_state(), positions={"MSFT": 10},
        cash=100000, available_positions=None,
    )
    assert frozen["actions"] == []
    assert immediate["actions"][0]["shares"] == 10


def test_sellable_cap_does_not_leak_into_buy_branch():
    """SELL is an ``elif``; a frozen holding must not fall through to BUY."""
    state = _state(signals={"MSFT": _signal(price=110, rsi=80, sma20=100, sma50=100)})
    out = make_rule_based_decision(
        portfolio_state=state,
        positions={"MSFT": 10},
        cash=100000,
        available_positions={"MSFT": 0},
    )
    assert out["actions"] == []


def test_inputs_are_not_mutated_by_the_sellable_cap():
    positions = {"MSFT": 10}
    available = {"MSFT": 4}
    positions_before = copy.deepcopy(positions)
    available_before = copy.deepcopy(available)
    make_rule_based_decision(
        portfolio_state=_sell_state(), positions=positions,
        cash=100000, available_positions=available,
    )
    assert positions == positions_before
    assert available == available_before
