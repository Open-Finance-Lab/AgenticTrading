"""The scorer and the reference ledger on hand-built results, no engine.

Each test pins one way the harness could blame a correct engine (or raise
instead of reporting), so an adapter author can trust a red line to be the
engine's.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from . import cases
from .cases import DO, fl, mkt
from .model import SHARPE_NULL, Expected, Fill, OrderIntent
from .reference import replay
from .scoring import _check_avg_cost, _check_positions, _check_sharpe, actual_invariants


def _lines(problems, prefix):
    return [line for line in problems if line.startswith(prefix)]


def test_replay_reports_a_fill_for_an_unknown_order_instead_of_raising():
    case = cases.T01
    stray = replace(case.expected.fills[0], order_id="zz")
    ledger = replay(case, list(case.expected.fills) + [stray])
    assert "fill for unknown order zz" in ledger.problems


def test_replay_reports_a_target_weight_fill_on_a_bar_its_symbol_lacks():
    """The sort key sizes a target-weight fill off its symbol's bar, so this
    is the shape that raised ``KeyError`` before the guard ran."""
    case = cases.T01
    last = max(case.bars, key=lambda bar: bar.ts)
    elsewhere = replace(last, symbol="MSFT", ts=last.ts + timedelta(hours=1))
    weight = OrderIntent("tw", last.ts, last.symbol, target_weight=Decimal("0.5"))
    case = replace(case, bars=tuple(case.bars) + (elsewhere,), intents=tuple(case.intents) + (weight,))
    ledger = replay(case, list(case.expected.fills) + [Fill("tw", elsewhere.ts, elsewhere.open, Decimal(1))])
    assert f"tw: {last.symbol} has no bar at {elsewhere.ts}" in ledger.problems


def test_invariants_apply_splits_before_a_post_split_sale():
    """CA01 with the sell raised to 6 of the 8 post-split shares."""
    case = cases.CA01
    o2 = replace(case.intents[1], qty=Decimal(6))
    case = replace(case, intents=(case.intents[0], o2))
    fills = (case.expected.fills[0], fl("o2", DO("2026-06-04"), "102", "6"))
    actual = replace(case.expected, fills=fills)
    assert _lines(actual_invariants(case, actual), "invariant 2") == []


def test_invariants_credit_dividend_cash_before_it_is_spent():
    """CA02 plus a pay-date buy that spends exactly the dividend-funded cash."""
    case = cases.CA02
    dividend = case.dividends[0]
    bar = next(b for b in case.bars if b.ts.date() == dividend.pay_date)
    buy = case.expected.fills[0]
    credited = case.initial_cash - buy.qty * buy.price + buy.qty * dividend.amount
    qty = credited / bar.open
    later = mkt("o2", DO(dividend.pay_date.isoformat()), "AAPL", "buy", "1")
    case = replace(case, intents=tuple(case.intents) + (replace(later, qty=qty),))
    fills = (buy, Fill("o2", bar.ts, bar.open, qty))
    problems = actual_invariants(case, replace(case.expected, fills=fills))
    assert _lines(problems, "invariant 1") == []


def test_invariant_3_backs_out_the_engines_own_slippage():
    """F04 states 10 bps; an engine that applies none reports slippage 0 at
    the raw open, which is a fills/cash defect -- not a price-range breach."""
    case = cases.F04
    raw = tuple(
        replace(fill, price=next(b.open for b in case.bars if b.ts == fill.bar), slippage_cost=Decimal(0))
        for fill in case.expected.fills
    )
    assert _lines(actual_invariants(case, replace(case.expected, fills=raw)), "invariant 3") == []
    # The case's own slipped fills, with their stated slippage, stay in range too.
    assert _lines(actual_invariants(case, case.expected), "invariant 3") == []


def test_float_dust_is_not_a_held_position():
    case = cases.F04  # expects the position closed
    dust = replace(case.expected, positions={"AAPL": Decimal("1e-15")}, avg_cost={"AAPL": Decimal("100.1")})
    assert _check_positions(case, case.expected, dust) == []
    assert _check_avg_cost(case, case.expected, dust) == []
    held = replace(dust, positions={"AAPL": Decimal("1e-6")})
    assert _check_positions(case, case.expected, held)


def test_an_unstated_sharpe_is_not_compared():
    case = cases.T01
    unstated = replace(case.expected, sharpe=None)
    assert _check_sharpe(case, unstated, replace(unstated, sharpe=SHARPE_NULL)) == []
    assert _check_sharpe(case, unstated, replace(unstated, sharpe=Decimal("1.5"))) == []
    stated: Expected = replace(case.expected, sharpe=SHARPE_NULL)
    assert _check_sharpe(case, stated, replace(stated, sharpe=Decimal(0)))
