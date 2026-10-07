"""Self-consistency of the conformance case data, with no engine involved.

Every case's expected fills are replayed through the reference ledger
(``reference.replay``) in policy order, and every value the case states --
cash, positions, average cost, realized P&L, the equity curve, the metrics,
each fill's fees and price -- must equal what the ledger derives. The §5
invariants run on the way. A transcription error in ``cases.py`` therefore
goes red here before any adapter can mistake it for an engine defect.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from . import cases
from .cases import ALL_CASES, CASES, PREDICTIONS_D, PREDICTIONS_DL, PREDICTION_DELTAS
from .model import CHECKS, SHARPE_NULL, Case, CostModel, OrderIntent
from .reference import (
    ZERO,
    check_orders,
    daily_sharpe,
    max_drawdown,
    replay,
    sec_fee,
    session_close_equity,
)

_IDS = [case.id for case in ALL_CASES]


def _same(label, stated, derived, tol, problems):
    if stated is not None and abs(stated - derived) > tol:
        problems.append(f"{label}: stated {stated}, derived {derived}")


@pytest.mark.parametrize("case", ALL_CASES, ids=_IDS)
def test_expected_outputs_replay_exactly(case: Case):
    exp, tol = case.expected, case.tol
    ledger = replay(case, exp.fills)
    problems = list(ledger.problems)
    problems += check_orders(case, exp.fills, exp.orders)

    _same("cash", exp.cash, ledger.cash, tol.money, problems)
    held = {s: q for s, q in ledger.positions.items() if q}
    if {s: q for s, q in exp.positions.items() if q} != held:
        problems.append(f"positions: stated {dict(exp.positions)}, derived {held}")
    for symbol in held:
        _same(f"avg_cost[{symbol}]", exp.avg_cost.get(symbol), ledger.avg_cost[symbol], tol.price, problems)
    _same("realized_pnl", exp.realized_pnl, ledger.realized, tol.money, problems)
    for order_id, value in exp.avg_cost_path:
        _same(f"avg_cost after {order_id}", value, dict(ledger.avg_cost_path)[order_id], tol.price, problems)

    for ts, value in exp.equity:
        if ts not in ledger.equity:
            problems.append(f"equity[{ts.isoformat()}]: no case bar closes then")
        else:
            _same(f"equity[{ts.isoformat()}]", value, ledger.equity[ts], tol.money, problems)
    final = ledger.equity[max(ledger.equity)]
    _same("final_equity", exp.final_equity, final, tol.money, problems)
    _same("initial_equity", exp.initial_equity, case.initial_cash, tol.money, problems)
    _same("total_return", exp.total_return, final / case.initial_cash - 1, tol.ratio, problems)
    curve = [case.initial_cash] + [ledger.equity[ts] for ts in sorted(ledger.equity)]
    _same("max_drawdown", exp.max_drawdown, max_drawdown(curve), tol.ratio, problems)
    if exp.sharpe is not None:
        derived = daily_sharpe(session_close_equity(case, ledger.equity), case.initial_cash, case.rf_annual)
        if (exp.sharpe == SHARPE_NULL) != (derived == SHARPE_NULL):
            problems.append(f"sharpe: stated {exp.sharpe}, derived {derived}")
        elif derived != SHARPE_NULL:
            _same("sharpe", exp.sharpe, derived, tol.sharpe, problems)

    # Invariant 5: E_final - E_init = realized + unrealized + dividends.
    if exp.realized_pnl is not None and exp.final_equity is not None:
        unrealized = sum(
            (q * (ledger.last_close[s] - exp.avg_cost[s]) for s, q in exp.positions.items() if q), ZERO
        )
        identity = exp.realized_pnl + unrealized + ledger.dividends
        _same("invariant 5", exp.final_equity - case.initial_cash, identity, tol.money, problems)
    # Invariant 6: no fills and no corporate-action cash -> flat book.
    if not exp.fills and not (case.splits or case.dividends):
        moved = [v for v in ledger.equity.values() if v != case.initial_cash]
        if moved or ledger.cash != case.initial_cash:
            problems.append("invariant 6: a book with no fills moved")

    assert not problems, f"{case.id}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("case", ALL_CASES, ids=_IDS)
def test_case_shape(case: Case):
    assert case.checks and case.checks <= CHECKS, case.checks
    assert len({i.id for i in case.intents}) == len(case.intents)
    for intent in case.intents:
        assert isinstance(intent, OrderIntent)
        assert (intent.side is None) == (intent.target_weight is not None), intent.id
    bars = {(bar.symbol, bar.ts) for bar in case.bars}
    assert len(bars) == len(case.bars), f"{case.id}: duplicate bar"
    for bar in case.bars:
        assert bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high, bar
        assert bar.ts.tzinfo is not None
    for value in (case.expected.cash, case.initial_cash):
        assert isinstance(value, Decimal)


def test_twenty_eight_cases_plus_two_variants():
    assert len(CASES) == 28
    assert [c.id for c in cases.VARIANTS] == ["T02d", "M03b"]


def test_every_case_has_a_prediction():
    assert set(PREDICTIONS_D) == {case.id for case in ALL_CASES}
    assert set(PREDICTIONS_DL) == {"C01", "C04", "S01"}
    for table in (PREDICTIONS_D, PREDICTIONS_DL):
        for case_id, prediction in table.items():
            assert prediction.verdict in {"PASS", "FAIL", "N/E"}, case_id
            assert (prediction.verdict == "FAIL") == bool(prediction.fails), case_id
            assert prediction.fails <= CHECKS | {"invariants"}, case_id
            if prediction.verdict != "PASS":
                assert prediction.defects and prediction.note, case_id


def test_prediction_deltas_name_a_known_case():
    for key in PREDICTION_DELTAS:
        case_id, _, path = key.partition("/")
        assert case_id in cases.BY_ID and path in {"D", "DL"}, key


def test_sec_fee_function_table():
    """F02's fee-function table at the $20.60 rate (2026-04-04 onward)."""
    costs = CostModel(sec_fee=cases.F02.costs.sec_fee)
    for principal, fee in cases.F02_SEC_FEE_TABLE.items():
        assert sec_fee(costs, "sell", principal, cases.F02.bars[-1].ts.date()) == fee
        assert sec_fee(costs, "buy", principal, cases.F02.bars[-1].ts.date()) == ZERO


def test_f05_summed_eod_rounding_variant():
    """F05's unscored alternative: one ceiling over the summed regulatory fees."""
    from .reference import ceil_cent

    principal, qty = Decimal("1248.75"), Decimal("4.995")
    summed = ceil_cent(principal * Decimal("20.60") / 1_000_000 + qty * Decimal("0.000195"))
    fees = Decimal("1.00") + summed
    assert fees == cases.F05_SUMMED_EOD_ROUNDING["o2_fees"]
    assert principal - fees == cases.F05_SUMMED_EOD_ROUNDING["cash"]
    assert Decimal("-1.00") + qty * 50 - fees == cases.F05_SUMMED_EOD_ROUNDING["realized_pnl"]


def test_exactness_is_not_float():
    """The reason the model is Decimal (§1): a float floor is one step short."""
    import math

    from .reference import floor_step

    assert math.floor(8.0 / 1e-9) == 7999999999
    assert floor_step(Decimal(1000) / Decimal(125), Decimal("1e-9")) == 8
