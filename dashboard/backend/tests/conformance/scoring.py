"""Score an adapter's result against a case (§1 scoring), engine-neutral.

``score`` returns ``{check: [mismatch, ...]}`` holding only the checks that
failed. It compares exactly the checks the case names, plus ``invariants``
(§5), which run on every case. Every mismatch line names the field, the
expected value and the actual value.

An actual field of ``None`` means the engine has no such output; it fails any
check that names it. Fills and orders are matched by ``order_id`` (adapters for
id-less engines resolve ids before returning).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional

from .model import SHARPE_NULL, Case, Expected, Fill
from .reference import ZERO, Ledger, apply_corporate_actions, bar_index, grid

_FEE_FIELDS = ("commission", "sec_fee", "taf", "slippage_cost")


class ConformanceFailure(AssertionError):
    """A case failed on exactly the checks its prediction names."""


def _off(expected: Optional[Decimal], actual: Optional[Decimal], tol: Decimal) -> bool:
    if expected is None:
        return False
    return actual is None or abs(actual - expected) > tol


def _fmt(value) -> str:
    return "None (engine has no such output)" if value is None else str(value)


def _value(label: str, expected, actual, tol) -> List[str]:
    if _off(expected, actual, tol):
        return [f"{label}: expected {_fmt(expected)}, actual {_fmt(actual)}"]
    return []


def _fee_fields(case: Case) -> List[str]:
    costs = case.costs
    names = []
    if costs.has_commission:
        names.append("commission")
    if costs.sec_fee:
        names.append("sec_fee")
    if costs.taf:
        names.append("taf")
    if costs.slippage_bps:
        names.append("slippage_cost")
    return names


def _check_fills(case: Case, exp: Expected, act: Expected) -> List[str]:
    tol, out = case.tol, []
    actual: Dict[tuple, List[Fill]] = {}
    for fill in act.fills:
        actual.setdefault((fill.order_id, fill.bar), []).append(fill)
    for fill in exp.fills:
        key = (fill.order_id, fill.bar)
        found = actual.pop(key, [])
        label = f"fills[{fill.order_id}@{fill.bar.isoformat()}]"
        if not found:
            out.append(f"{label}: expected price {fill.price} qty {fill.qty}, actual no fill on that bar")
            continue
        if len(found) > 1:
            out.append(f"{label}: expected one fill, actual {len(found)}")
        got = found[0]
        out += _value(f"{label}.price", fill.price, got.price, tol.price)
        out += _value(f"{label}.qty", fill.qty, got.qty, tol.qty)
        for name in _fee_fields(case):
            out += _value(f"{label}.{name}", getattr(fill, name), getattr(got, name), tol.money)
    for (order_id, bar), extra in actual.items():
        for fill in extra:
            out.append(
                f"fills[{order_id}@{bar.isoformat()}]: expected no fill, "
                f"actual price {fill.price} qty {fill.qty}"
            )
    return out


def _check_orders(case: Case, exp: Expected, act: Expected) -> List[str]:
    out = []
    actual: Dict[str, list] = {}
    for final in act.orders:
        actual.setdefault(final.order_id, []).append(final)
    for final in exp.orders:
        got = actual.pop(final.order_id, [])
        label = f"orders[{final.order_id}]"
        want = f"{final.status}/{final.reason or '-'} filled {final.filled_qty}"
        if not got:
            out.append(f"{label}: expected {want}, actual no terminal OrderFinal")
            continue
        if len(got) > 1:
            out.append(f"{label}: expected one OrderFinal, actual {len(got)}")
        g = got[0]
        if (g.status, g.reason) != (final.status, final.reason) or _off(
            final.filled_qty, g.filled_qty, case.tol.qty
        ):
            out.append(f"{label}: expected {want}, actual {g.status}/{g.reason or '-'} filled {g.filled_qty}")
    for order_id in actual:
        out.append(f"orders[{order_id}]: expected no such order, actual {actual[order_id]}")
    return out


def _held(positions, tol: Decimal) -> set:
    """Symbols with a position beyond ``tol``: float dust is not a holding."""
    return {s for s, q in (positions or {}).items() if q is not None and abs(q) > tol}


def _check_book(label, want: Dict[str, Decimal], got, tol) -> List[str]:
    """Compare two per-symbol books, each already limited to held symbols."""
    if got is None:
        return [f"{label}: expected {dict(want)}, actual {_fmt(None)}"]
    out = []
    for symbol in sorted(set(want) | set(got)):
        if symbol not in got:
            out.append(f"{label}[{symbol}]: expected {want[symbol]}, actual not held")
        elif symbol not in want:
            out.append(f"{label}[{symbol}]: expected not held, actual {got[symbol]}")
        else:
            out += _value(f"{label}[{symbol}]", want[symbol], got[symbol], tol)
    return out


def _held_only(book, held: set):
    return None if book is None else {s: v for s, v in book.items() if s in held}


def _check_positions(case, exp, act):
    tol = case.tol.qty
    return _check_book(
        "positions",
        _held_only(exp.positions, _held(exp.positions, tol)),
        _held_only(act.positions, _held(act.positions, tol)),
        tol,
    )


def _check_avg_cost(case, exp, act):
    tol = case.tol.price
    want = _held_only(exp.avg_cost, _held(exp.positions, case.tol.qty))
    got = _held_only(act.avg_cost, _held(act.positions, case.tol.qty))
    out = _check_book("avg_cost", want, got, tol)
    if exp.avg_cost_path:
        got_path = dict(act.avg_cost_path)
        for order_id, value in exp.avg_cost_path:
            out += _value(f"avg_cost after {order_id}", value, got_path.get(order_id), tol)
    return out


def _check_equity(case, exp, act):
    got = dict(act.equity)
    return [
        line
        for ts, value in exp.equity
        for line in _value(f"equity[{ts.isoformat()}]", value, got.get(ts), case.tol.money)
    ]


#: Scalar checks: the ``Expected`` field (named like the check) and the
#: ``Tol`` attribute it is compared under.
_SCALARS = {
    "cash": "money",
    "realized_pnl": "money",
    "final_equity": "money",
    "initial_equity": "money",
    "total_return": "ratio",
    "max_drawdown": "ratio",
}


def _scalar(name: str):
    tol_attr = _SCALARS[name]

    def check(case, exp, act):
        return _value(name, getattr(exp, name), getattr(act, name), getattr(case.tol, tol_attr))

    return check


def _check_sharpe(case, exp, act):
    want, got = exp.sharpe, act.sharpe
    if want is None:  # not stated by the design doc: not compared
        return []
    if want == SHARPE_NULL or got == SHARPE_NULL or got is None:
        if want != got:
            return [f"sharpe: expected {_fmt(want)}, actual {_fmt(got)}"]
        return []
    return _value("sharpe", want, got, case.tol.sharpe)


_CHECKERS = {
    "fills": _check_fills,
    "orders": _check_orders,
    "positions": _check_positions,
    "avg_cost": _check_avg_cost,
    "equity": _check_equity,
    "sharpe": _check_sharpe,
    **{name: _scalar(name) for name in _SCALARS},
}


def _reference_price(case: Case, intent, side: str, fill: Fill) -> Decimal:
    """The bar price a fill was taken against, before slippage (invariant 3).

    Backed out of the engine's *own* reported slippage, never the case's
    ``slippage_bps``: the engine may apply a different rate (or none), which is
    a fills/cash defect, not a price-range breach. No reported slippage (the
    engine has no such field) means the fill price is the reference.
    """
    if intent.type == "limit" or fill.slippage_cost is None or not fill.qty:
        return fill.price
    per_share = fill.slippage_cost / fill.qty
    return fill.price - per_share if side == "buy" else fill.price + per_share


def actual_invariants(case: Case, act: Expected) -> List[str]:
    """§5 invariants 1, 2, 3, 4, 6 and 7 on an engine's own output.

    The running book applies the case's splits and dividends (via the
    reference ledger's ``apply_corporate_actions``) at the open of each ex/pay
    date, before that day's fills, so a post-split sale or spent dividend cash
    is not scored as a short position or negative cash. Invariant 4 (cash
    conservation) still skips corporate-action cases: whether the engine
    credited them is what the ``cash`` check measures.

    Invariant 5 needs realized P&L and receivables, which an engine may not
    report; it is applied to the expected data instead (``test_case_data``).
    """
    out: List[str] = []
    tol = case.tol
    intents = {intent.id: intent for intent in case.intents}
    bars = bar_index(case)
    has_corporate_cash = bool(case.splits or case.dividends)
    first_stamp: Dict = {}
    for ts, _minutes in grid(case):
        first_stamp.setdefault(ts.date(), ts)
    pending_days = sorted(first_stamp)
    book = Ledger(cash=case.initial_cash)
    held = book.positions

    def advance_to(day) -> None:
        while pending_days and pending_days[0] <= day:
            d = pending_days.pop(0)
            apply_corporate_actions(case, book, d, first_stamp[d], bars)

    for fill in act.fills:  # in the engine's execution order
        intent = intents.get(fill.order_id)
        if intent is None:
            out.append(f"invariant 7: fill for unknown order {fill.order_id}")
            continue
        advance_to(fill.bar.date())
        side = fill.side or intent.side or "buy"
        fees = sum((getattr(fill, f) or ZERO for f in _FEE_FIELDS if f != "slippage_cost"), ZERO)
        notional = fill.qty * fill.price
        book.avg_cost.setdefault(intent.symbol, ZERO)  # a split rescales it; its value is unused here
        if side == "buy":
            book.cash -= notional + fees
            held[intent.symbol] = held.get(intent.symbol, ZERO) + fill.qty
        else:
            book.cash += notional - fees
            held[intent.symbol] = held.get(intent.symbol, ZERO) - fill.qty
        if book.cash < -tol.money:
            out.append(f"invariant 1: cash {book.cash} after {fill.order_id}@{fill.bar.isoformat()}")
        if held[intent.symbol] < -tol.qty:
            out.append(f"invariant 2: {intent.symbol} position {held[intent.symbol]} after {fill.order_id}")
        bar = bars.get((intent.symbol, fill.bar))
        if bar is None:
            out.append(f"invariant 3: {fill.order_id} fills at {fill.bar.isoformat()}, not a {intent.symbol} bar")
        else:
            ref = _reference_price(case, intent, side, fill)
            if not bar.low <= ref <= bar.high:
                out.append(f"invariant 3: {fill.order_id} reference price {ref} outside [{bar.low}, {bar.high}]")
    cash = book.cash
    if act.cash is not None and not has_corporate_cash and abs(act.cash - cash) > case.tol.money:
        out.append(f"invariant 4: cash {act.cash}, the fills conserve to {cash}")
    if not act.fills and not has_corporate_cash:
        drifted = [(ts, v) for ts, v in act.equity if abs(v - case.initial_cash) > case.tol.money]
        if drifted:
            out.append(f"invariant 6: no fills, yet equity moved: {drifted[0][0].isoformat()} = {drifted[0][1]}")
    counts: Dict[str, int] = {}
    for final in act.orders:
        counts[final.order_id] = counts.get(final.order_id, 0) + 1
    for intent in case.intents:
        if counts.get(intent.id, 0) != 1:
            out.append(f"invariant 7: {intent.id} has {counts.get(intent.id, 0)} terminal OrderFinal(s), expected 1")
    return out


def score(case: Case, actual: Expected) -> Dict[str, List[str]]:
    failures: Dict[str, List[str]] = {}
    for check in sorted(case.checks):
        lines = _CHECKERS[check](case, case.expected, actual)
        if lines:
            failures[check] = lines
    lines = actual_invariants(case, actual)
    if lines:
        failures["invariants"] = lines
    return failures


def report(case_id: str, failures: Dict[str, List[str]]) -> str:
    lines = [f"{case_id}: {len(failures)} failing check(s)"]
    for check in sorted(failures):
        lines.append(f"  [{check}]")
        lines += [f"    {line}" for line in failures[check]]
    return "\n".join(lines)
