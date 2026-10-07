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
from .reference import ZERO, bar_index, unslipped

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


def _check_book(label, expected: Dict[str, Decimal], actual, tol) -> List[str]:
    if actual is None:
        return [f"{label}: expected {dict(expected)}, actual {_fmt(None)}"]
    out = []
    want = {s: q for s, q in expected.items() if q}
    got = {s: q for s, q in actual.items() if q}
    for symbol in sorted(set(want) | set(got)):
        if symbol not in got:
            out.append(f"{label}[{symbol}]: expected {want[symbol]}, actual not held")
        elif symbol not in want:
            out.append(f"{label}[{symbol}]: expected not held, actual {got[symbol]}")
        else:
            out += _value(f"{label}[{symbol}]", want[symbol], got[symbol], tol)
    return out


def _check_cash(case, exp, act):
    return _value("cash", exp.cash, act.cash, case.tol.money)


def _check_positions(case, exp, act):
    return _check_book("positions", exp.positions, act.positions, case.tol.qty)


def _check_avg_cost(case, exp, act):
    tol = case.tol.price
    held = {s for s, q in (exp.positions or {}).items() if q}
    want = {s: v for s, v in exp.avg_cost.items() if s in held}
    got = None if act.avg_cost is None else {
        s: v for s, v in act.avg_cost.items() if (act.positions or {}).get(s)
    }
    out = _check_book("avg_cost", want, got, tol)
    if exp.avg_cost_path:
        got_path = dict(act.avg_cost_path)
        for order_id, value in exp.avg_cost_path:
            out += _value(f"avg_cost after {order_id}", value, got_path.get(order_id), tol)
    return out


def _check_realized(case, exp, act):
    return _value("realized_pnl", exp.realized_pnl, act.realized_pnl, case.tol.money)


def _check_equity(case, exp, act):
    got = dict(act.equity)
    return [
        line
        for ts, value in exp.equity
        for line in _value(f"equity[{ts.isoformat()}]", value, got.get(ts), case.tol.money)
    ]


def _check_final_equity(case, exp, act):
    return _value("final_equity", exp.final_equity, act.final_equity, case.tol.money)


def _check_initial_equity(case, exp, act):
    return _value("initial_equity", exp.initial_equity, act.initial_equity, case.tol.money)


def _check_total_return(case, exp, act):
    return _value("total_return", exp.total_return, act.total_return, case.tol.ratio)


def _check_max_drawdown(case, exp, act):
    return _value("max_drawdown", exp.max_drawdown, act.max_drawdown, case.tol.ratio)


def _check_sharpe(case, exp, act):
    want, got = exp.sharpe, act.sharpe
    if want == SHARPE_NULL or got == SHARPE_NULL or got is None:
        if want != got:
            return [f"sharpe: expected {_fmt(want)}, actual {_fmt(got)}"]
        return []
    return _value("sharpe", want, got, case.tol.sharpe)


_CHECKERS = {
    "fills": _check_fills,
    "orders": _check_orders,
    "cash": _check_cash,
    "positions": _check_positions,
    "avg_cost": _check_avg_cost,
    "realized_pnl": _check_realized,
    "equity": _check_equity,
    "final_equity": _check_final_equity,
    "initial_equity": _check_initial_equity,
    "total_return": _check_total_return,
    "max_drawdown": _check_max_drawdown,
    "sharpe": _check_sharpe,
}


def actual_invariants(case: Case, act: Expected) -> List[str]:
    """§5 invariants 1, 2, 3, 4, 6 and 7 on an engine's own output.

    Invariant 5 needs realized P&L and receivables, which an engine may not
    report; it is applied to the expected data instead (``test_case_data``).
    """
    out: List[str] = []
    intents = {intent.id: intent for intent in case.intents}
    bars = bar_index(case)
    has_corporate_cash = bool(case.splits or case.dividends)
    cash, held = case.initial_cash, {}
    for fill in act.fills:  # in the engine's execution order
        intent = intents.get(fill.order_id)
        if intent is None:
            out.append(f"invariant 7: fill for unknown order {fill.order_id}")
            continue
        side = fill.side or intent.side or "buy"
        fees = sum((getattr(fill, f) or ZERO for f in _FEE_FIELDS if f != "slippage_cost"), ZERO)
        notional = fill.qty * fill.price
        if side == "buy":
            cash -= notional + fees
            held[intent.symbol] = held.get(intent.symbol, ZERO) + fill.qty
        else:
            cash += notional - fees
            held[intent.symbol] = held.get(intent.symbol, ZERO) - fill.qty
        if cash < 0:
            out.append(f"invariant 1: cash {cash} after {fill.order_id}@{fill.bar.isoformat()}")
        if held[intent.symbol] < 0:
            out.append(f"invariant 2: {intent.symbol} position {held[intent.symbol]} after {fill.order_id}")
        bar = bars.get((intent.symbol, fill.bar))
        if bar is None:
            out.append(f"invariant 3: {fill.order_id} fills at {fill.bar.isoformat()}, not a {intent.symbol} bar")
        else:
            ref = fill.price if intent.type == "limit" else unslipped(case.costs, side, fill.price)
            if fill.slippage_cost is None:
                ref = fill.price
            if not bar.low <= ref <= bar.high:
                out.append(f"invariant 3: {fill.order_id} reference price {ref} outside [{bar.low}, {bar.high}]")
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
