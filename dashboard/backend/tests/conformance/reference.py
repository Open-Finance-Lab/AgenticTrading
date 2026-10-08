"""Reference arithmetic for the conformance policy (§2), in exact Decimal.

Two jobs, both engine-neutral (no ATL imports):

* the policy's fee, rounding and metric formulas (P5, P7, P8, P10-P12, P19);
* ``replay``: a ledger that applies a case's *expected* fills, splits and
  dividends in policy order and re-derives cash, positions, average cost,
  realized P&L and the equity curve. ``test_case_data.py`` compares that
  against every value the case states, so a transcription error goes red with
  no engine involved.

``replay`` is not a simulator: it never decides whether or where an order
fills. It takes the fills as given and checks that they are consistent with
the policy (timing, price, fees, sizing, the §5 invariants).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Dict, List, Optional, Sequence, Tuple

from .model import (
    ORDER_REASONS,
    SHARPE_NULL,
    Bar,
    Case,
    CostModel,
    Fill,
    OrderIntent,
    rate_on,
)

CENT = Decimal("0.01")
ZERO = Decimal(0)


def ceil_cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_CEILING)


def floor_step(value: Decimal, step: Decimal) -> Decimal:
    """Floor toward zero on the qty grid (P7). Exact: no float floor."""
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def commission(costs: CostModel, qty: Decimal, notional: Decimal) -> Decimal:
    """P10: ``max(min, per_share*q + pct*notional)``, rounded up to the cent."""
    if not costs.has_commission:
        return ZERO
    raw = costs.commission_per_share * qty + costs.commission_pct * notional
    return ceil_cent(max(costs.commission_min, raw))


def sec_fee(costs: CostModel, side: str, principal: Decimal, trade_date: date) -> Decimal:
    """P11: sells only, ``principal * rate / 1e6``, rounded up, dated by trade date."""
    row = rate_on(costs.sec_fee, trade_date) if side == "sell" else None
    if row is None:
        return ZERO
    return ceil_cent(principal * row.rate / Decimal(1_000_000))


def taf_fee(costs: CostModel, side: str, qty: Decimal, trade_date: date) -> Decimal:
    """P11: sells only, ``min(q * rate, cap)``, rounded up, dated by trade date."""
    row = rate_on(costs.taf, trade_date) if side == "sell" else None
    if row is None:
        return ZERO
    fee = qty * row.rate
    if row.cap is not None:
        fee = min(fee, row.cap)
    return ceil_cent(fee)


def cat_fee(costs: CostModel, qty: Decimal, trade_date: date) -> Decimal:
    row = rate_on(costs.cat_fee, trade_date)
    return ZERO if row is None else ceil_cent(qty * row.rate)


def slippage_rate(costs: CostModel) -> Decimal:
    return costs.slippage_bps / Decimal(10_000)


def slipped(costs: CostModel, side: str, ref: Decimal) -> Decimal:
    """P12: adverse, not tick-rounded."""
    s = slippage_rate(costs)
    return ref * (1 + s) if side == "buy" else ref * (1 - s)


def unslipped(costs: CostModel, side: str, price: Decimal) -> Decimal:
    s = slippage_rate(costs)
    return price / (1 + s) if side == "buy" else price / (1 - s)


def max_drawdown(values: Sequence[Decimal]) -> Decimal:
    """P19: ``min_t (E_t - peak)/peak`` over ``[E_init, E(b0), ...]``; <= 0."""
    peak, worst = None, ZERO
    for value in values:
        peak = value if peak is None or value > peak else peak
        if peak:
            worst = min(worst, (value - peak) / peak)
    return worst


def daily_sharpe(session_closes: Sequence[Decimal], initial_cash: Decimal, rf_annual: Decimal):
    """P19 Sharpe over session-close equity, or ``SHARPE_NULL``."""
    if len(session_closes) < 2:
        return SHARPE_NULL
    rf_d = rf_annual / 252
    prev, excess = initial_cash, []
    for value in session_closes:
        excess.append(value / prev - 1 - rf_d)
        prev = value
    mean = sum(excess) / len(excess)
    var = sum((x - mean) ** 2 for x in excess) / (len(excess) - 1)
    if var == 0:
        return SHARPE_NULL
    return mean / var.sqrt() * Decimal(252).sqrt()


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


@dataclass
class Ledger:
    cash: Decimal
    positions: Dict[str, Decimal] = field(default_factory=dict)
    avg_cost: Dict[str, Decimal] = field(default_factory=dict)
    realized: Decimal = ZERO
    receivables: Dict[int, Decimal] = field(default_factory=dict)
    dividends: Decimal = ZERO  # booked (receivable + paid)
    last_close: Dict[str, Decimal] = field(default_factory=dict)
    equity: Dict[datetime, Decimal] = field(default_factory=dict)  # close stamp -> E
    avg_cost_path: List[Tuple[str, Optional[Decimal]]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def grid(case: Case) -> List[Tuple[datetime, int]]:
    """The case's bar grid: ``(open stamp, minutes)`` across all symbols."""
    minutes: Dict[datetime, int] = {}
    for bar in case.bars:
        if minutes.setdefault(bar.ts, bar.minutes) != bar.minutes:
            raise ValueError(f"{case.id}: bars at {bar.ts} disagree on length")
    return sorted(minutes.items())


def bar_index(case: Case) -> Dict[Tuple[str, datetime], Bar]:
    return {(bar.symbol, bar.ts): bar for bar in case.bars}


def _intent_order(case: Case) -> Dict[str, int]:
    return {intent.id: index for index, intent in enumerate(case.intents)}


def _check_price_rule(intent: OrderIntent, bar: Bar, ref: Decimal, side: str) -> Optional[str]:
    """P2/P15: the reference price an order type may fill at on this bar."""
    o = bar.open
    if intent.type == "market":
        expected = o
    elif intent.type == "limit":
        limit = intent.limit_price
        if side == "buy":
            expected = o if o <= limit else (limit if bar.low <= limit else None)
        else:
            expected = o if o >= limit else (limit if bar.high >= limit else None)
    else:
        stop = intent.stop_price
        if side == "sell":
            expected = o if o <= stop else (stop if bar.low <= stop else None)
        else:
            expected = o if o >= stop else (stop if bar.high >= stop else None)
    if expected is None:
        return f"{intent.id}: a {intent.type} {side} cannot fill on the bar at {bar.ts}"
    if ref != expected:
        return f"{intent.id}: {intent.type} {side} reference price {ref}, policy gives {expected}"
    return None


def replay(case: Case, fills: Sequence[Fill]) -> Ledger:
    """Apply ``fills`` plus the case's corporate actions in policy order."""
    costs = case.costs
    bars = bar_index(case)
    stamps = grid(case)
    intents = {intent.id: intent for intent in case.intents}
    order = _intent_order(case)
    statuses = {final.order_id: final for final in case.expected.orders}
    ledger = Ledger(cash=case.initial_cash)
    problems = ledger.problems
    fills_at: Dict[datetime, List[Fill]] = {}
    for fill in fills:
        fills_at.setdefault(fill.bar, []).append(fill)
    known = {ts for ts, _ in stamps}
    for ts in fills_at:
        if ts not in known:
            problems.append(f"fill on {ts}, which is not a case bar")

    seen_days = set()
    for position_in_grid, (ts, minutes) in enumerate(stamps):
        day = ts.date()
        if day not in seen_days:
            seen_days.add(day)
            apply_corporate_actions(case, ledger, day, ts, bars)

        # P8: one E_open snapshot per bar, after corporate actions, before fills.
        e_open = ledger.cash + sum(ledger.receivables.values(), ZERO)
        for symbol, qty in ledger.positions.items():
            bar = bars.get((symbol, ts))
            e_open += qty * (bar.open if bar else ledger.last_close[symbol])

        held_at_open = dict(ledger.positions)

        def side_of(fill: Fill) -> str:
            intent = intents[fill.order_id]
            if intent.side is not None:
                return intent.side
            target = _target_qty(intent, e_open, bars[(intent.symbol, ts)].open, costs)
            return "buy" if target > held_at_open.get(intent.symbol, ZERO) else "sell"

        # Guard before sorting: the sort key reads the intent and its bar, so a
        # fill naming an unknown order or a bar its symbol lacks must be
        # recorded and dropped here, not raise a KeyError from inside sorted().
        valid = []
        for fill in fills_at.get(ts, ()):
            intent = intents.get(fill.order_id)
            if intent is None:
                problems.append(f"fill for unknown order {fill.order_id}")
            elif (intent.symbol, ts) not in bars:
                problems.append(f"{fill.order_id}: {intent.symbol} has no bar at {ts}")
            else:
                valid.append(fill)
        todays = sorted(
            valid,
            # P4: sells before buys, each side in submission order.
            key=lambda f: (0 if side_of(f) == "sell" else 1, order[f.order_id]),
        )
        for fill in todays:
            intent = intents[fill.order_id]
            bar = bars[(intent.symbol, ts)]
            _apply_fill(
                case, ledger, intent, fill, bar, side_of(fill), e_open,
                held_at_open, statuses.get(fill.order_id), stamps, position_in_grid,
            )

        for symbol in {s for (s, t) in bars if t == ts}:
            ledger.last_close[symbol] = bars[(symbol, ts)].close
        ledger.equity[ts + timedelta(minutes=minutes)] = _mark(ledger)
    return ledger


def _mark(ledger: Ledger) -> Decimal:
    """P18: cash + sum(q * last close) + receivables."""
    value = ledger.cash + sum(ledger.receivables.values(), ZERO)
    for symbol, qty in ledger.positions.items():
        value += qty * ledger.last_close[symbol]
    return value


def _target_qty(intent: OrderIntent, e_open: Decimal, open_price: Decimal, costs: CostModel) -> Decimal:
    return floor_step(intent.target_weight * e_open / open_price, costs.qty_step)


def apply_corporate_actions(case, ledger: Ledger, day: date, ts: datetime, bars) -> None:
    """P16/P17 at the open of the first bar of ``day`` (stamped ``ts``), before
    any fill. Shared with ``scoring.actual_invariants``, which runs it over an
    engine's fills so a split or dividend is not read as a cash/position breach."""
    for split in case.splits:
        if split.ex_date != day or split.symbol not in ledger.positions:
            continue
        qty = ledger.positions[split.symbol] * split.ratio
        avg = ledger.avg_cost[split.symbol] / split.ratio
        whole = floor_step(qty, case.costs.qty_step)
        remainder = qty - whole
        if remainder:
            bar = bars.get((split.symbol, ts))
            if bar is None:
                ledger.problems.append(f"cash in lieu for {split.symbol} needs a bar at {ts}")
            else:
                ledger.cash += remainder * bar.open
                ledger.realized += remainder * (bar.open - avg)
        ledger.positions[split.symbol] = whole
        ledger.avg_cost[split.symbol] = avg
    for index, dividend in enumerate(case.dividends):
        if dividend.ex_date == day:
            # Entitled: the position at the previous session's close, which is
            # the position now -- nothing has filled yet today.
            amount = ledger.positions.get(dividend.symbol, ZERO) * dividend.amount
            ledger.receivables[index] = amount
            ledger.dividends += amount
        if dividend.pay_date == day:
            ledger.cash += ledger.receivables.pop(index, ZERO)


def _apply_fill(case, ledger, intent, fill, bar, side, e_open, held_at_open, final, stamps, position_in_grid):
    costs, problems = case.costs, ledger.problems
    oid, symbol, qty, price = fill.order_id, intent.symbol, fill.qty, fill.price
    trade_date = bar.ts.date()

    # Timing (P1-P3): never on the decision bar; a market order on the next one.
    if fill.bar <= intent.decision_bar:
        problems.append(f"{oid}: fills at {fill.bar}, not after its decision bar {intent.decision_bar}")
    if intent.type == "market":
        later = [ts for ts, _ in stamps if ts > intent.decision_bar]
        if not later or later[0] != fill.bar:
            problems.append(f"{oid}: a market order must fill on the next bar after {intent.decision_bar}")

    # Price, slippage, invariant 3.
    is_limit = intent.type == "limit"
    ref = price if is_limit else unslipped(costs, side, price)
    if not is_limit and slipped(costs, side, ref) != price:
        problems.append(f"{oid}: price {price} is not the slipped reference {ref}")
    rule = _check_price_rule(intent, bar, ref, side)
    if rule:
        problems.append(rule)
    if not bar.low <= ref <= bar.high:
        problems.append(f"invariant 3: {oid} reference price {ref} outside [{bar.low}, {bar.high}]")
    slip = qty * abs(price - ref)
    if fill.slippage_cost != slip:
        problems.append(f"{oid}: slippage_cost {fill.slippage_cost}, policy gives {slip}")

    # Fees (P10, P11).
    notional = qty * price
    fees = {
        "commission": commission(costs, qty, notional),
        "sec_fee": sec_fee(costs, side, notional, trade_date),
        "taf": taf_fee(costs, side, qty, trade_date),
    }
    for name, value in fees.items():
        if getattr(fill, name) != value:
            problems.append(f"{oid}: {name} {getattr(fill, name)}, policy gives {value}")
    total_fees = sum(fees.values(), ZERO) + cat_fee(costs, qty, trade_date)

    # Sizing (P5-P8).
    if qty <= 0 or floor_step(qty, costs.qty_step) != qty:
        problems.append(f"{oid}: qty {qty} is not a positive point on the {costs.qty_step} grid")
    held = ledger.positions.get(symbol, ZERO)
    if intent.target_weight is not None:
        target = _target_qty(intent, e_open, bar.open, costs)
        delta = target - held_at_open.get(symbol, ZERO)
        if abs(delta) != qty:
            problems.append(f"{oid}: target {target} from E_open {e_open} gives |delta| {abs(delta)}, fill is {qty}")
    if final is not None and final.reason == "clipped_to_position" and qty != held:
        problems.append(f"{oid}: clipped sell of {qty}, holding was {held}")
    if final is not None and final.reason == "volume_cap" and costs.participation_cap is not None:
        cap = floor_step(costs.participation_cap * bar.volume, costs.qty_step)
        if qty != cap:
            problems.append(f"{oid}: volume-capped fill {qty}, cap gives {cap}")

    cash_before = ledger.cash
    if side == "buy":
        ledger.cash -= notional + total_fees
        if final is not None and final.reason == "insufficient_cash":
            # P5: the largest grid quantity that fits -- one more step must not.
            step = qty + costs.qty_step
            cost_next = step * price + commission(costs, step, step * price) + cat_fee(costs, step, trade_date)
            if cost_next <= cash_before:
                problems.append(f"{oid}: reduced to {qty}, but {step} still fits {cash_before}")
        ledger.avg_cost[symbol] = (held * ledger.avg_cost.get(symbol, ZERO) + qty * price) / (held + qty)
        ledger.positions[symbol] = held + qty
        ledger.realized -= total_fees  # P9: buy fees are realized at the buy
    else:
        if qty > held:
            problems.append(f"invariant 2: {oid} sells {qty} with {held} held")
        avg = ledger.avg_cost.get(symbol, ZERO)
        ledger.cash += notional - total_fees
        ledger.realized += qty * (price - avg) - total_fees
        ledger.positions[symbol] = held - qty
        if ledger.positions[symbol] == 0:
            del ledger.positions[symbol]
            del ledger.avg_cost[symbol]
    if ledger.cash < 0:
        problems.append(f"invariant 1: cash {ledger.cash} after {oid}")
    ledger.avg_cost_path.append((oid, ledger.avg_cost.get(symbol)))


def check_orders(case: Case, fills: Sequence[Fill], orders) -> List[str]:
    """Invariant 7 plus status/quantity coherence of the stated outcomes."""
    problems = []
    by_id: Dict[str, list] = {}
    for final in orders:
        by_id.setdefault(final.order_id, []).append(final)
    filled: Dict[str, Decimal] = {}
    for fill in fills:
        filled[fill.order_id] = filled.get(fill.order_id, ZERO) + fill.qty
    for intent in case.intents:
        finals = by_id.pop(intent.id, [])
        if len(finals) != 1:
            problems.append(f"invariant 7: {intent.id} has {len(finals)} terminal OrderFinal(s)")
            continue
        final = finals[0]
        if final.reason not in ORDER_REASONS:
            problems.append(f"{intent.id}: reason {final.reason!r} is not in the vocabulary")
        if (final.status == "filled") != (final.reason == ""):
            problems.append(f"{intent.id}: status {final.status} with reason {final.reason!r}")
        if final.filled_qty != filled.get(intent.id, ZERO):
            problems.append(
                f"{intent.id}: filled_qty {final.filled_qty}, fills sum to {filled.get(intent.id, ZERO)}"
            )
        if final.status in {"rejected", "cancelled", "expired"} and final.filled_qty:
            problems.append(f"{intent.id}: {final.status} with a fill")
        if final.status in {"filled", "partially_filled"} and not final.filled_qty:
            problems.append(f"{intent.id}: {final.status} with nothing filled")
        if isinstance(intent.qty, Decimal) and intent.qty > 0:
            if final.status == "filled" and final.filled_qty != intent.qty:
                problems.append(f"{intent.id}: filled {final.filled_qty} of {intent.qty}")
            if final.status == "partially_filled" and not final.filled_qty < intent.qty:
                problems.append(f"{intent.id}: partial fill {final.filled_qty} of {intent.qty}")
    for order_id in by_id:
        problems.append(f"invariant 7: OrderFinal for unknown order {order_id}")
    return problems


def session_close_equity(case: Case, equity: Dict[datetime, Decimal]) -> List[Decimal]:
    """E_d for each session: the equity at the close of its last bar."""
    last: Dict[date, datetime] = {}
    for ts, minutes in grid(case):
        last[ts.date()] = ts + timedelta(minutes=minutes)
    return [equity[last[day]] for day in sorted(last)]
