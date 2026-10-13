"""The 28 US conformance cases (plus variants T02d and M03b) as plain data.

Every expected value is copied from the conformance-suite design doc
(2026-10-07, §4), where each is derived by hand from the stated policy (§2) --
none comes from any engine's output. ``test_case_data.py`` replays every case's
expected fills through the reference ledger (``reference.py``) so a
transcription error here goes red without any engine involved.

Each case also carries the design doc's prediction for the current dashboard
engine (column D of §7) and, for C01/C04/S01, for its LLM translator (DL). A
prediction is a verdict plus the set of checks expected to fail; both are read
off the doc's "Current:" prose. ``invariants`` is in a predicted set where that
prose implies an invariant breach (a missing ``OrderFinal`` breaks invariant 7,
negative cash breaks invariant 1). Where a run disagrees with the doc, the
marker follows the run and ``PREDICTION_DELTAS`` records why, at ``file:line``.

Ported scenarios. Several cases port scenarios from ml4t-backtest
(MIT License, Copyright (c) 2025-2026 Stefan Jansen): ``tests/oracle/test_oracle_self.py``
and ``tests/test_direction_matrix.py`` (T01), ``tests/property/test_order_sequence_invariants.py``
(C01), ``tests/accounting/test_cash_account_policy.py`` (C01, C02, C04),
``validation/scenarios/definitions.py`` scenarios 03, 04, 06, 08 and 10
(F01, F04, O02, O03), ``tests/scenarios/factory.py`` (F04),
``validation/native/evidence/zipline-3.1.1.json`` (L01),
``tests/test_order_type_matrix.py`` (O01) and ``tests/test_extreme_conditions.py``
(O02). The ported invariants (§5) come from its ``tests/property/*`` and
``tests/contracts/test_ledger_invariants.py``. C04 diverges from ml4t on
purpose: an oversell is clipped (P6), not rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Tuple

from .model import (
    FINRA_TAF,
    NEW_YORK,
    SEC_SECTION_31,
    SHARPE_NULL,
    Bar,
    Case,
    CashDividend,
    CostModel,
    D,
    DatedRate,
    Expected,
    Fill,
    OrderFinal,
    OrderIntent,
    Split,
)

# --------------------------------------------------------------------------
# Builders (§3 conventions)
# --------------------------------------------------------------------------

_HOURLY_OPENS = ((9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30))


def H(day: str, k: int) -> datetime:
    """Open stamp of hourly bar ``bk`` on ``day`` (b6 is the 15:30 bucket)."""
    hour, minute = _HOURLY_OPENS[k]
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=NEW_YORK)


def HC(day: str, k: int, minutes: Optional[int] = None) -> datetime:
    """Close stamp of hourly bar ``bk``: open + 60m, or 30m for b6."""
    return H(day, k) + timedelta(minutes=minutes or (30 if k == 6 else 60))


def DO(day: str) -> datetime:
    """Open stamp of a daily bar."""
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, 9, 30, tzinfo=NEW_YORK)


def DC(day: str) -> datetime:
    """Close stamp of a daily bar (16:00)."""
    return DO(day) + timedelta(minutes=390)


def _ohlc(spec: str) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    """``"102/104/101/103"`` or a flat ``"100"``."""
    parts = spec.split("/")
    if len(parts) == 1:
        parts = parts * 4
    o, h, l, c = (D(p) for p in parts)
    return o, h, l, c


def hbar(symbol, day, k, spec, *, volume="1000000", minutes=None) -> Bar:
    o, h, l, c = _ohlc(spec)
    return Bar(symbol, H(day, k), minutes or (30 if k == 6 else 60), o, h, l, c, D(volume))


def dbar(symbol, day, spec) -> Bar:
    o, h, l, c = _ohlc(spec)
    return Bar(symbol, DO(day), 390, o, h, l, c)


def mkt(oid, at, symbol, side, qty) -> OrderIntent:
    return OrderIntent(oid, at, symbol, side=side, qty=qty if qty == "all" else D(qty))


def tgt(oid, at, symbol, weight) -> OrderIntent:
    return OrderIntent(oid, at, symbol, target_weight=D(weight))


def lmt(oid, at, symbol, side, qty, price, *, tif="day", oco=None) -> OrderIntent:
    return OrderIntent(
        oid, at, symbol, side=side, qty=D(qty), type="limit",
        limit_price=D(price), tif=tif, oco_group=oco,
    )


def stp(oid, at, symbol, side, qty, price, *, tif="day", oco=None) -> OrderIntent:
    return OrderIntent(
        oid, at, symbol, side=side, qty=D(qty), type="stop",
        stop_price=D(price), tif=tif, oco_group=oco,
    )


def fl(oid, bar, price, qty, *, commission="0", sec="0", taf="0", slip="0") -> Fill:
    return Fill(oid, bar, D(price), D(qty), D(commission), D(sec), D(taf), D(slip))


def of(oid, status, filled, reason="") -> OrderFinal:
    return OrderFinal(oid, status, reason, D(filled))


def eq(*points) -> Tuple[Tuple[datetime, Decimal], ...]:
    return tuple((ts, D(value)) for ts, value in points)


def _pos(**held) -> Dict[str, Decimal]:
    return {symbol: D(qty) for symbol, qty in held.items()}


def _checks(*names) -> frozenset:
    return frozenset(names)


CASH = D("1000.00")  # ATL default capital (constants.py:52)
DAY = "2026-03-02"  # the hourly grid's session, a Monday


# --------------------------------------------------------------------------
# Group T: fill timing
# --------------------------------------------------------------------------

T01 = Case(
    "T01", "market order fills at the next bar's open",
    bars=(
        hbar("AAPL", DAY, 0, "100/101/99/100"),
        hbar("AAPL", DAY, 1, "102/104/101/103"),
        hbar("AAPL", DAY, 2, "103/105/102/104"),
    ),
    initial_cash=CASH,
    intents=(mkt("o1", H(DAY, 0), "AAPL", "buy", "5"),),
    checks=_checks("fills", "cash", "positions", "final_equity"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "102.00", "5"),),
        orders=(of("o1", "filled", "5"),),
        cash=D("490.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="102.00"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1005.00"), (HC(DAY, 2), "1010.00")),
        final_equity=D("1010.00"),
        total_return=D("0.010"),
    ),
)

T02 = Case(
    "T02", "a session-final decision fills at the next session's open",
    bars=(
        hbar("AAPL", "2026-03-02", 6, "100/101/99/100"),
        hbar("AAPL", "2026-03-03", 0, "95/97/94/96"),
    ),
    initial_cash=CASH,
    intents=(mkt("o1", H("2026-03-02", 6), "AAPL", "buy", "5"),),
    checks=_checks("fills", "cash", "final_equity"),
    expected=Expected(
        fills=(fl("o1", H("2026-03-03", 0), "95.00", "5"),),
        orders=(of("o1", "filled", "5"),),
        cash=D("525.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="95.00"),
        initial_equity=CASH,
        equity=eq((HC("2026-03-02", 6), "1000.00"), (HC("2026-03-03", 0), "1005.00")),
        final_equity=D("1005.00"),
    ),
)

T02d = Case(
    "T02d", "T02 on daily bars",
    bars=(dbar("AAPL", "2026-03-02", "100"), dbar("AAPL", "2026-03-03", "95/97/94/96")),
    initial_cash=CASH,
    intents=(mkt("o1", DO("2026-03-02"), "AAPL", "buy", "5"),),
    checks=_checks("fills", "cash", "final_equity"),
    expected=Expected(
        fills=(fl("o1", DO("2026-03-03"), "95.00", "5"),),
        orders=(of("o1", "filled", "5"),),
        cash=D("525.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="95.00"),
        initial_equity=CASH,
        equity=eq((DC("2026-03-02"), "1000.00"), (DC("2026-03-03"), "1005.00")),
        final_equity=D("1005.00"),
    ),
)

T03 = Case(
    "T03", "an order on the final bar of data expires unfilled",
    bars=(hbar("AAPL", DAY, 0, "100"), hbar("AAPL", DAY, 1, "100/101/99/101")),
    initial_cash=CASH,
    intents=(mkt("o1", H(DAY, 1), "AAPL", "buy", "5"),),
    checks=_checks("fills", "orders", "cash"),
    expected=Expected(
        fills=(),
        orders=(of("o1", "expired", "0", "no_next_bar"),),
        cash=D("1000.00"),
        positions={},
        avg_cost={},
        initial_equity=CASH,
        equity=eq((HC(DAY, 0), "1000.00"), (HC(DAY, 1), "1000.00")),
        final_equity=D("1000.00"),
    ),
)

T04 = Case(
    "T04", "a missing fill bar cancels the market order; the held position marks stale",
    bars=(
        hbar("AAPL", DAY, 0, "50"),
        hbar("AAPL", DAY, 1, "50/52/50/52"),
        # b2: no AAPL bar
        hbar("AAPL", DAY, 3, "53/55/53/55"),
        hbar("MSFT", DAY, 0, "100"),
        hbar("MSFT", DAY, 1, "100"),
        hbar("MSFT", DAY, 2, "100"),
        hbar("MSFT", DAY, 3, "100"),
    ),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "10"),
        mkt("o2", H(DAY, 1), "AAPL", "buy", "5"),
    ),
    checks=_checks("fills", "orders", "cash", "positions", "equity"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "50.00", "10"),),
        orders=(of("o1", "filled", "10"), of("o2", "cancelled", "0", "no_bar")),
        cash=D("500.00"),
        positions=_pos(AAPL="10"),
        avg_cost=_pos(AAPL="50.00"),
        initial_equity=CASH,
        equity=eq(
            (HC(DAY, 0), "1000.00"),
            (HC(DAY, 1), "1020.00"),
            (HC(DAY, 2), "1020.00"),
            (HC(DAY, 3), "1050.00"),
        ),
        final_equity=D("1050.00"),
    ),
)

_HALF_DAY = "2026-11-27"  # Friday after Thanksgiving, closes 13:00
T05 = Case(
    "T05", "an early-close day's 12:30 bucket is final and fills at the next session's open",
    bars=(
        hbar("AAPL", _HALF_DAY, 0, "100"),
        hbar("AAPL", _HALF_DAY, 1, "100"),
        hbar("AAPL", _HALF_DAY, 2, "100"),
        hbar("AAPL", _HALF_DAY, 3, "100", minutes=30),  # closes 13:00
        hbar("AAPL", "2026-11-30", 0, "102/104/101/103"),
    ),
    initial_cash=CASH,
    intents=(mkt("o1", H(_HALF_DAY, 3), "AAPL", "buy", "5"),),
    checks=_checks("fills", "orders", "cash", "positions", "equity"),
    expected=Expected(
        fills=(fl("o1", H("2026-11-30", 0), "102.00", "5"),),
        orders=(of("o1", "filled", "5"),),
        cash=D("490.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="102.00"),
        initial_equity=CASH,
        equity=eq((HC(_HALF_DAY, 3, 30), "1000.00"), (HC("2026-11-30", 0), "1005.00")),
        final_equity=D("1005.00"),
    ),
)

# --------------------------------------------------------------------------
# Group C: cash and ordering
# --------------------------------------------------------------------------


def _flat_pair(last_k, aapl="100", msft="50"):
    return tuple(hbar("AAPL", DAY, k, aapl) for k in range(last_k + 1)) + tuple(
        hbar("MSFT", DAY, k, msft) for k in range(last_k + 1)
    )


C01 = Case(
    "C01", "sells execute before buys within a bar",
    bars=_flat_pair(2),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "10"),
        mkt("o2", H(DAY, 1), "MSFT", "buy", "10"),  # listed first
        mkt("o3", H(DAY, 1), "AAPL", "sell", "5"),
    ),
    checks=_checks("fills", "orders", "cash", "positions"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100", "10"),
            fl("o3", H(DAY, 2), "100", "5"),
            fl("o2", H(DAY, 2), "50", "10"),
        ),
        orders=(of("o1", "filled", "10"), of("o2", "filled", "10"), of("o3", "filled", "5")),
        cash=D("0.00"),
        positions=_pos(AAPL="5", MSFT="10"),
        avg_cost=_pos(AAPL="100", MSFT="50"),
        initial_equity=CASH,
        final_equity=D("1000.00"),
    ),
)

C02 = Case(
    "C02", "a buy that no longer fits at the fill price is reduced",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "125/126/124/125"),
        hbar("AAPL", DAY, 2, "125"),
    ),
    initial_cash=CASH,
    intents=(mkt("o1", H(DAY, 0), "AAPL", "buy", "10"),),
    checks=_checks("fills", "orders", "cash", "positions"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "125", "8"),),
        orders=(of("o1", "partially_filled", "8", "insufficient_cash"),),
        cash=D("0.00"),
        positions=_pos(AAPL="8"),
        avg_cost=_pos(AAPL="125"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1000.00")),
    ),
)

C03 = Case(
    "C03", "two buys exceed cash in aggregate; the second in submission order is reduced",
    bars=_flat_pair(1),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "6"),
        mkt("o2", H(DAY, 0), "MSFT", "buy", "10"),
    ),
    checks=_checks("fills", "orders", "cash", "positions"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "100", "6"), fl("o2", H(DAY, 1), "50", "8")),
        orders=(of("o1", "filled", "6"), of("o2", "partially_filled", "8", "insufficient_cash")),
        cash=D("0.00"),
        positions=_pos(AAPL="6", MSFT="8"),
        avg_cost=_pos(AAPL="100", MSFT="50"),
        initial_equity=CASH,
        final_equity=D("1000.00"),
    ),
)

C04 = Case(
    "C04", "sell quantity guards: negative, zero, oversell, unheld",
    bars=_flat_pair(2),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "10"),
        mkt("o2", H(DAY, 1), "AAPL", "sell", "-5"),
        mkt("o3", H(DAY, 1), "AAPL", "sell", "0"),
        mkt("o4", H(DAY, 1), "AAPL", "sell", "12"),
        mkt("o5", H(DAY, 1), "MSFT", "sell", "5"),
    ),
    checks=_checks("fills", "orders", "cash", "positions"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "100", "10"), fl("o4", H(DAY, 2), "100", "10")),
        orders=(
            of("o1", "filled", "10"),
            of("o2", "rejected", "0", "invalid_quantity"),
            of("o3", "rejected", "0", "invalid_quantity"),
            of("o4", "partially_filled", "10", "clipped_to_position"),
            of("o5", "rejected", "0", "no_position"),
        ),
        cash=D("1000.00"),
        positions={},
        avg_cost={},
        initial_equity=CASH,
        final_equity=D("1000.00"),
    ),
)

# --------------------------------------------------------------------------
# Group S: sizing
# --------------------------------------------------------------------------

S01 = Case(
    "S01", "target-weight orders with fractional quantities",
    bars=(
        hbar("AAPL", DAY, 0, "300"),
        hbar("AAPL", DAY, 1, "320/330/320/330"),
        hbar("AAPL", DAY, 2, "375"),
    ),
    initial_cash=CASH,
    intents=(tgt("o1", H(DAY, 0), "AAPL", "0.25"), tgt("o2", H(DAY, 1), "AAPL", "0.25")),
    checks=_checks("fills", "cash", "positions", "avg_cost", "realized_pnl", "equity"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "320", "0.78125"), fl("o2", H(DAY, 2), "375", "0.0859375")),
        orders=(of("o1", "filled", "0.78125"), of("o2", "filled", "0.0859375")),
        cash=D("782.2265625"),
        positions=_pos(AAPL="0.6953125"),
        avg_cost=_pos(AAPL="320.00"),
        realized_pnl=D("4.7265625"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1007.8125"), (HC(DAY, 2), "1042.96875")),
        final_equity=D("1042.96875"),
    ),
)

S02 = Case(
    "S02", "whole-share mode floors target weights and leaves the residual in cash",
    bars=(
        hbar("AAPL", DAY, 0, "300"),
        hbar("AAPL", DAY, 1, "310/320/310/320"),
        hbar("MSFT", DAY, 0, "90"),
        hbar("MSFT", DAY, 1, "90/92/90/92"),
    ),
    initial_cash=CASH,
    costs=CostModel(qty_step=D("1")),
    intents=(tgt("o1", H(DAY, 0), "AAPL", "0.5"), tgt("o2", H(DAY, 0), "MSFT", "0.5")),
    checks=_checks("fills", "cash", "positions"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "310", "1"), fl("o2", H(DAY, 1), "90", "5")),
        orders=(of("o1", "filled", "1"), of("o2", "filled", "5")),
        cash=D("240.00"),
        positions=_pos(AAPL="1", MSFT="5"),
        avg_cost=_pos(AAPL="310", MSFT="90"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1020.00")),
    ),
)

# --------------------------------------------------------------------------
# Group F: costs
# --------------------------------------------------------------------------

F01 = Case(
    "F01", "commission per share with a per-order minimum",
    bars=_flat_pair(2, aapl="100", msft="2.00"),
    initial_cash=CASH,
    costs=CostModel(commission_per_share=D("0.005"), commission_min=D("1.00")),
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "4"),
        mkt("o2", H(DAY, 0), "MSFT", "buy", "250"),
        mkt("o3", H(DAY, 1), "AAPL", "sell", "all"),
        mkt("o4", H(DAY, 1), "MSFT", "sell", "all"),
    ),
    checks=_checks("fills", "cash", "realized_pnl"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100", "4", commission="1.00"),
            fl("o2", H(DAY, 1), "2.00", "250", commission="1.25"),
            fl("o3", H(DAY, 2), "100", "4", commission="1.00"),
            fl("o4", H(DAY, 2), "2.00", "250", commission="1.25"),
        ),
        orders=(
            of("o1", "filled", "4"),
            of("o2", "filled", "250"),
            of("o3", "filled", "4"),
            of("o4", "filled", "250"),
        ),
        cash=D("995.50"),
        positions={},
        avg_cost={},
        realized_pnl=D("-4.50"),
        initial_equity=CASH,
        final_equity=D("995.50"),
    ),
)

_SEC_F02 = (DatedRate(date(2025, 5, 14), D("0.00")), DatedRate(date(2026, 4, 4), D("20.60")))

F02 = Case(
    "F02", "SEC §31 fee: sells only, dated by fill trade date, rounded up to the cent",
    bars=tuple(dbar("AAPL", d, "100") for d in ("2026-03-31", "2026-04-01", "2026-04-02", "2026-04-06")),
    initial_cash=CASH,
    costs=CostModel(sec_fee=_SEC_F02),
    intents=(
        mkt("o1", DO("2026-03-31"), "AAPL", "buy", "10"),
        mkt("o2", DO("2026-04-01"), "AAPL", "sell", "5"),
        mkt("o3", DO("2026-04-02"), "AAPL", "sell", "5"),
    ),
    checks=_checks("fills", "cash"),
    expected=Expected(
        fills=(
            fl("o1", DO("2026-04-01"), "100", "10"),
            fl("o2", DO("2026-04-02"), "100", "5", sec="0.00"),
            fl("o3", DO("2026-04-06"), "100", "5", sec="0.02"),
        ),
        orders=(of("o1", "filled", "10"), of("o2", "filled", "5"), of("o3", "filled", "5")),
        cash=D("999.98"),
        positions={},
        avg_cost={},
        initial_equity=CASH,
        final_equity=D("999.98"),
    ),
)

#: F02's fee-function table, for adapters that expose the fee function:
#: principal -> SEC fee at the $20.60 rate, after the per-fill ceiling.
F02_SEC_FEE_TABLE = {D("50000"): D("1.03"), D("1000"): D("0.03"), D("500"): D("0.02")}

F03 = Case(
    "F03", "FINRA TAF: per share, sells only, cap per trade, Jan-1 dating",
    bars=tuple(dbar("AAPL", d, "1.00") for d in ("2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02")),
    initial_cash=D("100000"),
    costs=CostModel(
        sec_fee=_SEC_F02,
        taf=(
            DatedRate(date(2025, 1, 1), D("0.000166"), D("8.30")),
            DatedRate(date(2026, 1, 1), D("0.000195"), D("9.79")),
        ),
    ),
    intents=(
        mkt("o1", DO("2025-12-29"), "AAPL", "buy", "90000"),
        mkt("o2", DO("2025-12-30"), "AAPL", "sell", "30000"),
        mkt("o3", DO("2025-12-31"), "AAPL", "sell", "59990"),
        mkt("o4", DO("2025-12-31"), "AAPL", "sell", "10"),
    ),
    checks=_checks("fills", "cash"),
    expected=Expected(
        fills=(
            fl("o1", DO("2025-12-30"), "1.00", "90000"),
            fl("o2", DO("2025-12-31"), "1.00", "30000", taf="4.98"),
            fl("o3", DO("2026-01-02"), "1.00", "59990", taf="9.79"),
            fl("o4", DO("2026-01-02"), "1.00", "10", taf="0.01"),
        ),
        orders=(
            of("o1", "filled", "90000"),
            of("o2", "filled", "30000"),
            of("o3", "filled", "59990"),
            of("o4", "filled", "10"),
        ),
        cash=D("99985.22"),
        positions={},
        avg_cost={},
        initial_equity=D("100000"),
        final_equity=D("99985.22"),
    ),
)

F04 = Case(
    "F04", "slippage in bps, adverse, not tick-rounded",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "100/110/100/110"),
        hbar("AAPL", DAY, 2, "110"),
    ),
    initial_cash=CASH,
    costs=CostModel(slippage_bps=D("10")),
    intents=(mkt("o1", H(DAY, 0), "AAPL", "buy", "5"), mkt("o2", H(DAY, 1), "AAPL", "sell", "all")),
    checks=_checks("fills", "cash", "avg_cost", "realized_pnl"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100.10", "5", slip="0.50"),
            fl("o2", H(DAY, 2), "109.89", "5", slip="0.55"),
        ),
        orders=(of("o1", "filled", "5"), of("o2", "filled", "5")),
        cash=D("1048.95"),
        positions={},
        avg_cost={},  # the position is closed, so the final check is vacuous (see doc)
        realized_pnl=D("48.95"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1049.50")),
    ),
)

F05 = Case(
    "F05", "a buy reduced to fit its commission, then a full-stack round-trip sell",
    bars=(
        hbar("AAPL", "2026-05-04", 0, "200"),
        hbar("AAPL", "2026-05-04", 1, "200/250/200/250"),
        hbar("AAPL", "2026-05-04", 2, "250"),
    ),
    initial_cash=CASH,
    costs=CostModel(commission_min=D("1.00"), sec_fee=SEC_SECTION_31, taf=FINRA_TAF),
    intents=(
        mkt("o1", H("2026-05-04", 0), "AAPL", "buy", "5"),
        mkt("o2", H("2026-05-04", 1), "AAPL", "sell", "all"),
    ),
    checks=_checks("fills", "orders", "cash", "realized_pnl"),
    expected=Expected(
        fills=(
            fl("o1", H("2026-05-04", 1), "200", "4.995", commission="1.00"),
            fl("o2", H("2026-05-04", 2), "250", "4.995", commission="1.00", sec="0.03", taf="0.01"),
        ),
        orders=(
            of("o1", "partially_filled", "4.995", "insufficient_cash"),
            of("o2", "filled", "4.995"),
        ),
        cash=D("1247.71"),
        positions={},
        avg_cost={},
        realized_pnl=D("247.71"),
        initial_equity=CASH,
        equity=eq((HC("2026-05-04", 1), "1248.75")),
        final_equity=D("1247.71"),
    ),
)

#: F05 under Alpaca's summed-EOD fee rounding (not scored; open question Q3).
F05_SUMMED_EOD_ROUNDING = {"o2_fees": D("1.03"), "cash": D("1247.72"), "realized_pnl": D("247.72")}

# --------------------------------------------------------------------------
# Group L: liquidity
# --------------------------------------------------------------------------

L01 = Case(
    "L01", "the participation cap gives a partial fill; the remainder is cancelled",
    bars=(
        hbar("AAPL", DAY, 0, "100", volume="1000"),
        hbar("AAPL", DAY, 1, "100", volume="50"),
        hbar("AAPL", DAY, 2, "100/102/100/102", volume="1000"),
    ),
    initial_cash=CASH,
    costs=CostModel(participation_cap=D("0.10")),
    intents=(mkt("o1", H(DAY, 0), "AAPL", "buy", "8"),),
    checks=_checks("fills", "orders", "cash", "equity"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "100", "5"),),
        orders=(of("o1", "partially_filled", "5", "volume_cap"),),
        cash=D("500.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="100"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1000.00"), (HC(DAY, 2), "1010.00")),
        final_equity=D("1010.00"),
    ),
)

# --------------------------------------------------------------------------
# Group O: order types
# --------------------------------------------------------------------------

O01 = Case(
    "O01", "limit buys: one touched intrabar, one gapping through at the open",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "99/100/97/98"),
        hbar("AAPL", DAY, 2, "95/97/94/96"),
        hbar("MSFT", DAY, 0, "50"),
        hbar("MSFT", DAY, 1, "50/51/47/49"),
        hbar("MSFT", DAY, 2, "49/50/48/50"),
    ),
    initial_cash=CASH,
    intents=(
        lmt("o1", H(DAY, 0), "AAPL", "buy", "5", "96"),
        lmt("o2", H(DAY, 0), "MSFT", "buy", "5", "48"),
    ),
    checks=_checks("fills", "orders", "cash", "equity"),
    expected=Expected(
        fills=(fl("o2", H(DAY, 1), "48.00", "5"), fl("o1", H(DAY, 2), "95.00", "5")),
        orders=(of("o1", "filled", "5"), of("o2", "filled", "5")),
        cash=D("285.00"),
        positions=_pos(AAPL="5", MSFT="5"),
        avg_cost=_pos(AAPL="95.00", MSFT="48.00"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1005.00"), (HC(DAY, 2), "1015.00")),
        final_equity=D("1015.00"),
    ),
)

O02 = Case(
    "O02", "stop-loss sells: one triggered intrabar, one gapping through",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "100/101/99/100"),
        hbar("AAPL", DAY, 2, "98/99/93/94"),
        hbar("MSFT", DAY, 0, "50"),
        hbar("MSFT", DAY, 1, "50"),
        hbar("MSFT", DAY, 2, "44/46/43/45"),
    ),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "5"),
        mkt("o2", H(DAY, 0), "MSFT", "buy", "5"),
        stp("o3", H(DAY, 1), "AAPL", "sell", "5", "95", tif="gtc"),
        stp("o4", H(DAY, 1), "MSFT", "sell", "5", "47", tif="gtc"),
    ),
    checks=_checks("fills", "cash", "equity"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100", "5"),
            fl("o2", H(DAY, 1), "50", "5"),
            fl("o3", H(DAY, 2), "95.00", "5"),
            fl("o4", H(DAY, 2), "44.00", "5"),
        ),
        orders=(
            of("o1", "filled", "5"),
            of("o2", "filled", "5"),
            of("o3", "filled", "5"),
            of("o4", "filled", "5"),
        ),
        cash=D("945.00"),
        positions={},
        avg_cost={},
        initial_equity=CASH,
        equity=eq((HC(DAY, 1), "1000.00"), (HC(DAY, 2), "945.00")),
        final_equity=D("945.00"),
    ),
)

O03 = Case(
    "O03", "bracket (OCO): take-profit hit, and same-bar ambiguity resolved stop-first",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "100"),
        hbar("AAPL", DAY, 2, "101/111/99/105"),
        hbar("MSFT", DAY, 0, "50"),
        hbar("MSFT", DAY, 1, "50"),
        hbar("MSFT", DAY, 2, "50/56/44/52"),
    ),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "5"),
        mkt("o2", H(DAY, 0), "MSFT", "buy", "5"),
        stp("o3", H(DAY, 1), "AAPL", "sell", "5", "95", oco="g1"),
        lmt("o4", H(DAY, 1), "AAPL", "sell", "5", "110", oco="g1"),
        stp("o5", H(DAY, 1), "MSFT", "sell", "5", "45", oco="g2"),
        lmt("o6", H(DAY, 1), "MSFT", "sell", "5", "55", oco="g2"),
    ),
    checks=_checks("fills", "orders", "cash"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100", "5"),
            fl("o2", H(DAY, 1), "50", "5"),
            fl("o4", H(DAY, 2), "110.00", "5"),
            fl("o5", H(DAY, 2), "45.00", "5"),
        ),
        orders=(
            of("o1", "filled", "5"),
            of("o2", "filled", "5"),
            of("o3", "cancelled", "0", "oco_cancelled"),
            of("o4", "filled", "5"),
            of("o5", "filled", "5"),
            of("o6", "cancelled", "0", "oco_cancelled"),
        ),
        cash=D("1025.00"),
        positions={},
        avg_cost={},
        initial_equity=CASH,
        final_equity=D("1025.00"),
    ),
)

O04 = Case(
    "O04", "time in force: DAY expires, GTC carries over, fractional GTC is rejected",
    bars=(
        hbar("AAPL", "2026-03-02", 5, "100"),
        hbar("AAPL", "2026-03-02", 6, "100/100/99/99.5"),
        hbar("AAPL", "2026-03-03", 0, "99/99/96/97"),
    ),
    initial_cash=CASH,
    intents=(
        lmt("o1", H("2026-03-02", 5), "AAPL", "buy", "2", "97", tif="day"),
        lmt("o2", H("2026-03-02", 5), "AAPL", "buy", "3", "97", tif="gtc"),
        lmt("o3", H("2026-03-02", 5), "AAPL", "buy", "0.5", "97", tif="gtc"),
    ),
    checks=_checks("fills", "orders", "cash"),
    expected=Expected(
        fills=(fl("o2", H("2026-03-03", 0), "97.00", "3"),),
        orders=(
            of("o1", "expired", "0", "session_end"),
            of("o2", "filled", "3"),
            of("o3", "rejected", "0", "fractional_requires_day_tif"),
        ),
        cash=D("709.00"),
        positions=_pos(AAPL="3"),
        avg_cost=_pos(AAPL="97.00"),
        initial_equity=CASH,
        equity=eq((HC("2026-03-03", 0), "1000.00")),
        final_equity=D("1000.00"),
    ),
)

# --------------------------------------------------------------------------
# Group CA: corporate actions (daily bars)
# --------------------------------------------------------------------------

CA01 = Case(
    "CA01", "a 2-for-1 split doubles the quantity, halves the avg cost, books no P&L",
    bars=(
        dbar("AAPL", "2026-06-01", "200"),
        dbar("AAPL", "2026-06-02", "200"),
        dbar("AAPL", "2026-06-03", "101/102/100/102"),
        dbar("AAPL", "2026-06-04", "102/104/102/104"),
    ),
    initial_cash=CASH,
    splits=(Split("AAPL", date(2026, 6, 3), D("2")),),
    intents=(
        mkt("o1", DO("2026-06-01"), "AAPL", "buy", "4"),
        mkt("o2", DO("2026-06-03"), "AAPL", "sell", "3"),  # post-split shares
    ),
    checks=_checks("positions", "avg_cost", "cash", "realized_pnl", "equity"),
    expected=Expected(
        fills=(fl("o1", DO("2026-06-02"), "200", "4"), fl("o2", DO("2026-06-04"), "102", "3")),
        orders=(of("o1", "filled", "4"), of("o2", "filled", "3")),
        cash=D("506.00"),
        positions=_pos(AAPL="5"),
        avg_cost=_pos(AAPL="100.00"),
        realized_pnl=D("6.00"),
        initial_equity=CASH,
        equity=eq(
            (DC("2026-06-02"), "1000.00"),
            (DC("2026-06-03"), "1016.00"),
            (DC("2026-06-04"), "1026.00"),
        ),
        final_equity=D("1026.00"),
    ),
)

_CA02_AAPL = (
    dbar("AAPL", "2026-05-04", "50"),
    dbar("AAPL", "2026-05-05", "50"),
    dbar("AAPL", "2026-05-06", "49/49.5/48.5/49"),
    dbar("AAPL", "2026-05-07", "49"),
    dbar("AAPL", "2026-05-08", "49"),
)
_AAPL_DIV = CashDividend("AAPL", date(2026, 5, 6), date(2026, 5, 8), D("1.00"))

CA02 = Case(
    "CA02", "cash dividend: receivable at the ex-date, cash at the pay-date",
    bars=_CA02_AAPL,
    initial_cash=CASH,
    dividends=(_AAPL_DIV,),
    intents=(mkt("o1", DO("2026-05-04"), "AAPL", "buy", "10"),),
    checks=_checks("cash", "equity"),
    expected=Expected(
        fills=(fl("o1", DO("2026-05-05"), "50", "10"),),
        orders=(of("o1", "filled", "10"),),
        cash=D("510.00"),
        positions=_pos(AAPL="10"),
        avg_cost=_pos(AAPL="50"),
        initial_equity=CASH,
        equity=eq(
            (DC("2026-05-05"), "1000.00"),
            (DC("2026-05-06"), "1000.00"),
            (DC("2026-05-07"), "1000.00"),
            (DC("2026-05-08"), "1000.00"),
        ),
        final_equity=D("1000.00"),
        total_return=D("0"),
    ),
)

CA03 = Case(
    "CA03", "dividend entitlement edges: sell at the ex open keeps it, buy there does not earn it",
    bars=_CA02_AAPL + (
        dbar("MSFT", "2026-05-04", "100"),
        dbar("MSFT", "2026-05-05", "100"),
        dbar("MSFT", "2026-05-06", "98"),
        dbar("MSFT", "2026-05-07", "98"),
        dbar("MSFT", "2026-05-08", "98"),
    ),
    initial_cash=CASH,
    dividends=(_AAPL_DIV, CashDividend("MSFT", date(2026, 5, 6), date(2026, 5, 8), D("2.00"))),
    intents=(
        mkt("o1", DO("2026-05-04"), "MSFT", "buy", "5"),
        mkt("o2", DO("2026-05-05"), "MSFT", "sell", "5"),
        mkt("o3", DO("2026-05-05"), "AAPL", "buy", "10"),
    ),
    checks=_checks("cash", "positions", "equity"),
    expected=Expected(
        fills=(
            fl("o1", DO("2026-05-05"), "100", "5"),
            fl("o2", DO("2026-05-06"), "98", "5"),
            fl("o3", DO("2026-05-06"), "49", "10"),
        ),
        orders=(of("o1", "filled", "5"), of("o2", "filled", "5"), of("o3", "filled", "10")),
        cash=D("510.00"),
        positions=_pos(AAPL="10"),
        avg_cost=_pos(AAPL="49"),
        initial_equity=CASH,
        equity=eq((DC("2026-05-06"), "1000.00"), (DC("2026-05-08"), "1000.00")),
        final_equity=D("1000.00"),
    ),
)

CA04 = Case(
    "CA04", "a split cancels working orders; whole-share mode pays cash in lieu",
    bars=(
        dbar("AAPL", "2026-06-01", "150"),
        dbar("AAPL", "2026-06-02", "150"),
        dbar("AAPL", "2026-06-03", "101/102/99/100"),
        dbar("AAPL", "2026-06-04", "100"),
    ),
    initial_cash=CASH,
    costs=CostModel(qty_step=D("1")),
    splits=(Split("AAPL", date(2026, 6, 3), D("1.5")),),
    intents=(
        mkt("o1", DO("2026-06-01"), "AAPL", "buy", "5"),
        lmt("o2", DO("2026-06-02"), "AAPL", "buy", "2", "140", tif="gtc"),
    ),
    checks=_checks("fills", "orders", "cash", "positions", "avg_cost", "realized_pnl", "equity"),
    expected=Expected(
        fills=(fl("o1", DO("2026-06-02"), "150", "5"),),
        orders=(of("o1", "filled", "5"), of("o2", "cancelled", "0", "corporate_action")),
        cash=D("300.50"),
        positions=_pos(AAPL="7"),
        avg_cost=_pos(AAPL="100.00"),
        realized_pnl=D("0.50"),
        initial_equity=CASH,
        equity=eq(
            (DC("2026-06-02"), "1000.00"),
            (DC("2026-06-03"), "1000.50"),
            (DC("2026-06-04"), "1000.50"),
        ),
        final_equity=D("1000.50"),
    ),
)

# --------------------------------------------------------------------------
# Group M: marking, cost basis, metrics
# --------------------------------------------------------------------------

# The step-session path pins the same back-marking in an existing test, which
# asserts 13 post-trade marks (tests/test_external_minute_source.py:117).
M01 = Case(
    "M01", "the equity curve is pre-trade per bar; initial_equity is the starting cash",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "102/111/101/110"),
        hbar("AAPL", DAY, 2, "120/121/119/120"),
        hbar("AAPL", DAY, 3, "120/122/120/121"),
    ),
    initial_cash=CASH,
    intents=(mkt("o1", H(DAY, 0), "AAPL", "buy", "5"), mkt("o2", H(DAY, 1), "AAPL", "buy", "4")),
    checks=_checks("equity", "initial_equity", "max_drawdown", "total_return", "avg_cost"),
    expected=Expected(
        fills=(fl("o1", H(DAY, 1), "102", "5"), fl("o2", H(DAY, 2), "120", "4")),
        orders=(of("o1", "filled", "5"), of("o2", "filled", "4")),
        cash=D("10.00"),
        positions=_pos(AAPL="9"),
        avg_cost=_pos(AAPL="110.00"),
        initial_equity=D("1000.00"),
        equity=eq(
            (HC(DAY, 0), "1000.00"),
            (HC(DAY, 1), "1040.00"),
            (HC(DAY, 2), "1090.00"),
            (HC(DAY, 3), "1099.00"),
        ),
        final_equity=D("1099.00"),
        total_return=D("0.099"),
        max_drawdown=D("0"),
    ),
)

M02 = Case(
    "M02", "average cost and realized P&L across partial sells and re-buys",
    bars=(
        hbar("AAPL", DAY, 0, "100"),
        hbar("AAPL", DAY, 1, "100/120/100/120"),
        hbar("AAPL", DAY, 2, "120/130/120/130"),
        hbar("AAPL", DAY, 3, "130/130/90/90"),
        hbar("AAPL", DAY, 4, "90"),
    ),
    initial_cash=CASH,
    intents=(
        mkt("o1", H(DAY, 0), "AAPL", "buy", "4"),
        mkt("o2", H(DAY, 1), "AAPL", "buy", "4"),
        mkt("o3", H(DAY, 2), "AAPL", "sell", "2"),
        mkt("o4", H(DAY, 3), "AAPL", "buy", "2"),
    ),
    checks=_checks("avg_cost", "realized_pnl", "final_equity"),
    expected=Expected(
        fills=(
            fl("o1", H(DAY, 1), "100", "4"),
            fl("o2", H(DAY, 2), "120", "4"),
            fl("o3", H(DAY, 3), "130", "2"),
            fl("o4", H(DAY, 4), "90", "2"),
        ),
        orders=(
            of("o1", "filled", "4"),
            of("o2", "filled", "4"),
            of("o3", "filled", "2"),
            of("o4", "filled", "2"),
        ),
        cash=D("200"),
        positions=_pos(AAPL="8"),
        avg_cost=_pos(AAPL="105.00"),
        realized_pnl=D("40.00"),
        initial_equity=CASH,
        equity=eq((HC(DAY, 4), "920.00")),
        final_equity=D("920.00"),
        avg_cost_path=(("o1", D("100")), ("o2", D("110.00")), ("o3", D("110")), ("o4", D("105.00"))),
    ),
)

_M03_BARS = (
    dbar("AAPL", "2026-03-02", "100"),
    dbar("AAPL", "2026-03-03", "100/102/100/102"),
    dbar("AAPL", "2026-03-04", "102/102/96.9/96.9"),
    dbar("AAPL", "2026-03-05", "96.9/101.745/96.9/101.745"),
    dbar("AAPL", "2026-03-06", "101.745/102.76245/101.745/102.76245"),
)

M03 = Case(
    "M03", "daily metrics: TR, MDD, and Sharpe with rf and ddof=1",
    bars=_M03_BARS,
    initial_cash=CASH,
    rf_annual=D("0.0252"),
    intents=(mkt("o1", DO("2026-03-02"), "AAPL", "buy", "10"),),
    checks=_checks("total_return", "max_drawdown", "sharpe"),
    expected=Expected(
        fills=(fl("o1", DO("2026-03-03"), "100", "10"),),
        orders=(of("o1", "filled", "10"),),
        cash=D("0"),
        positions=_pos(AAPL="10"),
        avg_cost=_pos(AAPL="100"),
        initial_equity=CASH,
        equity=eq(
            (DC("2026-03-02"), "1000"),
            (DC("2026-03-03"), "1020"),
            (DC("2026-03-04"), "969"),
            (DC("2026-03-05"), "1017.45"),
            (DC("2026-03-06"), "1027.6245"),
        ),
        final_equity=D("1027.6245"),
        total_return=D("0.0276245"),
        max_drawdown=D("-0.05"),
        sharpe=D("2.568186"),
    ),
)

#: The design doc gives M03b as "cash only, 3 sessions, no intents" without
#: bars; it reuses M03's first three sessions, which leaves every expected
#: number unchanged (a cash-only book is flat on any prices).
M03b = Case(
    "M03b", "M03 sub-case: cash only, 3 sessions, undefined Sharpe is null",
    bars=_M03_BARS[:3],
    initial_cash=CASH,
    rf_annual=D("0.0252"),
    intents=(),
    checks=_checks("total_return", "max_drawdown", "sharpe"),
    expected=Expected(
        fills=(),
        orders=(),
        cash=CASH,
        positions={},
        avg_cost={},
        initial_equity=CASH,
        final_equity=CASH,
        total_return=D("0"),
        max_drawdown=D("0"),
        sharpe=SHARPE_NULL,
    ),
)

#: The 28 cases of §4, in doc order.
CASES = (
    T01, T02, T03, T04, T05,
    C01, C02, C03, C04,
    S01, S02,
    F01, F02, F03, F04, F05,
    L01,
    O01, O02, O03, O04,
    CA01, CA02, CA03, CA04,
    M01, M02, M03,
)
#: Doc-named variants scored alongside, but outside the 28-case tally.
VARIANTS = (T02d, M03b)
ALL_CASES = CASES + VARIANTS
BY_ID = {case.id: case for case in ALL_CASES}

# --------------------------------------------------------------------------
# Predictions for the current engine (§7) and the deltas measured against them
# --------------------------------------------------------------------------

PASS, FAIL, NOT_EXPRESSIBLE = "PASS", "FAIL", "N/E"


@dataclass(frozen=True)
class Prediction:
    verdict: str  # PASS | FAIL | N/E
    fails: frozenset = frozenset()  # checks expected to fail (FAIL only)
    defects: Tuple[str, ...] = ()  # §7 root-defect ids
    note: str = ""  # one line: the mechanism (FAIL) or the reason (N/E)


def _p(verdict, fails=(), defects=(), note=""):
    return Prediction(verdict, frozenset(fails), tuple(defects), note)


_NE_SIZING = "D9: actions carry whole shares only; no target-weight or fractional intent (execution.py:442-445)"
_NE_ORDERS = "D12: actions are market-only; no limit, stop, OCO or TIF (execution.py:442-445)"

#: Column D: the dashboard engine through the scripted seam (§8.2).
PREDICTIONS_D: Dict[str, Prediction] = {
    "T01": _p(PASS, note="fills at the 5m bar opening at the decision stamp"),
    "T02": _p(FAIL, ("fills", "cash", "final_equity"), ("D2", "D16"),
              "the 16:00 bucket fills at the 15:55 bar's close (bar_aggregation.py:402-410)"),
    "T02d": _p(FAIL, ("fills", "cash", "final_equity"), ("D2", "D16"),
               "no daily timeframe; the 16:00 decision fills at the same close"),
    "T03": _p(FAIL, ("orders", "invariants"), ("D8",),
              "a decision with no fill bar is not a step, so o1 leaves no record (engine.py:1949)"),
    "T04": _p(FAIL, ("orders", "invariants"), ("D8",),
              "a fill with no source bar hits a bare continue, no order event (execution.py:526-527)"),
    "T05": _p(FAIL, ("fills", "orders", "cash", "positions", "equity", "invariants"), ("D19", "D8"),
              "the session is hard-coded to 16:00, so the 13:00 bucket never fills (sessions.py:40-44)"),
    "C01": _p(FAIL, ("fills", "orders", "cash", "positions"), ("D5",),
              "actions run in listed order; the buy is rejected before the sell funds it (execution.py:442)"),
    "C02": _p(FAIL, ("fills", "orders", "cash", "positions"), ("D6", "D8"),
              "buys are all-or-nothing, so o1 is rejected (execution.py:612-626)"),
    "C03": _p(FAIL, ("fills", "orders", "cash", "positions"), ("D6",),
              "buys are all-or-nothing, so o2 is rejected outright (execution.py:612-626)"),
    "C04": _p(FAIL, ("fills", "orders", "cash", "positions", "invariants"), ("D7", "D8"),
              "a -5 sell adds shares and drives cash to -500 (execution.py:812-826)"),
    "S01": _p(NOT_EXPRESSIBLE, defects=("D9",), note=_NE_SIZING),
    "S02": _p(NOT_EXPRESSIBLE, defects=("D9",), note=_NE_SIZING),
    "F01": _p(FAIL, ("fills", "cash", "realized_pnl"), ("D10", "D14"),
              "the US profile has no costs (profiles.py:198-218); no realized P&L field"),
    "F02": _p(FAIL, ("fills", "cash"), ("D10", "D2", "D16"),
              "no SEC fee; daily decisions fill at the same close"),
    "F03": _p(FAIL, ("fills", "cash"), ("D10", "D2", "D16"),
              "no per-share or capped fee component (execution.py:336-352)"),
    "F04": _p(FAIL, ("fills", "cash", "realized_pnl"), ("D10", "D14"),
              "no slippage on the US profile; no realized P&L field"),
    "F05": _p(FAIL, ("fills", "orders", "cash", "realized_pnl"), ("D10", "D6", "D14"),
              "no fees, so the unreduced buy of 5 passes and sells at 250"),
    "L01": _p(FAIL, ("fills", "orders", "cash", "equity"), ("D11",),
              "the executor never reads volume, so all 8 fill"),
    "O01": _p(NOT_EXPRESSIBLE, defects=("D12",), note=_NE_ORDERS),
    "O02": _p(NOT_EXPRESSIBLE, defects=("D12",), note=_NE_ORDERS),
    "O03": _p(NOT_EXPRESSIBLE, defects=("D12",), note=_NE_ORDERS),
    "O04": _p(NOT_EXPRESSIBLE, defects=("D12",), note=_NE_ORDERS),
    "CA01": _p(FAIL, ("positions", "avg_cost", "realized_pnl", "equity"), ("D13", "D14"),
               "raw bars and no split ledger: the position stays 4 (alpaca_bars.py:732)"),
    "CA02": _p(FAIL, ("cash", "equity"), ("D13",), "no dividend is credited"),
    "CA03": _p(FAIL, ("cash", "equity"), ("D13", "D2"),
               "no dividends; the 05-05 orders fill at the 05-05 close"),
    "CA04": _p(NOT_EXPRESSIBLE, defects=("D12", "D13"), note=_NE_ORDERS),
    "M01": _p(FAIL, ("equity", "initial_equity", "max_drawdown", "avg_cost"), ("D1", "D3", "D14"),
              "each decision's trade is marked back over earlier bars (engine.py:2267-2302)"),
    "M02": _p(FAIL, ("avg_cost", "realized_pnl"), ("D14",),
              "entry_prices is overwritten by each buy (execution.py:631); no realized P&L"),
    "M03": _p(FAIL, ("sharpe",), ("D15",),
              "per-5m returns, ddof 0, rf 0, sqrt(19,656) annualization (metrics.py:17-64)"),
    "M03b": _p(FAIL, ("sharpe",), ("D15",), "an undefined Sharpe is reported as 0 (metrics.py:57-58)"),
}

#: Column DL: the same engine with the LLM translator
#: (``PortfolioManager.make_trading_decision_with_llm``) fed scripted JSON.
#: Non-strict, the path the doc's C04 prediction describes.
PREDICTIONS_DL: Dict[str, Prediction] = {
    "C01": _p(FAIL, ("fills", "orders", "cash", "positions", "invariants"), ("D5", "D18"),
              "the buy is pre-checked on pre-batch cash and dropped; the SELL liquidates all 10"),
    "C04": _p(FAIL, ("fills", "orders", "invariants"), ("D18",),
              "SELL size is ignored and the batch is cut to one action per symbol"),
    "S01": _p(FAIL, ("fills", "cash", "positions", "avg_cost", "realized_pnl", "equity", "invariants"),
              ("D9",), "int(position_size) truncates the fractional size to 0"),
}

#: Where a run disagreed with the design doc's prediction. Keyed
#: ``"<case>/<path>"``; each entry is one line, verified at file:line. The
#: marker on the test follows the run, not the doc.
PREDICTION_DELTAS: Dict[str, str] = {
    "C04/DL": (
        "also fails `fills`, not only `orders`: the translator sizes o2's -5 SELL at the "
        "full holding (portfolio_manager.py:918-931), so the 10-share liquidation is o2's "
        "fill and §1's (order_id, bar) matching finds no o4 fill -- the doc's mechanism, "
        "one more failing check than its prose names"
    ),
}
