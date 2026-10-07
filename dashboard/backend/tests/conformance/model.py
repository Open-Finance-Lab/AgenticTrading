"""Engine-neutral data model for the US backtest conformance suite.

The seed of a future ``SimulatorPort``: the same case data is meant to score the
current ATL engine, AKQuant, NautilusTrader or an ATL-owned core, each through
its own adapter. This module therefore imports **nothing from ATL** -- stdlib
only -- so a candidate engine's adapter can depend on it without pulling in the
dashboard backend.

Every money, price and quantity value is a ``Decimal``. Binary floats cannot
carry the suite's exactness contract: ``math.floor(8.0 / 1e-9)`` is
``7999999999``, so a float floor returns 7.999999999 shares for C02 instead of
8, one full ``Tol.qty`` step short. Adapters convert an engine's floats with
``to_decimal`` (``Decimal(repr(x))``, the shortest round-tripping decimal).

Design: the conformance-suite design doc of 2026-10-07 (§1 harness interface,
§2.1 dated fee table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Literal, Mapping, Optional, Protocol, Sequence, Tuple, Union
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")

#: Sentinel for an *undefined* Sharpe (P19: N < 2 or zero stdev). Distinct from
#: ``None``, which means "the engine reports no Sharpe at all".
SHARPE_NULL = "null"

Qty = Union[Decimal, Literal["all"]]


def D(value) -> Decimal:
    """Exact decimal from a literal. Strings and ints only: a float literal is
    already rounded by the time it gets here."""
    if isinstance(value, float):
        raise TypeError(f"pass {value!r} as a string; floats are not exact")
    return Decimal(value)


def to_decimal(value) -> Optional[Decimal]:
    """An engine's numeric output as a Decimal (``None`` passes through)."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(repr(float(value))) if isinstance(value, float) else Decimal(value)


class NotExpressible(Exception):
    """The engine cannot accept one of the case's *intents* as written (§1).

    Environment facts (costs, splits, dividends) never raise this: an engine
    with no input for them is run with them ignored and scored on its output.
    """


# --------------------------------------------------------------------------
# Market data and environment facts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """One case bar, open-stamped, tz-aware America/New_York."""

    symbol: str
    ts: datetime
    minutes: int  # 60, 30 (the 15:30 / early-close bucket), 390 (daily)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal(1_000_000)

    @property
    def close_ts(self) -> datetime:
        return self.ts + timedelta(minutes=self.minutes)


@dataclass(frozen=True)
class Split:
    symbol: str
    ex_date: date
    ratio: Decimal  # new shares per old: 2 = 2-for-1, 0.5 = 1-for-2


@dataclass(frozen=True)
class CashDividend:
    symbol: str
    ex_date: date
    pay_date: date
    amount: Decimal  # per share


@dataclass(frozen=True)
class DatedRate:
    effective: date
    rate: Decimal
    cap: Optional[Decimal] = None  # per trade (TAF)


def rate_on(table: Sequence[DatedRate], day: date) -> Optional[DatedRate]:
    """The row in force on ``day`` (the latest ``effective <= day``)."""
    current = None
    for row in sorted(table, key=lambda r: r.effective):
        if row.effective <= day:
            current = row
    return current


# --------------------------------------------------------------------------
# Regulatory fee parameters, dated by TRADE date (§2.1, verified 2026-10-07)
# --------------------------------------------------------------------------

#: SEC Section 31 fee, sells only, dollars per $1,000,000 of principal.
SEC_SECTION_31: Tuple[DatedRate, ...] = (
    # $27.80 was in force through 2025-05-13. The design doc records only its
    # end, so the row is open-ended backwards rather than given a start date
    # nobody verified. Source: SEC Fee Rate Advisory 2025-2,
    # https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2
    DatedRate(date.min, Decimal("27.80")),
    # $0.00 from 2025-05-14 through 2026-04-03. Sources: Advisory 2025-2 and
    # Advisory 2026-2, https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2
    DatedRate(date(2025, 5, 14), Decimal("0.00")),
    # $20.60 from 2026-04-04 until 60 days after the FY2027 appropriation (no
    # FY2027 advisory existed as of 2026-10-07 -- re-check the advisory index
    # before every refresh). Sources: Advisory 2026-2 (2026-02-27); FINRA
    # Information Notice 2026-03-17 (charge date = trade date),
    # https://www.finra.org/rules-guidance/notices/information-notice-20260317
    DatedRate(date(2026, 4, 4), Decimal("20.60")),
)

#: FINRA Trading Activity Fee, covered equity, sells only, dollars per share,
#: capped per trade. Source for every row: FINRA fee adjustment schedule,
#: SR-FINRA-2024-019 ("increases take effect on January 1 of the year
#: stated"), https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule
#: FINRA's Schedule A §1 rulebook page still showed the 2024 text when
#: checked; the schedule is the operative source. Confirm against one real
#: Alpaca sell confirmation before shipping.
FINRA_TAF: Tuple[DatedRate, ...] = (
    DatedRate(date(2025, 1, 1), Decimal("0.000166"), Decimal("8.30")),
    DatedRate(date(2026, 1, 1), Decimal("0.000195"), Decimal("9.79")),
    DatedRate(date(2027, 1, 1), Decimal("0.000232"), Decimal("11.61")),  # scheduled
)

#: CAT fee: Alpaca passes one through on buys and sells
#: (https://alpaca.markets/support/regulatory-fees) but its rate is
#: UNVERIFIED, so no row is hard-coded and ``CostModel.cat_fee`` defaults off.
CAT_FEE: Tuple[DatedRate, ...] = ()


@dataclass(frozen=True)
class CostModel:
    commission_per_share: Decimal = Decimal(0)
    commission_pct: Decimal = Decimal(0)  # of notional
    commission_min: Decimal = Decimal(0)  # per order
    slippage_bps: Decimal = Decimal(0)
    participation_cap: Optional[Decimal] = None  # fraction of fill-bar volume
    sec_fee: Tuple[DatedRate, ...] = ()  # $ per $1M principal, sells only
    taf: Tuple[DatedRate, ...] = ()  # $ per share, sells only, cap per trade
    cat_fee: Tuple[DatedRate, ...] = ()  # $ per share, buys AND sells; off
    qty_step: Decimal = Decimal("1e-9")  # 1 = whole-share mode

    @property
    def has_commission(self) -> bool:
        return bool(self.commission_per_share or self.commission_pct or self.commission_min)

    @property
    def is_default(self) -> bool:
        return self == CostModel()


# --------------------------------------------------------------------------
# Intents and outcomes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderIntent:
    id: str
    decision_bar: datetime  # open stamp of bar t; decided at its close
    symbol: str
    side: Optional[Literal["buy", "sell"]] = None  # None when target_weight is set
    qty: Optional[Qty] = None
    target_weight: Optional[Decimal] = None
    type: Literal["market", "limit", "stop"] = "market"
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    tif: Literal["day", "gtc"] = "day"
    oco_group: Optional[str] = None


OrderStatus = Literal["filled", "partially_filled", "rejected", "cancelled", "expired"]

#: The ``OrderFinal.reason`` vocabulary (§1). An adapter maps its engine's own
#: strings onto these and passes an unmapped one through verbatim, which then
#: fails the ``orders`` check rather than being silently coerced.
ORDER_REASONS = frozenset({
    "",
    "insufficient_cash",
    "invalid_quantity",
    "no_position",
    "clipped_to_position",
    "no_bar",
    "no_next_bar",
    "volume_cap",
    "session_end",
    "oco_cancelled",
    "fractional_requires_day_tif",
    "fractional_requires_simple_order",
    "below_min_notional",
    "corporate_action",
})


@dataclass(frozen=True)
class Fill:
    """``price`` is after slippage. A fee is ``None`` when the engine has no
    such field (an actual result only; expected fills always state it)."""

    order_id: str
    bar: datetime  # open stamp of the case bar the fill happened on
    price: Decimal
    qty: Decimal
    commission: Optional[Decimal] = Decimal(0)
    sec_fee: Optional[Decimal] = Decimal(0)
    taf: Optional[Decimal] = Decimal(0)
    slippage_cost: Optional[Decimal] = Decimal(0)
    #: Engine-reported side, needed only for a target-weight order, whose
    #: intent carries no side. Expected fills leave it ``None``.
    side: Optional[Literal["buy", "sell"]] = None


@dataclass(frozen=True)
class OrderFinal:
    order_id: str
    status: OrderStatus
    reason: str = ""
    filled_qty: Decimal = Decimal(0)


@dataclass(frozen=True)
class Expected:
    """A case's expected outcome -- and the adapter's RunResult shape.

    In an actual result, ``None`` means "the engine has no such output", which
    fails any check that names the field. In an expected result, ``None`` means
    "the design doc does not state it" and the field is not compared.
    """

    fills: Sequence[Fill]
    orders: Sequence[OrderFinal]
    cash: Optional[Decimal]
    positions: Optional[Mapping[str, Decimal]]
    avg_cost: Optional[Mapping[str, Decimal]]
    realized_pnl: Optional[Decimal] = None
    initial_equity: Optional[Decimal] = None
    equity: Sequence[Tuple[datetime, Decimal]] = ()  # (bar close time, value)
    final_equity: Optional[Decimal] = None
    total_return: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    sharpe: Union[Decimal, Literal["null"], None] = None
    #: (order_id, avg cost of that symbol right after the fill), for checks
    #: that pin the average cost after every fill (M02), not just at the end.
    avg_cost_path: Sequence[Tuple[str, Decimal]] = ()


@dataclass(frozen=True)
class Tol:
    money: Decimal = Decimal("1e-6")
    price: Decimal = Decimal("1e-9")
    qty: Decimal = Decimal("1e-9")  # = one grid step: one step short FAILS
    ratio: Decimal = Decimal("1e-9")
    sharpe: Decimal = Decimal("1e-6")


#: Every check name a case may list. ``invariants`` is not listed by cases: it
#: is added to every scored run (§5).
CHECKS = frozenset({
    "fills",
    "orders",
    "cash",
    "positions",
    "avg_cost",
    "realized_pnl",
    "equity",
    "final_equity",
    "initial_equity",
    "total_return",
    "max_drawdown",
    "sharpe",
})
INVARIANTS = "invariants"


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    bars: Sequence[Bar]
    initial_cash: Decimal
    intents: Sequence[OrderIntent]
    expected: Expected
    checks: frozenset
    splits: Sequence[Split] = ()
    dividends: Sequence[CashDividend] = ()
    costs: CostModel = field(default_factory=CostModel)
    rf_annual: Decimal = Decimal(0)
    tol: Tol = field(default_factory=Tol)


class Adapter(Protocol):
    """One per engine. ``run`` raises ``NotExpressible(reason)``."""

    name: str
    capabilities: frozenset

    def run(self, case: Case) -> Expected: ...
