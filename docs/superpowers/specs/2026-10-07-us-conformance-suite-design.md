> **Status (added 2026-10-13).** Written 2026-10-07 by workflow `wf_bd4c3031-d4b`, then amended by its two verifier reviews and by the owner decisions at the end, all in a session scratchpad that was later wiped. Recovered verbatim by replaying every recorded edit, and checked against four snapshots of the file captured during that session. Everything below the line is the original. Later facts:
>
> - **Implemented in #623** under `dashboard/backend/tests/conformance/`: case data, an independent Decimal reference ledger, the current-engine adapter, and a strict `known_failures.json`. Observed result: D is **1 PASS / 20 FAIL / 7 not expressible**, as predicted in §0. The DL (LLM translator) probe fails C01, C04 and S01. One deviation from §7: C04-DL also fails on fills. No expected value was disputed during implementation.
> - **Column S (step sessions) and D17 are out of scope** (see "Owner decisions"). They are kept here for reference.
> - **Open PR #626** changes the comparison view's Sharpe annualization (D20) by inferring the bar interval from timestamps. That does not match owner decision 5 (P19: daily Sharpe, computed once by the engine and displayed, not recomputed, by the frontend).
> - Path mapping for references below: `scratchpad/engine-report.md` → `2026-10-07-backtest-engine-adoption-research.md` in this directory. `scratchpad/oss2/ml4t-backtest` was a throwaway local checkout of the public ml4t-backtest repo. `calc_check.py` (the arithmetic re-check) was not recovered; `tests/conformance/reference.py` now does that job.
> - Code citations point at `origin/main` @ `e8736d07` and have drifted since.

---

# US Backtest Conformance Suite: design (Phase 1, item 2)

- **Date:** 2026-10-07
- **Scope:** US equities only. Alpaca SIP data, hourly and daily bars, open-stamped, regular hours, long-only cash account. A-share is out of scope.
- **Purpose:** this is the yardstick that every candidate engine is scored against: the current ATL engine, AKQuant, NautilusTrader, and an ATL-owned core. Every expected number below is derived by hand from the stated policy. None of them comes from any engine's output.
- **Code cited:** citations point at `origin/main` @ `e8736d07`. The working tree is on an older branch, and its `engine.py` lines sit about 17 lines lower than `origin/main` after line 512. Paths are relative to `dashboard/backend/`.

---

## 0. Summary

- **Size:** 28 cases in 8 groups: timing (T), cash and ordering (C), sizing (S), costs (F), liquidity (L), order types (O), corporate actions (CA), and marking and metrics (M). On top of these sit 7 invariants that run on every case and an optional parametric family ported from ml4t.
- **Shape:** each case uses 2 to 5 bars, 1 or 2 symbols and round numbers. Every expected value has its arithmetic shown.
- **Predicted result for the current engine:**
  - Dashboard engine (D): **1 PASS, 20 FAIL, 7 not expressible**.
  - Step sessions (S): **1 PASS, 19 FAIL, 8 not expressible**.
  - The two paths share one executor, so their verdicts differ only on F03. Their mechanisms differ in C01, C04, S01 and S02.
  - The error inventory in §7 lists 19 defects.
- **Revision:** amended 2026-10-07 after two independent verifier reviews; see the Review log at the end.

---

## 1. Harness interface (the seed of `SimulatorPort`)

Case files are plain data: one module per group, `pytest`, and no engine imports. Each engine has one adapter that translates the case data.

```python
# conformance/model.py  (no ATL imports)
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Optional, Sequence

@dataclass(frozen=True)
class Bar:            # open-stamped, tz-aware America/New_York
    symbol: str; ts: datetime; minutes: int          # 60, 30 (15:30 bucket), 390 (daily)
    open: float; high: float; low: float; close: float; volume: float = 1_000_000

@dataclass(frozen=True)
class Split:
    symbol: str; ex_date: date; ratio: float         # new shares per old (2.0 = 2-for-1, 0.5 = 1-for-2)

@dataclass(frozen=True)
class CashDividend:
    symbol: str; ex_date: date; pay_date: date; amount: float   # per share

@dataclass(frozen=True)
class DatedRate:
    effective: date; rate: float; cap: Optional[float] = None   # cap per trade (TAF)

@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0
    commission_pct: float = 0.0          # of notional
    commission_min: float = 0.0          # per order
    slippage_bps: float = 0.0
    participation_cap: Optional[float] = None    # fraction of fill-bar volume
    sec_fee: tuple[DatedRate, ...] = ()          # $ per $1M principal, sells only
    taf: tuple[DatedRate, ...] = ()              # $ per share, sells only, cap per trade
    cat_fee: tuple[DatedRate, ...] = ()          # $ per share, buys AND sells; rate unverified, off by default
    qty_step: float = 1e-9                       # 1.0 = whole-share mode

@dataclass(frozen=True)
class OrderIntent:
    id: str; decision_bar: datetime              # open stamp of bar t; decided at its close
    symbol: str
    side: Literal["buy", "sell"] | None = None   # None when target_weight is set
    qty: float | Literal["all"] | None = None
    target_weight: Optional[float] = None
    type: Literal["market", "limit", "stop"] = "market"
    limit_price: Optional[float] = None; stop_price: Optional[float] = None
    tif: Literal["day", "gtc"] = "day"
    oco_group: Optional[str] = None              # bracket legs share a group

@dataclass(frozen=True)
class Case:
    id: str; bars: Sequence[Bar]; initial_cash: float; intents: Sequence[OrderIntent]
    splits: Sequence[Split] = (); dividends: Sequence[CashDividend] = ()
    costs: CostModel = CostModel(); rf_annual: float = 0.0
    checks: frozenset[str] = frozenset({"fills", "orders", "cash", "positions", "final_equity"})
    expected: "Expected" = None; tol: "Tol" = None

@dataclass(frozen=True)
class Fill:
    order_id: str; bar: datetime; price: float; qty: float      # price after slippage
    commission: float = 0.0; sec_fee: float = 0.0; taf: float = 0.0; slippage_cost: float = 0.0

@dataclass(frozen=True)
class OrderFinal:
    order_id: str
    status: Literal["filled", "partially_filled", "rejected", "cancelled", "expired"]
    reason: str = ""      # insufficient_cash | invalid_quantity | no_position | clipped_to_position |
                          # no_bar | no_next_bar | volume_cap | session_end | oco_cancelled |
                          # fractional_requires_day_tif | fractional_requires_simple_order |
                          # below_min_notional | corporate_action
    filled_qty: float = 0.0

@dataclass(frozen=True)
class Expected:       # also the adapter's RunResult shape
    fills: Sequence[Fill]; orders: Sequence[OrderFinal]
    cash: float; positions: dict[str, float]; avg_cost: dict[str, float]
    realized_pnl: Optional[float] = None
    initial_equity: Optional[float] = None
    equity: Sequence[tuple[datetime, float]] = ()    # (bar close time, value)
    total_return: Optional[float] = None; max_drawdown: Optional[float] = None
    sharpe: Optional[float] | Literal["null"] = None

@dataclass(frozen=True)
class Tol:
    money: float = 1e-6; price: float = 1e-9; qty: float = 1e-9; ratio: float = 1e-9; sharpe: float = 1e-6
```

```python
# conformance/port.py
class Adapter(Protocol):
    name: str
    capabilities: frozenset[str]   # e.g. {"market", "limit", "stop", "oco", "tif", "fractional",
                                   #       "target_weight", "daily", "splits", "dividends",
                                   #       "costs", "volume_cap", "order_lifecycle"}
    def run(self, case: Case) -> Expected: ...   # raises NotExpressible(reason)
```

**Scoring.**
- Each case compares only the fields named in `case.checks`. A case is PASS, FAIL, or NOT_EXPRESSIBLE.
- **NOT_EXPRESSIBLE** means the engine's input surface cannot accept the case's **intents** as written, or refuses or alters one of its explicit inputs (an order type, TIF, target weight or fractional size it cannot submit; a capital or order-size clamp). The adapter raises `NotExpressible(reason)`.
- **Environment facts are not intents.** Costs (`CostModel`) and corporate actions (`splits`, `dividends`) describe the market, not the agent. An engine with no input for them is still run, with those facts ignored, and scored on its output, which is then a FAIL. Treating them as NOT_EXPRESSIBLE would let an engine that silently ignores fees or splits look merely incomplete rather than wrong.
- NOT_EXPRESSIBLE counts against the engine for adoption, but it is reported separately from FAIL. A field the adapter cannot read (it returns `None`) is a FAIL on any check that names that field.
- Fills are matched by `(order_id, bar)`. An adapter for an engine without order ids matches by `(bar, symbol, side, submission sequence)`.
- **Orders** (`OrderFinal`) are matched the same way: by `order_id`, or for an engine without ids by `(decision bar, symbol, side, submission sequence)`. Every intent must produce exactly one `OrderFinal`; a missing one is a FAIL on `orders`. The adapter maps the engine's own status and reason strings onto the `OrderFinal` vocabulary and states the mapping in its module.
- Equity is compared at each case-bar close. If an engine emits finer marks, the adapter samples the last mark whose price is that bar's close.

**Exact arithmetic.** Every expected value in §4 is exact. `floor_step` (P5, P7, P8) and ceil-to-cent (P10, P11) are defined over exact decimal arithmetic: the reference computation uses `Decimal`, or integer nano-shares and integer cents. In binary floating point `math.floor(8.0/1e-9) = 7999999999`, so a naive float floor returns 7.999999999 for C02 instead of 8, and a float product such as `30000 × 0.000166 × 100 = 497.99999999999994` can push a ceil one cent the wrong way. `Tol.qty = 1e-9` equals one grid step on purpose: an engine that floors one step short **fails**, because that is a real (if tiny) policy violation. An engine may use floats internally only with a guarded floor and ceil that reproduce the exact results.

---

## 2. Target semantics (policy)

Every line is a policy the suite asserts, with one sentence of why.

| # | Policy | Why |
|---|---|---|
| P1 | A decision at the close of bar `t` sees only bars `≤ t`. | No look-ahead. |
| P2 | A market order decided on bar `t` fills at the **open of bar `t+1`**. After the session's final bar, "t+1" is the next session's first bar. Sessions follow the exchange calendar, including early closes (13:00 ET), so on a half day the final bar is the 12:30 bucket. | It is the first price observable after the decision. One rule for every bar removes today's zero-latency 16:00 fill (`bar_aggregation.py:402-410`). *Labelled optimism:* bar `t` closes at the instant bar `t+1` opens, so this is still **zero latency** relative to the decision stamp; a live LLM decision takes seconds to minutes and cannot reach the opening print. No look-ahead is involved. A latency parameter (e.g. fill at the open of the first 5m bar after the decision plus a delay) is future work; case bars cannot express sub-bar timing. |
| P3 | Every order becomes active at the open of `t+1`. Nothing fills on bar `t`. | Same reason as P2. Limit and stop orders follow it too. |
| P4 | Within a bar, corporate actions apply first, at the open of the ex-date session. Then every fill on that bar runs **sells before buys**, each side in submission order, with cash re-checked after every fill. | The outcome must not depend on the order of the agent's JSON array. A rebalance's sells must fund its buys. *Known divergences:* (a) on Alpaca, a buy submitted at the same open as a sell is checked against buying power at submission and can be rejected before the sell fills; (b) sale proceeds are reusable immediately, which models a 1x-buying-power margin-type account, **not** a US cash account. T+1 settlement, good-faith violations and pattern-day-trader rules are out of scope. |
| P5 | Cash never goes below 0 (cash account, no margin, no shorting). If a buy does not fit at its fill price including fees and slippage, it is **reduced** to the largest quantity on the qty grid that fits. If that quantity is 0, the buy is rejected with `insufficient_cash`. The status is `partially_filled` with reason `insufficient_cash`. **Solver:** `cost(q) = q·p_f + buy-side fees(q)` is monotone non-decreasing in `q`, so the largest fitting `q` is found exactly by binary search over the integer grid index `n` (`q = n·qty_step`, `0 ≤ n ≤ requested/qty_step`), about 40 iterations at the default step. A closed form may seed the search, but with ceil-to-cent fees and a commission minimum it is not exact on its own. | The decision-time price is stale by construction, because the fill is at the next open. Rejection would turn opening-gap noise into an all-or-nothing outcome; reduction keeps the intent and records the gap. *Known divergence:* Alpaca live rejects such orders at submission. |
| P6 | Sells are long-only. A sell for more than the holding is **clipped** to the holding (`clipped_to_position`). A sell with no position is rejected (`no_position`). A quantity `≤ 0` or non-finite is rejected (`invalid_quantity`). | The same "reduce and record" rule as P5. Shorting is out of scope. |
| P7 | Quantities are **fractional** by default: `qty_step = 1e-9`, floored toward zero. A fractional quantity (one not on the integer grid) requires `tif=day`; a fractional GTC order is rejected with `fractional_requires_day_tif`. A fractional order cannot carry an `oco_group` (`fractional_requires_simple_order`). A fractional order, or a buy that P5 reduces to a fractional quantity, whose notional at the fill price is below **$1.00** is rejected (`below_min_notional`). Whole-share mode is `qty_step = 1`. | Alpaca supports fractional market, limit, stop and stop-limit **DAY** orders only (so no bracket/OCO order class), with qty to 9 decimals and "as little as $1 worth" ([Alpaca fractional trading](https://docs.alpaca.markets/docs/fractional-trading)). Without these rules, an order Alpaca would refuse fills in the harness. ATL capital is $1k–$3k (`constants.py:52-59`), and a whole-share quantum at that size forces the ~85% single-name concentration recorded in memory. |
| P8 | A `target_weight` order resolves **at the fill**: `target_qty = floor_step(w × E_open / O(t+1))`, where `E_open = cash + Σ q_i × O_i(t+1)`, using the last close for any symbol without a bar. `E_open` is a **snapshot computed once per bar**, after corporate actions and before any fill on that bar; every target-weight order on the bar resolves against the same snapshot, and fills, fees and slippage on that bar do not move it. `delta = target_qty − held` becomes a sell or a buy, then P4–P5 apply. | The weight is the intent, and the price is known only at the fill. One snapshot keeps the result independent of fill order (S02 relies on it). |
| P9 | **Average cost.** On a buy, `avg ← (q·avg + q_f·p_f)/(q + q_f)`, where `p_f` includes slippage and excludes fees. A sell leaves `avg` unchanged. **Realized P&L.** A sell realizes `q_f·(p_f − avg) − sell fees`, and buy fees are realized at the buy. Identity: `E − E_init = realized + Σ q·(close − avg) + dividends`. | This is the same convention as Nautilus (`avg_px_open`) and ml4t. It keeps the `pnl_pct` shown to the agent price-only. |
| P10 | `commission = max(min, per_share·q + pct·notional)`, charged per fill and rounded **up** to the cent. Every parameter defaults to 0. | Alpaca equities are commission-free, but the comparison must be able to price other brokers. |
| P11 | Sells pay regulatory fees: **SEC** `= principal × rate(trade date)/1e6` and **TAF** `= min(q × rate(trade date), cap)`. Each fee is rounded **up** to the cent **per fill and per component**. Buys pay neither SEC nor TAF. The TAF cap applies per execution; because an order produces at most one fill (P14), per fill, per order and per execution coincide here. A separate **CAT** fee (`cat_fee`, per share, buys and sells) is supported but defaults to 0 because its rate is unverified. The trade date is the fill bar's session date, not the decision date. | Alpaca's support page says SEC and TAF are each "rounded up to the nearest penny", with TAF "applied on a per-trade basis … and capped" ([Alpaca support](https://alpaca.markets/support/regulatory-fees)). Alpaca's docs page instead says fees accrue intraday and the day's **total** is charged per account at EOD and rounded up ([docs](https://docs.alpaca.markets/docs/regulatory-fees)). Per-fill per-component ceilings are chosen because they are attributable to one fill and are an **upper bound** on the EOD-total method (a sum of ceilings ≥ the ceiling of the sum); see §2.1 and Q3. The SEC charge date is the trade date ([FINRA Information Notice 2026-03-17](https://www.finra.org/rules-guidance/notices/information-notice-20260317)). |
| P12 | Slippage `s = bps/1e4`. A buy fills at `ref × (1+s)`, a sell at `ref × (1−s)`. It applies to market fills (ref = open) and stop fills (ref = open or stop price), not to limit fills. The price is **not** tick-rounded, and the reported `slippage_cost = q × |p_f − ref|`. | Slippage models the half-spread and impact as a cost. A limit price is a hard bound. |
| P13 | If the symbol has **no bar** at a market order's fill bar, the order is `cancelled/no_bar` and **not carried** forward. Limit and stop orders stay working under their TIF. A held symbol with no bar is marked at its last close and flagged stale. | The agent re-decides every bar, so carrying a market order to a later price it never saw creates fills nobody priced. *Labelled simplification:* a real DAY market order into a halt stays working until the reopen. Limit and stop orders are carried because their price bound is part of the intent; a market order has none. |
| P14 | Participation cap: the fill qty is at most `floor_step(cap × V(fill bar))`. The remainder is `cancelled/volume_cap` (status `partially_filled`). The cap is off by default. | It bounds capacity. Cancelling the remainder keeps one fill per order (see Q2). |
| P15 | **Limit buy L:** if `O ≤ L`, fill at `O`; else if `Low ≤ L`, fill at `L`. **Limit sell L:** if `O ≥ L`, fill at `O`; else if `High ≥ L`, fill at `L`. **Stop sell S:** if `O ≤ S`, fill at `O`; else if `Low ≤ S`, fill at `S` (stop buy mirrored). **OCO:** if both legs are touched on one bar with no gap at the open, the **stop wins**. **DAY** expires after the last bar of the session in which the order became active (`expired/session_end`); **GTC** persists. | Bars give no intrabar order, so OCO stop-first and gap-through fills at the open are the conservative choices; gap-through-stop fills at the open, as in ml4t's `test_extreme_conditions.py`. *Labelled optimism:* a limit fills on a mere **touch** (`Low = L`), with certainty; a real touch fills only with queue-priority probability below 1, and the conservative convention is a trade-through (`Low < L`). Every limit fill in §4 trades strictly through its limit (O01 47 < 48, O03 111 > 110, O04 96 < 97), so the cases pass under either convention and the choice stays open. A stop fills exactly at `S` unless `slippage_bps > 0` (P12), which understates stop-market slippage at the default. |
| P16 | **Split** (ratio `k`), at the ex-date open before any fills: `q ← q·k`, `avg ← avg/k`. Every **working order** on that symbol is `cancelled/corporate_action`. No other cash moves and no P&L is booked, except that in whole-share mode the fractional remainder `r` is sold as **cash in lieu** at `O(ex)`: cash `+r × O(ex)`, realized `r × (O(ex) − avg)`, no fees (Q4). Dividends do not touch working orders. | Removes the phantom P&L that a split inside the window creates on raw prices. Cancelling working orders follows Alpaca, which cancels every GTC order on a reverse split and, where enabled, every open GTC order on a symbol with an upcoming mandatory corporate action ([Alpaca mandatory corporate actions FAQ](https://docs.alpaca.markets/docs/mandatory-corporate-actions)). *Known divergence:* on a forward split Alpaca instead adjusts GTC buy limits and sell stops and leaves other orders unadjusted. A pre-split sell limit left unadjusted would fill at a price nobody set, and the agent re-decides every bar, so cancel is the safe uniform rule. DAY orders never span an ex-date open, so only GTC orders are affected. |
| P17 | **Cash dividend:** the entitled quantity is the position at the **close of the session before the ex-date**. A receivable `q × amount` is booked at the ex-date open, and cash is credited at the pay-date open. Equity includes receivables. | The ex-date price drop is offset by the receivable, so the drop is not a loss in total equity. |
| P18 | **Equity at bar close:** `E(t) = cash + Σ q × close_t + receivables`. It includes fills on bar `t` and never fills of later bars. `E_init = initial cash`, stamped at the first bar's open; it is also `initial_equity`. | A curve must show the book as it stood at that close. Today the next decision's trade is written backwards over the curve (`engine.py:2267-2302`). |
| P19 | **Metrics.** `total_return = E_final / initial_cash − 1`. `max_drawdown = min_t (E_t − max_{s≤t} E_s)/max_{s≤t} E_s` over `[E_init, E(b0), E(b1), …]`, which is ≤ 0. **Sharpe:** for each session `d = 1..N` in the window, `E_d` is the equity at the close of that session's last bar, with `E_0 = initial_cash`. `r_d = E_d/E_{d−1} − 1`, `rf_d = rf_annual/252`, `x_d = r_d − rf_d`, and `Sharpe = mean(x)/stdev(x, ddof=1) × √252`. If `N < 2` or `stdev = 0`, Sharpe is **null** (Q5). Every session in the window counts, including idle sessions before the first fill (`r = 0`); `rf_d` is simple (`/252`), not compounded. Agent and baseline curves use the same session grid and the same `rf_annual`. | Sharpe is computed on a daily grid, the same for agent and baselines, with a stated risk-free rate. Including idle sessions deflates volatility but keeps every run on one grid. With N = 5 (M03) the value is statistically meaningless; it is there only to pin the formula. |
| P20 | Every order ends in exactly one terminal state with a reason, and that state is engine output. | Today rejections are missing or reconstructed after the fact by the API layer (`execution.py:526-527`; `domain/runs/service.py:1027, 1333`). |

### 2.1 Regulatory fee parameters (dated, verified 2026-10-07)

| Fee | Rate | Effective (trade date) | Source |
|---|---|---|---|
| SEC §31 (sells, per $1M principal) | $27.80 | through 2025-05-13 | [SEC Fee Rate Advisory 2025-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2) |
| | **$0.00** | 2025-05-14 → 2026-04-03 | Advisory 2025-2; [Advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) |
| | **$20.60** | 2026-04-04 → 60 days after the FY2027 appropriation | [Advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) (2026-02-27); FINRA [Info Notice 2026-03-17](https://www.finra.org/rules-guidance/notices/information-notice-20260317) (charge date = trade date) |
| FINRA TAF, covered equity (sells, per share) | $0.000166, max $8.30/trade | 2025 | [FINRA fee adjustment schedule, SR-FINRA-2024-019](https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule) |
| | **$0.000195, max $9.79/trade** | 2026-01-01 | same ("increases take effect on January 1 of the year stated") |
| | $0.000232, max $11.61 | 2027-01-01 (scheduled) | same |

**Caveats.**
- **SEC FY2027.** I found no SEC FY2027 advisory as of 2026-10-07; $20.60 holds until one is issued, so check the advisory index before every refresh.
- **FINRA rulebook page is stale.** FINRA's [Schedule A §1 rulebook page](https://www.finra.org/rules-guidance/rulebooks/corporate-organization/section-1-member-regulatory-fees) still showed the 2024 text ($0.000166/$8.30) when I fetched it. The fee adjustment schedule is the operative source; confirm the rate against one real Alpaca sell confirmation before shipping.
- **Alpaca rounding: two Alpaca sources disagree.** The support page says each fee is rounded up per trade; the docs page says the day's fees are accrued, summed per account and rounded up once at EOD ([docs](https://docs.alpaca.markets/docs/regulatory-fees), re-read 2026-10-07). The gap is **up to $0.01 per fee component per fill**, not "less than $0.01 per fill": F05's o2 owes SEC 0.02572425 + TAF 0.000974025 = 0.026698275, which is 0.04 under per-component ceilings and 0.03 under a summed ceiling. On a tiny sell the over-charge can exceed 100% of the true fee (0.0113 becomes 0.03). Per-fill per-component rounding is the chosen policy because it is attributable and is an upper bound; the summed-EOD alternative is recorded per case where it changes a number (F05) and is part of Q3. Settle it against one real Alpaca sell confirmation before shipping.
- **CAT fee.** Alpaca also passes through a CAT fee, per trade, on buys and sells ([Alpaca support](https://alpaca.markets/support/regulatory-fees)), but I did not verify its rate. `cat_fee` defaults to 0 and is flagged; it is not hard-coded. Until it is set, costs published "net" (Q3) omit it.

---

## 3. Case conventions

- **Hourly grid (`H`, session 2026-03-02, a Monday):** b0 09:30, b1 10:30, b2 11:30, b3 12:30, b4 13:30, b5 14:30, b6 15:30 (a 30-minute bucket that closes at 16:00). Bars are written `O/H/L/C`, with volume 1,000,000 unless stated. Flat `100` means `100/100/100/100`.
- **Daily grid (`D`):** one bar per session, open-stamped 09:30, 390 minutes.
- **Intents:** `@bk BUY 5 AAPL MKT DAY [o1]` means order o1, decided at the close of bk.
- **Early close (`H½`, session 2026-11-27, the Friday after Thanksgiving, closes 13:00):** b0 09:30, b1 10:30, b2 11:30, b3 12:30 (a 30-minute bucket that closes at 13:00).
- **Symbols:** AAPL and MSFT, both in DJIA-30, so protocol paths accept them (`api/v2/models.py:90-95`, the `ActionItem` universe validator, restricts to DJIA-30). Prices are synthetic.
- **Defaults:** cash 1,000.00 (ATL default, `constants.py:52`); costs 0; `qty_step = 1e-9`; `rf = 0`; the `Tol()` defaults; exact arithmetic (§1).
- **Equity notation:** `E(bk)` is the equity at bk's close and `E_init = 1,000.00`.
- **Isolating defects:** unless the gap is the point of the case, every case sets `O(t+1) = C(t)` at fill bars. That keeps the back-marking defect (§7 D1) out of cases that test something else; M01 owns that defect. The cases where a gap at a fill bar **is** the point are T01, T02, T05, C02, S01, S02, O01–O04, CA03 and M01. Every other case (including M02, F04 and F05 after this revision) holds the convention.

---

## 4. Cases

### Group T: fill timing

**T01: market order fills at the next bar's open.**
- Checks: fills, cash, positions, final equity.
- Bars, AAPL H: b0 `100/101/99/100`; b1 `102/104/101/103`; b2 `103/105/102/104`.
- Intent: `@b0 BUY 5 AAPL MKT DAY [o1]`.
- Expected:
  - o1 fills at b1 at `O(b1) = 102.00`, qty 5, cost `5×102 = 510.00`.
  - Cash `1,000 − 510 = 490.00`. AAPL 5 at avg 102.00.
  - `E(b1) = 490 + 5×103 = 1,005.00`; `E(b2) = 490 + 5×104 = 1,010.00`, which is final. TR = `10/1,000 = 0.010`.
- Credit: ml4t `tests/oracle/test_oracle_self.py` (NEXT_BAR timing), `tests/test_direction_matrix.py` (next-bar-open).
- **Current:**
  - D **PASS**: it fills at the 5m bar opening at the decision stamp (`bar_aggregation.py:394-401`; `engine.py:2244-2273`).
  - S **PASS** (`external_run_service.py:786-806`).
  - Variant: a 60m-source or `vnpy_simulation` run **FAILS**. It fills at 100, the decision bar's close (`engine.py:1928-1936`; `external_run_service.py:446-453`).

**T02: a session-final decision fills at the next session's open.** It has a daily variant, T02d.
- Checks: fills, cash, final equity.
- Bars, AAPL: d1 = 2026-03-02 b6 `100/101/99/100`; d2 = 2026-03-03 b0 `95/97/94/96`. T02d uses daily bars D1 `100/100/100/100` and D2 `95/97/94/96`.
- Intent: `@d1.b6 BUY 5 AAPL MKT DAY [o1]` (T02d: `@D1`).
- Expected:
  - o1 fills at d2.b0 at **95.00**, cost 475.00. Cash 525.00.
  - `E(d1.b6) = 1,000.00`; `E(d2.b0) = 525 + 5×96 = 1,005.00`.
- **Current:**
  - D **FAIL**. The 16:00 bucket fills at the 15:55 bar's close, 100 (`bar_aggregation.py:402-410`). Cash 500 and `E(d2.b0) = 500 + 480 = 980.00`.
  - S **FAIL**, same result.
  - T02d: neither path has a daily decision timeframe (`profiles.py:198-218`; `external_run_service.py:297-299`). The adapter maps it to the 16:00 decision, so it fails the same way.

**T03: an order on the final bar of data expires unfilled.**
- Checks: fills, orders, cash.
- Bars, AAPL H: b0 flat 100; b1 `100/101/99/101`. The data ends after b1.
- Intent: `@b1 BUY 5 AAPL MKT DAY [o1]`.
- Expected: no fill; o1 `expired/no_next_bar`, filled 0. Cash 1,000.00; `E(b0) = E(b1) = 1,000.00`.
- **Current:**
  - D **FAIL** on orders. A decision with no fill is not a step (`engine.py:1949`; planner docstring `bar_aggregation.py:384-385`), so o1 is never offered and nothing records it. Cash passes.
  - If the last bar is a session-final bucket, it fills at that same close instead (`bar_aggregation.py:402-410`).
  - S **FAIL**, same result (`market_data_store.py:510-515`).

**T04: a missing fill bar cancels the market order, and the held position is marked at its last close.**
- Checks: fills, orders, cash, positions, equity.
- Bars, H:

  | bar | AAPL | MSFT |
  |---|---|---|
  | b0 | flat 50 | flat 100 |
  | b1 | `50/52/50/52` | flat 100 |
  | b2 | **no bar** | flat 100 |
  | b3 | `53/55/53/55` | flat 100 |

- Intents: `@b0 BUY 10 AAPL [o1]`; `@b1 BUY 5 AAPL [o2]`.
- Expected:
  - o1 fills at b1 @ 50.00, cost 500, so cash 500.00.
  - o2's fill bar is b2, which is missing, so o2 is `cancelled/no_bar` and is **not** filled at b3.
  - Equity: `E(b0) = 1,000`; `E(b1) = 500 + 10×52 = 1,020.00`; `E(b2) = 500 + 10×52 (stale) = 1,020.00`; `E(b3) = 500 + 10×55 = 1,050.00`.
  - AAPL 10 at avg 50.00.
- **Current:**
  - D **FAIL** on orders only. The 80% coverage rule drops the b2 decision step (`engine.py:2071-2076`). o2's fill finds no AAPL source bar and hits a bare `continue` with no order event (`execution.py:526-527`). There is no fill (correct) and no carry (correct), and the stale mark is correct via the forward-fill cache (`engine.py:2016-2040`; `domain/trading/portfolio.py:27-38`).
  - S **FAIL** on orders: the same silent skip, and `order_events` is never persisted (`external_run_service.py:906-909`).

**T05: on an early-close day the 12:30 bucket is the session's final bar, and its decision fills at the next session's open.**
- Checks: fills, orders, cash, positions, equity.
- Bars, AAPL `H½`: 2026-11-27 b0..b2 flat 100; b3 (12:30, 30 minutes, closes 13:00) flat 100. Next session 2026-11-30 (Monday) b0 `102/104/101/103`. The data ends after 11-30.b0.
- Intent: `@11-27.b3 BUY 5 AAPL MKT DAY [o1]`.
- Expected:
  - b3 closes at 13:00 and is the session's final bar (P2), so o1 fills at 11-30.b0 at **102.00**, cost 510.00. Cash 490.00, AAPL 5 at avg 102.00. o1 `filled`.
  - `E(11-27.b3) = 1,000.00` (equity stamp 13:00, not 13:30); `E(11-30.b0) = 490 + 5×103 = 1,005.00`.
- **Current:**
  - D **FAIL**. The US session window is hard-coded to 09:30–16:00 with no early-close calendar (`infrastructure/market_data/sessions.py:40-44`; `domain/leaderboard/us_market_calendar.py:9` states early closes are not modelled). The 12:30 bucket is therefore stamped 13:30, a time when no source bar opens and which is not a session close, so the decision gets no fill and is not a step (`bar_aggregation.py:394-410`; `engine.py:1949`). o1 never fills and nothing records it: cash 1,000.00, final equity 1,000.00, and the equity stamp is 13:30.
  - S **FAIL**, same planner (`market_data_store.py:510-515`).

### Group C: cash and ordering

**C01: sells execute before buys within a bar.**
- Checks: fills, orders, cash, positions.
- Bars, H: AAPL flat 100 and MSFT flat 50 at b0..b2.
- Intents: `@b0 BUY 10 AAPL [o1]`; `@b1 [BUY 10 MSFT [o2], SELL 5 AAPL [o3]]`. The buy is listed first.
- Expected:
  - o1 fills at b1, `10×100 = 1,000.00`, leaving cash 0.00 (an exact-cash fill is allowed).
  - At b2 the sell goes first: o3 sells 5 @ 100, `+500`, cash 500. Then o2 buys 10 @ 50, `−500`, cash 0.00.
  - Final: AAPL 5 (avg 100), MSFT 10 (avg 50), cash 0.00, E 1,000.00.
- Credit: ml4t `tests/property/test_order_sequence_invariants.py` (exit-first); `tests/accounting/test_cash_account_policy.py` (exact cash).
- **Current:**
  - D **FAIL**. It runs actions in listed order (`execution.py:442`). o2 needs 500 against cash 0 and is rejected `insufficient_cash` (`execution.py:612-626, 663-698`). Then o3 fills. Final MSFT 0, cash 500.
  - DL **FAIL**. The decision-time pre-check uses pre-batch cash (`portfolio_manager.py:904-906`) and drops o2. The SELL ignores size and liquidates all 10 (`portfolio_manager.py:918-931`). Final AAPL 0, cash 1,000.
  - S **FAIL**. The pre-check against cash 0 drops o2 silently (`infrastructure/llm/validator.py:379-380`). Final MSFT 0, cash 500.

**C02: a buy that no longer fits at the fill price is reduced.**
- Checks: fills, orders, cash, positions.
- Bars, AAPL H: b0 flat 100; b1 `125/126/124/125`; b2 flat 125.
- Intent: `@b0 BUY 10 AAPL [o1]`. It fits at the decision price: `10×100 = 1,000`.
- Expected:
  - At the b1 open, 125, the full order needs `1,250 > 1,000`.
  - Reduce to `floor_step(1,000/125) = 8`. Cost 1,000.00, cash 0.00.
  - o1 is `partially_filled/insufficient_cash`, filled 8. `E(b1) = 8×125 = 1,000.00`.
- Credit: ml4t `tests/accounting/test_cash_account_policy.py` (insufficient cash).
- **Current:**
  - D **FAIL**. Buys are all-or-nothing (`execution.py:612-626`). o1 is rejected, its event is recorded in run metadata (`engine.py:1818-1824`), and the account ends with AAPL 0 and cash 1,000.
  - S **FAIL**. The order passes the decision check (`validator.py:379-380`) and the executor rejects it. The rejection is invisible: `order_events` is not persisted (`external_run_service.py:906-909`) and v2 hard-codes `rejected: []` (`execution/backtest_backend.py:255-264`).

**C03: two buys exceed cash in aggregate; the first in submission order fills and the second is reduced.**
- Checks: fills, orders, cash, positions.
- Bars, H: AAPL flat 100 and MSFT flat 50 at b0..b1.
- Intents: `@b0 [BUY 6 AAPL [o1], BUY 10 MSFT [o2]]`.
- Expected:
  - At b1, o1 costs `6×100 = 600`, leaving cash 400.
  - o2 needs `500 > 400`, so it is reduced to `400/50 = 8`, `−400`, leaving cash 0.00. o2 is `partially_filled/insufficient_cash`.
  - Final: AAPL 6, MSFT 8, E 1,000.00.
- **Current:**
  - D **FAIL**. o2 is rejected outright, leaving MSFT 0 and cash 400 (`execution.py:612-626`).
  - DL **FAIL**. Each buy passes its own un-netted pre-check (`portfolio_manager.py:904-906`), and the executor result is the same.
  - S **FAIL**. Same, because both pre-checks use one cash snapshot (`validator.py:379-380`; `external_run_service.py:748-753`).

**C04: sell quantity guards (negative, zero, oversell, unheld).**
- Checks: fills, orders, cash, positions, plus an invariant that cash stays ≥ 0 at every fill.
- Bars, H: AAPL flat 100 and MSFT flat 50 at b0..b2.
- Intents: `@b0 BUY 10 AAPL [o1]`; `@b1 [SELL −5 AAPL [o2], SELL 0 AAPL [o3], SELL 12 AAPL [o4], SELL 5 MSFT [o5]]`.
- Expected:
  - o1 fills at b1 @ 100, leaving cash 0.
  - At b2: o2 and o3 are `rejected/invalid_quantity`.
  - o4 is clipped to 10 and fills `+1,000`, giving cash 1,000.00. Its status is `partially_filled/clipped_to_position`, filled 10.
  - o5 is `rejected/no_position`.
  - Final: no positions, cash 1,000.00, E 1,000.00. The fills are exactly o1 and o4.
- Credit: ml4t `test_cash_account_policy.py` (reject overselling). **We diverge on purpose (P6: clip).**
- **Current:**
  - D **FAIL**. o2 runs `min(−5, 10) = −5`, which **adds** 5 shares and sets cash to `0 − 500 = −500` (`execution.py:812-826`; bug documented at `:36-40`). o3 appends a 0-share trade. o4 sells `min(12, 15) = 12` for `+1,200`. o5 records `insufficient_position`. Final: **cash 700, AAPL 3**, with cash negative partway through the batch.
  - DL **FAIL** on orders (predicted). The dashboard translator never schema-checks a SELL's size: it does not use `LLMTradingDecision` at all, and a SELL is always sized at the full sellable holding, whatever `position_size` says (`portfolio_manager.py:918-931`). It also truncates the batch to one action per universe symbol (2 here; `portfolio_manager.py:802-812`; under `strict_llm` the whole batch raises instead). So o2 (−5) liquidates all 10 AAPL for `+1,000`, o3 (0) becomes a second 10-share sell that the executor rejects `insufficient_position`, and o4 and o5 are dropped with no record. The final state (no positions, cash 1,000.00) is right by accident.
  - S (session or legacy) **FAIL**. The −5 fails the schema check, so the **whole batch** becomes a `validation_hold` (`external_run_service.py:725-734`). Final AAPL 10, cash 0.
  - S-v2 **FAIL** on orders only. The −5 is dropped per item with a reason (`api/v2/models.py` `validate_actions`), and the `0` is executed as a full liquidation (`validator.py:391`) instead of being rejected. The final state comes out right by accident.

### Group S: sizing

**S01: target-weight orders with fractional quantities.**
- Checks: fills, cash, positions, avg cost, realized P&L, equity.
- Bars, AAPL H: b0 flat 300; b1 `320/330/320/330`; b2 flat 375.
- Intents: `@b0 TARGET AAPL 0.25 [o1]`; `@b1 TARGET AAPL 0.25 [o2]`.
- Expected:
  - **o1** at the b1 open, 320: `E_open = 1,000`, notional `0.25×1,000 = 250`, qty `250/320 = 0.78125`. Cost 250.00, leaving cash 750.00 and avg 320.
  - `E(b1) = 750 + 0.78125×330 = 750 + 257.8125 = 1,007.8125`.
  - **o2** at the b2 open, 375: `E_open = 750 + 0.78125×375 = 1,042.96875`. Target `0.25×1,042.96875/375 = 0.6953125`. Delta `= 0.6953125 − 0.78125 = −0.0859375`, so this is a SELL.
  - Proceeds `0.0859375×375 = 32.2265625`, giving cash 782.2265625.
  - Realized `0.0859375×(375−320) = 4.7265625`. Avg stays 320.00 and the position is 0.6953125.
  - `E(b2) = 782.2265625 + 0.6953125×375 = 1,042.96875`. Check: `260.7421875/1,042.96875 = 0.25`.
- **Current:**
  - D **NOT EXPRESSIBLE**: actions carry shares only (`execution.py:442-445`).
  - DL would truncate `int(position_size)` to 0 (`portfolio_manager.py:859-871`).
  - S session, legacy and v2 **NOT EXPRESSIBLE**: `position_size` is an int (`validator.py:85`; `api/v2/models.py:86`).
  - S-v1 **FAIL**. The weight resolves at the decision close with floor division: `int(250 // 300) = 0`, so the order is rejected as `zero_quantity` (`domain/runs/protocol.py:112-114`; `domain/runs/service.py:963`).

**S02: whole-share mode (`qty_step = 1`) floors target weights and leaves the residual in cash.**
- Checks: fills, cash, positions.
- Bars, H: AAPL b0 flat 300 and b1 `310/320/310/320`; MSFT b0 flat 90 and b1 `90/92/90/92`.
- Intents: `@b0 [TARGET AAPL 0.5 [o1], TARGET MSFT 0.5 [o2]]`.
- Expected, at `E_open = 1,000`:
  - o1: `floor(500/310) = 1` share, cost 310.
  - o2: `floor(500/90) = 5` shares, cost 450.
  - Cash `1,000 − 760 = 240.00`. Rounding residuals: AAPL `500−310 = 190`, MSFT `500−450 = 50`.
  - `E(b1) = 240 + 320 + 5×92 = 1,020.00`.
- **Current:**
  - D **NOT EXPRESSIBLE**.
  - S-v1 **FAIL**. It sizes `int(500 // 300) = 1` and `int(500 // 90) = 5`, then rejects both against the 25% position cap (`300 > 250`, `450 > 250`; `domain/runs/service.py:916-985, 1311-1329`).

### Group F: costs

**F01: commission per share with a per-order minimum.**
- Checks: fills incl. commission, cash, realized P&L.
- Costs: `commission_per_share = 0.005`, `commission_min = 1.00`, no regulatory fees.
- Bars, H: AAPL flat 100 and MSFT flat 2.00 at b0..b2.
- Intents: `@b0 [BUY 4 AAPL [o1], BUY 250 MSFT [o2]]`; `@b1 [SELL all AAPL [o3], SELL all MSFT [o4]]`.
- Expected:
  - o1: `max(1.00, 4×0.005 = 0.02) = 1.00`, so cash impact `−401.00`.
  - o2: `max(1.00, 250×0.005 = 1.25) = 1.25`, so cash impact `−501.25`. Cash after b1: `1,000 − 401 − 501.25 = 97.75`.
  - o3: `+400 − 1.00`; o4: `+500 − 1.25`. Cash after b2: `97.75 + 399 + 498.75 = 995.50`.
  - Realized P&L `= −(1+1.25+1+1.25) = −4.50`, which equals `995.50 − 1,000`.
- Credit: ml4t `validation/scenarios/definitions.py` scenario 06 (per-share commission).
- **Current:**
  - D **FAIL**. The US profile has no costs (`profiles.py:198-218`; `execution.py:295-308`), so cash is 1,000.00.
  - A probe run with `commission_rate = 0.0025` and `minimum_commission = 1.00` (`execution.py:336-341`) would match fills, commissions and cash, but only because both prices are flat (0.0025 equals `0.005 × q / notional` on every leg here); a per-share fee cannot be expressed as a fixed rate. The probe still **FAILS** on realized P&L, which D has no field for (D14).
  - S **FAIL**: the session builds `PortfolioManager` with no cost profile (`external_run_service.py:301-304`).

**F02: SEC §31 fee, sells only, dated by fill trade date, rounded up to the cent.**
- Checks: fills incl. sec_fee, cash.
- Costs: `sec_fee = (2025-05-14: 0.00, 2026-04-04: 20.60)`; TAF off.
- Bars, AAPL D: 2026-03-31, 04-01, 04-02, 04-06, all flat 100. The 04-03 Good Friday and the weekend have no session.
- Intents: `@03-31 BUY 10 [o1]`; `@04-01 SELL 5 [o2]`; `@04-02 SELL 5 [o3]`.
- Expected:
  - o1 fills 04-01 at `−1,000`, leaving cash 0. Buys pay no SEC fee.
  - o2 fills 04-02 (rate 0.00): SEC `500×0/1e6 = 0.00`, leaving cash 500.00.
  - o3 is **decided** on 04-02 but **fills** on 04-06 (rate 20.60). SEC `= 500×20.60/1e6 = 0.0103`, rounded up to **0.02**. Cash `500 + 500 − 0.02 = 999.98`, and that is the final equity.
  - Fee-function table (adapters that expose the fee function): principal 50,000 gives `1.03` exactly, and principal 1,000 gives `0.0206`, rounded up to `0.03`.
- **Current:**
  - D **FAIL**: no fee, cash 1,000.00.
  - A probe using `stamp_duty_sell_rate = 20.6e-6` still fails. It rounds half-up, `0.0103 → 0.01` (`execution.py:246-248, 343-346`). It is undated, so it also charges o2. And under the daily-to-16:00 mapping, o3 trades on 04-02.
  - S **FAIL**.

**F03: FINRA TAF, per share, sells only, cap per trade, Jan-1 dating.**
- Checks: fills incl. taf, cash.
- Setup: cash **100,000**. Costs `taf = (2025-01-01: 0.000166 cap 8.30, 2026-01-01: 0.000195 cap 9.79)`; SEC as in F02, which is 0.00 on these dates.
- Bars, AAPL D: 2025-12-29, 12-30, 12-31, 2026-01-02, all flat 1.00.
- Intents: `@12-29 BUY 90,000 [o1]`; `@12-30 SELL 30,000 [o2]`; `@12-31 [SELL 59,990 [o3], SELL 10 [o4]]`.
- Expected:
  - o1 fills 12-30 at `−90,000`, leaving cash 10,000.
  - o2 fills 12-31 (2025 rate): TAF `30,000×0.000166 = 4.98`, under the cap. Cash `10,000 + 30,000 − 4.98 = 39,995.02`.
  - o3 fills 2026-01-02: `59,990×0.000195 = 11.69805`, so it hits the **cap of 9.79**.
  - o4 fills 2026-01-02: `10×0.000195 = 0.00195`, rounded up to **0.01**.
  - Cash `39,995.02 + 59,990 − 9.79 + 10 − 0.01 = 99,985.22`, and that is the final equity. Total fees `4.98 + 9.80 = 14.78`.
- **Current:**
  - D **FAIL**: cash 100,000.00. No per-share or capped fee component exists, so this cannot be probed (`execution.py:336-352`).
  - S **NOT EXPRESSIBLE**: capital is clamped to 3,000 (`constants.py:96-115`; `external_run_service.py:289`) and order size is capped at 10,000 (`validator.py:69`; `api/v2/models.py:86`).

**F04: slippage in bps, adverse, not tick-rounded.**
- Checks: fills incl. slippage_cost, cash, avg cost, realized P&L.
- Costs: `slippage_bps = 10`.
- Bars, AAPL H: b0 flat 100; b1 `100/110/100/110`; b2 flat 110.
- Intents: `@b0 BUY 5 [o1]`; `@b1 SELL all [o2]`.
- Expected:
  - o1: `100×1.001 = 100.10`, cost 500.50, cash 499.50, slippage_cost `5×0.10 = 0.50`, avg 100.10.
  - `E(b1) = 499.50 + 5×110 = 1,049.50`.
  - o2: `110×0.999 = 109.89`, proceeds 549.45, cash **1,048.95**, slippage_cost `5×0.11 = 0.55`.
  - Realized `5×(109.89−100.10) = 48.95`.
- Credit: ml4t scenario 08 (% slippage); `tests/scenarios/factory.py` long round trip.
- **Current:**
  - D **FAIL**: it fills at 100 and 110, giving cash 1,050.00.
  - A probe with `buy/sell_slippage_rate = 0.001` would match fills and cash, because both slipped prices land on whole cents and survive the probe's 0.01 tick rounding (`execution.py:321-333`). It still **FAILS** on realized P&L, which D has no field for (D14). The avg-cost check compares the final state, where the position is closed (D deletes `entry_prices` on a full sell, `execution.py:826-827`), so it is vacuous there.
  - S **FAIL**.

**F05: a buy reduced to fit its commission, then a full-stack round-trip sell after 2026-04-04.**
- Checks: fills incl. all fee components, orders, cash, realized P&L.
- Costs: `commission_min = 1.00` (a flat $1 per order); SEC and TAF as in the table; slippage 0.
- Bars, AAPL H on **2026-05-04**: b0 flat 200; b1 `200/250/200/250`; b2 flat 250.
- Intents: `@b0 BUY 5 [o1]`; `@b1 SELL all [o2]`.
- Expected:
  - o1: the full order needs `5×200 + 1 = 1,001 > 1,000`, so reduce to `q = (1,000−1)/200 = 4.995` (exact; the P5 binary search lands on the same grid point). Gross 999.00 plus 1.00 commission leaves cash 0.00. Status `partially_filled/insufficient_cash`, filled 4.995.
  - `E(b1) = 0 + 4.995×250 = 1,248.75`.
  - o2: gross `4.995×250 = 1,248.75`.
  - o2 fees: commission 1.00; SEC `1,248.75×20.6e-6 = 0.02572425`, rounded up to **0.03**; TAF `4.995×0.000195 = 0.000974025`, rounded up to **0.01**. Total 1.04.
  - Cash `1,248.75 − 1.04 = 1,247.71`, and that is the final equity.
  - Realized `= −1.00 + [4.995×(250−200) − 1.04] = −1.00 + 248.71 = 247.71`.
  - *Rounding alternative (not scored, Q3):* under Alpaca's summed-EOD rounding the regulatory fees are `ceil(0.026698275) = 0.03`, so o2's fees total 1.03, cash 1,247.72 and realized 247.72. An engine configured for that method should be scored against this variant.
- **Current:**
  - D **FAIL**. It buys 5 at 200 (no fees, all-or-nothing passes), then sells 5 at 250, giving cash 1,250.00.
  - S **FAIL**, same result.

### Group L: liquidity

**L01: the participation cap gives a partial fill and the remainder is cancelled.**
- Checks: fills, orders, cash, equity.
- Costs: `participation_cap = 0.10`.
- Bars, AAPL H: b0 flat 100 with V 1,000; b1 flat 100 with **V 50**; b2 `100/102/100/102` with V 1,000.
- Intent: `@b0 BUY 8 [o1]`.
- Expected:
  - The cap is `0.10×50 = 5`, so 5 fill at 100, leaving cash 500.00. o1 is `partially_filled/volume_cap`, filled 5, and the remaining 3 are cancelled.
  - `E(b1) = 1,000.00`; `E(b2) = 500 + 5×102 = 1,010.00`.
- Credit: ml4t native oracle `zipline volume_share partial fills` (`validation/native/evidence/zipline-3.1.1.json`).
- **Current:**
  - D **FAIL**. The executor never reads volume, so 8 fill, cash is 200, and `E(b2) = 1,016.00`.
  - S **FAIL**.

### Group O: order types

**O01: limit buys, one touched intrabar and one gapping through at the open.**
- Checks: fills, orders, cash, equity.
- Bars, H:

  | bar | AAPL | MSFT |
  |---|---|---|
  | b0 | flat 100 | flat 50 |
  | b1 | `99/100/97/98` | `50/51/47/49` |
  | b2 | `95/97/94/96` | `49/50/48/50` |

- Intents: `@b0 [LMT BUY 5 AAPL @96 DAY [o1], LMT BUY 5 MSFT @48 DAY [o2]]`.
- Expected:
  - o2 at b1: `O = 50 > 48` but `L = 47 ≤ 48`, so it fills at **48.00**, `−240`, leaving cash 760.
  - o1 at b1: `O = 99 > 96` and `L = 97 > 96`, so no fill.
  - o1 at b2: `O = 95 ≤ 96`, so it fills at the **open, 95.00**, `−475`, leaving cash 285.00.
  - `E(b1) = 760 + 5×49 = 1,005.00`; `E(b2) = 285 + 5×96 + 5×50 = 1,015.00`.
- Credit: ml4t `tests/test_order_type_matrix.py` (`test_limit_fills_at_limit_price`, `test_limit_no_fill_if_not_touched`).
- **Current:** D and S **NOT EXPRESSIBLE**. Actions are market-only (`execution.py:442-445`), and v1's `order_type` field accepts `market` only (`VALID_ORDER_TYPES = {"market"}`, `domain/runs/protocol.py:24`; anything else is rejected `unsupported_order_type` at `:99-100`). Downgrading these orders to market would make them fill at 100 and 50, which is wrong.

**O02: stop-loss sells, one triggered intrabar and one gapping through.**
- Checks: fills, cash, equity.
- Bars, H:

  | bar | AAPL | MSFT |
  |---|---|---|
  | b0 | flat 100 | flat 50 |
  | b1 | `100/101/99/100` | flat 50 |
  | b2 | `98/99/93/94` | `44/46/43/45` |

- Intents: `@b0 [BUY 5 AAPL [o1], BUY 5 MSFT [o2]]`; `@b1 [STP SELL 5 AAPL @95 GTC [o3], STP SELL 5 MSFT @47 GTC [o4]]`.
- Expected:
  - At b1: `−500 − 250`, leaving cash 250.00. `E(b1) = 250 + 500 + 250 = 1,000.00`.
  - At b2, o3: `O = 98 > 95` and `L = 93 ≤ 95`, so it fills at **95.00** for `+475`.
  - At b2, o4: `O = 44 ≤ 47`, so it fills at the **open, 44.00** for `+220`.
  - Cash `250 + 475 + 220 = 945.00 = E(b2)`. Neither stop could trigger at b1, because both were decided at b1's close.
- Credit: ml4t scenario 03 (stop loss), `tests/test_extreme_conditions.py` (gap down through stop fills at open).
- **Current:** **NOT EXPRESSIBLE** on both paths. `stop_loss_price` is accepted and ignored (`validator.py:86-87`; v1 forces it to `None` at `domain/runs/protocol.py:133-134`), even though the LLM prompt says stops are "enforced separately in code" (`infrastructure/llm/prompts.py:42-46`).

**O03: bracket (OCO), take-profit hit, and same-bar ambiguity resolved as stop-first.**
- Checks: fills, orders, cash.
- Bars, H:

  | bar | AAPL | MSFT |
  |---|---|---|
  | b0 | flat 100 | flat 50 |
  | b1 | flat 100 | flat 50 |
  | b2 | `101/111/99/105` | `50/56/44/52` |

- Intents: `@b0 [BUY 5 AAPL [o1], BUY 5 MSFT [o2]]`. At `@b1`:
  - AAPL OCO group `g1`: `[STP SELL 5 @95 [o3], LMT SELL 5 @110 [o4]]`.
  - MSFT OCO group `g2`: `[STP SELL 5 @45 [o5], LMT SELL 5 @55 [o6]]`.
- Expected:
  - Cash after b1: `1,000 − 500 − 250 = 250`.
  - AAPL at b2: the open 101 sits inside (95, 110). `H = 111 ≥ 110` and `L = 99 > 95`, so o4 fills at **110.00** (`+550`) and o3 is `cancelled/oco_cancelled`.
  - MSFT at b2: both legs are touched (`H = 56 ≥ 55`, `L = 44 ≤ 45`) with no gap, so the **stop wins**. o5 fills at **45.00** (`+225`) and o6 is `cancelled/oco_cancelled`.
  - Cash `250 + 550 + 225 = 1,025.00`. No positions.
- Credit: ml4t scenario 04 (TP) and scenario 10 (bracket SL5/TP10).
- **Current:** **NOT EXPRESSIBLE**, as in O02.

**O04: time in force. A DAY order expires at the session end, a GTC order carries over, and a fractional GTC order is rejected.**
- Checks: fills, orders, cash.
- Bars, AAPL: d1 (2026-03-02) b5 flat 100; d1 b6 `100/100/99/99.5`; d2 (03-03) b0 `99/99/96/97`.
- Intents: `@d1.b5 [LMT BUY 2 @97 DAY [o1], LMT BUY 3 @97 GTC [o2], LMT BUY 0.5 @97 GTC [o3]]`.
- Expected:
  - o3 is `rejected/fractional_requires_day_tif` at submission.
  - At d1.b6, `L = 99 > 97`, so nothing fills. o1 is `expired/session_end` after d1 closes.
  - At d2.b0, `O = 99 > 97` and `L = 96 ≤ 97`, so o2 fills 3 @ **97.00** (`−291`), leaving cash 709.00.
  - `E(d2.b0) = 709 + 3×97 = 1,000.00`.
- **Current:** **NOT EXPRESSIBLE**, because there are no limit orders and no TIF.

### Group CA: corporate actions (daily bars)

**CA01: a 2-for-1 split doubles the quantity, halves the average cost, and books no P&L.**
- Checks: positions, avg cost, cash, realized P&L, equity.
- Bars, AAPL D: 06-01 flat 200; 06-02 flat 200; **06-03 (ex-date)** `101/102/100/102`; 06-04 `102/104/102/104`.
- Split: AAPL, ex-date 2026-06-03, ratio 2.
- Intents: `@06-01 BUY 4 [o1]`; `@06-03 SELL 3 [o2]`. o2's quantity is in post-split shares.
- Expected:
  - o1 fills 06-02 @ 200 for `−800`, leaving cash 200.00. `E(06-02) = 200 + 4×200 = 1,000.00`.
  - At the 06-03 open the position becomes 8 at avg 100.00. Equity is unchanged at the split, because the pre-split value `4×200 = 800` equals the post-split value `8×100`.
  - `E(06-03) = 200 + 8×102 = 1,016.00`.
  - o2 fills 06-04 @ 102 for `+306`, leaving cash 506.00 and AAPL 5 at avg 100.00. Realized `3×(102−100) = 6.00`.
  - `E(06-04) = 506 + 5×104 = 1,026.00`.
- **Current:**
  - D **FAIL**. Bars are requested raw: the `StockBarsRequest` at `infrastructure/market_data/alpaca_bars.py:732-738` passes no `adjustment` argument, so alpaca-py's raw default applies. There is no split ledger. The position stays 4, so `E(06-03) = 200 + 4×102 = 608.00` (a phantom loss of −39.2%). o2 leaves 1 share, so `E(06-04) = 506 + 104 = 610.00`.
  - The daily mapping does not confound this result, because `O(t+1) = C(t)` at both fills.
  - S **FAIL**.

**CA02: a cash dividend. The receivable is booked at the ex-date, cash arrives at the pay-date, and equity is continuous.**
- Checks: cash, equity.
- Bars, AAPL D: 05-04 flat 50; 05-05 flat 50; **05-06 (ex-date)** `49/49.5/48.5/49`; 05-07 flat 49; **05-08 (pay-date)** flat 49.
- Dividend: AAPL $1.00, ex 2026-05-06, pay 2026-05-08.
- Intent: `@05-04 BUY 10 [o1]`.
- Expected:
  - o1 fills 05-05 @ 50, leaving cash 500. The position held at the 05-05 close is 10, so the receivable is `10×1.00 = 10.00`.
  - `E(05-05) = 1,000`.
  - `E(05-06) = 500 + 490 + 10 = 1,000.00` and `E(05-07) = 1,000.00`.
  - On 05-08 the cash becomes 510.00 and `E = 510 + 490 = 1,000.00`. TR = 0.
- **Current:** D and S **FAIL**. No dividend is credited (`alpaca_bars.py:732`; no ledger), so `E(05-06) = 990.00` and the final equity is 990.00.

**CA03: dividend entitlement edges. Selling at the ex-date open keeps the dividend; buying at the ex-date open does not earn it.**
- Checks: cash, positions, equity.
- Bars, D: AAPL as in CA02, with the same $1.00 dividend. MSFT 05-04 flat 100, 05-05 flat 100, **05-06 (ex-date)** flat 98, 05-07 flat 98, 05-08 flat 98. MSFT dividend $2.00, ex 05-06, pay 05-08.
- Intents: `@05-04 BUY 5 MSFT [o1]`; `@05-05 [SELL 5 MSFT [o2], BUY 10 AAPL [o3]]`.
- Expected:
  - o1 fills 05-05 @ 100, leaving cash 500.
  - At the 05-06 open, o2 sells @ 98 (`+490`, cash 990) and o3 buys @ 49 (`−490`, cash 500).
  - Entitlement: MSFT held at the 05-05 close is 5, giving a receivable of 10.00. The AAPL position bought on the ex-date is not entitled, so its dividend is 0.
  - `E(05-06) = 500 + 10×49 + 10 = 1,000.00`. On 05-08 cash is 510.00 and `E = 1,000.00`.
- **Current:**
  - D **FAIL**. Under the 16:00 mapping, o2 and o3 fill at the 05-05 close (`bar_aggregation.py:402-410`), and there are no dividends. Cash is 500, AAPL 10, `E(05-06) = 990.00`, final 990.00.
  - S **FAIL**.

**CA04: a split cancels working orders, and whole-share mode pays the fractional remainder as cash in lieu.**
- Checks: fills, orders, cash, positions, avg cost, realized P&L, equity.
- Costs: `qty_step = 1` (whole-share mode).
- Bars, AAPL D: 06-01 flat 150; 06-02 flat 150; **06-03 (ex-date)** `101/102/99/100`; 06-04 flat 100.
- Split: AAPL, ex-date 2026-06-03, ratio 1.5 (3-for-2).
- Intents: `@06-01 BUY 5 [o1]`; `@06-02 LMT BUY 2 @140 GTC [o2]`.
- Expected:
  - o1 fills 06-02 @ 150 for `−750`, leaving cash 250.00. `E(06-02) = 250 + 5×150 = 1,000.00`. o2 is working from the 06-03 open.
  - At the 06-03 open, before any fill: the position becomes `5×1.5 = 7.5` at avg `150/1.5 = 100.00`. Whole-share mode keeps 7 and sells the remainder 0.5 as cash in lieu at `O(ex) = 101`: cash `250 + 50.50 = 300.50`, realized `0.5×(101−100) = 0.50`.
  - o2 is `cancelled/corporate_action`, filled 0. Left working at its pre-split price it would fill at the open, 101 ≤ 140, buying 2 shares nobody priced.
  - `E(06-03) = 300.50 + 7×100 = 1,000.50`; `E(06-04) = 1,000.50`. Identity: `1,000.50 − 1,000 = 0.50` realized `+ 7×(100−100)` unrealized.
- **Current:** D and S **NOT EXPRESSIBLE**: no limit orders or TIF (D12). The split half alone would fail as in CA01 (D13).

### Group M: marking, cost basis, metrics

**M01: the equity curve is pre-trade per bar, `initial_equity` equals the starting cash, and the average cost is weighted.**
- Checks: equity (all bars), initial_equity, max_drawdown, total_return, avg cost.
- Bars, AAPL H: b0 flat 100; b1 `102/111/101/110`; b2 `120/121/119/120`; b3 `120/122/120/121`. There are opening gaps at b1 (100 → 102) and b2 (110 → 120).
- Intents: `@b0 BUY 5 [o1]`; `@b1 BUY 4 [o2]`.
- Expected:
  - o1 fills 5 @ 102 for `−510`, leaving cash 490. o2 fills 4 @ 120 for `−480`, leaving cash 10.00.
  - Avg cost `(510 + 480)/9 = 110.00`.
  - `E_init = 1,000.00`; `E(b0) = 1,000.00`; `E(b1) = 490 + 5×110 = 1,040.00`; `E(b2) = 10 + 9×120 = 1,090.00`; `E(b3) = 10 + 9×121 = 1,099.00`.
  - TR = 0.099. MDD = 0, because the curve is monotone.
- **Current:**
  - D **FAIL**. Execution runs **before** the source bars since the previous fill are marked (`engine.py:2267-2302`), so each decision's trade is written back over the preceding marks:
    - `E(b0) = 490 + 5×100 = 990.00`;
    - `E(b1) = 10 + 9×110 = 1,000.00`;
    - `initial_equity = equity_curve[0] = 990.00` (`engine.py:2360`);
    - MDD `= (1,000 − 1,040)/1,040 = −0.0384615`, measured from the 10:30 mark of 1,040;
    - `entry_prices = 120`, the last buy (`execution.py:631`).
  - TR (0.099), `E(b2)` and `E(b3)` pass.
  - S **FAIL**, identically (`external_run_service.py:797, 817, 852`). An existing test already pins 13 post-trade marks (`tests/test_external_minute_source.py:117`).

**M02: average cost and realized P&L across partial sells and re-buys.**
- Checks: avg cost after each fill, realized P&L, final equity.
- Bars, AAPL H: b0 flat 100; b1 `100/120/100/120`; b2 `120/130/120/130`; b3 `130/130/90/90`; b4 flat 90. Each fill bar opens at the previous close (§3 convention).
- Intents: `@b0 BUY 4`; `@b1 BUY 4`; `@b2 SELL 2`; `@b3 BUY 2`.
- Expected:
  - b1: 4 @ 100, leaving cash 600 and avg 100.
  - b2: 4 @ 120, leaving cash 120 and avg `(400+480)/8 = 110.00`.
  - b3: sell 2 @ 130 for `+260`, leaving cash 380. Realized `2×(130−110) = 40.00`. Avg stays 110 and the position is 6.
  - b4: 2 @ 90 for `−180`, leaving cash 200. Avg `(6×110 + 2×90)/8 = 840/8 = 105.00`.
  - `E(b4) = 200 + 8×90 = 920.00`. Unrealized `8×(90−105) = −120`. Realized plus unrealized `= −80 = 920 − 1,000`.
- **Current:**
  - D **FAIL** on avg cost and realized P&L. `entry_prices` is overwritten by each buy, ending at 90 instead of 105 (`execution.py:631`), and no realized P&L field exists (`domain/trading/portfolio.py:63-87`). Final equity 920 passes.
  - S **FAIL**, same.

**M03: daily metrics (TR, MDD, and Sharpe with rf and ddof=1), plus an undefined-Sharpe sub-case.**
- Checks: total_return, max_drawdown, sharpe.
- Setup: `rf_annual = 0.0252`, so `rf_d = 0.0252/252 = 0.0001`.
- Bars, AAPL D: 03-02 flat 100; 03-03 `100/102/100/102`; 03-04 `102/102/96.9/96.9`; 03-05 `96.9/101.745/96.9/101.745`; 03-06 `101.745/102.76245/101.745/102.76245`.
- Intent: `@03-02 BUY 10 [o1]`. It fills 03-03 @ 100, leaving cash 0.
- Expected:
  - Daily equity: `E_0 = 1,000`, `E_1 = 1,000`, `E_2 = 1,020`, `E_3 = 969`, `E_4 = 1,017.45`, `E_5 = 1,027.6245`.
  - `r = [0, 0.02, −0.05, 0.05, 0.01]`.
  - **TR** `= 27.6245/1,000 = 0.0276245`.
  - **MDD:** the peak is 1,020 and the trough 969, giving `−51/1,020 = −0.05`.
  - **Sharpe:**
    - `mean(x) = 0.03/5 − 0.0001 = 0.0059`;
    - deviations of `r` from 0.006: `[−0.006, 0.014, −0.056, 0.044, 0.004]`;
    - squared and summed: `0.000036 + 0.000196 + 0.003136 + 0.001936 + 0.000016 = 0.00532`;
    - `var = 0.00532/4 = 0.00133` and `stdev = 0.0364692`;
    - `Sharpe = 0.0059/0.0364692 × √252 = 0.161781 × 15.874508 = `**`2.568186`**.
  - **Sub-case M03b:** cash only, 3 sessions, no intents. TR 0, MDD 0, Sharpe **null**.
- Tolerance: Sharpe 1e-6.
- **Current:**
  - D **FAIL** on Sharpe. It uses per-5m-mark returns, ddof 0, rf 0 and `√(252·6.5·12)` (`metrics.py:17-64`; `engine.py:2031-2036, 2452-2453`). On the adapter's step-function curve (390 marks) that gives **≈ 2.876100**.
  - D passes TR and MDD.
  - M03b: D returns `0`, not null (`metrics.py:57-58`).
  - S **FAIL**, same (`external_run_service.py:870-875`).

---

## 5. Invariants checked on every case

These are ported from ml4t `tests/property/*` and `tests/contracts/test_ledger_invariants.py` (MIT, © Stefan Jansen). The case's own inputs drive them; they do not use Hypothesis.

1. Cash ≥ 0 after every fill.
2. Positions ≥ 0 at all times.
3. Every fill's reference price, before slippage, lies within `[Low, High]` of its fill bar.
4. Cash conservation: `cash_final = cash_0 − Σbuy(q·p) + Σsell(q·p) − Σfees + Σdividends_paid + Σcash_in_lieu`.
5. Equity identity: `E(t) = cash + Σ q·close + receivables`, and `E_final − E_init = realized + unrealized + dividends` (realized includes cash-in-lieu P&L, P16).
6. A run with no fills and no corporate-action cash keeps cash and equity constant.
7. Every intent reaches exactly one terminal `OrderFinal` (P20).

### 5.1 Known coverage gaps (deliberately deferred)

The review found policies the cases do not yet exercise. T05 (early close) and CA04 (split vs working orders, cash in lieu) were added; the rest are backlog, each a one-case addition once the policy it tests is settled:
- a sell limit that gaps up through its limit at the open (mirror of O01's o1);
- slippage applied to a stop fill (P12);
- the P8 `E_open` fallback to the last close for a symbol with no bar;
- commission rounding on a product that is not an exact cent (P10);
- a DAY order working across several bars of one session before it expires;
- a reverse split, and a split and a dividend on the same ex-date;
- a DST-transition week. The grid is written in New York local time, so DST is invisible to the case data; the case would test the adapter's and engine's UTC conversion, not a policy;
- a fractional order below the $1 minimum notional, and a fractional OCO (P7).

## 6. Optional parametric family (ml4t factory port)

This ports `tests/scenarios/factory.py` and `tests/oracle/engine.py` (MIT), long-only.
- The grid is 5 price pairs `[(100,110), (100,90), (100,100), (50,75), (200,180)]` × qty `[10, 100, 500]` × `commission_pct` `[0, 0.001, 0.005]`, which gives 45 round trips, plus 4 slippage cases.
- Every case uses cash 100,000 and the P2 timing: entry decided at b0, filled at the b1 open; exit decided at b1, filled at the b2 open.
- Expected values come from the closed form in `factory.py`, computed before the run.
- For Alpaca realism, re-run the grid with the dated SEC/TAF table at a 2026-05 trade date.

---

## 7. Prediction summary and error inventory

| Case | D (dashboard, scripted seam) | S (step session) | Root defect |
|---|---|---|---|
| T01 | PASS | PASS | (fallback: D4) |
| T02, T02d | FAIL | FAIL | D2, D16 |
| T03 | FAIL (orders) | FAIL (orders) | D8 |
| T04 | FAIL (orders) | FAIL (orders) | D8 |
| T05 | FAIL | FAIL | D19, D8 |
| C01 | FAIL | FAIL | D5, D18 |
| C02 | FAIL | FAIL | D6, D8 |
| C03 | FAIL | FAIL | D6 |
| C04 | FAIL | FAIL (batch voided) | D7, D8 |
| S01 | N/E | N/E (v1 FAIL) | D9, D17 |
| S02 | N/E | N/E (v1 FAIL) | D9, D17 |
| F01, F02, F04, F05 | FAIL | FAIL | D10 (+D6 in F05) |
| F03 | FAIL | N/E | D10, D17 |
| L01 | FAIL | FAIL | D11 |
| O01–O04 | N/E | N/E | D12 |
| CA01–CA03 | FAIL | FAIL | D13 |
| CA04 | N/E | N/E | D12, D13 |
| M01 | FAIL | FAIL | D1, D3, D14 |
| M02 | FAIL | FAIL | D14 |
| M03 | FAIL (Sharpe) | FAIL | D15 |

Tally over 28 cases:
- **D:** PASS 1 (T01), FAIL 20, N/E 7 (S01, S02, O01–O04, CA04).
- **S:** PASS 1, FAIL 19, N/E 8 (adds F03).

F01–F05 and CA01–CA03 count as FAIL rather than N/E under the §1 rule: costs and corporate actions are environment facts, not intents, so the adapter runs the case with them ignored and the output is wrong. CA04 is N/E because its GTC limit **intent** cannot be submitted.

| ID | Defect | Where (origin/main) | Paths |
|---|---|---|---|
| D1 | Back-marking: execution runs before the bars since the previous fill are marked, so the next decision's trade is written over earlier marks | `engine.py:2267-2302`; `external_run_service.py:797, 817` | D, S |
| D2 | The 16:00 decision fills at the same close (zero latency, asymmetric with the other six) | `bar_aggregation.py:402-410` | D, S |
| D3 | `initial_equity = equity_curve[0]`, which is a post-trade mark | `engine.py:2360`; `external_run_service.py:852` | D, S |
| D4 | A non-intraday source fills at the decision bar's close (look-ahead) | `engine.py:1928-1936`; `external_run_service.py:446-453` | D, S |
| D5 | No sells-before-buys ordering; decision-time pre-checks use one stale cash snapshot | `execution.py:442`; `portfolio_manager.py:904-906`; `validator.py:379-380` | D, DL, S |
| D6 | Buys are all-or-nothing, with no reduction | `execution.py:612-626` | D, S |
| D7 | A negative sell adds shares and drives cash negative; a 0-share sell records a no-op trade; one bad item voids the whole batch | `execution.py:36-40, 812-826`; `external_run_service.py:725-734` | D, S |
| D8 | Missing fill bars are dropped silently; S never persists or returns `order_events`; v2 returns `rejected: []`; unfillable decisions are not steps | `execution.py:526-527`; `external_run_service.py:906-909`; `backtest_backend.py:255-264`; `engine.py:1949` | D, S |
| D9 | No target-weight or fractional intents; sizes are truncated to int | `execution.py:442-445`; `portfolio_manager.py:859-871`; `validator.py:85`; `api/v2/models.py:86` | D, S |
| D10 | US costs are zero. The cost machinery is CN-shaped: % commission with a minimum, half-up rounding, undated, with no per-share or capped fee | `profiles.py:198-218`; `execution.py:246-248, 295-358`; `external_run_service.py:301-304` | D, S |
| D11 | No volume or participation limit | `execution.py` (never reads volume) | D, S |
| D12 | No limit, stop, OCO or TIF orders; `stop_loss_price`/`take_profit_price` are accepted and ignored, while the prompt claims they are enforced | `validator.py:86-87`; `domain/runs/protocol.py:133-134`; `prompts.py:42-46` | D, S |
| D13 | No splits or dividends; bars are raw | `alpaca_bars.py:732` | D, S |
| D14 | `entry_prices` is overwritten by the last buy; no realized P&L | `execution.py:631`; `portfolio.py:63-87` | D, S |
| D15 | Sharpe uses per-5m returns × √19,656, ddof 0, rf 0, and returns 0 when undefined; baselines use √1,638 | `metrics.py:17-64`; `engine.py:2031-2036, 2452-2453, 2571, 2654` | D, S |
| D16 | No daily decision timeframe | `profiles.py:198-218`; `external_run_service.py:297-299` | D, S |
| D17 | S caps capital at ≤ 3,000 and order size at ≤ 10,000; v1 resolves weights at the decision close with floor division and applies a 25% cap | `constants.py:96-115`; `validator.py:69`; `domain/runs/protocol.py:112-114`; `domain/runs/service.py:916-985` | S |
| D18 | The LLM translator's SELL always liquidates the full position, whatever its size; batches are truncated to one action per universe symbol | `portfolio_manager.py:802-812, 918-931` | DL |
| D19 | The US session is hard-coded to 09:30–16:00 with no early-close calendar, so a half day's final bucket is mis-stamped and its decision never fills | `infrastructure/market_data/sessions.py:40-44`; `domain/leaderboard/us_market_calendar.py:9` | D, S |

---

## 8. Adapter plan for the current engine

### 8.1 Bar synthesis (shared by both paths)

- **Hourly bars.** A case bar `(O,H,L,C,V)` lasting `n × 5m` becomes `n` open-stamped 5m bars. `sub0 = (O,H,L,C,V/n)` and every later `sub_k = (C,C,C,C,V/n)`.
- **Why this aggregates back exactly.** `bar_aggregation.py:236-264` reproduces `(O,H,L,C,V)` from these sub-bars. The 5m bar opening at the decision stamp is `sub0` of `t+1`, so its open is `O(t+1)`. Every intrabar mark equals the bar-close mark, so the 5m curve is a step function of the case curve, and MDD and sampling are faithful to the case grid.
- **What is lost.** The order of high and low within the bar. That is irrelevant here, because the engine has no stops.
- **Daily bars.** Each becomes the 78 sub-bars from 09:30 to 15:55, with `sub0 = (O,H,L,C)` and the rest flat at C. The daily decision is scripted at the 16:00 decision stamp.
- **Padding.** Pad only *before* the case's first bar within its first session: flat at `O(first bar)`, with no intents. Never pad after the last bar (T03 depends on this).
- **Missing bar.** Omit all of that bucket's sub-bars for the symbol.
- **Frame attributes.** Set `FRAME_ATTR_OPEN_STAMPED_MINUTES = 5`, `FRAME_ATTR_FEED = "sip"`, `FRAME_ATTR_SIP_FALLBACK = False`, `FRAME_ATTR_END_CLAMPED = False`. Use a `DatetimeIndex` tz-aware in America/New_York. The template is `tests/backtesting/test_engine_minute_source.py:20-81`.

### 8.2 D: dashboard engine, scripted seam (the scored run)

```python
monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)   # loader: source_timeframe,
                                                                           # configure_source_timeframe(), fetch_bars()
monkeypatch.setattr(engine_mod, "db", RecordingDB())                       # insert_run/_equity_points/_trades/_decisions
holder = {}
def scripted(self, state):                                                 # bypasses decision-time sizing & pre-check
    holder["pm"] = self
    return {"actions": translate(intents_at[state["timestamp"]], self.positions)}
monkeypatch.setattr(PortfolioManager, "make_trading_decision", scripted)  # called at engine.py:2189
bt = HourlyBacktester(start, end, use_llm=False, symbols=case_syms, initial_capital=case.initial_cash)
bt.load_data(); bt.calculate_indicators(); run_id, curve = bt.run_agent_backtest()
```

**Mapping.** `intents_at` is keyed on the decision-bar **close** stamp: case bar `bk` (open stamp `ts`) maps to `ts + minutes`.

**`translate()` supports:**
- MKT BUY/SELL with `qty`, which becomes `{"symbol", "action", "shares": qty, "reason": id}`. The executor accepts float shares.
- `qty="all"`, which becomes `shares = positions[sym]`. This is the same value at decision time and at fill time, since nothing fills in between.

Any other **intent** raises `NotExpressible`: target weight, limit, stop, OCO or GTC. Costs, splits and dividends do **not** raise (§1): the adapter runs the case without them, because the scored engine has no input for them, and the case is scored FAIL on its output. Probe mode (below) is the only place a cost profile is passed in.

**Reading the outputs:**
- Fills come from `RecordingDB.trades` (price, shares, and timestamp, which equals `filled_at`).
- Orders come from the **in-memory ledger** `holder["pm"].order_events`, not from run metadata. `insert_run(...).metadata["order_events"]` (`engine.py:1818-1824`) is unusable as a source: it keeps only non-filled events (`_unfilled_order_events`, `engine.py:193-217`), is capped at `REJECTED_ORDER_SAMPLE_LIMIT = 200` (`engine.py:173`), and stores no order ids. The in-memory ledger is complete, but pure rejections are still collapsed per `(symbol, side, reason, trading date)` into one event with a `repeat_count` (`execution.py:197-240`, `portfolio_manager.py:150-155`); the adapter expands `repeat_count` back into separate entries and matches entries to intents by `(decision bar, symbol, side, submission sequence)`.
- Status mapping: executor `filled` → `filled`; `partial` → `partially_filled`; `rejected` → `rejected`. Reasons map `insufficient_cash` → `insufficient_cash` and `insufficient_position` → `no_position`; an unmapped reason is passed through verbatim (and so fails the check). An intent with no ledger entry (e.g. the bare `continue` at `execution.py:526-527`, or a decision that is not a step) yields no `OrderFinal`, which FAILs `orders` and invariant 7.
- Cash, positions and avg cost (`entry_prices`) come from `holder["pm"]`.
- Equity comes from `curve`, sampled at the last 5m mark of each case bar (stamp = close − 5m).
- `initial_equity`, `total_return`, `sharpe_ratio` and `max_drawdown` come from the `insert_run` kwargs.

**Environment.** `tests/conftest.py` already sets `DATABASE_PATH` to a temp file and `ATL_BAR_CACHE=0`. Never use ad-hoc `python -c` (memory note `adhoc-imports-mutate-the-seed-db`).

**Probe mode (informational, not scored).** Rebuild the manager with a `TransactionCostProfile(version, "USD", commission_rate, minimum_commission, stamp_duty_sell_rate, 0, buy_slippage_rate, sell_slippage_rate, 0.01)`, either by patching `profile.transaction_cost_profile` or by using `PortfolioManager(..., transaction_cost_profile=...)`. This measures F01, F02 and F04 against the existing mechanism; §4 gives the predictions.

**DL variant (optional).** Pass a stub `llm_client` whose reply is the intents as `LLMTradingDecision` JSON, running through `make_trading_decision_with_llm` (`portfolio_manager.py:376`). Run it only for C01, C04 and S01, to measure translator drift (D18, D5, D9).

### 8.3 S: step session, `ExternalBacktestSession.submit_decisions`

```python
mds._reset_for_tests()
monkeypatch.setattr(ebs, "DJIA_30", case_syms)                  # universe + 80% coverage base (market_data_store.py:540-565)
monkeypatch.setattr(ebs, "DECISION_TIMEOUT_SECONDS", 10**9)     # read once at import (external_run_service.py:80)
monkeypatch.setattr(ebs, "AlpacaDataLoader", FakeLoader)        # resolved at call time (external_run_service.py:346-351)
monkeypatch.setattr(ebs, "db", BacktestDatabase(db_path=tmp_path / "conf.db"))
monkeypatch.setattr(baseline_worker, "submit", lambda job: None)  # finalize enqueues baselines (:934)
s = ExternalBacktestSession(backtest_id="conf", session_id="conf", agent_name="conformance",
                            model_name="scripted", start_date=..., end_date=...,
                            symbols=case_syms, initial_capital=case.initial_cash)
s.load_market_data()
while s.status == "waiting_decision":
    ts = s.timestamps[s.step_index]
    s.submit_decisions({"actions": [
        {"action": side, "symbol": sym, "confidence": 1.0,
         "reasoning": "conformance", "position_size": int(q)} for ...]})   # [] = hold
```

**Outputs:**
- Fills, orders, cash and positions come from `s.manager.{trades, order_events, cash, positions, entry_prices, equity_history}`. Read `order_events` in memory, because S never persists them (D8). The same `repeat_count` expansion and status mapping as §8.2 apply.
- Metrics come from `db.get_run(s.run_id)`.

**Capabilities (otherwise the case is NOT_EXPRESSIBLE):**
- integer `q` with `0 ≤ q ≤ 10,000`;
- `initial_cash ≤ 3,000`;
- MKT orders only;
- `qty="all"` maps to `position_size = 0`, which the session reads as liquidate (`validator.py:391`).

**Protocol variants (optional).**
- **v1** goes through `domain/runs/service.submit_decision(run_id, step_id, DecisionIn(...))`, with `quantity_type="weight"` for S01 and S02.
- **v2** goes through `BacktestBackend.apply_decisions(actions)`.
- These add each surface's own caps and validation, which is D17.

### 8.4 Executor-only unit seam

Use `domain/trading/execution.py:371` `execute_actions(...)` with `execution_prices={sym: O(t+1)}` and the probe cost profiles. It is good for F01, F02, F04, C01–C04 and M02 as pure accounting checks. It cannot exercise timing, marking or metrics.

### 8.5 Cases that cannot be expressed on the current engine

| Cases | Reason |
|---|---|
| S01, S02 on D, legacy S and v2 S | No target-weight or fractional intent (D9). v1 can express them but sizes them wrongly. |
| O01–O04, CA04 | No order types or TIF (D12). Downgrading to market would test the adapter, not the engine. |
| F03 on S | Capital and per-order share caps (D17). |
| Daily cases (T02d, F02, F03, CA*, M03) | No daily timeframe (D16). They are expressed via the 16:00 hourly decision, which is itself a defect (D2). Cases are designed with `O(t+1) = C(t)` so the mapping does not confound the other checks, except where noted. |

---

## 9. Open policy questions (max 5)

1. **Session-final and daily fills.** Should a decision on the last bar of a session fill at the next session's open (P2), or market-on-close at the official close?
   - **Recommend the next open.** MOC needs the SIP closing-auction print, and IEX is thin there (memory note `alpaca-bar-stamps-and-closing-auction`). It would also bring back zero latency.
   - Effect: every daily strategy and the 7th hourly decision now carry overnight gap risk.
2. **Participation-cap remainder.** Should it be cancelled (P14) or carried within the session as a DAY order?
   - **Recommend cancel.** It keeps one fill per order, the agent re-decides every bar, and there is no per-order commission-floor fragmentation (see the `calculate_transaction_costs` docstring, `execution.py:279-285`).
3. **Production cost defaults and publication.**
   - Defaults: commission 0, the dated SEC/TAF table on, slippage **5 bps** per side, participation cap **10%**.
   - **Recommend** publishing results **net** of costs and showing gross alongside. This closes #461, and it means re-running leaderboard entries (a one-off cost of about 21 billable backtests, per `engine-report.md` §6).
   - **Fee rounding:** per-fill per-component ceilings (P11, the upper bound) or Alpaca's summed-EOD-per-account ceiling (F05 differs by $0.01)? **Recommend** per-fill until one real Alpaca sell confirmation settles which Alpaca page is right. The CAT rate must also be verified before "net" can claim to be complete.
4. **Corporate-action source and representation.**
   - **Recommend** Alpaca's corporate-actions API, with raw prices for fills and marks plus ledger events (P16, P17), rather than adjusted bars.
   - In whole-share mode, pay cash in lieu at the ex-date open.
   - Cancel every working order on a split (P16, CA04) rather than adjusting it as Alpaca does for forward-split buy limits and sell stops.
   - Indicators use a back-adjusted series built from the same events.
5. **Risk-free rate and undefined metrics.**
   - **Recommend** a constant `rf_annual` per run, defaulting to the 3-month T-bill yield at the window start and recorded in `agent_runs.metadata`.
   - An undefined Sharpe should be **null**, not 0. That needs the API and DB to accept null; CLAUDE.md records NaN breaking Postgres and Starlette, and null avoids that.

---

## 10. Sources

- Repository `origin/main` `e8736d07`, for every file:line above.
- ml4t-backtest (MIT, © 2025-2026 Stefan Jansen), local checkout at `scratchpad/oss2/ml4t-backtest`: `tests/scenarios/factory.py`, `tests/oracle/*`, `tests/accounting/*`, `tests/property/*`, `tests/test_order_type_matrix.py`, `tests/test_extreme_conditions.py`, `validation/scenarios/definitions.py`, `validation/native/evidence/*`.
- [SEC Fee Rate Advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2); [SEC Fee Rate Advisory 2025-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2025-2).
- [FINRA Information Notice 2026-03-17](https://www.finra.org/rules-guidance/notices/information-notice-20260317).
- [FINRA fee adjustment schedule, SR-FINRA-2024-019](https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule); [FINRA Schedule A §1](https://www.finra.org/rules-guidance/rulebooks/corporate-organization/section-1-member-regulatory-fees); [FINRA TAF FAQ](https://www.finra.org/rules-guidance/guidance/faqs/trading-activity-fee).
- [Alpaca regulatory fees (support)](https://alpaca.markets/support/regulatory-fees); [Alpaca regulatory fees (docs)](https://docs.alpaca.markets/docs/regulatory-fees); [Alpaca fractional trading](https://docs.alpaca.markets/docs/fractional-trading); [Alpaca mandatory corporate actions FAQ](https://docs.alpaca.markets/docs/mandatory-corporate-actions).
- Prior report: `scratchpad/engine-report.md` §4.3–§6.

---

## Review log (2026-10-07)

Two independent verifiers reviewed v1. Each finding was re-derived against `origin/main` @ `e8736d07`, the ml4t checkout, Alpaca's live docs and exact arithmetic before it was accepted. **Accepted 28, rejected 2** (3 informational confirmations need no action and are not counted).

| # | Source | Finding | Verdict | Reason |
|---|---|---|---|---|
| 1 | V1 | M02 breaks the `O(t+1)=C(t)` convention without saying so | **Accept** | Re-derived; M02's bars were changed to hold the convention (same fills, same numbers). Also found F04 and F05 broke it silently; fixed the same way, and §3 now lists every case where the gap is the point. |
| 2 | V1 | F04 (and F01) probe cannot PASS, because D has no realized P&L | **Accept** | D14 confirmed; the probe sentences now say fills and cash match but realized P&L FAILs. |
| 3 | V1 | C04 DL mechanism misdescribed (size 0 passes the schema) | **Accept, corrected further** | Both spec and verifier were wrong: the DL translator never uses `LLMTradingDecision`/`validator.py`. A SELL is sized from holdings regardless of size, and the batch is truncated to one action per universe symbol (`portfolio_manager.py:802-812, 918-931`). C04 DL now predicts a FAIL on orders; D18 widened. |
| 4 | V1 | Decimal needed for floor/ceil | **Accept** | Same as #6; merged into §1 "Exact arithmetic". |
| 5 | V1 | Scoring rule for matching `OrderFinal` unstated | **Accept** | §1 now states the order-matching rule and that a missing `OrderFinal` is a FAIL. |
| 6 | V2 | C02 float floor gives 7.999999999, outside `Tol.qty` | **Accept** | Reproduced (`math.floor(8.0/1e-9) = 7999999999`). Exact arithmetic is now mandated; `Tol.qty` is kept at one grid step on purpose, so a one-step-short engine fails. |
| 7 | V2 | Tally inconsistent: §8.2 makes costs/CA N/E, §7 counts them FAIL | **Accept** | §1 now defines N/E on intents only; environment facts (costs, splits, dividends) are run and ignored, scored FAIL. §8.2 no longer raises for them. The tally is reproducible from the spec. |
| 8 | V2 | D adapter reads orders from capped, fills-free metadata | **Accept** | Confirmed `_unfilled_order_events` (`engine.py:193-217`) and cap 200. §8.2 reads `pm.order_events` in memory, expands `repeat_count` and maps statuses (`partial` → `partially_filled`, `insufficient_position` → `no_position`). |
| 9 | V2 | F05 per-fill ceil vs Alpaca summed-EOD ceil | **Accept as explicit decision** | Re-read both Alpaca pages: they do conflict. Per-fill per-component is kept (attributable, upper bound); the caveat is corrected to "up to $0.01 per component per fill", F05 records the 1,247.72 alternative, and Q3 carries the decision. |
| 10 | V2 | P4 optimistic vs Alpaca buying power; settlement/GFV/PDT unstated | **Accept (label)** | P4 now states both divergences and puts settlement, GFV and PDT out of scope. Policy unchanged. |
| 11 | V2 | P2 next-open fill is effectively zero latency | **Accept (label)** | Labelled in P2; latency parameter recorded as future work. No case: bars cannot express sub-bar delay. |
| 12 | V2 | CAT on both sides; TAF cap per execution; "net" incomplete | **Accept** | Added `cat_fee` (default 0, unverified); P11 states the cap is per execution, which equals per fill here; Q3 notes "net" is incomplete without CAT. |
| 13 | V2 | Limit touch-fill and exact stop fill are optimistic | **Accept (label)** | P15 relabelled. Checked that every limit fill in §4 trades strictly through, so the cases pass under either convention. |
| 14 | V2 | P7 omits fractional-OCO ban and $1 minimum | **Accept** | Confirmed on Alpaca's fractional page (DAY market/limit/stop/stop-limit only; "as little as $1"). Added to P7 with two new reasons. No existing case is affected; a case is listed in §5.1. |
| 15 | V2 | P16 working-order rescale untested; Alpaca behaviour unverified | **Accept, policy changed** | Alpaca FAQ: a reverse split cancels all GTC orders; a forward split adjusts only buy limits and sell stops; CA cancels can be enabled for all GTC. P16 now cancels every working order (`corporate_action`), with the forward-split divergence labelled. CA04 added (cancel + cash in lieu). Reverse split and split+dividend same date are listed in §5.1. |
| 16 | V2 | P8 `E_open` once per bar or per fill; P5 solver unspecified | **Accept** | P8 pins a single snapshot per bar (as S02 assumes); P5 specifies binary search over the grid index, exact because cost is monotone in qty. |
| 17 | V2 | P13 market-order cancel on halt is a simplification | **Accept (label)** | Labelled in P13 with the reason limit/stop orders are carried. |
| 18 | V2 | P19 unstated Sharpe choices | **Accept** | P19 now states idle sessions count, rf is simple, and agent and baselines share the grid and rf. |
| 19 | V2 | No half-day or DST case | **Accept half-day, defer DST** | T05 added; it exposes a new defect D19 (session window hard-coded to 16:00, `sessions.py:40-44`). DST is listed in §5.1: local-time grids make it invisible to the case data, so it tests conversion code, not policy. |
| 20 | V2 | Six further coverage gaps | **Accept as backlog** | Listed in §5.1. Each needs only one case once its policy is settled; adding all six now would grow the suite without moving any adoption decision. |
| 21 | V1 | `protocol.py:92-135` "v1 has no order type" imprecise | **Accept** | Reworded to "market only" with `protocol.py:24, 99-100`. |
| 22 | V1 | `external_run_service.py:290` should be 289 | **Accept** | Confirmed `resolve_initial_capital` at 289. |
| 23 | V1 | `validator.py:379-380` should be 378-381 | **Reject** | On `origin/main` line 379 is `cost = shares * price` and 380 is `if cost <= cash:`; the citation is exact. |
| 24 | V1 | `alpaca_bars.py:732` rests on an absent argument | **Accept** | Reworded: the request at 732-738 passes no `adjustment`, so the raw default applies. |
| 25 | V1 | Uncited-but-unopened files (`bar_aggregation.py:236-264`, test template) | **Reject (no action)** | Not a defect claim. `tests/backtesting/test_engine_minute_source.py` exists at that path on `origin/main`. |
| 26 | V2 | M01 test path `tests/backtesting/test_external_minute_source.py:113` wrong | **Accept** | The file is `tests/test_external_minute_source.py`, and on `origin/main` the `== 13` assertion is at line **117**, not 112-113. |
| 27 | V2 | `test_direction_matrix.py` is under `tests/`, not `tests/oracle/` | **Accept** | Confirmed in the ml4t checkout; path made explicit. |
| 28 | V2 | `engine.py:1926-1935` drift | **Accept** | Changed to 1928-1936 in T01 and D4. |
| 29 | V2 | `api/v2/models.py:79-86` cites the class head, not the universe validator | **Accept** | Changed to `:90-95` (`_symbol_in_universe`). |
| 30 | V1+V2 | External fee rates and dates | **Accept (confirmation)** | Both verifiers confirmed them independently; V2 also confirmed SEC $0.00 for 2025-05-14 → 2026-04-03. No change. |

Informational confirmations needing no action: V1 found no case-arithmetic errors and every case discriminating its target; V1 and V2 confirmed the other ~40 file:line citations.

## Owner decisions (2026-10-07)

| Q | Decision |
|---|---|
| 1 | The session-final decision fills at the next session's open (P2 as written). |
| 2 | The participation-cap remainder is **cancelled** (P14 as written). |
| 3 | Defaults: commission 0, the dated SEC/TAF table on, 5 bps slippage per side, a 10% participation cap. Results are published **net**, with gross alongside. Fees use a per-fill, per-component ceiling until a real Alpaca confirmation is checked. The CAT rate is to be verified. |
| 4 | Alpaca's corporate-actions API supplies the events, applied as ledger events on raw prices, with cash in lieu and a back-adjusted series for indicators (P16, P17 as written). |
| 5 | Sharpe follows P19. **Display requirement (plain language, revised by owner):** label `Sharpe ratio`. Tooltip: "Return above what cash would have earned, divided by how much the daily returns swung. Based on {N} trading days; cash rate {rf}%/yr; scaled to a yearly figure." No jargon on screen (no "T-bill", "excess", "√252", "sample SD"); those details go in the docs/methodology page. When N < 2 or the returns never moved it shows "—" with a plain reason, e.g. "Needs at least 2 trading days". |

**Scope narrowed by the owner:** only the My Agents backtest is in scope (`POST /backtest/run` → `HourlyBacktester`). Column S (step sessions) and D17 are out of scope and kept for reference only. DL (the LLM translator, D18) **is** in scope, because My Agents runs are LLM-driven.

**D20 (found 2026-10-07, after review):** `dashboard/frontend/js/backtest-comparison.js:30-66` recomputes Sharpe in the browser on whatever curve points it receives, using ddof 0, rf 0 and √(252 × 6.5). The comparison view can therefore show a Sharpe that disagrees with the backend's (D15, √19,656), with a third annualization. Fix: the engine computes metrics once (P19), and the frontend displays them and never recomputes.
