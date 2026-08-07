"""Deterministic rule-based reference trading agent.

Extracted (Phase 2C1) from ``PortfolioManager.make_trading_decision`` in
``dashboard/scripts/backtest_hourly_agent.py``. This is pure, deterministic,
domain-level decision logic over an explicit portfolio-state snapshot plus the
current positions/cash. The legacy method now delegates here.

Behavior is byte-for-byte identical to the original method. In particular:

* position sizing, BUY/SELL thresholds, reason strings, and the action schema
  are unchanged;
* symbols are iterated in ``portfolio_state["market_signals"]`` insertion order;
* a symbol is skipped when ``pd.isna([rsi, sma20]).any()`` (missing/None/NaN
  ``rsi`` or ``sma20``);
* BUY requires no existing position, ``rsi < 30`` and ``price < sma20``. With
  the default ``lot_size=1``, size is ``int(total_equity * 0.02 / price)`` and
  the existing cash pre-check is unchanged. A market with ``lot_size > 1``
  requests exactly one lot and leaves affordability to the shared executor so
  a rejected order remains auditable;
* SELL (an ``elif``, so mutually exclusive with BUY) requires an existing
  position and (``rsi > 70`` or (``sma50`` truthy and ``price > sma50 * 1.02``));
  it sells the full held quantity;
* the return value is ``{"actions": [...]}`` and inputs are not mutated.

No thresholds, formulas, strings, or default quantities are changed. The LLM
decision workflow is intentionally NOT extracted here.

The one addition is the optional ``available_positions`` argument (T+1 markets).
Left ``None`` — every non-A-share run — the held quantity *is* the sellable
quantity and the emitted actions are unchanged. Supplied, the SELL size is
capped at what would actually fill, and a fully-frozen holding emits no action
at all. Without that cap the agent re-proposes an unfillable full-holding sell
on every bar of the buy date, and each one becomes a ``t1_frozen`` audit record
for a constraint the agent was never shown.

The cap alone would trade a too-loud audit for a silent one: a capped order
fills exactly, so the executor sees nothing to record, and "how often did T+1
stop this agent exiting?" becomes unanswerable. So a capped SELL also reports
the discarded intent — as ``t1_deferrals`` on the return value (the caller
dedupes it per trading day) and as ``requested_shares`` on the action itself.

This module is domain-only: it must not import Anthropic, Alpaca, the database,
FastAPI, API routers, or scripts.
"""

from typing import Dict, List, Optional

import pandas as pd


def make_rule_based_decision(
    *,
    portfolio_state: Dict,
    positions: Dict,
    cash: float,
    available_positions: Optional[Dict] = None,
    lot_size: int = 1,
) -> Dict:
    """Produce rule-based trading actions for the given portfolio state.

    ``portfolio_state`` must provide ``total_equity`` and ``market_signals`` (a
    mapping of symbol -> indicator dict). ``positions`` and ``cash`` reflect the
    current holdings and available cash. ``available_positions`` is the T+1
    sellable balance, or ``None`` when settlement is immediate. ``lot_size`` is
    1 for legacy markets; values above 1 request exactly one market lot on a
    BUY signal. Inputs are read only, never mutated. Returns
    ``{"actions": [...]}`` with the same action dictionaries the original
    method produced, plus a ``t1_deferrals`` key that appears **only** when the
    T+1 cap actually shrank an intended exit.
    """
    actions: List[Dict] = []
    deferrals: List[Dict] = []

    # Calculate total portfolio equity for consistent position sizing
    total_equity = portfolio_state["total_equity"]

    for symbol, signal in portfolio_state["market_signals"].items():
        rsi = signal.get("rsi")
        price = signal.get("price")
        sma20 = signal.get("sma20")
        sma50 = signal.get("sma50")

        # Skip if indicators not ready
        if pd.isna([rsi, sma20]).any():
            continue

        has_position = symbol in positions and positions[symbol] > 0

        # BUY logic: RSI < 30 (oversold)
        if not has_position and rsi < 30 and price < sma20:
            if lot_size > 1:
                # The market lot is the intended order. The executor owns the
                # cash gate so an unaffordable A-share signal remains visible.
                shares_to_buy = lot_size
                should_submit = True
            else:
                # Preserve the legacy 2%-of-equity sizing and cash pre-check.
                risk_amount = total_equity * 0.02
                shares_to_buy = int(risk_amount / price)
                should_submit = (
                    shares_to_buy > 0 and shares_to_buy * price <= cash
                )
            if should_submit:
                actions.append({
                    "symbol": symbol,
                    "action": "buy",
                    "shares": shares_to_buy,
                    "reason": f"RSI oversold ({rsi:.0f}), price below MA"
                })

        # SELL logic: RSI > 70 (overbought) or price above SMA50
        elif has_position and (rsi > 70 or (sma50 and price > sma50 * 1.02)):
            # Under T+1 the held quantity is not necessarily the sellable one.
            # Proposing more than that would only mint rejection records.
            sellable = (
                positions[symbol]
                if available_positions is None
                else min(available_positions.get(symbol, 0), positions[symbol])
            )
            reason = (
                f"RSI overbought ({rsi:.0f})" if rsi > 70 else "Price above MA50"
            )
            if sellable < positions[symbol]:
                # Report the intent the cap discarded. Capping alone would make
                # "T+1 stopped this agent exiting" unobservable: the order fills
                # exactly, so the executor has nothing to audit.
                deferrals.append({
                    "symbol": symbol,
                    "requested_shares": positions[symbol],
                    "sellable_shares": sellable,
                })
            if sellable > 0:
                action = {
                    "symbol": symbol,
                    "action": "sell",
                    "shares": sellable,
                    "reason": reason,
                }
                if sellable < positions[symbol]:
                    # Keep the untruncated intent on the action so the decision
                    # log does not read as though the agent asked for `sellable`.
                    action["requested_shares"] = positions[symbol]
                actions.append(action)

    result: Dict = {"actions": actions}
    if deferrals:
        # Only present when T+1 actually bound, so the returned shape is
        # unchanged for every non-A-share caller.
        result["t1_deferrals"] = deferrals
    return result
