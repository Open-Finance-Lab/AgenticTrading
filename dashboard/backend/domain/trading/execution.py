"""In-memory order-execution / trade-mutation helpers.

Extracted (Phase 2B3) from ``PortfolioManager.execute_actions`` in
``dashboard/scripts/backtest_hourly_agent.py``. This is pure, domain-level
execution logic over explicit state (cash, positions, entry prices, trades). The
legacy method now delegates here.

With ``t_plus_one_enabled=False`` (the default), behavior remains identical to
the original method. In particular:

* action fields are read with ``action.get("symbol")``, ``action.get("action")``,
  ``action.get("shares", 0)``, ``action.get("reason", "")``;
* a symbol absent from ``market_data`` is skipped (the original ``continue``);
  ``price_cache`` is intentionally NOT consulted here, exactly as before, so this
  module deliberately does NOT reuse ``portfolio.resolve_price``;
* execution price is always ``market_data[symbol]["close"]``;
* BUY executes only when ``cost <= cash and shares > 0``, sets
  ``entry_prices[symbol] = price``, and records a trade with a ``cost`` field;
* SELL executes only when the symbol is held with a positive quantity, sells
  ``min(shares, positions[symbol])``, removes the position and its entry price
  when the holding reaches zero, and records a trade with a ``proceeds`` field
  and ``shares`` equal to the executed quantity;
* HOLD / unknown action types are no-ops;
* a later action is never blocked by an earlier skipped/failed one;
* ``positions``, ``entry_prices`` and ``trades`` are mutated in place; ``cash`` is
  a scalar and is returned so the caller can reassign it.

The optional T+1 path adds an available-position balance, date-stamped frozen
buy lots, partial fills, and rejected-order audit records. It is enabled only by
the iFinD A-share market profiles.

The T+1 SELL branch is a rewrite rather than a wrapper, so it also diverges from
the legacy branch in two ways that are NOT T+1 semantics. Both are deliberate;
neither can reach a run with ``t_plus_one_enabled=False``:

* it guards ``shares <= 0`` and skips. The legacy branch does not: a sell of
  ``-5`` evaluates ``min(-5, 100) == -5``, which *adds* 5 shares to the position
  and *subtracts* the "proceeds" from cash. That is a latent bug, kept only
  because the legacy path is frozen;
* a sell of a symbol that is not held emits an ``insufficient_position`` audit
  record, where the legacy branch skips silently. The audit trail is the point
  of the T+1 path, so "the agent asked for something impossible" is recorded
  rather than dropped.

This module is domain-only: it must not import FastAPI, Anthropic, Alpaca
clients, the database singleton, API routers, or scripts.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional


# Share quantities are integral for A-shares, but the ledger arithmetic is
# float, so a fully-filled sell can leave residue on the order of 1e-17
# (0.3 - 0.1 - 0.2). Without a tolerance that residue is "unfilled" and mints a
# rejection record for ~0 shares, which reads as a real constraint violation.
_SHARE_EPSILON = 1e-9


def _trading_date(timestamp: Any) -> date:
    """Return the market date carried by one backtest timestamp."""
    if isinstance(timestamp, datetime):
        return timestamp.date()
    if isinstance(timestamp, date):
        return timestamp
    date_method = getattr(timestamp, "date", None)
    if callable(date_method):
        value = date_method()
        if isinstance(value, date):
            return value
    raise TypeError("T+1 execution requires a date-like timestamp")


def _release_frozen_lots(
    *,
    current_date: date,
    available_positions: Dict,
    frozen_lots: Dict,
) -> None:
    """Move every prior-date buy lot into the sellable balance."""
    for symbol, lots in list(frozen_lots.items()):
        remaining = []
        released = 0
        for lot in lots:
            if lot["buy_date"] < current_date:
                released += lot["quantity"]
            else:
                remaining.append(lot)
        if released:
            available_positions[symbol] = (
                available_positions.get(symbol, 0) + released
            )
        if remaining:
            frozen_lots[symbol] = remaining
        else:
            frozen_lots.pop(symbol, None)


def _append_rejection(
    rejected_orders: List[Dict],
    *,
    timestamp: Any,
    symbol: str,
    requested_shares: float,
    executed_shares: float,
    unfilled_shares: float,
    reason: str,
) -> None:
    rejected_orders.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "action": "sell",
        "requested_shares": requested_shares,
        "executed_shares": executed_shares,
        "unfilled_shares": unfilled_shares,
        "status": "partial" if executed_shares > 0 else "rejected",
        "reason": reason,
    })


def execute_actions(
    *,
    actions: List[Dict],
    market_data: Dict,
    timestamp: datetime,
    cash: float,
    positions: Dict,
    entry_prices: Dict,
    trades: List[Dict],
    t_plus_one_enabled: bool = False,
    available_positions: Optional[Dict] = None,
    frozen_lots: Optional[Dict] = None,
    rejected_orders: Optional[List[Dict]] = None,
) -> float:
    """Apply ``actions`` to the given portfolio state in place.

    ``positions``, ``entry_prices`` and ``trades`` are mutated in place. ``cash``
    is a scalar and the (possibly updated) value is returned; callers must
    reassign it. The return value is the new cash balance, matching the original
    method's mutation of ``self.cash``.
    """
    current_date = None
    if t_plus_one_enabled:
        if (
            available_positions is None
            or frozen_lots is None
            or rejected_orders is None
        ):
            raise ValueError(
                "T+1 execution requires available positions, frozen lots, "
                "and rejected-order state"
            )
        current_date = _trading_date(timestamp)
        _release_frozen_lots(
            current_date=current_date,
            available_positions=available_positions,
            frozen_lots=frozen_lots,
        )

    for action in actions:
        symbol = action.get("symbol")
        action_type = action.get("action")
        shares = action.get("shares", 0)
        reason = action.get("reason", "")

        if symbol not in market_data:
            continue

        price = market_data[symbol]["close"]

        if action_type == "buy":
            cost = shares * price
            if cost <= cash and shares > 0:
                cash -= cost
                positions[symbol] = positions.get(symbol, 0) + shares
                entry_prices[symbol] = price
                if t_plus_one_enabled:
                    frozen_lots.setdefault(symbol, []).append({
                        "quantity": shares,
                        "buy_date": current_date,
                    })
                trades.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "BUY",
                    "shares": shares,
                    "price": price,
                    "cost": cost,
                    "reason": reason
                })

        elif action_type == "sell" and t_plus_one_enabled:
            if shares <= 0:
                continue
            held_shares = positions.get(symbol, 0)
            sellable_shares = min(
                available_positions.get(symbol, 0),
                held_shares,
            )
            sell_shares = min(shares, sellable_shares)

            if sell_shares > 0:
                proceeds = sell_shares * price
                cash += proceeds
                positions[symbol] -= sell_shares
                available_positions[symbol] -= sell_shares
                if available_positions[symbol] == 0:
                    available_positions.pop(symbol, None)
                if positions[symbol] == 0:
                    positions.pop(symbol, None)
                    entry_prices.pop(symbol, None)
                trades.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "SELL",
                    "shares": sell_shares,
                    "price": price,
                    "proceeds": proceeds,
                    "reason": reason,
                })

            unfilled_shares = shares - sell_shares
            if unfilled_shares <= _SHARE_EPSILON:
                continue
            frozen_shares = max(held_shares - sellable_shares, 0)
            t1_unfilled = min(unfilled_shares, frozen_shares)
            if t1_unfilled > _SHARE_EPSILON:
                _append_rejection(
                    rejected_orders,
                    timestamp=timestamp,
                    symbol=symbol,
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    unfilled_shares=t1_unfilled,
                    reason="t1_frozen",
                )
            insufficient_shares = unfilled_shares - t1_unfilled
            if insufficient_shares > _SHARE_EPSILON:
                _append_rejection(
                    rejected_orders,
                    timestamp=timestamp,
                    symbol=symbol,
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    unfilled_shares=insufficient_shares,
                    reason="insufficient_position",
                )

        elif action_type == "sell":
            if symbol in positions and positions[symbol] > 0:
                sell_shares = min(shares, positions[symbol])
                proceeds = sell_shares * price
                cash += proceeds
                positions[symbol] -= sell_shares
                if positions[symbol] == 0:
                    del positions[symbol]
                    if symbol in entry_prices:
                        del entry_prices[symbol]
                trades.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": "SELL",
                    "shares": sell_shares,
                    "price": price,
                    "proceeds": proceeds,
                    "reason": reason
                })

    return cash
