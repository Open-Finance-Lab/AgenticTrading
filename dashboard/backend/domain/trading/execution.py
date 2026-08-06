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
* a sell of a symbol that is not held emits an ``insufficient_position``
  ``rejected_orders`` record, where the legacy branch does not. That list is
  the T+1 audit trail and stays A-share-only.

``order_events`` is the exception to "legacy behaviour is byte-identical": it
is populated for *every* market, including unfilled orders on the legacy
branches. Nothing about execution changes -- cash, positions, and ``trades``
are untouched -- but an order that could not fill now leaves a trace instead of
vanishing, because the UI built on this list claims to show all orders and a
DJIA run that silently drops its unaffordable buys makes that claim false.

This module is domain-only: it must not import FastAPI, Anthropic, Alpaca
clients, the database singleton, API routers, or scripts.
"""

import math
import numbers
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
    action: str,
    requested_shares: float,
    executed_shares: float,
    unfilled_shares: float,
    reason: str,
) -> None:
    rejected_orders.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "action": action,
        "requested_shares": requested_shares,
        "executed_shares": executed_shares,
        "unfilled_shares": unfilled_shares,
        "status": "partial" if executed_shares > 0 else "rejected",
        "reason": reason,
    })


def _is_valid_lot_quantity(shares: Any, lot_size: int) -> bool:
    """Return whether an A-share quantity is a positive whole lot.

    Rejects non-numeric types outright rather than coercing them. ``float()``
    would happily accept the string ``"100"``, which passes every lot check and
    then raises ``TypeError`` downstream at ``shares * price`` -- so a guard
    that coerced would *look* like validation while letting the one input it
    should have caught reach the arithmetic.
    """
    # ``numbers.Real`` rather than ``(int, float)``: numpy registers its scalar
    # types with the numeric ABCs, and ``np.int64`` is not an ``int`` subclass.
    if isinstance(shares, bool) or not isinstance(shares, numbers.Real):
        return False
    try:
        numeric_shares = float(shares)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(numeric_shares)
        and numeric_shares > 0
        and numeric_shares.is_integer()
        and int(numeric_shares) % lot_size == 0
    )


def _repeat_key(
    *, timestamp: Any, symbol: str, side: str, reason: str
) -> Optional[tuple]:
    """Identity of a rejection that repeating bars would re-emit verbatim.

    ``None`` when no market date can be read off the timestamp, in which case
    the caller must record the event rather than risk collapsing two genuinely
    distinct rejections into one.
    """
    try:
        trading_date = _trading_date(timestamp)
    except TypeError:
        return None
    return (symbol, side, reason, trading_date)


def _append_order_event(
    order_events: Optional[List[Dict]],
    *,
    timestamp: Any,
    symbol: str,
    side: str,
    requested_shares: Any,
    executed_shares: Any,
    price: Any,
    status: str,
    reason: str,
    strategy_reason: str,
    repeat_index: Optional[Dict] = None,
) -> None:
    if order_events is None:
        return
    unfilled_shares = requested_shares - executed_shares
    if abs(unfilled_shares) <= _SHARE_EPSILON:
        unfilled_shares = 0

    # A signal the agent cannot act on does not go away: an unaffordable BUY
    # re-fires on every bar for as long as the indicator holds, and each bar
    # would otherwise mint an identical rejection. Left unchecked those
    # duplicates fill the persisted head sample end to end, so the audit is
    # least informative exactly when the constraint bound hardest -- the same
    # failure `t1_deferrals` was added to avoid, and it is deduped the same
    # way, per symbol-trading-day.
    #
    # Only *pure* rejections collapse. A fill or partial fill moved the ledger,
    # so it is a distinct event however much it looks like the last one.
    repeat_key = None
    if repeat_index is not None and executed_shares == 0:
        repeat_key = _repeat_key(
            timestamp=timestamp, symbol=symbol, side=side, reason=reason
        )
    if repeat_key is not None:
        existing = repeat_index.get(repeat_key)
        if existing is not None:
            # Carry the scale rather than dropping it: "this happened 47 times
            # today" is the part a reader of a collapsed record needs.
            existing["repeat_count"] = existing.get("repeat_count", 1) + 1
            return

    order_events.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "requested_shares": requested_shares,
        "executed_shares": executed_shares,
        "unfilled_shares": unfilled_shares,
        "price": price,
        "executed_value": executed_shares * price,
        "status": status,
        "reason": reason,
        "strategy_reason": strategy_reason,
    })
    if repeat_key is not None:
        repeat_index[repeat_key] = order_events[-1]


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
    lot_size: int = 1,
    order_events: Optional[List[Dict]] = None,
    order_event_repeats: Optional[Dict] = None,
) -> float:
    """Apply ``actions`` to the given portfolio state in place.

    ``positions``, ``entry_prices`` and ``trades`` are mutated in place. ``cash``
    is a scalar and the (possibly updated) value is returned; callers must
    reassign it. The return value is the new cash balance, matching the original
    method's mutation of ``self.cash``.

    ``order_event_repeats`` is the caller-owned index that lets a repeating
    rejection collapse across calls (see ``_append_order_event``). Omitting it
    keeps every event, which is what a single-call unit test wants; the
    ``PortfolioManager`` owns one for the lifetime of a run.
    """
    def _record_order_event(**fields) -> None:
        _append_order_event(
            order_events, repeat_index=order_event_repeats, **fields
        )

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

        if (
            lot_size > 1
            and action_type in {"buy", "sell"}
            and not _is_valid_lot_quantity(shares, lot_size)
        ):
            if rejected_orders is not None:
                _append_rejection(
                    rejected_orders,
                    timestamp=timestamp,
                    symbol=symbol,
                    action=action_type,
                    requested_shares=shares,
                    executed_shares=0,
                    unfilled_shares=shares,
                    reason="invalid_lot_size",
                )
            _record_order_event(
                timestamp=timestamp,
                symbol=symbol,
                side=action_type.upper(),
                requested_shares=shares,
                executed_shares=0,
                price=price,
                status="rejected",
                reason="invalid_lot_size",
                strategy_reason=reason,
            )
            continue

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
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="BUY",
                    requested_shares=shares,
                    executed_shares=shares,
                    price=price,
                    status="filled",
                    reason="",
                    strategy_reason=reason,
                )
            elif shares > 0:
                # The order-event ledger records this for every market, not
                # just A-shares: a buy the portfolio could not afford is an
                # outcome the "All Orders" log promises to show, and dropping
                # it silently on DJIA is the same fail-closed-but-invisible
                # gap the A-share path was built to close. `rejected_orders`
                # stays A-share-only -- it is the T+1 audit trail, and the
                # legacy single-share branch's semantics are frozen.
                if lot_size > 1:
                    if rejected_orders is not None:
                        _append_rejection(
                            rejected_orders,
                            timestamp=timestamp,
                            symbol=symbol,
                            action="buy",
                            requested_shares=shares,
                            executed_shares=0,
                            unfilled_shares=shares,
                            reason="insufficient_cash_for_lot",
                        )
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="BUY",
                    requested_shares=shares,
                    executed_shares=0,
                    price=price,
                    status="rejected",
                    reason=(
                        "insufficient_cash_for_lot"
                        if lot_size > 1
                        else "insufficient_cash"
                    ),
                    strategy_reason=reason,
                )

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
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="SELL",
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    price=price,
                    status="filled",
                    reason="",
                    strategy_reason=reason,
                )
                continue
            frozen_shares = max(held_shares - sellable_shares, 0)
            t1_unfilled = min(unfilled_shares, frozen_shares)
            if t1_unfilled > _SHARE_EPSILON:
                _append_rejection(
                    rejected_orders,
                    timestamp=timestamp,
                    symbol=symbol,
                    action="sell",
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
                    action="sell",
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    unfilled_shares=insufficient_shares,
                    reason="insufficient_position",
                )
            primary_reason = (
                "t1_frozen"
                if t1_unfilled > _SHARE_EPSILON
                else "insufficient_position"
            )
            _record_order_event(
                timestamp=timestamp,
                symbol=symbol,
                side="SELL",
                requested_shares=shares,
                executed_shares=sell_shares,
                price=price,
                status="partial" if sell_shares > 0 else "rejected",
                reason=primary_reason,
                strategy_reason=reason,
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
                unfilled_shares = shares - sell_shares
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="SELL",
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    price=price,
                    status=(
                        "filled"
                        if unfilled_shares <= _SHARE_EPSILON
                        else "partial"
                    ),
                    reason=(
                        ""
                        if unfilled_shares <= _SHARE_EPSILON
                        else "insufficient_position"
                    ),
                    strategy_reason=reason,
                )
            elif shares > 0:
                # Recorded for every market, for the reason given on the BUY
                # branch above. Execution behaviour is unchanged: the legacy
                # branch still skips silently, it just no longer skips
                # *invisibly*.
                if lot_size > 1 and rejected_orders is not None:
                    _append_rejection(
                        rejected_orders,
                        timestamp=timestamp,
                        symbol=symbol,
                        action="sell",
                        requested_shares=shares,
                        executed_shares=0,
                        unfilled_shares=shares,
                        reason="insufficient_position",
                    )
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="SELL",
                    requested_shares=shares,
                    executed_shares=0,
                    price=price,
                    status="rejected",
                    reason="insufficient_position",
                    strategy_reason=reason,
                )

    return cash
