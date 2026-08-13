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
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
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
    audit: Optional[Dict[str, Any]] = None,
) -> None:
    record = {
        "timestamp": timestamp,
        "symbol": symbol,
        "action": action,
        "requested_shares": requested_shares,
        "executed_shares": executed_shares,
        "unfilled_shares": unfilled_shares,
        "status": "partial" if executed_shares > 0 else "rejected",
        "reason": reason,
    }
    if audit is not None:
        record.update(audit)
    rejected_orders.append(record)


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
    costs: Optional[Dict[str, float]] = None,
    repeat_index: Optional[Dict] = None,
    audit: Optional[Dict[str, Any]] = None,
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

    event = {
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
    }
    if costs is not None:
        event.update(costs)
    if audit is not None:
        event.update(audit)
    order_events.append(event)
    if repeat_key is not None:
        repeat_index[repeat_key] = order_events[-1]


_MONEY_QUANTUM = Decimal("0.01")


def _round_money(value: Decimal) -> Decimal:
    """Round a monetary amount to CNY cents deterministically."""
    return value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _zero_transaction_costs(reference_price: Any) -> Dict[str, float]:
    """Return an explicit zero-cost breakdown for a rejected A-share order."""
    price = float(reference_price)
    return {
        "reference_price": price,
        "slippage_amount": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "total_fees": 0.0,
        "net_cash_impact": 0.0,
    }


def calculate_transaction_costs(
    *,
    side: str,
    reference_price: Any,
    shares: Any,
    transaction_cost_profile: Any = None,
) -> Dict[str, float]:
    """Calculate one deterministic simulated fill and its cash impact.

    A ``None`` profile deliberately preserves the legacy US execution path. A
    configured profile uses adverse-direction slippage, market price-tick
    rounding, and cent-rounded fees. The function does not mutate portfolio
    state, making the accounting independently testable.

    ONE CALL IS ONE SUBMITTED ORDER. There is no order identity here, so the
    per-order commission floor (¥5 on A-shares) is charged once per call. Every
    caller today executes an action in a single call, which is what makes that
    correct. Anything that later fills an order in pieces must cost the whole
    order and take differences between quotes, never call this per piece —
    otherwise the floor is charged once per fragment.

    ``gross_value`` is ``price * shares`` at the SLIPPED execution price, so it
    already contains ``slippage_amount``; the latter is reported separately for
    audit display only. Cash moved is ``net_cash_impact`` — adding slippage to
    ``gross_value`` or to ``total_fees`` double-counts it.
    """
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError(f"unsupported transaction side: {side!r}")

    if transaction_cost_profile is None:
        price = float(reference_price)
        gross = float(shares) * price
        return {
            "reference_price": price,
            "price": price,
            "gross_value": gross,
            "slippage_amount": 0.0,
            "commission": 0.0,
            "stamp_duty": 0.0,
            "transfer_fee": 0.0,
            "total_fees": 0.0,
            "net_cash_impact": -gross if normalized_side == "buy" else gross,
        }

    try:
        reference = Decimal(str(reference_price))
        quantity = Decimal(str(shares))
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError("price and shares must be numeric") from exc
    if not reference.is_finite() or reference <= 0:
        raise ValueError("reference price must be positive and finite")
    if not quantity.is_finite() or quantity <= 0:
        raise ValueError("filled shares must be positive and finite")

    rate_name = (
        "buy_slippage_rate"
        if normalized_side == "buy"
        else "sell_slippage_rate"
    )
    slippage_rate = Decimal(str(getattr(transaction_cost_profile, rate_name)))
    direction = Decimal(1) + slippage_rate
    if normalized_side == "sell":
        direction = Decimal(1) - slippage_rate
    raw_price = reference * direction
    tick = Decimal(str(transaction_cost_profile.price_tick))
    rounding = ROUND_CEILING if normalized_side == "buy" else ROUND_FLOOR
    execution_price = (raw_price / tick).to_integral_value(rounding=rounding) * tick
    execution_price = _round_money(execution_price)

    gross_value = _round_money(execution_price * quantity)
    slippage_amount = _round_money(abs(execution_price - reference) * quantity)
    commission = max(
        _round_money(
            gross_value * Decimal(str(transaction_cost_profile.commission_rate))
        ),
        _round_money(Decimal(str(transaction_cost_profile.minimum_commission))),
    )
    stamp_duty = Decimal("0")
    if normalized_side == "sell":
        stamp_duty = _round_money(
            gross_value * Decimal(str(transaction_cost_profile.stamp_duty_sell_rate))
        )
    transfer_fee = _round_money(
        gross_value * Decimal(str(transaction_cost_profile.transfer_fee_rate))
    )
    total_fees = commission + stamp_duty + transfer_fee
    net_cash_impact = (
        -(gross_value + total_fees)
        if normalized_side == "buy"
        else gross_value - total_fees
    )

    return {
        "reference_price": float(reference),
        "price": float(execution_price),
        "gross_value": float(gross_value),
        "slippage_amount": float(slippage_amount),
        "commission": float(commission),
        "stamp_duty": float(stamp_duty),
        "transfer_fee": float(transfer_fee),
        "total_fees": float(total_fees),
        "net_cash_impact": float(_round_money(net_cash_impact)),
    }


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
    transaction_cost_profile: Any = None,
    market_rules: Optional[Dict[str, Any]] = None,
    fallback_prices: Optional[Dict[str, Any]] = None,
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
    costs_enabled = transaction_cost_profile is not None

    current_market_rule_audit = None

    def _record_order_event(**fields) -> None:
        _append_order_event(
            order_events,
            repeat_index=order_event_repeats,
            audit=current_market_rule_audit,
            **fields,
        )

    def _record_rejection(**fields) -> None:
        if rejected_orders is not None:
            _append_rejection(
                rejected_orders,
                audit=current_market_rule_audit,
                **fields,
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

        current_market_rule_audit = None
        market_rule = None
        if market_rules is not None and action_type in {"buy", "sell"}:
            market_rule = market_rules.get(symbol)

            reference_price = None
            if symbol in market_data:
                reference_price = market_data[symbol]["close"]
            elif fallback_prices is not None:
                reference_price = fallback_prices.get(symbol)
            display_price = 0.0 if reference_price is None else reference_price

            if market_rule is None:
                # Fail closed, but do not take the run down with it. Every
                # producer filters to the allowed universe before reaching here,
                # so this is unreachable today — and if one ever stops, letting
                # a symbol trade ungated would defeat the calendar, while
                # raising would discard every bar already simulated for a single
                # stray action. Reject it and leave a record instead.
                _record_rejection(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=action_type,
                    requested_shares=shares,
                    executed_shares=0,
                    unfilled_shares=shares,
                    reason="market_rule_unavailable",
                )
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=action_type.upper(),
                    requested_shares=shares,
                    executed_shares=0,
                    price=display_price,
                    status="rejected",
                    reason="market_rule_unavailable",
                    strategy_reason=reason,
                    costs=(
                        _zero_transaction_costs(display_price)
                        if costs_enabled
                        else None
                    ),
                )
                continue

            current_market_rule_audit = market_rule.to_audit()

            if market_rule.suspended:
                _record_rejection(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=action_type,
                    requested_shares=shares,
                    executed_shares=0,
                    unfilled_shares=shares,
                    reason="suspended",
                )
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=action_type.upper(),
                    requested_shares=shares,
                    executed_shares=0,
                    price=display_price,
                    status="rejected",
                    reason="suspended",
                    strategy_reason=reason,
                    costs=(
                        _zero_transaction_costs(display_price)
                        if costs_enabled
                        else None
                    ),
                )
                continue

        if symbol not in market_data:
            continue

        price = market_data[symbol]["close"]

        if market_rule is not None:
            price_tick = getattr(transaction_cost_profile, "price_tick", 0.01)
            closing_gate_effective = market_rule.closing_gate_effective(
                timestamp=timestamp,
                reference_price=price,
                price_tick=price_tick,
            )
            current_market_rule_audit = market_rule.to_audit(
                closing_gate_effective=closing_gate_effective
            )
            limit_state = market_rule.closing_limit_state.value
            market_rejection_reason = None
            if closing_gate_effective and action_type == "buy" and limit_state == "upper":
                market_rejection_reason = "limit_up_buy_blocked"
            elif (
                closing_gate_effective
                and action_type == "sell"
                and limit_state == "lower"
            ):
                market_rejection_reason = "limit_down_sell_blocked"
            if market_rejection_reason is not None:
                _record_rejection(
                    timestamp=timestamp,
                    symbol=symbol,
                    action=action_type,
                    requested_shares=shares,
                    executed_shares=0,
                    unfilled_shares=shares,
                    reason=market_rejection_reason,
                )
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=action_type.upper(),
                    requested_shares=shares,
                    executed_shares=0,
                    price=price,
                    status="rejected",
                    reason=market_rejection_reason,
                    strategy_reason=reason,
                    costs=(
                        _zero_transaction_costs(price) if costs_enabled else None
                    ),
                )
                continue

        if (
            lot_size > 1
            and action_type in {"buy", "sell"}
            and not _is_valid_lot_quantity(shares, lot_size)
        ):
            if rejected_orders is not None:
                _record_rejection(
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
                costs=(
                    _zero_transaction_costs(price) if costs_enabled else None
                ),
            )
            continue

        if action_type == "buy":
            calculated_costs = None
            if shares > 0:
                calculated_costs = calculate_transaction_costs(
                    side="buy",
                    reference_price=price,
                    shares=shares,
                    transaction_cost_profile=transaction_cost_profile,
                )
            required_cash = (
                -calculated_costs["net_cash_impact"]
                if calculated_costs is not None
                else 0.0
            )
            if calculated_costs is not None and required_cash <= cash:
                execution_price = calculated_costs["price"]
                gross_value = calculated_costs["gross_value"]
                cash += calculated_costs["net_cash_impact"]
                positions[symbol] = positions.get(symbol, 0) + shares
                entry_prices[symbol] = execution_price
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
                    "price": execution_price,
                    "cost": gross_value,
                    "reason": reason
                })
                if costs_enabled:
                    trades[-1].update(calculated_costs)
                if current_market_rule_audit is not None:
                    trades[-1].update(current_market_rule_audit)
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="BUY",
                    requested_shares=shares,
                    executed_shares=shares,
                    price=execution_price,
                    status="filled",
                    reason="",
                    strategy_reason=reason,
                    costs=(calculated_costs if costs_enabled else None),
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
                        _record_rejection(
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
                    costs=(
                        _zero_transaction_costs(price) if costs_enabled else None
                    ),
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
                calculated_costs = calculate_transaction_costs(
                    side="sell",
                    reference_price=price,
                    shares=sell_shares,
                    transaction_cost_profile=transaction_cost_profile,
                )
                proceeds = calculated_costs["gross_value"]
                cash += calculated_costs["net_cash_impact"]
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
                    "price": calculated_costs["price"],
                    "proceeds": proceeds,
                    "reason": reason,
                })
                if costs_enabled:
                    trades[-1].update(calculated_costs)
                if current_market_rule_audit is not None:
                    trades[-1].update(current_market_rule_audit)

            unfilled_shares = shares - sell_shares
            if unfilled_shares <= _SHARE_EPSILON:
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="SELL",
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    price=(
                        calculated_costs["price"]
                        if sell_shares > 0
                        else price
                    ),
                    status="filled",
                    reason="",
                    strategy_reason=reason,
                    costs=(
                        calculated_costs
                        if costs_enabled and sell_shares > 0
                        else (_zero_transaction_costs(price) if costs_enabled else None)
                    ),
                )
                continue
            frozen_shares = max(held_shares - sellable_shares, 0)
            t1_unfilled = min(unfilled_shares, frozen_shares)
            if t1_unfilled > _SHARE_EPSILON:
                _record_rejection(
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
                _record_rejection(
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
                price=(
                    calculated_costs["price"]
                    if sell_shares > 0
                    else price
                ),
                status="partial" if sell_shares > 0 else "rejected",
                reason=primary_reason,
                strategy_reason=reason,
                costs=(
                    calculated_costs
                    if costs_enabled and sell_shares > 0
                    else (_zero_transaction_costs(price) if costs_enabled else None)
                ),
            )

        elif action_type == "sell":
            if symbol in positions and positions[symbol] > 0:
                sell_shares = min(shares, positions[symbol])
                calculated_costs = calculate_transaction_costs(
                    side="sell",
                    reference_price=price,
                    shares=sell_shares,
                    transaction_cost_profile=transaction_cost_profile,
                )
                proceeds = calculated_costs["gross_value"]
                cash += calculated_costs["net_cash_impact"]
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
                    "price": calculated_costs["price"],
                    "proceeds": proceeds,
                    "reason": reason
                })
                if costs_enabled:
                    trades[-1].update(calculated_costs)
                if current_market_rule_audit is not None:
                    trades[-1].update(current_market_rule_audit)
                unfilled_shares = shares - sell_shares
                _record_order_event(
                    timestamp=timestamp,
                    symbol=symbol,
                    side="SELL",
                    requested_shares=shares,
                    executed_shares=sell_shares,
                    price=calculated_costs["price"],
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
                    costs=(calculated_costs if costs_enabled else None),
                )
            elif shares > 0:
                # Recorded for every market, for the reason given on the BUY
                # branch above. Execution behaviour is unchanged: the legacy
                # branch still skips silently, it just no longer skips
                # *invisibly*.
                if lot_size > 1 and rejected_orders is not None:
                    _record_rejection(
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
                    costs=(
                        _zero_transaction_costs(price) if costs_enabled else None
                    ),
                )

    return cash
