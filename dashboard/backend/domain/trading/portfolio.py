"""Portfolio state, valuation, and equity-curve helpers.

Extracted (Phase 2B2) from ``PortfolioManager`` in
``dashboard/scripts/backtest_hourly_agent.py``. These are pure functions over
explicit state (cash, positions, entry prices, market data, price cache) plus an
in-place equity-history mutator. Behavior is byte-for-byte identical to the
original methods, which now delegate here. In particular:

* price resolution order is real ``market_data[symbol]["close"]`` first, then
  forward-filled ``price_cache[symbol][timestamp]``, otherwise the symbol is
  skipped (matching the original ``continue``);
* position-list, market-signal, and portfolio-state dictionary schemas are
  unchanged;
* positions-value and total-equity formulas are unchanged (``positions_value``
  starts at int ``0`` and accumulates, ``total_equity = cash + positions_value``);
* the equity-curve record schema is unchanged and records are appended in place;
* equity-curve retrieval returns the SAME list object (an alias, NOT a copy),
  exactly as the original ``get_equity_curve``.

``PortfolioManager`` itself is intentionally NOT moved; order execution, trade
recording, and decision logic are out of scope for this phase.
"""

from typing import Dict, List, Optional


def resolve_price(symbol, market_data: Dict, price_cache: Optional[Dict] = None, timestamp=None):
    """Resolve a symbol's valuation price, or ``None`` if unavailable.

    Order: real ``market_data[symbol]["close"]`` first, then forward-filled
    ``price_cache[symbol][timestamp]``. Returns ``None`` when no price is
    available so callers can skip the symbol (matching the original ``continue``).
    """
    if symbol in market_data:
        return market_data[symbol]["close"]
    if price_cache and symbol in price_cache and timestamp in price_cache[symbol]:
        return price_cache[symbol][timestamp]
    return None


def build_position_list(
    positions,
    entry_prices,
    market_data,
    price_cache=None,
    timestamp=None,
    available_positions=None,
):
    """Build the per-position detail list and the aggregate positions value.

    Returns ``(position_list, positions_value)``. Symbols without an available
    price are skipped. ``positions_value`` starts at int ``0`` and accumulates
    ``shares * current_price``.

    ``available_positions`` is the T+1 sellable balance. Pass ``None`` (the
    default) when settlement is immediate and ``shares`` is itself sellable —
    the position dicts are then byte-identical to the pre-T+1 schema, so
    non-A-share LLM prompts are unchanged.
    """
    positions_value = 0
    position_list = []

    for symbol, shares in positions.items():
        current_price = resolve_price(symbol, market_data, price_cache, timestamp)
        if current_price is None:
            continue

        position_value = shares * current_price
        positions_value += position_value
        entry_price = entry_prices.get(symbol, current_price)
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        position = {
            "symbol": symbol,
            "shares": shares,
            "entry_price": entry_price,
            "current_price": current_price,
            "position_value": position_value,
            "pnl_pct": pnl_pct,
        }
        if available_positions is not None:
            position["sellable_shares"] = min(
                available_positions.get(symbol, 0), shares
            )
        position_list.append(position)

    return position_list, positions_value


def build_market_signals(market_data):
    """Build the per-symbol market-signal dict from real market data rows."""
    market_signals = {}
    for symbol, row in market_data.items():
        market_signals[symbol] = {
            "price": row["close"],
            "rsi": row.get("rsi_14"),
            "macd": row.get("macd"),
            "macd_signal": row.get("macd_signal"),
            "sma20": row.get("sma20"),
            "sma50": row.get("sma50"),
            "bb_upper": row.get("bb_upper"),
            "bb_lower": row.get("bb_lower"),
        }
    return market_signals


def build_portfolio_state(
    cash,
    positions,
    entry_prices,
    market_data,
    price_cache=None,
    timestamp=None,
    available_positions=None,
) -> Dict:
    """Assemble the full portfolio-state snapshot dictionary.

    ``available_positions`` (T+1 only) adds a ``sellable_shares`` key to each
    position so a decision-maker can see what an order submitted now could
    actually fill. Left ``None`` the snapshot schema is unchanged.
    """
    position_list, positions_value = build_position_list(
        positions,
        entry_prices,
        market_data,
        price_cache,
        timestamp,
        available_positions,
    )
    market_signals = build_market_signals(market_data)

    return {
        "cash": cash,
        "positions": position_list,
        "positions_value": positions_value,
        "total_equity": cash + positions_value,
        "market_signals": market_signals,
    }


def calculate_positions_value(positions, market_data, price_cache=None, timestamp=None):
    """Sum ``shares * price`` across positions, skipping unpriced symbols."""
    positions_value = 0
    for symbol, shares in positions.items():
        price = resolve_price(symbol, market_data, price_cache, timestamp)
        if price is None:
            continue
        positions_value += shares * price
    return positions_value


def build_equity_record(cash, positions, market_data, price_cache=None, timestamp=None) -> Dict:
    """Build a single equity-curve record for the given valuation inputs."""
    positions_value = calculate_positions_value(positions, market_data, price_cache, timestamp)
    return {
        "timestamp": timestamp,
        "equity": cash + positions_value,
        "cash": cash,
        "positions_value": positions_value,
    }


def append_equity_record(equity_history: List[Dict], cash, positions, market_data, price_cache=None, timestamp=None) -> Dict:
    """Append an equity-curve record to ``equity_history`` in place.

    Mutates ``equity_history`` exactly as the original ``update_equity`` did and
    returns the appended record for convenience (the legacy method ignores it).
    """
    record = build_equity_record(cash, positions, market_data, price_cache, timestamp)
    equity_history.append(record)
    return record


def get_equity_curve(equity_history: List[Dict]) -> List[Dict]:
    """Return the equity history list as-is (an alias, NOT a copy)."""
    return equity_history
