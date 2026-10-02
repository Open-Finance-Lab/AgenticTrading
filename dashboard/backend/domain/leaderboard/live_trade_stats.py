"""Cumulative trade statistics for a Live board entry.

A live month is many segments (one per nightly increment), each a fresh
``PortfolioManager`` whose ``trades`` list starts empty. Storing every trade on
every checkpoint row would grow the row with the month, so a checkpoint stores
this fold instead: counters plus the FIFO lots still open, which is all the
next segment needs to keep matching sells to the buys that opened them.

Holding time is calendar time between a buy fill and the sell fill that
closes it, weighted by shares. Closed trades only: a position still open has
no holding time yet.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

TRADE_STATS_KEY = "live_trade_stats"
STATS_VERSION = 1
_EPS = 1e-9


def empty_trade_stats() -> Dict[str, Any]:
    return {
        "version": STATS_VERSION,
        "complete": True,
        "buys": 0,
        "sells": 0,
        "closed_sells": 0,
        "winning_sells": 0,
        "held_share_hours": 0.0,
        "held_shares": 0.0,
        "open_lots": {},
    }


def _ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def fold_trade_stats(
    prior: Optional[Dict[str, Any]],
    trades: Iterable[Dict[str, Any]],
    *,
    prior_has_history: bool = False,
) -> Dict[str, Any]:
    """``prior``'s stats with ``trades`` applied in order.

    ``prior`` of None starts flat. When it is None but the resumed row already
    traded (``prior_has_history``: a row written before this fold existed), the
    result is marked incomplete — its counters would silently undercount the
    month, and a sell could close a lot this fold never saw.
    """
    if prior and prior.get("version") == STATS_VERSION:
        stats = {
            **prior,
            "open_lots": {s: [list(l) for l in lots] for s, lots in (prior.get("open_lots") or {}).items()},
        }
    else:
        stats = empty_trade_stats()
        if prior_has_history:
            stats["complete"] = False

    lots: Dict[str, List[List[Any]]] = stats["open_lots"]
    for trade in trades:
        side = str(trade.get("side") or "").upper()
        symbol = str(trade.get("symbol") or "")
        qty = float(trade.get("shares") or trade.get("quantity") or 0)
        price = float(trade.get("price") or 0)
        when = _ts(trade.get("timestamp"))
        if not symbol or qty <= _EPS or when is None:
            continue
        if side == "BUY":
            stats["buys"] += 1
            lots.setdefault(symbol, []).append([qty, when.isoformat(), price])
            continue
        if side != "SELL":
            continue
        stats["sells"] += 1
        remaining = qty
        pnl = 0.0
        book = lots.get(symbol, [])
        while remaining > _EPS and book:
            lot_qty, lot_ts, lot_price = book[0]
            take = min(remaining, float(lot_qty))
            opened = _ts(lot_ts)
            if opened is not None:
                hours = max((when - opened).total_seconds() / 3600.0, 0.0)
                stats["held_share_hours"] += hours * take
                stats["held_shares"] += take
            pnl += (price - float(lot_price)) * take
            remaining -= take
            if float(lot_qty) - take <= _EPS:
                book.pop(0)
            else:
                book[0][0] = float(lot_qty) - take
        if remaining > _EPS:
            stats["complete"] = False
        if remaining < qty - _EPS:
            stats["closed_sells"] += 1
            if pnl > 0:
                stats["winning_sells"] += 1
        if not book:
            lots.pop(symbol, None)
    return stats


def summarize_trade_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Public figures for one entry; None where the fold cannot vouch for them."""
    if not stats or stats.get("version") != STATS_VERSION or not stats.get("complete"):
        return {"win_rate": None, "avg_hold_hours": None, "closed_trades": None}
    closed = int(stats.get("closed_sells") or 0)
    held = float(stats.get("held_shares") or 0)
    return {
        "win_rate": (int(stats.get("winning_sells") or 0) / closed) if closed else None,
        "avg_hold_hours": (float(stats.get("held_share_hours") or 0) / held) if held > _EPS else None,
        "closed_trades": closed,
    }
