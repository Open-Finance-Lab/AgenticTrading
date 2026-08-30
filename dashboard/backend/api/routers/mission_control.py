"""Mission Control overview: real-money and paper wallet balances and
holdings side by side, in one response, for the dedicated dashboard page.

Deliberately read-only. Placing an order is out of scope for this router --
see ``dashboard.backend.execution.alpaca_live_service`` for the risk-gated
live-trading path.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from dashboard.backend.api.auth import require_admin
from dashboard.backend.cache import paper_trading_cache
from dashboard.backend.execution.alpaca_live_service import execute_enabled, max_order_usd
from dashboard.backend.infrastructure.brokers.alpaca_live import (
    AlpacaLiveCredentialsError,
    AlpacaLiveTradingClient,
)
from dashboard.backend.infrastructure.brokers.alpaca_paper import AlpacaPaperTradingClient

router = APIRouter(prefix="/v1/mission-control", tags=["mission-control"])

# Namespaced like paper_trading.py's own keys, in the same shared TTL cache
# (dashboard.backend.cache.paper_trading_cache is a generic cache -- paper
# trading was merely its first consumer). Only a *successful* snapshot is
# cached: an error/not-configured branch reflects a state an operator may be
# actively fixing (e.g. dropping in credentials/alpaca_live.json), and caching
# that would paper over the fix for up to 30s after it lands.
_CACHE_KEY_PAPER = "mission_control:paper"
_CACHE_KEY_LIVE = "mission_control:live"
_CACHE_TTL_SECONDS = 30


def _paper_snapshot() -> dict:
    cached = paper_trading_cache.get(_CACHE_KEY_PAPER)
    if cached is not None:
        return cached

    try:
        client = AlpacaPaperTradingClient()
    except Exception as e:
        print(f"⚠️ Mission Control: paper account unavailable: {e}")
        return {"configured": False, "error": "Paper account not configured", "account": None, "positions": []}

    account = client.get_account()
    positions = client.get_positions()
    snapshot = {
        "configured": account is not None,
        "error": None if account is not None else "Failed to fetch paper account",
        "account": account,
        "positions": [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_fill_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_plpc": p.unrealized_plpc,
                "side": p.side,
            }
            for p in positions
        ],
    }
    if snapshot["configured"]:
        paper_trading_cache.set(_CACHE_KEY_PAPER, snapshot, ttl_seconds=_CACHE_TTL_SECONDS)
    return snapshot


def _live_snapshot() -> dict:
    cached = paper_trading_cache.get(_CACHE_KEY_LIVE)
    if cached is not None:
        return cached

    try:
        client = AlpacaLiveTradingClient()
    except AlpacaLiveCredentialsError as e:
        print(f"⚠️ Mission Control: live account not configured: {e}")
        return {"configured": False, "error": "Live account not configured", "account": None, "positions": []}
    except Exception as e:
        # _load_from_credentials can also raise json.JSONDecodeError / OSError /
        # AttributeError reading credentials/alpaca_live.json -- anything other
        # than the "just not configured" case above. Left uncaught, that used
        # to 500 the whole /overview response and take the paper-wallet half
        # down with it. The sanitized message goes to the client; the real
        # exception stays server-side (test_error_detail_sanitization.py).
        print(f"⚠️ Mission Control: failed to connect to live account: {e}")
        return {"configured": False, "error": "Failed to connect to live account", "account": None, "positions": []}

    account = client.get_account()
    positions = client.get_positions_detailed()
    snapshot = {
        "configured": account is not None,
        "error": None if account is not None else "Failed to fetch live account",
        "account": account,
        "positions": positions,
    }
    if snapshot["configured"]:
        paper_trading_cache.set(_CACHE_KEY_LIVE, snapshot, ttl_seconds=_CACHE_TTL_SECONDS)
    return snapshot


@router.get("/overview")
def get_overview(current_user: dict = Depends(require_admin)):
    """Everything the Mission Control page needs in one call: both wallets,
    both holdings lists, and whether live execution is actually armed.

    Admin-only: this is real-money account data, and unlike every other
    surface behind /api, nothing here is scoped to the caller's own session --
    there is exactly one operator-owned live account. ``require_admin`` (the
    one admin gate; see api/routers/admin.py) covers both "no session" (401)
    and "signed in but not an admin" (403).
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": _paper_snapshot(),
        "live": _live_snapshot(),
        "live_execute_enabled": execute_enabled(),
        "live_max_order_usd": max_order_usd(),
    }
