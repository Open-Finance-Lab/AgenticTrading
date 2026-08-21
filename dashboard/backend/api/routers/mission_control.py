"""Mission Control overview: real-money and paper wallet balances and
holdings side by side, in one response, for the dedicated dashboard page.

Deliberately read-only. Placing an order is out of scope for this router --
see ``dashboard.backend.execution.alpaca_live_service`` for the risk-gated
live-trading path.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from dashboard.backend.execution.alpaca_live_service import execute_enabled, max_order_usd
from dashboard.backend.infrastructure.brokers.alpaca_live import (
    AlpacaLiveCredentialsError,
    AlpacaLiveTradingClient,
)
from dashboard.backend.infrastructure.brokers.alpaca_paper import AlpacaPaperTradingClient

router = APIRouter(prefix="/v1/mission-control", tags=["mission-control"])


def _paper_snapshot() -> dict:
    try:
        client = AlpacaPaperTradingClient()
    except Exception as e:
        print(f"⚠️ Mission Control: paper account unavailable: {e}")
        return {"configured": False, "error": "Paper account not configured", "account": None, "positions": []}

    account = client.get_account()
    positions = client.get_positions()
    return {
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


def _live_snapshot() -> dict:
    try:
        client = AlpacaLiveTradingClient()
    except AlpacaLiveCredentialsError as e:
        print(f"⚠️ Mission Control: live account not configured: {e}")
        return {"configured": False, "error": "Live account not configured", "account": None, "positions": []}

    account = client.get_account()
    positions = client.get_positions_detailed()
    return {
        "configured": account is not None,
        "error": None if account is not None else "Failed to fetch live account",
        "account": account,
        "positions": positions,
    }


@router.get("/overview")
def get_overview():
    """Everything the Mission Control page needs in one call: both wallets,
    both holdings lists, and whether live execution is actually armed."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper": _paper_snapshot(),
        "live": _live_snapshot(),
        "live_execute_enabled": execute_enabled(),
        "live_max_order_usd": max_order_usd(),
    }
