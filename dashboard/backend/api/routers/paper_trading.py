"""Paper Trading API router.

Canonical location (Phase 3C5C). Moved verbatim from inline route definitions in
``dashboard/backend/app.py``. External paths are preserved exactly
(``/paper/...``) via a router prefix; this router is registered directly on the
app (not under the ``/api`` prefix) to keep the route table identical. Endpoint
names, query parameters/defaults, response envelopes, status codes, exception
messages, cache keys/TTLs, and timestamps are unchanged.

Consumes the canonical provider/domain modules:
    * dashboard.backend.infrastructure.brokers.alpaca_paper
    * dashboard.backend.domain.trading.paper_session
    * dashboard.backend.domain.backtesting.baselines.paper (baseline init helper)
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.database import db
from dashboard.backend.baselines_endpoint import get_baselines_from_db
from dashboard.backend.cache import (
    paper_trading_cache,
    CACHE_KEY_ACCOUNT,
    CACHE_KEY_POSITIONS,
    CACHE_KEY_TRADES,
    CACHE_KEY_PORTFOLIO_HISTORY,
    CACHE_KEY_BASELINES,
    TTL_ACCOUNT,
    TTL_POSITIONS,
    TTL_TRADES,
    TTL_PORTFOLIO_HISTORY,
    TTL_BASELINES,
)
from dashboard.backend.infrastructure.brokers.alpaca_paper import AlpacaPaperTradingClient
from dashboard.backend.domain.trading.paper_session import create_paper_trading_session

router = APIRouter(prefix="/paper")


@router.get("/account")
def get_paper_account():
    """
    Get live paper trading account info from Alpaca.
    
    Returns:
        Account details: cash, equity, buying_power, portfolio_value, etc.
    """
    try:
        cached = paper_trading_cache.get(CACHE_KEY_ACCOUNT)
        if cached:
            return {
                "success": True,
                "account": cached,
                "timestamp": datetime.now().isoformat(),
                "cached": True
            }

        client = AlpacaPaperTradingClient()
        account = client.get_account()

        if account:
            paper_trading_cache.set(CACHE_KEY_ACCOUNT, account, TTL_ACCOUNT)
            return {
                "success": True,
                "account": account,
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
        else:
            return {
                "success": False,
                "error": "Failed to fetch account",
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
    except Exception as e:
        print(f"/paper/account fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch account",
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }


@router.get("/positions")
def get_paper_positions():
    """
    Get current positions from Alpaca paper trading account.
    Cached for 30 seconds (prices update frequently).
    
    Returns:
        List of positions: {symbol, qty, avg_fill_price, current_price, unrealized_pl, unrealized_plpc}
    """
    try:
        # Check cache first
        cached = paper_trading_cache.get(CACHE_KEY_POSITIONS)
        if cached:
            return {
                "success": True,
                "count": len(cached),
                "positions": cached,
                "timestamp": datetime.now().isoformat(),
                "cached": True
            }
        
        client = AlpacaPaperTradingClient()
        positions = client.get_positions()
        
        positions_data = [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_fill_price": p.avg_fill_price,
                "current_price": p.current_price,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_plpc": p.unrealized_plpc,
                "side": p.side,
                "market_value": p.market_value
            }
            for p in positions
        ]
        
        # Cache for 30 seconds
        paper_trading_cache.set(CACHE_KEY_POSITIONS, positions_data, TTL_POSITIONS)
        
        return {
            "success": True,
            "count": len(positions_data),
            "positions": positions_data,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    except Exception as e:
        print(f"/paper/positions fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch positions",
            "positions": [],
            "timestamp": datetime.now().isoformat()
        }


@router.get("/trades")
def get_paper_trades(limit: int = 50):
    """
    Get recent trades/fills from Alpaca paper trading account.
    Cached for 60 seconds (trade history changes infrequently).
    
    Query params:
    - limit: Max number of trades to return (default 50)
    
    Returns:
        List of trades with symbol, qty, side, price, timestamp
    """
    try:
        # Check cache first
        cached = paper_trading_cache.get(CACHE_KEY_TRADES)
        if cached:
            return {
                "success": True,
                "count": len(cached),
                "trades": cached,
                "timestamp": datetime.now().isoformat(),
                "cached": True
            }
        
        client = AlpacaPaperTradingClient()
        activities = client.get_activities(activity_type="FILL", limit=limit)
        
        trades = []
        for activity in activities:
            trades.append({
                "id": activity.get("id"),
                "symbol": activity.get("symbol"),
                "qty": float(activity.get("qty", 0)),
                "side": activity.get("side"),
                "price": float(activity.get("price", 0)),
                "timestamp": activity.get("created_at"),
                "order_status": activity.get("order_status")
            })
        
        # Cache for 60 seconds
        paper_trading_cache.set(CACHE_KEY_TRADES, trades, TTL_TRADES)
        
        return {
            "success": True,
            "count": len(trades),
            "trades": trades,
            "timestamp": datetime.now().isoformat(),
            "cached": False
        }
    except Exception as e:
        print(f"/paper/trades fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch trades",
            "trades": [],
            "timestamp": datetime.now().isoformat()
        }


@router.get("/portfolio-history")
def get_paper_portfolio_history(timeframe: str = "1D"):
    """
    Get portfolio history/equity curve from Alpaca.
    Cached for 2 minutes (updated frequently but not every second).
    
    Query params:
    - timeframe: '1D' (day), '1W' (week), '1M' (month), '3M', '1A' (all), 'all'
    
    Returns:
        Dict with equity curve: timestamp, equity for each data point
    """
    try:
        # Check cache first
        cached = paper_trading_cache.get(CACHE_KEY_PORTFOLIO_HISTORY)
        if cached:
            return {
                "success": True,
                "timeframe": timeframe,
                "equity_curve": cached,
                "timestamp": datetime.now().isoformat(),
                "cached": True
            }
        
        client = AlpacaPaperTradingClient()
        history = client.get_portfolio_history(timeframe=timeframe)
        
        if history:
            # Convert timestamps and build equity curve
            equity_curve = []
            
            if "equity" in history and "timestamp" in history:
                for ts, equity in zip(history.get("timestamp", []), history.get("equity", [])):
                    equity_curve.append({
                        "timestamp": datetime.fromtimestamp(ts).isoformat() if isinstance(ts, int) else str(ts),
                        "equity": equity
                    })
            
            # Cache for 2 minutes
            paper_trading_cache.set(CACHE_KEY_PORTFOLIO_HISTORY, equity_curve, TTL_PORTFOLIO_HISTORY)
            
            return {
                "success": True,
                "timeframe": timeframe,
                "equity_curve": equity_curve,
                "base_value": history.get("base_value"),
                "timestamp": datetime.now().isoformat(),
                "cached": False
            }
        else:
            return {
                "success": False,
                "error": "No portfolio history available",
                "equity_curve": [],
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"/paper/portfolio-history fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch portfolio history",
            "equity_curve": [],
            "timestamp": datetime.now().isoformat()
        }


@router.post("/start-session")
def start_paper_session(
    agent_name: str = "Agent",
    current_user: dict = Depends(get_current_user),
):
    """
    Start a new paper trading session and return run_id.

    Requires a signed-in account. Read-only ``/paper/*`` GETs remain open for
    the shared Alpaca paper demo; mutating session creation does not. Stays a
    plain (sync) ``def`` — not ``async`` — so FastAPI keeps running it in the
    threadpool rather than the event loop (#292); ``get_current_user`` is
    itself sync, so the dependency doesn't require an async route.

    Query params:
    - agent_name: Name of agent/strategy (default: "Agent")

    Returns:
        run_id for tracking this session
    """
    _ = current_user  # auth gate only; shared Alpaca account is not per-user yet
    try:
        run_id = create_paper_trading_session(agent_name)
        
        # Get current account state
        client = AlpacaPaperTradingClient()
        account = client.get_account()
        
        if account:
            initial_equity = account.get("equity", 100000)
            
            # Create run record in database. The paper flow has no separate
            # protocol session, so the run doubles as its own session.
            db.insert_run(
                run_id=run_id,
                session_id=run_id,
                agent_name=agent_name,
                mode="paper",
                start_date=datetime.now().isoformat(),
                end_date="",
                initial_equity=initial_equity
            )
            
            return {
                "success": True,
                "run_id": run_id,
                "agent_name": agent_name,
                "initial_equity": initial_equity,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": "Failed to fetch initial account state",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"/paper/start-session failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to start paper trading session",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Baseline Routes (for comparing paper trading against benchmarks)
# ============================================================================

@router.get("/baselines")
def get_paper_baselines(days: int = 31):
    """
    Get baseline equity curves from database (real historical data).
    Pre-computed from same data source as backtesting.
    Cached for 1 hour.
    
    Query params:
    - days: Ignored (uses full history from database)
    
    Returns:
        Dict with 'djia' and 'buy_and_hold' equity curves
    """
    try:
        # Check cache first
        cached = paper_trading_cache.get(CACHE_KEY_BASELINES)
        if cached:
            return {
                "success": True,
                "baselines": cached,
                "timestamp": datetime.now().isoformat(),
                "cached": True,
                "note": "Real historical data (same as backtesting)"
            }
        
        # Fetch from database
        result = get_baselines_from_db()
        
        if result.get("success"):
            baselines = result.get("baselines", {})
            # Cache for 1 hour
            paper_trading_cache.set(CACHE_KEY_BASELINES, baselines, TTL_BASELINES)
            
            return {
                "success": True,
                "baselines": baselines,
                "timestamp": datetime.now().isoformat(),
                "cached": False,
                "note": "Real historical data (same as backtesting)"
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "No baseline data available"),
                "baselines": {},
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        print(f"/paper/baselines fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch baselines",
            "baselines": {},
            "timestamp": datetime.now().isoformat()
        }
