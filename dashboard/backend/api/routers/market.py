"""Live market-data ticker route (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. The external path ``/ticker``
and its response schema are unchanged; registered directly on the app.
"""

from datetime import datetime

from fastapi import APIRouter

from dashboard.backend.infrastructure.market_data.quotes import get_market_quotes

router = APIRouter()


@router.get("/ticker")
def get_ticker(symbols: str = "AAPL,NVDA,MSFT,BTC"):
    """
    Get live market quotes for symbols.
    
    Query params:
    - symbols: comma-separated list of symbols (default: AAPL,NVDA,MSFT,BTC)
    
    Returns:
        List of quotes with symbol, price, change%, timestamp
    """
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        return {"error": "No symbols provided", "quotes": []}
    
    try:
        quotes = get_market_quotes(symbol_list)
        return {
            "success": True,
            "count": len(quotes),
            "quotes": quotes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"/ticker quote fetch failed: {e!r}")
        return {
            "success": False,
            "error": "Failed to fetch market quotes",
            "quotes": []
        }
