"""Live market-data ticker route (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. The external path ``/ticker``
and its response schema are unchanged; registered directly on the app.
"""

from datetime import datetime

from fastapi import APIRouter

from dashboard.backend.infrastructure.market_data.quotes import get_market_quotes

router = APIRouter()

# /ticker is session-exempt, so ``symbols`` is unauthenticated input that both
# fans out into one provider call per symbol and becomes a quote-cache key. Cap
# the fan-out per request; quotes.py caps the number of distinct keys retained.
MAX_TICKER_SYMBOLS = 25


@router.get("/ticker")
def get_ticker(symbols: str = "AAPL,NVDA,MSFT,BTC"):
    """
    Get live market quotes for symbols.
    
    Query params:
    - symbols: comma-separated list of symbols (default: AAPL,NVDA,MSFT,BTC)
    
    Returns:
        List of quotes with symbol, price, change%, timestamp
    """
    # dict.fromkeys dedupes while preserving order, so "AAPL,AAPL" is one fetch
    # and one cache key rather than two.
    symbol_list = list(dict.fromkeys(s.strip().upper() for s in symbols.split(',') if s.strip()))

    if not symbol_list:
        return {"error": "No symbols provided", "quotes": []}

    if len(symbol_list) > MAX_TICKER_SYMBOLS:
        return {
            "error": f"Too many symbols (max {MAX_TICKER_SYMBOLS})",
            "quotes": [],
        }


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
