"""
MCP Client Example: Batch fetch 3 years of OHLCV data for 30 stocks using polygon agent via DataAgentPool MCP server.

Updated architecture (WORKING):
- PolygonAgent runs as independent MCP server (configured from polygon.yaml) on port 8003 with SSE transport
- DataAgentPool runs as coordinator MCP server on port 8001 with SSE transport  
- DataAgentPool proxies requests to PolygonAgent via proper MCP SSE client
- Natural language interface is the primary method for data queries
- Uses proper MCP client libraries instead of HTTP requests
- PolygonAgent API configuration loaded from config/polygon.yaml file

Tools available on DataAgentPool (port 8001):
- process_market_query: Natural language market data queries (proxied to PolygonAgent)
- fetch_market_data: Direct market data fetching (proxied to PolygonAgent)  
- get_company_info: Company information lookup (proxied to PolygonAgent)
- batch_fetch_market_data: Batch processing for multiple symbols
- list_agents: List configured agents and endpoints
- health_check: Health status of pool and connected agents

Tools available on PolygonAgent (configured from polygon.yaml):
- process_market_query: Natural language market data queries
- fetch_market_data: Direct market data fetching
- get_company_info: Company information lookup
- health_check: Agent health status
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

# Updated to use proper MCP SSE clients
DATA_AGENT_POOL_URL = "http://localhost:8001/sse"  # DataAgentPool coordinator
POLYGON_DIRECT_URL = "http://localhost:8003/sse"   # PolygonAgent direct access (configured from polygon.yaml)

# List of 30 stock symbols (example)
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
    "HD", "MA", "PG", "LLY", "XOM", "MRK", "ABBV", "AVGO", "COST", "PEP",
    "KO", "ADBE", "CSCO", "WMT", "BAC", "MCD", "DIS", "CRM", "ACN", "TMO"
]

# Date range: last 3 years
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=3*365)

# Date range: last 3 years  
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=3*365)

# Stock symbols for testing
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH",
    "HD", "MA", "PG", "LLY", "XOM", "MRK", "ABBV", "AVGO", "COST", "PEP",
    "KO", "ADBE", "CSCO", "WMT", "BAC", "MCD", "DIS", "CRM", "ACN", "TMO"
]


async def fetch_via_data_agent_pool(symbol, method="process_market_query"):
    """
    Fetch market data via DataAgentPool coordinator using proper MCP client.
    
    Args:
        symbol: Stock symbol to fetch
        method: Tool to use ("process_market_query" or "fetch_market_data")
    """
    try:
        async with sse_client(DATA_AGENT_POOL_URL, timeout=30) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                if method == "process_market_query":
                    # Use natural language query
                    query = f"Get daily price data for {symbol} from {START_DATE} to {END_DATE}"
                    result = await session.call_tool("process_market_query", {"query": query})
                else:
                    # Use direct fetch
                    result = await session.call_tool("fetch_market_data", {
                        "symbol": symbol,
                        "start": str(START_DATE),
                        "end": str(END_DATE),
                        "interval": "1d"
                    })
                
                # Parse result
                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        data = json.loads(content_item.text)
                        return data
                
                return {"status": "error", "error": "No content in response"}
                
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def batch_fetch_via_data_agent_pool(symbols):
    """
    Use the batch_fetch_market_data tool on DataAgentPool.
    """
    try:
        async with sse_client(DATA_AGENT_POOL_URL, timeout=60) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool("batch_fetch_market_data", {
                    "symbols": symbols,
                    "start": str(START_DATE),
                    "end": str(END_DATE),
                    "interval": "1d"
                })
                
                # Parse result
                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        data = json.loads(content_item.text)
                        return data
                
                return {"status": "error", "error": "No content in response"}
                
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def fetch_via_polygon_direct(symbol):
    """
    Fetch market data directly from PolygonAgent (bypass coordinator).
    """
    try:
        async with sse_client(POLYGON_DIRECT_URL, timeout=30) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Use natural language query
                query = f"Get daily price data for {symbol} from {START_DATE} to {END_DATE}"
                result = await session.call_tool("process_market_query", {"query": query})
                
                # Parse result
                if result.content and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, 'text'):
                        data = json.loads(content_item.text)
                        return data
                
                return {"status": "error", "error": "No content in response"}
                
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def test_connectivity():
    """Test connectivity to both MCP servers."""
    print("=== Testing MCP Server Connectivity ===")
    
    # Test DataAgentPool
    try:
        async with sse_client(DATA_AGENT_POOL_URL, timeout=10) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("health_check", {})
                print("✅ DataAgentPool: Connected")
                if result.content:
                    health_data = json.loads(result.content[0].text)
                    print(f"   Pool status: {health_data.get('pool_status', 'unknown')}")
    except Exception as e:
        print(f"❌ DataAgentPool: Failed - {e}")
    
    # Test PolygonAgent
    try:
        async with sse_client(POLYGON_DIRECT_URL, timeout=10) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("health_check", {})
                print("✅ PolygonAgent: Connected")
                if result.content:
                    health_data = json.loads(result.content[0].text)
                    print(f"   Agent status: {health_data.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ PolygonAgent: Failed - {e}")


async def main():
    """Main execution function."""
    print("Starting batch fetch using proper MCP client architecture...")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"DataAgentPool: {DATA_AGENT_POOL_URL}")
    print(f"PolygonAgent Direct: {POLYGON_DIRECT_URL}")
    
    # Test connectivity first
    await test_connectivity()
    
    # Option 1: Test batch fetch (recommended)
    print("\n=== Option 1: Batch fetch via DataAgentPool ===")
    batch_symbols = STOCK_SYMBOLS[:5]  # Test with first 5 symbols
    batch_result = await batch_fetch_via_data_agent_pool(batch_symbols)
    
    if batch_result.get("status") == "success":
        batch_results = batch_result.get("batch_results", [])
        all_rows = []
        
        for item in batch_results:
            symbol = item["symbol"]
            result_data = item["result"]
            
            if result_data.get("status") == "success":
                # Extract data depending on structure
                if "result" in result_data:
                    nested_result = result_data["result"]
                    if isinstance(nested_result, dict) and "data" in nested_result:
                        rows = nested_result["data"]
                    elif isinstance(nested_result, list):
                        rows = nested_result
                    else:
                        print(f"  Unexpected result structure for {symbol}")
                        continue
                elif "data" in result_data:
                    rows = result_data["data"]
                else:
                    print(f"  No data found for {symbol}")
                    continue
                
                # Add symbol to each row and collect
                for row in rows:
                    if isinstance(row, dict):
                        row["symbol"] = symbol
                all_rows.extend(rows)
                print(f"  ✅ {symbol}: {len(rows)} data points")
            else:
                print(f"  ❌ {symbol}: {result_data.get('error', 'Unknown error')}")
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            filename = "polygon_batch_mcp_coordinated.csv"
            df.to_csv(filename, index=False)
            print(f"\n🎉 Batch success! Saved {len(all_rows)} data points to {filename}")
            print(f"   Data for {len(df['symbol'].unique())} unique symbols")
    else:
        print(f"❌ Batch fetch failed: {batch_result.get('error', 'Unknown error')}")
    
    # Option 2: Individual fetches via coordinator
    print("\n=== Option 2: Individual fetches via DataAgentPool ===")
    all_rows = []
    test_symbols = STOCK_SYMBOLS[:3]  # Test with first 3 symbols
    
    for i, symbol in enumerate(test_symbols, 1):
        print(f"[{i}/{len(test_symbols)}] Fetching {symbol}...")
        
        # Try direct fetch first, then natural language
        result = await fetch_via_data_agent_pool(symbol, "fetch_market_data")
        if result.get("status") != "success":
            print(f"  Direct fetch failed, trying natural language...")
            result = await fetch_via_data_agent_pool(symbol, "process_market_query")
        
        if result.get("status") == "success":
            # Extract data
            if "data" in result:
                rows = result["data"]
            elif "result" in result and isinstance(result["result"], dict):
                nested = result["result"]
                if "data" in nested:
                    rows = nested["data"]
                else:
                    rows = nested if isinstance(nested, list) else []
            else:
                rows = []
            
            if rows:
                for row in rows:
                    if isinstance(row, dict):
                        row["symbol"] = symbol
                all_rows.extend(rows)
                print(f"  ✅ {symbol}: {len(rows)} data points")
            else:
                print(f"  ⚠️ {symbol}: No data in response")
        else:
            print(f"  ❌ {symbol}: {result.get('error', 'Unknown error')}")
        
        await asyncio.sleep(0.5)  # Rate limiting
    
    if all_rows:
        df = pd.DataFrame(all_rows)
        filename = "polygon_individual_mcp_coordinated.csv"
        df.to_csv(filename, index=False)
        print(f"\n🎉 Individual fetch success! Saved {len(all_rows)} data points to {filename}")
        print(f"   Data for {len(df['symbol'].unique())} unique symbols")
    
    # Option 3: Direct PolygonAgent access (bypass coordinator)
    print("\n=== Option 3: Direct PolygonAgent access ===")
    test_symbol = "AAPL"
    print(f"Testing direct fetch for {test_symbol}...")
    
    result = await fetch_via_polygon_direct(test_symbol)
    if result.get("status") == "success":
        print(f"✅ Direct access successful!")
        if "result" in result:
            data = result["result"]
            if isinstance(data, dict) and "data" in data:
                print(f"   Got {len(data['data'])} data points")
            elif isinstance(data, list):
                print(f"   Got {len(data)} data points")
    else:
        print(f"❌ Direct access failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
