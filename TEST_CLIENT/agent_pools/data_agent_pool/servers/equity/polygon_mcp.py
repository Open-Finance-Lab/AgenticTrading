#
# add attribution from https://github.com/polygon-io/mcp_polygon/tree/master
#

import os
import json
from typing import Optional, Any, Dict, Union, List
from mcp.server.fastmcp import FastMCP
from polygon import RESTClient
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
polygon_client = RESTClient(POLYGON_API_KEY)

mcp = FastMCP("Polygon", dependencies=["polygon"])

@mcp.tool()
async def get_aggs(
        ticker: str,
        multiplier: int,
        timespan: str,
        from_: Union[str, int, datetime, date],
        to: Union[str, int, datetime, date],
        adjusted: Optional[bool] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    List aggregate bars for a ticker over a given date range in custom time window sizes.
    """
    try:
        results = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_,
            to=to,
            adjusted=adjusted,
            sort=sort,
            limit=limit,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_aggs(
        ticker: str,
        multiplier: int,
        timespan: str,
        from_: Union[str, int, datetime, date],
        to: Union[str, int, datetime, date],
        adjusted: Optional[bool] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Iterate through aggregate bars for a ticker over a given date range.
    """
    try:
        results = polygon_client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_,
            to=to,
            adjusted=adjusted,
            sort=sort,
            limit=limit,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_grouped_daily_aggs(
        date: str,
        adjusted: Optional[bool] = None,
        include_otc: Optional[bool] = None,
        locale: Optional[str] = None,
        market_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get grouped daily bars for entire market for a specific date.
    """
    try:
        results = polygon_client.get_grouped_daily_aggs(
            date=date,
            adjusted=adjusted,
            include_otc=include_otc,
            locale=locale,
            market_type=market_type,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_daily_open_close_agg(
        ticker: str,
        date: str,
        adjusted: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get daily open, close, high, and low for a specific ticker and date.
    """
    try:
        results = polygon_client.get_daily_open_close_agg(
            ticker=ticker,
            date=date,
            adjusted=adjusted,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_previous_close_agg(
        ticker: str,
        adjusted: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get previous day's open, close, high, and low for a specific ticker.
    """
    try:
        results = polygon_client.get_previous_close_agg(
            ticker=ticker,
            adjusted=adjusted,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_trades(
        ticker: str,
        timestamp: Optional[Union[str, int, datetime, date]] = None,
        timestamp_lt: Optional[Union[str, int, datetime, date]] = None,
        timestamp_lte: Optional[Union[str, int, datetime, date]] = None,
        timestamp_gt: Optional[Union[str, int, datetime, date]] = None,
        timestamp_gte: Optional[Union[str, int, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get trades for a ticker symbol.
    """
    try:
        results = polygon_client.list_trades(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_last_trade(
        ticker: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get the most recent trade for a ticker symbol.
    """
    try:
        results = polygon_client.get_last_trade(
            ticker=ticker,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_last_crypto_trade(
        from_: str,
        to: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get the most recent trade for a crypto pair.
    """
    try:
        results = polygon_client.get_last_crypto_trade(
            from_=from_,
            to=to,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_quotes(
        ticker: str,
        timestamp: Optional[Union[str, int, datetime, date]] = None,
        timestamp_lt: Optional[Union[str, int, datetime, date]] = None,
        timestamp_lte: Optional[Union[str, int, datetime, date]] = None,
        timestamp_gt: Optional[Union[str, int, datetime, date]] = None,
        timestamp_gte: Optional[Union[str, int, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get quotes for a ticker symbol.
    """
    try:
        results = polygon_client.list_quotes(
            ticker=ticker,
            timestamp=timestamp,
            timestamp_lt=timestamp_lt,
            timestamp_lte=timestamp_lte,
            timestamp_gt=timestamp_gt,
            timestamp_gte=timestamp_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_last_quote(
        ticker: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get the most recent quote for a ticker symbol.
    """
    try:
        results = polygon_client.get_last_quote(
            ticker=ticker,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_last_forex_quote(
        from_: str,
        to: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get the most recent forex quote.
    """
    try:
        results = polygon_client.get_last_forex_quote(
            from_=from_,
            to=to,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_real_time_currency_conversion(
        from_: str,
        to: str,
        amount: Optional[float] = None,
        precision: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get real-time currency conversion.
    """
    try:
        results = polygon_client.get_real_time_currency_conversion(
            from_=from_,
            to=to,
            amount=amount,
            precision=precision,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_universal_snapshots(
        type: str,
        ticker_any_of: Optional[List[str]] = None,
        order: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get universal snapshots for multiple assets of a specific type.
    """
    try:
        results = polygon_client.list_universal_snapshots(
            type=type,
            ticker_any_of=ticker_any_of,
            order=order,
            limit=limit,
            sort=sort,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_snapshot_all(
        market_type: str,
        tickers: Optional[List[str]] = None,
        include_otc: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get a snapshot of all tickers in a market.
    """
    try:
        results = polygon_client.get_snapshot_all(
            market_type=market_type,
            tickers=tickers,
            include_otc=include_otc,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_snapshot_direction(
        market_type: str,
        direction: str,
        include_otc: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get gainers or losers for a market.
    """
    try:
        results = polygon_client.get_snapshot_direction(
            market_type=market_type,
            direction=direction,
            include_otc=include_otc,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_snapshot_ticker(
        market_type: str,
        ticker: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get snapshot for a specific ticker.
    """
    try:
        results = polygon_client.get_snapshot_ticker(
            market_type=market_type,
            ticker=ticker,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_snapshot_option(
        underlying_asset: str,
        option_contract: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get snapshot for a specific option contract.
    """
    try:
        results = polygon_client.get_snapshot_option(
            underlying_asset=underlying_asset,
            option_contract=option_contract,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_snapshot_crypto_book(
        ticker: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get snapshot for a crypto ticker's order book.
    """
    try:
        results = polygon_client.get_snapshot_crypto_book(
            ticker=ticker,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_market_holidays(
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get upcoming market holidays and their open/close times.
    """
    try:
        results = polygon_client.get_market_holidays(
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_market_status(
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get current trading status of exchanges and financial markets.
    """
    try:
        results = polygon_client.get_market_status(
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_tickers(
        ticker: Optional[str] = None,
        type: Optional[str] = None,
        market: Optional[str] = None,
        exchange: Optional[str] = None,
        cusip: Optional[str] = None,
        cik: Optional[str] = None,
        date: Optional[Union[str, datetime, date]] = None,
        search: Optional[str] = None,
        active: Optional[bool] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Query supported ticker symbols across stocks, indices, forex, and crypto.
    """
    try:
        results = polygon_client.list_tickers(
            ticker=ticker,
            type=type,
            market=market,
            exchange=exchange,
            cusip=cusip,
            cik=cik,
            date=date,
            search=search,
            active=active,
            sort=sort,
            order=order,
            limit=limit,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_ticker_details(
        ticker: str,
        date: Optional[Union[str, datetime, date]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get detailed information about a specific ticker.
    """
    try:
        results = polygon_client.get_ticker_details(
            ticker=ticker,
            date=date,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_ticker_news(
        ticker: Optional[str] = None,
        published_utc: Optional[Union[str, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get recent news articles for a stock ticker.
    """
    try:
        results = polygon_client.list_ticker_news(
            ticker=ticker,
            published_utc=published_utc,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_ticker_types(
        asset_class: Optional[str] = None,
        locale: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    List all ticker types supported by Polygon.io.
    """
    try:
        results = polygon_client.get_ticker_types(
            asset_class=asset_class,
            locale=locale,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_splits(
        ticker: Optional[str] = None,
        execution_date: Optional[Union[str, datetime, date]] = None,
        reverse_split: Optional[bool] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get historical stock splits.
    """
    try:
        results = polygon_client.list_splits(
            ticker=ticker,
            execution_date=execution_date,
            reverse_split=reverse_split,
            limit=limit,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_dividends(
        ticker: Optional[str] = None,
        ex_dividend_date: Optional[Union[str, datetime, date]] = None,
        frequency: Optional[int] = None,
        dividend_type: Optional[str] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get historical cash dividends.
    """
    try:
        results = polygon_client.list_dividends(
            ticker=ticker,
            ex_dividend_date=ex_dividend_date,
            frequency=frequency,
            dividend_type=dividend_type,
            limit=limit,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_conditions(
        asset_class: Optional[str] = None,
        data_type: Optional[str] = None,
        id: Optional[int] = None,
        sip: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    List conditions used by Polygon.io.
    """
    try:
        results = polygon_client.list_conditions(
            asset_class=asset_class,
            data_type=data_type,
            id=id,
            sip=sip,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_exchanges(
        asset_class: Optional[str] = None,
        locale: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    List exchanges known by Polygon.io.
    """
    try:
        results = polygon_client.get_exchanges(
            asset_class=asset_class,
            locale=locale,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_stock_financials(
        ticker: Optional[str] = None,
        cik: Optional[str] = None,
        company_name: Optional[str] = None,
        company_name_search: Optional[str] = None,
        sic: Optional[str] = None,
        filing_date: Optional[Union[str, datetime, date]] = None,
        filing_date_lt: Optional[Union[str, datetime, date]] = None,
        filing_date_lte: Optional[Union[str, datetime, date]] = None,
        filing_date_gt: Optional[Union[str, datetime, date]] = None,
        filing_date_gte: Optional[Union[str, datetime, date]] = None,
        period_of_report_date: Optional[Union[str, datetime, date]] = None,
        period_of_report_date_lt: Optional[Union[str, datetime, date]] = None,
        period_of_report_date_lte: Optional[Union[str, datetime, date]] = None,
        period_of_report_date_gt: Optional[Union[str, datetime, date]] = None,
        period_of_report_date_gte: Optional[Union[str, datetime, date]] = None,
        timeframe: Optional[str] = None,
        include_sources: Optional[bool] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Get fundamental financial data for companies.
    """
    try:
        results = polygon_client.vx.list_stock_financials(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            company_name_search=company_name_search,
            sic=sic,
            filing_date=filing_date,
            filing_date_lt=filing_date_lt,
            filing_date_lte=filing_date_lte,
            filing_date_gt=filing_date_gt,
            filing_date_gte=filing_date_gte,
            period_of_report_date=period_of_report_date,
            period_of_report_date_lt=period_of_report_date_lt,
            period_of_report_date_lte=period_of_report_date_lte,
            period_of_report_date_gt=period_of_report_date_gt,
            period_of_report_date_gte=period_of_report_date_gte,
            timeframe=timeframe,
            include_sources=include_sources,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_ipos(
        ticker: Optional[str] = None,
        listing_date: Optional[Union[str, datetime, date]] = None,
        listing_date_lt: Optional[Union[str, datetime, date]] = None,
        listing_date_lte: Optional[Union[str, datetime, date]] = None,
        listing_date_gt: Optional[Union[str, datetime, date]] = None,
        listing_date_gte: Optional[Union[str, datetime, date]] = None,
        ipo_status: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Retrieve upcoming or historical IPOs.
    """
    try:
        results = polygon_client.vx.list_ipos(
            ticker=ticker,
            listing_date=listing_date,
            listing_date_lt=listing_date_lt,
            listing_date_lte=listing_date_lte,
            listing_date_gt=listing_date_gt,
            listing_date_gte=listing_date_gte,
            ipo_status=ipo_status,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_short_interest(
        ticker: Optional[str] = None,
        settlement_date: Optional[Union[str, datetime, date]] = None,
        settlement_date_lt: Optional[Union[str, datetime, date]] = None,
        settlement_date_lte: Optional[Union[str, datetime, date]] = None,
        settlement_date_gt: Optional[Union[str, datetime, date]] = None,
        settlement_date_gte: Optional[Union[str, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Retrieve short interest data for stocks.
    """
    try:
        results = polygon_client.vx.list_short_interest(
            ticker=ticker,
            settlement_date=settlement_date,
            settlement_date_lt=settlement_date_lt,
            settlement_date_lte=settlement_date_lte,
            settlement_date_gt=settlement_date_gt,
            settlement_date_gte=settlement_date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_short_volume(
        ticker: Optional[str] = None,
        date: Optional[Union[str, datetime, date]] = None,
        date_lt: Optional[Union[str, datetime, date]] = None,
        date_lte: Optional[Union[str, datetime, date]] = None,
        date_gt: Optional[Union[str, datetime, date]] = None,
        date_gte: Optional[Union[str, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Retrieve short volume data for stocks.
    """
    try:
        results = polygon_client.vx.list_short_volume(
            ticker=ticker,
            date=date,
            date_lt=date_lt,
            date_lte=date_lte,
            date_gt=date_gt,
            date_gte=date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_treasury_yields(
        date: Optional[Union[str, datetime, date]] = None,
        date_lt: Optional[Union[str, datetime, date]] = None,
        date_lte: Optional[Union[str, datetime, date]] = None,
        date_gt: Optional[Union[str, datetime, date]] = None,
        date_gte: Optional[Union[str, datetime, date]] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
        order: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
    """
    Retrieve treasury yield data.
    """
    try:
        results = polygon_client.vx.list_treasury_yields(
            date=date,
            date_lt=date_lt,
            date_lte=date_lte,
            date_gt=date_gt,
            date_gte=date_gte,
            limit=limit,
            sort=sort,
            order=order,
            params=params,
            raw=True
        )
        
        data_str = results.data.decode('utf-8')
        return json.loads(data_str)
    except Exception as e:
        return {"error": str(e)}

def run(transport):
    """Run the Polygon MCP server."""
    mcp.run(transport)