import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from collections.abc import Sequence as SequenceABC

import yfinance as yf
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yfinance-server")

# Default settings
DEFAULT_SYMBOL = "AAPL"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("YFINANCE_OUTPUT_DIR", "./outputs"))

app = Server("yfinance-server")

async def fetch_stock_info(symbol: str) -> dict[str, Any]:
    """Fetch current stock information."""
    stock = yf.Ticker(symbol)
    info = stock.info
    return info
    
@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available financial resources."""
    uri = AnyUrl(f"finance://{DEFAULT_SYMBOL}/info")
    return [
        Resource(
            uri=uri,
            name=f"Current stock information for {DEFAULT_SYMBOL}",
            mimeType="application/json",
            description="Real-time stock market data"
        )
    ]

@app.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read current stock information."""
    symbol = DEFAULT_SYMBOL
    if str(uri).startswith("finance://") and str(uri).endswith("/info"):
        symbol = str(uri).split("/")[-2]
    else:
        raise ValueError(f"Unknown resource: {uri}")

    try:
        stock_data = await fetch_stock_info(symbol)
        return json.dumps(stock_data, indent=2)
    except Exception as e:
        raise RuntimeError(f"Stock API error: {str(e)}")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available financial tools."""
    return [
        Tool(
            name="get_historical_data",
            description="Get historical stock data for a symbol and optionally save it as CSV",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)",
                        "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
                    },
                    "start": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) marking the start of the range"
                    },
                    "end": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) marking the end of the range"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path where the CSV should be saved"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory where the CSV will be saved when output_path is not provided"
                    }
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_stock_metric",
            description="""Get a specific metric for a stock using yfinance field names.
            Common requests and their exact field names:
            
            Stock Price & Trading Info:
            - Current/Stock Price: currentPrice
            - Opening Price: open
            - Day's High: dayHigh
            - Day's Low: dayLow
            - Previous Close: previousClose
            - 52 Week High: fiftyTwoWeekHigh
            - 52 Week Low: fiftyTwoWeekLow
            - 50 Day Average: fiftyDayAverage
            - 200 Day Average: twoHundredDayAverage
            - Trading Volume: volume
            - Average Volume: averageVolume
            - Average Daily Volume (10 day): averageDailyVolume10Day
            - Market Cap/Capitalization: marketCap
            - Beta: beta
            - Bid Price: bid
            - Ask Price: ask
            - Bid Size: bidSize
            - Ask Size: askSize
            
            Company Information:
            - Company Name: longName
            - Short Name: shortName
            - Business Description/About/Summary: longBusinessSummary
            - Industry: industry
            - Sector: sector
            - Website: website
            - Number of Employees: fullTimeEmployees
            - Country: country
            - State: state
            - City: city
            - Address: address1
            
            Financial Metrics:
            - PE Ratio: trailingPE
            - Forward PE: forwardPE
            - Price to Book: priceToBook
            - Price to Sales: priceToSalesTrailing12Months
            - Enterprise Value: enterpriseValue
            - Enterprise to EBITDA: enterpriseToEbitda
            - Enterprise to Revenue: enterpriseToRevenue
            - Book Value: bookValue
            
            Earnings & Revenue:
            - Revenue/Total Revenue: totalRevenue
            - Revenue Growth: revenueGrowth
            - Revenue Per Share: revenuePerShare
            - EBITDA: ebitda
            - EBITDA Margins: ebitdaMargins
            - Net Income: netIncomeToCommon
            - Earnings Growth: earningsGrowth
            - Quarterly Earnings Growth: earningsQuarterlyGrowth
            - Forward EPS: forwardEps
            - Trailing EPS: trailingEps
            
            Margins & Returns:
            - Profit Margin: profitMargins
            - Operating Margin: operatingMargins
            - Gross Margins: grossMargins
            - Return on Equity/ROE: returnOnEquity
            - Return on Assets/ROA: returnOnAssets
            
            Dividends:
            - Dividend Yield: dividendYield
            - Dividend Rate: dividendRate
            - Dividend Date: lastDividendDate
            - Ex-Dividend Date: exDividendDate
            - Payout Ratio: payoutRatio
            
            Balance Sheet:
            - Total Cash: totalCash
            - Cash Per Share: totalCashPerShare
            - Total Debt: totalDebt
            - Debt to Equity: debtToEquity
            - Current Ratio: currentRatio
            - Quick Ratio: quickRatio
            
            Ownership:
            - Institutional Ownership: heldPercentInstitutions
            - Insider Ownership: heldPercentInsiders
            - Float Shares: floatShares
            - Shares Outstanding: sharesOutstanding
            - Short Ratio: shortRatio
            
            Analyst Coverage:
            - Analyst Recommendation: recommendationKey
            - Number of Analysts: numberOfAnalystOpinions
            - Price Target Mean: targetMeanPrice
            - Price Target High: targetHighPrice
            - Price Target Low: targetLowPrice
            - Price Target Median: targetMedianPrice
            
            Risk Metrics:
            - Overall Risk: overallRisk
            - Audit Risk: auditRisk
            - Board Risk: boardRisk
            - Compensation Risk: compensationRisk
            
            Other:
            - Currency: currency
            - Exchange: exchange
            - Year Change/52 Week Change: 52WeekChange
            - S&P 500 Year Change: SandP52WeekChange""",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "metric": {
                        "type": "string",
                        "description": "The metric to retrieve, use camelCase"
                    }
                },
                "required": ["symbol", "metric"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls."""
    if name == "get_stock_metric":
        symbol = arguments["symbol"]
        metric = arguments["metric"]
        
        try:
            stock_data = await fetch_stock_info(symbol)
            if metric in stock_data:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({metric: stock_data[metric]}, indent=2)
                    )
                ]
            else:
                raise ValueError(f"Metric {metric} not found")
        except Exception as e:
            logger.error(f"Stock API error: {str(e)}")
            raise RuntimeError(f"Stock API error: {str(e)}")
            
    elif name == "get_historical_data":

        if not isinstance(arguments, dict) or "symbol" not in arguments:
            raise ValueError("Invalid arguments")

        symbol = arguments["symbol"]
        period = arguments.get("period")
        start = arguments.get("start")
        end = arguments.get("end")
        output_path_arg = arguments.get("output_path")
        output_dir_arg = arguments.get("output_dir")

        try:
            stock = yf.Ticker(symbol)
            history_kwargs: dict[str, Any] = {}
            if start or end:
                history_kwargs["start"] = start
                history_kwargs["end"] = end
            else:
                history_kwargs["period"] = period or "1mo"

            history = stock.history(**history_kwargs)

            if history.empty:
                raise ValueError("No historical data returned for the specified parameters")

            history_df = history.reset_index()
            date_series = history_df.iloc[:, 0]
            if hasattr(date_series, "dt"):
                history_df["date"] = date_series.dt.strftime("%Y-%m-%d")
            else:
                history_df["date"] = date_series.astype(str)

            history_df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            expected_columns = ["date", "open", "high", "low", "close", "volume"]
            history_df = history_df[[col for col in expected_columns if col in history_df.columns]]
            data = history_df.to_dict(orient="records")

            if output_path_arg:
                csv_path = Path(output_path_arg).expanduser()
            else:
                base_dir = Path(output_dir_arg).expanduser() if output_dir_arg else DEFAULT_OUTPUT_DIR
                base_dir.mkdir(parents=True, exist_ok=True)
                if start or end:
                    start_fragment = start.replace("-", "") if start else "start"
                    end_fragment = end.replace("-", "") if end else "end"
                    range_fragment = f"{start_fragment}_{end_fragment}"
                else:
                    range_fragment = (period or "1mo").replace(" ", "")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"{symbol.lower()}_{range_fragment}_{timestamp}.csv"
                csv_path = base_dir / csv_filename

            csv_path = csv_path.expanduser()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            history_df.to_csv(csv_path, index=False)
            
            response_payload = {
                "symbol": symbol,
                "csv_path": str(csv_path.resolve()),
                "row_count": len(data),
                "columns": [col for col in expected_columns if col in history_df.columns],
                "range": {
                    "period": history_kwargs.get("period"),
                    "start": start,
                    "end": end
                },
                "data": data
            }

            return [
                TextContent(
                    type="text",
                    text=json.dumps(response_payload, indent=2)
                )
            ]
        except Exception as e:
            logger.error(f"Stock API error: {str(e)}")
            raise RuntimeError(f"Stock API error: {str(e)}")

async def main():
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())