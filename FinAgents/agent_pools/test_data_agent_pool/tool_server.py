from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd

mcp = FastMCP("DataTools")

@mcp.tool()
def ingest_ticker_data(ticker: str, period: str = "5d", interval: str = "1d") -> list[dict]:
    """
    Download historical price data for 'ticker' using yfinance.
    Returns a list of dicts (one per row) with Date, Open, High, Low, Close, Volume, etc.
    """
    if not ticker or not ticker.isalnum():
        raise ValueError("Invalid ticker symbol. Please provide an alphanumeric ticker (e.g., AAPL).")

    try:
        data = yf.download(ticker, period=period, interval=interval)
    except Exception as e:
        raise RuntimeError(f"Error downloading data for {ticker} with period={period}, interval={interval}: {e}")

    if data.empty:
        raise ValueError(f"No data returned for {ticker} with period={period} and interval={interval}.")

    return data.reset_index().to_dict(orient="records")

@mcp.tool()
def validate_data(data: list[dict]) -> dict:
    """
    Check for nulls in each column and return { 'nulls': {...}, 'is_valid': True/False }.
    """
    df = pd.DataFrame(data)

    if df.empty:
        return {"nulls": {}, "is_valid": False}

    null_counts = df.isnull().sum().to_dict()
    is_valid = df.notnull().all().all()

    return {"nulls": null_counts, "is_valid": is_valid}

@mcp.tool()
def transform_data(data: list[dict]) -> str:
    """
    Format all rows as a Markdown table (no return computation).
    """
    df = pd.DataFrame(data)

    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(missing)}. Cannot format data.")

    return df.to_markdown(index=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")
