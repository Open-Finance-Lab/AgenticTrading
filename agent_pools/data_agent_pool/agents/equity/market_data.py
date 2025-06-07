import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

INTERVAL_MAP = {
    '1d': (1, 'day'),
    '1h': (1, 'hour'),
    '1m': (1, 'minute'),
}

def load_market_data(symbol: str, start_date: str, end_date: str, lookback_days: int = 30, interval: str = '1d', force_refresh: bool = False) -> pd.DataFrame:
    """
    Load historical price data for the given stock symbol using Polygon.io.
    Downloads data from (start_date - lookback_days) to end_date.
    Returns a DataFrame indexed by datetime, with OHLCV, VWAP, trade count, dividends, splits, pre/post-market prices.
    """
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{symbol}_{start_date}_{end_date}_{interval}_vwap.csv')
    if not force_refresh and os.path.exists(cache_file):
        print(f"Loading cached OHLCV data for {symbol} from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    if POLYGON_API_KEY is None:
        raise ValueError("POLYGON_API_KEY not set in environment.")

    # Compute lookback start
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_days)
    end_dt = pd.to_datetime(end_date)
    from_date = start_dt.strftime('%Y-%m-%d')
    to_date = end_dt.strftime('%Y-%m-%d')

    # Map interval
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Unsupported interval: {interval}")
    multiplier, timespan = INTERVAL_MAP[interval]

    url = BASE_URL.format(
        symbol=symbol.upper(),
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )
    url += f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"

    print(f"Fetching Polygon data: {url}")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Polygon API error: {resp.status_code} {resp.text}")
        return pd.DataFrame()
    data = resp.json()
    if 'results' not in data:
        print(f"No results in Polygon response: {data}")
        return pd.DataFrame()
    results = data['results']
    if not results:
        print(f"No data returned from Polygon for {symbol}")
        return pd.DataFrame()

    # Build DataFrame
    df = pd.DataFrame(results)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume',
        'n': 'transactions'
    })
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # --- ENHANCEMENT: Fetch VWAP, trade count, pre/post-market, dividends, splits ---
    vwap_list = []
    trades_list = []
    pre_market_list = []
    after_market_list = []
    dividend_list = []
    split_list = []
    for date in df.index.date:
        # Open/close endpoint for VWAP, pre/post-market
        openclose_url = f"https://api.polygon.io/v1/open-close/{symbol.upper()}/{date}?adjusted=true&apiKey={POLYGON_API_KEY}"
        resp2 = requests.get(openclose_url)
        if resp2.status_code == 200:
            oc_data = resp2.json()
            vwap = oc_data.get('vwap', None)
            trades = oc_data.get('volume', None)
            pre_market = oc_data.get('preMarket', None)
            after_market = oc_data.get('afterHours', None)
        else:
            vwap = None
            trades = None
            pre_market = None
            after_market = None
        vwap_list.append(vwap)
        trades_list.append(trades)
        pre_market_list.append(pre_market)
        after_market_list.append(after_market)
        # Dividends endpoint
        dividend_url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol.upper()}&ex_dividend_date={date}&apiKey={POLYGON_API_KEY}"
        resp3 = requests.get(dividend_url)
        if resp3.status_code == 200:
            div_data = resp3.json()
            if 'results' in div_data and div_data['results']:
                dividend = div_data['results'][0].get('cash_amount', None)
            else:
                dividend = None
        else:
            dividend = None
        dividend_list.append(dividend)
        # Splits endpoint
        split_url = f"https://api.polygon.io/v3/reference/splits?ticker={symbol.upper()}&execution_date={date}&apiKey={POLYGON_API_KEY}"
        resp4 = requests.get(split_url)
        if resp4.status_code == 200:
            split_data = resp4.json()
            if 'results' in split_data and split_data['results']:
                split = split_data['results'][0].get('split_from', None)
            else:
                split = None
        else:
            split = None
        split_list.append(split)
    df['vwap'] = vwap_list
    df['trades'] = trades_list
    df['pre_market'] = pre_market_list
    df['after_market'] = after_market_list
    df['dividend'] = dividend_list
    df['split'] = split_list

    df.to_csv(cache_file)
    print(f"Cached OHLCV+VWAP+dividends+splits data for {symbol} to {cache_file}")
    return df

# def load_market_data(symbol: str, force_refresh: bool = False) -> dict:
#     """
#     Load 7-day hourly price data for the given stock symbol using yfinance.
#     Returns a dictionary with timestamps and OHLCV data.
#     """
#     # Create cache directory if it doesn't exist
#     cache_dir = 'data/cache'
#     os.makedirs(cache_dir, exist_ok=True)
    
#     # Check for cached data
#     cache_file = os.path.join(cache_dir, f'{symbol}_ohlcv.csv')
#     if not force_refresh and os.path.exists(cache_file):
#         print(f"Loading cached OHLCV data for {symbol}")
#         df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
#     else:
#         end = datetime.now()
#         start = end - timedelta(days=7)
#         df = yf.download(symbol, start=start, end=end, interval='1h', auto_adjust=False)
#         if df.empty:
#             return {}
#         # Save to cache
#         df.to_csv(cache_file)
#         print(f"Cached OHLCV data for {symbol} to {cache_file}")
    
#     # Flatten columns if MultiIndex, then lowercase
#     df.columns = [
#         col[0].lower() if isinstance(col, tuple) else col.lower()
#         for col in df.columns
#     ]
    
#     data = {
#         'symbol': symbol,
#         'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
#         'open': df['open'].tolist() if 'open' in df else [],
#         'high': df['high'].tolist() if 'high' in df else [],
#         'low': df['low'].tolist() if 'low' in df else [],
#         'close': df['close'].tolist() if 'close' in df else [],
#         'volume': df['volume'].tolist() if 'volume' in df else [],
#     }
#     return data 