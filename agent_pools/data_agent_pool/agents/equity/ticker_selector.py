import yfinance as yf
import pandas as pd
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback tickers if no qualified tickers found
FALLBACK_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

def get_sp500_tickers():
    """
    Fetch S&P 500 tickers from Wikipedia using pandas.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].tolist()
    # Some tickers have dots (e.g., BRK.B), yfinance expects dashes (BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

def select_top_tickers(n=5, force_refresh=False):
    """
    Select top n tickers from S&P 500 based on avg volume > 500k, price > $1, and highest volatility.
    Returns a list of ticker symbols.
    """
    # Create cache directory if it doesn't exist
    cache_dir = 'data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check for cached results
    today = datetime.now().strftime('%Y%m%d')
    cache_file = os.path.join(cache_dir, f'selected_tickers_{today}.json')
    
    if not force_refresh and os.path.exists(cache_file):
        logger.info(f"Loading cached tickers from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_tickers = json.load(f)
            if cached_tickers:  # Only use cache if it's not empty
                return cached_tickers
            logger.warning("Cached tickers list is empty, regenerating...")
    
    logger.info("Fetching and filtering S&P 500 tickers...")
    tickers = get_sp500_tickers()
    stats = []
    skipped_empty = 0
    skipped_volume = 0
    skipped_price = 0
    passed_empty = 0
    passed_volume = 0
    passed_price = 0
    
    for symbol in tqdm(tickers, desc='Filtering S&P 500'):
        try:
            df = yf.download(symbol, period='3mo', interval='1d', progress=False)
            if df.empty or len(df) < 20:
                skipped_empty += 1
                continue
            passed_empty += 1
            
            avg_vol = df['Volume'].mean()
            last_price = df['Close'].iloc[-1]
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std()
            
            if avg_vol <= 500_000:
                skipped_volume += 1
                continue
            passed_volume += 1
            
            if last_price <= 1.0:
                skipped_price += 1
                continue
            passed_price += 1
            
            stats.append({
                'symbol': symbol,
                'volatility': volatility,
                'avg_vol': avg_vol,
                'price': last_price
            })
        except Exception as e:
            logger.warning(f"Error processing {symbol}: {str(e)}")
            continue
    
    logger.info(f"\nFiltering results:")
    logger.info(f"- Total tickers processed: {len(tickers)}")
    logger.info(f"- Passed (not empty): {passed_empty}")
    logger.info(f"- Passed (volume > 500k): {passed_volume}")
    logger.info(f"- Passed (price > $1): {passed_price}")
    logger.info(f"- Skipped (empty/incomplete data): {skipped_empty}")
    logger.info(f"- Skipped (low volume): {skipped_volume}")
    logger.info(f"- Skipped (low price): {skipped_price}")
    logger.info(f"- Qualified tickers: {len(stats)}")
    
    # Sort by volatility descending
    stats = sorted(stats, key=lambda x: x['volatility'], reverse=True)
    selected = [x['symbol'] for x in stats[:n]]
    
    # If no tickers qualified, use fallback
    if not selected:
        logger.warning("No tickers qualified after filtering. Using fallback tickers.")
        selected = FALLBACK_TICKERS[:n]
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(selected, f)
    logger.info(f"\nSelected top {n} tickers: {selected}")
    logger.info(f"Results cached to {cache_file}")
    
    return selected 