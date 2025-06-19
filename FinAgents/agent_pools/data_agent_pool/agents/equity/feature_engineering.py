import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

def compute_features(df, symbol: str = None, force_refresh: bool = False) -> dict:
    """
    Compute features for a given ticker DataFrame.
    Returns a dict with RSI, MACD, z-score, Bollinger Bands, avg volume ratio, typical price, dollar volume, prev day's data, dividends, splits, pre/post-market prices if available.
    """
    if symbol is not None:
        # Create cache directory if it doesn't exist
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check for cached features
        cache_file = os.path.join(cache_dir, f'{symbol}_features.json')
        if not force_refresh and os.path.exists(cache_file):
            print(f"Loading cached features for {symbol}")
            with open(cache_file, 'r') as f:
                return json.load(f)
    
    features = {}
    # Use lower-case column names for compatibility
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    open_ = df['open']
    # Typical price
    typical_price = (high + low + close) / 3
    features['typical_price'] = float(typical_price.iloc[-1]) if not typical_price.empty else np.nan
    # Dollar volume
    dollar_volume = close * volume
    features['dollar_volume'] = float(dollar_volume.iloc[-1]) if not dollar_volume.empty else np.nan
    # Previous day's close, high, low
    features['prev_close'] = float(close.shift(1).iloc[-1]) if len(close) > 1 else np.nan
    features['prev_high'] = float(high.shift(1).iloc[-1]) if len(high) > 1 else np.nan
    features['prev_low'] = float(low.shift(1).iloc[-1]) if len(low) > 1 else np.nan
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    features['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else np.nan
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    features['macd'] = float(macd.iloc[-1]) if not macd.empty else np.nan
    features['macd_signal'] = float(signal.iloc[-1]) if not signal.empty else np.nan
    # z-score of returns
    returns = close.pct_change().dropna()
    zscore = (returns.iloc[-1] - returns.mean()) / (returns.std() + 1e-9) if not returns.empty else np.nan
    features['zscore'] = float(zscore)
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_upper'] = float((ma20 + 2 * std20).iloc[-1]) if not ma20.empty else np.nan
    features['bb_lower'] = float((ma20 - 2 * std20).iloc[-1]) if not ma20.empty else np.nan
    features['bb_ma'] = float(ma20.iloc[-1]) if not ma20.empty else np.nan
    # Avg volume ratio
    avg_vol_5 = volume.rolling(5).mean().iloc[-1] if len(volume) >= 5 else np.nan
    avg_vol_30 = volume.rolling(30).mean().iloc[-1] if len(volume) >= 30 else np.nan
    try:
        avg_vol_5_val = float(avg_vol_5)
        avg_vol_30_val = float(avg_vol_30)
    except (ValueError, TypeError):
        avg_vol_5_val = avg_vol_30_val = np.nan
    if not np.isnan(avg_vol_5_val) and not np.isnan(avg_vol_30_val):
        features['avg_vol_ratio'] = avg_vol_5_val / (avg_vol_30_val + 1e-9)
    else:
        features['avg_vol_ratio'] = np.nan
    # VWAP and trades (from enhanced loader)
    if 'vwap' in df.columns:
        features['vwap'] = float(df['vwap'].iloc[-1]) if not df['vwap'].empty else np.nan
    if 'trades' in df.columns:
        features['trades'] = float(df['trades'].iloc[-1]) if not df['trades'].empty else np.nan
    # Pre/post-market prices (from open-close endpoint, if merged in loader)
    if 'pre_market' in df.columns:
        features['pre_market'] = float(df['pre_market'].iloc[-1]) if not df['pre_market'].empty else np.nan
    if 'after_market' in df.columns:
        features['after_market'] = float(df['after_market'].iloc[-1]) if not df['after_market'].empty else np.nan
    # Dividends and splits (if merged in loader)
    if 'dividend' in df.columns:
        features['dividend'] = float(df['dividend'].iloc[-1]) if not df['dividend'].empty else np.nan
    if 'split' in df.columns:
        features['split'] = float(df['split'].iloc[-1]) if not df['split'].empty else np.nan
    # Cache features if symbol provided
    if symbol is not None:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
        print(f"Cached features for {symbol} to {cache_file}")
    return features 