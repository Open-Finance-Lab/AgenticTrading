"""Shared technical-indicator formulas for the daily-signal baseline strategies.

Pure functions over a wide DataFrame (one column per symbol) or a single
Series. No strategy logic and no I/O, matching ``_common.py``'s convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(series: pd.DataFrame, span: int) -> pd.DataFrame:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: pd.DataFrame, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower)
    return upper, mid, lower, pct_b


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return np.maximum(np.maximum(tr1, tr2), tr3)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), plus_di, minus_di


def supertrend_single(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, multiplier: float = 4.0) -> pd.Series:
    """Standard iterative Supertrend for one symbol's series. Returns an 'up'/'down' trend Series."""
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr_
    basic_lower = hl2 - multiplier * atr_

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    for i in range(1, len(close)):
        if pd.isna(atr_.iloc[i - 1]):
            continue
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    trend = pd.Series(index=close.index, dtype=object)
    trend.iloc[0] = "up"
    for i in range(1, len(close)):
        prev_trend = trend.iloc[i - 1]
        if prev_trend == "up":
            trend.iloc[i] = "down" if close.iloc[i] < final_lower.iloc[i] else "up"
        else:
            trend.iloc[i] = "up" if close.iloc[i] > final_upper.iloc[i] else "down"
    return trend


def zscore_row(row: pd.Series) -> pd.Series:
    m, s = row.mean(), row.std()
    if not s or np.isnan(s):
        return row * 0
    return (row - m) / s
