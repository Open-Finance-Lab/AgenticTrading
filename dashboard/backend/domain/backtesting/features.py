"""Technical indicator feature computation.

Extracted (Phase 2A) from ``TechnicalIndicators`` in
``dashboard/scripts/backtest_hourly_agent.py``. Feature names, dataframe column
names, indicator parameters, and the returned dataframe shape are preserved.

The whole backtest window arrives at once, so every indicator is computed
causally: a row depends only on the closes at or before it, never on the
closes after it or on how many bars follow it. A row takes the pandas-ta value
only once its own prefix is long enough for pandas-ta to answer (``_GROUPS``);
earlier rows, rows the library leaves empty, and groups the library failed to
produce use causal fallbacks -- neutral RSI (50) and MACD (0), an expanding
mean for the SMAs, and expanding mean +/- 2 sample standard deviations for the
Bollinger Bands, which is the library's own formula over the bars seen so far.
Indicator columns are NaN only on rows with no usable close at or before them;
non-numeric and non-finite closes count as missing.
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:
    raise ImportError(
        "pandas-ta is required for technical indicators; "
        "install the project dependencies from requirements.txt"
    ) from exc


RSI_LENGTH = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_LENGTH, BB_STD, BB_DDOF = 20, 2.0, 1

# Columns that switch between library and fallback together, and the shortest
# frame pandas-ta answers for them (it returns None below it). Gating on the
# row's own prefix rather than on len(df) is what keeps a row independent of
# the bars after it: pandas-ta fills early rows of a long enough frame (RSI
# from bar 2, the MACD line from bar 26) that it leaves empty in a shorter one.
# Pairs share one gate so a MACD line is never read against a signal line, or
# an upper band against a lower band, from the other source.
_GROUPS = (
    (("rsi_14",), RSI_LENGTH + 1),
    (("macd", "macd_signal"), MACD_SLOW + MACD_SIGNAL - 1),
    (("bb_upper", "bb_lower"), BB_LENGTH),
    (("sma20",), 20),
    (("sma50",), 50),
)


def _usable_close(close: pd.Series) -> pd.Series:
    try:
        values = pd.to_numeric(close, errors="coerce").astype(float)
    except (TypeError, ValueError):
        return pd.Series(np.nan, index=close.index)
    return values.where(np.isfinite(values))


def _fallbacks(close: pd.Series) -> dict:
    seen = close.expanding(min_periods=1)
    mean = seen.mean()
    # One bar has no dispersion yet: a zero-width band there, not a missing one.
    std = seen.std(ddof=BB_DDOF).fillna(0.0).where(mean.notna())
    return {
        "rsi_14": pd.Series(50.0, index=close.index),
        "macd": pd.Series(0.0, index=close.index),
        "macd_signal": pd.Series(0.0, index=close.index),
        "bb_upper": mean + BB_STD * std,
        "bb_lower": mean - BB_STD * std,
        "sma20": mean,
        "sma50": mean,
    }


def _column(frame, marker: str):
    if not isinstance(frame, pd.DataFrame):
        return None
    return next((frame[name] for name in frame.columns if marker in name), None)


def _library_indicators(close: pd.Series, out: dict) -> None:
    """Fill ``out`` in order, so an exception keeps what was computed before it."""
    out["rsi_14"] = ta.rsi(close, length=RSI_LENGTH)
    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    out["macd"] = _column(macd, f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}")
    out["macd_signal"] = _column(macd, f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}")
    bbands = ta.bbands(
        close, length=BB_LENGTH, lower_std=BB_STD, upper_std=BB_STD, ddof=BB_DDOF
    )
    out["bb_upper"] = _column(bbands, "BBU")
    out["bb_lower"] = _column(bbands, "BBL")
    out["sma20"] = ta.sma(close, length=20)
    out["sma50"] = ta.sma(close, length=50)


class TechnicalIndicators:
    """Calculates technical indicators for trading signals."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Indicators:
        - RSI (14-period)
        - MACD (12/26/9)
        - Bollinger Bands (20/2)
        - SMA (20 & 50-period)

        IMPORTANT: Requires minimum 50 bars for reliable signals.
        Backtests shorter than 1 month will have unreliable indicators.
        """
        if df is None or df.empty:
            print(f"Warning: Empty or None dataframe, skipping indicators")
            return df

        df = df.copy()

        # Preserve an actual source-bar turnover when aggregation supplied one.
        # Legacy hourly providers can still expose an approximation from their
        # own bar VWAP (preferred) or close, always in the market's native currency.
        if "turnover" not in df.columns and {"close", "volume"} <= set(df.columns):
            prices = pd.to_numeric(df["close"], errors="coerce")
            if "vwap" in df.columns:
                vwap = pd.to_numeric(df["vwap"], errors="coerce")
                prices = vwap.where(vwap.notna() & vwap.abs().lt(float("inf")), prices)
            volume = pd.to_numeric(df["volume"], errors="coerce")
            df["turnover"] = prices * volume

        # Check if we have enough data for indicators
        min_required = 50  # Need at least 50 bars for SMA50
        if len(df) < min_required:
            print(f"\n⚠️  DATA WARNING: Only {len(df)} bars, need {min_required}!")
            print(f"   Indicators will be unreliable. Backtest needs at least 1 month of data.")
            print(f"   Recommended: 3+ months for meaningful results.\n")
            # Still calculate what we can

        close = _usable_close(df["close"])
        fallback = _fallbacks(close)
        library: dict = {}
        try:
            _library_indicators(close, library)
        except Exception as e:
            print(f"Warning: Error calculating indicators: {e}")

        rows = np.arange(len(df))
        for columns, min_bars in _GROUPS:
            ready = rows >= min_bars - 1
            for column in columns:
                series = library.get(column)
                if series is None or len(series) != len(df):
                    ready = np.zeros(len(df), dtype=bool)
                    break
                ready = ready & series.notna().to_numpy()
            for column in columns:
                values = fallback[column].to_numpy()
                if ready.any():
                    values = np.where(ready, library[column].to_numpy(dtype=float), values)
                df[column] = values

        return df
