"""Technical indicator feature computation.

Extracted (Phase 2A) verbatim from ``TechnicalIndicators`` in
``dashboard/scripts/backtest_hourly_agent.py``. Feature names, dataframe column
names, NaN behavior, minimum-history assumptions, indicator parameters, and the
returned dataframe shape are unchanged.
"""

import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:
    raise ImportError(
        "pandas-ta is required for technical indicators; "
        "install the project dependencies from requirements.txt"
    ) from exc


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

        # Check if we have enough data for indicators
        min_required = 50  # Need at least 50 bars for SMA50
        if len(df) < min_required:
            print(f"\n⚠️  DATA WARNING: Only {len(df)} bars, need {min_required}!")
            print(f"   Indicators will be unreliable. Backtest needs at least 1 month of data.")
            print(f"   Recommended: 3+ months for meaningful results.\n")
            # Still calculate what we can

        try:
            # RSI (14-period requires 14+ bars)
            if len(df) >= 14:
                rsi = ta.rsi(df["close"], length=14)
                if rsi is not None:
                    df["rsi_14"] = rsi
                else:
                    df["rsi_14"] = 50.0  # Default neutral RSI
            else:
                df["rsi_14"] = 50.0  # Not enough data

            # MACD (26-period required)
            if len(df) >= 26:
                macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
                if macd is not None and isinstance(macd, pd.DataFrame):
                    macd_cols = [c for c in macd.columns if "MACD_12_26_9" in c]
                    signal_cols = [c for c in macd.columns if "MACDs_12_26_9" in c]
                    if macd_cols:
                        df["macd"] = macd[macd_cols[0]]
                    else:
                        df["macd"] = 0.0
                    if signal_cols:
                        df["macd_signal"] = macd[signal_cols[0]]
                    else:
                        df["macd_signal"] = 0.0
                else:
                    df["macd"] = 0.0
                    df["macd_signal"] = 0.0
            else:
                df["macd"] = 0.0
                df["macd_signal"] = 0.0

            # Bollinger Bands (20-period required)
            if len(df) >= 20:
                bbands = ta.bbands(df["close"], length=20, std=2)
                if bbands is not None and isinstance(bbands, pd.DataFrame):
                    bbu_cols = [c for c in bbands.columns if "BBU" in c]
                    bbl_cols = [c for c in bbands.columns if "BBL" in c]
                    if bbu_cols:
                        df["bb_upper"] = bbands[bbu_cols[0]]
                    else:
                        df["bb_upper"] = df["close"].max()
                    if bbl_cols:
                        df["bb_lower"] = bbands[bbl_cols[0]]
                    else:
                        df["bb_lower"] = df["close"].min()
                else:
                    df["bb_upper"] = df["close"].max()
                    df["bb_lower"] = df["close"].min()
            else:
                df["bb_upper"] = df["close"].max()
                df["bb_lower"] = df["close"].min()

            # SMAs
            if len(df) >= 20:
                sma20 = ta.sma(df["close"], length=20)
                df["sma20"] = sma20 if sma20 is not None else df["close"].mean()
            else:
                df["sma20"] = df["close"].mean()

            if len(df) >= 50:
                sma50 = ta.sma(df["close"], length=50)
                df["sma50"] = sma50 if sma50 is not None else df["close"].mean()
            else:
                df["sma50"] = df["close"].mean()

        except Exception as e:
            print(f"Warning: Error calculating indicators: {e}")
            # Fill in defaults
            for col in ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "sma20", "sma50"]:
                if col not in df.columns:
                    df[col] = df["close"].mean() if col != "rsi_14" else 50.0

        return df
