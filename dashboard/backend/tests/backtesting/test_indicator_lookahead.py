"""Future-price perturbation checks inspired by Freqtrade lookahead-analysis.

https://www.freqtrade.io/en/stable/lookahead-analysis/
These test ATL's feature calculator, not Freqtrade or portfolio performance.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import features


COLUMNS = ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "sma20", "sma50"]


def prices(n):
    close = 100 + np.arange(n) * 0.1 + np.sin(np.arange(n))
    return pd.DataFrame({"close": close}, index=pd.date_range("2026-01-01", periods=n, freq="h"))


@pytest.mark.parametrize("n", [10, 19, 20, 25, 26, 33, 34, 49, 50, 80])
@pytest.mark.parametrize("failure", ["none", "missing", "columns", "exception", "late_exception"])
def test_future_prices_do_not_change_past_indicators(monkeypatch, n, failure):
    if failure == "missing":
        for name in ("rsi", "macd", "bbands", "sma"):
            monkeypatch.setattr(features.ta, name, lambda *a, **k: None)
    elif failure == "columns":
        for name in ("macd", "bbands"):
            monkeypatch.setattr(features.ta, name, lambda series, **k: pd.DataFrame(index=series.index))
    elif failure in ("exception", "late_exception"):
        def fail(*args, **kwargs):
            raise RuntimeError("simulated indicator failure")
        monkeypatch.setattr(features.ta, "rsi" if failure == "exception" else "bbands", fail)

    original = prices(n)
    cutoff = n // 2
    changed = original.copy()
    # Introduce both a new minimum and maximum strictly after the cutoff.
    changed.iloc[cutoff:, 0] = np.where(np.arange(n - cutoff) % 2, 1000.0, 1.0)
    before = features.TechnicalIndicators.calculate_indicators(original)
    after = features.TechnicalIndicators.calculate_indicators(changed)
    pd.testing.assert_frame_equal(
        before[COLUMNS].iloc[:cutoff], after[COLUMNS].iloc[:cutoff], check_exact=True
    )


def test_short_history_fallback_uses_only_observed_prices():
    frame = pd.DataFrame({"close": [100.0, 110.0, 90.0]})
    result = features.TechnicalIndicators.calculate_indicators(frame)
    assert result["sma20"].tolist() == [100.0, 105.0, 100.0]
    assert result["sma50"].tolist() == [100.0, 105.0, 100.0]
    assert result["bb_upper"].tolist() == [100.0, 110.0, 110.0]
    assert result["bb_lower"].tolist() == [100.0, 100.0, 90.0]


def test_ready_indicators_still_match_library_output():
    frame = prices(80)
    close = frame["close"]
    result = features.TechnicalIndicators.calculate_indicators(frame)
    expected = {
        "rsi_14": features.ta.rsi(close, length=14),
        "sma20": features.ta.sma(close, length=20),
        "sma50": features.ta.sma(close, length=50),
    }
    macd = features.ta.macd(close, fast=12, slow=26, signal=9)
    bands = features.ta.bbands(close, length=20, std=2)
    expected["macd"] = macd["MACD_12_26_9"]
    expected["macd_signal"] = macd["MACDs_12_26_9"]
    expected["bb_upper"] = bands[next(c for c in bands if "BBU" in c)]
    expected["bb_lower"] = bands[next(c for c in bands if "BBL" in c)]
    for column, values in expected.items():
        pd.testing.assert_series_equal(result[column], values, check_names=False)
