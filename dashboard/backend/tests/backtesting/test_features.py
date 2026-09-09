"""Characterization tests for extracted TechnicalIndicators (Phase 2A).

Uses a fixed local dataframe (no Alpaca/network) and locks in column names,
shape/index behavior, NaN behavior, minimum-history defaults, and a couple of
deterministic indicator values.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting.features import TechnicalIndicators

EXPECTED_COLS = {"rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "sma20", "sma50"}


def _df(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"close": close}, index=idx)


def test_output_columns_and_shape():
    df = _df(60)
    out = TechnicalIndicators.calculate_indicators(df)
    assert EXPECTED_COLS.issubset(set(out.columns))
    assert "close" in out.columns
    assert len(out) == 60
    assert list(out.index) == list(df.index)


def test_does_not_mutate_input():
    df = _df(60)
    before = df.copy()
    TechnicalIndicators.calculate_indicators(df)
    pd.testing.assert_frame_equal(df, before)


def test_turnover_prefers_bar_vwap_and_falls_back_to_close():
    frame = pd.DataFrame(
        {
            "close": [10.0, 20.0],
            "volume": [100.0, 50.0],
            "vwap": [11.0, None],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="h"),
    )

    out = TechnicalIndicators.calculate_indicators(frame)

    assert out["turnover"].tolist() == [1_100.0, 1_000.0]


def test_sma_values_are_rolling_means():
    df = _df(60, seed=1)
    out = TechnicalIndicators.calculate_indicators(df)
    assert out["sma20"].iloc[-1] == pytest.approx(df["close"].iloc[-20:].mean())
    assert out["sma50"].iloc[-1] == pytest.approx(df["close"].iloc[-50:].mean())


def test_nan_behavior_on_early_rows():
    df = _df(60, seed=2)
    out = TechnicalIndicators.calculate_indicators(df)
    # rolling indicators are NaN until they have enough history
    assert np.isnan(out["rsi_14"].iloc[0])
    assert np.isnan(out["sma20"].iloc[0])
    assert np.isnan(out["sma50"].iloc[0])


def test_insufficient_data_uses_defaults():
    df = _df(10, seed=3)
    out = TechnicalIndicators.calculate_indicators(df)
    assert EXPECTED_COLS.issubset(set(out.columns))
    assert len(out) == 10
    assert (out["rsi_14"] == 50.0).all()
    assert (out["macd"] == 0.0).all()
    assert (out["macd_signal"] == 0.0).all()


def test_empty_dataframe_returned_unchanged():
    df = pd.DataFrame({"close": []})
    out = TechnicalIndicators.calculate_indicators(df)
    assert list(out.columns) == ["close"]
    assert len(out) == 0


def test_two_dataframes_computed_independently():
    out_a = TechnicalIndicators.calculate_indicators(_df(60, seed=4))
    out_b = TechnicalIndicators.calculate_indicators(_df(60, seed=5))
    assert out_a["sma20"].iloc[-1] != out_b["sma20"].iloc[-1]
