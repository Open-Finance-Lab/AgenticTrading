"""Unit tests for ``strategies/_indicators.py``: the edge cases where an
indicator must produce a number rather than NaN (a NaN silently turns every
threshold comparison False), and the causality property the precomputing
strategies (hlhb, trendrider, supertrend_triple) rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.leaderboard.strategies._indicators import (
    adx,
    ema,
    macd,
    rsi,
    supertrend_single,
    zscore_row,
)


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame({"X": np.asarray(values, dtype=float)})


def test_rsi_is_100_not_nan_when_there_are_no_down_moves():
    rising = _frame(np.linspace(100, 160, 40))
    out = rsi(rising, 14)
    assert out.iloc[:14, 0].isna().all()  # warm-up rows stay NaN
    assert out.iloc[14:, 0].eq(100.0).all()


def test_rsi_ordinary_series_stays_in_range():
    rng = np.random.default_rng(0)
    out = rsi(_frame(100 * np.cumprod(1 + rng.normal(0, 0.01, 60))), 14).iloc[14:, 0]
    assert out.notna().all()
    assert ((out >= 0) & (out <= 100)).all()


def test_adx_is_zero_not_nan_on_a_range_bound_flat_stretch():
    n = 60
    high, low, close = _frame([101.0] * n), _frame([99.0] * n), _frame([100.0] * n)
    adx14, plus_di, minus_di = adx(high, low, close, 14)
    assert (plus_di.iloc[14:, 0] == 0).all() and (minus_di.iloc[14:, 0] == 0).all()
    assert adx14.iloc[-1, 0] == 0.0
    assert not adx14.iloc[-1, 0] != adx14.iloc[-1, 0]  # i.e. not NaN


def test_adx_recovers_after_a_flat_stretch():
    close = np.concatenate([np.full(30, 100.0), np.linspace(100, 130, 30)])
    high, low = close + 1, close - 1
    adx14, _, _ = adx(_frame(high), _frame(low), _frame(close), 14)
    assert adx14.iloc[-1, 0] > 25


def test_zscore_row_degenerate_input_is_zeros_not_nan():
    all_nan = pd.Series([np.nan, np.nan, np.nan], index=list("abc"))
    out = zscore_row(all_nan)
    assert list(out.index) == list("abc")
    assert (out == 0.0).all()
    flat = pd.Series([2.0, 2.0, 2.0], index=list("abc"))
    assert (zscore_row(flat) == 0.0).all()


@pytest.mark.parametrize("k", [20, 27, 33, 40])
def test_indicators_are_causal_so_precompute_equals_per_day_recompute(k):
    """Position ``k-1`` of an indicator over the full series equals the last
    value of the same indicator over the first ``k`` rows alone. hlhb,
    trendrider and supertrend_triple compute once over the full daily series
    and index into it instead of recomputing on every truncated history;
    this is the property that makes that bit-identical and free of look-ahead."""
    rng = np.random.default_rng(3)
    n = 40
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    full_c, full_h, full_l = _frame(close), _frame(high), _frame(low)
    part_c, part_h, part_l = full_c.iloc[:k], full_h.iloc[:k], full_l.iloc[:k]

    def last(frame):
        return float(frame.iloc[-1, 0])

    def at(frame):
        return float(frame.iloc[k - 1, 0])

    for name, full, part in [
        ("rsi", rsi(full_c, 14), rsi(part_c, 14)),
        ("ema5", ema(full_c, 5), ema(part_c, 5)),
        ("ema50", ema(full_c, 50), ema(part_c, 50)),
        ("macd_hist", macd(full_c)[2], macd(part_c)[2]),
        ("adx", adx(full_h, full_l, full_c, 14)[0], adx(part_h, part_l, part_c, 14)[0]),
    ]:
        a, b = at(full), last(part)
        assert (np.isnan(a) and np.isnan(b)) or a == b, name

    for period, mult in [(7, 3.0), (10, 3.0), (14, 4.0)]:
        full_trend = supertrend_single(full_h["X"], full_l["X"], full_c["X"], period, mult)
        part_trend = supertrend_single(part_h["X"], part_l["X"], part_c["X"], period, mult)
        assert full_trend.iloc[k - 1] == part_trend.iloc[-1], (period, mult)
