"""Look-ahead checks for the indicator calculator, inspired by Freqtrade.

https://www.freqtrade.io/en/stable/lookahead-analysis/
These test ATL's feature calculator, not Freqtrade or portfolio performance.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard.backend.domain.backtesting import features


COLUMNS = ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "sma20", "sma50"]
FAILURES = ["none", "missing", "columns", "exception", "late_exception"]
# The pandas-ta functions each failure mode replaces. Every case asserts they
# were reached, so no mode can quietly degrade into a repeat of "none".
PATCHED = {
    "missing": ("rsi", "macd", "bbands", "sma"),
    "columns": ("macd", "bbands"),
    "exception": ("rsi",),
    "late_exception": ("bbands",),
}
# First row whose own prefix pandas-ta answers for (see features._GROUPS).
FIRST_READY_ROW = {
    "rsi_14": 14, "macd": 33, "macd_signal": 33,
    "bb_upper": 19, "bb_lower": 19, "sma20": 19, "sma50": 49,
}


def prices(n):
    close = 100 + np.arange(n) * 0.1 + np.sin(np.arange(n))
    return pd.DataFrame({"close": close}, index=pd.date_range("2026-01-01", periods=n, freq="h"))


def indicators(frame):
    return features.TechnicalIndicators.calculate_indicators(frame)[COLUMNS]


def install_failure(monkeypatch, failure):
    calls = []

    def patch(name, replacement):
        def recorded(*args, **kwargs):
            calls.append(name)
            return replacement(*args, **kwargs)
        monkeypatch.setattr(features.ta, name, recorded)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated indicator failure")

    for name in PATCHED.get(failure, ()):
        if failure == "missing":
            patch(name, lambda *a, **k: None)
        elif failure == "columns":
            patch(name, lambda series, **k: pd.DataFrame(index=series.index))
        else:
            patch(name, fail)
    return calls


def assert_failure_reached(failure, calls, output):
    assert set(PATCHED.get(failure, ())) <= set(calls)
    raised = "Warning: Error calculating indicators: simulated indicator failure" in output
    assert raised == failure.endswith("exception")


@pytest.mark.parametrize("failure", FAILURES)
def test_each_row_depends_only_on_the_bars_up_to_it(monkeypatch, capsys, failure):
    """Row i must not change with how many bars follow it, not just their prices."""
    calls = install_failure(monkeypatch, failure)
    frame = prices(60)
    full = indicators(frame)
    for n in range(1, len(frame) + 1):
        # pandas-ta's rolling mean differs in the last bit with frame length;
        # a leak moves a value by orders of magnitude more than this tolerance.
        pd.testing.assert_frame_equal(
            indicators(frame.iloc[:n]), full.iloc[:n], rtol=1e-12, atol=0, obj=f"first {n} bars"
        )
    assert not full.isna().any().any()
    assert_failure_reached(failure, calls, capsys.readouterr().out)


@pytest.mark.parametrize("n", [10, 19, 20, 25, 26, 33, 34, 49, 50, 80])
@pytest.mark.parametrize("failure", FAILURES)
def test_future_prices_do_not_change_past_indicators(monkeypatch, capsys, n, failure):
    calls = install_failure(monkeypatch, failure)
    original = prices(n)
    cutoff = n // 2
    changed = original.copy()
    # Introduce both a new minimum and maximum strictly after the cutoff.
    changed.iloc[cutoff:, 0] = np.where(np.arange(n - cutoff) % 2, 1000.0, 1.0)
    before = indicators(original)
    after = indicators(changed)
    pd.testing.assert_frame_equal(before.iloc[:cutoff], after.iloc[:cutoff], check_exact=True)
    assert_failure_reached(failure, calls, capsys.readouterr().out)


@pytest.mark.parametrize("n", [10, 30, 80])
def test_a_library_error_falls_back_like_a_missing_library(monkeypatch, n):
    frame = prices(n)
    install_failure(monkeypatch, "missing")
    expected = indicators(frame)
    monkeypatch.undo()
    install_failure(monkeypatch, "exception")
    pd.testing.assert_frame_equal(indicators(frame), expected, check_exact=True)


def test_short_history_fallbacks_use_only_observed_prices():
    result = indicators(pd.DataFrame({"close": [100.0, 110.0, 90.0]}))
    assert result["sma20"].tolist() == [100.0, 105.0, 100.0]
    assert result["sma50"].tolist() == [100.0, 105.0, 100.0]
    assert result["rsi_14"].tolist() == [50.0, 50.0, 50.0]
    assert result["macd"].tolist() == result["macd_signal"].tolist() == [0.0, 0.0, 0.0]
    # Expanding mean +/- 2 sample std; a single bar has no dispersion yet.
    spread = [0.0, 2 * np.std([100.0, 110.0], ddof=1), 20.0]
    np.testing.assert_allclose(result["bb_upper"], [100.0, 105.0 + spread[1], 120.0])
    np.testing.assert_allclose(result["bb_lower"], [100.0, 105.0 - spread[1], 80.0])


def test_fallback_band_does_not_pin_a_rising_close_to_its_upper_edge():
    frame = pd.DataFrame({"close": 100.0 + np.arange(15)})
    result = indicators(frame)
    assert (frame["close"].iloc[1:] < result["bb_upper"].iloc[1:]).all()
    assert (result["bb_upper"] - result["bb_lower"]).iloc[1:].gt(0).all()


def test_fallbacks_meet_the_library_at_its_first_ready_row():
    """The fallback formula is the library's own over the bars seen so far."""
    close = prices(80)["close"]
    result = indicators(close.to_frame())
    first20, first50 = close.iloc[:20], close.iloc[:50]
    band = 2 * first20.std(ddof=1)
    assert result["sma20"].iloc[19] == pytest.approx(first20.mean())
    assert result["sma50"].iloc[49] == pytest.approx(first50.mean())
    assert result["bb_upper"].iloc[19] == pytest.approx(first20.mean() + band)
    assert result["bb_lower"].iloc[19] == pytest.approx(first20.mean() - band)


def test_ready_rows_match_library_output():
    frame = prices(80)
    close = frame["close"]
    result = indicators(frame)
    macd = features.ta.macd(close, fast=12, slow=26, signal=9)
    # Called exactly as before this change, to pin that ready rows did not move.
    bands = features.ta.bbands(close, length=20, std=2)
    expected = {
        "rsi_14": features.ta.rsi(close, length=14),
        "macd": macd["MACD_12_26_9"],
        "macd_signal": macd["MACDs_12_26_9"],
        "bb_upper": bands[next(c for c in bands if "BBU" in c)],
        "bb_lower": bands[next(c for c in bands if "BBL" in c)],
        "sma20": features.ta.sma(close, length=20),
        "sma50": features.ta.sma(close, length=50),
    }
    for column, values in expected.items():
        first = FIRST_READY_ROW[column]
        pd.testing.assert_series_equal(
            result[column].iloc[first:], values.iloc[first:], check_names=False
        )


def test_unusable_closes_count_as_missing_instead_of_raising():
    frame = pd.DataFrame({"close": [np.nan, "100", "110", "n/a", float("inf"), "90"]}, dtype=object)
    result = indicators(frame)
    assert np.isnan(result["sma20"].iloc[0])
    assert result["sma20"].iloc[1:].tolist() == [100.0, 105.0, 105.0, 105.0, 100.0]
    assert result["rsi_14"].tolist() == [50.0] * 6

    junk = indicators(pd.DataFrame({"close": ["a", "b"]}))
    assert junk["sma20"].isna().all()
    assert junk["macd"].tolist() == [0.0, 0.0]
