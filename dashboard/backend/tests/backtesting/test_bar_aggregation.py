from datetime import datetime

import pandas as pd
import pytz

from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars


def _bars(timestamps):
    prices = list(range(100, 100 + len(timestamps)))
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [10] * len(prices),
            "vwap": [price + 0.5 for price in prices],
        },
        index=pd.DatetimeIndex(timestamps),
    )


def test_us_bars_are_anchored_to_0930_and_labeled_at_bucket_end():
    eastern = pytz.timezone("US/Eastern")
    timestamps = pd.date_range(
        eastern.localize(datetime(2026, 3, 2, 9, 30)),
        eastern.localize(datetime(2026, 3, 2, 15, 55)),
        freq="5min",
    )

    result = aggregate_bars(
        _bars(timestamps),
        source_timeframe="5m",
        decision_timeframe="60m",
        market="US",
        timezone="US/Eastern",
    )

    assert list(result.index[:2]) == [
        pd.Timestamp("2026-03-02 15:30:00", tz="UTC"),
        pd.Timestamp("2026-03-02 16:30:00", tz="UTC"),
    ]
    assert result.iloc[0]["open"] == 100
    assert result.iloc[0]["close"] == 111
    assert result.iloc[0]["source_bar_count"] == 12
    assert result.iloc[0]["expected_source_bars"] == 12
    assert bool(result.iloc[0]["is_complete"]) is True
    assert result.iloc[0]["vwap"] == 106.0
    # The final 15:30-16:00 bucket is complete, but its 16:00 label has no
    # following source bar and is filtered from the execution plan by the engine.
    assert result.index[-1] == pd.Timestamp("2026-03-02 21:00:00", tz="UTC")


def test_cn_lunch_break_does_not_create_a_cross_session_bucket():
    shanghai = pytz.timezone("Asia/Shanghai")
    morning = pd.date_range(
        shanghai.localize(datetime(2026, 3, 2, 9, 30)),
        shanghai.localize(datetime(2026, 3, 2, 11, 25)),
        freq="5min",
    )
    afternoon = pd.date_range(
        shanghai.localize(datetime(2026, 3, 2, 13, 0)),
        shanghai.localize(datetime(2026, 3, 2, 14, 55)),
        freq="5min",
    )

    result = aggregate_bars(
        _bars(morning.append(afternoon)),
        source_timeframe="5m",
        decision_timeframe="60m",
        market="CN",
        timezone="Asia/Shanghai",
    )

    assert [timestamp.tz_convert("Asia/Shanghai").strftime("%H:%M") for timestamp in result.index] == [
        "10:30",
        "11:30",
        "14:00",
        "15:00",
    ]
    assert all(result["source_bar_count"] == 12)


def test_missing_source_bar_is_visible_in_quality_columns():
    eastern = pytz.timezone("US/Eastern")
    timestamps = pd.date_range(
        eastern.localize(datetime(2026, 3, 2, 9, 30)),
        eastern.localize(datetime(2026, 3, 2, 10, 25)),
        freq="5min",
    ).delete(3)

    result = aggregate_bars(
        _bars(timestamps),
        source_timeframe="5m",
        decision_timeframe="60m",
        market="US",
        timezone="US/Eastern",
    )

    assert result.iloc[0]["source_bar_count"] == 11
    assert result.iloc[0]["expected_source_bars"] == 12
    assert bool(result.iloc[0]["is_complete"]) is False
    assert bool(result.iloc[0]["has_gap"]) is True
