"""Baseline calculations must remain offline when bars are already supplied."""

from __future__ import annotations

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

import dashboard.backend.baseline_generator as baseline_module
from dashboard.backend.baseline_generator import BaselineGenerator
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
)


def sample_bars() -> dict[str, pd.DataFrame]:
    index = pd.date_range(
        "2026-04-01 10:00",
        periods=8,
        freq="h",
        tz="US/Eastern",
        name="timestamp",
    )
    return {
        "AAPL": pd.DataFrame(
            {
                "open": range(100, 108),
                "high": range(102, 110),
                "low": range(99, 107),
                "close": range(101, 109),
                "volume": [1_000] * 8,
            },
            index=index,
        ),
        "MSFT": pd.DataFrame(
            {
                "open": range(200, 208),
                "high": range(202, 210),
                "low": range(199, 207),
                "close": range(201, 209),
                "volume": [2_000] * 8,
            },
            index=index,
        ),
    }


def sample_cn_bars() -> dict[str, pd.DataFrame]:
    index = pd.DatetimeIndex(
        [
            "2026-04-01 10:30:00",
            "2026-04-01 11:30:00",
            "2026-04-01 14:00:00",
            "2026-04-01 15:00:00",
            "2026-04-02 10:30:00",
            "2026-04-02 11:30:00",
            "2026-04-02 14:00:00",
            "2026-04-02 15:00:00",
        ],
        tz=ZoneInfo("Asia/Shanghai"),
        name="timestamp",
    )
    return {
        symbol: pd.DataFrame(
            {
                "open": [100] * len(index),
                "high": [101] * len(index),
                "low": [99] * len(index),
                "close": [100 + row for row in range(len(index))],
                "volume": [1_000] * len(index),
            },
            index=index,
        )
        for symbol in ("600519.SH", "601318.SH")
    }


def test_constructor_and_supplied_bar_calculations_do_not_load_credentials(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Alpaca credentials must not be loaded")

    monkeypatch.setattr(BaselineGenerator, "_load_credentials", fail_if_called)
    generator = BaselineGenerator()
    bars = sample_bars()

    buyhold = generator.generate_buyhold_baseline(
        bars, "2026-04-01", "2026-04-02", initial_capital=100_000
    )
    index = generator.generate_index_baseline(
        bars, "2026-04-01", "2026-04-02", initial_capital=100_000
    )

    assert buyhold
    assert index
    assert buyhold[0]["equity"] > 0
    assert index[0]["equity"] > 0


def test_real_alpaca_fetch_loads_credentials_lazily(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(baseline_module, "CREDENTIALS_DIR", tmp_path)

    generator = BaselineGenerator()

    with pytest.raises(MarketDataUnavailableError, match="credentials"):
        generator._fetch_bars_for_symbol("AAPL", "2026-04-01", "2026-04-02")


def test_cn_baselines_keep_shanghai_session_timestamps():
    bars = sample_cn_bars()

    buyhold, index = baseline_module.generate_baselines(
        bars,
        "2026-04-01",
        "2026-04-02",
        initial_capital=100_000,
        symbols_list=list(bars),
        market_timezone="Asia/Shanghai",
    )

    assert buyhold
    assert index
    assert buyhold[0]["timestamp"].startswith("2026-04-01T10:30:00+08:00")
    assert index[0]["timestamp"].startswith("2026-04-01T10:30:00+08:00")
