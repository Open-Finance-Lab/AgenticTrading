"""Baseline calculations must remain offline when bars are already supplied."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.backend import baseline_generator as baseline_module
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
