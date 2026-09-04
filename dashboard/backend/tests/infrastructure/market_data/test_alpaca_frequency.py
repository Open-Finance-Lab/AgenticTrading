"""Tests for configurable Alpaca source bar timeframes."""

from __future__ import annotations

import pandas as pd

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
from dashboard.backend.infrastructure.market_data import provider


def _bars_df(symbol: str = "AAPL"):
    index = pd.MultiIndex.from_tuples(
        [(symbol, pd.Timestamp("2026-01-02 14:30:00Z"))],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [100]},
        index=index,
    )


def test_alpaca_loader_maps_5m_to_sdk_timeframe(monkeypatch):
    state = {"requests": []}

    class FakeBars:
        df = _bars_df()

    class FakeSession:
        def request(self, *args, **kwargs):
            raise AssertionError("HTTP session should not be used")

    class FakeClient:
        def __init__(self, api_key, secret_key):
            self._session = FakeSession()

        def get_stock_bars(self, request):
            state["requests"].append(request)
            return FakeBars()

    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient",
        FakeClient,
    )

    loader = AlpacaDataLoader(
        api_key="key",
        secret_key="secret",
        source_timeframe="5m",
    )
    result = loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")

    request = state["requests"][0]
    assert request.timeframe.value == TimeFrame(5, TimeFrameUnit.Minute).value
    assert result["AAPL"].attrs
    assert loader.last_fetch["source_timeframe"] == "5m"


def test_alpaca_loader_maps_1m_to_sdk_timeframe(monkeypatch):
    state = {"requests": []}

    class FakeBars:
        df = _bars_df()

    class FakeSession:
        def request(self, *args, **kwargs):
            raise AssertionError("HTTP session should not be used")

    class FakeClient:
        def __init__(self, api_key, secret_key):
            self._session = FakeSession()

        def get_stock_bars(self, request):
            state["requests"].append(request)
            return FakeBars()

    monkeypatch.setattr(
        "alpaca.data.historical.StockHistoricalDataClient",
        FakeClient,
    )

    loader = AlpacaDataLoader(
        api_key="key",
        secret_key="secret",
        source_timeframe="1min",
    )
    loader.fetch_bars(["AAPL"], "2026-01-01", "2026-01-03")

    assert state["requests"][0].timeframe.value == TimeFrame.Minute.value
    assert loader.last_fetch["source_timeframe"] == "1m"


def test_provider_factory_configures_explicit_source_timeframe(monkeypatch):
    created = []

    class FakeAlpacaLoader:
        def __init__(self):
            self.configured = None
            created.append(self)

        def configure_source_timeframe(self, value):
            self.configured = value

    monkeypatch.setattr(provider, "AlpacaDataLoader", FakeAlpacaLoader)

    loader = provider.create_market_data_provider(
        provider.ALPACA,
        source_timeframe="5Min",
    )

    assert loader is created[0]
    assert loader.configured == "5m"


def test_provider_factory_default_remains_profile_decision_timeframe(monkeypatch):
    created = []

    class FakeAlpacaLoader:
        def __init__(self):
            self.configured = None
            created.append(self)

        def configure_source_timeframe(self, value):
            self.configured = value

    monkeypatch.setattr(provider, "AlpacaDataLoader", FakeAlpacaLoader)

    loader = provider.create_market_data_provider(provider.ALPACA)

    assert loader.configured == "60m"
