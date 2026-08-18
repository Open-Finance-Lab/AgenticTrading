"""Market-data provider boundary and feature-gate tests."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_provider_module_import_does_not_import_vnpy():
    code = (
        "import sys\n"
        "import dashboard.backend.infrastructure.market_data.provider\n"
        "assert not any(name == 'vnpy' or name.startswith('vnpy.') "
        "for name in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_provider_module_import_has_no_ifind_side_effects():
    code = (
        "import os\n"
        "os.environ['IFIND_ACCESS_TOKEN'] = 'must-not-be-printed'\n"
        "import dashboard.backend.infrastructure.market_data.provider\n"
        "print('imported')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
    assert "must-not-be-printed" not in result.stdout + result.stderr


def test_alpaca_is_the_default_provider(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    class FakeAlpacaLoader:
        pass

    monkeypatch.setattr(provider, "AlpacaDataLoader", FakeAlpacaLoader)

    created = provider.create_market_data_provider(provider.ALPACA)

    assert isinstance(created, FakeAlpacaLoader)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_vnpy_simulation_truthy_feature_values(monkeypatch, value):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_VNPY_SIMULATION", value)

    assert provider.vnpy_simulation_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "disabled"])
def test_vnpy_simulation_falsey_feature_values(monkeypatch, value):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_VNPY_SIMULATION", value)

    assert provider.vnpy_simulation_enabled() is False


def test_unknown_data_source_is_rejected():
    from dashboard.backend.infrastructure.market_data import provider

    with pytest.raises(provider.UnsupportedMarketDataSource, match="unknown"):
        provider.validate_market_data_source("unknown")


def test_disabled_vnpy_simulation_is_rejected(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.delenv("ENABLE_VNPY_SIMULATION", raising=False)

    with pytest.raises(provider.MarketDataSourceDisabled, match="disabled"):
        provider.validate_market_data_source(provider.VNPY_SIMULATION)


def test_ifind_is_a_supported_data_source():
    from dashboard.backend.infrastructure.market_data import provider

    assert provider.IFIND_ASHARE == "ifind_ashare"
    assert provider.IFIND_ASHARE in provider.SUPPORTED_DATA_SOURCES


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_ifind_truthy_feature_values(monkeypatch, value):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", value)

    assert provider.ifind_ashare_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "disabled"])
def test_ifind_falsey_feature_values(monkeypatch, value):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", value)

    assert provider.ifind_ashare_enabled() is False


def test_disabled_ifind_is_rejected(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.delenv("ENABLE_IFIND_ASHARE", raising=False)

    with pytest.raises(provider.MarketDataSourceDisabled, match="iFinD.*disabled"):
        provider.validate_market_data_source(provider.IFIND_ASHARE)


def test_missing_ifind_token_is_rejected_without_leaking_secrets(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-leak")

    with pytest.raises(provider.MarketDataCredentialsError) as exc_info:
        provider.ensure_market_data_source_available(provider.IFIND_ASHARE)

    message = str(exc_info.value)
    assert "IFIND_REFRESH_TOKEN" in message
    assert "IFIND_ACCESS_TOKEN" in message
    assert "must-not-leak" not in message


def test_refresh_token_is_sufficient_for_ifind_availability(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_REFRESH_TOKEN", "refresh-token-canary")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    provider.ensure_market_data_source_available(provider.IFIND_ASHARE)


def test_invalid_ifind_universe_is_rejected_before_credentials(monkeypatch):
    from dashboard.backend.infrastructure.market_data import provider

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Unknown market profile"):
        provider.create_market_data_provider(
            provider.IFIND_ASHARE,
            universe="not_a_registered_universe",
        )


def test_ifind_profile_describes_the_fixed_a_share_market():
    from dashboard.backend.infrastructure.market_data.profiles import (
        A_SHARE_DEMO_6_SYMBOLS,
        get_market_profile,
    )
    from dashboard.backend.infrastructure.market_data.provider import IFIND_ASHARE

    profile = get_market_profile(IFIND_ASHARE)

    assert profile.data_source == IFIND_ASHARE
    assert profile.market == "CN"
    assert profile.timezone == "Asia/Shanghai"
    assert profile.timeframe == "60m"
    assert profile.universe == "a_share_demo_6"
    assert profile.symbols == A_SHARE_DEMO_6_SYMBOLS
    assert profile.symbols == (
        "600519.SH",
        "601318.SH",
        "600036.SH",
        "000001.SZ",
        "000858.SZ",
        "300750.SZ",
    )
    assert profile.decision_source == "llm"
    assert profile.benchmark == "equal_weight_buyhold"
    assert profile.llm_enabled is True
    assert profile.index_baseline_enabled is False
