"""Tests for the Phase 0/1 market-data and trading frequency contract."""

from __future__ import annotations

import pytest

from dashboard.backend.infrastructure.market_data.frequency import (
    FrequencyConfigError,
    TradingFrequency,
    build_verified_intraday_contract,
    normalize_bar_timeframe,
    normalize_decision_frequency,
    verify_source_timeframe,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    ALPACA,
    get_market_profile,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1m", "1m"),
        ("1Min", "1m"),
        ("5MIN", "5m"),
        ("hourly", "60m"),
        ("1h", "60m"),
    ],
)
def test_bar_timeframe_aliases_are_canonical(value, expected):
    assert normalize_bar_timeframe(value) == expected


def test_decision_frequency_is_canonical():
    assert normalize_decision_frequency(" hourly ") == "1h"
    assert normalize_decision_frequency("60m") == "1h"


def test_minute_source_hourly_decision_contract():
    contract = TradingFrequency.minute_source_hourly_decisions()

    assert contract.to_metadata() == {
        "source_timeframe": "5m",
        "decision_timeframe": "60m",
        "decision_frequency": "1h",
        "execution_timeframe": "5m",
        "valuation_frequency": "5m",
    }


def test_frequency_contract_rejects_coarser_source_than_decision():
    with pytest.raises(FrequencyConfigError, match="coarser"):
        TradingFrequency(source_timeframe="60m", decision_timeframe="5m")


def test_frequency_contract_rejects_mismatched_decision_cadence():
    with pytest.raises(FrequencyConfigError, match="match"):
        TradingFrequency(
            source_timeframe="5m",
            decision_timeframe="5m",
            decision_frequency="1h",
        )


def test_runtime_contract_attests_fixed_intraday_execution_policy():
    assert build_verified_intraday_contract(
        source_timeframe="5m",
        decision_timeframe="60m",
        decision_frequency="1h",
    ) == {
        "source_timeframe": "5m",
        "decision_timeframe": "60m",
        "decision_frequency": "1h",
        "execution_timeframe": "5m",
        "valuation_frequency": "5m",
        "aggregation": "session_anchored_completed_bars",
        "fill_policy": "next_source_bar_open",
        "verification_status": "verified",
    }


def test_runtime_contract_rejects_non_intraday_source():
    with pytest.raises(FrequencyConfigError, match="finer"):
        build_verified_intraday_contract(
            source_timeframe="60m",
            decision_timeframe="60m",
            decision_frequency="1h",
        )


def test_source_timeframe_verification_rejects_provider_drift():
    with pytest.raises(FrequencyConfigError, match="requested 5m.*reported 60m"):
        verify_source_timeframe("5m", "60m", evidence="provider")


def test_alpaca_profile_records_minute_source_and_hourly_decisions():
    profile = get_market_profile(ALPACA)

    assert profile.timeframe == "60m"
    assert profile.decision_timeframe == "60m"
    assert profile.source_timeframe == "5m"
    assert profile.decision_frequency == "1h"
    assert profile.execution_timeframe == "5m"
    assert profile.valuation_frequency == "5m"
    assert profile.frequency_contract == TradingFrequency.minute_source_hourly_decisions()
