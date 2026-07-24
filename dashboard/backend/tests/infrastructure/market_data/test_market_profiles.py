"""Market profile registry contracts for source and universe selection."""

from __future__ import annotations

import pytest

from dashboard.backend.infrastructure.market_data.profiles import (
    ALPACA,
    A_SHARE_DEMO_6,
    CSI300_SAMPLE_20_2026H2,
    CSI300_SAMPLE_20_2026H2_SYMBOLS,
    IFIND_ASHARE,
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
    VNPY_SIMULATION,
    get_market_profile,
    resolve_decision_source,
)


@pytest.mark.parametrize(
    ("data_source", "universe", "expected_count"),
    [
        (ALPACA, "djia_30", 30),
        (VNPY_SIMULATION, "djia_30", 30),
        (IFIND_ASHARE, A_SHARE_DEMO_6, 6),
        (IFIND_ASHARE, CSI300_SAMPLE_20_2026H2, 20),
    ],
)
def test_registry_resolves_source_and_universe_pair(
    data_source, universe, expected_count
):
    profile = get_market_profile(data_source, universe)

    assert profile.data_source == data_source
    assert profile.universe == universe
    assert len(profile.symbols) == expected_count


@pytest.mark.parametrize(
    ("data_source", "default_universe"),
    [
        (ALPACA, "djia_30"),
        (VNPY_SIMULATION, "djia_30"),
        (IFIND_ASHARE, A_SHARE_DEMO_6),
    ],
)
def test_registry_preserves_default_universe(data_source, default_universe):
    assert get_market_profile(data_source).universe == default_universe


@pytest.mark.parametrize(
    ("data_source", "universe"),
    [
        (ALPACA, CSI300_SAMPLE_20_2026H2),
        (VNPY_SIMULATION, A_SHARE_DEMO_6),
        (IFIND_ASHARE, "unknown_a_share_pool"),
        ("unknown_source", "djia_30"),
    ],
)
def test_registry_rejects_unknown_or_mismatched_pair(data_source, universe):
    with pytest.raises(ValueError, match="market profile"):
        get_market_profile(data_source, universe)


def test_csi300_sample_20_is_exact_and_versioned():
    assert CSI300_SAMPLE_20_2026H2_SYMBOLS == (
        "600519.SH",
        "601318.SH",
        "600036.SH",
        "300750.SZ",
        "000333.SZ",
        "002594.SZ",
        "600276.SH",
        "300760.SZ",
        "688981.SH",
        "002415.SZ",
        "601766.SH",
        "600309.SH",
        "601899.SH",
        "601857.SH",
        "600900.SH",
        "600050.SH",
        "000725.SZ",
        "600030.SH",
        "600887.SH",
        "600048.SH",
    )
    assert len(set(CSI300_SAMPLE_20_2026H2_SYMBOLS)) == 20


def test_a_share_demo_6_defaults_to_rules_but_allows_llm():
    profile = get_market_profile(IFIND_ASHARE, A_SHARE_DEMO_6)

    assert profile.default_decision_source == RULE_BASED_DECISION_SOURCE
    assert profile.allowed_decision_sources == (
        RULE_BASED_DECISION_SOURCE,
        LLM_DECISION_SOURCE,
    )
    assert profile.decision_source == RULE_BASED_DECISION_SOURCE
    assert profile.llm_enabled is False


def test_csi300_sample_20_defaults_to_rules_but_allows_llm():
    profile = get_market_profile(IFIND_ASHARE, CSI300_SAMPLE_20_2026H2)

    assert profile.default_decision_source == RULE_BASED_DECISION_SOURCE
    assert profile.allowed_decision_sources == (
        RULE_BASED_DECISION_SOURCE,
        LLM_DECISION_SOURCE,
    )
    assert profile.llm_enabled is False


def test_us_profiles_preserve_their_default_decision_behavior():
    alpaca = get_market_profile(ALPACA)
    simulation = get_market_profile(VNPY_SIMULATION)

    assert alpaca.default_decision_source == LLM_DECISION_SOURCE
    assert alpaca.allowed_decision_sources == (
        RULE_BASED_DECISION_SOURCE,
        LLM_DECISION_SOURCE,
    )
    assert alpaca.llm_enabled is True
    assert simulation.default_decision_source == RULE_BASED_DECISION_SOURCE
    assert simulation.allowed_decision_sources == (RULE_BASED_DECISION_SOURCE,)
    assert simulation.llm_enabled is False


@pytest.mark.parametrize(
    ("universe", "requested", "expected"),
    [
        (A_SHARE_DEMO_6, None, RULE_BASED_DECISION_SOURCE),
        (A_SHARE_DEMO_6, RULE_BASED_DECISION_SOURCE, RULE_BASED_DECISION_SOURCE),
        (A_SHARE_DEMO_6, LLM_DECISION_SOURCE, LLM_DECISION_SOURCE),
        (CSI300_SAMPLE_20_2026H2, None, RULE_BASED_DECISION_SOURCE),
        (CSI300_SAMPLE_20_2026H2, LLM_DECISION_SOURCE, LLM_DECISION_SOURCE),
    ],
)
def test_resolve_decision_source_uses_profile_capabilities(
    universe, requested, expected
):
    profile = get_market_profile(IFIND_ASHARE, universe)

    assert resolve_decision_source(profile, requested) == expected


@pytest.mark.parametrize(
    ("universe", "requested"),
    [
        (A_SHARE_DEMO_6, "unknown_decision_source"),
        (CSI300_SAMPLE_20_2026H2, "unknown_decision_source"),
    ],
)
def test_resolve_decision_source_rejects_unsupported_values(universe, requested):
    profile = get_market_profile(IFIND_ASHARE, universe)

    with pytest.raises(
        ValueError,
        match=rf"{IFIND_ASHARE}.*{universe}.*{requested}",
    ):
        resolve_decision_source(profile, requested)
