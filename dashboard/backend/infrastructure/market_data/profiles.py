"""Canonical market profiles for selectable backtest data sources."""

from __future__ import annotations

from dataclasses import dataclass

from dashboard.backend.infrastructure.llm.validator import DJIA_30


ALPACA = "alpaca"
VNPY_SIMULATION = "vnpy_simulation"
IFIND_ASHARE = "ifind_ashare"

RULE_BASED_DECISION_SOURCE = "rule_based"
LLM_DECISION_SOURCE = "llm"

A_SHARE_DEMO_6 = "a_share_demo_6"
A_SHARE_DEMO_6_SYMBOLS = (
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "000001.SZ",
    "000858.SZ",
    "300750.SZ",
)

CSI300_SAMPLE_20_2026H2 = "csi300_sample_20_2026h2"
CSI300_SAMPLE_20_2026H2_SYMBOLS = (
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


@dataclass(frozen=True)
class MarketProfile:
    """Backtest market behavior selected together with a data source."""

    data_source: str
    market: str
    universe: str
    timezone: str
    timeframe: str
    symbols: tuple[str, ...]
    benchmark: str
    default_decision_source: str
    allowed_decision_sources: tuple[str, ...]
    index_baseline_enabled: bool
    native_currency: str
    reporting_currency: str
    t_plus_one_enabled: bool
    lot_size: int = 1

    @property
    def decision_source(self) -> str:
        """Backward-compatible name for the profile's default decision source."""
        return self.default_decision_source

    @property
    def llm_enabled(self) -> bool:
        """Preserve whether legacy requests default to the LLM path."""
        return self.default_decision_source == LLM_DECISION_SOURCE


_DJIA_30_SYMBOLS = tuple(DJIA_30)

_MARKET_PROFILES = {
    (ALPACA, "djia_30"): MarketProfile(
        data_source=ALPACA,
        market="US",
        universe="djia_30",
        timezone="US/Eastern",
        timeframe="60m",
        symbols=_DJIA_30_SYMBOLS,
        benchmark="top10_buyhold_and_djia",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=True,
        native_currency="USD",
        reporting_currency="USD",
        t_plus_one_enabled=False,
    ),
    (VNPY_SIMULATION, "djia_30"): MarketProfile(
        data_source=VNPY_SIMULATION,
        market="US",
        universe="djia_30",
        timezone="US/Eastern",
        timeframe="60m",
        symbols=_DJIA_30_SYMBOLS,
        benchmark="top10_buyhold_and_djia",
        default_decision_source=RULE_BASED_DECISION_SOURCE,
        allowed_decision_sources=(RULE_BASED_DECISION_SOURCE,),
        index_baseline_enabled=True,
        native_currency="USD",
        reporting_currency="USD",
        t_plus_one_enabled=False,
    ),
    (IFIND_ASHARE, A_SHARE_DEMO_6): MarketProfile(
        data_source=IFIND_ASHARE,
        market="CN",
        universe=A_SHARE_DEMO_6,
        timezone="Asia/Shanghai",
        timeframe="60m",
        symbols=A_SHARE_DEMO_6_SYMBOLS,
        benchmark="equal_weight_buyhold",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=False,
        native_currency="CNY",
        reporting_currency="USD",
        t_plus_one_enabled=True,
        lot_size=100,
    ),
    (IFIND_ASHARE, CSI300_SAMPLE_20_2026H2): MarketProfile(
        data_source=IFIND_ASHARE,
        market="CN",
        universe=CSI300_SAMPLE_20_2026H2,
        timezone="Asia/Shanghai",
        timeframe="60m",
        symbols=CSI300_SAMPLE_20_2026H2_SYMBOLS,
        benchmark="equal_weight_buyhold",
        default_decision_source=LLM_DECISION_SOURCE,
        allowed_decision_sources=(
            RULE_BASED_DECISION_SOURCE,
            LLM_DECISION_SOURCE,
        ),
        index_baseline_enabled=False,
        native_currency="CNY",
        reporting_currency="USD",
        t_plus_one_enabled=True,
        lot_size=100,
    ),
}


_DEFAULT_UNIVERSES = {
    ALPACA: "djia_30",
    VNPY_SIMULATION: "djia_30",
    IFIND_ASHARE: A_SHARE_DEMO_6,
}


def get_market_profile(
    data_source: str,
    universe: str | None = None,
) -> MarketProfile:
    """Return the immutable profile for a registered source/universe pair."""
    resolved_universe = universe or _DEFAULT_UNIVERSES.get(data_source)
    try:
        return _MARKET_PROFILES[(data_source, resolved_universe)]
    except KeyError as exc:
        available = sorted(
            registered_universe
            for registered_source, registered_universe in _MARKET_PROFILES
            if registered_source == data_source
        )
        raise ValueError(
            "Unknown market profile: "
            f"data_source={data_source!r}, universe={resolved_universe!r}, "
            f"available_universes={available!r}"
        ) from exc


def resolve_decision_source(
    profile: MarketProfile,
    requested: str | None = None,
) -> str:
    """Resolve and validate a decision source against one market profile."""
    resolved = (
        str(requested).strip().lower()
        if requested is not None
        else profile.default_decision_source
    )
    if resolved not in profile.allowed_decision_sources:
        raise ValueError(
            "Unsupported decision source: "
            f"data_source={profile.data_source!r}, "
            f"universe={profile.universe!r}, decision_source={resolved!r}, "
            f"allowed_decision_sources={list(profile.allowed_decision_sources)!r}"
        )
    return resolved
