"""Explicit time-frequency contracts for market-data and trading loops.

The backtest historically used one ``timeframe`` value for several different
concepts.  Minute-data support needs those concepts to be explicit:

* ``source_timeframe``: the resolution fetched from a provider;
* ``decision_timeframe`` / ``decision_frequency``: the completed bar and
  cadence supplied to a strategy;
* ``execution_timeframe``: the resolution used to model a fill; and
* ``valuation_frequency``: the resolution used for mark-to-market updates.

This module is deliberately provider-neutral.  It validates configuration but
does not aggregate bars; aggregation belongs to the backtesting domain layer.
"""

from __future__ import annotations

from dataclasses import dataclass


class FrequencyConfigError(ValueError):
    """Raised when a frequency value is unknown or internally inconsistent."""


SUPPORTED_BAR_TIMEFRAMES = ("1m", "5m", "60m")

_BAR_TIMEFRAME_ALIASES = {
    "1m": "1m",
    "1min": "1m",
    "1minute": "1m",
    "minute": "1m",
    "min": "1m",
    "5m": "5m",
    "5min": "5m",
    "5minute": "5m",
    "60m": "60m",
    "60min": "60m",
    "60minute": "60m",
    "1h": "60m",
    "1hour": "60m",
    "hour": "60m",
    "hourly": "60m",
}

_DECISION_FREQUENCY_ALIASES = {
    "1h": "1h",
    "60m": "1h",
    "60min": "1h",
    "1hour": "1h",
    "hour": "1h",
    "hourly": "1h",
}


def _clean(value: str, field_name: str) -> str:
    if value is None:
        raise FrequencyConfigError(f"{field_name} must be non-empty")
    cleaned = str(value).strip().lower()
    if not cleaned:
        raise FrequencyConfigError(f"{field_name} must be non-empty")
    return cleaned


def normalize_bar_timeframe(value: str) -> str:
    """Return the canonical bar timeframe used by the application."""
    cleaned = _clean(value, "bar timeframe")
    try:
        return _BAR_TIMEFRAME_ALIASES[cleaned]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_BAR_TIMEFRAMES)
        raise FrequencyConfigError(
            f"Unsupported bar timeframe {value!r}; expected one of {allowed}"
        ) from exc


def normalize_decision_frequency(value: str) -> str:
    """Return the canonical strategy decision cadence."""
    cleaned = _clean(value, "decision frequency")
    try:
        return _DECISION_FREQUENCY_ALIASES[cleaned]
    except KeyError as exc:
        raise FrequencyConfigError(
            f"Unsupported decision frequency {value!r}; expected '1h'"
        ) from exc


def timeframe_minutes(value: str) -> int:
    """Return the duration, in minutes, of a canonical bar timeframe."""
    canonical = normalize_bar_timeframe(value)
    return {"1m": 1, "5m": 5, "60m": 60}[canonical]


def decision_frequency_minutes(value: str) -> int:
    """Return the duration, in minutes, of a canonical decision cadence."""
    return {"1h": 60}[normalize_decision_frequency(value)]


def verify_source_timeframe(
    requested: str,
    reported: str,
    *,
    evidence: str,
) -> str:
    """Return the reported canonical timeframe or fail on provider drift.

    A run must not be labelled as minute-sourced when a provider ignored the
    requested resolution. ``evidence`` names the runtime signal in the error
    (for example ``configured`` or ``fetch``); it is not user configuration.
    """
    expected = normalize_bar_timeframe(requested)
    actual = normalize_bar_timeframe(reported)
    if actual != expected:
        raise FrequencyConfigError(
            f"{evidence} source timeframe mismatch: requested {expected}, "
            f"reported {actual}"
        )
    return actual


@dataclass(frozen=True)
class TradingFrequency:
    """Validated frequency contract for one market profile.

    The default remains the legacy all-hourly configuration.  Use
    :meth:`minute_source_hourly_decisions` for the Phase 0/1 target contract.
    """

    source_timeframe: str = "60m"
    decision_timeframe: str = "60m"
    decision_frequency: str = "1h"
    execution_timeframe: str = "60m"
    valuation_frequency: str = "60m"

    def __post_init__(self) -> None:
        source = normalize_bar_timeframe(self.source_timeframe)
        decision = normalize_bar_timeframe(self.decision_timeframe)
        decision_frequency = normalize_decision_frequency(self.decision_frequency)
        execution = normalize_bar_timeframe(self.execution_timeframe)
        valuation = normalize_bar_timeframe(self.valuation_frequency)

        if timeframe_minutes(source) > timeframe_minutes(decision):
            raise FrequencyConfigError(
                "source_timeframe cannot be coarser than decision_timeframe"
            )
        if decision_frequency_minutes(decision_frequency) != timeframe_minutes(decision):
            raise FrequencyConfigError(
                "decision_frequency must match decision_timeframe"
            )

        object.__setattr__(self, "source_timeframe", source)
        object.__setattr__(self, "decision_timeframe", decision)
        object.__setattr__(self, "decision_frequency", decision_frequency)
        object.__setattr__(self, "execution_timeframe", execution)
        object.__setattr__(self, "valuation_frequency", valuation)

    @classmethod
    def minute_source_hourly_decisions(cls) -> "TradingFrequency":
        """Return the target 5m-source / 1h-decision Phase 0/1 contract."""
        return cls(
            source_timeframe="5m",
            decision_timeframe="60m",
            decision_frequency="1h",
            execution_timeframe="5m",
            valuation_frequency="5m",
        )

    def to_metadata(self) -> dict[str, str]:
        """Return a stable JSON-safe representation for run metadata."""
        return {
            "source_timeframe": self.source_timeframe,
            "decision_timeframe": self.decision_timeframe,
            "decision_frequency": self.decision_frequency,
            "execution_timeframe": self.execution_timeframe,
            "valuation_frequency": self.valuation_frequency,
        }


def build_verified_intraday_contract(
    *,
    source_timeframe: str,
    decision_timeframe: str,
    decision_frequency: str,
) -> dict[str, str]:
    """Build the fixed runtime contract for finer-source hourly backtests.

    Execution and valuation deliberately inherit the source resolution. The
    aggregation and fill policy are constants, not strategy options. Returning
    ``verification_status=verified`` attests that these invariants were checked
    from the runtime's effective timeframes before result persistence.
    """
    source = normalize_bar_timeframe(source_timeframe)
    decision = normalize_bar_timeframe(decision_timeframe)
    if timeframe_minutes(source) >= timeframe_minutes(decision):
        raise FrequencyConfigError(
            "Verified intraday contracts require source_timeframe to be finer "
            "than decision_timeframe"
        )
    contract = TradingFrequency(
        source_timeframe=source,
        decision_timeframe=decision,
        decision_frequency=decision_frequency,
        execution_timeframe=source,
        valuation_frequency=source,
    ).to_metadata()
    contract.update(
        {
            "aggregation": "session_anchored_completed_bars",
            "fill_policy": "next_source_bar_open",
            "verification_status": "verified",
        }
    )
    return contract
