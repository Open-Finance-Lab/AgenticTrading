"""Immutable market-rule contracts used by historical backtests.

The domain deliberately knows nothing about iFinD indicator names or response shapes.
Infrastructure adapters translate vendor observations into this contract before an
execution loop starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


MARKET_RULE_PROFILE_VERSION = "ifind-ashare-closing-rules-v1"
MARKET_RULE_SOURCE = "ifind_http"


class MarketRuleDataError(RuntimeError):
    """Raised when required market-rule observations are unavailable or invalid."""


# Shared by every place that prints or raises a sample of gaps (this module and
# ``ifind_market_rules.py``), so the sample size cannot drift between the
# exception message, the allowed-run warning, and the UI row.
GAP_SAMPLE_LIMIT = 5


@dataclass(frozen=True)
class CorporateActionGap:
    """A signed overnight move no A-share daily limit band can produce.

    The widest band is 20% (STAR/ChiNext), so a larger move is not trading: it
    is most often a 除权除息 (ex-rights/ex-dividend) date seen through
    unadjusted prices, though an unbanded first-week move on a newly listed
    STAR/ChiNext symbol can also cross it legitimately. ``overnight_move`` is
    signed (positive = price rose) because the corporate action itself can go
    either way — a share consolidation raises the unadjusted price, a split,
    dividend, or rights issue lowers it — and mislabeling a rise as "a loss"
    is itself a wrong number shown to the operator.
    """

    symbol: str
    trading_date: date
    overnight_move: Decimal

    def to_metadata(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "date": self.trading_date.isoformat(),
            "overnight_move": float(self.overnight_move),
        }


class CorporateActionGapError(ValueError):
    """Raised when the requested window (dates × symbols) cannot be backtested
    on unadjusted prices as given — a bad-input condition, not a case of the
    rule data itself being unavailable or invalid, so this deliberately does
    NOT subclass ``MarketRuleDataError``. Callers that need to catch it
    alongside that family list both explicitly.

    Carries the gaps rather than only a message because the same observations
    are what a permitted run records in its metadata -- one producer, two
    consumers. The message itself is built lazily in ``__str__`` rather than
    passed to ``super().__init__()``: ``BaseException.__reduce__`` reconstructs
    an exception as ``cls(*self.args)``, so if ``args`` held the formatted
    string, unpickling would call ``__init__`` with a string where ``gaps`` is
    expected. Keeping ``args == (gaps,)`` makes reconstruction call
    ``CorporateActionGapError(gaps)`` again, which is exactly what pickling an
    exception is supposed to do.
    """

    def __init__(self, gaps: Iterable[CorporateActionGap]) -> None:
        gaps = tuple(gaps)
        super().__init__(gaps)
        self.gaps = gaps

    def __str__(self) -> str:
        gaps = self.gaps
        sample = ", ".join(
            f"{gap.symbol} on {gap.trading_date.isoformat()} "
            f"({gap.overnight_move:+.0%})"
            for gap in gaps[:GAP_SAMPLE_LIMIT]
        )
        return (
            f"This window crosses {len(gaps)} date(s) where the overnight "
            f"move exceeds every A-share daily limit band: {sample}"
            f"{' and others' if len(gaps) > GAP_SAMPLE_LIMIT else ''}. That is "
            "usually a 除权除息 (ex-rights/ex-dividend) date, but can also be "
            "an unbanded first-week move on a newly listed STAR/ChiNext "
            "symbol. A-share prices are requested unadjusted, so it would be "
            "charted as a real gain or loss that never happened — in the "
            "agent curve and in the buy-and-hold baseline alike, and across "
            "every symbol in this run, not only the ones listed above. "
            "Choose a window that does not cross it, or set "
            "IFIND_ALLOW_CORPORATE_ACTION_GAPS=1 to run anyway and have the "
            "affected dates recorded on the result."
        )


class ClosingLimitState(str, Enum):
    """Official security state at the daily close."""

    NONE = "none"
    UPPER = "upper"
    LOWER = "lower"


def _as_decimal(value: Any, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _as_datetime(value: Any, *, field: str) -> datetime:
    converter = getattr(value, "to_pydatetime", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def _tick_units(value: Any, price_tick: Any) -> Decimal:
    price = _as_decimal(value, field="price")
    tick = _as_decimal(price_tick, field="price_tick")
    if tick <= 0:
        raise ValueError("price_tick must be positive")
    return (price / tick).to_integral_value(rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class DailyMarketRule:
    """One validated A-share rule observation for one Shanghai market date."""

    symbol: str
    trading_date: date
    suspended: bool
    closing_limit_state: ClosingLimitState = ClosingLimitState.NONE
    official_close_price: Decimal | None = None
    final_bar_timestamp: datetime | None = None
    source: str = MARKET_RULE_SOURCE
    version: str = MARKET_RULE_PROFILE_VERSION

    def __post_init__(self) -> None:
        normalized_symbol = str(self.symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be non-empty")
        object.__setattr__(self, "symbol", normalized_symbol)

        if not isinstance(self.trading_date, date) or isinstance(
            self.trading_date, datetime
        ):
            raise ValueError("trading_date must be a date")
        if not isinstance(self.suspended, bool):
            raise ValueError("suspended must be a boolean")
        try:
            limit_state = ClosingLimitState(self.closing_limit_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("closing_limit_state is invalid") from exc
        object.__setattr__(self, "closing_limit_state", limit_state)

        if not str(self.source or "").strip() or not str(self.version or "").strip():
            raise ValueError("market-rule source and version must be non-empty")

        if self.suspended:
            if limit_state is not ClosingLimitState.NONE:
                raise ValueError("a suspended rule cannot carry a closing limit state")
            if self.official_close_price is not None:
                raise ValueError("a suspended rule cannot carry an official close")
            if self.final_bar_timestamp is not None:
                raise ValueError("a suspended rule cannot carry a final bar timestamp")
            return

        close = _as_decimal(self.official_close_price, field="official_close_price")
        if close <= 0:
            raise ValueError("official_close_price must be positive")
        if self.final_bar_timestamp is None:
            # The official feed says this security traded, but this run holds no
            # intraday series for the day — a symbol whose hourly history starts
            # late, or has a gap, inside a universe that traded through it. The
            # observation is still worth carrying for the audit, but it can never
            # gate: ``closing_gate_effective`` needs a bar to compare against,
            # and no order can execute on a bar that does not exist either.
            object.__setattr__(self, "official_close_price", close)
            return
        timestamp = _as_datetime(
            self.final_bar_timestamp, field="final_bar_timestamp"
        )
        if timestamp.date() != self.trading_date:
            raise ValueError("final_bar_timestamp must fall on trading_date")
        object.__setattr__(self, "official_close_price", close)
        object.__setattr__(self, "final_bar_timestamp", timestamp)

    def closing_gate_effective(
        self,
        *,
        timestamp: Any,
        reference_price: Any,
        price_tick: Any,
    ) -> bool:
        """Return whether this order is at the verified end-of-day limit close."""
        if self.suspended or self.closing_limit_state is ClosingLimitState.NONE:
            return False
        if self.final_bar_timestamp is None:
            # Unverifiable rather than false: without the day's final bar there
            # is no observation proving this order sits at the limit close.
            return False
        current = _as_datetime(timestamp, field="timestamp")
        if current != self.final_bar_timestamp:
            return False
        return _tick_units(reference_price, price_tick) == _tick_units(
            self.official_close_price, price_tick
        )

    def to_audit(self, *, closing_gate_effective: bool = False) -> dict[str, object]:
        """Return stable JSON-safe fields for an order-event audit record."""
        return {
            "market_rule_date": self.trading_date.isoformat(),
            "market_rule_suspended": self.suspended,
            "market_rule_closing_limit_state": self.closing_limit_state.value,
            "market_rule_official_close": (
                float(self.official_close_price)
                if self.official_close_price is not None
                else None
            ),
            "market_rule_closing_gate_effective": bool(closing_gate_effective),
            "market_rule_source": self.source,
            "market_rule_version": self.version,
        }


class MarketRuleCalendar:
    """Read-only lookup of validated rules keyed by symbol and market date."""

    def __init__(
        self,
        rules: Iterable[DailyMarketRule],
        *,
        corporate_action_gaps: Iterable[CorporateActionGap] = (),
    ) -> None:
        # Carried on the calendar rather than raised, because these reach here
        # only on the permitted path: a run that was allowed to cross an
        # ex-rights date still has to say so on its own result, where the
        # subprocess's stdout never reaches anyone.
        self.corporate_action_gaps = tuple(corporate_action_gaps)
        normalized: dict[tuple[str, date], DailyMarketRule] = {}
        for rule in rules:
            if not isinstance(rule, DailyMarketRule):
                raise TypeError("market-rule calendar accepts DailyMarketRule values")
            key = (rule.symbol, rule.trading_date)
            if key in normalized:
                raise MarketRuleDataError(
                    "Market rule data unavailable: duplicate symbol-date observation"
                )
            normalized[key] = rule
        if not normalized:
            raise MarketRuleDataError("Market rule data unavailable: calendar is empty")
        self._rules: Mapping[tuple[str, date], DailyMarketRule] = MappingProxyType(
            normalized
        )

    def rule_for(self, symbol: str, trading_date: date) -> DailyMarketRule:
        key = (str(symbol or "").strip().upper(), trading_date)
        try:
            return self._rules[key]
        except KeyError:
            raise MarketRuleDataError(
                "Market rule data unavailable: missing symbol-date observation"
            ) from None

    def rule_for_timestamp(self, symbol: str, timestamp: Any) -> DailyMarketRule:
        current = _as_datetime(timestamp, field="timestamp")
        return self.rule_for(symbol, current.date())

    def __len__(self) -> int:
        return len(self._rules)

    def to_metadata(self) -> dict[str, object]:
        sample = next(iter(self._rules.values()))
        metadata: dict[str, object] = {
            "enabled": True,
            "source": sample.source,
            "version": sample.version,
            "observations": len(self._rules),
            "scope": "full_day_suspension_and_closing_limits_and_corporate_action_gaps",
        }
        # Omitted when empty rather than written as [] — the overwhelming
        # majority of runs cross no ex-rights date, and a key present on every
        # run trains the reader to skip it.
        if self.corporate_action_gaps:
            metadata["corporate_action_gaps"] = [
                gap.to_metadata() for gap in self.corporate_action_gaps
            ]
        return metadata
