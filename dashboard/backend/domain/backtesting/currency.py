"""Native-ledger and reporting-currency conversion for backtests."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
import math
from types import MappingProxyType
from typing import Mapping
from zoneinfo import ZoneInfo


class CurrencyContextError(ValueError):
    """Raised when a backtest cannot resolve a valid historical FX rate."""


@dataclass(frozen=True)
class CurrencyContext:
    """Convert a native trading ledger into one reporting currency.

    Rates are expressed as native currency units per one reporting currency
    unit. Missing market dates use only the most recent earlier observation.
    """

    native_currency: str
    reporting_currency: str
    timezone: str
    rates: Mapping[date, float]
    fx_source: str | None = None
    fx_policy: str | None = None

    def __post_init__(self) -> None:
        native = str(self.native_currency or "").strip().upper()
        reporting = str(self.reporting_currency or "").strip().upper()
        if not native or not reporting:
            raise CurrencyContextError("currency codes must be non-empty")
        try:
            ZoneInfo(self.timezone)
        except Exception:
            raise CurrencyContextError("market timezone is invalid") from None

        normalized: dict[date, float] = {}
        for observed_date, raw_rate in self.rates.items():
            if not isinstance(observed_date, date) or isinstance(observed_date, datetime):
                raise CurrencyContextError("FX rate keys must be date values")
            if isinstance(raw_rate, bool):
                raise CurrencyContextError("FX rates must be positive and finite")
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError, OverflowError):
                raise CurrencyContextError("FX rates must be positive and finite") from None
            if not math.isfinite(rate) or rate <= 0:
                raise CurrencyContextError("FX rates must be positive and finite")
            normalized[observed_date] = rate

        if native != reporting and not normalized:
            raise CurrencyContextError("cross-currency backtests require historical FX rates")

        object.__setattr__(self, "native_currency", native)
        object.__setattr__(self, "reporting_currency", reporting)
        object.__setattr__(self, "rates", MappingProxyType(dict(sorted(normalized.items()))))
        object.__setattr__(self, "_rate_dates", tuple(sorted(normalized)))

    def __hash__(self) -> int:
        # frozen=True generates a __hash__ over the declared fields, and `rates`
        # is a MappingProxyType — unhashable, so the generated one raises. Hash
        # the (sorted) rate items instead so the value stays usable as a dict
        # key / set member and remains consistent with the generated __eq__.
        return hash(
            (
                self.native_currency,
                self.reporting_currency,
                self.timezone,
                tuple(self.rates.items()),
                self.fx_source,
                self.fx_policy,
            )
        )

    @classmethod
    def identity(cls, currency: str, timezone: str) -> "CurrencyContext":
        """Create a no-conversion context for a single-currency market."""
        return cls(currency, currency, timezone, {})

    @property
    def requires_conversion(self) -> bool:
        return self.native_currency != self.reporting_currency

    @property
    def fx_pair(self) -> str | None:
        if not self.requires_conversion:
            return None
        return f"{self.reporting_currency}/{self.native_currency}"

    def market_date(self, value: date | datetime) -> date:
        if isinstance(value, datetime):
            if value.tzinfo is not None and value.utcoffset() is not None:
                value = value.astimezone(ZoneInfo(self.timezone))
            return value.date()
        if isinstance(value, date):
            return value
        raise CurrencyContextError("FX lookup requires a date or datetime")

    def rate_at(self, value: date | datetime) -> float:
        """Return the same-day rate or the nearest earlier rate."""
        if not self.requires_conversion:
            return 1.0
        market_date = self.market_date(value)
        position = bisect_right(self._rate_dates, market_date) - 1
        if position < 0:
            raise CurrencyContextError(
                "no historical FX rate is available on or before the first market bar"
            )
        return self.rates[self._rate_dates[position]]

    def to_native(self, amount: float, value: date | datetime) -> float:
        return float(amount) * self.rate_at(value)

    def to_reporting(self, amount: float, value: date | datetime) -> float:
        return float(amount) / self.rate_at(value)

    def reporting_equity_record(self, record: Mapping[str, object]) -> dict:
        """Return a USD-compatible record while preserving native audit values."""
        result = dict(record)
        if not self.requires_conversion:
            return result
        timestamp = result.get("timestamp")
        rate = self.rate_at(timestamp)  # type: ignore[arg-type]
        native_equity = float(result.get("equity") or 0)
        native_cash = float(result.get("cash") or 0)
        native_positions = float(result.get("positions_value") or 0)
        result.update(
            {
                "equity": native_equity / rate,
                "cash": native_cash / rate,
                "positions_value": native_positions / rate,
                "native_equity": native_equity,
                "native_cash": native_cash,
                "native_positions_value": native_positions,
                "fx_rate": rate,
            }
        )
        return result

    def reporting_trade(self, trade: Mapping[str, object]) -> dict:
        """Return a reporting-currency trade and retain the native execution."""
        result = dict(trade)
        if not self.requires_conversion:
            return result
        timestamp = result.get("timestamp")
        rate = self.rate_at(timestamp)  # type: ignore[arg-type]
        quantity = int(result.get("shares") or result.get("quantity") or 0)
        native_price = float(result.get("price") or 0)
        native_value = float(
            result.get("cost")
            or result.get("proceeds")
            or result.get("value")
            or quantity * native_price
        )
        monetary_fields = (
            "price",
            "reference_price",
            "cost",
            "proceeds",
            "value",
            "gross_value",
            "slippage_amount",
            "commission",
            "stamp_duty",
            "transfer_fee",
            "total_fees",
            "net_cash_impact",
        )
        for field in monetary_fields:
            if field not in result:
                continue
            native_amount = float(result[field] or 0)
            result[f"native_{field}"] = native_amount
            result[field] = native_amount / rate
        reporting_value = native_value / rate
        result.update(
            {
                "price": native_price / rate,
                "value": reporting_value,
                "native_price": native_price,
                "native_value": native_value,
                "fx_rate": rate,
            }
        )
        return result

    def reporting_order_event(self, event: Mapping[str, object]) -> dict:
        """Return a reporting-currency order outcome with native audit values."""
        result = dict(event)
        if not self.requires_conversion:
            return result
        timestamp = result.get("timestamp")
        rate = self.rate_at(timestamp)  # type: ignore[arg-type]
        native_price = float(result.get("price") or 0)
        # A rejected order executes no value. Never derive this field from the
        # requested quantity, because that would make an unfilled order look paid.
        native_value = float(result.get("executed_value") or 0)
        monetary_fields = (
            "price",
            "reference_price",
            "executed_value",
            "gross_value",
            "slippage_amount",
            "commission",
            "stamp_duty",
            "transfer_fee",
            "total_fees",
            "net_cash_impact",
        )
        for field in monetary_fields:
            if field not in result:
                continue
            native_amount = float(result[field] or 0)
            result[f"native_{field}"] = native_amount
            result[field] = native_amount / rate
        result.update(
            {
                "price": native_price / rate,
                "executed_value": native_value / rate,
                "native_price": native_price,
                "native_value": native_value,
                "fx_rate": rate,
            }
        )
        return result
