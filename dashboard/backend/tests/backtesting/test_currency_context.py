"""Historical currency accounting for native-market backtests."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from dashboard.backend.domain.backtesting.currency import (
    CurrencyContext,
    CurrencyContextError,
)


CN = ZoneInfo("Asia/Shanghai")


def make_context() -> CurrencyContext:
    return CurrencyContext(
        native_currency="CNY",
        reporting_currency="USD",
        timezone="Asia/Shanghai",
        rates={date(2026, 4, 1): 7.0, date(2026, 4, 3): 7.1},
        fx_source="ifind_history_currency_conversion",
        fx_policy="daily_implied_median_forward_fill",
    )


def test_initial_capital_and_unchanged_native_equity_convert_to_usd():
    context = make_context()
    first_bar = datetime(2026, 4, 1, 10, 30, tzinfo=CN)
    later_bar = datetime(2026, 4, 3, 10, 30, tzinfo=CN)

    assert context.to_native(1_000, first_bar) == pytest.approx(7_000)
    assert context.to_reporting(7_000, first_bar) == pytest.approx(1_000)
    assert context.to_reporting(7_000, later_bar) == pytest.approx(985.91549296)


def test_missing_date_uses_previous_rate_and_never_future_rate():
    context = make_context()

    assert context.rate_at(date(2026, 4, 2)) == pytest.approx(7.0)
    with pytest.raises(CurrencyContextError, match="first market bar"):
        context.rate_at(date(2026, 3, 31))


def test_reporting_records_preserve_native_equity_and_trade_values():
    context = make_context()
    timestamp = datetime(2026, 4, 1, 10, 30, tzinfo=CN)

    equity = context.reporting_equity_record(
        {
            "timestamp": timestamp,
            "equity": 7_000,
            "cash": 5_600,
            "positions_value": 1_400,
        }
    )
    trade = context.reporting_trade(
        {
            "timestamp": timestamp,
            "symbol": "600519.SH",
            "side": "BUY",
            "shares": 1,
            "price": 1_400,
            "cost": 1_400,
        }
    )

    assert equity["equity"] == pytest.approx(1_000)
    assert equity["native_equity"] == pytest.approx(7_000)
    assert equity["fx_rate"] == pytest.approx(7.0)
    assert trade["price"] == pytest.approx(200)
    assert trade["value"] == pytest.approx(200)
    assert trade["native_price"] == pytest.approx(1_400)
    assert trade["native_value"] == pytest.approx(1_400)
    assert trade["fx_rate"] == pytest.approx(7.0)


def test_identity_context_keeps_legacy_usd_schema():
    context = CurrencyContext.identity("USD", "US/Eastern")
    record = {
        "timestamp": datetime(2026, 4, 1),
        "equity": 1_000,
        "cash": 1_000,
        "positions_value": 0,
    }

    assert context.reporting_equity_record(record) == record
    assert context.rate_at(date(2026, 4, 1)) == 1.0
