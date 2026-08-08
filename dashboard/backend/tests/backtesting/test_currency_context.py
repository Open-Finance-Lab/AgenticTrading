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


def test_reporting_order_event_converts_only_executed_value():
    context = make_context()
    timestamp = datetime(2026, 4, 1, 10, 30, tzinfo=CN)

    filled = context.reporting_order_event({
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": 100,
        "executed_shares": 100,
        "price": 700,
        "executed_value": 70_000,
        "status": "filled",
        "reason": "",
    })
    rejected = context.reporting_order_event({
        "timestamp": timestamp,
        "symbol": "600519.SH",
        "side": "BUY",
        "requested_shares": 100,
        "executed_shares": 0,
        "price": 700,
        "executed_value": 0,
        "status": "rejected",
        "reason": "insufficient_cash_for_lot",
    })

    assert filled["price"] == pytest.approx(100)
    assert filled["executed_value"] == pytest.approx(10_000)
    assert filled["native_price"] == pytest.approx(700)
    assert filled["native_value"] == pytest.approx(70_000)
    assert filled["fx_rate"] == pytest.approx(7.0)
    assert rejected["executed_value"] == 0
    assert rejected["native_value"] == 0


def test_reporting_trade_converts_a_share_cost_breakdown_and_preserves_cny_audit():
    context = make_context()
    timestamp = datetime(2026, 4, 1, 10, 30, tzinfo=CN)
    trade = context.reporting_trade(
        {
            "timestamp": timestamp,
            "symbol": "600519.SH",
            "side": "BUY",
            "shares": 100,
            "price": 100.05,
            "reference_price": 100.0,
            "cost": 10005.0,
            "gross_value": 10005.0,
            "slippage_amount": 5.0,
            "commission": 5.0,
            "stamp_duty": 0.0,
            "transfer_fee": 0.1,
            "total_fees": 5.1,
            "net_cash_impact": -10010.1,
        }
    )

    assert trade["price"] == pytest.approx(100.05 / 7.0)
    assert trade["reference_price"] == pytest.approx(100.0 / 7.0)
    assert trade["total_fees"] == pytest.approx(5.1 / 7.0)
    assert trade["net_cash_impact"] == pytest.approx(-10010.1 / 7.0)
    assert trade["native_price"] == pytest.approx(100.05)
    assert trade["native_reference_price"] == pytest.approx(100.0)
    assert trade["native_total_fees"] == pytest.approx(5.1)
    assert trade["native_net_cash_impact"] == pytest.approx(-10010.1)
    assert trade["native_value"] == pytest.approx(10005.0)


def test_reporting_order_event_converts_a_share_cost_fields():
    context = make_context()
    timestamp = datetime(2026, 4, 1, 10, 30, tzinfo=CN)
    event = context.reporting_order_event(
        {
            "timestamp": timestamp,
            "symbol": "600519.SH",
            "side": "SELL",
            "requested_shares": 100,
            "executed_shares": 100,
            "price": 99.95,
            "reference_price": 100.0,
            "executed_value": 9995.0,
            "gross_value": 9995.0,
            "slippage_amount": 5.0,
            "commission": 5.0,
            "stamp_duty": 5.0,
            "transfer_fee": 0.1,
            "total_fees": 10.1,
            "net_cash_impact": 9984.9,
            "status": "filled",
            "reason": "",
        }
    )

    assert event["executed_value"] == pytest.approx(9995.0 / 7.0)
    assert event["total_fees"] == pytest.approx(10.1 / 7.0)
    assert event["net_cash_impact"] == pytest.approx(9984.9 / 7.0)
    assert event["native_executed_value"] == pytest.approx(9995.0)
    assert event["native_total_fees"] == pytest.approx(10.1)
    assert event["native_net_cash_impact"] == pytest.approx(9984.9)


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

    order_event = {
        "timestamp": datetime(2026, 4, 1),
        "price": 100,
        "executed_value": 10_000,
    }
    assert context.reporting_order_event(order_event) == order_event


def test_currency_context_is_hashable():
    """frozen=True generates a __hash__ that would raise on the proxied rates."""
    from dashboard.backend.domain.backtesting.currency import CurrencyContext

    context = CurrencyContext(
        native_currency="CNY",
        reporting_currency="USD",
        timezone="Asia/Shanghai",
        rates={date(2026, 4, 1): 7.0},
    )
    twin = CurrencyContext(
        native_currency="CNY",
        reporting_currency="USD",
        timezone="Asia/Shanghai",
        rates={date(2026, 4, 1): 7.0},
    )

    assert hash(context) == hash(twin)
    assert len({context, twin}) == 1
    assert context == twin


def test_identity_contexts_hash_independently_of_rates():
    from dashboard.backend.domain.backtesting.currency import CurrencyContext

    usd = CurrencyContext.identity("USD", "US/Eastern")
    cny = CurrencyContext.identity("CNY", "Asia/Shanghai")

    assert len({usd, cny, CurrencyContext.identity("USD", "US/Eastern")}) == 2
