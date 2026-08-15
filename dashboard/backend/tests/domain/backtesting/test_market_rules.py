"""Domain tests for immutable A-share market-rule observations."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from dashboard.backend.domain.backtesting.market_rules import (
    ClosingLimitState,
    DailyMarketRule,
    MarketRuleCalendar,
    MarketRuleDataError,
)


CN = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 7, 13)
FINAL_BAR = datetime(2026, 7, 13, 15, 0, tzinfo=CN)


def active_rule(state=ClosingLimitState.NONE):
    return DailyMarketRule(
        symbol="000725.sz",
        trading_date=DAY,
        suspended=False,
        closing_limit_state=state,
        official_close_price=Decimal("6.83"),
        final_bar_timestamp=FINAL_BAR,
    )


def test_active_rule_is_normalized_immutable_and_auditable():
    rule = active_rule(ClosingLimitState.LOWER)

    assert rule.symbol == "000725.SZ"
    assert rule.official_close_price == Decimal("6.83")
    assert rule.to_audit(closing_gate_effective=True) == {
        "market_rule_date": "2026-07-13",
        "market_rule_suspended": False,
        "market_rule_closing_limit_state": "lower",
        "market_rule_official_close": 6.83,
        "market_rule_closing_gate_effective": True,
        "market_rule_source": "ifind_http",
        "market_rule_version": "ifind-ashare-closing-rules-v1",
    }
    with pytest.raises(FrozenInstanceError):
        rule.suspended = True


def test_suspended_rule_carries_no_fake_price_or_bar():
    rule = DailyMarketRule(
        symbol="688981.SH",
        trading_date=date(2025, 9, 1),
        suspended=True,
    )

    assert rule.official_close_price is None
    assert rule.final_bar_timestamp is None
    assert rule.closing_limit_state is ClosingLimitState.NONE


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"official_close_price": 0}, "positive"),
        ({"final_bar_timestamp": datetime(2026, 7, 13, 15)}, "timezone-aware"),
        (
            {"final_bar_timestamp": datetime(2026, 7, 14, 15, tzinfo=CN)},
            "trading_date",
        ),
    ],
)
def test_active_rule_rejects_invalid_price_or_final_bar(kwargs, match):
    values = {
        "symbol": "000725.SZ",
        "trading_date": DAY,
        "suspended": False,
        "official_close_price": Decimal("6.83"),
        "final_bar_timestamp": FINAL_BAR,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        DailyMarketRule(**values)


def test_closing_gate_is_tick_safe_and_never_applies_to_earlier_bar():
    rule = active_rule(ClosingLimitState.LOWER)

    assert rule.closing_gate_effective(
        timestamp=FINAL_BAR,
        reference_price=6.8300000001,
        price_tick=0.01,
    )
    assert not rule.closing_gate_effective(
        timestamp=datetime(2026, 7, 13, 14, 0, tzinfo=CN),
        reference_price=6.83,
        price_tick=0.01,
    )
    assert not rule.closing_gate_effective(
        timestamp=FINAL_BAR,
        reference_price=6.84,
        price_tick=0.01,
    )


def test_calendar_is_read_only_and_requires_exact_symbol_date():
    rule = active_rule()
    calendar = MarketRuleCalendar([rule])

    assert calendar.rule_for("000725.sz", DAY) is rule
    assert calendar.rule_for_timestamp("000725.SZ", FINAL_BAR) is rule
    assert calendar.to_metadata() == {
        "enabled": True,
        "source": "ifind_http",
        "version": "ifind-ashare-closing-rules-v1",
        "observations": 1,
        "scope": "full_day_suspension_and_closing_limits",
    }
    with pytest.raises(MarketRuleDataError, match="missing"):
        calendar.rule_for("600519.SH", DAY)


def test_calendar_rejects_duplicate_observation():
    rule = active_rule()
    with pytest.raises(MarketRuleDataError, match="duplicate"):
        MarketRuleCalendar([rule, rule])
