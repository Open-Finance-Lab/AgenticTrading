"""Sanitized adapter tests for official iFinD A-share market rules."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting.market_rules import (
    ClosingLimitState,
    MarketRuleDataError,
)
from dashboard.backend.infrastructure.market_data.ifind_market_rules import (
    response_to_market_rules,
)


CN = ZoneInfo("Asia/Shanghai")
DAY_1 = date(2025, 8, 29)
DAY_2 = date(2025, 9, 1)
SYMBOLS = ("688981.SH", "600519.SH")


def frame(rows):
    return pd.DataFrame(
        {"close": [price for _timestamp, price in rows]},
        index=pd.DatetimeIndex(
            [timestamp for timestamp, _price in rows],
            name="timestamp",
        ),
    )


def bars():
    return {
        "688981.SH": frame([
            (datetime(2025, 8, 29, 14, tzinfo=CN), 102.0),
            (datetime(2025, 8, 29, 15, tzinfo=CN), 101.5),
        ]),
        "600519.SH": frame([
            (datetime(2025, 8, 29, 15, tzinfo=CN), 1480.0),
            (datetime(2025, 9, 1, 14, tzinfo=CN), 1488.0),
            (datetime(2025, 9, 1, 15, tzinfo=CN), 1490.0),
        ]),
    }


def history_payload():
    return {
        "errorcode": 0,
        "tables": [
            {
                "thscode": "688981.SH",
                "time": ["2025-08-29", "2025-09-01"],
                "table": {
                    "close": [101.5, None],
                    "ths_trading_status_stock": ["交易", None],
                    "ths_up_and_down_status_stock": ["非涨跌停", None],
                },
            },
            {
                "thscode": "600519.SH",
                "time": ["2025-08-29", "2025-09-01"],
                "table": {
                    "close": [1480.0, 1490.0],
                    "ths_trading_status_stock": ["交易", "交易"],
                    "ths_up_and_down_status_stock": ["非涨跌停", "涨停"],
                },
            },
        ],
    }


def suspended_supplement(symbols, trading_date):
    assert symbols == ("688981.SH",)
    assert trading_date == DAY_2
    return {
        "errorcode": 0,
        "tables": [
            {
                "thscode": "688981.SH",
                "table": {
                    "ths_trading_status_stock": [
                        "Important announcement, suspended from 2025-09-01"
                    ],
                    "ths_up_and_down_status_stock": ["停牌"],
                },
            }
        ],
    }


def adapt(payload=None, **kwargs):
    return response_to_market_rules(
        history_payload() if payload is None else payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=bars(),
        fetch_basic_status=suspended_supplement,
        price_tick=0.01,
        **kwargs,
    )


def test_normalizes_active_suspended_and_closing_limit_observations():
    calendar = adapt()

    normal = calendar.rule_for("688981.SH", DAY_1)
    suspended = calendar.rule_for("688981.SH", DAY_2)
    upper = calendar.rule_for("600519.SH", DAY_2)
    assert not normal.suspended
    assert normal.closing_limit_state is ClosingLimitState.NONE
    assert suspended.suspended
    assert suspended.official_close_price is None
    assert upper.closing_limit_state is ClosingLimitState.UPPER
    assert upper.official_close_price == 1490
    assert upper.final_bar_timestamp == datetime(2025, 9, 1, 15, tzinfo=CN)


def test_does_not_call_basic_supplement_when_history_is_complete():
    payload = history_payload()
    payload["tables"][0]["table"]["close"][1] = 103.0
    payload["tables"][0]["table"]["ths_trading_status_stock"][1] = "交易"
    payload["tables"][0]["table"]["ths_up_and_down_status_stock"][1] = "非涨跌停"
    local_bars = bars()
    local_bars["688981.SH"] = pd.concat([
        local_bars["688981.SH"],
        frame([(datetime(2025, 9, 1, 15, tzinfo=CN), 103.0)]),
    ])

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=local_bars,
        fetch_basic_status=lambda *_args: pytest.fail("unexpected supplement"),
    )

    assert len(calendar) == 4


def test_supplements_blank_status_without_discarding_historical_close():
    payload = history_payload()
    payload["tables"][0]["table"]["close"][1] = 103.0
    local_bars = bars()
    local_bars["688981.SH"] = pd.concat([
        local_bars["688981.SH"],
        frame([(datetime(2025, 9, 1, 15, tzinfo=CN), 103.0)]),
    ])

    def active_supplement(symbols, trading_date):
        assert symbols == ("688981.SH",)
        assert trading_date == DAY_2
        return {
            "errorcode": 0,
            "tables": [
                {
                    "thscode": "688981.SH",
                    "table": {
                        "ths_trading_status_stock": ["交易"],
                        "ths_up_and_down_status_stock": ["非涨跌停"],
                    },
                }
            ],
        }

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=local_bars,
        fetch_basic_status=active_supplement,
    )

    rule = calendar.rule_for("688981.SH", DAY_2)
    assert not rule.suspended
    assert rule.official_close_price == 103


def test_supplements_only_the_blank_limit_status_field():
    payload = history_payload()
    payload["tables"][1]["table"]["ths_up_and_down_status_stock"][1] = None

    def combined_supplement(symbols, trading_date):
        assert symbols == ("688981.SH", "600519.SH")
        assert trading_date == DAY_2
        return {
            "errorcode": 0,
            "tables": [
                {
                    "thscode": "688981.SH",
                    "table": {
                        "ths_trading_status_stock": [
                            "Important announcement, suspended from 2025-09-01"
                        ],
                        "ths_up_and_down_status_stock": ["停牌"],
                    },
                },
                {
                    "thscode": "600519.SH",
                    "table": {
                        "ths_trading_status_stock": ["交易"],
                        "ths_up_and_down_status_stock": ["涨停"],
                    },
                }
            ],
        }

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=bars(),
        fetch_basic_status=combined_supplement,
    )

    rule = calendar.rule_for("600519.SH", DAY_2)
    assert rule.closing_limit_state is ClosingLimitState.UPPER
    assert rule.official_close_price == 1490


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda payload: payload["tables"].pop(),
            "missing symbols",
        ),
        (
            lambda payload: payload["tables"][1]["table"].__setitem__(
                "ths_up_and_down_status_stock", ["非涨跌停"]
            ),
            "lengths differ",
        ),
        (
            lambda payload: payload["tables"][1]["table"][
                "ths_up_and_down_status_stock"
            ].__setitem__(1, "unknown"),
            "unknown closing limit status",
        ),
        (
            lambda payload: payload["tables"][1]["table"]["close"].__setitem__(
                1, 1490.02
            ),
            "does not match",
        ),
        (
            lambda payload: (
                payload["tables"][1]["table"]["close"].__setitem__(1, 1490.004)
            ),
            "price tick",
        ),
    ],
)
def test_rejects_incomplete_or_misaligned_official_data(mutate, match):
    payload = history_payload()
    mutate(payload)

    with pytest.raises(MarketRuleDataError, match=match):
        adapt(payload)


def test_rejects_supplement_that_does_not_explicitly_confirm_suspension():
    def ambiguous_supplement(symbols, trading_date):
        payload = suspended_supplement(symbols, trading_date)
        payload["tables"][0]["table"] = {
            "ths_trading_status_stock": [None],
            "ths_up_and_down_status_stock": [None],
        }
        return payload

    with pytest.raises(MarketRuleDataError, match="unknown closing limit status"):
        response_to_market_rules(
            history_payload(),
            expected_symbols=SYMBOLS,
            required_dates=(DAY_1, DAY_2),
            bars_by_symbol=bars(),
            fetch_basic_status=ambiguous_supplement,
        )
