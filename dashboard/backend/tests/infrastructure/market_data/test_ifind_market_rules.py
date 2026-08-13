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


def test_active_symbol_date_without_hourly_bars_is_kept_ungated():
    """A symbol-date gap must not abort a run the loaders already tolerate.

    ``required_dates`` is the union across the universe, so one symbol's short
    or gappy hourly series drags in dates another symbol traded through. The
    engine deliberately tolerates that — 50-bar minimum, common-index start,
    80%-coverage bar filter — so failing the whole load here would reject
    universes that run fine today. Without a bar there is nothing to align
    against and nothing to execute, so the rule is recorded ungated.
    """
    payload = history_payload()
    # 688981 traded on DAY_2 per the official feed, but its hourly series
    # stops at DAY_1.
    payload["tables"][0]["table"]["close"][1] = 103.0
    payload["tables"][0]["table"]["ths_trading_status_stock"][1] = "交易"
    payload["tables"][0]["table"]["ths_up_and_down_status_stock"][1] = "涨停"

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=bars(),
        fetch_basic_status=lambda *_args: pytest.fail("unexpected supplement"),
    )

    unaligned = calendar.rule_for("688981.SH", DAY_2)
    assert not unaligned.suspended
    assert unaligned.official_close_price == 103
    assert unaligned.final_bar_timestamp is None
    # No bar means no closing gate can ever be proven, and no order can execute
    # on a bar that does not exist.
    assert not unaligned.closing_gate_effective(
        timestamp=datetime(2025, 9, 1, 15, tzinfo=CN),
        reference_price=103.0,
        price_tick=0.01,
    )
    # The symbol's own bar dates keep the full alignment check.
    assert calendar.rule_for("688981.SH", DAY_1).final_bar_timestamp == datetime(
        2025, 8, 29, 15, tzinfo=CN
    )


def test_close_mismatch_still_fails_on_a_date_the_symbol_has_bars_for():
    """The ungated escape hatch must not swallow a real contract break."""
    payload = history_payload()
    payload["tables"][1]["table"]["close"][1] = 1495.0

    with pytest.raises(MarketRuleDataError, match="does not match final hourly bar"):
        adapt(payload)


def test_no_supplement_round_trip_when_the_row_already_proves_suspension():
    """Suspension settles the row; the supplement cannot change the outcome.

    The supplement is one blocking vendor call per date, and a suspended
    security is exactly what leaves the other field blank — so the common case
    was paying a sequential round trip for an answer already in hand.
    """
    payload = history_payload()
    payload["tables"][0]["table"]["ths_trading_status_stock"][1] = "停牌"
    payload["tables"][0]["table"]["ths_up_and_down_status_stock"][1] = None

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=bars(),
        fetch_basic_status=lambda *_args: pytest.fail("unexpected supplement"),
    )

    assert calendar.rule_for("688981.SH", DAY_2).suspended


def test_unrecognised_vendor_status_names_the_value_symbol_and_date():
    """The gate stays keyed on exact strings, so the message has to diagnose.

    'unknown closing limit status' alone cannot tell an operator whether iFinD
    added a value, the account answered in English, or the field arrived blank.
    """
    payload = history_payload()
    payload["tables"][1]["table"]["ths_up_and_down_status_stock"][1] = "未知状态"

    with pytest.raises(MarketRuleDataError) as excinfo:
        adapt(payload)

    message = str(excinfo.value)
    assert "unknown closing limit status" in message
    assert "未知状态" in message
    assert "600519.SH" in message
    assert "2025-09-01" in message


def test_unexpected_trading_status_is_reported_apart_from_a_missing_close():
    payload = history_payload()
    payload["tables"][1]["table"]["ths_trading_status_stock"][1] = "Trading"

    with pytest.raises(MarketRuleDataError, match="unexpected trading status"):
        adapt(payload)


def test_unadjusted_ex_rights_gap_is_reported_not_swallowed(capsys):
    """Unadjusted prices put corporate actions straight into the equity curve.

    Both feeds are requested unadjusted so the audit can show the official close
    a limit was set against, and nothing in the backend applies dividends or
    splits. A 10-for-3 bonus issue therefore reads as a ~23% overnight loss in
    the agent curve and the buy-and-hold baseline alike, with no offsetting cash
    credit. This does not correct it — it stops it being invisible.
    """
    payload = history_payload()
    payload["tables"][1]["table"]["close"][1] = 1140.00
    local_bars = bars()
    local_bars["600519.SH"] = frame([
        (datetime(2025, 8, 29, 15, tzinfo=CN), 1480.0),
        (datetime(2025, 9, 1, 14, tzinfo=CN), 1138.0),
        (datetime(2025, 9, 1, 15, tzinfo=CN), 1140.0),
    ])

    response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=local_bars,
        fetch_basic_status=suspended_supplement,
    )

    warning = capsys.readouterr().out
    assert "unadjusted" in warning
    assert "600519.SH 2025-09-01" in warning


def test_a_gap_across_a_suspension_is_not_reported_as_a_corporate_action():
    """A halt legitimately lets the price gap when the security resumes."""
    payload = history_payload()
    payload["tables"][1]["table"]["ths_trading_status_stock"][0] = "停牌"
    payload["tables"][1]["table"]["ths_up_and_down_status_stock"][0] = "停牌"
    payload["tables"][1]["table"]["close"][0] = None
    payload["tables"][1]["table"]["close"][1] = 1140.00
    local_bars = bars()
    local_bars["600519.SH"] = frame([
        (datetime(2025, 9, 1, 14, tzinfo=CN), 1138.0),
        (datetime(2025, 9, 1, 15, tzinfo=CN), 1140.0),
    ])

    calendar = response_to_market_rules(
        payload,
        expected_symbols=SYMBOLS,
        required_dates=(DAY_1, DAY_2),
        bars_by_symbol=local_bars,
        fetch_basic_status=suspended_supplement,
    )

    assert calendar.rule_for("600519.SH", DAY_1).suspended
