"""Strict iFinD tables-to-OHLCV conversion tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, time, timedelta
from math import inf, nan
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6_SYMBOLS,
)


CN = ZoneInfo("Asia/Shanghai")
START = datetime(2026, 4, 1, tzinfo=CN)
END = datetime(2026, 5, 1, tzinfo=CN)
BAR_END_TIMES = (time(10, 30), time(11, 30), time(14, 0), time(15, 0))


def trading_timestamps(count: int) -> list[str]:
    values = []
    current = START.date()
    while len(values) < count:
        if current.weekday() < 5:
            for bar_time in BAR_END_TIMES:
                values.append(
                    datetime.combine(current, bar_time).strftime("%Y-%m-%d %H:%M:%S")
                )
                if len(values) == count:
                    break
        current += timedelta(days=1)
    return values


def make_table(symbol: str, count: int = 52) -> dict[str, object]:
    base = 100 + A_SHARE_DEMO_6_SYMBOLS.index(symbol) * 10
    opens = [base + index * 0.1 for index in range(count)]
    return {
        "thscode": symbol,
        "time": trading_timestamps(count),
        "table": {
            "open": [f"{value:.2f}" for value in opens],
            "high": [f"{value + 1:.2f}" for value in opens],
            "low": [f"{value - 1:.2f}" for value in opens],
            "close": [f"{value + 0.5:.2f}" for value in opens],
            "volume": [str(10_000 + index) for index in range(count)],
        },
    }


def make_payload(
    symbols=A_SHARE_DEMO_6_SYMBOLS,
    count: int = 52,
) -> dict[str, object]:
    return {
        "errorcode": 0,
        "errmsg": "",
        "tables": [make_table(symbol, count) for symbol in symbols],
    }


def adapt(
    payload,
    *,
    expected_symbols=A_SHARE_DEMO_6_SYMBOLS,
    start=START,
    end=END,
    min_bars=50,
):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        response_to_frames,
    )

    return response_to_frames(
        payload,
        expected_symbols=expected_symbols,
        start=start,
        end=end,
        min_bars=min_bars,
    )


def one_symbol_payload(count: int = 1):
    symbol = A_SHARE_DEMO_6_SYMBOLS[0]
    return make_payload((symbol,), count), (symbol,)


def test_maps_official_tables_to_six_numeric_ohlcv_frames():
    frames = adapt(make_payload())

    assert tuple(frames) == A_SHARE_DEMO_6_SYMBOLS
    for symbol, frame in frames.items():
        assert len(frame) == 52
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(frame.index, pd.DatetimeIndex)
        assert frame.index.name == "timestamp"
        assert str(frame.index.tz) == "Asia/Shanghai"
        assert frame.index.is_monotonic_increasing
        assert frame.index.is_unique
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)
        assert frame.iloc[0].to_dict() == {
            "open": float(100 + A_SHARE_DEMO_6_SYMBOLS.index(symbol) * 10),
            "high": float(101 + A_SHARE_DEMO_6_SYMBOLS.index(symbol) * 10),
            "low": float(99 + A_SHARE_DEMO_6_SYMBOLS.index(symbol) * 10),
            "close": float(100.5 + A_SHARE_DEMO_6_SYMBOLS.index(symbol) * 10),
            "volume": 10_000.0,
        }


def test_naive_boundaries_are_interpreted_as_shanghai_time():
    payload, symbols = one_symbol_payload()

    frames = adapt(
        payload,
        expected_symbols=symbols,
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 2),
        min_bars=1,
    )

    assert str(frames[symbols[0]].index.tz) == "Asia/Shanghai"


def test_zero_minimum_keeps_empty_frame_schema_and_timezone():
    payload, symbols = one_symbol_payload(count=0)

    frame = adapt(
        payload,
        expected_symbols=symbols,
        min_bars=0,
    )[symbols[0]]

    assert frame.empty
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "timestamp"
    assert str(frame.index.tz) == "Asia/Shanghai"


def test_business_error_is_sanitized():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBusinessResponseError,
    )

    secret = "upstream-secret-must-not-leak"
    payload = {"errorcode": -403, "errmsg": secret, "tables": []}

    with pytest.raises(IFindBusinessResponseError) as exc_info:
        adapt(payload)

    assert exc_info.value.errorcode == -403
    assert secret not in str(exc_info.value)


def test_untrusted_errorcode_is_not_echoed():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    secret = "malicious-errorcode-secret"
    payload = {"errorcode": secret, "tables": []}

    with pytest.raises(IFindResponseSchemaError) as exc_info:
        adapt(payload)

    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"errorcode": 0},
        {"errorcode": 0, "tables": {}},
        {"errorcode": False, "tables": []},
    ],
)
def test_rejects_non_official_top_level_shapes(payload):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    with pytest.raises(IFindResponseSchemaError):
        adapt(payload)


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_rejects_symbol_set_mismatch(mode):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    payload = make_payload()
    if mode == "missing":
        payload["tables"].pop()
    else:
        extra = deepcopy(payload["tables"][0])
        extra["thscode"] = "999999.SH"
        payload["tables"].append(extra)

    with pytest.raises(IFindResponseSchemaError, match="symbol set"):
        adapt(payload)


def test_rejects_duplicate_symbol_table():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    payload = make_payload()
    payload["tables"].append(deepcopy(payload["tables"][0]))

    with pytest.raises(IFindResponseSchemaError, match="duplicate"):
        adapt(payload)


def test_rejects_mismatched_array_lengths():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["table"]["volume"].append("10001")

    with pytest.raises(IFindResponseSchemaError, match="array lengths"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


@pytest.mark.parametrize("field", ["time", "open", "high", "low", "close", "volume"])
def test_rejects_non_array_series_fields(field):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindResponseSchemaError,
    )

    payload, symbols = one_symbol_payload()
    if field == "time":
        payload["tables"][0][field] = "2026-04-01 10:30:00"
    else:
        payload["tables"][0]["table"][field] = "100"

    with pytest.raises(IFindResponseSchemaError, match=field):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_invalid_timestamp_without_echoing_raw_value():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    raw_secret = "invalid-time-secret"
    payload["tables"][0]["time"][0] = raw_secret

    with pytest.raises(IFindBarValidationError) as exc_info:
        adapt(payload, expected_symbols=symbols, min_bars=1)

    assert raw_secret not in str(exc_info.value)


def test_rejects_timezone_aware_upstream_timestamp():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["time"][0] = "2026-04-01T10:30:00+08:00"

    with pytest.raises(IFindBarValidationError, match="LocalTime"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_duplicate_timestamps():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload(count=2)
    payload["tables"][0]["time"][1] = payload["tables"][0]["time"][0]

    with pytest.raises(IFindBarValidationError, match="duplicate timestamp"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_unsorted_timestamps():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload(count=2)
    payload["tables"][0]["time"].reverse()

    with pytest.raises(IFindBarValidationError, match="strictly increasing"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-03-31 15:00:00",
        "2026-05-01 10:30:00",
    ],
)
def test_rejects_timestamps_outside_half_open_window(timestamp):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["time"][0] = timestamp

    with pytest.raises(IFindBarValidationError, match="requested window"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-04-01 12:30:00",
        "2026-04-04 10:30:00",
    ],
)
def test_rejects_lunch_break_and_weekend_bars(timestamp):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["time"][0] = timestamp

    with pytest.raises(IFindBarValidationError, match="trading session"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open", "0"),
        ("open", "-1"),
        ("high", nan),
        ("low", inf),
        ("close", "not-a-number"),
        ("close", None),
    ],
)
def test_rejects_non_finite_or_non_positive_prices(field, value):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["table"][field][0] = value

    with pytest.raises(IFindBarValidationError, match=field):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_high_below_open_or_close():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["table"]["high"][0] = "100.25"

    with pytest.raises(IFindBarValidationError, match="high"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_low_above_open_or_close():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["table"]["low"][0] = "100.25"

    with pytest.raises(IFindBarValidationError, match="low"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


@pytest.mark.parametrize("value", ["-1", nan, inf, "not-a-number", None])
def test_rejects_invalid_volume(value):
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload()
    payload["tables"][0]["table"]["volume"][0] = value

    with pytest.raises(IFindBarValidationError, match="volume"):
        adapt(payload, expected_symbols=symbols, min_bars=1)


def test_rejects_symbol_with_fewer_than_fifty_valid_bars():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )

    payload, symbols = one_symbol_payload(count=49)

    with pytest.raises(
        IFindBarValidationError,
        match=r"600519\.SH.*49.*50",
    ):
        adapt(payload, expected_symbols=symbols)
