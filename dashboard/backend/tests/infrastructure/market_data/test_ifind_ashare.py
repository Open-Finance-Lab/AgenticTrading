"""Thin orchestration contract for the fixed-universe iFinD provider."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6_SYMBOLS,
    IFIND_ASHARE,
)


START = date(2026, 4, 1)
END = date(2026, 5, 1)
CN = ZoneInfo("Asia/Shanghai")


class SpyClient:
    def __init__(self, payload=None, error=None):
        self.payload = {"errorcode": 0, "tables": []} if payload is None else payload
        self.error = error
        self.calls = []
        self.market_rule_calls = []

    def fetch_hourly_bars(self, symbols, start, end):
        self.calls.append((symbols, start, end))
        if self.error is not None:
            raise self.error
        return self.payload

    def fetch_daily_market_rules(self, symbols, start, end):
        self.market_rule_calls.append((symbols, start, end))
        if self.error is not None:
            raise self.error
        return self.payload

    def fetch_basic_market_status(self, symbols, trading_date):
        raise AssertionError(
            f"unexpected basic status call for {symbols!r} on {trading_date}"
        )


class SpyAdapter:
    def __init__(self, result=None, error=None):
        self.result = {} if result is None else result
        self.error = error
        self.calls = []

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class SpyFxProvider:
    def __init__(self, result=None):
        self.result = {date(2026, 3, 31): 7.0} if result is None else result
        self.calls = []

    def fetch_usd_cny(self, symbols, start, end):
        self.calls.append((symbols, start, end))
        return self.result


def valid_bar_times(count=52):
    values = []
    current = START
    bar_times = (time(10, 30), time(11, 30), time(14), time(15))
    while len(values) < count:
        if current.weekday() < 5:
            for bar_time in bar_times:
                values.append(
                    datetime.combine(current, bar_time).strftime("%Y-%m-%d %H:%M:%S")
                )
                if len(values) == count:
                    break
        current += timedelta(days=1)
    return values


def valid_payload(count=52):
    timestamps = valid_bar_times(count)
    tables = []
    for offset, symbol in enumerate(A_SHARE_DEMO_6_SYMBOLS):
        base = 100 + offset * 10
        opens = [base + row * 0.1 for row in range(count)]
        tables.append(
            {
                "thscode": symbol,
                "time": timestamps.copy(),
                "table": {
                    "open": opens,
                    "high": [value + 1 for value in opens],
                    "low": [value - 1 for value in opens],
                    "close": [value + 0.5 for value in opens],
                    "volume": [10_000 + row for row in range(count)],
                },
            }
        )
    return {"errorcode": 0, "errmsg": "", "tables": tables}


def test_fetches_fixed_universe_once_and_passes_payload_to_adapter():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    payload = {"errorcode": 0, "tables": []}
    frames = {symbol: pd.DataFrame() for symbol in A_SHARE_DEMO_6_SYMBOLS}
    client = SpyClient(payload)
    adapter = SpyAdapter(frames)
    provider = IFindAshareProvider(client=client, adapter=adapter)

    result = provider.fetch_bars(
        list(reversed(A_SHARE_DEMO_6_SYMBOLS)),
        "2026-04-01",
        "2026-05-01",
    )

    assert result is frames
    assert client.calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    assert adapter.calls == [
        (
            payload,
            {
                "expected_symbols": A_SHARE_DEMO_6_SYMBOLS,
                "start": START,
                "end": END,
                "min_bars": 50,
            },
        )
    ]


def test_fetches_historical_fx_for_the_same_registered_universe():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    client = SpyClient()
    fx_provider = SpyFxProvider()
    provider = IFindAshareProvider(client=client, fx_provider=fx_provider)

    result = provider.fetch_usd_cny(
        list(reversed(A_SHARE_DEMO_6_SYMBOLS)),
        "2026-04-01",
        "2026-05-01",
    )

    assert result == {date(2026, 3, 31): 7.0}
    assert fx_provider.calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    assert client.calls == []


def test_fetches_daily_rules_once_and_passes_combined_clock_to_adapter():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    payload = {"errorcode": 0, "tables": []}
    client = SpyClient(payload)
    expected = object()
    adapter = SpyAdapter(expected)
    provider = IFindAshareProvider(client=client, market_rule_adapter=adapter)
    bars = {
        A_SHARE_DEMO_6_SYMBOLS[0]: pd.DataFrame(
            index=pd.DatetimeIndex(
                [
                    datetime(2026, 4, 1, 15, tzinfo=CN),
                    datetime(2026, 4, 2, 15, tzinfo=CN),
                ]
            )
        ),
        A_SHARE_DEMO_6_SYMBOLS[1]: pd.DataFrame(
            index=pd.DatetimeIndex([datetime(2026, 4, 2, 15, tzinfo=CN)])
        ),
    }

    result = provider.fetch_market_rules(
        A_SHARE_DEMO_6_SYMBOLS,
        START,
        END,
        bars_by_symbol=bars,
    )

    assert result is expected
    assert client.market_rule_calls == [(A_SHARE_DEMO_6_SYMBOLS, START, END)]
    adapted_payload, kwargs = adapter.calls[0]
    assert adapted_payload is payload
    assert kwargs["expected_symbols"] == A_SHARE_DEMO_6_SYMBOLS
    assert kwargs["required_dates"] == [date(2026, 4, 1), date(2026, 4, 2)]
    assert kwargs["bars_by_symbol"] is bars
    assert kwargs["fetch_basic_status"].__self__ is client
    assert kwargs["price_tick"] == 0.01


def test_fetches_selected_csi300_sample20_in_registered_order():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )
    from dashboard.backend.infrastructure.market_data.profiles import (
        CSI300_SAMPLE_20_2026H2,
        CSI300_SAMPLE_20_2026H2_SYMBOLS,
        IFIND_ASHARE,
        get_market_profile,
    )

    payload = {"errorcode": 0, "tables": []}
    frames = {
        symbol: pd.DataFrame() for symbol in CSI300_SAMPLE_20_2026H2_SYMBOLS
    }
    client = SpyClient(payload)
    adapter = SpyAdapter(frames)
    profile = get_market_profile(IFIND_ASHARE, CSI300_SAMPLE_20_2026H2)
    provider = IFindAshareProvider(
        profile=profile,
        client=client,
        adapter=adapter,
    )

    result = provider.fetch_bars(
        list(reversed(CSI300_SAMPLE_20_2026H2_SYMBOLS)),
        START,
        END,
    )

    assert result is frames
    assert client.calls == [(CSI300_SAMPLE_20_2026H2_SYMBOLS, START, END)]
    assert adapter.calls[0][1]["expected_symbols"] == (
        CSI300_SAMPLE_20_2026H2_SYMBOLS
    )


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_selected_csi300_sample20_rejects_inexact_symbol_set(mode):
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
        IFindUniverseError,
    )
    from dashboard.backend.infrastructure.market_data.profiles import (
        CSI300_SAMPLE_20_2026H2,
        CSI300_SAMPLE_20_2026H2_SYMBOLS,
        IFIND_ASHARE,
        get_market_profile,
    )

    requested = list(CSI300_SAMPLE_20_2026H2_SYMBOLS)
    if mode == "missing":
        requested.pop()
    elif mode == "extra":
        requested.append("999999.SH")
    else:
        requested[-1] = requested[0]

    client = SpyClient()
    provider = IFindAshareProvider(
        profile=get_market_profile(IFIND_ASHARE, CSI300_SAMPLE_20_2026H2),
        client=client,
    )

    with pytest.raises(IFindUniverseError, match=CSI300_SAMPLE_20_2026H2):
        provider.fetch_bars(requested, START, END)

    assert client.calls == []


def test_real_adapter_returns_six_valid_frames_from_one_client_response():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    client = SpyClient(valid_payload())

    frames = IFindAshareProvider(client=client).fetch_bars(
        A_SHARE_DEMO_6_SYMBOLS,
        START,
        END,
    )

    assert len(client.calls) == 1
    assert tuple(frames) == A_SHARE_DEMO_6_SYMBOLS
    assert all(len(frame) == 52 for frame in frames.values())
    assert all(str(frame.index.tz) == "Asia/Shanghai" for frame in frames.values())
    assert all(
        list(frame.columns) == ["open", "high", "low", "close", "volume"]
        for frame in frames.values()
    )


@pytest.mark.parametrize(
    "symbols",
    [
        A_SHARE_DEMO_6_SYMBOLS[:-1],
        (*A_SHARE_DEMO_6_SYMBOLS, "999999.SH"),
        (*A_SHARE_DEMO_6_SYMBOLS, A_SHARE_DEMO_6_SYMBOLS[0]),
        ("600519.SH",),
    ],
)
def test_rejects_any_universe_other_than_the_fixed_six(symbols):
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
        IFindUniverseError,
    )

    client = SpyClient()

    with pytest.raises(IFindUniverseError, match="a_share_demo_6"):
        IFindAshareProvider(client=client).fetch_bars(symbols, START, END)

    assert client.calls == []


def test_accepts_date_and_datetime_inputs_and_normalizes_to_cn_dates():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    client = SpyClient()
    adapter = SpyAdapter()
    provider = IFindAshareProvider(client=client, adapter=adapter)

    provider.fetch_bars(
        A_SHARE_DEMO_6_SYMBOLS,
        datetime(2026, 3, 31, 16, 30, tzinfo=ZoneInfo("UTC")),
        datetime(2026, 4, 30, 16, 30, tzinfo=ZoneInfo("UTC")),
    )

    assert client.calls == [
        (A_SHARE_DEMO_6_SYMBOLS, date(2026, 4, 1), date(2026, 5, 1))
    ]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2026/04/01", "2026-05-01"),
        ("not-a-date", "2026-05-01"),
        (object(), date(2026, 5, 1)),
        (date(2026, 5, 1), date(2026, 4, 1)),
        (date(2026, 5, 1), date(2026, 5, 1)),
    ],
)
def test_rejects_invalid_date_inputs_before_client_call(start, end):
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
        IFindDateInputError,
    )

    client = SpyClient()

    with pytest.raises(IFindDateInputError):
        IFindAshareProvider(client=client).fetch_bars(
            A_SHARE_DEMO_6_SYMBOLS, start, end
        )

    assert client.calls == []


def test_client_error_propagates_and_adapter_is_not_called():
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTransportError,
    )

    original = IFindTransportError("sanitized transport failure")
    client = SpyClient(error=original)
    adapter = SpyAdapter()

    with pytest.raises(IFindTransportError) as exc_info:
        IFindAshareProvider(client=client, adapter=adapter).fetch_bars(
            A_SHARE_DEMO_6_SYMBOLS, START, END
        )

    assert exc_info.value is original
    assert len(client.calls) == 1
    assert adapter.calls == []


def test_adapter_error_propagates_without_fallback_or_second_client_call():
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
    )
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )

    original = IFindBarValidationError("sanitized validation failure")
    client = SpyClient()
    adapter = SpyAdapter(error=original)

    with pytest.raises(IFindBarValidationError) as exc_info:
        IFindAshareProvider(client=client, adapter=adapter).fetch_bars(
            A_SHARE_DEMO_6_SYMBOLS, START, END
        )

    assert exc_info.value is original
    assert len(client.calls) == 1
    assert len(adapter.calls) == 1


def test_factory_creates_ifind_provider_without_making_http_request(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )
    from dashboard.backend.infrastructure.market_data.provider import (
        create_market_data_provider,
    )

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token")

    created = create_market_data_provider(IFIND_ASHARE)

    assert isinstance(created, IFindAshareProvider)


def test_factory_accepts_refresh_token_without_making_http_request(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        IFindAshareProvider,
    )
    from dashboard.backend.infrastructure.market_data.provider import (
        create_market_data_provider,
    )

    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_REFRESH_TOKEN", "refresh-token")
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    created = create_market_data_provider(IFIND_ASHARE)

    assert isinstance(created, IFindAshareProvider)


def test_ifind_provider_import_is_lazy_and_has_no_network_or_fallback_imports():
    env = {
        **os.environ,
        "ENABLE_IFIND_ASHARE": "true",
        "IFIND_ACCESS_TOKEN": "test-token",
    }
    code = (
        "import requests\n"
        "import sys\n"
        "def fail_session():\n"
        "    raise AssertionError('network session created during import')\n"
        "requests.Session = fail_session\n"
        "import dashboard.backend.infrastructure.market_data.ifind_ashare\n"
        "assert 'dashboard.backend.infrastructure.market_data.alpaca_bars' "
        "not in sys.modules\n"
        "assert 'dashboard.backend.infrastructure.market_data.quotes' "
        "not in sys.modules\n"
        "assert not any(name == 'vnpy' or name.startswith('vnpy.') "
        "for name in sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
