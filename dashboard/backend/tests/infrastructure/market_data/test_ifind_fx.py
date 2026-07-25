"""Historical USD/CNY inference from iFinD dual-currency closes."""

from __future__ import annotations

from datetime import date

import pytest


START = date(2026, 4, 1)
END = date(2026, 4, 4)
SYMBOLS = ("600519.SH", "601318.SH", "600036.SH")


class SpyClient:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def fetch_daily_closes(self, symbols, start, end, *, currency):
        self.calls.append((tuple(symbols), start, end, currency))
        return self.payloads[currency]


def payload(rows_by_symbol):
    return {
        "errorcode": 0,
        "tables": [
            {
                "thscode": symbol,
                "time": [row[0] for row in rows],
                "table": {"close": [row[1] for row in rows]},
            }
            for symbol, rows in rows_by_symbol.items()
        ],
    }


def dual_payloads(rates=(6.9025, 6.8876, 6.8929)):
    days = ("2026-04-01", "2026-04-02", "2026-04-03")
    rmb = {
        "600519.SH": list(zip(days, (1459.44, 1459.80, 1460.00))),
        "601318.SH": list(zip(days, (58.12, 58.40, 58.31))),
        "600036.SH": list(zip(days, (42.25, 42.19, 42.61))),
    }
    mhb = {
        symbol: [
            (day, round(float(close) / rate, 4))
            for (day, close), rate in zip(rows, rates)
        ]
        for symbol, rows in rmb.items()
    }
    return {"RMB": payload(rmb), "MHB": payload(mhb)}


def test_fetches_two_currencies_with_lookback_and_returns_daily_medians():
    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindHistoricalFxProvider,
    )

    client = SpyClient(dual_payloads())

    rates = IFindHistoricalFxProvider(client=client).fetch_usd_cny(
        SYMBOLS, START, END
    )

    assert client.calls == [
        (SYMBOLS, date(2026, 3, 18), END, "RMB"),
        (SYMBOLS, date(2026, 3, 18), END, "MHB"),
    ]
    assert rates[date(2026, 4, 1)] == pytest.approx(6.9025, rel=1e-5)
    assert rates[date(2026, 4, 2)] == pytest.approx(6.8876, rel=1e-5)
    assert rates[date(2026, 4, 3)] == pytest.approx(6.8929, rel=1e-5)


def test_real_supercommand_moutai_values_imply_verified_rate_direction():
    rmb_close = 1459.44
    usd_close = 211.4364

    assert rmb_close / usd_close == pytest.approx(6.90250118)


def test_uses_median_to_reduce_rounding_noise():
    payloads = dual_payloads(rates=(6.90, 6.90, 6.90))
    client = SpyClient(payloads)

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindHistoricalFxProvider,
    )

    rates = IFindHistoricalFxProvider(client=client).fetch_usd_cny(
        SYMBOLS, START, END
    )

    assert rates[START] == pytest.approx(6.90, rel=1e-4)


def test_rejects_conflicting_symbol_rates():
    payloads = dual_payloads()
    payloads["MHB"]["tables"][1]["table"]["close"][0] = 4.0

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
        IFindHistoricalFxProvider,
    )

    with pytest.raises(IFindFxValidationError, match="disagree"):
        IFindHistoricalFxProvider(client=SpyClient(payloads)).fetch_usd_cny(
            SYMBOLS, START, END
        )


def test_rejects_day_with_fewer_than_two_matched_symbols():
    payloads = dual_payloads()
    for table in payloads["MHB"]["tables"][1:]:
        table["time"] = table["time"][1:]
        table["table"]["close"] = table["table"]["close"][1:]

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
        IFindHistoricalFxProvider,
    )

    with pytest.raises(IFindFxValidationError, match="at least 2"):
        IFindHistoricalFxProvider(client=SpyClient(payloads)).fetch_usd_cny(
            SYMBOLS, START, END
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda p: p["RMB"]["tables"][0].update({"time": "2026-04-01"}),
            "array",
        ),
        (
            lambda p: p["MHB"]["tables"][0]["table"].update({"close": [1.0]}),
            "length",
        ),
        (
            lambda p: p["RMB"]["tables"][0]["table"]["close"].__setitem__(0, 0),
            "positive",
        ),
        (
            lambda p: p["MHB"].update({"errorcode": -1}),
            "business",
        ),
    ],
)
def test_rejects_invalid_or_failed_payloads_without_raw_values(mutate, match):
    payloads = dual_payloads()
    mutate(payloads)

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxResponseError,
        IFindFxValidationError,
        IFindHistoricalFxProvider,
    )

    with pytest.raises((IFindFxResponseError, IFindFxValidationError), match=match):
        IFindHistoricalFxProvider(client=SpyClient(payloads)).fetch_usd_cny(
            SYMBOLS, START, END
        )
