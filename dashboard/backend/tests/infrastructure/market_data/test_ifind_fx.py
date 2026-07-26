"""Historical USD/CNY inference from iFinD dual-currency closes."""

from __future__ import annotations

from datetime import date
import logging

import pytest


START = date(2026, 4, 1)
END = date(2026, 4, 4)
SYMBOLS = ("600519.SH", "601318.SH", "600036.SH")
DAY_ONE = date(2026, 4, 1)
DAY_TWO = date(2026, 4, 2)
DAY_THREE = date(2026, 4, 3)


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


def table_for(payloads, currency, symbol):
    return next(
        table
        for table in payloads[currency]["tables"]
        if table["thscode"] == symbol
    )


def blank_out(payloads, currency, symbol, day_index, value=None):
    """Simulate a suspended symbol: iFinD is asked to Fill=Blank those days."""
    table_for(payloads, currency, symbol)["table"]["close"][day_index] = value


def provider(payloads):
    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindHistoricalFxProvider,
    )

    return IFindHistoricalFxProvider(client=SpyClient(payloads))


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
    assert rates[DAY_ONE] == pytest.approx(6.9025, rel=1e-5)
    assert rates[DAY_TWO] == pytest.approx(6.8876, rel=1e-5)
    assert rates[DAY_THREE] == pytest.approx(6.8929, rel=1e-5)


def test_real_supercommand_moutai_values_imply_verified_rate_direction():
    rmb_close = 1459.44
    usd_close = 211.4364

    assert rmb_close / usd_close == pytest.approx(6.90250118)


def test_uses_median_to_reduce_rounding_noise():
    rates = provider(dual_payloads(rates=(6.90, 6.90, 6.90))).fetch_usd_cny(
        SYMBOLS, START, END
    )

    assert rates[START] == pytest.approx(6.90, rel=1e-4)


# --- Suspended symbols and single bad quotes must not kill the run ----------
# A-share names halt (停牌) routinely, and fetch_daily_closes asks iFinD for
# Fill=Blank, so gaps are expected data rather than a broken contract.


@pytest.mark.parametrize("blank_value", [None, "", "   "])
def test_blank_close_skips_only_that_symbol_day(blank_value):
    payloads = dual_payloads()
    blank_out(payloads, "RMB", "601318.SH", 0, blank_value)

    fx = provider(payloads)
    rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert set(rates) == {DAY_ONE, DAY_TWO, DAY_THREE}
    assert rates[DAY_ONE] == pytest.approx(6.9025, rel=1e-5)
    assert fx.last_stats.blank_closes == 1
    assert fx.last_stats.skipped_dates == 0


def test_symbol_suspended_for_whole_window_still_yields_rates_and_logs_error(caplog):
    payloads = dual_payloads()
    for day_index in range(3):
        blank_out(payloads, "RMB", "601318.SH", day_index)

    fx = provider(payloads)
    with caplog.at_level(logging.ERROR):
        rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert set(rates) == {DAY_ONE, DAY_TWO, DAY_THREE}
    assert fx.last_stats.symbols_without_usable_closes == ["601318.SH"]
    assert "wholesale" in caplog.text


def test_drops_single_disagreeing_symbol_instead_of_failing_the_date():
    payloads = dual_payloads()
    table_for(payloads, "MHB", "601318.SH")["table"]["close"][0] = 4.0

    fx = provider(payloads)
    rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert rates[DAY_ONE] == pytest.approx(6.9025, rel=1e-5)
    assert fx.last_stats.outliers_dropped_by_symbol == {"601318.SH": 1}


def test_skips_date_when_survivors_still_disagree(caplog):
    payloads = dual_payloads()
    for symbol, close in (("601318.SH", 4.0), ("600036.SH", 9.0)):
        table_for(payloads, "MHB", symbol)["table"]["close"][0] = close

    fx = provider(payloads)
    with caplog.at_level(logging.WARNING):
        rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert set(rates) == {DAY_TWO, DAY_THREE}
    assert fx.last_stats.dates_inconsistent == 1
    assert "dropped some data" in caplog.text


def test_skips_date_with_fewer_than_two_matched_symbols(caplog):
    payloads = dual_payloads()
    for symbol in ("601318.SH", "600036.SH"):
        blank_out(payloads, "MHB", symbol, 0)

    fx = provider(payloads)
    with caplog.at_level(logging.WARNING):
        rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert set(rates) == {DAY_TWO, DAY_THREE}
    assert fx.last_stats.dates_below_min_symbols == 1
    assert caplog.text


def test_unusable_close_is_dropped_per_datum_not_per_run():
    payloads = dual_payloads()
    table_for(payloads, "RMB", "600519.SH")["table"]["close"][0] = 0

    fx = provider(payloads)
    rates = fx.fetch_usd_cny(SYMBOLS, START, END)

    assert rates[DAY_ONE] == pytest.approx(6.9025, rel=1e-5)
    assert fx.last_stats.unusable_closes == 1


# --- Wholesale drift is still fatal ----------------------------------------


def test_raises_when_no_date_survives():
    payloads = dual_payloads()
    for symbol in ("601318.SH", "600036.SH"):
        for day_index in range(3):
            blank_out(payloads, "MHB", symbol, day_index)

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
    )

    with pytest.raises(IFindFxValidationError, match="no usable daily rates"):
        provider(payloads).fetch_usd_cny(SYMBOLS, START, END)


def test_raises_when_most_dates_are_dropped():
    payloads = dual_payloads()
    for symbol in ("601318.SH", "600036.SH"):
        for day_index in (0, 1):
            blank_out(payloads, "MHB", symbol, day_index)

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
    )

    with pytest.raises(IFindFxValidationError, match="changed daily-close contract"):
        provider(payloads).fetch_usd_cny(SYMBOLS, START, END)


def test_wholesale_failure_message_carries_counts_but_no_raw_values():
    payloads = dual_payloads()
    for symbol in ("601318.SH", "600036.SH"):
        for day_index in range(3):
            blank_out(payloads, "MHB", symbol, day_index)

    from dashboard.backend.infrastructure.market_data.ifind_fx import (
        IFindFxValidationError,
    )

    with pytest.raises(IFindFxValidationError) as excinfo:
        provider(payloads).fetch_usd_cny(SYMBOLS, START, END)

    message = str(excinfo.value)
    assert "blank_closes=6" in message
    assert "1459.44" not in message
    assert "211.4364" not in message


# --- Contract breaks stay fatal --------------------------------------------


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
            lambda p: p["MHB"].update({"errorcode": -1}),
            "business",
        ),
        (
            lambda p: p["RMB"]["tables"][0].update({"thscode": "000001.SZ"}),
            "unexpected symbol",
        ),
        (
            lambda p: p["RMB"]["tables"][0]["time"].__setitem__(1, "2026-04-01"),
            "duplicate date",
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
