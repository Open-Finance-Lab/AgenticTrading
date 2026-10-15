"""NYSE early closes, by rule, pinned to the exchange's published calendars.

The Live board's 2026-11-27 session is why this exists: the calendar knew the
holidays but not the 13:00 early closes, so that day would have had seven
chart nodes and three model decisions on late-session bars.
``PUBLISHED_EARLY_CLOSES`` is hand-copied from NYSE, never generated from the
code under test: the NYSE Group releases "2022, 2023 and 2024 Holiday and
Early Closings Calendar" (2021-12-27), "2025, 2026 and 2027 ..." (2024-11-08)
and "2026, 2027 and 2028 ..." (2025-12-23), and nyse.com "Holidays & Trading
Hours" (https://www.nyse.com/trade/hours-calendars), retrieved 2026-10-15.
Re-pin when NYSE publishes the next year, usually in December.

``XNYS_EARLY_CLOSES`` widens the pin to 2004-2035. It was generated once,
offline, from exchange_calendars 4.13.2 (``get_calendar("XNYS").closes``: every
session closing before 16:00, all of them at 13:00), the maintained calendar
library zipline's descendants use; it is not a dependency. For 2004-2025 it is
sessions that already happened; for 2026-2035 it is a second, independent
implementation of the same rules, so it catches a slip in ours rather than a
change in NYSE's policy. That only NYSE's next release can show.

A third calendar agrees too: QuantConnect LEAN's market-hours database
(``Equity-usa``) matches every early close and holiday here for 2004-2028
(diffed 2026-10-15). It is not the pin because it carries a Sunday early
close, 2017-12-24.
"""

from datetime import date, time, timedelta

import pytest

from dashboard.backend.infrastructure.market_data import sessions
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    UNSCHEDULED_CLOSURES,
    is_early_close,
    is_trading_day,
    nyse_early_closes,
    nyse_holidays,
)

PUBLISHED_EARLY_CLOSES = {
    2022: ["2022-11-25"],
    2023: ["2023-07-03", "2023-11-24"],
    2024: ["2024-07-03", "2024-11-29", "2024-12-24"],
    2025: ["2025-07-03", "2025-11-28", "2025-12-24"],
    2026: ["2026-11-27", "2026-12-24"],
    2027: ["2027-11-26"],
    2028: ["2028-07-03", "2028-11-24"],
}


@pytest.mark.parametrize("year", sorted(PUBLISHED_EARLY_CLOSES))
def test_early_closes_match_the_nyse_published_calendar(year):
    assert sorted(d.isoformat() for d in nyse_early_closes(year)) == (
        PUBLISHED_EARLY_CLOSES[year]
    )


XNYS_EARLY_CLOSES = {
    2004: ["11-26"],
    2005: ["11-25"],
    2006: ["07-03", "11-24"],
    2007: ["07-03", "11-23", "12-24"],
    2008: ["07-03", "11-28", "12-24"],
    2009: ["11-27", "12-24"],
    2010: ["11-26"],
    2011: ["11-25"],
    2012: ["07-03", "11-23", "12-24"],
    2013: ["07-03", "11-29", "12-24"],
    2014: ["07-03", "11-28", "12-24"],
    2015: ["11-27", "12-24"],
    2016: ["11-25"],
    2017: ["07-03", "11-24"],
    2018: ["07-03", "11-23", "12-24"],
    2019: ["07-03", "11-29", "12-24"],
    2020: ["11-27", "12-24"],
    2021: ["11-26"],
    2022: ["11-25"],
    2023: ["07-03", "11-24"],
    2024: ["07-03", "11-29", "12-24"],
    2025: ["07-03", "11-28", "12-24"],
    2026: ["11-27", "12-24"],
    2027: ["11-26"],
    2028: ["07-03", "11-24"],
    2029: ["07-03", "11-23", "12-24"],
    2030: ["07-03", "11-29", "12-24"],
    2031: ["07-03", "11-28", "12-24"],
    2032: ["11-26"],
    2033: ["11-25"],
    2034: ["07-03", "11-24"],
    2035: ["07-03", "11-23", "12-24"],
}


@pytest.mark.parametrize("year", sorted(XNYS_EARLY_CLOSES))
def test_early_closes_match_exchange_calendars_xnys(year):
    assert sorted(d.strftime("%m-%d") for d in nyse_early_closes(year)) == (
        XNYS_EARLY_CLOSES[year]
    )


#: Sessions per year in exchange_calendars 4.13.2's XNYS calendar, generated
#: the same way as ``XNYS_EARLY_CLOSES``. A count catches a missing or extra
#: closure anywhere in the year without pinning ~8,000 dates.
XNYS_SESSIONS_PER_YEAR = {
    2004: 252, 2005: 252, 2006: 251, 2007: 251, 2008: 253, 2009: 252,
    2010: 252, 2011: 252, 2012: 250, 2013: 252, 2014: 252, 2015: 252,
    2016: 252, 2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252,
    2022: 251, 2023: 250, 2024: 252, 2025: 250, 2026: 251, 2027: 251,
    2028: 251, 2029: 251, 2030: 251, 2031: 251, 2032: 252, 2033: 251,
    2034: 250, 2035: 251,
}


@pytest.mark.parametrize("year", sorted(XNYS_SESSIONS_PER_YEAR))
def test_sessions_per_year_match_exchange_calendars_xnys(year):
    day, sessions_in_year = date(year, 1, 1), 0
    while day.year == year:
        sessions_in_year += is_trading_day(day)
        day += timedelta(days=1)
    assert sessions_in_year == XNYS_SESSIONS_PER_YEAR[year]


def test_unscheduled_closures_are_not_trading_days():
    """No rule reproduces these. 2025-01-09 (President Carter's national day of
    mourning) is the one a current backtest window can reach."""
    assert date(2025, 1, 9) in UNSCHEDULED_CLOSURES
    for day in UNSCHEDULED_CLOSURES:
        assert day.weekday() < 5, day
        assert day in nyse_holidays(day.year)
        assert not is_trading_day(day)
    assert is_trading_day(date(2025, 1, 8))
    assert is_trading_day(date(2025, 1, 10))


def test_an_unscheduled_early_close_never_opens_a_closed_day(monkeypatch):
    """The dated set is empty today; adding to it must shorten a session, never
    create one on a holiday or a weekend."""
    from dashboard.backend.infrastructure.market_data import us_market_calendar

    assert us_market_calendar.UNSCHEDULED_EARLY_CLOSES == frozenset()
    monkeypatch.setattr(us_market_calendar, "UNSCHEDULED_EARLY_CLOSES", frozenset({
        date(2026, 3, 12),  # a Thursday
        date(2026, 4, 3),   # Good Friday
        date(2026, 3, 14),  # a Saturday
    }))
    nyse_early_closes.cache_clear()
    try:
        assert nyse_early_closes(2026) == {
            date(2026, 3, 12), date(2026, 11, 27), date(2026, 12, 24),
        }
        assert sessions.session_close_time(date(2026, 3, 12)) == time(13, 0)
        assert not is_trading_day(date(2026, 4, 3))
    finally:
        nyse_early_closes.cache_clear()


def test_a_holiday_or_weekend_is_never_an_early_close():
    # Independence Day 2026 is observed on Friday July 3; Christmas 2027 on
    # Friday December 24; December 24, 2028 is a Sunday.
    assert date(2026, 7, 3) in nyse_holidays(2026)
    assert date(2026, 7, 3) not in nyse_early_closes(2026)
    assert date(2027, 12, 24) in nyse_holidays(2027)
    assert date(2027, 12, 24) not in nyse_early_closes(2027)
    assert date(2028, 12, 24) not in nyse_early_closes(2028)


@pytest.mark.parametrize("year", range(2022, 2031))
def test_an_early_close_is_always_a_trading_day(year):
    for day in nyse_early_closes(year):
        assert is_trading_day(day), day


def test_is_early_close():
    assert is_early_close(date(2026, 11, 27))
    assert is_early_close(date(2026, 12, 24))
    assert not is_early_close(date(2026, 11, 25))
    # Thanksgiving itself is a holiday, not an early close.
    assert not is_early_close(date(2026, 11, 26))


def test_the_close_times_have_one_owner():
    assert sessions.EARLY_CLOSE_TIME == time(13, 0)
    assert sessions.REGULAR_CLOSE_TIME == time(16, 0)
    assert sessions.REGULAR_CLOSE_TIME == sessions.session_windows("US")[-1][1]


@pytest.mark.parametrize(
    ("day", "market", "close"),
    [
        (date(2026, 11, 27), "US", time(13, 0)),
        (date(2026, 12, 24), "us", time(13, 0)),
        (date(2026, 11, 25), "US", time(16, 0)),
        (date(2026, 11, 30), None, time(16, 0)),
        # A-shares keep their own close; the NYSE calendar is not theirs.
        (date(2026, 11, 27), "CN", time(15, 0)),
    ],
)
def test_session_close_time(day, market, close):
    assert sessions.session_close_time(day, market) == close


def test_session_close_time_defaults_to_the_us_market():
    assert sessions.session_close_time(date(2026, 11, 27)) == time(13, 0)
