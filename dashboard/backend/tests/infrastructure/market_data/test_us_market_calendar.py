"""NYSE early closes, by rule, pinned to the exchange's published calendars.

The Live board's 2026-11-27 session is why this exists: the calendar knew the
holidays but not the 13:00 early closes, so that day would have had seven
chart nodes and three model decisions on late-session bars. Sources: the NYSE
Group releases "2025, 2026 and 2027 Holiday and Early Closings Calendar"
(2024-11-08) and "2026, 2027 and 2028 ..." (2025-12-23), and nyse.com
"Holidays & Trading Hours".
"""

from datetime import date, time

import pytest

from dashboard.backend.infrastructure.market_data import sessions
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
    is_trading_day,
    nyse_early_closes,
    nyse_holidays,
)

PUBLISHED_EARLY_CLOSES = {
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
