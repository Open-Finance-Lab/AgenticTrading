"""NYSE full-day holidays and early closes, computed by rule (no data file,
no dependency).

Shared US market-data infrastructure: the Live Trading Leaderboard walks it
forward day by day, and the Run Backtest modal's default period
(``provider.default_backtest_window``) walks it back week by week. It lived
under ``domain/leaderboard/`` while the board was its only reader; a calendar
the market-data layer needs belongs beside ``sessions.py``, not behind an
import into ``domain/``.

The board is where a weekday-only calendar first broke: a holiday became a
freeze day whose one-day increment had no bars and failed every model, and it
padded "Day N of M" and the chart axis with a session that never trades.

Covers the ten NYSE holidays in force since 2022 (Juneteenth added that year),
the dated unscheduled closures since 2004 (``UNSCHEDULED_CLOSURES``) and the
recurring 13:00 ET early closes (``nyse_early_closes``). An early close is
still a trading day. The hour it ends at is a session bound, so it lives
with the other bounds in ``sessions.py`` (``session_close_time``).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import FrozenSet, Iterator, Tuple


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    wk = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * wk) // 451
    month, day = divmod(h + wk - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


#: Full-day closures NYSE called at short notice, which no rule reproduces:
#: national days of mourning and Hurricane Sandy. Dated data beside the rules,
#: the split exchange_calendars makes (``adhoc_holidays`` in its XNYS
#: calendar, where these come from). The next one is a one-line addition; until
#: it lands, the Live board's session-gap check (``live._session_gap``) fails
#: that night's refresh loudly rather than trading a closed day.
UNSCHEDULED_CLOSURES: FrozenSet[date] = frozenset({
    date(2004, 6, 11),   # President Reagan
    date(2007, 1, 2),    # President Ford
    date(2012, 10, 29),  # Hurricane Sandy
    date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),   # President G. H. W. Bush
    date(2025, 1, 9),    # President Carter
})

#: 13:00 closes NYSE called at short notice. None since 2004; the set exists
#: so the next one is a one-line addition, not a change to the rules.
UNSCHEDULED_EARLY_CLOSES: FrozenSet[date] = frozenset()


@lru_cache(maxsize=16)
def nyse_holidays(year: int) -> FrozenSet[date]:
    """NYSE's full-day closures in ``year``: the rule-based holidays plus any
    :data:`UNSCHEDULED_CLOSURES`. Matches every year 2004-2035 of
    exchange_calendars' XNYS sessions (the test module pins the count)."""
    days = {
        _nth_weekday(year, 1, 0, 3),   # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),   # Washington's Birthday
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),     # Memorial Day
        _observed(date(year, 7, 4)),   # Independence Day
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),  # Christmas
    }
    # NYSE Rule 7.2: a Saturday New Year's Day is not observed on the Friday
    # before (that Friday closes a year). A Sunday one moves to Monday.
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        days.add(_observed(new_year))
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))  # Juneteenth
    days.update(day for day in UNSCHEDULED_CLOSURES if day.year == year)
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


@lru_cache(maxsize=16)
def nyse_early_closes(year: int) -> FrozenSet[date]:
    """NYSE's 13:00 ET early closes in ``year``: by rule, plus any
    :data:`UNSCHEDULED_EARLY_CLOSES`.

    - the day after Thanksgiving;
    - December 24 when it falls Monday to Thursday (a Friday one is the
      observed Christmas holiday, a weekend one is no session);
    - July 3 when Independence Day falls Tuesday to Friday (a Saturday July 4
      makes July 3 the observed holiday; a Sunday or Monday one puts July 3 on
      a weekend).

    NYSE Rule 7.2 lists holidays only; early closes are announced year by
    year, so these rules restate the announced pattern. They reproduce NYSE's
    published 2022-2028 calendars and every year 2004-2035 of
    exchange_calendars' XNYS calendar (the test module pins both); a year past
    NYSE's latest release is extrapolated. Earlier years followed other rules:
    a Thursday July 4 put the half day on Friday July 5 (1996, 2002), and
    1997-12-26, 1999-12-31 and 2003-12-26 were one-off half days.

    A candidate that is not a trading day is dropped, so this set can never
    contradict ``nyse_holidays``. An unscheduled early close missing from
    :data:`UNSCHEDULED_EARLY_CLOSES` goes unnoticed: bars still arrive after
    the bell, so the Live board's session-gap check sees a normal day.
    """
    candidates = {_nth_weekday(year, 11, 3, 4) + timedelta(days=1)}
    christmas_eve = date(year, 12, 24)
    if christmas_eve.weekday() < 4:
        candidates.add(christmas_eve)
    if 1 <= date(year, 7, 4).weekday() <= 4:
        candidates.add(date(year, 7, 3))
    candidates.update(day for day in UNSCHEDULED_EARLY_CLOSES if day.year == year)
    return frozenset(day for day in candidates if is_trading_day(day))


def is_early_close(day: date) -> bool:
    """Whether ``day`` is an NYSE session that ends at 13:00 ET."""
    return day in nyse_early_closes(day.year)


def trading_weeks_back(day: date) -> Iterator[Tuple[date, date]]:
    """First and last session of the Mon-Fri week holding ``day``, then of each
    earlier week, newest first. Unbounded: the caller stops it.

    Holidays trim the ends (Good Friday makes a week end on Thursday); a week
    with no session at all is skipped. Whether a week has *finished* is not a
    calendar question -- it depends on the clock and on when its data settles
    -- so this yields the current, still-trading week too and leaves that
    judgement to the caller (``provider.default_backtest_window``).
    """
    monday = day - timedelta(days=day.weekday())
    while True:
        sessions = [
            session
            for session in (monday + timedelta(days=offset) for offset in range(5))
            if is_trading_day(session)
        ]
        if sessions:
            yield sessions[0], sessions[-1]
        monday -= timedelta(days=7)
