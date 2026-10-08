"""NYSE full-day holidays, computed by rule (no data file, no dependency).

Shared US market-data infrastructure: the Live Trading Leaderboard walks it
forward day by day, and the Run Backtest modal's default period
(``provider.default_backtest_window``) walks it back week by week. It lived
under ``domain/leaderboard/`` while the board was its only reader; a calendar
the market-data layer needs belongs beside ``sessions.py``, not behind an
import into ``domain/``.

The board is where a weekday-only calendar first broke: a holiday became a
freeze day whose one-day increment had no bars and failed every model, and it
padded "Day N of M" and the chart axis with a session that never trades.

Covers the ten NYSE holidays in force since 2022 (Juneteenth added that year).
Early closes (13:00 ET) are not modelled; they are still trading days.
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


@lru_cache(maxsize=16)
def nyse_holidays(year: int) -> FrozenSet[date]:
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
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


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
