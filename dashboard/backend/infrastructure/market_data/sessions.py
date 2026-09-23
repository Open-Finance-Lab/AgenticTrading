"""The one owner of "which market, in which timezone, and is it open".

Every in-session filter in the backend used to carry its own copy of the
bounds -- the engine, the baseline generator, the leaderboard strategies, the
equity-curve route and the dataset store each spelled 09:30-16:00 as a literal,
and each normalised (or failed to normalise) the market name its own way. The
copies were not merely duplicates: they disagreed at the edges (`hour == 16 and
minute == 0` admits 16:00:59, `local_time <= time(16, 0)` does not), and the
store's was unconditionally US, so a CN profile aggregated on CN sessions and
was then filtered against 09:30-16:00 *ET* -- every bar dropped.

Lives in ``infrastructure/market_data`` rather than ``domain/`` so every layer
can import it: domain, the backend-root baseline generator and ``api/`` alike.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pytz

from dashboard.backend.infrastructure.market_data.profiles import (
    registered_market_timezones,
)

DEFAULT_MARKET = "US"


def canonical_market(market: object) -> str:
    """``None``/empty -> ``"US"``; otherwise trimmed and upper-cased.

    Anything that keys, branches or caches on a market must go through this:
    ``"us"``, ``"US "`` and ``None`` all select the same sessions, so treating
    them as distinct values splits one dataset into several cache entries.
    """
    return str(market or DEFAULT_MARKET).strip().upper() or DEFAULT_MARKET


def session_windows(market: object) -> tuple[tuple[time, time], ...]:
    """The market's trading sessions, in its own local time. Unknown -> US."""
    if canonical_market(market) == "CN":
        return ((time(9, 30), time(11, 30)), (time(13, 0), time(15, 0)))
    return ((time(9, 30), time(16, 0)),)


def timezone_for_market(market: object) -> str:
    """The timezone a registered profile pairs with ``market``. Unknown -> US.

    Derived from the profile registry rather than restated, so a new market
    gets its timezone from the same place its profile does.
    """
    zones = registered_market_timezones()
    return zones.get(canonical_market(market), zones[DEFAULT_MARKET])


def market_for_timezone(timezone: str) -> str:
    """Inverse of :func:`timezone_for_market`, for callers holding only a zone
    (the baseline generator's public API takes ``market_timezone`` alone)."""
    for market, zone in registered_market_timezones().items():
        if zone == timezone:
            return market
    return DEFAULT_MARKET


def time_in_session(local_time: time, market: object) -> bool:
    """Whether a wall-clock time in the market's own zone is in session.

    Minute resolution, inclusive at both ends. This is the rule for a bar
    stamped at the END of its bucket -- an aggregated decision bar, or an
    iFinD bar -- so the last US bucket is stamped 16:00 and must survive. A bar
    stamped at its OPEN (every Alpaca bar) is a different question; pass
    ``open_stamped_minutes`` to :func:`is_in_session` for it.
    Seconds are dropped because every copy this replaced that the live US path
    ran on (`hour == 16 and minute == 0`) admitted the whole 16:00 minute; bars
    are minute-aligned, so nothing real sits inside that minute either way.
    """
    minute = local_time.replace(second=0, microsecond=0)
    return any(start <= minute <= end for start, end in session_windows(market))


def _market_local(timestamp: datetime, timezone: str) -> datetime:
    """A naive timestamp is read as market-local time, matching
    ``bar_aggregation._as_local_index``; ``astimezone`` on a naive pandas
    ``Timestamp`` raises instead, which made the filter and the aggregation
    disagree on exactly the data a local-time feed returns."""
    zone = pytz.timezone(timezone)
    if timestamp.tzinfo is None:
        return zone.localize(timestamp)
    return timestamp.astimezone(zone)


def is_in_session(
    timestamp: datetime,
    *,
    market: object,
    timezone: str,
    open_stamped_minutes: int | None = None,
) -> bool:
    """:func:`time_in_session` for a timestamp in any zone.

    ``open_stamped_minutes`` says the timestamp is a bar's OPEN and the bar
    spans that many minutes; the bar is then in session when it CLOSES inside
    one (``start < close <= end``). Alpaca stamps every bar at its open and
    returns extended hours, so the close-stamp rule kept its 16:00 bar -- the
    16:00-16:05 (or 16:00-17:00) after-hours bar -- and, for clock-aligned
    hourly bars, dropped the 09:00 bar that holds the 09:30 open.
    """
    local = _market_local(timestamp, timezone)
    if open_stamped_minutes is None:
        return time_in_session(local.time(), market)
    close = local + timedelta(minutes=open_stamped_minutes)
    if close.date() != local.date():
        return False
    close_minute = close.time().replace(second=0, microsecond=0)
    return any(
        start < close_minute <= end for start, end in session_windows(market)
    )


def is_session_close(timestamp: datetime, *, market: object, timezone: str) -> bool:
    """Whether ``timestamp`` falls in the minute a session ends (16:00 ET,
    11:30 and 15:00 CST): the stamp of a final bucket, which no in-session
    source bar opens at."""
    minute = _market_local(timestamp, timezone).time().replace(
        second=0, microsecond=0
    )
    return any(minute == end for _start, end in session_windows(market))
