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

from datetime import datetime, time

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

    Minute resolution, inclusive at both ends: a decision bar is stamped at the
    END of its bucket, so the last US bucket is stamped 16:00 and must survive.
    Seconds are dropped because every copy this replaced that the live US path
    ran on (`hour == 16 and minute == 0`) admitted the whole 16:00 minute; bars
    are minute-aligned, so nothing real sits inside that minute either way.
    """
    minute = local_time.replace(second=0, microsecond=0)
    return any(start <= minute <= end for start, end in session_windows(market))


def is_in_session(timestamp: datetime, *, market: object, timezone: str) -> bool:
    """:func:`time_in_session` for a timestamp in any zone.

    A naive timestamp is read as market-local time, matching
    ``bar_aggregation._as_local_index``; ``astimezone`` on a naive pandas
    ``Timestamp`` raises instead, which made the filter and the aggregation
    disagree on exactly the data a local-time feed returns.
    """
    zone = pytz.timezone(timezone)
    if timestamp.tzinfo is None:
        local = zone.localize(timestamp)
    else:
        local = timestamp.astimezone(zone)
    return time_in_session(local.time(), market)
