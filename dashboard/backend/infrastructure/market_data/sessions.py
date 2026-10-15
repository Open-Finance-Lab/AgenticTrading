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

from datetime import date, datetime, time, timedelta
from typing import Any, Mapping

import pytz

from dashboard.backend.infrastructure.market_data.profiles import (
    registered_market_timezones,
)
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
)

DEFAULT_MARKET = "US"


def canonical_market(market: object) -> str:
    """``None``/empty -> ``"US"``; otherwise trimmed and upper-cased.

    Anything that keys, branches or caches on a market must go through this:
    ``"us"``, ``"US "`` and ``None`` all select the same sessions, so treating
    them as distinct values splits one dataset into several cache entries.
    """
    return str(market or DEFAULT_MARKET).strip().upper() or DEFAULT_MARKET


_US_SESSIONS: tuple[tuple[time, time], ...] = ((time(9, 30), time(16, 0)),)
_CN_SESSIONS: tuple[tuple[time, time], ...] = (
    (time(9, 30), time(11, 30)),
    (time(13, 0), time(15, 0)),
)

#: The US cash close on an NYSE early-close day
#: (``us_market_calendar.nyse_early_closes``).
EARLY_CLOSE_TIME: time = time(13, 0)


def _trades_us_sessions(market: object) -> bool:
    """Whether ``market`` trades the US cash session. Unknown -> US."""
    return canonical_market(market) != "CN"


def session_windows(market: object) -> tuple[tuple[time, time], ...]:
    """The market's trading sessions on a full day, in its own local time.
    Unknown -> US. For a particular date, use :func:`day_session_windows`:
    an NYSE early close ends the US session at 13:00."""
    return _US_SESSIONS if _trades_us_sessions(market) else _CN_SESSIONS


def day_session_windows(day: date, market: object) -> tuple[tuple[time, time], ...]:
    """:func:`session_windows` for ``day``: a US day ends at
    :data:`EARLY_CLOSE_TIME` on the NYSE early-close days the calendar lists.
    Any other market's sessions do not vary by date here. Whether ``day``
    trades at all is the calendar's question
    (``us_market_calendar.is_trading_day``), not this one's.

    Read by every rule that places a bar against the day's close: the
    open-stamped filter (:func:`is_in_session`), the session-close fill
    (:func:`is_session_close`) and the bucketing in
    ``bar_aggregation.aggregate_bars``. They have to agree. When only the
    filter knew about the 13:00 close, aggregation still built a 13:30 bucket
    from the 12:30-12:59 minutes, nothing filled it, and an early-close day lost
    its closing decision. The close-stamped rule (:func:`time_in_session`) does
    not read it and still admits a bar stamped up to 16:00 (D19 in the US
    conformance design, owned by the engine rewrite). No US source this repo
    loads is close-stamped, and aggregation no longer produces a decision bar
    after 13:00 on an early close, so nothing reaches that gap today.
    """
    if _trades_us_sessions(market) and is_early_close(day):
        return ((_US_SESSIONS[0][0], EARLY_CLOSE_TIME),)
    return session_windows(market)


def session_close_time(day: date, market: object = DEFAULT_MARKET) -> time:
    """When ``market``'s last session ends on ``day``, in its own local time:
    16:00 ET, or :data:`EARLY_CLOSE_TIME` on an NYSE early close, for US
    sessions; the last window's end for any other market."""
    return day_session_windows(day, market)[-1][1]


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


#: ``DataFrame.attrs`` key a loader sets on every frame whose bars it stamps at
#: their OPEN, holding the bar span in minutes. Absent means stamped at the
#: close -- an aggregated decision bar, an iFinD bar, a legacy double. It lives
#: on the frame because the convention is a fact about the data, not about
#: whoever is filtering it: passing it by hand let the protocol baseline worker,
#: which builds its backtester without loading anything, filter aggregated
#: close-stamped bars under the raw-bar rule and drop every 16:00 close.
FRAME_ATTR_OPEN_STAMPED_MINUTES = "bar_open_stamped_minutes"


def frames_open_stamped_minutes(frames: Mapping[str, Any]) -> int | None:
    """The span the loader stamped on ``frames``, or ``None`` if their bars are
    stamped at the close. Empty frames carry no bars and are ignored.

    Raises ``ValueError`` on a mix: no one rule filters both conventions, and
    choosing one would silently misfilter the other half of the universe.
    """
    spans = {
        frame.attrs.get(FRAME_ATTR_OPEN_STAMPED_MINUTES)
        for frame in frames.values()
        if len(frame)
    }
    if len(spans) > 1:
        raise ValueError(
            f"bars mix stamp conventions (open-stamped spans {sorted(spans, key=str)})"
        )
    return spans.pop() if spans else None


def market_local(timestamp: datetime, timezone: str) -> datetime:
    """``timestamp`` in the market's zone. A naive timestamp is read as
    market-local time, matching ``bar_aggregation._as_local_index``;
    ``astimezone`` on a naive pandas ``Timestamp`` raises instead, which made
    the filter and the aggregation disagree on exactly the data a local-time
    feed returns."""
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
    spans that many minutes (read it off the frames with
    :func:`frames_open_stamped_minutes`). Such a bar is in session only when it
    lies wholly inside one: it opens at or after the session starts and closes
    at or before it ends. Alpaca stamps bars at their open and serves extended
    hours, so the close-stamp rule kept its 16:00 bar -- 16:00-16:05, or
    16:00-17:00, after hours. Clock-aligned hourly bars straddle the 09:30
    open, and that 09:00 bar is out too: its open, high, low and volume include
    half an hour of pre-market trading, so admitting it would price the day's
    first fill off a pre-market print.

    The day's sessions come from :func:`day_session_windows`, so the last one
    ends at 13:00 ET on an NYSE early close. Trading goes on after that bell (NYSE's late sessions,
    FINRA TRF post-market reporting and Alpaca's half-day extended session
    all run to 17:00), so bars stamped 13:00 and later can carry real prints:
    the rule drops them by their stamp, never by their absence. The
    close-stamped rule (:func:`time_in_session`) still ends every US day at
    16:00 (see :func:`day_session_windows`).

    The close mark this leaves is pre-auction on every US day. The last kept
    bar (the 12:00 bar on an early close, 15:00 otherwise) closes on the last
    trade before the bell, not the official close: the closing-auction print,
    stamped at or just after 16:00:00, opens the next bar, which is dropped
    (measured on SIP), and the 13:00 auction is expected to do the same on an
    early close. Marking on that bar instead was rejected, because post-close
    prints in the same hour can set its close.
    """
    local = market_local(timestamp, timezone)
    if open_stamped_minutes is None:
        return time_in_session(local.time(), market)
    close = local + timedelta(minutes=open_stamped_minutes)
    if close.date() != local.date():
        return False
    open_minute = local.time().replace(second=0, microsecond=0)
    close_minute = close.time().replace(second=0, microsecond=0)
    return any(
        start <= open_minute and close_minute <= end
        for start, end in day_session_windows(local.date(), market)
    )


def is_session_close(timestamp: datetime, *, market: object, timezone: str) -> bool:
    """Whether ``timestamp`` falls in the minute one of that day's sessions
    ends (16:00 ET, 13:00 on an NYSE early close; 11:30 and 15:00 CST): the
    stamp of a final bucket, which no in-session source bar opens at."""
    local = market_local(timestamp, timezone)
    minute = local.time().replace(second=0, microsecond=0)
    return any(
        minute == end for _start, end in day_session_windows(local.date(), market)
    )
