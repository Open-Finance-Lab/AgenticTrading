"""Fetch real market-index levels (e.g. ^DJI, ^GSPC) from Yahoo Finance.

Alpaca only serves tradeable securities (ETFs like DIA/SPY), not the underlying
index. ETFs drift off the index (dividends, NAV premium), so for a true
"market index" baseline we pull the index series directly from Yahoo.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

from dashboard.backend.infrastructure.market_data.sessions import (
    session_close_time,
    session_windows,
)
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
)

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_US_EXCHANGE_ZONE = "America/New_York"


class YahooChartError(requests.RequestException):
    """Yahoo answered, but with a failure envelope instead of chart data.

    A ``RequestException`` subclass on purpose: to every caller this is the same
    class of problem as a 429 or a timeout — the upstream did not deliver, a
    retry may help, and whatever the caller renders is *incomplete* rather than
    simply empty. The status code cannot make that call on its own, because
    Yahoo reports some outages *inside* a 200 body (``chart.error`` set with
    ``chart.result`` null) and serves consent/interstitial pages carrying no
    ``chart`` key at all. Classifying those as "no data for this window" is
    exactly the absent-vs-broken collapse CLAUDE.md warns about.
    """


def _epoch(date_str: str) -> int:
    """Epoch seconds for ``YYYY-MM-DD`` (midnight UTC) or a full ISO-8601 stamp.

    Run windows arrive here in two shapes: plain dates typed into a backtest,
    and full ``datetime.isoformat()`` stamps written by the paper-trading
    baselines (``domain/backtesting/baselines/paper.py``) and by
    ``api/routers/paper_trading.py``. Parsing only the former made an ISO stamp
    raise ``ValueError`` *before* any HTTP call — so it could never be a
    ``RequestException``, and it escaped every transport-level guard as a 500.
    """
    text = (date_str or "").strip()
    if not text:
        raise ValueError("empty date string")
    try:
        parsed = dt.datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp())


def usable_window(start_date: str, end_date: str) -> bool:
    """Whether both ends of a run window parse into something Yahoo can be asked for.

    Lets a caller skip the request entirely rather than discover the problem as
    an exception from inside the fetch. A window that does not parse is
    *permanently* unusable — unlike an outage, retrying it never helps.
    """
    try:
        _epoch(start_date)
        _epoch(end_date)
    except ValueError:
        return False
    return True


def fetch_index_hourly(
    symbol: str,
    start_date: str,
    end_date: str,
    timeout: int = 20,
) -> List[Tuple[dt.datetime, float]]:
    """Return [(timestamp_utc, close)] hourly index points within [start, end].

    Yahoo's hourly endpoint ignores period2 and returns through the present, so
    results are filtered to the requested window here.
    """
    start_e = _epoch(start_date)
    end_e = _epoch(end_date) + 86400  # inclusive of end_date

    resp = requests.get(
        _CHART_URL.format(symbol=symbol),
        params={"period1": start_e, "period2": end_e, "interval": "1h"},
        headers=_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()

    payload = resp.json()
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        # No chart object at all: a consent page, an interstitial, or an error
        # shape we don't know. Never "the window was empty".
        raise YahooChartError(f"{symbol}: response carried no chart object")

    error = chart.get("error")
    if error:
        detail = error
        if isinstance(error, dict):
            detail = error.get("description") or error.get("code") or error
        raise YahooChartError(f"{symbol}: {detail}")

    results = chart.get("result")
    if results is None:
        # Yahoo pairs a null result with an error; a null result *without* one
        # is a contract break, not an empty window.
        raise YahooChartError(f"{symbol}: chart.result was null with no error")
    if not results:
        return []

    res = results[0]
    timestamps = res.get("timestamp") or []
    quote = (res.get("indicators") or {}).get("quote") or [{}]
    closes = quote[0].get("close") or []

    points: List[Tuple[dt.datetime, float]] = []
    for ts, close in zip(timestamps, closes):
        if close is None or not (start_e <= ts < end_e):
            continue
        points.append((dt.datetime.fromtimestamp(ts, dt.timezone.utc), float(close)))
    return _restamp_close_levels(points, (res.get("meta") or {}).get("exchangeTimezoneName"))


def _restamp_close_levels(
    points: List[Tuple[dt.datetime, float]], exchange_zone: Optional[str]
) -> List[Tuple[dt.datetime, float]]:
    """Move a US index level stamped *at* the session close into the half bar it closes.

    Every other point here is open-stamped on Yahoo's hourly grid, which starts
    at the 09:30 open, so a session's last slot is a half bar (15:30-16:00). On
    an NYSE early close Yahoo returns that half bar (12:30-13:00) with a null
    close and sends the closing level as a separate point stamped 13:00. That
    was recorded for ^DJI and ^GSPC on every early close from 2024-11-29 to
    2025-12-24; full sessions carry no such point. Any reader that treats the
    series as open-stamped (the Live board's ``_axis_node_key``) drops a 13:00
    stamp as after-hours, so the day would end on its 12:30 level. Restamped to
    the half bar's open, the closing level lands where that half bar's close
    would. If Yahoo also filled the half bar, the closing level replaces it.

    That level is the official close, auction included: ^DJI 47,716.42 on
    2025-11-28, Yahoo's daily close to the cent. On a full session the 15:30
    half bar's close is pre-auction instead (47,419.67 against an official
    47,427.12 on 2025-11-26), as every agent and Alpaca-baseline mark is on
    both kinds of day (``sessions.is_in_session``). The index curve therefore
    ends a half day about 0.02% off the basis it ends a full day on. That is
    the lesser gap: the only pre-auction level on offer is the 11:30 bar's,
    30 minutes stale.

    Only on an NYSE early close, for a US exchange. On a full session the one
    point stamped at the close is the latest close Yahoo appends to every
    response, inside the window whenever a fetch reaches today; moving it would
    mark today on the official close and a replay of the same day pre-auction.
    """
    if exchange_zone != _US_EXCHANGE_ZONE:
        return points
    zone = ZoneInfo(_US_EXCHANGE_ZONE)
    session_open = session_windows("US")[0][0]
    hour = dt.timedelta(hours=1)
    by_stamp: Dict[dt.datetime, float] = {}
    for stamp, level in points:
        local = stamp.astimezone(zone)
        if is_early_close(local.date()) and local.time() == session_close_time(
            local.date(), "US"
        ):
            opened = dt.datetime.combine(local.date(), session_open, tzinfo=zone)
            last_slot = (local - opened) % hour or hour
            stamp = (local - last_slot).astimezone(dt.timezone.utc)
        by_stamp[stamp] = level
    return sorted(by_stamp.items())
