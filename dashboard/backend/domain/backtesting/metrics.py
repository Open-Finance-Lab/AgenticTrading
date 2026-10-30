"""Backtest performance metrics.

Extracted (Phase 2A) from ``HourlyBacktester._calc_sharpe`` and
``HourlyBacktester._calc_max_dd`` in ``dashboard/scripts/backtest_hourly_agent.py``.

These are pure functions over an equity curve represented as a list of dicts,
each containing an ``"equity"`` value. Inputs, outputs, edge-case behavior, and
the hourly annualization assumptions are identical to the original methods; the
legacy methods now delegate here.
"""

import logging
import math
import statistics
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger(__name__)


def calculate_sharpe(
    equity_curve: List[Dict], periods_per_year: Optional[float] = None
) -> float:
    """
    Calculate Sharpe ratio from hourly equity curve.

    Formula:
        sharpe = (mean(returns) / std(returns)) * sqrt(periods_per_year)

    ``periods_per_year`` defaults to the legacy hourly assumption.  A 5-minute
    valuation curve can pass ``252 * 6.5 * 12`` without changing the historical
    hourly behavior.

    Returns: float
        Annualized Sharpe ratio. Returns 0 if insufficient data, zero volatility,
        or an undefined per-step return (see below).
    """
    if len(equity_curve) < 2:
        return 0

    equities = np.array([e["equity"] for e in equity_curve], dtype=float)
    # A step whose *opening* equity is 0 has no defined return: the $0 runs made
    # legal on 2026-09-10 produce an all-zero curve, so every step is 0/0.
    #
    # This cannot be left to the zero-volatility exit below, because
    # ``np.std(nan) == 0`` is False -- NaN would fall straight through it and be
    # written to ``agent_runs.sharpe_ratio``. That is invisible on SQLite, which
    # coerces NaN to NULL on write, and fatal on the Postgres run-history
    # backend (``AGENT_RUNS_DATABASE_URL``), which round-trips it: Starlette
    # encodes with ``allow_nan=False``, so every route returning a plain dict
    # rather than a ``response_model`` answers 500 for that run.
    #
    # 0 is the same convention ``fractional_return`` settled on for the return
    # itself: with no capital there is nothing to have been at risk, so the
    # honest answer to "how was it risk-adjusted" is "it wasn't".
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equities) / equities[:-1]

    if len(returns) == 0 or not np.all(np.isfinite(returns)):
        return 0
    if np.std(returns) == 0:
        return 0

    periods_per_year = 252 * 6.5 if periods_per_year is None else periods_per_year
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    annualization_factor = np.sqrt(periods_per_year)
    return (np.mean(returns) / np.std(returns)) * annualization_factor


def calculate_max_drawdown(equity_curve: List[Dict]) -> float:
    """Calculate max drawdown."""
    if not equity_curve:
        return 0

    equities = np.array([e["equity"] for e in equity_curve])
    running_max = equities[0]
    max_dd = 0

    for equity in equities:
        if equity > running_max:
            running_max = equity
        # No peak means no drawdown from it. An unfunded run sits at 0 for its
        # whole curve, making this ``(0 - 0) / 0``. The old code already landed
        # on the right answer, but only by accident -- ``nan < max_dd`` is
        # False, so max_dd stayed 0 -- while numpy printed a RuntimeWarning into
        # the stdout of every $0 backtest.
        if running_max == 0:
            continue
        dd = (equity - running_max) / running_max
        if dd < max_dd:
            max_dd = dd

    return max_dd


# ---------------------------------------------------------------------------
# Daily Sharpe with its 95% interval (P19 on the boards; spec 2026-10-14 §4)
# ---------------------------------------------------------------------------
#
# ``calculate_sharpe`` above annualises whatever steps a curve holds (hourly
# marks, ddof 0, rf 0, undefined -> 0). The leaderboards publish this one
# instead: one return per US trading day, ddof 1, a stated cash rate, an
# interval, and None -- never 0, never NaN -- when it is undefined. It is read
# at request time from a stored curve, so it must stay pure arithmetic (no
# fetch, no DB) and never raise on stored data; an unreadable curve or seed is
# logged at ERROR rather than passed off as "pending" or "no variation".

SHARPE_TRADING_DAYS_PER_YEAR = 252
SHARPE_MIN_DAYS = 5
SHARPE_CONFIDENCE = 0.95
# The legacy engine has no T-bill source, so the boards state 0%/yr and say so.
SHARPE_RF_ANNUAL = 0.0

SHARPE_STATUS_OK = "ok"
SHARPE_STATUS_INSUFFICIENT_DAYS = "insufficient_days"
SHARPE_STATUS_NO_VARIATION = "no_variation"
SHARPE_STATUS_PENDING = "pending"

_SHARPE_TZ = ZoneInfo("America/New_York")
# Below this a daily-return stdev is float noise, not variation. A flat curve
# rescaled onto the board's capital (the committed contest board carries a
# 100000.00000000003 seed) can sit one ulp off E_0 on day 1, and that alone
# would publish a Sharpe of about 6.5 for an account that never moved.
_SHARPE_MIN_STDEV = 1e-12


def sharpe_standard_error_daily(sharpe_daily: float, n: int) -> float:
    """Standard error of a per-day Sharpe estimated from ``n`` iid daily returns.

    Lo (2002), iid case: ``sqrt((1 + SR_d**2 / 2) / n)``. It is Mertens'
    formula with skew 0 and kurtosis 3; both moment estimates are noise at the
    5-20 returns a month holds, so the iid form is the stable, hand-checkable
    choice. Not annualised: multiply by ``sqrt(252)`` for the yearly figure.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    return math.sqrt((1.0 + sharpe_daily * sharpe_daily / 2.0) / n)


# Both boards run this on every GET, so one broken row would otherwise log the
# same ERROR on every page load. Once per process per distinct break, the way
# live.py's ``_warned_live_config`` does; bounded so junk cannot grow it.
_logged_drift: set = set()
_LOGGED_DRIFT_MAX = 1000


def _log_drift_once(key: Tuple, message: str, *args: Any) -> None:
    if key in _logged_drift:
        return
    if len(_logged_drift) >= _LOGGED_DRIFT_MAX:
        _logged_drift.clear()
    _logged_drift.add(key)
    logger.error(message, *args)


def _point_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        stamp = value
    elif value is None:
        return None
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    # Naive stamps are UTC: the convention of every stored curve in this repo
    # (live.py ``_parse_to_et``, service.py ``_curve_span``).
    if stamp.tzinfo is None:
        try:
            stamp = stamp.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):  # pandas NaT is a datetime that refuses
            return None
    return stamp


def _finite_equity(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):  # an int past 1.8e308
        return None
    return number if math.isfinite(number) else None


def daily_session_closes(
    points: Iterable[Dict[str, Any]], *, tz: ZoneInfo = _SHARPE_TZ
) -> List[Tuple[date, float]]:
    """The last finite equity of each ``tz`` calendar date, in date order.

    Points are ordered by timestamp, not by list position. A point with no
    parseable timestamp, or an equity that is None or not finite, is "no
    observation" and skipped; 0 and negatives are real observations and kept.
    It never raises on stored data: the boards call it on their GET path.
    """
    observed: List[Tuple[datetime, int, date, float]] = []
    for index, point in enumerate(points or []):
        if not isinstance(point, Mapping):
            continue
        stamp = _point_time(point.get("timestamp"))
        equity = _finite_equity(point.get("equity"))
        if stamp is None or equity is None:
            continue
        try:
            day = stamp.astimezone(tz).date()
        except (OverflowError, ValueError):  # e.g. year 1 shifted west of UTC
            continue
        observed.append((stamp, index, day, equity))
    observed.sort(key=lambda item: (item[0], item[1]))
    closes: Dict[date, float] = {}
    for _stamp, _index, day, equity in observed:
        closes[day] = equity
    return sorted(closes.items())


def daily_sharpe_basis(
    *,
    rf_annual: float = SHARPE_RF_ANNUAL,
    confidence: float = SHARPE_CONFIDENCE,
    min_days: int = SHARPE_MIN_DAYS,
) -> Dict[str, Any]:
    """What a published ``sharpe_ratio`` means: one owner for the board copy."""
    return {
        "basis": "daily_returns",
        "annualization_days": SHARPE_TRADING_DAYS_PER_YEAR,
        "confidence": confidence,
        "min_days": min_days,
        "rf_annual": rf_annual,
        "stderr": "iid",
    }


def daily_sharpe_interval(
    points: Iterable[Dict[str, Any]],
    initial_equity: Any,
    *,
    rf_annual: float = SHARPE_RF_ANNUAL,
    confidence: float = SHARPE_CONFIDENCE,
    min_days: int = SHARPE_MIN_DAYS,
) -> Dict[str, Any]:
    """Annualised daily Sharpe and its interval, from an equity curve.

    ``E_0 = initial_equity``; ``E_d`` is the last finite point of US/Eastern
    date ``d``. ``r_d = E_d / E_{d-1} - 1``, ``x_d = r_d - rf_annual / 252``,
    ``SR = mean(x) / stdev(x, ddof=1) * sqrt(252)`` and the interval is
    ``SR -/+ z * sharpe_standard_error_daily(SR_d, n) * sqrt(252)``.

    Returns ``{"sharpe", "ci_low", "ci_high", "days", "status", "rf_annual"}``.
    ``days`` is the number of daily returns. The three values are floats only
    when ``status == "ok"`` and None otherwise -- never NaN, which Postgres
    round-trips and Starlette refuses to encode (``allow_nan=False``):

    * ``pending``: no daily close at all;
    * ``insufficient_days``: fewer than ``min_days`` returns;
    * ``no_variation``: a return is undefined (an ``E_{d-1}`` that is 0 or
      negative, or a seed that is not a positive number) or the returns never
      moved (stdev 0, or float noise below ``_SHARPE_MIN_STDEV``), or the
      arithmetic overflowed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if min_days < 2:
        raise ValueError("min_days must be at least 2: a sample stdev needs two returns")
    if not math.isfinite(rf_annual):
        raise ValueError("rf_annual must be finite")

    points = list(points or [])
    closes = [equity for _day, equity in daily_session_closes(points)]
    days = len(closes)

    def _result(status: str, sharpe=None, low=None, high=None) -> Dict[str, Any]:
        return {
            "sharpe": sharpe,
            "ci_low": low,
            "ci_high": high,
            "days": days,
            "status": status,
            "rf_annual": rf_annual,
        }

    if days == 0:
        if points:
            # Absent and broken must not look alike: a curve that has points but
            # no usable close is a stored-shape break, not a board still pending.
            _log_drift_once(
                ("no_close", len(points), repr(points[0])[:200]),
                "daily_sharpe: %d curve points but no usable daily close "
                "(timestamps or equity unreadable; first point %s); publishing "
                "status pending",
                len(points),
                repr(points[0])[:200],
            )
        return _result(SHARPE_STATUS_PENDING)
    if days < min_days:
        return _result(SHARPE_STATUS_INSUFFICIENT_DAYS)
    seed = _finite_equity(initial_equity)
    if seed is None or seed <= 0:
        _log_drift_once(
            ("seed", repr(initial_equity)[:200], days),
            "daily_sharpe: seed %r is not a positive number; publishing status "
            "no_variation for a %d-day curve",
            initial_equity,
            days,
        )
    previous_levels = [seed] + closes[:-1]
    if any(level is None or level <= 0 for level in previous_levels):
        return _result(SHARPE_STATUS_NO_VARIATION)

    rf_daily = rf_annual / SHARPE_TRADING_DAYS_PER_YEAR
    excess = [
        close / previous - 1.0 - rf_daily
        for previous, close in zip(previous_levels, closes)
    ]
    if not all(math.isfinite(value) for value in excess):
        return _result(SHARPE_STATUS_NO_VARIATION)
    try:
        mean = statistics.fmean(excess)
        stdev = statistics.stdev(excess)
    except (OverflowError, ValueError):
        return _result(SHARPE_STATUS_NO_VARIATION)
    if not math.isfinite(mean) or not math.isfinite(stdev) or stdev <= _SHARPE_MIN_STDEV:
        return _result(SHARPE_STATUS_NO_VARIATION)

    sharpe_daily = mean / stdev
    annualise = math.sqrt(SHARPE_TRADING_DAYS_PER_YEAR)
    sharpe = sharpe_daily * annualise
    half_width = (
        statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
        * sharpe_standard_error_daily(sharpe_daily, days)
        * annualise
    )
    low, high = sharpe - half_width, sharpe + half_width
    if not all(math.isfinite(value) for value in (sharpe, low, high)):
        return _result(SHARPE_STATUS_NO_VARIATION)
    return _result(SHARPE_STATUS_OK, sharpe, low, high)
