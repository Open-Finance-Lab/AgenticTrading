"""`market_data.sessions` is the one owner of the trading-session bounds.

Every in-session filter used to carry its own copy, and the copies disagreed at
the edges and on how the market name was spelled. These pin the helper's
semantics, that the filters built on it agree with each other, and that no new
copy of the bounds appears beside it.
"""

import ast
from datetime import time as clock_time
from pathlib import Path

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data import profiles, sessions

BACKEND = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("spelling", ["US", "us", " US ", None, ""])
def test_canonical_market_collapses_spellings_of_us(spelling):
    assert sessions.canonical_market(spelling) == "US"


def test_canonical_market_keeps_other_markets():
    assert sessions.canonical_market(" cn") == "CN"


def test_timezone_follows_the_profile_registry():
    assert sessions.timezone_for_market("us") == "US/Eastern"
    assert sessions.timezone_for_market("CN") == "Asia/Shanghai"
    # Unknown markets get US sessions from `session_windows`, so they get the
    # US zone too -- never one without the other.
    assert sessions.timezone_for_market("XX") == "US/Eastern"
    assert sessions.market_for_timezone("Asia/Shanghai") == "CN"
    assert sessions.market_for_timezone("US/Eastern") == "US"


def test_a_market_registered_with_two_timezones_is_refused(monkeypatch):
    us = profiles.get_market_profile(profiles.ALPACA)
    clash = profiles.MarketProfile(**{
        **{f: getattr(us, f) for f in us.__dataclass_fields__},
        "timezone": "America/Chicago",
    })
    monkeypatch.setitem(profiles._MARKET_PROFILES, ("clash", "x"), clash)
    with pytest.raises(ValueError, match="two timezones"):
        sessions.timezone_for_market("US")


@pytest.mark.parametrize(
    ("market", "local", "expected"),
    [
        ("US", clock_time(9, 29, 59), False),
        ("US", clock_time(9, 30), True),
        ("US", clock_time(16, 0), True),
        # The whole 16:00 minute, as the `hour == 16 and minute == 0` copies
        # the live US path ran on always admitted.
        ("US", clock_time(16, 0, 59), True),
        ("US", clock_time(16, 1), False),
        ("CN", clock_time(11, 30), True),
        ("CN", clock_time(12, 0), False),
        ("CN", clock_time(13, 0), True),
        ("CN", clock_time(15, 1), False),
        ("cn", clock_time(12, 0), False),
    ],
)
def test_time_in_session(market, local, expected):
    assert sessions.time_in_session(local, market) is expected


def test_naive_timestamps_are_read_as_market_local():
    naive = pd.Timestamp("2026-04-15 10:30")
    aware = pd.Timestamp("2026-04-15 10:30", tz="Asia/Shanghai")
    assert sessions.is_in_session(naive, market="CN", timezone="Asia/Shanghai")
    assert sessions.is_in_session(aware, market="CN", timezone="Asia/Shanghai")
    # The same instant is 22:30 the night before in New York.
    assert not sessions.is_in_session(aware, market="US", timezone="US/Eastern")


def _sample_timestamps():
    """Every 15 minutes over a US day, in UTC -- straddles both closes."""
    return list(
        pd.date_range("2026-04-15 12:00", "2026-04-15 22:00", freq="15min",
                      tz="UTC")
    ) + [pd.Timestamp("2026-04-15 20:00:30", tz="UTC")]  # 16:00:30 ET


def test_every_us_filter_agrees():
    """The engine, the baseline generator, the leaderboard strategies and the
    dataset store were four copies; a bar the store dropped and the engine kept
    was a step the protocol and dashboard paths counted differently."""
    from dashboard.backend import baseline_generator
    from dashboard.backend.domain.backtesting import market_data_store as mds
    from dashboard.backend.domain.leaderboard.strategies import _common

    stamps = _sample_timestamps()
    expected = [
        ts for ts in stamps
        if sessions.is_in_session(ts, market="US", timezone="US/Eastern")
    ]
    assert expected, "the sample must straddle the session"
    assert baseline_generator._market_hours_only(stamps, "US/Eastern") == expected
    assert _common.filter_market_hours(stamps) == expected
    frame = pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(stamps))
    assert mds._build_trading_timestamps(
        {"X": frame}, market="US", timezone="US/Eastern"
    ) == expected
    # The 16:00:30 stamp is in: the store's own first rewrite excluded it
    # while the engine kept it.
    assert pd.Timestamp("2026-04-15 20:00:30", tz="UTC") in expected


def test_the_engine_filter_agrees_for_both_markets():
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    stamps = _sample_timestamps()
    for data_source, market, zone in (
        (profiles.ALPACA, "US", "US/Eastern"),
        (profiles.IFIND_ASHARE, "CN", "Asia/Shanghai"),
    ):
        engine = HourlyBacktester.__new__(HourlyBacktester)
        engine.profile = profiles.get_market_profile(data_source)
        assert engine._market_hours_only(stamps) == [
            ts for ts in stamps
            if sessions.is_in_session(ts, market=market, timezone=zone)
        ]


def test_the_route_filter_uses_the_shared_bounds():
    from dashboard.backend.api.routers.backtests import filter_market_hours

    points = [
        {"timestamp": "2026-04-15T20:00:00Z"},  # 16:00 ET
        {"timestamp": "2026-04-15T20:15:00Z"},  # 16:15 ET
        {"timestamp": "2026-04-15T04:00:00Z"},  # 12:00 CST, CN lunch
        {"timestamp": "2026-04-15T02:30:00Z"},  # 10:30 CST
    ]
    assert filter_market_hours(points) == points[:1]
    assert filter_market_hours(
        points, market="cn", market_timezone="Asia/Shanghai"
    ) == points[3:]


_SESSION_LITERALS = {
    (9, 30), (11, 30), (13, 0), (15, 0), (16, 0),
}


def _session_bound_literals(tree):
    """`time(9, 30)`-style calls whose (hour, minute) is a session bound."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "time"
            and len(node.args) == 2
            and all(isinstance(a, ast.Constant) for a in node.args)
            and tuple(a.value for a in node.args) in _SESSION_LITERALS
        ):
            yield node.lineno
        # `hour == 16` / `16 * 60`: the integer spellings of the same bounds.
        if isinstance(node, ast.Compare) and any(
            isinstance(side, ast.Attribute) and side.attr == "hour"
            or isinstance(side, ast.Name) and side.id == "hour"
            for side in [node.left, *node.comparators]
        ) and any(
            isinstance(side, ast.Constant) and side.value in (9, 16)
            for side in [node.left, *node.comparators]
        ):
            yield node.lineno
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 60
            and isinstance(node.left, ast.Constant)
            and node.left.value in (9, 11, 13, 15, 16)
        ):
            yield node.lineno


def test_no_module_restates_the_session_bounds():
    """A new copy is how this broke: each looked right on its own. Build the
    check on the AST, not a substring, so a respelling cannot slip past."""
    owner = BACKEND / "infrastructure" / "market_data" / "sessions.py"
    offenders = []
    for path in BACKEND.rglob("*.py"):
        if "tests" in path.relative_to(BACKEND).parts or path == owner:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.relative_to(BACKEND)}:{line}"
            for line in _session_bound_literals(tree)
        ]
    assert offenders == [], (
        "session bounds restated outside market_data/sessions.py: "
        + ", ".join(offenders)
    )
