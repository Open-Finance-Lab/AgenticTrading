"""`market_data.sessions` is the one owner of the trading-session bounds.

Every in-session filter used to carry its own copy, and the copies disagreed at
the edges and on how the market name was spelled. These pin the helper's
semantics, that the filters built on it agree with each other, and that no new
copy of the bounds appears beside it.
"""

import ast
import functools
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
    was a step the protocol and dashboard paths counted differently. Each
    stamp convention is one rule, whichever of them applies it."""
    from dashboard.backend import baseline_generator
    from dashboard.backend.domain.backtesting import market_data_store as mds
    from dashboard.backend.domain.leaderboard.strategies import _common

    stamps = _sample_timestamps()
    for open_minutes in (None, 60):
        frame = pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(stamps))
        if open_minutes is not None:
            frame.attrs[sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES] = open_minutes
        expected = [
            ts for ts in stamps
            if sessions.is_in_session(
                ts,
                market="US",
                timezone="US/Eastern",
                open_stamped_minutes=open_minutes,
            )
        ]
        assert expected, "the sample must straddle the session"
        # Each reads the convention off the frame, as its callers hand it one.
        assert baseline_generator._market_hours_only(
            stamps, "US/Eastern", {"X": frame}
        ) == expected
        assert mds._build_trading_timestamps(
            {"X": frame}, market="US", timezone="US/Eastern"
        ) == expected
        assert _common.market_timestamps({"X": frame}) == expected
        assert _common.filter_market_hours(
            stamps, open_stamped_minutes=open_minutes
        ) == expected
    # The 16:00:30 stamp is in for a close-stamped bar: the store's own first
    # rewrite excluded it while the engine kept it.
    assert sessions.is_in_session(
        pd.Timestamp("2026-04-15 20:00:30", tz="UTC"),
        market="US",
        timezone="US/Eastern",
    )


def _et(clock):
    return pd.Timestamp(f"2026-04-15 {clock}", tz="US/Eastern")


@pytest.mark.parametrize(
    ("clock", "minutes", "kept"),
    [
        # Alpaca 5m, stamped at the open: 09:30 through 15:55.
        ("09:25", 5, False),
        ("09:30", 5, True),
        ("15:55", 5, True),
        ("16:00", 5, False),  # 16:00-16:05 is after hours
        # Alpaca 1h, clock-aligned: 10:00 through 15:00.
        ("08:00", 60, False),
        ("09:00", 60, False),  # 09:00-10:00 holds 30 minutes of pre-market
        ("10:00", 60, True),
        ("15:00", 60, True),
        ("16:00", 60, False),  # the bar the close-stamp rule used to keep
    ],
)
def test_an_open_stamped_bar_is_in_session_only_when_wholly_inside_one(
    clock, minutes, kept
):
    assert sessions.is_in_session(
        _et(clock),
        market="US",
        timezone="US/Eastern",
        open_stamped_minutes=minutes,
    ) is kept


def test_open_stamped_hourly_day_is_six_bars():
    """The raw 1h path keeps 10:00-15:00. The close-stamp rule kept 10:00-16:00,
    the last of which is after hours; 09:00 straddles the open, so its prices
    are part pre-market and it cannot stand in for the 09:30-10:00 half hour."""
    day = pd.date_range("2026-04-15 04:00", "2026-04-15 19:00", freq="h",
                        tz="US/Eastern")
    kept = [
        ts.strftime("%H:%M") for ts in day
        if sessions.is_in_session(
            ts, market="US", timezone="US/Eastern", open_stamped_minutes=60
        )
    ]
    assert kept == ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def test_session_close_is_each_window_end():
    assert sessions.is_session_close(_et("16:00"), market="US", timezone="US/Eastern")
    assert not sessions.is_session_close(
        _et("15:30"), market="US", timezone="US/Eastern"
    )
    for clock, is_close in (("11:30", True), ("15:00", True), ("13:00", False)):
        assert sessions.is_session_close(
            pd.Timestamp(f"2026-04-15 {clock}", tz="Asia/Shanghai"),
            market="CN",
            timezone="Asia/Shanghai",
        ) is is_close


def test_the_alpaca_loader_stamps_every_frame_it_returns(monkeypatch):
    """Stamped above the bar cache, so a hit carries the convention too."""
    from dashboard.backend.infrastructure.market_data.alpaca_bars import (
        AlpacaDataLoader,
    )

    loader = AlpacaDataLoader.__new__(AlpacaDataLoader)
    for timeframe, span in (("5m", 5), ("60m", 60)):
        loader.source_timeframe = timeframe
        monkeypatch.setattr(
            loader,
            "_fetch_bars_resolved",
            lambda symbols, start, end: {
                symbol: pd.DataFrame({"close": [1.0]}) for symbol in symbols
            },
        )
        frames = loader.fetch_bars(["AAPL", "MSFT"], "2026-04-15", "2026-04-16")
        assert sessions.frames_open_stamped_minutes(frames) == span


def test_frames_open_stamped_minutes_refuses_a_mix():
    def frame(span=None, rows=1):
        result = pd.DataFrame({"close": [1.0] * rows})
        if span is not None:
            result.attrs[sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES] = span
        return result

    assert sessions.frames_open_stamped_minutes({}) is None
    assert sessions.frames_open_stamped_minutes({"A": frame()}) is None
    # An empty frame holds no bars to misfilter, whatever it carries.
    assert sessions.frames_open_stamped_minutes(
        {"A": frame(5), "B": frame(rows=0)}
    ) == 5
    with pytest.raises(ValueError, match="mix stamp conventions"):
        sessions.frames_open_stamped_minutes({"A": frame(5), "B": frame()})


def test_the_final_bucket_fills_at_the_last_in_session_close():
    from dashboard.backend.domain.backtesting.bar_aggregation import (
        plan_execution_fills,
    )

    plan = functools.partial(
        plan_execution_fills, source_minutes=5, market="US", timezone="US/Eastern"
    )
    source = [_et("15:25"), _et("15:30"), _et("15:55")]
    decisions = [_et("15:30"), _et("16:00"), _et("15:45")]
    assert plan(decisions, source) == {
        _et("15:30"): (_et("15:30"), "open", _et("15:30")),
        # Priced at the 15:55 bar's close and stamped then: 16:00, never before
        # the decision it fills.
        _et("16:00"): (_et("15:55"), "close", _et("16:00")),
        # Not a session close and no bar opens there: no fill.
    }
    # A day with no source bars has nothing to fill the close on.
    assert plan([_et("16:00")], []) == {}
    # Nor does one whose last bar ends before the close: 15:50's close is 15:55.
    assert plan([_et("16:00")], [_et("15:45"), _et("15:50")]) == {}


def test_an_exact_fill_is_the_source_bar_not_the_equal_decision_stamp():
    from dashboard.backend.domain.backtesting.bar_aggregation import (
        plan_execution_fills,
    )

    # Aggregated decisions arrive in UTC, source bars in ET: equal instants,
    # but the trade is stamped with the fill bar, so its tz must survive.
    decision = _et("15:30").tz_convert("UTC")
    fills = plan_execution_fills(
        [decision],
        [_et("15:30")],
        source_minutes=5,
        market="US",
        timezone="US/Eastern",
    )
    assert str(fills[decision].bar.tz) == "US/Eastern"
    assert str(fills[decision].filled_at.tz) == "US/Eastern"


def test_the_engine_filter_agrees_for_both_markets():
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    stamps = _sample_timestamps()
    for data_source, market, zone in (
        (profiles.ALPACA, "US", "US/Eastern"),
        (profiles.IFIND_ASHARE, "CN", "Asia/Shanghai"),
    ):
        engine = HourlyBacktester.__new__(HourlyBacktester)
        engine.profile = profiles.get_market_profile(data_source)
        close_stamped = {"X": pd.DataFrame({"close": 1.0}, index=pd.DatetimeIndex(stamps))}
        assert engine._market_hours_only(stamps, close_stamped) == [
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


@pytest.mark.parametrize(
    "strategy_key", ["buy_hold", "equal_weight_buyhold", "equal_weight_index"]
)
def test_leaderboard_baselines_read_the_board_bars_as_open_stamped(strategy_key):
    """These three hand the board's raw Alpaca hourly bars to the baseline
    generator, which reads the stamp convention the loader left on them."""
    from dashboard.backend.domain.leaderboard.strategies import get_strategy

    hours = pd.date_range("2026-04-15 08:00", "2026-04-15 18:00", freq="h",
                          tz="US/Eastern")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + i for i in range(len(hours))],
            "volume": 1000,
        },
        index=hours.tz_convert("UTC"),
    )
    frame.attrs[sessions.FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    strategy = get_strategy({"strategy": strategy_key, "symbols": ["AAPL", "MSFT"]})
    curve = strategy.run(
        {"AAPL": frame, "MSFT": frame.copy()}, "2026-04-15", "2026-04-15", 10_000.0
    )
    clocks = [
        pd.Timestamp(point["timestamp"]).tz_convert("US/Eastern").strftime("%H:%M")
        for point in curve
    ]
    assert clocks[0] == "10:00" and clocks[-1] == "15:00"
