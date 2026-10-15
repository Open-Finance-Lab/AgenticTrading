"""The Live board's decision cadence, pinned (atl-direction spec §6, §7).

Six model decisions per full session, on the open-stamped hourly bars 10:00 to
15:00 ET, and three on an NYSE early close (10:00 to 12:00). The budget in spec
§7 (126 calls for a 21-session month) is this number times the calendar, so a
filter change that adds or drops one bar a day moves the bill by a sixth.
Nothing here is new behaviour: these stop the cadence drifting.
"""

from collections import Counter
from datetime import date

import pandas as pd
import pytest

from dashboard.backend.domain.leaderboard import live
from dashboard.backend.domain.leaderboard.strategies import _common, llm_agent
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
)
from dashboard.backend.infrastructure.market_data.us_market_calendar import (
    is_early_close,
)

FULL_SESSION = ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]
EARLY_CLOSE = ["10:00", "11:00", "12:00"]


def _alpaca_hourly(days):
    """Open-stamped hourly bars, 04:00-19:00 ET on each day: Alpaca serves
    extended hours, so the late session after an early close is present."""
    index = pd.DatetimeIndex([
        ts
        for day in days
        for ts in pd.date_range(f"{day} 04:00", f"{day} 19:00", freq="h",
                                tz="US/Eastern")
    ]).tz_convert("UTC")
    frame = pd.DataFrame({"close": 100.0}, index=index)
    frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    return {"AAPL": frame}


def _clocks_by_day(stamps):
    days = {}
    for ts in stamps:
        et = ts.tz_convert("US/Eastern")
        days.setdefault(et.date(), []).append(et.strftime("%H:%M"))
    return days


def test_six_decision_bars_on_a_full_session_and_three_on_an_early_close():
    stamps = _common.market_timestamps(_alpaca_hourly(["2026-11-25", "2026-11-27"]))
    assert _clocks_by_day(stamps) == {
        date(2026, 11, 25): FULL_SESSION,
        date(2026, 11, 27): EARLY_CLOSE,
    }


class _CountingManager:
    """Stands in for PortfolioManager: records every model call."""

    calls = []

    def __init__(self, **kwargs):
        self.equity_history = []
        self.trades = []
        self.llm_calls = self.llm_decisions = 0
        self.input_tokens = self.output_tokens = 0

    def restore_state(self, snapshot):
        pass

    def snapshot_state(self):
        return {"cash": 1.0, "positions": {}}

    def get_portfolio_state(self, *args):
        return {}

    def make_trading_decision_with_llm(self, state, client, **kwargs):
        _CountingManager.calls.append(state["timestamp"])
        self.llm_calls += 1
        self.llm_decisions += 1
        return {"actions": []}

    def execute_actions(self, *args):
        pass

    def update_equity(self, market_data, price_cache, ts):
        self.equity_history.append({"timestamp": ts, "equity": 1.0})

    def get_equity_curve(self):
        return self.equity_history


def test_the_decision_loop_asks_the_model_once_per_kept_bar(monkeypatch):
    _CountingManager.calls = []
    monkeypatch.setattr(llm_agent, "PortfolioManager", _CountingManager)
    stamps = _common.market_timestamps(_alpaca_hourly(["2026-11-25", "2026-11-27"]))
    strategy = llm_agent.LLMAgentStrategy({"strategy": "llm_agent", "model_id": "m"})
    closes = []
    strategy._run_decision_loop(
        client=object(),
        timestamps=stamps,
        symbols=["AAPL"],
        data={},
        price_cache={},
        initial_capital=1.0,
        model_id="m",
        on_session_close=lambda day, curve: closes.append((day, strategy.llm_calls)),
    )
    assert Counter(
        ts.tz_convert("US/Eastern").date() for ts in _CountingManager.calls
    ) == {date(2026, 11, 25): 6, date(2026, 11, 27): 3}
    # One checkpoint per session, the counters cumulative when it fires.
    assert closes == [(date(2026, 11, 25), 6), (date(2026, 11, 27), 9)]


@pytest.mark.parametrize(
    ("month_start", "month_end", "sessions", "early_closes", "decisions"),
    [
        ("2026-08-01", "2026-08-31", 21, 0, 126),
        ("2026-11-01", "2026-11-30", 20, 1, 117),
        ("2026-12-01", "2026-12-31", 22, 1, 129),
    ],
)
def test_a_months_decisions_follow_the_calendar(
    month_start, month_end, sessions, early_closes, decisions
):
    """Spec §7's 126 calls is 6 x 21 sessions. An early close takes three off;
    retries are billed on top of this floor."""
    days = live._trading_days_inclusive(
        date.fromisoformat(month_start), date.fromisoformat(month_end)
    )
    assert len(days) == sessions
    assert sum(is_early_close(day) for day in days) == early_closes
    stamps = _common.market_timestamps(_alpaca_hourly([d.isoformat() for d in days]))
    assert len(stamps) == decisions == 6 * sessions - 3 * early_closes


def test_every_axis_node_after_a_days_first_is_where_a_decision_bar_closes():
    """The 10:00 node carries the prior close (the 09:00 bar straddles the open
    and is never a decision); every later node is one decision bar's close.
    Pins the chart axis and the decision filter to one calendar."""
    days = live._trading_days_inclusive(date(2026, 11, 1), date(2026, 11, 30))
    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    stamps = _common.market_timestamps(_alpaca_hourly([d.isoformat() for d in days]))
    decision_nodes = [live._axis_node_key(live._parse_to_et(ts.isoformat())) for ts in stamps]
    first_nodes = {f"{day.isoformat()}T10:00" for day in days}
    assert decision_nodes == [ts[:16] for ts in axis if ts[:16] not in first_nodes]
