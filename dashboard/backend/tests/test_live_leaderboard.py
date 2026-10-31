"""Tests for the calendar-month Live Trading Leaderboard.

Isolated from contest/daily: live freeze uses ``leaderboard-live`` and must
not rewrite the contest window.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.database import db
import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.infrastructure.llm.validator import LLM_SELL_SIZING

SEED = float(lb_service.load_leaderboard_config().get("initial_capital", 100_000))

_ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _isolated_live_session(monkeypatch, tmp_path):
    """The test DB is session-scoped: start every test with no live rows, and
    never let a refresh write the real state file under DATA_DIR."""
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    _clear_live_and_spend_rows()
    yield
    _clear_live_and_spend_rows()


def _clear_live_and_spend_rows():
    # Every live deploy also prices itself into the spend ledger's session.
    for session in ("leaderboard-live", "leaderboard-live-spend"):
        for run in db.get_runs_by_session(session) or []:
            db.delete_run(run["run_id"])


@pytest.fixture(autouse=True)
def _roster_without_join_dates(monkeypatch):
    """These cases are dated August 2026, before any entry's ``live_from``
    month, and are about the board, not about who joins when: read without
    that key, the whole roster is live. The join gate has its own cases in
    test_live_roster.py, against the real config."""
    real = lb_service.load_leaderboard_config

    def without_join_dates():
        cfg = real()
        return {
            **cfg,
            "strategies": [
                {k: v for k, v in s.items() if k != live.LIVE_FROM_KEY}
                for s in cfg["strategies"]
            ],
        }

    monkeypatch.setattr(lb_service, "load_leaderboard_config", without_join_dates)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def no_alpaca(monkeypatch):
    """The read path must never reach market data or the baseline recompute."""

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the live GET path reached the compute path")

    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", _must_not_run)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _must_not_run)


def _insert_live_run(run_id, llm_model, end_date, points, *, final=None,
                     total_return=0.0, sharpe=0.0, max_dd=0.0, metadata=None,
                     start_date="2026-08-01", num_trades=0,
                     sell_rule=LLM_SELL_SIZING, **extra):
    if metadata is not None and live.LIVE_SNAPSHOT_KEY in metadata and sell_rule:
        # A checkpoint this code writes records its sell rule, and one without
        # it is read as pre-#661 and replayed: pass sell_rule=None for that.
        metadata = {"llm_sell_sizing": sell_rule, **metadata}
    db.insert_run(
        run_id=run_id,
        session_id="leaderboard-live",
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date=start_date,
        end_date=end_date,
        initial_equity=extra.pop("initial_equity", points[0]["equity"]),
        final_equity=points[-1]["equity"] if final is None else final,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=num_trades,
        llm_model=llm_model,
        metadata=metadata,
        **extra,
    )
    db.insert_equity_points(run_id, points)


def _pt(ts, equity):
    return {"timestamp": ts, "equity": equity, "cash": equity, "positions_value": 0}


class _ScriptedAgent:
    """Stands in for LLMAgentStrategy: one continuous run over the segment, one
    bar per session, the session-close hook fired at each close with the
    counters already cumulative, as the real decision loop does."""

    used_llm = True
    model_id = "deepseek/deepseek-v4-pro"

    def __init__(self, *, steps=7, misses=None, crash_on=(), no_bars=(), runs=None,
                 step_tokens=(10, 10)):
        self.steps = steps
        # (input, output) tokens billed per traded session.
        self.step_tokens = step_tokens
        self.misses = misses or {}
        self.crash_on = set(crash_on)
        self.no_bars = set(no_bars)
        self.runs = [] if runs is None else runs
        self.traded = []
        self.llm_calls = self.llm_decisions = self.decision_steps = 0
        self.input_tokens = self.output_tokens = 0
        self.planned_decision_steps = 0
        self.last_portfolio_snapshot = None
        self._trades = 0

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start, end, capital, starting_snapshot=None, on_session_close=None):
        self.runs.append((start, end, (starting_snapshot or {}).get("day")))
        days = [
            day for day in live._trading_days_inclusive(
                date.fromisoformat(start), date.fromisoformat(end)
            )
            if day.isoformat() not in self.no_bars
        ]
        self.planned_decision_steps = self.steps * len(days)
        curve = []
        for day in days:
            iso = day.isoformat()
            if iso in self.crash_on:
                raise RuntimeError("worker killed")
            self.traded.append(iso)
            self.llm_calls += self.steps
            self.decision_steps += self.steps
            self.llm_decisions += self.steps - self.misses.get(iso, 0)
            self.input_tokens += self.step_tokens[0]
            self.output_tokens += self.step_tokens[1]
            self._trades += 1
            curve.append(_pt(f"{iso}T19:00:00+00:00", SEED + day.day))
            self.last_portfolio_snapshot = {"cash": SEED, "positions": {}, "day": iso[5:]}
            if on_session_close is not None:
                on_session_close(day, list(curve))
        return curve

    def num_trades(self):
        return self._trades


def _one_bar_frame():
    return {"AAPL": type("Frame", (), {"attrs": {}})()}


def _deepseek(freeze_cfg):
    return next(e for e in live.live_llm_entries(freeze_cfg) if e["id"] == "deepseek_v4_pro")


def _month_rows(entry_id):
    return sorted(
        (r for r in db.get_runs_by_session("leaderboard-live") or []
         if r.get("llm_model") == entry_id),
        key=lambda r: r["end_date"],
    )


def test_september_window_is_a_new_calendar_month():
    start, end = live.live_month_dates(datetime(2026, 9, 1, 10, 0, tzinfo=_ET))
    assert start == "2026-09-01"
    assert end == "2026-09-30"


def test_august_window_is_the_calendar_month():
    start, end = live.live_month_dates(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert start == "2026-08-01"
    assert end == "2026-08-31"


def test_hourly_axis_skips_weekends_and_covers_rth():
    axis = live.live_month_hourly_axis("2026-08-01", "2026-08-31")
    assert len(axis) == 21 * 7
    assert all(not ts.startswith("2026-08-01T") for ts in axis)
    assert all(not ts.startswith("2026-08-02T") for ts in axis)
    assert axis[0].startswith("2026-08-03T10:00")
    assert axis[-1].startswith("2026-08-31T16:00")
    hours = {datetime.fromisoformat(ts).hour for ts in axis}
    assert hours == {10, 11, 12, 13, 14, 15, 16}


def test_clock_after_close_freezes_today():
    clock = live.live_clock(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert clock["session_state"] == "closed"
    assert clock["frozen_through"] == "2026-08-27"
    assert clock["live_day"] is None


def test_clock_during_rth_keeps_yesterday_frozen():
    clock = live.live_clock(datetime(2026, 8, 27, 14, 0, tzinfo=_ET))
    assert clock["session_state"] == "rth"
    assert clock["frozen_through"] == "2026-08-26"
    assert clock["live_day"] == "2026-08-27"


def test_clock_weekend_rolls_to_friday():
    clock = live.live_clock(datetime(2026, 8, 29, 12, 0, tzinfo=_ET))
    assert clock["session_state"] == "weekend"
    assert clock["frozen_through"] == "2026-08-28"


def test_clock_does_not_freeze_today_until_the_close_settles():
    """At 16:02 a Basic-plan SIP fetch is clamped to 15:47: the closing bar is
    partial, and a freeze taken then would cache the truncated curve."""
    clock = live.live_clock(datetime(2026, 8, 27, 16, 2, tzinfo=_ET))
    assert clock["session_state"] == "settling"
    assert clock["frozen_through"] == "2026-08-26"
    settled = live.freeze_settle_time(date(2026, 8, 27))
    assert settled > datetime(2026, 8, 27, 16, 15, tzinfo=_ET)
    after = live.live_clock(settled)
    assert after["session_state"] == "closed"
    assert after["frozen_through"] == "2026-08-27"


def test_settle_time_follows_the_sip_delay(monkeypatch):
    monkeypatch.setenv("ALPACA_SIP_DELAY_MINUTES", "40")
    assert live.freeze_settle_time(date(2026, 8, 27)) == datetime(
        2026, 8, 27, 16, 55, tzinfo=_ET
    )


def test_clock_preopen_has_no_session_in_progress():
    clock = live.live_clock(datetime(2026, 8, 27, 8, 0, tzinfo=_ET))
    assert clock["session_state"] == "preopen"
    assert clock["frozen_through"] == "2026-08-26"
    assert clock["live_day"] is None


def test_holiday_is_not_a_session():
    """Labor Day 2026: not a freeze day, not an axis day, not a progress day."""
    clock = live.live_clock(datetime(2026, 9, 7, 17, 0, tzinfo=_ET))
    assert clock["session_state"] == "holiday"
    assert clock["frozen_through"] == "2026-09-04"
    axis = live.live_month_hourly_axis("2026-09-01", "2026-09-30")
    assert not any(ts.startswith("2026-09-07") for ts in axis)
    assert len(axis) == 21 * 7
    assert live.next_session_day(date(2026, 9, 4)) == date(2026, 9, 8)
    assert live.live_increment_bounds(
        "2026-09-04", month_start="2026-09-01", freeze_end="2026-09-08"
    ) == ("2026-09-08", "2026-09-08")


def test_month_opening_on_a_holiday_has_no_freeze_yet():
    assert live.live_freeze_config(datetime(2027, 1, 1, 18, 0, tzinfo=_ET)) is None


def test_nyse_holiday_rules():
    from dashboard.backend.infrastructure.market_data.us_market_calendar import nyse_holidays

    assert sorted(d.isoformat() for d in nyse_holidays(2026)) == [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ]
    # A Saturday New Year's Day is not observed on the Friday before.
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)


# ── NYSE early closes (13:00 ET) ────────────────────────────────────────────


@pytest.fixture
def sip_delay_15(monkeypatch):
    monkeypatch.delenv("ALPACA_ALLOW_RECENT_SIP", raising=False)
    monkeypatch.setenv("ALPACA_SIP_DELAY_MINUTES", "15")


def _node_clocks(axis, day):
    return [ts[11:16] for ts in axis if ts.startswith(day)]


def test_hourly_axis_ends_at_13_00_on_an_early_close():
    """2026-11-27 closes at 13:00: four nodes (10:00 carries the prior close,
    11:00-13:00 hold the 10:00-12:00 bars), not seven with three phantoms."""
    from dashboard.backend.infrastructure.market_data.us_market_calendar import (
        is_early_close,
    )

    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    days = live._trading_days_inclusive(date(2026, 11, 1), date(2026, 11, 30))
    assert len(days) == 20
    assert [day for day in days if is_early_close(day)] == [date(2026, 11, 27)]
    assert len(axis) == 19 * 7 + 4
    assert _node_clocks(axis, "2026-11-27") == ["10:00", "11:00", "12:00", "13:00"]
    assert _node_clocks(axis, "2026-11-25") == [
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00",
    ]
    assert _node_clocks(axis, "2026-11-26") == []


def test_december_axis_has_one_early_close():
    axis = live.live_month_hourly_axis("2026-12-01", "2026-12-31")
    days = live._trading_days_inclusive(date(2026, 12, 1), date(2026, 12, 31))
    assert len(days) == 22
    assert len(axis) == 21 * 7 + 4
    assert _node_clocks(axis, "2026-12-24") == ["10:00", "11:00", "12:00", "13:00"]
    assert axis[-1].startswith("2026-12-31T16:00")


def test_an_early_close_settles_30_minutes_after_13_00(sip_delay_15):
    assert live.freeze_settle_time(date(2026, 11, 27)) == datetime(
        2026, 11, 27, 13, 30, tzinfo=_ET
    )
    assert live.freeze_settle_time(date(2026, 11, 25)) == datetime(
        2026, 11, 25, 16, 30, tzinfo=_ET
    )


@pytest.mark.parametrize(
    ("clock", "state", "frozen_through", "live_day"),
    [
        ((12, 59), "rth", "2026-11-25", "2026-11-27"),
        ((13, 5), "settling", "2026-11-25", "2026-11-27"),
        ((13, 30), "closed", "2026-11-27", None),
        ((15, 0), "closed", "2026-11-27", None),
    ],
)
def test_the_clock_closes_at_13_00_on_an_early_close(
    sip_delay_15, clock, state, frozen_through, live_day
):
    """Thanksgiving (11-26) is skipped: before the half day settles, the
    freeze is still Wednesday's."""
    status = live.live_clock(datetime(2026, 11, 27, *clock, tzinfo=_ET))
    assert (
        status["session_state"], status["frozen_through"], status["live_day"]
    ) == (state, frozen_through, live_day)


def test_a_bare_date_means_that_days_cash_close():
    assert live._coerce_as_of_eastern(date(2026, 11, 27)) == datetime(
        2026, 11, 27, 13, 0, tzinfo=_ET
    )
    assert live._coerce_as_of_eastern(date(2026, 11, 25)) == datetime(
        2026, 11, 25, 16, 0, tzinfo=_ET
    )


def test_axis_node_key_caps_at_the_early_close():
    def key(hour, minute=0, day=27):
        return live._axis_node_key(datetime(2026, 11, day, hour, minute, tzinfo=_ET))

    assert key(12) == "2026-11-27T13:00"  # Alpaca's last cash-session bar
    # Yahoo's closing level, restamped to 12:30 by _yahoo from its 13:00 stub
    # (test_yahoo_chart pins that against recorded responses).
    assert key(12, 30) == "2026-11-27T13:00"
    assert key(13) is None  # the late session has no node
    assert key(15) is None
    assert key(15, day=25) == "2026-11-25T16:00"


def test_frozen_curve_through_an_early_close_stops_at_13_00():
    """Open-stamped bars 10:00-15:00 ET (15:00Z-20:00Z in EST): the three
    late-session prints must not paint 14:00-16:00 nodes."""
    axis = live.live_month_hourly_axis("2026-11-01", "2026-11-30")
    points = [
        _pt(f"2026-11-27T{15 + i}:00:00+00:00", SEED + 100 * i) for i in range(6)
    ]
    curve = live.reindex_frozen_curve(points, axis, "2026-11-27", SEED)
    assert len(curve) == 18 * 7 + 4
    assert [
        (p["timestamp"][11:16], p["equity"])
        for p in curve if p["timestamp"].startswith("2026-11-27")
    ] == [
        ("10:00", SEED), ("11:00", SEED), ("12:00", SEED + 100), ("13:00", SEED + 200),
    ]


def test_the_window_description_names_the_early_close(no_alpaca):
    payload = live.get_live_leaderboard(as_of=datetime(2026, 11, 27, 17, 39, tzinfo=_ET))
    assert "10:00–13:00 on NYSE early-close days" in payload["window"]["description"]


def test_the_live_board_keeps_no_fixed_session_close():
    """A module-level 16:00 close and a fixed seven-hour axis are how the early
    close slipped through: every reader asks for the day's close instead."""
    assert not hasattr(live, "_US_CASH_CLOSE")
    assert not hasattr(live, "_RTH_HOURS")


def _alpaca_hourly_bars(*days):
    """Open-stamped hourly AAPL bars, every hour 04:00-19:00 ET: a deliberate
    superset of what Alpaca sends. It emits no bar for an hour without a valid
    trade, and its half-day extended session ends at 17:00. Dense, so the late
    session after an early close is certainly in the response."""
    import pandas as pd

    from dashboard.backend.infrastructure.market_data.sessions import (
        FRAME_ATTR_OPEN_STAMPED_MINUTES,
    )

    index = pd.DatetimeIndex([
        ts
        for day in days
        for ts in pd.date_range(f"{day} 04:00", f"{day} 19:00", freq="h",
                                tz="US/Eastern")
    ]).tz_convert("UTC")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        index=index,
    )
    frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = 60
    return {"AAPL": frame}


def test_a_replay_across_thanksgiving_and_the_early_close_is_not_a_gap(monkeypatch):
    """The real decision loop over 11-25 (full), 11-26 (holiday) and 11-27
    (13:00 close): six decisions then three, a checkpoint per session, no
    LiveSessionGapError, and four nodes on the half day."""
    from dashboard.backend.domain.leaderboard.strategies import llm_agent

    decided = []

    class Manager:
        def __init__(self, **kwargs):
            self.equity_history = []
            self.trades = []
            self.llm_calls = self.llm_decisions = 0
            self.input_tokens = self.output_tokens = 0

        def restore_state(self, snapshot):
            pass

        def snapshot_state(self):
            return {"cash": SEED, "positions": {}}

        def get_portfolio_state(self, *args):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            decided.append(state["timestamp"])
            self.llm_calls += 1
            self.llm_decisions += 1
            return {"actions": []}

        def execute_actions(self, *args):
            pass

        def update_equity(self, market_data, price_cache, ts):
            equity = SEED + 1000 * (len(self.equity_history) + 1)
            self.equity_history.append(
                {"timestamp": ts, "equity": equity, "cash": equity, "positions_value": 0}
            )

        def get_equity_curve(self):
            return self.equity_history

    monkeypatch.setattr(llm_agent, "PortfolioManager", Manager)
    monkeypatch.setattr(llm_agent.LLMAgentStrategy, "_make_client", lambda self: object())
    monkeypatch.setattr(
        lb_service,
        "fetch_hourly_bars",
        lambda symbols, start, end: _alpaca_hourly_bars("2026-11-25", "2026-11-27"),
    )
    _insert_live_run(
        "lb_deepseek_v4_pro_20261101_20261124", "deepseek_v4_pro", "2026-11-24",
        [_pt("2026-11-24T20:00:00+00:00", SEED)],
        start_date="2026-11-01",
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
    )
    as_of = datetime(2026, 11, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    assert freeze_cfg["end_date"] == "2026-11-27"

    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    assert row["segment"] == {"start_date": "2026-11-25", "end_date": "2026-11-27"}
    by_day = {}
    for ts in decided:
        et = ts.tz_convert("US/Eastern")
        by_day.setdefault(et.date().isoformat(), []).append(et.strftime("%H:%M"))
    assert by_day == {
        "2026-11-25": ["10:00", "11:00", "12:00", "13:00", "14:00", "15:00"],
        "2026-11-27": ["10:00", "11:00", "12:00"],
    }
    stored = db.get_run(row["run_id"])
    assert stored["end_date"] == "2026-11-27"
    assert stored["llm_calls"] == 9
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-11-27"]

    payload = live.get_live_leaderboard(as_of=as_of)
    assert _node_clocks(payload["chart_axis"], "2026-11-27") == [
        "10:00", "11:00", "12:00", "13:00",
    ]
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert [
        (p["timestamp"][11:16], p["equity"])
        for p in deepseek["equity_curve"] if p["timestamp"].startswith("2026-11-27")
    ] == [
        ("10:00", SEED + 6000),
        ("11:00", SEED + 7000),
        ("12:00", SEED + 8000),
        ("13:00", SEED + 9000),
    ]
    assert deepseek["portfolio_value"] == SEED + 9000


def test_payload_without_runs_keeps_empty_curves(no_alpaca):
    payload = live.get_live_leaderboard(
        as_of=datetime(2026, 8, 27, 14, 0, tzinfo=_ET)
    )
    assert payload["period"] == "live"
    assert payload["window"]["start_date"] == "2026-08-01"
    assert payload["window"]["end_date"] == "2026-08-31"
    assert payload["leader"] == "—"
    status = payload["live_status"]
    assert status["phase"] == 1
    assert status["has_prints"] is False
    assert status["session_state"] == "rth"
    assert status["freeze_end"] == "2026-08-26"
    assert len(payload["chart_axis"]) == 21 * 7
    assert payload["chart_axis"][-1].startswith("2026-08-31T16:00")
    for entry in payload["entries"]:
        assert entry["equity_curve"] == []
        assert entry["rank"] is None
        assert entry["portfolio_value"] is None
        assert entry["cumulative_return"] is None


def test_frozen_curve_maps_utc_hours_onto_et_axis_and_stops(no_alpaca):
    """Alpaca stores open-stamped UTC hours; the live axis is ET bar closes.

    The 09:00 ET bar (13:00Z) closes at 10:00 and the 15:00 ET bar (19:00Z)
    holds the 16:00 close. Do not paint past the freeze."""
    db.insert_run(
        run_id="lb_spy_index_20260801_20260826",
        session_id="leaderboard-live",
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-26",
        initial_equity=SEED,
        final_equity=SEED * 1.01,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.01,
        num_trades=0,
        llm_model="spy_index",
    )
    db.insert_equity_points(
        "lb_spy_index_20260801_20260826",
        [
            {
                "timestamp": "2026-08-03T13:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T19:00:00+00:00",
                "equity": SEED * 1.01,
                "cash": 0,
                "positions_value": SEED * 1.01,
            },
        ],
    )
    payload = live.get_live_leaderboard(
        as_of=datetime(2026, 8, 27, 14, 0, tzinfo=_ET)
    )
    spy = next(e for e in payload["entries"] if e["entry_id"] == "spy_index")
    assert spy["equity_curve"], "freeze window must produce a printed series"
    assert spy["equity_curve"][0]["timestamp"].startswith("2026-08-03T10:00")
    assert spy["equity_curve"][0]["equity"] == SEED
    assert spy["equity_curve"][-1]["timestamp"].startswith("2026-08-26T16:00")
    assert spy["equity_curve"][-1]["equity"] == SEED * 1.01
    assert all(not p["timestamp"].startswith("2026-08-27") for p in spy["equity_curve"])
    assert all(not p["timestamp"].startswith("2026-08-31") for p in spy["equity_curve"])
    assert payload["live_status"]["has_prints"] is True
    assert payload["leader"] != "—"


def test_live_api_get_is_read_only(client, no_alpaca, monkeypatch):
    """The window moves daily, so a computing GET would miss cache on the first
    request of every day and fetch bars inside a public request thread."""
    monkeypatch.setattr(
        live,
        "_coerce_as_of_eastern",
        lambda as_of=None: datetime(2026, 8, 27, 17, 39, tzinfo=_ET),
    )

    resp = client.get("/api/v1/leaderboard?period=live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "live"
    assert body["window"]["label"] == "2026-08-01 — 2026-08-31"
    status = body["live_status"]
    assert status["freeze_start"] == "2026-08-01"
    assert status["freeze_end"] == "2026-08-27"
    assert status["snapshot_stale"] is True


def test_live_freeze_config_uses_live_session_not_contest():
    cfg = live.live_freeze_config(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert cfg["session_id"] == "leaderboard-live"
    assert cfg["start_date"] == "2026-08-01"
    assert cfg["end_date"] == "2026-08-27"


def test_stale_freeze_snapshot_still_prints_and_does_not_invent_next_day(no_alpaca):
    """A 1–26 snapshot must still show after freeze rolls to 27, without filling 27."""
    db.insert_run(
        run_id="lb_spy_index_20260801_20260826",
        session_id="leaderboard-live",
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-26",
        initial_equity=SEED,
        final_equity=SEED * 1.01,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.01,
        num_trades=0,
        llm_model="spy_index",
    )
    db.insert_equity_points(
        "lb_spy_index_20260801_20260826",
        [
            {
                "timestamp": "2026-08-03T13:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T19:00:00+00:00",
                "equity": SEED * 1.01,
                "cash": 0,
                "positions_value": SEED * 1.01,
            },
        ],
    )
    payload = live.get_live_leaderboard(
        as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    spy = next(e for e in payload["entries"] if e["entry_id"] == "spy_index")
    assert spy["equity_curve"]
    assert spy["equity_curve"][-1]["timestamp"].startswith("2026-08-26T16:00")
    assert all(not p["timestamp"].startswith("2026-08-27") for p in spy["equity_curve"])
    assert payload["live_status"]["freeze_end"] == "2026-08-27"
    assert payload["live_status"]["snapshot_end"] == "2026-08-26"
    assert payload["live_status"]["snapshot_stale"] is True


def test_live_month_run_shows_llm_snapshot(no_alpaca):
    db.insert_run(
        run_id="lb_qwen3_7_plus_20260801_20260827",
        session_id="leaderboard-live",
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-27",
        initial_equity=SEED,
        final_equity=SEED * 1.005,
        total_return=0.005,
        sharpe_ratio=0.2,
        max_drawdown=-0.01,
        num_trades=2,
        llm_model="qwen3_7_plus",
        llm_calls=10,
    )
    db.insert_equity_points(
        "lb_qwen3_7_plus_20260801_20260827",
        [
            {
                "timestamp": "2026-08-03T13:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-27T19:00:00+00:00",
                "equity": SEED * 1.005,
                "cash": 0,
                "positions_value": SEED * 1.005,
            },
        ],
    )
    payload = live.get_live_leaderboard(
        as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    model = next(e for e in payload["entries"] if e["entry_id"] == "qwen3_7_plus")
    assert model["is_model"] is True
    assert model["equity_curve"]
    assert model["equity_curve"][-1]["timestamp"].startswith("2026-08-27T16:00")
    assert model["equity_curve"][-1]["equity"] == pytest.approx(SEED * 1.005)
    assert payload["live_status"]["models_cached"] >= 1
    assert payload["live_status"]["models_pending"] == 2
    pending = [e for e in payload["entries"] if e["status"] == "pending"]
    assert pending and all(e["rank"] is None for e in pending)
    assert payload["live_status"]["roster"] == [
        "deepseek_v4_pro",
        "qwen3_7_plus",
        "claude_haiku_4_5",
    ]
    model_ids = {e["entry_id"] for e in payload["entries"] if e.get("is_model")}
    assert model_ids == {"deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"}


def test_get_live_never_deploys_models(no_alpaca, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("GET must not deploy models")

    monkeypatch.setattr(lb_service, "deploy_model_run", boom)
    monkeypatch.setattr(live, "deploy_live_model_increment", boom)
    live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert calls == []


def test_refresh_live_leaderboard_deploys_models(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_refresh.json")
    deployed = []

    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda **kwargs: {"created": 0, "cache_hit": True},
    )

    windows = []

    # Records rather than asserts: the roster loop catches any Exception per
    # entry, so an assert in here would be filed as a model failure.
    def fake_deploy(entry, freeze_cfg, **kwargs):
        deployed.append(entry["id"])
        windows.append(
            (freeze_cfg.get("session_id"), freeze_cfg.get("start_date"), freeze_cfg.get("end_date"))
        )
        return {"entry_id": entry["id"], "run_id": f"lb_{entry['id']}", "cached": False}

    monkeypatch.setattr(live, "deploy_live_model_increment", fake_deploy)
    result = live.refresh_live_leaderboard(
        deploy_models=True,
        as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET),
    )
    assert result["model_failures"] == []
    assert set(windows) == {("leaderboard-live", "2026-08-01", "2026-08-27")}
    assert result["skipped"] is False
    assert result["models_deployed"] is True
    assert "gpt_5_5" not in deployed
    assert "nemotron_3_nano_30b" not in deployed
    # Roster order, not file order: the file lists Haiku first.
    assert deployed == ["deepseek_v4_pro", "qwen3_7_plus", "claude_haiku_4_5"]


def test_model_deploys_default_off_everywhere(client, monkeypatch):
    """Model deploys bill the operator; nothing may default them on."""
    import inspect

    for fn in (
        live.refresh_live_leaderboard,
        live.maybe_schedule_live_leaderboard_refresh,
        live.enqueue_live_leaderboard_refresh,
    ):
        assert inspect.signature(fn).parameters["deploy_models"].default is False

    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    seen = {}

    def fake_enqueue(**kwargs):
        seen.update(kwargs)
        return {"accepted": True, "period": "live"}

    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.enqueue_live_leaderboard_refresh",
        fake_enqueue,
    )
    resp = client.post(
        "/api/v1/leaderboard/live/refresh",
        headers={"X-Leaderboard-Refresh-Secret": "cron-secret"},
    )
    assert resp.status_code == 202
    assert seen["deploy_models"] is False


def test_workflow_schedule_only_bills_behind_an_explicit_variable():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "live-leaderboard.yml"
    ).read_text(encoding="utf-8")
    assert "vars.LIVE_LEADERBOARD_DEPLOY_MODELS == 'true'" in workflow
    assert "github.event_name != 'workflow_dispatch' && 'true'" not in workflow
    dispatch = workflow.split("deploy_models:", 1)[1].split("force:", 1)[0]
    assert "default: false" in dispatch


def test_clear_forgets_the_refresh_state(monkeypatch, tmp_path):
    """--clear without --force used to delete every row and then skip the
    refresh, because the state file still claimed this window was done."""
    state_path = tmp_path / "live_refresh.json"
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", state_path)
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    calls = []
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda **kwargs: calls.append(kwargs) or {"created": 0},
    )
    live.refresh_live_leaderboard(as_of=as_of)
    assert live.refresh_live_leaderboard(as_of=as_of)["skipped"] is True
    live.clear_live_session_runs()
    assert not state_path.exists()
    assert live.refresh_live_leaderboard(as_of=as_of)["skipped"] is False
    assert len(calls) == 2


def test_refresh_prunes_superseded_freeze_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "s.json")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    for end in ("2026-08-25", "2026-08-26", "2026-08-27"):
        _insert_live_run(
            f"lb_spy_index_prune_{end}", "spy_index", end,
            [_pt(f"{end}T13:00:00+00:00", SEED)],
        )
    _insert_live_run(
        "lb_spy_index_prune_july", "spy_index", "2026-07-31",
        [_pt("2026-07-31T13:00:00+00:00", SEED)], start_date="2026-07-01",
    )
    result = live.refresh_live_leaderboard(
        as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["pruned_runs"] == 2
    left = {r["run_id"] for r in db.get_runs_by_session("leaderboard-live")}
    assert "lb_spy_index_prune_2026-08-27" in left
    assert "lb_spy_index_prune_2026-08-25" not in left
    assert "lb_spy_index_prune_2026-08-26" not in left
    assert "lb_spy_index_prune_july" in left


def test_contest_period_still_uses_get_leaderboard(client, monkeypatch):
    import dashboard.backend.api.routers.leaderboard as router_mod

    monkeypatch.setattr(
        router_mod,
        "get_leaderboard",
        lambda **kwargs: {"period": "contest", "entries": [], "sentinel": True},
    )
    resp = client.get("/api/v1/leaderboard?period=contest")
    assert resp.status_code == 200
    assert resp.json()["sentinel"] is True


def test_live_increment_bounds_appends_next_session_only():
    assert live.live_increment_bounds(
        "2026-08-27", month_start="2026-08-01", freeze_end="2026-08-27"
    ) is None
    assert live.live_increment_bounds(
        "2026-08-26", month_start="2026-08-01", freeze_end="2026-08-27"
    ) == ("2026-08-27", "2026-08-27")
    assert live.live_increment_bounds(
        "2026-08-28", month_start="2026-08-01", freeze_end="2026-08-31"
    ) == ("2026-08-31", "2026-08-31")
    assert live.live_increment_bounds(
        None, month_start="2026-08-01", freeze_end="2026-08-03"
    ) == ("2026-08-01", "2026-08-03")


def test_portfolio_snapshot_roundtrip_keeps_cash_and_lots():
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

    manager = PortfolioManager(initial_capital=SEED, t_plus_one_enabled=True)
    manager.cash = 1234.5
    manager.positions = {"AAPL": 10}
    manager.entry_prices = {"AAPL": 100.0}
    manager.available_positions = {"AAPL": 0}
    manager.frozen_lots = {"AAPL": [{"quantity": 10, "buy_date": date(2026, 8, 26)}]}
    snap = manager.snapshot_state()
    restored = PortfolioManager(initial_capital=SEED, t_plus_one_enabled=True)
    restored.restore_state(snap)
    assert restored.cash == 1234.5
    assert restored.positions == {"AAPL": 10.0}
    assert restored.frozen_lots["AAPL"][0]["quantity"] == 10.0
    assert restored.frozen_lots["AAPL"][0]["buy_date"] == date(2026, 8, 26)


def test_deploy_live_increment_trades_only_the_new_session(monkeypatch):
    freeze_cfg = live.live_freeze_config(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert freeze_cfg is not None
    snapshot = {"cash": SEED * 0.2, "positions": {"AAPL": 3}, "entry_prices": {"AAPL": 50}}
    db.insert_run(
        run_id="lb_qwen3_7_plus_20260801_20260826",
        session_id="leaderboard-live",
        agent_name="Qwen3.7 Plus",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-26",
        initial_equity=SEED,
        final_equity=SEED * 1.01,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.01,
        num_trades=2,
        llm_model="qwen3_7_plus",
        llm_calls=70,
        llm_decisions=70,
        input_tokens=1000,
        output_tokens=200,
        metadata={"llm_sell_sizing": LLM_SELL_SIZING, live.LIVE_SNAPSHOT_KEY: snapshot},
    )
    db.insert_equity_points(
        "lb_qwen3_7_plus_20260801_20260826",
        [
            {
                "timestamp": "2026-08-03T13:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T19:00:00+00:00",
                "equity": SEED * 1.01,
                "cash": snapshot["cash"],
                "positions_value": SEED * 1.01 - snapshot["cash"],
            },
        ],
    )

    class FakeAgent:
        def __init__(self):
            self.windows = []
            self.input_tokens = 11
            self.output_tokens = 3
            self.llm_calls = 7
            self.llm_decisions = 7
            self.decision_steps = 7
            self.used_llm = True
            self.model_id = "qwen/qwen3.7-plus"
            self.last_portfolio_snapshot = {
                "cash": 500.0,
                "positions": {"AAPL": 4},
            }

        def required_symbols(self):
            return ["AAPL"]

        def run(self, bars, start, end, capital, starting_snapshot=None,
                on_session_close=None):
            self.windows.append((start, end, starting_snapshot))
            curve = [
                {
                    "timestamp": "2026-08-27T14:00:00+00:00",
                    "equity": SEED * 1.02,
                    "cash": 500.0,
                    "positions_value": SEED * 1.02 - 500.0,
                }
            ]
            on_session_close(date(2026, 8, 27), curve)
            return curve

        def num_trades(self):
            return 1

    fake = FakeAgent()
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: fake)
    monkeypatch.setattr(
        lb_service,
        "fetch_hourly_bars",
        lambda symbols, start, end: {"AAPL": type("Frame", (), {"attrs": {}})()},
    )

    entry = next(e for e in live.live_llm_entries(freeze_cfg) if e["id"] == "qwen3_7_plus")
    row = live.deploy_live_model_increment(entry, freeze_cfg)
    assert fake.windows == [("2026-08-27", "2026-08-27", snapshot)]
    assert row["increment"] is True
    assert row["segment"] == {"start_date": "2026-08-27", "end_date": "2026-08-27"}
    assert row["window"] == {"start_date": "2026-08-01", "end_date": "2026-08-27"}
    curve = db.get_equity_curve(row["run_id"])
    assert len(curve) == 3
    assert curve[-1]["equity"] == pytest.approx(SEED * 1.02)
    stored = db.get_run(row["run_id"])
    assert stored["end_date"] == "2026-08-27"
    assert stored["llm_calls"] == 77
    meta = stored["metadata"]
    assert meta[live.LIVE_SNAPSHOT_KEY]["positions"]["AAPL"] == 4


def test_live_refresh_endpoint_requires_secret(client, monkeypatch):
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    resp = client.post("/api/v1/leaderboard/live/refresh")
    assert resp.status_code == 401

    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.enqueue_live_leaderboard_refresh",
        lambda **_: {
            "accepted": True,
            "started": True,
            "refresh_in_progress": True,
            "period": "live",
            "window": {"start_date": "2026-08-01", "end_date": "2026-08-27", "label": "2026-08-01 → 2026-08-27"},
            "message": "Live leaderboard refresh started in the background.",
        },
    )
    ok = client.post(
        "/api/v1/leaderboard/live/refresh?deploy_models=true",
        headers={"X-Leaderboard-Refresh-Secret": "cron-secret"},
    )
    assert ok.status_code == 202
    body = ok.json()
    assert body["accepted"] is True
    assert body["period"] == "live"


def test_open_stamped_bars_land_on_the_node_their_bar_closes_in():
    """Alpaca 1h bars open on the hour (09:00 … 15:00), Yahoo's on the half
    hour (09:30 … 15:30). Each belongs to the node its bar closes in, so the
    16:00 node carries the real close instead of a flat copy of 15:00."""
    axis = live.live_month_hourly_axis("2026-08-03", "2026-08-03")
    alpaca = [
        _pt(f"2026-08-03T{13 + i:02d}:00:00+00:00", SEED + i) for i in range(7)
    ]
    curve = live.reindex_frozen_curve(alpaca, axis, "2026-08-03", SEED)
    assert [p["timestamp"][11:16] for p in curve] == [
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00",
    ]
    assert [p["equity"] for p in curve] == [SEED + i for i in range(7)]

    yahoo = [
        _pt(f"2026-08-03T{13 + i:02d}:30:00+00:00", SEED + 10 * i) for i in range(7)
    ]
    curve = live.reindex_frozen_curve(yahoo, axis, "2026-08-03", SEED)
    assert [p["equity"] for p in curve] == [SEED + 10 * i for i in range(7)]


def test_null_equity_is_not_read_as_zero():
    axis = live.live_month_hourly_axis("2026-08-03", "2026-08-03")
    points = [
        _pt("2026-08-03T13:00:00+00:00", SEED),
        {"timestamp": "2026-08-03T14:00:00+00:00", "equity": None},
    ]
    curve = live.reindex_frozen_curve(points, axis, "2026-08-03", SEED)
    assert all(p["equity"] == SEED for p in curve)


def test_a_model_with_no_run_is_never_ranked(no_alpaca):
    """In a down month every printed entry is below the seed; a pending model
    published at the seed and 0% used to rank #1 without ever trading."""
    for entry_id in ("spy_index", "qwen3_7_plus"):
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260826", entry_id, "2026-08-26",
            [_pt("2026-08-03T13:00:00+00:00", SEED),
             _pt("2026-08-26T19:00:00+00:00", SEED * 0.9)],
            total_return=-0.1,
        )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    ranked = [e for e in payload["entries"] if e["rank"] is not None]
    assert {e["entry_id"] for e in ranked} == {"spy_index", "qwen3_7_plus"}
    assert [e["rank"] for e in ranked] == [1, 2]
    assert payload["entries"][: len(ranked)] == ranked
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["status"] == "pending"
    assert deepseek["portfolio_value"] is None
    qwen = next(e for e in ranked if e["entry_id"] == "qwen3_7_plus")
    assert payload["leader"] == qwen["model"]


def test_returns_come_off_the_stored_run_and_keep_the_first_hour(no_alpaca):
    """The first mark already carries the first hour's P&L. Rescaling by it
    (``initial_equity``) inflated every point and erased that hour."""
    first_mark = SEED * 0.995
    _insert_live_run(
        "lb_qwen3_7_plus_20260801_20260826", "qwen3_7_plus", "2026-08-26",
        [_pt("2026-08-03T13:00:00+00:00", first_mark),
         _pt("2026-08-26T19:00:00+00:00", SEED * 1.02)],
        total_return=0.02, sharpe=1.1, max_dd=-0.03,
        metadata={"initial_capital": SEED},
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    qwen = next(e for e in payload["entries"] if e["entry_id"] == "qwen3_7_plus")
    assert qwen["cumulative_return"] == pytest.approx(0.02)
    # Sharpe is not the stored 1.1: it is the daily figure off the plotted
    # curve. Aug 3-26 is 18 sessions; the as-of fill holds 0.995 x seed from
    # Aug 3 to Aug 25, so r = [-0.005, 0 x 16, 1.02/0.995 - 1]. Measured from the
    # first mark instead of the seed, r_1 would be 0 and this would read 3.74.
    assert qwen["sharpe_ratio"] == pytest.approx(2.906881, abs=1e-6)
    assert qwen["sharpe_ci_low"] == pytest.approx(-4.487853, abs=1e-6)
    assert qwen["sharpe_ci_high"] == pytest.approx(10.301616, abs=1e-6)
    assert qwen["sharpe_days"] == 18
    assert qwen["sharpe_status"] == "ok"
    assert qwen["max_drawdown"] == pytest.approx(-0.03)
    assert qwen["portfolio_value"] == pytest.approx(SEED * 1.02)
    assert qwen["equity_curve"][0]["equity"] == pytest.approx(first_mark)


def test_live_entry_publishes_trade_stats_and_invested(no_alpaca):
    """The Performance table reads these off the GET entry, not the rank."""
    from dashboard.backend.domain.leaderboard.live_trade_stats import TRADE_STATS_KEY

    _insert_live_run(
        "lb_qwen3_7_plus_20260801_20260826", "qwen3_7_plus", "2026-08-26",
        [_pt("2026-08-03T13:00:00+00:00", SEED),
         _pt("2026-08-26T19:00:00+00:00", SEED * 1.02)],
        total_return=0.02, sharpe=1.1, max_dd=-0.03,
        num_trades=3,
        metadata={
            "initial_capital": SEED,
            live.LIVE_SNAPSHOT_KEY: {"cash": SEED * 1.02 * 0.4, "positions": {"AAPL": 10}},
            TRADE_STATS_KEY: {
                "version": 1,
                "complete": True,
                "buys": 2,
                "sells": 1,
                "closed_sells": 1,
                "winning_sells": 1,
                "held_share_hours": 12.0,
                "held_shares": 4.0,
                "open_lots": {},
            },
        },
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    assert payload["standings_label"] == "Performance"
    qwen = next(e for e in payload["entries"] if e["entry_id"] == "qwen3_7_plus")
    assert qwen["num_trades"] == 3
    assert qwen["win_rate"] == pytest.approx(1.0)
    assert qwen["avg_hold_hours"] == pytest.approx(3.0)
    assert qwen["closed_trades"] == 1
    assert qwen["invested_pct"] == pytest.approx(0.6)
    spy = next(e for e in payload["entries"] if e["entry_id"] == "spy_index")
    assert spy["status"] == "pending"
    assert spy["num_trades"] is None
    assert spy["win_rate"] is None
    assert spy["invested_pct"] is None


def test_dollar_axis_scales_by_the_recorded_seed_only(no_alpaca):
    seed = SEED / 10
    _insert_live_run(
        "lb_qwen3_7_plus_20260801_20260826", "qwen3_7_plus", "2026-08-26",
        [_pt("2026-08-03T13:00:00+00:00", seed * 0.99),
         _pt("2026-08-26T19:00:00+00:00", seed * 1.05)],
        total_return=0.05, metadata={"initial_capital": seed},
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    qwen = next(e for e in payload["entries"] if e["entry_id"] == "qwen3_7_plus")
    assert qwen["portfolio_value"] == pytest.approx(SEED * 1.05)
    assert qwen["cumulative_return"] == pytest.approx(0.05)
    assert qwen["equity_curve"][0]["equity"] == pytest.approx(SEED * 0.99)


def test_increment_lookback_is_relative_to_the_segment_and_shared(monkeypatch):
    freeze_cfg = live.live_freeze_config(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    fetches = []

    class Agent:
        input_tokens = output_tokens = llm_calls = llm_decisions = decision_steps = 1
        used_llm = True
        model_id = "m"
        last_portfolio_snapshot = {"cash": 1.0, "positions": {}}

        def required_symbols(self):
            return ["AAPL", "MSFT"]

        def run(self, bars, start, end, capital, starting_snapshot=None,
                on_session_close=None):
            curve = [_pt(f"{end}T19:00:00+00:00", SEED)]
            on_session_close(date.fromisoformat(end), curve)
            return curve

        def num_trades(self):
            return 0

    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: Agent())
    monkeypatch.setattr(lb_service, "_reject_if_llm_fallback", lambda *a, **k: None)

    def fake_fetch(symbols, start, end):
        fetches.append((tuple(symbols), start, end))
        return {"AAPL": type("Frame", (), {"attrs": {}})()}

    monkeypatch.setattr(lb_service, "fetch_hourly_bars", fake_fetch)
    for entry_id in ("qwen3_7_plus", "deepseek_v4_pro"):
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260826", entry_id, "2026-08-26",
            [_pt("2026-08-26T19:00:00+00:00", SEED)],
            metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
        )
    memo = {}
    for entry in live.live_llm_entries(freeze_cfg):
        if entry["id"] in ("qwen3_7_plus", "deepseek_v4_pro"):
            live.deploy_live_model_increment(entry, freeze_cfg, bars_memo=memo)
    assert fetches == [(("AAPL", "MSFT"), "2026-07-27", "2026-08-27")]


def _live_aug27():
    return live.live_freeze_config(datetime(2026, 8, 27, 17, 39, tzinfo=_ET))


def _insert_checkpoint(end_date, day, equity=SEED):
    _insert_live_run(
        f"lb_deepseek_v4_pro_20260801_{end_date.replace('-', '')}",
        "deepseek_v4_pro",
        end_date,
        [_pt(f"{end_date}T19:00:00+00:00", equity)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": day}},
    )


def _use_agent(monkeypatch, agent):
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: agent)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    return agent


def test_catch_up_is_one_run_checkpointed_per_session_and_resumes_after_a_crash(
    monkeypatch,
):
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-24", "08-24")
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(crash_on={"2026-08-26"}, runs=runs))

    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # One continuous run for the segment: the book, prices and trade memory
    # are never rebuilt at a day boundary; only the writes are per session.
    assert runs == [("2026-08-25", "2026-08-27", "08-24")]
    rows = _month_rows("deepseek_v4_pro")
    assert [r["end_date"] for r in rows] == ["2026-08-25"]
    assert live._snapshot_from_run(rows[0])["day"] == "08-25"

    runs.clear()
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-26", "2026-08-27", "08-25")]
    assert row["segment"] == {"start_date": "2026-08-26", "end_date": "2026-08-27"}
    assert row["window"] == {"start_date": "2026-08-01", "end_date": "2026-08-27"}
    stored = db.get_run(row["run_id"])
    assert stored["end_date"] == "2026-08-27"
    assert stored["llm_calls"] == 21
    assert stored["num_trades"] == 3
    assert stored["metadata"]["live_increment"] == {
        "segment_start": "2026-08-26",
        "segment_end": "2026-08-27",
        "full_replay": False,
        "forced": False,
        "config_changed": None,
        "resumed_from_run_id": "lb_deepseek_v4_pro_20260801_20260825",
        "resumed_from_end_date": "2026-08-25",
    }
    assert live.LIVE_CHECKPOINT_PENDING_KEY not in stored["metadata"]
    curve = db.get_equity_curve(row["run_id"])
    assert [p["equity"] for p in curve] == [SEED, SEED + 25, SEED + 26, SEED + 27]
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row["run_id"]]


def test_a_checkpoint_cut_between_its_row_and_curve_is_never_resumed_or_shown(
    monkeypatch,
):
    """The row and its curve are separate writes. A writer killed between them
    used to leave a row with a snapshot and no curve, which the next refresh
    resumed from, stitching the rest of the month onto nothing."""
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-24", "08-24")
    real_points = db.insert_equity_points

    def killed_on_the_26th(run_id, points, *args, **kwargs):
        if run_id.endswith("_20260826"):
            raise RuntimeError("killed between the row and its curve")
        return real_points(run_id, points, *args, **kwargs)

    monkeypatch.setattr(db, "insert_equity_points", killed_on_the_26th)
    _use_agent(monkeypatch, _ScriptedAgent())
    with pytest.raises(RuntimeError, match="killed between"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    shown = live.latest_live_month_runs("2026-08-01", "2026-08-27")["deepseek_v4_pro"]
    assert shown["end_date"] == "2026-08-25"
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["snapshot_end"] == "2026-08-25"
    assert deepseek["equity_curve"]

    monkeypatch.setattr(db, "insert_equity_points", real_points)
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-26", "2026-08-27", "08-25")]
    curve = db.get_equity_curve(row["run_id"])
    assert [p["equity"] for p in curve] == [SEED, SEED + 25, SEED + 26, SEED + 27]
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row["run_id"]]


def test_a_snapshot_row_with_no_curve_is_dropped_not_resumed(monkeypatch):
    """Rows written before the pending flag have no other tell of a cut write."""
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-24", "08-24")
    db.insert_run(
        run_id="lb_deepseek_v4_pro_20260801_20260826",
        session_id="leaderboard-live",
        agent_name="DeepSeek V4 Pro",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-26",
        initial_equity=SEED,
        llm_model="deepseek_v4_pro",
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-26"}},
    )
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-25", "2026-08-27", "08-24")]


def test_a_replay_with_no_snapshot_resumes_from_its_own_checkpoints(monkeypatch):
    """A newer row without a snapshot used to win every resume, so each retry of
    the month-open replay started from the 1st and re-billed the days done."""
    freeze_cfg = _live_aug27()
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260826", "deepseek_v4_pro", "2026-08-26",
        [_pt("2026-08-26T19:00:00+00:00", SEED * 1.1)],
    )
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(crash_on={"2026-08-12"}, runs=runs))
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-01", "2026-08-27", None)]
    # The old freeze keeps the board company while the replay catches up.
    shown = live.latest_live_month_runs("2026-08-01", "2026-08-27")["deepseek_v4_pro"]
    assert shown["end_date"] == "2026-08-26"

    runs.clear()
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-12", "2026-08-27", "08-11")]
    curve = db.get_equity_curve(row["run_id"])
    assert len(curve) == 19
    assert curve[0]["equity"] == SEED + 3
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row["run_id"]]


def test_a_forced_replay_killed_part_way_is_not_undone_by_the_next_refresh(monkeypatch):
    """The pre-force row ended at the freeze, so after a crash the next normal
    refresh called the window done and pruned the forced rows."""
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-27", "old", equity=SEED * 2)
    _use_agent(monkeypatch, _ScriptedAgent(crash_on={"2026-08-12"}))
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(
            _deepseek(freeze_cfg), freeze_cfg, force_refresh=True
        )
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-11"]

    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is False
    assert runs == [("2026-08-12", "2026-08-27", "08-11")]
    assert db.get_run(row["run_id"])["metadata"]["live_increment"]["forced"] is False


def _recorded_config(freeze_cfg, **overrides):
    """The config keys a checkpoint records, as the current entry would write them."""
    entry = _deepseek(freeze_cfg)
    return {
        "entry_id": entry["id"],
        "model_id": entry["model_id"],
        "integration": entry.get("integration"),
        "temperature": entry.get("temperature"),
        "reasoning_effort": entry.get("reasoning_effort"),
        "strategy_prompt": None,
        "llm_sell_sizing": LLM_SELL_SIZING,
        **overrides,
    }


def _insert_configured_checkpoint(end_date, day, config, **kwargs):
    _insert_live_run(
        f"lb_deepseek_v4_pro_20260801_{end_date.replace('-', '')}",
        "deepseek_v4_pro",
        end_date,
        [_pt(f"{end_date}T19:00:00+00:00", SEED)],
        metadata={
            **config,
            live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": day},
        },
        **kwargs,
    )


def test_a_checkpoint_from_before_sells_honoured_their_size_replays_the_month(
    monkeypatch, capsys
):
    """A checkpoint with no ``llm_sell_sizing`` sold whole positions whatever
    the size (#661). Its absence is that rule, not an unknown: resuming would
    splice a month sized two ways into one curve and one set of trade stats."""
    freeze_cfg = _live_aug27()
    legacy = _recorded_config(freeze_cfg)
    del legacy["llm_sell_sizing"]
    _insert_configured_checkpoint("2026-08-24", "08-24", legacy, sell_rule=None)
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    assert runs == [("2026-08-01", "2026-08-27", None)]
    assert "keys=llm_sell_sizing" in capsys.readouterr().out
    stored = db.get_run(row["run_id"])
    assert stored["metadata"]["live_increment"]["config_changed"] == ["llm_sell_sizing"]
    assert stored["metadata"]["llm_sell_sizing"] == LLM_SELL_SIZING


def test_a_checkpoint_under_the_same_config_resumes(monkeypatch):
    freeze_cfg = _live_aug27()
    _insert_configured_checkpoint("2026-08-24", "08-24", _recorded_config(freeze_cfg))
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-25", "2026-08-27", "08-24")]
    assert db.get_run(row["run_id"])["metadata"]["live_increment"]["config_changed"] is None


def test_an_entry_whose_config_changed_replays_the_month_instead_of_resuming(
    monkeypatch, capsys
):
    """Resuming would trade the rest of the month under the new sampling and
    re-record it over the whole curve, labelling sessions that ran under the
    old one. PR #605 pinned DeepSeek thinking-off one session into October."""
    freeze_cfg = _live_aug27()
    _insert_configured_checkpoint(
        "2026-08-24",
        "08-24",
        _recorded_config(freeze_cfg, temperature=None, reasoning_effort=None),
    )
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    assert runs == [("2026-08-01", "2026-08-27", None)]
    out = capsys.readouterr().out
    assert "WARNING: live.config_changed entry=deepseek_v4_pro" in out
    assert "keys=temperature,reasoning_effort" in out
    stored = db.get_run(row["run_id"])
    assert stored["metadata"]["live_increment"]["full_replay"] is True
    assert stored["metadata"]["live_increment"]["config_changed"] == [
        "temperature",
        "reasoning_effort",
    ]
    assert stored["metadata"]["temperature"] == 0
    assert stored["metadata"]["reasoning_effort"] == "none"
    # Counters start over with the replay, not on top of the old run's.
    sessions = live._trading_days_inclusive(date(2026, 8, 1), date(2026, 8, 27))
    assert stored["llm_calls"] == 7 * len(sessions)
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row["run_id"]]


def test_a_config_replay_killed_part_way_never_resumes_the_old_config(monkeypatch):
    """The old row outlives a replay killed before its first checkpoint, and the
    next refresh must reach the same verdict rather than resume from it."""
    freeze_cfg = _live_aug27()
    _insert_configured_checkpoint(
        "2026-08-24", "08-24", _recorded_config(freeze_cfg, temperature=0.7)
    )
    _use_agent(monkeypatch, _ScriptedAgent(crash_on={"2026-08-03"}))
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)

    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-01", "2026-08-27", None)]


def test_a_checkpoint_that_predates_a_config_key_is_not_a_change(monkeypatch):
    """Absent is unknown: rows written before a key existed must not each force
    a month-open replay of every model. ``llm_sell_sizing`` is the one
    exception, on purpose: its absence is a rule (``_RESUME_KEY_DEFAULTS``)."""
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-24", "08-24")
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-25", "2026-08-27", "08-24")]


def test_h6_judges_the_segment_not_each_session(monkeypatch):
    """One unusable reply in a 7-step session is 85.7%. Judged per session it
    failed a catch-up that clears 95% over the segment, as it always had to."""
    freeze_cfg = _live_aug27()
    agent = _use_agent(monkeypatch, _ScriptedAgent(misses={"2026-08-03": 1}))
    written = []
    real_write = live._write_live_checkpoint

    def spy(*args, **kwargs):
        written.append(kwargs["day_iso"])
        return real_write(*args, **kwargs)

    monkeypatch.setattr(live, "_write_live_checkpoint", spy)
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert len(agent.traded) == 19
    # 6/7 and 13/14 are short of 95%, so those closes are not stored; 20/21 is.
    assert written[0] == "2026-08-05"
    assert written[-1] == "2026-08-27"
    assert db.get_run(row["run_id"])["llm_decisions"] == 132


def test_a_segment_that_can_no_longer_clear_h6_stops_billing(monkeypatch):
    freeze_cfg = _live_aug27()
    agent = _use_agent(monkeypatch, _ScriptedAgent(misses={"2026-08-03": 7}))
    with pytest.raises(lb_service.LeaderboardFallbackError):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # 7 misses of 133 steps already caps the segment at 94.7%.
    assert agent.traded == ["2026-08-03"]
    assert _month_rows("deepseek_v4_pro") == []


# What a failed night leaves behind, pinned as it is (2026-10-14). The rule
# these encode: a session is checkpointed only once the segment traded so far
# clears H6, so billed sessions H6 has not yet accepted are lost when the run
# dies, and the next night re-bills them. Storing that prefix anyway would
# publish a curve the model did not drive; do not "fix" these by doing so.


class _DeadModelAgent(_ScriptedAgent):
    """Every call fails before generating anything (a gateway 404): each step
    falls back to rule-based, so sessions are offered and traded while
    ``llm_calls`` and ``llm_decisions`` never move."""

    def run(self, bars, start, end, capital, starting_snapshot=None, on_session_close=None):
        self.runs.append((start, end, (starting_snapshot or {}).get("day")))
        days = live._trading_days_inclusive(date.fromisoformat(start), date.fromisoformat(end))
        self.planned_decision_steps = self.steps * len(days)
        curve = []
        for day in days:
            iso = day.isoformat()
            self.traded.append(iso)
            self.decision_steps += self.steps
            curve.append(_pt(f"{iso}T19:00:00+00:00", SEED))
            self.last_portfolio_snapshot = {"cash": SEED, "positions": {}, "day": iso[5:]}
            on_session_close(day, list(curve))
        return curve


def test_a_dead_model_stores_nothing_and_restarts_at_the_1st_every_night(monkeypatch):
    freeze_cfg = _live_aug27()
    for _night in range(2):
        runs = []
        agent = _use_agent(monkeypatch, _DeadModelAgent(runs=runs))
        with pytest.raises(lb_service.LeaderboardFallbackError, match="llm_calls=0"):
            live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
        # Aug 1 is a Saturday: the first session is the 3rd, and it is the
        # only one traded before the guard stops the run.
        assert runs == [("2026-08-01", "2026-08-27", None)]
        assert agent.traded == ["2026-08-03"]
        assert agent.llm_calls == 0
        assert _month_rows("deepseek_v4_pro") == []


def test_an_early_miss_then_a_crash_loses_the_unchecked_prefix(monkeypatch):
    freeze_cfg = _live_aug27()
    agent = _use_agent(
        monkeypatch,
        _ScriptedAgent(misses={"2026-08-03": 1}, crash_on={"2026-08-05"}),
    )
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # 6/7 and 13/14 are under 95%, so neither close was stored, yet both
    # sessions were billed.
    assert agent.traded == ["2026-08-03", "2026-08-04"]
    assert agent.llm_calls == 14
    assert _month_rows("deepseek_v4_pro") == []

    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-01", "2026-08-27", None)]


def test_a_single_miss_on_a_nightly_increment_keeps_the_prior_checkpoint(monkeypatch):
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-26", "08-26")
    agent = _use_agent(monkeypatch, _ScriptedAgent(misses={"2026-08-27": 1}))
    with pytest.raises(lb_service.LeaderboardFallbackError, match="6/7"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert agent.llm_calls == 7
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-26"]

    # The retry resumes from that checkpoint, not from the 1st.
    runs = []
    _use_agent(monkeypatch, _ScriptedAgent(runs=runs))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert runs == [("2026-08-27", "2026-08-27", "08-26")]


def test_a_session_with_no_bars_stops_the_run_loudly(monkeypatch, capsys):
    """A trading day with no bars is broken data. Skipping it carried the book
    across a session it never traded, and the window was marked deployed."""
    freeze_cfg = _live_aug27()
    _insert_checkpoint("2026-08-24", "08-24")
    agent = _use_agent(monkeypatch, _ScriptedAgent(no_bars={"2026-08-26"}))
    with pytest.raises(live.LiveSessionGapError, match="2026-08-26"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert agent.traded == ["2026-08-25", "2026-08-27"]
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-25"]
    assert "ERROR: live.session_gap entry=deepseek_v4_pro missing=2026-08-26" in (
        capsys.readouterr().out
    )


def test_a_missing_last_session_fails_the_refresh_instead_of_marking_it_done(
    monkeypatch, capsys,
):
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    for entry_id in live.LIVE_MODEL_IDS:
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260825", entry_id, "2026-08-25",
            [_pt("2026-08-25T19:00:00+00:00", SEED)],
            metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
        )
    monkeypatch.setattr(
        lb_service, "get_strategy",
        lambda entry: _ScriptedAgent(no_bars={"2026-08-27"}),
    )
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)
    assert result["models_deployed"] is False
    assert {f["entry_id"] for f in result["model_failures"]} == set(live.LIVE_MODEL_IDS)
    assert result["model_results"] == []
    out = capsys.readouterr().out
    assert "ERROR: live.model_deploy_failed entry=deepseek_v4_pro" in out
    # The session that did land is stored; the window is retried next time. The
    # first two entries fail alike, so the roster stops and the third is never
    # attempted: it stays at its checkpoint.
    shown = live.latest_live_month_runs("2026-08-01", "2026-08-27")
    assert {
        row["llm_model"]: row["end_date"]
        for row in shown.values()
        if row["llm_model"] in live.LIVE_MODEL_IDS
    } == {
        "deepseek_v4_pro": "2026-08-26",
        "qwen3_7_plus": "2026-08-26",
        "claude_haiku_4_5": "2026-08-25",
    }
    assert live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)["skipped"] is False


def test_an_unexpected_error_from_one_model_does_not_stop_the_roster(
    monkeypatch, capsys,
):
    """The per-entry catch named three types. Anything else (a dropped database
    connection, a market-data socket error, a bug after the model calls)
    escaped it: every later entry went untraded, and the prune and the state
    save never ran. LLM provider errors never got this far; the non-strict
    decision loop turns them into rule-based steps, which H6 refuses."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    deepseek = _deepseek(freeze_cfg)
    second = {**deepseek, "id": "second_model", "name": "Second Model"}
    monkeypatch.setattr(live, "live_llm_entries", lambda *args, **kwargs: [deepseek, second])
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    healthy = _ScriptedAgent()

    def get_strategy(entry):
        if entry["id"] == "deepseek_v4_pro":
            raise ConnectionError("server closed the connection unexpectedly")
        return healthy

    monkeypatch.setattr(lb_service, "get_strategy", get_strategy)
    # A superseded row the end-of-refresh prune must still remove.
    for end in ("2026-08-25", "2026-08-26"):
        _insert_live_run(
            f"lb_spy_index_20260801_{end.replace('-', '')}", "spy_index", end,
            [_pt(f"{end}T19:00:00+00:00", SEED)],
        )

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)

    assert result["models_deployed"] is False
    assert result["model_failures"] == [
        {
            "entry_id": "deepseek_v4_pro",
            "error": "ConnectionError: server closed the connection unexpectedly",
        }
    ]
    assert [row["entry_id"] for row in result["model_results"]] == ["second_model"]
    assert healthy.traded[-1] == "2026-08-27"
    assert (
        "ERROR: live.model_deploy_failed entry=deepseek_v4_pro "
        "error=ConnectionError: server closed the connection unexpectedly"
    ) in capsys.readouterr().out
    assert result["pruned_runs"] == 1
    assert db.get_run("lb_spy_index_20260801_20260825") is None
    saved = live._live_refresh_state()
    assert saved["window_key"] == result["window_key"]
    assert saved["model_failures"] == result["model_failures"]


def test_a_database_error_mid_checkpoint_is_isolated_and_its_pending_row_pruned(
    monkeypatch, capsys,
):
    """The failure the wide catch is for lands after the model calls: the
    checkpoint's pending row is written, then the curve write raises. The next
    entry must still trade off the shared bar fetch, and the end-of-refresh
    prune must take the half-written row."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    deepseek = _deepseek(freeze_cfg)
    second = {**deepseek, "id": "second_model", "name": "Second Model"}
    monkeypatch.setattr(live, "live_llm_entries", lambda *args, **kwargs: [deepseek, second])
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    fetches = []
    monkeypatch.setattr(
        lb_service, "fetch_hourly_bars",
        lambda *args: fetches.append(args) or _one_bar_frame(),
    )
    agents = {"deepseek_v4_pro": _ScriptedAgent(), "second_model": _ScriptedAgent()}
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: agents[entry["id"]])
    for entry_id in agents:
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260826", entry_id, "2026-08-26",
            [_pt("2026-08-26T19:00:00+00:00", SEED)],
            metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
        )

    class OperationalError(Exception):
        pass

    OperationalError.__module__ = "psycopg"
    OperationalError.__qualname__ = "OperationalError"
    real_insert_points = db.insert_equity_points

    def insert_equity_points(run_id, points):
        if "deepseek_v4_pro" in run_id:
            # psycopg's real message for a dropped server, newlines and all.
            raise OperationalError(
                "server closed the connection unexpectedly\n"
                "\tThis probably means the server terminated abnormally\n"
                "\tbefore or while processing the request."
            )
        return real_insert_points(run_id, points)

    monkeypatch.setattr(db, "insert_equity_points", insert_equity_points)

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)

    error = (
        "psycopg.OperationalError: server closed the connection unexpectedly "
        "This probably means the server terminated abnormally before or while "
        "processing the request."
    )
    assert result["model_failures"] == [{"entry_id": "deepseek_v4_pro", "error": error}]
    assert agents["deepseek_v4_pro"].traded == ["2026-08-27"]  # billed, then failed
    assert [row["entry_id"] for row in result["model_results"]] == ["second_model"]
    assert agents["second_model"].traded == ["2026-08-27"]
    assert len(fetches) == 1  # the failed entry's bars still served the next one
    out = capsys.readouterr().out
    line = next(
        text for text in out.splitlines()
        if text.startswith("ERROR: live.model_deploy_failed")
    )
    assert line == f"ERROR: live.model_deploy_failed entry=deepseek_v4_pro error={error}"
    # Not one of the types the catch used to name, so the traceback that used to
    # escape the loop is still printed, after the one-line summary.
    after = out.split(line, 1)[1]
    assert "Traceback (most recent call last)" in after
    assert "in _write_live_checkpoint" in after
    deepseek_rows = _month_rows("deepseek_v4_pro")
    assert [row["end_date"] for row in deepseek_rows] == ["2026-08-26"]
    assert not any(live._is_pending_checkpoint(row) for row in deepseek_rows)
    assert [row["end_date"] for row in _month_rows("second_model")] == ["2026-08-27"]


def test_a_fault_every_entry_shares_stops_the_roster_after_two(monkeypatch, capsys):
    """A bug in the checkpoint path raises for every model after its session
    was billed. Carrying on would bill the whole roster for nothing, every
    night until the fix ships; two entries in a row failing the same way stop
    the rest."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    freeze_cfg = live.live_freeze_config(as_of)
    deepseek = _deepseek(freeze_cfg)
    roster = [deepseek] + [
        {**deepseek, "id": f"model_{n}", "name": f"Model {n}"} for n in (2, 3)
    ]
    monkeypatch.setattr(live, "live_llm_entries", lambda *args, **kwargs: roster)
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    agents = {entry["id"]: _ScriptedAgent() for entry in roster}
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: agents[entry["id"]])

    def broken_metadata(*args, **kwargs):
        raise KeyError("cash")

    monkeypatch.setattr(lb_service, "_llm_run_metadata", broken_metadata)

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)

    assert agents["deepseek_v4_pro"].traded and agents["model_2"].traded
    assert agents["model_3"].traded == []
    assert result["model_failures"] == [
        {"entry_id": "deepseek_v4_pro", "error": "KeyError: 'cash'"},
        {"entry_id": "model_2", "error": "KeyError: 'cash'"},
        {
            "entry_id": "model_3",
            "error": (
                "not attempted: deepseek_v4_pro and model_2 failed in a row "
                "with the same KeyError"
            ),
        },
    ]
    assert result["models_deployed"] is False
    assert (
        "ERROR: live.model_not_attempted entry=model_3 "
        "after=deepseek_v4_pro,model_2 error=KeyError"
    ) in capsys.readouterr().out
    assert live._live_refresh_state()["model_failures"] == result["model_failures"]


def test_a_base_exception_still_stops_the_refresh(monkeypatch):
    """Exception, not BaseException: a shutdown must not be logged as one
    model's failure while the loop carries on billing the rest."""
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())

    def get_strategy(entry):
        raise SystemExit("worker shutting down")

    monkeypatch.setattr(lb_service, "get_strategy", get_strategy)
    with pytest.raises(SystemExit):
        live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)
    assert live._live_refresh_state() == {}


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2026-12-31", "2026-11-30"),
        ("2026-03-31", "2026-02-28"),
        ("2028-03-30", "2028-02-29"),
        ("2026-01-31", "2025-12-31"),
        ("2026-08-27", "2026-07-27"),
    ],
)
def test_reference_start_clamps_to_the_prior_month(start, expected):
    from dashboard.backend.domain.leaderboard.strategies._common import (
        reference_start_date,
    )

    assert reference_start_date(start, None) == expected


def test_an_entry_several_sessions_behind_is_shown_but_not_ranked(no_alpaca):
    """A replay checkpoints as it goes, so hours into one the model's newest
    row is weeks behind the baselines. Ranking that value against theirs
    compares different days."""
    for entry_id, end in (
        ("spy_index", "2026-08-27"),
        ("deepseek_v4_pro", "2026-08-20"),
        ("qwen3_7_plus", "2026-08-26"),
    ):
        _insert_live_run(
            f"lb_{entry_id}_20260801_{end.replace('-', '')}", entry_id, end,
            [_pt("2026-08-03T13:00:00+00:00", SEED),
             _pt(f"{end}T19:00:00+00:00", SEED * 1.5)],
            total_return=0.5,
        )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    by_id = {e["entry_id"]: e for e in payload["entries"]}
    deepseek = by_id["deepseek_v4_pro"]
    assert deepseek["status"] == "catching_up"
    assert deepseek["rank"] is None
    assert deepseek["snapshot_end"] == "2026-08-20"
    assert deepseek["equity_curve"]
    # One session behind is the nightly append in flight, ranked as before.
    assert by_id["qwen3_7_plus"]["status"] == "frozen"
    assert by_id["qwen3_7_plus"]["rank"] is not None
    assert payload["live_status"]["catching_up"] == ["deepseek_v4_pro"]
    assert payload["leader"] == by_id["qwen3_7_plus"]["model"]


def test_llm_agent_fires_the_hook_per_session_from_one_book(monkeypatch):
    import pandas as pd

    from dashboard.backend.domain.leaderboard.strategies import llm_agent

    class Manager:
        instances = []

        def __init__(self, **kwargs):
            self.equity_history = []
            self.trades = []
            self.llm_calls = self.llm_decisions = 0
            self.input_tokens = self.output_tokens = 0
            Manager.instances.append(self)

        def restore_state(self, snapshot):
            pass

        def snapshot_state(self):
            return {"cash": 1.0, "positions": {}, "steps": len(self.equity_history)}

        def get_portfolio_state(self, *args):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            self.llm_calls += 1
            self.llm_decisions += 1
            return {"actions": []}

        def execute_actions(self, *args):
            pass

        def update_equity(self, market_data, price_cache, ts):
            self.equity_history.append({"timestamp": ts, "equity": 100.0})

        def get_equity_curve(self):
            return self.equity_history

    monkeypatch.setattr(llm_agent, "PortfolioManager", Manager)
    stamps = [pd.Timestamp(f"2026-08-03T{h}:00:00Z") for h in range(13, 20)]
    stamps += [pd.Timestamp(f"2026-08-04T{h}:00:00Z") for h in (13, 14)]
    strategy = llm_agent.LLMAgentStrategy({"strategy": "llm_agent", "model_id": "m"})
    seen = []

    def hook(day, curve):
        seen.append((
            day, len(curve), strategy.decision_steps, strategy.llm_calls,
            strategy.last_portfolio_snapshot["steps"], curve[-1]["timestamp"],
        ))

    strategy._run_decision_loop(
        client=object(), timestamps=stamps, symbols=["AAPL"], data={},
        price_cache={}, initial_capital=100.0, model_id="m", on_session_close=hook,
    )
    assert seen == [
        (date(2026, 8, 3), 7, 7, 7, 7, "2026-08-03T19:00:00+00:00"),
        (date(2026, 8, 4), 9, 9, 9, 9, "2026-08-04T14:00:00+00:00"),
    ]
    assert len(Manager.instances) == 1
    # The hook gets a copy; the live history keeps its Timestamps.
    assert isinstance(Manager.instances[0].equity_history[0]["timestamp"], pd.Timestamp)


def test_restored_lot_with_an_unreadable_date_is_released_not_kept():
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

    manager = PortfolioManager(initial_capital=SEED, t_plus_one_enabled=True)
    manager.restore_state({
        "cash": 100.0,
        "positions": {"AAPL": 10},
        "available_positions": {"AAPL": 4},
        "frozen_lots": {"AAPL": [
            {"quantity": 5, "buy_date": "not-a-date"},
            {"quantity": 1, "buy_date": None},
        ]},
    })
    assert manager.available_positions["AAPL"] == 10.0
    assert "AAPL" not in manager.frozen_lots


# --- Daily Sharpe with its 95% interval (P19; spec 2026-10-14 §4) -------------

# Conformance case M03 as session closes: r = [0, 0.02, -0.05, 0.05, 0.01].
_M03_FACTORS = (1.0, 1.02, 0.969, 1.01745, 1.0276245)


def _insert_sharpe_run(entry_id, days, factors, *, seed=SEED, sharpe=9.99):
    """One 15:00 ET bar per August session (19:00Z, open-stamped -> the 16:00
    node). ``sharpe`` is the stored column, which the board must ignore."""
    end = f"2026-08-{days[-1]:02d}"
    _insert_live_run(
        f"lb_{entry_id}_20260801_{end.replace('-', '')}", entry_id, end,
        [_pt(f"2026-08-{d:02d}T19:00:00+00:00", seed * f) for d, f in zip(days, factors)],
        total_return=factors[-1] - 1, sharpe=sharpe,
        metadata={"initial_capital": seed},
    )


def _sharpe_entry(payload, entry_id):
    return next(e for e in payload["entries"] if e["entry_id"] == entry_id)


def test_live_sharpe_is_daily_with_its_interval(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), _M03_FACTORS)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)
    assert deepseek["sharpe_ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert deepseek["sharpe_ci_high"] == pytest.approx(16.619919, abs=1e-6)
    assert deepseek["sharpe_days"] == 5
    assert deepseek["sharpe_status"] == "ok"


def test_live_sharpe_is_hidden_below_five_days(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 7), _M03_FACTORS[:4])
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 6, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["status"] == "frozen"
    assert deepseek["sharpe_ratio"] is None
    assert deepseek["sharpe_ci_low"] is None and deepseek["sharpe_ci_high"] is None
    assert deepseek["sharpe_days"] == 4
    assert deepseek["sharpe_status"] == "insufficient_days"


def test_live_sessions_before_the_first_print_count_as_idle_days(no_alpaca):
    """P19 counts every session in the window. Printing first on Aug 5 leaves
    Aug 3-4 at the seed: two zero returns, so five days, not three."""
    _insert_sharpe_run("deepseek_v4_pro", (5, 6, 7), (1.01, 1.0, 1.02))
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_days"] == 5
    assert deepseek["sharpe_status"] == "ok"
    assert deepseek["sharpe_ratio"] == pytest.approx(5.611652, abs=1e-6)


def test_live_sharpe_is_the_same_at_any_seed(no_alpaca):
    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), _M03_FACTORS, seed=10_000.0)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)


def test_a_pending_entry_has_no_sharpe(no_alpaca):
    _insert_sharpe_run("spy_index", range(3, 8), _M03_FACTORS)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["status"] == "pending"
    assert deepseek["sharpe_ratio"] is None
    assert deepseek["sharpe_ci_low"] is None and deepseek["sharpe_ci_high"] is None
    assert deepseek["sharpe_days"] == 0
    assert deepseek["sharpe_status"] == "pending"


def test_live_payload_states_its_sharpe_basis(no_alpaca):
    from dashboard.backend.domain.backtesting.metrics import daily_sharpe_basis

    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    assert payload["live_status"]["sharpe"] == {
        "basis": "daily_returns",
        "annualization_days": 252,
        "confidence": 0.95,
        "min_days": 5,
        "rf_annual": 0.0,
        "stderr": "iid",
    }
    assert payload["live_status"]["sharpe"] == daily_sharpe_basis()


def test_live_payload_survives_strict_json(no_alpaca):
    """A flat curve is no_variation, and nothing on the wire is NaN."""
    import json

    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), (1.0,) * 5)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_status"] == "no_variation"
    assert deepseek["sharpe_ratio"] is None
    json.dumps(payload, allow_nan=False)


# November 2026's NYSE sessions: Thanksgiving (the 26th) shut, the 27th an
# early close at 13:00.
_NOV_2026_SESSIONS = (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 27, 30)


def test_live_sharpe_counts_an_early_close_once(no_alpaca):
    """November 2026 through the 30th: 20 sessions, Thanksgiving (11-26) shut,
    11-27 closing at 13:00. Each session returns the next of conformance case
    M03's five returns, four times over, so the figure is hand-checkable:
    mean 0.006, stdev 0.0334664 (ddof 1), SR = 0.006 / 0.0334664 x sqrt(252)
    = 2.846050, and E_0 = the seed gives 20 returns, not 19.

    One 15:00 ET bar per full session (20:00Z in EST, open-stamped -> the 16:00
    node). On 11-27 the last bar is the 12:00 one (17:00Z), which closes at
    13:00, so the 13:00 node is that day's close. An axis with a fixed 16:00
    close draws 14:00-16:00 nodes after it carrying the same equity, so the
    Sharpe figure alone cannot tell the two axes apart; the curve assertions can.
    """
    from dashboard.backend.domain.backtesting.metrics import daily_session_closes

    closes, level = [], SEED
    for r in (0.0, 0.02, -0.05, 0.05, 0.01) * 4:
        level *= 1 + r
        closes.append(level)
    close_by_day = dict(zip(_NOV_2026_SESSIONS, closes))
    points = []
    for day, equity in close_by_day.items():
        if day == 27:
            # The 11:00 bar lands on the 12:00 node, the 12:00 bar on 13:00.
            points.append(_pt("2026-11-27T16:00:00+00:00", equity * 0.99))
            points.append(_pt("2026-11-27T17:00:00+00:00", equity))
        else:
            points.append(_pt(f"2026-11-{day:02d}T20:00:00+00:00", equity))
    _insert_live_run(
        "lb_deepseek_v4_pro_20261101_20261130", "deepseek_v4_pro", "2026-11-30", points,
        total_return=closes[-1] / SEED - 1, sharpe=9.99,
        metadata={"initial_capital": SEED}, start_date="2026-11-01",
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 11, 30, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")

    half_day = [
        (p["timestamp"][11:16], p["equity"])
        for p in deepseek["equity_curve"] if p["timestamp"].startswith("2026-11-27")
    ]
    assert half_day == [
        ("10:00", close_by_day[25]),
        ("11:00", close_by_day[25]),
        ("12:00", close_by_day[27] * 0.99),
        ("13:00", close_by_day[27]),
    ]
    daily = dict(daily_session_closes(deepseek["equity_curve"]))
    assert [day.day for day in daily] == list(_NOV_2026_SESSIONS)
    assert daily[date(2026, 11, 27)] == close_by_day[27]

    assert deepseek["sharpe_ratio"] == pytest.approx(2.846050, abs=1e-6)
    assert deepseek["sharpe_ci_low"] == pytest.approx(-4.166815, abs=1e-6)
    assert deepseek["sharpe_ci_high"] == pytest.approx(9.858915, abs=1e-6)
    assert deepseek["sharpe_days"] == 20
    assert deepseek["sharpe_status"] == "ok"


def test_a_baseline_row_publishes_the_same_daily_sharpe(no_alpaca):
    """Baselines and models share one reader; pin the baseline side too."""
    _insert_sharpe_run("spy_index", range(3, 8), _M03_FACTORS)
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    spy = _sharpe_entry(payload, "spy_index")
    assert spy["sharpe_status"] == "ok"
    assert spy["sharpe_days"] == 5
    assert spy["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)
    assert spy["sharpe_ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert spy["sharpe_ci_high"] == pytest.approx(16.619919, abs=1e-6)


def test_a_stored_curve_with_nothing_to_plot_is_pending_and_logged(no_alpaca, monkeypatch, caplog):
    """Absent and broken must differ (CLAUDE.md, fail-closed is not fail-visible).

    Every stored point is after the bell, so none lands on an axis node. Filling
    the axis from the seed drew a flat line, ranked it, and published Sharpe as
    "Returns never moved" with nothing logged."""
    monkeypatch.setattr(live, "_reported_unplottable", set())
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260807", "deepseek_v4_pro", "2026-08-07",
        [_pt(f"2026-08-{d:02d}T21:00:00+00:00", SEED * f) for d, f in zip(range(3, 8), _M03_FACTORS)],
        total_return=_M03_FACTORS[-1] - 1, sharpe=9.99,
        metadata={"initial_capital": SEED},
    )
    with caplog.at_level("ERROR"):
        for _ in range(2):
            payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["status"] == "pending"
    assert deepseek["rank"] is None
    assert deepseek["equity_curve"] == []
    assert deepseek["sharpe_status"] == "pending"
    assert deepseek["sharpe_ratio"] is None
    logged = [r.getMessage() for r in caplog.records if "live.curve_unplottable" in r.getMessage()]
    assert len(logged) == 1, "once per process, not on every public GET"
    assert "lb_deepseek_v4_pro_20260801_20260807" in logged[0]


def test_a_healthy_curve_logs_no_unplottable_error(no_alpaca, monkeypatch, caplog):
    monkeypatch.setattr(live, "_reported_unplottable", set())
    _insert_sharpe_run("deepseek_v4_pro", range(3, 8), _M03_FACTORS)
    with caplog.at_level("ERROR"):
        live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    assert not [r for r in caplog.records if "curve_unplottable" in r.getMessage()]


def test_a_row_with_no_recorded_seed_measures_day_one_in_its_own_dollars(no_alpaca):
    """A $10k run on the $100k board that did not record its seed is not
    rescaled, so its E_0 must be in its own dollars too: the first mark, as on
    the Competition board. Against display_capital, day 1 read about -90% and
    published a large negative Sharpe with status ok."""
    seed = 10_000.0
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260807", "deepseek_v4_pro", "2026-08-07",
        [_pt(f"2026-08-{d:02d}T19:00:00+00:00", seed * f) for d, f in zip(range(3, 8), _M03_FACTORS)],
        total_return=_M03_FACTORS[-1] - 1, sharpe=9.99, metadata=None,
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    # M03's first return is 0, so E_0 = the first mark changes nothing else.
    assert deepseek["sharpe_status"] == "ok"
    assert deepseek["sharpe_ratio"] == pytest.approx(2.611715, abs=1e-6)
    assert deepseek["sharpe_days"] == 5
    # The axis before the first print opens there as well, not at $100k.
    assert deepseek["equity_curve"][0]["equity"] == pytest.approx(seed)


def test_a_row_with_no_seed_at_all_is_undefined_and_logged(no_alpaca, monkeypatch, caplog):
    from dashboard.backend.domain.backtesting import metrics

    monkeypatch.setattr(metrics, "_logged_drift", set())
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260807", "deepseek_v4_pro", "2026-08-07",
        [_pt(f"2026-08-{d:02d}T19:00:00+00:00", SEED * f) for d, f in zip(range(3, 8), _M03_FACTORS)],
        total_return=_M03_FACTORS[-1] - 1, metadata=None, initial_equity=0.0,
    )
    with caplog.at_level("ERROR"):
        payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 7, 17, 39, tzinfo=_ET))
    deepseek = _sharpe_entry(payload, "deepseek_v4_pro")
    assert deepseek["sharpe_status"] == "undefined"
    assert deepseek["sharpe_ratio"] is None
    assert any("not a positive number" in r.getMessage() for r in caplog.records)
