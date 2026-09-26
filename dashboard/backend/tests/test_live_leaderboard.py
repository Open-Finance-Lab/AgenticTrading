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

SEED = float(lb_service.load_leaderboard_config().get("initial_capital", 100_000))

_ET = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _isolated_live_session(monkeypatch, tmp_path):
    """The test DB is session-scoped: start every test with no live rows, and
    never let a refresh write the real state file under DATA_DIR."""
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    for run in db.get_runs_by_session("leaderboard-live") or []:
        db.delete_run(run["run_id"])
    yield
    for run in db.get_runs_by_session("leaderboard-live") or []:
        db.delete_run(run["run_id"])


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
                     start_date="2026-08-01", **extra):
    db.insert_run(
        run_id=run_id,
        session_id="leaderboard-live",
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date=start_date,
        end_date=end_date,
        initial_equity=points[0]["equity"],
        final_equity=points[-1]["equity"] if final is None else final,
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        num_trades=0,
        llm_model=llm_model,
        metadata=metadata,
        **extra,
    )
    db.insert_equity_points(run_id, points)


def _pt(ts, equity):
    return {"timestamp": ts, "equity": equity, "cash": equity, "positions_value": 0}


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
    from dashboard.backend.domain.leaderboard.us_market_calendar import nyse_holidays

    assert sorted(d.isoformat() for d in nyse_holidays(2026)) == [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    ]
    # A Saturday New Year's Day is not observed on the Friday before.
    assert date(2021, 12, 31) not in nyse_holidays(2021)
    assert date(2022, 1, 1) not in nyse_holidays(2022)


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
        run_id="lb_nemotron_3_nano_30b_20260801_20260827",
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
        llm_model="nemotron_3_nano_30b",
        llm_calls=10,
    )
    db.insert_equity_points(
        "lb_nemotron_3_nano_30b_20260801_20260827",
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
    model = next(e for e in payload["entries"] if e["entry_id"] == "nemotron_3_nano_30b")
    assert model["is_model"] is True
    assert model["equity_curve"]
    assert model["equity_curve"][-1]["timestamp"].startswith("2026-08-27T16:00")
    assert model["equity_curve"][-1]["equity"] == pytest.approx(SEED * 1.005)
    assert payload["live_status"]["models_cached"] >= 1
    assert payload["live_status"]["models_pending"] == 2
    pending = [e for e in payload["entries"] if e["status"] == "pending"]
    assert pending and all(e["rank"] is None for e in pending)
    assert payload["live_status"]["roster"] == [
        "gpt_5_5",
        "deepseek_v4_pro",
        "nemotron_3_nano_30b",
    ]
    model_ids = {e["entry_id"] for e in payload["entries"] if e.get("is_model")}
    assert model_ids == {"gpt_5_5", "deepseek_v4_pro", "nemotron_3_nano_30b"}


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

    def fake_deploy(entry, freeze_cfg, **kwargs):
        deployed.append(entry["id"])
        assert freeze_cfg.get("session_id") == "leaderboard-live"
        assert freeze_cfg.get("start_date") == "2026-08-01"
        assert freeze_cfg.get("end_date") == "2026-08-27"
        return {"entry_id": entry["id"], "run_id": f"lb_{entry['id']}", "cached": False}

    monkeypatch.setattr(live, "deploy_live_model_increment", fake_deploy)
    result = live.refresh_live_leaderboard(
        deploy_models=True,
        as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET),
    )
    assert result["skipped"] is False
    assert result["models_deployed"] is True
    assert "claude_haiku_4_5" not in deployed
    assert set(deployed) == {"gpt_5_5", "deepseek_v4_pro", "nemotron_3_nano_30b"}


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
        run_id="lb_gpt_5_5_20260801_20260826",
        session_id="leaderboard-live",
        agent_name="GPT-5.5",
        mode="leaderboard",
        start_date="2026-08-01",
        end_date="2026-08-26",
        initial_equity=SEED,
        final_equity=SEED * 1.01,
        total_return=0.01,
        sharpe_ratio=0.5,
        max_drawdown=-0.01,
        num_trades=2,
        llm_model="gpt_5_5",
        llm_calls=70,
        llm_decisions=70,
        input_tokens=1000,
        output_tokens=200,
        metadata={live.LIVE_SNAPSHOT_KEY: snapshot},
    )
    db.insert_equity_points(
        "lb_gpt_5_5_20260801_20260826",
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
            self.model_id = "openai/gpt-5.5"
            self.last_portfolio_snapshot = {
                "cash": 500.0,
                "positions": {"AAPL": 4},
            }

        def required_symbols(self):
            return ["AAPL"]

        def run(self, bars, start, end, capital, starting_snapshot=None):
            self.windows.append((start, end, starting_snapshot))
            return [
                {
                    "timestamp": "2026-08-27T14:00:00+00:00",
                    "equity": SEED * 1.02,
                    "cash": 500.0,
                    "positions_value": SEED * 1.02 - 500.0,
                }
            ]

        def num_trades(self):
            return 1

    fake = FakeAgent()
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: fake)
    monkeypatch.setattr(
        lb_service,
        "fetch_hourly_bars",
        lambda symbols, start, end: {"AAPL": type("Frame", (), {"attrs": {}})()},
    )

    entry = next(e for e in live.live_llm_entries(freeze_cfg) if e["id"] == "gpt_5_5")
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
    for entry_id in ("spy_index", "gpt_5_5"):
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260826", entry_id, "2026-08-26",
            [_pt("2026-08-03T13:00:00+00:00", SEED),
             _pt("2026-08-26T19:00:00+00:00", SEED * 0.9)],
            total_return=-0.1,
        )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    ranked = [e for e in payload["entries"] if e["rank"] is not None]
    assert {e["entry_id"] for e in ranked} == {"spy_index", "gpt_5_5"}
    assert [e["rank"] for e in ranked] == [1, 2]
    assert payload["entries"][: len(ranked)] == ranked
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["status"] == "pending"
    assert deepseek["portfolio_value"] is None
    gpt = next(e for e in ranked if e["entry_id"] == "gpt_5_5")
    assert payload["leader"] == gpt["model"]


def test_returns_come_off_the_stored_run_and_keep_the_first_hour(no_alpaca):
    """The first mark already carries the first hour's P&L. Rescaling by it
    (``initial_equity``) inflated every point and erased that hour."""
    first_mark = SEED * 0.995
    _insert_live_run(
        "lb_gpt_5_5_20260801_20260826", "gpt_5_5", "2026-08-26",
        [_pt("2026-08-03T13:00:00+00:00", first_mark),
         _pt("2026-08-26T19:00:00+00:00", SEED * 1.02)],
        total_return=0.02, sharpe=1.1, max_dd=-0.03,
        metadata={"initial_capital": SEED},
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    gpt = next(e for e in payload["entries"] if e["entry_id"] == "gpt_5_5")
    assert gpt["cumulative_return"] == pytest.approx(0.02)
    assert gpt["sharpe_ratio"] == pytest.approx(1.1)
    assert gpt["max_drawdown"] == pytest.approx(-0.03)
    assert gpt["portfolio_value"] == pytest.approx(SEED * 1.02)
    assert gpt["equity_curve"][0]["equity"] == pytest.approx(first_mark)


def test_dollar_axis_scales_by_the_recorded_seed_only(no_alpaca):
    seed = SEED / 10
    _insert_live_run(
        "lb_gpt_5_5_20260801_20260826", "gpt_5_5", "2026-08-26",
        [_pt("2026-08-03T13:00:00+00:00", seed * 0.99),
         _pt("2026-08-26T19:00:00+00:00", seed * 1.05)],
        total_return=0.05, metadata={"initial_capital": seed},
    )
    payload = live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))
    gpt = next(e for e in payload["entries"] if e["entry_id"] == "gpt_5_5")
    assert gpt["portfolio_value"] == pytest.approx(SEED * 1.05)
    assert gpt["cumulative_return"] == pytest.approx(0.05)
    assert gpt["equity_curve"][0]["equity"] == pytest.approx(SEED * 0.99)


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

        def run(self, bars, start, end, capital, starting_snapshot=None):
            return [_pt(f"{end}T19:00:00+00:00", SEED)]

        def num_trades(self):
            return 0

    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: Agent())
    monkeypatch.setattr(lb_service, "_reject_if_llm_fallback", lambda *a, **k: None)

    def fake_fetch(symbols, start, end):
        fetches.append((tuple(symbols), start, end))
        return {"AAPL": type("Frame", (), {"attrs": {}})()}

    monkeypatch.setattr(lb_service, "fetch_hourly_bars", fake_fetch)
    for entry_id in ("gpt_5_5", "deepseek_v4_pro"):
        _insert_live_run(
            f"lb_{entry_id}_20260801_20260826", entry_id, "2026-08-26",
            [_pt("2026-08-26T19:00:00+00:00", SEED)],
            metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}}},
        )
    memo = {}
    for entry in live.live_llm_entries(freeze_cfg):
        if entry["id"] in ("gpt_5_5", "deepseek_v4_pro"):
            live.deploy_live_model_increment(entry, freeze_cfg, bars_memo=memo)
    assert fetches == [(("AAPL", "MSFT"), "2026-07-27", "2026-08-27")]
