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


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def no_alpaca(monkeypatch):
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda **kwargs: {
            "created": 0,
            "cache_hit": True,
            "refreshed_at": "2026-08-27T00:00:00+00:00",
        },
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


def test_frozen_curve_maps_utc_hours_onto_et_axis_and_stops(no_alpaca):
    """Alpaca stores UTC hours; the live axis is ET. Do not paint past freeze."""
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
                "timestamp": "2026-08-03T14:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T20:00:00+00:00",
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


def test_live_api_freeze_uses_live_session_not_contest(client, monkeypatch):
    seen = {}

    def fake_ensure(**kwargs):
        seen.update(kwargs)
        return {"created": 0, "cache_hit": True, "refreshed_at": "t"}

    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", fake_ensure)
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
    cfg = seen.get("config") or {}
    assert cfg.get("session_id") == "leaderboard-live"
    assert cfg.get("start_date") == "2026-08-01"
    assert cfg.get("end_date") == "2026-08-27"
    assert cfg.get("start_date") != "2026-04-15"


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
                "timestamp": "2026-08-03T14:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T20:00:00+00:00",
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
                "timestamp": "2026-08-03T14:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-27T20:00:00+00:00",
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
                "timestamp": "2026-08-03T14:00:00+00:00",
                "equity": SEED,
                "cash": SEED,
                "positions_value": 0,
            },
            {
                "timestamp": "2026-08-26T20:00:00+00:00",
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
