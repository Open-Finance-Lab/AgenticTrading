"""Tests for the calendar-month Live Trading Leaderboard.

Isolated from contest/daily: live freeze uses ``leaderboard-live`` and must
not rewrite the contest window.
"""

from datetime import datetime
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

    def fake_deploy(entry_id, **kwargs):
        deployed.append(entry_id)
        assert kwargs.get("config", {}).get("session_id") == "leaderboard-live"
        assert kwargs.get("start_date") == "2026-08-01"
        assert kwargs.get("end_date") == "2026-08-27"
        return {"entry_id": entry_id, "run_id": f"lb_{entry_id}", "cached": False}

    monkeypatch.setattr(lb_service, "deploy_model_run", fake_deploy)
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
