"""Tests for leaderboard API."""

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.database as db_module
import dashboard.backend.domain.leaderboard.service as lb_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "leaderboard.db"
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(lb_service, "db", test_db)
    return TestClient(app)


def _seed_leaderboard_runs(db, session_id="leaderboard-contest"):
    start = "2026-04-15"
    end = "2026-05-15"

    db.insert_run(
        run_id="lb_djia_index_20260415_20260515",
        session_id=session_id,
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date=start,
        end_date=end,
        initial_equity=100000,
        final_equity=105000,
        total_return=0.05,
        sharpe_ratio=1.2,
        max_drawdown=-0.02,
        num_trades=1,
        llm_model="djia_index",
    )
    db.insert_equity_points(
        "lb_djia_index_20260415_20260515",
        [
            {"timestamp": "2026-04-15T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-05-15T20:00:00", "equity": 105000, "cash": 0, "positions_value": 105000},
        ],
    )

    db.insert_run(
        run_id="lb_spy_index_20260415_20260515",
        session_id=session_id,
        agent_name="Agentic Trading Lab",
        mode="leaderboard",
        start_date=start,
        end_date=end,
        initial_equity=100000,
        final_equity=103000,
        total_return=0.03,
        sharpe_ratio=0.9,
        max_drawdown=-0.03,
        num_trades=1,
        llm_model="spy_index",
    )
    db.insert_equity_points(
        "lb_spy_index_20260415_20260515",
        [
            {"timestamp": "2026-04-15T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
            {"timestamp": "2026-05-15T20:00:00", "equity": 103000, "cash": 0, "positions_value": 103000},
        ],
    )


def test_leaderboard_api_returns_baselines(client, monkeypatch):
    _seed_leaderboard_runs(lb_service.db)

    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-contest",
            "start_date": "2026-04-15",
            "end_date": "2026-05-15",
            "period": "contest",
            "created": 0,
            "refreshed_at": "2026-06-18T00:00:00+00:00",
        },
    )

    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_entries"] == 2
    assert body["period"] == "contest"
    assert body["phase_label"] == "Preseason"
    assert len(body["entries"]) == 2
    names = {e["team_name"] for e in body["entries"]}
    assert names == {"Agentic Trading Lab"}
    models = {e["model"] for e in body["entries"]}
    assert "DJIA" in models
    assert "SPY" in models
    assert body["entries"][0]["rank"] == 1
    assert body["entries"][0]["entry_type"] == "baseline"


def test_daily_leaderboard_api_uses_daily_window(client, monkeypatch):
    day = "2026-07-14"  # Tuesday
    monkeypatch.setattr(lb_service, "daily_window_dates", lambda as_of=None: (day, day))

    start = day
    end = day
    for strategy_id, final, ret, sharpe in (
        ("djia_index", 101000, 0.01, 0.5),
        ("spy_index", 100500, 0.005, 0.4),
    ):
        run_id = f"lb_{strategy_id}_{start.replace('-', '')}_{end.replace('-', '')}"
        lb_service.db.insert_run(
            run_id=run_id,
            session_id="leaderboard-daily",
            agent_name="Agentic Trading Lab",
            mode="leaderboard",
            start_date=start,
            end_date=end,
            initial_equity=100000,
            final_equity=final,
            total_return=ret,
            sharpe_ratio=sharpe,
            max_drawdown=-0.01,
            num_trades=1,
            llm_model=strategy_id,
        )
        lb_service.db.insert_equity_points(
            run_id,
            [
                {"timestamp": f"{start}T14:00:00", "equity": 100000, "cash": 0, "positions_value": 100000},
                {"timestamp": f"{end}T20:00:00", "equity": final, "cash": 0, "positions_value": final},
            ],
        )

    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-daily",
            "start_date": day,
            "end_date": day,
            "period": "daily",
            "created": 0,
            "refreshed_at": "2026-07-15T00:00:00+00:00",
        },
    )

    resp = client.get("/api/v1/leaderboard?period=daily")
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "daily"
    assert body["phase_label"] == "Daily"
    assert body["standings_label"] == "Ranking"
    assert body["window"]["start_date"] == day
    assert body["window"]["end_date"] == day
    assert body["total_entries"] == 2
    assert body["entries"][0]["rank"] == 1


def test_daily_window_dates_skips_weekend():
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    # Monday before the cash close → previous Friday
    start, end = lb_service.daily_window_dates(
        as_of=datetime(2026, 7, 13, 15, 59, tzinfo=et)
    )
    assert start == end == "2026-07-10"
    # Monday at/after 16:00 ET → Monday (same-day board after close)
    start, end = lb_service.daily_window_dates(
        as_of=datetime(2026, 7, 13, 16, 0, tzinfo=et)
    )
    assert start == end == "2026-07-13"
    # Tuesday morning → Monday
    start, end = lb_service.daily_window_dates(
        as_of=datetime(2026, 7, 14, 9, 30, tzinfo=et)
    )
    assert start == end == "2026-07-13"
    # Bare date is treated as that ET day at cash close → weekday = that session
    start, end = lb_service.daily_window_dates(as_of=date(2026, 7, 14))
    assert start == end == "2026-07-14"


def test_daily_window_dates_cron_moment_is_todays_session():
    """22:30 UTC after a weekday close must select that US session, not Friday."""
    from datetime import datetime, timezone

    # Monday 22:30 UTC = Monday 18:30 EDT (after 16:00 close)
    start, end = lb_service.daily_window_dates(
        as_of=datetime(2026, 7, 13, 22, 30, tzinfo=timezone.utc)
    )
    assert start == end == "2026-07-13"


def test_ensure_leaderboard_runs_skips_refetch_after_empty_curve(tmp_path, monkeypatch):
    """A failed baseline must not force Alpaca refetch on every page load."""
    db_path = tmp_path / "lb.db"
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(lb_service, "db", test_db)
    monkeypatch.setattr(lb_service, "_SKIP_CACHE_PATH", tmp_path / "skips.json")

    config = {
        "session_id": "leaderboard-daily",
        "start_date": "2026-07-14",
        "end_date": "2026-07-14",
        "initial_capital": 1000,
        "period": "daily",
        "strategies": [
            {
                "id": "equal_weight_djia",
                "name": "Agentic Trading Lab",
                "label": "Baseline Strategy",
                "model": "Equal-Weight",
                "strategy": "equal_weight_index",
                "symbols": [],
            },
            {
                "id": "mean_variance_djia",
                "name": "Agentic Trading Lab",
                "label": "Baseline Strategy",
                "model": "Mean-Variance",
                "strategy": "mean_variance",
                "symbols": [],
            },
        ],
    }

    class _Ok:
        used_llm = None

        def required_symbols(self):
            return ["AAPL"]

        def run(self, bars, start, end, capital):
            return [
                {"timestamp": f"{start}T14:00:00", "equity": capital, "cash": 0, "positions_value": capital},
                {"timestamp": f"{end}T20:00:00", "equity": capital * 1.01, "cash": 0, "positions_value": capital * 1.01},
            ]

        def num_trades(self):
            return 1

    class _Empty:
        used_llm = None

        def required_symbols(self):
            return ["AAPL"]

        def run(self, bars, start, end, capital):
            return []

        def num_trades(self):
            return 0

    def fake_get_strategy(strategy):
        return _Ok() if strategy["id"] == "equal_weight_djia" else _Empty()

    fetch_calls = {"n": 0}

    def fake_fetch(symbols, start, end):
        fetch_calls["n"] += 1
        return {"AAPL": object()}

    monkeypatch.setattr(lb_service, "get_strategy", fake_get_strategy)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", fake_fetch)
    monkeypatch.setattr(lb_service, "_config_needs_alpaca", lambda cfg: True)
    monkeypatch.setattr(lb_service, "_alpaca_bars_start", lambda cfg: cfg["start_date"])
    monkeypatch.setattr(lb_service, "_symbols_for_config", lambda cfg: ["AAPL"])
    monkeypatch.setattr(
        lb_service,
        "calc_metrics",
        lambda curve, capital: {
            "initial_equity": capital,
            "final_equity": capital * 1.01,
            "total_return": 0.01,
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.01,
        },
    )

    first = lb_service.ensure_leaderboard_runs(config=config)
    assert first["created"] == 1
    assert first["skipped"] == 1
    assert fetch_calls["n"] == 1

    second = lb_service.ensure_leaderboard_runs(config=config)
    assert second.get("cache_hit") is True
    assert second["created"] == 0
    assert fetch_calls["n"] == 1  # no second Alpaca pull


def test_prune_stale_window_skips_bounds_daily_sidecar(tmp_path, monkeypatch):
    """Rolling daily windows must not accumulate skip entries forever."""
    import json as _json

    monkeypatch.setattr(lb_service, "_SKIP_CACHE_PATH", tmp_path / "skips.json")

    cache = {
        # stale daily windows — should be dropped
        "leaderboard-daily|2026-07-10|2026-07-10|mean_variance_djia": "empty_curve",
        "leaderboard-daily|2026-07-13|2026-07-13|mean_variance_djia": "no_bars",
        # current daily window — must be kept
        "leaderboard-daily|2026-07-14|2026-07-14|mean_variance_djia": "empty_curve",
        # a different session (contest) — must always be kept
        "leaderboard|2025-01-01|2025-03-31|mean_variance_djia": "no_bars",
    }

    kept = lb_service._prune_stale_window_skips(
        "leaderboard-daily", "2026-07-14", "2026-07-14", cache
    )

    assert set(kept) == {
        "leaderboard-daily|2026-07-14|2026-07-14|mean_variance_djia",
        "leaderboard|2025-01-01|2025-03-31|mean_variance_djia",
    }
    # the reduced set was persisted to disk
    on_disk = _json.loads((tmp_path / "skips.json").read_text(encoding="utf-8"))
    assert set(on_disk) == set(kept)


def test_prune_stale_window_skips_noop_for_fixed_window(tmp_path, monkeypatch):
    """A fixed-window (contest) board has no other-window keys under its session,
    so nothing is pruned and no needless disk write happens."""
    monkeypatch.setattr(lb_service, "_SKIP_CACHE_PATH", tmp_path / "skips.json")

    cache = {
        "leaderboard|2025-01-01|2025-03-31|mean_variance_djia": "no_bars",
        # a daily entry is a *different* session (the trailing '|' guards against
        # the "leaderboard" prefix matching "leaderboard-daily") → preserved.
        "leaderboard-daily|2026-07-13|2026-07-13|mean_variance_djia": "empty_curve",
    }
    kept = lb_service._prune_stale_window_skips(
        "leaderboard", "2025-01-01", "2025-03-31", dict(cache)
    )
    assert kept == cache  # unchanged
    assert not (tmp_path / "skips.json").exists()  # no write when nothing pruned


def test_daily_leaderboard_includes_status_block(client, monkeypatch):
    day = "2026-07-14"
    monkeypatch.setattr(lb_service, "daily_window_dates", lambda as_of=None: (day, day))
    monkeypatch.setattr(lb_service, "maybe_schedule_daily_leaderboard_refresh", lambda **_: False)
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-daily",
            "start_date": day,
            "end_date": day,
            "period": "daily",
            "created": 0,
            "refreshed_at": "2026-07-15T00:00:00+00:00",
        },
    )

    resp = client.get("/api/v1/leaderboard?period=daily")
    assert resp.status_code == 200
    body = resp.json()
    status = body["daily_status"]
    assert status["trading_date"] == day
    assert status["models_total"] >= 1
    assert status["models_pending"] == status["models_total"]


def test_daily_refresh_endpoint_requires_secret(client, monkeypatch):
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    resp = client.post("/api/v1/leaderboard/daily/refresh")
    assert resp.status_code == 401

    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.enqueue_daily_leaderboard_refresh",
        lambda **_: {
            "accepted": True,
            "started": True,
            "refresh_in_progress": True,
            "window": {"start_date": "2026-07-14", "end_date": "2026-07-14", "label": "2026-07-14"},
            "message": "Daily leaderboard refresh started in the background.",
        },
    )
    ok = client.post(
        "/api/v1/leaderboard/daily/refresh?deploy_models=false",
        headers={"X-Leaderboard-Refresh-Secret": "cron-secret"},
    )
    assert ok.status_code == 202
    body = ok.json()
    assert body["accepted"] is True
    assert body["started"] is True
    assert body["window"]["start_date"] == "2026-07-14"


def test_enqueue_daily_refresh_returns_immediately(monkeypatch):
    """Cron path must schedule a thread, not block on deploy_model_run."""
    day = "2026-07-14"
    monkeypatch.setattr(lb_service, "daily_window_dates", lambda as_of=None: (day, day))
    monkeypatch.setattr(lb_service, "_daily_refresh_running", False)
    monkeypatch.setattr(
        lb_service,
        "_daily_models_status",
        lambda config: {
            "trading_date": day,
            "models_total": 1,
            "models_cached": 0,
            "models_pending": 1,
            "pending_entry_ids": ["m1"],
            "refresh_in_progress": False,
        },
    )

    called = {}

    def _bg(**kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr(lb_service, "_run_daily_refresh_background", _bg)

    # Avoid actually starting a real Thread; invoke target inline via stub.
    class _ImmediateThread:
        def __init__(self, target=None, kwargs=None, **_):
            self._target = target
            self._kwargs = kwargs or {}

        def start(self):
            self._target(**self._kwargs)

    monkeypatch.setattr(lb_service.threading, "Thread", _ImmediateThread)

    payload = lb_service.enqueue_daily_leaderboard_refresh(
        deploy_models=True, force_refresh=False, allow_fallback=False
    )
    assert payload["accepted"] is True
    assert payload["started"] is True
    assert called["kwargs"]["deploy_models"] is True


def test_daily_window_dates_on_saturday_shows_friday():
    from datetime import date

    start, end = lb_service.daily_window_dates(as_of=date(2026, 7, 18))  # Saturday
    assert start == end == "2026-07-17"  # Friday


def test_partial_model_deploy_does_not_mark_window_complete(tmp_path, monkeypatch):
    """A mixed success/failure run must not skip remaining models on the next cron."""
    day = "2026-07-14"
    state_path = tmp_path / "daily_refresh.json"
    monkeypatch.setattr(lb_service, "_DAILY_REFRESH_STATE_PATH", state_path)
    monkeypatch.setattr(lb_service, "daily_window_dates", lambda as_of=None: (day, day))
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-daily",
            "start_date": day,
            "end_date": day,
            "period": "daily",
            "created": 0,
            "refreshed_at": "2026-07-15T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        lb_service,
        "llm_leaderboard_entries",
        lambda config=None: [{"id": "ok-model"}, {"id": "fail-model"}],
    )

    def _deploy(entry_id, **_kwargs):
        if entry_id == "fail-model":
            raise RuntimeError("boom")
        return {"entry_id": entry_id, "run_id": f"run-{entry_id}", "total_return": 0.01}

    monkeypatch.setattr(lb_service, "deploy_model_run", _deploy)

    first = lb_service.refresh_daily_leaderboard(deploy_models=True, force_refresh=True)
    assert first["models_deployed"] is False
    assert len(first["model_failures"]) == 1
    assert len(first["model_results"]) == 1

    # Without force, a wrongly-true models_deployed flag would skip forever.
    second = lb_service.refresh_daily_leaderboard(deploy_models=True, force_refresh=False)
    assert second.get("skipped") is not True
