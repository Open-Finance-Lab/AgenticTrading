"""Tests for leaderboard API."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.api.routers.leaderboard as leaderboard_router
import dashboard.backend.database as db_module
import dashboard.backend.domain.leaderboard.service as lb_service
import dashboard.backend.infrastructure.market_data.alpaca_bars as alpaca_bars


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


def test_leaderboard_api_does_not_leak_exception_text(client, monkeypatch):
    """Public GET must answer 500 with a static message, never raw exception text.

    Regression for the Neon-host leak: a psycopg connection failure message
    embeds the database endpoint id, and this endpoint is public +
    unauthenticated, so `detail=str(exc)` would hand that to any visitor.
    The traceback still goes to the server log via print().
    """
    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.get_leaderboard",
        lambda **kwargs: (_ for _ in ()).throw(
            # A psycopg connection failure is a plain Exception (OperationalError),
            # so it lands in the generic branch. The RuntimeError branch is
            # covered separately below — it is equally unsafe to echo.
            Exception("stale connection: could not connect to "
                      "ep-abc123.us-east-2.aws.neon.tech:5432"),
        ),
    )

    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Failed to load leaderboard"
    # The internal endpoint id must never appear in the public response.
    assert "neon.tech" not in resp.text
    assert "ep-abc123" not in resp.text


def test_leaderboard_api_does_not_leak_runtime_error_text(client, monkeypatch, tmp_path):
    """The 503 branch must be static too — RuntimeError is not a safe allowlist.

    ``MarketDataUnavailableError`` -> ``AlpacaCredentialsError`` subclass
    ``RuntimeError`` (they were made RuntimeErrors to escape SystemExit, not as
    a safety marker), and reach this handler through ``get_leaderboard`` ->
    ``ensure_leaderboard_runs`` -> ``fetch_hourly_bars``. Their message embeds
    the server-side credentials path, so ``detail=str(exc)`` on the 503 handed
    that path to anonymous callers exactly like the 500 branch did — the daily
    board hits it on any window with no cached runs yet.

    The exception is raised by the *real* loader rather than hand-written, so a
    reworded upstream message cannot drift away from this assertion.
    """
    creds_dir = tmp_path / "server-only" / "credentials"
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_bars, "CREDENTIALS_DIR", creds_dir)

    def _raise_from_real_loader(**kwargs):
        alpaca_bars.AlpacaDataLoader()  # raises AlpacaCredentialsError

    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.get_leaderboard",
        _raise_from_real_loader,
    )

    # Sanity: the error really is a RuntimeError, so it really does take the
    # 503 branch rather than the already-covered generic one.
    with pytest.raises(RuntimeError) as raised:
        alpaca_bars.AlpacaDataLoader()
    assert str(creds_dir) in str(raised.value)

    resp = client.get("/api/v1/leaderboard")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Leaderboard is temporarily unavailable"
    # The server-side credentials path must never reach a public caller.
    assert str(creds_dir) not in resp.text
    assert "alpaca.json" not in resp.text


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


@pytest.mark.daily_refresh_background
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
    done = threading.Event()

    # Stub the *work*, not threading.Thread. Patching lb_service.threading.Thread
    # would rebind the attribute on the stdlib threading module itself -- every
    # thread created anywhere in the process for the duration of this test. The
    # real _run_daily_refresh_background still runs here, so its try/finally
    # still releases _daily_refresh_running exactly as in production.
    def _fake_refresh(**kwargs):
        called["kwargs"] = kwargs
        done.set()
        return {}

    monkeypatch.setattr(lb_service, "refresh_daily_leaderboard", _fake_refresh)

    payload = lb_service.enqueue_daily_leaderboard_refresh(
        deploy_models=True, force_refresh=False
    )
    assert payload["accepted"] is True
    assert payload["started"] is True

    # enqueue returned without waiting for the worker -- that is the property
    # under test -- so join before asserting on what the worker received.
    assert done.wait(5), "background refresh worker never ran"
    assert called["kwargs"]["deploy_models"] is True
    assert "allow_fallback" not in called["kwargs"], (
        "the background path must never be able to waive the H6 integrity guard"
    )

    # The worker's finally clause must hand the flag back, or every later
    # refresh in this process returns False and the UI polls a dead worker.
    for _ in range(50):
        if not lb_service._daily_refresh_running:
            break
        time.sleep(0.01)
    assert lb_service._daily_refresh_running is False


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


# --- Review hardening: the daily refresh path spends real money -------------
# GET /api/v1/leaderboard?period=daily is public and unauthenticated, and it
# calls maybe_schedule_daily_leaderboard_refresh() on every request. These pin
# the guards that stop an anonymous request turning into LLM billing.


class _StubThreading:
    """Stand-in for the threading module, scoped to lb_service only.

    Patching ``lb_service.threading.Thread`` would rebind the attribute on the
    real stdlib module -- affecting every thread created anywhere in the process
    for the duration of the test. Replacing the module *reference* this one
    module holds does not.
    """

    def __init__(self, thread_cls):
        self.Thread = thread_cls

    def __getattr__(self, name):
        return getattr(threading, name)


def test_auto_deploy_is_strict_opt_in(monkeypatch):
    """Unset must mean OFF, including where RENDER is absent (Docker, CI, forks)."""
    monkeypatch.delenv("LEADERBOARD_DAILY_AUTO_DEPLOY", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    assert lb_service._auto_deploy_daily_models_enabled() is False

    monkeypatch.setenv("LEADERBOARD_DAILY_AUTO_DEPLOY", "true")
    assert lb_service._auto_deploy_daily_models_enabled() is True

    monkeypatch.setenv("LEADERBOARD_DAILY_AUTO_DEPLOY", "false")
    assert lb_service._auto_deploy_daily_models_enabled() is False


@pytest.mark.daily_refresh_background
def test_public_daily_get_does_not_schedule_model_deploys(monkeypatch):
    """An anonymous GET must not be able to start a paid model deploy."""
    monkeypatch.delenv("LEADERBOARD_DAILY_AUTO_DEPLOY", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setattr(
        lb_service,
        "_daily_models_status",
        lambda config: {
            "trading_date": "2026-07-14",
            "models_total": 7,
            "models_cached": 0,
            "models_pending": 7,
            "pending_entry_ids": ["a"],
            "refresh_in_progress": False,
        },
    )
    started = {}

    def _fake_refresh(**kwargs):
        started["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(lb_service, "refresh_daily_leaderboard", _fake_refresh)

    lb_service.maybe_schedule_daily_leaderboard_refresh()
    for _ in range(200):
        if "kwargs" in started:
            break
        time.sleep(0.01)

    # A baselines-only refresh is fine; deploying models is not.
    assert started.get("kwargs", {}).get("deploy_models") is False


def test_daily_refresh_endpoint_hides_whether_secret_is_configured(client, monkeypatch):
    """Unconfigured must answer 401 like a wrong secret, not advertise 503."""
    monkeypatch.delenv("LEADERBOARD_DAILY_REFRESH_SECRET", raising=False)
    resp = client.post(
        "/api/v1/leaderboard/daily/refresh",
        headers={"X-Leaderboard-Refresh-Secret": "anything"},
    )
    assert resp.status_code == 401
    assert "LEADERBOARD_DAILY_REFRESH_SECRET" not in resp.text


def test_daily_refresh_endpoint_rejects_non_ascii_secret_header(client, monkeypatch):
    """compare_digest raises TypeError on a non-ASCII str -> would surface as 500.

    Sent as raw bytes because httpx refuses to encode a non-ASCII *str* header;
    a real HTTP client is under no such constraint, and Starlette latin-1
    decodes whatever arrives into chars above 127.
    """
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    resp = client.post(
        "/api/v1/leaderboard/daily/refresh",
        headers={"X-Leaderboard-Refresh-Secret": "sécret".encode("latin-1")},
    )
    assert resp.status_code == 401


def test_verify_daily_refresh_secret_handles_non_ascii(monkeypatch):
    """The boundary itself: a wrong secret is a PermissionError, never TypeError."""
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    with pytest.raises(PermissionError):
        lb_service.verify_daily_refresh_secret("sécret")


def test_daily_refresh_endpoint_has_no_h6_fallback_bypass(client, monkeypatch):
    """allow_fallback waives the H6 integrity guard; keep it off the HTTP surface."""
    import inspect

    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    seen = {}

    def _fake_enqueue(**kwargs):
        seen["kwargs"] = kwargs
        return {
            "accepted": True,
            "started": True,
            "refresh_in_progress": True,
            "window": {"start_date": "d", "end_date": "d", "label": "d"},
            "message": "ok",
        }

    monkeypatch.setattr(
        "dashboard.backend.api.routers.leaderboard.enqueue_daily_leaderboard_refresh",
        _fake_enqueue,
    )
    resp = client.post(
        "/api/v1/leaderboard/daily/refresh?deploy_models=false&allow_fallback=true",
        headers={"X-Leaderboard-Refresh-Secret": "cron-secret"},
    )
    assert resp.status_code == 202
    # The query param is ignored, never forwarded.
    assert "allow_fallback" not in seen["kwargs"]

    # And no function on the background path can accept it either.
    for fn in (
        lb_service.enqueue_daily_leaderboard_refresh,
        lb_service.maybe_schedule_daily_leaderboard_refresh,
        lb_service._run_daily_refresh_background,
    ):
        assert "allow_fallback" not in inspect.signature(fn).parameters, fn.__name__


def test_daily_refresh_endpoint_is_rate_limited(client, monkeypatch):
    """One shared secret plus unlimited attempts is an open guessing budget."""
    monkeypatch.setenv("LEADERBOARD_DAILY_REFRESH_SECRET", "cron-secret")
    limiter = leaderboard_router._daily_refresh_rate_limiter
    codes = {
        client.post(
            "/api/v1/leaderboard/daily/refresh",
            headers={"X-Leaderboard-Refresh-Secret": "wrong"},
        ).status_code
        for _ in range(limiter.max_events + 3)
    }
    assert 429 in codes


def test_daily_models_status_scans_runs_once(client, monkeypatch):
    """N+1: this used to be one full session scan per LLM entry, per public GET."""
    config = lb_service.resolve_leaderboard_config("daily")
    assert len(lb_service.llm_leaderboard_entries(config)) > 1  # else the bug hides
    calls = {"n": 0}
    real = lb_service.db.get_runs_by_session

    def _counting(session_id):
        calls["n"] += 1
        return real(session_id)

    monkeypatch.setattr(lb_service.db, "get_runs_by_session", _counting)
    lb_service._daily_models_status(config)
    assert calls["n"] == 1


@pytest.mark.daily_refresh_background
def test_thread_start_failure_releases_the_in_progress_flag(monkeypatch):
    """A stuck True flag blocks every later refresh and makes the UI poll forever."""
    monkeypatch.setattr(lb_service, "_daily_refresh_running", False)
    monkeypatch.setattr(
        lb_service,
        "_daily_models_status",
        lambda config: {
            "trading_date": "2026-07-14",
            "models_total": 0,
            "models_cached": 0,
            "models_pending": 0,
            "pending_entry_ids": [],
            "refresh_in_progress": False,
        },
    )

    class _ExplodingThread:
        def __init__(self, **_):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(lb_service, "threading", _StubThreading(_ExplodingThread))

    with pytest.raises(RuntimeError):
        lb_service.maybe_schedule_daily_leaderboard_refresh(force_refresh=True)
    assert lb_service._daily_refresh_running is False


def test_live_leaderboard_api_serves_the_contest_curves_under_season_chrome(client, monkeypatch):
    """The route, not the service function. `?period=live` was previously coerced
    to 'contest' *inside* `_normalize_period`, so every service-level assertion
    about the live board passed identically before and after the period existed.
    Only a request through the router proves FastAPI accepts the value, that it
    survives the `Query` declaration, and that `season` reaches the wire.
    """
    _seed_leaderboard_runs(lb_service.db)
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-contest",
            "start_date": "2026-04-15",
            "end_date": "2026-05-15",
            "period": "live",
            "created": 0,
            "refreshed_at": "2026-06-18T00:00:00+00:00",
        },
    )

    resp = client.get("/api/v1/leaderboard?period=live")
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "live"
    assert body["board_title"] == "Live Trading Leaderboard"
    # Same window as the contest board, deliberately: a distinct window would
    # miss `_find_cached_run` on every entry and start recomputing baselines --
    # and, with LEADERBOARD_DAILY_AUTO_DEPLOY armed, billable LLM deploys -- from
    # a public, unauthenticated GET.
    assert body["window"]["start_date"] == "2026-04-15"
    assert body["window"]["end_date"] == "2026-05-15"
    assert body["total_entries"] == 2

    # The board's own description, not the Competition board's rules.
    assert "contest window" not in body["window"]["description"]
    assert "preview" in body["window"]["description"].lower()

    # And it still reads as a preview to `seasonHasAdvanced()`.
    season = body["season"]
    assert season["number"] == 0
    assert season["last_advanced_date"] is None
    assert season["trading_days_elapsed"] == 0


def test_contest_leaderboard_api_carries_no_season(client, monkeypatch):
    """One fixed historical window is not a season. Attaching one would render
    the season strip on a board that has none."""
    _seed_leaderboard_runs(lb_service.db)
    monkeypatch.setattr(
        lb_service,
        "ensure_leaderboard_runs",
        lambda force_refresh=False, period="contest", config=None: {
            "session_id": "leaderboard-contest",
            "created": 0,
            "refreshed_at": "2026-06-18T00:00:00+00:00",
        },
    )
    assert "season" not in client.get("/api/v1/leaderboard").json()
