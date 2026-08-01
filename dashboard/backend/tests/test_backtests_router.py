"""MEDIUM #2 / #3 — /backtest routes hardening.

#3: GET /runs/{run_id}/plot.png must not block the event loop (sync handler ->
    threadpool), must not re-import/re-configure matplotlib per request, and
    should cache the immutable rendered PNG per run_id.
#2: POST /backtest/run must not let an anonymous caller burn operator LLM
    credits — model allowlist, prompt size cap, date-range cap, write rate limit.
"""

import inspect
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
import dashboard.backend.api.routers.backtests as bt


# ===========================================================================
# #3 — plot.png: event loop + caching
# ===========================================================================

def test_plot_png_handler_is_sync_offloaded():
    # Sync def -> FastAPI runs the CPU-bound render in a threadpool, not on the
    # event loop. (Was `async def`, which blocked the loop for the whole render.)
    assert not inspect.iscoroutinefunction(bt.get_run_plot)


def test_plot_png_matplotlib_hoisted_to_module():
    # The renderer no longer imports/configures matplotlib per call.
    src = inspect.getsource(bt._render_run_plot_png)
    assert "import matplotlib" not in src
    assert 'matplotlib.use(' not in src
    # It's configured once at module import instead.
    assert bt.matplotlib.get_backend().lower() == "agg"


def test_plot_png_cached_per_run(monkeypatch):
    bt._render_run_plot_png.cache_clear()
    calls = {"get_run": 0, "equity": 0}
    fake_run = {
        "session_id": None, "created_at": "2026-05-01T10:00:00", "agent_name": "Agent",
        "start_date": "2026-05-01", "end_date": "2026-05-07", "mode": "safe_trading",
        "baseline_buyhold_run_id": None, "baseline_djia_run_id": None,
    }

    def fake_get_run(rid):
        calls["get_run"] += 1
        return fake_run

    def fake_equity(rid):
        calls["equity"] += 1
        return [{"timestamp": "2026-05-01T10:00:00", "equity": 100000},
                {"timestamp": "2026-05-01T11:00:00", "equity": 101000}]

    monkeypatch.setattr(bt.db, "get_run", fake_get_run)
    monkeypatch.setattr(bt.db, "get_equity_curve", fake_equity)
    monkeypatch.setattr(bt, "filter_market_hours", lambda pts: pts)  # isolate caching

    first = bt._render_run_plot_png("run_x")
    second = bt._render_run_plot_png("run_x")

    assert first == second
    assert first[:8] == b"\x89PNG\r\n\x1a\n"      # valid PNG
    assert calls["get_run"] == 1                  # 2nd call served from cache
    bt._render_run_plot_png.cache_clear()


def test_plot_png_missing_run_not_cached(monkeypatch):
    # A 404 must not be cached: a run that appears later should still render.
    bt._render_run_plot_png.cache_clear()
    from fastapi import HTTPException
    monkeypatch.setattr(bt.db, "get_run", lambda rid: None)
    with pytest.raises(HTTPException):
        bt._render_run_plot_png("missing")
    # Nothing cached -> a second call re-queries (would render if data existed).
    hits = {"n": 0}

    def counting_get_run(rid):
        hits["n"] += 1
        return None

    monkeypatch.setattr(bt.db, "get_run", counting_get_run)
    with pytest.raises(HTTPException):
        bt._render_run_plot_png("missing")
    assert hits["n"] == 1  # re-evaluated, not served from a cached exception
    bt._render_run_plot_png.cache_clear()


# ===========================================================================
# #2 — /backtest/run: cost-abuse hardening
# ===========================================================================

class _Spy:
    def __init__(self):
        self.calls = 0
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *a, **k):
        self.calls += 1
        self.last_args = a
        self.last_kwargs = k


def _run_record(metadata=None):
    return {
        "run_id": "run_source",
        "agent_name": "Agent",
        "mode": "backtest",
        "start_date": "2026-04-01",
        "end_date": "2026-04-23",
        "initial_equity": 100_000,
        "num_trades": 1,
        "created_at": "2026-04-23T16:00:00",
        "metadata": metadata,
    }


def test_run_metadata_response_exposes_simulation_source():
    response = bt._run_metadata_response(
        _run_record({"data_source": "vnpy_simulation"})
    )

    assert response.data_source == "vnpy_simulation"


def test_run_metadata_response_defaults_legacy_runs_to_alpaca():
    assert bt._run_metadata_response(_run_record()).data_source == "alpaca"


def test_run_metadata_response_exposes_complete_ifind_profile():
    response = bt._run_metadata_response(
        _run_record(
            {
                "data_source": "ifind_ashare",
                "market": "CN",
                "universe": "a_share_demo_6",
                "timeframe": "60m",
                "timezone": "Asia/Shanghai",
                "decision_source": "rule_based",
                "benchmark": "equal_weight_buyhold",
                "symbols": ["600519.SH", "601318.SH"],
                "native_currency": "CNY",
                "reporting_currency": "USD",
                "native_initial_capital": 7_000,
                "fx_pair": "USD/CNY",
                "fx_source": "ifind_history_currency_conversion",
                "fx_policy": "daily_implied_median_forward_fill",
                "fx_start_rate": 7.0,
                "fx_end_rate": 7.1,
            }
        )
    )

    assert response.data_source == "ifind_ashare"
    assert response.market == "CN"
    assert response.universe == "a_share_demo_6"
    assert response.timeframe == "60m"
    assert response.timezone == "Asia/Shanghai"
    assert response.decision_source == "rule_based"
    assert response.benchmark == "equal_weight_buyhold"
    assert response.symbols == ["600519.SH", "601318.SH"]
    assert response.native_currency == "CNY"
    assert response.reporting_currency == "USD"
    assert response.native_initial_capital == 7_000
    assert response.fx_pair == "USD/CNY"
    assert response.fx_source == "ifind_history_currency_conversion"
    assert response.fx_start_rate == 7.0
    assert response.fx_end_rate == 7.1


def test_run_metadata_response_keeps_new_fields_optional_for_legacy_runs():
    response = bt._run_metadata_response(_run_record())

    assert response.market is None
    assert response.universe is None
    assert response.timeframe is None
    assert response.timezone is None
    assert response.decision_source is None
    assert response.benchmark is None
    assert response.symbols is None
    assert response.native_currency is None
    assert response.reporting_currency is None
    assert response.native_initial_capital is None
    assert response.fx_pair is None
    assert response.fx_source is None
    assert response.fx_start_rate is None


@pytest.fixture(autouse=True)
def _reset_backtest_guards(monkeypatch):
    bt._backtest_rate_limiter.reset()
    bt.backtest_status.update({
        "running": False,
        "error": None,
        "runs_count": 0,
        "started_at": None,
        "progress_file": None,
        "live_run_id": None,
    })
    # Safety net: no test in this file may launch a real backtest thread.
    monkeypatch.setattr(bt, "run_backtest_background", lambda *a, **k: None)
    yield
    bt._backtest_rate_limiter.reset()


def _sess():
    return {"X-Session-Id": str(uuid.uuid4())}


def test_backtest_run_valid_request_ok():
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-07"},
        headers=_sess(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "session_id" in body


def test_backtest_run_targets_builtin_agent_session(client, monkeypatch):
    """Discord (and website) can pass agent_id so runs land on the agent card."""
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "Discord Card Bot", "agent_type": "builtin"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_session = created["session_id"]
    agent_id = created["agent"]["agent_id"]

    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "strategy_prompt": "buy low sell high",
            "agent_id": agent_id,
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == agent_session
    assert spy.calls == 1
    assert spy.last_kwargs["session_id"] == agent_session
    assert spy.last_kwargs["runtime_type"] == "pipeline"
    assert spy.last_kwargs["runtime_config"] == {}


def _stub_hosted_runtime_installed(monkeypatch):
    """Pretend the isolated upstream venv exists on this deployment.

    CI installs core requirements only, so the real check always reports the
    runtime as missing. Tests about *other* preconditions have to say which
    deployment they are describing.
    """
    monkeypatch.setattr(bt, "runtime_unavailable_reason", lambda: None)


def test_backtest_run_dispatches_ai_hedge_fund_runtime(client, monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    _stub_hosted_runtime_installed(monkeypatch)
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: "user-financial-datasets-test-key",
    )
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    cloned = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    )
    assert cloned.status_code == 200
    agent = cloned.json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "llm",
            "model": "claude-haiku-4.5",
            "pipeline": [{"label": "must be ignored"}],
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["runtime_type"] == "ai_hedge_fund"
    assert set(response.json()["ignored_fields"]) == {"model", "pipeline"}
    assert spy.calls == 1
    assert spy.last_kwargs["runtime_type"] == "ai_hedge_fund"
    assert spy.last_kwargs["runtime_config"]["analysts"]
    assert (
        spy.last_kwargs["financial_datasets_api_key"]
        == "user-financial-datasets-test-key"
    )
    assert spy.last_kwargs["model"] is None
    assert spy.last_kwargs["pipeline"] is None


def test_ai_hedge_fund_backtest_requires_owned_agent_credential(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    _stub_hosted_runtime_installed(monkeypatch)
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: None,
    )

    missing = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )
    assert missing.status_code == 422
    assert "Financial Datasets API key" in missing.text

    unauthorized = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert unauthorized.status_code == 403


def test_ai_hedge_fund_backtest_rejects_run_when_runtime_not_installed(
    client, monkeypatch
):
    """A missing isolated venv must fail at request time, not 30 minutes later.

    render.yaml is documentation rather than the deploy mechanism here, so a
    service without the venv is a live possibility. Without this the run is
    accepted, backgrounded, and dies on its first decision step.
    """
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-openrouter-test-key")
    monkeypatch.setattr(
        bt,
        "runtime_unavailable_reason",
        lambda: "AI Hedge Fund runtime is not installed; configure AI_HEDGE_FUND_PYTHON",
    )
    monkeypatch.setattr(
        bt.agent_credential_store,
        "get_secret",
        lambda agent_id, credential_name: "user-financial-datasets-test-key",
    )
    headers = {"X-Session-Id": str(uuid.uuid4())}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 503, response.text
    assert "AI_HEDGE_FUND_PYTHON" in response.text
    assert spy.calls == 0


def test_hosted_backtest_timeout_covers_every_decision_step():
    """The parent timeout must not be the binding constraint on a hosted run.

    A fixed 1800s cap over a month of trading days leaves ~85s per step while
    the runtime is configured for 300s, so the parent kills the child mid-run
    and discards every completed step.
    """
    step_seconds = bt.resolve_step_timeout_seconds()
    decision_days = bt._estimated_decision_days("2026-01-01", "2026-01-31")
    assert decision_days == 22

    hosted = bt._backtest_subprocess_timeout(
        "ai_hedge_fund", "2026-01-01", "2026-01-31"
    )
    assert hosted >= step_seconds * decision_days
    assert hosted > bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS

    # Pipeline runs keep their established budget exactly.
    assert (
        bt._backtest_subprocess_timeout("pipeline", "2026-01-01", "2026-01-31")
        == bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
    )


def test_hosted_backtest_timeout_is_capped_and_never_below_pipeline():
    """A long range is capped rather than pinning a worker thread forever."""
    capped = bt._backtest_subprocess_timeout(
        "ai_hedge_fund", "2020-01-01", "2030-01-01"
    )
    assert capped == bt.MAX_SUBPROCESS_TIMEOUT_SECONDS

    # Unparseable dates must not collapse the budget to the overhead constant.
    assert (
        bt._backtest_subprocess_timeout("ai_hedge_fund", "not-a-date", "also-bad")
        == bt.PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
    )


def test_ai_hedge_fund_requires_openrouter_not_direct_openai(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-authorize-hosted-runtime")
    owner = str(uuid.uuid4())
    headers = {"X-Session-Id": owner}
    agent = client.post(
        "/api/v1/agents/marketplace/ai-hedge-fund/clone",
        json={},
        headers=headers,
    ).json()["agent"]

    response = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "agent_id": agent["agent_id"],
        },
        headers=headers,
    )

    assert response.status_code == 503
    assert "platform-managed OpenRouter provider" in response.text


def test_backtest_run_forwards_selected_assets(client, monkeypatch):
    """UI Asset Universe must reach the background worker (not stay mocked/DJIA-only)."""
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    mag7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "assets": mag7,
        },
        headers=_sess(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["assets"] == mag7
    assert spy.calls == 1
    # By name, not position: universe/timeframe were inserted mid-signature
    # once, and an index-based assertion would have kept passing on the wrong
    # argument.
    assert spy.last_args == ()
    assert spy.last_kwargs["assets"] == mag7
    assert spy.last_kwargs["decision_source"] == "llm"


def test_backtest_run_rejects_bad_assets(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "assets": ["NOT A TICKER!!!"],
        },
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_external_agent_id(client):
    owner = str(uuid.uuid4())
    created = client.post(
        "/api/v1/agents",
        json={"name": "External Only", "agent_type": "external"},
        headers={"X-Session-Id": owner},
    ).json()
    agent_id = created["agent"]["agent_id"]

    resp = client.post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02", "agent_id": agent_id},
        headers=_sess(),
    )
    assert resp.status_code == 422


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize("model", [
    # Exactly the options the dashboard UI dropdown (app.html) offers. A pricing-
    # table allowlist previously 422'd gpt-5.2 / gpt-5-mini / deepseek-* / gemini-*,
    # breaking the UI's own model choices.
    "claude-haiku-4.5", "claude-sonnet-4.6", "claude-opus-4.7",
    "gpt-5.2", "gpt-5-mini", "deepseek-v4-flash", "deepseek-v4-pro",
    "gemini-3.5-flash", "gemini-2.5-pro",
    "openai/gpt-5.5", "google/gemini-3.1-pro-preview",
])
def test_backtest_run_accepts_frontend_model_options(model):
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02", "model": model},
        headers=_sess(),
    )
    assert resp.status_code == 200, (model, resp.text)


@pytest.mark.parametrize("model", [
    "claude-haiku-4.5", "claude-sonnet-4.6", "claude-opus-4.7",
    "gpt-5.2", "gpt-5-mini", "deepseek-v4-flash", "deepseek-v4-pro",
    "gemini-3.5-flash", "gemini-2.5-pro",
])
def test_ifind_llm_accepts_every_frontend_model(monkeypatch, model):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    monkeypatch.setattr(
        bt,
        "ensure_llm_client_available",
        object,
        raising=False,
    )

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "llm",
            "model": model,
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, (model, resp.text)
    assert resp.json()["decision_source"] == "llm"


def test_explicit_llm_requires_model_before_scheduling(monkeypatch):
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    monkeypatch.setattr(
        bt,
        "ensure_llm_client_available",
        object,
        raising=False,
    )

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "llm",
        },
        headers=_sess(),
    )

    assert resp.status_code == 422
    assert "model" in resp.text.lower()
    assert spy.calls == 0


@pytest.mark.parametrize("model", [
    "bad model with spaces", "x; rm -rf /", "a" * 100, "m\nnewline", "café",
])
def test_backtest_run_rejects_malformed_model(monkeypatch, model):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02", "model": model},
        headers=_sess(),
    )
    assert resp.status_code == 422, (model, resp.text)
    assert spy.calls == 0  # nothing scheduled


def test_backtest_run_rejects_oversized_prompt(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02",
              "strategy_prompt": "x" * 5000},
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_excessive_date_range(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2020-01-01", "end_date": "2026-01-01"},
        headers=_sess(),
    )
    assert resp.status_code == 422
    assert spy.calls == 0


def test_backtest_run_rejects_bad_date_format():
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "05/01/2026", "end_date": "2026-05-02"},
        headers=_sess(),
    )
    assert resp.status_code == 422


def test_backtest_status_includes_live_progress(tmp_path):
    progress_file = tmp_path / "progress.json"
    progress_file.write_text(json.dumps({
        "run_id": "agent_test",
        "step": 5,
        "total_steps": 100,
        "equity_curve": [{"timestamp": "2026-05-01T10:00:00", "equity": 100500, "cash": 50000, "positions_value": 50500}],
        "trades": [{
            "timestamp": "2026-05-01T10:00:00",
            "symbol": "AAPL",
            "side": "BUY",
            "quantity": 10,
            "price": 150.25,
            "value": 1502.5,
        }],
    }), encoding="utf-8")
    bt.backtest_status.update({
        "running": True,
        "error": None,
        "started_at": time.time(),
        "progress_file": str(progress_file),
        "live_run_id": "agent_test",
    })
    resp = TestClient(app).get("/backtest/status", headers=_sess())
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is True
    assert body["progress"]["step"] == 5
    assert body["progress"]["total_steps"] == 100
    assert len(body["progress"]["equity_curve"]) == 1
    assert len(body["progress"]["equity_curve"]) == 1
    assert len(body["progress"]["trades"]) == 1
    assert body["progress"]["trades"][0]["symbol"] == "AAPL"
    assert "step 5/100" in body["message"]


def test_get_run_trades_endpoint(client, monkeypatch):
    session_id = str(uuid.uuid4())
    run_id = "agent_test_trades"

    def fake_get_run_with_session(rid, sid):
        if rid == run_id and sid == session_id:
            return {"run_id": run_id, "agent_name": "Agent", "mode": "backtest"}
        return None

    def fake_get_trades(rid):
        if rid == run_id:
            return [{
                "timestamp": "2026-05-01T10:00:00",
                "symbol": "MSFT",
                "quantity": 5,
                "side": "BUY",
                "price": 380.5,
                "value": 1902.5,
            }]
        return []

    monkeypatch.setattr(bt.db, "get_run_with_session", fake_get_run_with_session)
    monkeypatch.setattr(bt.db, "get_trades", fake_get_trades)

    resp = client.get(f"/runs/{run_id}/trades", headers={"X-Session-Id": session_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["count"] == 1
    assert body["trades"][0]["symbol"] == "MSFT"


def test_backtest_run_rate_limited_per_client(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        bt, "_backtest_rate_limiter",
        FixedWindowRateLimiter(max_events=2, window_seconds=3600, clock=lambda: now[0]),
    )
    client = TestClient(app)
    headers = _sess()  # same session -> same rate key across the three calls
    body = {"start_date": "2026-05-01", "end_date": "2026-05-02"}
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 200
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 200
    assert client.post("/backtest/run", json=body, headers=headers).status_code == 429


def test_rule_based_still_validates_llm_only_fields_before_dropping(monkeypatch):
    """Dropping them before validation answered 200 to a malformed model."""
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "rule_based",
            "model": "x; rm -rf /",
        },
        headers=_sess(),
    )

    assert resp.status_code == 422
    assert "Invalid model id" in resp.text
    assert spy.calls == 0


def test_rule_based_reports_the_llm_fields_it_dropped(monkeypatch):
    """Dropping them is right; doing it invisibly is what hid the bad input."""
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    spy = _Spy()
    monkeypatch.setattr(bt, "run_backtest_background", spy)

    resp = TestClient(app).post(
        "/backtest/run",
        json={
            "data_source": "ifind_ashare",
            "universe": "a_share_demo_6",
            "timeframe": "60m",
            "decision_source": "rule_based",
            "model": "gpt-5.2",
            "strategy_prompt": "buy low",
        },
        headers=_sess(),
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["decision_source"] == "rule_based"
    assert sorted(resp.json()["ignored_fields"]) == ["model", "strategy_prompt"]


def test_llm_run_reports_no_ignored_fields(monkeypatch):
    resp = TestClient(app).post(
        "/backtest/run",
        json={"start_date": "2026-05-01", "end_date": "2026-05-02"},
        headers=_sess(),
    )

    assert resp.status_code == 200, resp.text
    assert "ignored_fields" not in resp.json()


def test_subprocess_log_dump_is_redacted_but_not_truncated(monkeypatch):
    """print() is the only prod log channel; trimming drops every run's head."""
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "super-secret-token")
    head = "UNIVERSE-LINE-AT-THE-VERY-TOP"
    noise = "x" * 8000
    log = f"{head}\n{noise}\naccess_token=super-secret-token\n"

    redacted = bt._redact_credentials(log)

    assert head in redacted
    assert len(redacted) > 8000
    assert "super-secret-token" not in redacted
    assert "[REDACTED]" in redacted


def test_subprocess_log_redacts_user_financial_datasets_credential():
    credential = "user-financial-datasets-plaintext-canary"

    redacted = bt._redact_credentials(
        f"upstream failed with key={credential}", credential
    )

    assert credential not in redacted
    assert "[REDACTED]" in redacted


def test_background_error_redacts_user_financial_datasets_credential():
    credential = "user-financial-datasets-test-key"
    summary = bt._sanitize_backtest_error(
        f"runtime rejected credential={credential}",
        extra_secret=credential,
    )

    assert credential not in summary
    assert "[REDACTED]" in summary


def test_error_summary_stays_bounded(monkeypatch):
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "super-secret-token")

    summary = bt._sanitize_backtest_error("y" * 5000 + " super-secret-token", 500)

    assert len(summary) == 500
    assert "super-secret-token" not in summary
