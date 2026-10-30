"""HTTP-level tests for the v2 runs endpoints.

These drive the real FastAPI stack (auth dependency -> scope -> session
ownership -> error envelope -> rate limit) by registering an offline
FakeBackend under a run_id whose session_id matches a freshly-registered
agent. This avoids Alpaca/network while still exercising the wire contract.
"""

import time

from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
from dashboard.backend.api.v2 import rate_limit
from dashboard.backend.domain.agents.repository import agent_store
from dashboard.backend.app import app
from dashboard.backend.tests._v2_fakes import FakeBackend

client = TestClient(app)


def _agent(name):
    """Register an agent; return (api_key, session_id, agent_id)."""
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


def _register_run(run_id, session_id, total_steps=2):
    backend = FakeBackend(run_id=run_id, total_steps=total_steps, session_id=session_id)
    runs_mod.register_run(run_id, backend, session_id)
    return backend


# -- security boundary: cross-agent access ---------------------------------

def test_cross_agent_cannot_read_run_returns_404_envelope():
    key_a, sid_a, _ = _agent("owner-a")
    key_b, _, _ = _agent("intruder-b")
    _register_run("run_iso_http", sid_a)
    r = client.get("/api/v2/runs/run_iso_http/context", headers={"X-API-Key": key_b})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "run_not_found"


# -- scope enforcement at the HTTP layer -----------------------------------

def test_missing_scope_returns_403_envelope():
    key, _, agent_id = _agent("scoped-agent")
    conn = agent_store._get_connection()
    conn.execute("UPDATE external_agents SET scopes=? WHERE agent_id=?",
                 ("runs:read", agent_id))
    conn.commit()
    # context requires context:read, which we just removed
    r = client.get("/api/v2/runs/any_run/context", headers={"X-API-Key": key})
    assert r.status_code == 403
    body = r.json()
    assert body["error"]["code"] == "forbidden_scope"
    assert body["error"]["details"]["required"] == "context:read"


# -- rate limiting at the HTTP layer ---------------------------------------

def test_rate_limit_returns_429_envelope_with_retry_after():
    key, sid, agent_id = _agent("rl-agent")
    _register_run("run_rl_http", sid)
    # Drain the bucket so the next call is denied.
    # Keep the bucket empty for the whole request even on a slow CI runner;
    # with 120 tokens/minute, a real-time timestamp could refill one token
    # during TestClient startup and make this assertion flaky.
    rate_limit.limiter._buckets[agent_id] = (0.0, time.monotonic() + 60.0)
    r = client.get("/api/v2/runs/run_rl_http/context", headers={"X-API-Key": key})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "rate_limited"
    assert "Retry-After" in r.headers


# -- error envelope: result before completion ------------------------------

def test_result_before_completion_returns_409_envelope():
    key, sid, _ = _agent("res-agent")
    _register_run("run_res_http", sid)
    r = client.get("/api/v2/runs/run_res_http/result", headers={"X-API-Key": key})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "invalid_status"


# -- M2: request-body validation flows through the uniform envelope --------

def test_invalid_create_body_returns_validation_failed_envelope():
    key, _, _ = _agent("val-agent")
    r = client.post("/api/v2/runs", headers={"X-API-Key": key},
                    json={"mode": "live", "start_date": "2026-01-01",
                          "end_date": "2026-01-02"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_failed"
    assert "details" in body["error"]


# -- M1: cancel transitions context to a 'closed' envelope -----------------

def test_cancel_then_context_reports_closed():
    key, sid, _ = _agent("cancel-agent")
    _register_run("run_cancel_http", sid)
    h = {"X-API-Key": key}
    c = client.post("/api/v2/runs/run_cancel_http/cancel", headers=h)
    assert c.status_code == 200
    assert c.json()["status"] == "closed"
    ctx = client.get("/api/v2/runs/run_cancel_http/context", headers=h)
    assert ctx.status_code == 200
    assert ctx.json()["status"] == "closed"


# -- partial execution (spec §5.3): invalid actions dropped with reasons ----

def test_mixed_payload_executes_valid_and_rejects_invalid():
    key, sid, _ = _agent("partial-agent")
    _register_run("run_partial_http", sid)
    h = {"X-API-Key": key}
    r = client.post("/api/v2/runs/run_partial_http/decisions", headers=h, json={
        "idempotency_key": "mix-1",
        "actions": [
            {"action": "buy", "symbol": "AAPL", "confidence": 0.7,
             "reasoning": "valid momentum play", "position_size": 3},
            {"action": "buy", "symbol": "ZZZ", "confidence": 0.7,
             "reasoning": "not in the universe", "position_size": 3},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    assert any(e["symbol"] == "AAPL" for e in body["executed"])
    assert any(rj["reason"] == "universe_violation" for rj in body["rejected"])


def test_lowercase_symbol_executes_as_its_canonical_ticker():
    # v1 accepts "aapl"; v2 used to hold the step as a universe_violation.
    key, sid, _ = _agent("case-agent")
    _register_run("run_case_http", sid)
    r = client.post("/api/v2/runs/run_case_http/decisions",
                    headers={"X-API-Key": key}, json={
        "idempotency_key": "case-1",
        "actions": [{"action": "buy", "symbol": " aapl ", "confidence": 0.7,
                     "reasoning": "valid momentum play", "position_size": 3}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["rejected"] == []
    assert body["decision_source"] == "external_agent"
    assert [e["symbol"] for e in body["executed"]] == ["AAPL"]


def test_lowercase_symbol_trades_through_the_real_backtest_engine(monkeypatch):
    """FakeBackend echoes whatever the boundary hands it, so the test above
    cannot see the engine. This one runs the real chain: BacktestBackend ->
    ExternalBacktestSession.submit_decisions -> parse_actions_payload ->
    actions_to_executable, whose price lookup is keyed by the symbol."""
    import pandas as pd

    import dashboard.backend.execution.backtest_backend as bb_mod

    symbols = ["AAPL", "MSFT", "JPM"]
    idx = pd.date_range("2026-04-15 13:30", periods=40, freq="h", tz="UTC")
    bars = {
        sym: pd.DataFrame({
            "open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i,
            "close": 100.0 + i, "volume": 1000,
        }, index=idx)
        for i, sym in enumerate(symbols)
    }

    class _Loader:
        def fetch_bars(self, syms, start, end):
            return bars

    monkeypatch.setattr(bb_mod.ext, "AlpacaDataLoader", lambda: _Loader())
    monkeypatch.setattr(bb_mod, "DJIA_30", symbols, raising=False)

    key, sid, _ = _agent("real-engine-case-agent")
    backend = bb_mod.BacktestBackend(
        run_id="run_case_engine", session_id=sid, agent_name="a",
        model_name="m", start_date="2026-04-15", end_date="2026-04-16",
    )
    backend.load_blocking()
    runs_mod.register_run("run_case_engine", backend, sid)

    r = client.post("/api/v2/runs/run_case_engine/decisions",
                    headers={"X-API-Key": key}, json={
        "idempotency_key": "case-engine-1",
        "actions": [{"action": "buy", "symbol": " aapl ", "confidence": 0.7,
                     "reasoning": "valid momentum play", "position_size": 3}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rejected"] == []
    assert [(e["symbol"], e["shares"]) for e in body["executed"]] == [("AAPL", 3)]
    assert backend.session.manager.positions.get("AAPL") == 3


def test_all_invalid_payload_auto_holds():
    key, sid, _ = _agent("allbad-agent")
    _register_run("run_allbad_http", sid)
    h = {"X-API-Key": key}
    r = client.post("/api/v2/runs/run_allbad_http/decisions", headers=h, json={
        "idempotency_key": "bad-1",
        "actions": [{"action": "buy", "symbol": "ZZZ", "confidence": 0.7,
                     "reasoning": "not in the universe", "position_size": 3}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["executed"] == []
    assert body["decision_source"] == "validation_hold"
    assert any(rj["reason"] == "universe_violation" for rj in body["rejected"])


# -- N1: decisions log reads through the backend interface ------------------

def test_decisions_log_returns_recorded_decisions():
    key, sid, _ = _agent("log-agent")
    _register_run("run_log_http", sid)
    h = {"X-API-Key": key}
    sub = client.post("/api/v2/runs/run_log_http/decisions", headers=h, json={
        "idempotency_key": "k1",
        "actions": [{"action": "buy", "symbol": "AAPL", "confidence": 0.7,
                     "reasoning": "momentum looks strong", "position_size": 3}],
    })
    assert sub.status_code == 200
    r = client.get("/api/v2/runs/run_log_http/decisions", headers=h)
    assert r.status_code == 200
    decisions = r.json()["decisions"]
    assert isinstance(decisions, list)
    assert len(decisions) >= 1
