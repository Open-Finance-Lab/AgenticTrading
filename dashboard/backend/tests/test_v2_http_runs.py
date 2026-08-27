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
