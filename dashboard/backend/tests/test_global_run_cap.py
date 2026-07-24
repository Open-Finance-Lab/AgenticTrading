"""Global active-run backstop cap (T3): 429 too_many_active_runs_global with
Retry-After on BOTH create surfaces; 0 disables; per-agent cap unchanged."""

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.v2.runs as runs_mod
import dashboard.backend.domain.runs.service as run_service
from dashboard.backend.app import app
from dashboard.backend.domain.runs.repository import run_store

client = TestClient(app)


def _agent(name):
    r = client.post("/api/v2/agents", json={"name": name}).json()
    return r["api_key"], r["session_id"], r["agent_id"]


class _StubBackend:
    loop = "lockstep"
    news_sentiment_source = None

    def __init__(self, **kwargs):
        self._active = True

    def start_background_load(self):
        pass

    def is_active(self):
        return self._active

    def status(self):
        return {"status": "waiting_decision"}

    def advance(self):
        pass

    def cancel(self):
        self._active = False


def _seed_active_run(agent_id, sid, n=1):
    for i in range(n):
        run_store.create_run(
            agent_id=agent_id, agent_version_id=None, session_id=sid,
            environment_id="us-equity-hourly-v1", environment_type="backtest",
            config={}, backtest_id=f"bt_cap_{agent_id}_{i}", status="running",
        )


def test_count_active_runs_total_spans_agents():
    _, sid_a, aid_a = _agent("cap-count-a")
    _, sid_b, aid_b = _agent("cap-count-b")
    before = run_store.count_active_runs_total()
    _seed_active_run(aid_a, sid_a)
    _seed_active_run(aid_b, sid_b)
    assert run_store.count_active_runs_total() == before + 2


def test_v2_global_cap_rejects_with_retry_after(monkeypatch):
    key_b, _, _ = _agent("cap-v2-victim")
    _, sid_a, aid_a = _agent("cap-v2-filler")
    _seed_active_run(aid_a, sid_a)
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key_b},
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["error"]["code"] == "too_many_active_runs_global"
    assert resp.json()["error"]["retryable"] is True
    assert resp.headers.get("Retry-After") == "30"


def test_v1_global_cap_rejects_with_retry_after(monkeypatch):
    _, sid_a, aid_a = _agent("cap-v1-filler")
    _, sid_b, aid_b = _agent("cap-v1-victim")
    _seed_active_run(aid_a, sid_a)
    monkeypatch.setattr(run_service, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    with pytest.raises(run_service.ProtocolError) as ei:
        run_service.create_run(
            agent={"agent_id": aid_b, "session_id": sid_b, "name": "x"},
            agent_version=None,
            environment_id="us-equity-hourly-v1",
            config={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        )
    assert ei.value.code == "too_many_active_runs_global"
    assert ei.value.status_code == 429
    assert ei.value.headers == {"Retry-After": "30"}


def test_zero_disables_the_global_cap(monkeypatch):
    key, _, _ = _agent("cap-disabled")
    _, sid_a, aid_a = _agent("cap-disabled-filler")
    _seed_active_run(aid_a, sid_a, n=3)
    monkeypatch.setattr(runs_mod, "MAX_ACTIVE_RUNS_GLOBAL", 0)
    monkeypatch.setattr(runs_mod, "BacktestBackend", _StubBackend)
    resp = client.post(
        "/api/v2/runs",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 200, resp.text


def test_schema_endpoint_reports_the_new_code():
    resp = client.get("/api/v2/schema")
    assert "too_many_active_runs_global" in resp.json()["error_codes"]


def test_v1_global_cap_over_http_carries_retry_after(monkeypatch):
    """End-to-end v1: _handle_protocol_error must forward ProtocolError.headers
    onto the HTTPException so the 429 actually carries Retry-After."""
    key, _, _ = _agent("cap-v1-http")          # v2-registered key works on v1 too
    _, sid_f, aid_f = _agent("cap-v1-http-filler")
    _seed_active_run(aid_f, sid_f)
    monkeypatch.setattr(run_service, "MAX_ACTIVE_RUNS_GLOBAL",
                        run_store.count_active_runs_total())
    resp = client.post(
        "/api/v1/runs",
        json={"environment": {"type": "backtest",
                              "environment_id": "us-equity-hourly-v1"},
              "config": {"start_date": "2026-04-15", "end_date": "2026-04-16"}},
        headers={"X-API-Key": key},
    )
    assert resp.status_code == 429, resp.text
    assert resp.json()["detail"]["error"]["code"] == "too_many_active_runs_global"
    assert resp.headers.get("Retry-After") == "30"
