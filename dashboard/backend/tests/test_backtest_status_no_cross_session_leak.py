"""/backtest/status must never answer one visitor with another's running run."""

import time
import uuid

from fastapi.testclient import TestClient

import dashboard.backend.api.routers.backtests as bt
from dashboard.backend.app import app


def test_stranger_does_not_see_the_mirrored_run(monkeypatch):
    owner = str(uuid.uuid4())
    monkeypatch.setattr(bt, "backtest_session_id", owner)
    monkeypatch.setitem(bt.backtest_status, "running", True)
    monkeypatch.setitem(bt.backtest_status, "started_at", time.time())
    monkeypatch.setitem(bt.backtest_status, "live_run_id", "agent_victim_live")

    client = TestClient(app)
    stranger = client.get("/backtest/status", headers={"X-Session-Id": str(uuid.uuid4())}).json()
    assert stranger["running"] is False
    assert owner not in str(stranger)
    assert "agent_victim_live" not in str(stranger)

    own = client.get("/backtest/status", headers={"X-Session-Id": owner}).json()
    assert own["running"] is True
    assert own["live_run_id"] == "agent_victim_live"
