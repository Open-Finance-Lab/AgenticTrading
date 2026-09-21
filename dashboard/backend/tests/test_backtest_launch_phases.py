"""The parent tells the child when it was launched.

The child's first progress write can only account for the gap before it
(imports, store startup) if it knows when the parent spawned it. That clock
rides argv beside --run-id, so a run's `starting` phase is measured rather
than missing.
"""
import os
import subprocess
import sys
import time

import dashboard.backend.api.routers.backtests as backtests
from dashboard.backend.tests._fake_child import FakeChild

REAL_RUN_BACKTEST_BACKGROUND = backtests.run_backtest_background


def _launch(monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        return FakeChild()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(backtests.db, "get_runs_by_mode", lambda mode: [])
    before = time.time()
    REAL_RUN_BACKTEST_BACKGROUND(
        "2026-04-01",
        "2026-04-08",
        "session-id",
        decision_source="rule_based",
    )
    captured["before"] = before
    captured["after"] = time.time()
    return captured


def test_child_argv_carries_the_launch_time(monkeypatch):
    captured = _launch(monkeypatch)
    command = captured["command"]
    launched_at = float(command[command.index("--launched-at") + 1])
    assert captured["before"] - 1 <= launched_at <= captured["after"] + 1


def test_script_accepts_launched_at(tmp_path):
    result = subprocess.run(
        [sys.executable, "dashboard/scripts/backtest_hourly_agent.py", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_PATH": str(tmp_path / "backtest.db")},
    )
    assert result.returncode == 0, result.stderr
    assert "--launched-at" in result.stdout
