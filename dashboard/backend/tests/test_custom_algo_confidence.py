"""The Trading Algo loop reads confidence with the backtest gate's rule.

``backtest_custom_algo`` imports ``backtest_hourly_agent`` as a top-level module,
so it is exercised in its own run context (``dashboard/scripts`` as the working
directory), the way ``test_engine_move`` checks its subclassing.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"

_CODE = r'''
import json
from datetime import datetime

import backtest_custom_algo as bca

bca.HAS_ANTHROPIC = True


class _Reply:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class _Client:
    def __init__(self, actions):
        reply = json.dumps({"actions": actions})

        class _Messages:
            @staticmethod
            def create(**kwargs):
                return _Reply(reply)

        self.messages = _Messages()


STATE = {
    "timestamp": datetime(2026, 1, 2, 10),
    "cash": 100000,
    "total_equity": 100000,
    "positions": [],
    "market_signals": {
        "AAPL": {"price": 100.0, "rsi": 25.0, "sma20": 110.0, "sma50": 120.0},
    },
}


def shares(**fields):
    manager = bca.CustomAlgoPortfolioManager(100000, {})
    action = {"symbol": "AAPL", "action": "buy", "reasoning": "x"}
    action.update(fields)
    decision = manager.make_trading_decision_with_custom_algo(STATE, _Client([action]))
    return [item["shares"] for item in decision["actions"]]


print(json.dumps({
    "unsized_25": shares(confidence=25),
    "unsized_true": shares(confidence=True),
    "unsized_text": shares(confidence="high"),
    "sized_25": shares(confidence=25, position_size=5),
    "unsized_half": shares(confidence=0.5),
    "sized_low": shares(confidence=0.1, position_size=5),
}))
'''


def _run():
    with tempfile.TemporaryDirectory(prefix="atl_algo_") as tmp:
        env = {**os.environ, "DATABASE_PATH": os.path.join(tmp, "t.db")}
        proc = subprocess.run(
            [sys.executable, "-c", _CODE],
            cwd=str(_SCRIPTS_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_trading_algo_reads_confidence_like_the_backtest_gate():
    """25 used to read as 2500% and buy 500 shares (half the account) where a
    0.25 would not have traded. Unsized buys with no 0-1 confidence are now
    skipped; a buy that names its size trades it; the 0.3 floor still holds
    for a confidence that can be read."""
    assert _run() == {
        "unsized_25": [],
        "unsized_true": [],
        "unsized_text": [],
        "sized_25": [5],
        "unsized_half": [10],
        "sized_low": [],
    }
