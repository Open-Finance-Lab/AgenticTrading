"""ETA and staleness formatting for a running backtest.

Both are honesty-constrained rather than precision-constrained:

* an ETA derived from two or three steps is wild, and a number that visibly
  jumps reads as broken -- so it is suppressed early and coarse thereafter;
* a stale progress file means the numbers are old, NOT that the run is stuck.
  An LLM pipeline step can legitimately take minutes. Claiming "stuck" would be
  the same class of error as the fabricated Performance Drivers card.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import fn_body, js_const

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _eval(expr: str) -> object:
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            fn_body("function formatBacktestEta("),
            fn_body("function formatProgressStaleness("),
            f"console.log(JSON.stringify({expr}));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_staleness_threshold_is_two_minutes():
    """Pins the shipped constant, since every staleness case below is scaled to
    it. Lowering it would make the UI cry wolf on ordinary model latency."""
    assert js_const("BACKTEST_STALE_SECONDS") == "const BACKTEST_STALE_SECONDS = 120;"


def test_eta_is_suppressed_for_the_first_two_steps():
    assert _eval("formatBacktestEta(4, 1, 240)") is None
    assert _eval("formatBacktestEta(8, 2, 240)") is None


def test_eta_appears_from_step_three():
    # 30s for 3 of 243 steps -> 10s/step -> 2400s remaining -> "~40m left"
    assert _eval("formatBacktestEta(30, 3, 243)") == "~40m left"


def test_eta_is_coarse_under_a_minute():
    # 100s for 100 of 130 steps -> 1s/step -> 30s remaining
    assert _eval("formatBacktestEta(100, 100, 130)") == "<1m left"


def test_eta_rounds_to_whole_minutes():
    # 125s for 25 of 50 steps -> 5s/step -> 125s remaining -> ~2m
    assert _eval("formatBacktestEta(125, 25, 50)") == "~2m left"


def test_eta_is_null_without_totals():
    assert _eval("formatBacktestEta(60, 10, 0)") is None
    assert _eval("formatBacktestEta(60, 10, null)") is None
    assert _eval("formatBacktestEta(60, null, 240)") is None


def test_eta_is_null_on_the_final_step():
    """No remaining work to estimate; the completion path takes over."""
    assert _eval("formatBacktestEta(600, 240, 240)") is None


def test_staleness_is_silent_below_the_threshold():
    assert _eval("formatProgressStaleness(0)") is None
    assert _eval("formatProgressStaleness(119)") is None


def test_staleness_reports_the_actual_gap_not_the_threshold():
    """A message frozen at '2m' while the real gap grows to ten is worse than
    no message -- it actively misinforms."""
    assert "2m" in _eval("formatProgressStaleness(130)")
    assert "9m" in _eval("formatProgressStaleness(560)")


def test_staleness_wording_does_not_claim_the_run_is_stuck():
    message = _eval("formatProgressStaleness(300)")
    assert "stuck" not in message.lower()
    assert "fail" not in message.lower()
