"""The resend button's countdown label, executed under node.

Seconds up to a minute, whole minutes above it: the hourly 429 carries a
Retry-After of up to 3600, and "Resend code (3600s)" is not a label anyone
should read. Ceil, never floor -- a label that reads "0 min" while the button
is still disabled contradicts itself.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import fn_body

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _eval(expr: str) -> object:
    script = "\n".join(
        [fn_body("function formatResendCountdown("), f"console.log(JSON.stringify({expr}));"]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_under_a_minute_counts_seconds():
    assert _eval("formatResendCountdown(59)") == "59s"
    assert _eval("formatResendCountdown(1)") == "1s"


def test_above_a_minute_counts_whole_minutes_rounded_up():
    assert _eval("formatResendCountdown(60)") == "60s"
    assert _eval("formatResendCountdown(61)") == "2 min"
    assert _eval("formatResendCountdown(3600)") == "60 min"


def test_never_shows_zero_while_still_counting():
    assert _eval("formatResendCountdown(0)") == "1s"
