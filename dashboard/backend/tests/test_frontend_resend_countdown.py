"""The resend button's countdown label, executed under node.

Seconds under a minute, whole minutes under an hour, whole hours above (the
daily 429 carries a Retry-After of up to 86400, and "Resend code (86400s)" is
not a label anyone should read) -- rounded to the NEAREST unit, not ceiled:
a ceiled "2 min" at 61 s that became "60s" a second later doubled the
apparent wait at the exact moment a waiting user was watching the button.
The label never reads "0", and it never goes UP as the countdown ticks down.
"""

import json
import re
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


def test_a_minute_and_up_counts_whole_minutes_rounded_to_nearest():
    assert _eval("formatResendCountdown(60)") == "1 min"
    assert _eval("formatResendCountdown(61)") == "1 min"
    assert _eval("formatResendCountdown(90)") == "2 min"
    assert _eval("formatResendCountdown(3599)") == "60 min"


def test_an_hour_and_up_counts_whole_hours():
    assert _eval("formatResendCountdown(3600)") == "1 h"
    assert _eval("formatResendCountdown(86400 - 7200)") == "22 h"


def test_never_shows_zero_while_still_counting():
    assert _eval("formatResendCountdown(0)") == "1s"


def test_label_never_rises_as_the_countdown_falls():
    labels = _eval(
        "Array.from({length: 7300}, (_, i) => formatResendCountdown(7300 - i))"
    )
    unit = {"s": 1, "min": 60, "h": 3600}

    def seconds(label):
        match = re.fullmatch(r"(\d+) ?(s|min|h)", label)
        assert match, label
        return int(match.group(1)) * unit[match.group(2)]

    values = [seconds(label) for label in labels]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
