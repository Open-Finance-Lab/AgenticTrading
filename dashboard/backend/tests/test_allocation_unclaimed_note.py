"""``portfolio.unclaimed_allocated`` must have an observer.

The backend splits the sleeve total into ``owned`` and ``visible`` and reports
the gap as ``unclaimed_allocated`` -- capital allocated to agents this browser
created before sign-in, which ``POST /api/v1/agents/claim-account`` links to
the account at login.

Reporting it was only half the job. The claim is fired-and-forgotten from the
client, so when it fails the account's spendable cash silently drops by the
guest sleeve and the next allocation comes back 400 "Insufficient unallocated
cash" -- while the one field that explains why was rendered by nothing, logged
by nothing, and alerted on by nothing. A number nobody reads cannot tell
"nothing to claim" apart from "the claim failed": that is precisely the
fail-closed-is-not-fail-visible shape CLAUDE.md names.

These pin the two observers: the panel states the amount, and a failed claim
logs at error level rather than as a "skipped" warning.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import fn_body

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_PORTFOLIO_JS = (_FRONTEND / "js" / "portfolio.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")

_NODE_MISSING = shutil.which("node") is None


def _run_node(script: str):
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _note_harness() -> str:
    return "\n".join(
        [
            fn_body("function pfMoney", _PORTFOLIO_JS),
            fn_body("function allocationUnclaimedNoteHtml", _PORTFOLIO_JS),
        ]
    )


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_nothing_unclaimed_says_nothing():
    """The steady state. An account with no guest agents must see no caveat."""
    script = _note_harness() + """
const quiet = [
  { unclaimed_allocated: 0 },
  { unclaimed_allocated: null },
  { unclaimed_allocated: -1 },
  {},
  null,
];
console.log(JSON.stringify(quiet.map(allocationUnclaimedNoteHtml)));
"""
    assert _run_node(script) == ["", "", "", "", ""]


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_an_unclaimed_sleeve_is_named_with_its_amount():
    script = _note_harness() + """
console.log(JSON.stringify(
  allocationUnclaimedNoteHtml({ unclaimed_allocated: 9000 })
));
"""
    note = _run_node(script)
    assert "$9,000.00" in note
    assert "allocation-legend-hint--unclaimed" in note
    # It must say what the user can do about it, not merely that it exists.
    assert "reload" in note.lower()


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_a_sub_cent_remainder_is_not_worth_a_caveat():
    """Float noise on a derived figure must not raise a banner over $0.00."""
    script = _note_harness() + """
console.log(JSON.stringify(allocationUnclaimedNoteHtml({ unclaimed_allocated: 1e-9 })));
"""
    assert _run_node(script) == ""


def test_legend_renders_the_unclaimed_note():
    assert "allocationUnclaimedNoteHtml(" in fn_body(
        "function renderAllocationLegend", _PORTFOLIO_JS
    )


def test_live_portfolio_keeps_the_field_the_note_reads():
    """The cached summary drops every field it does not name, so this one has
    to be named or the note goes quiet on the instant-paint path."""
    body = fn_body("function renderPortfolioFromLive", _PORTFOLIO_JS)
    assert "unclaimed_allocated" in body


def test_unclaimed_note_style_exists():
    assert ".allocation-legend-hint--unclaimed {" in _STYLES


def test_failed_claim_logs_at_error_level():
    """A swallowed claim is the drift boundary: the sleeve stays unclaimed and
    the account's cash silently drops. A warning saying "skipped" read as a
    no-op, which is the one thing it is not."""
    body = fn_body("function claimAgentsForUser")
    assert "console.error(" in body
    assert "console.warn(" not in body
