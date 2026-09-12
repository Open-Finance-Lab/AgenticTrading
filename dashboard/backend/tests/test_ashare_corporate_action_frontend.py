"""The ex-rights row on a results panel, and why it is a row at all.

Issue #346. A window crossing a 除权除息 date is refused by default; the
override runs it and the run records the dates. This is the half a reader
actually sees -- the curve still charts the drop as a loss, and this row is
what stops them taking that number at face value.

The label is executed under node rather than pattern-matched: "drop is not a
real loss" is the entire reason the row exists, and a string check cannot tell
whether it survives a truncation that keeps the symbols.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    fn_body,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _format(gaps: list[dict]) -> str:
    script = "\n".join(
        [
            fn_body("function formatCorporateActionGaps("),
            "console.log(JSON.stringify(formatCorporateActionGaps("
            f"{json.dumps(gaps)})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_one_gap_names_the_symbol_the_date_and_the_caveat():
    label = _format([{"symbol": "600519.SH", "date": "2025-09-01"}])
    assert "600519.SH" in label
    assert "2025-09-01" in label
    assert "not a real loss" in label


def test_many_gaps_are_truncated_without_losing_the_caveat():
    """The count may be elided; the reason the row exists may not.

    Truncating to three symbols and dropping the clause would leave a row that
    reads as a neutral list of dates -- which is the silent version this issue
    is about, restated in the UI.
    """
    label = _format(
        [{"symbol": f"S{n}", "date": f"2025-09-0{n}"} for n in range(1, 6)]
    )
    assert label.count("·") == 3
    assert "+2 more" in label
    assert "not a real loss" in label


def test_the_row_exists_and_is_hidden_by_default():
    """Hidden in the markup, so a run that crossed nothing shows nothing.

    Rendering a reassuring "None" on every A-share run is the other way to make
    this invisible: a row that is always there stops being read.
    """
    assert 'id="backtestConfigCorporateActionsRow" hidden' in APP_HTML
    assert 'id="backtestConfigCorporateActions"' in APP_HTML


def test_an_absent_gap_list_hides_the_row_rather_than_defaulting_it():
    """`corporate_action_gaps` is omitted from metadata when empty.

    So the client must read an absence as "nothing crossed" *and* hide the row.
    Asserted on the Array.isArray narrowing plus the hide, because a `?? []`
    without the type check would accept a non-array and throw in `.slice`.
    """
    body = strip_comments(
        APP_JS[APP_JS.index("const corporateActionGaps") :][:600]
    )
    assert "Array.isArray(marketRuleProfile?.corporate_action_gaps)" in body
    assert "corporateActionsRow.hidden = corporateActionGaps.length === 0" in body
