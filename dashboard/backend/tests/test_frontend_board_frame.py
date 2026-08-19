"""Guards for the shared nof1-derived board frame (2026-08-19 spec §4).

One visual contract, two vanilla Chart.js implementations plus a Recharts one.
The duplication is forced by the stacks and accepted; leaving the *numbers*
unguarded is not, which is what test_the_two_surfaces_agree_on_the_numbers_that
_must_agree already establishes for this pair.

The geometry is exercised under node against the SHIPPING source, extracted by
name -- so a rename or deletion reddens these instead of leaving them passing
against a copy that no longer runs. Same harness shape as
test_frontend_leaderboard_hover.py.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_LEADERBOARD_JS = _FRONTEND / "js" / "leaderboard.js"
_HOME_JS = _FRONTEND / "home-page.js"
_SRC = _LEADERBOARD_JS.read_text(encoding="utf-8")
_HOME_SRC = _HOME_JS.read_text(encoding="utf-8")


def _extract_function(name: str) -> str:
    """The source of ``function <name>(...) { ... }``, brace-matched."""
    marker = f"function {name}("
    start = _SRC.index(marker)
    depth = 0
    index = _SRC.index("{", _SRC.index(")", start))
    while True:
        if _SRC[index] == "{":
            depth += 1
        elif _SRC[index] == "}":
            depth -= 1
            if depth == 0:
                return _SRC[start : index + 1]
        index += 1


def _board_constants() -> str:
    """Every ``const BOARD_* = <literal>;`` line, in source order.

    Extracted rather than restated: a number that only exists in this file is a
    number the guard cannot be wrong about.
    """
    lines = [
        ln
        for ln in _SRC.splitlines()
        if re.match(r"^const BOARD_[A-Z_]+ = .+;$", ln)
    ]
    assert lines, "no BOARD_* constants found -- the frame was renamed or deleted"
    return "\n".join(lines)


def _run_node(script: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [
            _board_constants(),
            _extract_function("boardFrameLayout"),
            _extract_function("boardLabelBlockWidth"),
            _extract_function("boardPillTextColor"),
            # Stub 2d context: every glyph is 6px wide. Enough to exercise the
            # width arithmetic without a canvas, and deliberately NOT a real
            # measurement -- the point of measuring at runtime is that no test
            # has to know the font metrics.
            """
function makeChart(width, height) {
  return {
    width,
    height,
    ctx: {
      save() {}, restore() {}, font: '',
      measureText(text) { return { width: String(text).length * 6 }; },
    },
  };
}
function makeLabels(n, name, value) {
  return Array.from({ length: n }, (_, i) => ({
    i, name: name || 'Model ' + i, value: value || '+1.00%',
  }));
}
""",
            script,
        ]
    )
    proc = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_gutter_is_two_fifths_of_a_wide_chart():
    """Spec §4.1: plot 60%, gutter 40%, as a FRACTION of measured width so the
    ratio survives every breakpoint rather than holding at one design size."""
    result = _run_node(
        """
const frame = boardFrameLayout(makeChart(1200, 420), makeLabels(9), 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels }));
"""
    )
    assert result["gutter"] == pytest.approx(480.0)
    assert result["draw"] is True


def test_a_long_label_raises_the_gutter_above_the_fraction():
    """The 40% is a target, not a ceiling: at middling widths it clips, so the
    measured label block is a floor under it. Same width for both, so the only
    variable is the label."""
    result = _run_node(
        """
const short = boardFrameLayout(makeChart(400, 420), makeLabels(3, 'AI', '+1%'), 0.4);
const long = boardFrameLayout(
  makeChart(400, 420), makeLabels(3, 'DeepSeek V4 Pro', '-12.34%'), 0.4);
console.log(JSON.stringify({ short: short.gutter, long: long.gutter }));
"""
    )
    assert result["short"] == pytest.approx(160.0), "40% of 400 clears the short label"
    assert result["long"] > 160.0, "the measured block must be able to push past 40%"


def test_a_chart_too_narrow_for_its_labels_drops_them_rather_than_clipping():
    """A 390px phone card cannot carry `DeepSeek V4 Pro -12.34%`. Clipping is the
    failure this repo keeps re-learning (the chip strip cut four of five names
    with no scrollbar and nothing failing), so the frame gives the space back and
    draws the arrow alone. Both surfaces keep a complete key elsewhere."""
    result = _run_node(
        """
const frame = boardFrameLayout(
  makeChart(300, 420), makeLabels(5, 'DeepSeek V4 Pro', '-12.34%'), 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels }));
"""
    )
    assert result["draw"] is False
    assert result["gutter"] == pytest.approx(18.0), "arrow padding only"


def test_a_chart_too_short_to_stack_its_labels_drops_them_too():
    """Screen 0's panel clamps to `clamp(140px, 26vh, 280px)` and draws nine
    curves. Nine labels at the 13px minimum need 117px of plot; at the 140px
    floor there is not that much once the x-axis is taken out."""
    result = _run_node(
        """
console.log(JSON.stringify({
  tall: boardFrameLayout(makeChart(900, 280), makeLabels(9), 0.4).drawLabels,
  short: boardFrameLayout(makeChart(900, 140), makeLabels(9), 0.4).drawLabels,
}));
"""
    )
    assert result["tall"] is True
    assert result["short"] is False


def test_the_stagger_gap_tightens_before_it_gives_up():
    """20px is the comfortable gap and 13px the legibility floor. Between them
    the gap shrinks to fit rather than jumping straight to no labels."""
    result = _run_node(
        """
console.log(JSON.stringify({
  roomy: boardFrameLayout(makeChart(900, 600), makeLabels(4), 0.4).gap,
  tight: boardFrameLayout(makeChart(900, 200), makeLabels(9), 0.4).gap,
}));
"""
    )
    assert result["roomy"] == pytest.approx(20.0)
    assert 13.0 <= result["tight"] < 20.0


def test_no_labels_means_no_gap_to_stagger_by():
    """An empty board reserves nothing. Guards the divide-by-zero the gap
    formula would otherwise hit on `labels.length === 0`."""
    result = _run_node(
        """
const frame = boardFrameLayout(makeChart(900, 400), [], 0.4);
console.log(JSON.stringify({ gutter: frame.gutter, draw: frame.drawLabels, gap: frame.gap }));
"""
    )
    assert result["draw"] is False
    assert result["gap"] == 0


def test_pill_ink_follows_the_swatch_luminance():
    """Every palette entry today is a light tint on a dark page, so hardcoded
    dark ink would read -- until the first mid-dark colour lands, which is a
    one-line edit to dashboard/config/leaderboard.json away and would produce
    navy-on-navy with nothing failing."""
    result = _run_node(
        """
console.log(JSON.stringify({
  amber: boardPillTextColor('#FBBF24'),
  slate: boardPillTextColor('#94A3B8'),
  deep: boardPillTextColor('#1E3A8A'),
}));
"""
    )
    assert result["amber"] == "#0b1220"
    assert result["slate"] == "#0b1220"
    assert result["deep"] == "#f8fafc"
