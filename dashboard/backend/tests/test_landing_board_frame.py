"""The board frame is implemented twice; these pin the two copies together.

`js/leaderboard.js` (vanilla Chart.js, two surfaces) and
`landing/src/lib/boardFrame.ts` (Recharts, one surface) draw the same contract
on different stacks. The duplication is forced by the stacks and accepted --
leaving the NUMBERS unguarded is not, which is the arrangement
test_the_two_surfaces_agree_on_the_numbers_that_must_agree already establishes
for the other pair.

TWO TIERS, AND THE SECOND SKIPS IN CI. The constant mirror is a source scan and
runs everywhere. The behavioural test transpiles the TS with the esbuild inside
dashboard/landing/node_modules and runs it under node, so it needs an `npm
install` CI does not do. A green CI therefore says the numbers agree, NOT that
the geometry was exercised -- run this suite locally before shipping a change to
either copy.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LEADERBOARD_JS = (_ROOT / "frontend" / "js" / "leaderboard.js").read_text(encoding="utf-8")
_BOARD_FRAME_TS_PATH = _ROOT / "landing" / "src" / "lib" / "boardFrame.ts"
_ESBUILD = _ROOT / "landing" / "node_modules" / ".bin" / "esbuild"

_JS_CONST = re.compile(r"^const (BOARD_[A-Z_]+) = (.+);$", re.M)
_TS_CONST = re.compile(r"^export const (BOARD_[A-Z_]+) = (.+);$", re.M)


def _constants(source: str, pattern: re.Pattern) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(source)}


def test_both_copies_declare_the_same_frame_constants():
    js = _constants(_LEADERBOARD_JS, _JS_CONST)
    ts = _constants(_BOARD_FRAME_TS_PATH.read_text(encoding="utf-8"), _TS_CONST)
    assert js, "no BOARD_* constants in js/leaderboard.js"
    assert set(js) == set(ts), (
        f"the two copies of the frame declare different constants; "
        f"only in js: {sorted(set(js) - set(ts))}, only in ts: {sorted(set(ts) - set(js))}"
    )
    for name in sorted(js):
        assert js[name] == ts[name], (
            f"{name} disagrees: js={js[name]} ts={ts[name]}"
        )


def _run_ts(script: str):
    """Transpile boardFrame.ts to CJS and run `script` against it under node."""
    node = shutil.which("node")
    if not node or not _ESBUILD.is_file():
        pytest.skip("node and dashboard/landing/node_modules are required")
    bundled = subprocess.run(
        [str(_ESBUILD), str(_BOARD_FRAME_TS_PATH), "--bundle", "--format=cjs",
         "--platform=node", "--log-level=error"],
        capture_output=True, text=True, timeout=60,
    )
    assert bundled.returncode == 0, bundled.stderr
    proc = subprocess.run(
        [node, "-e", bundled.stdout + "\n" + script],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_a_wide_card_reserves_the_measured_floor_plus_slack_not_two_fifths_of_the_width():
    """Renamed from a superseded assertion. `frameLayout`'s gutter formula is
    `max(floor, min(width * fraction, floor + BOARD_GUTTER_SLACK))` -- fraction
    is a ceiling on the measured floor, not the gutter's width (see the comment
    above BOARD_GUTTER_FRACTION in js/leaderboard.js). At width 900 with these
    nine labels the floor is 144px (12 inset + 104 widest block + 12 tick
    clearance + 16 trailing pad); width * 0.4 = 360 but is clamped to
    floor + BOARD_GUTTER_SLACK = 144 + 36 = 180, not the raw 40% figure the old
    test name asserted."""
    result = _run_ts(
        """
const labels = Array.from({length: 9}, (_, i) => ({name: 'Model ' + i, value: '+1.00%'}));
const frame = module.exports.frameLayout({width: 900, height: 420, labels});
console.log(JSON.stringify({gutter: frame.gutter, draw: frame.drawLabels}));
"""
    )
    assert result["gutter"] == pytest.approx(180.0)
    assert result["draw"] is True


def test_a_narrow_card_drops_the_labels_rather_than_clipping_them():
    """390px is the stacked phone width the card is measured at. Verified
    against the shipped formula: at width 300 the measured floor (198px, from
    15-char names and 7-char values) exceeds width * BOARD_GUTTER_MAX_FRACTION
    (150px), so frameLayout gives up and reserves BOARD_ARROW_PAD (18px) alone
    -- the same value the superseded formula produced, so no correction needed
    here."""
    result = _run_ts(
        """
const labels = Array.from({length: 9}, () => ({name: 'DeepSeek V4 Pro', value: '-12.34%'}));
const frame = module.exports.frameLayout({width: 300, height: 420, labels});
console.log(JSON.stringify({gutter: frame.gutter, draw: frame.drawLabels}));
"""
    )
    assert result["draw"] is False
    assert result["gutter"] == pytest.approx(18.0)


def test_the_stack_fit_guard_is_wired_in_and_does_not_misfire_at_its_tightest_margin():
    """`frameLayout` refuses ((n-1)*gap + BOARD_PILL_HEIGHT > height) before it
    ever measures a floor -- the brief this test replaces had no counterpart for
    it at all. This test reaches that line at the tightest legal margin: 20
    labels packed at exactly BOARD_LABEL_GAP_MIN (16px, chosen so height = 34 +
    16*20 = 354 makes the ratio branch land exactly on the gap floor) leaves
    (20-1)*16 + 15 = 319px of required stack against a 354px canvas -- inside by
    35px, so drawLabels stays True.

    An exhaustive local search over n in [1, 500] and height in [1, 2000] (with
    an oversized width so no other guard can intervene) found no combination
    where this refusal fires on its own under today's constants -- matching the
    shipped comment in js/leaderboard.js: 'With today's constants it cannot
    fire.' The guard exists as a safety net for a future edit to
    BOARD_LABEL_GAP_MAX or BOARD_XAXIS_ALLOWANCE, not because it is reachable
    today. Porting it without exercising the line at all would leave that net
    silently absent from the mirror; this is the closest a legal frameLayout
    call gets to it."""
    result = _run_ts(
        """
const labels = Array.from({length: 20}, (_, i) => ({name: 'M' + i, value: '+1%'}));
const frame = module.exports.frameLayout({width: 2000, height: 354, labels});
console.log(JSON.stringify({gutter: frame.gutter, draw: frame.drawLabels, gap: frame.gap}));
"""
    )
    assert result["draw"] is True
    assert result["gap"] == pytest.approx(16.0)


def test_the_stagger_separates_coincident_endpoints():
    """The real board spans -0.43% to +7.49%, so nine endpoints land within a
    few pixels of each other. Without the stagger they are one smear."""
    result = _run_ts(
        """
const anchors = Array.from({length: 5}, (_, i) => ({key: 'k' + i, anchorX: 500, anchorY: 200 + i}));
const placed = module.exports.stackLabels(anchors, {gap: 20, top: 0, bottom: 400});
console.log(JSON.stringify(placed.map((p) => ({y: p.y, displaced: p.displaced}))));
"""
    )
    ys = [row["y"] for row in result]
    assert ys == sorted(ys)
    assert all(b - a >= 20 for a, b in zip(ys, ys[1:])), "every pair clears the gap"
    assert result[0]["displaced"] is False, "the top label did not move"
    assert result[-1]["displaced"] is True


def test_an_overflowing_stack_is_pushed_back_inside_the_plot():
    """Both bounds, not just the bottom. Pushing an overflowing stack up can
    drive its head above the plot top, and a label drawn above the chart is not
    a smaller bug than one drawn below it."""
    result = _run_ts(
        """
const anchors = Array.from({length: 6}, (_, i) => ({key: 'k' + i, anchorX: 500, anchorY: 390 + i}));
const placed = module.exports.stackLabels(anchors, {gap: 20, top: 0, bottom: 400});
console.log(JSON.stringify(placed.map((p) => p.y)));
"""
    )
    assert min(result) >= 0
    assert max(result) <= 400
