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


def _run_endpoints_node(script: str):
    """Same node-subprocess shape as ``_run_node``, built for
    ``boardVisibleEndpoints`` instead: a fake ``chart.data.datasets`` +
    ``chart.getDatasetMeta(i)`` pair rather than the width/height canvas stub
    the layout tests above use, since this function never touches a canvas.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [
            _extract_function("shortName"),
            _extract_function("boardSeriesColor"),
            _extract_function("boardVisibleEndpoints"),
            """
function makeChart(datasets, metas) {
  return {
    data: { datasets },
    getDatasetMeta(i) { return (metas && metas[i]) || { data: [] }; },
  };
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


def test_the_frame_is_built_by_factories_not_a_singleton():
    """Screen 0 draws the same frame over datasets that carry none of this tab's
    private fields (`_raw`, `_entry`, `_style`) and has no hover gate. A shared
    singleton would have to close over this module's `hoveredDatasetIndex` and
    `currentChartView`, so the only way to share it is to parameterise it."""
    assert "function createEndpointLabelPlugin(options)" in _SRC
    assert "function createAxisArrowPlugin(" in _SRC
    assert "const endpointLabelPlugin = {" not in _SRC, (
        "the singleton is replaced by a factory call, not kept beside it"
    )


def test_the_gutter_is_reserved_in_beforelayout_and_never_in_the_domain():
    """`layout.padding`, not scale domain. `resolveHoverTarget` rejects
    `x > chartArea.right` outright, so padding leaves the hover gate's whole
    premise true; a domain padded with future slots would move empty territory
    inside the plot and make the gutter hoverable.

    beforeLayout because the width is a FRACTION of the rendered width, which is
    not known at config time -- and beforeLayout is the last hook that can still
    move chartArea."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "beforeLayout(chart)" in factory
    assert "padding.right = frame.gutter" in factory
    assert "data.labels" not in factory, (
        "reserving space by appending empty category labels is the design that "
        "was dropped -- it puts the gutter INSIDE chartArea"
    )


def test_the_tab_lets_the_plugin_own_the_right_padding():
    """A literal `right: 120` left in the config is dead but not inert: it is
    what renders on the first frame before beforeLayout runs, and it is what a
    reader will believe."""
    layout = re.search(r"layout:\s*\{\s*padding:\s*\{[^}]*\}", _SRC)
    assert layout, "the tab's layout.padding block moved or was deleted"
    assert "right:" not in layout.group(0), (
        "the gutter is the frame's to compute; leaving a literal here renders it "
        "for one frame and misinforms every reader after that"
    )
    assert "top: 8" in layout.group(0), "the top padding is unrelated and stays"


def test_the_tab_pill_follows_the_axis_unit():
    """Spec §4.5: no surface invents a unit its axis does not show. This tab
    defaults to `$` (`currentChartView = 'absolute'`) and the endpoint label
    printed `+7.49%` in that view -- a percent beside a dollar axis."""
    call = _SRC[_SRC.index("createEndpointLabelPlugin({") :][:800]
    assert "currentChartView === 'absolute'" in call
    assert "formatLeaderboardNumber" in call, "the money branch reuses the axis formatter"
    assert "cumulative_return" in call, (
        "the percent branch keeps preferring the entry's stored return over the "
        "last plotted point"
    )


def test_the_hover_fade_stays_this_tabs_business():
    """`hoveredDatasetIndex` is this module's pointer-gate state. Screen 0 has no
    hover gate at all, so it must reach the factory as an injected predicate
    rather than a closed-over global."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "hoveredDatasetIndex" not in factory, (
        "the factory must not close over this tab's hover state"
    )
    assert "isFaded" in factory
    # 1000, not the 800 used above: `isFaded` is the last option in the call,
    # after the longer `formatValue` body, so it needs the wider window to
    # stay inside the same call site rather than spilling into whatever
    # follows it.
    call = _SRC[_SRC.index("createEndpointLabelPlugin({") :][:1000]
    assert "hoveredDatasetIndex" in call, "the tab injects it at the call site"


def test_each_curve_ends_in_a_dot_and_a_dotted_stub():
    """The handwritten note's `•⋯` mark: the curve carries on, and the stub
    asserts no value for where it goes."""
    factory = _extract_function("createEndpointLabelPlugin")
    assert "BOARD_DOT_RADIUS" in factory
    assert "BOARD_STUB_LENGTH" in factory
    assert factory.count("setLineDash([1, 3])") == 2, "the stub and the leader line"


def test_the_arrow_is_drawn_past_the_plot_and_not_as_an_axis_tick():
    """Chart.js can draw anywhere on the canvas, not only inside chartArea, so
    the forward affordance costs no scale configuration at all -- which is the
    whole reason the future-tick design was dropped."""
    arrow = _extract_function("createAxisArrowPlugin")
    assert "chart.width" in arrow, "the arrow tip is on the canvas edge, not chartArea"
    assert "BOARD_ARROW_HEAD_LENGTH" in arrow and "BOARD_ARROW_HEAD_HALF" in arrow
    assert "afterDraw(chart)" in arrow, (
        "chrome above the data: afterDatasetsDraw would let a curve running along "
        "the floor sit on top of the baseline"
    )


def test_a_hidden_dataset_contributes_no_endpoint():
    """`hiddenSeries` toggling (the tab's legend) sets `ds.hidden` on the
    dataset the tab builds -- a hidden curve must not get a label in the
    gutter either."""
    result = _run_endpoints_node(
        """
const chart = makeChart(
  [{ label: 'A', data: [1, 2, 3], hidden: true }],
  [{ data: [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }] }],
);
const out = boardVisibleEndpoints(chart, (ds, idx) => String(ds.data[idx]));
console.log(JSON.stringify(out));
"""
    )
    assert result == []


def test_an_empty_dataset_contributes_no_endpoint():
    """Series use different hour grids (spanGaps' own justification); a curve
    that has no data at all yet must be dropped, not throw on an empty
    backward scan."""
    result = _run_endpoints_node(
        """
const chart = makeChart(
  [{ label: 'A', data: [] }],
  [{ data: [] }],
);
const out = boardVisibleEndpoints(chart, (ds, idx) => String(ds.data[idx]));
console.log(JSON.stringify(out));
"""
    )
    assert result == []


def test_an_all_null_dataset_contributes_no_endpoint():
    """A curve that is entirely gaps (no real value has arrived yet) must
    terminate the backward scan at lastIdx = -1 and drop out, not anchor on a
    null point or throw."""
    result = _run_endpoints_node(
        """
const chart = makeChart(
  [{ label: 'A', data: [null, null, null] }],
  [{ data: [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }] }],
);
const out = boardVisibleEndpoints(chart, (ds, idx) => String(ds.data[idx]));
console.log(JSON.stringify(out));
"""
    )
    assert result == []


def test_trailing_nulls_anchor_on_the_last_real_point():
    """Series use different hour grids (e.g. SPY :30 vs an LLM's :00), so a
    curve that stopped early still trails nulls out to the end of the shared
    axis. The endpoint must anchor on the last REAL value -- both the index
    and the x/y read off it -- not the last array slot."""
    result = _run_endpoints_node(
        """
const chart = makeChart(
  [{ label: 'A', data: [100, 110, 105, null, null] }],
  [{ data: [
    { x: 0, y: 50 }, { x: 1, y: 40 }, { x: 2, y: 45 }, { x: 3, y: 0 }, { x: 4, y: 0 },
  ] }],
);
const out = boardVisibleEndpoints(chart, (ds, idx) => String(ds.data[idx]));
console.log(JSON.stringify(out));
"""
    )
    assert len(result) == 1
    entry = result[0]
    assert entry["lastIdx"] == 2, "must land on the last non-null slot, index 2"
    assert entry["anchorX"] == 2 and entry["anchorY"] == 45, (
        "anchor must read meta.data[2], not the trailing null slots at 3/4"
    )
    assert entry["value"] == "105", "formatValue must be called with lastIdx, not data.length - 1"


def test_the_happy_path_returns_one_entry_per_visible_dataset():
    """A normal multi-dataset chart: each visible curve gets one endpoint,
    carrying the index, name, color and formatted value a caller (the layout
    pass, the draw hook) actually reads -- and in dataset order."""
    result = _run_endpoints_node(
        """
const chart = makeChart(
  [
    { label: 'DeepSeek V4 Pro', data: [1, 2, 3], _style: { color: '#ff0000' } },
    { label: 'SPY', data: [4, 5], borderColor: '#00ff00' },
  ],
  [
    { data: [{ x: 0, y: 9 }, { x: 1, y: 8 }, { x: 2, y: 7 }] },
    { data: [{ x: 0, y: 6 }, { x: 1, y: 5 }] },
  ],
);
const out = boardVisibleEndpoints(chart, (ds, idx) => 'V' + ds.data[idx]);
console.log(JSON.stringify(out));
"""
    )
    assert len(result) == 2
    first, second = result
    assert first["i"] == 0
    assert first["lastIdx"] == 2
    assert first["name"] == "DeepSeek V4 Pro"
    assert first["value"] == "V3"
    assert first["color"] == "#ff0000"
    assert second["i"] == 1
    assert second["lastIdx"] == 1
    assert second["name"] == "SPY"
    assert second["value"] == "V5"
    assert second["color"] == "#00ff00", "no _style on this dataset -- falls back to borderColor"
