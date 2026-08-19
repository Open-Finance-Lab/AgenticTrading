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


def test_the_frame_factories_are_explicit_cross_file_exports():
    """Same contract as `buildEquityCurvesFromEntries`: explicit, not the
    implicit global these classic scripts share. On rename the implicit form
    degrades to a chart with no frame, and a frame that silently stops drawing
    looks exactly like a frame nobody asked for."""
    assert "window.createEndpointLabelPlugin = createEndpointLabelPlugin;" in _SRC
    assert "window.createAxisArrowPlugin = createAxisArrowPlugin;" in _SRC
    assert "window.createEndpointLabelPlugin" in _HOME_SRC
    assert "window.createAxisArrowPlugin" in _HOME_SRC


def test_screen_zero_installs_the_frame_on_its_chart():
    """Not merely defined next to it. The plugins array is what makes it draw."""
    assert "homeBoardFramePlugins()" in _HOME_SRC
    chart_call = _HOME_SRC[_HOME_SRC.index("new window.Chart(") :][:400]
    assert "plugins: homeBoardFramePlugins()" in chart_call


def test_screen_zero_says_so_when_the_frame_is_missing():
    """A missing export degrades to a frameless chart, which is a plausible
    design rather than a break -- so it needs a signal, exactly like the missing
    curve-builder case this module already warns about."""
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:700]
    assert "console.warn" in fn
    assert "return []" in fn


def test_screen_zero_does_not_grow_a_pointer_gate():
    """`Interaction.modes.nearest` delegates to getNearestItems, which returns []
    unless `chart.isPointInArea(position)` -- so the widened gutter is already
    inert for this panel's tooltip. A hand-rolled gate here would be dead code
    imitating the Leaderboard tab, which needs one only because it sets
    `events: []`."""
    assert "pointermove" not in _HOME_SRC
    assert "resolveHoverTarget" not in _HOME_SRC


def test_screen_zero_keeps_its_percent_pill_by_taking_the_default():
    """The factory's default formatter is percent to two decimals, which is what
    the rank row beside each curve renders (`homeFormatReturnPct`). Passing a
    formatter here would be a second chance to render the same number two ways
    -- the reason this module borrows both axis formatters rather than writing
    its own."""
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:700]
    assert "formatValue" not in fn


def test_the_shared_fraction_is_the_default_and_any_override_says_why():
    """Both surfaces take 0.4 unless a rendered check found otherwise. If screen
    0 overrides, the number must arrive as the factory's documented option --
    not as a second constant, and not by moving the shared default, which would
    silently narrow the plot on a full-width board to fix a panel."""
    assert "const BOARD_GUTTER_FRACTION = 0.4;" in _SRC
    fn = _HOME_SRC[_HOME_SRC.index("function homeBoardFramePlugins()") :][:900]
    if "gutterFraction" in fn:
        assert re.search(r"Measured at \S", fn), (
            "an override must carry the measurement that justified it"
        )
    assert "BOARD_GUTTER_FRACTION" not in _HOME_SRC, (
        "the fraction is the frame's; screen 0 either takes it or passes an option"
    )


# ---------------------------------------------------------------------------
# The label stack's vertical layout. Behavioural, not source-shape: the defect
# these pin rendered clipped labels on BOTH surfaces at 1280/1440/1920 while
# every source-shape guard in this module stayed green.
# ---------------------------------------------------------------------------

def _run_stack_node(script: str):
    """``boardStackLabels`` under node. It is pure in (labels, gap, top, bottom)
    and touches no canvas, so it needs only the BOARD_* constants -- not the
    width/height chart stub the layout tests use."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [_board_constants(), _extract_function("boardStackLabels"), script]
    )
    proc = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _stack(anchors, gap, canvas_height):
    """Lay `anchors` out in the band the plugin uses: the CANVAS, inset by half
    a pill at each edge, which is what makes a pill's own edges land on canvas."""
    script = """
const half = BOARD_PILL_HEIGHT / 2;
const labels = %s.map((y, i) => ({ i, y }));
const fits = boardStackLabels(labels, %r, half, %r - half);
const ys = labels.map((l) => l.y);
console.log(JSON.stringify({
  fits,
  top: +(ys[0] - half).toFixed(2),
  bottom: +(ys[ys.length - 1] + half).toFixed(2),
  minGap: +Math.min(...ys.slice(1).map((y, i) => y - ys[i])).toFixed(2),
}));
""" % (json.dumps(anchors), gap, canvas_height)
    return _run_stack_node(script)


# Endpoint y positions RECORDED from the live render at 1440px, not invented:
# a fixture built from the plugin's own arithmetic would drift with the code and
# stay green through exactly the kind of regression these exist to catch.
_HOME_1440 = ([36.82, 81.13, 121.29, 124.1, 125.51, 138.83, 161.39, 162.81, 167.03], 19.67, 211)
_TAB_1440 = (
    [45.95, 82.5, 104.17, 108.27, 164.75, 166.1, 168.71, 170.68, 189.42, 221.15, 223.14, 229.08],
    19.5,
    268,
)


@pytest.mark.parametrize(
    "anchors,gap,height,surface",
    [_HOME_1440 + ("screen 0",), _TAB_1440 + ("Leaderboard tab",)],
)
def test_no_label_is_laid_out_past_either_canvas_edge(anchors, gap, height, surface):
    """THE REGRESSION. Both surfaces' endpoints cluster low, so the staggered
    stack is taller than the PLOT -- 202.5px against 168.8px on screen 0, 255.3px
    against 237.4px on the tab. The version this replaces clamped to `chartArea`
    with two whole-stack shifts that cancelled exactly, and drew the last label
    10.4px past the canvas bottom on screen 0 and 5px past it on the tab.

    The bound is the CANVAS: a gutter label sits right of the plot, where the
    x-axis tick strip is empty, so hanging below `chartArea.bottom` is fine and
    only the canvas edge clips."""
    out = _stack(anchors, gap, height)
    assert out["fits"] is True, f"{surface}: the stack should fit this canvas"
    assert out["top"] >= 0, f"{surface}: top label {out['top']}px above the canvas"
    assert out["bottom"] <= height, (
        f"{surface}: bottom label ends at {out['bottom']} on a {height}px canvas"
    )
    assert out["minGap"] >= gap - 0.01, (
        f"{surface}: gap compressed to {out['minGap']} -- nothing may be squeezed "
        "to buy the room; the frame drops labels instead"
    )


def test_a_band_too_small_for_the_stack_is_refused_rather_than_clipped():
    """The degradation, at the helper. Nine pills at a 19.67px pitch need
    8*19.67 + 15 = 172.4px; offered 120, the stack cannot fit, and the contract
    is to SAY SO so the plugin draws nothing -- not to return a stack whose tail
    hangs off the canvas. An earlier draft of this guard asserted the overhang
    was 'only as large as the real shortfall', which would have certified that
    clipping the top edge is acceptable once the bottom edge was fixed."""
    out = _stack([20, 24, 28, 33, 39, 44, 50, 56, 61], 19.67, 120)
    assert out["fits"] is False, "a stack that cannot fit must report it"


def test_a_panel_too_short_for_its_labels_reserves_the_arrow_and_nothing_else():
    """The degradation, at the layout hook -- and the reachable one. Screen 0's
    chart is 132px tall at any viewport <= 700px high (measured), where nine
    labels want a 10.9px pitch against a 13px legibility floor. The frame gives
    the gutter back rather than stacking unreadable text, which is what a
    rendered check at 390px and at 1440x600 showed it doing."""
    out = _run_node(
        """
const chart = makeChart(550, 132);
const frame = boardFrameLayout(chart, makeLabels(9), 0.4);
console.log(JSON.stringify(frame));
"""
    )
    assert out["drawLabels"] is False
    assert out["gutter"] == 18, "arrow-only reserves BOARD_ARROW_PAD, not a gutter"
