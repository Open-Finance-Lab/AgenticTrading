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
            _extract_function("hexToRgb"),
            _extract_function("boardXAxisHeight"),
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
    // Chart.js v4's own rule, not a convenience: `meta.hidden` wins only when
    // it is an actual boolean, otherwise the dataset's flag. Stubbing this
    // rather than a bare `!hidden` is what keeps the test honest about the
    // contract boardVisibleEndpoints now delegates to.
    isDatasetVisible(i) {
      const meta = this.getDatasetMeta(i);
      return typeof meta.hidden === 'boolean' ? !meta.hidden : !datasets[i].hidden;
    },
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


def _run_plugin_node(script: str):
    """Runs `createEndpointLabelPlugin`'s own hooks end to end -- `beforeLayout`
    reserves the gutter via `boardFrameLayout`, then `afterDatasetsDraw` walks
    it -- through a recording ctx stub, rather than re-deriving what the draw
    hook does by hand. Pulls in every helper the draw hook calls by name
    (`boardRoundRect`, `boardStackLabels`, `boardSignedPercent`, ...) so a
    rename or a signature change reddens this instead of a hand-copied
    stand-in quietly drifting from the shipping code.
    """
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [
            _board_constants(),
            _extract_function("hexToRgb"),
            _extract_function("hexToRgba"),
            _extract_function("boardXAxisHeight"),
            _extract_function("shortName"),
            _extract_function("boardSeriesColor"),
            _extract_function("boardPillTextColor"),
            _extract_function("boardLabelBlockWidth"),
            _extract_function("boardFrameLayout"),
            _extract_function("boardVisibleEndpoints"),
            _extract_function("boardLayoutLabels"),
            _extract_function("boardWatchGutterFont"),
            _extract_function("boardSignedPercent"),
            _extract_function("boardDefaultValueText"),
            _extract_function("boardRoundRect"),
            _extract_function("boardStackLabels"),
            _extract_function("createEndpointLabelPlugin"),
            """
// A ctx stub that also RECORDS every fillText call -- (text, x, y) -- since
// that is the only place the draw hook's accumulated x-offset becomes
// observable from outside. Same 6px/char rule as the layout tests' stub.
function makeCtx() {
  const fillTextCalls = [];
  return {
    fillTextCalls,
    save() {}, restore() {}, beginPath() {}, closePath() {},
    arc() {}, fill() {}, stroke() {}, moveTo() {}, lineTo() {},
    quadraticCurveTo() {}, setLineDash() {},
    font: '', textBaseline: '', textAlign: '',
    fillStyle: '', strokeStyle: '', lineWidth: 0,
    measureText(text) { return { width: String(text).length * 6 }; },
    fillText(text, x, y) { fillTextCalls.push({ text, x, y }); },
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


def test_the_two_fifths_split_is_a_ceiling_not_the_gutter_width():
    """Post-review change (2026-08-19, fix-gutter-cap): a 1600px Leaderboard tab
    at the raw 40% reserved 640px for a ~200px label block -- 440px of the plot
    rendering as an empty column. The design owner's fix keeps 40% as the
    UPPER BOUND the rail may take, not the width it always takes: past the
    measured floor, the gutter only grows `BOARD_GUTTER_SLACK` further, then
    stops, and the plot gets the rest back.

    This is the WIDE regime: `width * fraction` (1200 * 0.4 = 480) is far past
    `floor + BOARD_GUTTER_SLACK`, so the ceiling binds and the gutter is
    `floor + slack`, not 480."""
    result = _run_node(
        """
const floor = boardLabelBlockWidth(makeChart(1200, 420), makeLabels(9));
const frame = boardFrameLayout(makeChart(1200, 420), makeLabels(9), 0.4);
console.log(JSON.stringify({ floor, gutter: frame.gutter, draw: frame.drawLabels }));
"""
    )
    assert result["floor"] == pytest.approx(144.0)
    assert result["gutter"] == pytest.approx(result["floor"] + 36.0), (
        "the wide-chart gutter must be floor + BOARD_GUTTER_SLACK, not 40% of the "
        "canvas -- a regression back to the raw fraction reintroduces the 440px "
        "dead column this change exists to remove"
    )
    assert result["draw"] is True


def test_a_long_label_raises_the_gutter_above_the_fraction():
    """The measured floor is a hard lower bound on the gutter no matter how the
    fraction/slack ceiling above it is computed: a label block too big to fit
    under `width * fraction` still gets the room it measures, never less.

    Same width for both labels, so the only variable is the label. The SHORT
    label lands in the wide regime (`width * fraction` clears `floor + slack`,
    so the ceiling binds at `floor + slack`, not at 40% of 400). The LONG
    label's floor (198) alone exceeds `width * fraction` (160), so the ceiling
    never intervenes: case 3 (floor > width * fraction) is the floor's own.

    Both floors carry `BOARD_TICK_CLEARANCE` since 2026-08-19 -- the gutter is
    not the empty canvas the band note used to claim -- and it is reserved for
    every label so that only the descending ones have to spend it. That moved
    the short floor 84 -> 96 and the long one 186 -> 198."""
    result = _run_node(
        """
const short = boardFrameLayout(makeChart(400, 420), makeLabels(3, 'AI', '+1%'), 0.4);
const long = boardFrameLayout(
  makeChart(400, 420), makeLabels(3, 'DeepSeek V4 Pro', '-12.34%'), 0.4);
console.log(JSON.stringify({ short: short.gutter, long: long.gutter }));
"""
    )
    assert result["short"] == pytest.approx(132.0), (
        "40% of 400 (160) is past floor(96) + slack(36) = 132, so the wide-regime "
        "ceiling caps it at 132, not the raw 160"
    )
    assert result["long"] == pytest.approx(198.0), (
        "the long label's floor alone (198) already exceeds 40% of 400 (160), "
        "so this is the floor case and the ceiling never applies"
    )
    assert result["long"] > result["short"], "the measured block must be able to push past 40%"


def test_the_ceiling_has_three_regimes():
    """The post-review formula (`fix-gutter-cap-brief.md`) is
    `max(floor, min(width * fraction, floor + BOARD_GUTTER_SLACK))`, read as
    three regimes over the same 9-label set (floor == 144 throughout, only the
    canvas width changes):

    1. WIDE (`width * fraction >= floor + slack`, 1200px): the ceiling binds
       at `floor + slack` = 180, well under the raw 40% (480).
    2. TIGHT (`floor <= width * fraction < floor + slack`, 400px): 40% of 400
       is 160, inside `[144, 180)` -- the fraction itself binds, because the
       ceiling never has to intervene here. (This case was 350px until
       `BOARD_TICK_CLEARANCE` raised the floor 132 -> 144: 40% of 350 is 140,
       which is now BELOW the floor and therefore regime 3, not regime 2. The
       width moved so the case still exercises the regime it names.)
    3. OVERFLOWING (`width * fraction < floor`, 300px): 40% of 300 is 120,
       under the floor -- `Math.max(floor, ...)` wins and the gutter is
       exactly 144, the floor, not 120. This is the load-bearing outer
       `Math.max` from the brief: dropping it would clip the label text,
       which is the one failure this whole frame exists to avoid."""
    result = _run_node(
        """
console.log(JSON.stringify({
  wide: boardFrameLayout(makeChart(1200, 420), makeLabels(9), 0.4).gutter,
  tight: boardFrameLayout(makeChart(400, 420), makeLabels(9), 0.4).gutter,
  overflowing: boardFrameLayout(makeChart(300, 420), makeLabels(9), 0.4).gutter,
}));
"""
    )
    assert result["wide"] == pytest.approx(180.0), "floor(144) + slack(36), not 40% of 1200 (480)"
    assert result["tight"] == pytest.approx(160.0), "40% of 400 -- the fraction itself, uncapped"
    assert result["overflowing"] == pytest.approx(144.0), (
        "exactly the floor, not 40% of 300 (120) -- the clipping guard"
    )


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


def test_the_axis_baseline_is_drawn_under_the_endpoint_labels():
    """The baseline runs the FULL canvas width, straight through the reserved
    gutter -- so whichever plugin draws last owns those pixels.

    On `afterDraw` the arrow drew after every `afterDatasetsDraw` hook and
    struck a line through any label the stack pushes below `chartArea.bottom`
    (measured on a 1440x268 tab, 12 low-clustered curves: chartArea.bottom
    247.6, a label at y=241 whose pill spans 233.5-248.5). Two things have to
    hold for the fix, and only together: the hook is `afterDatasetsDraw`, and
    the arrow is registered BEFORE the label plugin at every call site, since
    Chart.js runs a hook in array order.

    Moving the hook does not cost the property the old comment protected --
    datasets are already drawn by `afterDatasetsDraw`, so a curve running along
    the floor still sits under the baseline."""
    arrow = _extract_function("createAxisArrowPlugin")
    assert "afterDatasetsDraw(chart)" in arrow, (
        "afterDraw put the baseline on top of every descended endpoint label"
    )
    assert "afterDraw(chart)" not in arrow

    # The tab's `plugins: [...]` array, not the function definitions above it.
    registered = _SRC[_SRC.index("plugins: ["):]
    assert registered.index("createAxisArrowPlugin") < registered.index(
        "createEndpointLabelPlugin"
    ), "leaderboard.js lists the label plugin first -- the baseline paints over it"

    # Screen 0 aliases both factories off `window` before returning them.
    assert "return [arrow(), labels()];" in _HOME_SRC, (
        "home-page.js reordered or renamed its frame array -- the arrow must "
        "still be listed before the labels"
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


# ---------------------------------------------------------------------------
# boardStackLabels, five more edge cases the forward/backward/forward passes
# were hand-traced against but never run: a single label (both loops that
# assume >= 2 elements skip entirely), two labels, unsorted input, and the two
# band-size extremes. All five are traced correct in the review that asked for
# them; a failure here is a real defect in boardStackLabels, not in the test.
# ---------------------------------------------------------------------------

def test_a_single_label_is_left_alone_when_it_already_fits():
    """`last = labels.length - 1` is 0 here, so every `for (k = 1; k <= last; …)`
    loop -- both stagger passes -- never executes, and the only code that can
    still run is the two clamp checks. A label already inside the band must
    come out exactly where it went in."""
    out = _stack([100], 20, 200)
    assert out["fits"] is True
    assert out["top"] == pytest.approx(92.5)
    assert out["bottom"] == pytest.approx(107.5)


def test_a_single_label_still_clamps_to_the_top_edge():
    """The clamp checks are reachable even with nothing to stagger against --
    an anchor above the band must still land on the band's top edge, not sail
    past it."""
    out = _stack([5], 20, 200)
    assert out["fits"] is True
    assert out["top"] == pytest.approx(0.0)


def test_two_labels_open_up_to_the_gap_and_no_further():
    """The simplest real collision: two endpoints 5px apart must separate to
    exactly `gap`, not more (nothing here should trigger the reverse or
    third-pass rework two labels are too few to need)."""
    out = _stack([100, 105], 20, 300)
    assert out["fits"] is True
    assert out["minGap"] == pytest.approx(20.0)


def test_unsorted_input_is_sorted_before_it_is_staggered():
    """The function's first line is `labels.sort((a, b) => a.y - b.y)`. Feed it
    labels tagged with a stable index but out of y order, and read the index
    order back afterward: it must come back y-ascending, not in call order.

    `_stack`'s fits/minGap summary (used everywhere else in this file) cannot
    tell this apart from a bug: the forward stagger pass enforces `gap` between
    consecutive ARRAY entries regardless of whether the array is y-sorted, so a
    missing sort would still report a healthy minGap -- just with labels
    stacked in the wrong order, which only the `i` order below exposes."""
    out = _run_stack_node(
        """
const labels = [{ i: 0, y: 100 }, { i: 1, y: 80 }, { i: 2, y: 90 }];
const fits = boardStackLabels(labels, 20, 0, 300);
console.log(JSON.stringify({ fits, order: labels.map((l) => l.i) }));
"""
    )
    assert out["fits"] is True
    assert out["order"] == [1, 2, 0], (
        "labels must come back y-ascending (80, 90, 100 -> i 1, 2, 0), not call "
        "order -- the function's own sort is what guarantees this"
    )


def test_a_band_exactly_one_pill_tall_still_fits_flush_to_the_edge():
    """`bottom - top == BOARD_PILL_HEIGHT` exactly -- the boundary a `>` instead
    of `>=` off-by-one would miss. One label, anchored above the band, must
    still clamp to the top edge and be reported as fitting rather than
    refused."""
    out = _stack([0], 20, 30)  # canvas_height = 2 * BOARD_PILL_HEIGHT
    assert out["fits"] is True
    assert out["top"] == pytest.approx(0.0)


def test_a_band_smaller_than_one_pill_is_refused_rather_than_clipped():
    """Shrink the canvas below one pill's own height and the half-pill insets
    `_stack` uses (matching the real caller) cross over: `top > bottom`. The
    contract this pins is the same as the multi-label case above -- refuse the
    layout rather than hand back a position squeezed into a band that cannot
    hold it -- but reached here through the single-label path, which has no
    stagger pass to catch it and must rely on the clamp checks alone."""
    out = _stack([5], 20, 10)  # canvas_height < BOARD_PILL_HEIGHT
    assert out["fits"] is False


# ---------------------------------------------------------------------------
# The measured floor (`boardLabelBlockWidth`) and the draw hook inside
# `createEndpointLabelPlugin` each total the "dot name pill" block from
# scratch, ~1400 lines apart, sharing only `BOARD_DOT_GAP`/`BOARD_NAME_GAP` by
# promise. They agree today -- the point of this guard is to keep it that way
# numerically, by running the REAL draw hook and reading back where it put the
# pixels, rather than trusting that a future edit to either copy remembers the
# other.
# ---------------------------------------------------------------------------

def test_the_draw_hooks_block_matches_the_measured_floor():
    """The measured block is the FLOOR under the gutter -- what stops the
    frame from clipping at narrow widths. The Leaderboard tab has 280-536px of
    slack under it, so an under-measure would stay invisible there
    indefinitely; the home panel binds to within 16-52px (measured in a
    browser), so the same drift clips text on the very next redeploy with
    every source-shape guard in this file still green.

    Runs `createEndpointLabelPlugin()` for real against one label, then
    recomputes the block width two ways: once from `boardLabelBlockWidth` (the
    floor `beforeLayout` reserved room for), and once from where
    `afterDatasetsDraw` actually put the second `fillText` call (the value
    pill) plus the pill's own width. If the two literals drift apart, this
    numeric comparison fails; the source-shape checks elsewhere in this file
    (`BOARD_DOT_GAP` / `BOARD_NAME_GAP` appearing in both places) cannot."""
    out = _run_plugin_node(
        """
const NAME = 'DeepSeek V4 Pro';
const chart = {
  width: 900, height: 420,
  options: {},
  ctx: makeCtx(),
  $boardFrame: null,
  data: { datasets: [
    { label: NAME, hidden: false, data: [0, 0.1, -0.1234], borderColor: '#F97316' },
  ] },
  getDatasetMeta(i) {
    if (i !== 0) return { hidden: true, data: [] };
    return { hidden: false, data: [{ x: 10, y: 50 }, { x: 20, y: 40 }, { x: 30, y: 60 }] };
  },
  isDatasetVisible(i) { return !this.getDatasetMeta(i).hidden; },
  chartArea: { left: 0, top: 0, right: 540, bottom: 386 },
};

const plugin = createEndpointLabelPlugin();
plugin.beforeLayout(chart);
if (!chart.$boardFrame.drawLabels) throw new Error('frame declined to draw labels');
chart.ctx = makeCtx();
plugin.afterDatasetsDraw(chart);

const calls = chart.ctx.fillTextCalls;
if (calls.length !== 2) throw new Error('expected exactly 2 fillText calls, got ' + calls.length);
const [nameCall, valueCall] = calls;
const labelX = chart.chartArea.right + BOARD_GUTTER_TEXT_INSET;
const pillWidth = valueCall.text.length * 6 + BOARD_PILL_PAD_X * 2;
const drawBlockWidth = (valueCall.x - BOARD_PILL_PAD_X + pillWidth) - labelX;

const measuredBlock =
  boardLabelBlockWidth(chart, [{ name: nameCall.text, value: valueCall.text }]) -
  BOARD_GUTTER_TEXT_INSET - BOARD_TICK_CLEARANCE - BOARD_GUTTER_TRAILING_PAD;

console.log(JSON.stringify({
  nameText: nameCall.text, valueText: valueCall.text, drawBlockWidth, measuredBlock,
}));
"""
    )
    assert out["nameText"] == "DeepSeek V4 Pro"
    assert out["valueText"] == "-12.34%"
    assert out["drawBlockWidth"] == pytest.approx(out["measuredBlock"]), (
        "the draw hook painted a different width than boardLabelBlockWidth "
        "measured -- the floor no longer matches what actually renders"
    )


def test_one_nan_endpoint_does_not_blank_the_whole_rail():
    """`anchorY` comes from `meta.data[lastIdx].y` (leaderboard.js), which can
    be `NaN` for a malformed point -- and `NaN != null` is `false`, so the old
    `!= null` filter let it through where `Number.isFinite` rejects it.

    A `NaN` anchor left in the array does not merely draw one garbage label:
    `Array.prototype.sort` treats a `NaN` comparator result as "equal" (V8
    verified directly -- a `NaN`-valued entry keeps its ORIGINAL array
    position rather than sorting by value), so a `NaN` that starts out last
    stays last. From there every `boardStackLabels` comparison against it is
    `false` (`NaN > x` and `NaN <= x` both are), including the final
    `labels[last].y <= bottom` the function reports its verdict with -- so
    `fits` comes back `false` and `afterDatasetsDraw`'s `if (!boardStackLabels
    (...)) return;` bails before drawing ANYTHING. One bad series, with the old
    filter, silently deleted every other curve's label -- confirmed below: the
    unfixed filter on this exact fixture draws zero labels, not merely one.

    Three datasets, the NaN one last (the ordering that reaches the failing
    branch): the other two must still get their dot, name and value pill."""
    out = _run_plugin_node(
        """
const chart = {
  width: 900, height: 420,
  options: {},
  ctx: makeCtx(),
  $boardFrame: null,
  data: { datasets: [
    { label: 'Alpha', hidden: false, data: [0, 0.05, 0.10], borderColor: '#F97316' },
    { label: 'Charlie', hidden: false, data: [0, -0.01, 0.07], borderColor: '#4ADE80' },
    { label: 'Bravo', hidden: false, data: [0, 0.02, -0.03], borderColor: '#38BDF8' },
  ] },
  getDatasetMeta(i) {
    const y = [50, 90, NaN][i]; // Bravo (index 2, last) is the malformed point
    return { hidden: false, data: [{ x: 10, y: 10 }, { x: 20, y: 20 }, { x: 30, y }] };
  },
  isDatasetVisible(i) { return !this.getDatasetMeta(i).hidden; },
  chartArea: { left: 0, top: 0, right: 540, bottom: 386 },
};

const plugin = createEndpointLabelPlugin();
plugin.beforeLayout(chart);
chart.ctx = makeCtx();
plugin.afterDatasetsDraw(chart);

const texts = chart.ctx.fillTextCalls.map((c) => c.text);
console.log(JSON.stringify({ drawLabels: chart.$boardFrame.drawLabels, texts }));
"""
    )
    assert out["drawLabels"] is True, (
        "the gutter is still reserved -- beforeLayout drops the NaN series too "
        "now (boardLayoutLabels), leaving 2 real labels rather than 3"
    )
    assert "Alpha" in out["texts"], "the good series before the NaN one must still draw"
    assert "Charlie" in out["texts"], "the good series after the NaN one must still draw"
    assert not any("Bravo" in t for t in out["texts"]), (
        "the NaN series itself must not draw a label at a garbage position"
    )
    assert len(out["texts"]) == 4, "exactly 2 surviving labels x (name, value)"


# ---------------------------------------------------------------------------
# Post-review fixes, 2026-08-19. Each case below is a defect the review found
# in the frame as first written; they are grouped here rather than filed beside
# their neighbours so the reasoning stays with the change that motivated it.
# ---------------------------------------------------------------------------

def test_the_minimum_pitch_is_at_least_one_pill_tall():
    """`BOARD_LABEL_GAP_MIN` was 13 against a 15px pill, so at any pitch in
    `[13, 15)` the guard passed and consecutive FILLED colour pills overlapped
    by up to 2px.

    That is not a legibility judgement anyone can tune -- the guard was reading
    "is 11px text still separable?" when the binding constraint is geometric:
    two opaque rounded rects at a pitch under their own height intersect, at
    any font size. Screen 0 reaches it for real (9 series into `clamp(140px,
    26vh, 280px)` is a 152-168px canvas at a 585-645px viewport), which is the
    case below. Deriving the constant is what stops the two being edited apart
    again."""
    result = _run_node(
        """
console.log(JSON.stringify({
  gapMin: BOARD_LABEL_GAP_MIN,
  pill: BOARD_PILL_HEIGHT,
  // Screen 0 at a 600px-tall viewport: 9 curves, 26vh -> a 156px canvas.
  screenZero: boardFrameLayout(makeChart(420, 156), makeLabels(9), 0.4),
}));
"""
    )
    assert result["gapMin"] >= result["pill"], (
        "a pitch shorter than the pill itself makes adjacent pills overlap; the "
        "frame must degrade to arrow-only instead"
    )
    assert result["screenZero"]["drawLabels"] is False, (
        "9 labels into a 156px canvas is a ~13px pitch -- pills would overlap, "
        "so the frame gives the gutter back"
    )
    assert result["screenZero"]["gutter"] == 18, "arrow-only reserves BOARD_ARROW_PAD"


def test_the_axis_allowance_yields_to_the_rendered_scale_height():
    """`BOARD_XAXIS_ALLOWANCE` is the FIRST-FRAME estimate, not a permanent
    stand-in. 34 was a guess and the tab's axis renders at 20.4px, so it spent
    14px of stacking room that exists -- and that height is what both the gap
    divisor and the fits-the-canvas guard are computed from, so the frame gave
    up on labels earlier than its own geometry required.

    Chart.js leaves the previous layout's scale on the chart, so from the second
    update onwards the real number is readable. Reading it also means a change
    to the tick font moves this by itself rather than leaving a literal wrong --
    the `BoardPreview.tsx` `width={56}` failure, in the other direction."""
    result = _run_node(
        """
const bare = { width: 900, height: 268, ctx: makeChart(0, 0).ctx };
const laidOut = { ...bare, scales: { x: { height: 20.4 } } };
const junk = { ...bare, scales: { x: { height: NaN } } };
console.log(JSON.stringify({
  fallback: boardXAxisHeight(bare),
  rendered: boardXAxisHeight(laidOut),
  junk: boardXAxisHeight(junk),
  gapBefore: boardFrameLayout(bare, makeLabels(12), 0.4).gap,
  gapAfter: boardFrameLayout(laidOut, makeLabels(12), 0.4).gap,
}));
"""
    )
    assert result["fallback"] == 34, "no scale yet -- the first-frame estimate stands in"
    assert result["rendered"] == pytest.approx(20.4), "the rendered axis height wins"
    assert result["junk"] == 34, "a non-finite scale height falls back rather than poisoning the gap"
    assert result["gapAfter"] > result["gapBefore"], (
        "reading the real axis height must give the stack back the room the "
        "34px estimate was over-reserving"
    )


def test_a_numeric_layout_padding_survives_the_gutter_write():
    """`layout: { padding: 12 }` is legal Chart.js shorthand for uniform
    padding. `typeof 12 === 'object'` is false, so the hook replaced it with
    `{}` and set only `.right` -- the chart silently lost its top, left and
    bottom padding and rendered flush to those edges.

    Neither caller in this repo passes a number, which is exactly why this
    needs a test: `createEndpointLabelPlugin` is exported on `window` so other
    surfaces can adopt the frame, and this is the shape one of them will use."""
    out = _run_plugin_node(
        """
function chartWith(padding) {
  return {
    width: 900, height: 420, options: { layout: { padding } }, ctx: makeCtx(),
    $boardFrame: null,
    data: { datasets: [{ label: 'A', hidden: false, data: [0, 0.1], borderColor: '#F97316' }] },
    getDatasetMeta() { return { hidden: false, data: [{ x: 10, y: 50 }, { x: 20, y: 40 }] }; },
    isDatasetVisible() { return true; },
    chartArea: { left: 0, top: 0, right: 540, bottom: 386 },
  };
}
const plugin = createEndpointLabelPlugin();
const numeric = chartWith(12);
plugin.beforeLayout(numeric);
const object = chartWith({ top: 8 });
plugin.beforeLayout(object);
console.log(JSON.stringify({
  numeric: numeric.options.layout.padding,
  object: object.options.layout.padding,
}));
"""
    )
    assert out["numeric"]["top"] == 12, "the numeric shorthand's other three sides were dropped"
    assert out["numeric"]["left"] == 12
    assert out["numeric"]["bottom"] == 12
    assert out["numeric"]["right"] > 12, "the gutter still has to be written"
    assert out["object"]["top"] == 8, "an object padding is still mutated in place"


def test_a_label_hanging_into_the_axis_strip_clears_the_last_tick():
    """The gutter is not the empty canvas the band note used to claim. Chart.js
    centres the LAST x tick on `chartArea.right` and reserves its right-half
    overhang inside our own `layout.padding.right` -- ~18px here, ~21px on
    screen 0's larger ticks -- so a label that hangs below `chartArea.bottom`
    lands in the tick strip at overlapping x. Scales draw before
    `afterDatasetsDraw`, so the label wins those pixels rather than being
    occluded; a colour dot on the tail of a tick label is still a collision.

    Only the descending labels spend the clearance, and it is reserved for all
    of them, so paying it here can never push a block out of the gutter."""
    out = _run_plugin_node(
        """
function chartAt(anchorY) {
  return {
    width: 900, height: 200, options: {}, ctx: makeCtx(), $boardFrame: null,
    data: { datasets: [{ label: 'A', hidden: false, data: [0, 0.1], borderColor: '#F97316' }] },
    getDatasetMeta() { return { hidden: false, data: [{ x: 10, y: 20 }, { x: 530, y: anchorY }] }; },
    isDatasetVisible() { return true; },
    chartArea: { left: 0, top: 0, right: 540, bottom: 150 },
  };
}
const plugin = createEndpointLabelPlugin();
function nameX(anchorY) {
  const chart = chartAt(anchorY);
  plugin.beforeLayout(chart);
  chart.ctx = makeCtx();
  plugin.afterDatasetsDraw(chart);
  return chart.ctx.fillTextCalls[0].x;
}
console.log(JSON.stringify({
  inside: nameX(60),      // well above chartArea.bottom
  descending: nameX(190), // pill spans 182.5-197.5, past bottom (150)
  clearance: BOARD_TICK_CLEARANCE,
}));
"""
    )
    assert out["descending"] - out["inside"] == out["clearance"], (
        "a label hanging into the x-axis strip must be indented past the last "
        "tick's overhang; one inside the plot must not move"
    )


def test_a_non_finite_anchor_x_drops_its_label_too():
    """`anchorY` was finite-checked and `anchorX` was not, though the draw hook
    uses `anchorX` for the endpoint dot, the dotted stub and the leader line.

    The canvas spec silently DISCARDS an `arc`/`moveTo`/`lineTo` containing
    `NaN` -- no throw, no warning -- so a point with a finite y and a NaN x drew
    a name and a value pill with no dot and no stub beside them. An
    inconsistent row, and one no existing test could see: they all stub
    `anchorY` alone."""
    out = _run_plugin_node(
        """
const chart = {
  width: 900, height: 420, options: {}, ctx: makeCtx(), $boardFrame: null,
  data: { datasets: [
    { label: 'Alpha', hidden: false, data: [0, 0.05], borderColor: '#F97316' },
    { label: 'Bravo', hidden: false, data: [0, 0.02], borderColor: '#38BDF8' },
  ] },
  getDatasetMeta(i) {
    // Bravo's last point has a finite y and a NaN x.
    return { hidden: false, data: [{ x: 10, y: 10 }, { x: [530, NaN][i], y: [50, 90][i] }] };
  },
  isDatasetVisible() { return true; },
  chartArea: { left: 0, top: 0, right: 540, bottom: 386 },
};
const plugin = createEndpointLabelPlugin();
plugin.beforeLayout(chart);
chart.ctx = makeCtx();
plugin.afterDatasetsDraw(chart);
console.log(JSON.stringify({ texts: chart.ctx.fillTextCalls.map((c) => c.text) }));
"""
    )
    assert "Alpha" in out["texts"], "the well-formed series must still draw"
    assert "Bravo" not in out["texts"], (
        "a NaN anchorX draws a name and a pill with no dot and no stub -- drop "
        "the row instead of rendering half of it"
    )


def test_the_draw_hook_reuses_the_set_beforelayout_measured():
    """`formatValue` per series plus 2N `measureText` is the whole cost of a
    layout, and the draw hook used to rebuild all of it.

    That put the entire measurement on the mousemove path: `applyHoverTarget`
    calls `chart.update('none')` on every change of hovered curve, and a bare
    `chart.render()` on the frames between re-ran the draw hook's own copy --
    ~24 `measureText` calls and N format passes per hover frame, all producing
    values that cannot have changed. Reuse is safe precisely because nothing
    changes a label's text without an update, and an update re-runs
    `beforeLayout`.

    Asserted by mutating the data BETWEEN the two hooks: the drawn text must be
    what `beforeLayout` measured, not what a re-derivation would produce."""
    out = _run_plugin_node(
        """
const chart = {
  width: 900, height: 420, options: {}, ctx: makeCtx(), $boardFrame: null,
  data: { datasets: [{ label: 'Alpha', hidden: false, data: [0, 0.05], borderColor: '#F97316' }] },
  getDatasetMeta() { return { hidden: false, data: [{ x: 10, y: 10 }, { x: 530, y: 50 }] }; },
  isDatasetVisible() { return true; },
  chartArea: { left: 0, top: 0, right: 540, bottom: 386 },
};
const plugin = createEndpointLabelPlugin();
plugin.beforeLayout(chart);
chart.data.datasets[0].data[1] = 0.99; // no update() -- the draw hook must not see this
chart.ctx = makeCtx();
plugin.afterDatasetsDraw(chart);
console.log(JSON.stringify({
  stashed: chart.$boardFrame.labels.map((l) => l.value),
  drawn: chart.ctx.fillTextCalls.map((c) => c.text),
}));
"""
    )
    assert out["stashed"] == ["+5.00%"], "beforeLayout stashes the measured texts"
    assert out["drawn"] == ["Alpha", "+5.00%"], (
        "the draw hook re-derived the value instead of reusing the measured set"
    )


def test_both_colour_readers_share_one_hex_parse():
    """`boardPillTextColor` carried a verbatim copy of `hexToRgba`'s four-line
    parse, and neither handled the 3-digit `#rgb` form: `#fff` read as
    r=255 g=15 b=0, which is a light-ink verdict on a near-white pill in one and
    an orange-red stroke in the other.

    Latent -- every palette entry in this file is 6-digit -- but the defect is
    the duplication, not the arithmetic: a fix to one copy would not have
    reached the other. Asserted through both consumers, not through `hexToRgb`
    alone, since sharing the parse is the actual property."""
    result = _run_node(
        """
console.log(JSON.stringify({
  short: hexToRgb('#fff'),
  long: hexToRgb('#ffffff'),
  inkShort: boardPillTextColor('#fff'),
  inkLong: boardPillTextColor('#ffffff'),
  inkDark: boardPillTextColor('#000'),
  empty: hexToRgb(''),
  junk: hexToRgb('nonsense'),
}));
"""
    )
    assert result["short"] == result["long"], "`#fff` must read as white, not r=255 g=15 b=0"
    assert result["inkShort"] == result["inkLong"] == "#0b1220", "dark ink on a white pill"
    assert result["inkDark"] == "#f8fafc", "light ink on a black pill"
    assert result["empty"] == {"r": 0, "g": 0, "b": 0}, "no colour at all reads as black"
    # Not an exact triple: `parseInt` stops at the first bad digit, so a junk
    # string yields whatever prefix happened to be hex. The property that
    # matters is that no channel is NaN -- a NaN would poison the luminance
    # comparison into always choosing light ink. Unchanged from `hexToRgba`.
    assert all(isinstance(v, int) for v in result["junk"].values()), (
        "an unparseable colour must degrade to numbers, never NaN"
    )


def test_visibility_is_asked_of_chart_js_not_re_implemented():
    """`resolveHoverTarget` calls `chart.isDatasetVisible(i)`; the label rail
    hand-rolled `meta.hidden || ds.hidden` for the identical question.

    Two predicates for one chart is how the hover gate and the rail end up
    disagreeing about which curves are on screen -- a pill drawn for a series
    that is not there. Chart.js also resolves the pair properly (`meta.hidden`
    wins only when it is an actual boolean) where `||` reads an explicit
    `meta.hidden === false` as "fall through to `ds.hidden`". Behavioural
    coverage lives in the hidden/empty/all-null cases above; this pins the
    delegation itself, which those cannot see."""
    body = _extract_function("boardVisibleEndpoints")
    # STRIPPED, because the negative assertion below is otherwise satisfied by
    # the comment that explains the fix -- which is how a guard in this repo
    # has passed on prose before.
    code = "\n".join(
        ln for ln in body.splitlines() if not ln.strip().startswith(("//", "*", "/*"))
    )
    assert "chart.isDatasetVisible(i)" in code, (
        "ask Chart.js, so the rail and the hover gate cannot diverge"
    )
    assert "meta.hidden ||" not in code, "the hand-rolled predicate is back"
