"""Guards for the leaderboard equity chart's hover gate (js/leaderboard.js).

The chart emphasizes one curve and dims the rest while the pointer is on a line.
Getting the *gate* wrong is invisible to every other test in the suite, and two
of the ways to get it wrong are not obvious from reading the code:

* Measuring to the nearest **data point** rather than to the rendered line looks
  right on the Contest board (162 samples per curve, ~4px apart) and breaks the
  Daily board (8 samples, ~97px apart), where most of every segment then sits
  further than the hit radius from either endpoint. `resolveHoverTarget` is
  therefore exercised below at Daily-board spacing.
* Chart.js only fires `onHover` inside `chartArea`, and `chart.update()` replays
  the last in-plot event. Emphasis driven from `onHover` consequently sticks in
  the endpoint-label gutter *and* re-arms itself when cleared. (That gutter is
  measured per layout by `boardFrameLayout`, not a literal: 18px when the frame
  draws no labels, up to the measured block plus slack otherwise. It read
  "120px" here until 2026-08-19, which was the pre-frame `layout.padding.right`
  and is wrong in both directions now.) The fix is
  to take Chart.js out of event handling entirely (`events: []`) and drive hover
  from a canvas pointer listener; the source guards below hold that in place,
  since restoring either default would silently bring the bug back.

The behavioural cases run the real extracted functions under node against a stub
chart, following the convention set by test_frontend_xss_guards.py.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_LEADERBOARD_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "leaderboard.js"
)
_SRC = _LEADERBOARD_JS.read_text(encoding="utf-8")

# Daily-board geometry: 8 samples across a ~606px plot.
_POINT_SPACING = 97.0
_PLOT_LEFT = 74.0
_PLOT_RIGHT = 680.0


def _extract_function(name: str) -> str:
    """The source of ``function <name>(...) { ... }``, brace-matched.

    Extracted rather than restated so a rename or deletion fails these tests
    instead of leaving them passing against a copy that no longer ships.
    """
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


def _hit_radius_const() -> str:
    line = next(
        ln for ln in _SRC.splitlines() if ln.startswith("const HOVER_HIT_RADIUS_PX")
    )
    return line


def _run_node(script: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    harness = "\n".join(
        [
            _hit_radius_const(),
            _extract_function("resolveHoverTarget"),
            _extract_function("nearestDataIndex"),
            # Stub chart: straight segments between evenly spaced samples, which
            # is what Chart.js' LineElement.interpolate returns for tension 0.
            """
function makeChart(curves, hidden = []) {
  const area = { left: %(left)s, right: %(right)s, top: 0, bottom: 400 };
  const metas = curves.map((ys) => {
    const data = ys.map((y, i) => ({ x: area.left + i * %(spacing)s, y }));
    return {
      data,
      dataset: {
        interpolate({ x }) {
          for (let i = 1; i < data.length; i++) {
            if (x >= data[i - 1].x && x <= data[i].x) {
              const t = (x - data[i - 1].x) / (data[i].x - data[i - 1].x);
              return { x, y: data[i - 1].y + t * (data[i].y - data[i - 1].y) };
            }
          }
          return undefined;
        },
      },
    };
  });
  return {
    chartArea: area,
    data: { datasets: curves.map((_, i) => ({ label: 'curve' + i })) },
    isDatasetVisible: (i) => !hidden.includes(i),
    getDatasetMeta: (i) => metas[i],
  };
}
"""
            % {"left": _PLOT_LEFT, "right": _PLOT_RIGHT, "spacing": _POINT_SPACING},
            script,
        ]
    )
    proc = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_hit_radius_is_a_declared_constant():
    assert "const HOVER_HIT_RADIUS_PX" in _SRC


def test_on_curve_between_sparse_points_is_a_hit():
    """The Daily-board case: cursor on the line, ~48px from either sample."""
    result = _run_node(
        """
const chart = makeChart([[100, 200, 100, 200, 100, 200, 100, 200]]);
const x = chart.chartArea.left + %(spacing)s * 1.5;   // midway between samples 1 and 2
const y = 150;                                        // exactly on the line there
const target = resolveHoverTarget(chart, x, y);
console.log(JSON.stringify({
  hit: target ? target.datasetIndex : null,
  distance: target ? target.distance : null,
  gapToNearestSample: %(spacing)s / 2,
}));
"""
        % {"spacing": _POINT_SPACING}
    )
    assert result["hit"] == 0, (
        "hovering directly on the line must emphasize it even when the nearest "
        f"data point is {result['gapToNearestSample']}px away"
    )
    assert result["distance"] < 1


def test_off_curve_beyond_the_radius_is_a_miss():
    result = _run_node(
        """
const chart = makeChart([[100, 100, 100, 100, 100, 100, 100, 100]]);
const x = chart.chartArea.left + %(spacing)s * 1.5;
console.log(JSON.stringify({
  justInside: resolveHoverTarget(chart, x, 100 + HOVER_HIT_RADIUS_PX - 1) !== null,
  justOutside: resolveHoverTarget(chart, x, 100 + HOVER_HIT_RADIUS_PX + 1) !== null,
}));
"""
        % {"spacing": _POINT_SPACING}
    )
    assert result["justInside"] is True
    assert result["justOutside"] is False


def test_endpoint_label_gutter_is_never_a_hit():
    """Right of chartArea is the reserved label gutter, not part of the plot."""
    result = _run_node(
        """
const chart = makeChart([[100, 100, 100, 100, 100, 100, 100, 100]]);
const a = chart.chartArea;
console.log(JSON.stringify({
  insidePlot: resolveHoverTarget(chart, a.right - 5, 100) !== null,
  inGutter: resolveHoverTarget(chart, a.right + 40, 100) !== null,
  farGutter: resolveHoverTarget(chart, a.right + 110, 100) !== null,
  abovePlot: resolveHoverTarget(chart, a.right - 50, a.top - 5) !== null,
}));
"""
    )
    assert result["insidePlot"] is True
    assert result["inGutter"] is False
    assert result["farGutter"] is False
    assert result["abovePlot"] is False


def test_nearest_curve_wins_and_hidden_series_are_ignored():
    result = _run_node(
        """
const flat = (v) => [v, v, v, v, v, v, v, v];
const chart = makeChart([flat(100), flat(130)]);
const x = chart.chartArea.left + %(spacing)s * 1.5;
const hiddenChart = makeChart([flat(100), flat(130)], [1]);
console.log(JSON.stringify({
  nearTop: resolveHoverTarget(chart, x, 104).datasetIndex,
  nearBottom: resolveHoverTarget(chart, x, 126).datasetIndex,
  hiddenIgnored: resolveHoverTarget(hiddenChart, x, 126),
}));
"""
        % {"spacing": _POINT_SPACING}
    )
    assert result["nearTop"] == 0
    assert result["nearBottom"] == 1
    assert result["hiddenIgnored"] is None, "a hidden curve must not be hoverable"


def test_target_carries_the_nearest_sample_index_for_the_tooltip():
    """The tooltip reads `_raw[dataIndex]`, so the index must track the pointer."""
    result = _run_node(
        """
const chart = makeChart([[100, 100, 100, 100, 100, 100, 100, 100]]);
const a = chart.chartArea;
console.log(JSON.stringify({
  nearFirst: resolveHoverTarget(chart, a.left + 5, 100).dataIndex,
  nearThird: resolveHoverTarget(chart, a.left + %(spacing)s * 2 + 3, 100).dataIndex,
}));
"""
        % {"spacing": _POINT_SPACING}
    )
    assert result["nearFirst"] == 0
    assert result["nearThird"] == 2


def test_chartjs_event_handling_stays_disabled():
    """`events: []` is what stops onHover replay from re-arming a cleared hover."""
    assert "events: []" in _SRC
    assert "hover: { mode: null }" in _SRC
    assert "onHover(" not in _SRC, (
        "hover must not be driven from Chart.js' onHover: it never fires in the "
        "label gutter, and chart.update() replays the last in-plot event"
    )


def test_hover_is_driven_by_canvas_pointer_events():
    assert "addEventListener('pointermove', handleCanvasPointerMove)" in _SRC
    assert "addEventListener('pointerleave', clearChartHoverEmphasis)" in _SRC


def test_builtin_point_hover_marker_is_off():
    """Chart.js' marker ignores the proximity gate; hoverMarkerPlugin replaces it."""
    assert "pointHoverRadius: 0" in _SRC
    assert "id: 'hoverMarker'" in _SRC


def test_idle_chart_does_not_auto_emphasize_the_leader():
    body = _extract_function("getEmphasisLabel")
    assert "selectedLeaderboardEntry" in body
    assert "getEntryKind" not in body, (
        "idle view must show every curve at its kind weight; only an explicit "
        "row selection is emphasized"
    )
