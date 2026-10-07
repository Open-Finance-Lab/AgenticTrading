"""The leaderboard chart's run envelope ("band") behind a median curve (#602).

PR #604 made an LLM entry with repeat runs publish its median run and the
range of returns; the table says "Median of 3 runs · lo to hi", but the chart
drew the median alone, so the one view that shows a run's path hid that it is
one draw of several. The server now sends `entry.sample_band` -- per-index
min/max equity across the pooled runs, parallel to `equity_curve` -- and the
tab shades it behind the median line.

Executed under node against the shipped functions (lifted with `fn_body`, the
rest stubbed), plus source guards for the two properties a run cannot see: the
band is a plugin registered on the tab, and never a dataset -- hover, tooltip,
endpoint labels and the legend all iterate `chart.data.datasets`.
"""

import json
import shutil

import pytest

from dashboard.backend.tests._frontend_source import fn_body, run_node, strip_comments
from dashboard.backend.tests.test_leaderboard_curve_integrity import _LEADERBOARD_JS

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _const(name: str) -> str:
    start = _LEADERBOARD_JS.index(f"const {name} = ")
    return _LEADERBOARD_JS[start : _LEADERBOARD_JS.index(";", start) + 1]


def _prelude() -> str:
    return "\n".join(
        [
            "let hoveredDatasetIndex = null;",
            _const("SAMPLE_BAND_ALPHA"),
            fn_body("function finiteNumber(", _LEADERBOARD_JS),
            fn_body("function chartTimeKey(", _LEADERBOARD_JS),
            fn_body("function hexToRgb(", _LEADERBOARD_JS),
            fn_body("function hexToRgba(", _LEADERBOARD_JS),
            fn_body("function formatLeaderboardNumber(", _LEADERBOARD_JS),
            fn_body("function boardSignedPercent(", _LEADERBOARD_JS),
            fn_body("function transformLeaderboardChartData(", _LEADERBOARD_JS),
            fn_body("function buildSampleBandSeries(", _LEADERBOARD_JS),
            fn_body("function sampleBandAlpha(", _LEADERBOARD_JS),
            fn_body("function sampleBandRuns(", _LEADERBOARD_JS),
            fn_body("function createSampleBandPlugin(", _LEADERBOARD_JS),
            fn_body("function formatSampleBandTooltipLine(", _LEADERBOARD_JS),
        ]
    )


def _run(body: str):
    return run_node(_prelude() + "\n" + body)


def _curve(*points):
    return [{"timestamp": t, "equity": e} for t, e in points]


_T = ["2026-04-15T13:00", "2026-04-15T14:00", "2026-04-15T15:00", "2026-04-15T16:00"]
_ENTRY = {
    "model": "DeepSeek V4 Pro",
    "equity_curve": _curve(
        ("2026-04-15T13:00:00+00:00", 100000),
        ("2026-04-15T14:00:00+00:00", 100500),
        ("2026-04-15T15:00:00+00:00", 101000),
        ("2026-04-15T16:00:00+00:00", 100800),
    ),
    "sample_band": {
        "lower": [100000, 100100, 100400, 100200],
        "upper": [100000, 100900, 101500, 101300],
        "runs": 3,
    },
}


def _series(entry, times=_T):
    return _run(
        f"console.log(JSON.stringify(buildSampleBandSeries({json.dumps(entry)}, "
        f"{json.dumps(times)})));"
    )


# --------------------------------------------------------------------------
# buildSampleBandSeries
# --------------------------------------------------------------------------


@needs_node
def test_a_band_maps_onto_the_shared_axis():
    out = _series(_ENTRY)
    assert out == {
        "lower": [100000, 100100, 100400, 100200],
        "upper": [100000, 100900, 101500, 101300],
        "onGrid": [True, True, True, True],
        "runs": 3,
    }


@needs_node
def test_a_slot_another_series_owns_is_off_grid_not_a_gap():
    """SPY's :30 grid interleaves with the LLMs' :00 one on the shared axis.
    Those slots are null and `onGrid: false` -- bridged at draw time, the way
    the median's own line spans them."""
    times = ["2026-04-15T13:00", "2026-04-15T13:30", "2026-04-15T14:00",
             "2026-04-15T14:30", "2026-04-15T15:00", "2026-04-15T16:00"]
    out = _series(_ENTRY, times)
    assert out["lower"] == [100000, None, 100100, None, 100400, 100200]
    assert out["upper"] == [100000, None, 100900, None, 101500, 101300]
    assert out["onGrid"] == [True, False, True, False, True, True]


@needs_node
@pytest.mark.parametrize(
    "band",
    [
        None,
        {"lower": [1, 2, 3, 4], "upper": [1, 2, 3, 4], "runs": 1},
        {"lower": [1, 2, 3, 4], "upper": [1, 2, 3, 4]},
        {"lower": [1, 2, 3], "upper": [1, 2, 3, 4], "runs": 3},
        {"lower": [1, 2, 3, 4], "upper": [1, 2, 3, 4, 5], "runs": 3},
        {"lower": "nope", "upper": [1, 2, 3, 4], "runs": 3},
        {"lower": [None, None, None, None], "upper": [None, None, None, None], "runs": 3},
    ],
    ids=["absent", "one-run", "no-runs", "short-lower", "long-upper", "not-array", "all-null"],
)
def test_no_band_when_there_is_nothing_honest_to_draw(band):
    entry = dict(_ENTRY)
    if band is None:
        entry.pop("sample_band")
    else:
        entry["sample_band"] = band
    assert _series(entry) is None


@needs_node
def test_a_band_over_fewer_runs_than_the_label_counts_is_not_drawn():
    """"Median of 4 runs · lo to hi" beside a 3-run band would print one range
    and shade a narrower one; the server withholds that band, and so does the
    chart if a payload ever carries one."""
    entry = dict(_ENTRY, samples={"count": 4, "min_return": -0.01, "max_return": 0.02})
    assert _series(entry) is None
    entry["samples"] = dict(entry["samples"], count=3)
    assert _series(entry)["runs"] == 3


@needs_node
def test_a_missing_bound_stays_null_never_zero():
    """A $0 bound is a -100% envelope, and on a shared y-axis it flattens
    every curve on the board (#390)."""
    entry = dict(_ENTRY)
    entry["sample_band"] = {
        "lower": [100000, None, "", 100200],
        "upper": [100000, 100900, 101500, None],
        "runs": 2,
    }
    out = _series(entry)
    assert out["lower"] == [100000, None, None, None]
    assert out["upper"] == [100000, None, None, None]
    assert out["onGrid"] == [True, True, True, True]


# --------------------------------------------------------------------------
# the plugin's draw
# --------------------------------------------------------------------------

_FAKE_CHART = """
function fakeChart(datasets, opts) {
  opts = opts || {};
  const calls = [];
  const ctx = {};
  ['save', 'restore', 'beginPath', 'closePath', 'fill', 'clip'].forEach((m) => {
    ctx[m] = () => calls.push([m]);
  });
  ['moveTo', 'lineTo', 'rect'].forEach((m) => {
    ctx[m] = (...a) => calls.push([m, ...a]);
  });
  Object.defineProperty(ctx, 'fillStyle', {
    set(v) { calls.push(['fillStyle', v]); }, get() { return ''; },
  });
  const hidden = new Set(opts.hidden || []);
  const chart = {
    ctx,
    chartArea: { left: 0, top: 0, right: 100, bottom: 100 },
    scales: {
      x: { getPixelForValue: (i) => i * 10 },
      y: { getPixelForValue: (v) => 1000 - v },
    },
    data: { datasets },
    isDatasetVisible: (i) => !hidden.has(i),
    $emphasisLabel: opts.emphasis || null,
  };
  return { chart, calls };
}
function polygons(calls) {
  const out = [];
  let cur = null;
  calls.forEach((c) => {
    if (c[0] === 'moveTo') { cur = [[c[1], c[2]]]; out.push(cur); }
    else if (c[0] === 'lineTo' && cur) cur.push([c[1], c[2]]);
    else if (c[0] === 'closePath') cur = null;
  });
  return out;
}
function bandDs(label, lower, upper, onGrid) {
  return {
    label,
    _style: { color: '#ff0000' },
    _band: { lower, upper, onGrid: onGrid || lower.map(() => true), runs: 3 },
  };
}
"""


@needs_node
def test_the_band_fills_upper_forward_then_lower_back():
    out = _run(
        _FAKE_CHART
        + """
const { chart, calls } = fakeChart([bandDs('A', [10, 20, 30], [15, 25, 35])]);
createSampleBandPlugin().beforeDatasetsDraw(chart);
console.log(JSON.stringify({
  polys: polygons(calls),
  fills: calls.filter((c) => c[0] === 'fill').length,
  clipped: calls.some((c) => c[0] === 'clip'),
  balanced: calls.filter((c) => c[0] === 'save').length
    === calls.filter((c) => c[0] === 'restore').length,
}));
"""
    )
    assert out["polys"] == [[[0, 985], [10, 975], [20, 965], [20, 970], [10, 980], [0, 990]]]
    assert out["fills"] == 1
    assert out["clipped"] is True
    assert out["balanced"] is True


@needs_node
def test_a_gap_on_the_grid_splits_the_band_and_an_off_grid_slot_does_not():
    out = _run(
        _FAKE_CHART
        + """
// slot 1: another series' clock (bridged); slot 3: no run recorded (split).
const ds = bandDs('A',
  [10, null, 20, null, 30, 40],
  [15, null, 25, null, 35, 45],
  [true, false, true, true, true, true]);
const { chart, calls } = fakeChart([ds]);
createSampleBandPlugin().beforeDatasetsDraw(chart);
console.log(JSON.stringify(polygons(calls).map((p) => p.map((pt) => pt[0]))));
"""
    )
    assert out == [[0, 20, 20, 0], [40, 50, 50, 40]]


@needs_node
def test_a_hidden_curve_draws_no_band():
    out = _run(
        _FAKE_CHART
        + """
const { chart, calls } = fakeChart(
  [bandDs('A', [10, 20], [15, 25]), { label: 'SPY', _style: { color: '#00ff00' } }],
  { hidden: [0] });
createSampleBandPlugin().beforeDatasetsDraw(chart);
console.log(JSON.stringify(calls.length));
"""
    )
    assert out == 0


@needs_node
def test_band_opacity_follows_hover_and_selection():
    out = _run(
        _FAKE_CHART
        + """
function fill(hover, emphasis) {
  hoveredDatasetIndex = hover;
  const { chart, calls } = fakeChart(
    [bandDs('A', [10, 20], [15, 25]), bandDs('B', [10, 20], [15, 25])],
    { emphasis });
  createSampleBandPlugin().beforeDatasetsDraw(chart);
  return calls.filter((c) => c[0] === 'fillStyle').map((c) => c[1]);
}
console.log(JSON.stringify({
  idle: fill(null, null),
  selected: fill(null, 'B'),
  hoverA: fill(0, 'B'),
}));
"""
    )
    base, emph, faded = "rgba(255, 0, 0, 0.14)", "rgba(255, 0, 0, 0.22)", "rgba(255, 0, 0, 0.05)"
    assert out["idle"] == [base, base]
    assert out["selected"] == [base, emph]
    # Hover outranks selection, exactly as it does for the strokes.
    assert out["hoverA"] == [emph, faded]


@needs_node
def test_the_y_scale_widens_to_visible_bands_only():
    out = _run(
        _FAKE_CHART
        + """
function limits(hidden) {
  const { chart } = fakeChart(
    [bandDs('A', [90, null], [110, 130]), bandDs('B', [50, 60], [70, 80])],
    { hidden });
  const plugin = createSampleBandPlugin();
  const y = { id: 'y', min: 95, max: 120 };
  const x = { id: 'x', min: 0, max: 5 };
  plugin.afterDataLimits(chart, { scale: y });
  plugin.afterDataLimits(chart, { scale: x });
  return { y: [y.min, y.max], x: [x.min, x.max] };
}
console.log(JSON.stringify({ both: limits([]), onlyA: limits([1]), none: limits([0, 1]) }));
"""
    )
    assert out["both"] == {"y": [50, 130], "x": [0, 5]}
    assert out["onlyA"] == {"y": [90, 130], "x": [0, 5]}
    assert out["none"] == {"y": [95, 120], "x": [0, 5]}


@needs_node
def test_percent_view_transforms_the_band_like_the_curve():
    out = _run(
        f"""
const s = buildSampleBandSeries({json.dumps(_ENTRY)}, {json.dumps(_T)});
console.log(JSON.stringify(transformLeaderboardChartData(s.upper, 'cumulative', 100000)));
"""
    )
    assert out == pytest.approx([0.0, 0.009, 0.015, 0.013])


# --------------------------------------------------------------------------
# tooltip line
# --------------------------------------------------------------------------


@needs_node
def test_the_tooltip_names_the_range_in_the_current_unit():
    out = _run(
        """
const ds = { _initial: 100000,
  _band: { rawLower: [100000, 99000], rawUpper: [100000, 102500], runs: 3 } };
console.log(JSON.stringify({
  money: formatSampleBandTooltipLine(ds, 1, 'absolute'),
  pct: formatSampleBandTooltipLine(ds, 1, 'cumulative'),
  missing: formatSampleBandTooltipLine({ _initial: 1, _band: { rawLower: [null], rawUpper: [1], runs: 3 } }, 0, 'absolute'),
  none: formatSampleBandTooltipLine({ _initial: 1, _band: null }, 0, 'absolute'),
}));
"""
    )
    assert out == {
        "money": "Range of 3 runs: $99,000.00 to $102,500.00",
        # Signed like the "lo to hi" label beside the same curve.
        "pct": "Range of 3 runs: -1.00% to +2.50%",
        "missing": "",
        "none": "",
    }


# --------------------------------------------------------------------------
# source guards
# --------------------------------------------------------------------------

_RENDER = strip_comments(fn_body("async function renderEquityCurvesChart(", _LEADERBOARD_JS))


def test_the_band_plugin_is_registered_on_the_tab_and_drawn_under_the_lines():
    plugins = _RENDER[_RENDER.index("plugins: ["):]
    assert "createSampleBandPlugin()" in plugins
    # Still the pinned frame order (test_frontend_board_frame.py).
    assert plugins.index("createAxisArrowPlugin") < plugins.index("createEndpointLabelPlugin")
    plugin = fn_body("function createSampleBandPlugin(", _LEADERBOARD_JS)
    assert "beforeDatasetsDraw(chart)" in plugin
    assert "afterDatasetsDraw" not in plugin


def test_the_band_is_never_a_dataset():
    """Hover, tooltip, endpoint labels, legend and the curve picker all iterate
    `chart.data.datasets`; a band there would be hoverable and labelled."""
    assert _RENDER.count("datasets.push(") == 1
    assert "_band: band ?" in _RENDER
    assert "buildSampleBandSeries(entry, axisLabels)" in _RENDER
    # Each bound goes through the curve's own transform, so percent view works.
    assert "transformLeaderboardChartData(band.lower, currentChartView, initial)" in _RENDER
    assert "transformLeaderboardChartData(band.upper, currentChartView, initial)" in _RENDER


def test_the_tooltip_reads_the_band_line():
    assert "formatSampleBandTooltipLine(ds, idx, currentChartView)" in _RENDER


def test_a_banded_median_is_drawn_straight_like_its_band():
    """The plugin joins the bounds with lineTo; a bezier median (tension 0.1)
    overshoots between points and pokes outside its own envelope wherever it
    is itself the min or max -- always, with two runs."""
    assert "tension: band ? 0 : 0.1," in _RENDER
    plugin = strip_comments(fn_body("function createSampleBandPlugin(", _LEADERBOARD_JS))
    assert "lineTo(" in plugin
    assert "bezierCurveTo" not in plugin and "quadraticCurveTo" not in plugin


def test_the_shared_curve_builder_does_not_carry_the_band():
    """`buildEquityCurvesFromEntries` is screen 0's too; the band stays a
    leaderboard-tab concern and that return shape stays fixed."""
    builder = fn_body("function buildEquityCurvesFromEntries(", _LEADERBOARD_JS)
    assert "sample_band" not in builder
    assert "return { times, days: times, curves, trajectories, initials };" in builder
