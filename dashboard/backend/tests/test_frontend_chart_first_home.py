"""Guards for the chart-first rebuild of /app screen 0 (2026-08-15 spec).

Screen 0 lives inside `.home-pager-screen`, which is `height:100%;
overflow:hidden` in a scroll-snap pager: it CLIPS rather than scrolls, with no
scrollbar and no error. Every constraint here exists because the failure mode is
silent -- rows vanish, the chart is a blank box, and nothing logs.

The behavioural cases run the real extracted functions under node, following
test_frontend_leaderboard_hover.py. The source-shape cases guard the seams that
node cannot see (CSS, DOM insertion points, cross-file globals).
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import STYLES, css_blocks

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_CONFIG = Path(__file__).resolve().parents[2] / "config"


def _strip_comments(source: str) -> str:
    """JS with its comments removed, so a scan reads code and never prose.

    NOT optional, and the reason is specific to this file: the two functions
    guarded below are the most heavily commented in home-page.js, and those
    comments quote the guarded strings almost verbatim -- "is toFixed(2), so
    `+7.49%`", "Pinned by test_the_chart_readout...". Every `in` assertion here
    was therefore satisfiable by a comment ABOUT an implementation that had been
    deleted: remove the tooltip callback, leave the paragraph explaining it, and
    the guard stays green over the regression it exists to catch.

    test_landing_chart_first.py has stripped for exactly this reason since it
    was written; the /app half of the same pass did not. Brace matching also
    gets safer as a side effect -- `_extract` counts braces, and a comment
    containing an unbalanced one silently returns the wrong region.

    Whole-line `//` only: an inline `//` would eat the tail of any line holding
    a URL, and this file has ten of them in HOME_MOCK_NEWS.
    """
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", source)


_HOME_JS = _strip_comments((_FRONTEND / "home-page.js").read_text(encoding="utf-8"))
_LEADERBOARD_JS = _strip_comments(
    (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")
)

_PANEL_SELECTOR = (
    'html[data-nav-page="home"] #homeView .home-landing-board .home-module'
)


def _panel_block() -> str:
    """The unscoped (>1200px) rule for the board panel.

    `css_blocks` returns every block with this prelude; the <=1200px media query
    re-declares the same selector, so taking [0] rather than the whole list is
    what makes "the cap is gone" mean the desktop cap and not the stacked one.
    """
    blocks = css_blocks(_PANEL_SELECTOR)
    assert blocks, "the board panel rule was renamed or deleted"
    return blocks[0]


def test_board_panel_is_not_capped_at_a_fixed_height():
    """Measured at 1440x900, the panel's own chrome (head, meta, table head,
    Season-0 note, footer button, padding) consumes 253px of a 520px cap, and
    seven standings rows need 202px -- leaving ~0px for a chart, and a negative
    budget at 1366x768. The cap was a card-proportion choice from when the panel
    held only a table; the board is the screen's subject now and takes the row.
    """
    block = _panel_block()
    assert "height: 100%" in block
    assert "min-height: 0" in block
    assert "max-height: none" in block
    assert "520px" not in block, (
        "the 520px cap leaves ~0px for the chart at 1440x900 and is negative at 1366x768"
    )


def test_the_board_column_is_stretched_so_the_panels_height_resolves():
    """`height: 100%` on the panel is inert without this, and inert SILENTLY.

    `.home-landing-hero-inner` is `align-items: center`, so the board's cross
    size comes from its content unless it is stretched; the percentage then
    resolves against an indefinite height and CSS falls back to `auto`. The
    panel sized itself to its content, overran `.home-landing-hero` -- which is
    `overflow: hidden` -- and was cut with no scrollbar: measured 62px of
    overflow at 1280x720, 44px at 1366x768, 56px at 1201x760, 70px at 1240x700,
    with the panel header and the footer button off-screen at each.

    Asserted on the board rather than the panel because the panel's own rule
    reads perfectly correct in isolation. That is what made this survive a
    measurement pass: nothing about `height: 100%` looks wrong, and the probe
    that was supposed to catch it measured `#homeScreenLanding`, whose own
    overflow stays 0 because the hero absorbs and hides the excess.
    """
    blocks = css_blocks(
        'html[data-nav-page="home"] #homeView .home-landing-board'
    )
    assert blocks, "the board column rule was renamed or deleted"
    assert "align-self: stretch" in blocks[0], (
        "without this the panel's height: 100% resolves to auto and the hero clips it"
    )


def test_the_chart_yields_height_before_the_standings_do():
    """A rigid chart plus a bounded panel left the list showing ONE row of seven.

    Once the panel stops overrunning the hero there is a real deficit at short
    viewports -- 509px of panel against 637px of content at 1240x700 -- and
    something has to absorb it. `flex-shrink: 0` on the chart meant the list
    absorbed all of it. The standings are this panel's subject and the chart is
    its illustration, so the illustration gives way, down to a floor.
    """
    blocks = css_blocks(".hm-rank-chart")
    assert blocks, ".hm-rank-chart was renamed or deleted"
    assert "flex: 0 1 auto" in blocks[0], (
        "flex-grow must stay 0 (the list absorbs surplus) but shrink must not"
    )
    assert re.search(r"min-height:\s*\d+px", blocks[0]), (
        "a shrinkable chart needs a floor, or it collapses to a sliver"
    )


_requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract(source: str, name: str) -> str:
    """`function <name>(...) { ... }`, brace-matched, from `source`.

    Extracted rather than restated so a rename or deletion fails these tests
    instead of leaving them green against a copy that no longer ships.
    """
    marker = f"function {name}("
    start = source.index(marker)
    index = source.index("{", source.index(")", start))
    depth = 0
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1


def _const_block(source: str, name: str) -> str:
    """`const <name> = {...};` or `= [...];`, bracket-matched.

    A line-based reader returns the first line of a multi-line literal, which is
    syntactically incomplete: node then fails with a SyntaxError that reads like
    the code under test is broken. MODEL_COLOR_PALETTE and LEADERBOARD_STYLES
    are both multi-line.
    """
    start = source.index(f"const {name}")
    opener = min(
        i for i in (source.find("{", start), source.find("[", start)) if i != -1
    )
    closer = {"{": "}", "[": "]"}[source[opener]]
    depth, index = 0, opener
    while True:
        if source[index] == source[opener]:
            depth += 1
        elif source[index] == closer:
            depth -= 1
            if depth == 0:
                return source[start : index + 1] + ";"
        index += 1


def _harness() -> str:
    """Everything the extracted functions close over, in dependency order.

    Lifted from the shipped files rather than stubbed: a stub of
    `LEADERBOARD_STYLES` would quietly test the stub's dash patterns instead of
    the ones that ship, which is exactly the assertion these tests exist to make.
    """
    return "\n".join(
        [
            _const_block(_LEADERBOARD_JS, "LEADERBOARD_STYLES"),
            _const_block(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE"),
            _const_block(_LEADERBOARD_JS, "TEAM_COLOR_PALETTE"),
            "const modelColorMap = {}; const teamColorMap = {};",
            _extract(_LEADERBOARD_JS, "isModelEntry"),
            _extract(_LEADERBOARD_JS, "getModelColor"),
            _extract(_LEADERBOARD_JS, "getTeamColor"),
            _extract(_LEADERBOARD_JS, "getSeriesStyle"),
            _extract(_LEADERBOARD_JS, "chartTimeKey"),
            # The real builder, so the gate is exercised against the actual
            # "silently drops curveless entries" behaviour rather than a stub
            # that would drop them the way the test author assumed.
            _extract(_LEADERBOARD_JS, "buildEquityCurvesFromEntries"),
            # The leaderboard tab's percent formula, so screen 0's copy of it
            # can be checked for equivalence rather than eyeballed. Pure and
            # closure-free (curveValues, viewType, initialValue).
            _extract(_LEADERBOARD_JS, "transformLeaderboardChartData"),
            _const_block(_HOME_JS, "HOME_CHART_BASELINE_IDS"),
            _extract(_HOME_JS, "homeChartEntries"),
            _extract(_HOME_JS, "homeChartSeries"),
        ]
    )


def _run_node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    script = f"{_harness()}\nconsole.log(JSON.stringify({expr}));"
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


def _entry(entry_id, model, *, is_model, curve, initial_equity=10000):
    """`initial_equity` is a parameter, not a constant, on purpose.

    A fixture where every row shares one capital base cannot fail on mixed
    capital, and mixed capital is a live condition on this board: the config
    says 10000, the published curves were computed at 100000, and
    `_find_cached_run` does not key on it (issue #365). The default keeps the
    existing cases unchanged; the mixed case below passes 100000 explicitly.
    """
    points = [
        {"timestamp": f"2026-04-{15 + i:02d}T14:00:00+00:00", "equity": v}
        for i, v in enumerate(curve)
    ]
    return {
        "entry_id": entry_id,
        "model": model,
        "team_name": model,
        "is_model": is_model,
        "team_badge": "Model" if is_model else "Baseline Strategy",
        "equity_curve": points,
        "initial_equity": initial_equity,
    }


@_requires_node
def test_chart_draws_the_baselines_the_rank_list_filters_out():
    """`isHomeModelEntry` is `is_model || team_badge === 'Model'`, so the rank
    list's source has no baselines in it at all. A chart built from that source
    draws seven model curves with nothing to judge them against, which fails the
    one question the chart exists to answer: is +21.0% good?
    """
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100]),
        _entry("buy_hold_djia", "Buy & Hold", is_model=False, curve=[10000, 10550]),
        _entry("djia_index", "DJIA", is_model=False, curve=[10000, 10280]),
        _entry("mean_variance_djia", "Mean-Variance", is_model=False, curve=[10000, 10100]),
    ]
    labels = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries)"
        ".series.map(s => s.label)"
    )
    assert "DeepSeek V4 Pro" in labels
    assert "Buy & Hold" in labels, "the chart must carry a strategy baseline"
    assert "DJIA" in labels, "the chart must carry an index baseline"
    assert "Mean-Variance" not in labels, (
        "two reference curves, not five -- the panel's chart is 187-280px tall"
    )


@_requires_node
def test_baselines_are_dashed_and_models_are_not():
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100]),
        _entry("buy_hold_djia", "Buy & Hold", is_model=False, curve=[10000, 10550]),
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    by_label = {s["label"]: s for s in series}
    assert by_label["Buy & Hold"]["dash"], "baselines read as reference curves, not entrants"
    assert by_label["Buy & Hold"]["isBaseline"] is True
    assert not by_label["DeepSeek V4 Pro"]["dash"]
    assert by_label["DeepSeek V4 Pro"]["isBaseline"] is False


@_requires_node
def test_real_entries_with_no_curves_yield_no_series():
    """The third fallback state, and the reason the gate is NOT keyed on
    `sample`. `renderEntries` runs with `sample: null` whenever
    `models.length > 0`, regardless of whether any entry carries an
    `equity_curve`, and the builder silently drops curveless entries
    (`if (!points.length) return;`). Real entries + no curves therefore produce
    an empty chart with axes, under a real standings list, carrying no sample
    note -- absent and broken rendering identically.
    """
    entries = [
        {
            "entry_id": "deepseek_v4_pro",
            "model": "DeepSeek V4 Pro",
            "is_model": True,
            "team_badge": "Model",
            "equity_curve": [],
        }
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    assert series == [], "no curves must mean no chart, not an empty chart"


@_requires_node
def test_a_series_with_no_usable_points_is_dropped_rather_than_drawn_flat():
    """The one case the `.filter()` tail alone catches.

    An entry whose `equity_curve` is non-empty but whose timestamps are all
    unusable survives into `perEntry` -- `chartTimeKey` returns '' for them so
    nothing lands in `byTime`, but the `if (!points.length) return;` skip never
    fires. `curves[label]` is then a row of nulls the length of some OTHER
    entry's time axis, which Chart.js draws as a labelled dataset with no line.
    Task 4 makes the rank-row swatch this chart's only key, so that is a swatch
    pointing at nothing.

    Written after mutation-testing Step 6: removing the `.filter()` tail left
    every other case in this file green, so nothing else pinned it. The sibling
    `if (!times.length)` early return is fully subsumed by that filter -- it is
    a cheap exit, not the guard.
    """
    good = _entry(
        "deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100]
    )
    blank = _entry("buy_hold_djia", "Buy & Hold", is_model=False, curve=[10000, 10550])
    for point in blank["equity_curve"]:
        point["timestamp"] = ""

    labels = _run_node(
        f"homeChartSeries({json.dumps([good, blank])}, buildEquityCurvesFromEntries)"
        ".series.map(s => s.label)"
    )
    assert labels == ["DeepSeek V4 Pro"], (
        "a series with no usable points must be dropped, not drawn as an empty line"
    )


@_requires_node
def test_a_missing_builder_yields_no_series_rather_than_throwing():
    """`buildEquityCurvesFromEntries` lives in another file and reaches this one
    as a global. If it is ever renamed, the panel must degrade to today's
    layout, not throw inside the leaderboard load and leave the list on
    "Loading the standings...". The source guard below is what makes the rename
    loud; this is the runtime floor under it.
    """
    entries = [
        _entry("deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True, curve=[10000, 12100])
    ]
    assert _run_node(f"homeChartSeries({json.dumps(entries)}, undefined).series") == []


@_requires_node
def test_mixed_initial_equity_does_not_break_the_chart():
    """`homeChartSeries` normalises per series, so a payload whose rows carry
    different capital bases still plots on one axis.

    DEFENCE IN DEPTH, NOT A LIVE BUG -- and the distinction is the point of this
    docstring. `/api/v1/leaderboard` does not currently emit the payload built
    below: `get_leaderboard` rescales every entry by its own stored
    `initial_equity` (`service.py:1204-1211`) and then reports the same
    `display_capital` as every entry's `initial_equity` (`:1240`), so the bases
    agree on the wire and a dollar axis would draw no scale break. That was
    measured against a hand-built mixed-capital database, not assumed.

    An earlier draft of this docstring claimed the opposite -- that issue #365
    makes a 10x break reachable today. It does not. #365 is real but its damage
    is to the RETURNS: a baseline recomputed at $10k trades in a much coarser
    share quantum (one DJIA share is ~2.5% of equity, several names unbuyable),
    so its curve genuinely differs from the $100k curves it is ranked against.
    No y-axis repairs that, and this test does not claim to.

    What this pins is that `homeChartSeries` is correct as a PURE FUNCTION of
    its input, and never silently acquires a dependency on the backend
    happening to pre-normalise for it.
    """
    entries = [
        _entry(
            "deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True,
            curve=[100000, 107490], initial_equity=100000,
        ),
        _entry(
            "buy_hold_djia", "Buy & Hold", is_model=False,
            curve=[10000, 10550], initial_equity=10000,
        ),
    ]
    series = _run_node(
        f"homeChartSeries({json.dumps(entries)}, buildEquityCurvesFromEntries).series"
    )
    by_label = {s["label"]: s["values"] for s in series}
    assert set(by_label) == {"DeepSeek V4 Pro", "Buy & Hold"}

    # Fractions, so the two are on one axis despite a 10x difference in base.
    # Raw dollars would put these finals 96,940 apart.
    assert by_label["DeepSeek V4 Pro"][-1] == pytest.approx(0.0749, abs=1e-4)
    assert by_label["Buy & Hold"][-1] == pytest.approx(0.0550, abs=1e-4)
    assert all(
        abs(v) < 1 for values in by_label.values() for v in values if v is not None
    ), "a value outside +/-100% means the series is still in dollars"


@_requires_node
def test_home_chart_matches_the_leaderboards_percent_formula():
    """Screen 0 and the Leaderboard tab must compute percent identically.

    They are two files with no shared module, so the formula is duplicated by
    force. Pinning it as a STRING in either file would pass while the other
    drifted; this runs both against the same input and compares outputs, which
    is the only version of this assertion that can fail for the right reason.
    """
    entries = [
        _entry(
            "deepseek_v4_pro", "DeepSeek V4 Pro", is_model=True,
            curve=[100000, 103000, 107490], initial_equity=100000,
        ),
        _entry(
            "buy_hold_djia", "Buy & Hold", is_model=False,
            curve=[10000, 9800, 10550], initial_equity=10000,
        ),
    ]
    pairs = _run_node(
        "(() => {"
        f"  const entries = {json.dumps(entries)};"
        "   const built = buildEquityCurvesFromEntries(entries);"
        "   return homeChartSeries(entries, buildEquityCurvesFromEntries).series.map("
        "     (s) => ({"
        "       label: s.label,"
        "       home: s.values,"
        "       leaderboard: transformLeaderboardChartData("
        "         built.curves[s.label], 'cumulative', built.initials[s.label]"
        "       ),"
        "     })"
        "   );"
        "})()"
    )
    assert pairs, "no series produced -- the fixture or the harness is wrong"
    for pair in pairs:
        assert pair["home"] == pair["leaderboard"], (
            f"{pair['label']}: screen 0 and the leaderboard tab disagree on percent"
        )


def test_the_curve_builder_is_an_explicit_cross_file_export():
    """home-page.js consumes this from js/leaderboard.js. Both are classic
    scripts sharing global scope, so an implicit top-level function would work
    -- and would break silently on rename, degrading to "no chart", which is
    indistinguishable from the honest no-curves state by design (see above).
    Pinning both sides of the seam is what turns that into a red test.

    Sits with the render guards rather than the gate's, because the consuming
    half is the call site in `loadHomeLeaderboardModule`, which the render task
    adds. Asserting it a task earlier pins a seam that only has one side.
    """
    assert (
        "window.buildEquityCurvesFromEntries = buildEquityCurvesFromEntries;"
        in _LEADERBOARD_JS
    )
    assert "window.buildEquityCurvesFromEntries" in _HOME_JS


def test_the_chart_element_is_created_only_when_there_are_series():
    """Reserve nothing. Chart.js is a deferred third-party script and screen 0 is
    now the first thing /app paints, so there is a window -- longer on a
    free-tier cold start -- where the panel knows its chart's height and has
    nothing to draw in it. A reserved-but-blank 234px box looks like a chart that
    FAILED rather than one that has not arrived, which is the same absent-vs-
    broken confusion the gate exists to prevent. One downward layout shift, of
    content nobody has started reading, is the cheaper cost.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "if (!series.length) {" in body, (
        "no series must mean no canvas -- not an empty canvas"
    )
    # The insertion is guarded, not unconditional at module scope.
    assert "document.createElement('canvas')" in body or "<canvas" in body
    assert "typeof window.Chart" in body, (
        "Chart.js is deferred; the render path must tolerate it not having landed"
    )


def test_no_chart_paths_take_an_existing_chart_down_with_them():
    """"No chart this time" is a state this panel arrives at WITH ONE DRAWN.

    `onHomePageShow` calls `refreshHomeModules()` on every return to Home, and
    an IntersectionObserver calls it again, so every no-chart path is a
    re-render path. Returning early left the previous window's nine real curves
    on screen above five invented sample rows -- and because the mock roster is
    a different set of models, each row's swatch then keyed the reader to a
    different model's line than the one it named.

    Both halves are asserted: the sample branches must clear (they return before
    the chart call and are the only place that can), and the render function's
    own early exits must clear too (real entries whose `equity_curve`s are all
    empty reach it and yield no series).
    """
    teardown = _extract(_HOME_JS, "clearHomeLeaderboardChart")
    assert "homeRankChart.destroy()" in teardown, "the Chart.js instance must be released"
    assert "removeChild" in teardown or "wrap.remove()" in teardown, (
        "the wrapper element must go too -- a destroyed chart still leaves its box"
    )

    render = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    head = render[: render.index("const panel")]
    assert head.count("clearHomeLeaderboardChart()") == 2, (
        "both early exits (no series, no Chart.js) must tear an existing chart down"
    )

    entries = _extract(_HOME_JS, "renderEntries")
    assert "if (sample) clearHomeLeaderboardChart();" in entries, (
        "the three sample paths return before the chart call, so this is the only "
        "place that can take the chart down with the standings"
    )


def test_the_chart_axis_reads_dates_and_not_raw_stamps():
    """`times` are raw hourly `equity_curve` stamps -- `2026-04-15T14:00`.

    Chart.js renders an unrecognised string label verbatim and auto-rotates to
    fit, so with no callback this axis printed six ISO timestamps at ~45
    degrees, colliding with each other and running past the canvas edge, across
    a plot 132-280px tall. The formatter is borrowed from js/leaderboard.js
    rather than reimplemented: both surfaces plot the same field, and a second
    formatter is a second chance to render it two ways.
    """
    assert "window.formatShortDate = formatShortDate;" in _LEADERBOARD_JS
    assert (
        "window.formatChartTooltipLabel = formatChartTooltipLabel;" in _LEADERBOARD_JS
    )
    stamp = _extract(_HOME_JS, "homeFormatChartStamp")
    assert "window.formatChartTooltipLabel" in stamp and "window.formatShortDate" in stamp
    assert "return raw" in stamp, "a missing export must degrade to the stamp, not blank"

    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "homeFormatChartStamp(this.getLabelForValue(value), false)" in body, (
        "the x ticks must be formatted, not printed raw"
    )
    assert re.search(r"maxRotation:\s*0", body) and re.search(r"minRotation:\s*0", body), (
        "auto-rotation is what made the raw labels collide; flat ticks are the fix"
    )


def test_the_tooltip_reads_one_series_not_all_nine():
    """An index-mode tooltip over nine series is taller than the plot it sits in.

    Measured at 1440x900 before the fix: nine rows, 178px, inside a 234px
    canvas. The Leaderboard tab keeps 'index' only because it also ships a
    `tooltip.filter` bound to an explicit `hoveredDatasetIndex`; this panel has
    no such hover gate, so it uses 'nearest'.

    The `filter` is still required on top. 'nearest' returns EVERY item at the
    minimum distance, and at the leftmost tick that is all nine -- `values[0]`
    is `(base-base)/base` for every series, so the curves genuinely coincide
    there and the nine-row tooltip came back at the one x a reader starts from.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "mode: 'nearest'" in body, "'index' lists every series in one tooltip"
    assert "filter: (item, index) => index === 0" in body, (
        "every series shares its first value, so 'nearest' ties nine ways at x=0"
    )


def test_the_tooltip_signs_zero_the_way_the_rank_row_does():
    """The two sit side by side showing the same number, and the first point of
    every series is exactly zero -- so the sign rule is not a detail.

    `homeFormatReturnPct` is `> 0`, which renders `0.00%`. A `>= 0` test in the
    tooltip rendered `+0.00%` for the identical value. The precision guard above
    compares decimals and is structurally blind to this.

    The two can no longer disagree by construction -- the tooltip calls the rank
    row's formatter, which calls the board frame's `boardSignedPercent` -- so
    what is asserted here is the sign rule at the ONE place that now renders it,
    plus the local fallback that stands in when leaderboard.js has not landed.
    """
    assert "v > 0 ? '+' : ''" in _extract(_LEADERBOARD_JS, "boardSignedPercent"), (
        "the shared formatter's sign rule changed -- it renders the pill, the "
        "rank row and the tooltip at once now"
    )
    assert "pct > 0 ? '+' : ''" in _extract(_HOME_JS, "homeFormatReturnPct"), (
        "the rank list's leaderboard.js-is-absent fallback must keep the rule too"
    )


def test_the_canvas_label_names_the_baselines_it_draws():
    """The two reference curves are the reason the chart exists, and the only
    thing marking them is that their lines are dashed -- which is not
    information a screen reader receives. A label reading "for each AI model"
    told that reader the image contains exactly what the baselines were added to
    correct.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    label = re.search(r"aria-label',\s*\n?\s*'([^']+)'", body)
    assert label, "the canvas must carry an aria-label"
    text = label.group(1).lower()
    assert "baseline" in text or "buy-and-hold" in text, (
        f"the label names only the models: {label.group(1)!r}"
    )


def test_the_sample_rows_carry_real_entry_ids():
    """`getSeriesStyle` resolves a model's colour through
    `getModelColor(entry.entry_id || label)`, which mints a palette slot per
    unseen key -- and `modelColorMap` is module-level state in js/leaderboard.js
    that the Leaderboard tab shares.

    Id-less mock rows therefore entered that map under their display labels
    while the real entries enter under their ids: one model, two slots, twelve
    keys chasing a ten-colour palette, and a mock row handed the colour already
    assigned to a different real model's curve. The shift outlived this panel --
    the Leaderboard tab's own colours came to depend on whether the home module
    had failed earlier in the session.
    """
    roster = {
        s["id"]
        for s in json.loads(
            (_CONFIG / "leaderboard.json").read_text(encoding="utf-8")
        )["strategies"]
    }
    mock = _const_block(_HOME_JS, "HOME_MOCK_LEADERBOARD")
    ids = re.findall(r"entry_id:\s*'([^']+)'", mock)
    assert len(ids) == mock.count("rank:"), "every sample row needs an entry_id"
    unknown = sorted(set(ids) - roster)
    assert not unknown, (
        f"sample rows carry ids that are not on the board: {unknown} -- "
        "a plausible-looking id mints its own palette slot exactly like no id at all"
    )


def test_the_charts_baseline_ids_are_on_the_board():
    """`HOME_CHART_BASELINE_IDS` hardcodes two primary keys from
    dashboard/config/leaderboard.json.

    Ids rather than labels is the right call -- labels are renameable copy --
    but ids are editable in that same file, and nothing else connected the two.
    Rename either and screen 0 draws seven model curves with nothing to judge
    them against, no console warning, and a green suite: every fixture in this
    module hand-writes the ids it expects, so those cases assert the constant
    against itself. This is the one case that reads the roster.
    """
    roster = {
        s["id"]
        for s in json.loads(
            (_CONFIG / "leaderboard.json").read_text(encoding="utf-8")
        )["strategies"]
    }
    ids = re.findall(
        r"'([^']+)'", _const_block(_HOME_JS, "HOME_CHART_BASELINE_IDS")
    )
    assert ids, "the baseline id list was renamed or emptied"
    missing = sorted(set(ids) - roster)
    assert not missing, (
        f"the chart's reference curves are not on the board: {missing} -- "
        "screen 0 would draw the models against nothing"
    )


def test_chart_axis_ticks_are_14px():
    """Spec §2's type scale is the only thing keeping the two surfaces looking
    like one product, and nothing enforces it across stacks -- so it is pinned on
    each. The cross-surface pair check is in test_landing_chart_first.py.
    """
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert re.search(r"font:\s*\{\s*size:\s*14\s*\}", body), (
        "11px axis ticks were one of the three reported problems"
    )


def test_the_chart_readout_matches_the_rank_lists_own_precision():
    """The tooltip is a per-series readout sitting beside the rank row showing
    the same number, so the two must not disagree on decimals.

    The rank list renders `homeFormatReturnPct` (home-page.js), which is
    `toFixed(2)` -- `+7.49%`, not `+7.5%`. The AXIS keeps one decimal, for the
    unrelated reason in its own comment: tick labels over a narrow domain
    collapse into duplicates at zero decimals and turn noisy at two. Different
    jobs, different precision; only the tooltip has a neighbour to match.

    MATCHING IS NOW DELEGATION, not two expressions that agree. The tooltip
    inlined its own `(y * 100).toFixed(2)`, which is a fourth copy of a rule
    that also lives in `homeFormatReturnPct` and (twice) in leaderboard.js --
    so this guard asserted that two literals were equal rather than that one
    number had one source. It now asserts the call, and that the tooltip does
    NOT re-derive; the precision itself is pinned where it is implemented.
    """
    assert "toFixed(2)" in _extract(_HOME_JS, "homeFormatReturnPct"), (
        "the rank list's formatter changed -- re-check the tooltip's precision"
    )
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "homeFormatReturnPct(c.parsed.y)" in body, (
        "the tooltip must render through the row's own formatter"
    )
    # Scoped to the tooltip's OWN expression, not to `toFixed` anywhere in the
    # function: the axis tick callback legitimately carries `(v * 100)
    # .toFixed(1)`, which is the one-decimal rule this docstring separates out.
    assert "c.parsed.y * 100" not in body, (
        "the tooltip re-derived the percent instead of delegating -- that is "
        "exactly the drift this guard exists to prevent"
    )


def test_chart_height_is_the_app_clamp_and_not_the_landing_one():
    """The surfaces have different vertical envelopes and therefore different
    formulas. /app's panel is bounded by the pager; /'s card by the document.
    A shared assertion here would be a bug, not a simplification.
    """
    blocks = css_blocks(".hm-rank-chart")
    assert blocks, ".hm-rank-chart was renamed or deleted"
    assert "clamp(140px, 26vh, 280px)" in blocks[0]
    assert "100dvh" not in blocks[0], "that is /'s formula, measured against the fold"


def test_the_model_palette_has_a_distinct_colour_for_every_board_model():
    """`getModelColor` assigns `MODEL_COLOR_PALETTE[n % len]` in first-seen
    order. The board carries seven models and the palette had five, so models 6
    and 7 got models 1 and 2's colours -- two pairs of identically coloured
    curves. Harmless while the swatch was decoration; not harmless now that the
    swatch is the chart's only key.
    """
    block = _const_block(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE")
    colours = re.findall(r"#[0-9A-Fa-f]{6}", block)
    assert len(colours) >= 7, "seven models are on the board"
    assert len(set(c.lower() for c in colours)) == len(colours), "duplicate colours"


def test_the_model_palette_does_not_collide_with_the_baseline_styles():
    """The chart draws models and baselines in one plot area, so their colours
    share a namespace even though the palettes do not.

    `LEADERBOARD_STYLES` fixes the two reference curves' colours by label, and
    `getModelColor` hands out `MODEL_COLOR_PALETTE` entries by arrival order --
    nothing consults the other. A model handed DJIA's grey reads as a second
    index line, and the dash pattern is the only thing left distinguishing
    them at 187px tall.
    """
    models = {
        c.lower()
        for c in re.findall(
            r"#[0-9A-Fa-f]{6}", _const_block(_LEADERBOARD_JS, "MODEL_COLOR_PALETTE")
        )
    }
    baselines = {
        c.lower()
        for c in re.findall(
            r"#[0-9A-Fa-f]{6}", _const_block(_LEADERBOARD_JS, "LEADERBOARD_STYLES")
        )
    }
    assert not (models & baselines), (
        f"model palette collides with a baseline colour: {sorted(models & baselines)}"
    )


def test_rank_rows_carry_the_swatch_from_the_same_source_as_the_curve():
    """A row whose swatch disagrees with its curve is worse than no swatch: it
    points the reader at the wrong line. Both sides therefore read
    `getSeriesStyle`, rather than the list picking its own colour.
    """
    body = _extract(_HOME_JS, "renderEntries")
    assert "getSeriesStyle" in body
    assert "hm-rank-swatch" in body


def test_the_swatch_sits_inside_the_name_cell_not_in_the_row_grid():
    """`.home-module-rank-list li` is a five-column GRID whose template mirrors
    `.hm-rank-table-head` column for column. A swatch added as a direct child of
    the `<li>` therefore takes column 1 and shifts every real cell one right --
    the rank badge into the name's `1.2fr`, the name into a 72px slot -- and
    spills Sharpe into an implicit sixth column that the header does not have.
    Nothing throws; the table just stops lining up with its own head.

    `.hm-rank-entry` is already `display:flex; align-items:center; gap:4px` with
    the name ellipsising inside it, so the swatch belongs there: one flex child,
    no grid change, no header change, and it keys the model NAME rather than the
    rank number.
    """
    body = _extract(_HOME_JS, "renderEntries")
    entry_cell = body[body.index('class="hm-rank-entry"') :]
    swatch = body.index("hm-rank-swatch")
    assert swatch > body.index('class="hm-rank-entry"'), (
        "the swatch must be inside .hm-rank-entry, not a sixth grid child of the <li>"
    )
    assert entry_cell.index("hm-rank-swatch") < entry_cell.index(
        "home-module-rank-name"
    ), "the swatch reads as a key only if it precedes the name it keys"

    row = css_blocks(".home-module-rank-list li")
    assert row, "the rank row rule was renamed or deleted"
    assert row[0].count("px") >= 1 and "grid-template-columns" in row[0], (
        "this guard assumes the row is still a fixed-column grid"
    )
    head = css_blocks(".hm-rank-table-head")
    assert head, "the table head rule was renamed or deleted"

    def _columns(block: str) -> str:
        return re.search(r"grid-template-columns:([^;]+);", block).group(1).strip()

    assert _columns(row[0]) == _columns(head[0]), (
        "the row and its header must declare the same columns -- if you add one "
        "to either, add it to both"
    )


def test_the_swatch_colour_is_escaped_before_it_reaches_a_style_attribute():
    """The colour lands in an inline `style` attribute built by string
    concatenation, and it comes from a payload field: `getSeriesStyle` falls
    through to `getTeamColor(entry?.entry_id || label)` for anything it does not
    recognise, and both of those are server-supplied. Unescaped, a crafted
    label closes the attribute.
    """
    body = _extract(_HOME_JS, "renderEntries")
    assert re.search(r"style=\"background:\$\{homeEscape\(", body), (
        "the swatch colour must go through homeEscape on its way into style="
    )


def test_rank_rows_keep_ending_value_and_sharpe():
    """/ demotes its table to a legend strip because it has Race.tsx to hold the
    detail. /app has no such page, and these are real numbers a signed-in user
    came for.
    """
    body = _extract(_HOME_JS, "renderEntries")
    assert "hm-rank-value" in body
    assert "hm-rank-sharpe" in body


def test_the_screen_zero_lede_challenges_and_then_names_the_mechanism():
    """A challenge alone promises a place on a board that takes no entries.

    Two earlier ledes tried to describe: one glossed "agent" AND pre-empted "is
    my agent on this list?", the next narrated what the board shows. The
    headline states the offer and the board states the no-entry fact
    ("AI models only - ranked by return"), so a describing lede was the third
    element saying the same thing -- and dropping the description is what the
    current copy is for.

    But it cannot drop ALL the way to a bare challenge, and that is the
    assertion below that is not a string pin. Neither leaderboard accepts
    entries: `get_leaderboard` builds every row from the curated roster in
    config/leaderboard.json and `api/routers/leaderboard.py` exposes no
    submission route. So "Think you can beat them?" with nothing after it, sat
    under a ranking headline and above "Create a free account", reads as an
    invitation to join the ranking -- exactly the copy CLAUDE.md forbids. The
    trailing clause is what converts it back into something true (run the same
    window yourself), and it aims at the CTA while doing so.

    Hence the shape check as well as the literal: a future edit that tightens
    this to the punchier half sentence is the regression, and it would keep
    every string assertion here green if only the challenge were pinned.
    """
    from dashboard.backend.tests._frontend_source import APP_HTML

    html = re.sub(r"<!--.*?-->", "", APP_HTML, flags=re.DOTALL)
    lede = re.search(r'<p class="home-landing-lede">([^<]+)</p>', html)
    assert lede, "screen 0 no longer has a `home-landing-lede` paragraph"
    text = lede.group(1).strip()

    assert text == "Think you can beat them? Test your own idea on the same days."
    challenge, _, mechanism = text.partition("?")
    assert challenge and mechanism.strip(), (
        f"the lede is a bare challenge ({text!r}). Neither board takes entries, "
        "so a challenge with no mechanism after it promises a place on the "
        "ranking. Keep the clause that says what the reader actually does."
    )

    # Both retired ledes, so a revert shows up here rather than as duplicated
    # description on screen.
    assert "in a test of its own" not in html
    assert "See how the AI models did" not in html
    # The fact only the FIRST of those two carried must still be on screen, on
    # the board making the claim -- otherwise this is a deletion, not a split.
    assert "AI models only" in html


def test_the_series_style_helper_is_an_explicit_cross_file_export():
    """The same seam as the curve builder: home-page.js reads this off `window`
    and falls back to a transparent swatch when it is missing, so a rename
    degrades to colourless rows rather than an error. Pinned from both sides.
    """
    assert "window.getSeriesStyle = getSeriesStyle;" in _LEADERBOARD_JS
    assert "window.getSeriesStyle" in _HOME_JS


_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)

# `.home-scroll-hint` is absolute at `bottom: 18px` and measures 56px tall, so
# the band it occupies is 74px. The shipped reserve is larger; this is the floor
# below which the hint provably sits on top of whatever the hero laid out.
_HINT_STRIP_FLOOR_PX = 74

# The viewport height at or above which the reserve measured FREE: the populated
# panel keeps all seven rows and overflows by 0, exactly as it did before the
# reserve existed. At 760 it starts costing rows, and an UNCONDITIONAL reserve
# overflowed the panel by 12/32/52/72px at 680/660/640/620 and put the footer
# button outside its own `overflow: hidden` box.
_RESERVE_MAX_GATE_PX = 768

# Above this the headline's second line wraps -- at the WIDEST screens, because
# the gap keeps growing while the rail is pinned. TWO bounds stack: this ratio
# (which sets the copy column) and the headline's own `max-width: 36rem` = 576px.
# Below ~1650px of viewport the column is the narrower of the two and the ratio
# binds; above it the column clears 576, the h1 stops at its cap, and the ratio
# stops mattering. Measured against the h1's laid-out width -- min(column, 576)
# -- at 2176px with the current headline: 1.25 -> +26.5px, 1.35 -> +8.5px,
# 1.38 -> +1.5px, 1.40 -> wraps. The ceiling is ~1.38 and is deliberately NOT
# taken here: the copy got shorter, the constraint did not get looser. The
# previous headline ("AI models finished", 25.6px wider) put the ceiling at 1.28
# and cleared its cap by +1.0px at 1744 and up.
_MAX_BOARD_GROW = 1.25

_HINT_SELECTOR_HINT = "scroll-hint"
_HORIZONTAL_ANCHORS = ("left:", "right:", "inset:", "inset-inline")


def _brace_end(index: int) -> int:
    """Index just past the `}` closing the `{` at or after `index`."""
    index = STYLES.index("{", index)
    depth = 0
    while True:
        if STYLES[index] == "{":
            depth += 1
        elif STYLES[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1


def _at_rule_spans() -> list[tuple[int, int]]:
    """(start, end) of every `@media` block, so a rule can be told from an override."""
    return [
        (match.start(), _brace_end(match.start()))
        for match in re.finditer(r"@media[^{]*\{", STYLES)
    ]


def _exact_rule_blocks(prelude: str) -> list[str]:
    """`css_blocks`, minus blocks where `prelude` is only a selector's TAIL.

    `css_blocks` searches for the prelude followed by `{`, so a *scoped* rule --
    `html[data-nav-page="home"] #homeView .home-scroll-hint {` -- also matches a
    bare `.home-scroll-hint` and comes back as a block that begins mid-selector.
    A guard asking "does the UNSCOPED rule still viewport-centre the hint" was
    therefore satisfiable by the scoped override: by the very rule it exists to
    be independent of.

    `,` is in the accepted set deliberately. A prelude appearing as the second
    member of a selector list is a real rule for that selector, and dropping it
    turns any "no rule does X" assertion into a guaranteed pass -- the failure
    mode is silent and always in the permissive direction.
    """
    starts = [
        match.start() for match in re.finditer(re.escape(prelude) + r"\s*\{", STYLES)
    ]
    blocks = css_blocks(prelude)
    assert len(starts) == len(blocks), "css_blocks stopped matching its own regex"
    return [block for _, block in _exact_rule_sites(prelude)]


def _exact_rule_sites(prelude: str) -> list[tuple[int, str]]:
    """`(start offset, block)` for each rule whose selector is exactly `prelude`."""
    starts = [
        match.start() for match in re.finditer(re.escape(prelude) + r"\s*\{", STYLES)
    ]
    blocks = css_blocks(prelude)
    kept = []
    for start, block in zip(starts, blocks):
        before = STYLES[:start].rstrip()
        if before == "" or before[-1] in "{}," or before.endswith("*/"):
            kept.append((start, block))
    return kept


def _declarations(block: str, prop: str) -> list[str]:
    """Every value `block` declares for `prop`, read from code and never prose.

    Comments are stripped first because the rules guarded here carry long notes
    quoting their own numbers -- the trap this file's `_strip_comments`
    docstring describes for the JS side. Without the strip, moving a value and
    leaving the note behind keeps the guard green on the regression it exists
    to catch.

    The terminator is `;` OR `}`: a final declaration written without a
    semicolon is legal CSS that minifiers and hand-edits both produce, and a
    `;`-only regex silently reports it as absent -- which reaches the callers
    as `ValueError: not enough values to unpack` instead of the actionable
    message each assertion carefully spells out.

    The declaration is anchored to `{` or `;` rather than to start-of-line for
    the same reason. styles.css writes plenty of rules on one line
    (`.home-landing-hero-inner { flex-direction: column; align-items: stretch; }`),
    and a `^`-anchored pattern reads NOTHING from those -- so re-writing a
    guarded rule as a one-liner would quietly empty every assertion built on it.
    Anchoring also keeps `padding` from matching `padding-block`, and
    `max-width` from matching `min-width`, since the colon must follow the name.
    """
    body = _CSS_COMMENT.sub("", block)
    return [
        match.group(1).strip()
        for match in re.finditer(rf"(?:^|[{{;])\s*{re.escape(prop)}:\s*([^;}}]+)[;}}]", body)
    ]


def _shorthand_parts(value: str) -> list[str]:
    """Split a shorthand on TOP-LEVEL whitespace only.

    A plain `.split()` is wrong here and fails in the permissive direction: the
    hero's padding is `clamp(20px, 3vh, 40px) clamp(40px, 5vw, 80px)`, and the
    spaces inside those `clamp()` calls turn a two-value shorthand into six
    tokens, so the value read back is `3vh,` rather than a length.
    """
    parts: list[str] = []
    depth = 0
    current = ""
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char.isspace() and depth == 0:
            if current:
                parts.append(current)
                current = ""
            continue
        current += char
    if current:
        parts.append(current)
    return parts


def _px(value: str) -> float:
    match = re.fullmatch(r"(-?[\d.]+)px", value.strip())
    assert match, f"expected a plain px length, got {value!r}"
    return float(match.group(1))


def _at_rule_blocks(condition: re.Pattern[str]) -> list[tuple[str, str]]:
    """(condition text, brace-matched body) for every `@media` whose condition matches."""
    found = []
    for match in re.finditer(r"@media\s*([^{]+)\{", STYLES):
        if not condition.search(match.group(1)):
            continue
        index = STYLES.index("{", match.start())
        depth = 0
        while True:
            if STYLES[index] == "{":
                depth += 1
            elif STYLES[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        found.append((match.group(1).strip(), STYLES[match.start() : index + 1]))
    return found


def _hint_strip_px() -> float:
    """The one declared width of the scroll hint's reserved band."""
    values = [
        match.group(1).strip()
        for match in re.finditer(r"--home-hint-strip:\s*([^;}]+)[;}]", _CSS_COMMENT.sub("", STYLES))
    ]
    assert len(values) == 1, (
        f"--home-hint-strip is declared {len(values)} times; it exists so the "
        "hint's band is one number -- re-point this guard, do not fan it out"
    )
    return _px(values[0])


def _unique_block(prelude: str, prop: str) -> str:
    """The one UNCONDITIONAL rule for `prelude` that declares `prop`.

    "Unconditional" -- outside every `@media` -- rather than `blocks[0]`.
    Position is not a property of a rule: authoring a narrow-viewport override
    above the base silently changes which one `[0]` returns, and the guard then
    reports a stacked value as the desktop one and passes whatever the desktop
    rule says. Selecting on "declares this property" is not enough either,
    because the `<=1200px` overrides re-declare `flex` and `max-width` too.
    """
    spans = _at_rule_spans()
    matching = [
        block
        for start, block in _exact_rule_sites(prelude)
        if _declarations(block, prop)
        and not any(begin <= start < end for begin, end in spans)
    ]
    assert len(matching) == 1, (
        f"{prelude} declares `{prop}` in {len(matching)} unconditional rules; "
        "this guard needs exactly one -- disambiguate it here rather than "
        "deleting the case"
    )
    return matching[0]


def _rail_geometry(prelude: str) -> tuple[float, float]:
    """(rail width, horizontal inset) for the one rule that declares the rail.

    Both numbers come from the SAME block: screen 1 re-declares its padding
    inside `@media (max-width: 820px)`, so reading width and padding from
    different rules compares a desktop rail against a stacked inset.
    """
    block = _unique_block(prelude, "max-width")
    assert "box-sizing: border-box" in _CSS_COMMENT.sub("", block), (
        f"{prelude} is no longer border-box, so its max-width stopped including "
        "its padding and the arithmetic below is wrong"
    )
    (width,) = _declarations(block, "max-width")
    padding = _declarations(block, "padding")
    if not padding:
        return _px(width), 0.0
    parts = _shorthand_parts(padding[0])
    inline = parts[1] if len(parts) > 1 else parts[0]
    return _px(width), _px(inline)


def test_the_two_pager_screens_share_one_content_rail():
    """The two screens put their CONTENT on one rail, not their border boxes.

    `#homeView` is a scroll-snap pager: the screens are never on-screen
    together, so a rail mismatch is invisible in any single view and shows up
    only as content sliding sideways on every snap.

    The quantity is the content rail, and getting it wrong is not theoretical:
    screen 1 caps at 1500px then insets 28px a side, so equalising the two
    `max-width` declarations at 1500 lands the border boxes on the same x while
    moving the content edges 28px APART -- measured, against 2px for the 1440
    rail that preceded it, whose 60px under-size was almost exactly screen 1's
    56px of padding. A guard on the raw declarations calls that a fix.

    SCOPE, because the assertion message must not overclaim: equal content rails
    give equal x only where BOTH caps bind, measured 1744px and up. Below that
    the hero's own `padding-inline: clamp(40px, 5vw, 80px)` sets its rail
    instead and the screens still differ -- 32px at 1201 rising to 47px at 1500,
    then 9.5px at 1600. That is pre-existing and not what this case covers; it
    covers the capped range, where the two numbers are free to disagree and a
    change to either would silently reopen the jump.
    """
    hero_rail, hero_inset = _rail_geometry(
        'html[data-nav-page="home"] #homeView .home-landing-hero-inner'
    )
    dash_rail, dash_inset = _rail_geometry(
        'html[data-nav-page="home"] #homeView .home-dashboard-screen-inner'
    )
    hero_content = hero_rail - 2 * hero_inset
    dash_content = dash_rail - 2 * dash_inset
    assert hero_content == dash_content, (
        f"where both rails are cap-limited (measured {max(hero_rail, dash_rail):.0f}px "
        f"of viewport and up) screen 0 shows {hero_content}px of content "
        f"({hero_rail} rail less {hero_inset}px a side) against screen 1's "
        f"{dash_content}px ({dash_rail} less {dash_inset}px a side), so content "
        f"jumps {abs(hero_content - dash_content) / 2}px sideways on every "
        "pager snap -- move the rails together, or absorb the difference in the "
        "other screen's padding"
    )


def test_the_hero_reserves_the_scroll_hints_strip():
    """The hint gets a reserved band -- but only where the hero can afford it.

    Both halves are the guard. The hint is `z-index: 3` with live pointer-events
    over an absolutely-positioned band, so content sharing that band is
    un-clickable rather than merely overlapped. But the board panel is
    `overflow: hidden` around a chart with a hard `min-height: 132px`, so an
    UNCONDITIONAL reserve buys the band out of the panel: measured with the
    panel populated, it overflowed by 12/32/52/72px at viewport heights
    680/660/640/620 and pushed the "See both leaderboards" button outside its
    own clipping box, unreachable by `elementFromPoint`. That shipped once.

    So the reserve must be gated on viewport height, and the gate must be no
    higher than the height at which it measured free. An unconditional reserve
    fails here, which is the specific regression this case exists to catch.
    """
    strip = _hint_strip_px()
    assert strip >= _HINT_STRIP_FLOOR_PX, (
        f"the reserved band is {strip}px, under the {_HINT_STRIP_FLOOR_PX}px the "
        "hint occupies (56px tall at `bottom: 18px`), so it no longer clears it"
    )

    hero = 'html[data-nav-page="home"] #homeView .home-landing-hero'
    for block in _exact_rule_blocks(hero):
        body = _CSS_COMMENT.sub("", block)
        for prop, index in (("padding", 2), ("padding-block", 1), ("padding-bottom", 0)):
            for value in _declarations(block, prop):
                parts = _shorthand_parts(value)
                if len(parts) <= index:
                    continue
                assert "--home-hint-strip" in parts[index] or _px(parts[index]) < strip, (
                    f"an ungated hero rule reserves {parts[index]} at the bottom. "
                    "The reserve must live in the height-gated block: unconditional, "
                    "it clips the board panel's footer button at short viewports"
                )

    gated = [
        (condition, body)
        for condition, body in _at_rule_blocks(re.compile(r"min-height"))
        if "home-landing-hero" in body and "padding-bottom" in body
    ]
    assert gated, (
        "nothing reserves the hint's band behind a min-height gate any more -- "
        "re-point this case at however the reserve is now expressed, do not "
        "delete it; the hint sits on top of the copy column's CTAs without it"
    )
    for condition, body in gated:
        (declared,) = _declarations(body, "padding-bottom")
        assert "--home-hint-strip" in declared, (
            f"the gated reserve declares {declared!r} rather than the shared "
            "token, so it can drift from the band the hint actually occupies"
        )
        gate = _px(re.search(r"min-height:\s*([^)]+)\)", condition).group(1))
        assert gate <= _RESERVE_MAX_GATE_PX, (
            f"the reserve is gated at {gate}px, above the {_RESERVE_MAX_GATE_PX}px "
            "at which it measured free -- between the two the panel loses rows "
            "for a band nothing needed"
        )

    base = _exact_rule_blocks(".home-scroll-hint")
    assert base, "the unscoped scroll-hint rule was renamed or deleted"
    assert any("left: 50%" in _CSS_COMMENT.sub("", b) for b in base), (
        "the scroll hint no longer viewport-centres -- that is the anchor both "
        "layouts rely on now that the hero reserves a band instead of moving it"
    )

    # Any rule that re-anchors the hint horizontally, however it is spelled.
    offenders = []
    for match in re.finditer(r"(?m)^([^{}@/\n][^{}]*)\{", STYLES):
        selector = match.group(1).strip()
        if _HINT_SELECTOR_HINT not in selector and "homeScrollHint" not in selector:
            continue
        if selector == ".home-scroll-hint":
            continue  # the base rule IS the viewport-centring anchor
        index = STYLES.index("{", match.start())
        depth = 0
        while True:
            if STYLES[index] == "{":
                depth += 1
            elif STYLES[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        body = _CSS_COMMENT.sub("", STYLES[match.start() : index + 1])
        if any(anchor in body for anchor in _HORIZONTAL_ANCHORS):
            offenders.append(selector)
    assert not offenders, (
        f"{offenders} re-anchor the scroll hint horizontally. That is the fix "
        "this case exists to prevent: it moves the hint off the board card and "
        "onto the copy column's CTAs, where `elementFromPoint` returned the "
        "hint instead of the Discord link at four measured viewports -- and it "
        "needs half the rail as a hard-coded constant to do it"
    )


def test_the_board_ratio_leaves_the_headline_room():
    """The board may not grow so fast that the headline gains a third line.

    The copy column is whatever the board leaves, and the binding width is the
    WIDEST screen, not the narrowest: `gap: clamp(2.5rem, 5vw, 6.8rem)` keeps
    growing to 108.8px while the rail is pinned at 1444, so the column shrinks
    above ~1744 and bottoms out at 2176+.

    A ceiling of 1.35 once passed values that visibly wrap, because it had been
    measured against a table that stopped at 1920. The other way to get this
    wrong is to measure the copy column and call it the headroom: the headline
    also carries `max-width: 36rem`, and past ~1650px of viewport THAT is the
    narrower bound, so the column-only reading overstates the margin by however
    far the column clears 576px. See `_MAX_BOARD_GROW` for the current grid.
    Raising this means re-measuring at 2176 and up, against the cap, not editing
    the number.
    """
    block = _unique_block(
        'html[data-nav-page="home"] #homeView .home-landing-board', "flex"
    )
    (flex,) = _declarations(block, "flex")
    grow_text = _shorthand_parts(flex)[0]
    try:
        grow = float(grow_text)
    except ValueError:
        raise AssertionError(
            f"the board's desktop rule declares `flex: {flex}`, whose grow term "
            f"({grow_text!r}) is not a number -- this guard bounds the grow "
            "factor, so re-point it at whatever now controls the split"
        ) from None
    assert grow <= _MAX_BOARD_GROW, (
        f"the board grows at {grow}, above the measured {_MAX_BOARD_GROW} "
        "ceiling; the hero headline wraps to a third line at 2176px and wider"
    )


def test_the_ratio_grid_measures_the_headline_that_ships():
    """The measured grid beside the board's `flex` must name the CURRENT headline.

    This is the one assertion in this file that reads a comment ON PURPOSE,
    against `_strip_comments`' rule, because here the comment IS the artifact
    under guard: the ceiling is a measurement of one specific string, and a copy
    edit that leaves the grid behind produces a table that is precise,
    authoritative, and about a headline nobody ships any more. Nothing else can
    catch that -- the CSS still parses, the ratio still passes its bound, and the
    stale numbers read exactly like fresh ones.

    Pairs with `_MAX_BOARD_GROW`: that bounds the number, this bounds what the
    number was measured against.

    It reads a MARKER line rather than searching the block, and that is the
    whole difficulty of the case. The comment legitimately names the RETIRED
    headline as well -- the ceiling only means something stated against both --
    so `assert headline in block` is satisfied by the history: revert screen 0
    to "AI models finished" and the check still passes, because the grid quotes
    that string while explaining why the ratio was not raised. A guard that
    cannot fail on the exact edit it exists to catch is worse than none, since
    it reports the grid as verified. One `MEASURED-HEADLINE:` line, holding the
    only quoted string on it, is the smallest thing that cannot be satisfied by
    prose about some other headline.
    """
    from dashboard.backend.tests._frontend_source import APP_HTML

    accent = re.search(r"home-headline-line--2[^>]*>([^<]+)<", APP_HTML)
    assert accent, "screen 0 has no `home-headline-line--2` span for the grid to measure"
    headline = accent.group(1).strip()

    grid = next(
        (
            block
            for block in re.findall(r"/\*.*?\*/", STYLES, re.S)
            if "test_the_board_ratio_leaves_the_headline_room" in block
        ),
        None,
    )
    assert grid, (
        "the board's ratio comment no longer names its guard, so this test cannot "
        "find the grid -- re-point it at whatever now carries the measurements"
    )
    # Collapse the comment's own line wrapping first: the grid is prose in a CSS
    # block comment, so a re-flow can split the marker across lines without
    # changing a word. A guard that fails on re-indentation gets deleted by the
    # next person who touches it.
    flattened = " ".join(grid.split())
    marker = re.search(r'MEASURED-HEADLINE:\s*"([^"]*)"', flattened)
    assert marker, (
        'the ratio grid no longer carries a `MEASURED-HEADLINE: "..."` line. It '
        "is not decoration: the block also quotes the retired headline, so "
        "without the marker this case passes on that instead and reports a "
        "stale grid as measured. Re-add it naming whatever the grid measures."
    )
    measured = marker.group(1)
    assert measured == headline, (
        f"screen 0 ships the headline {headline!r}, but the ratio grid was "
        f"measured against {measured!r}. The grid is what licenses "
        "_MAX_BOARD_GROW, so re-measure it against the new copy (the comment "
        "carries the method) rather than editing the marker to match."
    )


def test_the_board_column_is_not_recapped_by_the_unscoped_rule():
    """`max-width: none` here is load-bearing, not a leftover.

    The unscoped `.home-landing-board` rule carries `max-width: 42rem`. This
    scoped rule used to mask it, so DELETING the cap here -- the obvious
    tidy-up, since no cap ever binds above 1200px -- does not uncap the column:
    it hands it to that 42rem and measured pins the board to 672px at
    1600/1920/2560, silently undoing most of the widening.
    """
    block = _unique_block(
        'html[data-nav-page="home"] #homeView .home-landing-board', "max-width"
    )
    (declared,) = _declarations(block, "max-width")
    assert declared == "none", (
        f"the board's desktop rule declares `max-width: {declared}` instead of "
        "`none`; if that is narrower than the column wants it re-caps the board"
    )
    unscoped = [b for b in _exact_rule_blocks(".home-landing-board") if _declarations(b, "max-width")]
    assert unscoped, (
        "the unscoped .home-landing-board no longer declares a max-width, so "
        "`max-width: none` above may now be removable -- verify and update both"
    )
