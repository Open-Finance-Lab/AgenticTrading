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

from dashboard.backend.tests._frontend_source import css_blocks

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


def test_the_screen_zero_lede_is_a_fact_then_a_call_to_action():
    """The old sentence did two jobs at once -- glossing "agent" AND pre-empting
    "is my agent on this list?" -- which is why it read as neither marketing nor
    a CTA. The no-entry fact is already stated on the board itself
    ("AI models only - ranked by return"), so the lede is freed to be one plain
    thing. The gloss drops on this surface: the reader is signed in and inside
    the app, where the word is glossed throughout.
    """
    from dashboard.backend.tests._frontend_source import APP_HTML

    html = re.sub(r"<!--.*?-->", "", APP_HTML, flags=re.DOTALL)
    assert (
        "See how the AI models did. Then test your own idea on the same days."
        in html
    )
    assert "in a test of its own" not in html
    # The fact it used to carry must still be on screen, on the board making the
    # claim -- otherwise this is a deletion, not a split.
    assert "AI models only" in html


def test_the_series_style_helper_is_an_explicit_cross_file_export():
    """The same seam as the curve builder: home-page.js reads this off `window`
    and falls back to a transparent swatch when it is missing, so a rename
    degrades to colourless rows rather than an error. Pinned from both sides.
    """
    assert "window.getSeriesStyle = getSeriesStyle;" in _LEADERBOARD_JS
    assert "window.getSeriesStyle" in _HOME_JS


_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _declared_max_width(prelude: str) -> str | None:
    """The `max-width` this selector declares, read from code and not prose.

    Comments are stripped first because the two rules guarded below now carry
    long notes that quote their own numbers ("1720 here put the hero rail at
    x=93", "caps its own rail at 1500") -- exactly the trap this file's
    `_strip_comments` docstring describes for the JS side. Without the strip,
    editing one rail to 1720 and leaving the note behind keeps the guard green
    on the regression it exists to catch.

    Returns the first block that declares one: `css_blocks` also returns the
    narrow-viewport override of each selector, and only one of the two sets a
    rail width.
    """
    for block in css_blocks(prelude):
        match = re.search(r"\bmax-width:\s*([^;]+);", _CSS_COMMENT.sub("", block))
        if match:
            return match.group(1).strip()
    return None


def test_the_two_pager_screens_share_one_content_rail():
    """Screen 0's rail and screen 1's rail are one number, declared twice.

    `#homeView` is a scroll-snap pager: the two screens are never on-screen
    together, so a rail mismatch is invisible in any single view and shows up
    only as the content sliding sideways on every snap. That is precisely the
    kind of defect nobody files. Measured at 1920 while the hero rail was
    1720px against screen 1's 1500px: x=93 vs x=203, a 110px jump.

    Kept as an equality between two independently-read declarations rather than
    a literal, so widening the app later stays a one-line change that this case
    forces you to make in both places -- pinning `1500px` here would just make
    the guard the third thing to update.
    """
    hero = _declared_max_width(
        'html[data-nav-page="home"] #homeView .home-landing-hero-inner'
    )
    dash = _declared_max_width(
        'html[data-nav-page="home"] #homeView .home-dashboard-screen-inner'
    )
    assert hero and dash, (
        "one of the two pager rails no longer declares a max-width where this "
        f"guard reads it (hero={hero!r}, dashboard={dash!r}) -- re-point it at "
        "however the rail is now expressed, do not delete the case"
    )
    assert hero == dash, (
        f"screen 0's rail is {hero} but screen 1's is {dash}, so the content "
        "shifts sideways on every pager snap -- move both or neither"
    )


def test_the_scroll_hint_steps_out_of_the_board_column():
    """The hint is centred on the VIEWPORT, and the hero is two columns.

    Fine while the hero was one column; beside a board card it puts "SEE YOUR
    DASHBOARD" on top of the card -- measured 165x42px of overlap at 1920, and
    the hint's own `z-index: 3` means it wins the pixels. The base rule keeps
    viewport-centring because below 1201px the hero really is one column.

    Both halves are asserted: an override that sets `left` but leaves the base
    `translateX(-50%)` in place pulls the hint a half-width back toward the
    card, which is the natural way to half-fix this.
    """
    base = [_CSS_COMMENT.sub("", block) for block in css_blocks(".home-scroll-hint")]
    assert any("left: 50%" in block for block in base), (
        "the scroll hint no longer viewport-centres by default -- the stacked "
        "layout below 1201px relies on that; re-point this case if the anchor moved"
    )
    override = [
        _CSS_COMMENT.sub("", block)
        for block in css_blocks(
            'html[data-nav-page="home"] #homeView .home-scroll-hint'
        )
    ]
    assert override, (
        "the two-column override is gone, so the hint is viewport-centred beside "
        "the board card again -- it overlapped it by 165x42px at 1920 before this"
    )
    assert any(
        "left:" in block and "transform: none" in block for block in override
    ), (
        "the override must reset BOTH `left` and the base rule's "
        "`translateX(-50%)`; resetting only `left` leaves the hint pulled a "
        f"half-width back over the card (blocks: {override!r})"
    )
