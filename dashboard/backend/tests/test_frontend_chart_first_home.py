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
_HOME_JS = (_FRONTEND / "home-page.js").read_text(encoding="utf-8")
_LEADERBOARD_JS = (_FRONTEND / "js" / "leaderboard.js").read_text(encoding="utf-8")

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
    """The board's rows do NOT share a capital base, so the chart cannot
    assume one.

    `dashboard/config/leaderboard.json` says `initial_capital: 10000` while
    every published curve was computed at $100,000, and `_find_cached_run`
    (`service.py:615`) does not key on `initial_equity` -- so one
    `?refresh=true` recomputes the five `auto_compute` baselines at $10k and
    leaves the seven model entries at $100k (issue #365, open). Plotted in
    dollars that renders as a 10x scale break: models near 100000, the
    reference baselines flat on the floor at 10000.

    This is the case the old fixture could not express, because it hardcoded
    one `initial_equity` for every row.
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
    assert "if (!series.length) return null;" in body, (
        "no series must mean no canvas -- not an empty canvas"
    )
    # The insertion is guarded, not unconditional at module scope.
    assert "document.createElement('canvas')" in body or "<canvas" in body
    assert "typeof window.Chart" in body, (
        "Chart.js is deferred; the render path must tolerate it not having landed"
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
    """
    assert "toFixed(2)" in _extract(_HOME_JS, "homeFormatReturnPct"), (
        "the rank list's formatter changed -- re-check the tooltip's precision"
    )
    body = _extract(_HOME_JS, "renderHomeLeaderboardChart")
    assert "(c.parsed.y * 100).toFixed(2)" in body, (
        "the tooltip must read in the same precision as the row beside it"
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


def test_the_series_style_helper_is_an_explicit_cross_file_export():
    """The same seam as the curve builder: home-page.js reads this off `window`
    and falls back to a transparent swatch when it is missing, so a rename
    degrades to colourless rows rather than an error. Pinned from both sides.
    """
    assert "window.getSeriesStyle = getSeriesStyle;" in _LEADERBOARD_JS
    assert "window.getSeriesStyle" in _HOME_JS
