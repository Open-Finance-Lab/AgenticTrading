"""The My Agents card charts the run it is already watching.

Feedback, 2026-09-03: *"The waiting for 'Run Backtest' function is way too long
and often runs overtime. Maybe putting a little auto-updating interactive plot
on the side could make it less boring (and even runs overtime, some results
could be obtained)."*

Nothing here adds a data source. ``engine._publish_live_progress`` has always
written the whole ``equity_curve`` to the progress file every step -- its
docstring says "for live dashboard charting" -- and ``/backtest/status`` has
always served it. Two things stood between that payload and the card the user
is actually looking at:

1. ``advanceBacktestProgress()`` folded the poll into ``{step, totalSteps,
   ageSeconds, ...anchors}`` and dropped ``equity_curve`` on the floor, so the
   card path never saw a number it could plot.
2. ``renderAgentRunningActions()`` offered Configure and a disabled "Running…"
   pill, so during the wait this feature exists for there was no route to the
   live chart on the Backtest tab -- which has drawn this same curve all along.

The two-renderer rule from ``test_backtest_progress_card`` applies unchanged and
is the reason the spark and equity nodes are emitted even when empty: a full
re-render fires only when the *set* of running agents changes, so a node that
appears only once it has content can never be filled in mid-run.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_JS,
    css_blocks,
    fn_body,
    js_const,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_SPARK_HELPERS = (
    "function hashStringSeed(",
    "function renderAgentSparklineFromValues(",
    "function formatAgentMoney(",
    "function formatSignedMoney(",
    "function formatBacktestEta(",
    "function resolveBacktestEta(",
    "function resolveProgressAgeSeconds(",
    "function formatProgressStaleness(",
    "function formatStartupStaleness(",
    "function resolveRunningNotice(",
    "function deriveRunningProgress(",
)

#: A rising four-point curve off $1,000 of starting capital.
_CURVE = "[1000, 1010, 1025, 1043.2]"

_LIVE = (
    "{step: 84, totalSteps: 240, ageSeconds: 1, ageAt: Date.now(),"
    " firstStep: 4, firstStepAt: Date.now() - 184000,"
    f" equityCurve: {_CURVE}}}"
)


def _node(script: str) -> object:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# --- The fold: does the curve survive the poll? -------------------------------


def _fold(progress_js: str, previous_js: str = "null") -> object:
    script = "\n".join(
        [
            js_const("LIVE_SPARK_MAX_POINTS"),
            fn_body("function advanceBacktestProgress("),
            f"console.log(JSON.stringify(advanceBacktestProgress("
            f"{previous_js}, {progress_js}, Date.now())));",
        ]
    )
    return _node(script)


def test_fold_carries_the_equity_curve():
    """The defect in one assertion: the payload arrives, the fold discards it."""
    folded = _fold("{step: 3, total_steps: 10, equity_curve: "
                   "[{equity: 1000}, {equity: 1010}, {equity: 1025}]}")
    assert folded["equityCurve"] == [1000, 1010, 1025]


def test_fold_caps_the_curve_at_the_tail():
    """An hourly year is ~1,750 points. The card draws an 80px sparkline, so
    every point beyond the cap costs JSON parsing and a path segment nobody can
    resolve -- and this object is re-folded once a second for the whole run.

    Tail, not head: the newest points are the ones the user is waiting on.
    """
    curve = ", ".join(f"{{equity: {i}}}" for i in range(400))
    folded = _fold(f"{{step: 3, total_steps: 10, equity_curve: [{curve}]}}")
    assert len(folded["equityCurve"]) == 120
    assert folded["equityCurve"][-1] == 399


def test_fold_survives_a_payload_with_no_curve():
    """Normal for the opening ticks of every run, and for a file caught
    mid-rewrite. Empty, never null: the renderers test length, and a null here
    would make the sparkline branch throw instead of drawing nothing."""
    folded = _fold("{step: 3, total_steps: 10}")
    assert folded["equityCurve"] == []


def test_fold_still_refuses_a_tick_with_no_step():
    """Regression guard on the existing contract. A pre-confirmation entry must
    stay indeterminate -- spreading numbers onto a launch that is about to be
    refused is a bug this function already fixed once."""
    assert _fold("{equity_curve: [{equity: 1000}, {equity: 1010}]}") is None


def test_fold_ignores_non_numeric_equity_points():
    """The curve is JSON off disk, written by a subprocess mid-run. A null
    equity reaching the SVG path builder renders `LNaN,NaN` and blanks the
    whole card."""
    folded = _fold("{step: 3, total_steps: 10, equity_curve: "
                   "[{equity: 1000}, {equity: null}, {equity: 1025}]}")
    assert folded["equityCurve"] == [1000, 1025]


# --- The card -----------------------------------------------------------------


def _render(running_js: str) -> str:
    """`_render_raw` with an elapsed timer spread over it.

    It used to interpolate the module-level `_LIVE` and drop its own argument on
    the floor, which is invisible while every caller passes `_LIVE` and silently
    vacuous for the first one that does not -- including the long-curve case
    below, whose entire point is a curve other than `_LIVE`'s.
    """
    return _render_raw(f"{{elapsedSeconds: 185, ...{running_js}}}")


def _render_raw(running_js: str) -> str:
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            *[fn_body(signature) for signature in _SPARK_HELPERS],
            fn_body("function renderAgentRunningBody("),
            "console.log(JSON.stringify(renderAgentRunningBody("
            f"{{agent_id: 'a1'}}, {running_js})));",
        ]
    )
    return _node(script)


def test_card_draws_the_live_curve():
    html = _render(_LIVE)
    assert 'data-running-spark="a1"' in html
    assert "<svg" in html.split('data-running-spark="a1">')[1]
    # A real polyline, not the flat placeholder dash.
    assert "agent-card-sparkline--empty" not in html


def test_card_spark_node_exists_before_any_data():
    """Same rule the detail and staleness nodes follow: the per-second patch
    finds nodes by attribute, and only a change to the set of running agents
    re-renders. A node conjured on first data never appears at all."""
    html = _render_raw("{elapsedSeconds: 2}")
    assert 'data-running-spark="a1"></div>' in html
    blocks = css_blocks(".agent-card-running-spark:empty")
    assert any("display: none" in block for block in blocks), blocks


def test_card_reports_current_equity_against_the_start():
    """"Some results could be obtained" -- the number is the result; the curve
    is only its shape. A sparkline with no value beside it cannot be read."""
    html = _render(_LIVE)
    equity = html.split('data-running-equity="a1">')[1].split("</p>")[0]
    assert "$1,043.20" in equity
    assert "+4.32%" in equity


def test_card_marks_a_losing_run_as_losing():
    losing = _LIVE.replace(_CURVE, "[1000, 980, 960.5]")
    html = _render_raw(f"{{elapsedSeconds: 185, ...{losing}}}")
    assert "is-neg" in html
    equity = html.split('data-running-equity="a1">')[1].split("</p>")[0]
    assert "-3.95%" in equity


def test_card_equity_node_is_empty_and_hidden_before_any_data():
    html = _render_raw("{elapsedSeconds: 2}")
    assert 'data-running-equity="a1"></p>' in html
    blocks = css_blocks(".agent-card-running-equity:empty")
    assert any("display: none" in block for block in blocks), blocks


def test_card_does_not_draw_a_curve_from_a_single_point():
    """One published step is one point. Two are needed for a line, and the
    helper's placeholder dash is the honest render of "not yet"."""
    single = _LIVE.replace(_CURVE, "[1000]")
    html = _render_raw(f"{{elapsedSeconds: 185, ...{single}}}")
    assert "agent-card-sparkline--empty" in html


# --- The way out while it runs ------------------------------------------------


def _actions() -> str:
    script = "\n".join(
        [
            "function escapeHtml(s) { return String(s); }",
            fn_body("function renderAgentRunningActions("),
            "console.log(JSON.stringify(renderAgentRunningActions({agent_id: 'a1'})));",
        ]
    )
    return _node(script)


def test_running_card_offers_a_route_to_the_live_chart():
    """The Backtest tab has drawn this run all along; during the wait the card
    offered no way to reach it. Configure and a disabled pill were the only
    controls on screen for the entire run."""
    html = _actions()
    assert "agent-view-live-btn" in html
    assert 'data-agent-id="a1"' in html


def test_running_card_keeps_the_run_button_disabled():
    """Regression guard: the status pill stays. Turning it back into a live
    control here would bypass openRunBacktestModal, the single funnel that
    refuses a launch once this browser is at the concurrency limit."""
    html = _actions()
    assert "agent-card-cta--disabled" in html


def test_live_chart_button_is_wired_to_the_backtest_surface():
    """The handler must pass the *running* run id. openAgentInBacktest falls
    back to resolveLatestAgentRunId when given none, which during a run pins
    the previous finished run -- the one view the user did not ask for."""
    source = fn_body("function renderAgentCards(")
    assert ".agent-view-live-btn" in source
    assert "openAgentInBacktest" in source


# --- The run history that was there all along ---------------------------------
#
# Feedback: *"Backtest history are not saved. It only returns the most recent
# graph if I'm not misunderstanding(?)"* -- and that "(?)" is the finding. The
# history is saved (``agent_runs``, read back through
# ``AgentService.list_external_runs``) and ``populateBacktestRunSelector`` has
# always listed every run with localStorage persistence. It rendered as a bare
# <select> between the chart title and the 1D/1W/1M/ALL buttons, so it reads as
# a chart control. A user concluded the data did not exist.


def test_run_selector_carries_a_visible_label():
    """aria-label alone is invisible to the sighted user who could not find it,
    which is the entire reported defect."""
    markup = pathlib.Path("dashboard/frontend/app.html").read_text(encoding="utf-8")
    group = markup.split('id="backtestRunHistory"')[1].split("</div>")[0]
    assert 'for="backtestRunSelect"' in group
    assert "Run history" in group


def test_run_selector_is_not_inside_the_time_range_row():
    """Grouped with 1D/1W/1M/ALL it inherits their meaning: another way to
    reframe the chart you are already looking at, rather than a way to open a
    different run."""
    markup = pathlib.Path("dashboard/frontend/app.html").read_text(encoding="utf-8")
    controls = markup.split('class="chart-controls"')[1].split("</div>\n                    </div>")[0]
    history_at = controls.find("backtestRunHistory")
    first_time_btn = controls.find("time-btn")
    assert history_at != -1 and first_time_btn != -1
    assert history_at < first_time_btn


def test_selector_visibility_has_one_owner():
    """The label must disappear with the selector it names. Toggling
    `select.hidden` alone strands "Run history" over nothing on a session with
    no runs -- so every site that hid the select must hide the group instead."""
    source = (
        fn_body("function populateBacktestRunSelector(")
        + fn_body("function attachToLiveBacktest(")
    )
    assert "setBacktestRunSelectorVisible" in source
    # The old direct writes are what this replaces; leaving one behind puts the
    # group and the select back under separate owners.
    assert "select.hidden =" not in source
    assert "runSelect.hidden =" not in source


def test_a_zero_capital_run_reports_no_percentage():
    """PR #449 made `0` a legal backtest capital and describes the result as "a
    real, flat, 0.00% run", so the opening equity this card divides by can now
    genuinely be zero.

    The `&& opening` guard in deriveRunningProgress is what stops that: removed,
    the same input renders `$0.00 · NaN%` (verified directly against the shipped
    function, not reasoned about). A dollar figure alone is the honest render --
    there is no percentage of nothing.
    """
    zero = _LIVE.replace(_CURVE, "[0, 0, 0]")
    html = _render_raw(f"{{elapsedSeconds: 185, ...{zero}}}")
    equity = html.split('data-running-equity="a1">')[1].split("</p>")[0]
    assert equity == "$0.00", equity
    assert "NaN" not in html
    # Flat is not losing: a zero-capital run must not paint itself red.
    assert "is-neg" not in html


# --- The baseline the percentage is measured against --------------------------
#
# `advanceBacktestProgress` trims the curve to LIVE_SPARK_MAX_POINTS, so the
# first point of what the card receives is the opening equity only for the first
# 120 published steps. Past that it is the equity 120 steps ago, and a gain
# measured against it is a rolling-window return wearing the run's label.


def test_fold_carries_the_runs_true_opening():
    """Read before the trim, carried beside it. The trim is what destroys it."""
    curve = ", ".join(f"{{equity: {1000 + i}}}" for i in range(400))
    folded = _fold(f"{{step: 3, total_steps: 10, equity_curve: [{curve}]}}")
    assert folded["openingEquity"] == 1000
    # ...and the trimmed curve genuinely no longer holds it.
    assert folded["equityCurve"][0] == 1280


def test_opening_is_the_first_plottable_point_not_the_first_point():
    """Same filter the curve gets: a null opening reaching the division renders
    `NaN%`, and `Number(null)` is a finite 0 that would render `Infinity%`."""
    folded = _fold(
        "{step: 3, total_steps: 10, equity_curve: "
        "[{equity: null}, {equity: 1000}, {equity: 1010}]}"
    )
    assert folded["openingEquity"] == 1000


def test_opening_is_null_before_any_point_arrives():
    """Normal for the opening ticks. Null, not 0: a zero baseline is a legal
    run since PR #449, so the two must stay distinguishable."""
    folded = _fold("{step: 3, total_steps: 10}")
    assert folded["openingEquity"] is None


def test_card_measures_gain_against_the_opening_not_the_window():
    """The defect in one assertion. A long run down 8% overall, up over its last
    120 bars, reported `+0.33%` in green -- the number, the sign and the colour
    all taken from a window the label never mentions."""
    windowed = (
        "{step: 400, totalSteps: 500, ageSeconds: 1, ageAt: Date.now(),"
        " firstStep: 4, firstStepAt: Date.now() - 184000,"
        " openingEquity: 1000, equityCurve: [917, 918, 920]}"
    )
    html = _render(windowed)
    equity = html.split('data-running-equity="a1">')[1].split("</p>")[0]
    assert "$920.00" in equity
    assert "-8.00%" in equity
    # The colour is derived from the same gain, so it moves with it or the card
    # paints a loss green.
    assert "is-neg" in html


def test_card_falls_back_to_the_curve_head_without_a_carried_opening():
    """One poll's worth of back-compat: an entry folded by the previous build
    has no `openingEquity`, and for a run short enough to be untrimmed the two
    baselines are the same number anyway."""
    html = _render(_LIVE)
    equity = html.split('data-running-equity="a1">')[1].split("</p>")[0]
    assert "+4.32%" in equity


# --- The box the sparkline is told to fill ------------------------------------


def test_sparkline_stretches_to_a_box_wider_than_its_viewbox():
    """`width: 100%` over a `viewBox="0 0 80 36"` does nothing on its own: the
    default `xMidYMid meet` scales by min(W/80, H/36), which is 1 for every box
    wider than 80px at this height. The curve drew at 80px, centred, with ~110px
    of blank card on each side of it."""
    source = fn_body("function renderAgentSparklineFromValues(")
    # Both SVGs -- the curve and the placeholder dash share the box.
    assert source.count('preserveAspectRatio="none"') == 2
    # Non-uniform scale thickens a stroke along one axis only, so a steep step
    # would draw several times heavier than a flat one.
    assert source.count('vector-effect="non-scaling-stroke"') == 2
    blocks = css_blocks(".agent-card-running-spark .agent-card-sparkline")
    assert any("width: 100%" in block for block in blocks), blocks


# --- The narrow-width layout --------------------------------------------------


def test_narrow_layout_widens_the_group_not_the_select():
    """The select sits in a labelled flex group now. `flex: 1 0 100%` on the
    select claims the whole group box while a nowrap label and an 8px gap still
    need room in it, and `flex-shrink: 0` forbids the give -- the performance
    card overflows sideways at phone width. Worse, that rule is the most
    specific one targeting the select, so a later, narrower breakpoint trying to
    relax it loses silently."""
    select = css_blocks(".performance-card .chart-controls .backtest-run-select")
    assert select, "the narrow-width select rule is gone entirely"
    assert not any("flex: 1 0 100%" in block for block in select), select
    group = css_blocks(".performance-card .chart-controls .backtest-run-history")
    assert any("flex: 1 0 100%" in block for block in group), group


# --- Which run the live view attaches to --------------------------------------


def _status_url(live_run_id: str) -> str:
    script = "\n".join(
        [
            "const API_BASE = 'http://x';",
            fn_body("function backtestStatusUrl("),
            f"console.log(JSON.stringify(backtestStatusUrl({live_run_id})));",
        ]
    )
    return _node(script)


def test_status_url_carries_the_run_when_one_is_named():
    assert _status_url("'run-b/1'") == (
        "http://x/backtest/status?live_run_id=run-b%2F1"
    )


def test_status_url_omits_the_parameter_when_no_run_is_named():
    assert _status_url("null") == "http://x/backtest/status"


def test_loading_the_backtest_tab_asks_about_the_run_it_was_given():
    """`/backtest/status` with no `live_run_id` answers with the *newest* active
    slot this session owns. openAgentInBacktest pinned the running run and then
    called loadData(), which asked the unqualified question and overwrote the
    pin -- so with two agents running, "View live chart" on A opened B's chart,
    under B's "Running..." option, with HTTP 200 throughout."""
    source = strip_comments(fn_body("async function loadData("))
    assert "backtestStatusUrl(liveRunId)" in source
    opener = strip_comments(fn_body("async function openAgentInBacktest("))
    assert "loadData({" in opener and "liveRunId" in opener


def test_every_status_poll_goes_through_the_one_url_builder():
    """The poller passed the id and loadData did not: two spellings of the same
    request, one of them wrong, and nothing at either call site to show which."""
    builder = strip_comments(fn_body("function backtestStatusUrl("))
    # The qualified spelling and the bare one, both inside the builder...
    assert builder.count("/backtest/status") == 2
    # ...and nowhere else in app.js. Counted rather than searched-and-removed: a
    # `.replace()` of the canonical literal deletes a *duplicate* of it too, so
    # the guard would go green on precisely the regression it exists to catch.
    assert strip_comments(APP_JS).count("/backtest/status") == 2


# --- The selector and the label that names it ---------------------------------


def _visibility(visible: str, *, with_group: bool) -> dict:
    group = "backtestRunHistory: {hidden: false}," if with_group else ""
    script = "\n".join(
        [
            f"const nodes = {{backtestRunSelect: {{hidden: false}}, {group}}};",
            "const document = {getElementById: (id) => nodes[id] || null};",
            fn_body("function setBacktestRunSelectorVisible("),
            f"setBacktestRunSelectorVisible({visible});",
            (
                "console.log(JSON.stringify({select: nodes.backtestRunSelect.hidden,"
                " group: nodes.backtestRunHistory"
                " ? nodes.backtestRunHistory.hidden : null}));"
            ),
        ]
    )
    return _node(script)


def test_hiding_the_selector_hides_the_group_that_holds_it():
    assert _visibility("false", with_group=True)["group"] is True


def test_showing_the_selector_clears_a_stale_hidden_on_the_select():
    """The reason the select is written at all: an older cached app.html can
    have left `hidden` on it, inside a group that is now visible."""
    shown = _visibility("true", with_group=True)
    assert shown["group"] is False
    assert shown["select"] is False


def test_hiding_falls_back_to_the_select_when_the_group_is_missing():
    """Stale markup is exactly the case the unconditional `select.hidden = false`
    was written for, and exactly the case it broke: with no group, hide() hid
    nothing and *unhid* the select -- which populateBacktestRunSelector has just
    emptied -- so a session with no runs rendered a blank dropdown."""
    assert _visibility("false", with_group=False)["select"] is True
    assert _visibility("true", with_group=False)["select"] is False


def _populate(*, pin: str, select_value: str, running_id: str, runs: str) -> dict:
    """Run the shipped populateBacktestRunSelector over a stubbed DOM."""
    script = "\n".join(
        [
            js_const("SELECTED_BACKTEST_RUN_KEY"),
            f"const store = {{[SELECTED_BACKTEST_RUN_KEY]: {pin}}};",
            (
                "const localStorage = {getItem: (k) => store[k] ?? null,"
                " setItem: (k, v) => { store[k] = v; },"
                " removeItem: (k) => { delete store[k]; }};"
            ),
            f"const select = {{value: {select_value}, innerHTML: '', hidden: false}};",
            "const group = {hidden: false};",
            (
                "const document = {getElementById: (id) => id === 'backtestRunSelect'"
                " ? select : (id === 'backtestRunHistory' ? group : null)};"
            ),
            "function escapeHtml(s) { return String(s); }",
            "function formatBacktestRunPrimary(r) { return r.agent_name || 'Agent'; }",
            "function formatBacktestRunLabel(r) { return r.run_id; }",
            (
                "function getBacktestLaunchConfig() {"
                " return {agentName: 'A', startedAt: ''}; }"
            ),
            fn_body("function setBacktestRunSelectorVisible("),
            fn_body("function populateBacktestRunSelector("),
            f"populateBacktestRunSelector({runs}, {{runningId: {running_id}}});",
            (
                "console.log(JSON.stringify({selected: select.value,"
                " pin: store[SELECTED_BACKTEST_RUN_KEY] ?? null,"
                " options: select.innerHTML}));"
            ),
        ]
    )
    return _node(script)


_FINISHED = "[{run_id: 'X', agent_name: 'A', created_at: '2026-09-01'}]"


def test_an_explicit_pin_outranks_whatever_the_tab_was_showing():
    """The other half of the wrong-run defect, and the one that survives fixing
    the status call. openAgentInBacktest writes the pin and navigates; the
    Backtest tab's <select> is not destroyed in between, so it still holds the
    finished run the user was looking at ten seconds ago. Reading the DOM first
    made that stale value outrank "open this run" -- and then rewrote the pin to
    match, so the live view was never attached."""
    result = _populate(
        pin="'runA'", select_value="'X'", running_id="'runA'", runs=_FINISHED
    )
    assert result["selected"] == "runA"
    assert result["pin"] == "runA"


def test_the_dom_value_still_wins_when_nothing_is_pinned():
    """The fallback is real: localStorage can be unavailable or cleared, and the
    selection the user is looking at is a better answer than the newest run."""
    result = _populate(
        pin="undefined", select_value="'X'", running_id="null", runs=_FINISHED
    )
    assert result["selected"] == "X"
