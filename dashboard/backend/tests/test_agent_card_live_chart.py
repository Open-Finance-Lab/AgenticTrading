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

from dashboard.backend.tests._frontend_source import css_blocks, fn_body, js_const

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
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            *[fn_body(signature) for signature in _SPARK_HELPERS],
            fn_body("function renderAgentRunningBody("),
            "console.log(JSON.stringify(renderAgentRunningBody("
            f"{{agent_id: 'a1'}}, {{elapsedSeconds: 185, ...{_LIVE}}})));",
        ]
    )
    return _node(script)


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
