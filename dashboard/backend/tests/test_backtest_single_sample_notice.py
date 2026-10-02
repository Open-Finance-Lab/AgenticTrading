"""A finished model-driven run is labelled as one sample (#602, option 1).

Three DeepSeek V4 Pro reruns with identical inputs and pinned sampling
diverged on the first bar and finished at -1.25%, -0.48% and -0.37% (#539).
One curve is therefore one draw, and the chart says so. A rule-based curve is
repeatable and gets no note.

The note describes the curve, so the chart path owns it: these drive the real
`loadHistoricalBacktestSurfaces` -> `initializeCharts` path over a fake DOM.
Each run is built by the list route's own serializer
(`_run_metadata_response(...).model_dump()`), so the explicit nulls and the
server's `decision_provenance` verdict are exactly what production sends. A
hand-written fixture omits keys the route always sends, which is the
mismatch that once hid the Sampling row in production while source-shape
tests stayed green.
"""
import json
import re
import shutil

import pytest

from dashboard.backend.api.routers.backtests import _run_metadata_response
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    fn_body,
    js_const,
    run_node,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_NOTICE = "chartSingleSampleNotice"
_PINNED_SAMPLING = {
    "temperature": 0,
    "reasoning_effort": "disabled",
    "policy": "pinned_v1",
    "model": "deepseek/deepseek-v4-pro",
}


def _list_row(run_id="run_1", *, llm_model, llm_calls, llm_decisions, metadata):
    """One run exactly as `GET /api/backtest/runs` answers with it."""
    row = _run_metadata_response(
        {
            "run_id": run_id,
            "agent_name": "Agent",
            "mode": "backtest",
            "start_date": "2026-09-07",
            "end_date": "2026-09-11",
            "initial_equity": 100_000,
            "num_trades": 4,
            "created_at": "2026-10-01T23:00:00",
            "llm_model": llm_model,
            "llm_calls": llm_calls,
            "llm_decisions": llm_decisions,
            "metadata": metadata,
        }
    )
    return row.model_dump(mode="json")


def _model_requested(**extra):
    """Metadata of a run that asked for the model under a recorded ceiling."""
    return {
        "decision_steps": 35,
        "requested_decision_source": "llm",
        "decision_source": "llm",
        "llm_max_output_tokens": 2000,
        "llm_sampling": _PINNED_SAMPLING,
        **extra,
    }


_RUNS = {
    "pinned_model_run": _list_row(
        llm_model="deepseek/deepseek-v4-pro",
        llm_calls=35,
        llm_decisions=35,
        metadata=_model_requested(),
    ),
    "partial_run": _list_row(
        llm_model="deepseek/deepseek-v4-pro",
        llm_calls=35,
        llm_decisions=12,
        metadata=_model_requested(),
    ),
    # Every response was billed and unusable, so every step traded
    # rule-based. It carries every field that says "asked for a model": the
    # model name, the calls, the ceiling and the sampling block.
    "total_fallback": _list_row(
        llm_model="deepseek/deepseek-v4-pro",
        llm_calls=35,
        llm_decisions=0,
        metadata=_model_requested(),
    ),
    # A deterministic SMA agent driven through the SDK. External runs count
    # every submitted decision as an `llm_call` and record no decision count.
    "external_sdk_run": _list_row(
        "ext_1",
        llm_model="local-model",
        llm_calls=35,
        llm_decisions=0,
        metadata=None,
    ),
    "rule_based_run": _list_row(
        llm_model="rule-based",
        llm_calls=0,
        llm_decisions=0,
        metadata={"decision_steps": 35, "decision_source": "rule_based"},
    ),
}


def test_the_rows_reach_the_branch_each_case_is_named_for():
    """The cases below are only as good as the verdicts these rows carry."""
    assert {name: run["decision_provenance"] for name, run in _RUNS.items()} == {
        "pinned_model_run": "llm",
        "partial_run": "partial",
        "total_fallback": "rule_based",
        "external_sdk_run": "unknown",
        "rule_based_run": "rule_based",
    }
    # The route sends these keys as explicit nulls on a rule-based row, so a
    # gate written as `!== undefined` would label every rule-based run.
    rule_based = _RUNS["rule_based_run"]
    for key in ("llm_max_output_tokens", "llm_sampling", "llm_execution"):
        assert key in rule_based and rule_based[key] is None


# Stubs for everything the chart path calls that is not under test. Chart.js
# and the comparison model are replaced; the notice, the comparison state and
# the request tokens run as shipped.
_HARNESS = [
    "const print = console.log.bind(console);",
    "console.log = () => {};",
    "console.warn = () => {};",
    "const cells = {};",
    "const el = (id) => cells[id] || (cells[id] = {",
    "  id, hidden: undefined, dataset: {},",
    "  replaceChildren() {}, getContext() { return {}; },",
    "});",
    "const document = { getElementById: el };",
    "const window = { BacktestComparison: { buildModel: () => ({ columns: [] }) } };",
    "class Chart { destroy() {} }",
    "const API_BASE = '';",
    "let chartResponse = null;",
    "const API = { get: () => chartResponse };",
    "let liveBacktestChartActive = false;",
    "let backtestSurfaceRequestSeq = 0;",
    "let backtestChartData = null;",
    "let chartInstance = null;",
    "const filterIfindChartSeries = (series) => series;",
    "let renderPerformanceComparison = () => {};",
    "const renderPerformanceLegend = () => {};",
    "const clearTradingLog = () => {};",
    "const loadTradingLogForRun = async () => {};",
    js_const("LLM_DECISION_SOURCE"),
    fn_body("function curveIsOneModelSample("),
    fn_body("function renderChartSingleSampleNotice("),
    fn_body("function setPerformanceComparisonState("),
    fn_body("function clearPerformanceComparison("),
    fn_body("function beginBacktestSurfaceRequest("),
    fn_body("function isCurrentBacktestSurfaceRequest("),
    fn_body("function initializeCharts()"),
    fn_body("async function loadHistoricalBacktestSurfaces("),
]

_CHART_RESPONSES = {
    "paint": "Promise.resolve({ series: [{}], timestamps: [], x_labels: [] })",
    "empty": "Promise.resolve({ series: [], timestamps: [], x_labels: [] })",
    "fail": "Promise.reject(new Error('chart-data 502'))",
    "pending": "new Promise(() => {})",
}


def _load(run: dict, chart: str = "paint", *, setup: str = "") -> dict:
    """Select `run` and load its chart over a note a previous run left up.

    Returns the note's `hidden` state while the chart request is in flight and
    once it has settled, plus the comparison region's state. The note starts
    visible, so a path that forgets to hide it shows up as `False`.
    """
    script = "\n".join(
        [
            *_HARNESS,
            setup,
            f"el('{_NOTICE}').hidden = false;",
            f"chartResponse = {_CHART_RESPONSES[chart]};",
            "chartResponse.catch(() => {});",
            f"const run = {json.dumps(run)};",
            "window.SELECTED_RUN = run;",
            "loadHistoricalBacktestSurfaces(run);",
            f"const during = el('{_NOTICE}').hidden;",
            # A macrotask runs after every settled promise's callbacks.
            "setTimeout(() => print(JSON.stringify({",
            f"  during, after: el('{_NOTICE}').hidden,",
            "  state: el('performanceComparison').dataset.state,",
            "})), 0);",
        ]
    )
    return run_node(script)


def test_markup_ships_hidden_beside_the_chart():
    assert f'id="{_NOTICE}"' in APP_HTML
    tag = APP_HTML[APP_HTML.index(f'id="{_NOTICE}"') - 3 :].split(">", 1)[0]
    assert "hidden" in tag
    # Above the chart, not inside "Show advanced details": the label exists
    # to be read before the curve is.
    assert APP_HTML.index(f'id="{_NOTICE}"') < APP_HTML.index('id="performanceChart"')


@pytest.mark.parametrize("name", ["pinned_model_run", "partial_run"])
def test_a_curve_the_model_drove_is_labelled_once_it_is_painted(name):
    """Pinned sampling included: it is the run a reader most expects to repeat."""
    state = _load(_RUNS[name])
    assert state == {"during": True, "after": False, "state": "loading"}


@pytest.mark.parametrize(
    "name", ["total_fallback", "external_sdk_run", "rule_based_run"]
)
def test_a_repeatable_curve_is_never_labelled(name):
    """Model fields on the row are not enough: these curves would repeat.

    The total fallback and the SDK run both carry `llm_calls > 0`, and the
    fallback also carries the output ceiling and a sampling block. All of
    those record what was *asked* for. Only the server's verdict says what
    drove the curve.
    """
    assert _load(_RUNS[name])["after"] is True


def test_a_chart_still_loading_carries_no_note():
    """The previous run's curve is still painted; the new run's note waits."""
    state = _load(_RUNS["pinned_model_run"], chart="pending")
    assert state["during"] is True
    assert state["after"] is True


def test_a_chart_answer_with_no_series_leaves_no_note():
    """initializeCharts returns before painting, so nothing new is on screen."""
    assert _load(_RUNS["pinned_model_run"], chart="empty")["after"] is True


def test_a_failed_chart_load_leaves_no_note():
    state = _load(_RUNS["pinned_model_run"], chart="fail")
    assert state["after"] is True
    assert state["state"] == "error"


def test_a_paint_that_throws_after_labelling_leaves_no_note():
    """The loader's catch destroys the chart, so it must take the note too.

    The note goes up as soon as the curve is drawn, before the comparison
    table renders. If that render throws, the request's catch tears the chart
    down, and the loading-time clear is long past.
    """
    state = _load(
        _RUNS["pinned_model_run"],
        setup="renderPerformanceComparison = () => { throw new Error('boom'); };",
    )
    assert state["after"] is True
    assert state["state"] == "error"


def test_initialize_charts_hides_the_note_on_its_own_early_returns():
    """Its own contract, not the loader's: no paint, no note, whoever calls it."""
    hidden = run_node(
        "\n".join(
            [
                *_HARNESS,
                f"el('{_NOTICE}').hidden = false;",
                "window.SELECTED_RUN = { decision_provenance: 'llm' };",
                "backtestChartData = { series: [] };",
                "initializeCharts();",
                f"print(JSON.stringify(el('{_NOTICE}').hidden));",
            ]
        )
    )
    assert hidden is True


@pytest.mark.parametrize("state", ["loading", "live", "empty", "error"])
def test_every_comparison_clear_hides_the_note(state):
    """Covers a running backtest ('live') and a cleared selection ('empty')."""
    hidden = run_node(
        "\n".join(
            [
                *_HARNESS,
                f"el('{_NOTICE}').hidden = false;",
                f"clearPerformanceComparison('{state}', '');",
                f"print(JSON.stringify(el('{_NOTICE}').hidden));",
            ]
        )
    )
    assert hidden is True


def test_only_the_chart_paint_can_show_the_note():
    """One writer, and one caller that passes it a run.

    The note was first toggled from the config-panel renderer, which repaints
    before the chart request settles and again whether or not a curve ever
    arrives. A second caller passing a run would rebuild that bug.
    """
    code = strip_comments(APP_JS)
    writer = strip_comments(fn_body("function renderChartSingleSampleNotice("))
    assert code.count(f"'{_NOTICE}'") == 1
    assert f"'{_NOTICE}'" in writer

    calls = re.findall(r"(?<!function )renderChartSingleSampleNotice\(([^)]*)\)", code)
    assert sorted(set(calls)) == ["null", "window.SELECTED_RUN"]
    assert calls.count("window.SELECTED_RUN") == 1
    painter = strip_comments(fn_body("function initializeCharts()"))
    assert "renderChartSingleSampleNotice(window.SELECTED_RUN)" in painter
    # After the Chart is built, so an exception while building it cannot
    # leave a note above a chart that was never drawn.
    assert painter.index("new Chart(") < painter.index(
        "renderChartSingleSampleNotice(window.SELECTED_RUN)"
    )
