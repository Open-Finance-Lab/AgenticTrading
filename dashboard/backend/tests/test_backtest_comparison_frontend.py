"""Browser-runtime contracts for Backtest benchmark comparison."""

import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    STYLES,
    fn_body,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "frontend" / "js" / "backtest-comparison.js"
FIXTURE = Path(__file__).parent / "fixtures" / "backtest-comparison.json"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def run_helper(expression):
    fixture = FIXTURE.read_text(encoding="utf-8")
    source = SCRIPT.read_text(encoding="utf-8")
    result = subprocess.run(
        [
            "node",
            "-e",
            "\n".join(
                [
                    "global.window = global;",
                    source,
                    f"const fixture = {fixture};",
                    f"console.log(JSON.stringify({expression}));",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def extract_app_function(source, name):
    for marker in (f"async function {name}(", f"function {name}("):
        start = source.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in app.js")
    depth = 0
    index = source.index("{", start)
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
        index += 1
    raise AssertionError(f"unterminated function {name}")


def run_app_functions(source, names, harness_lines):
    script = "\n".join(
        [*(extract_app_function(source, name) for name in names), *harness_lines]
    )
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_model_canonicalizes_shuffled_series_and_calculates_deltas():
    model = run_helper("BacktestComparison.buildModel(fixture.chart, fixture.run)")
    assert [column["key"] for column in model["columns"]] == [
        "agent",
        "djia",
        "nasdaq",
        "buyhold",
    ]
    assert [column["label"] for column in model["columns"]] == [
        "Your Agent",
        "DJIA",
        "Nasdaq-100",
        "Buy & Hold",
    ]
    assert model["agentDeltas"]["djia"] == pytest.approx(0.63)
    assert model["agentDeltas"]["nasdaq"] == pytest.approx(-4.26)
    assert model["agentDeltas"]["buyhold"] == pytest.approx(-0.66)
    assert model["bestByMetric"]["totalReturn"] == ["nasdaq"]


def test_metrics_use_population_stddev_and_hourly_annualization():
    metrics = run_helper(
        "BacktestComparison.calculateMetrics([100, 110, 99, 120])"
    )
    returns = [0.1, -0.1, 120 / 99 - 1]
    expected_sharpe = (
        statistics.fmean(returns)
        / statistics.pstdev(returns)
        * math.sqrt(252 * 6.5)
    )
    assert metrics["finalValue"] == pytest.approx(120.0)
    assert metrics["totalReturn"] == pytest.approx(20.0)
    assert metrics["maxDrawdown"] == pytest.approx(-10.0)
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)


def test_invalid_and_flat_series_are_unavailable_not_zero():
    result = run_helper(
        "({"
        "short: BacktestComparison.calculateMetrics([100]),"
        "flat: BacktestComparison.calculateMetrics([100, 100, 100]),"
        "zeroStart: BacktestComparison.calculateMetrics([0, 10, 12]),"
        "missing: BacktestComparison.calculateMetrics([null, undefined]),"
        "gapped: BacktestComparison.calculateMetrics([100, null, 110]),"
        "nonFinite: BacktestComparison.calculateMetrics([100, Number.NaN, 110])"
        "})"
    )
    assert result["short"] == {
        "finalValue": 100,
        "totalReturn": None,
        "maxDrawdown": None,
        "sharpe": None,
    }
    assert result["flat"]["sharpe"] is None
    assert result["zeroStart"]["totalReturn"] is None
    assert result["missing"] == {
        "finalValue": None,
        "totalReturn": None,
        "maxDrawdown": None,
        "sharpe": None,
    }
    assert result["gapped"]["totalReturn"] == pytest.approx(10.0)
    assert result["nonFinite"]["totalReturn"] == pytest.approx(10.0)


def test_ifind_model_omits_inapplicable_us_indexes():
    keys = run_helper(
        "BacktestComparison.buildModel(fixture.chart, "
        "{...fixture.run, data_source: 'ifind_ashare'}).columns.map(c => c.key)"
    )
    assert keys == ["agent", "buyhold"]


def test_degraded_indexes_and_missing_buyhold_stay_explicitly_unavailable():
    model = run_helper(
        "BacktestComparison.buildModel("
        "{...fixture.chart, index_baselines_ok: false, "
        "series: fixture.chart.series.filter(s => s.run_id !== 'buyhold-fixture')}, "
        "fixture.run)"
    )
    columns = {column["key"]: column for column in model["columns"]}
    assert columns["agent"]["available"] is True
    assert columns["djia"]["available"] is False
    assert columns["nasdaq"]["available"] is False
    assert columns["buyhold"]["available"] is False
    assert model["agentDeltas"] == {
        "djia": None,
        "nasdaq": None,
        "buyhold": None,
    }


def test_exact_raw_ties_mark_every_tied_series_best():
    best = run_helper(
        "(() => { const agent = fixture.chart.series.find("
        "s => s.run_id === fixture.chart.agent_run_id); "
        "const chart = {...fixture.chart, series: fixture.chart.series"
        ".filter(s => ['agent-fixture', 'index:^DJI'].includes(s.run_id))"
        ".map(s => s.run_id === 'index:^DJI' ? {...s, values: agent.values} : s)}; "
        "return BacktestComparison.buildModel(chart, fixture.run).bestByMetric; })()"
    )
    assert best["totalReturn"] == ["agent", "djia"]
    assert best["maxDrawdown"] == ["agent", "djia"]


def test_comparison_script_and_semantic_table_ship_before_app():
    helper = '<script src="js/backtest-comparison.js?v=1" defer></script>'
    app = '<script src="app.js?v=125" defer></script>'
    assert 'href="styles.css?v=130"' in APP_HTML
    assert APP_HTML.index(helper) < APP_HTML.index(app)
    for element_id in (
        "performanceLegend",
        "performanceComparison",
        "performanceComparisonHead",
        "performanceComparisonBody",
        "performanceComparisonStatus",
    ):
        assert f'id="{element_id}"' in APP_HTML
    assert 'aria-describedby="performanceComparisonCaption"' in APP_HTML
    assert 'aria-details="performanceComparison"' in APP_HTML
    assert 'role="img"' in APP_HTML


def test_chart_uses_canonical_model_and_dom_legend():
    body = fn_body("function initializeCharts()")
    assert "BacktestComparison.buildModel" in body
    assert "renderPerformanceComparison" in body
    assert "renderPerformanceLegend" in body
    assert "comparisonKey" in body
    assert "display: false" in body


def test_renderers_create_semantic_headers_and_native_legend_buttons():
    comparison = fn_body("function renderPerformanceComparison(")
    legend = fn_body("function renderPerformanceLegend(")
    assert ".scope = 'col'" in comparison
    assert ".scope = 'row'" in comparison
    assert "Best" in comparison
    assert "document.createElement('button')" in legend
    assert "button.type = 'button'" in legend
    assert "aria-pressed" in legend
    assert "setDatasetVisibility" in legend


def test_comparison_states_and_focus_styles_ship():
    assert ".performance-comparison" in STYLES
    assert ".performance-legend-button:focus-visible" in STYLES
    assert ".performance-comparison-scroll:focus-visible" in STYLES
    assert 'data-metric="final-value"' not in APP_HTML
    for removed in (
        "loadPerformanceMetrics",
        "displayPerformanceMetrics",
        "displayNoMetrics",
    ):
        assert removed not in APP_JS


def test_request_token_rejects_previous_run_after_selection_changes():
    result = run_app_functions(
        APP_JS,
        ["beginBacktestSurfaceRequest", "isCurrentBacktestSurfaceRequest"],
        [
            "global.window = { SELECTED_RUN: { run_id: 'run-a' } };",
            "let liveBacktestChartActive = false;",
            "let backtestSurfaceRequestSeq = 0;",
            "const first = beginBacktestSurfaceRequest('run-a');",
            "window.SELECTED_RUN = { run_id: 'run-b' };",
            "const second = beginBacktestSurfaceRequest('run-b');",
            "console.log(JSON.stringify({",
            "  first: isCurrentBacktestSurfaceRequest(first),",
            "  second: isCurrentBacktestSurfaceRequest(second),",
            "}));",
        ],
    )
    assert result == {"first": False, "second": True}


def test_historical_surfaces_settle_independently():
    body = fn_body("async function loadHistoricalBacktestSurfaces(")
    assert "Promise.allSettled" in body
    assert "setPerformanceComparisonState(" in body
    assert "'error'" in body
    assert "loadTradingLogForRun" in body
    assert "isCurrentBacktestSurfaceRequest" in body
