"""Frontend contracts for five-minute source data and hourly decisions."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, FRONTEND, fn_body


def _run_formatters(expression: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = "\n".join(
        [
            fn_body("function formatBacktestFrequencyContract("),
            fn_body("function formatBacktestMarketDataQuality("),
            f"console.log(JSON.stringify({expression}));",
        ]
    )
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _render_data_source_badge(run: dict):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    script = "\n".join(
        [
            "const badge = {};",
            "const document = { getElementById: () => badge };",
            fn_body("function renderBacktestDataSourceBadge("),
            f"renderBacktestDataSourceBadge({json.dumps(run)});",
            "console.log(JSON.stringify(badge));",
        ]
    )
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_backtest_details_have_frequency_and_quality_rows():
    assert 'id="backtestConfigFrequencyRow"' in APP_HTML
    assert 'id="backtestConfigFrequency"' in APP_HTML
    assert 'id="backtestConfigDataQualityRow"' in APP_HTML
    assert 'id="backtestConfigDataQuality"' in APP_HTML


def test_minute_frequency_formatter_states_fixed_execution_policy():
    value = _run_formatters(
        "formatBacktestFrequencyContract({"
        "source_timeframe:'5m',decision_timeframe:'60m',"
        "decision_frequency:'1h',execution_timeframe:'5m',"
        "valuation_frequency:'5m',fill_policy:'next_source_bar_open'"
        "})"
    )

    assert value == "5m source · 1h decisions · next 5m open fills · 5m valuation"


def test_quality_formatter_reports_dropped_and_problem_counts():
    value = _run_formatters(
        "formatBacktestMarketDataQuality({"
        "total_decision_bars:210,usable_decision_bars:207,"
        "dropped_decision_bars:3,missing_source_bars:2,"
        "duplicate_source_bars:1,off_grid_source_bars:0,invalid_source_bars:0"
        "})"
    )

    assert value == "207/210 usable · 3 dropped · 2 missing · 1 duplicate"


def test_alpaca_badge_and_strategy_page_describe_minute_source_hourly_decisions():
    badge = _render_data_source_badge(
        {
            "data_source": "alpaca",
            "frequency_contract": {
                "source_timeframe": "5m",
                "decision_frequency": "1h",
            },
        }
    )
    render = fn_body("function renderBacktestRunConfig(")
    assert badge["textContent"] == "Alpaca · 5m source · hourly decisions"
    assert badge["hidden"] is False
    assert "run?.frequency_contract" in render
    assert "run?.market_data_quality" in render

    strategy_source = (FRONTEND / "strategy.html").read_text(encoding="utf-8")
    assert "Alpaca 5-minute data + hourly decisions + hosted model" in strategy_source
