"""Source-level contract for the vanilla-JS market-data selector.

The dashboard is vanilla JS with no build step and no DOM test runner, so these
assert against source text. That makes *how* they match load-bearing: matching
exact formatting locks the file's whitespace, and a reformat or a quote-style
change turns them red without any behavior changing. Match structure instead --
a declaration inside a rule, an identifier in an expression -- via the helpers
below.

Know what these cannot prove. CSS text shows a rule exists, never that anything
is visible at a given viewport: the ``@media (max-width: 900px)`` rules below
coexist with a pre-existing ``.left-panel { display: none }`` at
``max-width: 1200px``, so the setup panel is still hidden from 901-1200px and
these assertions pass anyway. Layout needs a real browser; treat this file as a
wiring guard only.
"""

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_APP_JS = _FRONTEND / "app.js"
_STYLES = _FRONTEND / "styles.css"


def _media_block(css: str, query: str) -> str:
    """Return the body of the first ``@media`` block matching ``query``.

    Brace-counted rather than regexed to the closing ``}``, since the block
    contains nested rules.
    """
    start = re.search(
        r"@media\s*\(\s*max-width:\s*%s\s*\)\s*\{" % re.escape(query), css
    )
    assert start, f"no @media (max-width: {query}) block in styles.css"

    depth, i = 1, start.end()
    while i < len(css) and depth:
        depth += {"{": 1, "}": -1}.get(css[i], 0)
        i += 1
    assert not depth, f"unbalanced braces in @media (max-width: {query})"
    return css[start.end() : i - 1]


def _declarations(css: str, selector: str) -> str:
    """Return the declaration block for ``selector``, whitespace-insensitive."""
    pattern = re.sub(r"\s+", r"\\s+", re.escape(selector).replace(r"\ ", " "))
    match = re.search(pattern + r"\s*\{([^}]*)\}", css)
    assert match, f"no rule for {selector!r}"
    return match.group(1)


def _has_declaration(css: str, selector: str, prop: str, value: str) -> bool:
    return bool(
        re.search(
            rf"{re.escape(prop)}\s*:\s*{re.escape(value)}\s*;",
            _declarations(css, selector),
        )
    )


@pytest.fixture(scope="module")
def html() -> str:
    return _APP_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return _APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return _STYLES.read_text(encoding="utf-8")


def _attr(html: str, attr: str, value: str) -> bool:
    return bool(re.search(rf"{attr}\s*=\s*[\"']{re.escape(value)}[\"']", html))


def test_market_data_controls_and_provenance_badge_exist(html):
    assert _attr(html, "id", "marketDataSourceSelect")
    assert _attr(html, "id", "vnpySimulationNotice")
    assert _attr(html, "id", "backtestDataSourceBadge")
    assert _attr(html, "value", "alpaca")


def test_vnpy_option_is_not_hardcoded_in_markup(html):
    """The simulation option is injected by JS only when the feature endpoint
    says it is enabled; shipping it in the markup would expose it to everyone."""
    assert not _attr(html, "value", "vnpy_simulation")


def test_vnpy_option_is_feature_gated_and_updates_model_state(js):
    assert re.search(r"async\s+function\s+loadMarketDataFeatures\s*\(", js)
    assert re.search(r"features\.vnpy_simulation_enabled\s*===\s*true", js)
    assert re.search(r"option\.value\s*=\s*['\"]vnpy_simulation['\"]", js)
    assert re.search(r"modelSelect\.disabled\s*=\s*isSimulation", js)
    assert re.search(
        r"decisionSource\s*=\s*isSimulation\s*\?\s*RULE_BASED_DECISION_SOURCE",
        js,
    )


def test_backtest_request_and_result_labels_include_data_source(js):
    assert re.search(r"data_source:\s*dataSource", js)
    assert re.search(r"renderBacktestDataSourceBadge\(\s*selectedRun\s*\)", js)
    assert re.search(r"run\.data_source\s*===\s*['\"]vnpy_simulation['\"]", js)
    assert "vn.py simulated data" in js


def test_mobile_backtest_exposes_setup_controls(css):
    """Wiring guard only -- see the module docstring on what CSS text cannot
    prove about actual visibility."""
    mobile = _media_block(css, "900px")

    assert _has_declaration(mobile, ".playground-backtest-panel .left-panel", "display", "flex")
    assert _has_declaration(mobile, ".playground-backtest-panel .right-panel", "display", "flex")
    assert _has_declaration(mobile, ".performance-card .section-header", "flex-direction", "column")


def test_backtest_refresh_reloads_agent_selector(js):
    assert re.search(r"if\s*\(\s*!allAgents\.length\s*\)\s*loadAgents\(\s*\)", js)
