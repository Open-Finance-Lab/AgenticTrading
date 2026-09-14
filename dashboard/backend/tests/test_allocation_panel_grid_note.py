"""The Capital Allocation panel must say when the grid beside it shows less.

The panel is portfolio-wide by necessity: a pie that dropped agents would no
longer add up to the portfolio, so the legend lists every agent holding a
sleeve. The My Agents grid is not -- a search term, a market chip and the
per-shelf page cap each hide cards. The two therefore disagree on *count* by
design, which reads as the panel inventing agents that "don't exist".

These pin the note that reconciles them, and the one thing that makes the note
trustworthy: it is anchored on how many cards were actually painted, not on
whether some filter happens to be set. A filter that hides nothing must not
raise a caveat, and the page cap must raise one with no filter set at all.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_JS,
    fn_body,
    js_const,
)

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_PORTFOLIO_JS = (_FRONTEND / "js" / "portfolio.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")

_NODE_MISSING = shutil.which("node") is None


def _portfolio_fn(signature: str) -> str:
    return fn_body(signature, _PORTFOLIO_JS)


def _run_node(script: str):
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_GRID_HARNESS = """
const AGENT_GRID_PAGE_SIZE = {page_size};
{shelves}
{shelf_key}
let allAgents = [];
let agentGridShownCount = 0;
let agentMarketFilter = 'all';
let searchValue = '';
const document = {{ getElementById: () => ({{ value: searchValue }}) }};
{describe}
"""


def _grid_harness() -> str:
    page_size = re.search(r"const AGENT_GRID_PAGE_SIZE = (\d+)", APP_JS).group(1)
    return _GRID_HARNESS.format(
        page_size=page_size,
        shelves=js_const("AGENT_SHELVES"),
        shelf_key=fn_body("function agentShelfKey"),
        describe=fn_body("function describeAgentGridVisibility"),
    )


def _agents(count: int, **overrides) -> str:
    base = {"agent_type": "builtin", "runtime_type": "pipeline"}
    base.update(overrides)
    return json.dumps([dict(base, agent_id=f"a{i}") for i in range(count)])


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_page_cap_alone_is_reported_even_with_no_filter_set():
    """Six LLM agents, page size 5: five cards, six legend rows, no filter."""
    script = _grid_harness() + f"""
allAgents = {_agents(6)};
agentGridShownCount = {5};
const view = describeAgentGridVisibility();
console.log(JSON.stringify(view));
"""
    view = _run_node(script)
    assert view["total"] == 6
    assert view["shown"] == 5
    assert view["paged"] is True
    assert view["searching"] is False
    assert view["filtered"] is False


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_shown_never_exceeds_the_roster():
    """A stale tally must not produce "shows 7 of 6" — clamped, so the note
    simply disappears rather than printing a number that cannot be true."""
    script = _grid_harness() + f"""
allAgents = {_agents(3)};
agentGridShownCount = 9;
console.log(JSON.stringify(describeAgentGridVisibility()));
"""
    view = _run_node(script)
    assert view["shown"] == 3
    assert view["total"] == 3


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_is_silent_when_every_agent_has_a_card():
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{ describeAgentGridVisibility: () => (
  {{ shown: 4, total: 4, searching: true, filtered: true, paged: true }}
) }};
console.log(JSON.stringify(allocationGridNoteHtml()));
"""
    assert _run_node(script) == ""


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_names_the_counts_and_the_reason():
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{ describeAgentGridVisibility: () => (
  {{ shown: 4, total: 6, searching: false, filtered: true, paged: true }}
) }};
console.log(JSON.stringify(allocationGridNoteHtml()));
"""
    note = _run_node(script)
    assert "4 of 6" in note
    assert "the market filter" in note
    assert "paging" in note
    assert "your search" not in note


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_degrades_to_silence_without_the_grid_helper():
    """portfolio.js loads before app.js. A legend painted in that window must
    not throw, and must not guess — no helper means no claim about the grid."""
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{}};
console.log(JSON.stringify([allocationGridNoteHtml()]));
"""
    assert _run_node(script) == [""]


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_empty_roster_makes_no_claim():
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{ describeAgentGridVisibility: () => (
  {{ shown: 0, total: 0, searching: false, filtered: false, paged: false }}
) }};
console.log(JSON.stringify(allocationGridNoteHtml()));
"""
    assert _run_node(script) == ""


def test_tally_is_reset_and_published_by_the_grid_render():
    """Counted during the render, so it cannot drift from what was painted."""
    body = fn_body("function renderAgentCategories")
    assert "agentGridShownCount = 0;" in body
    assert "window.refreshAllocationLegendNote" in body
    assert "agentGridShownCount += visibleAgents.length;" in fn_body(
        "function renderAgentCards"
    )


def test_legend_renders_the_note_and_exposes_its_refresh():
    assert "allocationGridNoteHtml()" in fn_body(
        "function renderAllocationLegend", _PORTFOLIO_JS
    )
    assert "window.refreshAllocationLegendNote = repaintAllocationLegend;" in _PORTFOLIO_JS


def test_note_style_exists():
    assert ".allocation-legend-hint--grid {" in _STYLES
