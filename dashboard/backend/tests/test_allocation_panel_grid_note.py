"""The Capital Allocation panel must say when the grid beside it shows less.

The panel is portfolio-wide by necessity: a pie that dropped agents would no
longer add up to the portfolio, so the legend lists every agent holding a
sleeve. The My Agents grid is not -- a search term, a market chip and the
per-shelf page cap each hide cards. The two therefore disagree on *count* by
design, which reads as the panel inventing agents that "don't exist".

These pin the note that reconciles them, and the two things that make the note
trustworthy:

* Both counts are over the set the legend actually draws -- agents carrying a
  sleeve. The legend has no row for an agent with no capital, so that agent's
  absence from the grid reconciles nothing. Counting the whole roster printed a
  caveat for a disagreement the panel did not have, against a total matching
  neither the legend nor anything else on screen.
* Each named cause is measured by what it *dropped*, never by whether the
  control happens to be set. A search that matches everything, a chip that
  excludes nothing and a page cap above the shelf size are all invisible to the
  user; naming one of them explains the wrong thing.

Together those give the invariant the note leans on: ``shown < total`` holds if
and only if at least one cause is true, so the sentence can never trail off
into a bare "because of" nothing.
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


def _visibility_harness() -> str:
    """The two pure helpers the grid render feeds, lifted out of app.js."""
    return "\n".join(
        [
            fn_body("function holdsAllocatedCapital"),
            fn_body("function countAllocatedCapital"),
            fn_body("function agentGridVisibilityFrom"),
        ]
    )


# --- what counts as a row the legend draws ---------------------------------


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_only_agents_carrying_a_sleeve_are_counted():
    """Mirrors buildAgentAllocationData's `cash_allocation > 0` filter.

    An agent with no sleeve draws no legend row, so its absence from the grid
    is nothing for the note to reconcile.
    """
    script = _visibility_harness() + """
console.log(JSON.stringify(countAllocatedCapital([
  { cash_allocation: 1000 },
  { cash_allocation: 0 },
  { cash_allocation: null },
  {},
  { cash_allocation: '2500' },
  { cash_allocation: -5 },
])));
"""
    assert _run_node(script) == 2


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_counting_survives_a_null_roster():
    script = _visibility_harness() + """
console.log(JSON.stringify([countAllocatedCapital(null), countAllocatedCapital([null])]));
"""
    assert _run_node(script) == [0, 0]


# --- cause attribution ------------------------------------------------------


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_no_caveat_when_every_capital_holder_has_a_card():
    """Ten agents, three carrying sleeves, page size 5, all three on page one.

    Paging really did withhold seven cards -- and hid nothing the panel lists,
    so there is no disagreement to explain. Counting the whole roster here
    printed "shows 5 of 10", a total matching neither the three legend rows nor
    anything else on the panel.
    """
    script = _visibility_harness() + """
console.log(JSON.stringify(
  agentGridVisibilityFrom({ roster: 3, searched: 3, chipped: 3, painted: 3 })
));
"""
    view = _run_node(script)
    assert view["shown"] == view["total"] == 3
    assert (view["searching"], view["filtered"], view["paged"]) == (False, False, False)


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_paging_is_not_blamed_when_it_withheld_nothing():
    """Seven sleeve-holders, a search matching two: one page, no pager drawn."""
    script = _visibility_harness() + """
console.log(JSON.stringify(
  agentGridVisibilityFrom({ roster: 7, searched: 2, chipped: 2, painted: 2 })
));
"""
    view = _run_node(script)
    assert view["shown"] == 2 and view["total"] == 7
    assert view["searching"] is True
    assert view["paged"] is False
    assert view["filtered"] is False


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_search_is_not_blamed_when_it_matched_everything():
    """A term matching every agent is invisible to the user; the page cap is not."""
    script = _visibility_harness() + """
console.log(JSON.stringify(
  agentGridVisibilityFrom({ roster: 7, searched: 7, chipped: 7, painted: 5 })
));
"""
    view = _run_node(script)
    assert view["searching"] is False
    assert view["filtered"] is False
    assert view["paged"] is True


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_market_chip_is_not_blamed_when_it_excluded_nothing():
    script = _visibility_harness() + """
console.log(JSON.stringify(
  agentGridVisibilityFrom({ roster: 6, searched: 4, chipped: 4, painted: 4 })
));
"""
    view = _run_node(script)
    assert view["filtered"] is False
    assert view["searching"] is True
    assert view["paged"] is False


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_every_stage_that_dropped_a_row_is_named():
    script = _visibility_harness() + """
console.log(JSON.stringify(
  agentGridVisibilityFrom({ roster: 9, searched: 7, chipped: 5, painted: 3 })
));
"""
    view = _run_node(script)
    assert view["shown"] == 3 and view["total"] == 9
    assert (view["searching"], view["filtered"], view["paged"]) == (True, True, True)


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_a_hidden_row_always_names_at_least_one_cause():
    """The invariant the note leans on: no "because of" with nothing after it.

    Every stage is measured as a drop, and the stages are consecutive, so
    ``painted < roster`` cannot happen without some stage having dropped a row.
    """
    script = _visibility_harness() + """
const out = [];
for (let roster = 0; roster <= 4; roster += 1) {
  for (let searched = 0; searched <= roster; searched += 1) {
    for (let chipped = 0; chipped <= searched; chipped += 1) {
      for (let painted = 0; painted <= chipped; painted += 1) {
        const v = agentGridVisibilityFrom({ roster, searched, chipped, painted });
        const named = v.searching || v.filtered || v.paged;
        if ((v.shown < v.total) !== named) out.push(v);
      }
    }
  }
}
console.log(JSON.stringify(out));
"""
    assert _run_node(script) == []


# --- the rendered sentence --------------------------------------------------


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_is_silent_when_every_listed_agent_has_a_card():
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
    assert "4 of" in note and "6" in note
    assert "the market filter" in note
    assert "paging" in note
    assert "your search" not in note


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_degrades_to_silence_without_the_grid_helper():
    """portfolio.js loads before app.js. A legend painted in that window must
    not throw, and must not guess -- no helper means no claim about the grid."""
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{}};
console.log(JSON.stringify([allocationGridNoteHtml()]));
"""
    assert _run_node(script) == [""]


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_note_makes_no_claim_when_no_listed_agent_holds_capital():
    script = f"""
{_portfolio_fn("function allocationGridNoteHtml")}
globalThis.window = {{ describeAgentGridVisibility: () => (
  {{ shown: 0, total: 0, searching: false, filtered: false, paged: false }}
) }};
console.log(JSON.stringify(allocationGridNoteHtml()));
"""
    assert _run_node(script) == ""


# --- wiring -----------------------------------------------------------------


def test_render_measures_each_stage_and_publishes_the_result():
    """Measured during the render, so no count can drift from what was painted."""
    body = fn_body("function renderAgentCategories")
    for stage in ("rosterWithCapital", "searchedWithCapital", "chippedWithCapital"):
        assert stage in body, stage
    assert "agentGridVisibility = agentGridVisibilityFrom(" in body
    # The roster total must be decorated: the legend is drawn from
    # allAgents.map(decorateAgent), and decorateAgent is what restores a
    # locally-overridden sleeve.
    assert "countAllocatedCapital(allAgents.map(decorateAgent))" in body
    assert "return visibleAgents;" in fn_body("function renderAgentCards")


def test_page_size_is_shared_with_the_grid():
    """The page cap the note calls `paged` is still a real cap.

    It is no longer one number: the grid pins its columns via a CSS ladder and
    `agentGridPageSizeFor` sizes each page to whole rows of whatever rung is
    live. What this guard cares about is only that a cap exists and that the
    render applies it -- `test_agent_grid_pagination.py` owns its shape.
    """
    assert re.search(r"const AGENT_GRID_TARGET_PAGE_SIZE = \d+", APP_JS)
    body = fn_body("function renderAgentCards")
    assert "const pageSize = agentGridPageSize(grid);" in body
    assert "agents.slice(start, start + pageSize)" in body


def test_legend_renders_the_note_and_exposes_its_refresh():
    assert "allocationGridNoteHtml()" in fn_body(
        "function renderAllocationLegend", _PORTFOLIO_JS
    )
    assert "window.refreshAllocationLegendNote = repaintAllocationLegend;" in _PORTFOLIO_JS


def test_note_style_exists():
    assert ".allocation-legend-hint--grid {" in _STYLES
