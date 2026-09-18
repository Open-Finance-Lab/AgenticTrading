"""A My Agents shelf page must be a whole number of grid rows.

The page cap counted a fixed 5 *items* while the grid laid out *tracks*:
``repeat(auto-fill, minmax(300px, 1fr))`` resolves to 4 columns across the
whole 1264-1580px band -- most laptops -- so 5 % 4 left the fifth card alone on
a second row. A lone card under a full row reads as "the rest are on the next
page" even when the page is already complete. That is the widow these pin.

Two halves, and they have to stay in step:

* styles.css pins the columns (4 -> 3 -> 2 -> 1 down the ladder) instead of
  letting ``auto-fill`` float them.
* app.js MEASURES that column count back off the computed style and sizes the
  page to whole rows of it, rather than mirroring the breakpoints in JS. A
  mirror is a second copy of the ladder, and a second copy drifts.

The height guard is the other half of the requirement. "Two rows per page" must
come from IMPLICIT rows -- ``grid-template-rows: repeat(2, 1fr)`` is the
obvious way to write it and is exactly wrong, because declared tracks
materialize whether or not a card lands in them, holding a second row open and
blank under a page of four. Implicit rows only exist for cards that exist, so a
short page collapses to one row by itself.
"""

import json
import re
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_JS, STYLES, fn_body

_NODE_MISSING = shutil.which("node") is None

# Every rung of the ladder in styles.css, widest first.
_LADDER = (4, 3, 2, 1)


def _run_node(script: str):
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _page_size_harness() -> str:
    """The pure page-size helper, lifted out of app.js with its constant."""
    target = re.search(r"const AGENT_GRID_TARGET_PAGE_SIZE = \d+;", APP_JS)
    assert target, "AGENT_GRID_TARGET_PAGE_SIZE is gone"
    return target.group(0) + "\n" + fn_body("function agentGridPageSizeFor")


# --- the invariant that kills the widow -------------------------------------


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_every_page_is_whole_rows():
    """`pageSize % cols == 0` at every rung, and a page holds at least one row.

    This is the whole fix in one line. A page that divides evenly into the live
    column count cannot end mid-row, so no card is ever left alone under a gap.
    Swept over more columns than the ladder ships so that widening it later
    cannot quietly reintroduce a remainder.
    """
    script = _page_size_harness() + """
const out = {};
for (let cols = 1; cols <= 8; cols += 1) out[cols] = agentGridPageSizeFor(cols);
console.log(JSON.stringify(out));
"""
    sizes = _run_node(script)
    for cols in range(1, 9):
        size = sizes[str(cols)]
        assert size % cols == 0, f"{size} cards does not divide into {cols} columns"
        assert size >= cols, f"{size} cards is less than one row of {cols}"


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_the_desktop_layout_is_four_by_two():
    """The shipped spec: 4 columns, 2 rows, 8 agents on a page.

    Narrower rungs are named here too because they are a deliberate trade, not
    an accident. 3 columns holds 6 (two rows) -- 8 there would be 3 + 3 + 2,
    the widow again. 1 and 2 columns hold 8 rather than two rows' worth,
    because two rows of one card is not a page.
    """
    script = _page_size_harness() + """
console.log(JSON.stringify([4, 3, 2, 1].map(agentGridPageSizeFor)));
"""
    assert _run_node(script) == [8, 6, 8, 8]


@pytest.mark.skipif(_NODE_MISSING, reason="node is not installed")
def test_a_junk_column_count_still_yields_a_usable_page():
    """A measurement that fails must not paginate the shelf into nothing.

    agentGridColumnCount returns a fallback rather than 0 when it cannot read
    px tracks, but this is the backstop for that backstop: a page size of 0
    would make agentGridPageCount divide by zero and a negative one would slice
    backwards, so clamp at the arithmetic rather than at the caller.
    """
    script = _page_size_harness() + """
console.log(JSON.stringify([0, -3, NaN, undefined, null, 'x'].map(agentGridPageSizeFor)));
"""
    for size in _run_node(script):
        assert size >= 1, size


# --- the JS reads the CSS, rather than restating it -------------------------


def test_column_count_is_measured_not_mirrored():
    body = fn_body("function agentGridMeasuredColumns")
    assert "gridTemplateColumns" in body
    # matchMedia here would be a second copy of the ladder in styles.css, and
    # two copies of a breakpoint list disagree the first time one is edited.
    assert "matchMedia" not in body


def test_a_hidden_grid_falls_back_rather_than_reading_one_column():
    """Every shelf is hidden until you navigate to My Agents.

    getComputedStyle on a grid that was never laid out returns the SPECIFIED
    value -- "repeat(4, minmax(0, 1fr))" -- not used pixel tracks. Split on
    whitespace that is 2 tokens, so a naive count paginates the whole shelf
    into pairs. Accept the measurement only when every token is a px length.
    """
    assert "px" in fn_body("function agentGridMeasuredColumns")
    assert "AGENT_GRID_FALLBACK_COLUMNS" in fn_body("function agentGridColumnCount")
    assert re.search(r"const AGENT_GRID_FALLBACK_COLUMNS = \d+;", APP_JS)


def test_render_records_the_measurement_not_the_fallback():
    """A hidden paint must stay distinguishable from a real 4-column one.

    loadAgents() renders these shelves while the panel is still hidden --
    showPlaygroundPanel does it on the Backtest subtab too -- where the
    measurement cannot succeed. Recording the FALLBACK there would mean a
    viewport actually on the 3-column rung had 4 stored against it, and the
    resize guard below would see no step to correct.
    """
    assert "agentGridColumns[categoryKey] = agentGridMeasuredColumns(grid);" in fn_body(
        "function renderAgentCards"
    )


def test_resize_repaints_only_when_the_ladder_steps():
    """Guarded on the column count changing, not on the resize firing.

    Dragging a window edge emits a resize per frame while the ladder holds at
    one rung for hundreds of pixels, and an unconditional repaint changes the
    grid's own height -- which is how a render loop that only reproduces on
    someone else's machine gets written.
    """
    body = fn_body("function setupAgentGridResizeHandler")
    assert "agentGridColumnCount(grid) !== agentGridColumns[shelf.key]" in body
    assert "if (!stepped) return;" in body
    # Keep the page index: a repaint caused by a window resize must not scroll
    # the user back to page 1 of every shelf.
    assert "applyAgentFilters(false)" in body
    assert "setupAgentGridResizeHandler();" in APP_JS


def test_view_mode_switch_repaints():
    """List view is one column, so the toggle changes the page size.

    Toggling .agents-grid--list without repainting left the cards on screen
    paginated for the other view.
    """
    assert "applyAgentFilters(false);" in fn_body("function setAgentViewMode")


# --- the CSS half -----------------------------------------------------------


def _agents_grid_rules() -> list[str]:
    """Every declaration block governing a My Agents grid, comments stripped.

    Comments are stripped because these guards assert on the ABSENCE of
    properties, and prose explaining why a property is forbidden contains the
    property's name -- a guard that reads its own rationale as a violation
    fails the moment someone documents it.
    """
    source = re.sub(r"/\*.*?\*/", "", STYLES, flags=re.S)
    return [
        source[m.end():source.index("}", m.end())]
        for m in re.finditer(
            r"\.agents-section \.agents-grid:not\(\.agents-grid--list\)\s*\{", source
        )
    ]


def test_the_ladder_pins_every_rung():
    rules = _agents_grid_rules()
    assert len(rules) == len(_LADDER), f"expected {len(_LADDER)} rungs, found {len(rules)}"
    found = []
    for rule in rules:
        template = re.search(r"grid-template-columns:\s*([^;]+);", rule)
        assert template, rule
        value = template.group(1)
        if "repeat(" in value:
            found.append(int(re.search(r"repeat\((\d+)", value).group(1)))
        else:
            found.append(1)  # the single-column rung needs no repeat()
    assert found == list(_LADDER), found
    # auto-fill is what let the column count float free of the page size.
    assert not any("auto-fill" in rule or "auto-fit" in rule for rule in rules)


def test_nothing_reserves_a_second_row():
    """The no-giant-blanks requirement, as a prohibition.

    Declared rows exist whether or not a card lands in them, so
    `grid-template-rows: repeat(2, 1fr)` would hold a blank row open under a
    page of four -- the exact thing two-rows-per-page must not cost. Implicit
    rows only materialize for cards that exist. Same for a min-height floor on
    the grid, which reserves the space without even naming a row.
    """
    for rule in _agents_grid_rules():
        assert "grid-template-rows" not in rule, rule
        assert "min-height" not in rule, rule
        assert re.search(r"(^|[^-\w])height\s*:", rule) is None, rule


# --- the pager label --------------------------------------------------------


def test_the_pager_names_the_range_not_the_ordinal():
    """The page size moves with the ladder, so an ordinal describes nothing.

    "Page 2 of 3" covers a different number of agents at different widths;
    "Showing 9-16 of 21 agents" is true at every rung and answers the question
    the widow bug made people ask.
    """
    body = fn_body("function renderAgentGridFooter")
    assert "Showing ${first}" in body
    assert "of ${total} agents" in body
    assert "Page ${page + 1}" not in body


def test_the_pager_label_is_legible():
    """13px of --text-secondary was the smallest, lowest-contrast text here."""
    block = re.search(
        r"\.agents-grid-footer-count \{(.*?)\}", STYLES, flags=re.S
    )
    assert block
    rule = block.group(1)
    size = re.search(r"font-size:\s*(\d+)px", rule)
    assert size and int(size.group(1)) >= 14, rule
    assert "var(--text-primary)" in rule

    disabled = re.search(
        r"\.agents-grid-footer-btn--nav:disabled \{(.*?)\}", STYLES, flags=re.S
    )
    assert disabled
    opacity = re.search(r"opacity:\s*([\d.]+)", disabled.group(1))
    # Below ~0.4 the glyph and its border fall under the 3:1 WCAG non-text
    # contrast floor and read as absent rather than disabled.
    assert opacity and float(opacity.group(1)) >= 0.4, disabled.group(1)
