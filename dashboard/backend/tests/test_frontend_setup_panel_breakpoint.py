"""Source guard for the backtest setup panel's responsive dead band.

Issue #129: `.left-panel` (the backtest setup/config panel on /app's Playground
tab) was hidden at `@media (max-width: 1200px)`, but the rule that restores it
-- `.playground-backtest-panel .left-panel { display: flex; }`, which wins on
specificity -- lived in the separate `@media (max-width: 900px)` block. Since
the two breakpoints differ, a viewport between 901px and 1200px matched only
the hide rule: the panel disappeared with no replacement across most laptops
in a split window and every tablet in landscape. That panel is where a user
configures and launches a backtest (the primary onboarding task as of PR
#451), so the dead band made the product's main task unreachable there.

/app has no build step and no CSS test toolchain, so this asserts against the
shipped styles.css as text -- the convention set by test_ai_hedge_fund_frontend.py.
The helpers below are adapted from test_vnpy_simulation_frontend.py, the
closer precedent for asserting on CSS specifically: same `_media_block` brace
counting and `_declarations`/`_has_declaration` shape, generalized to scan
every breakpoint rather than one named up front, since the whole point here is
to discover which breakpoint a rule lives in and compare it to another rule's.
"""

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
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


def _declaration_block(css: str, selector: str, *, line_anchored: bool = False) -> str | None:
    """The declaration body for ``selector`` in ``css``, or None if absent.

    Adapted from test_vnpy_simulation_frontend.py's ``_declarations``: this
    returns None instead of asserting, because ``_breakpoint_declaring`` below
    calls it once per candidate breakpoint and most breakpoints legitimately
    do not contain the rule at all -- an assertion there would abort the scan
    instead of just ruling that breakpoint out.

    ``line_anchored=True`` requires the selector to start its own line (only
    leading whitespace before it). Needed for the bare ``.left-panel`` lookup:
    it is a literal substring of the longer
    ``.playground-backtest-panel .left-panel``, so an unanchored search can
    match inside that compound selector instead of the standalone rule. Every
    selector in this file starts its own line, so the anchor reliably tells
    the two apart.
    """
    pattern = re.sub(r"\s+", r"\\s+", re.escape(selector).replace(r"\ ", " "))
    if line_anchored:
        match = re.search(r"^[ \t]*" + pattern + r"\s*\{([^}]*)\}", css, re.MULTILINE)
    else:
        match = re.search(pattern + r"\s*\{([^}]*)\}", css)
    return match.group(1) if match else None


def _has_declaration(css: str, selector: str, prop: str, value: str, **kwargs) -> bool:
    block = _declaration_block(css, selector, **kwargs)
    if block is None:
        return False
    return bool(re.search(rf"{re.escape(prop)}\s*:\s*{re.escape(value)}\s*;", block))


def _all_max_width_breakpoints(css: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"@media\s*\(\s*max-width:\s*(\d+)px\s*\)", css)})


def _breakpoint_declaring(css: str, selector: str, prop: str, value: str, **kwargs) -> int:
    """The one max-width breakpoint whose (first) block declares
    ``selector { prop: value }``.

    Scans every distinct breakpoint present in the file rather than checking
    one named up front (as ``test_mobile_backtest_exposes_setup_controls``
    does): the point of this guard is to catch the rule moving to the *wrong*
    breakpoint, not to re-assert a hardcoded one it might move to along with
    the rule -- which would defeat the point of comparing two breakpoints.
    """
    hits = [
        bp
        for bp in _all_max_width_breakpoints(css)
        if _has_declaration(_media_block(css, f"{bp}px"), selector, prop, value, **kwargs)
    ]
    assert hits, f"no @media (max-width: ...) block declares {selector} {{ {prop}: {value} }}"
    assert len(hits) == 1, (
        f"{selector} {{ {prop}: {value} }} appears in more than one max-width "
        f"breakpoint: {hits}"
    )
    return hits[0]


@pytest.fixture(scope="module")
def css() -> str:
    return _STYLES.read_text(encoding="utf-8")


def test_setup_panel_reshow_breakpoint_matches_its_hide_breakpoint(css):
    """The re-show rule's breakpoint must equal the hide rule's.

    A test that only asserted both rules exist would have passed before this
    fix too -- that is exactly the regression this must catch, so it compares
    the two breakpoints rather than just their presence.
    """
    hide_at = _breakpoint_declaring(css, ".left-panel", "display", "none", line_anchored=True)
    reshow_at = _breakpoint_declaring(
        css, ".playground-backtest-panel .left-panel", "display", "flex"
    )
    assert reshow_at == hide_at, (
        f".left-panel is hidden at max-width:{hide_at}px but only restored "
        f"(.playground-backtest-panel .left-panel) at max-width:{reshow_at}px -- "
        "every viewport width between the two leaves the setup panel gone with "
        "no replacement (issue #129)"
    )
