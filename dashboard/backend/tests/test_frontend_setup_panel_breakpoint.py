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
shipped styles.css as text -- the convention set by test_ai_hedge_fund_frontend.py
(see _frontend_source.py for the shared reader).
"""

import re

from ._frontend_source import STYLES


def _strip_css_comments(source: str) -> str:
    """`/* ... */` removed. styles.css has no `//` comments and no URL literals
    that would make a smarter, quote-aware stripper necessary here."""
    return re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)


_SOURCE = _strip_css_comments(STYLES)


def _media_max_width_blocks(source: str) -> list[tuple[int, str]]:
    """Every top-level `@media (max-width: Npx) { ... }` block, brace-matched.

    Returns (breakpoint, body) pairs. Brace-matched rather than sliced to a
    fixed width: a rule's own `{ ... }` body is itself nested one level inside
    the @media block, so a naive "up to the next '}'" slice would truncate at
    the first rule instead of the block's real end.
    """
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"@media\s*\(max-width:\s*(\d+)px\)\s*\{", source):
        breakpoint = int(match.group(1))
        open_brace = match.end() - 1
        depth = 0
        index = open_brace
        while True:
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        blocks.append((breakpoint, source[open_brace + 1 : index]))
    return blocks


def _breakpoint_of(selector_pattern: str, declaration: str) -> int:
    """The max-width breakpoint of the one `@media` block holding a rule whose
    selector matches `selector_pattern` and whose body contains `declaration`.

    `selector_pattern` is anchored to the start of its line (only leading
    whitespace before it): a bare `\\.left-panel` must not match inside the
    longer `.playground-backtest-panel .left-panel`, which contains it as a
    literal substring. Every selector in this file starts its own line, so the
    anchor reliably tells the two apart.
    """
    rule_re = re.compile(
        r"^[ \t]*" + selector_pattern + r"\s*\{[^}]*" + re.escape(declaration) + r"[^}]*\}",
        re.MULTILINE,
    )
    hits = [bp for bp, body in _media_max_width_blocks(_SOURCE) if rule_re.search(body)]
    assert hits, (
        f"no @media (max-width: ...) block has a rule matching /{selector_pattern}/ "
        f"with `{declaration}` -- has the selector or declaration changed?"
    )
    assert len(hits) == 1, (
        f"rule matching /{selector_pattern}/ with `{declaration}` appears in more "
        f"than one max-width breakpoint: {hits}"
    )
    return hits[0]


def test_setup_panel_reshow_breakpoint_matches_its_hide_breakpoint():
    """The re-show rule's breakpoint must equal the hide rule's.

    A test that only asserted both rules exist would have passed before this
    fix too -- that is exactly the regression this must catch, so it compares
    the two breakpoints rather than just their presence.
    """
    hide_at = _breakpoint_of(r"\.left-panel", "display: none")
    reshow_at = _breakpoint_of(r"\.playground-backtest-panel \.left-panel", "display: flex")
    assert reshow_at == hide_at, (
        f".left-panel is hidden at max-width:{hide_at}px but only restored "
        f"(.playground-backtest-panel .left-panel) at max-width:{reshow_at}px -- "
        "every viewport width between the two leaves the setup panel gone with "
        "no replacement (issue #129)"
    )
