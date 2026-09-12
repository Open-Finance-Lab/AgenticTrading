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
The helpers below started from test_vnpy_simulation_frontend.py, the closer
precedent for asserting on CSS specifically, and differ in three ways that the
dead-band question forces:

1. They scan **every** `@media` block at a breakpoint, not the first. styles.css
   has four `@media (max-width: 1200px)` blocks, five at 1100px and eight at
   600px, so "the 1200px block" does not exist. A first-block-only scan reports
   the re-show rule present while an identical-specificity `display: none` in a
   *later* 1200px block silently wins the cascade and reinstates issue #129.
2. They report a **list** of breakpoints rather than asserting a unique one, and
   the test below compares coverage instead of equality (see its docstring).
3. They run on comment-stripped CSS. This stylesheet's house style quotes rules
   verbatim in comments (`/* Beat `.section-card h3 { font-size: 12px }` */` at
   lines 9345, 10869, 11103, 11203), so scanning raw text lets a *comment*
   register as a declaration -- and a `{` or `}` inside one also desynchronises
   the brace counter in `_media_blocks`.

Know what this still cannot prove. Resolution here is last-wins among
*textually identical* selectors, where specificity is equal by construction.
It does not model the cascade: a hide rule written at higher specificity
(`.main-container.playground-backtest-panel .left-panel`), inside a `min-width`
block, or with `!important` would defeat the re-show rule and pass this file.
Only a real browser can settle layout; treat this as a wiring guard.
"""

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_STYLES = _FRONTEND / "styles.css"


def _strip_comments(css: str) -> str:
    """Blank out `/* ... */` comments, preserving the file's line structure.

    Line count is preserved so that `line_anchored=True` below keeps meaning
    what it says; the content is dropped so that a rule quoted inside a comment
    -- this stylesheet does that habitually -- cannot register as a declaration,
    and so that a brace inside one cannot desynchronise `_media_blocks`.
    """
    return re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), css, flags=re.DOTALL
    )


def _media_blocks(css: str, query: str) -> list[str]:
    """The bodies of *every* `@media` block matching `query`, in source order.

    Brace-counted rather than regexed to the closing `}`, since each block
    contains nested rules. Returning all of them (not the first) is the point:
    styles.css repeats most breakpoints several times, and only the last
    declaration among them survives the cascade.
    """
    bodies = []
    for start in re.finditer(
        r"@media\s*\(\s*max-width:\s*%s\s*\)\s*\{" % re.escape(query), css
    ):
        depth, i = 1, start.end()
        while i < len(css) and depth:
            depth += {"{": 1, "}": -1}.get(css[i], 0)
            i += 1
        assert not depth, f"unbalanced braces in @media (max-width: {query})"
        bodies.append(css[start.end() : i - 1])
    return bodies


def _declaration_blocks(css: str, selector: str, *, line_anchored: bool = False) -> list[str]:
    """Every declaration body for `selector` in `css`, in source order.

    `line_anchored=True` requires the selector to start its own line (only
    leading whitespace before it). Needed for the bare `.left-panel` lookup: it
    is a literal substring of the longer `.playground-backtest-panel
    .left-panel`, so an unanchored search can match inside that compound
    selector instead of the standalone rule. Every selector in this file starts
    its own line, so the anchor reliably tells the two apart. The anchor also
    excludes `.left-panel,` at the head of a grouped base rule (styles.css:918),
    which is intended -- this guard is about the media-query overrides.
    """
    pattern = re.sub(r"\s+", r"\\s+", re.escape(selector).replace(r"\ ", " "))
    if line_anchored:
        rx = re.compile(r"^[ \t]*" + pattern + r"\s*\{([^}]*)\}", re.MULTILINE)
    else:
        rx = re.compile(pattern + r"\s*\{([^}]*)\}")
    return [m.group(1) for m in rx.finditer(css)]


def _effective_value(css: str, selector: str, prop: str, **kwargs) -> str | None:
    """What `prop` resolves to for `selector` in `css`, or None if never set.

    Among rules whose selector text is identical, specificity is identical too,
    so the last declaration in source order is the one that wins. That is the
    whole reason `_media_blocks` must return every block rather than the first.
    """
    value = None
    for block in _declaration_blocks(css, selector, **kwargs):
        for match in re.finditer(rf"{re.escape(prop)}\s*:\s*([^;]+);", block):
            value = match.group(1).strip()
    return value


def _all_max_width_breakpoints(css: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"@media\s*\(\s*max-width:\s*(\d+)px\s*\)", css)})


def _breakpoints_declaring(css: str, selector: str, prop: str, value: str, **kwargs) -> list[int]:
    """Max-width breakpoints at which `selector { prop: value }` is in effect.

    Scans every distinct breakpoint present in the file rather than checking one
    named up front (as `test_narrow_viewport_backtest_exposes_setup_controls` does): the
    point of this guard is to catch the rule moving to the *wrong* breakpoint,
    not to re-assert a hardcoded one it might move to along with the rule --
    which would defeat the point of comparing two breakpoints.

    Returns a list rather than asserting a unique hit. Declaring the same rule
    at two breakpoints is redundant but harmless, and failing on it would make
    this guard reject correct stylesheets.
    """
    return [
        bp
        for bp in _all_max_width_breakpoints(css)
        if _effective_value("\n".join(_media_blocks(css, f"{bp}px")), selector, prop, **kwargs)
        == value
    ]


@pytest.fixture(scope="module")
def css() -> str:
    return _strip_comments(_STYLES.read_text(encoding="utf-8"))


def test_setup_panel_reshow_covers_every_breakpoint_that_hides_it(css):
    """Every breakpoint that hides `.left-panel` must be covered by a re-show.

    A test that only asserted both rules exist would have passed before this fix
    too -- that is exactly the regression this must catch, so it compares the
    two breakpoints rather than just their presence.

    Coverage, not equality: `max-width: R` is active at every width where
    `max-width: H` is, as long as R >= H, and the re-show selector
    (`.playground-backtest-panel .left-panel`, specificity 0-2-0) outranks the
    hide selector (`.left-panel`, 0-1-0) regardless of source order. Demanding
    R == H would reject a correct stylesheet that re-showed the panel at a wider
    breakpoint.
    """
    hidden_at = _breakpoints_declaring(css, ".left-panel", "display", "none", line_anchored=True)
    reshown_at = _breakpoints_declaring(
        css, ".playground-backtest-panel .left-panel", "display", "flex"
    )

    if not hidden_at:
        # No breakpoint hides the panel, so no dead band can exist. This is a
        # legitimate end state, not a hole: once the hide and re-show rules sit
        # at the same breakpoint the hide is dead code, and deleting both is
        # strictly more correct than keeping them. Asserting the hide rule must
        # exist would fail that cleanup -- the opposite of this file's purpose.
        return

    for hide_at in hidden_at:
        assert [bp for bp in reshown_at if bp >= hide_at], (
            f".left-panel is hidden at max-width:{hide_at}px but only restored "
            f"(.playground-backtest-panel .left-panel) at {reshown_at or 'no breakpoint'} -- "
            "every viewport width between the two leaves the setup panel gone with "
            "no replacement (issue #129)"
        )


def test_breakpoint_scan_reads_every_block_not_just_the_first(css):
    """Pins the helper contract whose absence was the original guard's bug.

    styles.css repeats breakpoints, so a scan that stopped at the first matching
    `@media` block would read a `display: none` added to a *later* block at the
    same breakpoint as absent -- reporting the panel restored while the cascade
    hides it. Asserting the repetition exists keeps that failure mode from
    quietly becoming untestable if the helper regresses.
    """
    assert len(_media_blocks(css, "1200px")) > 1, (
        "expected styles.css to declare @media (max-width: 1200px) more than once; "
        "if that is no longer true, _media_blocks' scan-every-block contract still "
        "must hold for the breakpoints that do repeat"
    )
