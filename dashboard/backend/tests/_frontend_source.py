"""Readers for the frontend static-text guards.

/app has no build step and no JS test toolchain, so its contracts are guarded by
asserting against the shipped source as text (the convention set by
test_ai_hedge_fund_frontend.py). This module is the half those guards share:
load each file once, and slice a named region out of it by brace matching.

Shared rather than copied because the slicing is where the subtle bugs live -- a
helper that returns the wrong region makes every assertion built on it vacuous,
and a copy in each test file would have to be fixed in each test file.
"""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
STYLES = (FRONTEND / "styles.css").read_text(encoding="utf-8")


def _match_brace(source: str, index: int) -> int:
    """Index of the "}" closing the "{" at `index`."""
    depth = 0
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1


def fn_body(signature: str) -> str:
    """The named function's source, brace-matched to its real closing brace.

    Brace-matching rather than a fixed-width slice: a `[start:start + 900]`
    window over-reads into whatever unrelated top-level code happens to follow,
    so an assertion can pass on a neighbour's source instead of the function
    under test.

    The parameter list is walked by paren *before* the first brace is taken. A
    signature may legally contain braces of its own -- `f(msg, { opt = 1 } = {})`
    is a destructured parameter -- and matching from the textually-first "{"
    returns that parameter block instead of the body: a short, plausible-looking
    string in which every `assert "..." in body` fails, or worse, passes.
    """
    start = APP_JS.index(signature)
    index = APP_JS.index("(", start)
    depth = 0
    while True:
        if APP_JS[index] == "(":
            depth += 1
        elif APP_JS[index] == ")":
            depth -= 1
            if depth == 0:
                break
        index += 1
    open_brace = APP_JS.index("{", index)
    return APP_JS[start : _match_brace(APP_JS, open_brace) + 1]


def js_const(name: str) -> str:
    """The named top-level `const` declaration, verbatim including the `;`.

    For guards that execute app.js source under node: a harness that restates a
    threshold instead of lifting it tests the code against the harness's own
    value, so changing the shipped constant silently stops being covered while
    every case stays green.

    The initializer stops at the first `;`, so a value that *contains* one is
    truncated. That fails loudly rather than vacuously -- the truncation is not
    valid JS, so node exits non-zero and the harness's `returncode == 0` assert
    reports it. Use `js_string_const` for string constants.
    """
    match = re.search(rf"^const {re.escape(name)} = [^;]+;", APP_JS, re.MULTILINE)
    assert match, f"{name} not found in app.js"
    return match.group(0)


def js_string_const(name: str) -> str:
    """The *value* of a single-quoted JS string constant in app.js.

    The sibling of `js_const`, which returns the whole declaration: guards that
    execute app.js under node need the declaration verbatim, guards that compare
    a frontend copy against its Python original need the string itself.
    """
    match = re.search(
        rf"const\s+{re.escape(name)}\s*=\s*\n?\s*'((?:[^'\\]|\\.)*)'", APP_JS
    )
    assert match, f"{name} is no longer a single-quoted const in app.js"
    return match.group(1).replace("\\'", "'")


def css_blocks(prelude: str) -> list[str]:
    """Every styles.css block introduced by this prelude, brace-matched.

    styles.css carries eight separate reduced-motion blocks. Slicing from a
    class name to end-of-file would sweep in all the later ones, so any test
    asking "does *this* rule have a fallback" has to isolate the real block.

    Returns every match rather than the first: a selector commonly appears both
    as a plain rule and again inside a media query, and a test that silently
    took whichever came first would depend on authoring order.
    """
    return [
        STYLES[match.start() : _match_brace(STYLES, STYLES.index("{", match.start())) + 1]
        for match in re.finditer(re.escape(prelude) + r"\s*\{", STYLES)
    ]


def at_rule_blocks(prelude: str) -> list[str]:
    """`css_blocks` under its original name, kept for the Phase A guards."""
    return css_blocks(prelude)
