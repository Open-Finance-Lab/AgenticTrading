"""Readers for the frontend static-text guards.

/app has no build step and no JS test toolchain, so its contracts are guarded by
asserting against the shipped source as text (the convention set by
test_ai_hedge_fund_frontend.py). This module is the half those guards share:
load each file once, and slice a named region out of it by brace matching.

Shared rather than copied because the slicing is where the subtle bugs live -- a
helper that returns the wrong region makes every assertion built on it vacuous,
and a copy in each test file would have to be fixed in each test file.

The three module constants are /app's files, but the scanners below take their
source as an argument: the landing guards (test_landing_*.py) slice TSX with the
same primitives, and the first copy of `match_paren` to live in one of those
files also shipped its own `//`-stripping regex that ate the second slash of any
URL. That is the failure this module's docstring is about, one directory over.
"""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
STYLES = (FRONTEND / "styles.css").read_text(encoding="utf-8")


def match_paren(source: str, index: int) -> int:
    """Index of the ")" closing the "(" at `index`.

    The paren twin of `_match_brace`, exported because an argument list is the
    unit the landing guards assert over: "which collection is this call
    measured across" is answerable only from the whole, balanced call, and a
    `source[start:start + 200]` window over it silently includes the next
    statement or cuts the argument in half.
    """
    depth = 0
    while index < len(source):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise AssertionError("unbalanced parentheses -- no ')' closes this '('")


def call_args(source: str, callee: str) -> str:
    """The first call to `callee` in `source`: its argument list, parens included.

    Strip comments first (see `strip_comments`) when the assertion is about what
    the call *does*. A call site worth guarding usually carries a note naming
    which of two collections it must not use, and an un-stripped scan is then
    satisfied by the prose explaining the bug rather than by the code avoiding
    it -- the assertion passes on the comment, and would go on passing if the
    code were reverted.
    """
    match = re.search(rf"\b{re.escape(callee)}\s*\(", source)
    assert match, f"{callee} is not called in this source"
    open_paren = source.index("(", match.start())
    return source[open_paren : match_paren(source, open_paren) + 1]


def strip_comments(source: str) -> str:
    r"""JS/TS source with `//` and `/* */` comments removed, strings left intact.

    A plain `re.sub(r"//[^\n]*", "", src)` also deletes the rest of any line
    holding a URL -- `"https://example.com/x"` becomes `"https:` -- which does
    not raise: it silently shortens the region a guard then asserts over, and
    the guard reports whatever survives. Nothing in the sources scanned today
    has a URL, so that is a trap set for the edit which adds one.

    A scanner rather than a regex, because "is this `//` a comment" is a
    question about what came before it. Two known limits, both checked against
    the files scanned today and both stated here rather than left to be
    rediscovered the way the `//` one was:

    * a regex *literal* holding an escaped slash (`/https?:\/\//`) is not
      tracked, so its trailing `\//` reads as a comment start;
    * an apostrophe in bare JSX TEXT (`<p>didn't load</p>`) opens a string that
      only closes at the next one, and any comment in between survives. Every
      apostrophe in the scanned TSX is inside a comment or a double-quoted
      string, which is safe by construction -- a comment is skipped whole,
      before quote tracking sees it.

    A file that grows either needs this taught about it, not a caller working
    around it: the caller's workaround is what this function replaced.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    quote: str | None = None
    while index < length:
        char = source[index]
        if quote is not None:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
        elif char in "'\"`":
            quote = char
            out.append(char)
            index += 1
        elif source.startswith("//", index):
            newline = source.find("\n", index)
            index = length if newline == -1 else newline
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = length if end == -1 else end + 2
        else:
            out.append(char)
            index += 1
    return "".join(out)


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
    open_brace = APP_JS.index("{", match_paren(APP_JS, APP_JS.index("(", start)))
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
