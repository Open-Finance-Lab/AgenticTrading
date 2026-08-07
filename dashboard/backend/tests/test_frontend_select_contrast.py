"""Guards for dark-theme readability of native form controls.

A ``<select>`` popup is drawn by the browser, not by us: the list panel is
painted from the UA's *color scheme*, while the text inherits our
``--text-primary`` (#e5e7eb). With no ``color-scheme`` declared, the UA defaults
to light -- near-white text on a white panel, i.e. an unreadable menu. The fix
is ``color-scheme: dark`` on the control (plus explicit ``option`` colors, which
is what Firefox actually honours).

None of this is observable from Python, and the frontend has no JS test harness,
so -- following the convention of the other ``test_frontend_*`` modules here --
these contracts are asserted against the shipped stylesheet directly.

The picker-indicator case is the subtle one, and it is a regression that shipped
inside the very PR that added ``color-scheme`` (#265): ``color-scheme`` is
declared on the *shared* ``.date-input, .control-select`` rule, so it also
changed how the UA draws the date field's calendar glyph -- from dark (which the
old ``filter: invert(0.8)`` hack existed to flip *light*) to light. The invert
then ran backwards and made the button near-black on --bg-input. Any invert()
hack over a native control's internals is coupled to color-scheme this way.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_STYLES_CSS = _FRONTEND / "styles.css"

# Selectors that style a <select>. Each must opt into the dark UA scheme.
_SELECT_SELECTORS = (
    ".auth-field select",
    ".control-select",
    ".filter-select",
    ".agent-editor-model-select",
)

# Classes whose rule carries `color-scheme: dark`, keyed by how markup spells
# them. `.auth-field select` is a descendant rule and so has no class of its own.
_COVERED_CLASSES = {"control-select", "filter-select", "agent-editor-model-select"}

# <select>s covered by a descendant rule rather than their own class.
_DESCENDANT_COVERED_IDS = {
    "builtinAgentModel",  # inside <label class="auth-field"> -> `.auth-field select`
    "duplicateAgentModel",  # inside <label class="auth-field"> -> `.auth-field select`
}

# The only <select> that is never rendered: no code path clears its `hidden`
# attribute, so it is pure JS-side state. Do NOT extend this set just because an
# element ships `hidden` in the markup -- `#modelSelect` (app.js: `modelSelect
# .hidden = !isIFind`) and `#backtestRunSelect` both start hidden and are later
# revealed, which is precisely the case this guard has to keep covering.
_NEVER_RENDERED_IDS = {"backtestAgentSelect"}

_RULE_RE = re.compile(r"(?P<sel>[^{}]+)\{(?P<body>[^{}]*)\}", re.S)
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_SELECT_TAG_RE = re.compile(r"<select\b[^>]*>", re.S)


def _rule_bodies(selector: str) -> list[str]:
    """Bodies of rules that declare `selector` as a whole item of their list.

    Matching on list items rather than substrings keeps `.control-select:focus`
    from masquerading as `.control-select`.

    Comments are stripped first: these assertions are about declarations, and a
    rule's own prose explaining `invert()` or `color-scheme` must neither
    satisfy nor break them.
    """
    css = _COMMENT_RE.sub("", _STYLES_CSS.read_text(encoding="utf-8"))
    bodies = []
    for match in _RULE_RE.finditer(css):
        items = [item.strip() for item in match.group("sel").split(",")]
        if selector in items:
            bodies.append(match.group("body"))
    return bodies


def test_every_styled_select_declares_the_dark_ua_scheme():
    for selector in _SELECT_SELECTORS:
        bodies = _rule_bodies(selector)
        assert bodies, f"no rule declares {selector} in styles.css"
        assert any("color-scheme: dark" in body for body in bodies), (
            f"{selector} has no `color-scheme: dark`; its popup will render as "
            "light-on-white in the dark dashboard"
        )


def test_every_styled_select_colors_its_options():
    """Firefox ignores color-scheme for option paint; these rules are what it uses."""
    for selector in _SELECT_SELECTORS:
        bodies = _rule_bodies(f"{selector} option")
        assert bodies, f"no `{selector} option` rule in styles.css"
        assert any("var(--bg-card)" in body for body in bodies), (
            f"`{selector} option` must set an explicit dark background"
        )


def test_calendar_picker_indicator_is_not_inverted():
    """Regression guard -- see this module's docstring.

    `.date-input` shares its rule with `.control-select`, so it inherits
    `color-scheme: dark` and the UA already draws this glyph light. Re-adding an
    invert() would turn it near-black on --bg-input and hide the picker button.
    """
    bodies = _rule_bodies(".date-input::-webkit-calendar-picker-indicator")
    assert bodies, "the calendar picker indicator rule vanished from styles.css"
    for body in bodies:
        assert "invert(" not in body, (
            "the calendar glyph is already light under `color-scheme: dark`; "
            "inverting it makes the date picker button invisible"
        )


def test_no_select_ships_without_dark_coverage():
    """A new <select> that nobody styled is exactly how #265's bug got in.

    Every <select> is checked, including the ones that ship `hidden`. Exempting
    those would be the tempting shortcut and it is wrong: `#modelSelect` and
    `#backtestRunSelect` are both hidden in the markup and both revealed by
    app.js later, so a hidden-skip would wave through the exact elements whose
    popup a user does eventually open.
    """
    html = _APP_HTML.read_text(encoding="utf-8")

    uncovered = []
    for tag in _SELECT_TAG_RE.findall(html):
        element_id = (re.search(r'id="([^"]*)"', tag) or [None, ""])[1]
        if element_id in _DESCENDANT_COVERED_IDS or element_id in _NEVER_RENDERED_IDS:
            continue
        classes = set((re.search(r'class="([^"]*)"', tag) or [None, ""])[1].split())
        if not classes & _COVERED_CLASSES:
            uncovered.append(element_id or tag)

    assert not uncovered, (
        f"<select>s with no dark-scheme rule: {uncovered}. Give each a class "
        f"from {sorted(_COVERED_CLASSES)}, or add a rule and list it here."
    )
