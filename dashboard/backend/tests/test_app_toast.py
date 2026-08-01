"""The /app toast primitive: a non-blocking success channel.

Agent creation previously closed its modal and refreshed the grid with no
confirmation at all, so a slow create read as a dead click. `alert()` -- this
file's 18-times-over convention -- is modal and blocking, which is a worse
answer for a *success* than the silence it replaces.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")

_REDUCED_MOTION = "@media (prefers-reduced-motion: reduce)"


def _at_rule_blocks(prelude: str) -> list[str]:
    """Every at-rule block with this prelude, brace-matched to its own end.

    styles.css carries eight separate reduced-motion blocks. Slicing from a
    class name to end-of-file would sweep in all the later ones, so any test
    asking "does *this* rule have a fallback" has to isolate the real block.
    """
    blocks = []
    for match in re.finditer(re.escape(prelude) + r"\s*\{", _STYLES):
        index = _STYLES.index("{", match.start())
        depth = 0
        while True:
            if _STYLES[index] == "{":
                depth += 1
            elif _STYLES[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(_STYLES[match.start() : index + 1])
                    break
            index += 1
    return blocks


def test_toast_container_is_a_polite_live_region():
    """A success message screen readers never announce is not a confirmation.

    Asserted as one whole tag, not three independent substrings: app.html:377
    (the ticker) already carries role="status" and aria-live="polite", so
    file-wide substring checks for those two would pass before the toast
    exists -- two thirds of a vacuous test.
    """
    assert (
        '<div id="appToast" class="app-toast" role="status" aria-live="polite" hidden>'
        in _APP_HTML
    )


def test_toast_helper_exists():
    assert "function showAppToast(" in _APP_JS


def test_toast_is_not_the_home_live_toast():
    """Distinct class: .home-live-toast is the Home live-decision widget in the
    same shared stylesheet, and conflating them couples two unrelated features.

    Asserts the two are never selected *together*, not merely that the string
    exists somewhere: a later `.app-toast, .home-live-toast { ... }` merge is
    precisely the coupling this guards against, and a bare substring check
    would wave it through.
    """
    assert ".app-toast" in _STYLES
    conflated = [
        line
        for line in _STYLES.splitlines()
        if ".app-toast" in line and "home-live-toast" in line
    ]
    assert not conflated, conflated


def test_toast_animation_has_a_reduced_motion_fallback():
    """Scoped to the reduced-motion block that actually names .app-toast.

    Slicing from the first ".app-toast" to end-of-file would also cover the
    unrelated reduced-motion blocks that follow it (.is-pending and
    .agent-card.is-just-created), so deleting the toast's own fallback would
    leave this passing on somebody else's rule.
    """
    blocks = _at_rule_blocks(_REDUCED_MOTION)
    assert any(".app-toast" in block for block in blocks), blocks
