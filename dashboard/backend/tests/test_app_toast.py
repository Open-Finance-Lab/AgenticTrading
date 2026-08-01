"""The /app toast primitive: a non-blocking success channel.

Agent creation previously closed its modal and refreshed the grid with no
confirmation at all, so a slow create read as a dead click. `alert()` -- this
file's 18-times-over convention -- is modal and blocking, which is a worse
answer for a *success* than the silence it replaces.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")


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
    same shared stylesheet, and conflating them couples two unrelated features."""
    assert ".app-toast" in _STYLES


def test_toast_animation_has_a_reduced_motion_fallback():
    block = _STYLES[_STYLES.index(".app-toast") :]
    assert "prefers-reduced-motion" in block
