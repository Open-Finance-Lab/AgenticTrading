"""The /app toast primitive: a non-blocking success channel.

Agent creation previously closed its modal and refreshed the grid with no
confirmation at all, so a slow create read as a dead click. `alert()` -- this
file's 18-times-over convention -- is modal and blocking, which is a worse
answer for a *success* than the silence it replaces.
"""

import re

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    STYLES,
    at_rule_blocks,
    fn_body,
)

_REDUCED_MOTION = "@media (prefers-reduced-motion: reduce)"


def _toast_tag() -> str:
    match = re.search(r"<div id=\"appToast\"[^>]*>", APP_HTML)
    assert match, "no #appToast container in app.html"
    return match.group(0)


def test_toast_container_is_a_polite_live_region():
    """A success message screen readers never announce is not a confirmation.

    Asserted within the one matched tag, not as file-wide substrings: app.html:377
    (the ticker) already carries role="status" and aria-live="polite", so
    independent substring checks for those two would pass before the toast
    exists -- two thirds of a vacuous test. Matching the tag by id and asserting
    inside it keeps that property without also pinning attribute order.
    """
    tag = _toast_tag()
    assert 'role="status"' in tag
    assert 'aria-live="polite"' in tag


def test_toast_container_is_never_display_none():
    """`hidden` is display:none, and an unrendered live region is not monitored.

    Populate-then-unhide is the intuitive order and it announces nothing on most
    screen readers -- a bug invisible to every other test here, because the
    markup and the CSS are both correct and only the call order is wrong. The
    container hides itself via .app-toast's opacity, so it must never be given
    `hidden`, in markup or at runtime.
    """
    assert "hidden" not in _toast_tag()
    assert "hidden" not in fn_body("function showAppToast(")


def test_toast_message_is_cleared_rather_than_hidden():
    """What replaces `hidden` on the way out.

    Leaving the last message in the DOM strands it for anyone browsing the page
    afterwards; emptying the region keeps it registered for the next write.
    """
    assert "el.textContent = ''" in fn_body("function showAppToast(")


def test_toast_helper_exists():
    assert "function showAppToast(" in APP_JS


def test_toast_is_not_the_home_live_toast():
    """Distinct class: .home-live-toast is a leftover selector family in the same
    shared stylesheet -- no markup applies it any more -- and reviving a dead
    name is how two unrelated features end up sharing one rule.

    Asserts the two are never selected *together*, not merely that the string
    exists somewhere: a later `.app-toast, .home-live-toast { ... }` merge is
    precisely the coupling this guards against, and a bare substring check
    would wave it through.
    """
    assert ".app-toast" in STYLES
    conflated = [
        line
        for line in STYLES.splitlines()
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
    blocks = at_rule_blocks(_REDUCED_MOTION)
    assert any(".app-toast" in block for block in blocks), blocks
