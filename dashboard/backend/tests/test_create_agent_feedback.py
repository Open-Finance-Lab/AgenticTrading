"""Create-agent gives feedback within one frame of the click.

A tester reported ~5 seconds of apparently-dead UI after clicking "Create
built-in agent". The agent was created correctly every time; the button just
never changed and nothing confirmed success. The POST itself is genuinely slow
(see the round-trip note in the spec), so the fix is feedback, not latency.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _fn_body(signature: str) -> str:
    """The named function's source, brace-matched to its real closing brace.

    Brace-matching rather than a fixed-width slice: a `[start:start + 900]`
    window over-reads into whatever unrelated top-level code happens to follow,
    so an assertion can pass on a neighbour's source instead of the function
    under test.
    """
    start = _APP_JS.index(signature)
    index = _APP_JS.index("{", start)
    depth = 0
    while True:
        if _APP_JS[index] == "{":
            depth += 1
        elif _APP_JS[index] == "}":
            depth -= 1
            if depth == 0:
                return _APP_JS[start : index + 1]
        index += 1


def _submit_fn() -> str:
    return _fn_body("async function submitCreateBuiltinAgent(")


def test_helpers_exist():
    assert "function setButtonPending(" in _APP_JS
    assert "function restoreButton(" in _APP_JS


def test_pending_state_is_set_before_the_await():
    """Set after the await, the label would appear only once the POST returned --
    exactly the window the tester experienced as dead."""
    fn = _submit_fn()
    assert "setButtonPending(" in fn
    assert fn.index("setButtonPending(") < fn.index("await API.post")


def test_pending_label_is_creating():
    assert "'Creating…'" in _submit_fn()


def test_success_confirmation_is_not_gated_on_the_grid_refresh():
    """loadAgents() is a second round trip. Confirming after it would reintroduce
    most of the delay the toast exists to cover."""
    fn = _submit_fn()
    assert "showAppToast(" in fn
    assert fn.index("showAppToast(") < fn.index("await loadAgents()")


def test_button_is_restored_on_every_path():
    """finally, not the success branch: an error must not strand a dead button."""
    fn = _submit_fn()
    finally_block = fn[fn.index("} finally {") :]
    assert "restoreButton(" in finally_block


def test_aria_busy_is_toggled_in_both_directions():
    """Set *and* removed, each checked inside the function that owns it.

    A whole-file search for "aria-busy" passes on the set call alone, so a
    restoreButton() that stopped removing the attribute -- leaving the button
    announcing itself as busy to a screen reader for the rest of the session --
    would not be caught.
    """
    assert "setAttribute('aria-busy', 'true')" in _fn_body("function setButtonPending(")
    assert "removeAttribute('aria-busy')" in _fn_body("function restoreButton(")


def test_new_agent_card_is_located_after_creation():
    assert "function highlightAgentCard(" in _APP_JS
    assert "highlightAgentCard(" in _submit_fn()


def test_highlight_uses_attribute_lookup_not_selector_interpolation():
    """Same rule refreshRunningAgentCards() follows (app.js:3370): agent ids are
    server-supplied, so never interpolate one into a selector string."""
    body = _fn_body("function highlightAgentCard(")
    assert "querySelectorAll('.agent-card[data-agent-id]')" in body
