"""Create-agent gives feedback within one frame of the click.

A tester reported ~5 seconds of apparently-dead UI after clicking "Create
built-in agent". The agent was created correctly every time; the button just
never changed and nothing confirmed success. The POST itself is genuinely slow
(see the round-trip note in the spec), so the fix is feedback, not latency.
"""

from dashboard.backend.tests._frontend_source import APP_JS, fn_body


def _submit_fn() -> str:
    return fn_body("async function submitCreateBuiltinAgent(")


def test_helpers_exist():
    assert "function setButtonPending(" in APP_JS
    assert "function restoreButton(" in APP_JS


def test_pending_state_is_set_before_the_await():
    """Set after the await, the label would appear only once the POST returned --
    exactly the window the tester experienced as dead."""
    fn = _submit_fn()
    assert "setButtonPending(" in fn
    assert fn.index("setButtonPending(") < fn.index("await API.post")


def test_pending_label_is_creating():
    assert "'Creating…'" in _submit_fn()


def test_idle_label_is_captured_and_restored():
    """Both halves of the round trip, each inside the function that owns it.

    Losing either one is worse than the dead click this whole file is about: the
    button reads "Creating…" for the rest of the session, and cannot recover.
    setButtonPending captures the idle label once, behind an `=== undefined`
    guard, so a later create finds dataset.idleLabel already set to "Creating…"
    (or never set at all) and has nothing to restore from. A whole-file search
    for either string passes on the other one's line.
    """
    assert "btn.dataset.idleLabel = btn.textContent" in fn_body("function setButtonPending(")
    assert "btn.textContent = btn.dataset.idleLabel" in fn_body("function restoreButton(")


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
    assert "setAttribute('aria-busy', 'true')" in fn_body("function setButtonPending(")
    assert "removeAttribute('aria-busy')" in fn_body("function restoreButton(")


def test_new_agent_card_is_located_after_creation():
    assert "function highlightAgentCard(" in APP_JS
    assert "highlightAgentCard(" in _submit_fn()


def test_highlight_uses_attribute_lookup_not_selector_interpolation():
    """Same rule refreshRunningAgentCards() follows (app.js:3370): agent ids are
    server-supplied, so never interpolate one into a selector string."""
    body = fn_body("function highlightAgentCard(")
    assert "querySelectorAll('.agent-card[data-agent-id]')" in body


def test_cards_carry_the_attribute_the_highlight_reads():
    """The writer half. Without it the reader matches nothing, forever.

    highlightAgentCard() is a lookup against an attribute that renderAgentCards()
    has to put there; the two are only connected through the DOM, so every other
    test in this file passes with the setAttribute call deleted and the feature a
    silent no-op. Asserted inside renderAgentCards because ~8 *buttons* per card
    carry a data-agent-id too -- a whole-file search matches those instead.
    """
    assert (
        "card.setAttribute('data-agent-id', agent.agent_id)"
        in fn_body("function renderAgentCards(")
    )


def test_external_create_uses_the_same_feedback_primitives():
    """The sibling flow posts to the same slow /api/v1/agents endpoint.

    Left on a bare `disabled = true` it reproduces the exact dead click this file
    exists to fix, and its confirmation (the API key, shown once and recoverable
    only by rotating) sat behind loadAgents().
    """
    fn = fn_body("async function submitCreateExternalAgent(")
    assert "setButtonPending(" in fn
    assert "restoreButton(" in fn[fn.index("} finally {") :]
    assert fn.index("showAgentCredentials(") < fn.index("await loadAgents()")
