"""The slicing helpers behind the frontend static-text guards.

Worth testing because of how they fail. Every guard in test_app_toast.py and
test_create_agent_feedback.py is `assert "..." in fn_body(...)`; a helper that
returns the wrong region does not error, it just makes those assertions describe
somebody else's source -- vacuous in one direction, red for no reason in the
other. Nothing downstream can tell the difference.
"""

from dashboard.backend.tests._frontend_source import at_rule_blocks, fn_body


def test_fn_body_walks_past_the_parameter_list():
    """Matching from the textually-first "{" lands in the parameter list.

    showAgentCredentials(apiKey, options = {}) is a live instance in app.js: the
    default value's braces come before the body's, so a first-brace matcher
    returns the two-character string "{}" and every assertion against it is
    trivially false. Anchored on real source rather than a synthetic fixture so
    the case cannot quietly stop existing.
    """
    body = fn_body("function showAgentCredentials(")
    assert "getElementById('agentCredentialsModal')" in body


def test_fn_body_stops_at_the_functions_own_closing_brace():
    """The other direction: over-reading into whatever follows.

    setButtonPending is immediately followed by restoreButton, and the two are
    near-mirror images, so a slice that runs past the closing brace lets a
    "removed in restoreButton" assertion pass on setButtonPending's source.
    """
    body = fn_body("function setButtonPending(")
    assert body.endswith("}")
    assert "function restoreButton(" not in body


def test_at_rule_blocks_isolates_each_block():
    """styles.css has several reduced-motion blocks; a fallback test that swept
    them all together would pass on any one of them."""
    blocks = at_rule_blocks("@media (prefers-reduced-motion: reduce)")
    assert len(blocks) > 1
    assert all(block.endswith("}") for block in blocks)
    assert all(block.count("{") == block.count("}") for block in blocks)
