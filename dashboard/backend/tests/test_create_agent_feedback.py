"""Create-agent gives feedback within one frame of the click.

A tester reported ~5 seconds of apparently-dead UI after clicking "Create
built-in agent". The agent was created correctly every time; the button just
never changed and nothing confirmed success. The POST itself is genuinely slow
(see the round-trip note in the spec), so the fix is feedback, not latency.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _submit_fn() -> str:
    start = _APP_JS.index("async function submitCreateBuiltinAgent(")
    depth = 0
    i = _APP_JS.index("{", start)
    while True:
        if _APP_JS[i] == "{":
            depth += 1
        elif _APP_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return _APP_JS[start : i + 1]
        i += 1


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


def test_aria_busy_is_toggled():
    assert re.search(r"aria-busy", _APP_JS)
