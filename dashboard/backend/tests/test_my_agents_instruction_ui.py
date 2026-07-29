"""An empty Trading Instruction is a supported state (2026-07-29).

It used to be a silent no-op: ``getEditorState()`` set ``sendPipeline = false``
on an empty box, so the stored pipeline was never touched and the user got a
success toast for a save that changed nothing -- with no way to return an agent
to the platform's default strategy.

Empty now clears the pipeline, which makes the backend take the
``create_prompt`` branch in portfolio_manager. Two things must hold:

1. The UI says so, and shows what the default actually is.
2. The starter *backfill* is gone. It re-injected the default text whenever a
   pipeline-less agent was opened, which under the new semantics would silently
   undo a deliberate empty save on the next visit.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def test_the_empty_state_is_explained_next_to_the_textarea():
    assert (
        "Leave this empty and the agent uses the platform's default trading strategy."
        in _APP_HTML
    )


def test_the_default_instruction_is_inspectable():
    assert "See the default instruction" in _APP_HTML
    assert 'id="agentEditorDefaultInstructionText"' in _APP_HTML


def test_saving_empty_reports_the_default_was_applied():
    assert "Saved — using the default trading instruction." in _EDITOR_JS


def test_the_starter_backfill_is_gone():
    """The old backfill fought a deliberate empty save on reopen."""
    assert "if (!subAgents.length && instructionEl" not in _EDITOR_JS


def test_empty_instruction_sends_an_empty_pipeline():
    """sendPipeline must be true on empty, or the clear never reaches the API."""
    assert "sendPipeline = false" not in _EDITOR_JS
