"""Launch-time visibility guards (PR-1 of the 2026-08-04 backtest spec)."""

from dashboard.backend.tests._frontend_source import fn_body


def test_run_backtest_closes_editor_overlay_before_navigating():
    """A run launched from inside the agent editor must close the overlay.

    The editor is position:fixed inset:0 z-index:1200; navigateToPage()
    repaints My Agents underneath it, invisibly (spec Finding 4).
    """
    body = fn_body("async function runBacktest")
    # Asserted before indexing: a bare .index() on a removed call raises
    # ValueError("substring not found"), which names neither the call nor why
    # it has to be there.
    assert "window.AgentEditor.close(true)" in body, (
        "runBacktest() must close the agent editor overlay; force=true is safe "
        "because the editor's own Run Backtest button refuses to open the modal "
        "while isDirty (js/agent-editor.js), so there is nothing to discard"
    )
    assert "navigateToPage('playground'" in body, (
        "runBacktest() no longer navigates — re-anchor this guard on whatever "
        "replaced it, or the ordering below is vacuous"
    )
    assert body.index("window.AgentEditor.close(true)") < body.index(
        "navigateToPage('playground'"
    ), "the overlay must be closed before the repaint it would otherwise hide"
