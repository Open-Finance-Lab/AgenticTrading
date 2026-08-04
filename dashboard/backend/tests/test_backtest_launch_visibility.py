"""Launch-time visibility guards (PR-1 of the 2026-08-04 backtest spec)."""

from dashboard.backend.tests._frontend_source import fn_body


def test_run_backtest_closes_editor_overlay_before_navigating():
    """A run launched from inside the agent editor must close the overlay.

    The editor is position:fixed inset:0 z-index:1200; navigateToPage()
    repaints My Agents underneath it, invisibly (spec Finding 4).
    """
    body = fn_body("async function runBacktest")
    close_at = body.index("window.AgentEditor.close(true)")
    navigate_at = body.index("navigateToPage('playground'")
    assert close_at < navigate_at
