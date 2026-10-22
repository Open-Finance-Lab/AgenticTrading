"""The Robinhood real-money path is commented out, not deleted.

Real capital is out of scope (docs/superpowers/specs/2026-10-14-atl-direction.md),
so on 2026-10-22 every entry point to the path was commented out and everything
behind them was left in place: ``execution/robinhood_live_service.py``,
``infrastructure/brokers/robinhood_oauth.py`` / ``robinhood_mcp.py`` and the
broker store still import, and their own suites (test_robinhood_*.py) still run.

These tests pin the disabled state from outside. A partial restore is the
failure they exist for: the panel back with its routes still unmounted gives a
user a Connect button that 404s, and the routes back without the panel mount a
real-money API whose UI nobody re-reviewed. The source guards strip comments
first, because the commented-out lines still contain every name checked here.

The restore steps live in one place, CLAUDE.md (`execution/`); deleting this
file is the last of them.
"""

import re

from dashboard.backend.app import app
from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    FRONTEND,
    fn_body,
    strip_comments,
)

_EDITOR_JS = (FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def _without_html_comments(html: str) -> str:
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def test_no_robinhood_route_is_mounted():
    mounted = sorted(
        getattr(route, "path", "")
        for route in app.routes
        if "robinhood" in getattr(route, "path", "").lower()
    )
    assert mounted == []


def test_the_editor_broker_panel_is_commented_out():
    live_markup = _without_html_comments(APP_HTML)
    for element_id in (
        "agentEditorBrokerPanel",
        "agentEditorConnectRobinhoodBtn",
        "agentEditorLiveTradingEnabled",
        "agentEditorRunLiveBtn",
    ):
        assert f'id="{element_id}"' not in live_markup
    # Commented out rather than deleted, so a restore is an uncomment.
    assert 'id="agentEditorBrokerPanel"' in APP_HTML


def test_no_script_calls_into_the_robinhood_path():
    assert "handleRobinhoodOAuthReturn();" not in strip_comments(APP_JS)

    editor = strip_comments(_EDITOR_JS)
    open_editor = fn_body("function open(agent)", editor)
    bind_events = fn_body("function bindEvents()", editor)
    assert "refreshRobinhoodStatus" not in open_editor
    for handler in ("connectRobinhood", "disconnectRobinhood", "runLive"):
        assert f"'click', {handler})" not in bind_events
