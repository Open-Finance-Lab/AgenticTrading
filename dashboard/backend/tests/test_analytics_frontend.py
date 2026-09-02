"""Static safety and lifecycle contracts for browser Analytics events."""

from __future__ import annotations

import re
from pathlib import Path

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body


_ROOT = Path(__file__).resolve().parents[2]
_ANALYTICS_JS = (_ROOT / "frontend" / "js" / "analytics.js").read_text()
_AGENT_EDITOR_JS = (_ROOT / "frontend" / "js" / "agent-editor.js").read_text()


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    quote = None
    escaped = False
    for index in range(brace, len(source)):
        char = source[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"Unclosed function: {signature}")


def test_analytics_script_loads_between_app_and_page_scripts():
    app_at = APP_HTML.index('<script src="app.js?v=125" defer></script>')
    analytics_at = APP_HTML.index(
        '<script src="js/analytics.js?v=1" defer></script>'
    )
    editor_at = APP_HTML.index(
        '<script src="js/agent-editor.js?v=30" defer></script>'
    )
    assert app_at < analytics_at < editor_at


def test_analytics_session_and_heartbeat_constants_are_pinned():
    assert "const SESSION_STORAGE_KEY = 'atl-analytics-session-v1';" in _ANALYTICS_JS
    assert "const SESSION_TIMEOUT_MS = 30 * 60 * 1000;" in _ANALYTICS_JS
    assert "const HEARTBEAT_INTERVAL_MS = 30 * 1000;" in _ANALYTICS_JS


def test_page_view_map_contains_only_approved_identifiers():
    match = re.search(
        r"const PAGE_VIEW_MAP = Object\.freeze\(\{(?P<body>.*?)\}\);",
        _ANALYTICS_JS,
        re.DOTALL,
    )
    assert match
    values = set(re.findall(r":\s*'([a-z_]+)'", match.group("body")))
    assert values == {
        "home",
        "agents",
        "agent_editor",
        "backtest",
        "paper_trading",
        "competition",
        "community",
        "credits",
        "account",
    }


def test_event_payload_has_only_the_seven_approved_fields():
    body = _function_body(_ANALYTICS_JS, "function queueEvent(")
    payload = body[body.index("const payload = {") : body.index("};", body.index("const payload = {"))]
    keys = set(re.findall(r"^\s{6}([a-z_]+):", payload, re.MULTILINE))
    assert keys == {
        "event_id",
        "schema_version",
        "event_name",
        "session_id",
        "occurred_at",
        "page_view",
        "properties",
    }
    assert "credentials: 'include'" in body
    assert "keepalive: true" in body
    assert "window.csrfHeaders()" in body


def test_analytics_module_has_no_sensitive_payload_fields():
    forbidden = {
        "email",
        "display_name",
        "role",
        "api_key",
        "token",
        "prompt",
        "strategy",
        "provider_response",
        "user_agent",
    }
    for name in forbidden:
        assert not re.search(rf"[\"']{name}[\"']\s*:", _ANALYTICS_JS), name


def test_navigation_records_only_after_final_normalization():
    body = fn_body("function navigateToPage(")
    call = "window.ATLAnalytics?.recordNavigation(page, { playgroundTab, competitionTab })"
    assert call in body
    assert body.index(call) > body.index("competitionTab === 'daily'")
    assert body.index(call) > body.index("persistNavigation()")


def test_same_page_playground_tab_switch_records_navigation_after_panel_update():
    body = fn_body("function switchPlaygroundTab(")
    call = "window.ATLAnalytics?.recordNavigation('playground', { playgroundTab })"
    assert call in body
    assert body.index(call) > body.index("showPlaygroundPanel(tab)")
    assert body.index(call) < body.index("syncNavigationHistory({ replace: false })")


def test_agent_editor_enters_and_leaves_transient_view():
    open_body = _function_body(_AGENT_EDITOR_JS, "function open(")
    close_body = _function_body(_AGENT_EDITOR_JS, "function close(")
    assert "window.ATLAnalytics?.enterTransientView('agent_editor')" in open_body
    assert open_body.index("view.hidden = false") < open_body.index(
        "enterTransientView('agent_editor')"
    )
    assert "window.ATLAnalytics?.leaveTransientView()" in close_body
    assert close_body.index("view.hidden = true") < close_body.index(
        "leaveTransientView()"
    )


def test_visibility_and_heartbeat_lifecycle_is_safe():
    visibility = _function_body(_ANALYTICS_JS, "function handleVisibilityChange(")
    hide_view = _function_body(_ANALYTICS_JS, "function hideCurrentView(")
    show_view = _function_body(_ANALYTICS_JS, "function showCurrentView(")
    signed_in = _function_body(_ANALYTICS_JS, "function signedIn(")
    heartbeat = _function_body(_ANALYTICS_JS, "function sendHeartbeat(")
    assert "hideCurrentView" in visibility
    assert "showCurrentView" in visibility
    assert "page_hidden" in hide_view
    assert "page_viewed" in show_view
    assert "signedIn()" in heartbeat
    assert "getStoredAuthUser" in signed_in
    assert "document.visibilityState !== 'visible'" in heartbeat
    assert "session_heartbeat" in heartbeat


def test_ingestion_failure_never_escapes_into_navigation():
    body = _function_body(_ANALYTICS_JS, "function queueEvent(")
    assert ".catch(() =>" in body
    assert "throw" not in body
