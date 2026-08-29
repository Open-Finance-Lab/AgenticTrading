"""Configure header chrome: fields and actions stay in separate header areas."""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_AGENT_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_configure_fields_stay_out_of_header_actions():
    title_wrap = _slice(
        _APP_HTML, 'class="agent-editor-title-wrap"', 'class="agent-editor-header-actions"'
    )
    actions = _slice(_APP_HTML, 'class="agent-editor-header-actions"', "</header>")

    assert 'id="agentEditorNameInput"' in title_wrap
    assert 'id="agentEditorModelField"' in title_wrap
    assert 'id="agentEditorCategoryField"' in title_wrap
    assert 'id="agentEditorDescription"' in title_wrap
    assert 'id="agentEditorBrokerPanel"' in title_wrap

    assert 'id="agentEditorDirtyBadge"' in actions
    assert 'id="agentEditorRunBacktestBtn"' in actions
    assert 'id="agentEditorSaveBtn"' in actions
    assert 'id="agentEditorModelField"' not in actions
    assert 'id="agentEditorCategoryField"' not in actions
    assert 'id="agentEditorDescription"' not in actions
    assert 'id="agentEditorBrokerPanel"' not in actions


def test_dirty_badge_is_hidden_by_default_and_toggled_by_editor_state():
    badge = _slice(_APP_HTML, 'id="agentEditorDirtyBadge"', ">")
    assert 'class="agent-editor-dirty-badge"' in badge
    assert " hidden" in badge
    assert "badge.hidden = !dirty" in _AGENT_EDITOR_JS


def test_prompted_model_name_follows_the_model_dropdown():
    """Cards whose title is already a model label lock the name to the
    dropdown so 'DeepSeek V4 Pro' cannot drift from the selected model.
    """
    assert "function nameFollowsModel(" in _AGENT_EDITOR_JS
    assert "function syncBoundAgentName(" in _AGENT_EDITOR_JS
    assert "nameInput.readOnly = bound" in _AGENT_EDITOR_JS
    assert "agentEditorModelSelect" in _AGENT_EDITOR_JS
    assert "syncBoundAgentName();" in _AGENT_EDITOR_JS
    assert "agent-editor-name-input--bound" in _AGENT_EDITOR_JS
    styles = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    assert ".agent-editor-name-input--bound" in styles
