"""Source-level guards for the hosted AI Hedge Fund editor mode.

The dashboard ships as vanilla JavaScript, so these checks protect the UI
contract without introducing a second frontend test toolchain.
"""

from pathlib import Path


_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")
_STYLES_CSS = (_FRONTEND / "styles.css").read_text(encoding="utf-8")


def _slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_hosted_editor_replaces_model_picker_with_managed_metadata():
    assert 'id="agentEditorModelField"' in _APP_HTML
    assert 'id="agentEditorManagedModelField"' in _APP_HTML
    assert "OpenAI · GPT-4.1" in _APP_HTML

    configure = _slice(
        _EDITOR_JS,
        "function configureEditorMode(agent)",
        "function populateModelSelect(agent)",
    )
    assert "modelField.hidden = hostedAiHedgeFund" in configure
    assert "managedModelField.hidden = !hostedAiHedgeFund" in configure
    assert ".agent-editor-model-field[hidden]" in _STYLES_CSS
    assert ".agent-editor-managed-model-field[hidden]" in _STYLES_CSS


def test_hosted_editor_never_submits_a_model_override():
    editor_state = _slice(
        _EDITOR_JS,
        "function getEditorState()",
        "function snapshotState()",
    )
    assert "model_name: hostedAiHedgeFund\n        ? ''" in editor_state


def test_robinhood_editor_behavior_is_shared_by_both_runtimes():
    configure = _slice(
        _EDITOR_JS,
        "function configureEditorMode(agent)",
        "function populateModelSelect(agent)",
    )
    open_editor = _slice(_EDITOR_JS, "function open(agent)", "function close(force)")
    editor_state = _slice(
        _EDITOR_JS,
        "function getEditorState()",
        "function snapshotState()",
    )

    assert "agentEditorBrokerPanel" not in configure
    assert "refreshRobinhoodStatus();" in open_editor
    assert "live_trading_enabled: Boolean(liveToggle?.checked)" in editor_state
