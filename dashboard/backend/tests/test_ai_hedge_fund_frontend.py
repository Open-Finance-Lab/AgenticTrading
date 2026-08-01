"""Source-level guards for the hosted AI Hedge Fund editor mode.

The dashboard ships as vanilla JavaScript, so these checks protect the UI
contract without introducing a second frontend test toolchain.
"""

from pathlib import Path


_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")
_STYLES_CSS = (_FRONTEND / "styles.css").read_text(encoding="utf-8")


def _slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_hosted_editor_replaces_model_picker_with_managed_metadata():
    assert 'id="agentEditorModelField"' in _APP_HTML
    assert 'id="agentEditorManagedModelField"' in _APP_HTML
    assert "OpenRouter · nvidia/nemotron-3-nano-30b-a3b" in _APP_HTML
    assert "hosted OpenRouter model" in _APP_HTML

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


def test_marketplace_copy_label_is_scoped_to_ai_hedge_fund():
    assert (
        "const cloneLabel = isAiHedgeFundTemplate "
        "? 'Copy to My Agents' : 'Add to My Agents';"
    ) in _APP_JS


def test_stored_credential_can_be_removed_from_the_editor():
    """The DELETE credential route needs a UI path.

    Without one a user can store a third-party API key and has no way to take
    it back out -- the endpoint exists but nothing ever calls it.
    """
    assert 'id="agentEditorFinancialDatasetsRemove"' in _APP_HTML
    assert "removeFinancialDatasetsCredential" in _EDITOR_JS
    assert "credentialRequest(agent, 'DELETE')" in _EDITOR_JS
    assert (
        "getElementById('agentEditorFinancialDatasetsRemove')?.addEventListener"
        in _EDITOR_JS
    )
    assert ".agent-editor-credential-remove" in _STYLES_CSS


def test_editor_asset_cache_bust_advances_with_its_source():
    """A stale ?v= serves the old editor to every returning browser."""
    assert "js/agent-editor.js?v=17" in _APP_HTML
    assert "styles.css?v=71" in _APP_HTML
