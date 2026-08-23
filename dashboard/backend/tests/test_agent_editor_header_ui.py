"""Configure header chrome: Unsaved changes must not shove fields sideways.

Model / Market / Description / Robinhood used to live in the same flex row as
the dirty badge. Showing the badge grew the action column and narrowed every
field beside it. The toolbar is now its own row; those fields sit full-width
below it. The hidden badge still occupies its slot so the name row does not
jump either.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_configure_fields_are_not_in_the_toolbar_row():
    toolbar = _slice(
        _APP_HTML, 'class="agent-editor-toolbar"', 'class="agent-editor-header-fields"'
    )
    assert 'id="agentEditorDirtyBadge"' in toolbar
    assert 'id="agentEditorRunBacktestBtn"' in toolbar
    assert 'id="agentEditorSaveBtn"' in toolbar
    assert 'id="agentEditorNameInput"' in toolbar
    assert 'id="agentEditorModelField"' not in toolbar
    assert 'id="agentEditorCategoryField"' not in toolbar
    assert 'id="agentEditorDescription"' not in toolbar
    assert 'id="agentEditorBrokerPanel"' not in toolbar

    fields = _slice(_APP_HTML, 'class="agent-editor-header-fields"', "</header>")
    assert 'id="agentEditorModelField"' in fields
    assert 'id="agentEditorCategoryField"' in fields
    assert 'id="agentEditorDescription"' in fields
    assert 'id="agentEditorBrokerPanel"' in fields
    assert 'id="agentEditorDirtyBadge"' not in fields


def test_dirty_badge_keeps_its_slot_when_hidden():
    hidden = _slice(_STYLES, ".agent-editor-dirty-badge[hidden] {", "}")
    assert "visibility: hidden" in hidden
    assert "display: inline-flex !important" in hidden
