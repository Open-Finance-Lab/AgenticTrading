"""Both allocated-capital fields live in one Configure card (2026-07-29).

Paper-trading capital used to sit in the agent editor's *header* as a 12.5px
uppercase field while backtest capital was a separate input inside the Run
Backtest modal, with nothing connecting them. They are now one card in the
editor's main column, and the modal shows the saved backtest figure read-only.

These are contract guards, not style assertions: they pin *where the inputs
live* and *that the modal no longer edits capital*, which is the behaviour the
consolidation delivers.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_configure_has_one_allocated_capital_card_with_both_inputs():
    card = _slice(_APP_HTML, 'class="agent-capital-card', "</div>\n                    <div")
    assert "Allocated Capital" in card
    assert 'id="agentEditorCashAllocation"' in card
    assert 'id="agentEditorBacktestAllocation"' in card


def test_capital_inputs_are_no_longer_in_the_editor_header():
    """The header held a cramped uppercase field; the card replaces it."""
    header = _slice(_APP_HTML, 'class="agent-editor-title-wrap"', "</header>")
    assert 'id="agentEditorCashAllocation"' not in header
    assert 'id="agentEditorBacktestAllocation"' not in header


def test_capital_limits_are_stated_next_to_each_field():
    assert _APP_HTML.count("max $3,000") >= 2
    assert "max $10,000" not in _APP_HTML


def test_editor_state_carries_backtest_allocation():
    assert "backtest_allocation" in _EDITOR_JS


_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def test_run_backtest_modal_has_no_editable_capital_input():
    """Capital is set in Configure now; the modal only reports it."""
    modal = _slice(_APP_HTML, 'id="runBacktestModal"', 'id="runBacktestModalSubmit"')
    assert modal.find('id="runBacktestBillingGroup"') >= 0
    assert 'id="backtestInitialCapital"' not in modal
    assert 'type="number"' not in modal


def test_run_backtest_modal_links_to_configure():
    modal = _slice(_APP_HTML, 'id="runBacktestModal"', 'id="runBacktestModalSubmit"')
    assert 'id="runBacktestEditCapitalBtn"' in modal
    assert "Edit in Configure" in modal


def test_backtest_capital_resolution_helper_exists():
    assert "function resolveBacktestCapital(" in _APP_JS
