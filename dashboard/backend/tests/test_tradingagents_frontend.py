"""Source-level guards for the hosted TradingAgents editor mode.

The dashboard ships as vanilla JavaScript, so these checks protect the UI
contract without a second frontend test toolchain -- the same approach
``test_ai_hedge_fund_frontend.py`` takes for the runtime that shipped first.

What this file is really pinning: the editor used to key every hosted decision
off one AI-Hedge-Fund-shaped boolean, so a TradingAgents agent fell through to
the pipeline branch and was *mis-saved* -- a model picker it does not own, a
prompt pipeline it never consumes, and no ``runtime_config`` at all.
"""

import re
from pathlib import Path


_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")

_UPSTREAM_ANALYSTS = ("market", "social", "news", "fundamentals")


def _slice(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def _tradingagents_analyst_records() -> list[dict[str, str]]:
    """Every entry of the shipped roster, as a field map.

    Field-by-field rather than one fixed-shape regex: the previous version
    required ``}`` immediately after ``description``, so adding the
    ``unavailable`` reason to an analyst silently dropped it from the records
    and the "exactly four analysts" guard started passing on one.
    """
    metadata = _slice(_EDITOR_JS, "const TRADINGAGENTS_ANALYSTS = [", "\n  ];")
    records = []
    for block in re.findall(r"\{([^{}]+)\}", metadata):
        fields = dict(re.findall(r"(\w+): '((?:[^'\\]|\\.)*)'", block))
        assert "id" in fields, block
        records.append(fields)
    return records


def test_editor_offers_exactly_the_four_upstream_analysts():
    """Upstream's registry has four entries (``tradingagents/graph/setup.py``).
    Offering a fifth would be rejected by ``normalize_runtime_config`` only
    after the user pressed save."""
    records = _tradingagents_analyst_records()
    assert [record["id"] for record in records] == list(_UPSTREAM_ANALYSTS)


def test_every_tradingagents_analyst_has_a_concise_description():
    for record in _tradingagents_analyst_records():
        assert record["label"].strip(), record["id"]
        assert record["description"].strip().endswith("."), record["id"]
        assert len(record["description"]) <= 200, record["id"]


def test_the_lookahead_unsafe_analysts_carry_a_reason():
    """The editor's copy of the withdrawal must match the server's.

    The server owns the decision (``TRADINGAGENTS_LOOKAHEAD_EVIDENCE`` in
    ``domain/agents/runtime.py``) and refuses the roster with a 422; this pins
    that the UI withdraws exactly the same three, so it cannot offer a run that
    always fails. The *behaviour* -- disabled checkbox, visible reason, cleared
    selection -- is covered by executing the renderer in
    ``test_agent_editor_hosted_behavior.py``.
    """
    from dashboard.backend.domain.agents.runtime import (
        TRADINGAGENTS_LOOKAHEAD_EVIDENCE,
    )

    records = _tradingagents_analyst_records()
    withdrawn = {r["id"] for r in records if r.get("unavailable")}
    assert withdrawn == set(TRADINGAGENTS_LOOKAHEAD_EVIDENCE)
    for record in records:
        if record.get("unavailable"):
            # Shown to the user verbatim, so it has to read as a sentence.
            assert record["unavailable"].strip().endswith("."), record["id"]
            assert len(record["unavailable"]) <= 200, record["id"]


def test_hosted_runtimes_share_one_analyst_roster_table():
    """The gate is a per-runtime roster, not a per-runtime branch: adding a
    hosted runtime that forgets the editor should be impossible to miss."""
    roster = _slice(
        _EDITOR_JS,
        "const HOSTED_RUNTIME_ANALYSTS = {",
        "};",
    )
    assert "[AI_HEDGE_FUND_RUNTIME]: AI_HEDGE_FUND_ANALYSTS" in roster
    assert "[TRADINGAGENTS_RUNTIME]: TRADINGAGENTS_ANALYSTS" in roster


def test_the_editor_gate_is_runtime_agnostic():
    body = _slice(
        _EDITOR_JS,
        "function hostedAnalystRoster(",
        "function escapeHtml(",
    )
    assert "HOSTED_RUNTIME_ANALYSTS[" in body
    assert "function isHostedRuntimeAgent(" in body


def test_tradingagents_saves_runtime_config_and_no_model_override():
    editor_state = _slice(
        _EDITOR_JS,
        "function getEditorState()",
        "function snapshotState()",
    )
    # Model and pipeline are platform-owned for every hosted runtime.
    assert "model_name: hostedRuntime\n        ? ''" in editor_state
    assert "let sendPipeline = !hostedRuntime;" in editor_state
    assert "runtime_config: hostedRuntime" in editor_state
    assert "analysts: selectedHostedAnalysts()" in editor_state


def test_the_financial_datasets_field_stays_ai_hedge_fund_only():
    """TradingAgents reads keyless yfinance data. Showing an agent-scoped
    market-data key on its editor would invite a credential it never uses --
    and it is a different key from both the platform OpenRouter key and the
    ATL_API_KEY that Connect Agent issues to external agents."""
    assert 'id="agentEditorFinancialDatasetsField"' in _APP_HTML

    configure = _slice(
        _EDITOR_JS,
        "function configureEditorMode(agent)",
        "function populateModelSelect(agent)",
    )
    assert "isAiHedgeFundAgent(agent)" in configure
    assert "credentialField.hidden = !aiHedgeFund" in configure

    editor_state = _slice(
        _EDITOR_JS,
        "function getEditorState()",
        "function snapshotState()",
    )
    assert "financial_datasets_api_key: isAiHedgeFundAgent()" in editor_state


def test_hosted_panel_names_the_runtime_it_is_configuring():
    """One shared panel serves both runtimes, so its heading has to follow the
    agent rather than stay hardcoded to the runtime that shipped first."""
    assert 'id="agentEditorHostedPanelTitle"' in _APP_HTML

    configure = _slice(
        _EDITOR_JS,
        "function configureEditorMode(agent)",
        "function populateModelSelect(agent)",
    )
    assert "HOSTED_RUNTIME_LABELS" in configure


def test_the_managed_model_line_is_not_hard_coded():
    """This test previously asserted the opposite, and was wrong.

    It pinned the literal "OpenRouter - nvidia/nemotron-3-nano-30b-a3b" in
    app.html and claimed in its own docstring that both hosted runtimes share
    one platform-managed model. Neither held: nothing ever wrote that element,
    production resolves CommonStack first, and TradingAgents runs on
    claude-haiku-4-5 there. So the panel named the wrong provider and the
    wrong model for every TradingAgents agent, and this guard kept it that
    way.

    The value now comes from GET /api/v1/agents/hosted-runtimes. The
    behaviour -- which text is rendered for which runtime -- is executed in
    test_agent_editor_hosted_behavior.py; what is checked here is that no
    hard-coded pair has come back.
    """
    assert "nemotron" not in _APP_HTML, (
        "a model name in app.html is a claim nothing verifies"
    )
    assert "claude-haiku" not in _APP_HTML
    configure = _slice(
        _EDITOR_JS,
        "function configureEditorMode(agent)",
        "function populateModelSelect(agent)",
    )
    assert "managedModelField.hidden = !hostedRuntime" in configure
    # The panel is filled from the server, not from a constant.
    assert "renderManagedModel(agent)" in configure
    assert "loadHostedRuntimeModels()" in configure
