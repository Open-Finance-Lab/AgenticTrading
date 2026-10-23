"""Source-level guards for the hosted AI Hedge Fund editor mode.

The dashboard ships as vanilla JavaScript, so these checks protect the UI
contract without introducing a second frontend test toolchain.
"""

import re
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


def _analyst_metadata_records() -> list[tuple[str, str, str]]:
    metadata = _slice(
        _EDITOR_JS,
        "const AI_HEDGE_FUND_ANALYSTS = [",
        "];",
    )
    return re.findall(
        r"\{\s*id: '([^']+)',\s*label: '([^']+)',\s*"
        r"description: '([^']+)',\s*\}",
        metadata,
    )


def test_every_selectable_analyst_has_a_concise_description():
    records = _analyst_metadata_records()
    expected_ids_and_labels = [
        ("aswath_damodaran", "Aswath Damodaran"),
        ("ben_graham", "Ben Graham"),
        ("bill_ackman", "Bill Ackman"),
        ("cathie_wood", "Cathie Wood"),
        ("charlie_munger", "Charlie Munger"),
        ("michael_burry", "Michael Burry"),
        ("mohnish_pabrai", "Mohnish Pabrai"),
        ("nassim_taleb", "Nassim Taleb"),
        ("peter_lynch", "Peter Lynch"),
        ("phil_fisher", "Phil Fisher"),
        ("rakesh_jhunjhunwala", "Rakesh Jhunjhunwala"),
        ("stanley_druckenmiller", "Stanley Druckenmiller"),
        ("warren_buffett", "Warren Buffett"),
        ("technical_analyst", "Technical"),
        ("fundamentals_analyst", "Fundamentals"),
        ("growth_analyst", "Growth"),
        ("news_sentiment_analyst", "News Sentiment"),
        ("sentiment_analyst", "Sentiment"),
        ("valuation_analyst", "Valuation"),
    ]

    assert [(analyst_id, label) for analyst_id, label, _ in records] == (
        expected_ids_and_labels
    )
    for _, _, description in records:
        assert description == description.strip()
        assert description
        assert len(description.split()) <= 30
        assert description[-1] in ".!?"


def test_analyst_tooltips_support_pointer_and_keyboard_users():
    renderer = _slice(
        _EDITOR_JS,
        "function renderAiHedgeFundAnalysts(agent)",
        "function selectedAiHedgeFundAnalysts()",
    )

    assert 'value="${escapeHtml(id)}"' in renderer
    assert 'aria-labelledby="${escapeHtml(labelId)}"' in renderer
    assert 'aria-describedby="${escapeHtml(tooltipId)}"' in renderer
    assert 'role="tooltip"' in renderer
    assert "${escapeHtml(description)}" in renderer
    assert "${selected.has(id) ? 'checked' : ''}" in renderer
    assert "option.addEventListener('mouseenter'" in renderer
    assert "option.addEventListener('mouseleave'" in renderer
    assert "option.addEventListener('focusin'" in renderer
    assert "option.addEventListener('focusout'" in renderer

    positioner = _slice(
        _EDITOR_JS,
        "function positionAiHedgeFundTooltip(option)",
        "function renderAiHedgeFundAnalysts(agent)",
    )
    assert "getBoundingClientRect()" in positioner
    assert "window.visualViewport" in positioner
    assert "--analyst-tooltip-left" in positioner
    assert "--analyst-tooltip-top" in positioner
    assert "'resize'" in _EDITOR_JS
    assert "'scroll'" in _EDITOR_JS

    assert ".agent-editor-analyst-option:has(input:focus-visible)" in _STYLES_CSS
    assert ".agent-editor-analyst-option.is-tooltip-visible" in _STYLES_CSS
    tooltip_styles = _slice(
        _STYLES_CSS,
        ".agent-editor-analyst-tooltip {",
        ".agent-editor-analyst-option.is-tooltip-visible",
    )
    reveal_styles = _slice(
        _STYLES_CSS,
        ".agent-editor-analyst-option.is-tooltip-visible",
        "@media (prefers-reduced-motion: reduce)",
    )
    assert "position: fixed" in tooltip_styles
    assert "visibility: hidden" in tooltip_styles
    assert "pointer-events: none" in tooltip_styles
    assert "max-width:" in tooltip_styles
    assert "overflow-wrap: anywhere" in tooltip_styles
    assert "opacity: 1" in reveal_styles
    assert "visibility: visible" in reveal_styles


def test_analyst_tooltips_do_not_change_runtime_config_submission():
    selected = _slice(
        _EDITOR_JS,
        "function selectedAiHedgeFundAnalysts()",
        "function setFinancialDatasetsStatus",
    )
    editor_state = _slice(
        _EDITOR_JS,
        "function getEditorState()",
        "function snapshotState()",
    )

    assert ').map((input) => input.value);' in selected
    assert "? { analysts: selectedAiHedgeFundAnalysts() }" in editor_state


def test_hosted_editor_replaces_model_picker_with_managed_metadata():
    assert 'id="agentEditorModelField"' in _APP_HTML
    assert 'id="agentEditorManagedModelField"' in _APP_HTML
    assert "OpenRouter · nvidia/nemotron-3-nano-30b-a3b" in _APP_HTML
    assert "hosted and managed by Agentic Trading Lab" in _APP_HTML

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


def test_marketplace_cta_is_unified_add_to_my_agents():
    """Superseded 2026-08-05 (Task C4): "Copy to My Agents" was scoped to the
    AI Hedge Fund template by an `isAiHedgeFundTemplate` ternary; PR #253's
    canonical CTA is "Add to My Agents" everywhere, so the ternary is gone
    and every card -- AI Hedge Fund included -- renders the one string.
    """
    assert "const cloneLabel = 'Add to My Agents';" in _APP_JS
    assert "isAiHedgeFundTemplate" not in _APP_JS
    assert "Copy to My Agents" not in _APP_JS


def test_hosted_agents_land_on_the_open_agents_shelf():
    """Hosted runtimes (AI Hedge Fund) render under Open Agents, not mixed
    into LLMs. Shelves still resolve through one function, so no
    predicate can double-count or drop an agent.

    runtime_type is always present and truthy (server-defaulted to 'pipeline'),
    so the hosted check MUST be an inequality against 'pipeline'.
    """
    assert "Foundation Agents" not in _APP_HTML
    assert ">Open Agents</h3>" in _APP_HTML
    assert ">LLMs</h3>" in _APP_HTML
    assert ">Prompting LLMs</h3>" not in _APP_HTML
    assert ">U.S. Stock Trading</h3>" not in _APP_HTML
    assert ">Stocks</h3>" not in _APP_HTML
    assert 'id="agentsGridOpen"' in _APP_HTML
    assert 'id="agentsGridPrompted"' in _APP_HTML

    assert "isOpenAgent" not in _APP_JS
    assert "const AGENT_SHELVES = [" in _APP_JS
    assert "function agentShelfKey(agent)" in _APP_JS
    assert "agentShelfKey(a) === 'prompted'" in _APP_JS
    assert "agentShelfKey(a) === 'open'" in _APP_JS
    key_fn = _APP_JS.split("function agentShelfKey(agent)", 1)[1].split("\n}", 1)[0]
    assert "!== 'pipeline'" in key_fn
    assert "return 'open'" in key_fn
    assert "return 'prompted'" in key_fn


def test_uncategorized_hosted_agents_still_resolve_to_the_us_market():
    """Every AI Hedge Fund agent cloned before shelving shipped carries
    `category: null`, and those rows are durable (CONTENT_DATABASE_URL).

    Without a runtime fallback those agents resolve to no market at all, so the
    U.S. chip would hide the very hosted agents it exists to show. This is
    deliberately not a SQL backfill: the fallback also covers rows served by a
    backend that predates the column and sends no `category` field at all,
    which a one-shot migration cannot reach.
    """
    assert "const LEGACY_RUNTIME_MARKET = { ai_hedge_fund: 'us_stocks' }" in _APP_JS
    # ...and it must be consulted only after a real category, so a hosted agent
    # explicitly filed on another market stays where the user put it.
    key_fn = _APP_JS.split("function agentMarketKey(agent)", 1)[1].split("\n}", 1)[0]
    assert key_fn.index("MARKET_LABELS[slug]") < key_fn.index("LEGACY_RUNTIME_MARKET")


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
    """A stale ?v= serves the old editor to every returning browser.

    Parsed >= rather than literal pins, matching
    test_frontend_account_page.py::test_cache_bust_versions_were_bumped and for
    the reason that test already documents: styles.css is the single shared
    stylesheet, so its counter is bumped by unrelated work too, and an equality
    assert turns CI red on whichever PR happens to bump it next (the failure
    mode that blocked #88-#91). The floors are the versions that shipped the
    hosted editor -- going backwards would serve a stale editor, which is what
    this guard is actually for.
    """
    editor_version = int(re.search(r"js/agent-editor\.js\?v=(\d+)", _APP_HTML).group(1))
    styles_version = int(re.search(r"styles\.css\?v=(\d+)", _APP_HTML).group(1))

    assert editor_version >= 27
    assert styles_version >= 87
