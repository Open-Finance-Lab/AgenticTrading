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


def test_hosted_agents_are_not_given_a_runtime_flavoured_section():
    """The hosted AI Hedge Fund agent shares a shelf with every other stock
    strategy rather than getting a runtime-flavoured section of its own.

    #309 briefly split My Agents on `runtime_type === 'ai_hedge_fund'` into an
    "Open Agents" section. Rewritten 2026-08-05: the shelves are asset-class
    based now (Stocks / Crypto / Futures / Connected Agents), so a hosted agent
    lands on Stocks like everything else and its *market* -- a separate axis --
    comes from `category`. The separation the runtime split was after still
    holds, and it now generalizes: a hosted A-share agent shelves correctly and
    files under the China A-Share chip, which the runtime test could not do.
    """
    assert "Foundation Agents" not in _APP_HTML
    assert "Open Agents" not in _APP_HTML
    # The retired axes: "Prompting LLMs" was how-it-decides (now a card label),
    # "U.S. Stock Trading" was geography (now a market chip).
    assert ">Prompting LLMs</h3>" not in _APP_HTML
    assert ">U.S. Stock Trading</h3>" not in _APP_HTML
    assert ">Stocks</h3>" in _APP_HTML
    assert 'id="agentsGridStocks"' in _APP_HTML

    # The runtime-keyed *sections* are gone: shelves resolve through one
    # function, so no predicate can double-count or drop.
    assert "isOpenAgent" not in _APP_JS
    assert "agentsGridOpen" not in _APP_JS
    assert "const AGENT_SHELVES = [" in _APP_JS
    assert "function agentShelfKey(agent)" in _APP_JS
    assert "agentShelfKey(a) === 'stocks'" in _APP_JS


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

    assert editor_version >= 17
    assert styles_version >= 71
