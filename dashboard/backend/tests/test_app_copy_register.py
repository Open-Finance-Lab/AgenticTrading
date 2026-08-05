"""Guards for the app.html/app.js plain-language copy sweep (Task C5, 2026-08-05).

C1-C4 categorized the marketplace catalog and rebuilt the My Agents shelves;
this task is the wide, mechanical sweep over the rest of the dashboard app --
modals, toasts, the Competition/Ranking pages, Account, and Playground -- that
replaces developer-register strings ("API key", "session", "pipeline",
"frontier model", LLM) with the plan's private-banker register. app.html/app.js
have no JS test harness, so, per this suite's frontend convention
(`_frontend_source`), these are asserted against the shipped source directly.
"""

import re

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body


def _strip_html_comments(html: str) -> str:
    """`html` with its `<!-- -->` comments removed, so `not in` assertions
    read live markup rather than commentary that might echo a banned phrase.
    """
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _strip_js_comments(source: str) -> str:
    """`source` with `//` and `/* */` comments removed, for the same reason."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL))


_HTML = _strip_html_comments(APP_HTML)
_JS = _strip_js_comments(APP_JS)


def _html_region(start_marker: str) -> str:
    """The `<div ...>` starting at `start_marker`, through its next `</div>`.

    Not brace-matched (this is HTML, not JS) -- safe only for a region with no
    nested `<div>` of its own, which the backtest-config row markup is.
    """
    start = _HTML.index(start_marker)
    end = _HTML.index("</div>", start)
    return _HTML[start:end]


# --- Step 1 required guards (the plan's Step 1 red set) --------------------


def test_prompting_game_is_absent_belt_and_braces():
    """C2 already retired this trust-killer subtitle; re-checked here
    case-insensitively as a second guard against it resurfacing anywhere else
    app.html's copy sweep touches.
    """
    assert "prompting game" not in _HTML.lower()


def test_real_sleeve_note_is_gone():
    assert "Real sleeve." not in _HTML


def test_llm_powered_is_absent_from_app_html():
    """LLM is developer vocabulary; "Prompting LLMs" is the plan's sole named
    exception (the category name + its description), and that string doesn't
    contain "LLM-powered" so this assertion doesn't need to carve it out.
    """
    assert "LLM-powered" not in _HTML


def test_daily_deploy_job_is_absent():
    """The daily refresh job is unwired (issue #145) -- the old copy claimed
    a mechanism that doesn't run; never say "automatically each trading day"
    either (both would be a truth violation, not just a register one).
    """
    assert "daily deploy job" not in _HTML
    assert "automatically each trading day" not in _HTML


def test_add_agent_subtitle_old_developer_register_is_gone():
    assert "persistent session and API key" not in _HTML


def test_backtest_config_prompt_label_is_replaced_with_instruction():
    """Scoped to the single config row, not the whole document -- "Prompt"
    could otherwise match unrelated substrings ("Prompting LLMs" etc.).
    """
    row = _html_region('<div class="backtest-config-row" id="backtestConfigPromptRow"')
    assert "<dt>Instruction</dt>" in row
    assert "<dt>Prompt</dt>" not in row


# --- Presence guards for the highest-value new strings ----------------------


def test_competition_organizer_line_is_present():
    assert "Organized by SecureFinAI Lab with Agentic Trading Lab." in _HTML


def test_capital_note_new_text_is_present():
    assert (
        "Reserved from your My Portfolio balance while this agent paper-trades. "
        "Backtests use a separate simulated amount and never touch it."
    ) in _HTML


# --- Additional register guards ---------------------------------------------


def test_access_key_label_replaces_api_key_in_agent_credentials_modal():
    """The one-time-key modal relabels to "Access key"; the SDK/docs bridge
    sentence in the subtitle is the deliberate exception that keeps saying
    "API key" (glossary: "API key -> 'access key' in app UI, developer bridge
    once"), so this checks the label specifically, not a blanket absence.
    """
    assert '<span>Access key</span>' in _HTML
    assert 'aria-label="Access key"' in _HTML
    assert (
        "Use the access key below to connect your own program to Agentic "
        "Trading Lab. (This is the API key in the SDK and docs.)"
    ) in _HTML


def test_new_access_key_button_and_toast_strings():
    assert ">New access key</button>" in _JS
    assert "title: 'New access key created'," in _JS
    assert "New API key" not in _JS


def test_rotate_key_confirm_and_created_toast_strings():
    assert (
        'Create a new access key for "${agent.name}"? The current key stops '
        "working right away — any connected program must switch to the new key."
    ) in _JS
    assert "title: 'New access key created'" in _JS
    assert 'Update your program — the old key no longer works.' in _JS


def test_error_alert_strings_are_plain_language():
    assert "Couldn't create a new access key. Please try again." in _JS
    assert "Couldn't start Discord linking. Please sign in and try again." in _JS
    assert "Couldn't add this template. Please try again." in _JS
    assert "Couldn't delete the agent. Please try again." in _JS
    assert "Robinhood connection failed. Please try again on a desktop computer." in _JS
    # The old developer-register alerts must not survive alongside the new ones.
    assert "Failed to create new API key" not in _JS
    assert "Could not start Discord linking. Are you signed in?" not in _JS
    assert "Failed to add template" not in _JS
    assert "Failed to delete agent" not in _JS
    assert "Use localhost and a desktop browser." not in _JS


def test_managed_model_field_is_hosted_ai_model():
    assert "Hosted AI model" in _HTML
    assert "Managed for you by Agentic Trading Lab" in _HTML
    assert "Managed provider / model" not in _HTML


def test_ai_hedge_fund_panel_copy_is_updated():
    assert "AI Hedge Fund analyst panel" in _HTML
    assert (
        "Choose the analysts that shape this agent's strategy. The AI model "
        "and its settings are hosted and managed by Agentic Trading Lab."
    ) in _HTML
    assert "analyst committee" not in _HTML
    assert "upstream analysts" not in _HTML


def test_financial_datasets_label_names_its_source():
    assert "Financial Datasets access key (from financialdatasets.ai)" in _HTML


def test_credential_storage_note_is_plain_language():
    assert (
        "Encrypted and stored securely. For your protection, it is never "
        "shown again after you save."
    ) in _HTML
    assert "Encrypted in credential storage." not in _HTML


def test_backtest_hint_uses_strategy_and_limit():
    assert "Multi-step strategies can take several minutes (limit: 10 minutes)." in _HTML
    assert "Multi-step agent pipelines" not in _HTML


def test_home_leaderboard_column_says_ai_model():
    """Scoped to the home leaderboard module -- other "Model" labels in the
    document (the model-select form field, the agent editor) are untouched
    by this row and must stay bare "Model".
    """
    start = _HTML.index('id="homeModuleRanking"')
    end = _HTML.index("</article>", start)
    region = _HTML[start:end]
    assert "<span>AI Model</span>" in region


def test_join_discord_expands_on_home_and_community_not_competition():
    """The row scopes this to "app.html home + community surfaces"; the
    Competition/Leaderboard page's own Discord link is a different surface
    and must stay bare, per this row's own precedent (leave what's out of
    scope alone rather than expand every occurrence blindly).
    """
    assert _HTML.count("Join our Discord community") == 2
    assert ">Join Discord<" in _HTML  # the untouched Competition instance


def test_playground_home_preview_title_is_plain_language():
    assert "agent-playground.exe" not in _HTML
    assert "Example: a conversation with your agent" in _HTML


def test_resource_subtitles_are_plain_language():
    assert "Guides and reference documentation" in _HTML
    assert "Talk strategy with other members" in _HTML
    assert "Source code and worked examples" in _HTML
    assert "Guides and API reference" not in _HTML
    assert "Chat with other builders" not in _HTML
    assert "Open source and examples" not in _HTML


def test_decision_source_row_says_decision_method():
    assert "<dt>Decision method</dt>" in _HTML
    assert "<dt>Decision source</dt>" not in _HTML


def test_market_data_notice_names_no_ai_not_no_llm():
    assert "Simulated practice data — repeatable results, rule-based decisions only (no AI)" in _HTML
    assert "vn.py simulated bars · deterministic · no LLM calls" not in _HTML


def test_sharpe_tooltip_is_plain_language():
    assert 'title="Risk-adjusted return, annualized from hourly results."' in _HTML
    assert "Annualized for hourly data (sqrt(252*6.5))" not in _HTML


def test_competition_about_copy_says_ai_not_llm():
    assert "a paper-trading competition for AI-powered agents." in _HTML
    assert "comparing leading AI models, baseline strategies, and market indices." in _HTML
    assert "paper-trading competition for LLM-powered agents" not in _HTML
    assert "comparing provided LLM models" not in _HTML


def test_participants_empty_state_says_ai_models_and_baseline_strategies():
    assert "The Ranking board shows AI models and baseline strategies only." in _HTML
    assert "The Ranking board shows models and baselines only." not in _HTML


def test_account_description_is_plain_language():
    assert "Your profile and sign-in details." in _HTML
    assert "Signed-in profile and session." not in _HTML


def test_home_mock_chat_gloss_matches_landing():
    """The row names "Paper Trading tab first mention," but the tab's own
    content (`#paperTradingView`) has no bare "paper trading" prose -- only
    metric labels ("Portfolio Value", "Cash Available", ...). An earlier bare
    mention exists in document order (the auth modal subtitle, "Optional --
    backtest and paper trading work without an account."), but that's a
    compact overlay shown only on user action, outside the page's normal
    reading flow -- the term there is incidental to the sentence's point
    ("works without an account"), and splicing in the long landing gloss
    would bloat a small dialog. The one this row's "gloss identical to
    landing" phrase actually points at is the home page's mock chat demo:
    the same chat-bubble sentence PR A1 already glossed on the landing
    page's Hero.tsx. Applying it here mirrors that precedent rather than
    inventing a new location.
    """
    start = _HTML.index('id="homePlaygroundChat"')
    end = _HTML.index("</section>", start)
    region = _HTML[start:end]
    assert '<span class="home-headline-accent">paper trading</span>' in region
    assert (
        "— practice trading with simulated money at live market prices —"
    ) in region
    assert "and alert you when Berkshire's next 13F drops?" in region


def test_builtin_agent_placeholder_asks_what_makes_it_different():
    assert 'placeholder="What makes this agent different?"' in _HTML
    assert "What is this agent's edge?" not in _HTML


# --- Fix round 1 (2026-08-05): six visible "LLM" strings left in app.js -----
#
# Review found six more user-visible strings using developer-register "LLM"
# that the first pass missed (they don't contain any of the ★ row phrases, so
# the original sweep's targeted greps didn't surface them). Curated exact
# strings, not a blanket "LLM absent from app.js" assertion -- app.js
# legitimately keeps "Prompting LLMs" (the glossary's one named exception,
# read off `AGENT_SHELVES`) and internal identifiers like
# `LLM_DECISION_SOURCE`/`allowsLLM`, and a blanket check would false-positive
# on both.


def test_token_cost_label_says_ai_not_llm():
    assert "est. AI cost" in _JS
    assert "est. LLM cost" not in _JS


def test_vnpy_readonly_label_says_no_ai_not_no_llm_calls():
    assert "Rule-based — simulated practice data, no AI involved" in _JS
    assert "vn.py simulation makes no LLM calls" not in _JS


def test_model_select_hint_is_glossary_compliant():
    assert (
        "Uses this agent's AI model by default. Choose Rule-based for "
        "repeatable decisions without AI."
    ) in _JS
    assert "deterministic decisions without LLM calls" not in _JS


def test_provider_not_configured_error_says_ai_not_llm():
    assert (
        "The selected AI provider is not configured. Configure the "
        "provider or choose Rule-based."
    ) in _JS
    assert "The selected LLM provider is not configured." not in _JS


def test_decision_source_fallback_label_says_ai_not_llm():
    """Rendered directly under the "Decision method" label (row 23's
    rename) -- the old fallback value would have contradicted that rename
    in the same panel.
    """
    assert "'AI / Rule-based'" in _JS
    assert "'LLM / Rule-based'" not in _JS


def test_algo_submit_status_says_ai_not_llm():
    assert "Submitting backtest — real market data + AI…" in _JS
    assert "Submitting real backtest (Alpaca + LLM)…" not in _JS
