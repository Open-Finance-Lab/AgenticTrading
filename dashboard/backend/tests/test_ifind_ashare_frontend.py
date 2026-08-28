"""Source contracts for the controlled iFinD A-share backtest UI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_APP_JS = _FRONTEND / "app.js"
_STYLES = _FRONTEND / "styles.css"


@pytest.fixture(scope="module")
def html() -> str:
    return _APP_HTML.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return _APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def css() -> str:
    return _STYLES.read_text(encoding="utf-8")


def _attr(source: str, attr: str, value: str) -> bool:
    return bool(re.search(rf'{attr}\s*=\s*["\']{re.escape(value)}["\']', source))


def test_ifind_option_is_feature_gated_in_javascript_not_html(html, js):
    assert not _attr(html, "value", "ifind_ashare")
    assert re.search(r"features\.ifind_ashare_enabled\s*===\s*true", js)
    assert re.search(r"option\.value\s*=\s*['\"]ifind_ashare['\"]", js)
    assert "iFinD China A-Shares (60 min)" in js


def test_registered_a_share_universes_are_visible_and_complete(html, js):
    assert _attr(html, "id", "ifindAshareUniverse")
    assert _attr(html, "id", "ifindAshareNotice")
    assert _attr(html, "id", "ifindAshareUniverseSelect")
    assert _attr(html, "value", "a_share_demo_6")
    assert _attr(html, "value", "csi300_sample_20_2026h2")
    for symbol, name in (
        ("600519.SH", "Kweichow Moutai"),
        ("601318.SH", "Ping An Insurance"),
        ("600036.SH", "China Merchants Bank"),
        ("000001.SZ", "Ping An Bank"),
        ("000858.SZ", "Wuliangye Yibin"),
        ("300750.SZ", "CATL"),
        ("000333.SZ", "Midea Group"),
        ("002594.SZ", "BYD"),
        ("600276.SH", "Hengrui Medicine"),
        ("300760.SZ", "Mindray"),
        ("688981.SH", "SMIC"),
        ("002415.SZ", "Hikvision"),
        ("601766.SH", "CRRC"),
        ("600309.SH", "Wanhua Chemical"),
        ("601899.SH", "Zijin Mining"),
        ("601857.SH", "PetroChina"),
        ("600900.SH", "China Yangtze Power"),
        ("600050.SH", "China Unicom"),
        ("000725.SZ", "BOE Technology"),
        ("600030.SH", "CITIC Securities"),
        ("600887.SH", "Yili"),
        ("600048.SH", "Poly Developments"),
    ):
        assert symbol in js
        assert name in js
    assert "A-Share Demo 6" in html
    assert "CSI 300 Sample 20 (2026 H2)" in html
    assert "60m" in html


def test_ifind_user_facing_copy_contains_no_chinese(html, js):
    for text in (
        "iFinD A股（60分钟）",
        "iFinD A股 · 60m",
        "A股代表6只",
        "贵州茅台",
        "中国平安",
        "招商银行",
        "平安银行",
        "五粮液",
        "宁德时代",
    ):
        assert text not in html
        assert text not in js


def test_ifind_mode_locks_us_universe_and_restores_previous_us_model(js):
    assert re.search(r"const\s+isIFind\s*=\s*[^;]*ifind_ashare", js)
    assert re.search(r"ifindUniverse\.hidden\s*=\s*!isIFind", js)
    assert re.search(r"universeTabs\.hidden\s*=\s*isIFind", js)
    assert "Rule-based" in js
    assert "previousUniverse" in js
    assert "previousModel" in js
    assert re.search(r"selectPreset\([^)]*previousUniverse", js)


def test_ifind_mode_applies_one_month_dates_without_changing_capital(html, js):
    assert re.search(
        r"IFIND_ASHARE_START_DATE\s*=\s*['\"]2026-04-01['\"]",
        js,
    )
    assert re.search(
        r"IFIND_ASHARE_END_DATE\s*=\s*['\"]2026-05-01['\"]",
        js,
    )
    assert "previousStartDate" in js
    assert "previousEndDate" in js
    assert re.search(r"startDateInput\.value\s*=\s*IFIND_ASHARE_START_DATE", js)
    assert re.search(r"endDateInput\.value\s*=\s*IFIND_ASHARE_END_DATE", js)
    assert re.search(r"startDateInput\.value\s*=\s*previousStartDate", js)
    assert re.search(r"endDateInput\.value\s*=\s*previousEndDate", js)

    # Capital is no longer a per-run input the mode switch could touch (Task 4,
    # 2026-07-29) — the modal reads it read-only from the agent's saved value.
    # The invariant this test guards ("switching to iFind mode doesn't perturb
    # capital") now holds structurally: there is no capital-related identifier
    # anywhere in the mode-switch path.
    assert 'id="runBacktestCapitalValue"' in html
    assert "IFIND_ASHARE_INITIAL_CAPITAL" not in js


def test_ifind_profiles_declare_llm_capability_and_sync_model_control(js):
    demo = re.search(
        r"a_share_demo_6\s*:\s*\{(?P<body>.*?)\n\s*\},\n\s*csi300_sample_20",
        js,
        re.S,
    )
    sample = re.search(
        r"csi300_sample_20_2026h2\s*:\s*\{(?P<body>.*?)\n\s*\},\n\s*\};",
        js,
        re.S,
    )
    assert demo and "allowedDecisionSources: ['rule_based', 'llm']" in demo.group("body")
    assert sample and "allowedDecisionSources: ['rule_based', 'llm']" in sample.group("body")
    assert re.search(r"function\s+syncIFindModelControl\s*\(", js)
    assert re.search(r"modelSelect\.disabled\s*=\s*!allowsLLM", js)
    assert "resetIFindDecisionSource" in js
    assert re.search(
        r"renderIFindAshareUniverse\s*\(\s*\{[^}]*resetDecisionSource",
        js,
        re.S,
    )
    assert "Uses this agent's AI model by default" in js
    assert re.search(r"function\s+normalizeBacktestModelId\s*\(", js)
    assert re.search(r"function\s+findBacktestModelOption\s*\(", js)
    assert re.search(
        r"findBacktestModelOption\(\s*modelSelect\s*,\s*preferredModel\s*\)",
        js,
    )
    assert not re.search(
        r"if\s*\(\s*resetDecisionSource\s*\|\|\s*!allowsLLM\s*\)\s*\{"
        r"\s*modelSelect\.value\s*=\s*RULE_BASED_DECISION_SOURCE",
        js,
        re.S,
    )


def test_agent_model_sync_accepts_provider_paths_and_version_separators(js):
    assert re.search(r"raw\.includes\(['\"]/['\"]\)", js)
    assert re.search(r"raw\.split\(['\"]/['\"]\)\.pop\(\)", js)
    assert ".replace(/_(?:" not in js
    assert ".replace(/_/g, '-')" in js
    assert ".replace(/-(\\d+)-(\\d+)(?=-|$)/g, '-$1.$2')" in js
    assert re.search(
        r"findBacktestModelOption\(\s*modelSelect\s*,\s*agent\.model_name\s*\)",
        js,
    )
    assert re.search(r"function\s+resolveBacktestModelRequest\s*\(", js)
    assert re.search(
        r"agentOption\?\.value\s*===\s*selectedModel[^}]*return\s+agent\.model_name",
        js,
        re.S,
    )
    assert re.search(
        r"resolveBacktestModelRequest\(\s*modelSelect\s*,\s*activeAgent\s*\)",
        js,
    )


def test_ifind_request_uses_selected_profile_and_execution_lane(js):
    assert re.search(r"payload\.universe\s*=\s*selectedIFindUniverse", js)
    assert re.search(r"payload\.timeframe\s*=\s*['\"]60m['\"]", js)
    assert re.search(r"payload\.decision_source\s*=\s*decisionSource", js)
    assert re.search(r"params\.set\(\s*['\"]decision_source['\"]\s*,\s*decisionSource", js)
    assert re.search(
        r"if\s*\(\s*decisionSource\s*===\s*LLM_DECISION_SOURCE"
        r"\s*&&\s*!isHostedRuntime\s*\)",
        js,
    )
    assert re.search(r"params\.set\(\s*['\"]billing_mode['\"]\s*,\s*selectedBillingMode", js)
    assert re.search(r"params\.set\(\s*['\"]provider_id['\"]\s*,\s*selectedProviderId", js)
    assert re.search(r"payload\.billing_mode\s*=\s*selectedBillingMode", js)
    assert re.search(r"payload\.provider_id\s*=\s*selectedProviderId", js)
    assert re.search(r"payload\.model\s*=\s*model", js)
    assert re.search(r"if\s*\(\s*decisionSource\s*===\s*LLM_DECISION_SOURCE\s*&&\s*pipeline\?\.length", js)
    assert re.search(r"const\s+pipeline\s*=\s*isRuleBasedDecision\s*\?\s*null", js)


def test_ifind_universe_change_preserves_decision_source(js):
    assert re.search(
        r"getElementById\(['\"]ifindAshareUniverseSelect['\"]\)"
        r"\?\.addEventListener\(\s*['\"]change['\"]\s*,"
        r"\s*\(\)\s*=>\s*renderIFindAshareUniverse\(\s*\)",
        js,
        re.S,
    )


def test_backtest_capital_input_stays_removed_default_unchanged(html, js):
    """Historically pinned native <input min/max/step/value> attributes.

    The input was removed in Task 4 (2026-07-29): capital is now set in
    Configure and the modal only reports it via ``resolveBacktestCapital``,
    so there is nothing left to natively validate. This guards that the old
    input stays gone and the fallback default is still $1,000.
    """
    assert 'id="backtestInitialCapital"' not in html
    assert re.search(r"DEFAULT_AGENT_CASH_ALLOCATION\s*=\s*1000", js)


def test_ifind_model_dropdown_is_populated_from_supported_models(html, js):
    """This used to pin nine hardcoded <option>s here -- six models the platform
    cannot run, in a bare-slug format ('gpt-5.2') the rest of the app does not
    use, while #builtinAgentModel used namespaced slugs. Both pickers now build
    from SUPPORTED_MODELS in app.js; the vocabulary itself is pinned by
    test_frontend_model_vocabulary.py.

    What still matters *here* is that the picker is never left empty: on the
    iFinD A-share path it is the live rule-based-vs-LLM decision-source control,
    so the populator has to be wired into boot, not merely defined.
    """
    assert re.search(r'<select[^>]*id="modelSelect"[^>]*>\s*</select>', html)
    assert "function populateSupportedModelSelects" in js
    # Wired into the pure-DOM boot block, not left merely defined.
    assert js.index("setupTickerScrollControls();") < js.index(
        "populateSupportedModelSelects();"
    )


def test_run_config_shows_ifind_source_universe_count_timeframe_and_decision(html, js):
    for element_id in (
        "backtestConfigMarketData",
        "backtestConfigUniverse",
        "backtestConfigSymbols",
        "backtestConfigTimeframe",
        "backtestConfigDecisionSource",
    ):
        assert _attr(html, "id", element_id)
        assert element_id in js
    assert "Rule-based" in js
    assert re.search(
        r"decisionSource\s*===\s*LLM_DECISION_SOURCE\s*\?\s*formatAgentModelLabel\(model\)",
        js,
    )
    assert "symbolCount" in js
    assert "timeframe" in js


def test_ifind_results_show_historical_fx_and_native_trade_audit(html, js, css):
    for element_id in (
        "backtestConfigNativeCapital",
        "backtestConfigFxSource",
        "backtestConfigFxRate",
    ):
        assert _attr(html, "id", element_id)
        assert element_id in js
    assert "iFinD Historical Conversion Rate" in js
    assert "native_price" in js
    assert "native_value" in js
    assert "fx_rate" in js
    assert "¥" in js
    assert ".trading-log-native" in css


def test_running_and_historical_results_show_ifind_provenance(js, css):
    assert js.count("iFinD China A-Shares · 60m") >= 2
    assert re.search(r"renderBacktestDataSourceBadge\(\s*\{[^}]*data_source:\s*dataSource", js, re.S)
    assert ".data-source-badge.is-ifind" in css
    assert re.search(r"run\.data_source\s*===\s*['\"]ifind_ashare['\"]", js)


def test_historical_run_config_reads_delay_summary_from_linked_buyhold(js):
    assert re.search(
        r"function\s+renderBacktestRunConfig\([^)]*baselineRun\s*=\s*null",
        js,
        re.S,
    )
    assert re.search(
        r"baselineAllocation\s*=\s*baselineMetadata\.baseline_allocation",
        js,
    )
    assert re.search(
        r"resolveBaselinesForRun\(selectedRun,\s*sessionRuns\)",
        js,
    )
    assert re.search(r"baselineRun:\s*selectedBuyholdRun", js)


def test_ifind_chart_does_not_render_us_index_series(js):
    assert "BacktestComparison.buildModel" in js
    assert re.search(r"filterIfindChartSeries\s*\(", js)
    assert "DJIA index" in js
    assert "Nasdaq-100" in js
    assert re.search(r"filterIfindChartSeries\(\s*series", js)


def test_us_index_filter_keys_on_structural_run_id_not_only_the_label(js):
    """A renamed chart label must not silently disable the filter."""
    assert "MARKET_INDEX_RUN_ID_PREFIX = 'index:'" in js
    assert re.search(
        r"run_id\.startsWith\(MARKET_INDEX_RUN_ID_PREFIX\)",
        js,
    )


def test_ifind_errors_are_mapped_to_short_actionable_messages(js):
    assert re.search(r"function\s+formatBacktestError\s*\(", js)
    for marker in (
        "403",
        "503",
        "429",
        "50 bars",
        "minimum=50",
        "valid bars",
        "authentication",
        "response format",
    ):
        assert marker in js
    assert re.search(r"formatBacktestError\(\s*error", js)
    assert "The selected AI provider is not configured" in js
    assert js.index("llm provider client is unavailable") < js.index("status === 503")


def test_backtest_launch_failure_remains_visible_instead_of_loading_history(js):
    assert "let liveBacktestLaunchPending = false" in js
    assert "let liveBacktestLaunchError = false" in js
    assert re.search(r"function\s+showBacktestLaunchFailure\s*\(", js)
    assert "statusLabel: 'Failed'" in js
    assert "Backtest did not start." in js
    assert "isError ? 'Backtest did not start' : 'Backtest in progress'" in js
    assert re.search(
        r"!runningId\s*&&\s*\(liveBacktestLaunchPending\s*\|\|\s*liveBacktestLaunchError\)",
        js,
    )
    assert "setTimeout(() => showBacktestRunProgress(false), 5000)" not in js


def test_completed_zero_trade_run_has_actionable_empty_state(js):
    assert "No orders were submitted by the selected strategy." in js


def test_frontend_never_collects_or_stores_ifind_credentials(html, js):
    combined = f"{html}\n{js}".lower()
    assert "ifind_access_token" not in combined
    assert "access_token" not in combined
    assert "refresh_token" not in combined


def test_ifind_fixed_universe_has_stable_responsive_layout(css):
    assert re.search(
        r"\.ifind-symbol-grid\s*\{[^}]*grid-template-columns\s*:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)",
        css,
        re.S,
    )
    assert re.search(r"\.ifind-symbol-item\s*\{[^}]*min-width\s*:\s*0", css, re.S)
