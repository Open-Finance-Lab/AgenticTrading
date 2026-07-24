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


def test_ifind_request_uses_selected_profile_and_explicit_decision_source(js):
    assert re.search(r"payload\.universe\s*=\s*selectedIFindUniverse", js)
    assert re.search(r"payload\.timeframe\s*=\s*['\"]60m['\"]", js)
    assert re.search(r"payload\.decision_source\s*=\s*decisionSource", js)
    assert re.search(r"params\.set\(\s*['\"]decision_source['\"]\s*,\s*decisionSource", js)
    assert re.search(r"if\s*\(\s*decisionSource\s*===\s*LLM_DECISION_SOURCE\s*&&\s*model", js)
    assert re.search(r"if\s*\(\s*decisionSource\s*===\s*LLM_DECISION_SOURCE\s*&&\s*pipeline\?\.length", js)
    assert re.search(r"const\s+pipeline\s*=\s*isRuleBasedDecision\s*\?\s*null", js)


def test_ifind_universe_change_resets_decision_source(js):
    assert re.search(
        r"getElementById\(['\"]ifindAshareUniverseSelect['\"]\)"
        r"\?\.addEventListener\(\s*['\"]change['\"]\s*,"
        r"\s*\(\)\s*=>\s*renderIFindAshareUniverse\(\s*\{"
        r"\s*resetDecisionSource\s*:\s*true\s*\}\s*\)",
        js,
        re.S,
    )


def test_backtest_default_capital_is_natively_valid(html):
    capital_input = re.search(
        r'<input\b(?=[^>]*id="backtestInitialCapital")[^>]*>',
        html,
        re.S,
    )

    assert capital_input
    tag = capital_input.group(0)
    assert _attr(tag, "min", "1")
    assert _attr(tag, "step", "1")
    assert _attr(tag, "max", "10000")
    assert _attr(tag, "value", "1000")


def test_ifind_model_dropdown_keeps_all_nine_models_available(html):
    model_options = re.findall(
        r'<option\s+value="([^"]+)">[^<]+</option>',
        re.search(r'<select[^>]+id="modelSelect"[^>]*>(.*?)</select>', html, re.S).group(1),
    )
    assert model_options == [
        "claude-haiku-4.5",
        "claude-sonnet-4.6",
        "claude-opus-4.7",
        "gpt-5.2",
        "gpt-5-mini",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
    ]


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


def test_running_and_historical_results_show_ifind_provenance(js, css):
    assert js.count("iFinD China A-Shares · 60m") >= 2
    assert re.search(r"renderBacktestDataSourceBadge\(\s*\{[^}]*data_source:\s*dataSource", js, re.S)
    assert ".data-source-badge.is-ifind" in css
    assert re.search(r"run\.data_source\s*===\s*['\"]ifind_ashare['\"]", js)


def test_ifind_chart_does_not_render_us_index_series(js):
    assert re.search(r"filterIfindChartSeries\s*\(", js)
    assert "DJIA index" in js
    assert "Nasdaq-100" in js
    assert re.search(r"filterIfindChartSeries\(\s*series", js)


def test_ifind_errors_are_mapped_to_short_actionable_messages(js):
    assert re.search(r"function\s+formatBacktestError\s*\(", js)
    for marker in ("403", "503", "429", "50 bars", "authentication", "response format"):
        assert marker in js
    assert re.search(r"formatBacktestError\(\s*error", js)


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
