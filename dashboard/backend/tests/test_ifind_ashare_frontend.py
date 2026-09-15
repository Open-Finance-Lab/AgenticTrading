"""Source contracts for the controlled iFinD A-share backtest UI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from dashboard.backend.tests._frontend_source import fn_body, js_const, strip_comments


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
        r"IFIND_ASHARE_END_DATE\s*=\s*['\"]2026-04-15['\"]",
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
        "authentication",
        "response format",
    ):
        assert marker in js
    assert re.search(r"formatBacktestError\(\s*error", js)
    assert "The selected AI provider is not configured" in js
    assert js.index("llm provider client is unavailable") < js.index("status === 503")


# ===========================================================================
# The bar-shortfall arm, executed against the REAL producer messages
# ===========================================================================
#
# This used to be four more markers in the list above -- "50 bars",
# "minimum=50" -- asserted to be PRESENT IN THE FILE. They stayed green through
# the change that made every one of them unreachable: the floor moved from a
# flat 50 to minimum_bars_for_window(), so the adapter began saying
# "minimum=20" and the engine "fewer than 20 bars", and the mapper's arms
# matched neither. The user was still told to widen to "about one month", a
# window MAX_BACKTEST_DAYS now refuses outright.
#
# Two things are therefore deliberate here:
#
# 1. The messages are PROVOKED from the real adapter and the real engine, never
#    typed as literals. A fixture written from the mapper's own expectations
#    tests the mapper against itself and cannot fail when the producer is
#    renamed -- the trap CLAUDE.md's "fail-closed is not fail-visible" section
#    names. Rename either producer's wording and these fail.
# 2. The assertions are on the MAPPED OUTPUT of the real function, run under
#    node, not on source text. /app has no build step, so executing it is the
#    only way to learn whether an arm actually fires.


_ASHARE_SYMBOL = "600519.SH"
_WINDOW_START = "2026-04-01"
_WINDOW_END = "2026-04-15"


def _real_adapter_shortfall_message() -> str:
    """The message the adapter itself raises when a reply is too shallow."""
    from dashboard.backend.infrastructure.market_data.ifind_adapter import (
        IFindBarValidationError,
        response_to_frames,
    )
    from dashboard.backend.infrastructure.market_data.ifind_ashare import (
        minimum_bars_for_window,
    )

    cn = ZoneInfo("Asia/Shanghai")
    start = date.fromisoformat(_WINDOW_START)
    end = date.fromisoformat(_WINDOW_END)
    payload = {
        "errorcode": 0,
        "errmsg": "",
        "tables": [
            {
                "thscode": _ASHARE_SYMBOL,
                "time": [f"{_WINDOW_START} 10:30:00"],
                "table": {
                    "open": ["100.00"],
                    "high": ["101.00"],
                    "low": ["99.00"],
                    "close": ["100.50"],
                    "volume": ["10000"],
                },
            }
        ],
    }
    with pytest.raises(IFindBarValidationError) as excinfo:
        response_to_frames(
            payload,
            expected_symbols=(_ASHARE_SYMBOL,),
            start=datetime.combine(start, time(0, 0), tzinfo=cn),
            end=datetime.combine(end, time(0, 0), tzinfo=cn),
            min_bars=minimum_bars_for_window(start, end),
        )
    return str(excinfo.value)


def _real_engine_shortfall_message() -> str:
    """The message the engine raises for the same condition, one layer up."""
    import pandas as pd

    from dashboard.backend.domain.backtesting.engine import (
        HourlyBacktester,
        MarketDataUnavailableError,
    )

    frame = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp(f"{_WINDOW_START} 10:30:00", tz="Asia/Shanghai")],
            name="timestamp",
        ),
    )
    # The validator reads four attributes and nothing else, so an unbound call
    # against a stand-in is enough to get the genuine string without booting a
    # backtester (which would need a provider, credentials and a tape).
    stand_in = SimpleNamespace(
        symbols=(_ASHARE_SYMBOL,),
        all_data={_ASHARE_SYMBOL: frame},
        start_date=_WINDOW_START,
        end_date=_WINDOW_END,
    )
    with pytest.raises(MarketDataUnavailableError) as excinfo:
        HourlyBacktester._validate_ifind_loaded_data(stand_in)
    return str(excinfo.value)


def _map_backtest_error(message: str) -> str:
    """Run the real formatBacktestError over ``message`` under node."""
    script = "\n".join(
        [
            js_const("IFIND_ASHARE_SOURCE"),
            "const window = {};",
            fn_body("function formatBacktestError("),
            "process.stdout.write(JSON.stringify(formatBacktestError("
            f"{json.dumps({'message': message})}, IFIND_ASHARE_SOURCE)));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    "producer",
    [_real_adapter_shortfall_message, _real_engine_shortfall_message],
    ids=["adapter", "engine"],
)
def test_real_bar_shortfall_messages_reach_the_shortfall_arm(producer):
    raw = producer()
    mapped = _map_backtest_error(raw)

    # The engine's message used to land here -- a failure that carried an
    # actionable hint arriving with none.
    assert "check the backend log" not in mapped.lower()
    assert "too few valid bars" in mapped.lower()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    "producer",
    [_real_adapter_shortfall_message, _real_engine_shortfall_message],
    ids=["adapter", "engine"],
)
def test_the_shortfall_message_does_not_advise_an_illegal_remedy(producer):
    """Widening is refused by MAX_BACKTEST_DAYS *and* raises the floor."""
    mapped = _map_backtest_error(producer()).lower()

    assert "wider" not in mapped
    assert "one month" not in mapped
    # No floor may be quoted at all: it is derived per window now, so any
    # number printed here is right for one window and wrong for every other.
    assert not re.search(r"\d+\s*(valid\s*)?bars", mapped)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_shortfall_arm_is_not_keyed_on_a_floor_value():
    """The arm must survive the floor moving, which is what broke it before."""
    for floor in (20, 44, 50, 137):
        adapter_shaped = f"symbol={_ASHARE_SYMBOL} has 3 valid bars; minimum={floor}"
        engine_shaped = (
            f"iFinD symbols have fewer than {floor} bars: "
            f"{{'{_ASHARE_SYMBOL}': 3}}"
        )
        for raw in (adapter_shaped, engine_shaped):
            assert "too few valid bars" in _map_backtest_error(raw).lower()


def test_backtest_launch_failure_remains_visible_instead_of_loading_history(js):
    assert "let liveBacktestLaunchPending = false" in js
    assert "let liveBacktestLaunchError = false" in js
    assert re.search(r"function\s+showBacktestLaunchFailure\s*\(", js)
    assert "statusLabel: 'Failed'" in js
    assert "Backtest did not start." in js
    # A failed launch retitles the panel, so the error state is never shown
    # under "Backtest in progress". Asserted as the branch rather than as one
    # ternary's spelling: the same helper now carries four titles (it also
    # outlives its run for a fallback completion and for a cancel, issue #273)
    # and an exact-source match failed on changes that kept this contract
    # intact.
    panel = strip_comments(fn_body("function showBacktestRunProgress", js))
    assert "if (isError) title.textContent = 'Backtest did not start';" in panel
    assert "title.textContent = 'Backtest in progress';" in panel
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


def test_the_ifind_prefill_window_is_runnable_under_the_server_cap(js):
    """The auto-applied A-share window must be one the server will accept.

    Pinning the two date literals -- which the test above does -- says nothing
    about whether they are legal. That is exactly how this broke: lowering
    MAX_BACKTEST_DAYS from 31 to 14 left this prefill at 30 days, so choosing
    the iFinD A-share source wrote an unrunnable window into the form and the
    next click answered 422. The literal test stayed green throughout.

    Asserted against the imported server constant rather than a copy of it, so
    the two cannot drift apart again.
    """
    from datetime import date

    from dashboard.backend.api.routers.backtests import MAX_BACKTEST_DAYS

    start = re.search(r"IFIND_ASHARE_START_DATE\s*=\s*['\"]([\d-]+)['\"]", js)
    end = re.search(r"IFIND_ASHARE_END_DATE\s*=\s*['\"]([\d-]+)['\"]", js)
    assert start and end, "iFinD prefill date constants not found in app.js"

    span = (date.fromisoformat(end.group(1)) - date.fromisoformat(start.group(1))).days
    assert 0 < span <= MAX_BACKTEST_DAYS, (
        f"iFinD prefill window is {span} days, but the server refuses anything "
        f"over {MAX_BACKTEST_DAYS}"
    )
