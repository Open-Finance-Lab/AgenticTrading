"""A finished model-driven run is labelled as one sample (#602, option 1).

Three DeepSeek V4 Pro reruns with identical inputs and pinned sampling
diverged on the first bar and finished at -1.25%, -0.48% and -0.37% (#539).
One curve is therefore one draw, and the results panel says so beside the
chart. A rule-based curve is repeatable and gets no note.

These execute the real `renderBacktestRunConfig` against a run in the list
route's shape (top-level fields, no `metadata` key) -- the shape that once
hid the Sampling row in production while source-shape tests stayed green.
"""
import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, fn_body

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_NOTICE = "chartSingleSampleNotice"


def _notice_hidden(run_js: str, *, running: bool = False, preset: bool = False) -> bool:
    """Render a run over a fake DOM; return the notice's `hidden` state.

    `preset` starts the notice visible, as a previously selected model run
    would have left it, so a path that forgets to hide it shows up.
    """
    script = "\n".join(
        [
            "const cells = {};",
            "const el = (id) => cells[id] || (cells[id] = {",
            "  id, hidden: undefined, textContent: undefined,",
            "  classList: { toggle() {} },",
            "});",
            "const document = { getElementById: el };",
            f"el('{_NOTICE}').hidden = {'false' if preset else 'undefined'};",
            "const IFIND_ASHARE_SOURCE = 'ifind_ashare';",
            "const LLM_DECISION_SOURCE = 'llm';",
            "const RULE_BASED_DECISION_SOURCE = 'rule_based';",
            "const getBacktestLaunchConfig = () => null;",
            "const formatBacktestFrequencyContract = () => null;",
            "const formatBacktestMarketDataQuality = () => null;",
            "const formatBacktestMarketDataProvenance = () => null;",
            "const getIFindUniverseProfile = () => null;",
            "const describeUniverseFromAssets = () => null;",
            "const formatAgentModelLabel = (m) => m;",
            "const formatTransactionCostProfile = () => '';",
            "const formatTransactionCostTotals = () => '';",
            "const formatCorporateActionGaps = () => '';",
            "const showBacktestRunProgress = () => {};",
            fn_body("function setBacktestConfigText("),
            fn_body("function formatBacktestSampling("),
            fn_body("function renderBacktestRunConfig("),
            f"renderBacktestRunConfig({run_js}, {{ running: {str(running).lower()} }});",
            f"console.log(JSON.stringify(el('{_NOTICE}').hidden));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_API_RUN = (
    "run_id: 'run_1', agent_name: 'Agent', mode: 'backtest', "
    "start_date: '2026-09-07', end_date: '2026-09-11', initial_equity: 1000, "
    "created_at: '2026-10-01T23:00:00', data_source: 'alpaca', "
)


def test_markup_ships_hidden_beside_the_chart():
    assert f'id="{_NOTICE}"' in APP_HTML
    tag = APP_HTML[APP_HTML.index(f'id="{_NOTICE}"') - 3 :].split(">", 1)[0]
    assert "hidden" in tag
    # Above the chart, not inside "Show advanced details": the label exists
    # to be read before the curve is.
    assert APP_HTML.index(f'id="{_NOTICE}"') < APP_HTML.index('id="performanceChart"')


def test_a_pinned_model_run_shows_the_notice():
    """Pinned sampling is the case a reader most expects to reproduce."""
    assert _notice_hidden(
        "{ " + _API_RUN + "llm_calls: 28, llm_max_output_tokens: 2000, "
        "llm_sampling: {temperature: 0, reasoning_effort: 'disabled', "
        "policy: 'pinned_v1', model: 'deepseek/deepseek-v4-pro'} }"
    ) is False


def test_an_llm_run_written_before_sampling_was_recorded_shows_the_notice():
    assert _notice_hidden("{ " + _API_RUN + "llm_calls: 40 }") is False


def test_a_rule_based_run_hides_a_notice_left_by_the_previous_run():
    assert _notice_hidden(
        "{ " + _API_RUN + "decision_source: 'rule_based', llm_calls: 0 }",
        preset=True,
    ) is True


def test_a_running_backtest_hides_the_notice():
    """No curve is final yet; the note belongs to a finished run.

    The run carries model fields on purpose: a bare `{run_id}` is hidden by
    the usedModel gate alone and would pass with the `running` guard deleted.
    """
    assert _notice_hidden(
        "{ " + _API_RUN + "llm_calls: 12, llm_max_output_tokens: 2000 }",
        running=True,
        preset=True,
    ) is True


def test_clearing_the_selection_hides_the_notice():
    """The no-run early return must not leave the last run's note up."""
    assert _notice_hidden("null", preset=True) is True


def test_a_failed_chart_load_hides_the_notice():
    """No curve on screen means nothing for the note to describe."""
    body = fn_body("async function loadHistoricalBacktestSurfaces(")
    catch = body[body.index(".catch(") :]
    assert f"getElementById('{_NOTICE}')" in catch
