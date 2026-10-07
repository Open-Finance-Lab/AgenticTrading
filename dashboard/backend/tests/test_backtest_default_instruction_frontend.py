"""The default instruction is shown only for a run that will execute it.

An LLM backtest with an empty instruction runs DEFAULT_STARTER_INSTRUCTION
(``domain/agents/defaults.py::effective_pipeline``). Two places in /app state
that, and each used to claim it for runs that did not use it:

1. **The Run Backtest preview** was computed once when the modal opened, without
   the decision source. A rule-based run (vn.py, a rule-only iFinD universe, or
   Rule-based picked on iFinD) previewed a strategy the server drops.
2. **The results panel's Instruction row** decided "empty" from the browser's
   cached agent, while the server reads the stored row. When the two disagreed
   the panel named the default for a run that executed something else. It now
   states the default only on the server's word: the launch response's
   ``default_instruction``, or the run's own ``default_instruction`` flag.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_JS,
    fn_body,
    js_const,
    js_string_const,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

DEFAULT = js_string_const("DEFAULT_STARTER_INSTRUCTION")

_LIFTED = "\n".join(
    [
        js_const("IFIND_ASHARE_SOURCE"),
        js_const("RULE_BASED_DECISION_SOURCE"),
        js_const("LLM_DECISION_SOURCE"),
        f"const DEFAULT_STARTER_INSTRUCTION = {json.dumps(DEFAULT)};",
        fn_body("function loadAgentPipelineForBacktest("),
        fn_body("function formatPromptFromPipeline("),
        fn_body("function pipelineHasTradingInstruction("),
        fn_body("function runBacktestModalDecisionSource("),
        fn_body("function syncRunBacktestInstructionPreview("),
    ]
)


def _preview(agent, *, source="alpaca", model="openai/gpt-5.5", ifind_llm=True):
    script = f"""
    const els = {{
      marketDataSourceSelect: {{ value: {json.dumps(source)} }},
      modelSelect: {{ value: {json.dumps(model)} }},
      runBacktestPromptGroup: {{ hidden: false }},
      runBacktestPromptPreview: {{ textContent: 'stale' }},
    }};
    const document = {{ getElementById: (id) => els[id] || null }};
    const localStorage = {{ getItem: () => null }};
    const getSelectedIFindUniverse = () => 'u';
    const getIFindUniverseProfile = () => ({{
      allowedDecisionSources: {json.dumps(['rule_based', 'llm'] if ifind_llm else ['rule_based'])},
    }});
    let runBacktestModalAgent = {json.dumps(agent)};
    {_LIFTED}
    syncRunBacktestInstructionPreview();
    process.stdout.write(JSON.stringify({{
      hidden: els.runBacktestPromptGroup.hidden,
      text: els.runBacktestPromptPreview.textContent,
    }}));
    """
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_EMPTY = {"agent_id": "a1", "runtime_type": "pipeline", "pipeline": []}
_OWN = {
    "agent_id": "a2",
    "runtime_type": "pipeline",
    "pipeline": [{"presetKey": "simple_instruction", "prompt": "Only buy AAPL."}],
}
_BLANK = {
    "agent_id": "a3",
    "runtime_type": "pipeline",
    "pipeline": [
        {"presetKey": "simple_instruction", "prompt": "  "},
        {"presetKey": "post_trade_analysis", "prompt": "Review the day."},
    ],
}


def test_empty_instruction_previews_the_default_on_an_llm_run():
    assert _preview(_EMPTY) == {"hidden": False, "text": DEFAULT}


def test_blank_prompts_preview_the_default_like_no_pipeline():
    """Mirrors has_trading_instruction: a post-trade prompt is not a trading
    instruction, and the server substitutes the default for the decision side."""
    assert _preview(_BLANK) == {"hidden": False, "text": DEFAULT}


def test_own_instruction_previews_itself():
    assert _preview(_OWN) == {"hidden": False, "text": "Only buy AAPL."}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "vnpy_simulation"},
        {"source": "ifind_ashare", "ifind_llm": False},
        {"source": "ifind_ashare", "model": "rule_based"},
    ],
    ids=["vnpy", "ifind-rule-only-universe", "ifind-rule-based-picked"],
)
@pytest.mark.parametrize("agent", [_EMPTY, _OWN], ids=["empty", "own"])
def test_rule_based_runs_preview_no_instruction(agent, kwargs):
    """The server drops the pipeline on a rule-based run, so naming any
    strategy -- the default or the agent's own -- describes a run it won't make."""
    assert _preview(agent, **kwargs) == {"hidden": True, "text": ""}


def test_hosted_runtime_previews_no_instruction():
    assert _preview({**_EMPTY, "runtime_type": "ai_hedge_fund"})["hidden"] is True


def test_preview_follows_every_control_that_moves_the_decision_source():
    """Computed on open alone, the preview outlived a switch to Rule-based."""
    assert "syncRunBacktestInstructionPreview();" in fn_body(
        "function syncBacktestModelFieldMode("
    )
    assert "syncRunBacktestInstructionPreview();" in fn_body(
        "function syncIFindModelControl("
    )
    assert "syncRunBacktestInstructionPreview();" in fn_body(
        "async function openRunBacktestModal("
    )


def test_launch_and_preview_share_one_decision_source():
    body = strip_comments(fn_body("async function runBacktest("))
    assert "const decisionSource = runBacktestModalDecisionSource();" in body


def test_results_panel_states_the_default_only_on_the_servers_word():
    run_body = strip_comments(fn_body("async function runBacktest("))
    # The launch config never assumes the default from the cached agent...
    summary = run_body[run_body.index("const promptSummary") :]
    summary = summary[: summary.index(";") + 1]
    assert "DEFAULT_STARTER_INSTRUCTION" not in summary
    # ...the response says so.
    assert "data.default_instruction === true" in run_body
    assert "launchConfigBase.prompt = DEFAULT_STARTER_INSTRUCTION" in run_body

    code = strip_comments(APP_JS)
    panel = code[code.index("const prompt = cfg?.prompt") :]
    panel = panel[: panel.index(";") + 1]
    assert "default_instruction" in panel
    assert "DEFAULT_STARTER_INSTRUCTION" in panel
