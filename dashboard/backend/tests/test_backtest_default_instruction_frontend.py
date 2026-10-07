"""The default instruction is shown only for a run that will execute it.

An LLM backtest with an empty instruction runs DEFAULT_STARTER_INSTRUCTION
(``domain/agents/defaults.py::effective_pipeline``). Two places in /app state
that, and each used to claim it for runs that did not use it:

1. **The Run Backtest preview** was computed once when the modal opened, without
   the decision source. A rule-based run (vn.py, a rule-only iFinD universe, or
   Rule-based picked on iFinD) previewed a strategy the server drops.
2. **The preview and the results panel** decided "empty" from the browser's
   cached agent, while a launch with no body pipeline made the server read the
   stored row. When the two disagreed, both named the default for a run that
   executed something else. An LLM launch now always sends a pipeline
   (``backtestRequestPipeline``), so what the modal previewed is what runs;
   ``effectiveBacktestPipeline`` mirrors the route's resolution of it and is
   checked against ``effective_pipeline`` on the same inputs below. A run
   reopened from history has no launch config and uses the server's
   ``default_instruction`` flag.
3. **The preview hid post-trade steps** that run beside the default; it now
   formats the effective pipeline the way it formats any other.
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
        js_const("SIMPLE_INSTRUCTION_PRESET_KEY"),
        "const SIMPLE_INSTRUCTION_OUTPUT_FORMAT = "
        f"{json.dumps(js_string_const('SIMPLE_INSTRUCTION_OUTPUT_FORMAT'))};",
        js_const("SIMPLE_INSTRUCTION_LABEL"),
        js_const("POST_TRADE_PRESET_KEY"),
        fn_body("function loadAgentPipelineForBacktest("),
        fn_body("function formatPromptFromPipeline("),
        fn_body("function pipelineHasTradingInstruction("),
        fn_body("function effectiveBacktestPipeline("),
        fn_body("function backtestRequestPipeline("),
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
      runBacktestPromptDefaultNote: {{ hidden: false }},
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
      note: !els.runBacktestPromptDefaultNote.hidden,
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
    assert _preview(_EMPTY) == {"hidden": False, "text": DEFAULT, "note": True}
    no_pipeline = {k: v for k, v in _EMPTY.items() if k != "pipeline"}
    assert _preview(no_pipeline) == {"hidden": False, "text": DEFAULT, "note": True}


def test_blank_prompts_preview_the_default_and_the_post_trade_step():
    """Mirrors effective_pipeline: the default replaces the decision side and
    the post-trade step still runs -- a billed call the preview must not hide."""
    assert _preview(_BLANK) == {
        "hidden": False,
        "text": f"• Trading instruction: {DEFAULT}\n• post_trade_analysis: Review the day.",
        "note": True,
    }


def test_own_instruction_previews_itself():
    assert _preview(_OWN) == {"hidden": False, "text": "Only buy AAPL.", "note": False}


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
    assert _preview(agent, **kwargs) == {"hidden": True, "text": "", "note": False}


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


def _run_js(body):
    script = f"""
    const localStorage = {{ getItem: () => null }};
    {_LIFTED}
    {body}
    """
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


_MIRROR_CASES = {
    "none": None,
    "blank": [{"presetKey": "simple_instruction", "prompt": " "}],
    "blank+post-trade": _BLANK["pipeline"],
    "own": _OWN["pipeline"],
    "own+post-trade": _OWN["pipeline"] + [_BLANK["pipeline"][1]],
    "multi-step": [
        {"label": "Screen", "prompt": "Pick candidates."},
        {"label": "Trade", "prompt": "Size the orders."},
    ],
}


def _without_ids(pipeline):
    return [{k: v for k, v in step.items() if k != "id"} for step in pipeline]


@pytest.mark.parametrize("case", list(_MIRROR_CASES))
def test_effective_pipeline_mirror_matches_the_route(case):
    """The preview and the launch summary describe what the route runs, so the
    JS copy must resolve every shape the dashboard sends exactly as
    effective_pipeline does (step ids are minted per call on both sides)."""
    from dashboard.backend.domain.agents.defaults import effective_pipeline

    pipeline = _MIRROR_CASES[case]
    js = _run_js(
        "process.stdout.write(JSON.stringify("
        f"effectiveBacktestPipeline({json.dumps(pipeline)})));"
    )
    assert _without_ids(js) == _without_ids(effective_pipeline(pipeline))


def test_launch_always_sends_the_pipeline_it_previewed():
    """With no body pipeline the server reads the stored row, which a stale tab
    can disagree with -- so an LLM launch sends one, and a cached pipeline goes
    as-is so the route's write-back baseline still matches the stored row."""
    sent = _run_js(
        "process.stdout.write(JSON.stringify(["
        f"backtestRequestPipeline({json.dumps(_EMPTY)}),"
        f"backtestRequestPipeline({json.dumps(_BLANK)}),"
        "]));"
    )
    assert [step["prompt"] for step in sent[0]] == [DEFAULT]
    assert sent[1] == _BLANK["pipeline"]

    run_body = strip_comments(fn_body("async function runBacktest("))
    assert "backtestRequestPipeline(activeAgent)" in run_body
    summary = run_body[run_body.index("const promptSummary") :]
    summary = summary[: summary.index(";") + 1]
    assert "effectiveBacktestPipeline(pipeline)" in summary


def test_history_run_states_the_default_on_the_servers_word():
    code = strip_comments(APP_JS)
    panel = code[code.index("const prompt = cfg?.prompt") :]
    panel = panel[: panel.index(";") + 1]
    assert "default_instruction" in panel
    assert "DEFAULT_STARTER_INSTRUCTION" in panel
