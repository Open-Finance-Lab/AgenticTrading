"""The results panel says what sampling a run asked for.

One state per *shape* of recorded request. Three of those shapes are pinned
ones, as the policy table stands: *Pinned · temperature 0*, *Pinned ·
reasoning effort low (no temperature sent)*, and *Pinned · temperature 0 ·
thinking off*. A recorded effort in the off-set (`none/off/false/0/disabled`)
reads "thinking off"; any other value reads "reasoning effort <value>".
Beside them sit *Provider default*, for a run that
recorded pinning nothing, and *Not recorded*, for an LLM run written before
the field existed -- or any block whose recorded `policy` the row does not
recognise. The `policy` decides the prefix, never the presence of values: a
pinned block whose values were dropped must not read as provider default.
Leaderboard rows read *Entry config*, and a run answered on more than one
lane names the lanes.

This file enumerates the shapes and never asserts how many there are. The
count moves with the policy table and with `SamplingPolicy` itself -- a new
catalog row reusing an existing policy adds no shape, while a third field on
that dataclass (a `top_p`, say) adds several at once -- so a test that
counted would go red on a change that broke nothing. Not hypothetical: this
docstring's own first draft opened "Three states, never a fourth" above a
module already testing five. The spec makes the rule explicit: "the count
moves with that table, so nothing downstream should assert a number."

The row is hidden for rule-based runs, which had no sampler. The copy never
says "deterministic": providers are not, at temperature 0 least of all with
a mixture-of-experts model; what is pinned is the request.
"""
import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, fn_body

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _format(sampling_js: str) -> str:
    script = "\n".join(
        [
            fn_body("function formatBacktestSampling("),
            f"console.log(JSON.stringify(formatBacktestSampling({sampling_js})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_temperature_only():
    assert _format("{temperature: 0, reasoning_effort: null, policy: 'pinned_v1'}") == (
        "Pinned · temperature 0"
    )


def test_reasoning_only_says_the_temperature_was_not_sent():
    """The note explains the *absence*, and says nothing about the model.

    An earlier draft rendered "(this model ignores temperature)" on exactly
    this row -- which is the one policy (`PINNED_REASONING_LOW`) assigned to
    the one model that does the opposite: GPT-5.5, which returns 400 for a
    non-default temperature, which is precisely why the catalog withholds one
    (the spec's catalog table says the same). The models that genuinely
    ignore temperature while thinking are DeepSeek V4 Pro and Qwen3.7 Plus,
    and they carry `PINNED_NO_THINKING` (both values set), so the sentence
    rendered on every run it could not be about.

    The replacement makes no claim about the provider at all: an absent
    temperature could otherwise read as a half-recorded row, and "not sent" is
    the one thing the recorded request can actually support.
    """
    assert _format("{temperature: null, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · reasoning effort low (no temperature sent)"
    )


def test_both():
    """The `PINNED_NO_THINKING` row: DeepSeek V4 Pro and Qwen3.7 Plus."""
    assert _format("{temperature: 0, reasoning_effort: 'none', policy: 'pinned_v1'}") == (
        "Pinned · temperature 0 · thinking off"
    )


@pytest.mark.parametrize("off", ["none", "off", "false", "0", "disabled", "NONE"])
def test_every_off_value_reads_thinking_off(off):
    """The off-set is the adapter's, value for value (Task 5). A recorded
    "none" rendered as "reasoning effort none" would read as an effort level
    on the one row whose request switched reasoning off."""
    assert _format(
        f"{{temperature: 0, reasoning_effort: '{off}', policy: 'pinned_v1'}}"
    ) == "Pinned · temperature 0 · thinking off"


def test_a_graduated_effort_beside_a_temperature_still_reads_as_an_effort():
    """No catalog row has this shape today; the formatter reads shapes, so it
    still has to say something true about one."""
    assert _format("{temperature: 0, reasoning_effort: 'low', policy: 'pinned_v1'}") == (
        "Pinned · temperature 0 · reasoning effort low"
    )


def test_both_carries_no_caveat_about_what_the_model_does_with_it():
    """No note on the both-values row, about the model or the provider.

    DeepSeek V4 Pro and Qwen3.7 Plus are the only `PINNED_NO_THINKING` rows.
    Before the 2026-10-01 amendment they carried an effort of `low` and
    ignored the temperature while thinking. With thinking off the temperature
    applies, but this formatter reads the *shape* of a recorded request, not
    the catalog, so any note would be asserted of every future both-set row,
    true or not. The row's one job is to say what the run asked for. That
    "temperature 0" is a request and not an outcome is a claim about every row
    on this panel, and it is made where it can stay true: the copy rule that
    this cell never says "deterministic", the CLAUDE.md caveat, and
    `diff_backtest_runs.py`, which measures the spread instead of asserting
    it.
    """
    assert "(" not in _format(
        "{temperature: 0, reasoning_effort: 'none', policy: 'pinned_v1'}"
    )


def test_provider_default():
    assert _format("{temperature: null, reasoning_effort: null, policy: 'provider_default'}") == (
        "Provider default"
    )


def test_not_recorded():
    assert _format("null") == "Not recorded"
    assert _format("undefined") == "Not recorded"


@pytest.mark.parametrize(
    "block",
    [
        # No policy: values alone do not say who chose them.
        "{temperature: 0, reasoning_effort: null}",
        "{temperature: null, reasoning_effort: null}",
        "{}",
        "{temperature: 0, policy: 'something_new'}",
        # A provider-default record carrying values contradicts itself.
        "{temperature: 0, reasoning_effort: null, policy: 'provider_default'}",
    ],
)
def test_a_block_the_row_cannot_vouch_for_reads_not_recorded(block):
    assert _format(block) == "Not recorded"


def test_a_pinned_block_whose_values_were_dropped_still_says_pinned():
    """The route drops a string temperature; the policy still says pinned."""
    assert _format("{policy: 'pinned_v1'}") == "Pinned · values not recorded"
    assert _format("{temperature: '0', policy: 'pinned_v1'}") == (
        "Pinned · values not recorded"
    )


def test_a_leaderboard_row_reads_entry_config():
    assert _format(
        "{temperature: 0.2, reasoning_effort: null, policy: 'leaderboard_entry'}"
    ) == "Entry config · temperature 0.2"
    assert _format(
        "{temperature: null, reasoning_effort: null, policy: 'leaderboard_entry'}"
    ) == "Entry config · integration default"


def test_a_run_answered_on_more_than_one_lane_names_them():
    """One policy takes a different shape per lane; say the run was split."""
    assert _format(
        "{temperature: 0, reasoning_effort: 'none', policy: 'pinned_v1', "
        "wire: {commonstack: 'temperature=0.0;thinking=disabled', "
        "openrouter: 'temperature=0.0;reasoning.effort=none,enabled=false'}}"
    ) == "Pinned · temperature 0 · thinking off · lanes: commonstack, openrouter"
    assert _format(
        "{temperature: 0, reasoning_effort: 'none', policy: 'pinned_v1', "
        "wire: {commonstack: 'temperature=0.0;thinking=disabled'}}"
    ) == "Pinned · temperature 0 · thinking off"


def test_the_off_set_matches_the_python_one():
    """The one copy that cannot import the Python set is held equal to it."""
    import re

    from dashboard.backend.infrastructure.llm.reasoning_controls import (
        REASONING_OFF_VALUES,
    )

    body = fn_body("function formatBacktestSampling(")
    match = re.search(r"THINKING_OFF_VALUES = new Set\(\[([^\]]*)\]\)", body)
    assert match, "THINKING_OFF_VALUES literal not found"
    js_values = set(re.findall(r"'([^']*)'", match.group(1)))
    assert js_values == set(REASONING_OFF_VALUES)


def test_copy_never_claims_determinism():
    body = fn_body("function formatBacktestSampling(")
    assert "determin" not in body.lower()


@pytest.mark.parametrize("claim", ["ignor", "reject", "honour", "honor"])
def test_copy_never_says_what_the_provider_did_with_the_value(claim):
    """The formatter reads a request shape; it cannot see a provider.

    This is the guard for the specific mistake that shipped once: a note
    reading "this model ignores temperature", rendered on the one policy given
    to the one model that refuses temperature outright. Any word in this list
    is a claim about behaviour the recorded request does not observe, and the
    row has no way to be wrong about a fact it never states.

    `fn_body` slices from the `function` keyword, so the JSDoc above the
    function is not scanned -- explanations belong there, and this list does
    not censor them.
    """
    body = fn_body("function formatBacktestSampling(")
    assert claim not in body.lower()


def test_markup_and_renderer_carry_the_row():
    assert 'id="backtestConfigSamplingRow"' in APP_HTML
    assert 'id="backtestConfigSampling"' in APP_HTML
    body = fn_body("function renderBacktestRunConfig(")
    assert "formatBacktestSampling(" in body
    assert "backtestConfigSamplingRow" in body
    # Hidden for rule-based runs: an LLM run is one that recorded a ceiling.
    assert "metadata.llm_max_output_tokens" in body
    # Written unconditionally. A write guarded on the label leaves the
    # previous run's text in the hidden cell, ready to be painted under the
    # next run's heading by anything that unhides the row first. The literal
    # is the cell's own markup default, so the two agree.
    assert (
        "setBacktestConfigText('backtestConfigSampling', samplingLabel || '—')"
        in body
    )


# ---------------------------------------------------------------------------
# The renderer, fed what the list route really sends.
#
# `GET /api/backtest/runs` answers with RunMetadata, which has no `metadata`
# key. The row's visibility gate once read `metadata.llm_sampling`, so it was
# hidden on every run in production while every test above -- formatter only,
# plus a source-shape check on the renderer -- stayed green. These execute
# `renderBacktestRunConfig` against a run object of the API's shape.
# ---------------------------------------------------------------------------


def _render_sampling_row(run_js: str) -> dict:
    """Run the real renderer over a fake DOM; return the Sampling row's state."""
    script = "\n".join(
        [
            "const cells = {};",
            "const el = (id) => cells[id] || (cells[id] = {",
            "  id, hidden: undefined, textContent: undefined,",
            "  classList: { toggle() {} },",
            "});",
            "const document = { getElementById: el };",
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
            f"renderBacktestRunConfig({run_js}, {{ running: false }});",
            "const row = el('backtestConfigSamplingRow');",
            "const cell = el('backtestConfigSampling');",
            "console.log(JSON.stringify({ hidden: row.hidden, text: cell.textContent }));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# A run in the shape `_run_metadata_response(...).model_dump()` produces: the
# fields are top-level and there is no `metadata` key.
_API_RUN = (
    "run_id: 'run_1', agent_name: 'Agent', mode: 'backtest', "
    "start_date: '2026-04-01', end_date: '2026-04-23', initial_equity: 100000, "
    "created_at: '2026-04-23T16:00:00', data_source: 'alpaca', "
)


def test_a_pinned_run_in_the_api_shape_shows_the_row():
    state = _render_sampling_row(
        "{ " + _API_RUN + "llm_calls: 40, llm_max_output_tokens: 2000, "
        "llm_sampling: {temperature: 0, reasoning_effort: 'none', "
        "policy: 'pinned_v1', model: 'deepseek/deepseek-v4-pro'} }"
    )
    assert state == {"hidden": False, "text": "Pinned · temperature 0 · thinking off"}


def test_a_provider_default_run_in_the_api_shape_shows_the_row():
    state = _render_sampling_row(
        "{ " + _API_RUN + "llm_calls: 40, llm_max_output_tokens: 2000, "
        "llm_sampling: {temperature: null, reasoning_effort: null, "
        "policy: 'provider_default', model: 'm'} }"
    )
    assert state == {"hidden": False, "text": "Provider default"}


def test_an_llm_run_written_before_the_field_says_not_recorded():
    """It made model calls, so the row applies; it just cannot say how."""
    state = _render_sampling_row("{ " + _API_RUN + "llm_calls: 40 }")
    assert state == {"hidden": False, "text": "Not recorded"}


def test_a_rule_based_run_in_the_api_shape_hides_the_row():
    state = _render_sampling_row(
        "{ " + _API_RUN + "decision_source: 'rule_based', llm_calls: 0 }"
    )
    assert state["hidden"] is True
    assert state["text"] == "—"
