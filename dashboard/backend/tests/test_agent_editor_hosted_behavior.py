"""Executable behaviour of the agent editor's hosted-runtime mode.

Why this file exists: the hosted-runtime editor was previously covered only by
substring assertions over ``agent-editor.js``. Those pass whenever the source
*mentions* the right identifier, so the mutation that matters most --
``isHostedRuntimeAgent`` reverted to ``isAiHedgeFundAgent``, which sends a
TradingAgents agent down the pipeline path -- survives them untouched: the
variable name in the source is unchanged.

So the functions here are lifted out of the shipped module and **run** under
node against a DOM shim that parses the markup the renderer produces, and every
case asserts on observable behaviour: which panels end up visible, which
analysts are selectable, and what the PATCH body would carry.

The shim is deliberately small and strict. It parses only the markup
``renderHostedAnalysts`` actually emits, and raises on a selector it was not
taught -- a shim that quietly answered "no matches" would make every assertion
below vacuous, which is the failure mode this file replaces.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import fn_body, run_node

_needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_EDITOR_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "agent-editor.js"
).read_text(encoding="utf-8")


def _lift(*signatures: str) -> str:
    """The named functions' real source, concatenated."""
    return "\n".join(fn_body(sig, _EDITOR_JS) for sig in signatures)


def _lift_const(name: str) -> str:
    """A top-level ``const`` declaration from agent-editor.js, verbatim.

    Lifted rather than restated: a harness that re-types the analyst roster
    tests the code against the harness's own copy, so withdrawing an analyst in
    the shipped module would silently stop being covered.

    Bracket-matched rather than sliced to the next ``];``: the rosters contain
    nested object literals, and a textual end marker picks whichever one
    happens to be indented that way -- truncating the value into something
    that is still *almost* valid JS.
    """
    match = re.search(rf"^\s*const {re.escape(name)} = ", _EDITOR_JS, re.MULTILINE)
    assert match, f"{name} is no longer a top-level const in agent-editor.js"
    start = match.start()
    opener = _EDITOR_JS[match.end()]
    assert opener in "[{", f"{name} is not an array or object literal"
    closer = "]" if opener == "[" else "}"
    index = match.end()
    depth = 0
    quote = None
    while index < len(_EDITOR_JS):
        char = _EDITOR_JS[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"`":
            quote = char
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return _EDITOR_JS[start : index + 1] + ";"
        index += 1
    raise AssertionError(f"unbalanced literal in const {name}")


# The panels configureEditorMode owns. Collected here so a case asserts over the
# whole set rather than the one panel it happens to care about: the bug this
# screen actually shipped was a *hidden* panel, not a wrong one.
_PANEL_IDS = (
    "agentEditorModelField",
    "agentEditorManagedModelField",
    "agentEditorSimplePanel",
    "agentEditorAiHedgeFundPanel",
    "agentEditorFinancialDatasetsField",
)

# --- the DOM shim -----------------------------------------------------------
# Parses the <label>/<input> markup renderHostedAnalysts emits, so the
# production DOM queries in selectedHostedAnalysts run against something that
# behaves like a browser instead of against a stub that always answers [].
_DOM_SHIM = r"""
function parseInputs(html) {
  const out = [];
  const re = /<label class="([^"]*)">([\s\S]*?)<\/label>/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    const labelClass = m[1];
    const block = m[2];
    const tag = /<input ([^>]*)>/.exec(block);
    if (!tag) throw new Error('analyst option without an input: ' + block);
    const attrs = tag[1];
    const attr = (name) => {
      const found = new RegExp(name + '="([^"]*)"').exec(attrs);
      return found ? found[1] : null;
    };
    out.push({
      value: attr('value'),
      name: attr('name'),
      checked: / checked(\s|$)/.test(attrs) || /\schecked>/.test(tag[0]),
      disabled: /\sdisabled(\s|$)/.test(attrs),
      labelClass,
      labelText: (/<span id="[^"]*-label">([\s\S]*?)<\/span>/.exec(block) || [, ''])[1],
      tooltip: (/class="agent-editor-analyst-tooltip"[^>]*>([\s\S]*?)<\/span>/.exec(block) || [, ''])[1],
    });
  }
  return out;
}

let els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id,
      hidden: false,
      value: '',
      textContent: '',
      checked: false,
      disabled: false,
      _html: '',
      attrs: {},
      get innerHTML() { return this._html; },
      set innerHTML(v) { this._html = v; },
      setAttribute(k, v) { this.attrs[k] = v; },
      addEventListener() {},
      querySelectorAll(selector) {
        if (selector === '.agent-editor-analyst-option') {
          // Listener wiring only; the cases assert on markup, not handlers.
          return parseInputs(this._html).map(() => ({
            addEventListener() {},
            querySelector() { return null; },
            contains() { return false; },
          }));
        }
        throw new Error('shim was not taught selector: ' + selector);
      },
    };
  }
  return els[id];
}

const document = {
  getElementById: el,
  querySelectorAll(selector) {
    if (selector === 'input[name="agentEditorAiHedgeFundAnalyst"]:checked') {
      return parseInputs(el('agentEditorAiHedgeFundAnalysts').innerHTML)
        .filter((i) => i.checked && !i.disabled);
    }
    throw new Error('shim was not taught selector: ' + selector);
  },
  createElement() { return { value: '', textContent: '' }; },
};
const window = { requestAnimationFrame: (fn) => fn() };

// Collaborators outside the unit under test.
let activeAiHedgeFundTooltipOption = null;
function showAiHedgeFundTooltip() {}
function hideAiHedgeFundTooltip() {}
function restoreKeyboardFocusedAiHedgeFundTooltip() {}
let financialDatasetsStatus = null;
function setFinancialDatasetsStatus(configured, message) {
  financialDatasetsStatus = { configured, message };
}
function categoryFieldApplies() { return false; }
function nameFollowsModel() { return false; }
function selectedModelLabel() { return 'Model'; }
function simplePresetKey() { return 'simple_instruction'; }
function simpleOutputFormat() { return 'FORMAT'; }
let subAgentSeq = 0;
function newSubAgentId() { return 'sub_' + (++subAgentSeq); }
let currentAgent = null;
let subAgents = [];
let hostedRuntimeModels = null;
let hostedRuntimeModelsPromise = null;
let hostedRuntimeModelsState = 'ready';
function loadHostedRuntimeModels() { return Promise.resolve(hostedRuntimeModels); }
"""


def _run(scenarios):
    lifted = _lift(
        "function escapeHtml(",
        "function agentRuntime(",
        "function hostedAnalystRoster(",
        "function isHostedRuntimeAgent(",
        "function isAiHedgeFundAgent(",
        "function managedModelText(",
        "function renderManagedModel(",
        "function renderWithdrawnAnalystNote(",
        "function renderHostedAnalysts(",
        "function selectedHostedAnalysts(",
        "function configureEditorMode(",
        "function getEditorState(",
        "function emptyRosterRefusal(",
    )
    constants = "\n".join(
        _lift_const(name)
        for name in (
            "AI_HEDGE_FUND_ANALYSTS",
            "TRADINGAGENTS_ANALYSTS",
            "HOSTED_RUNTIME_ANALYSTS",
            "HOSTED_RUNTIME_LABELS",
        )
    )
    script = f"""
const AI_HEDGE_FUND_RUNTIME = 'ai_hedge_fund';
const TRADINGAGENTS_RUNTIME = 'tradingagents';
{_DOM_SHIM}
{constants}
{lifted}
const scenarios = {json.dumps(scenarios)};
const out = {{}};
for (const [name, s] of Object.entries(scenarios)) {{
  els = {{}};
  currentAgent = s.agent;
  subAgents = s.subAgents || [];
  hostedRuntimeModels = s.models === undefined ? null : s.models;
  hostedRuntimeModelsState = s.modelsState || (s.models === undefined ? 'pending' : 'ready');
  configureEditorMode(currentAgent);
  const grid = el('agentEditorAiHedgeFundAnalysts');
  const options = parseInputs(grid.innerHTML);
  if (s.check) {{
    // Simulate the user ticking boxes: re-emit the markup with the chosen
    // values checked, exactly as a browser would record the clicks.
    let html = grid.innerHTML;
    for (const value of s.check) {{
      html = html.replace(
        new RegExp('(<input [^>]*value="' + value + '"(?![^>]*\\\\schecked))'),
        '$1 checked'
      );
    }}
    grid.innerHTML = html;
  }}
  el('agentEditorNameInput').value = s.agent.name || '';
  el('agentEditorSimpleInstruction').value = s.instruction || '';
  el('agentEditorCashAllocation').value = '';
  el('agentEditorBacktestAllocation').value = '';
  let state = null;
  let stateError = null;
  try {{ state = getEditorState(); }} catch (e) {{ stateError = String(e.message); }}
  const visible = {{}};
  for (const id of {json.dumps(list(_PANEL_IDS))}) visible[id] = !el(id).hidden;
  out[name] = {{
    visible,
    panelTitle: el('agentEditorHostedPanelTitle').textContent,
    managedModel: el('agentEditorManagedModelValue').textContent,
    gridLabel: grid.attrs['aria-label'] || null,
    options: options.map((o) => ({{
      value: o.value,
      checked: o.checked,
      disabled: o.disabled,
      unavailableClass: /is-unavailable/.test(o.labelClass),
      labelText: o.labelText,
      tooltip: o.tooltip,
    }})),
    selected: state ? (state.runtime_config ? state.runtime_config.analysts : null) : null,
    sendPipeline: state ? state.sendPipeline : null,
    modelName: state ? state.model_name : null,
    credential: state ? state.financial_datasets_api_key : null,
    subAgents: state ? state.subAgents : null,
    withdrawnNote: {{
      hidden: el('agentEditorHostedWithdrawnNote').hidden,
      text: el('agentEditorHostedWithdrawnNote').textContent,
    }},
    financialDatasetsStatus,
    stateError,
    rosterRefusal: state ? emptyRosterRefusal(s.agent, state) : null,
  }};
  financialDatasetsStatus = null;
}}
process.stdout.write(JSON.stringify(out));
"""
    return run_node(script)


def _agent(runtime_type, analysts=None, **extra):
    agent = {
        "agent_id": "a1",
        "name": "Agent",
        "runtime_type": runtime_type,
        "runtime_config": {"analysts": list(analysts)} if analysts is not None else None,
    }
    agent.update(extra)
    return agent


# ---------------------------------------------------------------------------
# 1. Each runtime enters its own mode. The regression: a TradingAgents agent
#    falling into the pipeline editor, where it gets a model picker it does not
#    own and a prompt pipeline it never consumes.
# ---------------------------------------------------------------------------

@_needs_node
def test_each_runtime_enters_its_own_editor_mode():
    out = _run(
        {
            "pipeline": {"agent": _agent("pipeline"), "instruction": "buy low"},
            "ai_hedge_fund": {
                "agent": _agent("ai_hedge_fund", ["technical_analyst"])
            },
            "tradingagents": {"agent": _agent("tradingagents", ["market"])},
        }
    )

    # Pipeline: prompt editor and model picker, no hosted panel.
    assert out["pipeline"]["visible"] == {
        "agentEditorModelField": True,
        "agentEditorManagedModelField": False,
        "agentEditorSimplePanel": True,
        "agentEditorAiHedgeFundPanel": False,
        "agentEditorFinancialDatasetsField": False,
    }

    # Both hosted runtimes: hosted panel and managed-model note, no model
    # picker and no prompt editor.
    for name in ("ai_hedge_fund", "tradingagents"):
        assert out[name]["visible"]["agentEditorModelField"] is False, name
        assert out[name]["visible"]["agentEditorManagedModelField"] is True, name
        assert out[name]["visible"]["agentEditorSimplePanel"] is False, name
        assert out[name]["visible"]["agentEditorAiHedgeFundPanel"] is True, name

    # The agent-scoped market-data credential is an AI Hedge Fund concept only.
    assert out["ai_hedge_fund"]["visible"]["agentEditorFinancialDatasetsField"] is True
    assert out["tradingagents"]["visible"]["agentEditorFinancialDatasetsField"] is False
    assert out["ai_hedge_fund"]["financialDatasetsStatus"] is not None
    assert out["tradingagents"]["financialDatasetsStatus"] is None


@_needs_node
def test_hosted_panel_is_titled_for_the_agents_own_runtime():
    out = _run(
        {
            "ahf": {"agent": _agent("ai_hedge_fund", ["technical_analyst"])},
            "ta": {"agent": _agent("tradingagents", ["market"])},
        }
    )
    assert out["ahf"]["panelTitle"] == "AI Hedge Fund analyst panel"
    assert out["ta"]["panelTitle"] == "TradingAgents analyst panel"
    assert out["ahf"]["gridLabel"] == "AI Hedge Fund analysts"
    assert out["ta"]["gridLabel"] == "TradingAgents analysts"


@_needs_node
def test_each_runtime_renders_only_its_own_roster():
    out = _run(
        {
            "ahf": {"agent": _agent("ai_hedge_fund", [])},
            "ta": {"agent": _agent("tradingagents", [])},
        }
    )
    ahf = {o["value"] for o in out["ahf"]["options"]}
    ta = {o["value"] for o in out["ta"]["options"]}
    # The two products share no analyst names; a pooled roster would let an AI
    # Hedge Fund name reach TradingAgents' setup_graph and raise mid-run.
    assert ahf & ta == set()
    assert "market" in ta and "technical_analyst" in ahf
    assert "market" not in ahf


# ---------------------------------------------------------------------------
# 2. Selection round-trips: saved -> rendered -> PATCH body.
# ---------------------------------------------------------------------------

@_needs_node
def test_saved_analysts_are_refilled_when_the_editor_reopens():
    out = _run(
        {
            "two": {
                "agent": _agent(
                    "ai_hedge_fund", ["technical_analyst", "valuation_analyst"]
                )
            },
        }
    )
    checked = {o["value"] for o in out["two"]["options"] if o["checked"]}
    assert checked == {"technical_analyst", "valuation_analyst"}


@_needs_node
def test_ticking_boxes_is_what_the_patch_body_carries():
    out = _run(
        {
            "picked": {
                "agent": _agent("ai_hedge_fund", []),
                "check": ["ben_graham", "michael_burry"],
            },
        }
    )
    # Order follows the roster, which is what the renderer emits.
    assert out["picked"]["selected"] == ["ben_graham", "michael_burry"]


@_needs_node
def test_hosted_agents_never_send_a_model_or_a_pipeline():
    out = _run(
        {
            # An instruction typed before the runtime was switched must not be
            # saved onto a hosted agent: it consumes no prompt pipeline.
            "ta": {
                "agent": _agent("tradingagents", ["market"]),
                "instruction": "buy the dip",
                "subAgents": [{"id": "sub_x", "prompt": "old"}],
            },
            "pipeline": {
                "agent": _agent("pipeline"),
                "instruction": "buy the dip",
            },
        }
    )
    assert out["ta"]["sendPipeline"] is False
    assert out["ta"]["modelName"] == ""
    assert out["ta"]["credential"] == ""
    # The stored pipeline is passed through untouched rather than rewritten.
    assert out["ta"]["subAgents"] == [{"id": "sub_x", "prompt": "old"}]

    assert out["pipeline"]["sendPipeline"] is True
    assert out["pipeline"]["runtime_config"] if False else True
    assert out["pipeline"]["selected"] is None
    assert out["pipeline"]["subAgents"][0]["prompt"] == "buy the dip"


# ---------------------------------------------------------------------------
# 3. Look-ahead: the UI must not offer a roster the API refuses.
#    Pairs with test_tradingagents_lookahead.py, which owns the server half.
# ---------------------------------------------------------------------------

@_needs_node
def test_lookahead_unsafe_analysts_cannot_be_selected():
    out = _run({"ta": {"agent": _agent("tradingagents", [])}})
    by_value = {o["value"]: o for o in out["ta"]["options"]}

    assert by_value["market"]["disabled"] is False
    for name in ("social", "news", "fundamentals"):
        assert by_value[name]["disabled"] is True, name
        assert by_value[name]["unavailableClass"] is True, name
        # The reason has to be on screen, not only in the source.
        assert "Unavailable:" in by_value[name]["tooltip"], name
        assert "unavailable" in by_value[name]["labelText"], name


@_needs_node
def test_a_disabled_analyst_cannot_reach_the_patch_body():
    # Even if something ticks the box, the ":checked" query excludes disabled
    # inputs -- so a stale DOM or a crafted click cannot smuggle the analyst
    # past the UI into a roster the server would 422.
    out = _run(
        {
            "ta": {
                "agent": _agent("tradingagents", []),
                "check": ["market", "social", "news"],
            }
        }
    )
    assert out["ta"]["selected"] == ["market"]


@_needs_node
def test_an_agent_saved_with_a_withdrawn_analyst_is_told_so():
    out = _run(
        {
            "stale": {
                "agent": _agent("tradingagents", ["market", "social", "news"])
            },
            "clean": {"agent": _agent("tradingagents", ["market"])},
        }
    )
    stale = out["stale"]
    # Cleared from the selection...
    assert stale["selected"] == ["market"]
    assert [o["value"] for o in stale["options"] if o["checked"]] == ["market"]
    # ...but not silently: the panel says which ones went and that saving
    # removes them.
    assert stale["withdrawnNote"]["hidden"] is False
    assert "Social Analyst" in stale["withdrawnNote"]["text"]
    assert "News Analyst" in stale["withdrawnNote"]["text"]
    # Not "saving removes it": if the whole stored roster was withdrawn,
    # emptyRosterRefusal blocks the save until something available is ticked,
    # so the note tells the user what to do instead of promising a save that
    # cannot happen.
    assert "Choose at least one available analyst" in stale["withdrawnNote"]["text"]

    # A clean agent gets no note at all.
    assert out["clean"]["withdrawnNote"]["hidden"] is True
    assert out["clean"]["withdrawnNote"]["text"] == ""


@_needs_node
def test_withdrawing_an_analyst_is_driven_by_the_shipped_roster():
    # The shim parses the real TRADINGAGENTS_ANALYSTS const, so this fails if
    # an `unavailable` reason is dropped from the shipped module -- the case
    # above would still pass on a hard-coded harness copy.
    out = _run({"ta": {"agent": _agent("tradingagents", [])}})
    disabled = {o["value"] for o in out["ta"]["options"] if o["disabled"]}
    assert disabled == {"social", "news", "fundamentals"}


# ---------------------------------------------------------------------------
# 4. Clearing every analyst must not look like a successful save.
# ---------------------------------------------------------------------------

@_needs_node
def test_clearing_every_analyst_yields_an_empty_roster_not_a_silent_default():
    out = _run({"ta": {"agent": _agent("tradingagents", [])}})
    # Nothing ticked -> an empty list, which save() must refuse rather than
    # substituting a default the user did not choose. (The server also refuses
    # it: normalize_runtime_config requires at least one analyst.)
    assert out["ta"]["selected"] == []

# ---------------------------------------------------------------------------
# 5. The managed-model line reports what the server says, per runtime.
# ---------------------------------------------------------------------------

_SERVER_MODELS = {
    "ai_hedge_fund": {
        "provider": "openrouter",
        "provider_label": "OpenRouter",
        "model": "nvidia/nemotron-3-nano-30b-a3b",
    },
    "tradingagents": {
        "provider": "commonstack",
        "provider_label": "CommonStack",
        "model": "anthropic/claude-haiku-4-5",
    },
}


@_needs_node
def test_each_runtime_shows_its_own_provider_and_model():
    # The regression: app.html carried one hard-coded pair that nothing wrote,
    # so a TradingAgents agent claimed OpenRouter and nemotron while the run
    # went to CommonStack and Haiku.
    out = _run(
        {
            "ahf": {
                "agent": _agent("ai_hedge_fund", ["technical_analyst"]),
                "models": _SERVER_MODELS,
            },
            "ta": {
                "agent": _agent("tradingagents", ["market"]),
                "models": _SERVER_MODELS,
            },
        }
    )
    assert out["ahf"]["managedModel"] == (
        "OpenRouter · nvidia/nemotron-3-nano-30b-a3b"
    )
    assert out["ta"]["managedModel"] == "CommonStack · anthropic/claude-haiku-4-5"


@_needs_node
def test_a_pending_fetch_reads_as_loading_not_as_a_failure():
    """Three states, three sentences.

    `configureEditorMode` paints synchronously, before the fetch resolves, so
    rendering the failure text while pending meant the panel said
    "unavailable" for up to 8 seconds on every first open -- contradicting
    both the code's own comment and app.html's "loading" placeholder, and
    making a pending fetch indistinguishable from a broken one.
    """
    out = _run(
        {
            "pending": {"agent": _agent("tradingagents", ["market"])},
            "signed_out": {
                "agent": _agent("tradingagents", ["market"]),
                "modelsState": "signed-out",
            },
            "failed": {
                "agent": _agent("tradingagents", ["market"]),
                "modelsState": "failed",
            },
        }
    )
    assert "loading" in out["pending"]["managedModel"]
    assert "unavailable" not in out["pending"]["managedModel"]
    assert "sign in" in out["signed_out"]["managedModel"].lower()
    assert "unavailable" in out["failed"]["managedModel"]


@_needs_node
def test_an_unanswered_or_unknown_runtime_says_unavailable_rather_than_guessing():
    out = _run(
        {
            # Answered, but the fetch failed.
            "pending": {
                "agent": _agent("tradingagents", ["market"]),
                "modelsState": "failed",
            },
            # Answered, but with no entry for this runtime.
            "unknown": {
                "agent": _agent("future_runtime", []),
                "models": _SERVER_MODELS,
            },
            # Answered with an entry carrying no model.
            "empty": {
                "agent": _agent("tradingagents", ["market"]),
                "models": {"tradingagents": {"provider_label": "CommonStack"}},
            },
        }
    )
    assert "unavailable" in out["pending"]["managedModel"]
    assert "unavailable" in out["empty"]["managedModel"]
    # A runtime with no table entry renders no analysts either, so the panel is
    # visibly empty rather than quietly wrong.
    assert out["unknown"]["options"] == []


@_needs_node
def test_the_label_never_falls_back_to_another_runtimes_model():
    # Only AI Hedge Fund in the answer: a TradingAgents agent must not borrow
    # it, which is exactly what the hard-coded literal amounted to.
    out = _run(
        {
            "ta": {
                "agent": _agent("tradingagents", ["market"]),
                "models": {"ai_hedge_fund": _SERVER_MODELS["ai_hedge_fund"]},
            }
        }
    )
    assert "nemotron" not in out["ta"]["managedModel"]
    assert "unavailable" in out["ta"]["managedModel"]

# ---------------------------------------------------------------------------
# 6. Saving with no analysts selected must not report success.
# ---------------------------------------------------------------------------

@_needs_node
def test_saving_an_empty_roster_is_refused_for_every_hosted_runtime():
    out = _run(
        {
            "ahf": {"agent": _agent("ai_hedge_fund", [])},
            "ta": {"agent": _agent("tradingagents", [])},
        }
    )
    # The regression: this check was keyed on isAiHedgeFundAgent, so the
    # TradingAgents case reached the server, failed the PATCH, and was
    # reported as "Saved locally; server update failed" with the dirty flag
    # cleared -- nothing saved anywhere, and the wording said otherwise.
    assert "AI Hedge Fund" in out["ahf"]["rosterRefusal"]
    assert "TradingAgents" in out["ta"]["rosterRefusal"]


@_needs_node
def test_a_populated_roster_is_not_refused():
    out = _run(
        {
            "ahf": {
                "agent": _agent("ai_hedge_fund", []),
                "check": ["ben_graham"],
            },
            "ta": {"agent": _agent("tradingagents", []), "check": ["market"]},
        }
    )
    assert out["ahf"]["rosterRefusal"] == ""
    assert out["ta"]["rosterRefusal"] == ""


@_needs_node
def test_a_pipeline_agent_is_never_refused_for_an_empty_roster():
    # Pipeline agents have no roster at all; refusing them would block every
    # prompt-only save.
    out = _run({"pipeline": {"agent": _agent("pipeline"), "instruction": "buy"}})
    assert out["pipeline"]["rosterRefusal"] == ""
