"""The Run Backtest modal must not open on a universe the runtime refuses.

The regression this pins: `openRunBacktestModal` selected the DJIA 30 preset
and the Built-in tab for *every* agent, and the Custom tab ships six chips in
the static HTML. TradingAgents accepts exactly one symbol (the router refuses
anything else with a 422, and `TradingAgentsRuntime._require_single_symbol`
refuses it again), so the headline flow -- Marketplace, Add, Configure, Run
Backtest -- failed on the user's first click, with a "Failed" card drawn on the
page for a request that could never have succeeded.

Executed under node rather than asserted as source text: the bug was a *value*
(`'djia'`) reaching a runtime that cannot use it, and no substring assertion
over app.js distinguishes the fixed version from the broken one.
"""

import json
import shutil

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    fn_body,
    js_const,
    run_node,
)

_needs_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


# The chips app.html ships inside #selectedChips. Lifted from the markup rather
# than restated: the trap is that the Custom tab is *not* empty by default, so a
# harness that assumed it was would prove nothing.
def _shipped_chips() -> list[str]:
    import re

    start = APP_HTML.index('id="selectedChips"')
    end = APP_HTML.index("</div>", APP_HTML.index("chips-helper", start))
    return re.findall(r'data-ticker="([A-Z0-9.]+)"', APP_HTML[start:end])


_DOM = r"""
let chips = [];
let tabs = {
  builtin: { dataset: { tab: 'builtin' }, disabled: false, classes: new Set(['active']), title: '' },
  custom: { dataset: { tab: 'custom' }, disabled: false, classes: new Set(), title: '' },
};
for (const t of Object.values(tabs)) {
  t.classList = {
    add: (c) => t.classes.add(c),
    remove: (c) => t.classes.delete(c),
    contains: (c) => t.classes.has(c),
  };
}
let els = {};
function el(id) {
  if (!els[id]) {
    els[id] = {
      id, hidden: false, value: '', textContent: '', disabled: false,
      style: {}, dataset: {},
      classes: new Set(id === 'builtinTab' ? ['active'] : []),
      attrs: {},
      setAttribute(k, v) { this.attrs[k] = v; },
      removeChild() {},
      appendChild() {},
      replaceChildren() {},
      createElement() { return {}; },
    };
    els[id].classList = {
      add: (c) => els[id].classes.add(c),
      remove: (c) => els[id].classes.delete(c),
      contains: (c) => els[id].classes.has(c),
    };
  }
  return els[id];
}

function chipEl(ticker) {
  return {
    dataset: { ticker },
    remove() { chips = chips.filter((c) => c !== ticker); },
  };
}

const document = {
  getElementById: el,
  createElement: () => ({ className: '', textContent: '', dataset: {}, appendChild() {} }),
  dispatchEvent() {},
  querySelector(selector) {
    const m = /\.universe-tab\[data-tab="(\w+)"\]/.exec(selector);
    if (m) return tabs[m[1]];
    if (/\[data-ticker="/.test(selector)) {
      const t = /\[data-ticker="([^"]+)"\]/.exec(selector)[1];
      return chips.includes(t) ? chipEl(t) : null;
    }
    throw new Error('shim was not taught querySelector: ' + selector);
  },
  querySelectorAll(selector) {
    if (selector === '.universe-tab') return Object.values(tabs);
    if (selector === '#selectedChips .chip') return chips.map(chipEl);
    throw new Error('shim was not taught querySelectorAll: ' + selector);
  },
};
const window = {};
function CustomEvent() {}
// agentDisplayName consults the model catalogue to decide whether a stored
// name is really a model label; empty here, so a stored name always wins.
function catalogModelLabels() { return new Set(); }
const CSS = { escape: (v) => v };
function renderSelectedUniversePreview() {}
function notifyAssetUniverseChanged() {}
console.log = () => {};
"""


def _lift_object_const(name: str) -> str:
    """A multi-line object ``const`` from app.js, brace-matched.

    ``js_const`` stops at the first ``;``, which is fine for a scalar but
    truncates nothing here and would silently return only the opening line.
    """
    import re

    from dashboard.backend.tests._frontend_source import APP_JS, _match_brace

    match = re.search(rf"^const {re.escape(name)} = \{{", APP_JS, re.MULTILINE)
    assert match, f"{name} is no longer a top-level object const in app.js"
    open_brace = APP_JS.index("{", match.start())
    return APP_JS[match.start() : _match_brace(APP_JS, open_brace) + 1] + ";"


def _run(scenarios, extra_js=""):
    lifted = "\n".join(
        [
            js_const("SINGLE_SYMBOL_RUNTIMES"),
            js_const("SIGN_IN_REQUIRED_RUNTIMES"),
            _lift_object_const("HOSTED_RUNTIME_LABELS"),
            fn_body("function agentRuntimeType("),
            fn_body("function runtimeTakesSingleSymbol("),
            fn_body("function hostedRuntimeLabel("),
            fn_body("function customUniverseSymbols("),
            fn_body("function trimCustomUniverseToOne("),
            fn_body("function runtimeRequiresSignIn("),
            fn_body("function hostedRuntimeAuthRefusal("),
            fn_body("function singleSymbolUniverseRefusal("),
            fn_body("function applyUniverseDefaultsForRuntime("),
            fn_body("function handleUniverseTabSwitch("),
            fn_body("function selectPreset("),
            fn_body("function getSelectedAssets("),
            fn_body("function formatAgentModelLabel("),
            fn_body("function agentDisplayName("),
            fn_body("function backtestAgentOptionLabel("),
        ]
    )
    script = f"""
{_DOM}
let selectedUniverse = 'djia';
let signedIn = true;
function isSignedIn() {{ return signedIn; }}
{js_const("ASSET_UNIVERSES")}
const IFIND_ASHARE_SOURCE = 'ifind_ashare';
function getSelectedIFindUniverse() {{ return null; }}
function getIFindUniverseProfile() {{ return null; }}
{lifted}
{extra_js}
const scenarios = {json.dumps(scenarios)};
const out = {{}};
for (const [name, s] of Object.entries(scenarios)) {{
  els = {{}};
  chips = {json.dumps(_shipped_chips())};
  selectedUniverse = 'djia';
  tabs.builtin.classes = new Set(['active']);
  tabs.custom.classes = new Set();
  tabs.builtin.disabled = false;
  tabs.builtin.title = '';
  el('builtinTab').classes = new Set(['active']);
  el('customTab').classes = new Set();
  signedIn = s.signedIn === undefined ? true : s.signedIn;
  if (s.chips !== undefined) chips = s.chips.slice();
  applyUniverseDefaultsForRuntime(s.agent);
  out[name] = {{
    chips: chips.slice(),
    activeTab: tabs.builtin.classList.contains('active') ? 'builtin' : 'custom',
    builtinTabDisabled: tabs.builtin.disabled,
    builtinTabTitle: tabs.builtin.title,
    selectedUniverse,
    assets: getSelectedAssets(),
    hint: {{
      hidden: el('backtestSingleSymbolHint').hidden,
      text: el('backtestSingleSymbolHint').textContent,
    }},
    label: hostedRuntimeLabel(s.agent),
    optionLabel: backtestAgentOptionLabel(
      Object.assign({{ agent_type: 'builtin', model_name: s.storedModel || '' }}, s.agent)
    ),
    singleSymbol: runtimeTakesSingleSymbol(s.agent),
    refusal: singleSymbolUniverseRefusal(s.agent),
    authRefusal: hostedRuntimeAuthRefusal(s.agent),
    refusalAfterEdits: (() => {{
      if (s.editTo === undefined) return null;
      chips = s.editTo.slice();
      return singleSymbolUniverseRefusal(s.agent);
    }})(),
    refusalOnBuiltinTab: (() => {{
      if (s.builtinAfter === undefined) return null;
      // What the iFinD restore path does: switch the pane back without
      // consulting whether the Built-in button is disabled.
      handleUniverseTabSwitch(tabs.builtin);
      selectPreset(s.builtinAfter);
      return singleSymbolUniverseRefusal(s.agent);
    }})(),
  }};
}}
process.stdout.write(JSON.stringify(out));
"""
    return run_node(script)


def _agent(runtime_type):
    return {"agent_id": "a1", "name": "A", "runtime_type": runtime_type}


# ---------------------------------------------------------------------------
# 1. The shipped markup is the hazard: assert it, so the cases below are not
#    quietly testing an empty Custom tab.
# ---------------------------------------------------------------------------

def test_the_custom_tab_ships_more_than_one_chip():
    chips = _shipped_chips()
    assert len(chips) > 1, (
        "this suite exists because the Custom tab is pre-populated; if it ships "
        "empty now, the single-symbol trimming cases need rewriting rather than "
        "silently passing"
    )


# ---------------------------------------------------------------------------
# 2. A single-symbol runtime opens on a universe it can actually run.
# ---------------------------------------------------------------------------

@_needs_node
def test_single_symbol_runtime_opens_with_exactly_one_asset():
    out = _run({"ta": {"agent": _agent("tradingagents")}})
    result = out["ta"]
    assert result["singleSymbol"] is True
    assert result["activeTab"] == "custom"
    assert len(result["chips"]) == 1
    # And what the request would carry is that one symbol, not a preset.
    assert len(result["assets"]) == 1
    assert result["assets"] == result["chips"]


@_needs_node
def test_single_symbol_runtime_keeps_the_first_shipped_chip():
    # Trimming rather than clearing: the first click should work, and the chip
    # stays visible and removable so the choice is the user's to change.
    out = _run({"ta": {"agent": _agent("tradingagents")}})
    assert out["ta"]["chips"] == [_shipped_chips()[0]]


@_needs_node
def test_single_symbol_runtime_cannot_reach_the_builtin_presets():
    out = _run({"ta": {"agent": _agent("tradingagents")}})
    # Neither preset is single-symbol (DJIA 30, Magnificent 7), so the tab is
    # closed off rather than left as a trap.
    assert out["ta"]["builtinTabDisabled"] is True
    assert "one" in out["ta"]["builtinTabTitle"].lower()


@_needs_node
def test_single_symbol_runtime_explains_the_constraint_on_screen():
    out = _run({"ta": {"agent": _agent("tradingagents")}})
    hint = out["ta"]["hint"]
    assert hint["hidden"] is False
    assert "TradingAgents" in hint["text"]
    assert "one" in hint["text"].lower()


@_needs_node
def test_an_already_single_custom_selection_is_left_alone():
    out = _run({"ta": {"agent": _agent("tradingagents"), "chips": ["NVDA"]}})
    assert out["ta"]["chips"] == ["NVDA"]


@_needs_node
def test_an_empty_custom_selection_is_not_silently_filled():
    # No chips means the user removed them. Inventing one would run a ticker
    # they did not choose; the submit guard refuses instead.
    out = _run({"ta": {"agent": _agent("tradingagents"), "chips": []}})
    assert out["ta"]["chips"] == []


# ---------------------------------------------------------------------------
# 3. Nothing changes for pipeline or AI Hedge Fund agents.
# ---------------------------------------------------------------------------

@_needs_node
def test_multi_symbol_runtimes_keep_the_djia_default():
    out = _run(
        {
            "pipeline": {"agent": _agent("pipeline")},
            "ahf": {"agent": _agent("ai_hedge_fund")},
        }
    )
    for name in ("pipeline", "ahf"):
        assert out[name]["singleSymbol"] is False, name
        assert out[name]["activeTab"] == "builtin", name
        assert out[name]["selectedUniverse"] == "djia", name
        assert len(out[name]["assets"]) == 30, name
        assert out[name]["builtinTabDisabled"] is False, name
        # Their chips are untouched, so switching to Custom still offers the
        # shipped set.
        assert out[name]["chips"] == _shipped_chips(), name
        assert out[name]["hint"]["hidden"] is True, name


# ---------------------------------------------------------------------------
# 3b. The submit guard: refuse while the modal is open, never silently.
# ---------------------------------------------------------------------------

@_needs_node
def test_submit_is_allowed_once_exactly_one_ticker_is_selected():
    out = _run({"ta": {"agent": _agent("tradingagents")}})
    # applyUniverseDefaultsForRuntime has trimmed to one, so the guard is clear.
    assert out["ta"]["refusal"] == ""


@_needs_node
def test_submit_is_refused_with_an_empty_custom_universe():
    out = _run({"ta": {"agent": _agent("tradingagents"), "editTo": []}})
    message = out["ta"]["refusalAfterEdits"]
    assert message
    assert "TradingAgents" in message
    # Says what to do, not just that it is wrong.
    assert "Add one" in message


@_needs_node
def test_submit_is_refused_with_several_tickers_and_counts_them():
    out = _run(
        {
            "ta": {
                "agent": _agent("tradingagents"),
                "editTo": ["AAPL", "MSFT", "NVDA", "TSLA"],
            }
        }
    )
    message = out["ta"]["refusalAfterEdits"]
    assert "Remove 3" in message, message


@_needs_node
def test_the_submit_guard_never_fires_for_multi_symbol_runtimes():
    out = _run(
        {
            "pipeline": {"agent": _agent("pipeline"), "editTo": []},
            "ahf": {"agent": _agent("ai_hedge_fund"), "editTo": ["A", "B", "C"]},
        }
    )
    # Even with zero or many chips: these runtimes take a whole universe, and
    # the guard must not start refusing runs that worked before.
    assert out["pipeline"]["refusalAfterEdits"] == ""
    assert out["ahf"]["refusalAfterEdits"] == ""


# ---------------------------------------------------------------------------
# 4. H4: a hosted run must be labelled with its own product's name.
# ---------------------------------------------------------------------------

@_needs_node
def test_each_hosted_runtime_reports_its_own_name():
    out = _run(
        {
            "ahf": {"agent": _agent("ai_hedge_fund")},
            "ta": {"agent": _agent("tradingagents")},
            "pipeline": {"agent": _agent("pipeline")},
            "unknown": {"agent": _agent("future_runtime")},
        }
    )
    assert out["ahf"]["label"] == "AI Hedge Fund"
    assert out["ta"]["label"] == "TradingAgents"
    # A pipeline agent is not a hosted runtime and has no hosted label.
    assert out["pipeline"]["label"] == ""
    # A runtime added without a label entry reports its own id rather than
    # another product's name -- visible immediately instead of wrong quietly.
    assert out["unknown"]["label"] == "future_runtime"

# ---------------------------------------------------------------------------
# 3c. Platform-paid runtimes need an account, and say so before the request.
# ---------------------------------------------------------------------------

@_needs_node
def test_a_signed_out_visitor_is_refused_before_the_request():
    out = _run(
        {
            "ta": {"agent": _agent("tradingagents"), "signedIn": False},
        }
    )
    message = out["ta"]["authRefusal"]
    assert message
    assert "Sign in" in message
    # Names the product and why, so the refusal is not a bare wall.
    assert "TradingAgents" in message
    assert "platform" in message


@_needs_node
def test_a_signed_in_visitor_is_not_refused():
    out = _run({"ta": {"agent": _agent("tradingagents"), "signedIn": True}})
    assert out["ta"]["authRefusal"] == ""


@_needs_node
def test_the_sign_in_gate_does_not_spread_to_other_runtimes():
    # AI Hedge Fund has the same server-side property but is shipped
    # behaviour; silently requiring an account for it would be a product
    # change made by a frontend edit. Pipeline runs are unaffected too.
    out = _run(
        {
            "ahf": {"agent": _agent("ai_hedge_fund"), "signedIn": False},
            "pipeline": {"agent": _agent("pipeline"), "signedIn": False},
        }
    )
    assert out["ahf"]["authRefusal"] == ""
    assert out["pipeline"]["authRefusal"] == ""

@_needs_node
def test_the_agent_picker_names_the_runtime_not_a_stale_stored_model():
    """A cloned hosted agent carries its template's `model_name`.

    `dashboard/config/marketplace.json` gives the TradingAgents template
    `nvidia/nemotron-3-nano-30b-a3b`, which is not what it runs on, and this
    picker was the one display path still repeating it. `agentDisplayName`
    already ignored the field for hosted agents.
    """
    out = _run(
        {
            "ta": {
                "agent": _agent("tradingagents"),
                "storedModel": "nvidia/nemotron-3-nano-30b-a3b",
            },
            "ahf": {
                "agent": _agent("ai_hedge_fund"),
                "storedModel": "nvidia/nemotron-3-nano-30b-a3b",
            },
            "pipeline": {
                "agent": _agent("pipeline"),
                "storedModel": "anthropic/claude-haiku-4-5",
            },
        }
    )
    assert "nemotron" not in out["ta"]["optionLabel"]
    assert "TradingAgents" in out["ta"]["optionLabel"]
    assert "AI Hedge Fund" in out["ahf"]["optionLabel"]
    # A pipeline agent's stored model is its real one, so it still shows.
    assert "claude-haiku-4-5" in out["pipeline"]["optionLabel"]

# ---------------------------------------------------------------------------
# 5. The model field the modal paints on open. This function had NO executing
#    test, which is how a ReferenceError in it shipped: the only reference was
#    a substring check on its name.
# ---------------------------------------------------------------------------

def _run_model_field(scenarios):
    lifted = "\n".join(
        [
            js_const("SINGLE_SYMBOL_RUNTIMES"),
            _lift_object_const("HOSTED_RUNTIME_LABELS"),
            fn_body("function agentRuntimeType("),
            fn_body("function hostedRuntimeLabel("),
            fn_body("function syncBacktestModelFieldMode("),
        ]
    )
    script = f"""
let els = {{}};
function el(id) {{
  if (!els[id]) els[id] = {{ id, hidden: false, value: '', textContent: '' }};
  return els[id];
}}
const document = {{ getElementById: el }};
const IFIND_ASHARE_SOURCE = 'ifind_ashare';
const RULE_BASED_DECISION_SOURCE = 'rule_based';
let runBacktestModalAgent = null;
// The two tail calls are other units; stubbed so a failure here is this
// function's.
function syncRunBacktestInstructionPreview() {{}}
function syncRunBacktestSubmitAvailability() {{}}
{lifted}
const scenarios = {json.dumps(scenarios)};
const out = {{}};
for (const [name, s] of Object.entries(scenarios)) {{
  els = {{}};
  el('marketDataSourceSelect').value = s.source || 'alpaca';
  el('modelSelect').value = s.model || 'anthropic/claude-haiku-4-5';
  runBacktestModalAgent = s.agent;
  let threw = null;
  try {{
    syncBacktestModelFieldMode();
  }} catch (error) {{
    threw = String(error && error.name) + ': ' + String(error && error.message);
  }}
  out[name] = {{
    threw,
    readonlyText: el('runBacktestModelReadonly').textContent,
    readonlyHidden: el('runBacktestModelReadonly').hidden,
    modelSelectHidden: el('modelSelect').hidden,
    billingHidden: el('runBacktestBillingGroup').hidden,
  }};
}}
process.stdout.write(JSON.stringify(out));
"""
    return run_node(script)


@_needs_node
def test_the_model_field_does_not_throw_for_any_runtime():
    """The Critical this file now guards.

    `syncBacktestModelFieldMode` is reached from `openRunBacktestModal` via
    `syncMarketDataSourceUI`, *before* `modal.hidden = false`. A throw here
    therefore means clicking Run Backtest does nothing at all -- no modal, no
    error on screen -- and it applied to AI Hedge Fund as much as to
    TradingAgents, so it was a regression of shipped behaviour.
    """
    out = _run_model_field(
        {
            "pipeline": {"agent": _agent("pipeline")},
            "ahf": {"agent": _agent("ai_hedge_fund")},
            "ta": {"agent": _agent("tradingagents")},
            "none": {"agent": None},
            "simulation": {
                "agent": _agent("pipeline"),
                "source": "vnpy_simulation",
            },
        }
    )
    for name, result in out.items():
        assert result["threw"] is None, f"{name}: {result['threw']}"


@_needs_node
def test_a_hosted_run_names_its_own_runtime_in_the_model_field():
    out = _run_model_field(
        {
            "ahf": {"agent": _agent("ai_hedge_fund")},
            "ta": {"agent": _agent("tradingagents")},
        }
    )
    assert out["ahf"]["readonlyText"] == "AI Hedge Fund — hosted runtime"
    assert out["ta"]["readonlyText"] == "TradingAgents — hosted runtime"
    for result in out.values():
        # The picker is replaced by the read-only line, and billing is hidden:
        # a hosted run has no model or lane to choose.
        assert result["modelSelectHidden"] is True
        assert result["readonlyHidden"] is False
        assert result["billingHidden"] is True


@_needs_node
def test_the_pipeline_model_field_is_unchanged():
    out = _run_model_field(
        {
            "pipeline": {"agent": _agent("pipeline")},
            "simulation": {
                "agent": _agent("pipeline"),
                "source": "vnpy_simulation",
            },
        }
    )
    # A pipeline agent picks its own model and lane.
    assert out["pipeline"]["modelSelectHidden"] is False
    assert out["pipeline"]["readonlyHidden"] is True
    assert out["pipeline"]["billingHidden"] is False
    # Simulated data is rule-based: no model, no billing, and the text says so
    # rather than naming a runtime.
    assert "Rule-based" in out["simulation"]["readonlyText"]
    assert out["simulation"]["billingHidden"] is True

@_needs_node
def test_the_guard_refuses_a_preset_the_request_would_actually_send():
    """The guard has to judge the pane the request will read, not the chips.

    `applyUniverseDefaultsForRuntime` opens on Custom, but
    `syncMarketDataSourceUI`'s iFinD restore path calls
    `handleUniverseTabSwitch(previousTab)` straight afterwards, and that
    function never consults the Built-in button's `disabled`. Reading the
    chips unconditionally then passed on one leftover chip while
    `getSelectedAssets()` sent all 30 DJIA names -- and the refusal arrived as
    the server's 422 after the modal had closed, which is the failure this
    guard exists to prevent.
    """
    out = _run(
        {
            "djia": {"agent": _agent("tradingagents"), "builtinAfter": "djia"},
            "mag7": {"agent": _agent("tradingagents"), "builtinAfter": "mag7"},
        }
    )
    djia = out["djia"]["refusalOnBuiltinTab"]
    assert djia, "a 30-name preset must be refused"
    assert "30" in djia
    # And it names the tab the user has to go to, rather than a tab they are
    # not on.
    assert "Custom" in djia
    assert "custom universe" not in djia

    assert "7" in out["mag7"]["refusalOnBuiltinTab"]


@_needs_node
def test_the_builtin_pane_does_not_make_multi_symbol_runtimes_complain():
    out = _run(
        {
            "pipeline": {"agent": _agent("pipeline"), "builtinAfter": "djia"},
            "ahf": {"agent": _agent("ai_hedge_fund"), "builtinAfter": "djia"},
        }
    )
    assert out["pipeline"]["refusalOnBuiltinTab"] == ""
    assert out["ahf"]["refusalOnBuiltinTab"] == ""
