"""Behaviour contracts for the Allocated Capital number inputs.

These inputs carry a hand-written spinner-correction layer (`bindCashStepInput`
in app.js) because native number spinners step by 1 on the first click even when
a larger `step` is configured. The correction has to tell two indistinguishable-
looking events apart: a spinner misfire, and a human editing the text.

The original discriminator was `Math.abs(newValue - lastValue) === 1`, which
cannot tell them apart at all -- deleting the last character of "1" produces
`Number('') === 0`, a diff of exactly -1, so the guard fired on a *deletion* and
rewrote the field by a whole step. Clamped to `min`, that put "1" back in the
backtest field and "0" in the paper field the instant the user tried to clear
them, which is what the 2026-09-03 user feedback reported and screenshotted
(both fields sitting at their minimum, un-clearable).

These run the shipped functions under node against a fake element rather than
asserting on source text: the bug was in the arithmetic, and only executing the
arithmetic can catch it. The convention (extract real source, execute under
node, never stub the thing under test) follows test_frontend_chart_first_home.py.
"""

import json
import re
import shutil
import subprocess

import pytest

from ._frontend_source import APP_HTML, APP_JS, FRONTEND, fn_body, strip_comments

_AGENT_EDITOR_JS = (FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")

_requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not available"
)

# A fake <input type=number>. Only the surface bindCashStepInput actually
# touches: value/step/min/max, dataset, attributes and event registration.
# `_fire` invokes the registered handlers directly so a test can send the exact
# event sequence a browser sends, in order, with no jsdom dependency.
#
# `validity.badInput` and the error slot are modelled because both are load-
# bearing and neither is observable from `.value`: a number input sanitises
# unparseable text to "", and `role="alert"` announces nothing if the text is
# written while the region is still `hidden`. The slot RECORDS the order of its
# own writes, since that ordering is the entire contract.
_FAKE_ELEMENT = """
function makeSlot() {
  const writes = [];
  return {
    writes,
    _hidden: true,
    _text: '',
    get hidden() { return this._hidden; },
    set hidden(v) { this._hidden = v; writes.push(['hidden', v]); },
    get textContent() { return this._text; },
    set textContent(v) { this._text = v; writes.push(['textContent', v]); },
  };
}
function makeInput({ value = '1000', step = 'any', min = '0', max = '3000',
                    cashStep = '100', withSlot = false } = {}) {
  const listeners = Object.create(null);
  const slot = withSlot ? makeSlot() : null;
  const dataset = Object.create(null);
  if (cashStep != null) dataset.cashStep = String(cashStep);
  return {
    value: String(value),
    step: String(step),
    min: String(min),
    max: String(max),
    validity: { badInput: false },
    _slot: slot,
    parentElement: slot
      ? { querySelector(sel) { return sel === '[data-cash-error-slot]' ? slot : null; } }
      : null,
    dataset,
    attributes: Object.create(null),
    setAttribute(name, val) { this.attributes[name] = String(val); },
    removeAttribute(name) { delete this.attributes[name]; },
    getAttribute(name) { return this.attributes[name] ?? null; },
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    dispatchEvent(event) {
      (listeners[event.type] || []).forEach((fn) => fn(event));
      return true;
    },
    _fire(type, extra = {}) {
      const event = { type, preventDefault() {}, ...extra };
      (listeners[type] || []).forEach((fn) => fn(event));
    },
  };
}
"""


def _harness() -> str:
    """The shipped functions plus everything they close over, in order."""
    return "\n".join(
        [
            fn_body("function cashStepMeta("),
            fn_body("function snapCashStepValue("),
            fn_body("function nudgeCashStepInput("),
            fn_body("function bindCashStepInput("),
            fn_body("function setCashInputValidity("),
            fn_body("function resetCashStepInput("),
            _FAKE_ELEMENT,
        ]
    )


def _cash_input_tags() -> dict[str, str]:
    """Every bound capital input's real `<input>` tag, straight from app.html."""
    ids = re.findall(r"'([A-Za-z]+)'", fn_body("const CASH_STEP_INPUT_IDS ="))
    assert len(ids) >= 4, "the bound-input list shrank; update this guard"
    tags = {}
    for input_id in ids:
        match = re.search(rf'<input id="{input_id}"[^>]*>', APP_HTML)
        assert match, f"{input_id} is bound in app.js but absent from app.html"
        tags[input_id] = match.group(0)
    return tags


def _shipped_cash_mins() -> dict[str, str]:
    """`min` as each bound field actually ships it.

    Read rather than written down. Every case below used to hardcode the min it
    expected, so when the backtest field moved from `min="1"` to `min="0"` on
    2026-09-10 its regression test went on probing a synthetic element no field
    in the app resembled -- passing, and covering nothing that shipped. A test
    whose fixture is copied out of the markup drifts the moment the markup does,
    and drifts silently, because the copy still describes *something*.
    """
    mins = {}
    for input_id, tag in _cash_input_tags().items():
        match = re.search(r'\bmin="([^"]*)"', tag)
        assert match, f"{input_id} ships no min= for the clamp to read"
        mins[input_id] = match.group(1)
    return mins


_SHIPPED_CASH_MINS = _shipped_cash_mins()


def _run_node(expr: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    script = f"{_harness()}\nconsole.log(JSON.stringify({expr}));"
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)


def _backspace_to_empty(min_value: str) -> str:
    """Value left in the field after backspacing "1000" away, character by character.

    This is the reported gesture: select nothing, hold backspace. The browser
    fires one `input` event per keystroke with the shortened text already in
    `.value`, and the last one carries the empty string.
    """
    return _run_node(
        f"""(() => {{
  const el = makeInput({{ value: '1000', min: '{min_value}' }});
  bindCashStepInput(el);
  for (const text of ['100', '10', '1', '']) {{
    el.value = text;
    el._fire('input', {{ inputType: 'deleteContentBackward' }});
  }}
  return el.value;
}})()"""
    )


@_requires_node
@pytest.mark.parametrize(
    ("input_id", "min_value"), sorted(_SHIPPED_CASH_MINS.items())
)
def test_backspacing_a_capital_field_empty_leaves_it_empty(input_id, min_value):
    """The reported bug, run against every field that actually ships the layer.

    A field that refills itself while you are deleting cannot be retyped: the
    user is left inserting digits around a value they never asked for, which is
    exactly how the feedback describes it ("the number 1 would be confusing").

    The refill lands on `min`, so `min` is what makes the symptom look like one
    thing or another -- "1" reads as broken, "0" reads as a deliberate choice of
    no capital. That is why the case is driven off each field's shipped `min`
    instead of a literal: the two originally hardcoded here (1 and 0) described
    the fields as they stood before 2026-09-10, and the backtest field's has
    since moved.
    """
    assert _backspace_to_empty(min_value) == "", (
        f"{input_id}: the spinner-correction guard fired on a deletion "
        "and refilled the field"
    )


@_requires_node
def test_a_spinner_click_still_moves_by_a_full_step():
    """The behaviour the correction layer exists to provide, kept honest.

    Without this, the simplest way to pass the two cases above is to delete the
    correction entirely -- which reintroduces the native +/-1 spinner glitch the
    layer was written for. A spinner click reaches us as an `input` event with
    no `inputType`, having already moved the value by 1.
    """
    result = _run_node(
        """(() => {
  const el = makeInput({ value: '1000' });
  bindCashStepInput(el);
  el.value = '1001';          // what the native spinner just did
  el._fire('input');          // no inputType: not a keystroke
  return el.value;
})()"""
    )
    assert result == "1100", "a spinner click must land on the configured step"


_DEFAULT_CASH_MIN = sorted(set(_SHIPPED_CASH_MINS.values()))[0]


def _type_and_commit(text: str, *, min_value: str = _DEFAULT_CASH_MIN) -> dict:
    """Type `text` into a cleared field and blur it, as a user filling it in does."""
    return _run_node(
        f"""(() => {{
  const el = makeInput({{ value: '1000', min: '{min_value}' }});
  bindCashStepInput(el);
  el.value = '';
  el._fire('input', {{ inputType: 'deleteContentBackward' }});
  el.value = '{text}';
  el._fire('input', {{ inputType: 'insertText' }});
  el._fire('change');
  return {{ value: el.value, invalid: el.getAttribute('aria-invalid'), error: el.dataset.cashError || null }};
}})()"""
    )


@_requires_node
def test_a_typed_amount_off_the_step_grid_is_kept_verbatim():
    """`step` is the spinner's increment, not a constraint on typed text.

    Snapping typed input to it rewrote every amount that was not a multiple of
    100, silently and in both directions: "1234" became "1200". The server takes
    any integer in range, so the grid was never anything but our own spinner's.
    """
    assert _type_and_commit("1234")["value"] == "1234", (
        "typed text was rounded to the spinner's grid"
    )


@_requires_node
def test_a_small_typed_amount_is_not_rounded_away_to_the_minimum():
    """The compound case, and the one that reads as the reported bug recurring.

    "5" rounded to 0 on the step grid and was then clamped up to `min`, so the
    field showed "1" -- the same wrong digit as the refill, arrived at by a
    different route. Fixing only the refill would leave this one shipping.
    """
    result = _type_and_commit("5")
    assert result["value"] == "5", "a valid small amount was rounded away"
    assert result["invalid"] is None, "a valid amount must not be flagged"


@_requires_node
def test_typing_below_the_minimum_is_flagged_rather_than_rewritten():
    """Their second ask: "if smaller than the minimum, return a red error message".

    The value is left on screen and the field marked invalid, so the message has
    something to render from and the user's own number is still there to fix.

    The probe is ``-1`` against ``min="0"``, not ``0`` against ``min="1"``: both
    shipped fields have carried ``min="0"`` since 2026-09-10 (see
    tests/test_zero_backtest_capital.py), so a case built on a min of 1 would go
    on passing against a synthetic element no field in the app resembles.
    """
    result = _type_and_commit("-1", min_value="0")
    assert result["value"] == "-1", "the typed value was overwritten instead of flagged"
    assert result["invalid"] == "true", "nothing marks the field invalid"
    assert result["error"], "no message for the inline error to render"


@_requires_node
def test_typing_above_the_maximum_is_flagged_rather_than_rewritten():
    """The other end of the same range, which the old clamp also swallowed.

    Typing 5000 into a max-3000 field silently produced 3000 -- a number the
    user never chose, in a field that governs how much capital a run is given.
    """
    result = _type_and_commit("5000")
    assert result["value"] == "5000"
    assert result["invalid"] == "true", "an over-max amount was clamped silently"


@_requires_node
def test_correcting_an_invalid_value_clears_the_flag():
    """An error that outlives its cause is worse than none: the field would sit
    red on a value that is now fine, training the user to ignore it.
    """
    result = _run_node(
        """(() => {
  const el = makeInput({ value: '1000', min: '0' });
  bindCashStepInput(el);
  el.value = '-5';
  el._fire('input', { inputType: 'insertText' });
  el._fire('change');
  el.value = '500';
  el._fire('input', { inputType: 'insertText' });
  el._fire('change');
  return { value: el.value, invalid: el.getAttribute('aria-invalid'), error: el.dataset.cashError || null };
})()"""
    )
    assert result["value"] == "500"
    assert result["invalid"] is None, "the invalid flag survived a corrected value"
    assert result["error"] is None, "the message survived a corrected value"


def test_every_bound_capital_input_has_somewhere_to_render_its_error():
    """`setCashInputValidity` writes into a `[data-cash-error-slot]` inside the
    input's own parent. A bound field without one still gets `aria-invalid`, so
    the rejection is announced to a screen reader and invisible to everyone
    else -- the field just refuses to save with no on-screen reason. That is a
    worse failure than the silent clamp this replaced, so the markup is pinned
    rather than left to whoever next edits the form.
    """
    ids = re.findall(r"'([A-Za-z]+)'", fn_body("const CASH_STEP_INPUT_IDS ="))
    assert len(ids) >= 4, "the bound-input list shrank; update this guard"
    for input_id in ids:
        match = re.search(rf'<input id="{input_id}"[^>]*>', APP_HTML)
        assert match, f"{input_id} is bound in app.js but absent from app.html"
        # The slot must be a sibling: setCashInputValidity looks it up through
        # `input.parentElement`, so one placed further out never resolves.
        label_start = APP_HTML.rfind("<label", 0, match.start())
        label_end = APP_HTML.index("</label>", match.end())
        assert "data-cash-error-slot" in APP_HTML[label_start:label_end], (
            f"{input_id} can be marked invalid with no way to say why"
        )


# ---------------------------------------------------------------------------
# The form-submit path: the half none of the cases above touch.
# ---------------------------------------------------------------------------


def test_the_spinner_increment_is_not_also_a_form_constraint():
    """The bug that survived every case above, because none of them submits.

    `externalAgentCashAllocation` and `builtinAgentCashAllocation` sit inside
    forms with `type="submit"` buttons and no `novalidate`. While the change
    handler snapped typed text onto the step grid, the field could not violate
    `step` -- so shipping `step="100"` cost nothing. Keeping the typed value
    (the fix this file is about) inverted that: the browser began refusing the
    submit for any amount off the grid, with "the two nearest valid values are
    1200 and 1300", and `submitCreateExternalAgent` never ran. Nothing in JS
    can report it, because the value is in range and perfectly valid to us.

    So the increment lives in `data-cash-step`, which only bindCashStepInput
    reads, and `step` stays `any` -- the one value that keeps `stepMismatch`
    out of the form's constraint set. `min`/`max`/`required` are deliberately
    left native: those are real constraints, and the other fields in the same
    forms still want the browser's own message.
    """
    for input_id, tag in _cash_input_tags().items():
        assert 'step="any"' in tag, (
            f"{input_id} carries a numeric step. Inside a form that is a native "
            "constraint, and the create-agent submit is refused for every amount "
            "off the grid -- silently, from this module's point of view"
        )
        assert re.search(r'data-cash-step="\d+"', tag), (
            f"{input_id} declares no data-cash-step, so cashStepMeta falls back "
            "to a literal and the increment is no longer stated in the markup"
        )


def test_binding_a_field_does_not_write_step_back_onto_it():
    """The other half of the same fix, and the easier one to undo by accident.

    `bindCashStepInput` used to normalise `step` to '100' at bind time, which
    put the constraint back on the element regardless of what the markup said.
    A guard on the markup alone would stay green through that.
    """
    # Comments stripped first: the line this guards against is now DESCRIBED in
    # a comment inside the same function, so an un-stripped scan is satisfied by
    # the prose explaining the bug rather than by the code avoiding it. That is
    # the failure `strip_comments` exists for, and this guard walked into it.
    body = strip_comments(fn_body("function bindCashStepInput("))
    assert not re.search(r"input\.step\s*=", body), (
        "bindCashStepInput assigns input.step again -- that reinstates the "
        "native step constraint the markup was changed to avoid"
    )


# ---------------------------------------------------------------------------
# "" is two states.
# ---------------------------------------------------------------------------


@_requires_node
def test_unparseable_text_is_reported_rather_than_read_as_cleared():
    """`type="number"` hands us "" for "1e" and "--" exactly as it does for a
    field the user emptied on purpose. Read as "cleared", junk showed no
    message and submitted `parseAgentCashAllocationInput('')` -- the 1000
    default -- under text still visibly sitting in the box.
    """
    result = _run_node(
        """(() => {
  const el = makeInput({ value: '1000' });
  bindCashStepInput(el);
  el.value = '';                 // what the control does with "1e"
  el.validity.badInput = true;
  el._fire('input', { inputType: 'insertText' });
  el._fire('change');
  return { invalid: el.getAttribute('aria-invalid'), error: el.dataset.cashError || null };
})()"""
    )
    assert result["invalid"] == "true", "unparseable text was accepted silently"
    assert result["error"], "no message for the inline error to render"


@_requires_node
def test_deliberately_clearing_the_field_still_reports_nothing():
    """The other side of the branch above, and the one the whole PR is for: an
    empty field with no bad input is a field mid-edit, not an error.
    """
    result = _run_node(
        """(() => {
  const el = makeInput({ value: '1000' });
  bindCashStepInput(el);
  el.value = '';
  el._fire('input', { inputType: 'deleteContentBackward' });
  el._fire('change');
  return { value: el.value, invalid: el.getAttribute('aria-invalid') };
})()"""
    )
    assert result["value"] == ""
    assert result["invalid"] is None, "clearing the field was flagged as an error"


# ---------------------------------------------------------------------------
# The message has to be announceable, and has to not outlive its value.
# ---------------------------------------------------------------------------


@_requires_node
def test_the_error_slot_is_revealed_before_its_text_is_written():
    """`role="alert"` is a live region, and a `hidden` element is not in the
    accessibility tree. Writing the text first mutates a region nobody is
    monitoring, so the reveal that follows announces nothing -- and the red
    border is the only other signal, which is the reader this markup exists for.
    """
    writes = _run_node(
        """(() => {
  const el = makeInput({ value: '1000', min: '1', withSlot: true });
  bindCashStepInput(el);
  el.value = '5000';
  el._fire('input', { inputType: 'insertText' });
  el._fire('change');
  return el._slot.writes;
})()"""
    )
    kinds = [w[0] for w in writes]
    assert "textContent" in kinds and "hidden" in kinds, writes
    assert kinds.index("hidden") < kinds.index("textContent"), (
        f"the slot's text was written before it was revealed ({writes}); a "
        "hidden live region announces nothing when it is later unhidden"
    )


@_requires_node
def test_refilling_the_field_from_code_clears_a_stale_error():
    """Switching agents in the editor, or reopening a create modal.

    Both assign `.value` directly, which fires no event -- so the previous
    agent's red border, its `aria-invalid` and its "Enter an amount between..."
    message all rode along onto the next agent's perfectly valid number and
    stayed until someone typed in the field. `form.reset()` is the same story:
    it restores `value="1000"` and touches none of the state above.
    """
    result = _run_node(
        """(() => {
  const el = makeInput({ value: '1000', min: '1', withSlot: true });
  bindCashStepInput(el);
  el.value = '5000';
  el._fire('input', { inputType: 'insertText' });
  el._fire('change');            // now red, with a message
  el.value = '1000';             // the next agent, assigned from code
  resetCashStepInput(el);
  return {
    invalid: el.getAttribute('aria-invalid'),
    error: el.dataset.cashError || null,
    slotText: el._slot.textContent,
    slotHidden: el._slot.hidden,
  };
})()"""
    )
    assert result["invalid"] is None, "aria-invalid survived the refill"
    assert result["error"] is None, "the message survived the refill"
    assert result["slotText"] == "" and result["slotHidden"] is True


@_requires_node
def test_the_reset_also_resyncs_the_spinner_baseline():
    """`lastValue` is closure state, and a refill that leaves it stale makes the
    NEXT spinner click compute its diff against a number no longer on screen --
    a +/-1 that is not a glitch, corrected by a whole step.
    """
    value = _run_node(
        """(() => {
  const el = makeInput({ value: '1000' });
  bindCashStepInput(el);
  el.value = '2000';             // assigned from code, no event
  resetCashStepInput(el);
  el.value = '2001';             // native spinner, +1
  el._fire('input');
  return el.value;
})()"""
    )
    assert value == "2100", (
        "the spinner baseline was not re-synced on refill, so the correction "
        "was computed against a stale value"
    )


def test_every_caller_that_refills_a_capital_field_resets_it():
    """The unit above is only worth having if the three refill sites call it.

    Source-shape rather than behavioural: `fillHeader` and the two modal
    openers reach for `document.getElementById`, which the node harness has no
    way to stand up. This is what fails when a fourth refill site is added.
    """
    fill_header = fn_body("function fillHeader(", _AGENT_EDITOR_JS)
    assert fill_header.count("resetCapitalInput(") == 2, (
        "agent-editor's fillHeader refills both capital fields; each one needs "
        "its validity state cleared or it inherits the previous agent's error"
    )
    for opener in ("function openCreateExternalAgentModal(",
                   "function openCreateBuiltinAgentModal("):
        body = fn_body(opener, APP_JS)
        assert "form.reset()" in body, f"{opener} no longer resets its form"
        assert "resetCashStepInput(" in body, (
            f"{opener} resets the form without clearing the capital field's "
            "error state, so the modal reopens red on a default value"
        )


def test_every_capital_field_accepts_zero():
    """Both halves of the Allocated Capital card agree about zero.

    They did not until 2026-09-10: the Paper Trading input shipped `min="0"`
    and the Backtesting input beside it shipped `min="1"`, so a typed 0 was
    accepted on the left and refused on the right with "Enter an amount between
    $1 and $3,000". Two boxes in one card disagreeing about the same number
    reads as the card being broken, which is how it was reported.

    `min` is also what `snapCashStepValue` clamps to, so the floor was doing
    double duty as the value the field refilled itself with -- see the module
    docstring. The backend floor moved with it
    (tests/test_zero_backtest_capital.py); this is the half a user can see.
    """
    for input_id, tag in _cash_input_tags().items():
        assert 'min="0"' in tag, (
            f"{input_id} does not accept $0. The two fields in the Allocated "
            "Capital card must agree, and the backend floor is now 0"
        )


def test_the_editor_does_not_reimpose_a_dollar_floor_in_js():
    """The `min` attribute is not the only gate -- getEditorState has its own.

    Relaxing the markup and leaving `value < 1` in the validator would swap a
    browser message for a thrown one: the field would accept the 0 and the save
    would fail, which is a worse version of the same bug.
    """
    source = strip_comments(_AGENT_EDITOR_JS)
    assert "value < 1" not in source, (
        "getEditorState still rejects amounts below $1 in JS, so the relaxed "
        "min attribute only moves where the refusal comes from"
    )
    assert "must be at least $1" not in source


def test_both_backtest_capital_resolvers_honour_a_saved_zero():
    """app.js renders the number; agent-editor.js refills the box with it.

    They are separate implementations of one fallback chain, and both used
    `value > 0` on the *saved* candidate. That made a $0 setting invisible in
    the card and self-erasing in the editor -- reopening Configure showed the
    fallback, and the next save wrote the fallback back. Fixing one and not the
    other leaves the erasure intact, so this asserts the shape in both.
    """
    for label, source in (
        ("app.js", strip_comments(APP_JS)),
        ("agent-editor.js", strip_comments(_AGENT_EDITOR_JS)),
    ):
        assert "backtest_allocation != null" in source, (
            f"{label} does not separate a saved backtest_allocation of 0 from a "
            "NULL column, so a deliberate $0 falls through to the paper sleeve"
        )
