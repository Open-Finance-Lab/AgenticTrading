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

from ._frontend_source import APP_HTML, fn_body

_requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not available"
)

# A fake <input type=number>. Only the surface bindCashStepInput actually
# touches: value/step/min/max, dataset, attributes and event registration.
# `_fire` invokes the registered handlers directly so a test can send the exact
# event sequence a browser sends, in order, with no jsdom dependency.
_FAKE_ELEMENT = """
function makeInput({ value = '1000', step = '100', min = '0', max = '3000' } = {}) {
  const listeners = Object.create(null);
  return {
    value: String(value),
    step: String(step),
    min: String(min),
    max: String(max),
    dataset: Object.create(null),
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
            _FAKE_ELEMENT,
        ]
    )


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
def test_backspacing_the_backtest_field_empty_leaves_it_empty():
    """The reported bug, at the field the user screenshotted.

    A field that refills itself while you are deleting cannot be retyped: the
    user is left inserting digits around a "1" they never asked for, which is
    exactly how the feedback describes it ("the number 1 would be confusing").
    """
    assert _backspace_to_empty("1") == "", (
        "the spinner-correction guard fired on a deletion and refilled the field"
    )


@_requires_node
def test_backspacing_the_paper_trading_field_empty_leaves_it_empty():
    """The same defect, one field over, and the reason it needs its own case.

    The paper field's `min` is 0, so the refill lands on "0" rather than "1" --
    a value that looks deliberate rather than broken. Asserting only the
    backtest field would leave the paper field free to regress into a state that
    reads as a real user choice of "no capital".
    """
    assert _backspace_to_empty("0") == "", (
        "the spinner-correction guard fired on a deletion and refilled the field"
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


def _type_and_commit(text: str, *, min_value: str = "1") -> dict:
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
    """Their second ask: "if smaller than 1, return a red error message".

    The value is left on screen and the field marked invalid, so the message has
    something to render from and the user's own number is still there to fix.
    """
    result = _type_and_commit("0")
    assert result["value"] == "0", "the typed value was overwritten instead of flagged"
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
  const el = makeInput({ value: '1000', min: '1' });
  bindCashStepInput(el);
  el.value = '0';
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
