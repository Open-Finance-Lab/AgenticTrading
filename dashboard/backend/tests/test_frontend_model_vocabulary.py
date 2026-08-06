"""Guards for the single frontend source of truth for runnable models.

app.html used to carry two hand-maintained model <option> lists that drifted
apart: the backtest picker offered six models this platform does not run, and
omitted four it does. Both selects are now built from SUPPORTED_MODELS in
app.js. /app has no JS test harness, so -- per this suite's convention
(_frontend_source) -- the contract is asserted against the shipped source, and
behaviour is asserted by running the real functions under node.
"""

import re
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, js_const

EXPECTED_MODELS = [
    ("anthropic/claude-haiku-4-5", "Claude Haiku 4.5", "anthropic"),
    ("anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6", "anthropic"),
    ("openai/gpt-5.5", "GPT-5.5", "openai"),
    ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", "google"),
    ("deepseek/deepseek-v4-pro", "DeepSeek V4 Pro", "deepseek"),
    ("qwen/qwen3.7-plus", "Qwen3.7 Plus", "qwen"),
]


def _select_markup(select_id: str) -> str:
    """The <select id="..."> element's own markup, up to its closing tag."""
    start = APP_HTML.index(f'id="{select_id}"')
    open_tag = APP_HTML.rindex("<select", 0, start)
    close = APP_HTML.index("</select>", start)
    return APP_HTML[open_tag:close]


@pytest.mark.parametrize("select_id", ["modelSelect", "builtinAgentModel"])
def test_model_selects_carry_no_hardcoded_options(select_id):
    """Neither picker may hold its own option list -- that is how they drifted."""
    assert "<option" not in _select_markup(select_id), (
        f"#{select_id} still hardcodes options; build it from SUPPORTED_MODELS"
    )


def test_supported_models_are_the_six_runnable_models():
    source = js_const("SUPPORTED_MODELS")
    found = re.findall(
        r"slug:\s*'([^']+)',\s*label:\s*'([^']+)',\s*vendor:\s*'([^']+)'", source
    )
    assert found == EXPECTED_MODELS


def test_retired_models_are_gone_from_the_frontend():
    """The six models the old picker offered that this platform cannot run."""
    for retired in (
        "claude-opus-4.7",
        "gpt-5.2",
        "gpt-5-mini",
        "deepseek-v4-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
    ):
        assert retired not in APP_HTML, f"{retired} still offered in app.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_model_options_html_renders_every_supported_model():
    script = f"""
function escapeHtml(s) {{ return String(s); }}
{js_const("SUPPORTED_MODELS")}
{fn_body("function modelOptionsHtml")}
console.log(modelOptionsHtml(SUPPORTED_MODELS));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    html = result.stdout
    for slug, label, _vendor in EXPECTED_MODELS:
        assert f'<option value="{slug}">{label}</option>' in html


_FAKE_SELECT = """
class FakeOption {
  constructor(value, text) { this.value = value; this.textContent = text; this.dataset = {}; }
}
class FakeSelect {
  constructor(values) {
    this.options = values.map((v) => new FakeOption(v, v));
    this.value = values[0] || '';
  }
  appendChild(option) { this.options.push(option); }
  querySelectorAll(selector) {
    if (selector !== 'option[data-injected-model]') throw new Error('unexpected: ' + selector);
    const self = this;
    const matches = this.options.filter((o) => o.dataset.injectedModel);
    matches.forEach((o) => { o.remove = () => {
      self.options = self.options.filter((x) => x !== o);
    }; });
    return matches;
  }
  // A real <select>'s value setter silently resolves to '' when the assigned
  // string matches no current <option>. Mirror that here so a future edit that
  // sets .value before the matching option exists (e.g. reordering appendChild
  // and the .value assignment) shows up as a blank value in this fake too,
  // instead of the fake accepting any string unconditionally.
  get value() { return this._value; }
  set value(v) {
    this._value = this.options.some((o) => o.value === v) ? v : '';
  }
}
"""


def _run_sync_harness(body: str) -> str:
    script = f"""
{_FAKE_SELECT}
let SELECT = null;
const document = {{
  getElementById: (id) => (id === 'modelSelect' ? SELECT : null),
  createElement: () => new FakeOption('', ''),
}};
function formatAgentModelLabel(m) {{ return String(m); }}
{fn_body("function normalizeBacktestModelId")}
{fn_body("function findBacktestModelOption")}
{fn_body("function syncModelSelectFromAgent")}
{body}
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unrepresentable_model_is_injected_not_left_stale():
    """The regression: agent B's run must never submit agent A's model."""
    out = _run_sync_harness(
        # Agent A leaves qwen in the select; agent B runs a legacy model.
        "SELECT = new FakeSelect(['anthropic/claude-haiku-4-5', 'qwen/qwen3.7-plus']);"
        "SELECT.value = 'qwen/qwen3.7-plus';"
        "syncModelSelectFromAgent({model_name: 'gpt-5.2'});"
        "console.log(SELECT.value);"
    )
    assert out == "gpt-5.2"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_injected_options_do_not_accumulate():
    out = _run_sync_harness(
        "SELECT = new FakeSelect(['anthropic/claude-haiku-4-5']);"
        "syncModelSelectFromAgent({model_name: 'gpt-5.2'});"
        "syncModelSelectFromAgent({model_name: 'local-model'});"
        "console.log(SELECT.options.map((o) => o.value).join(','));"
    )
    assert out == "anthropic/claude-haiku-4-5,local-model"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_representable_model_selects_its_own_option():
    out = _run_sync_harness(
        "SELECT = new FakeSelect(['anthropic/claude-haiku-4-5', 'qwen/qwen3.7-plus']);"
        "SELECT.value = 'qwen/qwen3.7-plus';"
        "syncModelSelectFromAgent({model_name: 'anthropic/claude-haiku-4-5'});"
        "console.log(SELECT.value + '|' + SELECT.options.length);"
    )
    assert out == "anthropic/claude-haiku-4-5|2"
