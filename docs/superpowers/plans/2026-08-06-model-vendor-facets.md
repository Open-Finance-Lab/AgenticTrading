# Model-Vendor Facets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AI *model* a browsable dimension of the platform — a vendor facet on Community, an open-source badge, four new templates, and two "make another one" hooks — after first repairing the backtest model picker that would silently mis-submit the new templates' models.

**Architecture:** Four independently-shippable PRs. The model axis is a **pure frontend derivation** from the `model_name` field every agent and template already carries — no database column, no migration, no Postgres-twin parity work, no `openapi` enum. Two new frontend constants (`SUPPORTED_MODELS`, `MODEL_VENDORS`) become the single source of truth for, respectively, *which models are runnable* and *who makes them / how they are licensed*. Two small backend additions (a `model_name` override on clone, a `duplicate` endpoint) back the conversion hooks.

**Tech Stack:** Vanilla JS (no build step, no JS test toolchain), FastAPI + Pydantic, pytest. Frontend contracts are guarded by **parsing the shipped source as text from Python**, and where behaviour matters, by **executing lifted functions under `node`** from pytest.

## Global Constraints

These apply to every task below. Values are copied verbatim from `docs/superpowers/specs/2026-08-06-model-vendor-facets-design.md`.

- **Run everything from the repo root**, with the venv active: `source ~/atl-venv/bin/activate`.
- **Never `git add -A` in this repo.** A bare `python -c "import dashboard.backend…"` runs lazy `ALTER`s against the committed prod seed DB (`dashboard/storage/data/backtest.db`). Stage files by explicit path, every time. Before every commit, run `git status --short` and confirm `dashboard/storage/data/backtest.db` is **not** listed.
- **The six runnable models** (`SUPPORTED_MODELS`), in this declaration order — declaration order is display order:
  | slug | label | vendor |
  |---|---|---|
  | `anthropic/claude-haiku-4-5` | `Claude Haiku 4.5` | `anthropic` |
  | `anthropic/claude-sonnet-4-6` | `Claude Sonnet 4.6` | `anthropic` |
  | `openai/gpt-5.5` | `GPT-5.5` | `openai` |
  | `google/gemini-3.1-pro-preview` | `Gemini 3.1 Pro Preview` | `google` |
  | `deepseek/deepseek-v4-pro` | `DeepSeek V4 Pro` | `deepseek` |
  | `qwen/qwen3.7-plus` | `Qwen3.7 Plus` | `qwen` |
- **The eight vendors** (`MODEL_VENDORS`), in this declaration order — declaration order is chip order:
  | key | prefix | label | licence |
  |---|---|---|---|
  | `anthropic` | `anthropic/` | `Claude` | `closed` |
  | `openai` | `openai/` | `GPT` | `closed` |
  | `google` | `google/` | `Gemini` | `closed` |
  | `deepseek` | `deepseek/` | `DeepSeek` | `open` |
  | `qwen` | `qwen/` | `Qwen` | `open` |
  | `nvidia` | `nvidia/nemotron` | `NVIDIA Nemotron` | `open` |
  | `meta` | `meta-llama/` | `Llama` | `open` |
  | `xai` | `x-ai/` | `Grok` | `closed` |
- **Unknown vendor resolves to `''`**, never to a default. `''` means "stays visible under All, excluded only by an explicit chip" — the contract `agentMarketKey` already documents at `app.js:529-539`.
- **No `model_name` whitelist on the API.** `create_agent` does not validate it today; validating only on clone/duplicate would be inconsistent and would drag in the `openapi` enum deploy gate #313 had to discharge.
- **Closed-source models get no badge.** Only `Open-source model` renders, and only on open-weight cards. Absence is not a negative claim about someone else's product.
- **Never auto-launch a backtest** from a duplicate/clone action. Auto-firing spends LLM credits on a click the user did not frame as "run".
- **Never loosen `dashboard/backend/infrastructure/llm/validator.py`.** It is a hard security boundary; nothing in this plan touches it.

### Deviations from the spec, decided during planning

1. **`MODEL_VENDORS` nvidia label is `NVIDIA Nemotron`, not `Nemotron`.** Spec §2 requires `formatModelProviderLabel`'s output stay **byte-identical**, and today it emits `"Powered by NVIDIA Nemotron"`. Deriving `"Powered by " + label` from a `Nemotron` label would silently change shipped copy. One field serving both the chip and the submeta is worth the slightly longer chip.
2. **`agent-editor.js`'s cache buster does not bump.** Spec §7 says all three bump; no task in this plan modifies that file, and bumping it would invalidate a cache for no change. `app.js` and `styles.css` bump in every PR that edits them — and in PR 1 that bump is **load-bearing**, not hygiene (see Task 1, Step 7).
3. **PR 3's catalog guard is the weaker of the two in spec §7.** The `MODEL_VENDORS`-prefix guard cannot exist until PR 4 defines `MODEL_VENDORS`. PR 3 ships the `SUPPORTED_MODELS` membership guard; PR 4 adds the vendor-prefix guard (Task 6).
4. **`duplicate` is restricted to built-in agents (400 otherwise).** Duplicating an *external* agent would mint a new API key through `create_agent`'s one-time-plaintext path, which is a credential-issuing surface the hook has no reason to open. Hook B is only offered on built-in cards anyway.

## Delivery order

| PR | Branch | Scope | Depends on |
|---|---|---|---|
| 1 | `fix/backtest-model-vocabulary` | Tasks 1–2 | — |
| 2 | `feat/agent-duplicate-and-clone-model` | Tasks 3–4 | — |
| 3 | `content/marketplace-vendor-spread` | Task 5 | PR 1 merged |
| 4 | `feat/community-model-facets` | Tasks 6–11 | PR 1 merged **and** PR 2 **live in prod** |

PRs 1 and 2 are independent and may run in parallel.

**PR 4's gate is a production probe, not a merge badge.** Merging to `main` auto-deploys, but Render lags Vercel by 10–40 minutes. Before merging PR 4, both of these must return the new shapes **from prod**:

```bash
curl -s https://agentictrading.onrender.com/openapi.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
print('duplicate route:', '/api/v1/agents/{agent_id}/duplicate' in d['paths']); \
print('clone model_name:', 'model_name' in d['components']['schemas']['CloneMarketplaceBody']['properties'])"
```

Both must print `True`. If PR 4 opens before that gate clears, it opens as a **draft** with `DO NOT MERGE until PR 2 is serving in prod (probe /openapi.json)` as the **first line of the body** — `main` has no branch protection and open clean PRs get merged by others.

---

# PR 1 — The model-vocabulary fix

Branch: `git switch -c fix/backtest-model-vocabulary`

### Task 1: One source of truth for runnable models

The backtest picker `#modelSelect` (`app.html:360`) offers nine models, six of which do not exist on this platform (`claude-opus-4.7`, `gpt-5.2`, `gpt-5-mini`, `deepseek-v4-flash`, `gemini-3.5-flash`, `gemini-2.5-pro`) and omits four that do. It is **not** dead markup: `syncBacktestModelFieldMode` (`app.js:1660`) unhides it for the iFinD A-share source, where it doubles as the rule-based-vs-LLM decision-source picker.

**Files:**
- Modify: `dashboard/frontend/app.js` (add constants + helpers after `MARKET_LABELS`, ~line 492; call the populator inside the `DOMContentLoaded` pure-DOM block, ~line 3954)
- Modify: `dashboard/frontend/app.html:360-370` (empty `#modelSelect`), `app.html:798-805` (empty `#builtinAgentModel`), `app.html:1733` (cache buster)
- Test: `dashboard/backend/tests/test_frontend_model_vocabulary.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `const SUPPORTED_MODELS` — array of `{ slug: string, label: string, vendor: string }`, six entries in the Global Constraints order.
  - `function modelOptionsHtml(models) -> string` — pure, `<option>` markup only, no DOM access.
  - `function populateSupportedModelSelects() -> void` — writes that markup into `#modelSelect` and `#builtinAgentModel`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_frontend_model_vocabulary.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest dashboard/backend/tests/test_frontend_model_vocabulary.py -v
```

Expected: FAIL. `test_model_selects_carry_no_hardcoded_options` fails for both ids, and `test_supported_models_are_the_six_runnable_models` fails with `AssertionError: SUPPORTED_MODELS not found in app.js` raised inside `js_const`.

- [ ] **Step 3: Add the constant and the two helpers to `app.js`**

Insert immediately after the `MARKET_LABELS` block and before the `window.AGENT_SHELF_LABELS` export (~line 493) — this is the taxonomy neighbourhood, where the market axis is already declared:

```js
/** Every model a user can actually pick and run here. The single source for
 * both model <select> elements: the Run Backtest picker (#modelSelect, live
 * only on the iFinD A-share path) and the Create Built-in picker
 * (#builtinAgentModel, which the Configure editor clones its own options from).
 *
 * These lists were hand-maintained separately and drifted: the backtest picker
 * offered six models this platform does not run and omitted four it does, and
 * an agent on an unlisted model silently submitted the *previous* agent's
 * selection (see syncModelSelectFromAgent). Declaration order is display order.
 *
 * The AI Hedge Fund runtime's Nemotron is deliberately absent: it is a property
 * of a hosted runtime, not a user choice, and syncBacktestModelFieldMode
 * already renders that case as "AI Hedge Fund — hosted runtime". */
const SUPPORTED_MODELS = [
  { slug: 'anthropic/claude-haiku-4-5', label: 'Claude Haiku 4.5', vendor: 'anthropic' },
  { slug: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6', vendor: 'anthropic' },
  { slug: 'openai/gpt-5.5', label: 'GPT-5.5', vendor: 'openai' },
  { slug: 'google/gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro Preview', vendor: 'google' },
  { slug: 'deepseek/deepseek-v4-pro', label: 'DeepSeek V4 Pro', vendor: 'deepseek' },
  { slug: 'qwen/qwen3.7-plus', label: 'Qwen3.7 Plus', vendor: 'qwen' },
];

/** Pure: no DOM, so the guards can run it under node. */
function modelOptionsHtml(models) {
  return models
    .map((model) => `<option value="${escapeHtml(model.slug)}">${escapeHtml(model.label)}</option>`)
    .join('');
}

/** Fill both model pickers. Runs once, in the pure-DOM boot block, which is
 * before syncIFindModelControl can prepend #modelSelect's "Rule-based" option
 * -- calling this again later would wipe that option out. */
function populateSupportedModelSelects() {
  const html = modelOptionsHtml(SUPPORTED_MODELS);
  const backtestPicker = document.getElementById('modelSelect');
  if (backtestPicker) backtestPicker.innerHTML = html;
  const createPicker = document.getElementById('builtinAgentModel');
  if (createPicker) createPicker.innerHTML = html;
}
```

- [ ] **Step 4: Empty both `<select>` elements in `app.html`**

Replace `app.html:360-370` with:

```html
                    <!-- Options are written by populateSupportedModelSelects() from
                         SUPPORTED_MODELS in app.js. Do not hardcode them here: this
                         list and #builtinAgentModel's drifted apart once already. -->
                    <select class="control-select" id="modelSelect" hidden></select>
```

Replace `app.html:798-805` with:

```html
                    <!-- Options are written by populateSupportedModelSelects() in app.js.
                         js/agent-editor.js clones this select's options for the Configure
                         screen's picker, at editor-open time, so it inherits them too. -->
                    <select id="builtinAgentModel"></select>
```

- [ ] **Step 5: Call the populator in the boot block**

In `app.js`'s `DOMContentLoaded` handler, inside the "Pure-DOM wiring before ANY network await" block, immediately after `setupTickerScrollControls();` (~line 3956):

```js
    populateSupportedModelSelects();
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
pytest dashboard/backend/tests/test_frontend_model_vocabulary.py -v
```

Expected: PASS (5 cases; the node case skips if `node` is absent).

- [ ] **Step 7: Bump `app.js`'s cache buster — this is load-bearing**

`app.html` is the document and is not cache-busted; `app.js` is. A browser holding the old `app.js` (`v=73`) alongside the new `app.html` would render **two empty model pickers**, because nothing would populate the now-optionless selects. Edit `dashboard/frontend/app.html:1733`:

```html
    <script src="app.js?v=74" defer></script>
```

- [ ] **Step 8: Run the full frontend guard suite**

```bash
pytest dashboard/backend/tests/ -k "frontend or model or my_agents" -q
```

Expected: PASS. Nothing else asserts on the old option list, but `test_frontend_marketplace_placement.py` and `test_my_agents_card_ui.py` parse the same files.

- [ ] **Step 9: Commit**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/frontend/app.js dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_model_vocabulary.py
git commit -m "fix(backtest): build both model pickers from one supported-model list"
```

---

### Task 2: Never submit a leftover model

`syncModelSelectFromAgent` (`app.js:1670`) is a **no-op** when the selected agent's model matches no option. On the hidden path `resolveBacktestModelRequest`'s `!backtestModelPickerIsLiveControl()` guard covers for that; on the **live iFinD path** it does not, so the run submits whatever the previously-selected agent left in the select — no error, no warning, wrong model in the run record. Task 1 shrinks the exposure but cannot close it: legacy agents carry values like `local-model` and `gpt-5.2` that will never be in `SUPPORTED_MODELS`.

**Files:**
- Modify: `dashboard/frontend/app.js:1670-1677` (`syncModelSelectFromAgent`), `app.js:1646-1650` (comment only)
- Test: `dashboard/backend/tests/test_frontend_model_vocabulary.py` (extend)

**Interfaces:**
- Consumes: `SUPPORTED_MODELS`, `populateSupportedModelSelects` (Task 1); existing `findBacktestModelOption`, `formatAgentModelLabel`.
- Produces: `syncModelSelectFromAgent(agent)` now guarantees the select's value is either the agent's own model or an explicit user choice — never a stale neighbour's.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_model_vocabulary.py`:

```python
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
```

The JS passed to `_run_sync_harness` is built from adjacent Python string literals, so it must carry **no comments of its own** — a `//` outside the quotes is a Python syntax error, and one inside would comment out the rest of the concatenated single line. Explain in Python comments above the call, as above.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_vocabulary.py -k "injected or representable" -v
```

Expected: FAIL. `test_unrepresentable_model_is_injected_not_left_stale` reports `qwen/qwen3.7-plus` (the leftover) instead of `gpt-5.2`, and the harness errors on the unknown `querySelectorAll` call because the function does not make it yet.

- [ ] **Step 3: Rewrite `syncModelSelectFromAgent`**

Replace `app.js:1670-1677` entirely:

```js
/**
 * Point the picker at this agent's model.
 *
 * A model the curated list cannot represent (a legacy value like 'gpt-5.2' or
 * 'local-model') is INJECTED as its own option rather than left unmatched.
 * Leaving it unmatched is a silent-wrong-value bug, not a cosmetic one: on the
 * live iFinD path resolveBacktestModelRequest returns the select's current
 * value, so the run would submit whatever the previously-selected agent left
 * there, recorded under this agent's name. js/agent-editor.js does the same
 * thing for the Configure picker.
 */
function syncModelSelectFromAgent(agent) {
  const modelSelect = document.getElementById('modelSelect');
  if (!modelSelect || !agent?.model_name) return;
  // Drop the previous agent's injected option first, so injections cannot pile
  // up across agent switches and cannot be matched as if they were curated.
  modelSelect.querySelectorAll('option[data-injected-model]').forEach((option) => option.remove());
  const option = findBacktestModelOption(modelSelect, agent.model_name);
  if (option) {
    modelSelect.value = option.value;
    return;
  }
  const injected = document.createElement('option');
  injected.value = agent.model_name;
  injected.textContent = formatAgentModelLabel(agent.model_name);
  injected.dataset.injectedModel = 'true';
  modelSelect.appendChild(injected);
  modelSelect.value = agent.model_name;
}
```

- [ ] **Step 4: Update the stale comment in `resolveBacktestModelRequest`**

The guard stays — it is now belt-and-braces rather than the only protection — but its comment names a nine-option list that no longer exists. Replace `app.js:1646-1647`:

```js
  // Belt-and-braces since syncModelSelectFromAgent started injecting an option
  // for unrepresentable models: on the hidden path the agent's saved model wins
  // outright, whatever the select happens to hold.
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_vocabulary.py -v
```

Expected: PASS (8 cases).

- [ ] **Step 6: Verify the deliberate rule-based path still wins**

An explicit `Rule-based` selection on the iFinD path must survive — `resolveBacktestModelRequest` returns the selected value there, and that is correct behaviour, not a leftover. Confirm by reading `app.js:1640-1652` and checking that with `selectedModel === 'rule_based'`, `agentOption?.value === selectedModel` is false and `backtestModelPickerIsLiveControl()` is true, so the function falls through to `return selectedModel`. No code change; this step is a read-and-confirm.

- [ ] **Step 7: Commit**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/frontend/app.js dashboard/backend/tests/test_frontend_model_vocabulary.py
git commit -m "fix(backtest): inject an agent's unlisted model instead of submitting a stale one"
```

- [ ] **Step 8: Open PR 1**

```bash
git push -u origin fix/backtest-model-vocabulary
gh pr create --title "fix(backtest): one model vocabulary, no stale submissions" --body "$(cat <<'EOF'
The Run Backtest picker offered six models this platform cannot run and omitted four it can. Both model `<select>` elements now build from one `SUPPORTED_MODELS` list in `app.js`.

Also closes a silent-wrong-value path: on the iFinD A-share route (the only place the picker is live) an agent whose model matched no option submitted the *previously selected* agent's model. It is now injected as its own option.

Prerequisite for the Community model-vendor facets and the Qwen A-share template.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 2 — Backend: clone override + duplicate

Branch (cut from `main`, not from PR 1): `git switch main && git switch -c feat/agent-duplicate-and-clone-model`

### Task 3: `model_name` override on marketplace clone

**Files:**
- Modify: `dashboard/backend/api/routers/agents.py:239-240` (`CloneMarketplaceBody`), `:253-259` (the call)
- Modify: `dashboard/backend/domain/agents/service.py:385-409` (`clone_marketplace_template`)
- Test: `dashboard/backend/tests/test_agents_api.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `POST /api/v1/agents/marketplace/{template_id}/clone` accepts `{"model_name": str | null}`; `AgentService.clone_marketplace_template(..., model_name: Optional[str] = None)`.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/test_agents_api.py`:

```python
def test_clone_honours_a_model_name_override(client):
    """Community's "Choose model" affordance clones a template onto another model."""
    cloned = client.post(
        "/api/v1/agents/marketplace/balanced-starter/clone",
        json={"model_name": "deepseek/deepseek-v4-pro"},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert cloned.status_code == 200, cloned.text
    assert cloned.json()["agent"]["model_name"] == "deepseek/deepseek-v4-pro"


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_clone_falls_back_to_the_template_model(client, blank):
    """Omitted or blank means "use the template's model", not "use empty"."""
    body = {} if blank is None else {"model_name": blank}
    cloned = client.post(
        "/api/v1/agents/marketplace/balanced-starter/clone",
        json=body,
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert cloned.status_code == 200, cloned.text
    assert cloned.json()["agent"]["model_name"] == "anthropic/claude-haiku-4-5"


def test_clone_does_not_validate_the_model_name(client):
    """No whitelist here: POST /agents and PATCH /agents/{id} don't have one either,
    and a Literal would drag in the openapi enum deploy gate #313 discharged."""
    cloned = client.post(
        "/api/v1/agents/marketplace/balanced-starter/clone",
        json={"model_name": "some/unreleased-model"},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert cloned.status_code == 200, cloned.text
    assert cloned.json()["agent"]["model_name"] == "some/unreleased-model"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_agents_api.py -k "clone_honours or clone_falls_back or clone_does_not_validate" -v
```

Expected: FAIL — all three return `anthropic/claude-haiku-4-5`, because `model_name` is dropped as an unknown body field.

- [ ] **Step 3: Widen the request body**

`dashboard/backend/api/routers/agents.py:239-240`:

```python
class CloneMarketplaceBody(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    # Deliberately unvalidated beyond length -- see clone_marketplace_template.
    model_name: Optional[str] = Field(default=None, max_length=100)
```

- [ ] **Step 4: Thread it through the route**

`dashboard/backend/api/routers/agents.py`, in `clone_marketplace_agent`, add one argument to the `agent_service.clone_marketplace_template(...)` call:

```python
            name=body.name.strip() if body.name else None,
            model_name=body.model_name,
        )
```

- [ ] **Step 5: Apply the override in the service**

`dashboard/backend/domain/agents/service.py`, extend the signature at line 385:

```python
    def clone_marketplace_template(
        self,
        *,
        template_id: str,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
```

and replace the hardcoded model argument at line 409:

```python
        # No whitelist, on purpose: create_agent doesn't validate model_name
        # either, so rejecting here would be inconsistent, and a Literal would
        # publish an openapi enum -- the deploy gate #313 had to discharge.
        # Blank/omitted means "the template's own model".
        resolved_model = (model_name or "").strip() or str(
            template.get("model_name") or "local-model"
        ).strip() or "local-model"
        agent = self.create_agent(
            name=resolved_name,
            model_name=resolved_model,
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_agents_api.py -k clone -v
```

Expected: PASS, including the pre-existing clone cases.

- [ ] **Step 7: Commit**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/backend/api/routers/agents.py \
        dashboard/backend/domain/agents/service.py \
        dashboard/backend/tests/test_agents_api.py
git commit -m "feat(agents): let a marketplace clone pick its model"
```

---

### Task 4: `POST /api/v1/agents/{agent_id}/duplicate`

Backs Hook B ("Run on another model"). Server-side so the pipeline copy is atomic and ownership is checked on one path.

**Files:**
- Modify: `dashboard/backend/domain/agents/service.py` (add `duplicate_agent` after `clone_marketplace_template`, ~line 430)
- Modify: `dashboard/backend/api/routers/agents.py` (add body model beside `CloneMarketplaceBody`; add route after `rotate_agent_api_key`, ~line 628)
- Modify: `dashboard/backend/tests/test_router_move.py:34` (`EXPECTED_AGENT_ROUTES`), `dashboard/backend/tests/test_app_composition.py:56` (`EXPECTED_FULL_CONTRACT`)
- Test: `dashboard/backend/tests/test_agent_duplicate.py` (create)

**Interfaces:**
- Consumes: `_require_owner_context`, `_require_agent_access`, `portfolio_service.ensure_cash_for_new_agent`, `DEFAULT_AGENT_CASH_ALLOCATION`, `normalize_runtime_type`, `normalize_runtime_config`, `normalize_category`, `PIPELINE_RUNTIME_TYPE` — all already imported in their respective modules.
- Produces:
  - `AgentService.duplicate_agent(*, agent_id: str, model_name: str, name: Optional[str], owner_user_id: Optional[int], owner_browser_session: Optional[str]) -> Dict[str, Any]`
  - Route `POST /api/v1/agents/{agent_id}/duplicate`, function name `duplicate_agent`, body `{"model_name": str, "name": str | null}`, response `{"agent": {...}}`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_agent_duplicate.py`:

```python
"""POST /api/v1/agents/{agent_id}/duplicate -- the "Run on another model" hook.

Copies an existing built-in agent onto a different model, server-side so the
pipeline copy and the ownership check are one path. It deliberately does NOT
start a backtest: auto-firing would spend LLM credits on a click the user did
not frame as "run".
"""

import uuid

import pytest


def _create_builtin(client, headers, *, model="anthropic/claude-haiku-4-5"):
    created = client.post(
        "/api/v1/agents",
        json={
            "name": "Source Agent",
            "model_name": model,
            "agent_type": "builtin",
            "description": "The original.",
        },
        headers=headers,
    )
    assert created.status_code == 200, created.text
    return created.json()["agent"]


def test_duplicate_copies_the_agent_onto_a_new_model(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)

    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "deepseek/deepseek-v4-pro", "name": "Source Agent (DeepSeek)"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    copy = response.json()["agent"]
    assert copy["agent_id"] != source["agent_id"]
    assert copy["model_name"] == "deepseek/deepseek-v4-pro"
    assert copy["name"] == "Source Agent (DeepSeek)"
    assert copy["description"] == "The original."


def test_duplicate_copies_the_pipeline(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    pipeline = [
        {
            "id": "sub_custom",
            "presetKey": "simple_instruction",
            "label": "Trading instruction",
            "prompt": "Buy only what you would hold for a week.",
            "outputFormat": "JSON: { \"orders\": [] }",
        }
    ]
    patched = client.patch(
        f"/api/v1/agents/{source['agent_id']}",
        json={"pipeline": pipeline},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text

    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "qwen/qwen3.7-plus"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    copy_id = response.json()["agent"]["agent_id"]
    fetched = client.get(f"/api/v1/agents/{copy_id}", headers=headers).json()["agent"]
    assert [step["prompt"] for step in fetched["pipeline"]] == [
        "Buy only what you would hold for a week."
    ]


def test_duplicate_defaults_the_name(client):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["agent"]["name"] == "Source Agent copy"


def test_duplicate_rejects_another_owners_agent(client):
    owner = {"X-Session-Id": str(uuid.uuid4())}
    attacker = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, owner)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=attacker,
    )
    assert response.status_code in (403, 404), response.text


def test_duplicate_rejects_an_unknown_agent(client):
    response = client.post(
        f"/api/v1/agents/{uuid.uuid4()}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert response.status_code == 404, response.text


def test_duplicate_rejects_an_external_agent(client):
    """External agents mint an API key on create -- not a surface this hook opens."""
    headers = {"X-Session-Id": str(uuid.uuid4())}
    created = client.post(
        "/api/v1/agents",
        json={"name": "Connected", "model_name": "local-model", "agent_type": "external"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    response = client.post(
        f"/api/v1/agents/{created.json()['agent']['agent_id']}/duplicate",
        json={"model_name": "openai/gpt-5.5"},
        headers=headers,
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("body", [{}, {"model_name": ""}])
def test_duplicate_requires_a_model_name(client, body):
    headers = {"X-Session-Id": str(uuid.uuid4())}
    source = _create_builtin(client, headers)
    response = client.post(
        f"/api/v1/agents/{source['agent_id']}/duplicate", json=body, headers=headers
    )
    assert response.status_code == 422, response.text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_agent_duplicate.py -v
```

Expected: FAIL — every case returns 405 (Method Not Allowed) or 404, because the route does not exist.

- [ ] **Step 3: Add the service method**

In `dashboard/backend/domain/agents/service.py`, immediately after `clone_marketplace_template` (after line 429):

```python
    def duplicate_agent(
        self,
        *,
        agent_id: str,
        model_name: str,
        name: Optional[str] = None,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
    ) -> Dict[str, Any]:
        """Copy an existing built-in agent onto a different model.

        The same two steps as ``clone_marketplace_template`` -- create, then
        write the pipeline -- so the copy carries the strategy, not just the
        name. Built-in only: ``create_agent`` mints a one-time plaintext API key
        for external agents, and duplicating one would open a credential-issuing
        path this hook has no reason to have.

        The generated name collides freely (two copies onto DeepSeek both read
        ``X copy``). Names are not unique anywhere else in this product, and
        de-duplicating would mean a lookup for a cosmetic gain.
        """
        source = self.agents.get_agent(agent_id)
        if not source:
            raise AgentNotFoundError()
        if (source.get("agent_type") or "builtin") != "builtin":
            raise AgentServiceError("Only built-in agents can be duplicated")

        resolved_name = (name or f"{source.get('name') or 'Agent'} copy").strip()
        pipeline = source.get("pipeline")
        has_own_pipeline = isinstance(pipeline, list) and bool(pipeline)
        runtime_type = normalize_runtime_type(source.get("runtime_type"))
        runtime_config = normalize_runtime_config(
            runtime_type, source.get("runtime_config") or {}
        )
        agent = self.create_agent(
            name=resolved_name,
            model_name=(model_name or "").strip()
            or str(source.get("model_name") or "local-model"),
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type="builtin",
            description=source.get("description"),
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            seed_default_pipeline=(
                runtime_type == PIPELINE_RUNTIME_TYPE and not has_own_pipeline
            ),
            # Lenient, matching clone_marketplace_template: a stored legacy
            # category must stamp None rather than reject the duplicate.
            category=normalize_category(source.get("category")),
        )
        if has_own_pipeline:
            agent = self.agents.update_agent(agent["agent_id"], pipeline=pipeline) or agent
        return self.attach_equity_sparklines([self.agent_with_stats(agent)])[0]
```

- [ ] **Step 4: Add the request body model**

In `dashboard/backend/api/routers/agents.py`, immediately after `CloneMarketplaceBody`:

```python
class DuplicateAgentBody(BaseModel):
    """``model_name`` is required here (unlike clone): the whole point of the
    action is running the same strategy somewhere else."""

    model_name: str = Field(min_length=1, max_length=100)
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
```

- [ ] **Step 5: Add the route**

In `dashboard/backend/api/routers/agents.py`, after `rotate_agent_api_key` (~line 628):

```python
@router.post("/{agent_id}/duplicate")
def duplicate_agent(
    agent_id: str,
    body: DuplicateAgentBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Copy a built-in agent onto a different model ("Run on another model").

    Does not start a backtest: the caller lands on the new agent with Run primed.
    """
    ctx = _require_owner_context(request, authorization)
    _require_agent_access(agent_id, ctx, reclaim_on_session_match=True)
    cash = float(DEFAULT_AGENT_CASH_ALLOCATION)
    if ctx["user_id"] and cash > 0:
        try:
            portfolio_service.ensure_cash_for_new_agent(ctx["user_id"], cash)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        agent = agent_service.duplicate_agent(
            agent_id=agent_id,
            model_name=body.model_name,
            name=body.name.strip() if body.name else None,
            owner_user_id=ctx["user_id"],
            owner_browser_session=ctx["browser_session"],
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if ctx["user_id"]:
        portfolio_service.get_or_create_portfolio(ctx["user_id"])
    return {"agent": agent}
```

Confirm `AgentServiceError` and `AgentNotFoundError` are in the module's imports from `dashboard.backend.domain.agents.service`; add whichever is missing.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_agent_duplicate.py -v
```

Expected: PASS (9 cases).

- [ ] **Step 7: Update the route-contract golden sets**

A new route reddens these on **every open PR** until they are updated. Add to `dashboard/backend/tests/test_router_move.py`'s `EXPECTED_AGENT_ROUTES` (line 34), next to the `rotate-api-key` entry:

```python
    ("POST", "/v1/agents/{agent_id}/duplicate", "duplicate_agent"),
```

Add to `dashboard/backend/tests/test_app_composition.py`'s `EXPECTED_FULL_CONTRACT` (line 56), keeping the file's sorted-by-path grouping — immediately after the `("PATCH", "/api/v1/agents/{agent_id}")` entry:

```python
    ("POST", "/api/v1/agents/{agent_id}/duplicate"),
```

- [ ] **Step 8: Run the contract guards and confirm no third golden set exists**

```bash
pytest dashboard/backend/tests/test_router_move.py dashboard/backend/tests/test_app_composition.py -v
command grep -rn "rotate-api-key" dashboard/backend/tests/*.py
```

Expected: PASS. The grep confirms which files hold route inventories; if it names a file other than `test_router_move.py`, `test_app_composition.py`, `test_agents_api.py` or `test_object_authz.py`, that file holds a third golden set and needs the same entry.

- [ ] **Step 9: Run the full backend suite**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: PASS. A red test here is a real regression — this suite has been green since PR #71. If `test_deleted_shim_is_not_importable` fails with `DID NOT RAISE ModuleNotFoundError`, that is stale bytecode, not a regression: `rm -rf dashboard/backend/engines dashboard/backend/services`.

- [ ] **Step 10: Commit and open PR 2**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/backend/api/routers/agents.py \
        dashboard/backend/domain/agents/service.py \
        dashboard/backend/tests/test_agent_duplicate.py \
        dashboard/backend/tests/test_router_move.py \
        dashboard/backend/tests/test_app_composition.py
git commit -m "feat(agents): duplicate an agent onto another model"
git push -u origin feat/agent-duplicate-and-clone-model
gh pr create --title "feat(agents): clone/duplicate onto a chosen model" --body "$(cat <<'EOF'
Two backend additions behind the upcoming Community model facets:

- `POST /api/v1/agents/marketplace/{id}/clone` accepts `model_name` (blank/omitted = template default).
- `POST /api/v1/agents/{agent_id}/duplicate` copies a built-in agent — pipeline included — onto another model. Built-in only; external agents mint an API key on create. It does not start a backtest.

No `model_name` whitelist, matching `POST /agents` and `PATCH /agents/{id}`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 3 — Content: four new templates

Branch (cut from `main` **after PR 1 has merged**): `git switch main && git pull && git switch -c content/marketplace-vendor-spread`

### Task 5: Populate every live vendor chip

The catalog is six Claude templates and one Nemotron. A vendor facet over it would show two populated chips and five empty ones. Four new templates take it to 11, populate all six pickable vendors, and take `cn_ashares` from 1 to 2.

**Files:**
- Modify: `dashboard/config/marketplace.json`
- Test: `dashboard/backend/tests/test_marketplace_catalog_models.py` (create)

**Interfaces:**
- Consumes: `SUPPORTED_MODELS` (Task 1) — the guard reads it out of `app.js`.
- Produces: template ids `contrarian-dip-buyer`, `sector-rotator`, `volatility-guard`, `ashare-momentum-t1`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_marketplace_catalog_models.py`:

```python
"""Every catalog template must run on a model the platform actually offers.

A template on an unlisted model is invisible trouble: it clones fine, then the
Run Backtest picker cannot represent its model. The only exception is a hosted
runtime, whose model is a property of the runtime rather than a user choice.
"""

import json
import re
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import js_const

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/marketplace.json").read_text(
        encoding="utf-8"
    )
)["templates"]

_SUPPORTED_SLUGS = set(re.findall(r"slug:\s*'([^']+)'", js_const("SUPPORTED_MODELS")))

_EXPECTED_NEW = {
    "contrarian-dip-buyer": ("openai/gpt-5.5", "us_stocks"),
    "sector-rotator": ("google/gemini-3.1-pro-preview", "us_stocks"),
    "volatility-guard": ("deepseek/deepseek-v4-pro", "us_stocks"),
    "ashare-momentum-t1": ("qwen/qwen3.7-plus", "cn_ashares"),
}


@pytest.mark.parametrize("template", _CATALOG, ids=lambda t: t["template_id"])
def test_every_template_runs_a_supported_or_hosted_model(template):
    if template.get("runtime_type"):
        return  # hosted runtime: its model is not user-selectable
    assert template["model_name"] in _SUPPORTED_SLUGS, (
        f"{template['template_id']} runs {template['model_name']!r}, "
        "which is not in SUPPORTED_MODELS"
    )


@pytest.mark.parametrize("template_id,expected", sorted(_EXPECTED_NEW.items()))
def test_new_templates_are_present_with_their_pairings(template_id, expected):
    found = next((t for t in _CATALOG if t["template_id"] == template_id), None)
    assert found is not None, f"{template_id} missing from marketplace.json"
    assert (found["model_name"], found["category"]) == expected


def test_catalog_covers_every_pickable_vendor():
    """The facet is decorative if most of its chips are empty."""
    vendors = {t["model_name"].split("/", 1)[0] for t in _CATALOG}
    assert {"anthropic", "openai", "google", "deepseek", "qwen"} <= vendors
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest dashboard/backend/tests/test_marketplace_catalog_models.py -v
```

Expected: FAIL — the four `test_new_templates_are_present_with_their_pairings` cases report `missing from marketplace.json`, and `test_catalog_covers_every_pickable_vendor` fails on the missing `openai`/`google`/`deepseek`/`qwen`.

- [ ] **Step 3: Add the four templates**

Append these four objects to the `templates` array in `dashboard/config/marketplace.json`, after `ashare-steady-t1`. They follow the shipped single-step `pipeline` shape and the plain-language register of `balanced-starter`; `outputFormat` is byte-identical to the existing templates' — the engine parses it.

```json
    {
      "template_id": "contrarian-dip-buyer",
      "name": "Contrarian Dip Buyer",
      "model_name": "openai/gpt-5.5",
      "description": "Buys stocks that have sold off hard and trims them back once they have recovered. The opposite instinct to a momentum strategy.",
      "category": "us_stocks",
      "tags": [
        "contrarian",
        "mean reversion"
      ],
      "author": "Agentic Trading Lab",
      "pipeline": [
        {
          "id": "sub_contrarian_dip",
          "presetKey": "simple_instruction",
          "label": "Trading instruction",
          "prompt": "Look for stocks that have fallen well below where they were trading recently and buy those, in small pieces rather than all at once. Sell back into strength once a position has recovered. Do not chase stocks that are already running.",
          "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
        }
      ]
    },
    {
      "template_id": "sector-rotator",
      "name": "Sector Rotator",
      "model_name": "google/gemini-3.1-pro-preview",
      "description": "Concentrates into whichever part of the market is leading, and moves on when leadership changes.",
      "category": "us_stocks",
      "tags": [
        "rotation",
        "trend"
      ],
      "author": "Agentic Trading Lab",
      "pipeline": [
        {
          "id": "sub_sector_rotator",
          "presetKey": "simple_instruction",
          "label": "Trading instruction",
          "prompt": "Group the available stocks by the kind of business they are in. Put most of the money into the group that has been performing best, and hold two or three names from it rather than one. When a different group takes the lead, sell out of the old one before building the new position.",
          "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
        }
      ]
    },
    {
      "template_id": "volatility-guard",
      "name": "Volatility Guard",
      "model_name": "deepseek/deepseek-v4-pro",
      "description": "Holds a steady portfolio in calm markets and cuts exposure when prices start swinging. Runs on the only model that has beaten the passive baselines on our leaderboard.",
      "category": "us_stocks",
      "tags": [
        "risk management",
        "defensive"
      ],
      "author": "Agentic Trading Lab",
      "pipeline": [
        {
          "id": "sub_volatility_guard",
          "presetKey": "simple_instruction",
          "label": "Trading instruction",
          "prompt": "Judge how violently prices have been moving lately compared with earlier in the period. While things are calm, stay invested across several stocks. When the swings get noticeably larger, sell part of every position and hold the cash rather than switching stocks. Rebuild the positions gradually once the market settles down. Protecting the money matters more here than catching every rally.",
          "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
        }
      ]
    },
    {
      "template_id": "ashare-momentum-t1",
      "name": "A-Share Momentum (T+1)",
      "model_name": "qwen/qwen3.7-plus",
      "description": "Rides the strongest Chinese A-shares while respecting that market's rule that shares bought today cannot be sold until the next trading day.",
      "category": "cn_ashares",
      "tags": [
        "a-shares",
        "momentum"
      ],
      "author": "Agentic Trading Lab",
      "pipeline": [
        {
          "id": "sub_ashare_momentum",
          "presetKey": "simple_instruction",
          "label": "Trading instruction",
          "prompt": "Buy the Chinese A-shares that have been climbing most steadily and hold them while they keep leading. Shares bought today cannot be sold until the next trading day, so only buy what you are happy to still own tomorrow, and plan any exit at least a day ahead. Sell a position once it stops leading.",
          "outputFormat": "JSON: { \"orders\": [{ \"symbol\": \"...\", \"side\": \"buy|sell|hold\", \"qty\": number, \"order_type\": \"market|limit\", \"limit_price\": number|null, \"reason\": \"...\" }] }"
        }
      ]
    }
```

- [ ] **Step 4: Verify the JSON parses and the catalog loads**

```bash
python3 -c "import json; d=json.load(open('dashboard/config/marketplace.json')); print(len(d['templates']), 'templates')"
```

Expected: `11 templates`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_marketplace_catalog_models.py dashboard/backend/tests/test_agents_api.py -q
```

Expected: PASS. `test_agents_api.py` asserts `len(templates) >= 7` and that every `template_id` is unique and every `category` recognized — all four new ones satisfy that.

- [ ] **Step 6: Clone one new template end-to-end**

```bash
pytest dashboard/backend/tests/test_agents_api.py -k marketplace -q
python3 - <<'PY'
from fastapi.testclient import TestClient
import os, tempfile, uuid
os.environ["DATABASE_PATH"] = tempfile.mktemp(suffix=".db")
from dashboard.backend.app import app
with TestClient(app) as c:
    r = c.post("/api/v1/agents/marketplace/ashare-momentum-t1/clone",
               json={}, headers={"X-Session-Id": str(uuid.uuid4())})
    print(r.status_code, r.json()["agent"]["model_name"], r.json()["agent"]["category"])
PY
```

Expected: `200 qwen/qwen3.7-plus cn_ashares`. Setting `DATABASE_PATH` first is what keeps this probe off the committed seed DB.

- [ ] **Step 7: Commit and open PR 3**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/config/marketplace.json \
        dashboard/backend/tests/test_marketplace_catalog_models.py
git commit -m "feat(community): four templates spanning GPT, Gemini, DeepSeek and Qwen"
git push -u origin content/marketplace-vendor-spread
gh pr create --title "feat(community): spread the catalog across vendors" --body "$(cat <<'EOF'
Catalog goes 7 → 11 templates and stops being a Claude monoculture: one each on GPT-5.5, Gemini 3.1 Pro, DeepSeek V4 Pro and Qwen3.7 Plus. China A-Share goes 1 → 2.

Content only. A new guard pins every template's model to `SUPPORTED_MODELS` (hosted runtimes excepted), so a future template cannot land on a model the Run Backtest picker cannot represent.

Two pairings are deliberate: Qwen (Alibaba) on the thin A-share shelf, DeepSeek on the risk strategy because it is the only leaderboard LLM that beat the passive baselines.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

# PR 4 — Frontend: vendor taxonomy, facets, hooks

Branch (cut from `main` **after PRs 1 and 3 have merged**): `git switch main && git pull && git switch -c feat/community-model-facets`

**Merge gate:** PR 2's endpoints must be **serving in prod** before this merges. Run the `/openapi.json` probe from the Delivery order section. If it does not print `True` twice, open this PR as a **draft** with `DO NOT MERGE until PR 2 is serving in prod (probe /openapi.json)` as the first line of the body.

### Task 6: `MODEL_VENDORS` — vendor identity and licence

`MODEL_PROVIDER_LABELS` (`app.js:1902`) is a prefix table that only ever produced card submeta. It is promoted to the source of truth for vendor identity and licence, and `formatModelProviderLabel` becomes a derivation over it with **byte-identical output**.

**Files:**
- Modify: `dashboard/frontend/app.js` (replace `MODEL_PROVIDER_LABELS` at ~line 1902; add `modelVendorKey`, `agentVendorKey`, `modelVendorLicence`)
- Test: `dashboard/backend/tests/test_frontend_model_facets.py` (create)

**Interfaces:**
- Consumes: `SUPPORTED_MODELS` (Task 1).
- Produces:
  - `const MODEL_VENDORS` — array of `{ key, prefix, label, licence }`, eight entries in the Global Constraints order.
  - `function modelVendorKey(modelName) -> string` — vendor key, or `''` when unknown.
  - `function agentVendorKey(agent) -> string` — `modelVendorKey(agent?.model_name)`.
  - `function modelVendorLicence(modelName) -> string` — `'open'`, `'closed'`, or `''`.
  - `function formatModelProviderLabel(modelName) -> string` — unchanged output.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_frontend_model_facets.py`:

```python
"""Guards for the Community model-vendor facet.

The vendor axis is a pure derivation from `model_name` -- no column, no
migration. MODEL_VENDORS is its single source of truth: chip order, display
label and open/closed licence all come from one table, so a badge cannot drift
from the vendor it describes. A wrong badge is a factual claim about someone
else's product.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, js_const

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/marketplace.json").read_text(
        encoding="utf-8"
    )
)["templates"]

EXPECTED_VENDORS = [
    ("anthropic", "anthropic/", "Claude", "closed"),
    ("openai", "openai/", "GPT", "closed"),
    ("google", "google/", "Gemini", "closed"),
    ("deepseek", "deepseek/", "DeepSeek", "open"),
    ("qwen", "qwen/", "Qwen", "open"),
    ("nvidia", "nvidia/nemotron", "NVIDIA Nemotron", "open"),
    ("meta", "meta-llama/", "Llama", "open"),
    ("xai", "x-ai/", "Grok", "closed"),
]


def _vendor_rows():
    return re.findall(
        r"key:\s*'([^']+)',\s*prefix:\s*'([^']+)',\s*label:\s*'([^']+)',\s*licence:\s*'([^']+)'",
        js_const("MODEL_VENDORS"),
    )


def test_vendor_table_is_pinned_including_licence():
    assert _vendor_rows() == EXPECTED_VENDORS


def test_every_catalog_model_matches_a_vendor_prefix():
    """The highest-value guard: a template on an unmatched prefix renders as
    "AI-powered" with no chip and no badge, which is otherwise invisible."""
    prefixes = [row[1] for row in _vendor_rows()]
    for template in _CATALOG:
        model = template["model_name"].lower()
        assert any(model.startswith(p) for p in prefixes), (
            f"{template['template_id']} runs {model!r}, which matches no MODEL_VENDORS prefix"
        )


def test_every_supported_model_matches_a_vendor_prefix():
    prefixes = [row[1] for row in _vendor_rows()]
    for slug in re.findall(r"slug:\s*'([^']+)'", js_const("SUPPORTED_MODELS")):
        assert any(slug.lower().startswith(p) for p in prefixes), slug


def test_supported_model_vendor_fields_agree_with_the_vendor_table():
    """SUPPORTED_MODELS carries its own `vendor` key; it must not drift."""
    by_prefix = {row[1]: row[0] for row in _vendor_rows()}
    pairs = re.findall(
        r"slug:\s*'([^']+)',\s*label:\s*'[^']+',\s*vendor:\s*'([^']+)'",
        js_const("SUPPORTED_MODELS"),
    )
    for slug, vendor in pairs:
        expected = next(k for p, k in by_prefix.items() if slug.lower().startswith(p))
        assert vendor == expected, f"{slug} is tagged {vendor!r}, table says {expected!r}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_provider_label_output_is_unchanged():
    """These six strings ship on cards today. The refactor must not touch them."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function formatModelProviderLabel")}
const cases = ['anthropic/claude-haiku-4-5', 'nvidia/nemotron-3-nano-30b-a3b',
               'deepseek/deepseek-v4-pro', 'openai/gpt-5.5',
               'google/gemini-3.1-pro-preview', 'qwen/qwen3.7-plus',
               'totally/unknown', ''];
console.log(JSON.stringify(cases.map(formatModelProviderLabel)));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "Powered by Claude",
        "Powered by NVIDIA Nemotron",
        "Powered by DeepSeek",
        "Powered by GPT",
        "Powered by Gemini",
        "Powered by Qwen",
        "AI-powered",
        "AI-powered",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unknown_vendor_resolves_to_empty_string():
    """Same contract as agentMarketKey: unknown stays visible under All and is
    excluded only by an explicit chip -- never hidden, never defaulted."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function agentVendorKey")}
{fn_body("function modelVendorLicence")}
console.log(JSON.stringify([
  modelVendorKey('totally/unknown'), modelVendorKey(null), modelVendorKey('local-model'),
  agentVendorKey({{model_name: 'qwen/qwen3.7-plus'}}), agentVendorKey(null),
  modelVendorLicence('deepseek/deepseek-v4-pro'),
  modelVendorLicence('anthropic/claude-haiku-4-5'),
  modelVendorLicence('totally/unknown'),
]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["", "", "", "qwen", "", "open", "closed", ""]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: FAIL with `AssertionError: MODEL_VENDORS not found in app.js` from `js_const`.

- [ ] **Step 3: Replace the label table with the vendor table**

Replace `app.js:1897-1920` (the `MODEL_PROVIDER_LABELS` doc comment, the table, and `formatModelProviderLabel`) with:

```js
/** The model-vendor axis: who makes a model, and how it is licensed.
 *
 * Promoted from a submeta label lookup into the source of truth for the whole
 * axis -- Community's vendor chips, the open-source badge and the card submeta
 * all derive from this one table, so a badge cannot drift from the vendor it
 * describes. A wrong badge is a factual claim about someone else's product.
 *
 * Matched by PREFIX, not exact slug, so a new model version under a known
 * vendor needs no entry here. Declaration order is chip order, mirroring how
 * MARKET_LABELS' key order mirrors the AgentCategory Literal.
 *
 * All eight are listed even though only six are pickable: a card whose model
 * matches nothing renders as the generic "AI-powered" with no chip and no
 * badge, which is invisible until someone notices. The chip ROW is still
 * derived from what the loaded catalog actually contains (see
 * renderMarketplaceVendorChips), so listing a vendor here never ships an
 * empty chip. */
const MODEL_VENDORS = [
  { key: 'anthropic', prefix: 'anthropic/', label: 'Claude', licence: 'closed' },
  { key: 'openai', prefix: 'openai/', label: 'GPT', licence: 'closed' },
  { key: 'google', prefix: 'google/', label: 'Gemini', licence: 'closed' },
  { key: 'deepseek', prefix: 'deepseek/', label: 'DeepSeek', licence: 'open' },
  { key: 'qwen', prefix: 'qwen/', label: 'Qwen', licence: 'open' },
  // "NVIDIA Nemotron", not "Nemotron": this label also feeds
  // formatModelProviderLabel, whose shipped output must not change.
  { key: 'nvidia', prefix: 'nvidia/nemotron', label: 'NVIDIA Nemotron', licence: 'open' },
  { key: 'meta', prefix: 'meta-llama/', label: 'Llama', licence: 'open' },
  { key: 'xai', prefix: 'x-ai/', label: 'Grok', licence: 'closed' },
];

/** Vendor key for a model slug, or '' when the platform genuinely doesn't know.
 *
 * '' is not a bug and must never hide the template: it stays visible under the
 * All chip and is excluded only by an explicit vendor chip -- the same contract
 * agentMarketKey documents for markets. */
function modelVendorKey(modelName) {
  const raw = String(modelName || '').trim().toLowerCase();
  if (!raw) return '';
  return (MODEL_VENDORS.find((vendor) => raw.startsWith(vendor.prefix)) || {}).key || '';
}

/** modelVendorKey for an agent record. The agent-facing twin of agentMarketKey. */
function agentVendorKey(agent) {
  return modelVendorKey(agent?.model_name);
}

/** 'open' | 'closed' | '' -- '' when the vendor is unknown. */
function modelVendorLicence(modelName) {
  const key = modelVendorKey(modelName);
  return (MODEL_VENDORS.find((vendor) => vendor.key === key) || {}).licence || '';
}

function formatModelProviderLabel(modelName) {
  const key = modelVendorKey(modelName);
  const vendor = MODEL_VENDORS.find((entry) => entry.key === key);
  return vendor ? `Powered by ${vendor.label}` : 'AI-powered';
}
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: PASS (7 cases).

- [ ] **Step 5: Commit**

```bash
git status --short
git add dashboard/frontend/app.js dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "refactor(community): make the vendor table the source of truth for the model axis"
```

---

### Task 7: The vendor chip row and the three empty states

Community gets a second chip row beneath the market row; the two filters AND together. The one deliberate asymmetry: **vendor chips render only for vendors present in the loaded catalog** — markets are a closed backend-validated enum, vendors are open-ended, and hardcoding all eight would ship permanently-empty chips.

**Files:**
- Modify: `dashboard/frontend/app.html:1616` (add the vendor chip container), `app.html:1618` (empty-state element)
- Modify: `dashboard/frontend/app.js` (add `marketplaceVendorFilter`, `setMarketplaceVendorFilter`, `renderMarketplaceVendorChips`, `marketplaceEmptyHtml`; extend `getFilteredMarketplaceTemplates` and `renderMarketplaceGrid`)
- Modify: `dashboard/frontend/styles.css` (vendor-row spacing)
- Test: `dashboard/backend/tests/test_frontend_model_facets.py` (extend)

**Interfaces:**
- Consumes: `MODEL_VENDORS`, `modelVendorKey` (Task 6); existing `marketplaceCategoryFilter`, `renderMarketplaceCategoryChips`, `escapeHtml`.
- Produces:
  - `let marketplaceVendorFilter` — `'all'` or a `MODEL_VENDORS` key.
  - `function setMarketplaceVendorFilter(vendorKey) -> void`
  - `function renderMarketplaceVendorChips() -> void`
  - `function marketplaceEmptyHtml({ searching, categoryFilter, vendorFilter }) -> string`

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_model_facets.py`:

```python
def test_vendor_chip_container_exists_in_the_community_view():
    community = APP_HTML[
        APP_HTML.index('<div id="communityView"') : APP_HTML.index('<div id="accountView"')
    ]
    assert 'id="marketplaceVendorChips"' in community
    assert 'id="marketplaceCategoryChips"' in community
    assert community.index('id="marketplaceCategoryChips"') < community.index(
        'id="marketplaceVendorChips"'
    ), "market row must render above the vendor row"


def test_vendor_chips_are_derived_not_hardcoded():
    """Chips come from MODEL_VENDORS intersected with the loaded catalog, so a
    vendor with no templates never ships an empty chip."""
    body = fn_body("function renderMarketplaceVendorChips")
    assert "MODEL_VENDORS" in body
    assert "marketplaceTemplates" in body
    for literal in ("'anthropic'", "'openai'", "'deepseek'", "'qwen'"):
        assert literal not in body, f"{literal} hardcoded in the chip builder"


def test_vendor_chips_are_built_once_then_toggled():
    """renderMarketplaceGrid runs on every search keystroke; rebuilding innerHTML
    per keystroke would blow away the focused chip."""
    body = fn_body("function renderMarketplaceVendorChips")
    assert "existing.length !== chips.length" in body


def test_three_empty_states_stay_distinguishable():
    body = fn_body("function marketplaceEmptyHtml")
    assert "No templates match your search." in body
    assert "No templates match both filters" in body
    assert "marketplace-clear-filters" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_empty_state_precedence():
    script = f"""
function escapeHtml(s) {{ return String(s); }}
const MARKET_LABELS = {{ us_stocks: 'U.S.', cn_ashares: 'China A-Share' }};
{js_const("MODEL_VENDORS")}
{fn_body("function marketplaceEmptyHtml")}
const out = [
  marketplaceEmptyHtml({{searching: true, categoryFilter: 'us_stocks', vendorFilter: 'qwen'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'us_stocks', vendorFilter: 'qwen'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'us_stocks', vendorFilter: 'all'}}),
  marketplaceEmptyHtml({{searching: false, categoryFilter: 'all', vendorFilter: 'all'}}),
];
console.log(JSON.stringify(out));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    search_empty, both, one_chip, none_at_all = json.loads(result.stdout)
    # A typed query wins: clearing the chips would not bring anything back.
    assert search_empty == "No templates match your search."
    assert "both filters" in both and "marketplace-clear-filters" in both
    assert "U.S." in one_chip and "both filters" not in one_chip
    assert none_at_all == "No templates match your search."


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unknown_vendor_survives_the_all_chip_and_only_that_chip():
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
const templates = [
  {{template_id: 'a', model_name: 'qwen/qwen3.7-plus'}},
  {{template_id: 'b', model_name: 'totally/unknown'}},
];
function visible(filter) {{
  return templates
    .filter((t) => filter === 'all' || modelVendorKey(t.model_name) === filter)
    .map((t) => t.template_id);
}}
console.log(JSON.stringify([visible('all'), visible('qwen')]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["a", "b"], ["a"]]
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -k "vendor_chip or empty" -v
```

Expected: FAIL — `marketplaceVendorChips` is absent from `app.html`, and `fn_body` raises `ValueError: substring not found` for the two functions that do not exist yet.

- [ ] **Step 3: Add the vendor chip container to `app.html`**

Replace `dashboard/frontend/app.html:1616` with:

```html
            <div id="marketplaceCategoryChips" class="marketplace-category-chips" role="group" aria-label="Filter templates by market"></div>
            <div id="marketplaceVendorChips" class="marketplace-category-chips marketplace-vendor-chips" role="group" aria-label="Filter templates by AI model"></div>
```

and replace line 1618's empty-state element (its copy is now written by `marketplaceEmptyHtml`):

```html
            <p id="marketplaceEmptyState" class="control-helper" hidden></p>
```

- [ ] **Step 4: Add the filter state and the chip builder to `app.js`**

Immediately after `marketplaceCategoryFilter`'s declaration (~line 1895):

```js
/** 'all' or one of MODEL_VENDORS' keys. ANDs with marketplaceCategoryFilter. */
let marketplaceVendorFilter = 'all';
```

After `setMarketplaceCategoryFilter` (~line 1935):

```js
/** Select a vendor chip and re-render. Mirrors setMarketplaceCategoryFilter,
 * including the reset-to-'all' fallback for an unrecognized key. */
function setMarketplaceVendorFilter(vendorKey) {
  marketplaceVendorFilter = MODEL_VENDORS.some((vendor) => vendor.key === vendorKey)
    ? vendorKey
    : 'all';
  renderMarketplaceGrid();
}
```

After `renderMarketplaceCategoryChips` (~line 1963):

```js
/** Second chip row: 'All' plus one chip per vendor PRESENT IN THE CATALOG.
 *
 * Deliberately asymmetric with the market row, which is hardcoded from
 * MARKET_LABELS: markets are a closed, backend-validated enum, vendors are
 * open-ended. Hardcoding all of MODEL_VENDORS would ship chips that can never
 * match anything. Order still comes from MODEL_VENDORS, not from catalog order,
 * so the row does not reshuffle when a template is added. */
function renderMarketplaceVendorChips() {
  const container = document.getElementById('marketplaceVendorChips');
  if (!container) return;
  const present = new Set(marketplaceTemplates.map((t) => modelVendorKey(t.model_name)));
  const chips = [
    { key: 'all', label: 'All models' },
    ...MODEL_VENDORS.filter((vendor) => present.has(vendor.key)).map((vendor) => ({
      key: vendor.key,
      label: vendor.label,
    })),
  ];
  // Build once, then only toggle state -- same reason as the market row: this
  // runs from renderMarketplaceGrid, which is bound to the search box's `input`.
  const existing = container.querySelectorAll('[data-marketplace-vendor]');
  if (existing.length !== chips.length) {
    container.innerHTML = chips
      .map((chip) => `<button type="button" class="marketplace-category-chip" data-marketplace-vendor="${escapeHtml(chip.key)}" aria-pressed="false">${escapeHtml(chip.label)}</button>`)
      .join('');
  }
  container.querySelectorAll('[data-marketplace-vendor]').forEach((button) => {
    const active = button.dataset.marketplaceVendor === marketplaceVendorFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

/** Empty-state copy. Three cases, deliberately worded apart -- the same concern
 * stocksEmptyHtml records for My Agents.
 *
 * A typed query wins over the facet case: when a search is what emptied the
 * grid, offering "Clear filters" sends the user to fix the wrong thing. */
function marketplaceEmptyHtml({ searching, categoryFilter, vendorFilter }) {
  if (searching) return 'No templates match your search.';
  if (categoryFilter !== 'all' && vendorFilter !== 'all') {
    return `No templates match both filters. <button type="button" class="marketplace-clear-filters">Clear filters</button>`;
  }
  if (categoryFilter !== 'all') {
    return `No ${escapeHtml(MARKET_LABELS[categoryFilter] || '')} templates yet.`;
  }
  if (vendorFilter !== 'all') {
    const vendor = MODEL_VENDORS.find((entry) => entry.key === vendorFilter);
    return `No ${escapeHtml(vendor?.label || '')} templates yet.`;
  }
  return 'No templates match your search.';
}
```

- [ ] **Step 5: AND the vendor filter into the template list**

In `getFilteredMarketplaceTemplates` (~line 1970), immediately after the category filter block:

```js
  if (marketplaceVendorFilter !== 'all') {
    list = list.filter((template) => modelVendorKey(template.model_name) === marketplaceVendorFilter);
  }
```

- [ ] **Step 6: Wire the row and the empty state into `renderMarketplaceGrid`**

In `renderMarketplaceGrid`, replace `renderMarketplaceCategoryChips();` (~line 1996) with:

```js
  renderMarketplaceCategoryChips();
  renderMarketplaceVendorChips();
```

and replace the empty-state block (~lines 2001-2007) with:

```js
  if (!templates.length) {
    // Keep it hidden before the first load, so it doesn't flash while
    // marketplaceTemplates is still empty.
    if (emptyEl) {
      emptyEl.hidden = marketplaceTemplates.length === 0;
      emptyEl.innerHTML = marketplaceEmptyHtml({
        searching: Boolean((document.getElementById('marketplaceSearchInput')?.value || '').trim()),
        categoryFilter: marketplaceCategoryFilter,
        vendorFilter: marketplaceVendorFilter,
      });
      emptyEl.querySelector('.marketplace-clear-filters')?.addEventListener('click', () => {
        marketplaceCategoryFilter = 'all';
        marketplaceVendorFilter = 'all';
        renderMarketplaceGrid();
      });
    }
    return;
  }
```

- [ ] **Step 7: Bind the chip row's clicks**

The market row is bound as one delegated listener on its container, inside `initNavigation` at `app.js:7187`. Add the sibling binding immediately after that block, so the two rows are wired identically:

```js
    document.getElementById('marketplaceVendorChips')?.addEventListener('click', (event) => {
        const chip = event.target.closest('[data-marketplace-vendor]');
        if (!chip) return;
        setMarketplaceVendorFilter(chip.dataset.marketplaceVendor);
    });
```

Delegation matters here: `renderMarketplaceVendorChips` replaces the container's `innerHTML` when the chip count changes, which would discard per-button listeners.

- [ ] **Step 8: Add the vendor-row spacing rule to `styles.css`**

The two rows share `.marketplace-category-chip`, so only the row's margin needs a rule. Insert immediately after the `.marketplace-category-chips` block (~line 9632):

```css
/* Second facet row (model vendor). Pulled up under the market row so the two
   read as one stacked filter block rather than two unrelated controls. */
.marketplace-vendor-chips {
    margin-top: -8px;
}
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: PASS (13 cases).

- [ ] **Step 10: Commit**

```bash
git status --short
git add dashboard/frontend/app.js dashboard/frontend/app.html dashboard/frontend/styles.css \
        dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "feat(community): stack a model-vendor facet under the market chips"
```

---

### Task 8: The open-source badge

**Files:**
- Modify: `dashboard/frontend/app.js` (`renderMarketplaceGrid`'s card template, ~lines 2018 and 2055)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_frontend_model_facets.py` (extend)

**Interfaces:**
- Consumes: `modelVendorLicence` (Task 6).
- Produces: `<span class="marketplace-licence-badge">Open-source model</span>` on open-weight cards.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_model_facets.py`:

```python
def test_only_open_weight_models_get_a_badge():
    """Closed models get NOTHING. A "Closed" label reads as a warning about
    someone else's product; absence is not a negative claim."""
    grid = fn_body("function renderMarketplaceGrid")
    assert "modelVendorLicence" in grid
    assert "Open-source model" in grid
    assert "Closed-source" not in APP_JS
    assert "Proprietary" not in APP_JS


def test_licence_badge_has_a_style_rule():
    from dashboard.backend.tests._frontend_source import css_blocks

    assert css_blocks(".marketplace-licence-badge"), "badge has no styles.css rule"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -k "badge or open_weight" -v
```

Expected: FAIL — `modelVendorLicence` is not referenced in `renderMarketplaceGrid`, and `css_blocks` returns `[]`.

- [ ] **Step 3: Compute the badge in the card loop**

In `renderMarketplaceGrid`, immediately after `const modelLabel = formatModelProviderLabel(template.model_name);` (~line 2018):

```js
    // Open weights get a badge; closed models get nothing. Licence comes from
    // MODEL_VENDORS so it cannot drift from the vendor it describes.
    const licenceBadge = modelVendorLicence(template.model_name) === 'open'
      ? '<span class="marketplace-licence-badge">Open-source model</span>'
      : '';
```

- [ ] **Step 4: Render it in the tag row**

Replace the tag-row line (~line 2055):

```js
        ${(licenceBadge || tags) ? `<div class="marketplace-tag-row">${licenceBadge}${tags}</div>` : ''}
```

- [ ] **Step 5: Style it**

Append to `dashboard/frontend/styles.css`, immediately after the `.marketplace-vendor-chips` rule from Task 7:

```css
/* Open-weight marker. Sits in the tag row so it wraps with the tags rather
   than competing with the mode chip for the card's top-right corner. */
.marketplace-licence-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--success-color);
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
```

- [ ] **Step 6: Confirm `--success-color` exists**

```bash
command grep -n -- "--success-color" dashboard/frontend/styles.css | head -3
```

Expected: at least one `:root` definition. If it is absent, use `var(--info-color)` (used by the active chip rule) instead.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: PASS (15 cases).

- [ ] **Step 8: Commit**

```bash
git status --short
git add dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "feat(community): badge open-weight models on template cards"
```

---

### Task 9: Hook A — choose the model when cloning

A **split button**: `Add to My Agents` keeps its exact current one-click behaviour on the template's default model, and a secondary `Choose model ▾` sits beside it. The primary CTA is the conversion click; putting a picker in front of it taxes precisely the interaction we most want to succeed. Model is the **only** thing the menu changes — no rename, no capital, no pipeline edits.

**Files:**
- Modify: `dashboard/frontend/app.js` (`renderMarketplaceGrid`'s actions block ~line 2057, its click bindings ~line 2063, and `cloneMarketplaceTemplate` ~line 2127)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_frontend_model_facets.py` (extend)

**Interfaces:**
- Consumes: `SUPPORTED_MODELS` (Task 1); `POST .../clone` with `model_name` (Task 3).
- Produces: `cloneMarketplaceTemplate(template, modelName)` — second parameter optional; omitted means the template's default.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_model_facets.py`:

```python
def test_primary_clone_cta_is_unchanged():
    """The conversion click keeps its label and its one-click behaviour."""
    grid = fn_body("function renderMarketplaceGrid")
    assert "const cloneLabel = 'Add to My Agents';" in grid
    assert "marketplace-clone-btn" in grid


def test_model_choice_is_a_secondary_affordance():
    grid = fn_body("function renderMarketplaceGrid")
    assert "marketplace-clone-model-btn" in grid
    assert "Choose model" in grid
    assert "SUPPORTED_MODELS" in grid


def test_clone_sends_the_chosen_model():
    body = fn_body("async function cloneMarketplaceTemplate")
    assert "model_name" in body


def test_clone_menu_changes_only_the_model():
    """A second half-Configure inside a clone menu is how two editing surfaces
    start drifting apart. Name, capital and pipeline stay in Configure."""
    grid = fn_body("function renderMarketplaceGrid")
    menu_start = grid.index("marketplace-model-menu")
    menu = grid[menu_start : menu_start + 800]
    for forbidden in ("cash_allocation", "backtest_allocation", "pipeline", "rename"):
        assert forbidden not in menu
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -k "clone or model_choice" -v
```

Expected: FAIL on `marketplace-clone-model-btn` and on `model_name` in `cloneMarketplaceTemplate`.

- [ ] **Step 3: Render the split button**

Replace the card's actions block (~lines 2057-2059):

```js
      <div class="agent-card-actions agent-card-actions--status">
        <div class="marketplace-clone-split">
          <button class="agent-card-cta marketplace-clone-btn" type="button" data-template-id="${escapeHtml(template.template_id)}">${cloneLabel}</button>
          <button class="agent-card-cta marketplace-clone-model-btn" type="button" data-template-id="${escapeHtml(template.template_id)}" aria-haspopup="true" aria-expanded="false" aria-label="Add on a different model">Choose model ▾</button>
          <div class="marketplace-model-menu" hidden>
            ${SUPPORTED_MODELS.map((model) => `<button type="button" class="agent-menu-item marketplace-model-option" data-template-id="${escapeHtml(template.template_id)}" data-model-slug="${escapeHtml(model.slug)}"${normalizeBacktestModelId(model.slug) === normalizeBacktestModelId(template.model_name) ? ' aria-current="true"' : ''}>${escapeHtml(model.label)}</button>`).join('')}
          </div>
        </div>
      </div>`;
```

- [ ] **Step 4: Bind the menu**

Immediately after the existing `.marketplace-clone-btn` binding block (~line 2082), add:

```js
  grid.querySelectorAll('.marketplace-clone-model-btn').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      const menu = btn.parentElement?.querySelector('.marketplace-model-menu');
      if (!menu) return;
      const opening = menu.hidden;
      // Close every other card's menu first: two open menus overlap.
      grid.querySelectorAll('.marketplace-model-menu').forEach((el) => { el.hidden = true; });
      grid.querySelectorAll('.marketplace-clone-model-btn').forEach((el) => el.setAttribute('aria-expanded', 'false'));
      menu.hidden = !opening;
      btn.setAttribute('aria-expanded', String(opening));
    });
  });

  grid.querySelectorAll('.marketplace-model-option').forEach((option) => {
    option.addEventListener('click', async () => {
      const template = marketplaceTemplates.find((item) => item.template_id === option.dataset.templateId);
      if (!template || marketplaceCloneInFlight) return;
      marketplaceCloneInFlight = true;
      option.disabled = true;
      try {
        await cloneMarketplaceTemplate(template, option.dataset.modelSlug);
      } catch (error) {
        alert(error.message || `Couldn't add this template. Please try again.`);
      } finally {
        marketplaceCloneInFlight = false;
        option.disabled = false;
      }
    });
  });
```

- [ ] **Step 5: Send the chosen model**

Replace `cloneMarketplaceTemplate` (~line 2127):

```js
/** `modelName` omitted means the template's own model -- the primary CTA's
 * path, whose behaviour is deliberately unchanged. */
async function cloneMarketplaceTemplate(template, modelName) {
  const data = await API.post(
    `${API_BASE}/api/v1/agents/marketplace/${encodeURIComponent(template.template_id)}/clone`,
    modelName ? { model_name: modelName } : {},
  );
  const agent = data?.agent;
  if (!agent?.agent_id) {
    throw new Error('Add failed — no agent returned');
  }
  applyActiveAgent(agent);
  await loadAgents();
  switchPlaygroundTab('agents');
  if (window.AgentEditor) {
    window.AgentEditor.open(agent);
  }
}
```

- [ ] **Step 6: Style the split button**

Append to `dashboard/frontend/styles.css`, after the licence-badge rule:

```css
/* Split CTA: the primary Add button keeps its weight; the model picker is a
   quieter sibling so it cannot compete with the conversion click. */
.marketplace-clone-split {
    position: relative;
    display: flex;
    gap: 6px;
    width: 100%;
}

.marketplace-clone-split .marketplace-clone-btn {
    flex: 1 1 auto;
}

.marketplace-clone-model-btn {
    flex: 0 0 auto;
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

.marketplace-clone-model-btn:hover {
    color: var(--text-primary);
    border-color: var(--info-color);
}

.marketplace-model-menu {
    position: absolute;
    right: 0;
    bottom: calc(100% + 6px);
    z-index: 20;
    min-width: 200px;
    padding: 6px;
    border-radius: 10px;
    background: var(--bg-elevated, #1e293b);
    border: 1px solid var(--border-color);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
}
```

- [ ] **Step 7: Confirm the elevated-background variable**

```bash
command grep -n -- "--bg-elevated" dashboard/frontend/styles.css | head -3
```

The rule already falls back to `#1e293b`, so no edit is required either way; this step just records which one is in play. If `--bg-elevated` is undefined, check that the existing `.agent-menu-dropdown` rule uses a different variable and match it instead.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: PASS (19 cases).

- [ ] **Step 9: Commit**

```bash
git status --short
git add dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "feat(community): add a template on a model of your choice"
```

---

### Task 10: Hook B — "Run on another model"

On a My Agents card whose status is `BACKTESTED` or `PAPER TRADING` — the user has already demonstrated intent — offer a menu action that duplicates the agent onto a different model. This is the highest-yield hook: it turns one agent into three and one run into three, among users who have already succeeded once. **It does not auto-launch a backtest.**

**Files:**
- Modify: `dashboard/frontend/app.html` (add `#duplicateAgentModal` beside `#createBuiltinAgentModal`, ~line 783)
- Modify: `dashboard/frontend/app.js` (`renderAgentCardActions` ~line 1044; card bindings ~line 1358; new `openDuplicateAgentModal`/`submitDuplicateAgent`; modal wiring in `DOMContentLoaded`)
- Test: `dashboard/backend/tests/test_frontend_model_facets.py` (extend)

**Interfaces:**
- Consumes: `SUPPORTED_MODELS`, `modelVendorKey`, `MODEL_VENDORS` (Tasks 1, 6); `POST /api/v1/agents/{id}/duplicate` (Task 4); existing `showAppToast`, `loadAgents`, `applyActiveAgent`, `highlightAgentCard` (`app.js:3909`).
- Produces:
  - `function duplicateModelChoices(agent) -> Array<{slug, label}>` — `SUPPORTED_MODELS` minus the agent's current model.
  - `function duplicateAgentName(agent, modelSlug) -> string` — `` `${agent.name} (${vendorLabel})` ``.
  - `function openDuplicateAgentModal(agent) -> void`

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_frontend_model_facets.py`:

```python
def test_duplicate_action_only_on_agents_that_have_run():
    """"Run on another model" is a follow-on offer, not a first action."""
    body = fn_body("function renderAgentCardActions")
    assert "agent-duplicate-model-btn" in body
    assert "'backtested'" in body and "'paper'" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_offers_every_model_except_the_current_one():
    script = f"""
{js_const("SUPPORTED_MODELS")}
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function duplicateModelChoices")}
console.log(JSON.stringify([
  duplicateModelChoices({{model_name: 'qwen/qwen3.7-plus'}}).map((m) => m.slug),
  duplicateModelChoices({{model_name: 'local-model'}}).map((m) => m.slug).length,
]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    without_qwen, legacy_count = json.loads(result.stdout)
    assert "qwen/qwen3.7-plus" not in without_qwen
    assert len(without_qwen) == 5
    # A legacy/hosted model isn't in the list, so nothing is filtered out.
    assert legacy_count == 6


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_duplicate_name_uses_the_vendor_label():
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function duplicateAgentName")}
console.log(duplicateAgentName({{name: 'Momentum Alpha'}}, 'deepseek/deepseek-v4-pro'));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Momentum Alpha (DeepSeek)"


def test_duplicate_does_not_start_a_backtest():
    """Auto-firing spends LLM credits on a click the user did not frame as run."""
    body = fn_body("async function submitDuplicateAgent")
    for forbidden in ("runBacktest(", "openRunBacktestModal("):
        assert forbidden not in body
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -k duplicate -v
```

Expected: FAIL — `agent-duplicate-model-btn` is absent and `fn_body` raises for the three new functions.

- [ ] **Step 3: Add the modal markup**

Insert into `dashboard/frontend/app.html` immediately after the `#createBuiltinAgentModal` block closes (~line 783):

```html
    <div id="duplicateAgentModal" class="auth-modal" hidden>
        <div class="auth-modal-backdrop" id="duplicateAgentModalBackdrop"></div>
        <div class="auth-modal-panel" role="dialog" aria-labelledby="duplicateAgentModalTitle">
            <button id="duplicateAgentModalClose" class="auth-modal-close" type="button" aria-label="Close">×</button>
            <h2 id="duplicateAgentModalTitle" class="auth-modal-title">Run on another model</h2>
            <p class="auth-modal-subtitle">Makes a copy of this agent — same strategy, same instructions — running on a different AI model, so you can compare them side by side. Nothing runs until you press Run Backtest.</p>
            <form id="duplicateAgentForm" class="auth-form">
                <label class="auth-field">
                    <span>Model</span>
                    <!-- Options are written by openDuplicateAgentModal(): the list
                         excludes the agent's own model, so it cannot be built here. -->
                    <select id="duplicateAgentModel"></select>
                </label>
                <p id="duplicateAgentError" class="auth-error" hidden></p>
                <button id="duplicateAgentSubmit" class="auth-submit-btn" type="submit">Create the copy</button>
            </form>
        </div>
    </div>
```

- [ ] **Step 4: Offer the action on the card**

In `renderAgentCardActions` (`app.js:1044`), immediately after the `rotate` constant:

```js
  // Only once the user has actually run this agent: "try it on another model"
  // is a follow-on offer, not a first action. Built-in only -- duplicating an
  // external agent would mint an API key (see the backend's duplicate route).
  const duplicate =
    agent.agent_type === 'builtin' && (statusKey === 'backtested' || statusKey === 'paper')
      ? `<button class="agent-menu-item agent-duplicate-model-btn" type="button" data-agent-id="${id}">Run on another model</button>`
      : '';
```

and add it to the dropdown, directly above the delete button (line 1059):

```js
          ${duplicate}
```

- [ ] **Step 5: Add the helpers and the modal logic**

Insert into `app.js` immediately before `renderAgentRunningActions` (~line 1065):

```js
/** SUPPORTED_MODELS minus the agent's current model -- an entry that duplicates
 * an agent onto the model it already runs is a no-op the user has to reason
 * about. A legacy or hosted-runtime model isn't in the list, so nothing is
 * filtered out and the full six are offered. */
function duplicateModelChoices(agent) {
  const current = String(agent?.model_name || '').trim().toLowerCase();
  return SUPPORTED_MODELS.filter((model) => model.slug.toLowerCase() !== current).map(
    (model) => ({ slug: model.slug, label: model.label }),
  );
}

/** "Momentum Alpha (DeepSeek)". Collides freely: two copies onto DeepSeek read
 * the same. Names are not unique anywhere else in this product, and
 * de-duplicating would mean a lookup for a cosmetic gain. */
function duplicateAgentName(agent, modelSlug) {
  const vendor = MODEL_VENDORS.find((entry) => entry.key === modelVendorKey(modelSlug));
  return `${agent?.name || 'Agent'} (${vendor?.label || 'new model'})`;
}

let duplicateAgentSource = null;

function openDuplicateAgentModal(agent) {
  const modal = document.getElementById('duplicateAgentModal');
  const select = document.getElementById('duplicateAgentModel');
  const error = document.getElementById('duplicateAgentError');
  if (!modal || !select || !agent) return;
  duplicateAgentSource = agent;
  select.innerHTML = duplicateModelChoices(agent)
    .map((model) => `<option value="${escapeHtml(model.slug)}">${escapeHtml(model.label)}</option>`)
    .join('');
  if (error) { error.hidden = true; error.textContent = ''; }
  modal.hidden = false;
}

function closeDuplicateAgentModal() {
  const modal = document.getElementById('duplicateAgentModal');
  if (modal) modal.hidden = true;
  duplicateAgentSource = null;
}

/** Lands the user on the new agent with Run primed. Deliberately does NOT start
 * a backtest: auto-firing would spend LLM credits on a click the user framed as
 * "make a copy". */
async function submitDuplicateAgent() {
  const agent = duplicateAgentSource;
  const select = document.getElementById('duplicateAgentModel');
  const error = document.getElementById('duplicateAgentError');
  const submit = document.getElementById('duplicateAgentSubmit');
  if (!agent || !select?.value) return;
  if (submit) submit.disabled = true;
  try {
    const data = await API.post(
      `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}/duplicate`,
      { model_name: select.value, name: duplicateAgentName(agent, select.value) },
    );
    const created = data?.agent;
    if (!created?.agent_id) throw new Error('Copy failed — no agent returned');
    closeDuplicateAgentModal();
    applyActiveAgent(created);
    await loadAgents();
    showAppToast(`${created.name} is ready. Press Run Backtest to compare them.`);
    highlightAgentCard(created.agent_id);
  } catch (err) {
    if (error) {
      error.textContent = err.message || `Couldn't create the copy. Please try again.`;
      error.hidden = false;
    }
  } finally {
    if (submit) submit.disabled = false;
  }
}
```

`highlightAgentCard` (`app.js:3909`) is the existing scroll-and-flash helper — it is scoped to `.agent-card`, because every card also carries the same `data-agent-id` on five to eight of its buttons.

- [ ] **Step 6: Bind the card action**

In the card-binding block, immediately after the `.agent-rotate-key-btn` binding (~line 1358):

```js
  grid.querySelectorAll('.agent-duplicate-model-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent) return;
      openDuplicateAgentModal(agent);
    });
  });
```

- [ ] **Step 7: Wire the modal in the boot block**

In `DOMContentLoaded`, next to the other modal wiring (~line 3966):

```js
    document.getElementById('duplicateAgentModalClose')?.addEventListener('click', closeDuplicateAgentModal);
    document.getElementById('duplicateAgentModalBackdrop')?.addEventListener('click', closeDuplicateAgentModal);
    document.getElementById('duplicateAgentForm')?.addEventListener('submit', (event) => {
        event.preventDefault();
        submitDuplicateAgent();
    });
```

- [ ] **Step 8: Run the tests to verify they pass**

```bash
pytest dashboard/backend/tests/test_frontend_model_facets.py -v
```

Expected: PASS (24 cases).

- [ ] **Step 9: Commit**

```bash
git status --short
git add dashboard/frontend/app.js dashboard/frontend/app.html \
        dashboard/backend/tests/test_frontend_model_facets.py
git commit -m "feat(agents): run an agent you've already tested on another model"
```

---

### Task 11: Cache busters, full verification, and the PR

**Files:**
- Modify: `dashboard/frontend/app.html:16` (styles.css buster), `:1733` (app.js buster)
- Modify: `docs/architecture/dashboard-target-structure.md` if it enumerates agent routes

- [ ] **Step 1: Bump both cache busters**

`app.js` and `styles.css` both changed in this PR. Edit `dashboard/frontend/app.html`:

```html
    <link rel="stylesheet" href="styles.css?v=85">
```

```html
    <script src="app.js?v=75" defer></script>
```

Both numbers assume PR 1 shipped `app.js?v=74`. **Check the current values first** (`command grep -n 'app.js?v=\|styles.css?v=' dashboard/frontend/app.html`) and increment from what is actually on `main` — a concurrent PR bumping to the same number is a known collision in this repo. `agent-editor.js` is untouched; leave its buster alone.

- [ ] **Step 2: Check whether any doc enumerates the agent routes**

```bash
command grep -rn "rotate-api-key\|marketplace/{template_id}/clone" docs/ --include=*.md --include=*.rst
```

If a developer-facing doc lists the `/api/v1/agents` routes, add `POST /api/v1/agents/{agent_id}/duplicate` to it. Leave user-facing/hosted docs alone — those are coordinated separately; record any staleness found for the session's follow-up list instead of editing.

- [ ] **Step 3: Run the full backend suite**

```bash
pytest dashboard/backend/tests/ -q
```

Expected: PASS. A red test is a real regression.

- [ ] **Step 4: Smoke-test the UI against a scratch database**

```bash
DATABASE_PATH=/tmp/claude-1000/-mnt-d-github-agent-trading-lab/scratch-facets.db \
  uvicorn dashboard.backend.app:app --port 8010
```

Then in a browser at `http://localhost:8010/app`, confirm by hand:
1. Community shows two chip rows; the vendor row lists exactly the vendors present in the catalog (six after PR 3), not all eight.
2. `China A-Share` + `Qwen` together show one card; `China A-Share` + `GPT` shows the **"No templates match both filters"** copy with a working **Clear filters** button.
3. DeepSeek/Qwen/Nemotron cards carry `Open-source model`; Claude/GPT/Gemini cards carry no licence badge at all.
4. `Add to My Agents` still clones in one click on the template's own model.
5. `Choose model ▾` → `DeepSeek V4 Pro` clones onto that model (check the model shown in Configure).
6. A backtested agent's `⋯` menu offers `Run on another model`; the copy appears named `… (DeepSeek)` and **no backtest starts**.
7. `DATABASE_PATH` was set, so the committed seed DB is untouched — confirm with `git status --short`.

- [ ] **Step 5: Commit and open PR 4**

```bash
git status --short   # confirm dashboard/storage/data/backtest.db is NOT listed
git add dashboard/frontend/app.html
git commit -m "chore(app): bump cache busters for the model-facet round"
git push -u origin feat/community-model-facets
```

Run the prod gate probe from the Delivery order section. **If it does not print `True` twice**, open as a draft:

```bash
gh pr create --draft --title "feat(community): browse and clone agents by AI model" --body "$(cat <<'EOF'
DO NOT MERGE until PR #<N> is serving in prod — probe `/openapi.json` for the `duplicate` route and `CloneMarketplaceBody.model_name`.

Community gains a model-vendor chip row that ANDs with the market row, an `Open-source model` badge, and two ways to make another agent: `Choose model ▾` beside the clone CTA, and `Run on another model` on any agent that has already been backtested. Neither auto-starts a run.

The vendor axis is derived from `model_name` — no column, no migration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

If the probe passes, drop `--draft` and the first body line.

---

## Self-review

**Spec coverage.** §1 → Tasks 1–2. §2 → Task 6. §3 (facets, three empty states, derived chips, page-index reset) → Task 7. §3's licence badge → Task 8. §4 → Task 5. §5 → Tasks 3, 9. §6 → Tasks 4, 10. §7 → the guard suites in every task, plus Task 11's manual pass. §8 → the Delivery order table and Task 11's gate.

Two spec items landed differently, both flagged in **Deviations** above: the nvidia label (`NVIDIA Nemotron`, to keep `formatModelProviderLabel` byte-identical as §2 requires), and `agent-editor.js`'s cache buster (no task edits that file).

**One spec line has no task, deliberately.** §3's "Selecting a vendor chip resets the grid's page index, matching `setAgentMarketFilter`'s existing behaviour" — the Community grid has **no pagination**; `renderMarketplaceGrid` renders every filtered template. `AGENT_GRID_PAGE_SIZE` and `agentGridPage` apply to **My Agents**, which gains no vendor chips (a spec non-goal). There is no page index to reset. If pagination is ever added to Community, `setMarketplaceVendorFilter` is where the reset goes.

**Type consistency.** `modelVendorKey`/`agentVendorKey`/`modelVendorLicence` are defined in Task 6 and used under those exact names in Tasks 7, 8 and 10. `SUPPORTED_MODELS` entries carry `{slug, label, vendor}` throughout; `MODEL_VENDORS` entries carry `{key, prefix, label, licence}` throughout. `cloneMarketplaceTemplate(template, modelName)` is defined and called with two arguments in Task 9 and with one in the untouched primary-CTA path. `duplicate_agent` is the route function name in both golden-set entries and the service method name in Task 4.
