# User-Feedback UX Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three defects a real tester reported — agent creation looks broken for ~5s, a running backtest gives no sense of position, and the landing page never says what the product is for.

**Architecture:** Three independent PRs over one spec. **A** adds a toast primitive and pending-button states to the vanilla-JS dashboard. **B** routes step-level progress that the backend *already emits* onto the My Agents card, adding one new backend datum (progress-file mtime) for staleness. **C** adds a value-proposition band to the React landing source and rebuilds the hand-patched bundle.

**Tech Stack:** FastAPI + vanilla JS (no build step) for the dashboard; React 18 + Vite 6 + Tailwind + framer-motion for the landing; pytest for all tests, including source-guards and a node-under-pytest harness for JS behaviour.

**Spec:** `docs/superpowers/specs/2026-08-01-user-feedback-ux-round-design.md`

## Global Constraints

- **Docs are a non-goal.** No task edits any file under `docs/` except this plan's own checkboxes. Staleness is recorded in the spec, not fixed here.
- **No paper-trading or real-capital claim** may appear in any user-visible string. The precise, defensible form of the claim: **no order-submission route exists.** `AlpacaPaperTradingClient` has no order method, `PaperTradingSession.add_trade` has zero production callers, `execution/paper_backend.py` raises `NotImplementedError`, and `ROBINHOOD_EXECUTE` defaults to `false`. Use *that* wording in PR bodies — **"`/paper/*` is read-only" is literally false**: `POST /paper/start-session` (`api/routers/paper_trading.py:253`) writes a run row via `db.insert_run(...)`. It places no order and no frontend calls it, so the conclusion holds; the shorthand does not, and a reviewer who greps will find the write.
- **Run every command from the repo root.** The backend is the `dashboard.backend` package; never run a backend file by path.
- **Tests:** `pytest dashboard/backend/tests/ -v`. Pytest is not in `requirements.txt` — install separately. The suite is green on `main`; a red test is a real regression.
- **Never `git add -A`.** Importing a backend module runs lazy `CREATE TABLE`/`ALTER` against the committed `dashboard/storage/data/backtest.db`, which is the real prod seed database. Stage named paths only.
- **`prefers-reduced-motion`:** every animating element added here needs a `@media (prefers-reduced-motion: reduce)` fallback. Carried forward from the 2026-07-29 running-state spec.
- **CSS tokens** (from `styles.css:14-26`): `--bg-card: #131a35`, `--bg-surface: #0f1328`, `--text-primary: #e5e7eb`, `--border-color: #1f2937`.
- **Branch discipline:** each phase gets its own branch off fresh `main`. Never push follow-up work to a merged branch.

---

# Phase A — Create-agent interaction feedback (PR 1)

Branch: `fix/create-agent-feedback`

`submitCreateBuiltinAgent` (`dashboard/frontend/app.js:1812`) already sets `submitBtn.disabled = true` at `:1838`. Everything the tester missed is what is not there: no label change, no spinner, no success confirmation.

---

### Task 1: Toast primitive

There is no toast system in `/app` — `alert()` appears 18 times and is the current convention. A blocking modal for a *success* is worse than the silence it replaces, so add a real one.

**Files:**
- Modify: `dashboard/frontend/app.html` (add container before `</body>`)
- Modify: `dashboard/frontend/app.js` (add `showAppToast`)
- Modify: `dashboard/frontend/styles.css` (append `.app-toast` block)
- Test: `dashboard/backend/tests/test_app_toast.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `showAppToast(message, { variant = 'success', timeoutMs = 4000 })` — global function in `app.js`. Task 2 and Task 3 call it.

**Do not reuse `.home-live-toast`** (`styles.css:6792`). That is the Home live-decision widget in the same shared stylesheet; a distinct class keeps the two from co-evolving.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_app_toast.py`:

```python
"""The /app toast primitive: a non-blocking success channel.

Agent creation previously closed its modal and refreshed the grid with no
confirmation at all, so a slow create read as a dead click. `alert()` -- this
file's 18-times-over convention -- is modal and blocking, which is a worse
answer for a *success* than the silence it replaces.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")


def test_toast_container_is_a_polite_live_region():
    """A success message screen readers never announce is not a confirmation.

    Asserted as one whole tag, not three independent substrings: app.html:377
    (the ticker) already carries role="status" and aria-live="polite", so
    file-wide substring checks for those two would pass before the toast
    exists -- two thirds of a vacuous test.
    """
    assert (
        '<div id="appToast" class="app-toast" role="status" aria-live="polite" hidden>'
        in _APP_HTML
    )


def test_toast_helper_exists():
    assert "function showAppToast(" in _APP_JS


def test_toast_is_not_the_home_live_toast():
    """Distinct class: .home-live-toast is the Home live-decision widget in the
    same shared stylesheet, and conflating them couples two unrelated features."""
    assert ".app-toast" in _STYLES


def test_toast_animation_has_a_reduced_motion_fallback():
    block = _STYLES[_STYLES.index(".app-toast") :]
    assert "prefers-reduced-motion" in block
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_app_toast.py -v`
Expected: FAIL — the container tag is absent from `_APP_HTML`

- [ ] **Step 3: Add the container to `app.html`**

Insert immediately before the closing `</body>` tag — **byte-for-byte as written**, since Step 1 pins the whole opening tag (attribute order included):

```html
<div id="appToast" class="app-toast" role="status" aria-live="polite" hidden></div>
```

- [ ] **Step 4: Add `showAppToast` to `app.js`**

Add near the other top-level UI helpers (anywhere above `submitCreateBuiltinAgent` at `:1812`):

```js
let appToastTimer = null;

/**
 * Non-blocking confirmation channel for /app.
 *
 * The pre-existing convention here is alert(), which is modal: acceptable for a
 * launch-time refusal the user must acknowledge, wrong for a success they only
 * need to notice. Text (not innerHTML) -- callers pass agent names.
 */
function showAppToast(message, { variant = 'success', timeoutMs = 4000 } = {}) {
  const el = document.getElementById('appToast');
  if (!el) return;
  el.textContent = String(message);
  el.className = `app-toast app-toast--${variant}`;
  el.hidden = false;
  // Force a reflow so re-showing an already-visible toast replays the transition.
  void el.offsetWidth;
  el.classList.add('is-visible');
  if (appToastTimer) clearTimeout(appToastTimer);
  appToastTimer = setTimeout(() => {
    el.classList.remove('is-visible');
    appToastTimer = setTimeout(() => { el.hidden = true; }, 240);
  }, timeoutMs);
}
```

- [ ] **Step 5: Add the styles**

Append to `dashboard/frontend/styles.css`:

```css
/* /app toast — success confirmations. Distinct from .home-live-toast, which is
   the Home live-decision widget. */
.app-toast {
    position: fixed;
    left: 50%;
    bottom: 32px;
    transform: translate(-50%, 12px);
    z-index: 9000;
    max-width: min(420px, calc(100vw - 32px));
    padding: 12px 18px;
    border-radius: 10px;
    border: 1px solid var(--border-color);
    background: var(--bg-card);
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
    opacity: 0;
    pointer-events: none;
    transition: opacity 220ms ease, transform 220ms ease;
}

.app-toast.is-visible {
    opacity: 1;
    transform: translate(-50%, 0);
}

.app-toast--success { border-color: rgba(34, 197, 94, 0.45); }
.app-toast--error { border-color: rgba(239, 68, 68, 0.45); }

@media (prefers-reduced-motion: reduce) {
    .app-toast { transition: none; transform: translate(-50%, 0); }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_app_toast.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_app_toast.py
git commit -m "feat(ui): add non-blocking toast primitive for /app"
```

---

### Task 2: Pending button state + confirm on POST resolution

**Files:**
- Modify: `dashboard/frontend/app.js:1812-1859` (`submitCreateBuiltinAgent`)
- Modify: `dashboard/frontend/styles.css` (spinner)
- Test: `dashboard/backend/tests/test_create_agent_feedback.py` (create)

**Interfaces:**
- Consumes: `showAppToast(message, opts)` from Task 1.
- Produces: `setButtonPending(btn, label)` and `restoreButton(btn)` — Task 3 does not use them, but future async submits will.

Two changes, and the ordering one matters as much as the visible one. Today the flow is: close modal → `applyActiveAgent` → `await loadAgents()`, with the button restored in `finally` *after* the refresh. The confirmation must not be gated on the grid refresh.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_create_agent_feedback.py`:

```python
"""Create-agent gives feedback within one frame of the click.

A tester reported ~5 seconds of apparently-dead UI after clicking "Create
built-in agent". The agent was created correctly every time; the button just
never changed and nothing confirmed success. The POST itself is genuinely slow
(see the round-trip note in the spec), so the fix is feedback, not latency.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def _submit_fn() -> str:
    start = _APP_JS.index("async function submitCreateBuiltinAgent(")
    depth = 0
    i = _APP_JS.index("{", start)
    while True:
        if _APP_JS[i] == "{":
            depth += 1
        elif _APP_JS[i] == "}":
            depth -= 1
            if depth == 0:
                return _APP_JS[start : i + 1]
        i += 1


def test_helpers_exist():
    assert "function setButtonPending(" in _APP_JS
    assert "function restoreButton(" in _APP_JS


def test_pending_state_is_set_before_the_await():
    """Set after the await, the label would appear only once the POST returned --
    exactly the window the tester experienced as dead."""
    fn = _submit_fn()
    assert "setButtonPending(" in fn
    assert fn.index("setButtonPending(") < fn.index("await API.post")


def test_pending_label_is_creating():
    assert "'Creating…'" in _submit_fn()


def test_success_confirmation_is_not_gated_on_the_grid_refresh():
    """loadAgents() is a second round trip. Confirming after it would reintroduce
    most of the delay the toast exists to cover."""
    fn = _submit_fn()
    assert "showAppToast(" in fn
    assert fn.index("showAppToast(") < fn.index("await loadAgents()")


def test_button_is_restored_on_every_path():
    """finally, not the success branch: an error must not strand a dead button."""
    fn = _submit_fn()
    finally_block = fn[fn.index("} finally {") :]
    assert "restoreButton(" in finally_block


def test_aria_busy_is_toggled():
    assert re.search(r"aria-busy", _APP_JS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_create_agent_feedback.py -v`
Expected: FAIL — `assert "function setButtonPending(" in _APP_JS`

- [ ] **Step 3: Add the helpers to `app.js`**

Add immediately above `openCreateBuiltinAgentModal` (`app.js:1797`):

```js
/**
 * Lock a submit button and say what it is doing.
 *
 * disabled alone is nearly invisible in this theme, which is why a create that
 * already set it still read as a dead click.
 */
function setButtonPending(btn, label) {
  if (!btn) return;
  if (btn.dataset.idleLabel === undefined) btn.dataset.idleLabel = btn.textContent;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.classList.add('is-pending');
  btn.textContent = label;
}

function restoreButton(btn) {
  if (!btn) return;
  btn.disabled = false;
  btn.removeAttribute('aria-busy');
  btn.classList.remove('is-pending');
  if (btn.dataset.idleLabel !== undefined) btn.textContent = btn.dataset.idleLabel;
}
```

- [ ] **Step 4: Rewrite the submit body**

In `submitCreateBuiltinAgent`, replace the block from `if (submitBtn) submitBtn.disabled = true;` (`:1838`) through the end of the function with:

```js
  setButtonPending(submitBtn, 'Creating…');

  try {
    const data = await API.post(`${API_BASE}/api/v1/agents`, {
      name,
      model_name,
      agent_type: 'builtin',
      description,
      cash_allocation,
    });
    // Confirm on the POST result, not after loadAgents(): that is a second
    // round trip, and gating the toast on it reinstates most of the delay.
    closeCreateBuiltinAgentModal();
    showAppToast(`"${name}" created`);
    if (data.agent) applyActiveAgent(data.agent);
    await loadAgents();
    if (data.agent) highlightAgentCard(data.agent.agent_id);
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.hidden = false;
    }
  } finally {
    restoreButton(submitBtn);
  }
}
```

`highlightAgentCard` is defined in **Task 3**, so this commit alone ships a live `ReferenceError` on the success path — after the toast fires, before `restoreButton`. The source-guard tests are string checks and stay green through it, so nothing will tell you. It is contained only because the PR opens at **phase exit**, never between these two commits: do not browser-test, push, or open a PR until Task 3 is committed.

- [ ] **Step 5: Add the pending spinner style**

Append to `dashboard/frontend/styles.css`:

```css
/* Pending submit buttons — disabled alone is nearly invisible in this theme. */
.auth-submit-btn.is-pending {
    opacity: 0.75;
    cursor: progress;
}

.auth-submit-btn.is-pending::before {
    content: '';
    display: inline-block;
    width: 12px;
    height: 12px;
    margin-right: 8px;
    vertical-align: -1px;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: app-btn-spin 640ms linear infinite;
}

@keyframes app-btn-spin { to { transform: rotate(360deg); } }

@media (prefers-reduced-motion: reduce) {
    .auth-submit-btn.is-pending::before { animation: none; }
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_create_agent_feedback.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_create_agent_feedback.py
git commit -m "fix(ui): lock and label the create-agent button, confirm on success"
```

---

### Task 3: Locate the newly created agent

Answers "did it work?" positionally, not just textually — the grid can be long and paginated.

**Files:**
- Modify: `dashboard/frontend/app.js` (add `highlightAgentCard`)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_create_agent_feedback.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `highlightAgentCard(agentId)` — called by Task 2's success path.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_create_agent_feedback.py`:

```python
def test_new_agent_card_is_located_after_creation():
    assert "function highlightAgentCard(" in _APP_JS
    assert "highlightAgentCard(" in _submit_fn()


def test_highlight_uses_attribute_lookup_not_selector_interpolation():
    """Same rule refreshRunningAgentCards() follows (app.js:3370): agent ids are
    server-supplied, so never interpolate one into a selector string."""
    start = _APP_JS.index("function highlightAgentCard(")
    body = _APP_JS[start : start + 900]
    assert "querySelectorAll('.agent-card[data-agent-id]')" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_create_agent_feedback.py -v -k highlight`
Expected: FAIL — `assert "function highlightAgentCard(" in _APP_JS`

- [ ] **Step 3: Tag the card element — and scope the selector to it**

Run: `command grep -n "data-agent-id" dashboard/frontend/app.js`

All ten existing occurrences are **`<button>` elements inside the card** (`:522`, `:892`, `:899`, `:902`, `:906`, `:912`, `:916`, `:918`, `:928`, `:1335` — Open Agent, Run Backtest, Configure, the menu items, the run links). **The outer card element carries no `data-agent-id`.**

Two consequences, both load-bearing:

1. Add it, alongside the existing `card.className = ...` at `app.js:1124` (which already includes the `agent-card` class):
   ```js
   card.setAttribute('data-agent-id', agent.agent_id);
   ```
   `setAttribute`, not HTML interpolation.
2. **The selector must be `.agent-card[data-agent-id]`, never the bare `[data-agent-id]`.** The bare form matches the card *and* its 5–8 child buttons, firing `scrollIntoView` six-to-eight times per creation — the smooth-scroll target changes mid-flight and the page visibly jitters. The class narrows it to the one element; marketplace cards (`:1666`) share `.agent-card` but carry no `data-agent-id`, so the conjunction matches exactly one node.

- [ ] **Step 4: Implement**

Add next to `refreshRunningAgentCards` in `app.js`:

```js
/**
 * Scroll the named agent's card into view and flash it.
 *
 * Attribute lookup then compare in JS -- no escaping, no CSS.escape feature
 * detection -- matching refreshRunningAgentCards() (app.js:3370).
 *
 * Scoped to .agent-card: every card also contains 5-8 buttons carrying the same
 * data-agent-id, and the unscoped selector would scroll to each of them in turn.
 */
function highlightAgentCard(agentId) {
  if (!agentId) return;
  document.querySelectorAll('.agent-card[data-agent-id]').forEach((card) => {
    if (card.getAttribute('data-agent-id') !== agentId) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('is-just-created');
    setTimeout(() => card.classList.remove('is-just-created'), 2400);
  });
}
```

- [ ] **Step 5: Add the flash style**

Append to `dashboard/frontend/styles.css`:

```css
.agent-card.is-just-created {
    animation: agent-card-flash 2.4s ease-out;
}

@keyframes agent-card-flash {
    0%, 40% { box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.55); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

@media (prefers-reduced-motion: reduce) {
    .agent-card.is-just-created { animation: none; box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.55); }
}
```

- [ ] **Step 6: Run the full Phase A suite**

Run: `pytest dashboard/backend/tests/test_app_toast.py dashboard/backend/tests/test_create_agent_feedback.py -v`
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_create_agent_feedback.py
git commit -m "feat(ui): scroll to and flash a newly created agent"
```

---

### Task 4: Record the create-path round-trip cost — no code change

The spec makes latency an investigation, not a deliverable. The investigation is already done; this task records it so nobody re-derives it, and explicitly forbids the tempting fix.

**Files:** none modified. Output goes in the PR body.

**Measured:** `POST /api/v1/agents` (`agents.py:129`) calls `portfolio_service._reconcile()` **twice** — once inside `ensure_cash_for_new_agent` → `_check_new_sleeve` (`service.py:120`, only when `delta > 0`) and once inside `get_or_create_portfolio` (`agents.py:176`). Each `_reconcile` is 2–3 DB round trips: `portfolio_store.get_or_create`, `agent_store.list_agents`, and conditionally `portfolio_store.set_cash_available`. Add `agent_service.create_agent` and `agent_with_stats` and the create path is roughly **5–7 Neon round trips**, then the frontend issues `loadAgents()` on top.

**Do not remove the second `_reconcile`.** It looks redundant — its return value is discarded at `agents.py:175-176` — but `service.py`'s module docstring states the design intent: *"`_reconcile` on every read and after every write"*, deriving `cash_available` from the sleeves so legacy drift self-corrects. The post-create call is the "after every write" half and it persists the new figure via `set_cash_available`. This is money accounting with documented failure modes (#175, `reclaim_on_session_match`); a latency micro-optimisation is not worth relitigating it inside a UX PR.

- [ ] **Step 1: Verify the round-trip claim still holds**

Run: `command grep -n "_reconcile" dashboard/backend/domain/portfolios/service.py`
Expected: `_reconcile` defined once, called from `_check_new_sleeve` and `get_or_create_portfolio`.

- [ ] **Step 2: Paste the paragraph above into the PR body under "Latency — investigated, not changed"**

- [ ] **Step 3: No commit** (documentation-in-PR-body only; the spec forbids doc edits)

---

**Phase A exit:** `pytest dashboard/backend/tests/ -v` fully green. Open PR `fix(ui): create-agent feedback`.

---

# Phase B — Backtest progress (PR 2)

Branch: `feat/backtest-progress-visibility`

**Read this first.** The 2026-07-29 spec (`docs/superpowers/specs/2026-07-29-my-agents-capital-instruction-running-design.md`) made this a non-goal, reasoning: *"deliberately not a percentage, since no honest completion estimate exists."* That premise is false — `engine.py:287` (`_publish_live_progress`) has been writing `step`/`total_steps` on every step, and `backtests.py:1272` already computes the percentage. We are correcting a factual error, not relitigating taste. Do not revert to indeterminate.

---

### Task 5: Backend — expose progress-file freshness

The only genuinely new datum in Phase B. Everything else already ships in the status payload.

**Files:**
- Modify: `dashboard/backend/api/routers/backtests.py:324-336` (`_read_backtest_progress`)
- Test: `dashboard/backend/tests/test_backtest_progress_status.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `progress.progress_updated_at` — a float, epoch seconds, present in `GET /backtest/status`'s `progress` object whenever a readable progress file exists. Task 7 consumes it.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_backtest_progress_status.py`:

```python
"""Progress-file freshness, so the UI can tell 'working' from 'stuck'.

The status payload already carried step/total_steps; what it could not answer
was whether those numbers were current. A run whose subprocess wedges keeps
reporting its last step forever, which reads identically to steady progress.
"""

import json
import time

from dashboard.backend.api.routers import backtests


def test_progress_carries_the_file_mtime(tmp_path, monkeypatch):
    progress_file = tmp_path / "backtest_progress_test.json"
    progress_file.write_text(json.dumps({"step": 7, "total_steps": 240}), encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))

    payload = backtests._read_backtest_progress()

    assert payload["step"] == 7
    assert payload["total_steps"] == 240
    assert payload["progress_updated_at"] == progress_file.stat().st_mtime
    assert payload["progress_updated_at"] <= time.time() + 1


def test_missing_progress_file_still_returns_none(tmp_path, monkeypatch):
    """Unchanged behaviour: the status payload omits `progress` entirely rather
    than shipping a half-populated object."""
    monkeypatch.setitem(
        backtests.backtest_status, "progress_file", str(tmp_path / "nope.json")
    )
    assert backtests._read_backtest_progress() is None


def test_malformed_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "broken.json"
    progress_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None


def test_non_dict_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "list.json"
    progress_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtest_progress_status.py -v`
Expected: FAIL — `KeyError: 'progress_updated_at'`

- [ ] **Step 3: Implement**

Replace the body of `_read_backtest_progress` (`backtests.py:324-336`):

```python
def _read_backtest_progress() -> Optional[Dict[str, Any]]:
    """Load incremental equity snapshots written by the backtest subprocess.

    ``progress_updated_at`` is the file's mtime, not a field the writer emits:
    it answers "are these numbers current?", which the payload alone cannot.
    stat() and read_text() are separate syscalls, so a file rewritten between
    them yields an mtime marginally older than the payload -- immaterial against
    a 120s staleness threshold, and not worth a lock to avoid.
    """
    progress_file = backtest_status.get("progress_file")
    if not progress_file:
        return None
    path = Path(progress_file)
    if not path.is_file():
        return None
    try:
        updated_at = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {**payload, "progress_updated_at": updated_at}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_status.py -v`
Expected: 4 passed

- [ ] **Step 5: Confirm no existing caller regressed**

Run: `pytest dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/integrations/test_discord_watcher.py -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtest_progress_status.py
git commit -m "feat(backtest): report progress-file freshness in status"
```

---

### Task 6: Progress formatters (pure functions, real behavioural tests)

These are pure, so they get executed under node rather than grepped. The harness pattern is established in `dashboard/backend/tests/test_my_agents_card_ui.py` — it lifts a function out of `app.js` by brace-matching and runs it with `node -e`.

**Files:**
- Modify: `dashboard/frontend/app.js` (add two functions near `formatBacktestElapsed`, `:4585`)
- Test: `dashboard/backend/tests/test_backtest_progress_format.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `formatBacktestEta(elapsedSeconds, step, totalSteps)` → `string | null`. `null` when `step < 3`, when `step`/`totalSteps` are missing or non-finite, when `totalSteps <= 0`, or when `step >= totalSteps`.
  - `formatProgressStaleness(secondsSinceUpdate)` → `string | null`. `null` below 120s.
  - `BACKTEST_STALE_SECONDS = 120` — module constant.

  Task 7 and Task 8 both call these.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_backtest_progress_format.py`:

```python
"""ETA and staleness formatting for a running backtest.

Both are honesty-constrained rather than precision-constrained:

* an ETA derived from two or three steps is wild, and a number that visibly
  jumps reads as broken -- so it is suppressed early and coarse thereafter;
* a stale progress file means the numbers are old, NOT that the run is stuck.
  An LLM pipeline step can legitimately take minutes. Claiming "stuck" would be
  the same class of error as the fabricated Performance Drivers card.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract_function(src: str, name: str) -> str:
    for marker in (f"async function {name}(", f"function {name}("):
        start = src.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in app.js")
    depth = 0
    i = src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1


def _eval(expr: str) -> object:
    script = "\n".join(
        [
            "const BACKTEST_STALE_SECONDS = 120;",
            _extract_function(_APP_JS, "formatBacktestEta"),
            _extract_function(_APP_JS, "formatProgressStaleness"),
            f"console.log(JSON.stringify({expr}));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_eta_is_suppressed_for_the_first_two_steps():
    assert _eval("formatBacktestEta(4, 1, 240)") is None
    assert _eval("formatBacktestEta(8, 2, 240)") is None


def test_eta_appears_from_step_three():
    # 30s for 3 of 243 steps -> 10s/step -> 2400s remaining -> "~40m left"
    assert _eval("formatBacktestEta(30, 3, 243)") == "~40m left"


def test_eta_is_coarse_under_a_minute():
    # 100s for 100 of 130 steps -> 1s/step -> 30s remaining
    assert _eval("formatBacktestEta(100, 100, 130)") == "<1m left"


def test_eta_rounds_to_whole_minutes():
    # 125s for 25 of 50 steps -> 5s/step -> 125s remaining -> ~2m
    assert _eval("formatBacktestEta(125, 25, 50)") == "~2m left"


def test_eta_is_null_without_totals():
    assert _eval("formatBacktestEta(60, 10, 0)") is None
    assert _eval("formatBacktestEta(60, 10, null)") is None
    assert _eval("formatBacktestEta(60, null, 240)") is None


def test_eta_is_null_on_the_final_step():
    """No remaining work to estimate; the completion path takes over."""
    assert _eval("formatBacktestEta(600, 240, 240)") is None


def test_staleness_is_silent_below_the_threshold():
    assert _eval("formatProgressStaleness(0)") is None
    assert _eval("formatProgressStaleness(119)") is None


def test_staleness_reports_the_actual_gap_not_the_threshold():
    """A message frozen at '2m' while the real gap grows to ten is worse than
    no message -- it actively misinforms."""
    assert "2m" in _eval("formatProgressStaleness(130)")
    assert "9m" in _eval("formatProgressStaleness(560)")


def test_staleness_wording_does_not_claim_the_run_is_stuck():
    message = _eval("formatProgressStaleness(300)")
    assert "stuck" not in message.lower()
    assert "fail" not in message.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtest_progress_format.py -v`
Expected: FAIL — `AssertionError: formatBacktestEta not found in app.js`

- [ ] **Step 3: Implement**

Add immediately after `formatBacktestElapsed` (`app.js:4590`):

```js
/** A progress file older than this is reported as stale (seconds). */
const BACKTEST_STALE_SECONDS = 120;

/**
 * Coarse remaining-time estimate, or null when no honest one exists.
 *
 * Suppressed below step 3: the first estimates swing wildly, and a number that
 * visibly jumps reads as broken. Coarse buckets thereafter -- a precise-looking
 * ETA that drifts is worse than an obviously approximate one.
 */
function formatBacktestEta(elapsedSeconds, step, totalSteps) {
  const elapsed = Number(elapsedSeconds);
  const done = Number(step);
  const total = Number(totalSteps);
  if (!Number.isFinite(elapsed) || elapsed <= 0) return null;
  if (!Number.isFinite(done) || !Number.isFinite(total)) return null;
  if (total <= 0 || done < 3 || done >= total) return null;
  const remaining = (elapsed / done) * (total - done);
  if (!Number.isFinite(remaining) || remaining <= 0) return null;
  if (remaining < 60) return '<1m left';
  return `~${Math.round(remaining / 60)}m left`;
}

/**
 * Staleness notice, or null while progress is fresh.
 *
 * Reports the *actual* gap, never the threshold: a message frozen at "2m" while
 * the real gap grows to ten actively misinforms. Deliberately does not say
 * "stuck" -- we know the file is old, not that the run died, and a long model
 * step looks exactly like this.
 */
function formatProgressStaleness(secondsSinceUpdate) {
  const gap = Number(secondsSinceUpdate);
  if (!Number.isFinite(gap) || gap < BACKTEST_STALE_SECONDS) return null;
  const minutes = Math.floor(gap / 60);
  return `No progress for ${minutes}m — long model steps can do this.`;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_format.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/frontend/app.js dashboard/backend/tests/test_backtest_progress_format.py
git commit -m "feat(backtest): add ETA and staleness formatters"
```

---

### Task 7: Determinate progress on the My Agents card

This is the surface the tester was actually standing on. The existing running-state store (`readRunningBacktests`, `app.js:3320`) holds only `{runId, startedAt}`; it needs the polled progress too.

**Files:**
- Modify: `dashboard/frontend/app.js:782-795` (`renderAgentRunningBody`)
- Modify: `dashboard/frontend/app.js:3362-3382` (`refreshRunningAgentCards`)
- Modify: `dashboard/frontend/app.js:4870-4901` (the `status.running` branch of `ensureBacktestPolling`)
- Modify: `dashboard/frontend/app.js:4942-4954` (the timeout branch — Step 4b)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_backtest_progress_card.py` (create)

**Interfaces:**
- Consumes: `formatBacktestEta`, `formatProgressStaleness`, `BACKTEST_STALE_SECONDS` (Task 6); `progress.progress_updated_at` (Task 5).
- Produces: `liveBacktestProgress` — a module-level `{ step, totalSteps, updatedAt }` object (or `null`), set by the poller and read by the renderer.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_backtest_progress_card.py`:

```python
"""The My Agents card shows step-level progress.

The backend has always emitted step/total_steps (engine.py `_publish_live_progress`,
surfaced by backtests.py `get_backtest_status`), and the Backtest tab has always
had a percentage bar. The card -- the page a user lands on after launching --
threw the data away and rendered an indeterminate bar plus an elapsed timer. A
tester watched it for 3m05s and could not tell running from stuck.

The 2026-07-29 spec called an indeterminate bar deliberate "since no honest
completion estimate exists". That premise was already false when written.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")
_STYLES = (_FRONTEND / "styles.css").read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract_function(src: str, name: str) -> str:
    for marker in (f"async function {name}(", f"function {name}("):
        start = src.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in app.js")
    depth = 0
    i = src.index("{", start)
    while True:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
        i += 1


def _render(running_js: str) -> str:
    script = "\n".join(
        [
            "const BACKTEST_STALE_SECONDS = 120;",
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            _extract_function(_APP_JS, "formatBacktestEta"),
            _extract_function(_APP_JS, "formatProgressStaleness"),
            _extract_function(_APP_JS, "renderAgentRunningBody"),
            f"console.log(JSON.stringify(renderAgentRunningBody("
            f"{{agent_id: 'a1'}}, {running_js})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_card_shows_step_and_percent_when_known():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "84/240" in html
    assert "35%" in html


def test_card_bar_is_determinate_when_step_is_known():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "is-determinate" in html
    assert "width: 35%" in html


def test_card_falls_back_to_indeterminate_before_the_first_step():
    """Not an error state: the progress file does not exist for the opening
    moments of every run."""
    html = _render("{elapsedSeconds: 2}")
    assert "is-determinate" not in html
    assert "Backtesting" in html


def test_card_shows_eta():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "left" in html


def test_card_warns_when_progress_is_stale():
    html = _render(
        "{elapsedSeconds: 600, step: 84, totalSteps: 240, updatedAt: Date.now() - 300000}"
    )
    assert "No progress for 5m" in html


def test_card_is_silent_when_progress_is_fresh():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "No progress for" not in html


def test_determinate_bar_keeps_a_reduced_motion_fallback():
    assert "prefers-reduced-motion" in _STYLES
    assert "agent-card-running-bar--is-determinate" in _STYLES or "is-determinate" in _STYLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py -v`
Expected: FAIL — `assert "84/240" in html`

- [ ] **Step 3: Rewrite `renderAgentRunningBody`**

Replace `app.js:775-795` (the JSDoc block and the function) with:

```js
/**
 * Card body for an agent with a backtest in flight.
 *
 * The bar is determinate whenever the engine has published a step: engine.py's
 * `_publish_live_progress` writes step/total_steps every step and the status
 * endpoint surfaces them. (The 2026-07-29 spec specified an indeterminate bar
 * "since no honest completion estimate exists" -- that was already untrue; see
 * the 2026-08-01 spec.) It falls back to indeterminate before the first step,
 * which is a normal state on every run, not an error.
 */
function renderAgentRunningBody(agent, running) {
  const step = Number(running.step);
  const total = Number(running.totalSteps);
  const determinate =
    Number.isFinite(step) && Number.isFinite(total) && total > 0 && step > 0;
  const pct = determinate ? Math.min(99, Math.round((100 * step) / total)) : null;
  const eta = determinate
    ? formatBacktestEta(running.elapsedSeconds, step, total)
    : null;
  const stale = running.updatedAt
    ? formatProgressStaleness((Date.now() - Number(running.updatedAt)) / 1000)
    : null;

  const stepLabel = determinate ? `${step}/${total}` : '';
  // Built from raw values and escaped once at the interpolation site below --
  // escaping here as well would double-encode.
  const detail = [
    determinate ? `${pct}%` : null,
    eta,
    `${formatBacktestElapsed(running.elapsedSeconds)} elapsed`,
  ]
    .filter(Boolean)
    .join(' · ');

  return `
    <div class="agent-card-running">
      <div class="agent-card-running-head">
        <span class="agent-card-running-dot" aria-hidden="true"></span>
        <span class="agent-card-running-label">Backtesting…</span>
        <span class="agent-card-running-step" data-running-step="${escapeHtml(agent.agent_id)}">${escapeHtml(stepLabel)}</span>
        <span class="agent-card-running-elapsed" data-running-elapsed="${escapeHtml(agent.agent_id)}">${escapeHtml(formatBacktestElapsed(running.elapsedSeconds))}</span>
      </div>
      <div class="agent-card-running-track" role="progressbar" aria-label="Backtest in progress"${determinate ? ` aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100"` : ''}>
        <div class="agent-card-running-bar${determinate ? ' is-determinate' : ''}"${determinate ? ` style="width: ${pct}%"` : ''}></div>
      </div>
      <p class="agent-card-running-detail" data-running-detail="${escapeHtml(agent.agent_id)}">${escapeHtml(detail)}</p>
      ${stale ? `<p class="agent-card-running-stale">${escapeHtml(stale)}</p>` : ''}
    </div>
    ${renderAgentAllocatedCapitalHero(agent)}`;
}
```

- [ ] **Step 4: Capture progress in the poller**

Declare the store **next to `lastRenderedRunningKey` (`app.js:3351`)**, not next to the poller. `getAgentBacktestRunning` (`:3340`) reads it, and a `let` declared 1500 lines below its reader is a temporal-dead-zone trap waiting for the first person who calls that function during script evaluation:

```js
/** Latest polled progress for the single in-flight run.
 *  backtest_status is one process-global dict on the server, so at most one
 *  backtest runs at a time and a single shared object is correct. */
let liveBacktestProgress = null;
```

Then in `ensureBacktestPolling` (`app.js:4859`), the `status.running` branch already computes `step`, `total` and `stepPct` at `:4879-4883`.

Then, inside the `if (status.running) {` branch, immediately after the existing `const stepPct = ...` assignment, insert:

```js
                liveBacktestProgress = Number.isFinite(step) && step > 0
                    ? { step, totalSteps: total, updatedAt: Number(status.progress?.progress_updated_at) * 1000 || Date.now() }
                    : null;
```

And in the `else` branch (`:4902`, run finished), immediately after `stopBacktestPolling();`, add:

```js
                liveBacktestProgress = null;
```

- [ ] **Step 4b: Close the timeout path — it is the one that leaks**

The finished branch above already clears every running entry (`Object.keys(readRunningBacktests()).forEach(clearAgentBacktestRunning);`, `:4907`). **The 10-minute timeout branch does not.** In `if (attempts >= maxAttempts) {` (`app.js:4942`), after `liveBacktestRunId = null;` (`:4953`), add the same two lines:

```js
                Object.keys(readRunningBacktests()).forEach(clearAgentBacktestRunning);
                liveBacktestProgress = null;
```

Why this is not optional once Step 5 lands. `liveBacktestProgress` is a single global merged into *every* entry in the running map. Today a timed-out entry only ever showed a wrong elapsed timer, and `getAgentBacktestRunning` self-expires it at `BACKTEST_POLL_MAX_SECONDS` (600s) anyway. After Step 5, that same orphaned entry renders **whatever the next run publishes**: agent A times out, agent B starts, and A's card shows B's step, percent and ETA until A's entry ages out. Clearing on timeout — the convention the finished branch already follows — closes it.

The `catch` at `:4955` needs nothing: it only logs, the interval keeps ticking, and the next successful poll re-establishes state.

- [ ] **Step 5: Merge progress into the running entry**

In `getAgentBacktestRunning` (`app.js:3340`), change the return to fold in the polled progress:

```js
    return { ...entry, ...(liveBacktestProgress || {}), elapsedSeconds: Math.floor(elapsed) };
```

`backtest_status` is a single process-global on the server, so at most one backtest runs at a time and one shared progress object is correct **for the run that is actually live**. The spread is unconditional, though, so it also lands on any *stale* entry still in the map — which is exactly why Step 4b exists. With Step 4b in place the only remaining window is a tab that was closed mid-run and reopened, where the entry survives in `sessionStorage` with no poller to clear it; `getAgentBacktestRunning`'s 600s expiry bounds that, and it is not worth per-agent progress keying for a server that runs one backtest at a time.

- [ ] **Step 6: Patch step and detail in place**

In `refreshRunningAgentCards` (`app.js:3362`), extend the patch loop. Replace the `elapsedNodes` block with:

```js
    const elapsedNodes = document.querySelectorAll('[data-running-elapsed]');
    const stepNodes = document.querySelectorAll('[data-running-step]');
    const detailNodes = document.querySelectorAll('[data-running-detail]');
    Object.keys(running).forEach((agentId) => {
        const entry = getAgentBacktestRunning(agentId);
        if (!entry) return;
        elapsedNodes.forEach((el) => {
            if (el.getAttribute('data-running-elapsed') !== agentId) return;
            el.textContent = formatBacktestElapsed(entry.elapsedSeconds);
        });
        const step = Number(entry.step);
        const total = Number(entry.totalSteps);
        const determinate = Number.isFinite(step) && Number.isFinite(total) && total > 0 && step > 0;
        if (!determinate) return;
        const pct = Math.min(99, Math.round((100 * step) / total));
        stepNodes.forEach((el) => {
            if (el.getAttribute('data-running-step') !== agentId) return;
            el.textContent = `${step}/${total}`;
        });
        detailNodes.forEach((el) => {
            if (el.getAttribute('data-running-detail') !== agentId) return;
            const eta = formatBacktestEta(entry.elapsedSeconds, step, total);
            el.textContent = [`${pct}%`, eta, `${formatBacktestElapsed(entry.elapsedSeconds)} elapsed`]
                .filter(Boolean)
                .join(' · ');
        });
    });
```

The bar width and the staleness line still move on the next full re-render; the per-second patch covers the numbers that change every tick.

- [ ] **Step 7: Add the styles**

Append to `dashboard/frontend/styles.css`:

```css
.agent-card-running-step {
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    opacity: 0.75;
}

.agent-card-running-detail {
    margin: 6px 0 0;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
    opacity: 0.7;
}

.agent-card-running-stale {
    margin: 4px 0 0;
    font-size: 12px;
    color: #fbbf24;
}

/* Determinate: width is data, so the indeterminate sweep must not also run. */
.agent-card-running-bar.is-determinate {
    animation: none;
    transition: width 400ms ease;
}

@media (prefers-reduced-motion: reduce) {
    .agent-card-running-bar.is-determinate { transition: none; }
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py dashboard/backend/tests/test_my_agents_card_ui.py -v`
Expected: all passed. `test_my_agents_card_ui.py` must stay green — it covers the same card.

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css dashboard/backend/tests/test_backtest_progress_card.py
git commit -m "feat(backtest): show step, percent, ETA and staleness on the agent card"
```

---

### Task 8: Keep the Backtest-tab panel consistent

Two surfaces showing different numbers for one run is worse than one surface showing none.

**Files:**
- Modify: `dashboard/frontend/app.js:4607-4625` (`updateBacktestRunProgress`)
- Modify: `dashboard/frontend/app.js` (the `viewingLive` call site, `:4891-4895`)
- Test: `dashboard/backend/tests/test_backtest_progress_card.py` (extend)

**Interfaces:**
- Consumes: `formatBacktestEta`, `formatProgressStaleness` (Task 6).
- Produces: nothing new.

The panel keeps its existing `viewingLive` gate. This changes *what* it shows, not *when*.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_backtest_progress_card.py`:

```python
def test_run_panel_shares_the_card_eta_helper():
    """One run, two surfaces. Divergent numbers are worse than one blank surface."""
    fn = _extract_function(_APP_JS, "updateBacktestRunProgress")
    assert "formatBacktestEta(" in fn


def test_run_panel_reports_staleness():
    fn = _extract_function(_APP_JS, "updateBacktestRunProgress")
    assert "formatProgressStaleness(" in fn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py -v -k panel`
Expected: FAIL — `assert "formatBacktestEta(" in fn`

- [ ] **Step 3: Implement**

Replace `updateBacktestRunProgress` (`app.js:4607-4625`) with:

```js
function updateBacktestRunProgress({
    elapsedSeconds,
    message = '',
    maxSeconds = BACKTEST_POLL_MAX_SECONDS,
    stepPct = null,
    step = null,
    totalSteps = null,
    updatedAt = null,
} = {}) {
    const elapsedEl = document.getElementById('backtestRunElapsed');
    const messageEl = document.getElementById('backtestRunProgressMessage');
    const barEl = document.getElementById('backtestRunProgressBar');

    if (elapsedEl && elapsedSeconds !== undefined && elapsedSeconds !== null) {
        const elapsed = Math.max(0, Number(elapsedSeconds) || 0);
        elapsedEl.textContent = formatBacktestElapsed(elapsed);
    }
    if (messageEl && message) {
        // Same two derived facts the card shows, from the same helpers, so the
        // two surfaces can never disagree about one run.
        const eta = formatBacktestEta(elapsedSeconds, step, totalSteps);
        const stale = updatedAt
            ? formatProgressStaleness((Date.now() - Number(updatedAt)) / 1000)
            : null;
        messageEl.textContent = [message, eta, stale].filter(Boolean).join(' · ');
    }
    if (barEl) {
        const pct = Number.isFinite(stepPct)
            ? Math.min(99, Math.round(stepPct))
            : (elapsedSeconds !== undefined && elapsedSeconds !== null
                ? Math.min(95, Math.round((Math.max(0, Number(elapsedSeconds) || 0) / maxSeconds) * 100))
                : null);
        if (pct != null) barEl.style.width = `${pct}%`;
    }
}
```

- [ ] **Step 4: Pass the new fields at the live call site**

In `ensureBacktestPolling`, the `if (viewingLive) {` block (`app.js:4891`) currently calls `updateBacktestRunProgress({ elapsedSeconds, message, stepPct })`. Extend it:

```js
                    updateBacktestRunProgress({
                        elapsedSeconds: displayElapsed,
                        message: status.message || 'Backtest is running…',
                        stepPct,
                        step,
                        totalSteps: total,
                        updatedAt: liveBacktestProgress?.updatedAt || null,
                    });
```

Leave the other **six** call sites unchanged. Verify the inventory before editing — `command grep -n "updateBacktestRunProgress(" dashboard/frontend/app.js` should list the definition at `:4607` plus seven callers:

| Site | Path | Why it stays |
|---|---|---|
| `:4806` | attach to an already-running backtest | no `step`/`total` in scope yet; the next poll tick paints them |
| `:4831` | launch, elapsed 0 | nothing to estimate from |
| **`:4891`** | **live poll — the one this step edits** | — |
| `:4917` | error | an ETA on a failed run is noise |
| `:4926` | completion | it is done |
| `:4947` | 10-minute timeout | the estimate is what failed |
| `:5491` | `runBacktest` re-entry | same as `:4831` |

All six are safe unedited because every new parameter defaults to `null` and `formatBacktestEta` returns `null` on missing input — the omission is a behaviour-preserving default, not an oversight to fix later.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtest_progress_card.py -v`
Expected: all passed

- [ ] **Step 6: Run the full suite**

Run: `pytest dashboard/backend/tests/ -v`
Expected: green (bar the documented `test_deleted_shim_is_not_importable` stale-bytecode case — if that fails, run `rm -rf dashboard/backend/engines dashboard/backend/services` and re-run)

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.js dashboard/backend/tests/test_backtest_progress_card.py
git commit -m "feat(backtest): mirror ETA and staleness in the run panel"
```

---

**Phase B exit:** full suite green. Open PR `feat(backtest): progress visibility`. The PR body must state that this deliberately reverses the 2026-07-29 non-goal and why the original premise was wrong, or a reviewer who knows that spec will read it as a regression.

---

# Phase C — Landing below-fold rework (PR 3)

Branch: `feat/landing-value-band`

**Highest-risk phase.** `dashboard/frontend/index.html` is **not** Vite output — it is a hand-patched artifact carrying ~370 lines Vite cannot emit. Copying `dist/index.html` over it silently kills every landing CTA with **no console error and a page that still loads**. Issue #225.

Audience for all copy: **has money to spare, not deep into trading.** Wants to test a market idea at low cost and friction.

---

### Task 9: The "Why you should care" band

**Files:**
- Create: `dashboard/landing/src/components/home/WhyCare.tsx`
- Modify: `dashboard/landing/src/pages/landing-page.tsx`
- Modify: `dashboard/landing/src/components/home/Talk.tsx:10` (move the scroll anchor)
- Test: `dashboard/backend/tests/test_landing_value_band.py` (create)

**Interfaces:**
- Consumes: `PRIMARY_LANDING_CTA` from `@/lib/cta`.
- Produces: `WhyCare` — default-exported named component, rendered between `<Hero />` and `<Talk />`.

Hero stays frozen. The band reuses the `01/02/03` mono-label pattern from Talk/Test/Race so it reads as one system, not a bolted-on marketing block.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_landing_value_band.py`:

```python
"""The landing states what the product is for, above the three-act narrative.

A tester could not tell what the platform's core advantage was without clicking
in and exploring. The narrative sections (Talk/Test/Race) each describe an act
but never state the problem being solved or who it is for.

Also pins the two claims that must never appear. Both contradict the code: no
order-submission route exists on any surface, and ROBINHOOD_EXECUTE defaults to
false. docs/source/lab/operating_modes.rst says the same. Copy that promises
either would be the fabricated-Performance-Drivers failure again.
"""

from pathlib import Path

_LANDING_SRC = Path(__file__).resolve().parents[2] / "landing" / "src"
_BAND = _LANDING_SRC / "components" / "home" / "WhyCare.tsx"
_PAGE = _LANDING_SRC / "pages" / "landing-page.tsx"


def test_band_component_exists():
    assert _BAND.is_file()


def test_band_is_rendered_between_hero_and_talk():
    page = _PAGE.read_text(encoding="utf-8")
    assert "WhyCare" in page
    assert page.index("<Hero />") < page.index("<WhyCare />") < page.index("<Talk />")


def test_band_states_the_problem_before_the_features():
    body = _BAND.read_text(encoding="utf-8")
    assert "Testing it properly is the expensive part" in body


def test_band_covers_the_three_acts():
    body = _BAND.read_text(encoding="utf-8")
    for heading in ("Describe it in plain English", "Prove it on real market data", "See how it ranks"):
        assert heading in body


def test_band_names_the_uncovered_capabilities():
    """Model choice and external agents are real and were absent from the landing."""
    body = _BAND.read_text(encoding="utf-8")
    assert "Pick the model" in body
    assert "Bring your own agent" in body


def test_band_makes_no_paper_trading_claim():
    body = _BAND.read_text(encoding="utf-8").lower()
    assert "paper trading" not in body
    assert "paper-trade" not in body


def test_band_makes_no_real_capital_claim():
    body = _BAND.read_text(encoding="utf-8").lower()
    for phrase in ("real capital", "real money", "go live", "trade live"):
        assert phrase not in body


def test_hero_scroll_anchor_still_resolves():
    """Hero.tsx scrolls to #landing-stats. If the band takes that anchor, Talk
    must give it up -- two elements with one id is a silent mis-scroll."""
    sources = [p.read_text(encoding="utf-8") for p in _LANDING_SRC.rglob("*.tsx")]
    total = sum(s.count('id="landing-stats"') for s in sources)
    assert total == 1, f"expected exactly one #landing-stats anchor, found {total}"
    assert 'id="landing-stats"' in _BAND.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v`
Expected: FAIL — `assert _BAND.is_file()`

- [ ] **Step 3: Create the component**

Create `dashboard/landing/src/components/home/WhyCare.tsx`:

```tsx
import { MessageSquare, LineChart, Trophy, Cpu, Code2, Hash } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PRIMARY_LANDING_CTA } from "@/lib/cta";

const ACTS = [
  {
    label: "01",
    icon: MessageSquare,
    title: "Describe it in plain English",
    body: "No code, no formulas. Write how you want to trade the way you would explain it to a person.",
  },
  {
    label: "02",
    icon: LineChart,
    title: "Prove it on real market data",
    body: "Real prices, real market hours, measured against buy-and-hold and the index — so you learn whether the idea was good, not whether it felt good.",
  },
  {
    label: "03",
    icon: Trophy,
    title: "See how it ranks",
    body: "Same window, same rules as everyone else's agents.",
  },
] as const;

const EXTRAS = [
  {
    icon: Cpu,
    title: "Pick the model",
    body: "Same idea, different brains: Claude, GPT, Gemini, DeepSeek, Qwen.",
  },
  {
    icon: Code2,
    title: "Bring your own agent",
    body: "A Python SDK and an API, if you would rather write the code.",
  },
  {
    icon: Hash,
    title: "Talk to it on Discord",
    body: "If you would rather just chat.",
  },
] as const;

export function WhyCare() {
  return (
    <section id="why" className="py-24 scroll-mt-40">
      {/* Hero's scroll target — moved here from Talk so the first scroll lands
          on the value proposition rather than past it. Hero.tsx still anchors
          to #landing-stats; do not rename without updating it. */}
      <div id="landing-stats" className="h-0 w-0 overflow-hidden" aria-hidden="true" />

      <div className="container mx-auto px-6">
        <div className="max-w-3xl mb-14">
          <h2 className="text-3xl md:text-4xl font-bold mb-4 tracking-tight">
            You have an idea about the market.
            <span className="block text-[#22d3ee]">Testing it properly is the expensive part.</span>
          </h2>
          <p className="text-foreground/80 text-lg">
            Normally that means writing code, buying data, and waiting months to find out you
            were wrong. Here it costs one sentence and a few minutes.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {ACTS.map(({ label, icon: Icon, title, body }) => (
            <div key={label}>
              <p className="text-sm font-mono tracking-wide text-primary mb-3">{label}</p>
              <Icon className="w-6 h-6 text-primary mb-3" aria-hidden="true" />
              <h3 className="text-lg font-bold mb-2">{title}</h3>
              <p className="text-sm text-foreground/70 leading-relaxed">{body}</p>
            </div>
          ))}
        </div>

        <div className="grid sm:grid-cols-3 gap-6 pt-10 border-t border-border">
          {EXTRAS.map(({ icon: Icon, title, body }) => (
            <div key={title} className="flex gap-3">
              <Icon className="w-5 h-5 text-primary shrink-0 mt-0.5" aria-hidden="true" />
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-1">{title}</h4>
                <p className="text-sm text-foreground/60 leading-relaxed">{body}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-14">
          <Button
            size="lg"
            type="button"
            data-landing-auth={PRIMARY_LANDING_CTA.authMode}
            className="bg-primary text-primary-foreground hover:bg-primary/90 text-base h-12 px-8"
          >
            {PRIMARY_LANDING_CTA.label}
          </Button>
        </div>
      </div>
    </section>
  );
}
```

This adds a **7th** `data-landing-auth` emitter (currently 6: Hero, Navbar, Talk, Test, Race, FooterCTA). The bundle guard counts emitters automatically and will fail until Task 12 rebuilds — that is the guard working, not a bug.

- [ ] **Step 4: Wire it into the page**

Replace `dashboard/landing/src/pages/landing-page.tsx` entirely:

```tsx
import { Navbar } from "../components/home/Navbar";
import { MarketTicker } from "../components/home/MarketTicker";
import { Hero } from "../components/home/Hero";
import { WhyCare } from "../components/home/WhyCare";
import { Talk } from "../components/home/Talk";
import { Test } from "../components/home/Test";
import { Race } from "../components/home/Race";
import { FooterCTA } from "../components/home/FooterCTA";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <div className="landing-chrome">
        <Navbar />
        <MarketTicker />
      </div>
      <main>
        <Hero />
        <WhyCare />
        <Talk />
        <Test />
        <Race />
      </main>
      <FooterCTA />
    </div>
  );
}
```

- [ ] **Step 5: Release the anchor from Talk**

In `dashboard/landing/src/components/home/Talk.tsx`, delete lines 9-10:

```tsx
      {/* Hero scroll target — do not remove; Hero.tsx still anchors here */}
      <div id="landing-stats" className="h-0 w-0 overflow-hidden" aria-hidden="true" />
```

Two elements sharing one id would make the scroll land on whichever the browser finds first — a silent mis-scroll, not an error.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add dashboard/landing/src/components/home/WhyCare.tsx dashboard/landing/src/pages/landing-page.tsx dashboard/landing/src/components/home/Talk.tsx dashboard/backend/tests/test_landing_value_band.py
git commit -m "feat(landing): add a value-proposition band below the hero"
```

---

### Task 10: Reframe `01 — Talk` away from Discord-first

The only place this round changes an existing deliberate decision rather than filling a gap. Justification: on-site plain-English authoring has existed since the agent editor shipped (`dashboard/frontend/app.html:972` — *"Tell the agent how to trade in plain language"*), and "join a Discord server" is exactly the friction this audience will not clear.

Structure, the `DiscordMock` visual and the `#talk` anchor all stay. Copy and CTA emphasis change.

**Files:**
- Modify: `dashboard/landing/src/components/home/Talk.tsx`
- Test: `dashboard/backend/tests/test_landing_value_band.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_landing_value_band.py`:

```python
_TALK = _LANDING_SRC / "components" / "home" / "Talk.tsx"


def test_talk_leads_with_the_on_site_path():
    """The heading no longer sells Discord as the way in. On-site plain-English
    authoring has existed since the agent editor shipped (app.html:972)."""
    body = _TALK.read_text(encoding="utf-8")
    assert "Talk to agents on Discord" not in body
    assert "Describe your idea" in body


def test_talk_keeps_discord_as_an_alternative():
    """Reframed, not removed -- the Discord path works and some users prefer it."""
    assert "Discord" in _TALK.read_text(encoding="utf-8")


def test_talk_keeps_its_anchor_and_visual():
    body = _TALK.read_text(encoding="utf-8")
    assert 'id="talk"' in body
    assert "<DiscordMock />" in body


def test_talk_has_exactly_one_section_label():
    """Step 3's replacement block *re-includes* the `01 — Talk` mono-label, so
    pasting it below the existing one stacks two identical labels. Every other
    assertion in this file is a substring check and would stay green."""
    assert _TALK.read_text(encoding="utf-8").count("01 — Talk") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v -k talk`
Expected: FAIL — `assert "Talk to agents on Discord" not in body`

- [ ] **Step 3: Implement**

In `Talk.tsx`, replace the contiguous span **from the `01 — Talk` mono-label `<p>` at `:15` through the closing `</ol>` at `:33`** — that is *four* elements (mono-label `<p>`, `<h2>`, body `<p>`, `<ol>`), not three. The block below **reproduces the mono-label unchanged**; keeping the original as well ships two stacked `01 — Talk` lines, which is what `test_talk_has_exactly_one_section_label` exists to catch. Leave `:34-41` (the `<Button>`) alone.

```tsx
            <p className="text-base md:text-lg font-mono tracking-wide text-primary mb-3">01 — Talk</p>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">Describe your idea, in a sentence</h2>
            <p className="text-foreground/80 mb-8 text-lg">
              Write how you want to trade. The agent follows it, hour by hour.
            </p>
            <ol className="space-y-3 mb-8 text-sm text-foreground/80">
              <li className="flex items-start gap-3">
                <MessageSquare className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">1.</span> Write your trading instruction in plain language</span>
              </li>
              <li className="flex items-start gap-3">
                <Bot className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">2.</span> Pick a model and how much simulated cash it gets</span>
              </li>
              <li className="flex items-start gap-3">
                <Hash className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                <span><span className="text-foreground font-medium">3.</span> Prefer chat? The same agent answers on Discord</span>
              </li>
            </ol>
```

The existing `MessageSquare`, `Bot` and `Hash` imports at the top of the file all remain in use — do not touch the import line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/landing/src/components/home/Talk.tsx dashboard/backend/tests/test_landing_value_band.py
git commit -m "feat(landing): lead Talk with the on-site path, Discord as alternative"
```

---

### Task 11: Fix the dead footer links

`Terms`, `Privacy` and `Documentation` are all `href="#"` (`FooterCTA.tsx:27-29`) — flagged in the 2026-07-25 UI audit, still open. `Documentation` gets the real docs site. `Terms` and `Privacy` have no destination, so they are **removed**: a link to nowhere is worse than an absent one, and inventing placeholder pages is out of scope.

**Files:**
- Modify: `dashboard/landing/src/components/home/FooterCTA.tsx`
- Test: `dashboard/backend/tests/test_landing_value_band.py` (extend)

- [ ] **Step 1: Verify the docs destination resolves**

Run: `curl -sS -o /dev/null -w '%{http_code}\n' https://finagent-orchestration.readthedocs.io/en/latest/`
Expected: `200`. If it is not 200, stop and report — replacing a dead link with another dead link is not a fix. This exact URL is the one the README's docs **badge** publishes (`README.md:15`); the prose link at `README.md:138` gives the bare host, which redirects to the same place. Use the badge's `/en/latest/` form so the footer link does not depend on a redirect.

- [ ] **Step 2: Write the failing test**

Append to `dashboard/backend/tests/test_landing_value_band.py`:

```python
_FOOTER = _LANDING_SRC / "components" / "home" / "FooterCTA.tsx"


def test_footer_has_no_dead_links():
    """Three href="#" anchors shipped since the 2026-07-25 audit. A link that
    goes nowhere costs more trust than an absent one."""
    assert 'href="#"' not in _FOOTER.read_text(encoding="utf-8")


def test_footer_documentation_points_at_the_docs_site():
    body = _FOOTER.read_text(encoding="utf-8")
    assert "finagent-orchestration.readthedocs.io" in body


def test_footer_external_link_is_safe():
    """target=_blank without rel=noopener hands the opener window to the target."""
    body = _FOOTER.read_text(encoding="utf-8")
    if 'target="_blank"' in body:
        assert "noopener" in body
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v -k footer`
Expected: FAIL — `assert 'href="#"' not in ...`

- [ ] **Step 4: Implement**

In `FooterCTA.tsx`, replace lines 26-30 (the `<div className="flex gap-6 ...">` block) with:

```tsx
          <div className="flex gap-6 mt-4 md:mt-0">
            <a
              href="https://finagent-orchestration.readthedocs.io/en/latest/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground"
            >
              Documentation
            </a>
          </div>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_landing_value_band.py -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add dashboard/landing/src/components/home/FooterCTA.tsx dashboard/backend/tests/test_landing_value_band.py
git commit -m "fix(landing): point Documentation at the docs site, drop dead links"
```

---

### Task 12: Rebuild and hand-patch the shipped bundle

**The dangerous one.** Do not shortcut any step.

**Files:**
- Modify: `dashboard/frontend/index.html` (asset refs only — never wholesale)
- Modify: `dashboard/frontend/assets/` (new hashed bundles in, superseded ones deleted)

- [ ] **Step 1: Record the current asset refs**

```bash
command grep -oE '/(assets|images)/[^"?#]+' dashboard/frontend/index.html | sort -u
```

Keep this output. Step 5 replaces exactly these `assets/*.js` and `assets/*.css` names, nothing else.

- [ ] **Step 2: Build**

```bash
cd dashboard/landing && npm ci && npm run build
```

Expected: ~90s install, ~3s build. Verified toolchain: node 22, npm 11, vite 6.4.3.

- [ ] **Step 3: Copy the new assets in, delete the superseded ones**

```bash
cd /mnt/d/github/agent-trading-lab
ls dashboard/landing/dist/public/assets/
cp dashboard/landing/dist/public/assets/index-*.js dashboard/landing/dist/public/assets/index-*.css dashboard/frontend/assets/
```

Then delete the *old* `index-*.js` and `index-*.css` from `dashboard/frontend/assets/` — the ones named in Step 1. `test_no_orphaned_assets` fails if a superseded bundle is left behind.

- [ ] **Step 4: DO NOT copy `dist/index.html`**

The shipped `dashboard/frontend/index.html` carries ~370 hand-written lines Vite cannot emit: the auth-gate script, `#landingAuthModal`, `<style id="landing-auth-patch">`, and the delegated `[data-landing-auth]` handler. Overwriting it turns all seven CTAs into dead buttons with **no console error** and a page that still renders.

- [ ] **Step 5: Edit only the asset references**

In `dashboard/frontend/index.html`, change the `<script src="/assets/index-XXXX.js">` and `<link href="/assets/index-XXXX.css">` hashes to the new filenames from Step 3. Change nothing else.

- [ ] **Step 6: Run the integrity guard**

Run: `pytest dashboard/backend/tests/test_frontend_bundle_integrity.py -v`
Expected: 6 passed. Specifically:
- `test_shipped_bundle_has_one_cta_per_landing_source_emitter` must report **7** in both source and bundle (was 6; Task 9's band adds one). The count is derived from the source at run time, not hardcoded, so no assertion needs editing.
- `test_hand_written_auth_layer_survives_a_bundle_refresh` proves Step 4 was honoured
- `test_no_orphaned_assets` proves Step 3's deletion happened

- [ ] **Step 6b: Correct the guard's stale prose**

Two docstrings in `test_frontend_bundle_integrity.py` (`:134` and `:187`) say **"six"** `data-landing-auth` CTAs. The assertions are dynamic and stay green, but the prose becomes wrong the moment Task 9 lands. Change both to **seven**. Comment-only edit; commit it with Step 10.

- [ ] **Step 7: Verify the rebuild is reproducible**

```bash
cd dashboard/landing && npm run build && sha256sum dist/public/assets/index-*.js
sha256sum ../frontend/assets/index-*.js
```

Expected: identical hashes, and the emitted filename equals the committed one. A mismatch means the committed bundle does not correspond to the committed source.

- [ ] **Step 8: Browser-verify (no npm needed)**

```bash
cd /mnt/d/github/agent-trading-lab
CHECK_DB="$YOUR_SCRATCHPAD/landing-check.db"   # substitute your own session scratchpad
mkdir -p "$(dirname "$CHECK_DB")"
DATABASE_PATH="$CHECK_DB" python -m dashboard.backend.app
```

`DATABASE_PATH` must point at a temp file **that you create the parent directory for** — the default is the committed prod seed DB, and merely importing the app runs lazy `CREATE TABLE`/`ALTER` against it. Do not paste a scratchpad path from this plan or any transcript: those belong to the session that wrote them and will not exist for you. Then drive `http://localhost:8000/` with `~/.venvs/htmlpdf/bin/python` (Playwright + Chromium already installed) and confirm: the band renders between hero and Talk, the first scroll lands on it, and **clicking a CTA opens the signup modal**. A `/_vercel/insights/script.js` 404 is expected off Vercel — not a defect.

- [ ] **Step 9: Run the full suite**

Run: `pytest dashboard/backend/tests/ -v`
Expected: green

- [ ] **Step 10: Commit**

```bash
git add dashboard/frontend/index.html dashboard/frontend/assets \
  dashboard/backend/tests/test_frontend_bundle_integrity.py
git commit -m "chore(landing): rebuild bundle for the value band"
```

---

**Phase C exit:** full suite green, CTA click verified in a browser. Open PR `feat(landing): below-fold value proposition`.

---

## Self-review — spec coverage

| Spec section | Task |
|---|---|
| A1 pending button state | Task 2 |
| A2 confirm on POST resolution | Task 2 |
| A3 success toast | Task 1 |
| A4 locate the new agent | Task 3 |
| A5 latency investigation | Task 4 |
| A6 error handling preserved | Task 2 (`finally` test) |
| A7 tests | Tasks 1-3 |
| B1 backend `progress_updated_at` | Task 5 |
| B2 determinate card bar | Task 7 |
| B3 in-place patching | Task 7 |
| B4 ETA | Task 6, Task 7 |
| B5 staleness | Task 6, Task 7 |
| B6 surface consistency | Task 8 |
| B7 cancel deferred | Issue #273 — no task, by design |
| B8 tests | Tasks 5-8 |
| C1 structure | Task 9 |
| C2 band copy | Task 9 |
| C3 Talk reframe | Task 10 |
| C4 forbidden claims | Task 9 (absence tests) |
| C5 build discipline | Task 12 |
| C6 footer links | Task 11 |
| C7 docs — not implemented | Global constraint; no task, by design |
| C8 tests | Tasks 9-12 |

## Known follow-ups

- **Issue #273** — cancel a running backtest. Out of scope.
- **`docs/landing-narrative-copy.md`** is superseded by Phase C and is deliberately not updated by these PRs. The requester coordinates doc updates separately.
