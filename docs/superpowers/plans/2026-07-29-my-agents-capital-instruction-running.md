# My Agents: unified capital, optional instruction, visible backtest state — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate both allocated-capital fields into one Configure card, let an empty Trading Instruction fall back to the platform default, and make a launched backtest visible on the My Agents page.

**Architecture:** A new persisted `backtest_allocation` column on `external_agents` turns per-run backtest capital into a saved agent setting; the Run Backtest modal becomes read-only for capital. The empty-instruction feature is subtractive — `PATCH {"pipeline": []}` already clears an agent server-side, so only the editor's refusal to send it is removed. The running indicator reuses the existing one-second `ensureBacktestPolling()` tick rather than adding a timer.

**Tech Stack:** FastAPI + Pydantic v2, SQLite with a Postgres twin (psycopg), vanilla JS (no build step), pytest. Frontend behavior is tested by lifting real functions out of the shipped source and running them under `node` — see `tests/test_frontend_portfolio_panel.py` for the idiom.

**Spec:** `docs/superpowers/specs/2026-07-29-my-agents-capital-instruction-running-design.md`

## Global Constraints

- **Every store change lands in BOTH twins.** `domain/agents/repository.py` (SQLite) and `domain/agents/repository_postgres.py` (Postgres). `tests/test_store_twin_parity.py` statically checks method sets, signature triples, declared columns, and lazy migrations. A column added to one twin only 500s every Configure save on prod while the SQLite suite stays green (PR #227).
- **`_public_agent` is shared.** `repository_postgres.py:24` imports it from `repository.py`. Edit it once, in the SQLite module.
- **SQLite declares new columns in the migration block only.** Its `CREATE TABLE` is the original shipped schema and is never extended — `cash_allocation`, `description`, `pipeline_config` all arrive via the `PRAGMA table_info` probe. Postgres declares new columns in **both** its `CREATE TABLE` and an `ADD COLUMN IF NOT EXISTS`. The parity test unions both sources, so this asymmetry is correct and expected.
- **Capital limits, verbatim:** paper sleeve `0`–`3000` (`MAX_AGENT_CASH_ALLOCATION`), backtest `1`–`10000` (`MAX_BACKTEST_INITIAL_CAPITAL`), default `1000` (`DEFAULT_AGENT_CASH_ALLOCATION`). Backend constants live in `domain/backtesting/constants.py`; the frontend mirrors them at `app.js:142-145`.
- **`backtest_allocation` never touches the portfolio ledger.** It is simulated money. Only `cash_allocation` routes through `portfolio_service.check_agent_allocation` / `set_agent_allocation`.
- **Resolution order everywhere:** `agent.backtest_allocation` → `agent.cash_allocation` → `1000`, clamped to `10000`.
- **Copy strings, verbatim** (used by guard tests):
  - `"Leave this empty and the agent uses the platform's default trading strategy."`
  - `"See the default instruction"`
  - `"Saved — using the default trading instruction."` (em dash, U+2014)
  - `"Paper trading is coming soon"`
  - Field labels: `"Paper Trading"`, `"Backtesting"`, card heading `"Allocated Capital"`
- **Do not edit `docs/source/`.** User-facing RTD docs are out of scope by request; stale lines are catalogued in the spec's followup section.
- **Never `git add -A`.** The committed `dashboard/storage/data/backtest.db` is the prod seed database. Stage named paths only. Before each commit verify: `git diff --stat origin/main -- dashboard/storage/data/backtest.db` prints nothing.
- **Tests:** run from repo root as `pytest dashboard/backend/tests/ -q`. The suite is green on `main`; a red test is a real regression.

---

## File Structure

**Backend — modify**
- `dashboard/backend/domain/agents/repository.py` — SQLite store: `_public_agent` projection, lazy migration, `create_agent`, `update_agent`
- `dashboard/backend/domain/agents/repository_postgres.py` — Postgres twin: `CREATE TABLE`, `ADD COLUMN IF NOT EXISTS`, `create_agent`, `update_agent`
- `dashboard/backend/domain/agents/service.py` — passthrough on `create_agent` / `update_agent`
- `dashboard/backend/api/routers/agents.py` — Pydantic fields, create route, PATCH route

**Backend — create**
- `dashboard/backend/tests/test_agent_backtest_allocation.py` — persistence + API contract for the new field, and the `pipeline: []` clearing contract

**Frontend — modify**
- `dashboard/frontend/app.html` — Configure capital card, Run Backtest modal capital row, instruction helper copy
- `dashboard/frontend/js/agent-editor.js` — capital read/write, empty-instruction semantics, backfill removal
- `dashboard/frontend/app.js` — card rendering, running-state store, navigation target, poller hook
- `dashboard/frontend/styles.css` — capital card typography, running indicator, disabled button

**Frontend — create**
- `dashboard/backend/tests/test_my_agents_capital_ui.py` — guards for Tasks 3–4
- `dashboard/backend/tests/test_my_agents_instruction_ui.py` — guards for Task 5
- `dashboard/backend/tests/test_my_agents_card_ui.py` — guards for Tasks 6–7

---

## Task 1: Persist `backtest_allocation` on both stores

**Files:**
- Modify: `dashboard/backend/domain/agents/repository.py` (`_public_agent` ~line 66, migration block ~line 141, `create_agent` ~line 174, `update_agent` ~line 480)
- Modify: `dashboard/backend/domain/agents/repository_postgres.py` (`CREATE TABLE` ~line 83, `ALTER` block ~line 105, `create_agent` ~line 144, `update_agent` ~line 443)
- Test: `dashboard/backend/tests/test_agent_backtest_allocation.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `AgentStore.create_agent(..., backtest_allocation: Optional[float] = None)` — same signature on `PostgresAgentStore`
  - `AgentStore.update_agent(agent_id, *, ..., backtest_allocation: Any = _UNSET)` — same on the twin
  - `_public_agent(row)` dict gains key `"backtest_allocation"` (float or `None`)

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_agent_backtest_allocation.py`:

```python
"""``backtest_allocation``: a saved per-agent simulated-capital setting.

Backtest capital used to be a per-run value reseeded from the paper sleeve on
every Run Backtest modal open. Consolidating both capital fields into one
Configure card (2026-07-29) makes it a stored column, which means it has to
exist on *both* twins -- see tests/test_store_twin_parity.py for why a
one-twin column is a prod-only 500.

Unlike ``cash_allocation`` this is simulated money: it must never move the
portfolio ledger.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.domain.agents.repository as agent_store_module
import dashboard.backend.database as db_module

AgentStore = agent_store_module.AgentStore


@pytest.fixture
def store(tmp_path):
    return AgentStore(db_path=tmp_path / "agents.db")


def test_backtest_allocation_round_trips_through_create(store):
    agent = store.create_agent(name="alpha", backtest_allocation=2500)
    assert agent["backtest_allocation"] == 2500

    reread = store.get_agent(agent["agent_id"])
    assert reread["backtest_allocation"] == 2500


def test_backtest_allocation_defaults_to_none(store):
    """Existing agents have a NULL column and must keep today's behaviour."""
    agent = store.create_agent(name="legacy")
    assert agent["backtest_allocation"] is None


def test_update_agent_sets_backtest_allocation(store):
    agent = store.create_agent(name="alpha")
    updated = store.update_agent(agent["agent_id"], backtest_allocation=4000)
    assert updated["backtest_allocation"] == 4000


def test_update_agent_leaves_backtest_allocation_alone_when_omitted(store):
    """The _UNSET sentinel means 'do not touch', not 'set to None'."""
    agent = store.create_agent(name="alpha", backtest_allocation=2500)
    updated = store.update_agent(agent["agent_id"], name="renamed")
    assert updated["backtest_allocation"] == 2500
    assert updated["name"] == "renamed"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_agent_backtest_allocation.py -q`
Expected: FAIL — `TypeError: create_agent() got an unexpected keyword argument 'backtest_allocation'`

- [ ] **Step 3: Add the column to the SQLite store**

In `dashboard/backend/domain/agents/repository.py`, inside `_public_agent`, add the key immediately after `cash_allocation`:

```python
        "cash_allocation": data.get("cash_allocation"),
        "backtest_allocation": data.get("backtest_allocation"),
```

In `_init_schema`, immediately after the existing `cash_allocation` probe:

```python
        if "backtest_allocation" not in existing_columns:
            cursor.execute(
                "ALTER TABLE external_agents ADD COLUMN backtest_allocation REAL"
            )
```

In `create_agent`, add the parameter after `cash_allocation`:

```python
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
```

and extend the INSERT — column list, placeholder count, and value tuple all move together:

```python
            INSERT INTO external_agents (
                agent_id, name, session_id, api_key_hash, api_key_prefix,
                model_name, agent_type, description, cash_allocation,
                backtest_allocation,
                owner_user_id, owner_browser_session, created_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

```python
                cash_allocation,
                backtest_allocation,
                owner_user_id,
```

In `update_agent`, add the parameter after `cash_allocation`:

```python
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
```

and the set clause after the `cash_allocation` block:

```python
        if backtest_allocation is not _UNSET:
            sets.append("backtest_allocation = ?")
            params.append(backtest_allocation)
```

- [ ] **Step 4: Mirror every change into the Postgres twin**

In `dashboard/backend/domain/agents/repository_postgres.py`, add to the `CREATE TABLE` body after `cash_allocation`:

```sql
                        cash_allocation DOUBLE PRECISION,
                        backtest_allocation DOUBLE PRECISION,
```

Add a lazy migration after the `cash_allocation` one:

```python
                cur.execute(
                    "ALTER TABLE external_agents "
                    "ADD COLUMN IF NOT EXISTS backtest_allocation DOUBLE PRECISION"
                )
```

In `create_agent`, add the same parameter in the same position:

```python
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
```

and extend the INSERT identically (note `%s` placeholders, and that the count must rise from 13 to 14):

```python
                    INSERT INTO external_agents (
                        agent_id, name, session_id, api_key_hash, api_key_prefix,
                        model_name, agent_type, description, cash_allocation,
                        backtest_allocation,
                        owner_user_id, owner_browser_session, created_at, last_used_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
```

```python
                        cash_allocation,
                        backtest_allocation,
                        owner_user_id,
```

In `update_agent`:

```python
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
```

```python
        if backtest_allocation is not _UNSET:
            sets.append("backtest_allocation = %s")
            params.append(backtest_allocation)
```

Do **not** touch `_public_agent` here — this module imports it from `repository.py`.

- [ ] **Step 5: Run the new test and the parity guard**

Run: `pytest dashboard/backend/tests/test_agent_backtest_allocation.py dashboard/backend/tests/test_store_twin_parity.py -q`
Expected: PASS — all four new cases plus every parity case.

If a parity case fails naming `backtest_allocation`, a twin edit was missed; re-read the failure, which prints both column sets.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/agents/repository.py \
        dashboard/backend/domain/agents/repository_postgres.py \
        dashboard/backend/tests/test_agent_backtest_allocation.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: persist backtest_allocation on both agent stores"
```

---

## Task 2: Expose `backtest_allocation` through the agents API

**Files:**
- Modify: `dashboard/backend/domain/agents/service.py` (`update_agent` ~line 248, `create_agent` ~line 272)
- Modify: `dashboard/backend/api/routers/agents.py` (imports ~line 16, `CreateAgentBody` ~line 37, `UpdateAgentBody` ~line 61, create route ~line 105, PATCH route ~line 320)
- Test: `dashboard/backend/tests/test_agent_backtest_allocation.py` (append)

**Interfaces:**
- Consumes: `AgentStore.create_agent(..., backtest_allocation=...)` and `update_agent(..., backtest_allocation=...)` from Task 1.
- Produces: `POST /api/v1/agents` and `PATCH /api/v1/agents/{id}` both accept an optional `backtest_allocation` (1–10000); every agent JSON payload carries a `backtest_allocation` key.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_agent_backtest_allocation.py`:

```python
@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.agents as agents_api

    db_path = tmp_path / "test.db"
    test_agents = AgentStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    return TestClient(app)


def _headers():
    return {"X-Browser-Session": str(uuid.uuid4())}


def test_create_accepts_backtest_allocation(client):
    headers = _headers()
    resp = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "backtest_allocation": 5000},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 5000


def test_patch_updates_backtest_allocation(client):
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 7500},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 7500


def test_patch_backtest_allocation_alone_is_not_no_fields_to_update(client):
    """It must satisfy the 'at least one field' guard on its own."""
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 2000},
        headers=headers,
    )
    assert resp.status_code != 400


@pytest.mark.parametrize("bad", [0, -100, 10001])
def test_backtest_allocation_out_of_range_is_rejected(client, bad):
    headers = _headers()
    resp = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "backtest_allocation": bad},
        headers=headers,
    )
    assert resp.status_code == 422


def test_backtest_allocation_does_not_change_the_paper_sleeve(client):
    """Simulated money must never move the real sleeve."""
    headers = _headers()
    created = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "cash_allocation": 1000},
        headers=headers,
    ).json()["agent"]

    updated = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 9000},
        headers=headers,
    ).json()["agent"]

    assert updated["cash_allocation"] == 1000
    assert updated["backtest_allocation"] == 9000
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_agent_backtest_allocation.py -q`
Expected: FAIL — the create test asserts `5000` but gets `None`, since Pydantic drops the unknown field.

- [ ] **Step 3: Add the service passthrough**

In `dashboard/backend/domain/agents/service.py`, `update_agent` — add the parameter after `cash_allocation` and forward it:

```python
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
        live_trading_enabled: Any = _UNSET,
    ) -> Dict[str, Any]:
        agent = self.agents.update_agent(
            agent_id,
            name=name,
            model_name=model_name,
            description=description,
            pipeline=pipeline,
            cash_allocation=cash_allocation,
            backtest_allocation=backtest_allocation,
            live_trading_enabled=live_trading_enabled,
        )
```

In `create_agent`, add the parameter after `cash_allocation` and forward it:

```python
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
        seed_default_pipeline: bool = True,
```

```python
        agent = self.agents.create_agent(
            name=name,
            model_name=model_name,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type=agent_type,
            description=description,
            cash_allocation=cash_allocation,
            backtest_allocation=backtest_allocation,
        )
```

Leave the `if cash_allocation is None: cash_allocation = float(DEFAULT_AGENT_CASH_ALLOCATION)` defaulting alone — `backtest_allocation` deliberately stays `None` so existing behavior (fall back to the sleeve) is preserved.

- [ ] **Step 4: Add the API surface**

In `dashboard/backend/api/routers/agents.py`, extend the constants import:

```python
from dashboard.backend.domain.backtesting.constants import (
    DEFAULT_AGENT_CASH_ALLOCATION,
    MAX_AGENT_CASH_ALLOCATION,
    MAX_BACKTEST_INITIAL_CAPITAL,
)
```

Add to `CreateAgentBody`, after `cash_allocation`:

```python
    backtest_allocation: Optional[float] = Field(
        default=None,
        ge=1,
        le=MAX_BACKTEST_INITIAL_CAPITAL,
    )
```

Add the identical field to `UpdateAgentBody`, after its `cash_allocation`.

In the create route, forward it to the service call:

```python
        cash_allocation=cash,
        backtest_allocation=body.backtest_allocation,
    )
```

In the PATCH route, register the field with the sentinel machinery. After `cash_allocation_provided`:

```python
    backtest_allocation_provided = "backtest_allocation" in fields_set
```

Add it to the "nothing to update" guard:

```python
    if (
        body.name is None
        and body.model_name is None
        and body.description is None
        and not pipeline_provided
        and not cash_allocation_provided
        and not backtest_allocation_provided
        and not live_trading_provided
    ):
        raise HTTPException(status_code=400, detail="No fields to update")
```

Build the argument next to the existing one:

```python
    backtest_allocation_arg = (
        body.backtest_allocation if backtest_allocation_provided else _UNSET
    )
```

and pass it to the service:

```python
            cash_allocation=cash_allocation_arg,
            backtest_allocation=backtest_allocation_arg,
            live_trading_enabled=live_trading_arg,
```

**Do not** add it to the `ledger_new_amount` branch. That block is `cash_allocation`-only by design.

- [ ] **Step 5: Run the tests**

Run: `pytest dashboard/backend/tests/test_agent_backtest_allocation.py dashboard/backend/tests/test_agents_api.py -q`
Expected: PASS — all new cases plus the existing agents-API suite unchanged.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/agents/service.py \
        dashboard/backend/api/routers/agents.py \
        dashboard/backend/tests/test_agent_backtest_allocation.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: accept backtest_allocation on agent create and patch"
```

---

## Task 3: Allocated Capital card in Configure

**Files:**
- Modify: `dashboard/frontend/app.html` (remove the cash field at ~line 936-939 from `.agent-editor-title-wrap`; add the new card in `.agent-editor-main` above `#agentEditorSimplePanel` at ~line 968)
- Modify: `dashboard/frontend/js/agent-editor.js` (`getEditorState` ~line 420, `fillHeader` ~line 715)
- Modify: `dashboard/frontend/styles.css` (retire `.agent-editor-cash-*` at ~line 8433-8463, add `.agent-capital-*`)
- Test: `dashboard/backend/tests/test_my_agents_capital_ui.py`

**Interfaces:**
- Consumes: `backtest_allocation` on the agent JSON (Task 2).
- Produces:
  - DOM ids `#agentEditorCashAllocation` (kept, moved) and `#agentEditorBacktestAllocation` (new)
  - `getEditorState()` return object gains `backtest_allocation: number`
  - CSS classes `.agent-capital-card`, `.agent-capital-title`, `.agent-capital-field`, `.agent-capital-label`, `.agent-capital-max`, `.agent-capital-input`, `.agent-capital-note`

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_my_agents_capital_ui.py`:

```python
"""Both allocated-capital fields live in one Configure card (2026-07-29).

Paper-trading capital used to sit in the agent editor's *header* as a 12.5px
uppercase field while backtest capital was a separate input inside the Run
Backtest modal, with nothing connecting them. They are now one card in the
editor's main column, and the modal shows the saved backtest figure read-only.

These are contract guards, not style assertions: they pin *where the inputs
live* and *that the modal no longer edits capital*, which is the behaviour the
consolidation delivers.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def _slice(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_configure_has_one_allocated_capital_card_with_both_inputs():
    card = _slice(_APP_HTML, 'class="agent-capital-card', "</div>\n                    <div")
    assert "Allocated Capital" in card
    assert 'id="agentEditorCashAllocation"' in card
    assert 'id="agentEditorBacktestAllocation"' in card


def test_capital_inputs_are_no_longer_in_the_editor_header():
    """The header held a cramped uppercase field; the card replaces it."""
    header = _slice(_APP_HTML, 'class="agent-editor-title-wrap"', "</header>")
    assert 'id="agentEditorCashAllocation"' not in header
    assert 'id="agentEditorBacktestAllocation"' not in header


def test_capital_limits_are_stated_next_to_each_field():
    assert "max $3,000" in _APP_HTML
    assert "max $10,000" in _APP_HTML


def test_editor_state_carries_backtest_allocation():
    assert "backtest_allocation" in _EDITOR_JS
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_my_agents_capital_ui.py -q`
Expected: FAIL — `ValueError: substring not found` on `agent-capital-card`.

- [ ] **Step 3: Move the markup**

In `dashboard/frontend/app.html`, **delete** this block from `.agent-editor-title-wrap`:

```html
                <label class="agent-editor-cash-field" for="agentEditorCashAllocation">
                    <span class="agent-editor-cash-label">Paper Trading Allocated Capital (max $3,000)</span>
                    <input id="agentEditorCashAllocation" class="agent-editor-cash-input" type="number" min="0" max="3000" step="100" placeholder="1000" aria-label="Paper Trading Allocated Capital">
                </label>
```

Then, inside `<div class="agent-editor-main">`, **immediately before** `<div id="agentEditorSimplePanel" ...>`, insert:

```html
                    <div class="agent-capital-card section-card compact">
                        <h3 class="agent-capital-title">Allocated Capital</h3>
                        <div class="agent-capital-grid">
                            <label class="agent-capital-field" for="agentEditorCashAllocation">
                                <span class="agent-capital-label">Paper Trading <span class="agent-capital-max">max $3,000</span></span>
                                <input id="agentEditorCashAllocation" class="agent-capital-input" type="number" min="0" max="3000" step="100" placeholder="1000" aria-label="Paper Trading Allocated Capital">
                                <span class="agent-capital-note">Reserved from My Portfolio. Real sleeve.</span>
                            </label>
                            <label class="agent-capital-field" for="agentEditorBacktestAllocation">
                                <span class="agent-capital-label">Backtesting <span class="agent-capital-max">max $10,000</span></span>
                                <input id="agentEditorBacktestAllocation" class="agent-capital-input" type="number" min="1" max="10000" step="100" placeholder="1000" aria-label="Backtest Allocated Capital">
                                <span class="agent-capital-note">Simulated only. Never spends real cash.</span>
                            </label>
                        </div>
                    </div>
```

- [ ] **Step 4: Read and write the new field in the editor**

In `dashboard/frontend/js/agent-editor.js`, inside `getEditorState()`, after the existing `cash_allocation` block, add:

```javascript
    const backtestInput = document.getElementById('agentEditorBacktestAllocation');
    let backtest_allocation = null;
    if (backtestInput && backtestInput.value !== '') {
      const value = Number(backtestInput.value);
      if (!Number.isFinite(value) || value < 1) {
        throw new Error('Backtest Allocated Capital must be at least $1.');
      }
      if (value > 10000) {
        throw new Error('Backtest Allocated Capital cannot exceed $10,000.');
      }
      backtest_allocation = Math.round(value);
    } else {
      backtest_allocation = cash_allocation;
    }
```

and add it to the returned object, after `cash_allocation`:

```javascript
      cash_allocation,
      backtest_allocation,
```

In `fillHeader(agent)`, alongside the existing `cashInput` population, add:

```javascript
    const backtestInput = document.getElementById('agentEditorBacktestAllocation');
    if (backtestInput) {
      const resolved =
        agent.backtest_allocation != null
          ? agent.backtest_allocation
          : agent.cash_allocation != null
            ? agent.cash_allocation
            : 1000;
      backtestInput.value = String(Math.min(Math.round(Number(resolved)), 10000));
    }
```

The PATCH goes through a **positional** helper, so the parameter has to be threaded through it. In `patchAgent` (~line 567), add the parameter after `cash_allocation` and put it in the payload:

```javascript
  async function patchAgent(agent, name, description, pipeline, cash_allocation, backtest_allocation, model_name, live_trading_enabled) {
    const payload = {
      name,
      description: description || null,
      cash_allocation,
      backtest_allocation,
      live_trading_enabled: Boolean(live_trading_enabled),
    };
```

Then update the single call site in `save()` (~line 841) — the argument order must match:

```javascript
      const updated = await patchAgent(
        currentAgent,
        state.name,
        state.description,
        state.sendPipeline ? subAgents : null,
        state.cash_allocation,
        state.backtest_allocation,
        state.model_name,
        state.live_trading_enabled
      );
```

**Do not** add a change listener for the new input. Dirty tracking is delegated — `bindEvents()` attaches `input` and `change` handlers to `#agentEditorView` itself, so any input inside the editor is covered automatically.

- [ ] **Step 5: Style the card**

In `dashboard/frontend/styles.css`, **replace** the `.agent-editor-cash-field`, `.agent-editor-cash-label`, `.agent-editor-cash-input`, and `.agent-editor-cash-input:focus` rules with:

```css
.agent-capital-card {
    margin-bottom: 12px;
}

.agent-capital-title {
    margin: 0 0 12px;
    font-size: 17px;
    font-weight: 700;
    color: var(--text-primary);
}

.agent-capital-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
}

.agent-capital-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
}

.agent-capital-label {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
}

.agent-capital-max {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    margin-left: 6px;
}

.agent-capital-input {
    width: 100%;
    box-sizing: border-box;
    padding: 9px 12px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background: var(--bg-input);
    color: var(--text-primary);
    font-size: 15px;
    font-family: var(--font-mono);
}

.agent-capital-input:focus {
    outline: none;
    border-color: rgba(34, 211, 238, 0.4);
}

.agent-capital-note {
    font-size: 13px;
    color: var(--text-secondary);
}
```

- [ ] **Step 6: Run the tests**

Run: `pytest dashboard/backend/tests/test_my_agents_capital_ui.py -q`
Expected: PASS — all four cases.

Then confirm no other test referenced the retired classes:
Run: `pytest dashboard/backend/tests/ -q -k "frontend or agent"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/js/agent-editor.js \
        dashboard/frontend/styles.css \
        dashboard/backend/tests/test_my_agents_capital_ui.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: one Allocated Capital card in agent Configure"
```

---

## Task 4: Run Backtest modal reads capital instead of editing it

**Files:**
- Modify: `dashboard/frontend/app.html` (`#backtestInitialCapital` control group, ~line 320-334)
- Modify: `dashboard/frontend/app.js` (`openRunBacktestModal` ~line 5104-5114, `runBacktest` ~line 5202-5218)
- Test: `dashboard/backend/tests/test_my_agents_capital_ui.py` (append)

**Interfaces:**
- Consumes: `#agentEditorBacktestAllocation` and the resolution order from Task 3.
- Produces: `resolveBacktestCapital(agent)` in `app.js` — takes an agent object, returns a number clamped to `MAX_BACKTEST_ALLOCATED_CAPITAL`. Task 6 reuses it for the card.

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_my_agents_capital_ui.py`:

```python
_APP_JS = (_FRONTEND / "app.js").read_text(encoding="utf-8")


def test_run_backtest_modal_has_no_editable_capital_input():
    """Capital is set in Configure now; the modal only reports it."""
    modal = _slice(_APP_HTML, 'id="runBacktestModal"', 'id="runBacktestModalSubmit"')
    assert 'id="backtestInitialCapital"' not in modal


def test_run_backtest_modal_links_to_configure():
    modal = _slice(_APP_HTML, 'id="runBacktestModal"', 'id="runBacktestModalSubmit"')
    assert 'id="runBacktestEditCapitalBtn"' in modal
    assert "Edit in Configure" in modal


def test_backtest_capital_resolution_helper_exists():
    assert "function resolveBacktestCapital(" in _APP_JS
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_my_agents_capital_ui.py -q`
Expected: FAIL — `assert 'id="backtestInitialCapital"' not in modal`.

- [ ] **Step 3: Replace the modal input with a read-only row**

In `dashboard/frontend/app.html`, replace the whole `control-group` containing `#backtestInitialCapital` with:

```html
                <div class="control-group">
                    <label>Allocated Capital</label>
                    <div class="run-backtest-capital-row">
                        <p id="runBacktestCapitalValue" class="run-backtest-readonly">—</p>
                        <button id="runBacktestEditCapitalBtn" class="run-backtest-edit-link" type="button">Edit in Configure</button>
                    </div>
                    <p id="runBacktestCapitalHint" class="control-helper">Simulated starting cash. Does not change Paper Trading Allocated Capital.</p>
                </div>
```

Add the supporting styles to `styles.css`:

```css
.run-backtest-capital-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
}

.run-backtest-edit-link {
    background: none;
    border: none;
    padding: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--accent-cyan, #22d3ee);
    cursor: pointer;
}

.run-backtest-edit-link:hover {
    text-decoration: underline;
}
```

- [ ] **Step 4: Add the resolver and rewire the modal**

In `dashboard/frontend/app.js`, add near the other agent helpers (above `renderAgentAllocatedCapitalHero`):

```javascript
/**
 * Saved simulated capital for an agent's backtests.
 *
 * Mirrors the backend fallback chain exactly: an agent created before
 * `backtest_allocation` existed has a NULL column and must keep behaving as it
 * did, i.e. starting from its paper sleeve.
 */
function resolveBacktestCapital(agent) {
  const candidates = [agent?.backtest_allocation, agent?.cash_allocation];
  for (const raw of candidates) {
    const value = Number(raw);
    if (Number.isFinite(value) && value > 0) {
      return Math.min(Math.round(value), MAX_BACKTEST_ALLOCATED_CAPITAL);
    }
  }
  return DEFAULT_AGENT_CASH_ALLOCATION;
}
```

In `openRunBacktestModal`, **replace** the block that reseeds `capitalInput` (the `// Reseed on every open.` comment and its body) with:

```javascript
    const capitalValue = document.getElementById('runBacktestCapitalValue');
    if (capitalValue) {
        capitalValue.textContent = `$${resolveBacktestCapital(agent).toLocaleString()}`;
    }
```

Keep the existing `runBacktestCapitalHint` block as-is.

Wire the link — add alongside the other modal listeners in the same init block that binds `runBacktestModalSubmit`:

```javascript
    document.getElementById('runBacktestEditCapitalBtn')?.addEventListener('click', () => {
        const agent = runBacktestModalAgent;
        closeRunBacktestModal();
        if (agent && window.AgentEditor?.open) window.AgentEditor.open(agent);
    });
```

In `runBacktest()`, **replace** the `capitalInput` validation block:

```javascript
    const capitalInput = document.getElementById('backtestInitialCapital');
    let initialCapital = DEFAULT_AGENT_CASH_ALLOCATION;
    if (capitalInput && capitalInput.value !== '') {
        const parsed = Number(capitalInput.value);
        if (!Number.isFinite(parsed) || parsed <= 0) {
            alert('Backtest Allocated Capital must be greater than 0.');
            return;
        }
        if (parsed > MAX_BACKTEST_ALLOCATED_CAPITAL) {
            alert(`Backtest Allocated Capital cannot exceed $${MAX_BACKTEST_ALLOCATED_CAPITAL.toLocaleString()}.`);
            return;
        }
        initialCapital = Math.round(parsed);
        capitalInput.value = String(initialCapital);
    }
```

with:

```javascript
    const initialCapital = resolveBacktestCapital(activeAgent);
```

This must be placed **after** `activeAgent` is resolved (i.e. after the `if (!activeAgent)` guard), since it now reads from the agent.

- [ ] **Step 5: Run the tests**

Run: `pytest dashboard/backend/tests/test_my_agents_capital_ui.py -q`
Expected: PASS — all seven cases.

Verify nothing else referenced the removed id:

Run: `command grep -rn "backtestInitialCapital" dashboard/ --include=*.js --include=*.html --include=*.py`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add dashboard/frontend/app.html dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_my_agents_capital_ui.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: Run Backtest modal reads saved capital, links to Configure"
```

---

## Task 5: An empty Trading Instruction uses the platform default

**Files:**
- Modify: `dashboard/frontend/js/agent-editor.js` (`getEditorState` empty branch ~line 454-461, `open()` backfill ~line 752-758, `save()` ~line 780)
- Modify: `dashboard/frontend/app.html` (`#agentEditorSimplePanel`, ~line 968-974)
- Modify: `dashboard/backend/tests/test_agent_starter_defaults.py` (docstring of `test_the_editor_can_read_the_starter_instruction`)
- Test: `dashboard/backend/tests/test_my_agents_instruction_ui.py`, plus one backend case appended to `test_agent_backtest_allocation.py`

**Interfaces:**
- Consumes: `window.DEFAULT_STARTER_INSTRUCTION` published by `app.js` (already exists).
- Produces: `getEditorState()` returns `subAgents: []` with `sendPipeline: true` when the instruction box is empty.

- [ ] **Step 1: Write the failing tests**

First, the backend contract. Append to `dashboard/backend/tests/test_agent_backtest_allocation.py`:

```python
def test_patch_with_an_empty_pipeline_clears_the_instruction(client):
    """Empty instruction -> no pipeline -> the backend's default hourly prompt.

    portfolio_manager takes the ``create_prompt`` branch when an agent has no
    pipeline, so clearing is what "use the default" means end to end.
    """
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]
    assert created["pipeline"], "builtin agents are seeded with a starter pipeline"

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}", json={"pipeline": []}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert not resp.json()["agent"]["pipeline"]

    reread = client.get(f"/api/v1/agents/{created['agent_id']}", headers=headers)
    assert not reread.json()["agent"]["pipeline"]
```

Then create `dashboard/backend/tests/test_my_agents_instruction_ui.py`:

```python
"""An empty Trading Instruction is a supported state (2026-07-29).

It used to be a silent no-op: ``getEditorState()`` set ``sendPipeline = false``
on an empty box, so the stored pipeline was never touched and the user got a
success toast for a save that changed nothing -- with no way to return an agent
to the platform's default strategy.

Empty now clears the pipeline, which makes the backend take the
``create_prompt`` branch in portfolio_manager. Two things must hold:

1. The UI says so, and shows what the default actually is.
2. The starter *backfill* is gone. It re-injected the default text whenever a
   pipeline-less agent was opened, which under the new semantics would silently
   undo a deliberate empty save on the next visit.
"""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = (_FRONTEND / "app.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")


def test_the_empty_state_is_explained_next_to_the_textarea():
    assert (
        "Leave this empty and the agent uses the platform's default trading strategy."
        in _APP_HTML
    )


def test_the_default_instruction_is_inspectable():
    assert "See the default instruction" in _APP_HTML
    assert 'id="agentEditorDefaultInstructionText"' in _APP_HTML


def test_saving_empty_reports_the_default_was_applied():
    assert "Saved — using the default trading instruction." in _EDITOR_JS


def test_the_starter_backfill_is_gone():
    """The old backfill fought a deliberate empty save on reopen."""
    assert "if (!subAgents.length && instructionEl" not in _EDITOR_JS


def test_empty_instruction_sends_an_empty_pipeline():
    """sendPipeline must be true on empty, or the clear never reaches the API."""
    assert "sendPipeline = false" not in _EDITOR_JS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_my_agents_instruction_ui.py dashboard/backend/tests/test_agent_backtest_allocation.py -q`
Expected: FAIL — the four UI cases fail on missing copy; `test_the_starter_backfill_is_gone` and `test_empty_instruction_sends_an_empty_pipeline` fail because both strings are still present. The backend case should **PASS** already (the clearing path exists); if it fails, stop and investigate before changing the frontend.

- [ ] **Step 3: Invert the empty branch**

In `dashboard/frontend/js/agent-editor.js`, replace the `else` branch of `getEditorState()`:

```javascript
    } else {
      // Empty instruction never touches the stored pipeline: not sent to the
      // server, not cached locally, not folded into currentAgent. This is what
      // stops a rename-only save from destroying a multi-step pipeline this
      // screen cannot display.
      subAgentsOut = subAgents;
      sendPipeline = false;
    }
```

with:

```javascript
    } else {
      // Empty means "use the platform default": clear the pipeline so the
      // backend takes its create_prompt branch. The multi-step pipeline this
      // screen cannot author is protected by a confirm in save(), not by
      // silently refusing to send -- which used to make an empty save a no-op
      // that still reported success.
      subAgentsOut = [];
      sendPipeline = true;
    }
```

- [ ] **Step 4: Guard destructive clears and report the outcome**

In `save()`, immediately after the `if (!state.name)` guard, add:

```javascript
    const clearingToDefault = state.sendPipeline && state.subAgents.length === 0;
    if (clearingToDefault && !isSimplePipeline(subAgents) && subAgents.length) {
      const ok = window.confirm(
        'This agent uses a custom multi-step pipeline. Saving an empty '
        + 'instruction replaces it with the platform default. Continue?',
      );
      if (!ok) return;
    }
```

Then branch the success message. In the server-PATCH path, replace:

```javascript
      showSaveStatus('Saved successfully');
```

with:

```javascript
      showSaveStatus(
        clearingToDefault
          ? 'Saved — using the default trading instruction.'
          : 'Saved successfully',
      );
```

Leave the demo-agent path's `'Saved (demo agent — stored locally)'` and the failure path's `'Saved locally; server update failed: …'` untouched.

**No change is needed in `patchAgent` for the empty list.** `[]` is truthy in JavaScript, so the existing `if (pipeline) payload.pipeline = serializePipeline(pipeline);` sends `pipeline: []` — exactly the clearing payload the backend wants. Verify this holds after editing rather than assuming it: an implementer who "fixes" that guard to `if (pipeline?.length)` silently breaks the whole feature, since the PATCH would then omit `pipeline` and the backend would leave the old instruction in place.

- [ ] **Step 5: Remove the backfill**

In `open()`, delete this block entirely:

```javascript
    if (!subAgents.length && instructionEl && defaultStarterInstruction()) {
      // Agent predates server-side seeding (its pipeline is empty). Offer the
      // starter instruction so Configure is never a blank box, and mark it dirty
      // so a Save persists it.
      instructionEl.value = defaultStarterInstruction();
      markDirtyFromInput();
    }
```

Keep the `captureSavedSnapshot();` call immediately above it, and update the comment above that call — it currently explains ordering relative to the backfill:

```javascript
    // Baseline the stored state so the dirty badge only fires on real edits.
    captureSavedSnapshot();
```

`defaultStarterInstruction()` stays — Step 6 uses it.

- [ ] **Step 6: Add the helper copy and disclosure**

In `dashboard/frontend/app.html`, inside `#agentEditorSimplePanel`, immediately after the `<textarea id="agentEditorSimpleInstruction" ...>` element, insert:

```html
                        <p class="agent-editor-simple-note">Leave this empty and the agent uses the platform's default trading strategy.</p>
                        <details class="agent-editor-default-details">
                            <summary>See the default instruction</summary>
                            <p id="agentEditorDefaultInstructionText" class="agent-editor-default-text"></p>
                        </details>
```

In `agent-editor.js`, populate it inside `open()`, right after the `instructionEl` is set:

```javascript
    const defaultText = document.getElementById('agentEditorDefaultInstructionText');
    if (defaultText) defaultText.textContent = defaultStarterInstruction();
```

Add the styles to `styles.css`:

```css
.agent-editor-default-details {
    margin-top: 8px;
    font-size: 13px;
    color: var(--text-secondary);
}

.agent-editor-default-details summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--text-secondary);
}

.agent-editor-default-text {
    margin: 8px 0 0;
    padding: 10px 12px;
    border-left: 2px solid var(--border-color);
    color: var(--text-secondary);
}
```

- [ ] **Step 7: Update the starter-defaults test rationale**

In `dashboard/backend/tests/test_agent_starter_defaults.py`, replace the docstring of `test_the_editor_can_read_the_starter_instruction`:

```python
    """agent-editor.js backfills from window, so app.js must publish the value."""
```

with:

```python
    """The editor shows the default in its disclosure, so app.js must publish it.

    (This used to back a *backfill* that injected the text into the textarea;
    that was removed when an empty instruction became a supported state --
    see tests/test_my_agents_instruction_ui.py.)
    """
```

- [ ] **Step 8: Run the tests**

Run: `pytest dashboard/backend/tests/test_my_agents_instruction_ui.py dashboard/backend/tests/test_agent_starter_defaults.py dashboard/backend/tests/test_agent_backtest_allocation.py -q`
Expected: PASS — all cases in all three files.

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/js/agent-editor.js dashboard/frontend/app.html \
        dashboard/frontend/styles.css \
        dashboard/backend/tests/test_my_agents_instruction_ui.py \
        dashboard/backend/tests/test_agent_starter_defaults.py \
        dashboard/backend/tests/test_agent_backtest_allocation.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: empty Trading Instruction falls back to the platform default"
```

---

## Task 6: Agent card shows both capitals and a disabled Run Paper Trading

**Files:**
- Modify: `dashboard/frontend/app.js` (`renderAgentAllocatedCapitalHero` ~line 733-748, `renderAgentCardBody` ~line 808-830, `renderAgentCardActions` ~line 842-870)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_my_agents_card_ui.py`

**Interfaces:**
- Consumes: `resolveBacktestCapital(agent)` from Task 4, `formatAgentCashAllocation` (existing).
- Produces: CSS classes `.agent-card-capitals`, `.agent-card-capital`, `.agent-card-cta--disabled`.

- [ ] **Step 1: Write the failing test**

Create `dashboard/backend/tests/test_my_agents_card_ui.py`:

```python
"""My Agents card: both capitals, and a signposted paper-trading affordance.

The card showed only the paper sleeve directly above a **Run Backtest** button,
which implied the figure was what the backtest would use -- it wasn't. Both
figures are now labelled side by side.

Run Paper Trading ships disabled: execution/paper_backend.py is still a stub
(Phase B), and a greyed button with no explanation reads as a bug.
"""

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


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _harness(body: str) -> str:
    """Real functions lifted from app.js, with their few dependencies stubbed."""
    return f"""
const MAX_BACKTEST_ALLOCATED_CAPITAL = 10000;
const DEFAULT_AGENT_CASH_ALLOCATION = 1000;
function escapeHtml(s) {{ return String(s); }}
function formatAgentCashAllocation(v) {{ return '$' + Number(v).toLocaleString(); }}
{_extract_function(_APP_JS, "resolveBacktestCapital")}
{_extract_function(_APP_JS, "renderAgentAllocatedCapitalHero")}
{body}
"""


def test_card_shows_both_capitals():
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 1000, backtest_allocation: 5000}));"
        )
    )
    assert "Paper Trading" in out
    assert "Backtesting" in out
    assert "$1,000" in out
    assert "$5,000" in out


def test_card_backtest_capital_falls_back_to_the_sleeve():
    """An agent predating the column must not render a dash."""
    out = _run_node(
        _harness(
            "console.log(renderAgentAllocatedCapitalHero("
            "{cash_allocation: 2000, backtest_allocation: null}));"
        )
    )
    assert out.count("$2,000") == 2


def test_run_paper_trading_button_is_disabled_and_explained():
    actions = _extract_function(_APP_JS, "renderAgentCardActions")
    assert "Run Paper Trading" in actions
    assert "disabled" in actions
    assert "Paper trading is coming soon" in actions


def test_run_paper_trading_is_absent_from_live_paper_cards():
    """Paper cards show Open Agent; a second paper button would be nonsense."""
    actions = _extract_function(_APP_JS, "renderAgentCardActions")
    head, _, tail = actions.partition("if (statusKey === 'paper')")
    branch, _, rest = tail.partition("} else {")
    assert "Run Paper Trading" not in branch
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_my_agents_card_ui.py -q`
Expected: FAIL — `AssertionError: resolveBacktestCapital not found in app.js` if Task 4 was skipped, otherwise `assert 'Backtesting' in out`.

- [ ] **Step 3: Render both capitals**

In `dashboard/frontend/app.js`, replace `renderAgentAllocatedCapitalHero` entirely:

```javascript
/** Shared top block: both capital figures, equal weight (draft + backtested). */
function renderAgentAllocatedCapitalHero(agent) {
  const paper =
    agent.cash_allocation != null
      ? formatAgentCashAllocation(agent.cash_allocation)
      : '$1,000';
  const backtest = formatAgentCashAllocation(resolveBacktestCapital(agent));
  return `
    <div class="agent-card-capitals">
      <div class="agent-card-capital">
        <span class="agent-card-metric-label">Paper Trading</span>
        <p class="agent-card-metric-value">${escapeHtml(paper)}</p>
        <p class="agent-card-capital-note">From My Portfolio</p>
      </div>
      <div class="agent-card-capital">
        <span class="agent-card-metric-label">Backtesting</span>
        <p class="agent-card-metric-value">${escapeHtml(backtest)}</p>
        <p class="agent-card-capital-note">Simulated</p>
      </div>
    </div>`;
}
```

In `renderAgentCardBody`, in the `backtested` branch, delete the now-redundant note line:

```javascript
        <p class="agent-card-latest-note">Simulated — separate from Paper Trading Allocated Capital.</p>
```

- [ ] **Step 4: Add the disabled Run Paper Trading button**

In `renderAgentCardActions`, replace the primary-action block:

```javascript
  let primary = '';
  if (statusKey === 'paper') {
    primary = `<button class="agent-card-cta agent-open-btn" type="button" data-agent-id="${id}">Open Agent</button>`;
  } else {
    primary = `<button class="agent-card-cta agent-run-backtest-btn" type="button" data-agent-id="${id}">Run Backtest</button>`;
  }
```

with:

```javascript
  let primary = '';
  if (statusKey === 'paper') {
    primary = `<button class="agent-card-cta agent-open-btn" type="button" data-agent-id="${id}">Open Agent</button>`;
  } else {
    // Paper trading is Phase B (execution/paper_backend.py is a stub). Ship the
    // affordance disabled *with a reason* -- an unexplained grey button reads as
    // a bug, and its absence hides that the two capital figures above map onto
    // two different things you can eventually run.
    primary = `
      <button class="agent-card-cta agent-run-backtest-btn" type="button" data-agent-id="${id}">Run Backtest</button>
      <button class="agent-card-cta agent-card-cta--disabled" type="button" disabled aria-disabled="true" title="Paper trading is coming soon" aria-label="Run Paper Trading — Paper trading is coming soon">Run Paper Trading</button>`;
  }
```

- [ ] **Step 5: Add the styles**

In `dashboard/frontend/styles.css`:

```css
.agent-card-capitals {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.agent-card-capital {
    min-width: 0;
}

.agent-card-capital .agent-card-metric-value {
    font-size: 22px;
}

.agent-card-cta--disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

.agent-card-cta--disabled:hover {
    background: inherit;
}
```

- [ ] **Step 6: Run the tests**

Run: `pytest dashboard/backend/tests/test_my_agents_card_ui.py -q`
Expected: PASS — all four cases.

- [ ] **Step 7: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_my_agents_card_ui.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: agent card shows both capitals and a paper-trading affordance"
```

---

## Task 7: Land on My Agents with a live running card

**Files:**
- Modify: `dashboard/frontend/app.js` (running-state store near `let backtestPollTimer` ~line 3206, `runBacktest` navigation ~line 5279, `renderAgentCards` ~line 1058, `ensureBacktestPolling` ~line 4660, `showBacktestLaunchFailure` ~line 4633)
- Modify: `dashboard/frontend/styles.css`
- Test: `dashboard/backend/tests/test_my_agents_card_ui.py` (append)

**Interfaces:**
- Consumes: `renderAgentCategories(agents)` and `allAgents` (existing), `formatBacktestElapsed` (existing, ~line 4402).
- Produces:
  - `markAgentBacktestRunning(agentId, runId)` / `clearAgentBacktestRunning(agentId)` / `getAgentBacktestRunning(agentId)`
  - `renderAgentRunningBody(agent, running)` → HTML string
  - CSS classes `.agent-card-running`, `.agent-card-running-dot`, `.agent-card-running-bar`

- [ ] **Step 1: Write the failing test**

Append to `dashboard/backend/tests/test_my_agents_card_ui.py`:

```python
def test_run_backtest_lands_on_my_agents():
    """The whole point: the user sees the agent they just started."""
    run_backtest = _extract_function(_APP_JS, "runBacktest")
    assert "playgroundTab: 'agents'" in run_backtest
    assert "playgroundTab: 'backtest'" not in run_backtest


def test_running_state_survives_a_refresh():
    assert "sessionStorage" in _APP_JS
    assert "function markAgentBacktestRunning(" in _APP_JS
    assert "function clearAgentBacktestRunning(" in _APP_JS


def test_running_card_shows_an_indicator_and_elapsed_time():
    body = _extract_function(_APP_JS, "renderAgentRunningBody")
    assert "Backtesting" in body
    assert "agent-card-running-dot" in body
    assert "agent-card-running-bar" in body
    assert "formatBacktestElapsed" in body


def test_running_animation_respects_reduced_motion():
    """First continuously-animating element on the page."""
    css = (_FRONTEND / "styles.css").read_text(encoding="utf-8")
    start = css.index(".agent-card-running-dot")
    assert "prefers-reduced-motion" in css[start:]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest dashboard/backend/tests/test_my_agents_card_ui.py -q`
Expected: FAIL — `assert "playgroundTab: 'agents'" in run_backtest`.

- [ ] **Step 3: Add the running-state store**

In `dashboard/frontend/app.js`, immediately after `let backtestPollTimer = null;`, add:

```javascript
// Which agents have a backtest in flight, so My Agents can show it. Mirrored to
// sessionStorage: a refresh mid-run must not silently drop the indicator and
// make a running backtest look like it never started.
const RUNNING_BACKTESTS_KEY = 'running-backtests';

function readRunningBacktests() {
    try {
        const raw = sessionStorage.getItem(RUNNING_BACKTESTS_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (error) {
        return {};
    }
}

function writeRunningBacktests(map) {
    try {
        sessionStorage.setItem(RUNNING_BACKTESTS_KEY, JSON.stringify(map));
    } catch (error) {
        /* sessionStorage unavailable — the in-page indicator still works */
    }
}

function markAgentBacktestRunning(agentId, runId) {
    if (!agentId) return;
    const map = readRunningBacktests();
    map[agentId] = { runId: runId || null, startedAt: Date.now() };
    writeRunningBacktests(map);
}

function clearAgentBacktestRunning(agentId) {
    if (!agentId) return;
    const map = readRunningBacktests();
    if (!(agentId in map)) return;
    delete map[agentId];
    writeRunningBacktests(map);
}

/**
 * Running entry for an agent, or null.
 *
 * Entries older than the poll ceiling are discarded: a run that died without a
 * terminal status would otherwise pin a card to "Backtesting…" forever.
 */
function getAgentBacktestRunning(agentId) {
    const entry = readRunningBacktests()[agentId];
    if (!entry) return null;
    const elapsed = (Date.now() - Number(entry.startedAt || 0)) / 1000;
    if (!Number.isFinite(elapsed) || elapsed > BACKTEST_POLL_MAX_SECONDS) {
        clearAgentBacktestRunning(agentId);
        return null;
    }
    return { ...entry, elapsedSeconds: Math.floor(elapsed) };
}
```

- [ ] **Step 4: Render the running card body**

Add next to the other card renderers, above `renderAgentCardBody`:

```javascript
/**
 * Card body for an agent with a backtest in flight.
 *
 * The bar is deliberately indeterminate rather than a percentage: the launch
 * path has no honest completion estimate, and a fake percentage that stalls is
 * worse than an animation that only claims "still working".
 *
 * Superseded 2026-08-01: the shipped bar is determinate once the engine has
 * published a step; see the 2026-08-01 UX-round spec §B2 and app.js's current
 * renderAgentRunningBody.
 */
function renderAgentRunningBody(agent, running) {
  return `
    <div class="agent-card-running">
      <div class="agent-card-running-head">
        <span class="agent-card-running-dot" aria-hidden="true"></span>
        <span class="agent-card-running-label">Backtesting…</span>
        <span class="agent-card-running-elapsed">${escapeHtml(formatBacktestElapsed(running.elapsedSeconds))}</span>
      </div>
      <div class="agent-card-running-track" role="progressbar" aria-label="Backtest in progress">
        <div class="agent-card-running-bar"></div>
      </div>
    </div>
    ${renderAgentAllocatedCapitalHero(agent)}`;
}
```

In `renderAgentCards`, inside the `visibleAgents.forEach` callback, replace the two render calls:

```javascript
      ${renderAgentCardBody(agent, statusBadge.key)}
      ${renderAgentCardActions(agent, statusBadge.key)}
```

with a running-aware version. Add just above `card.innerHTML = \``:

```javascript
    const running = getAgentBacktestRunning(agent.agent_id);
    if (running) card.classList.add('agent-card--running');
```

and inside the template:

```javascript
      ${running ? renderAgentRunningBody(agent, running) : renderAgentCardBody(agent, statusBadge.key)}
      ${running ? renderAgentRunningActions(agent) : renderAgentCardActions(agent, statusBadge.key)}
```

Add the running actions renderer next to `renderAgentCardActions`:

```javascript
function renderAgentRunningActions(agent) {
  const id = escapeHtml(agent.agent_id);
  return `
    <div class="agent-card-actions agent-card-actions--status">
      <button class="agent-card-cta agent-card-cta--configure agent-card-cta--disabled" type="button" disabled aria-disabled="true" data-agent-id="${id}">Configure</button>
      <button class="agent-card-cta agent-card-cta--disabled" type="button" disabled aria-disabled="true">Running…</button>
    </div>`;
}
```

- [ ] **Step 5: Change the landing page and mark the agent running**

In `runBacktest()`, replace:

```javascript
    closeRunBacktestModal();
    prepareLiveBacktestView(launchConfigBase);
    navigateToPage('playground', { playgroundTab: 'backtest' });
    currentMode = 'backtest';
```

with:

```javascript
    closeRunBacktestModal();
    prepareLiveBacktestView(launchConfigBase);
    markAgentBacktestRunning(activeAgent.agent_id, null);
    navigateToPage('playground', { playgroundTab: 'agents' });
    currentMode = 'backtest';
    renderAgentCategories(allAgents);
```

Then, where the launch response is handled, record the run id — after `stashBacktestLaunchConfig(liveRunId, launchConfigBase);` add:

```javascript
            markAgentBacktestRunning(activeAgent.agent_id, liveRunId);
```

In `showBacktestLaunchFailure(message, launchConfig)`, add at the top of the function:

```javascript
    if (launchConfig?.agentId) {
        clearAgentBacktestRunning(launchConfig.agentId);
        renderAgentCategories(allAgents);
    }
```

- [ ] **Step 6: Drive the card from the existing poller**

In `ensureBacktestPolling()`, inside the `setInterval` callback:

In the `if (status.running) {` branch, after the existing `if (liveId) liveBacktestRunId = liveId;`, add:

```javascript
                // Repaint the My Agents card even when the user is not on the
                // Backtest tab — that page is now the landing page after launch.
                if (playgroundTab === 'agents' && currentPage === 'playground') {
                    renderAgentCategories(allAgents);
                }
```

In the `} else {` (terminal) branch, immediately after `liveBacktestRunId = null;`, add:

```javascript
                Object.keys(readRunningBacktests()).forEach(clearAgentBacktestRunning);
                if (playgroundTab === 'agents' && currentPage === 'playground') {
                    loadAgents();
                }
```

`loadAgents()` refetches and re-renders, so the card flips to the real result rather than a stale local guess.

- [ ] **Step 7: Add the styles**

In `dashboard/frontend/styles.css`:

```css
.agent-card-running {
    margin-bottom: 14px;
}

.agent-card-running-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.agent-card-running-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-cyan, #22d3ee);
    animation: agent-card-running-pulse 1.4s ease-in-out infinite;
}

.agent-card-running-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
}

.agent-card-running-elapsed {
    margin-left: auto;
    font-size: 13px;
    font-family: var(--font-mono);
    color: var(--text-muted);
}

.agent-card-running-track {
    height: 4px;
    border-radius: 999px;
    background: var(--border-color);
    overflow: hidden;
}

.agent-card-running-bar {
    width: 40%;
    height: 100%;
    border-radius: 999px;
    background: var(--accent-cyan, #22d3ee);
    animation: agent-card-running-slide 1.6s ease-in-out infinite;
}

@keyframes agent-card-running-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.25; }
}

@keyframes agent-card-running-slide {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(250%); }
}

@media (prefers-reduced-motion: reduce) {
    .agent-card-running-dot,
    .agent-card-running-bar {
        animation: none;
    }
    .agent-card-running-bar {
        width: 100%;
        opacity: 0.5;
    }
}
```

- [ ] **Step 8: Run the tests**

Run: `pytest dashboard/backend/tests/test_my_agents_card_ui.py -q`
Expected: PASS — all eight cases.

- [ ] **Step 9: Commit**

```bash
git add dashboard/frontend/app.js dashboard/frontend/styles.css \
        dashboard/backend/tests/test_my_agents_card_ui.py
git diff --stat origin/main -- dashboard/storage/data/backtest.db   # must print nothing
git commit -m "feat: land on My Agents with a live backtest indicator"
```

---

## Task 8: Full-suite verification and manual walkthrough

**Files:** none modified unless a regression is found.

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: evidence that the branch is green and the flow works in a browser.

- [ ] **Step 1: Clear stale bytecode**

Phantom `test_deleted_shim_is_not_importable` failures come from leftover pre-refactor `__pycache__`, not from this work:

```bash
rm -rf dashboard/backend/engines dashboard/backend/services
```

- [ ] **Step 2: Run the whole backend suite**

Run: `pytest dashboard/backend/tests/ -q`
Expected: PASS. The `iFind csi300` case is a known full-suite flake — rerun it alone to confirm before treating it as a regression.

Any other failure is a real regression from this branch. Fix it before continuing; do not proceed with a red suite.

- [ ] **Step 3: Confirm the seed database is untouched**

```bash
git status --short dashboard/storage/data/
git diff --stat origin/main -- dashboard/storage/data/backtest.db
```

Expected: both print nothing. If `backtest.db` shows as modified, a command ran lazy `ALTER`s against the committed prod database — restore it with `git checkout origin/main -- dashboard/storage/data/backtest.db` and re-check for a stray `-wal` sidecar.

- [ ] **Step 4: Drive the real UI**

Start a backend against a scratch database so nothing touches the seed:

```bash
DATABASE_PATH=/tmp/claude-1000/atl-ui-check.db ~/atl-venv/bin/python -m uvicorn dashboard.backend.app:app --port 8010
```

Then walk the flow at `http://localhost:8010/app` and confirm each:

1. Configure shows **one** Allocated Capital card with both fields, headings clearly larger than before.
2. Setting Backtesting capital and saving persists it across a reload.
3. The Run Backtest modal shows that figure read-only, and **Edit in Configure** opens the editor.
4. Clearing the Trading Instruction and saving reports *"Saved — using the default trading instruction."*, and reopening Configure leaves the box **empty** (no backfill).
5. **See the default instruction** expands to the real default text.
6. Clicking Run Backtest closes the modal and lands on **My Agents**.
7. The launching agent's card shows the pulsing dot, moving bar, and a ticking elapsed timer.
8. On completion the card flips to the result.
9. Every card shows both capital figures and a greyed **Run Paper Trading** whose tooltip reads *"Paper trading is coming soon"*.

- [ ] **Step 5: Report**

State plainly which of the nine checks passed and which did not, with the actual output. Do not claim the flow works without having run it.

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| §1 backend `backtest_allocation`, both twins | Task 1 |
| §1 no ledger interaction | Task 2 (test + explicit non-change) |
| §1 Configure card + typography | Task 3 |
| §1 modal read-only + Edit in Configure | Task 4 |
| §2 empty → clear → platform default | Task 5 |
| §2 confirm on multi-step pipelines | Task 5 Step 4 |
| §2 backfill removal | Task 5 Step 5 |
| §2 copy + disclosure | Task 5 Step 6 |
| §3 navigation target change | Task 7 Step 5 |
| §3 running store + sessionStorage | Task 7 Step 3 |
| §3 no new polling loop | Task 7 Step 6 |
| §3 reduced-motion fallbacks | Task 7 Step 7 |
| §3 launch failure clears state | Task 7 Step 5 |
| §4 both capitals on the card | Task 6 Step 3 |
| §5 disabled Run Paper Trading | Task 6 Step 4 |
| Testing section | every task + Task 8 |
| Docs followup | out of scope by decision; catalogued in the spec |

**Placeholder scan:** none — every code step carries the literal code, every test step the literal test.

**Type consistency:** `backtest_allocation` (snake_case) is the wire and store name throughout; `resolveBacktestCapital` is defined in Task 4 and consumed in Tasks 4 and 6; `markAgentBacktestRunning` / `clearAgentBacktestRunning` / `getAgentBacktestRunning` are defined in Task 7 Step 3 and used in Steps 4–6; `renderAgentRunningBody` and `renderAgentRunningActions` are both defined in Task 7 Step 4 and referenced in the same step's template.
