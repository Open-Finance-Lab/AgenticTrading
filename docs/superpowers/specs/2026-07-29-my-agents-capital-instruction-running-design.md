# My Agents: unified capital, optional instruction, visible backtest state

**Date:** 2026-07-29
**Status:** Approved, ready for planning
**Surfaces:** `dashboard/frontend/` (app.html, app.js, js/agent-editor.js, styles.css),
`dashboard/backend/domain/agents/`, `dashboard/backend/api/routers/agents.py`

## Problem

Five distinct friction points on the My Agents page, reported together:

1. **Allocated capital is split across two screens.** The paper-trading sleeve is a
   cramped 12.5px uppercase field buried in the agent editor's header; backtest
   capital is a separate input inside the Run Backtest modal. Nothing tells the user
   the two are related, and neither label reads as a heading.
2. **An empty Trading Instruction silently does nothing.** `getEditorState()` sets
   `sendPipeline = false` on an empty box, so the stored pipeline is left untouched.
   The user gets a success toast for a save that changed nothing, and has no way to
   return an agent to the platform's default strategy.
3. **Clicking Run Backtest drops the user on a different page** (the Backtest tab)
   with no indication back on My Agents that anything is happening.
4. **The agent card shows only the paper sleeve, directly above a Run Backtest
   button** — implying, wrongly, that the number shown is what the backtest will use.
5. **There is no Paper Trading affordance on the card at all**, so the paper/backtest
   distinction the capital fields draw has no counterpart in the actions.

## Non-goals

- Building live paper trading. The new button ships **disabled**; `execution/paper_backend.py`
  remains a stub and this work does not touch it.
- Real-time backtest telemetry on the card. An indeterminate progress indicator plus an
  elapsed timer is the agreed bar; step-level data stays on the Backtest tab. **Superseded
  2026-08-01** — the card now shows a determinate step/percent/ETA bar once the engine has
  published a step; see the 2026-08-01 UX-round spec §B2.
- Any change to the Backtest tab's existing detailed progress panel.

## Decisions

Settled with the requester before design; recorded here so the plan does not relitigate them.

| Question | Decision |
|---|---|
| Where do the capital fields live? | One `Allocated Capital` card in the agent Configure screen. Backtest capital becomes a **saved per-agent setting**. |
| Can a backtest still use a one-off amount? | **No.** The Run Backtest modal shows the saved value read-only with an "Edit in Configure" link. Losing the per-run override is the accepted cost of "one place". |
| What does an empty instruction do? | **Clears** the agent's instruction; the agent then runs the platform's built-in default trading prompt. |
| What happens after Run Backtest? | Land on **My Agents**; the agent's card shows a live running state and auto-flips to the result on completion. |
| User-facing RTD docs | **Out of scope for this work.** Stale lines are catalogued below and handed to the requester as a followup. |

---

## 1. Allocated Capital in one place

### Backend: a new persisted field

`backtest_allocation` — nullable float on `external_agents`.

The store is dual-backend, so every change lands in **both** `domain/agents/repository.py`
(SQLite) and `domain/agents/repository_postgres.py` (Postgres twin), in all five places
`test_store_twin_parity.py` checks statically:

1. the `CREATE TABLE` column list,
2. the lazy `ALTER TABLE ... ADD COLUMN` migration (a column declared only in `CREATE`
   reaches a fresh database but never a deployed one),
3. `_public_agent` / row-to-dict projection,
4. the `create_agent` keyword signature,
5. the `update_agent` sentinel-guarded keyword signature.

PR #227 added a kwarg to one twin only and 500'd every Configure save on prod while the
SQLite suite stayed green. That is the failure this field must not repeat.

Then `domain/agents/service.py` (create + update passthrough) and the
`CreateAgentBody` / `UpdateAgentBody` Pydantic models in `api/routers/agents.py`,
validated `ge=1, le=10000`.

**Ledger interaction: none.** Unlike `cash_allocation`, this is simulated money. It must
*not* route through `portfolio_service.check_agent_allocation` / `set_agent_allocation`;
the `PATCH` handler's sleeve-reconciliation branch stays keyed on `cash_allocation` alone.

**Resolution order** wherever a backtest needs a starting amount:

```
agent.backtest_allocation  →  agent.cash_allocation  →  1000
```

clamped to `MAX_BACKTEST_ALLOCATED_CAPITAL` (10000). Existing agents have the column NULL
and therefore keep exactly today's behavior (seeded from the paper sleeve) until edited.

### Configure screen

The cash input moves out of `.agent-editor-title-wrap` into a new `Allocated Capital`
section card in the main column, placed above Trading instruction.

```
┌─ Allocated Capital ────────────────────────────┐   17px / 700 / --text-primary
│                                                │
│  Paper Trading            max $3,000           │   15px / 600 / --text-primary
│  [ $ 1,000 ]                                   │
│  Reserved from My Portfolio. Real sleeve.      │   13px / --text-secondary
│                                                │
│  Backtesting              max $10,000          │
│  [ $ 1,000 ]                                   │
│  Simulated only. Never spends real cash.       │
└────────────────────────────────────────────────┘
```

Typography is the explicit ask ("larger / clearer font"). Concretely: the card heading uses
the existing `.agent-editor-intro-title` treatment bumped to 17px; per-field labels go from
**12.5px uppercase `--text-muted`** to **15px sentence-case `--text-primary`**, i.e. they
read as headings rather than as form microcopy. The old `.agent-editor-cash-label`
uppercase style is retired.

Validation stays client-side as today (paper 0–3000, backtest 1–10000) with the server as
the authority.

### Run Backtest modal

`#backtestInitialCapital` stops being an input. It becomes a read-only value row plus an
"Edit in Configure" link that closes the modal and opens the editor for that agent.
`runBacktest()` reads the amount from the resolved agent object, not from the DOM.

`#runBacktestCapitalHint` keeps its role but restates the relationship: the amount shown is
simulated and does not touch the paper sleeve.

---

## 2. Empty Trading Instruction uses the platform default

### The path already exists

`PATCH /api/agents/{id}` with `{"pipeline": []}` reaches
`repository.update_agent`, where `json.dumps(pipeline) if pipeline else None` stores NULL.
At backtest time `portfolio_manager.py` takes the `else` branch and calls
`create_prompt(...)` — the built-in hourly trading prompt. No backend change is required;
the frontend simply refuses to send an empty list today.

### Frontend change

`getEditorState()` inverts its empty branch: empty instruction now yields
`subAgentsOut = []` and `sendPipeline = true`.

Two guards survive the inversion:

- **Custom multi-step pipelines.** The existing `updateSimpleReplaceNote()` warning stays,
  and an empty save against a non-simple pipeline additionally requires a `window.confirm`
  before it destroys work this screen cannot re-author. The original no-op existed to stop
  a rename-only save from wiping such a pipeline; the confirm replaces that protection with
  an explicit one.
- **The starter backfill is removed.** `agent-editor.js` currently injects
  `defaultStarterInstruction()` into the box whenever a pipeline-less agent is opened, and
  marks it dirty. Under the new semantics that directly fights the user: save empty,
  reopen, and the textarea has silently refilled itself and claims unsaved changes. A blank
  box is now a meaningful, explained state, so it stays blank.

### Copy

- Under the textarea: *"Leave this empty and the agent uses the platform's default trading
  strategy."*
- A collapsed `<details>` — *"See the default instruction"* — revealing the exact default
  text, so "default" is inspectable rather than a black box.
- Save status on an empty save: *"Saved — using the default trading instruction."*

`window.DEFAULT_STARTER_INSTRUCTION` is still published by `app.js`; its consumer changes
from the backfill to the disclosure. `test_agent_starter_defaults.py::test_the_editor_can_read_the_starter_instruction`
keeps asserting the value is reachable — only its rationale comment changes.

---

## 3. Run Backtest lands on My Agents with a live card

`runBacktest()` changes its destination from `navigateToPage('playground', {playgroundTab: 'backtest'})`
to `{playgroundTab: 'agents'}`. Everything before that — `closeRunBacktestModal()`,
`prepareLiveBacktestView()`, the launch-config stash — is unchanged, so the Backtest tab is
still fully populated for a user who navigates there.

### Running-state store

A module-level `runningBacktests: Map<agentId, {runId, startedAt, status}>`, mirrored to
`sessionStorage` so a mid-run page refresh keeps the indicator. Entries are written at
launch, updated by the existing poller, and removed on terminal status.

**No new polling loop.** `ensureBacktestPolling()` already ticks every second and already
resolves live run status; it gains a hook that updates the map and re-renders the affected
card. A run that finishes while the user is on another page is reconciled on next render.

### Card states

```
RUNNING                                  COMPLETE (brief highlight, then normal)
┌── Momentum Alpha ─────────────────┐    ┌── Momentum Alpha ─────────────────┐
│ ● Backtesting…             0:14   │    │ ✓ Backtest complete               │
│ ▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░          │ →  │ Latest Backtest          +4.2%    │
│                                   │    │ Ending Value $1,042 · Apr 15–23   │
│ Paper Trading    Backtesting      │    │ Paper Trading    Backtesting      │
│ $1,000           $1,000           │    │ $1,000           $1,000           │
│ [Configure]  [Running…]           │    │ [Configure]  [Run Backtest]       │
│                                   │    │              [Run Paper Trading]  │
└───────────────────────────────────┘    └───────────────────────────────────┘
```

The running indicator is a pulsing dot plus an **indeterminate** animated bar — deliberately
not a percentage, since no honest completion estimate exists. **Superseded 2026-08-01: see
the UX-round spec §B2 — the bar is determinate once `step`/`total_steps` are known.** The elapsed timer reuses
`formatBacktestElapsed()`. Card actions are disabled for the duration.

Both the pulse and the bar animation must be wrapped in `@media (prefers-reduced-motion: reduce)`
fallbacks (static dot, static bar) — this is the first continuously-animating element on the page.

### Failure

A launch that fails shows an error state on the card ("Backtest didn't start") with the
message, and clears the running entry. This mirrors the existing `showBacktestLaunchFailure`
path rather than replacing it.

---

## 4. Both capitals on the agent card

`renderAgentAllocatedCapitalHero(agent)` becomes a two-stat row of equal visual weight —
Paper Trading and Backtesting — replacing the single-metric hero. It resolves backtest
capital through the same fallback chain as the backend so a NULL column renders the paper
sleeve rather than a dash.

This is what removes the confusion in item 4 of the problem statement: the number sitting
above **Run Backtest** is now explicitly labelled as the backtest number.

The `backtested` card variant keeps its Latest Backtest block below the capital row; the
existing `agent-card-latest-note` ("Simulated — separate from Paper Trading Allocated
Capital") becomes redundant once both are labelled and is removed.

## 5. Run Paper Trading button (disabled)

`renderAgentCardActions()` gains a `Run Paper Trading` button rendered **below** Run
Backtest, permanently `disabled` for now, with `title` and `aria-label` reading
*"Paper trading is coming soon"* so the grey-out is explained rather than mysterious.
`aria-disabled` accompanies `disabled` for screen readers.

It renders on every card whose primary action is Run Backtest — both the `draft` (no
backtests yet) and `backtested` states. It is **not** rendered on cards in the `paper`
status state, which show **Open Agent** as their primary action and are already running.

---

## Testing

**Backend.** Twin parity is enforced automatically by the existing static tests. Add:

- `backtest_allocation` round-trips through `POST /api/agents` and `PATCH /api/agents/{id}`,
  including the NULL-on-existing-agents default.
- `backtest_allocation` out of range is rejected (0, negative, >10000).
- Changing `backtest_allocation` alone does **not** move the portfolio ledger.
- `PATCH {"pipeline": []}` clears a seeded agent, and a re-read confirms NULL.

**Frontend.** The repo has no JS test harness; the established convention is source-text
assertions from `dashboard/backend/tests/`. Add guards for:

- the Run Backtest modal no longer containing an editable capital input,
- the empty-instruction helper copy and the `<details>` disclosure being present,
- the starter backfill being gone from `agent-editor.js`,
- the disabled Run Paper Trading button carrying an explanatory label,
- `runBacktest()` navigating to the agents tab.

These guards pin *behavioral contracts*, not markup cosmetics, so they do not become
churn on every future style tweak.

**Manual verification.** Drive the flow with the WSL Playwright setup against a scratch-DB
backend: create an agent, save an empty instruction, confirm the default runs, click Run
Backtest, confirm the landing page and the running card, and confirm the completion flip.

## Risks

- **Per-run capital override is gone.** Accepted explicitly. If it proves painful, the
  escape hatch is a "use a different amount for this run" disclosure in the modal — designed
  and rejected for v1, not foreclosed.
- **Clearing an instruction is destructive.** Mitigated by the confirm on non-simple
  pipelines, but a user who clears a simple instruction and saves cannot undo it from the UI.
  Agent versions exist in `version_repository.py`; wiring undo to them is out of scope.
- **`sessionStorage` running-state can go stale** if a run dies without a terminal status.
  The poller's existing max-attempt bound clears it; entries older than the poll ceiling are
  discarded on read.

## Followup: user-facing docs to update

Not touched by this work, per the requester. These RTD pages describe behavior this change
removes:

- `docs/source/lab/getting_started.rst:9` — says Backtest Allocated Capital is set in the
  Run Backtest dialog.
- `docs/source/lab/getting_started.rst:30-32` — defines it as per-run simulated cash that
  defaults to the paper sleeve.
- `docs/source/lab/marketplace.rst:60-61` — says a backtest starts from the paper amount
  "unless you change it in the Run Backtest dialog".
- `docs/source/lab/external_agents.rst:378` — references card actions, which now include a
  second (disabled) button.

Also worth a line once shipped: an empty Trading Instruction is a supported, documented
state, not an error.
