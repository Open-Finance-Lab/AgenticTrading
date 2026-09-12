# Backtest P1 + P2 — implementation plan and live progress log

**Design:** `docs/superpowers/specs/2026-09-11-backtest-first-run-loop-design.md`
**Baseline:** `main` @ `193400d6`
**Standing instruction:** do **not** merge any PR in this workstream unless one branch
genuinely blocks another. The user is AFK; leave the merge decision to them.

> **Resuming after a session loss.** This file is the handoff. Read the **Status
> board** below, `git worktree list`, and `gh pr list --author @me`, in that order.
> Everything needed to continue is in git; nothing lives only in an agent's context.

---

## Status board

Update this table as work lands. `state` is one of:
`not-started` → `in-progress` → `pushed` → `pr-open` → `review-clean` → `blocked`.

| PR | Issues | Branch | Worktree | Base | State | PR # |
|---|---|---|---|---|---|---|
| 1 | #129 | `fix/backtest-setup-panel-dead-band` | `../ATL-worktrees/p1-setup-panel` | `main` | not-started | — |
| 2 | #169 | `fix/backtest-run-provenance` | `../ATL-worktrees/p1-provenance` | `main` | not-started | — |
| 3 | #273, #308 | `fix/backtest-cancel-and-memory` | `../ATL-worktrees/p1-cancel-memory` | PR 2 branch | not-started | — |
| 4 | #390, #365 | `fix/leaderboard-curve-integrity` | `../ATL-worktrees/p2-curve-integrity` | `main` | not-started | — |
| — | design docs | `docs/backtest-p1-p2-design` | `../ATL-worktrees/docs-design` | `main` | in-progress | — |

PR 3's worktree is created only after PR 2 has its first commit, since it branches off
PR 2.

---

## PR 1 — #129, setup panel dead band

**Files:** `dashboard/frontend/styles.css`, plus a source-shape test.

1. Read `styles.css:3652-3700` (the `1440`/`1200` blocks) and `:3755-3810` (the `900`
   block). Confirm the hide at `:3663` and the re-show at `:3804` before editing.
2. Move `.playground-backtest-panel .left-panel { display: flex; … }` out of the
   `max-width: 900px` block and into the `max-width: 1200px` block, immediately after
   the `.left-panel { display: none }` rule it corrects.
3. Check no other rule in the `901–1200` band re-hides it (`grep -n "left-panel"`).
4. Add a guard to `dashboard/backend/tests/` in the style of
   `test_frontend_live_trading_board.py`: parse `styles.css`, assert the re-show rule's
   enclosing breakpoint equals the hide rule's. A test that merely finds both rules
   passes today and would pass after a regression.

**Done when:** the setup panel is present at 1280px, 1100px, 1000px and 800px by rule
inspection, and the guard fails if the two breakpoints diverge again.

---

## PR 2 — #169, run provenance

**Files:** `dashboard/backend/database.py`, `domain/backtesting/engine.py`,
`api/routers/backtests.py`, `dashboard/frontend/app.js`, tests.

1. **Schema.** Add `llm_decisions INTEGER DEFAULT 0` to the `agent_runs` CREATE at
   `database.py:143` and to the self-migration list at `:355`. Mirror it in
   `database_postgres.py` — `AGENT_RUNS_DATABASE_URL` runs the Postgres twin in prod,
   and a column added to only one of them is a prod-only failure.
2. **Write path.** Thread `llm_decisions` through `insert_run` (`database.py:633`) and
   pass `manager.llm_decisions` from `engine.py`'s `db.insert_run` call (`:1650`),
   beside the existing `llm_calls=llm_calls_total`.
3. **Verdict.** Add one helper that maps
   `(llm_model, llm_calls, llm_decisions, decision_steps)` → `llm` | `partial` |
   `rule_based`. Reuse `MIN_LLM_DECISION_COVERAGE` from the leaderboard domain rather
   than restating `0.95` — two owners of that threshold will disagree eventually.
   Keep it in `domain/`, not in the router (`domain/` must not import `api/`).
4. **Surface.** In `/backtest/status` (`backtests.py:2166`), the completed branch looks
   up the run row and adds `decision_source` plus the counts. Additive only.
5. **Frontend.** `app.js` renders the verdict on the result. A `rule_based` run must
   not be able to show a bare success.
6. **Tests.** The load-bearing case: `llm_calls == decision_steps` with
   `llm_decisions == 0` classifies as `rule_based`. That is the case a check keyed on
   `llm_calls` passes, which is the whole reason the two counters exist.

**Done when:** a fallback run is distinguishable from a clean one in the API response
and on screen, and the coverage counter survives the subprocess boundary.

---

## PR 3 — #273 + #308, cancel and memory

**Base:** PR 2's branch. **Files:** `api/routers/backtests.py`, `domain/entitlements/`,
`app.js`, tests.

1. **`Popen`.** Replace `subprocess.run` at `backtests.py:1130`. Drain stdout/stderr on
   a reader thread. Retain a bounded head+tail rather than the whole stream — the head
   carries universe/decision-source/FX bootstrap, the tail carries the failure, and
   `_redact_credentials` still applies to what is retained. Enforce the existing
   timeout in the parent. This is #308's parent-side half.
2. **Handle on the slot.** Store the `Popen` in the `_active_slots` entry
   (`backtests.py:656`). It is process-local, which matches every other cap in this
   module; note that in a comment so nobody reads it as cluster-wide.
3. **`cancelled` state.** Extend `_finalize_slot` (`:748`) with a third outcome. Do not
   route cancel through `error` — reporting a user's deliberate action as a failure is
   the same class of lie PR 2 exists to fix.
4. **Route.** `POST /backtest/cancel`, taking `live_run_id`. Authorise with
   `_slot_visible_to` (`:804`) — reuse, do not restate. Unknown id and another user's
   id both 404, per `_resolve_status_slot`'s documented reasoning. Terminate, grace
   period, then kill.
5. **Refund.** A cancelled LLM run that made no billed call returns its credit, on the
   `llm_calls`-as-witness rule in `domain/entitlements/credits.py`.
6. **Status.** `/backtest/status` gains the `cancelled` branch.
7. **AHF pre-flight.** Refuse an over-bound window for the AHF runtime with a usable
   message. Bound is an env var, defensively parsed with a safe default and a log line —
   never a bare `int()` at module scope (this module has been killed at boot that way).
8. **Frontend.** A cancel control while a run is in flight, and a `cancelled` result
   state that is visibly not a failure.

**Done when:** a running backtest can be stopped from the UI in seconds, the slot and
credit are released, the outcome reads as cancelled, and the parent no longer buffers
unbounded child output.

---

## PR 4 — #390 + #365, leaderboard curve integrity

**Files:** `domain/leaderboard/service.py`, `domain/leaderboard/baselines.py`,
`dashboard/frontend/` chart layers, tests.

1. **#390.** At `service.py:1447-1449`, stop coercing `NULL` to `0`. Carry `None`
   through, render a gap. Log ERROR at the wholesale boundary — a whole series coming
   back empty is a contract break, and a per-point warning cannot report it.
2. **Readers.** Both chart layers read this payload. Update both in the same PR;
   a wire-format change with one reader updated is a half-shipped change.
3. **#365.** At `service.py:1436-1439`, stop falling back to config capital. Derive the
   seed from the curve's first point; if that is unavailable, **skip the entry and log**
   rather than publish a row scaled by a number nobody verified.
4. **Cache key.** Add seed capital to `_find_cached_run`'s key so runs at different
   seeds stop colliding.
5. **Comment the shim.** Scaling is only honest for scale-free strategies; position
   sizing, lot sizes and per-trade costs make a `$10k` run a different run, not a
   scaled one. Re-running at the published seed is the real fix.
6. **Board re-run.** Rows on prod were written under the old key. Overlaps open issue
   **#194** — link it, do not duplicate it. The re-run itself is an operator action, not
   part of this PR; note it in the PR body.

**Done when:** no published curve can claim a seed it did not run at, and a missing
observation renders as missing.

---

## Shared conventions

- **Every PR:** full suite green (`pytest dashboard/backend/tests/ -v`) before opening.
- **PR titles:** short, `fix:` prefix, detail in the body. Body concise — reviewers read
  the diff.
- **Never push to a branch whose PR has merged.** Cut a new one.
- **`llm_calls` vs `llm_decisions`** is load-bearing throughout. `llm_calls` bills and
  ticks on truncated responses; `llm_decisions` counts steps the model actually drove.
  Do not "simplify" one into the other.
- **`domain/` must not import `api/`** — enforced by
  `tests/test_architecture_boundaries.py`.
- **Postgres twins.** `database.py` has a `database_postgres.py` sibling. Prod runs the
  twin for `agent_runs`. Schema changes land in both.

---

## Progress log

Append newest last. One line per meaningful event.

- `2026-09-11` — Workstream opened. Design spec written, four worktrees created,
  baseline `main` @ `193400d6`. Nothing implemented yet.
