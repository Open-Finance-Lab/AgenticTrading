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
| 1 | #129 | `fix/backtest-setup-panel-dead-band` | `../ATL-worktrees/p1-setup-panel` | `main` | pr-open | #457 |
| 2 | #169 | `fix/backtest-run-provenance` | `../ATL-worktrees/p1-provenance` | `main` | pushed `0288ff79` | — |
| 3 | #273, #308 | `fix/backtest-cancel-and-memory` | `../ATL-worktrees/p1-cancel-memory` | `fix/backtest-run-provenance` @ `0288ff79` | in-progress | — |
| 4 | #390, #365 | `fix/leaderboard-curve-integrity` | `../ATL-worktrees/p2-curve-integrity` | `main` | pr-open (draft) | #459 |
| — | design docs | `docs/backtest-p1-p2-design` | `../ATL-worktrees/docs-design` | `main` | pr-open | #456 |

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
5. **Refund — DO NOT BUILD. There is nothing to refund.** Credit metering is currently
   **unwired from the production path**; see "Credit metering is disconnected" below.
   `POST /backtest/run` debits nothing, so a cancel has no charge to reverse. Do not
   invent a refund, and do **not** re-wire metering inside this PR — re-arming a spend
   control is a user-visible behaviour change that needs its own decision, not a rider
   on a cancel button.
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

**Revised 2026-09-11** after reading the issue bodies; the original #365 mechanism in
this plan was a guess the reporter's investigation contradicts. See the spec.

**Files:** `domain/leaderboard/service.py`, `domain/leaderboard/baselines.py`,
`dashboard/frontend/js/leaderboard.js` and the app chart layer, tests.

1. **#390.** At `service.py:1447-1449`, stop coercing `NULL` to `0`. Carry `None`
   through; render a gap. Log ERROR at the wholesale boundary — a series that comes back
   entirely empty is a contract break and a per-point warning cannot report it. Build on
   the existing `finiteNumber` helper in `js/leaderboard.js`; PR #387 hardened the
   landing side already. Update **both** chart readers in this PR.

2. **#365 — settle the facts before choosing the repair.** Query the committed seed DB
   (`dashboard/storage/data/backtest.db`, read-only, never commit a mutation) and find
   what the twelve `lb_*` runs actually hold for `initial_equity`: `100000`, `10000`, or
   `NULL`. If it is `100000`, the existing `scale` already normalises those curves and
   the practical defect is narrower than the issue implies. If `NULL`, they publish
   unscaled at `scale = 1.0`. The answer decides the fix.

3. **The real #365 defect is mixed capital across one board.** Baselines
   (`auto_compute: true`) recompute at the config's `$10k`; LLM entries
   (`auto_compute: false`) stay cached at the `$100k` they were written at, because
   `_find_cached_run`'s key omits seed capital.

4. **Widen the cache key — carefully.** Cover `initial_capital` *and* `strategy_prompt`
   (the issue author notes both are missing and one change fixes both;
   `strategy_prompt` arrives with PR #366 — include it only if that merged). **A miss
   must be inert:** skip-and-log, never recompute, never auto-deploy. CLAUDE.md warns a
   miss can trigger baseline recomputation and, with `LEADERBOARD_DAILY_AUTO_DEPLOY`
   armed, billable LLM deploys from a public unauthenticated GET. If inertness cannot be
   proven from the code, split the key change out of this PR.

5. **Do not assume a re-run.** #194 would repopulate the LLM entries and is **blocked on
   LLM API credits**. The board must be honest without it — publish fewer rows rather
   than mixed-capital rows.

6. **Tighten `service.py:1204-1206`** — a stored `0.0` is falsy and would collapse to
   config capital. `NOT NULL` makes it unreachable today; the author asked for an
   explicit `is None` check when the code is next touched, and this PR touches it.

7. **Comment the shim.** Scaling is only honest for a scale-free strategy; position
   sizing, lot sizes and per-trade costs make a `$10k` run a different run, not a scaled
   one.

8. **Tests:** a `NULL` point renders as a gap, never `0`, and never drags the shared
   axis; a run with no honest seed is skipped-and-logged rather than published at
   `scale = 1.0`; runs at different seeds do not collide; a legacy row does not cause a
   board-wide cache miss; a miss cannot reach `deploy_model_run`.

9. **Suite green:** `python -m pytest dashboard/backend/tests/ -q`.

**Fallback:** if #365 turns out to require the re-run #194 is blocked on, ship #390
alone and say so. A correct partial fix beats a confident wrong one.

## Out-of-scope finding: credit metering is disconnected

**Verified 2026-09-11 on `main` @ `193400d6`. Not one of the six issues; recorded here
because PR 3 was planned against behaviour that does not exist.**

`domain/entitlements/credits.py` implements the LLM-run metering policy and is fully
unit-tested. **Nothing in production calls it.** The only non-test import of
`domain/entitlements` in the entire backend is `api/routers/admin_users.py:179`, which
uses it solely to report `credits_metering_enabled` in admin stats.

- `authorize_llm_run` — called only by `tests/test_credit_metering.py`
- `refund_llm_run` — called only by `tests/test_credit_metering.py`

So no credit is ever debited for a dashboard LLM backtest, and setting
`CREDITS_METERING_ENABLED=1` changes nothing but an admin boolean.

**It was wired, and a later commit removed it.** `git log -S` isolates the change:

- `c8bdbcd6` *feat(admin): meter credits against LLM backtests* — added the wiring.
- `3eebb7da` *feat: add secure llm worker handoff* (2026-08-24) — **removed it.** The
  diff against `api/routers/backtests.py` deletes the `entitlements` import, the
  `credits.authorize_llm_run(owner_user_id)` debit at accept, and both
  `credits.refund_llm_run(...)` calls. All deletions; nothing replaced them.

The suite stayed green because the tests exercise the policy module directly and never
the route. CLAUDE.md still documents this metering as live and load-bearing — including
the ordering subtlety about the debit landing *after* the concurrency check — so the
documentation now describes a control that is switched off.

This is the workstream's own defect class in its purest form: a correct, tested policy
disconnected at the boundary, with everything around it still asserting it works. It
differs from the other five only in that the discarded value is **money**.

**Not actioned.** Re-arming a spend control changes behaviour for real users, and filing
on a shared repo assigns work to others — both are the user's call. Raised, awaiting a
decision.

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
- `2026-09-11` — Design spec + plan committed; PR #456 opened. Four worktrees created.
- `2026-09-11` — PRs 1, 2 and 4 dispatched in parallel worktrees.
- `2026-09-11` — Issue bodies read. **Two corrections landed in the spec.** #365's
  mechanism is mixed capital across one board (baselines recompute at $10k while LLM
  entries stay cached at $100k), not the NULL-fallback this plan first guessed; and #308
  is a hosting-capacity failure whose real exits are an ops plan upgrade or a product
  guard, so PR 3 ships the guard and must not claim to close it.
- `2026-09-11` — **Open decision for the user:** #308 option (a) is a Render plan
  upgrade, a recurring monthly cost. Not taken; flagged.
- `2026-09-11` — **Out-of-scope finding, verified:** credit metering is unwired from
  prod. `3eebb7da` (2026-08-24) deleted the debit and both refunds from
  `api/routers/backtests.py`; the policy module and its tests survived intact, so CI
  never noticed. PR 3's planned refund step is therefore unbuildable and has been struck.
  Awaiting the user's decision on whether to re-arm and whether to file an issue.
- `2026-09-11` — **PR #457 open (#129).** Re-show rule moved into the 1200px block.
  Suite 4438 passed / 163 skipped. Notable: `test_vnpy_simulation_frontend.py`'s
  docstring on `main` *documented the 901-1200px dead band as a known limitation its
  assertions pass anyway* — the guard was written to tolerate the bug, not catch it.
  That test now asserts the 1200px block, and a new guard
  (`test_frontend_setup_panel_breakpoint.py`) compares the two breakpoints and was
  confirmed to fail on the pre-fix CSS.
- `2026-09-11` — PR 2 pushed `0288ff79` (4464 passed / 163 skipped). Adds
  `domain/backtesting/provenance.py`, the `llm_decisions` column in **both** the SQLite
  and Postgres schemas, and 485 lines of tests.
- `2026-09-11` — PR 3 dispatched, stacked on `0288ff79`. Its refund step is struck (no
  debit exists to reverse) and it is briefed **not** to claim it closes #308 — it ships
  the product guard, not the capacity fix.
- `2026-09-11` — **PR #459 open (draft).** Seed DB settled #365: all twelve `lb_*` rows
  hold `initial_equity = 100000` (one as `100000.00000000003`), zero NULLs, first curve
  point exactly `100000.0`, `metadata` NULL on all twelve. Config declares `10000`, so
  `scale = 0.1` already normalises the board to a $10k display — **the mixed-capital
  board is latent, not live.** It begins at the next `force=true` refresh, when
  `auto_compute` baselines recompute at $10k while LLM entries stay at $100k. Even then
  the *display* would not diverge; the damage is to **returns**, because a $10k re-run
  trades in a coarser share quantum and is a different run rather than a rescaled one.
  That is what makes it a ranking defect. Resolution: align config to `100000` in an
  isolated, reversible commit.
- `2026-09-11` — **Verified spend path, worth keeping.** A leaderboard cache miss is
  **not** inert today. `maybe_schedule_daily_leaderboard_refresh` →
  `_run_daily_refresh_background` → `refresh_daily_leaderboard(deploy_models=True)`
  iterates *every* `llm_leaderboard_entries(config)` and calls `deploy_model_run` without
  consulting `pending_entry_ids`; `deploy_model_run` short-circuits only on a
  `_find_cached_run` hit. So widening the cache key without care converts a display bug
  into **billable LLM runs reachable from a public unauthenticated GET** whenever
  `LEADERBOARD_DAILY_AUTO_DEPLOY` is armed. PR #459 makes a seed mismatch inert at all
  four spend/serve points and pins each with a test.
- `2026-09-11` — Design note settled for #459: the defect is **mixed** capital, not
  capital disagreeing with config. A board whose entries all share one seed is
  internally consistent and merely mislabelled, so it is served with a warning; only
  genuinely mixed seeds drop the outliers. The skip rule must never empty the board — a
  one-line config typo would otherwise take down the acquisition hook.
- `2026-09-11` — Also found in `app.js`: the existing null guard was **inert, not
  absent**. `Number(point?.equity)` + `Number.isFinite` reads as a null check, but
  `Number(null)` is `0` and `0` is finite.
- `2026-09-11` — **#129 severity corrected.** Verified in `app.html`: the backtest launch
  flow is `#runBacktestModal`, opened from My Agents, and the `Market Data` selector the
  issue names lives inside it. The `.left-panel` this PR restores is a **read-only "Run
  config" summary**. A user at 1100px can still run a backtest; they lose the summary of
  what they ran. #129 is real but is **not** a precondition for the onboarding loop — my
  original "reachable / door to the same room" framing was wrong, and the issue body
  describes a pre-#450/#451 UI. Relay that to #129's triage.
- `2026-09-11` — PR #457 updated (`8f9c7787`): guard restyled onto
  `test_vnpy_simulation_frontend.py`'s `_media_block`/`_declarations` idiom, with one
  deliberate deviation — the bare `.left-panel` lookup is line-anchored, because
  `.left-panel` is a literal substring of `.playground-backtest-panel .left-panel` and an
  unanchored search matches inside the compound selector. Issue's cited `styles.css:2183`
  is `.chart-legend` on current main: stale, not a second hide site.
