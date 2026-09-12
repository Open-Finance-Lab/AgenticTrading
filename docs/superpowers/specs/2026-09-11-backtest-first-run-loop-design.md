# Backtest first-run loop — design

**Date:** 2026-09-11
**Status:** approved for implementation (P1 + P2)
**Baseline:** `main` @ `193400d6`

## Why now

PR #451 shipped a two-step checklist on **My Agents** — *"Run your first backtest"*
→ *"See your results"* — which makes the backtest the **declared primary onboarding
task**. The checklist is derived state: it disappears on the first finished run and
cannot be brought back. That is a deliberate design, and it is also a promise. A new
user gets exactly one pass through this loop, and whatever the loop hands them is what
they conclude the product is.

Six open issues break one of that promise's four preconditions. They are grouped into
two projects by what they protect.

- **P1 — the loop holds.** The run must be *reachable*, *honest*, *escapable* and
  *survivable*. (#129, #169, #273, #308)
- **P2 — the payoff is fair.** What the first result gets compared against must be the
  number it claims to be. (#390, #365)

Out of scope, filed separately: #346 (A-share 除权除息) and #123
(`RunManifest.news_sentiment_source`). Both are real, neither sits on the onboarding
path — A-share needs iFinD credentials that prod does not carry, and `RunManifest` is
the v2 agent surface, not the dashboard.

## The recurring defect class

Five of the six are the same shape, which the repo already has a name for:
**fail-closed is not fail-visible**. A correct value is computed and then discarded at
a boundary, and the discard is indistinguishable from the healthy case.

| Issue | Value computed | Where it is discarded |
|---|---|---|
| #169 | `llm_model = "rule-based"`, `llm_decisions` | never reaches `/backtest/status`; `llm_decisions` never reaches `agent_runs` |
| #390 | a stored `NULL` equity (= "no data") | `float(pt.get("equity") or 0)` → a real `$0` |
| #365 | a run's true seed capital | `run.get("initial_equity") or config[...]` → scale silently `1.0` |
| #273 | the child's liveness | `subprocess.run` returns no handle to act on |
| #308 | the child's output, incrementally | `capture_output=True` buffers it all in parent RAM |

`#129` is the odd one out: a plain CSS dead band. It is in P1 because it is the door to
the same room.

---

## P1 — The loop holds

### 1. Reachable — #129, setup panel invisible 901–1200px

**Verified on `main`.** `styles.css:3658` opens `@media (max-width: 1200px)` and
`:3663` sets `.left-panel { display: none }`. The rule that puts it back —
`.playground-backtest-panel .left-panel { display: flex }` at `:3804` — lives inside
`@media (max-width: 900px)`, which opens at `:3755`.

So the panel is hidden from 1200px down, and restored only from 900px down. **Between
901px and 1200px it is gone with no replacement** — a band that covers most laptops in
a split window and every tablet in landscape.

**Fix.** Move the re-show rule out of the 900px block and into the 1200px block, so the
restore has the same breakpoint as the hide. Verify the three bands (>1200, 901–1200,
≤900) each render the setup panel exactly once.

This is CSS only. No JS, no backend, no contract.

### 2. Honest — #169, a rule-based fallback reported as a clean result

There are **two** independent layers here, and the issue reads as one.

**Layer A — the run row is already honest; nothing reads it.**
`engine.py:1261` defaults `llm_model = "rule-based"` and `:1427` promotes it to the real
model on the first successful call. A pipeline run where every step fell back therefore
*already* persists `llm_model = "rule-based"` to `agent_runs`. `/backtest/status`
(`backtests.py:2166`) never looks: the completed branch returns a fixed
`"Backtest completed successfully"` and a `runs_count`. The honest label exists in the
database and dies at the HTTP boundary.

**Layer B — `llm_decisions` is never persisted at all.**
`llm_calls` is the *billing* counter: it ticks at `engine.py:1425` on any response
carrying usage, including a truncated one that then falls back to rule-based.
`llm_decisions` (`portfolio_manager.py:126`, written only at `:614` and `:782`) is the
counter that means *the model actually drove this step*. `agent_runs` has an
`llm_calls` column (`database.py:143`) and **no `llm_decisions` column**.

Consequence: the worst case is invisible. A run whose every step made a billed call
that failed to parse has `llm_calls = 30`, `llm_model = "claude-…"`,
`llm_decisions = 0`. The row looks clean. In-process the H6 guard would refuse it;
dashboard backtests are **subprocesses**, so the in-process counter never leaves the
child.

**Fix.**
- Add `llm_decisions` to the `agent_runs` schema (`database.py:143`) and to the
  self-migration list (`database.py:355`), thread it through `insert_run` and through
  `engine.py`'s save call alongside `llm_calls_total`.
- Compute a single `decision_source` verdict server-side — `llm`, `partial`, or
  `rule_based` — from `(llm_model, llm_calls, llm_decisions, decision_steps)`, reusing
  the H6 coverage threshold (`MIN_LLM_DECISION_COVERAGE = 0.95`) so the dashboard and
  the leaderboard cannot disagree about what "the model drove it" means.
- Return it in `/backtest/status`'s completed payload, and render it on the result. A
  run that fell back must not be able to display the words "completed successfully"
  with nothing else attached.

**Contract note.** This adds fields; it removes none. `/backtest/status` is polled by
`app.js` and by the legacy surface — additive only.

### 3. Escapable — #273, no cancel on a blocking subprocess

`backtests.py:1130` calls `subprocess.run(..., timeout=subprocess_timeout)`. That call
hands back no handle, so between launch and return there is nothing to act on. The
budget is `PIPELINE_SUBPROCESS_TIMEOUT_SECONDS = 3600` for the normal pipeline and up to
`MAX_SUBPROCESS_TIMEOUT_SECONDS = 14400` for hosted runtimes.

Before the checklist this was a bad experience. After it, a new user's *first and only*
pass through the onboarding loop can be a one-hour trap with no exit and one of their
concurrency slots held for the duration.

**Fix.**
- Replace `subprocess.run` with `Popen`, keep the handle on the slot dict in
  `_active_slots`, and enforce the timeout in the parent.
- Add `POST /backtest/cancel`, authorised by exactly the ownership rule
  `_slot_visible_to` already implements — reuse it, do not restate it. An unknown or
  someone else's `live_run_id` must answer 404 identically, for the same reason
  `_resolve_status_slot` documents: a session id is an access grant in this codebase.
- Terminate the child, then kill after a grace period.
- **A cancelled run needs a third terminal state.** `_finalize_slot` currently knows
  only *error* and *runs_count*, and routing a cancel through `error` would report the
  user's own deliberate action as a failure — a different lie in the same place we are
  fixing #169. Add `cancelled` and give `/backtest/status` a matching branch.
- Refund the credit. `POST /backtest/run` debits at accept; a cancelled LLM run that
  made no billed call must return it, on the same `llm_calls`-as-witness rule
  `domain/entitlements/credits.py` already documents.

### 4. Survivable — #308, an AHF backtest OOMs the instance

Two contributors, and the parent-side one is ours.

`subprocess.run(capture_output=True)` accumulates the **entire** child stdout and
stderr in parent memory before returning. The AHF runtime spawns one upstream
subprocess per trading day and each is verbose. On a 512MB Render free-tier instance
that buffer is a real share of the budget, and when the kernel picks a victim it is not
required to pick the backtest — it can take the web process, which is why a single
backtest downs the whole site.

**Fix.**
- The `Popen` change in #273 already replaces the unbounded buffer: drain the child's
  output on a reader thread, cap what is retained (keep head and tail, drop the
  middle — the head carries universe/decision-source/FX bootstrap, the tail carries the
  failure), and write through to the log as it arrives.
- Add a pre-flight refusal for the AHF runtime when the requested window would exceed a
  configured bound, returning a clear message rather than accepting a request that
  cannot survive. Bound is an env var with a safe default, parsed defensively — the
  module already documents an unparseable value killing app boot as a bug it has had
  once.

**Ordering.** #273 and #308 are one change to one call site. They ship together.

---

## P2 — The payoff is fair

Both defects live in `get_leaderboard` (`domain/leaderboard/service.py`), three lines
apart, and are the same `or`-swallows-`NULL` mistake. They ship together.

### 5. #390 — a NULL equity published as a real $0

`service.py:1447`:

```python
"equity": float(pt.get("equity") or 0) * scale,
```

A stored `NULL` means *no observation at this timestamp*. `or 0` converts it to a
genuine zero-dollar portfolio. `chart_equity_curve` (`baselines.py:115`) then prepends
an open point at `initial_equity`, so the series reads as a **−100% loss**.

It does not stop at one row. `align_equity_curves` puts every entry on one shared axis,
so a single −100% series sets the y-range and visually flattens every honest curve on
the board. The first thing a new user compares their result against is a chart where
nothing appears to move.

**Fix.** Distinguish absent from zero. Carry `None` through rather than coercing, let
the chart layer render a gap, and log at the wholesale boundary when an entire series
comes back empty — *absent* and *broken* must not be byte-identical. Both chart layers
read this payload, so the wire format change lands with both readers updated.

### 6. #365 — a $100k curve published as a $10k one

`service.py:1436-1439`:

```python
stored_initial = float(
    run.get("initial_equity") or config.get("initial_capital", INITIAL_CAPITAL)
)
scale = (display_capital / stored_initial) if stored_initial else 1.0
```

The scaling itself is correct and deliberate. The failure is the fallback: when a run's
`initial_equity` is `NULL`, `stored_initial` becomes the *config* capital (`$10,000`,
`dashboard/config/leaderboard.json:3`), so `scale` computes to exactly `1.0` and a curve
actually seeded at `$100,000` is published **unscaled**, labelled `$10,000`.

`_find_cached_run` matches on `(mode, start_date, end_date, llm_model)` only — seed
capital is not part of the key — so a stale row from a different seed is reused without
anything noticing.

**Fix.**
- Do not infer a seed from config. When `initial_equity` is missing, derive it from the
  curve's own first point; if that is unavailable too, skip the entry and log — a
  published row is a claim, and there is no honest number to make it with.
- Include seed capital in the cache key so runs at different seeds stop colliding.
- **Then re-run the board.** The rows on prod today were written under the old key.
  This overlaps open issue **#194**; link it rather than duplicating it.

**Scale-invariance caveat.** Scaling is only honest for a strategy whose behaviour is
scale-free. It is not, in general: position sizing, lot sizes and per-trade costs make a
`$10k` run a genuinely different run from a `$100k` one, not a scaled one. Re-running at
the published seed is the correct fix; scaling is the compatibility shim. Say so in the
code.

---

## Delivery

Four PRs. Two touch `backtests.py`'s slot machinery and status endpoint, so they stack
rather than merge.

| PR | Issues | Branch | Base |
|---|---|---|---|
| 1 | #129 | `fix/backtest-setup-panel-dead-band` | `main` |
| 2 | #169 | `fix/backtest-run-provenance` | `main` |
| 3 | #273, #308 | `fix/backtest-cancel-and-memory` | **PR 2's branch** |
| 4 | #390, #365 | `fix/leaderboard-curve-integrity` | `main` |

PRs 1, 2 and 4 are independent and are built in parallel worktrees under
`../ATL-worktrees/`. PR 3 branches off PR 2 because both change the status contract and
the slot dict; stacking keeps each reviewable without merging either. GitHub retargets
PR 3 automatically when PR 2 merges.

**Nothing merges** unless a branch genuinely blocks another — standing instruction for
this workstream.

## Testing

- **#129** — source-shape guard, in the style of `test_frontend_live_trading_board.py`:
  assert the re-show rule shares a breakpoint with the hide rule. A rendering test
  cannot run here (no browser in WSL), so the guard is on the CSS structure.
- **#169** — a run whose every step fell back must not report a clean success;
  `llm_calls > 0` with `llm_decisions == 0` must classify as `rule_based`, which is the
  case a coverage check keyed on `llm_calls` would pass.
- **#273** — cancel terminates the child, releases the slot, refunds the credit, and
  reports `cancelled` rather than `error`. A cancel against someone else's run and
  against an unknown id both answer 404.
- **#308** — the retained-output cap holds against a child that writes far more than it;
  the AHF pre-flight refuses an over-bound window with a usable message.
- **#390** — a `NULL` equity point renders as a gap, never as `0`, and never drags the
  shared axis.
- **#365** — a run with `NULL` `initial_equity` is skipped and logged rather than
  published at `scale = 1.0`; two runs at different seeds do not collide in the cache.

Full suite green before each PR opens: `pytest dashboard/backend/tests/ -v`.

## Follow-ups to file, not fix here

- **#346** — unadjusted A-share 除权除息 reads as a real loss. Needs iFinD credentials
  prod does not carry.
- **#123** — `RunManifest.news_sentiment_source` never assigned. v2 agent surface.
- **User-facing docs** — `docs/source/lab/getting_started.rst` gains a cancel control
  and a decision-source line on the result. Add to the batch backlog at
  `docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md`; do **not** fix
  piecemeal.
