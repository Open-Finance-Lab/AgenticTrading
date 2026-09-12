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

`#129` is the odd one out: a plain CSS dead band.

### Refined 2026-09-11, during implementation: absence is not a value

The table above says "computed, then discarded". Implementing #169 showed that is only
one direction of a single underlying error, and the other direction bit twice in one PR.

**The root error is conflating *absence of an observation* with *an observed value*.**

- **Discarding** — a real observation is thrown away and reads as nothing. `llm_model`
  carries `"rule-based"` and `/backtest/status` never looks.
- **Manufacturing** — an absence is read as a real observation, which is worse, because
  it fabricates a finding rather than losing one:
  - `float(pt.get("equity") or 0)` — *no data at this timestamp* becomes a real `$0`, a
    −100% curve. (#390)
  - `run.get("initial_equity") or config[...]` — *seed unrecorded* becomes *seed equals
    config*, and the scale silently computes to `1.0`. (#365)
  - `llm_decisions == 0` on a pre-migration row is `DEFAULT 0` backfill, not "the model
    drove nothing" — classifying it as a fallback would accuse the entire back
    catalogue.
  - `llm_calls == 0` from a usage-blind provider is "usage was unreadable", not "no call
    was made". `_record_llm_usage` (`portfolio_manager.py:803-817`) deliberately does not
    increment when `_extract_token_usage` raises, so a run the model drove every step of
    can carry `llm_calls == 0`. A classifier shortcutting on that reports a perfect run
    as a total fallback.

**The rule.** A classifier must distinguish *"I observed that nothing happened"* from
*"I have no observation"*, and may report a finding only from the first. Every counter
feeding a verdict therefore needs a **witness** for whether it was recorded at all —
`metadata.decision_steps` witnesses `llm_decisions`; `llm_decisions` in turn witnesses
`llm_calls`. Where no witness exists the honest output is a fourth state (`unknown`),
never the negative verdict.

**Why this matters more for a badge than for a guard, and asymmetrically.** The H6 guard
fails toward *refusing to publish*: a false positive costs one leaderboard entry. A badge
fails toward *telling a user their run was fake*: a false positive costs the badge's
credibility for every future run, after which the true ones are ignored too. A warning
nobody believes is worse than no warning, because it consumes the attention a real one
would need. That asymmetry is the whole argument for `unknown` existing.

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

**Severity corrected 2026-09-11, after verifying against `app.html`.** This spec first
called #129 "the door to the same room" — the control surface for launching a backtest.
That is no longer true on `main` and the issue text describes an earlier UI. The launch
flow now lives in `#runBacktestModal` (`app.html:~347`), opened from My Agents; the
`Market Data` selector the reporter names is inside that modal, not in the panel. The
current `<aside class="left-panel">` (`app.html:1208`) is a **read-only "Run config"
summary** — agent, model, market, date range, capital, billing — displayed beside
results, whose own empty state reads "No backtests yet. Run one from My Agents."

So a user at 1100px **can still start a backtest**; what they lose is the summary of what
they ran. The defect is real and worth fixing, but it does not block the onboarding loop,
and #129 stays in P1 as cheap adjacent work rather than as a precondition. Whoever
triages #129 should be told the body describes a pre-#450/#451 layout.

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
- **The issue's stated acceptance criterion is an "N of M steps were model-driven"
  badge shown whenever coverage is below 100%**, so the payload carries the raw
  numerator and denominator, not only the verdict. Note the asymmetry and keep it:
  H6's bar is 95% (*may this publish?*) while the badge fires below 100% (*should the
  user be told something degraded?*). Two different questions; do not collapse them
  into one number.

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
- Release the billing hold. ~~`POST /backtest/run` debits at accept~~ — **corrected
  2026-09-12**: it does not, and has not since `3eebb7da`. Billing is per call and
  usage-based (`infrastructure/llm/execution/service.py`), so a cancel must release the
  *reservation*, not refund a per-run credit. The parent's `finally` already calls
  `finalize_run` for exactly this case (`backtests.py:1627-1634`), which is why the
  shipped cancel needed no billing code of its own.
- **Both frontend surfaces** carry the control and the `cancelled` state — the My Agents
  card and the Backtest tab — per the issue's acceptance criteria. One surface updated
  is a half-shipped change.

### 4. Survivable — #308, an AHF backtest OOMs the instance

**The reporter frames this as a hosting-capacity failure, not an application bug, and
that framing is correct.** Their acceptance criteria offer three exits: (a) bump the
Render web service to 1–2GB, (b) a product guard that refuses AHF on the free tier with
a clear UI error, or (c) isolate AHF into its own service.

Note `render.yaml` already says `plan: standard`; per CLAUDE.md the live service runs on
the **free tier** and that file is documentation, not the deploy mechanism. So (a) is not
a code change at all — it is an operator action carrying a recurring monthly cost, and
it is the user's call, not this workstream's.

**This is on the onboarding path, which is why it is P1.** AI Hedge Fund is a built-in
agent on `/app?view=agents` — exactly where the first-loop checklist sends a new user. So
a new user's *first* backtest can be the one that downs the site for everybody.

**What this PR ships: (b), plus a real but secondary memory reduction.**

- A pre-flight refusal for the AHF runtime when the requested window exceeds a
  configured bound, with a clear message rather than an accepted request that cannot
  survive. The bound is an env var with a safe default, defensively parsed — never a
  bare `int()` at module scope, which has killed app boot in this module before.
- The `Popen` change from #273 also replaces the parent's unbounded output buffer:
  `subprocess.run(capture_output=True)` at `backtests.py:1130` accumulates the **entire**
  child stdout/stderr in parent memory for the whole run. Drain on a reader thread and
  retain a bounded head+tail — the head carries universe/decision-source/FX bootstrap,
  the tail carries the failure. A second `capture_output=True` exists at
  `adapter.py:240`, once per trading day; it is transient per call and freed, so the
  outer site is the real accumulator.

**This PR does not close #308.** The `AiHedgeFundSubprocessRunner`'s own footprint is the
driver; the parent-side buffer is a contributor, not the cause. The PR contains the
failure and keeps it off the onboarding path. Say exactly that in the PR body rather
than claiming a fix.

**The question worth putting to the user** is not "how do we fit AHF into 512MB" but
"should the heaviest runtime be reachable from the onboarding path at all". Gating it off
that path costs nothing; keeping it there costs a plan upgrade.

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
read this payload, so the wire format change lands with both readers updated; build on
the existing `finiteNumber` helper in `js/leaderboard.js` rather than adding a parallel
one. PR #387 hardened the landing side already, and the reporter confirms the fix
cannot happen client-side alone.

### 6. #365 — one board, two capital bases

**Corrected 2026-09-11 against the issue body.** An earlier draft of this spec blamed
the `or config[...]` NULL fallback below. That is a real latent defect, but it is not
what #365 reports. The reporter's mechanism:

`dashboard/config/leaderboard.json` declares `initial_capital: 10000`, but **all twelve
published contest-window curves in the committed seed DB were computed at $100,000**.
The config changed *after* those runs were written (commits `0cfc8fb`, `ea1bf2b`,
`1dd5816`, 2026-07-05/07-12).

Because `_find_cached_run`'s key omits seed capital, the two halves of the board then
diverge:

- baselines (`auto_compute: true`) **recompute at $10k**
- LLM entries (`auto_compute: false`) **stay cached at $100k**

leaving one board ranking entries against two different capital bases. That is the
headline defect — not a single mis-scaled curve.

The `or`-fallback at `service.py:1436-1439` remains worth fixing in the same pass: when
`initial_equity` is `NULL`, `stored_initial` becomes the *config* capital, `scale`
computes to exactly `1.0`, and a `$100k` curve publishes unscaled under a `$10k` label.
Whether that path is live depends on what the seed rows actually hold, which is an
empirical question to settle against `dashboard/storage/data/backtest.db` before
choosing the repair — read-only; never commit a mutation to that file.

**Fix.**
- Widen the cache key. The issue author notes `strategy_prompt` is missing from it too,
  and that covering **both** `initial_capital` and `strategy_prompt` fixes both problems
  in one change. (`strategy_prompt` arrives with PR #366; include it only if that has
  merged.)
- **A cache miss must be inert.** Adding seed capital to the key makes every LLM entry
  miss, and CLAUDE.md warns that a miss can trigger baseline recomputation — and, with
  `LEADERBOARD_DAILY_AUTO_DEPLOY` armed, billable LLM deploys from a public
  unauthenticated GET. A miss must skip-and-log, never recompute, never auto-deploy. If
  that cannot be made provably true, the key change splits out of this PR.
- Tighten `service.py:1204-1206`, where a stored `0.0` is falsy and would collapse to
  config capital. The column is `NOT NULL` so the state cannot occur today; the author
  asked for an explicit `is None` check whenever the code is next touched, and this PR
  touches it.
- Never infer a seed from config. Where no honest seed exists, skip the entry and log —
  a published row is a claim.

**The re-run cannot be assumed.** #194 would repopulate the LLM entries, and it is
**blocked on LLM API credits**. So this PR must leave the board honest *without* a
re-run: mixed-capital rows must not publish silently, even if that means publishing
fewer rows.

**Scale-invariance caveat.** Scaling is only honest for a scale-free strategy. Position
sizing, lot sizes and per-trade costs make a `$10k` run a genuinely different run from a
`$100k` one, not a scaled one. Re-running at the published seed is the real fix;
scaling is the compatibility shim. Say so in the code.

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
