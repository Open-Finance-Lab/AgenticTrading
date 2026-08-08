# Participatory competition: an Open Track users can enter

**Date:** 2026-08-09
**Status:** Design approved, pending implementation plan

## Goal

Give ATL a hook. Today a new visitor reads a tagline and lands on an empty
My Agents page with no stated objective; users report not knowing what the
platform wants from them. This design makes the **leaderboard the objective**:
a visitor sees a live board of models trading, is told the house instruction
lost money, and is invited to write a better one.

The differentiator is participation. Researched 2026-08-08: nof1.ai's Alpha
Arena is spectator-only (nof1 picks the models and writes the single shared
prompt) and has been dormant since its Season 1.5 ended 2025-12-03, with no
Season 2 as of 2026-08. TradeRank built a public Agent Builder and disabled it.
StrategyArena takes external strategies but is explicitly educational
simulation. **No public LLM-trading leaderboard currently accepts
user-submitted, prompt-defined agents.**

Alpha Arena varies the *model* across one shared prompt. This design does the
inverse: one shared model, N user instructions. That axis is unoccupied, and it
is also the cheap one.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Season window | Fixed historical month, re-runs capped by the attempt budget | Simplest to ship; no waiting for real time to pass |
| Board display | Best attempt only; attempt count in entry detail | User's call — a losing attempt never has to be worn |
| Competition axis | One pinned model, instructions compete | One variable; predictable cost; the unoccupied axis |
| Season 1 model | Nemotron 3 Nano 30B | Cheapest by 10x, and the house lost with it — the prompt carries the signal |
| Funding | Platform pays | No credit system exists yet |
| Attempt budget | Per-entry consumable balance | Is the credit ledger v0; billing tops it up later with no migration |

### The look-ahead trade-off, recorded deliberately

A fixed historical window means entrants can iterate against a known outcome,
and the current contest window (2026-04-15 → 2026-05-15) may fall inside the
training data of competing models. This was raised and accepted: the board
measures prompt engineering against a fixed replay, not alpha discovery, and
should be described that way in user-facing copy.

Two facts materially reduce the risk and are load-bearing:

1. **Nemotron is pinned to `temperature: 0`** — uniquely among the seven house
   entries, which otherwise run at provider default. Verified at
   `dashboard/config/leaderboard.json` (`"temperature": 0`,
   `"reasoning_effort": "none"`) and plumbed for real:
   `llm_agent.py:53-81` validates the value (finite, 0–1, rejects bools,
   rejects combining with extended thinking) and `llm_agent.py:181` passes it
   into the call. Re-running an unchanged instruction therefore returns a
   near-identical curve, so repeated attempts cannot mine sampling variance
   the way they could on a default-temperature model.
   This is *near*-deterministic, not bitwise: providers still vary through
   batching and MoE routing. Do not document it as reproducible.
2. **`instruction_sha256` is recorded per attempt** (see Data model), so two
   attempts sharing a hash are visibly re-runs rather than iterations.

The attempt cap therefore exists to bound cost and iteration count, not to stop
variance mining.

## Background: verified state of the code

Verified 2026-08-08/09 against `origin/main` @ `45ccbc0`. The local checkout was
seven commits behind at the time of writing; read leaderboard files at
`origin/main`.

### What already exists and is better than assumed

- **The board is a one-month backtest already.** `leaderboard.json` fixes
  `initial_capital: 10000`, `start_date: 2026-04-15`, `end_date: 2026-05-15`,
  `reference_start_date: 2026-03-15` (indicator warm-up, no in-window
  look-ahead). 12 entries live in prod: 5 baselines + 7 LLM models.
- **The chart is the most developed frontend in the repo.**
  `dashboard/frontend/js/leaderboard.js` (1,037 lines): three visual tiers by
  `kind` (benchmark / strategy / model, differing in width, dash and alpha),
  a hand-built custom legend (`buildCustomLegend`, `:810-847`) with Chart.js's
  own legend disabled, a grouped curve picker with per-group checkboxes
  (`renderCurvePicker`, `:185-230`) driving a shared `hiddenSeries` set,
  endpoint labels with collision avoidance in a reserved 120px gutter
  (`endpointLabelPlugin`, `:735-808`), a %/$ toggle, and tooltips carrying rank
  and delta-vs-SPY.
- **`buildEquityCurvesFromEntries` (`:477-512`) merges every entry onto one
  shared hour-precision time axis**, not by array index — because SPY ticks at
  `:30` and LLM agents at `:00`. Built for 12 entries, this is exactly what
  makes an N-entry board tractable.
- **The board is already fully public.** No auth dependency on
  `GET /api/v1/leaderboard`; `middleware.py:47-49` exempts `/api/*` from session
  enforcement; `applyInitialNavigation()` runs regardless of sign-in state.
  A logged-out visitor already sees the whole Competition tab.
- **Deep links exist**: `?view=contest|daily|competition|leaderboard|participants|about`
  via `NAV_VIEW_MAP` (`app.html:28-44`).
- **PR #325 is merged** (`45ccbc0`), shipping `POST /api/v1/leaderboard/daily/refresh`
  (secret-gated, 202 + background thread, rate-limited 20/hr), a nightly GitHub
  Actions cron at 22:30 UTC Mon–Fri, and an ET-cash-close-anchored rolling daily
  window. `LEADERBOARD_DAILY_AUTO_DEPLOY` is strict opt-in and must stay that way.
- **The landing page already has the shape of the hook**: a `Race` section
  (`landing/src/components/home/Race.tsx`) with a standings table and a
  leaderboard equity chart — rendering hardcoded `SAMPLE_CURVES`/`SAMPLE_STANDINGS`
  (`:17-32`) badged "Illustrative" (`:74,107`).

### Measured cost per entry, one month-window

From prod `GET /api/v1/leaderboard`, 160–161 LLM calls each:

| Model | est. cost |
|---|---|
| Nemotron 3 Nano 30B | **$0.072** |
| DeepSeek V4 Pro | $0.756 |
| Qwen3.7 Plus | $1.593 |
| Claude Haiku 4.5 | $1.726 |
| Claude Sonnet 4.6 | $4.941 |
| Gemini 3.1 Pro Preview | $11.263 |
| GPT-5.5 | $13.888 |

Full board rebuild: **$34.24**. At Nemotron, 100 entrants × 5 attempts is
**$36/season**; 1000 entrants × 5 is $360.

### What does not exist

- **No path from a user agent to the board.** The roster is hand-edited
  `dashboard/config/leaderboard.json`; entries resolve to one of six hardcoded
  classes in `_STRATEGY_CLASSES` (`registry.py:19-26`); `domain/leaderboard/`
  contains **zero** references to `external_agents`, `user_id`, `owner_id` or
  `agent_id`. Publishing today means editing JSON, running a CLI locally, and
  committing `backtest.db`.
- **No onboarding system of any kind.** No tour library, no `data-step`
  attributes, no spotlight overlay, no step sequencer. "onboarding" appears only
  in comments.
- **No user publish path to Community.** 7 templates, all lab-authored, from
  static `dashboard/config/marketplace.json`.
- **`#homeGetStartedBtn`** (`app.html:411`) has zero JS wiring. It is a dead
  button and has always been one.

### Why the two harnesses are not interchangeable

A 30-dimension reconciliation of the house path (`LLMAgentStrategy` via
`deploy_model_run`) against the user path (`HourlyBacktester` via
`POST /backtest/run`) found ~10 divergences. Four are individually fatal to
"just reuse the engine":

1. **No config key reaches the prompt on the house path.** `LLMAgentStrategy`
   reads only `mode`, `model_id`, `integration`, `reasoning_effort`,
   `temperature`, `symbols` (`llm_agent.py:47-53, 91-93`) and calls
   `make_trading_decision_with_llm` with no `strategy_prompt` and no `pipeline`
   (`llm_agent.py:176-182`), so `create_prompt`'s `custom_prompt` is always
   `None` and `SAFE_TRADING_PROMPT` (`validator.py:545-664`) is unconditional.
2. **The paths do not share a template**, so "only the instruction differs" is
   not expressible. The house sends `SYSTEM_PROMPT` + `SAFE_TRADING_PROMPT`;
   the pipeline sends `PIPELINE_SYSTEM_PROMPT` + `_build_step_prompt`.
3. **Capital.** House 10,000 (`service.py:948`); user runs are 422'd above 3,000
   (`backtests.py:1146-1149`) and clamped again in the worker.
4. **H6 has no user-side evidence.** The engine computes `llm_decisions` and
   never reads or persists it; `decision_steps` is never computed at all.

Six further divergences, none previously known, matter for comparability:
the house fetches bars from `reference_start_date` for indicator warm-up while
the engine fetches only `[start, end]` (leaving SMA50 immature for the first
~50 bars of a ≤31-day window); the house bumps the end date `+1 day`; the engine
applies an 80%-symbol-coverage filter before market hours; the engine passes a
`market` key into the snapshot the house omits; the engine plumbs no
`temperature` at all; the house retries 4× with a reasoning-disabled rescue call
while the pipeline falls back to rule-based on first unparseable response.

## Architecture: two loops

The design does **not** attempt to make one harness serve both purposes.

| | **Loop A — Iterate** (exists, unchanged) | **Loop B — Attempt** (new) |
|---|---|---|
| Path | `HourlyBacktester` | `LLMAgentStrategy` |
| Model | user's choice | Nemotron, `temperature: 0` |
| Window | ≤31 days, user-chosen | 2026-04-15 → 2026-05-15, warm-up from 2026-03-15 |
| Capital | ≤$3,000 | $10,000 |
| Universe | user-chosen, ≤30 | DJIA-30 |
| H6 | not enforced | enforced |
| Cost | existing controls | $0.072, decrements the ledger |
| Meaning | rehearsal | **the official score** |

**An attempt is the submission.** There is no separate publish step: spending an
attempt produces a real contest curve, and the entry's best attempt is what the
board shows. Entering a season is one explicit action; after that, best-of
publishes automatically.

**Rationale for not routing the score through the engine** (recorded so it is
not revisited): doing so requires changing the capital cap, the bar-fetch
window, the coverage filter and the `market_context` pass, plus adding
`decision_steps` and persisting `llm_decisions` — five behaviour changes to the
shipping product path. Raising `MAX_BACKTEST_INITIAL_CAPITAL` from 3,000 to
10,000 alone breaks `test_agents_api.py:485-494`,
`test_agent_backtest_allocation.py`, `test_my_agents_capital_ui.py` and the
frontend clamp at `agent-editor.js:628-660`, and reverses the paper-vs-backtest
capital invariant established by the earlier My Agents UX round.

## The Open Track

### Two tracks, one Competition page

- **Model Track** — the existing 7 LLM entries and 5 baselines. **Frozen.**
  Their published curves must not change.
- **Open Track** — a house reference entry plus all user entries, every one of
  them produced by the same new function.

The Open Track carries its **own house reference**: one entry run through the
identical submission path with the house instruction supplied as its
`strategy_prompt`. This costs $0.072/season and sidesteps the riskiest change in
the space — giving the existing seven entries an explicit `strategy_prompt`
would alter what they produce on any re-deploy, break reproducibility of
published curves, and flip tests across `test_prompts.py`, `test_validator.py`,
`test_llm_validator.py` and `test_portfolio_manager_move.py`.

Comparability inside the Open Track is exact by construction, because every
entry including the reference goes through one code path.

### Season configuration

Seasons live in config, not a table, until there is a second one:

```json
{
  "season_id": "s1",
  "label": "Season 1",
  "pinned_model": {
    "model_id": "nvidia/nemotron-3-nano-30b-a3b",
    "integration": "openrouter",
    "temperature": 0,
    "reasoning_effort": "none",
    "mode": "safe_trading"
  },
  "window": { "start_date": "2026-04-15", "end_date": "2026-05-15",
              "reference_start_date": "2026-03-15" },
  "initial_capital": 10000,
  "attempts_granted": 5,
  "house_reference_entry_id": "s1_house_reference"
}
```

`attempts_granted` is the single knob governing cost, iteration count and
throughput. It is deliberately one number.

### Data model

Two new tables beside `agent_runs` on the run-history database
(`AGENT_RUNS_DATABASE_URL`). **No foreign keys to `owner_user_id` or
`agent_id`**: accounts live behind `USERS_DATABASE_URL` and agents behind
`CONTENT_DATABASE_URL`, in different Neon projects. They are opaque strings by
necessity, and no code path may join across them.

```
leaderboard_entries
  entry_id           TEXT PK        -- namespaced, must not collide with lb_*
  season_id          TEXT NOT NULL
  owner_user_id      TEXT NOT NULL  -- opaque; different database
  agent_id           TEXT NOT NULL  -- opaque; different database
  display_name       TEXT NOT NULL
  best_run_id        TEXT           -- FK-in-spirit to agent_runs.run_id
  best_return        REAL
  attempts_granted   INTEGER NOT NULL
  attempts_used      INTEGER NOT NULL DEFAULT 0
  created_at, updated_at

leaderboard_attempts
  id                 INTEGER PK
  entry_id           TEXT NOT NULL
  run_id             TEXT NOT NULL
  attempt_no         INTEGER NOT NULL
  instruction_sha256 TEXT NOT NULL
  total_return       REAL
  h6_passed          INTEGER NOT NULL   -- BOOLEAN on the Postgres twin
  created_at
```

The `h6_passed` type divergence is called out explicitly because
`test_store_twin_parity.py` compares column *names* only and will not catch it.

One entry per user per season. `attempts_granted`/`attempts_used` **is** the
credit ledger: a season grants, a submission decrements, and a future billing
path tops up — same table, no migration.

`instruction_sha256` gives the integrity signal for free. Two attempts sharing a
hash were re-runs, not iterations, so the entry detail can honestly show
"5 attempts · 3 distinct instructions" without banning anything.

### Submission path

Six changes, modelled on the reconciliation's recommended shape:

- **C1 — `LLMAgentStrategy` accepts an instruction.** In
  `llm_agent.py:47-81` add
  `self.strategy_prompt = (self.config.get("strategy_prompt") or "").strip() or None`,
  and pass `strategy_prompt=self.strategy_prompt` at `llm_agent.py:176-182`.
  `make_trading_decision_with_llm` already declares the parameter
  (`portfolio_manager.py:243`) and already threads it (`:463`). No downstream
  signature changes.
- **C2 — deliberately void.** The reconciliation proposed giving both sides a
  shared template so house and user entries differ only by instruction. This
  design **rejects that** in favour of the Open Track house reference above:
  the existing seven entries are not touched, and comparability is achieved
  within the Open Track instead of across both tracks. The number is retained
  so this spec maps 1:1 onto the source analysis.
- **C3 — one submission entry point.** A new `deploy_user_entry(...)` in
  `domain/leaderboard/service.py`, modelled on `deploy_model_run`
  (`service.py:926-1041`). It builds a config dict carrying the season's pinned
  model plus `strategy_prompt`, calls the unchanged `get_strategy()`
  (`registry.py:37-47`), and **reuses `service.py:979-989` and `:996-1013`
  verbatim** for bar fetching, warm-up, the run, counters and the H6 guard.
  Capital, window, warm-up, the `+1 day` bump, the no-coverage-filter timestamp
  set and integrity all come for free because it is literally the same code.
- **C4 — pin what was submitted.** `_llm_run_metadata` (`service.py:894-921`)
  gains `agent_id`, `season_id` and `strategy_prompt_sha256`. Without this a
  published curve cannot be attributed to the string that produced it.
- **C5 — an authenticated ingress.** A new owner-authenticated, rate-limited
  route in `api/routers/leaderboard.py`, returning 202 and queueing, following
  the existing `/daily/refresh` shape (`:52-106`).
- **C6 — leave the engine alone.** No changes to `HourlyBacktester`,
  `MAX_BACKTEST_INITIAL_CAPITAL`, the fetch window or the coverage filter.

### Integrity

H6 applies unchanged and for free, because C3 reuses the guard call site.
`_reject_if_llm_fallback` (`service.py:833-892`) rejects an entry whose model
drove under `MIN_LLM_DECISION_COVERAGE = 0.95` of steps, keyed on
`llm_decisions` (success-exit counter) rather than `llm_calls` (billing
counter). A failed attempt still decrements the ledger and is recorded with
`h6_passed = 0`; it simply cannot become `best_run_id`.

Evidence this is survivable on a nano model: the house Nemotron entry is
published on the live board, so it cleared H6 through this same harness.

`allow_fallback` must remain absent from every HTTP surface, as it is today.

### Cost and throughput

Cost is bounded by construction: `attempts_granted × entrants × $0.072`.

Throughput is **not** solved by this design and is flagged for the plan.
Attempts do not contend with the engine's process-wide `backtest_status["running"]`
flag, but ~500 runs/season at 2–4 minutes each is roughly 25 hours serialized.
These are I/O-bound LLM calls, so modest concurrency (3–4 workers) is the
obvious answer, but issue **#202** reports blocking sync I/O on exactly the
leaderboard routes and the agent-scale investigation measured throughput
collapsing under load. The design is a FIFO queue with the position shown to the
user; **the worker count must be settled against #202 during planning, not
guessed here.**

## The board

Default visible series, never more than 5–8 lines:

1. Buy & Hold / DJIA baseline — the real question; 6 of 7 house models lost to it
2. The Open Track house reference — the bar being challenged
3. Top 3 Open Track entries
4. **The signed-in user's own entry, pinned unconditionally**, at any rank

Everything else remains available but hidden. This is a **default-selection
policy, not a charting feature**: it decides the initial contents of the
existing `hiddenSeries` set, which the existing curve picker and custom legend
already manage.

Pinning the user's own curve is deliberate. Top-N-only means most entrants see a
chart they are absent from, which is the moment they stop caring.

**Rank against the baseline, not only against each other.** Because Season 1
pins a model whose house result is `-0.22%`, the board will accumulate entries
that beat the house and still lose to buy-and-hold. Show return *and* a
"beat the market" marker so a win cannot quietly mean "lost money slower than a
bad model did".

**Not in v1:** a percentile band for the hidden field. Correct at 500 entries,
speculative at 12; the curve picker already surfaces any individual line.

## The funnel

### Landing

Wire the existing `Race` section to `GET /api/v1/leaderboard`, following the
fetch pattern `MarketTicker.tsx:13-19,77-117` already establishes. Two
constraints: remove the "Illustrative" badge once the data is real, and fetch
same-origin through the Vercel rewrite rather than hardcoding
`agentictrading.onrender.com` as `MarketTicker` does.

CTA copy derives from the platform's own result: *"Our instruction lost 0.22%
with this model. Write a better one."*

Shipping note: `Race.tsx` is Vite/React source but the served page is
hand-patched HTML carrying ~390 lines of auth code with no React counterpart.
The refresh is `npm run build` → copy hashed assets → hand-edit the script/link
hashes → preserve the four auth blocks verbatim, per
`dashboard/landing/README.md:31-81`, guarded mechanically by
`test_frontend_bundle_integrity.py`.

Post-signup redirect is unchanged: `/app?view=agents` with `nav-state`
pre-seeded (`index.html:307-328`).

### Onboarding

**A guided tour is explicitly rejected.** Tours explain the UI, but the observed
problem is that users do not know the goal; tours are stateless once dismissed,
so they do nothing for the user who leaves and returns; and the repo has zero
tour infrastructure, so it would mean building a spotlight/positioning framework
from scratch.

Instead, a **persistent Season checklist card** pinned at the top of My Agents —
a card in the existing shelf renderer, not an overlay framework:

```
Season 1 · Nemotron 3 Nano                        5 attempts left
The house instruction lost 0.22%. Beat it.

  [x] Your starter agent is ready
  [ ] Write its trading instruction        [ Configure ]
  [ ] Run an attempt on the season window  [ Run attempt ]
```

In Phase 1 the third step renders disabled with `Season 1 opens soon`; see
Rollout.

On first successful attempt it collapses into a permanent season HUD —
`Season 1 · rank 34 of 112 · 3 attempts left` — so it never becomes dead weight.

The checklist and the credit ledger are the **same state object**: the card
renders `attempts_remaining`, the integrity rule enforces it, and future billing
tops it up. Because it is server-side it survives the browser change that
today's `localStorage` onboarding guard (`ensureDefaultFoundationAgent`,
`app.js:1703-1776`) does not.

One additional surface: a single welcome screen shown once after signup, stating
the goal in a sentence with a button that scrolls to the checklist. One screen,
not a sequence. The auth-modal pattern in `index.html` is the model to copy.

**Fix while here:** wire `#homeGetStartedBtn` (`app.html:411`) to the checklist,
or remove it. It currently does nothing.

### Community → Agent Marketplace

Change exactly two user-visible strings: the nav label (`app.html:195`) and the
page title (`app.html:1600`). They must change together —
`test_frontend_marketplace_placement.py::test_community_page_header_matches_the_nav_button`
asserts they match, and passes if both move.

**Do not rename** the `community` page key, `#communityView`, `NAV_VIEW_MAP`
entries or the `nav-state` localStorage value: live bookmarks break for no
user-visible gain. Note that `?view=marketplace` already exists as a legacy
alias pointing at `community`, so the rename makes the legacy alias the accurate
one.

A user publish path is **out of scope for v1**. The competition creates the
natural demand for it ("publish my season entry as a template"), so it lands
better as v1.1 with a real corpus than as empty scaffolding now.

## Non-goals

- Live paper trading, real-time execution, or any broker order submission.
  `execution/paper_backend.py` stays a stub.
- Real capital. Ever, on this surface.
- Changing the published Model Track curves.
- Raising `MAX_BACKTEST_INITIAL_CAPITAL`.
- A credit/billing system. The ledger shape anticipates it; pricing is out of
  scope.
- A user publish path into the marketplace.
- Percentile/field bands on the chart.

## Verification

- **H6 on the new insert path.** The guard's docstring states it is applied on
  *both* insert paths; `deploy_user_entry` makes a third. `test_deploy_guard.py`
  needs the mirror coverage cases or an entry can bypass integrity.
- **Postgres twins for both new tables.** They sit on the run-history database,
  so both `BacktestDatabase` and `PostgresBacktestDatabase` need implementations,
  enforced by `test_store_twin_parity.py`. That guard parses source text, so it
  cannot see f-string DDL and compares column *names* only — `NOT NULL`
  divergence is invisible and must be checked by hand.
- **Route-contract freeze golden sets updated in the same commit as C5**, or
  every open PR reddens. Per-router and full-app freezes drift independently.
- **Frontend copy guards** for the checklist and CTA strings, following the
  existing `tests/_frontend_source.py` pattern.
- **Cache-buster bumps** for every touched frontend asset (`app.js?v=`,
  `js/leaderboard.js?v=`, `styles.css?v=`), each versioned independently.
- Cost assertion: a test pinning the season's pinned model to the config, so a
  silent switch to a frontier model cannot multiply spend 190x unnoticed.

### Deploy prerequisites

- **`OPENROUTER_API_KEY` must be set in the Render dashboard.** Nemotron is the
  only board entry not on CommonStack, and OpenRouter is never auto-selected
  (`providers/__init__.py:11`). Unset, submissions silently fail.
- `LEADERBOARD_DAILY_AUTO_DEPLOY` stays strict opt-in. Never restore any
  default that enables it implicitly.
- Render env writes are single-key PUT only; a bulk PUT wipes the list, and they
  do not trigger a redeploy.

## Rollout

**Phase 1 — display only.** Ship the board changes, the landing wiring, the CTA,
the welcome screen and the checklist, with **only the house reference entry**
present. Total cost: **$0.072**.

The checklist ships in a **preseason state**: steps 1 and 2 (starter agent,
write the instruction) are live and fully actionable; the attempt step renders
disabled with `Season 1 opens soon` and a working email-notify or Discord link.
This is the honest version — a checklist whose final step silently does nothing
is exactly the dead-CTA failure this design is fixing (`#homeGetStartedBtn`).

What Phase 1 therefore proves: whether the board and CTA convert a visitor into
a signup, and whether a signed-up user reaches a written instruction. What it
cannot prove: attempt throughput, H6 pass rates on user instructions, or queue
behaviour.

**Phase 2 — open attempts.** Ship C1/C3/C4/C5, the two tables and the queue.
The checklist's third step activates; nothing else about it changes.

This ordering is deliberate: every expensive or corrupting failure mode
(queue thrash, H6 rejections, unbounded spend) lives on the attempt path, while
everything that reveals whether the hook converts lives on the display path.

## Open items

1. **Attempt worker concurrency** — settle against #202 during planning.
2. **`attempts_granted` value** — 5 assumed throughout; confirm before build.
3. **Entry display names** — whether entries show account display name, a
   chosen alias, or are anonymous by default. Affects moderation surface.
4. **Season 1 dates** — whether to keep the current contest window or pick a
   window outside the pinned model's training data.

## References

- Central DB: `knowledge/opt-employment-2026.md` § "The 2026-08-08 Pivot" —
  professor's goal, nof1 competitive research, verified-unbuilt scope.
- Central DB: `decisions/2026-08.md`, entry 2026-08-08.
- `docs/superpowers/specs/2026-08-05-asset-class-shelves-design.md` — shelf
  rendering the checklist card sits inside.
- PR #325 (`45ccbc0`) — daily refresh cron and status UI.
- PR #326 — open at time of writing, touches `js/leaderboard.js`; rebase onto it.
- Issues #145 (scheduler), #202 (event-loop blocking), #230 (`decide()` seam).
