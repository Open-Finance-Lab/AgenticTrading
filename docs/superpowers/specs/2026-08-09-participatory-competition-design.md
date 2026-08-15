# Participatory competition: two boards users can enter

**Date:** 2026-08-09 (rewritten same day after a design grilling)
**Reconciled:** 2026-08-15 against `origin/main` @ `88c7b8c`
**Status:** Phase 0 pending (gated on spend). Phases 2–3 designed, unbuilt.

> ## Reconciliation notice — read before acting on this document
>
> This spec was written on 2026-08-09 and describes **two boards that did not
> exist yet**, named "Replay" and "Forward Season". Six days later, PR #352
> shipped two boards to prod with different names and a different cadence, and
> PR #357 rewrote the landing page around them. Neither PR was written against
> this spec.
>
> **The shipped boards are the house-only halves of the two designed here.** They
> carry the same windows and the same substrate; what they lack is the user entry
> path this document exists to specify. So this design is no longer "add two
> boards" — it is **"add user entry to the two boards that now exist"**, which is
> strictly less work and removes the risk of shipping a parallel, competing pair.
>
> | This spec called it | Prod calls it | Same? |
> |---|---|---|
> | Replay — perpetual qualifier over a fixed month | **Competition Leaderboard** | Same window and substrate; prod has no entry path and no attempt ledger |
> | Forward Season — forward execution on real bars | **Live Trading Leaderboard** | Same substrate and intent; **cadence changed weekly → two weeks**, and prod has no advance engine at all |
>
> The vocabulary throughout this document has been updated to the shipped names.
> Three substantive consequences are recorded inline where they bite, and are
> **not** cosmetic:
>
> 1. **The instruction lock is now two weeks, not one** (§Decisions taken). The
>    original argument for the lock rested on the next season being days away.
>    That argument is weaker at 10 trading days and is restated honestly rather
>    than carried over.
> 2. **C8 is no longer only a Phase 3 feature.** The Live Trading board is *in
>    prod today* in a Season 0 preview with no nightly advance. C8 is what makes
>    the already-shipped board real, independent of user entry (§Rollout).
> 3. **Phase 1 is largely obsolete.** #352/#357 delivered its landing-page half
>    by other means, and in one case in the opposite direction — the landing board
>    ships *illustrative* data under a guard test that requires the label. See the
>    plan document's task table.
>
> **Line numbers below are as of 2026-08-09 unless a note says otherwise.** The
> files they point into have since moved (`js/leaderboard.js` alone went 1,037 →
> 1,600 lines). Re-verify any reference before acting on it; the surrounding
> claims were checked at reconciliation and hold.
>
> ### Where authority now lives
>
> This document is no longer the only design doc covering these boards, and on the
> boards' own UI and payload contract it is **not** the authority:
>
> - **`docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md`** — the
>   newer spec, written alongside #352. It owns the Live Trading UI, the proposed
>   payload contract, the preview state and the settled data-feasibility question.
>   Where the two documents disagree about *the boards*, that one wins. This
>   document remains the authority on **user entry** — the attempt ledger, the
>   submission path, integrity and spend — which that one explicitly places out of
>   scope.
> - **Issue #354** — build the Live Trading season engine. This is C8, already
>   filed with the invariants and the `AGENT_RUNS_DATABASE_URL` placement. **Do not
>   file a second issue for C8**; the §Rollout note below proposing it be split out
>   ahead of Phase 2 is a scheduling argument to make *on #354*.
> - **Issue #355** — the two frontier questions the 2026-08-15 grilling left open.
>   It states outright that both **block this PR and Phase 2**: whether the
>   qualifier gate survives now that the practice board is unranked, and whether
>   `instruction_sha256` config-freeze means anything for user-owned, editable
>   entries. Neither is resolved here, and neither should be resolved by inference
>   from this document's older framing.

## Goal

Give ATL a hook that makes people **come** and **stay**.

Today a visitor reads a tagline, scrolls past four sections, and lands on an
empty My Agents page with no stated objective; users report not knowing what the
platform wants from them. This design makes the leaderboard the objective: a
visitor sees a live board of named frontier models mostly *losing* to
buy-and-hold, and is invited to beat them with one instruction.

The differentiator is participation. Researched 2026-08-08: nof1.ai's Alpha
Arena is spectator-only (nof1 picks the models and writes the single shared
prompt) and has been dormant since Season 1.5 ended 2025-12-03. TradeRank built
a public Agent Builder and disabled it. StrategyArena takes external strategies
but is explicitly educational simulation. **No public LLM-trading leaderboard
currently accepts user-submitted, prompt-defined agents.**

Alpha Arena varies the *model* across one shared prompt. This design does the
inverse: one pinned model, N user instructions. That axis is unoccupied, and it
is also the cheap one.

### Come, and stay — the two mechanisms, named

- **Come** is the Competition board: seven named frontier models, six of which
  lost to buy-and-hold over a real month, on one chart above the fold, with a CTA
  that points at a specific beatable number.
- **Stay** is the Live Trading season cycle: submit at season open, watch the
  board move daily, get a result email at season close, next season opens after.

A fixed replay alone cannot produce a return visit — the race ends the moment
you submit. That is why there are two boards and not one.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Board count | Two: Competition (perpetual) + Live Trading (seasonal) | Competition is the argument; Live Trading is the habit |
| Competition window | The existing fixed month, never reset | A fixed replay's result never changes; resetting deletes history and re-charges for the same curve |
| Live Trading substrate | **Simulated** forward execution on real bars | Decouples the season from broker paper trading, which does not exist |
| Live Trading cadence | **Two weeks — 10 trading days** | *Changed 2026-08-15.* Was "one week, Mon→Fri". PR #352 shipped two-week seasons and `SEASON_TRADING_DAYS = 10` (`js/leaderboard.js:276`); the season strip, progress meter and Season 0 chrome are all built to it. Matching prod beats re-litigating a cadence that already has UI |
| Instruction lock | Locks when the run begins | **Weakened by the cadence change.** The original argument was that a locked instruction costs little when the next season is days away; at 10 trading days the user waits twice as long to correct a bad instruction. Two mitigations, neither free: allow one instruction edit before the first advance, or grant a mid-season re-entry slot. **Resolve this in Phase 3 planning — do not treat the original "user's call" as still-decided** |
| Competition axis | One pinned model, instructions compete | One variable; predictable cost; the unoccupied axis |
| Season 1 model | Nemotron 3 Nano 30B, `temperature: 0` | Cheapest by 10×, and the house lost with it — the prompt carries the signal |
| Entry shape | An agent with **exactly one** pipeline step | Matches how the product already stores instructions |
| Attempts | 5 lifetime on Competition; Live Trading is a slot, not a ledger | Under a locked forward run there is nothing to spend a second attempt on |
| Abuse control | Email verification at **account creation** + a hard global monthly spend ceiling | Accounts are free and instant; attempts cost real money |
| Funding | Platform pays, capped at $50/month | No credit system exists yet |
| Graduation | Carry the instruction text only into Loop A | Loop A cannot reproduce season conditions; pretending otherwise invites bug reports |

### The look-ahead trade-off, resolved rather than accepted

The earlier draft accepted that a fixed historical window lets entrants iterate
against a known outcome, and that the window may sit inside competing models'
training data. **Live Trading resolves this** rather than merely disclosing it:
it runs on bars that do not exist when the instruction locks.

Competition keeps the known-outcome property and is therefore explicitly
positioned as a **qualifier**, not the competitive board. Its purpose is evidence
(the hero chart) and practice (the cheap, repeatable loop). A Competition rank is
a score against a fixed replay; a Live Trading rank is a forecast. User-facing
copy must say so.

Two facts still reduce Competition's gameability and are load-bearing:

1. **Nemotron is pinned to `temperature: 0`** — uniquely among the seven house
   entries, which otherwise run at provider default. Verified at
   `dashboard/config/leaderboard.json`, and plumbed for real: `llm_agent.py`
   validates the value (finite, 0–1, rejects bools, rejects combining with
   extended thinking) and passes it into the call at `:176-182`. Re-running an
   unchanged instruction therefore returns a near-identical curve, so repeated
   attempts cannot mine sampling variance. This is *near*-deterministic, not
   bitwise: providers still vary through batching and MoE routing. Do not
   document it as reproducible.
2. **`instruction_sha256` is recorded per attempt**, so two attempts sharing a
   hash are visibly re-runs rather than iterations.

## Background: verified state of the code

Verified 2026-08-08/09 against `origin/main` @ `45ccbc0`.

### There is one trading harness and two drivers

**This corrects the previous draft**, which claimed the paths "do not share a
template" and were "not interchangeable". That framing was wrong, and it argued
against a change that is in fact small.

Both paths construct the **same `PortfolioManager`** from
`domain/backtesting/portfolio_manager.py` — `engine.py:36,861` and
`llm_agent.py:27,153` — and drive it through the same four calls:
`get_portfolio_state` → `make_trading_decision_with_llm` → `execute_actions`
→ `update_equity`. Order fills, cash accounting, position tracking, T+1
settlement and trade records are **already identical**.

`make_trading_decision_with_llm` (`portfolio_manager.py:227-238`) is **one
method with one branch**:

- `if pipeline:` (`:428`) → `run_pipeline_decision` with `PIPELINE_SYSTEM_PROMPT`
  and a per-step `outputFormat` contract;
- `else:` (`:453`) → `create_prompt(custom_prompt=strategy_prompt)` with
  `SYSTEM_PROMPT` + `SAFE_TRADING_PROMPT`.

The house path passes neither `strategy_prompt` nor `pipeline`, so
`custom_prompt` is always `None`. **C1 fixes that in two lines.**

The divergences that remain are **parametric, not architectural**: which bars
are fed (warm-up from `reference_start_date`, the `+1 day` end bump, the engine's
80%-symbol-coverage filter), the capital cap, `llm_agent.py:152` hardcoding
`get_market_profile(ALPACA)`, the `market` snapshot key, and retry/rescue
behaviour (the house retries 4× with a reasoning-disabled rescue; the pipeline
falls back to rule-based on first unparseable response).

### The pipeline trap

In ATL today, **a trading instruction is structurally a one-step pipeline.**
Every marketplace template — including the simplest, "Balanced Starter" — stores
its instruction as `pipeline: [{presetKey: "simple_instruction", prompt: …,
outputFormat: …}]`, and `domain/agents/service.py:380` seeds
`default_starter_pipeline()` into every new pipeline-runtime agent.

So `if pipeline:` is **true for essentially every user agent**, including
single-instruction ones. "A pipeline-less user agent already matches the house
path" is technically true and practically vacuous — the product does not create
pipeline-less agents. This is why C7 (below) exists.

### What already exists and is better than assumed

- **The Competition board is a one-month backtest already.** `leaderboard.json`
  fixes `initial_capital: 10000`, `start_date: 2026-04-15`, `end_date:
  2026-05-15`, `reference_start_date: 2026-03-15`. 12 entries live in prod: 5
  baselines + 7 LLM models. *(Re-verified 2026-08-15: still exactly 12 entries,
  5 + 7, and still no `strategy_prompt`, `label: "Open Track"` or `authored_by`
  key anywhere in the config.)*
- **The chart is the most developed frontend in the repo.**
  `dashboard/frontend/js/leaderboard.js` (1,037 lines; **1,600 as of
  2026-08-15** — #326, #352 and #357 all landed in it): three visual tiers by
  `kind` (benchmark / strategy / model / **team**), a hand-built custom legend
  (`buildCustomLegend`, `:810-847`) with Chart.js's own legend disabled, a
  grouped curve picker (`renderCurvePicker`, `:185-230`) driving a shared
  `hiddenSeries` set, endpoint labels with collision avoidance in a reserved
  120px gutter (`endpointLabelPlugin`, `:735-808`), a %/$ toggle, and tooltips
  carrying rank and delta-vs-SPY. User entries map to the existing `team` tier,
  which already has a stable per-`entry_id` colour assignment.
- **`buildEquityCurvesFromEntries` (`:477-512`) merges every entry onto one
  shared hour-precision time axis**, not by array index — because SPY ticks at
  `:30` and LLM agents at `:00`.
- **The board is already fully public.** No auth dependency on
  `GET /api/v1/leaderboard`; `middleware.py:47-49` exempts `/api/*` from session
  enforcement.
- **One piece of dead scaffolding already exists for this feature.**
  `app.html:1403-1410` (**`:1428-1439` as of 2026-08-15**) renders a permanent
  empty state — *"Season hasn't started yet. Participating teams will appear here
  when the contest season opens."* It gets filled, not deleted.

  > **Correction (2026-08-15).** This bullet originally also claimed
  > `#homeGetStartedBtn` (`app.html:411`, now `:468`) "has zero JS wiring". That
  > was **already false when written**: `initHomeGetStarted()` was added to
  > `home-page.js` in `08c85aa` on **2026-07-25**, two weeks earlier. The button
  > opens the signup modal when signed out and navigates to the agents playground
  > when signed in, and its label swaps between the two states. Nothing here needs
  > wiring. The dead-CTA *lesson* stands and is still worth honouring — a CTA that
  > leads nowhere is a bug — but it is a past failure to avoid repeating, not an
  > open defect this design gets credit for fixing.
- **Simulated forward execution is nearly assembled.**
  `domain/trading/execution.py:1-6` is in-memory execution over explicit
  cash/positions/trades, extracted from the backtester, and `paper_session.py`
  tracks equity history.
- **A generic transactional email sender exists.**
  `infrastructure/email/sender.py:45` — `async send_email(to, subject,
  text_body)` over Brevo, 84 lines total.
- **Analytics partly exist.** `@vercel/analytics` ships via `App.tsx:2,30`
  (`package.json:75`), same-origin on Vercel so no CSP change was needed. It
  covers landing pageviews and traffic sources — **not** custom funnel events,
  and **not** `/app`, which does not load the React bundle.
- **A landing rebuild is close to mechanical.** `landing/README.md:56-63`:
  `npm run build`, copy content-hashed assets, delete superseded ones, repoint
  `<script>`/`<link>`, keep four hand-written auth markers. Content hashes are
  the cache bust — no `?v=`. `test_frontend_bundle_integrity.py` guards the
  mechanical half in CI.

### Measured cost per entry

From prod `GET /api/v1/leaderboard`, 160–161 LLM calls over the month window:

| Model | Competition (month, ~160 calls) | Live Trading (2 weeks, ~73 calls) |
|---|---|---|
| **Nemotron 3 Nano 30B** | **$0.072** | **~$0.033** |
| DeepSeek V4 Pro | $0.756 | ~$0.35 |
| Qwen3.7 Plus | $1.593 | — |
| Claude Haiku 4.5 | $1.726 | — |
| Claude Sonnet 4.6 | $4.941 | — |
| Gemini 3.1 Pro Preview | $11.263 | — |
| GPT-5.5 | $13.888 | — |

Full board rebuild: $34.24.

> **Cadence change, 2026-08-15.** The right-hand column was "Forward (week, ~36
> calls)". A two-week season is 10 trading days at the observed ~7.3 calls/day,
> so per-entry cost roughly doubles — but seasons arrive half as often, so
> **monthly spend per active user is unchanged** and every figure in §Cost model
> at scale still holds. This is the one place the cadence change is free; the
> instruction lock (§Decisions taken) is where it is not.

### What does not exist

- **No path from a user agent to the board.** The roster is hand-edited
  `leaderboard.json`; entries resolve to six hardcoded classes in
  `_STRATEGY_CLASSES` (`registry.py:19-26`); `domain/leaderboard/` contains
  **zero** references to `external_agents`, `user_id`, `owner_id` or `agent_id`.
- **No onboarding system of any kind.** No tour library, no `data-step`
  attributes, no spotlight overlay, no step sequencer.
- **No email verification at signup.** Every `verif` hit in `users.py` is
  password checking. An account is free, unverified and instant.
- **No order submission anywhere.** `submit_order|place_order|create_order`
  across `domain/trading/` and `infrastructure/brokers/` returns zero hits;
  `paper_backend.py:1-7` states the Alpaca client is read-only and that live
  paper trading "needs its own design pass."
- **No user publish path to the marketplace.** 7 templates, all lab-authored,
  from static `marketplace.json`.

## Architecture: two loops over one harness

| | **Loop A — Iterate** (exists, unchanged) | **Loop B — Compete** (new) |
|---|---|---|
| Driver | `HourlyBacktester` | `LLMAgentStrategy` |
| Harness | `PortfolioManager` | **the same `PortfolioManager`** |
| Model | user's choice | pinned per season |
| Window | ≤31 days, user-chosen | season-defined |
| Capital | ≤$3,000 | $10,000 |
| Universe | user-chosen, ≤30 | DJIA-30 |
| H6 integrity | not enforced | enforced |
| Meaning | rehearsal | **the official score** |

**An attempt is the submission.** There is no separate publish step: spending an
attempt produces a real board curve, and the entry's best attempt is what shows.

**Rationale for not routing the score through Loop A** (recorded so it is not
revisited): raising `MAX_BACKTEST_INITIAL_CAPITAL` from 3,000 to 10,000 alone
breaks `test_agents_api.py:485-494`, `test_agent_backtest_allocation.py`,
`test_my_agents_capital_ui.py` and the frontend clamp at
`agent-editor.js:628-660`, and reverses the paper-vs-backtest capital invariant
established by the My Agents UX round. Loop B also needs the warm-up window and
the no-coverage-filter timestamp set, neither of which Loop A has.

## The two boards

> **Both boards now exist in prod as house-only boards** (PR #352, merged
> 2026-08-15 as `7db504d`). What follows describes the *participatory* versions.
> Read every requirement below as an addition to a shipped board, not as a new
> board to stand up. What prod is missing in both cases is identical: an entry
> path, an attempt ledger, and a user-owned row.

### Competition — the qualifier

*(Designed here as "Replay". Shipped as the **Competition Leaderboard**;
`period=contest`, `VALID_PERIODS[0]`.)*

Perpetual, over the existing fixed month. Never resets: a fixed replay's result
never changes, so resetting would delete history and re-charge everyone to
reproduce identical curves. One long-lived season id; 5 lifetime attempts per
account; best attempt published.

Carries an **Open Track house reference**: one entry run through the identical
submission path with the house instruction supplied as its `strategy_prompt`.
This costs $0.072 and sidesteps the riskiest change in the space — giving the
existing seven entries an explicit `strategy_prompt` would alter what they
produce on any re-deploy, break reproducibility of published curves, and flip
tests across `test_prompts.py`, `test_validator.py`, `test_llm_validator.py` and
`test_portfolio_manager_move.py`.

**Prod gap:** `get_leaderboard` builds every row from the curated `strategies`
roster in `dashboard/config/leaderboard.json`, and `api/routers/leaderboard.py`
exposes no submission route. The board is display-only by construction.

### Live Trading — the competition

*(Designed here as "Forward Season". Shipped as the **Live Trading
Leaderboard**.)*

**Two weeks — 10 consecutive trading days** — of simulated forward execution on
real bars. The instruction locks when the run begins. Entry requires **at least
one completed Competition attempt** — Competition is the qualifier in fact, not
just in copy.

A nightly job advances each entry's persisted portfolio one trading day, reusing
`domain/trading/execution.py` and the shape of the existing daily-refresh cron.
Note the daily leaderboard is **not** reusable as-is: `daily_window_dates()`
(`service.py:79-91`) returns the last completed weekday as both start *and* end,
re-running a fresh one-day backtest each night with no portfolio carried across
days. Live Trading needs persisted state the daily board deliberately does not
keep.

**This is simulated, not brokered.** It must not be called "paper trading" in
the UI: `app.html:847` already has a **Paper Trading** subtab under My Agents,
and "Paper Trading Allocated Capital (max $3,000)" appears in four places tied
to reserving cash from My Portfolio. The two must stay distinguishable. *(Prod
already honours this: the shipped tab is "Live Trading", and the landing copy
states outright that "Live" names the direction the board runs, not brokered
execution.)*

#### What prod shipped, and the three invariants any advance engine must respect

The shipped board is a **Season 0 preview**: real Competition curves rendered
under season chrome, plus a banner saying nothing on it has advanced. Four facts
constrain Phase 3, and three of them are guarded by tests that will fail loudly:

1. **There is no season engine and no `live` period.** `VALID_PERIODS =
   ("contest", "daily")` (`service.py:50`); `live` and `season` are *frontend*
   vocabulary only (`normalizeBoardPeriod()`, `js/leaderboard.js:636`).
   `_normalize_period` coerces anything unrecognised back to `contest` rather
   than 4xx-ing, so `GET /api/v1/leaderboard?period=live` returns a perfectly
   successful **200 carrying the Competition board**. Adding `"live"` to
   `VALID_PERIODS` is the natural first commit and is safe — see (2).
2. **The preview banner is anchored on evidence of an advance, not on the
   period.** `isLivePreview()` is `isLiveBoard() && !seasonHasAdvanced(payload)`,
   and `seasonHasAdvanced()` tests `season.last_advanced_date` /
   `trading_days_elapsed > 0` — fields only a real advance can write. This is
   deliberate and is pinned by a source-shape test: an earlier `payload.period
   !== 'live'` form would have let the season engine's first commit silently
   clear every banner while nothing had run. **Do not simplify it back to a
   period check.** The happy consequence is that C8 needs no banner work at all —
   the first successful advance clears it by writing the field.
3. **Season 0 is falsy.** Every read of the season number goes through
   `displayedSeasonNumber()` + `Number.isFinite`, because `season?.number ? … :
   '—'` renders the shakedown season as no season at all. Any payload C8 emits
   must keep season 0 distinguishable from a missing season.
4. **Season length is `SEASON_TRADING_DAYS = 10`** (`js/leaderboard.js:276`),
   used as the denominator of the season progress meter when the payload omits
   `trading_days_total`. An advance engine emitting a different total will render
   a progress bar that disagrees with the chrome.

#### Seasons reset — a decision made after this document and adopted here

`2026-08-15-live-trading-leaderboard-ui.md`, decision 3: **every season resets.
Entries do not carry across seasons; joining is a per-season decision.** That
document records it as the cost control — perpetual entries would bill every
signup ever, every night, forever.

This is *compatible* with what this spec already said ("Live Trading is a slot,
not a ledger") and makes it sharper: a slot is per-season by construction, so
there is no carry-forward state to design and no lifetime grant to track on this
board. Two consequences to carry into Phase 3 rather than rediscover:

- The §Cost model figure of "1 entry per season" is now a *structural* property,
  not a policy that could be relaxed. Relaxing it re-introduces the unbounded
  nightly bill that decision 3 exists to prevent.
- Re-entry each season is the return-visit mechanic. That partially answers the
  watch item in §Cost model about the two-week gap — the user has to come back to
  re-enter, which a perpetual entry would not require.

**Consequence for sequencing:** C8 is currently the only thing standing between
prod and Season 1 of a board users can already open. Tracked as **issue #354**.
See §Rollout.

## Season configuration

**Competition lives in config. Live Trading seasons are derived, not
configured.**

A new season opens every two weeks, so hand-editing config per season would mean
a deploy per season — unworkable. Seasons are computed from a template plus date
arithmetic, in the spirit of `daily_window_dates()` (`service.py:79-91`), which
already derives the daily window.

**The derivation changed with the cadence (2026-08-15).** The original scheme was
`season_id = fwd-<ISO year>-W<week>`, which works only for week-aligned seasons —
a fortnight has no ISO name, and the ISO-week counter resets mid-season at every
year boundary. Derive from an explicit anchor instead:

```
season_number = floor(trading_days_between(anchor, today) / 10) + 1
season_id     = "live-s<season_number>"
window        = the 10 consecutive trading days starting at
                anchor + 14 calendar days × (season_number - 1)
```

Counting in **trading days, not calendar days**, is what keeps a holiday week
from silently shortening a season to 9 sessions. Before the anchor date the
derivation yields **season 0**, which is exactly the shakedown state prod renders
today — so the preview is a natural value of this function rather than a special
case bolted beside it. Keep it distinguishable from a *missing* season (see
invariant 3 above). No table, no per-season deploy.

The Competition config:

```json
{
  "season_id": "competition-s1",
  "kind": "competition",
  "label": "Competition Qualifier",
  "perpetual": true,
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
  "house_reference_entry_id": "competition_s1_house_reference"
}
```

The Live Trading template carries the same shape with `kind: "live"`,
`perpetual: false`, an `anchor_date`, `trading_days: 10`, no `window` (derived)
and no `attempts_granted` (a Live Trading entry is a slot, so the column defaults
to 0).

`kind: "live"` matches the period string the frontend already sends, so adding
`"live"` to `VALID_PERIODS` (`service.py:50`) is the one-line backend change that
makes the shipped tab stop silently serving the Competition board. It is safe to
land on its own — the preview banner does not key on the period.

## Data model

Three tables beside `agent_runs` on the run-history database
(`AGENT_RUNS_DATABASE_URL`). **No foreign keys to `owner_user_id` or `agent_id`**:
accounts live behind `USERS_DATABASE_URL` and agents behind
`CONTENT_DATABASE_URL`, in different Neon projects. They are opaque strings by
necessity, and no code path may join across them.

```
leaderboard_entries
  entry_id           TEXT PK        -- namespaced, must not collide with lb_*
  season_id          TEXT NOT NULL
  owner_user_id      TEXT NOT NULL  -- opaque; different database
  agent_id           TEXT NOT NULL  -- opaque; different database
  alias              TEXT NOT NULL  -- per-season display name; defaults to the
                                    -- account display name, editable until the
                                    -- entry's first attempt is submitted
  best_run_id        TEXT           -- FK-in-spirit to agent_runs.run_id
  best_return        REAL
  attempts_granted   INTEGER NOT NULL DEFAULT 0   -- competition seasons only
  attempts_used      INTEGER NOT NULL DEFAULT 0
  created_at, updated_at

leaderboard_attempts
  id                 INTEGER PK
  entry_id           TEXT NOT NULL
  run_id             TEXT NOT NULL
  attempt_no         INTEGER NOT NULL
  instruction_text   TEXT NOT NULL      -- the submitted string, snapshotted
  instruction_sha256 TEXT NOT NULL
  total_return       REAL
  h6_passed          INTEGER NOT NULL   -- BOOLEAN on the Postgres twin
  failure_kind       TEXT               -- NULL | 'h6_rejected' | 'infrastructure'
  created_at

forward_positions                       -- Live Trading seasons only
  entry_id           TEXT NOT NULL
  as_of_date         TEXT NOT NULL
  cash               REAL NOT NULL
  positions_json     TEXT NOT NULL
  equity             REAL NOT NULL
  PRIMARY KEY (entry_id, as_of_date)
```

Three notes:

- **`instruction_text` is not optional.** A hash is one-way; without the text you
  cannot re-run an entry, show a user what they submitted, or perform the
  graduation hand-off. The previous draft stored only the hash.
- **`failure_kind` must be set at the point of failure**, never inferred later
  from a null return. Per the fail-closed lesson in `CLAUDE.md`, the dangerous
  version is the one where "rejected on integrity" and "OpenRouter 502" look
  identical.
- **The `h6_passed` type divergence is called out explicitly** because
  `test_store_twin_parity.py` compares column *names* only and will not catch it.

One entry per user per season. `attempts_granted`/`attempts_used` **is** the
credit ledger v0: a season grants, a submission decrements, and a future billing
path tops up — same table, no migration.

## Submission path

- **C1 — `LLMAgentStrategy` accepts an instruction.** Add
  `self.strategy_prompt = (self.config.get("strategy_prompt") or "").strip() or None`
  and pass `strategy_prompt=self.strategy_prompt` at `llm_agent.py:176-182`.
  `make_trading_decision_with_llm` already declares the parameter
  (`portfolio_manager.py:233`) and already threads it (`:453`). No downstream
  signature changes.
- **C2 — deliberately void.** The source reconciliation proposed giving both
  sides a shared template so house and user entries differ only by instruction.
  This design rejects that in favour of the Open Track house reference: the
  existing seven entries are not touched, and comparability is achieved within
  the Open Track. The number is retained so this spec maps onto the analysis.
- **C3 — one submission entry point.** A new `deploy_user_entry(...)` in
  `domain/leaderboard/service.py`, modelled on `deploy_model_run`
  (`service.py:926-1041`). It builds a config carrying the season's pinned model
  plus `strategy_prompt`, calls the unchanged `get_strategy()`, and **reuses
  `service.py:979-989` and `:996-1013` verbatim** for bar fetching, warm-up, the
  run, counters and the H6 guard. Capital, window, warm-up, the `+1 day` bump,
  the no-coverage-filter timestamp set and integrity all come for free because
  it is literally the same code.
- **C4 — pin what was submitted.** `_llm_run_metadata` (`service.py:894-921`)
  gains `agent_id`, `season_id` and `strategy_prompt_sha256`.
- **C5 — an authenticated ingress.** A new owner-authenticated, rate-limited
  route in `api/routers/leaderboard.py`, returning 202 and queueing, following
  the existing `/daily/refresh` shape.
- **C6 — leave Loop A alone.** No changes to `HourlyBacktester`,
  `MAX_BACKTEST_INITIAL_CAPITAL`, the fetch window or the coverage filter.
- **C7 — collapse the one-step pipeline into `strategy_prompt`.** At submission,
  read `pipeline[0]["prompt"]` and pass it as `strategy_prompt`; **reject any
  entry with `len(pipeline) != 1`**. This is what makes "season entries are a
  single instruction" an enforceable validation rather than a copy promise, and
  it keeps user entries on the same `SAFE_TRADING_PROMPT` path as the house
  reference. The step's `outputFormat` is **ignored** — the UI must say so,
  because silently dropping a field the editor displays is exactly what produces
  "the leaderboard is broken" reports.
- **C8 — the nightly advance job.** A daily job that loads each active Live
  Trading entry's `forward_positions` row, fetches the day's bars, runs the
  pinned model over that day's steps, executes via
  `domain/trading/execution.py`, and writes the next row. Idempotent on
  `(entry_id, as_of_date)`.

  The table keeps the name `forward_positions` rather than `live_positions`
  deliberately: the whole point of §Live Trading is that "live" names the
  *direction* the board runs and never brokered execution, and a table called
  `live_positions` sitting next to `infrastructure/brokers/` is the one name most
  likely to be misread as real positions by someone skimming.

  **C8 is not only a Phase 3 concern.** It is the sole missing piece between prod
  and Season 1 of a board users can already open — see §Rollout.

## Integrity, abuse and spend

### H6

Applies unchanged and for free, because C3 reuses the guard call site.
`_reject_if_llm_fallback` (`service.py:833-892`) rejects an entry whose model
drove under `MIN_LLM_DECISION_COVERAGE = 0.95` of steps, keyed on
`llm_decisions` (success-exit counter) rather than `llm_calls` (billing
counter). Evidence this is survivable on a nano model: the house Nemotron entry
is published on the live board, so it cleared H6 through this same harness.

`allow_fallback` must remain absent from every HTTP surface, as it is today.

### Refunds

- **H6 rejection does not refund.** An instruction that makes a nano model emit
  unparseable output *is* a worse instruction, and refunding removes the only
  pressure toward instructions the model can follow. Recorded with
  `h6_passed = 0`, `failure_kind = 'h6_rejected'`; it can never become
  `best_run_id`.
- **Infrastructure failure refunds.** `failure_kind = 'infrastructure'`
  increments nothing.

### Abuse

Signup is free, instant and unverified today, while each attempt spends real
money. Ten accounts is ten minutes of work; a script makes it a thousand.

- **Email verification at account creation.** Reuses the Brevo code flow built
  for the email-change feature. **Existing accounts are grandfathered as
  verified** — forcing verification on next login would lock out real accounts
  in the prod database. This makes `BREVO_API_KEY` a hard signup dependency
  (see Deploy prerequisites).
- **A hard global spend ceiling**, `LEADERBOARD_MONTHLY_BUDGET_USD`, default
  **$50**, checked before dequeue and refusing with a clear user-facing message
  rather than failing silently. Verification raises the cost of abuse; it does
  not eliminate it. The precedent is on this repo — a public anonymous GET in
  PR #325 triggered seven billable LLM deploys, inside pytest.
- Additional per-run authorization checks may later be added at the
  auth/authz layer ahead of every backtest and paper-trading run; this design
  does not preclude them.

### Cost model at scale

Free tier: **5 lifetime Competition attempts + 1 Live Trading entry per season.**

| Policy | At 100 users | At 500 users |
|---|---|---|
| Competition, 5 lifetime | $36 one-time | **$180 one-time** |
| Live Trading, 1 entry/season | ~$7/month | **~$34/month** |
| *(rejected)* Competition +1/day refill | ~$216/month | ~$1,080/month |

The monthly figures are **unchanged by the two-week cadence**: a season costs
twice as much and arrives half as often (§Measured cost per entry).

A daily Competition refill was considered and rejected: it is 97% of total spend
and buys a return-visit mechanic the season cycle and its result email already
provide. Daily refills become the first thing credits buy once billing exists —
which is the shape the ledger was designed for.

> **Watch item added 2026-08-15.** The rejection assumed a *weekly* return-visit
> mechanic. A two-week cycle is a materially weaker habit loop, and the daily
> board that used to fill the gap between seasons is retired — PR #352 replaced
> the Daily Leaderboard tab and commented out its nightly cron, so **nothing
> refreshes automatically in prod today**. If retention is the goal, the gap
> between season close and next season open now needs its own answer. Do not
> close it by re-arming `LEADERBOARD_DAILY_AUTO_DEPLOY`: that flag lets an
> anonymous GET trigger seven billable model deploys, which is the exact hazard
> its strict opt-in exists to prevent.

**Contingency:** if the Phase 0 probe forces the pinned model to DeepSeek V4
Pro, every figure multiplies ~10.5× and the free tier at 500 users becomes
~$1,890. **The budget holds and the grant shrinks** (5 lifetime Competition
attempts → 1; Live Trading stays 1 per season).

## The board

Default visible series, never more than 5–8 lines:

1. DJIA / buy-and-hold baseline — the honest bar
2. The seven house models as a **thin, grey, unlabelled cluster** — "the models
   we tried"
3. The Open Track house reference — the specific number being challenged
4. Top 3 entries
5. **The signed-in user's own entry, pinned unconditionally**, at any rank

Everything else stays available but hidden. This is a **default-selection
policy, not a charting feature**: it decides the initial contents of the
existing `hiddenSeries` set, which the curve picker and custom legend already
manage.

Pinning the user's own curve is deliberate. Top-N-only means most entrants see a
chart they are absent from, which is the moment they stop caring.

**Rank against the pinned model's house curve as the headline; show
beat-the-market as a separate, scarce badge.** Season 1 pins a model whose house
result is `-0.22%`, so the board will accumulate entries that beat the house and
still lose to buy-and-hold — and it is likely that most of the field loses to
buy-and-hold, since six of seven house models already do. Headlining
"+1.4% vs. the house instruction" stays truthful when the field underperforms;
headlining a market-beating claim would not. **We are not in the business of
implying the market is easy to beat.**

Labels must distinguish *a different model* from *a different instruction*, or
"beat them" is ambiguous once both tracks share a chart.

**Not in v1:** a percentile band for the hidden field. Correct at 500 entries,
speculative at 12.

## Information architecture

The Competition tab keeps its four subtabs. **The roster changed under this
section on 2026-08-15** (PR #352): the Daily Leaderboard subtab is gone and Live
Trading took its slot, so the mapping below is rewritten against what prod
actually renders (`app.html:1431-1434`).

| Subtab (`data-competition-tab`) | Today in prod | After |
|---|---|---|
| `leaderboard` — Competition Leaderboard | fixed-window house board | + user entries and the attempt ledger |
| `live` — Live Trading Leaderboard | Season 0 preview, serving Competition curves under season chrome | the real season board, once C8 advances it |
| `participants` — Participating Teams | permanent empty state | **Participating Traders** — the entrant list, finally populated |
| `about` | contest rules | + season rules |

> **This is strictly less work than designed.** The original plan was to turn one
> subtab into a season board carrying a Replay/Forward switch modelled on the
> `%`/`$` toggle. Prod already ships the two boards as **two separate subtabs**,
> so the switch is unnecessary — delete that idea rather than building it. What
> remains is populating boards that already have their own navigation.

Two renames, no new navigation. **"Teams" becomes "Traders"** in user-visible
copy — each entrant is one person. Note the styling layer keeps `kind: 'team'`
and `TEAM_COLOR_PALETTE` internally; that is deliberate (renaming the series kind
would churn `js/leaderboard.js` for no user-visible gain), but it means the word
persists in code while the UI says "trader".

*(The Replay/Forward switch described here is withdrawn — prod ships the two
boards as separate subtabs. See the note above.)*

## Isolation contract

**Boards must not be able to break each other.** Binding requirements:

1. **Separate endpoints per board**, not one combined payload.
2. **Each landing section fetches its own data** and renders its own empty/error
   state. One failed fetch must not blank the page.
3. **A server-side kill switch per board** — two env-backed booleans making the
   board's endpoint return a *disabled* payload the frontend renders as a
   maintenance state. Designing the disabled state matters: an unhandled backend
   exception escapes `CORSMiddleware` un-headered and presents in the browser as
   a CORS error, so "we turned it off" and "it is broken" would otherwise look
   identical at exactly the moment they must be told apart.
4. **Separate queue paths**, so a Competition queue backup cannot stall the Live
   Trading advance job.

A consequence: swapping which chart leads the landing page is a **one-line move**
in `landing-page.tsx:18-22` plus the standard bundle refresh. No swap mechanism
is built — building one would be more complex than the swap.

## The funnel

### Landing

> **Largely delivered by PR #357 (2026-08-15), by different means. Read the
> status note before implementing anything in this subsection.**
>
> **Done:** the board is above the fold. #357 extracted the chart into a new
> `BoardPreview` component and mounted it in `Hero.tsx:144`, rather than
> reordering `landing-page.tsx`. `Race` never moved — it is still the last
> section, and `FooterCTA` still reads `Talk → Test → Race`.
>
> **Done differently, on the other surface:** the *live* board landed on the
> `/app` home screen, not the landing page. `home-page.js:1518` fetches
> `GET /api/v1/leaderboard` and renders real standings with a labelled sample
> fallback.
>
> **Reversed:** the Vercel landing page deliberately ships **illustrative**
> data. `BoardPreview.tsx` hardcodes `SAMPLE_CURVES`/`SAMPLE_STANDINGS` under a
> visible "Illustrative example" badge, and
> `test_landing_copy_register.py::test_illustrative_example_label_appears_at_least_twice`
> now **requires** that label to appear at least twice in the shipped bundle.
> The requirement below to remove the badges is therefore not merely undone —
> acting on it means deleting a guard a merged PR added on purpose.
>
> **If the live-data-on-landing idea is revived, argue it on the merits first.**
> The landing page is served from Vercel while the API is on Render's free tier,
> which spins down; an above-the-fold cross-origin fetch to a cold backend is the
> weakest possible first impression, and that is a real reason to prefer a
> labelled sample here even though `/app` gets live data. Whatever is decided,
> the honesty rule is non-negotiable: **sample data must be visibly labelled as
> sample.** Both current cards are.

Wire the existing `Race` section to `GET /api/v1/leaderboard`, following the
fetch pattern `MarketTicker.tsx:13-19,77-117` establishes. Remove the
"Illustrative" badges (`Race.tsx:74,107`) once the data is real, and fetch
same-origin through the Vercel rewrite rather than hardcoding
`agentictrading.onrender.com`.

**Layout:** move `Race` directly under the hero (it is currently section 5 of 5,
immediately before `FooterCTA`). Above the chart, a **season status strip** —
*"Season 12 · day 3 of 10 · 44 traders"* or *"Next season opens soon"* — which
doubles as the between-seasons CTA. Below the Competition chart, the Live Trading
chart as its own clearly-labelled section. When Live Trading proves out, the two
sections swap by reordering one line.

*(A season strip already exists on `/app` — `renderSeasonStrip`, hidden on the
Competition board. The landing equivalent should read from the same payload
fields, `trading_days_elapsed` / `trading_days_total`, rather than inventing a
second source of season truth.)*

`Race.tsx` draws with **recharts** (`:3-12`), not Chart.js. The landing board is
therefore a second implementation; none of `js/leaderboard.js`'s custom legend,
endpoint labels or tiering transfers. Accept the divergence for v1, but hold the
visual grammar constant — dashed grey baseline, bold entry lines — so it reads
as one product.

Copy: *"Most models lost to buy-and-hold. Can you beat them with a better
instruction?"* — "instruction" rather than "strategy", both because it is what
the editor already calls it (`label: "Trading instruction"`) and because
`domain/strategies/` is an existing product noun.

Renumber `Talk`/`Test` to 01/02 and make the board an unnumbered frame above
them. **This copy is Allan's** (`storyline.ts`, `WhyCare.tsx:5`,
`FooterCTA.tsx:10`) and needs his sign-off.

> **Mostly done as a side effect of #357, and one loose end.** `Talk.tsx:13` and
> `Test.tsx:143` carry `01 — Talk` / `02 — Test`, and the board's `03 — Race`
> line is gone — which is exactly the end state described here, reached without
> anyone asking for the sign-off. The loose end is `FooterCTA.tsx:10`, which
> still restates the old sequence as `Talk → Test → Race`. It is now the only
> place claiming an ordering the page no longer has.

Post-signup redirect is unchanged: `/app?view=agents` with `nav-state`
pre-seeded — plus a verification interstitial (see Abuse).

### Onboarding

**A guided tour is explicitly rejected.** Tours explain the UI, but the observed
problem is that users do not know the goal; tours are stateless once dismissed;
and the repo has zero tour infrastructure.

Instead, a **persistent Season checklist card** pinned at the top of My Agents —
a card in the existing shelf renderer, not an overlay framework:

```
Competition Qualifier · Nemotron 3 Nano           5 attempts left
The house instruction lost 0.22%. Beat it.

  [x] Your starter agent is ready
  [ ] Write its trading instruction        [ Configure ]
  [ ] Run a qualifying attempt             [ Run attempt ]
```

On first successful attempt it collapses into a permanent season HUD —
`Competition · rank 34 of 112 · 3 attempts left · next season opens soon` — so it
never becomes dead weight.

The checklist and the ledger are the **same state object**: the card renders
`attempts_remaining`, the integrity rule enforces it, and future billing tops it
up. Being server-side, it survives the browser change that today's
`localStorage` onboarding guard (`ensureDefaultFoundationAgent`,
`app.js:1703-1776`) does not.

Attempts must show **used, remaining, and that a started run cannot be
cancelled**, stated before submission rather than discovered after.

One welcome screen shown once after signup, stating the goal in a sentence with
a button that scrolls to the checklist. One screen, not a sequence.

~~**Fix while here:** wire `#homeGetStartedBtn` (`app.html:411`) to the
checklist, or remove it.~~ **Withdrawn 2026-08-15** — the button was already
wired on 2026-07-25 (`08c85aa`); see the correction in §Background. If the
checklist ships, deciding whether this CTA should point at it instead of the
agents playground is a live question, but it is a redirect, not a fix.

### Submission surface

A **"Submit to Season"** action on the agent configuration page, enabled only
when the agent has exactly one pipeline step and disabled with a stated reason
otherwise — which is also how future multi-step agents will present. Submission
**snapshots** the instruction into the attempt row; referencing the live agent
would let a Wednesday edit silently change a curve that is mid-flight.

While an attempt runs, the user sees **queue position and estimate**, not a
blocking modal, reusing the progress presentation My Agents backtests already
have.

### Graduation

On any completed attempt, the entry detail offers *"run this instruction on your
own model and universe"* — one click from the frozen season into Loop A, which
already exists. **Carry the instruction text only**, everything else at Loop A
defaults, with copy stating the difference outright: *"Different engine and
capital — this will not reproduce your season score."*

Carrying window/universe/scaled-capital instead was considered and rejected: it
looks more helpful and quietly implies a comparability that does not exist, so
the first user who gets a different number reasonably concludes the leaderboard
is broken.

### Community → Agent Marketplace

Change exactly two user-visible strings: the nav label (`app.html:195`) and the
page title (`app.html:1600`). They must change together —
`test_frontend_marketplace_placement.py::test_community_page_header_matches_the_nav_button`
asserts they match, and passes if both move.

**Do not rename** the `community` page key, `#communityView`, `NAV_VIEW_MAP`
entries or the `nav-state` localStorage value: live bookmarks break for no
user-visible gain. `?view=marketplace` already exists as a legacy alias, so the
rename makes the legacy alias the accurate one.

A user publish path is out of scope for v1. The competition creates the natural
demand for it, so it lands better as v1.1 with a real corpus.

## Measurement

Two systems with separate jobs:

- **Vercel Analytics** (already shipped) — acquisition: pageviews, traffic
  sources, landing behaviour.
- **A first-party event beacon** to the backend — the funnel, on both surfaces.
  Permitted by both CSPs' `connect-src` without any change, $0, and unaffected
  by the ad-blocking that a quant-adjacent academic audience does at high rates.

Exactly six events: `landing_view`, `board_interact`, `cta_click`,
`signup_complete`, `instruction_saved`, `attempt_submitted`. They answer every
Phase 1 question and nothing more.

**Acquisition channel:** the professor's and lab's academic network, plus the
existing Discord. One channel, designed for deliberately.

**Share artifacts:** a static site-wide `og:image` immediately (today
`index.html:12` requests `summary_large_image` and supplies none — the worst of
both), then a client-side canvas "download your result card" with Phase 3.
Dynamically rendered per-entry OG images are **rejected**: Render's free tier
spins down and link-preview crawlers abandon in seconds, so most first shares —
exactly when the link matters — would render nothing.

## Notifications

One email at season close: *"Season 12 closed. You placed 8th of 44. Next season
opens Monday."* Built on `send_email` (`sender.py:45`). Requires an opt-out from
the start. `send_email` is `async` while ATL routes must stay sync `def`
(#292), so it is called from the cron path, never from a request handler.

Discord posting of each season's top entries is **deferred but designed for**: the
notification layer takes a channel abstraction with email implemented and
Discord stubbed, so adding it later is a new channel rather than a refactor.

## Non-goals

- Broker-backed live paper trading, real-time execution, or order submission.
  `execution/paper_backend.py` stays a stub. Live Trading is **simulated** and
  does not depend on it.
- Real capital. Ever, on this surface.
- Changing the published Model Track curves.
- Raising `MAX_BACKTEST_INITIAL_CAPITAL`.
- A credit/billing system. The ledger anticipates it; pricing is out of scope.
- Multi-step pipelines as season entries.
- A user publish path into the marketplace.
- Percentile/field bands on the chart.
- A runtime chart-swap mechanism.

## Verification

- **H6 on the new insert path.** The guard's docstring states it is applied on
  *both* insert paths; `deploy_user_entry` makes a third. `test_deploy_guard.py`
  needs mirror coverage or an entry can bypass integrity.
- **Postgres twins for all three new tables**, enforced by
  `test_store_twin_parity.py`. That guard parses source text, so it cannot see
  f-string DDL and compares column *names* only — `NOT NULL` divergence is
  invisible and must be checked by hand.
- **Route-contract freeze golden sets updated in the same commit as C5**, or
  every open PR reddens. Per-router and full-app freezes drift independently.
- **`len(pipeline) != 1` rejection** tested at the API boundary, not only the UI.
- **Spend-ceiling test**: the dequeue refuses past the ceiling and the refusal is
  visible, not silent.
- **A pinned-model assertion**, so a silent switch to a frontier model cannot
  multiply spend 190× unnoticed.
- **Isolation tests**: each board's endpoint disabled independently, and the
  frontend renders the maintenance state rather than a blank page.
- **Frontend copy guards** for the checklist and CTA strings, following
  `tests/_frontend_source.py`.
- **Cache-buster bumps** for every touched frontend asset, each versioned
  independently. The landing bundle needs none — content hashes are the bust.

## Deploy prerequisites

- **`BREVO_API_KEY` + `ACCOUNT_EMAIL_FROM` become required for signup**, not
  just for email changes. Unset, account creation fails. This is a change in
  severity and must be set before the verification gate merges.
- **`OPENROUTER_API_KEY` must be set in the Render dashboard.** Nemotron is the
  only board entry not on CommonStack, and OpenRouter is never auto-selected
  (`providers/__init__.py:11`). Unset, submissions silently fail.
- `LEADERBOARD_MONTHLY_BUDGET_USD` — default $50.
- Two board kill-switch booleans.
- `LEADERBOARD_DAILY_AUTO_DEPLOY` stays strict opt-in. Never restore any default
  that enables it implicitly.
- Render env writes are single-key PUT only; a bulk PUT wipes the list, and they
  do **not** trigger a redeploy — so a ceiling change takes effect on next
  deploy, not immediately.

## Rollout

Each phase is a **separate PR**, and each must be independently shippable and
independently disableable.

**Phase 0 — the probe. A gate, not a phase.** C1 plus six deliberately opposite
instructions (aggressive momentum, defensive cash-heavy, equal-weight
buy-and-hold, contrarian mean-reversion, verbose-analytical, and one nonsense
control) run through `deploy_model_run` on **two models**: Nemotron and DeepSeek
V4 Pro. Total **$4.97**. Needs `OPENROUTER_API_KEY` locally.

The entire design rests on an unverified assumption — that better instructions
produce better returns on a 30B nano model. If the spread across genuinely
opposite instructions is under ~1pp on Nemotron but not DeepSeek, pin DeepSeek
and apply the contingency above. **If both are flat, the instruction axis does
not exist and Phase 2 must not be built.**

> **Correction (2026-08-15) — the stated reason for expecting failure was
> wrong, and the gate is more favourable than this document claimed.**
>
> The original sentence read "…a 30B nano model **whose prompt is dominated by an
> unconditional `SAFE_TRADING_PROMPT`**". That is not what the code does. A
> caller-supplied instruction **replaces** the `SAFE_TRADING_PROMPT` strategy body
> outright; only a fixed execution-contract scaffold is concatenated after it —
> `CUSTOM_STRATEGY_OUTPUT_CONTRACT` (`validator.py:667-711`), joined by
> `create_custom_prompt` (`:730`). The instruction gets the entire strategy slot,
> not a corner of one.
>
> This does not remove the need for the gate — a nano model may still fail to act
> on instruction content, which is the real question — but it does remove the
> specific mechanism this document predicted would suppress the signal, and it
> raises the prior on a pass. Two knock-on effects worth holding in mind when
> reading the probe results:
>
> - **A flat result is now more damning, not less.** With the instruction owning
>   the whole strategy body, "the model ignores it" is the remaining explanation,
>   and there is no prompt-dilution fix to try next.
> - **The control instruction matters even more than stated.** Since instructions
>   fully replace the strategy body, a nonsense control still produces a
>   *syntactically valid* run. A control landing mid-pack means the model responds
>   to having a strategy body at all rather than to its content — the outcome that
>   passes a naive spread check while the board ranks noise.

The five non-control instructions become the seed field, so this spend also
solves the cold-start problem below.

**Phase 1 — evidence and measurement.** **C1 only**, plus the six seed entries
(house reference + the probe's five non-control instructions) added as
`leaderboard.json` config entries carrying `strategy_prompt` — **no new tables,
no ingress, no queue**. Hero chart on real data, `Race` promoted above the fold,
seeds **visibly labelled house-authored**, the first-party beacon, static
`og:image`, CTA copy, Community→Agent Marketplace rename, "Teams"→"Traders".
No user entries yet.

> **Status 2026-08-15: mostly overtaken. Do not execute this phase as written.**
> The landing-page half was delivered by #352/#357 through different components,
> and the "hero chart on real data" item was resolved in the opposite direction
> on the landing page (labelled sample data, now guarded). What genuinely remains
> from Phase 1 is a short list — C1, the seed field, `og:image`, and the two
> renames — enumerated task-by-task with current line numbers in the companion
> plan document. Read that table before touching anything here.

**Incremental cost: ~$0.** The six curves are the Nemotron half of the Phase 0
probe; C1 is what makes a config entry able to carry an instruction at all, so a
seed and a probe run are the same object. Regenerating them costs $0.43.

A board with one entry is not a board, which is why the seeds matter: the
Open Track would otherwise ship containing only the house reference, in the
phase whose entire purpose is testing whether the board converts. Seeds are
config-driven house entries, not database rows — the committed `backtest.db` is
the prod database, so seeding by committing rows is not an option.

**Phase 2 — the Competition qualifier.** C3/C4/C5/C7 (C1 shipped in Phase 1),
`leaderboard_entries` and `leaderboard_attempts` with their Postgres twins, the
attempt ledger, email verification at signup, the checklist, the Submit action,
the queue (**two workers**, one env var, redeploy to change), graduation.

Two concurrent workers rather than the three or four originally floated: issue
**#202** reports blocking sync I/O on exactly the leaderboard routes, and the
agent-scale investigation measured throughput collapsing under load. Starting at
two makes the first real measurement cheap to recover from.

**Phase 3 — Live Trading seasons.** C8, `forward_positions` and its twin,
derived two-week seasons, the season-close email with opt-out, the share-card
download, the notification channel abstraction with Discord stubbed.

Ordering is forced, not chosen: Live Trading entry is gated on a completed
Competition attempt, so Phase 3 cannot precede Phase 2.

> **C8's priority changed on 2026-08-15, and the forced ordering above does not
> apply to it.** The Live Trading Leaderboard is *in prod now*, in a Season 0
> preview that has never advanced, because no advance engine exists. C8 is what
> makes that shipped board real — and it needs no user entries, no attempt
> ledger, and nothing from Phase 2 to run against the existing house roster.
>
> **Already filed as issue #354** — make the scheduling argument there rather
> than opening anything new.
>
> **Consider splitting C8 out and landing it before Phase 2.** The gating
> argument ("Live Trading entry requires a completed Competition attempt") is
> about *user entry*, and C8 with a house-only roster has no users in it. Landing
> it early converts a preview banner into a board that moves, which is the
> `Stay` half of the goal, and it does so without waiting on the Phase 0 probe —
> advancing seven existing house entries does not depend on whether *instructions*
> move returns.
>
> Two things to carry into that decision: the four invariants in §Live Trading
> (the banner clears itself when the advance writes `last_advanced_date`, so no
> frontend work is implied), and the cost. Advancing the full house roster nightly
> is the expensive shape — price it against the pinned-model-only alternative
> before arming any schedule, and remember that PR #352 paused the nightly cron
> precisely because it was deploying seven billable models for a board nobody
> could open.

**Watch item:** Phase 1 is the only phase shipping without users being able to
act. If Phase 0 comes back strong, consider collapsing 1 and 2 — a hero chart
with a CTA leading to a season nobody can enter is the dead-CTA failure at the
largest scale yet.

> **This watch item is now live, not hypothetical.** #352/#357 shipped the hero
> chart without the entry path, so the state this warned about is the state prod
> is in. It is handled honestly for the moment — the landing board is labelled
> illustrative and the Season 0 banner says nothing has advanced — but the
> mitigation is a disclaimer, not a destination. This strengthens the case for
> collapsing what remains of Phase 1 into Phase 2 rather than shipping another
> display-only increment.

## Recorded assumptions

Stated rather than left implicit; correct any that are wrong before planning.

1. **Existing prod accounts are grandfathered as verified.**
2. **Live Trading seasons source bars from Alpaca**, same as Competition.
3. ~~**The Daily Leaderboard is untouched** by this work.~~ **Void 2026-08-15.**
   PR #352 retired the Daily Leaderboard tab in favour of Live Trading and
   commented out its nightly `schedule:`. The route, the secret and the cron line
   all survive in `.github/workflows/daily-leaderboard.yml` because the season
   engine's nightly advance is the same call — re-enable by uncommenting, do not
   rewrite. Consequence: **nothing refreshes any board automatically in prod
   today.**
4. **`attempts_granted = 5`** for the Competition qualifier, lifetime.
5. **Competition keeps the current window** (`2026-04-15 → 2026-05-15`) — its
   look-ahead exposure is acceptable because Live Trading is the competitive
   board.
6. **The pinned model survives Phase 0.** If not, the contingency applies.
7. **Added 2026-08-15: the two-week cadence is taken as fixed**, because prod
   already renders it. If a future round wants weekly back, it is a UI change in
   `js/leaderboard.js` as well as a config one — reopen §Decisions taken, where
   the instruction-lock trade-off is recorded as unresolved.

## References

- Central DB: `knowledge/opt-employment-2026.md` § "The 2026-08-08 Pivot" —
  professor's goal, nof1 competitive research, verified-unbuilt scope.
- Central DB: `decisions/2026-08.md`, entry 2026-08-08.
- `docs/superpowers/specs/2026-08-05-asset-class-shelves-design.md` — the shelf
  renderer the checklist card sits inside.
- PR #325 (`45ccbc0`) — daily refresh cron and status UI.
- ~~PR #326 — open at time of writing, touches `js/leaderboard.js`; rebase onto
  it.~~ **Merged 2026-08-09**; the rebase constraint is void.
- **PR #352 (`7db504d`, merged 2026-08-15)** — the two-board redesign this spec
  was reconciled against: Competition + Live Trading, two-week seasons, Season 0
  preview, Daily Leaderboard tab retired and its cron paused.
- **PR #357 (`60fa01f`, merged 2026-08-15)** — leaderboard-first landing and
  `/app` home; added `BoardPreview.tsx` and the "Illustrative example" guard.
- **`docs/superpowers/specs/2026-08-15-live-trading-leaderboard-ui.md`** — the
  newer board-side spec; authoritative on the Live Trading UI and payload
  contract. See "Where authority now lives" at the top of this document.
- **Issues #354** (season engine = C8) and **#355** (qualifier gate and
  `instruction_sha256` scope — both stated to block this PR and Phase 2).
- Issues #145 (scheduler), #202 (event-loop blocking), #230 (`decide()` seam),
  #258 (run cancellation).
