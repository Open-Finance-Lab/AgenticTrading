# Live-Board Fixes Implementation Plan (index)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This file is the index: the tasks live in the seven per-PR files listed under **Phases**. Each per-PR file inherits this file's Global Constraints and Decisions.

**Goal:** Until Season 1 moves the Live Leaderboard to the new engine, keep the legacy-engine board correct and honest. That covers seven things:
- NYSE early closes;
- trade memory across nights;
- one model's failure no longer ends the night for the others;
- a $50/month spend stop enforced in code;
- a roster that CommonStack actually serves;
- daily Sharpe with its 95% interval;
- copy that says what the board is.

**Architecture:** Seven PRs on the legacy engine. None changes fees, slippage, fills or the prompt.
- **Calendar.** Early closes are added where the session calendar is owned: the dates in `us_market_calendar.py`, the close times in `sessions.py`. `live.py` reads the close per day.
- **Budget and roster.** Spend is tracked per month and per model in a new engine-neutral leaf module, `domain/leaderboard/live_budget.py`. It keeps one `agent_runs` row per month and model and adds no DB methods. A per-entry `live_from` key gates when a new model joins.
- **Sharpe.** It moves to a read-time `metrics.daily_sharpe_interval`, served on both boards. The browser only displays it.

**Tech Stack:** Python 3.11 FastAPI backend (`dashboard.backend`); SQLite locally and Neon Postgres in prod through the existing store twins; the vanilla-JS dashboard, tested under `node -e`; the Vite/React landing (P7 PR B only); pytest from the repo root.

**Spec:** `docs/superpowers/specs/2026-10-14-atl-direction.md`:
- §4: Sharpe and stop-loss;
- §6: the Live Leaderboard table and "Fixes to the board while it runs on the legacy engine";
- §7: budget and keys;
- §9: records to correct.

Five statements in the spec were found wrong while this plan was mapped. They are corrected in the spec's **Errata** block, added in the same change as this plan.

## Global Constraints

Every task in every per-PR file inherits these.

**Scope and behaviour**
- **Legacy engine only.** No fee, slippage, fill-rule or prompt change (spec §6: "None of these is a fee or slippage change, so none falls under the 10-08 'no retrofit' rule"). No phase edits `SAFE_TRADING_PROMPT`.
- **Naming.** The board is "**Live Leaderboard**", subtitled "**Simulated fills · forward-only · no real money**", and months stay labelled "**Season 0 · preview**" (spec §6).
- **Sharpe.** It is daily, computed once on the server and never recomputed in the browser (P19/D20). It is shown from **5 trading days**, **always with its 95% interval** (spec §4).
- **Spend.** A **$50/month hard stop enforced in code**, `LEADERBOARD_MONTHLY_BUDGET_USD` (spec §7):
  - the default is `50`, and the value is read on every call, never at import;
  - `0` blocks every billable live deploy;
  - blank uses the default silently;
  - a junk, NaN, infinite, negative or >10000 value prints one `WARNING` and uses the default;
  - it never raises.

  `tests/conftest.py` strips the variable.
- **Keys switch only between seasons** (spec §7). No phase changes the provider order or the key.
- **The public payload gains no dollar figure and no error text.** It gains only `live_status.budget_halted` (bool) and a per-entry `last_attempt = {on, outcome, kind}`. The per-entry `est_cost_usd` and token fields were already public on main@17c552c5 and stay as they are; see Decision 5.

**Safety while working**
- **Do not mutate the seed DB.** Never run `python -c` or an ad-hoc script that imports `dashboard.backend`: it mutates the committed seed `dashboard/storage/data/backtest.db`. pytest is safe. Stage files by name, and never commit `backtest.db`.

**Testing**
- Run pytest from the repo root, as `pytest <path> -q -p no:cacheprovider`. Do not add `-x` or `--timeout` (pytest-timeout is not installed). The full backend suite takes 7–8 minutes.
- Locally, `test_report_pdf.py` fails without `reportlab`, so judge the full suite by file: any failure outside that file is a regression. Node-gated frontend tests skip silently without `node`, so check that it is on `PATH` first.
- After every JS edit, run `node --check <file>`. No browser runs in the planning environment, so frontend behaviour is tested with the repo's `node -e` helpers. The 375px phone-width check is a human step.

**Cache-busters**
- Bump `js/live-leaderboard.js?v=` and `js/leaderboard.js?v=` in `app.html` by one from main's value at execution time.
- Update the exact pin in `tests/test_frontend_fast_boot.py` (≈:200) in the same commit.
- Re-check main's value right before merging (the Open-PR task of P3, P5 and P6 does this).
- Never touch `styles.css` or `app.js` (exact `?v=159` pins in 3–4 test files). Reuse existing CSS classes.

**Merging and records**
- **Merging to `main` auto-deploys prod.** Do not merge on a weekday between 22:15 and ~02:30 UTC. The deploy's restart kills the nightly live refresh thread mid-run, and that night's session is then billed again on the next night. The cron says 22:30 UTC, but GitHub starts scheduled runs late: the runs for 2026-10-23 and 2026-10-26 to 10-28 started between 00:50 and 01:52 UTC the next morning, hence the window's end.
- **The owner approves every outward-facing action:** filing issues, landing and RST copy (P7 PR B), and the post-merge billable check for Haiku (P4).
- A merged PR is write-only. Follow-ups go in issues, not comments.

## Phases (one PR each, in merge order)

| # | Plan file | Branch | PR title | Must land by |
|---|---|---|---|---|
| P1 | `2026-10-14-live-board-fixes-p1-early-close.md` | `fix/live-early-close` | `fix(live): NYSE early closes on the live board` | merged **and deployed** by **2026-11-20** (first early close 2026-11-27, then 2026-12-24) |
| P2 | `2026-10-14-live-board-fixes-p2-memory-robustness.md` | `fix/live-memory-robustness` | `fix(live): roster error isolation and nightly trade memory` | — |
| P3 | `2026-10-14-live-board-fixes-p3-attempt-ledger.md` | `feat/live-attempt-ledger` | `feat(live): monthly budget hard stop and attempt ledger` | before **2026-11-01** (the newcomers' first nights are the likeliest overspend) |
| P4 | `2026-10-14-live-board-fixes-p4-roster.md` | `feat/live-roster-qwen-haiku` | `feat(live): drop Nemotron, add Qwen and Haiku` | by **2026-10-31**, or bump both `live_from` values by hand |
| P5 | `2026-10-14-live-board-fixes-p5-daily-sharpe.md` | `feat/board-daily-sharpe` | `feat(leaderboard): daily Sharpe with its 95% interval` | — (needs P1; landing copy approved first) |
| P6 | `2026-10-14-live-board-fixes-p6-copy.md` | `feat/live-board-copy` | `feat(live): Live Leaderboard copy and catching-up badge` | the same week as P5 |
| P7 A | `2026-10-14-live-board-fixes-p7-records.md` | `docs/live-board-records` | `docs: correct the live board records` | after P1–P6 |
| P7 B | same file | `docs/live-board-public-copy` | `feat(landing): Live Leaderboard copy and Sharpe interval` (draft) | after P6, with owner-approved copy, the same week as P5 |

**Order rules**
- **Sequence:** P1 → P2 → P3 → P4, all before 2026-10-31 if possible, then P5 → P6 → P7 A, then P7 B.
- **P4 does not wait for P1–P3.** It has no code dependency on them, so it must not slip past 2026-10-31 if they do.
- **P3 does not wait for P2.** Its halt clause works with or without P2, so it can go ahead if P2 slips.
- **Phases are not safe to run in parallel.** P1, P2, P3, P5 and P6 all append to the end of `test_live_leaderboard.py` or `test_frontend_live_leaderboard.py`. P3, P5 and P6 bump the same cache-busters, and identical bumps from two branches merge with no conflict. A second executor takes a contiguous range (for example P1+P2, or P4 alone), never phases interleaved with the owner's.
- **P5 to P7 B window.** From P5's deploy until P7 PR B merges, the landing page shows the new daily Sharpe under its old "annualized from hourly results" tooltip. The owner therefore approves P7 Task 7.3 Step 0's copy before P5 merges, and P6 and P7 B follow within the same week.

## Decisions

These are the plan author's decisions from the 2026-10-14 mapping. Each per-PR file cites them as "Decision N in the index plan". The owner reviews them with this plan; items needing an explicit yes are collected under **Owner decisions** below.

1. **Early closes (P1).** NYSE early closes come from a rule:
   - the day after Thanksgiving;
   - Dec 24 when it falls Mon–Thu;
   - Jul 3 when Jul 4 falls Tue–Fri;
   - never on a holiday or weekend.

   Tests pin the rule to the 2025–2028 published lists. The per-day close is threaded through:
   - the **open-stamped** branch of `sessions.is_in_session` only. This is global, so My Agents and protocol runs change too, on about 3 days a year;
   - `live.py`'s axis, settle time, `live_clock`, node key and reindex.

   The close-stamped path stays untouched; D19 belongs to the engine rewrite. The conformance suite must show no XPASS.

   *As shipped (#638, 2026-10-15):* review widened this. `sessions.day_session_windows` ends an early close at 13:00 for aggregation, the open-stamped filter and `is_session_close` alike, so most of D19 shipped with P1 and conformance T05 now fails as D2/D16 instead of D19/D8. Only the close-stamped `time_in_session` still admits stamps to 16:00; it reaches close-stamped frames only (`vnpy_simulation`, test doubles). `REGULAR_CLOSE_TIME` was dropped as unused.

   *Deviation, recorded in P1:* the close **times** (`EARLY_CLOSE_TIME`, `REGULAR_CLOSE_TIME`, `session_close_time`) live in `sessions.py`, and only the **dates** live in the calendar. The reason is that `test_no_module_restates_the_session_bounds` forbids `time(13, 0)` / `time(16, 0)` anywhere else.

   *Amended 2026-10-15 after a prior-art scan (P1's last section).* Three changes:
   - The pins now cover NYSE's releases for 2022–2028 and exchange_calendars XNYS for 2004–2035. That library is used once, offline, and is not a dependency.
   - Dated unscheduled closures are added (2025-01-09 and five earlier dates).
   - The Yahoo index baselines get their early-close level restamped. Without it they would have ended 2026-11-27 at 12:00.
2. **Cadence (P1).** No code change. A test pins it at 6 decisions per full session and 3 on an early close.
3. **Trade memory (P2).** The new read-only `PortfolioManager.restored_trades` carries the 24h tail of trades, anchored on the newest trade and capped at 200, through `snapshot_state`. Restored trades never enter `self.trades`. The reason: `_record_progress` folds `manager.trades` into the nightly trade stats, so they would be counted twice. A `trade_memory` marker in the stored checkpoint records the session the memory started from, because the month P2 merges in mixes nights with and without memory.
4. **Roster-loop robustness (P2).**
   - `refresh_live_leaderboard` catches `Exception` per entry (not `BaseException`), names the exception type in its `ERROR` line, keeps going, and still prunes and saves.
   - Characterization tests pin what a failed night stores and where it resumes.
   - A prefix that H6 refused is never checkpointed.
5. **Attempt ledger (P3).** `domain/leaderboard/live_budget.py` keeps one `agent_runs` row per (month, entry):
   - session `leaderboard-live-spend`, mode `leaderboard_spend`;
   - cumulative cost and tokens across **all** attempts;
   - `metadata.last_attempt = {on, outcome, kind}`.

   The month-to-date spend per entry is `max(ledger, latest checkpoint)`, so the first night after rollout seeds itself. Enforcement:
   - a pre-check before any bar fetch or model build;
   - an in-run stop at session close;
   - a crash is still priced;
   - the stop prints `ERROR: live.budget_exhausted …` once.

   *Deviation, recorded in P3:* the in-run stop runs **after** the checkpoint decision, so the session just paid for is stored. That can overshoot the cap by about one session's cost per entry per month. Per-entry `est_cost_usd` was already public on main and stays; this plan adds no new dollar figure.
6. **Roster (P4).**
   - Nemotron leaves `LIVE_MODEL_IDS`. CommonStack serves no NVIDIA model, so its id cannot be "fixed" there. Its `leaderboard.json` entry is untouched.
   - Haiku is sent on the Live board as `anthropic/claude-haiku-4-5` at temperature 0.
   - Qwen and Haiku join on `live_from: 2026-11-01`.
   - The order becomes (deepseek_v4_pro, qwen3_7_plus, claude_haiku_4_5).

   *Deviation, recorded in P4:* Haiku's Live-only values sit under a `live_overrides` block rather than in the shared keys. Changing the shared keys would make `_entry_config_drift` demote Haiku's three Competition repeat runs.
7. **Sharpe (P5).** `metrics.daily_sharpe_interval` uses:
   - the last finite point per US/Eastern day, with E₀ = initial equity;
   - ddof 1, √252, cash rate 0 stated on screen;
   - an iid standard error and a 95% interval;
   - statuses `ok | insufficient_days | no_variation | pending`;
   - `None`, never 0 or NaN, when undefined.

   It is anchored on conformance M03 (rf 0 → 2.611715, CI [-11.396490, 16.619919]). Both boards publish `sharpe_ratio`, `sharpe_ci_low`, `sharpe_ci_high`, `sharpe_days` and `sharpe_status`, plus a board-level basis. Competition reads the scaled hourly curve and drops `run.get("sharpe_ratio") or 0`. `calculate_sharpe` and `calc_metrics` are untouched.
8. **Copy and labels (P6).**
   - The board is renamed "Live Leaderboard", with the static subtitle and a static "Nightly after the close" Updated stat.
   - "Season 0 · preview" comes from one owner pair, `LIVE_BOARD_TITLE` / `LIVE_PHASE_LABEL` in `service.py`.
   - The Competition board gets the legacy label.
   - Stop-loss and take-profit are disclaimed as **not executed** on both boards, with tests pinning that the legacy path ignores those fields; there is no prompt change.
   - For #586: a catching-up badge and its own sort tier, and the `models_cached` banner fix ("Closes #586").
9. **Records (P7).**
   - PR A: repo `CLAUDE.md` live-board passages, plus "superseded" banners on dated live-board specs and plans.
   - PR B: the landing (`Race.tsx` rename, the BoardPreview Sharpe tooltip and interval, rebuild) and the RST renames. PR B is owner-coordinated: execute it only after the owner approves the copy.

## Review Focus

These are the five inputs the spec implies, no task originally exercised, and that would most likely bite a person using the board. Each now has its test in the owning task.

1. **An early close inside the Sharpe window.** 2026-11-27 closes at 13:00. The daily Sharpe must count it as one session, with its 13:00 close as that day's return, never as a partial day or a missing one. Pinned by P5 Task 5.2, `test_live_sharpe_counts_an_early_close_once` (`test_live_leaderboard.py`). It reads the November 2026 board through Phase 1's early-close axis, gets 20 returns, and checks the hand-computed SR of 2.846050.
2. **A catching-up row under a column sort.** A catching-up row is shown but not ranked, and its figures come from an earlier session than its neighbours'. Sorting by Sharpe or any other column must not slot it among the current rows in either direction: current rows come first, then catching-up rows, then pending ones. Pinned by P6 Task 6.5, `test_catching_up_rows_sort_after_current_rows_and_before_pending_ones` (`test_frontend_live_leaderboard.py`). In that test, DeepSeek is catching up with a Sharpe that would otherwise sort first descending.
3. **A night that fails on a timeout or a dropped connection.** The attempt ledger classifies the failure by exception type, never by message text, and checks Timeout before Connect, so a provider or pool timeout reads "timed out", not "an internal error". Pinned by P3 Task 3.1, `test_an_attempt_is_classified_without_its_message`, with 14 cases including `anthropic.APITimeoutError`/`APIConnectionError`, `httpx.ReadTimeout`/`ConnectError` and `psycopg_pool.PoolTimeout`.
4. **A halt when the cap is 0, or after a mid-month raise.** The banner promises no resume date: "Model runs are paused: the monthly model budget is used up. Baselines keep updating." A cap of 0 halts the next month too, and a mid-month raise resumes the next night. Pinned by:
   - P3 Task 3.5: `test_a_budget_halt_shows_a_banner_line_on_a_healthy_board` and `test_the_budget_line_joins_an_existing_banner_message`, which both read that exact line;
   - P3 Tasks 3.1 and 3.3: `test_zero_is_a_cap_that_blocks_everything_not_off` and `test_a_zero_budget_blocks_before_any_bar_fetch_or_model_build`.
5. **Trade memory that starts mid-month.** P2 changes the prompt's inputs partway through a month, so the snapshot records the bar the memory started from, and a restore keeps that stamp. A night that ran without memory therefore stays visible. Pinned by P2 Task 2.3:
   - `test_snapshot_marks_the_memory_from_the_managers_first_bar` (`tests/backtesting/test_trade_memory.py`);
   - `test_the_checkpoint_row_records_where_the_memory_started` (`test_live_trade_memory.py`).

## Out of scope (and why)

- **Public per-trade lists** (spec §6 Format, "public trades") **and the reasoning/trade-log view** belong to their own plan. Nothing on the live path records a decision tape or reasoning trace yet; P7 PR A writes that fact into `CLAUDE.md` and pins it with a test.
- **The methodology page, and the subtitle link to it** (spec §4) belong to their own plan. The subtitle stays plain text here.
- **Nemotron's Season 1 source.** This is an owner decision (below).
- **#587** (a replay rewrites the month-to-date curve at every checkpoint) is perf only. P4's `live_from` makes the newcomers start on the 1st, so no replay happens.
- **The engine rewrite and everything it owns:** the close-stamped `time_in_session` remainder of D19 (#638 shipped the rest), stop-loss/take-profit execution, fees and costs, Season 1.
- **User-facing docs outside P7 PR B:** see the follow-up list in the PR that adds this plan.
- **A night lost to a total market-data failure reads "an internal error"** on the board. It reaches the attempt classifier as `RuntimeError('No market data returned…')`, and naming it properly needs a typed error raised in `live.py`. That is a follow-up issue, not part of P3.

## Owner decisions (needed before the phase that consumes them)

1. **Nemotron for Season 1** (spec §6 roster): fund OpenRouter for it, or replace it with another model. P4 only removes it from the live roster.
2. **`LEADERBOARD_MONTHLY_BUDGET_USD=0` means "block every billable live deploy"**, not "no cap" (P3). The reason: a typo that silently means "unlimited" fails in the expensive direction.
3. **The session filter is global** (P1). My Agents and protocol runs stop using post-13:00 bars on early-close days (about 3 days a year). Stored runs are not recomputed.
4. **The Sharpe rule also applies to the Competition board** (P5). Its published Sharpe values change from hourly-annualized to daily with an interval.
5. **New recurring cost: ≈ $2.1/month from November** (Qwen ≈ $0.63 + Haiku ≈ $1.45, on `COMMONSTACK_API_KEY`, outside Credits), for a Live board total of ≈ $2.4/month. The $50 cap bounds it.
6. **Keep the pre-existing public per-entry `est_cost_usd`** (P3). The alternative is to drop it from the Live payload; no JS reads it.
7. **Public copy for a timed-out night reads "a request timed out"** (P3 Task 3.5), not "the model provider timed out". On the live path a provider timeout is absorbed into a rule-based step and never reaches the classifier. What does reach it is a store or network timeout, so the old copy would have been false.
8. **`GET /runs?mode=leaderboard_spend` lists the ledger rows publicly** (P3). That includes each row's cumulative `llm_calls`, but no dollars or tokens. The plan leaves the route as it is. The alternative is to exclude that mode from `/runs`.
9. **Trade memory records `trade_memory_since`, not only a flag** (P2 Task 2.3). Superseded checkpoints are deleted, so a bare `trade_memory: 1` could not show which session the memory started from. If a bare flag is preferred, P2 lists what to remove.
