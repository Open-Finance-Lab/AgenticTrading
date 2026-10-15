# ATL direction: trustworthy backtests, a credible simulator, a model-vs-model Live Leaderboard

**Status:** owner-confirmed 2026-10-14, after a three-round alignment session. It supersedes earlier records wherever they disagree:
- the 08-15 two-week seasons;
- the 07-19 participatory scope, which is deferred rather than retired.

The 10-07 engine mandate is narrowed to the decisions below.

**Appendices** (in `2026-10-14-atl-direction/`):
- `context-brief.md`: an aggregation of every source on record. Read it with `context-brief-critique.md`, whose corrections take precedence.
- `engine-fix-cost-probe.md`: patch-in-place vs own core, 2026-10-13.
- `engine-live-board-reprice.md`: the same comparison re-priced with the live board in scope, 2026-10-14.

All three reports are estimates (±50%). They are not commitments.

**Errata (2026-10-14).** These were found while planning the live-board fixes (`docs/superpowers/plans/2026-10-14-live-board-fixes.md`) and are corrected in place below:
- §4, Sharpe interval: with 5 daily returns the 95% interval is about **±14** annualised. The ±7 written here was one standard error.
- §6, Cadence: 6 decisions per **full** session. A 13:00 early close has 3, on the bars stamped 10:00–12:00. The early closes on the calendar are 2026-11-27 and 2026-12-24.
- §6, Until Season 1: no model id can fix Nemotron. CommonStack serves no Nemotron or NVIDIA model (`GET /v1/models`, 2026-10-14).
- §6, Reasoning and trade-log view: nothing on the live path records a decision tape or reasoning trace yet. #628 was an admin icon fix.
- §6, Fixes: since #582 a model checkpoints each session that clears H6. Only a model with no checkpoint this month replays from the 1st.

## 1. Goals

1. **Backtests are trustworthy** (§3).
2. **A quant-literate reader does not treat ATL as a joke.** The goal is documented, checkable assumptions. Feature parity with institutional backtest engines is explicitly not the goal.
3. **A model-vs-model Live Leaderboard** in nof1's format. nof1 is never named in anything outward-facing.

**Audience, in priority order:**
1. the academic and lab network, who judge validity;
2. agent builders and My Agents users;
3. spectators.

## 2. Position and boundaries

- **Models compete on one shared prompt that ATL writes.** That makes the measured variable the model, as in nof1, which writes the prompt itself and has users submit nothing.
- **User participation stays the long-term direction** (07-19 / 08-08: one pinned model, many user instructions). It reopens after Season 1, and only once a test shows that comparing instructions is measurable at all. The motivation:
  - The noise between runs of a single model is large. Three pinned DeepSeek runs on identical inputs spread from +0.35% to +12.97%.
  - The Phase 0 floor of 0.20pp came from one pair of runs.
- **Out of scope, with no forward design:**
  - real capital;
  - broker paper trading;
  - shorts and leverage;
  - non-US markets.

## 3. Trustworthy means six things

1. **Engine correctness:** US costs, a volume cap and the market calendar, verified by the hand-computed conformance suite (`2026-10-07-us-conformance-suite-design.md`).
2. **Replay:** a run's recorded decisions (#627), re-run through a stamped engine version, reproduce the equity curve exactly. Re-running the model itself is not reproducible, because pinned sampling is not deterministic.
3. **No single-run headlines:** results come from repeat runs with a band, or carry an interval.
4. **Knowledge look-ahead:**
   - A per-model table of training cutoffs. When a provider publishes none, the model's public release date is used as its cutoff.
   - A backtest whose window falls before the cutoff is labelled.
   - The Live Leaderboard ranks on forward data only.
5. **No silent failures.** Every hold, fallback and timeout is recorded and visible (#573, #375, #600, #524, #601).
6. **Provenance:**
   - the data feed (done);
   - the news source (#123);
   - **point-in-time DJIA membership**: a backtest over an old window uses the index members as of each date, not today's.

## 4. Credibility bar

- **Methodology page.** It lives on Read the Docs and is linked from every backtest results panel and from the Live Leaderboard subtitle. It documents:
  - the fill rule;
  - costs;
  - the volume cap;
  - the calendar;
  - a "what we don't model" list.
- **What makes ATL distinctive:** it publishes the conformance suite and supports replay from recorded decisions.
- **User-overridable assumptions (My Agents).** An **Advanced** section collapsed at the bottom of the existing agent backtest settings page exposes four settings. Every run records the settings it used, and leaderboards always use the defaults.

  | Setting | Default |
  |---|---|
  | Fill timing | Next bar's open; same bar's close is the alternative |
  | Slippage | 5 bps per side |
  | Participation cap | 10% |
  | Costs | On |

- **Cost defaults**, per owner decision Q3 of 10-07:
  - commission 0;
  - dated SEC/TAF fees;
  - results published net, with gross alongside;
  - the CAT fee rate still needs to be verified.
- **Sharpe:** daily, computed once by the engine (P19), and never recomputed in the frontend (D20). It is shown from **5 trading days**, **always with its 95% interval**. With 5 daily returns the 95% interval is about ±14 annualised (±7 is one standard error; corrected 2026-10-14), and showing it is the point.
- **Stop-loss / take-profit:** the new core executes them for real. This item is not cuttable.

## 5. Engine

- **Ratified:** build ATL's own core, step-first:
  - The core API is `advance_to(ts)`, `submit` and `view`. A batch backtest is a loop over them.
  - It sits behind `SimulatorPort`.
  - No open-source engine is adopted, because adoption never pays back on this scope. This closes engine phase 1.
- **Order of work:**
  1. Option-neutral work first:
     - the calendar (D19);
     - fee tables;
     - P19 Sharpe and D20;
     - corporate actions;
     - the engine stamp;
     - point-in-time DJIA.
  2. **My Agents switches over** (`POST /backtest/run` → `HourlyBacktester`). It goes first because it is every new user's opening action, and because the engine should be proven where mistakes can be re-run.
  3. **The Live Leaderboard switches over**, about 25 more engineer-days. Its forward record is permanent.
- **Old My Agents runs are hidden after the switchover** rather than relabelled.
- **Time box:** 8 weeks fixed, with scope cut to fit. The estimate to the first switchover is about 37 engineer-days (±50%).

## 6. Live Leaderboard

| Item | Decision |
|---|---|
| Name | **Live Leaderboard**, subtitled "Simulated fills · forward-only · no real money". "Trading" is dropped because it reads as real orders and collides with the Robinhood live-trading docs. |
| Format | One shared prompt, identical data and decision times, public trades. |
| Instruments | Long-only DJIA-30 cash equities. |
| Seasons | Calendar month, with a hard stop and a capital reset. This formally replaces the 08-15 two-week decision. |
| Cadence | 6 decisions per full session, on bars stamped 10:00–15:00; 3 on a 13:00 early close (bars 10:00–12:00; 2026-11-27 and 2026-12-24), corrected 2026-10-14. Computed nightly on closed bars. Shown as "Updated nightly after the close". This is equivalent to real time because the model has no tools and the prompt carries nothing newer than the decision time; it was checked and there is no news in the prompt. |
| Season 1 gate | The board runs on the new engine **and** the methodology page is live. |
| Season 1 timing | One full Season 0 month on the new engine first, then Season 1 the month after. That is roughly Mar–Apr 2027. |
| Season 1 roster | 6 models without GPT-5.5 (Haiku 4.5, Sonnet 4.6, Gemini 3.1 Pro, Qwen3.7 Plus, DeepSeek V4 Pro, Nemotron 3 Nano), × 3 repeat runs each, ≈ $46/month. |
| Until Season 1 | Nemotron returns a CommonStack 404 every night, and no model id can fix that: CommonStack serves no Nemotron or NVIDIA model (corrected 2026-10-14). The live-board fixes take it off the live roster; its Season 1 source is an owner decision. Add Qwen and Haiku at 1 run each, ≈ $3/month. Months stay labelled "Season 0 · preview". |
| Reasoning and trade-log view | Its own plan and PR, inside the 8 weeks after the engine work. Nothing on the live path records a decision tape or reasoning trace yet (#628 was an admin icon fix; corrected 2026-10-14), so that plan adds the recording first. |
| Competition board | Labelled "legacy engine · historical window · may overlap model training data" now. Retired once Season 1 has data, not re-run. |

**Fixes to the board while it runs on the legacy engine.** None of these is a fee or slippage change, so none falls under the 10-08 "no retrofit" rule.
- The **2026-11-27** half-day session. This is a hard deadline: `live.py:56` assumes 7 fixed nodes.
- Trade memory across nights. `snapshot_state` omits `self.trades`.
- The stop-loss/take-profit disclaimer (D12 on the legacy engine).
- The Sharpe rule from §4.
- **A failed model replays from the 1st every night.** Since #582 a model checkpoints each session that clears H6, so only a model with no checkpoint this month replays the whole month from the 1st each night (corrected 2026-10-14; this said it "never checkpoints"). That replay is free only while the failure happens before anything is generated.

## 7. Budget and keys

- **Funding:** the owner pays, with a **$50/month hard stop enforced in code** before Season 1. `LEADERBOARD_MONTHLY_BUDGET_USD` was decided on 08-15 and never implemented.
- **Measured on 2026-10-14:**
  - The live board costs about $0.35/month (DeepSeek only).
  - The total since 09-29 is $0.51, with an upper bound of about $0.9.
  - The prices come from the `infrastructure/llm/pricing.py` snapshot, because CommonStack has no usage endpoint.
  - The calls go through `COMMONSTACK_API_KEY` and bypass the Credits layer.
- **Key order:**
  1. the owner's CommonStack key;
  2. a teammate's CommonStack key;
  3. OpenRouter, the long-term home.

  Keys switch **only between seasons**. A switch changes how "thinking off" is sent, and the OpenRouter version of that setting has never been tested. Before OpenRouter takes over, its key needs funding and that setting needs testing.

| Model | 1 run / month | 3 runs / month |
|---|---|---|
| GPT-5.5 | $10.87 | $32.61 |
| Gemini 3.1 Pro | $8.81 | $26.44 |
| Sonnet 4.6 | $3.87 | $11.60 |
| Haiku 4.5 | $1.45 | $4.35 |
| Qwen3.7 Plus | $0.63 | $1.88 |
| DeepSeek V4 Pro | $0.35 | $1.04 |
| Nemotron 3 Nano | $0.06 | $0.18 |

These assume 126 calls/month. Token counts for Sonnet, GPT-5.5, Gemini and Nemotron are from July runs (±25%).

## 8. Housekeeping done on 2026-10-14

- **#626** (browser Sharpe recompute): `blocked` label and a comment. It conflicts with P19/D20.
- **#395** (14 external baselines): `blocked` label and a comment. It would price baselines on the legacy engine, and it also bundles live-broker code.
- **#461** (US costs): answered by pointing to conformance group F. It stays open until the cost PR closes it.

## 9. Follow-ups

**Owner:**
- Ask Jinbo whether they can take the live-board fixes.

**Plans to write:**
- the engine stage 0 → My Agents switchover;
- the live-board fixes;
- the methodology page;
- the reasoning and trade-log view.

**Records to correct:**
- Repo `CLAUDE.md`: the "Season 0 preview / no advance engine" passages under `LEADERBOARD_DAILY_*` and "Two leaderboards" are stale, because `live.py` (#534) has advanced a calendar-month board nightly since 09-26. The central DB is stale on the same point and on `LIVE_LEADERBOARD_DEPLOY_MODELS`, which has been `true` since 2026-09-29.

**User-facing docs to update** (the owner coordinates these; nothing has been edited):
- `dashboard/frontend/app.html:1579, 1588, 1592-1595, 1605, 1798`;
- `docs/source/lab/operating_modes.rst:14`;
- `dashboard/landing/src/components/home/Race.tsx:27, 151, 159`;
- `docs/source/lab/live_trading.rst:11`.
