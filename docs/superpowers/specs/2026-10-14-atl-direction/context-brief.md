# ATL goal-alignment brief (2026-10-14)

Tags: [source-verified] = re-read at `engine-probe` HEAD `2dc8203a` this run. [gatherer-reported] = from a gatherer or the main session's established state, not re-checked. [web, conf=…] = web research. Paths are repo-relative.

## A. Situation

ATL runs three surfaces on one legacy engine. My Agents backtests fail ATL's own US conformance suite: 1 PASS / 20 FAIL / 7 not expressible, 25 strict xfails [gatherer-reported]. US profiles carry no cost model; `transaction_cost_profile` defaults to `None` (`infrastructure/market_data/profiles.py:144`) and is set only on the two A-share profiles [source-verified]. The Contest board publishes a median-of-runs band (#604), and no LLM median beats the passive baselines [gatherer-reported]. The calendar-month Live board advances nightly on the legacy PortfolioManager with zero costs and same-close fills, has two LLMs on the roster, and publishes Sharpe 5.8–11.5 off 9 days [gatherer-reported]. Analysis recommends an ATL-owned step-first core (option B). That is an analysis verdict; no owner decision ratifies it [gatherer-reported]. nof1's Alpha Arena has had no public run since Dec 2025, but it raised $15M in May 2026 for Season 2 [web, conf=med]. The owner carries almost all of the trust and engine work alone [gatherer-reported]. Several recurring costs have never been measured (§H).

## B. Per goal

### B1. Trustworthy backtest

**Where ATL is now**
- The conformance suite (#623) has 28 hand-derived cases, strict-pinned [gatherer-reported].
- The decision tape (#627) is the replay unit. It tags fallback bars as `llm_fallback` (`decision_tape.py:48`) [source-verified]. That partly closes the internal fix report's "fallback is invisible" gap for backtests. The leaderboard does not write the tape yet; Stage 0 prices that at 2.5 days [gatherer-reported].
- Look-ahead fixes are merged: #486 (indicator fallback leak), #563 (inclusive end date), #595 (warm-up pad) [gatherer-reported].
- The band contract is live (#603, #604, #613) [gatherer-reported].

**Gaps**
- D1–D20 are all open. Those that most affect the published number:
  - D2: same-close fill.
  - D10: zero costs.
  - D15: Sharpe annualised per 5 minutes.
  - D20: the browser recomputes Sharpe at `√(252×6.5)` (`backtest-comparison.js:65`) [source-verified].
- Owner decision Q5 (P19 daily Sharpe, computed by the engine) is not implemented. Open PR #626 is MERGEABLE and keeps the browser recompute [gatherer-reported].
- Silent non-trades [gatherer-reported]:
  - #573: outage holds are not persisted.
  - #375: deadline auto-holds are silent.
  - #600: the retry path is unreachable.
  - #524: a timed-out run writes no record.
  - #601: the reclaimer has never run in prod.
- LLM knowledge look-ahead has no policy, no test and no model-cutoff table [gatherer-reported]. The contest window is 2026-04-15 → 05-15 (`leaderboard.json`) [source-verified].
- Pinned sampling is not deterministic. Cross-process engine determinism is unproven [gatherer-reported].

**Risks**
- Fixing #461 (costs) or #346 (corporate actions) in the current engine duplicates the rewrite [gatherer-reported].
- #573, #600, #594 and #596 all change `portfolio_manager.py`, which the rewrite replaces [gatherer-reported].

### B2. Recognised as a valid simulator

**Where ATL is now**
- Internal specs exist: the conformance design records owner decisions Q1–Q5 [gatherer-reported].
- There is no public methodology page and no "what we don't model" list [gatherer-reported].

**Gaps** (peer bar in §C)
- Same-close fills, no costs, no volume cap, no US corporate actions.
- No early-close calendar: `_RTH_HOURS` at `live.py:56` is seven fixed nodes [source-verified].
- No tester has commented on fills, costs, timing or data. Feedback asks only for onboarding, speed and chart labels [gatherer-reported].
- Nobody external has been asked whether ATL is credible as a simulator.
- A Credits HTTP 429 (Cloudflare challenge page) blocked paid testers. The tracker is from 2026-09-10 and unverified since [gatherer-reported].

**Risks**
- "Like many other platforms" has no named comparator. Engine peers (LEAN, Backtrader) and arena peers (TradeRank, LiveTradeBench) set different bars and call for different work.

### B3. nof1-style live board

**Where ATL is now**
- Cron `30 22 * * 1-5` [source-verified]. Roster is `deepseek_v4_pro` and `nemotron_3_nano_30b` (`live.py:74-77`) [source-verified]. Nemotron is `pending` on prod with 0 points [gatherer-reported].
- Copy says "simulated trading on real market data", $100,000 per entry, resets on the 1st, and "Season 0 … preview" (`app.html:1592-1595`) [source-verified]. `phase_label: "Season 0"` is hard-coded at `live.py:283` and `:1517` [source-verified].
- Reasoning is stored (#628) but not shown. The board shows Sharpe, drawdown and trade count only. There is no per-trade log [gatherer-reported].

**Gaps**
- `snapshot_state` (`portfolio_manager.py:194`) saves cash, positions, entry prices, available positions and frozen lots, but not `self.trades`. `recent_trades` (`:445`) is rebuilt from `self.trades`, so it is empty every night (#585) [source-verified].
- The 2026-11-27 half day breaks `_RTH_HOURS` [source-verified constant; gatherer-reported impact].
- `stop_loss_price` and `take_profit_price` are accepted (`validator.py:86-87`) [source-verified]. They are then ignored while the prompt says they are enforced (D12) [gatherer-reported].
- The season engine (#354) is unbuilt. The live payload has no `season` block, and `build_season_payload` is dead code [gatherer-reported].
- Each model runs once per month. The Contest band contract does not exist here.
- Sharpe off 9 days is the short-sample noise the literature warns about [gatherer-reported; web, conf=high].
- The cron fires 2.5–4.5 h late [gatherer-reported].

**Risks**
- **Unrecorded recurring spend.** Prod shows a 63-point DeepSeek curve, but the central DB says `LIVE_LEADERBOARD_DEPLOY_MODELS` was unset on 2026-09-29 [gatherer-reported]. Either that variable is now `true` or someone dispatches deploys by hand (inference). Either way a nightly LLM cost is running that nobody has recorded.
- A funded nof1 Season 2 would reset the reference point [web, conf=med].

## C. Peer bar

### C1. Simulator features (simulator-bar gatherer)

| Feature | Peers | ATL |
|---|---|---|
| Fill no earlier than next data point, with a documented rule | Table stakes: LEAN, Backtrader, TradingView, Nautilus [web, conf=high] | Lacks. D2; T01 is the only pass [gatherer-reported] |
| Configurable commission | Table stakes [web, conf=high] | Lacks on US [source-verified]; A-share has one |
| Pluggable slippage (default is often 0) | Table stakes [web, conf=high] | Lacks (D10) [gatherer-reported] |
| Documented intrabar assumption | Common [web, conf=high] | Not published |
| Volume cap / partial fills | Common: Zipline 10% default, Backtrader filler [web, conf=high] | Lacks (D11). Owner chose a 10% cap with the remainder cancelled [gatherer-reported] |
| Corporate-action modes | Common in LEAN [web, conf=low] | Lacks on US (D13) [gatherer-reported] |
| Margin, short borrow | Common / differentiator [web, conf=med] | Long-only |
| SEC 31 / TAF fees | Default in none checked [web, conf=med] | Owner chose dated SEC/TAF on, which would exceed the bar [gatherer-reported] |
| Seeded determinism | Differentiator; Nautilus seeds the fill model only [web, conf=high] | Unproven |
| Published "what we don't model" list | LEAN, Nautilus, Backtrader and Alpaca paper all have one [web, conf=high] | None. The cheapest gap to close |
| Early-close calendar | Not retrieved for any peer | Lacks (D19). Has it since #638 (2026-10-15) |

Peers earn credibility by documenting assumptions that users can override, not by non-zero defaults [web, conf=high]. A defensible minimum for ATL is a methodology page plus fixes for D2, D10, D11 and D19 (inference).

### C2. nof1 vs ATL

| nof1 | Source | ATL |
|---|---|---|
| Real money, $10k per model, on-chain wallets | [web, conf=high] | No. The spec says "Real capital. Ever, on this surface." [gatherer-reported] |
| S1 crypto perps; S1.5 US equities in 4 modes | [web, conf=high/med] | DJIA-30 equities, long-only [gatherer-reported] |
| 6–8 models, identical prompt | [web, conf=high/med] | Contest 7 LLMs; Live 2, GPT-5.5 dropped for cost [source-verified] |
| About two-week seasons with a winner | [web, conf=high] | Calendar month, no winner, permanent "Season 0" [source-verified] |
| Model chats shown, trades verifiable | [web, conf=med] | Reasoning stored, not shown [gatherer-reported] |
| 2–3 minute cadence | [web, conf=low] | Nightly, hourly bars [source-verified] |
| One draw per season, no dispersion reported | [web, conf=med] | Contest band; Live single draw |
| Methodology unpublished | [web, conf=med] | Internal specs only |

ATL can beat nof1 on what critics attacked it for: replicas and bands, a minimum sample before Sharpe, and a published methodology [web, conf=med]. A forward-only board avoids contamination by construction, as LiveTradeBench does [web, conf=high].

## D. Owner decisions on record, and conflicts

| Date | Decision | Source |
|---|---|---|
| 07-19 | Hosting platform with a Trust/Commit boundary as the original contribution. The contest is the cold-start engine. Phase 2 needs ≥50 external submissions. | PJ:17-19 [gatherer-reported] |
| 08-08 | Copy from nof1: shared instruction, identical feed and cadence, fixed season with reset, public trades. No real capital. Do not name nof1 externally. | decisions/2026-08.md [gatherer-reported] |
| 08-15 | Two boards, two-week seasons, a failed-night policy, no broker paper trading, $50/mo `LEADERBOARD_MONTHLY_BUDGET_USD`. | decisions/2026-08.md [gatherer-reported] |
| 09-26 | Shipped a calendar-month freeze board with opt-in model deploys. | PJ:415-431 [gatherer-reported] |
| 10-03 | The result contract is a band. | memory [gatherer-reported] |
| 10-07 | Rewrite authorised: My Agents only, US only. Q1 next-open fill; Q2 cap remainder cancelled; Q3 cost defaults; Q4 Alpaca corporate actions; Q5 P19 Sharpe. | CS:914-926 [gatherer-reported] |
| 10-08 | No fees or slippage retrofitted into the current engine. PyPI out of scope. | EP §6; decisions/2026-10.md [gatherer-reported] |
| 10-14 | Live board back in eventual scope. | memory [gatherer-reported] |

**Conflicts**
1. Season length. Two weeks per the 08-15 decision and `leaderboard.json` (`length_trading_days: 10`) [source-verified]. Calendar month in the shipped code. No record shows anyone choosing the month.
2. North star. The 07-19 record makes the platform the product and the contest its cold-start engine. The Q1 recommendation makes the arena the product.
3. Sequencing. 10-07 says My Agents first; 10-14 adds the live board. Still open.
4. Retrofit. The 10-08 no-retrofit rule conflicts with the Q6 urgent fixes to legacy `live.py` and `portfolio_manager.py`.
5. Sharpe ownership. Q5 says the engine owns P19 Sharpe; open PR #626 keeps the browser recompute.
6. Budget. The 08-15 ceiling is $50/mo and was never built; the Q5 proposal is ~$100/mo.
7. Phase 0 gate. It "passes" with a 0.20pp noise floor from two runs (`probe-results/2026-08-09-instruction-sensitivity.md`) [source-verified]. Later pinned DeepSeek spreads ran +0.35% to +12.97% [gatherer-reported]. The gate's noise floor is understated by more than an order of magnitude.
8. Engine. B is recommended but not ratified, and RA still marks AKQuant "open again" [gatherer-reported].

## E. Stale or contradicted records (follow-ups)

**Repo dev docs (context lake)**
- `CLAUDE.md` [source-verified]:
  - `:217`: "two-week seasons".
  - `:227`: "Nothing advances anything".
  - `:232`: "reuses the contest session and window byte-for-byte".
  - `:243`: "only a real nightly advance can write".
  - `:259`: "Nothing advances Season 0".
- `CLAUDE.md:71` says "the season engine's nightly advance is this same call". The live advance actually calls `/live/refresh` [source-verified]. The same claim is in the `daily-leaderboard.yml` header [gatherer-reported].
- `CLAUDE.md` says "eleven of the twelve" twins are guarded. 12 `*_postgres.py` files call `init_schema_unless_worker` [source-verified].
- `CLAUDE.md`'s "All 7 LLM entries cleared H6" is true of the Contest board only [gatherer-reported].
- CS and LB cite `domain/backtesting/profiles.py`, which does not exist. The file is `infrastructure/market_data/profiles.py` [source-verified].
- ST status still says "approved" though it is largely implemented. RA line 3 says the live board is out of scope [gatherer-reported].
- Issues [gatherer-reported]:
  - #354's body is stale: it says live falls back to contest, and it is titled "season engine" though a month engine shipped.
  - #461 duplicates conformance group F.

**Central DB** [gatherer-reported]
- PJ:45 says Render FREE; `CLAUDE.md` says Standard 1c-2g since 09-11. Verify before any cost decision.
- PJ:61 says the decision tape is in progress; #627 is merged.
- PJ:437 says the deploy variable is unset, which a nightly DeepSeek curve contradicts.
- PJ:456 claims paper trading works; `PAPER_TRADING_ENABLED=False`. PJ:59's "1087 passed" is stale.
- Merges #625–#631 are unrecorded. The rewrite mandate exists only in auto-memory. The $50 ceiling is not in code.

**User-facing** (the owner coordinates these; do not edit them mid-session)
- `app.html:1579`: "Participating teams will appear…", but there is no entry path [source-verified].
- `app.html:1588`: "paper-trading competition … Preseason … registration Aug 2026" [source-verified].
- `app.html:1605` and `:1798`: "Preseason" for a window that ended 2026-05-15 [source-verified].
- `app.html:1595`: "Season 0, the shakedown season", with no season object behind it [source-verified].
- `docs/source/lab/live_trading.rst:11`: Robinhood "live trading" collides with the "Live" board's name [gatherer-reported].

## F. Round-1 questions

**Q1 North star.** Recommended: the arena is the product.
- For:
  - nof1 has had no public board since Dec 2025 [web, conf=med].
  - Forward data avoids contamination [web, conf=high].
  - Live came back into scope on 10-14.
  - A band would be a real differentiator.
- Against:
  - The 07-19 record makes a hosting platform the product, gated on ≥50 external submissions [gatherer-reported].
  - The participatory spec's axis is one model with N user instructions, a different arena from nof1's N models [gatherer-reported].
  - nof1 has $15M for Season 2 [web, conf=med].
- **Partly supported.** It holds only if the owner explicitly supersedes the 07-19 direction. Ask whether the arena is model-vs-model, instruction-vs-instruction, or both.

**Q2 Audience.** Recommended: researchers and labs primary, spectators secondary.
- For:
  - The lab's academic framing is on record [gatherer-reported].
  - The literature bar is what researchers apply [web, conf=high].
- Against:
  - "Model labs" appears in no record.
  - Recorded audiences are users and Discord, external agent builders, and the lab [gatherer-reported].
  - The only tester feedback is retail-shaped [gatherer-reported].
- **Partly supported.** The academic network is on record; model labs is a new audience.

**Q3 nof1 essentials.**
- For:
  - Real money is barred by the spec [gatherer-reported].
  - Perps raise LLM spend about 5.8× and break the nightly-close design [gatherer-reported].
  - The data for transparency already exists (#627, #628) [gatherer-reported].
  - Seasons with a reset were decided on 08-08 and 08-15.
- Against:
  - Two models, one of them pending, is not head-to-head at nof1 scale [source-verified].
  - Season length is unresolved.
  - The recommendation omits replicas. On a forward board, replicas multiply LLM cost by N.
- **Supported.** Add replicas and season length as explicit sub-questions.

**Q4 What "trustworthy" covers.**
- For:
  - Three senses have artifacts: the conformance suite, the decision tape, the band.
  - Knowledge look-ahead has nothing. Look-Ahead-Bench shows 15–22pp of in-sample alpha decay [web, conf=high, small sample].
- Caveats:
  - "Replay to identical equity" is reachable for the engine from the tape. It is not reachable for an LLM re-run [gatherer-reported].
  - Prod violates statistical honesty today (Sharpe off 9 days).
  - The four senses omit cost realism, H6 on Live, P20 fail-visible rejections, and survivorship ("DJIA" is today's 30 members) [gatherer-reported].
- **Supported**, with the definition widened.

**Q5 Appetite.**
- Time:
  - Judgment estimates for S1 are 103.5–123.5 days. Calibrated at 358 lines/day, it is about 28–44 days. Stage 0 alone is about 24 judgment days [gatherer-reported].
  - Eight weeks fits only the calibrated figure, and the calibration excludes design, review and prod verification [gatherer-reported]. Bus factor 1.
- Money:
  - The prior ceiling was $50/mo [gatherer-reported].
  - Recurring lines:
    - Render, whose tier is disputed.
    - Neon Launch.
    - Nightly LLM deploys, which are unmetered.
  - Hourly cadence needs a Render worker. Real-time SIP is $99/mo [gatherer-reported].
- **Weak.** $100 is unverified and double the prior figure. Measure the bill first.

**Q6 Live board meanwhile.**
- For:
  - Forward data only accrues while the board advances.
  - The 11-27 half day is a hard deadline.
  - The "preview" banner already exists [source-verified].
- Against:
  - The fixes edit legacy code the rewrite deletes, against the 10-08 no-retrofit rule. They cost about 3.9 days: early close 1, trade memory 1, D12 disclaimer 0.4, Sharpe 1.5 [gatherer-reported].
  - Each night spends LLM money on a curve that will be labelled legacy.
- **Supported** as the cheapest honest option. Add a minimum sample before Sharpe is shown, and confirm who pays for the nightly deploy.

**Q7 Housekeeping.**
- For: one document can retire the stale claims in §E.
- Against:
  - EP and LB live in a session scratchpad. Scratchpads are wiped within days [gatherer-reported], and HEAD `2dc8203a` is itself a "recover engine rewrite phase 1 records" commit [source-verified].
  - `main` is unprotected, so #626 needs a draft or `blocked` gate now. *(Superseded 2026-10-29: `main` is now gated by ruleset 26155796; see CLAUDE.md "Merge & branch discipline". A draft or `blocked` gate is still how to hold a PR, because one approval or an admin bypass still merges it.)*
- **Supported only if** the reports are copied into the repo this session.

## G. Decision tree for rounds 2+

| # | Decision | Prerequisites | Facts needed |
|---|---|---|---|
| 1 | North star and audience (Q1, Q2) | — | Does the 07-19 direction stand? Are user entries part of the arena? |
| 2 | Arena format: model-vs-model, instruction-vs-instruction, or both | 1 | A re-measured Phase 0 noise floor (0.20pp vs a ~12.6pp observed spread) |
| 3 | Season length (two weeks vs month), Season 1 start, winner rule | Q3 | LLM spend per entry-season |
| 4 | Replicas on Live | Q4, Q5 | Nightly cost per entry (unmeasured); memory per child process |
| 5 | Venue: S1 / S2 Alpaca paper / real money | Q3, Q5 | S2 adds ~25 days. Alpaca paper itself models no slippage and gives random partial fills [web, conf=high] |
| 6 | Instruments: long-only / shorts and leverage / perps | 5 | Shorts add 20–28 days and touch `validator.py`, a security boundary. Perps cost ~5.8× in LLM spend and need 24/7 cadence |
| 7 | Cadence: nightly vs hourly | 5, 6 | Render worker cost; Basic SIP has a 15-minute delay |
| 8 | Roster | 3, 4 | Per-model cost; why Nemotron is still pending after #617; model cutoffs |
| 9 | Ratify the engine (B / A / Nautilus / AKQuant) | 6 | Perps is the only lever toward Nautilus |
| 10 | Sequencing: leaderboard-first vs My-Agents-first | 1, 9 | About 37 days to first cutover either way |
| 11 | Comparability: 21 billable re-runs vs labelling rows legacy | 9, Q5 | Engine stamp in `_find_cached_run` / `_vetted_pool` |
| 12 | Look-ahead policy | Q4 | A per-model cutoff table |
| 13 | Transparency UI: reasoning, trade log | Q3 | Trace-store growth on Neon |
| 14 | Minimum N before Sharpe on Live | Q4 | — |
| 15 | Small forks: D16 (3.2 days), D12 enforce vs disclaim (3.5 vs 0.4 days), split-adjusted baselines, CAT rate | 9 | — |
| 16 | PR dispositions: #626, #395, and closing #461/#346 against the conformance suite | 9 | — |
| 17 | Naming: "Live" vs Robinhood live trading | 1 | — |

## H. Unknowns

- **Costs:**
  - The actual Render plan and bill.
  - The Neon bill.
  - Nightly LLM spend on Live.
  - The current `LIVE_LEADERBOARD_DEPLOY_MODELS` value.
- Each roster model's training cutoff against the contest window.
- Whether nof1 Season 2 has started, and its prompt, cadence and scoring [web gap].
- True throughput: 81 vs 358 lines/day, and whether calibration holds for live-clock and broker work.
- Cross-process engine determinism.
- The cause of the 2.5–4.5 h cron delay, and whether it threatens the freeze margin.
- Whether the Credits 429 and the escalate-on-retry fix (#448) are resolved.
- The CAT fee rate. The 2026 FINRA TAF figure is secondary-sourced only.
- Whether the Alpaca corporate-actions API covers the needed history on the free tier.
- Which platforms the owner means by "many other platforms".
- Whether any external user has been asked about realism.
