# Does powering the live leaderboard change the engine decision?

## 1. Direct answer

- **The recommendation stands: build B and do not adopt an engine. What changes is that it now holds without conditions.** The 2026-10-13 probe already said "If we build: use B's design" (report.md:10, :313). Its one exception was that A wins "only if the leaderboard and step sessions never move" (:328). The leaderboard now has to move, so that exception is gone. B's design also changes: it becomes step-first, the C6 snapshot comes into scope, and the leaderboard driver becomes a planned phase.
- **On cost, B does not win the long-only scenarios.**
  - S1 is a tie: A ≈102–103.5 likely days, B 104.5.
  - In S2, A is cheaper by 5–6 days on both pricers' figures. There the live board's state lives at the broker, so the probe's exception partly holds again. B is chosen there only on the probe's qualitative grounds: the loop surgery and the tape spec.
  - B wins on risk: moving the leaderboard has a high of 10.5 days under B against 26 under A. It also wins on cost once shorting or leverage is allowed: by about 5–6 days on equities and about 8 on perps.
- **Adopting an engine still does not pay back.** At least 82% of the leaderboard work is on ATL's side: the season clock, scheduler, comparability work and broker layer. An adopted engine absorbs none of it. Nautilus and AKQuant trail B by 6–19 likely days.
  - AKQuant drops off the leaderboard path. It costs 8–12 days more, and its A-share advantage is out of scope.
  - The one case that flips the order is crypto perps executed on Hyperliquid. There Nautilus leads by 2.5–9 days, which is within the noise. That justifies a spike, not adoption.
- **The premise was wrong.** A live engine already ships: `domain/leaderboard/live.py` (1,543 lines, #534). It appends one day at a time over a calendar month, on the shared `PortfolioManager` and `execution.py`, not on `HourlyBacktester`.
  - Prod is advancing: data through 2026-10-13, 9 of 22 days.
  - Only DeepSeek has a curve this month. Nemotron is `pending`.
  - The board publishes Sharpe ratios of 5.8–11.5 from 9 days of data, priced with zero costs and fills at each decision bar's own close.
- **The biggest levers are yours:** the venue and instruments (a 25–40 day swing), and whether the leaderboard or My Agents moves first.

## 2. Cost (engineer-days, low / likely / high)

S1 and S2 assume a long-only cash book. S3 means S1 plus crypto perps: it adds to equities rather than replacing them, which is how both pricers priced it.

| Option | S1: simulated season | S2: Alpaca paper per entrant | S3: S1 + perps | Calibrated likely S1 / S2 / S3 |
|---|---|---|---|---|
| **A** patch in place | 66 / 103.5 / 173 | 78 / 126 / 210 | 93 / 151 / 253 | 29–32 / 33–39 / 42–46 |
| **B** own core, step-first | 68 / 104.5 / 163 | 83 / 132 / 208 | 90 / 143 / 227 | 28–32 / 34–41 / 39–43 |
| **N** Nautilus | 79 / 123.5 / 195 | 93 / 149 / 236 | 93 / 149 / 240 | 41–44 / 45–52 / 48–51 |
| **AK** AKQuant | 72 / 113.5 / 182 | 87 / 140 / 225 | 96 / 155 / 251 | 38–42 / 44–50 / 47–54 |

**S3 with execution on Hyperliquid testnet (likely days):**

| | N | B | AK | A |
|---|---|---|---|---|
| Red-team pricer | 157 | 166 | 178 | 174 |
| Bottom-up pricer | 143.5 | 146 | 148 | 154 |

The Hyperliquid increment itself is unreconciled: +23.5 days red-team (ATL writes an adapter on Hyperliquid's SDK) against +10 bottom-up. Nautilus is first under either figure.

**How the pricers were reconciled:**
- **Which figures the table uses.** The table uses the red-team figures. They contain everything the bottom-up pricer priced, plus three items it missed:
  - a frozen bar archive for each season (1.5 days);
  - a period where two engines run side by side (2.5 days);
  - 50% rework on the live-clock code.
- **How far apart the pricers are.** S1 cells agree within 6–8%. Three cells disagree more:
  - N in S2: 130 vs 151 (16%);
  - AK in S3: 138 vs 155 (12%);
  - B in the Hyperliquid variant: 146 vs 166 (14%).
- **AK in S3.** I used the red-team figure. The bottom-up figure assumes AKQuant's CTP futures margin carries over to perps, and the web research found no perps support in AKQuant.
- **Double counts removed:**
  - N's dependency upkeep (1 / 2 / 3) appeared in both its derived base and its S1 delta. I removed it.
  - A's likely figure is about 1.5 days too high. That is 0.8 counted twice (the engine key duplicates INT-3, plus K11's +0.3) and 0.5–1 for a snapshot migration that a month-boundary cutover makes unnecessary. I left this in the table because it changes no ordering.
- **The N and AK bases** were derived by the pricers and never reviewed.
- **The calibrated column** is weaker for the leaderboard additions than for the base. Live-clock, broker and ops work is mostly design, review and prod verification, which a lines-per-day rate does not capture.

**Priced variant: shorting and leverage on US equities.** nof1's equities season (S1.5) used leverage and implied shorts, so the arena may want them. I derived a price from the bottom-up perps components (my derivation, judgement):
- **Engine work:** margin and leverage ledger 5, liquidation 4, shorting 1.5, oracle 4. That is about 14.5 days for B and about 19 for A (A pays 1.3× for bolting margin onto its ledger). Scaled red-team figures give B ≈10.5 and A ≈17.
- **ATL-side work:** validator and prompt changes plus frontend, about 6–9 days. The validator is a security boundary.
- **Net effect:** this adds about 20–28 days to S1 and puts B ahead of A by about 5–6. Nautilus's native margin narrows its gap to B to about 11–13. Borrow fees, locates and margin interest are not priced.

## 3. Recommendation and staged plan

### Build B step-first. Changes from the probe's design

- **Primary interface.** `advance_to(ts)` / `submit` / `view` becomes the main API, and a batch backtest is a loop over it. Adds about 1–1.5 days.
- **C6 versioned, serialisable state.** It must hold:
  - working orders, because under P1–P3 a 16:00 decision becomes a next-open order still pending at exit;
  - receivables;
  - the 24-hour `recent_trades` memory;
  - the H6 counters.

  Property test: N bars run straight must be byte-identical to N/2 bars, a restore, then N/2 more.
- **Two thin drivers on one core:** the C13 `HourlyBacktester` facade, and a leaderboard driver that replaces `llm_agent.py:187-262`.
- **No snapshot migration.** `live.py:1-8` runs one continuous month per entry, so cutting over on the 1st migrates nothing. Two-week seasons would give the same benefit; month boundaries already do.
- **Keep the C14 isolation flag until the leaderboard cutover.** C14 edits the translator in the shared `portfolio_manager.py`. Merging it mid-month would put two translator behaviours into one stitched curve.
- **Persistence.** Live state stays on `AGENT_RUNS_DATABASE_URL` (Postgres). Never build on `protocol_runs`, `idempotency_keys` or `PaperBackend`: all are SQLite and reset on every deploy. Run a single instance only, because replicas would bill the LLM calls twice.
- **Hedge (judgement):** do not hard-code a long-only ledger in C3.

**The A path for the leaderboard that was not priced.** Under A, the K2/K5/K8 executor fixes could reach the leaderboard through opt-in flags, and only K1/K3/K9 would be ported into the 76-line loop. That may undercut A's cost to move the leaderboard (15 likely, 26 high), which is what "B wins on risk" rests on. The price is a permanent second loop owning P1–P4.

### Stage 0: start now, about 24 likely days, needed under every option

- **The probe's 15.35 days:** D19 calendar 1.6, fee tables 1.0, P19 Sharpe ≈5.5, corporate actions 4.0, fractional and fee persistence 2.25, INT-2 stamp 1.0.
- **Leaderboard decision tape, 2.5. Do this first.** Tapes cannot be backfilled, and today no leaderboard bar is recorded. Audit, nof1-style per-model trade logs ("ModelChat") and differentials all depend on it.
- **Early closes on the live axis, about 1 (judgement). Deadline 2026-11-27.** `live.py:56` hardcodes seven hourly nodes. Write the half-day test first. Whether the effect is mis-stamped bars or empty nodes is unverified.
- **Carry trade memory in the snapshot, about 1 (judgement).** This is a live defect today. `restore_state` never restores `self.trades`, and `recent_trades` is built from it (`portfolio_manager.py:445`). So every nightly increment starts with an empty 24-hour memory.
- **D12 disclaimer, 0.4.** The prompt advertises stop and take-profit orders that are ignored, on a public board.
- **Leaderboard Sharpe on the P19 method, 1.5.**
- **Engine key in `_find_cached_run` / `_vetted_pool`, 0.5.** Under A this is INT-3, so it is not additional there.
- **Frozen bar archive, 1.5.**

The corporate-actions item sources the data and back-adjusts indicators. It does not credit dividends on the live board's ledger; that is engine work and waits for the leaderboard cutover. Season rollover and an hourly scheduler stay conditional on the cadence decision, at about 10–13.5 days once the two items above move into Stage 0.

### Later stages

1. **B prototype** (§5).
2. **Core:** C1–C7 plus C6.
3. **The first driver.** This is your choice:
   - **Leaderboard first.** About 37 likely days to cutover: C1–C7 15, plus 21.5 of bottom-up S1 engine-side items. Moving to B's own core is phase 3; Stage 0 runs alongside phases 1–2 on today's engine. That is comparable to My Agents' first ship (35–39). It serves the stated focus, and it proves B through a thin driver before the C13 facade (the probe's biggest risk, 4–8 days). Costs:
     - My Agents keeps its defects longer;
     - the C13 risk is deferred, not removed;
     - 21 billable contest re-baseline runs;
     - public curves change.
   - **My Agents first.** The 2026-10-07 mandate. The live board stays on same-close fills, zero costs and raw prices for about 70 or more days.
4. **The other driver.**
5. **Step sessions, later.**

**Optional A bridge:** 13 cases in about 15 days, shippable at about 17.5. It discards 8–12 days of executor patches later.

**Calendar floors:** a 10-session shakedown before ranking new-engine curves, and 2–4 weeks of paper shakedown under S2.

## 4. Decisions only you can make

| Fork | What it flips |
|---|---|
| **Venue: S1 simulated, S2 Alpaca paper, real money** | S2 adds about 25 days of broker work that does not depend on the engine, and narrows A vs B to a toss-up. Alpaca paper is not "real" either: 10% random partial fills, no regulatory fees, no dividends, and paper-only accounts get IEX data only. Real money (nof1's credibility claim) takes S2 as a floor, plus custody, legal and capital at risk, none of it priced. S1 with repeat-run bands is ATL's differentiator; nof1 never repeated a run. |
| **Instruments: long-only, equity shorts and leverage, or perps** | Shorts and leverage add about 20–28 days, change the validator, and put B ahead. Perps (S3): B ahead by about 8; with Hyperliquid execution, run the Nautilus spike first. LLM spend rises about 5.8× (203 against 35 decisions per model per week). |
| **Sequencing: leaderboard or My Agents first** | About 37 days to the first cutover either way. Trade-offs as in §3. |
| **Cadence and season length** | Hourly appends buy publication latency only: the same 7 decisions a day on settled bars. They cost 10–13.5 days plus a Render worker, because GitHub Actions fired the 22:30 cron 2.5–4.5 hours late on every scheduled run in the last week. Month or two-week seasons does not change the cutover cost. |
| **Comparability and history** | Today's live curves are optimistic: zero costs and same-close fills. nof1's fees reached about 13% of capital for one model. If yes: 21 billable re-baseline runs and a restart of the live month at cutover. If no: label the old curves legacy. |
| **Roster** | Curated or participatory. Entries run through `LLMAgentStrategy`, not My Agents. Each entrant multiplies the LLM spend and needs its own capital ledger. |
| **S2 only** | Whether one login holds N paper accounts (+3 days if the Broker API is needed). The PDT rule binds seeds under $25k. SIP is about $99 a month. |

**New recurring costs (flagged):**
- a Render worker (needed for hourly appends or S2);
- the SIP subscription (S2);
- a Nautilus node, which must use `run()` rather than `run_async()`, so it is a separate always-on service, plus Redis or Postgres for its cache;
- 21 billable runs for each contest re-baseline.

## 5. Cheapest next steps

1. **Decide venue, instruments and sequencing.** Under a day of your time; it settles the 25–40 day swing.
2. **Start the leaderboard tape (2.5 days) and the half-day fix (deadline 2026-11-27).** Separately, find out why Nemotron is pending. It is half the LLM roster.
3. **B prototype, 3–4 days.** Build C1 + C2 + C4 + C6 with versioned state, driven by the leaderboard loop over one recorded live month.
   - Kill it mid-session with a next-open order pending, resume, and require a bit-identical result.
   - Time the active days. That settles the throughput question: 358 against about 81 production lines per day moves every figure about 3.5×.
   - The bottom-up pricer scoped this at 2–3 days; the extra day covers the live-month replay.
4. **If shorts or perps are in scope:** a 3–5 day Nautilus spike instead. Cover Hyperliquid testnet, restart and resume, liquidation in the simulator (#3565), and #5280, where a periodic position check closes positions even in Sandbox.

## 6. Sources and uncertainty

- **Code at 2dc8203a,** checked by me or the critic:
  - `live.py` size, `:56`, `:74-77`;
  - the loop at `llm_agent.py:187-262`, with same-close fills;
  - nothing in `domain/leaderboard` references `HourlyBacktester` or a decision tape;
  - `snapshot_state` / `restore_state` (`:194` / `:228`) hold no trades;
  - `PaperBackend` raises `NotImplementedError` in all four methods;
  - the Alpaca paper client only reads;
  - no shorting or margin in the executor or validator;
  - AKQuant snapshots its ledger through msgpack `get_state_bytes` (adoption research :47, :96). Pickle applies only to its `save_checkpoint`.
- **Prod, checked 2026-10-14:**
  - `LIVE_LEADERBOARD_DEPLOY_MODELS=true` since 2026-09-29;
  - `live_status`: frozen through 2026-10-13, 9 of 22 days, `models_pending 1`, `snapshot_stale false`;
  - Sharpe 5.8–11.5;
  - scheduled runs fired at 01:17–03:04 UTC.
- **Web:**
  - **nof1:** secondary sources only. Cadence, leverage and SL/TP rules are unverified.
  - **Nautilus:** no Alpaca adapter ([#3374](https://github.com/nautechsystems/nautilus_trader/issues/3374)); [#5280](https://github.com/nautechsystems/nautilus_trader/issues/5280); [#3565](https://github.com/nautechsystems/nautilus_trader/issues/3565). It is LGPL, so any patches must be published. v2 is still in release candidates, and the maintainers advise against RC wheels for real capital.
  - **Alpaca paper:** checked today on its [docs](https://docs.alpaca.markets/docs/paper-trading).
  - **The $99 SIP price** comes from one pricer only.
- **Judgement:**
  - every engineer-day figure (±50%);
  - the N and AK bases;
  - the equity-shorting derivation;
  - the two roughly 1-day Stage 0 items;
  - the hedge against a long-only ledger.

  "#534 rework" is really about 44% follow-up churn: 2,956 production lines, then 1,301 within 6 days, including a new feature and review fixes.
- **Not verified:** whether Nautilus can give each strategy its own capital, and whether its margin model fits Reg-T.
- **Stale documents:**
  - Repo `CLAUDE.md`: "Season 0 preview".
  - The nof1-frame and participatory-competition specs.
  - User-facing (to update): `app.html:1578/1588/1595`, `operating_modes.rst:14`, `Race.tsx:27,151,159`.

The full cost probe is at `/tmp/claude-1000/-mnt-c-Users-27740-OneDrive-Documents-Github-AgenticTrading/fd372dd1-b067-4939-9adf-8b42585acdc6/scratchpad/probe/report.md`.
