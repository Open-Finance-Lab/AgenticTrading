> **Status (added 2026-10-13).** Written 2026-10-07 by workflow `wf_2676301e-bea` into a session scratchpad, which was later wiped. Recovered byte for byte from that run's saved result. Everything below the line is the original. Read it with these later facts in mind:
>
> - **Scope narrowed (owner, 2026-10-07).** The rewrite covers only the My Agents backtest (`POST /backtest/run` → `HourlyBacktester`, one subprocess per run), US only. A-share, paper/live trading and the live leaderboard are out. The step sessions (`/api/v1`, `/api/v2`) stay on the old engine. The contest leaderboard is expected to reuse the new engine later.
> - **The recommendation below is therefore open again.** AKQuant's main advantage was its A-share rules. NautilusTrader was ruled out mainly by one engine per process against 100 in-process sessions (R08), but the in-scope path already runs one process per run. R08 also doesn't bind in practice: prod peaked at 7 finalized sessions per 10 minutes. The choice is AKQuant vs NautilusTrader vs an ATL-owned core, scored on the conformance suite.
> - **Critic findings this report did not absorb:** in-process session memory is about 5× the estimate here (≈27 MB per parked session); AKQuant has bus factor 1 and its CI runs none of its 227 test files; about 10 recent AKQuant bugs hit exactly the T+1 + NextOpen path; adopting vs building was never costed head to head.
> - **The "golden conformance suite" proposed below now exists**: spec `2026-10-07-us-conformance-suite-design.md`, implemented in #623 under `dashboard/backend/tests/conformance/`.
> - ATL line citations point at the working tree as of 2026-10-07 and have drifted since.

---

# Backtest engine decision: adopt AKQuant behind an ATL-owned simulator port

Prepared 2026-10-07. The inputs were two ATL requirement surveys (engine-core and protocol-and-io), seven search angles, a shortlist of eight engines, and one source-level dossier per engine. Each dossier was checked twice, once for accuracy and once for adoption fit. Where a verifier refuted a dossier claim, this report uses the verifier's evidence and marks it **(verifier correction)**. Engine source was read at these commits: AKQuant `e86d611` (v0.3.65), NautilusTrader `49832e9` (develop = v2.0.0rc7; rc6 is the latest on PyPI), LEAN `80e7843`, Qlib `be72549`, Hikyuu `b2737ff`, QF-Lib `dfbcffc`, vnpy_portfoliostrategy `326ad69`, ml4t-backtest `672804b`. ATL references point at the working tree on branch `fix/empty-instruction-default-prompt`.

---

## 1. Bottom line

Bring in **AKQuant** (github.com/akfamily/akquant). It is MIT-licensed, has a Rust core with a PyO3 abi3 wheel, and should be pinned with a hash to `akquant==0.3.65`. Make it the single simulation core for dashboard runs, protocol/v2 sessions, leaderboard entries, Live-board resume and baselines. ATL drives AKQuant's low-level `Engine` directly, never `run_backtest`, and every call goes through an engine-agnostic `SimulatorPort` that ATL owns.

AKQuant is the only shortlisted engine whose built-in features cover what ATL keeps getting wrong in **both** markets:
- next-bar-open fills by default;
- sells processed before buys, with fill-time cash re-projection;
- a Decimal ledger with FIFO average cost;
- volume-capped partial fills whose remainder carries forward;
- T+1 sellable quantity;
- rejection of buys that are not a lot multiple;
- the A-share fee schedule;
- an ex-date dividend and split ledger on unadjusted prices.

It also meets the hard constraints:
- many engines can live in one process, which the 100 in-process protocol sessions need;
- it does not swallow `SystemExit`;
- it installs as a 7.7 MB wheel on `python:3.13-slim` with no conflicts against ATL's pinned dependencies;
- it can snapshot and restore the ledger.

No engine clears the bar unconditionally. AKQuant is eight months old, has effectively one maintainer, runs no tests in CI, and its verifiers found silent wrong-number defaults and two engine bugs. Commitment therefore waits on a two-week spike with explicit kill criteria. Every wheel upgrade must also pass an ATL-owned golden conformance suite, seeded from ml4t-backtest's MIT scenario matrix.

NautilusTrader is the stronger general-purpose engine, but it breaks the in-process protocol contract (one engine per process), gives no help with A-share rules or corporate actions, and cannot restore a ledger. If AKQuant fails the spike and the in-process protocol requirement (R08) stands, the fallback is a minimal ATL-owned core behind the same port, not Nautilus.

---

## 2. Ranked shortlist (all eight)

| # | Engine | Soundness / accuracy | A-share | External bar-by-bar control | License | Maintenance | Footprint (py3.13, 2 GB, 5 children) | Overall fit |
|---|---|---|---|---|---|---|---|---|
| 1 | **AKQuant** v0.3.65 | Sound core. NextOpen is the default and a guard stops an order filling on its own bar (`src/execution/simulated.rs:322-340`, #307). Sells go first with fill-time cash and T+1 re-check (`simulated.rs:265-277, 430-650`). Decimal ledger, FIFO average cost (`src/analysis/tracker.rs:575-600`). **Verifier-found defects:** `run_backtest` resolves fees to **zero** (`python/akquant/backtest/engine.py:1143-1150`); the volume cap default is never applied, and passing exactly 0.25 is ignored (`engine.py:4002-4009`); after a split, the split symbol is marked at the old price until its own bar arrives (`src/market/corporate_action.rs:57-100`); a next-open order is rejected if its symbol had no bar in the creation slice (`src/pipeline/stages/data.rs:97-139`); limit fills trigger on touch, and slippage can push them past the limit (`src/execution/common.rs:261-288, 347`). | **Best of the eight.** Native: T+1 (`src/market/china.rs:297-315`), buy-lot rejection (`common.rs:138-164`), ¥5-minimum commission plus sell stamp duty plus transfer fee (`src/market/stock.rs:36-68`), suspension treated as volume = 0, and a dividend/split ledger (`corporate_action.rs`, low-level API only). Missing: price-limit bands, ST, STAR 200-share lots, the odd-lot sell rule. | No step API: `run()` owns the loop (`src/engine/python.rs:1301-1396`). A blocking `on_bar` is fine. Multiple engines per process are allowed (no globals, no `Rc`/`RefCell`, not unsendable). Needs a worker-thread bridge per protocol session. Ledger snapshot via `get_state_bytes`/`load_state_bytes` (`python.rs:627-760`). | MIT | Bus factor 1 (1,215 commits against 1 external). 214 releases in 8 months. CI runs no tests (`.github/workflows/release.yml`). Rule version moved 1.3.7 → 1.7.0 in 7 weeks. Fixes land in days. | 7.7 MB abi3 wheel. ATL pins satisfy all floors. **Verifier correction:** `import akquant` does not import polars, and the plotly import is guarded (`plot/utils.py:3-11`). RSS not measured. | **Conditional adopt (recommended)** |
| 2 | **NautilusTrader** v2.0 rc | Most rigorous event ordering, a documented `ts_init` = bar-close contract, cash never goes negative, seeded fills. But a default `on_bar` order fills at the **same bar's close** at zero spread, and there is no next-open mode (#4063 closed as won't-do). **Verifier:** bars smaller than four size increments fill at the bar's low or high (`matching_engine/mod.rs:7676-7728, 2070-2078`). v2 **swallows** exceptions raised in `on_bar`, including `SystemExit` (`crates/common/src/actor/data_actor.rs:1078-1080`). | None: no T+1, no limits, no lot-multiple check, no calendar, no corporate actions (#3307 open). | `on_bar` may block. But the engine is `unsendable` (`crates/backtest/src/python/engine.rs:80-84`) and running engines concurrently in one process is unsupported (`docs/concepts/architecture.md:706-722`), which **blocks R08**. No state restore. | LGPL-3.0-only (fine as an unmodified pip dependency; the Docker image needs a notice and source pointer) | Company-backed with fast fixes, but about 82% of commits come from one author **(verifier correction from 87%)**. v1 is end-of-life; v2 is pre-release and every rc has a Breaking Changes section. | 69.9 MB x86_64 wheel **(verifier correction)**, no Python dependencies. v2 RSS not measured. | **Runner-up.** Wrong shape for the protocol path and for A-shares. |
| 3 | **ml4t-backtest** 0.1.12 | Sensible defaults: next-bar open, exits first, weighted cost basis. **Verifier:** Sharpe ignores the risk-free rate (`analytics/equity.py:141-154`); stops are not checked on the entry bar; pending orders never expire; a null OHLC field is silently replaced by the close. | None. No fill-time veto, and exit orders bypass the gatekeeper (`core/execution_engine.py:553-568`). | No step API; thread bridge needed; no restore. | MIT | One author (178 of 183 commits), 30 stars. | **Verifier:** needs `numpy>=2.3.2`, which conflicts with ATL's `numba==0.61.2` / `pandas-ta` pins (`requirements.txt:37,40`). | **Not an engine.** Harvest its scenario set and semantic matrix (`validation/METHODOLOGY.md:249-330`) as ATL's conformance suite. |
| 4 | **QuantConnect LEAN** | Best US corporate-action and reality models; data arrives at bar EndTime. But the default fill is the same bar's close with NullSlippage and 2x equity leverage. **Verifier:** dividends from factor files are rounded to 2 decimals (`Common/Data/Market/Dividend.cs:82-98`); 60-minute bars starting on the half hour get re-stamped, and the 15:30 bucket is emitted after the close. | None (no China market in `Common/Market.cs`). | One engine per process (static `Config`/`Composer`); no step API. | Apache-2.0 | Strongest team: five or more committers, public fixes within days. | .NET 10 plus pythonnet per process. **Verifier correction:** a NuGet package exists, and Python 3.11 can be avoided by writing the algorithm in C# and calling ATL over RPC. | **Design reference only** |
| 5 | **vn.py portfolio** | Next-bar crossing only. Futures-style P&L with no cash account. Synthetic stale bars fill at the decision bar's close. **Verifier:** warm-up is off by one (`backtesting.py:63, 217-229`) and the Sharpe risk-free term is dimensionally wrong (`:390`). | None. | `new_bars` / `send_order` can be stepped from outside. | MIT | Tests added only on 2026-10-04. | About 256 MB of PySide6. On import it adds a loguru file sink and creates `~/.vntrader`. | Reject: worth about 100-150 lines of logic. |
| 6 | **Qlib backtest** | **Verifier:** the `+0.1` lot rounding can push cash negative (`exchange.py:783`); the `$change` limit gate reads the fill bar's own close; `min_cost` is a single floor on the total cost (`:948`). | Partial: lots (with factor = 1), suspension, boolean limits. No T+1. | **Verifier correction:** a 6-line custom driver gives pull stepping. | MIT | No functional backtest change since 2025-08. | No cp313 wheel, and `gym` kills the process under NumPy 2 (#2018). | Reject |
| 7 | **QF-Lib** | **Verifier:** fixed-UTC session times misplace the winter close; the volume cap reads the whole day's volume; the remainder of a partial fill is dropped; no cash check; `MIN_5 = 1965` is a typo in the enum (`frequency.py:35`). | None. | `dispatch_next_event` can be stepped, but event state lives on classes (global). | Apache-2.0 | One main author. | Pins `numpy<=1.26.4`. WeasyPrint 52.5 fails at import when pango is missing. | Reject |
| 8 | **Hikyuu** | The TradeManager is only a ledger. **Verifier:** `getFunds` values an intraday query at the day's last close, which is look-ahead (`TradeManager.cpp:~1275`); built-in fee schedules are out of date; recent bugs in corporate-action handling (#509, #511). | Partial: lots, plus ex-rights weights loaded only from its own database. | `tm.buy` / `tm.sell` can be called from ATL's loop. | Apache-2.0 in the repo, MIT in the wheel. | One author. | 76 MB wheel, PySide6 is a hard dependency, and **a telemetry beacon fires on import** (`GlobalInitializer.cpp:92-94`). | Reject |

---

## 3. Wider landscape: found but not shortlisted

- **RQAlpha.** Most complete A-share rule set, but the LICENSE forbids use by any organization without Ricequant's written grant. Do not read its source, so a clean-room implementation stays clean.
- **zipline-reloaded.** No real commits in 2026, open correctness PRs (#313, #333, #335), daily or minute frequency only, and every live window needs a bundle ingest.
- **Lumibot.** GPL-3.0, with open bugs in fill timing (#1175) and dividends (#402). Its replay cache is a useful idea.
- **backtrader and its forks.** GPL-3.0, unmaintained since 2024-08, fill bugs never merged (#415).
- **backtesting.py.** AGPL-3.0, single asset, open look-ahead bug (#963).
- **vectorbt OSS/PRO.** Commons Clause blocks paid hosting, and the vectorized design cannot run an LLM per bar.
- **PyBroker.** Commons Clause, semi-vectorized, and its own multiprocessing.
- **SimTradeLab.** AGPL plus a paid licence. It emulates the PTrade API rather than being a library.
- **Vibe-Trading engines.** Embedded in an app, and decisions are precomputed weights. Keep `china_a.py` only as a citation for the limit-check look-ahead fix.
- **WonderTrader/wtpy.** C++ futures runner. Ex-rights handled only through adjustment factors; no T+1 or limit logic.
- **QUANTAXIS.** MongoDB monolith, last release 2020.
- **ATRS.** Repo not found; claims are self-reported; it rebalances weights rather than handling orders.
- **mx-backtest, a-share-honest-backtest.** 0-2 stars, never opened.
- **hftbacktest.** Tick and order-book market-making simulator, open partial-fill accounting bug (#316).
- **Barter-rs.** Rust only, crypto, disclaimed for production use.
- **QSTrader.** Dormant since 2024.
- **bt.** A weight allocator with no order or fill simulation.
- **Jesse, freqtrade, hummingbot.** Crypto only; freqtrade is GPL.
- **FinRL, Gym-Trading-Env, FinRL-X.** Fill at the same close the agent saw; RL toy environments.
- **FINSABER.** Daily bars, US, single ticker at a time. Its next-open default, 2.5% liquidity cap and LLM-cost accounting are folded into the checklist.
- **TradingAgents, ai-hedge-fund, StockBench, INVESTOR-BENCH/FinMem, QuantAgent.** No adoptable engine; useful only as a list of point-in-time leaks.
- **AI-Trader, LiveTradeBench, CLQT, PredictionMarketBench.** Live arenas or paper-only systems. CLQT's TimeGate is a practice reference.
- **GlowBack, tesser, RLXBT, RaptorBT, Observa, RustyBT.** Too young and unvetted.
- **PyAlgoTrade.** Archived in 2023, licence unclear.
- **Companions, not engines:** `exchange_calendars` (adopt it as the single session/holiday/half-day owner, R52), `quantstats` (optional), Look-Ahead-Bench (an input to leaderboard methodology on LLM memorisation).

---

## 4. Recommended engine: AKQuant

### 4.1 What moves out of ATL

| Today (ATL-owned) | After (AKQuant-owned) |
|---|---|
| `domain/trading/execution.py` (892 lines): fills, fees, T+1 frozen lots, lot checks, list-order execution | Matcher with NextOpen, sells first with cash re-projection (`simulated.rs:214-277, 866-915`), `StockConfig` fees, T+1 availability, buy-lot rejection |
| `domain/trading/portfolio.py` (189 lines): valuation, the `entry_prices` overwrite bug (`execution.py:631`) | Decimal portfolio, FIFO lots with weighted average price, realized and unrealized P&L (`tracker.rs:62-87, 575-600`) |
| The ledger half of `PortfolioManager` (`execute_actions`, `update_equity`, `snapshot_state`/`restore_state`, frozen lots) | Engine state plus `get_state_bytes`/`load_state_bytes` |
| Four decision loops with three fill models (`engine.py:2131-2297`, `llm_agent.py:215-250`, `external_run_service.py:773-840`, `backtest_custom_algo.py:243-366`) and the execution plan (`bar_aggregation.py:362-411`) | One engine loop; NextOpen everywhere; the last bar of a session fills at the next session's open |
| No liquidity model; orders up to 10,000 shares fill at a single price | `set_volume_limit(pct)` with the remainder carried forward (`common.rs:349-370`; `tests/test_partial_filled_status.py`) |
| A-share ex-rights refused or charted as a loss (`market_rules.py:58-101`) | `CorporateActionManager`: cash dividend on the ex-date, splits rescale position, availability and lots (`corporate_action.rs:31-104`) |
| Orders dropped silently with no `order_event` (`execution.py:526-527`) | Orders end in an explicit status with a `reject_reason` |

Estimate: about 3-4k of the 9.1k lines in `domain/backtesting` plus the 1.1k lines in `domain/trading/{execution,portfolio}.py` are retired or reduced to adapters. More important than line count, the classes of semantics that produced every patched bug stop being ATL code.

### 4.2 What ATL still owns

- Data layer: Alpaca/iFinD providers, bar cache, SIP clamp, inclusive end (`provider.py:24-109`), warm-up pad, features, 5m→60m aggregation, and stamp conversion to bar close.
- Decision provider: prompt, pipeline, strict-LLM strikes, `llm_calls`/`llm_decisions`/`decision_steps`, hosted runtime, reference agent, validator, execution_client and billing.
- A **pre-trade gate**:
  - price-limit bands against the prior **daily** close, per board and historical regime;
  - ST bands, STAR 200-share minimum with 1-share steps, the odd-lot sell rule;
  - symbol tradability at T, `max_position_weight`, `max_orders`;
  - ATL's own rejection vocabulary for all of the above.
- Corporate-action **event data**: iFinD ex-rights and Alpaca corporate actions, converted to `CorporateAction(symbol, date, type, value)` rows with the dividend ahead of the split.
- FX and USD reporting (`CurrencyContext`, unchanged).
- Per-bar equity for the progress file.
- Metrics on a declared daily grid, owned by ATL. Do not trust engine Sharpe: it drops the first day and defaults to 252 days (`src/analysis/result.rs:194-231`) **(verifier)**.
- Baseline orchestration and index series.
- The session facade: deadlines, holds, idempotency, caps, reaper.
- The result writer and the metadata allow-list.
- Engine-version stamping, migration of Live-board snapshots, and the golden conformance suite.

### 4.3 Integration seam

New layout inside `dashboard/backend/domain/backtesting/`. Existing import paths are kept as compatibility modules.

| Module | Fate |
|---|---|
| `sim/port.py` (new) | `SimulatorPort` protocol: configure, load bars, register strategy, run, state, snapshot/restore, result. Nothing outside `sim/` may name AKQuant. |
| `sim/akquant_engine.py` (new) | The **only** module that imports `akquant`, enforced by a new rule in `tests/test_architecture_boundaries.py`. Drives the low-level `Engine`. Every fee, T+1, lot, volume-limit and timezone setting is passed explicitly and then **read back and asserted at startup**, which neutralises the zero-fee and ignored-cap defaults **(verifier)** and symbol-suffix mismatches that silently drop per-instrument config (`engine.py:2089-2123`) **(verifier)**. |
| `markets/us.py`, `markets/cn.py` (new) | Dated cost profiles, the pre-trade gate, corporate-action event builders, and the rules for building limit/suspension bars. |
| `drivers/push.py`, `drivers/pull.py` (new) | The two drivers (below). |
| `decision/` (new, moved out of `portfolio_manager.py:376-1048`) | Prompt, pipeline, strict-LLM, counters. |
| `writer.py`, `progress.py` (new, moved out of `engine.py:992-1172, 2334-2453`) | `db.insert_*`, metadata keys, progress file, durable mirror every 10 steps. |
| `engine.py` | Rewritten as a thin `HourlyBacktester` compatibility class: same 23 constructor kwargs and the same `load_data → calculate_indicators → run_agent_backtest → run_buyhold_baseline → run_djia_baseline` lifecycle (C01). |
| `portfolio_manager.py` | Becomes a facade: read-only ledger views from captured engine state; `snapshot_state`/`restore_state` wrap engine bytes plus ATL JSON (C14/C15). |
| `external_run_service.py` | Facade keeps every public name and session attribute (C16); internals become the thread bridge. |
| `market_data_store.py`, `indicator_warmup.py`, `features.py`, `currency.py`, `market_rules.py`, `constants.py`, `provenance.py`, `partial_results.py`, `reference_agent.py`, `baseline_worker.py`, `algo_service.py`, `baselines/paper.py` | Kept at their paths. Engine-neutral ATL layers. |
| `bar_aggregation.py` | Aggregation kept; the execution-plan part (`:362-411`) is deleted. |
| `metrics.py` | Signatures kept (C09/R31); definitions redesigned on a daily grid with the risk-free rate; NaN-safe at $0. |

**How the LLM and external agents drive it:**

- **Push driver.** Used for dashboard LLM and pipeline runs, the hosted AI Hedge Fund, the rule-based agent, leaderboard entries and passive baselines.
  - `AtlStrategy(akquant.Strategy).on_bar` fires once per symbol per bar.
  - The driver knows from ATL's dataset which symbols are expected at each decision timestamp T. It calls the decision provider once all of them have arrived. It never infers "last symbol" from event order **(verifier)**.
  - The validated `{"actions":[...]}` goes through the pre-trade gate and becomes market orders via `self.buy/sell`. Gate rejections are recorded in ATL's vocabulary.
  - The engine fills on the next bar's open. A blocking LLM call inside `on_bar` is fine: blocking Python releases the GIL, and there is no engine-side clock.
- **Pull driver** (`/api/v1/runs`, `/api/v1/backtest/*`, `/api/v2`).
  - One worker thread per session runs `engine.run()`. At decision T the strategy publishes `{snapshot, fills since the previous step}` to the session and blocks on `queue.get()`.
  - `submit_decisions` (under `_step_lock`) validates and enqueues the actions, then waits until the engine parks at the next decision timestamp. That takes milliseconds of compute. It then returns `executed[]`, which are the next-open fills, plus `next_step`, so the R03 response shape is unchanged.
  - Deadline expiry and `drain_expired` enqueue an empty decision, which becomes `timeout_hold`. Cancel enqueues a sentinel; `on_bar` raises; `run()` unwinds.
  - The thread boundary catches `BaseException`, because PyO3 panics surface as `PanicException`, which is a `BaseException` (`china.rs:183-233`) **(verifier)**.
- **Subprocess driver.** argv, stdin handoff, progress file and exit codes are unchanged (R19/R20). `SystemExit` raised from ATL's SIGTERM handler inside `on_bar` propagates out of `run()` (`src/engine/core.rs:1297`, `python.rs:1374-1396`).

**Clock mapping:**
- **US.** Feed Alpaca 5m bars re-stamped to bar close (+5 min), tz-aware in America/New_York, and set the engine timezone explicitly. Naive timestamps would default to Asia/Shanghai and shift day boundaries (`docs/en/advanced/timezone.md`). Decide at ATL's 60m bucket ends and fill at the next 5m open. The 15:30-16:00 bucket fills at the next session's first open.
- **CN.** Feed close-stamped iFinD 60m bars. Decide at bar close and fill at the next hourly bar's open: the 11:30 decision fills at the 13:00 open, the 15:00 decision at the next 09:30 open. Do not use AKQuant's own multi-timeframe windows; they are anchored to midnight (`src/data/window.rs:209-216`).
- `Bar.extra` is visible to the strategy on the same bar, so it must never carry next-bar information **(verifier)**.

### 4.4 Accuracy bug classes closed

| Issue | Class | Closed by AKQuant? |
|---|---|---|
| #563 (merged) | Off-by-one on the inclusive end date | Not an engine concern. The port keeps calling `settled_exclusive_end` (C20). |
| #540 / #595 (merged) | Indicator warm-up look-ahead | Not an engine concern. `indicator_warmup.py` remains the single owner; the engine never sees pad bars. |
| #529 (merged) | Open-stamped bars | **Partially.** Next-bar fills are structural (#307 guard), but the engine trusts the caller's stamps. The feed adapter does the +5 min shift and a fixture test pins it. |
| #346 (open), #464 (merged guard) | A-share ex-rights not applied | **Yes, conditionally.** Ex-date cash dividends and split/bonus scaling of position, availability and lots. Retire the refusal guard and `IFIND_ALLOW_CORPORATE_ACTION_GAPS` only after the golden for 10送3转2派1 passes and the split stale-mark bug is fixed or shimmed **(verifier)**. |
| (no issue) | US splits and dividends ignored (`alpaca_bars.py:733-739`) | **Yes**, via the same ledger, fed from Alpaca corporate actions. |
| #597 (open) | Warm-up pad crosses an ex-rights date | Not by the engine. ATL computes indicators on a back-adjusted series built from the same factor table it feeds the engine, which replaces the 21%/30% trim heuristic. |
| #598 (open) | Protocol runs get no warm-up | Already fixed in code by `1c3b0b59`. Close it now. |
| #128 (open) | Simulated bars go flat | Moot. Drop `vnpy_simulation`, or keep it only as a test fixture. |
| #461 (open) | US backtests have no costs | **Mechanism yes:** SimpleMarket commission and minimum, a sell-side fee fraction, per-order slippage override, volume cap. Policy: publish net results. |
| #480 (open) | A-share bar floor has no unit coverage | Covered by the golden suite. |
| #573, #375 (open) | Holds leave no trace | Engine-neutral. Every hold now passes through one driver, so record holds by kind there; both fixes become cheap. |
| #600, #601, #524 (open) | Retry, reclaim and timeout records | Engine-neutral (decision provider and persistence). |
| Unfiled | CN fills at the same bar's close (`engine.py:1911-1919`); leaderboard same-close fills with no costs (`llm_agent.py:249`); US 16:00 same-instant fill (`bar_aggregation.py:395-408`); average cost overwritten (`execution.py:631`); buys rejected because they were listed before the funding sell (`execution.py:442-447`); orders dropped silently; no liquidity cap; equity dips on unpriced symbols (`portfolio.py:27-38`); `initial_equity` recorded after fees (`engine.py:2343-2345`) | **Yes.** |

### 4.5 Gaps ATL must fill

1. **Limit-up/limit-down at the fill bar.** AKQuant's default matcher ignores `Bar.extra`, and a custom matcher would replace the **entire** matcher and turn exceptions into no-fill (`src/execution/python.rs`). So: (a) the pre-trade gate blocks orders whose side is limit-locked at decision time, using prior-close bands; (b) open an upstream request for the default matcher to honour `can_buy`/`can_sell` (the maintainer fixes things in days); (c) until that lands, model one-price limit-locked fill bars as volume = 0. That approximation blocks both sides, and it must be recorded in `market_rule_profile`. Do not use a custom matcher.
2. STAR 200-share minimum with 1-share steps, the odd-lot sell rule (`common.rs:140` checks lots on buys only), ST bands, the ChiNext band change on 2020-08-24, and lot-aligned volume caps (`common.rs:349-356` is not lot-aligned). Either keep the volume cap off for CN and size orders to a participation limit in the gate, or get an upstream fix.
3. Dated fee schedules: 0.1% stamp duty before 2023-08-28. Spike question: can fees change mid-run? If not, refuse or split windows that straddle a rate change.
4. Map rejection reasons: free-text `reject_reason` to ATL's vocabulary, with a fixture per reason. An unknown reason maps to `engine_rejected` and the raw text is stored, so it stays visible.
5. Fee audit columns: `Trade.commission` is a single float, so ATL recomputes commission, stamp duty, transfer fee, slippage and reference price from its own `markets/*` profile and asserts the sum matches.
6. A shim for split stale marks until fixed upstream: rescale the split symbol's last price at the ex-date, or guarantee every held symbol has a bar at the decision timestamp **(verifier)**.
7. Same-slice next-open rejection: the gate refuses orders for symbols with no bar at T, with an explicit reason, until fixed upstream **(verifier)**.
8. A contract test that pins the underscore-prefixed `Strategy` internals the Rust core calls (`_on_bar_event_and_flush`, `_on_start_internal`, and others) **(verifier)**.

### 4.6 Main risks and mitigations

| Risk | Mitigation |
|---|---|
| Bus factor 1, 8 months old, project abandoned | Exact pin with hash. All code goes through `SimulatorPort`, so swapping to an ATL-owned core costs one adapter. ATL's golden suite is the safety net. Forking is legal (MIT) but means owning Rust, so prefer replacement through the port. |
| No tests in upstream CI; behaviour churn (no changelog since 0.2.14 while the package is at 0.3.65) **(verifier)** | Record `akquant.__version__` and `__engine_rule_version__` in metadata. An upgrade is allowed only after the golden suite and the differential harness pass. |
| Silent wrong-number defaults | Never call `run_backtest`. Set every parameter explicitly and fail at startup if a read-back value is zero or unlimited when it should not be. |
| Engine bugs found by verifiers | File them upstream during the spike and measure the turnaround. Use ATL shims (4.5 items 6-7) until fixed. Submit market orders only, which sidesteps the touch-fill limit and slippage-past-limit issues. |
| 100 in-process sessions in a 2 GB web process | Measure in the spike: one thread plus one copied `DataFeed` (~5 MB estimate) per session. If over budget, lower `MAX_ACTIVE_RUNS_GLOBAL`, which is a single constant. |
| Determinism across processes is unproven (std `HashMap` with a random seed) **(verifier)** | Two-process byte-identical golden on the rule-based agent in the spike. |
| Results change across the board | One discontinuity, stamped and refreshed (section 6). |

---

## 5. Rewrite boundary: consumers outside `domain/backtesting`

| Module (from `outer_contract_consumers`) | What it consumes | Verdict |
|---|---|---|
| `dashboard/scripts/backtest_hourly_agent.py:190,508-624` | `HourlyBacktester` lifecycle, `bha.*` re-exports | **Thin adapter change**: import the compatibility class; same kwargs; keep the re-exports tests patch. |
| `api/routers/backtests.py` | argv/stdin (1764-1860), progress reader (1579-1661), status (3844-3868), provenance (98), constants (125), equity filter (175-229), `RunMetadata` allow-list (~287-372, ~478-545), order-event routes (4229-4308), cadence preflights (2143-2200, 2352-2470) | **Thin adapter change**: add the `engine` block to the allow-list. Everything else is untouched if the writer emits the same keys. Preflights are unchanged because decision cadence is unchanged. |
| `app.py:33,298,324` | paper baselines, reclaim, sweep | **Untouched** |
| `domain/runs/service.py:29-35,314,680,790-1245` | session internals (`.manager.cash/positions/trades`, `protocol_*`, `fills_since`) | **Thin adapter change**: the facade keeps these attributes via a `manager` shim; optionally replace `_infer_rejection` with engine reasons. |
| `execution/backtest_backend.py:20,74-480` | session attributes, `peek`/`adopt_dataset`, `build_final_metrics` | **Thin adapter change** |
| `execution/base.py`, `api/v2/runs.py`, `api/routers/external_backtest.py`, `domain/runs/protocol.py`, `domain/runs/environment.py` | status vocabulary, `DECISION_TIMEOUT_SECONDS`, module functions, action dict | **Untouched** |
| `domain/leaderboard/strategies/llm_agent.py:27-28,172-364` | its own `PortfolioManager` loop | **Must change**: the internal loop becomes the push driver (I14). The `run()` signature, counters and `last_portfolio_snapshot` are kept (R27). |
| `domain/leaderboard/live.py:735-760,900-960` | snapshot resume | **Thin adapter change**, plus a one-time migration of stored prod snapshots by replaying decisions into a fresh engine. |
| `domain/leaderboard/service.py:98,1441,1635-1668` | H6 counters, `_find_cached_run` | **Thin change**: add the engine version to the cache key. H6 is untouched. |
| `domain/leaderboard/baselines.py:21-25`, `strategies/buy_hold.py` and `equal_weight_*` | `INITIAL_CAPITAL`, metric wrappers, `BaselineGenerator` | **Untouched**. The numbers change; the code does not. |
| `baseline_generator.py:19-27,244-746` | `calculate_transaction_costs`, `MarketRuleCalendar`, `CurrencyContext` | **Must change**: passive policies run through the engine with the same cost profile (C28/R30). This also removes the circular engine↔generator import. |
| `domain/trading/execution.py`, `domain/trading/portfolio.py` | engine logic that lives outside the package. `rg` shows the only importers are `portfolio_manager.py` and `baseline_generator.py`; `paper_session.py` imports neither. | **Replaced, then deleted** |
| `api/routers/paper_trading.py`, `domain/trading/paper_session.py`, `infrastructure/brokers/alpaca_paper.py` | paper trading | **Untouched** |
| `infrastructure/market_data/ifind_market_rules.py:13`, `provider.py`, `bar_cache_warm.py` | `market_rules` types, warm-up start | **Untouched** (paths kept) |
| `api/routers/agents.py:22`, `portfolio.py:9`, `domain/agents/service.py:407,462`, `domain/portfolios/*` | constants | **Untouched** |
| `api/routers/algo.py:18,92,103`, `algo_service.py` → `scripts/backtest_custom_algo.py` | subclasses `HourlyBacktester` and `PortfolioManager` | **Must change** (script port to a `Strategy` callback). Router and service untouched. |
| `database.py`, `database_postgres.py` | `insert_*`, live progress | **Untouched** (no new tables; whole shares, so the `int` cast is fine) |
| `frontend/app.js` | metadata keys, progress poller | **Untouched** (optionally add an "Engine" row) |
| `baseline_resolver.py:9`, `chart_style.py:44`, `domain/agents/service.py:120` | run-id prefixes | **Untouched** |
| `domain/analytics`, `domain/entitlements/credits.py` | `owner_user_id`, run events | **Untouched** |
| `infrastructure/llm/*` (validator, pipeline_runner, execution/billing) | safety boundary, billing | **Untouched** |
| `scripts/diff_backtest_runs.py:188-278` | comparability keys | **Thin change**: refuse pairs from different engines. |
| `scripts/loadtest/stress_serve.py:68-73` | patches engine internals | **Must change** (dev tool) |
| `scripts/backfill_runs_to_postgres.py`, `packaging/agentictrading`, `integrations/discord_*` | rows, HTTP | **Untouched** |
| `requirements.txt`, `Dockerfile` | dependencies | **Thin change**: add the hash-pinned `akquant==0.3.65`. Its floors (pandas ≥2.2, numpy ≥2.2.2, pyarrow ≥14, pydantic ≥2) are met by the existing pins (`requirements.txt:36,39,41,45`). It pulls polars and plotly wheels for image size only. |
| `dashboard/backend/tests` (83 of 249 files reference the engine) | internals | **Must change**: port them as facade acceptance tests. Assertions about fill semantics change on purpose. |

**Blast radius outside the engine package:** 3 production modules must change (the leaderboard loop, baseline_generator cost wiring, the custom-algo script); 2 are retired (`domain/trading/execution.py`, `portfolio.py`, about 1.1k lines removed); about 8 get thin edits, totalling low hundreds of lines; plus one dev tool, dependency files and the 83 test files. **Confirmed:** `/api/v1` and `/api/v2` routers and models, DB tables and both twins, leaderboard service and H6, frontend, LLM execution and billing, and market-data providers need no rewrite. The one place the boundary could break: if the spike shows in-process sessions cannot fit (R08), protocol sessions would move to a subprocess pool. That would materially change `domain/runs/service.py` caps and v2 capacity, and it is a product decision, not a refactor.

---

## 6. Naive semantics that will change on purpose

| Today | New | Effect on results |
|---|---|---|
| Three fill models. CN and the leaderboard fill at the same bar's close (look-ahead); US fills at the next 5m open; the US 16:00 bucket fills at the same instant | One model, next bar open, everywhere. The last bucket fills at the next session's open | CN and leaderboard returns lose a free look-ahead edge; overnight gap exposure appears. |
| US: zero commission and zero slippage (`profiles.py:198-218`) | Dated commission plus sell-side regulatory fee, plus per-order half-spread and participation slippage | US curves fall; high-turnover LLMs fall most. Publish net (closes the #461 decision). |
| No volume limit | Participation cap with the remainder carried | Large orders on thin names fill over several bars. |
| Orders executed in the order the LLM listed them | Sells first with cash re-projection (ties broken by symbol, alphabetically) | Fewer spurious `insufficient_cash` rejections. |
| `entry_price` overwritten by the last buy | FIFO weighted average cost | `pnl_pct` shown to the LLM and to agents changes, so decisions change. |
| Raw prices, no ledger events (US); CN refused or charted as a loss | Split/bonus scaling and ex-date cash dividends in both markets | Removes fake crashes; credits yield. |
| 80% coverage filter drops decision bars; orders on unpriced symbols dropped silently | Per-symbol tradability flags; explicit rejection reasons | More decision steps; visible rejections. |
| Unpriced position left out of equity; first equity point taken after fees | Last-price mark with a staleness flag; first point = initial capital | No spurious equity dips. |
| Sharpe = per-bar returns × √(252·6.5); agent on a 5m grid, baselines on an hourly grid; CN headline in USD | Daily grid, risk-free rate, sample std, observed bars per day, the same grid for agent and baselines; CN headline in CNY with FX shown separately | Sharpe becomes comparable; CN figures no longer include CNY/USD moves. |
| "DJIA" = equal weight of today's 30 members, priced at the first bar close, with no costs | Label it "Dow 30 equal-weight (current constituents)", add a real index ETF series on the same tape, and run buy-and-hold over the agent's universe from the agent's first fill time with the same costs | Baselines also pay costs; the comparison becomes like for like. |
| Cold-start indicators made up (RSI 50, MACD 0; `features.py:154-170`) | NaN or null until warm | Agents see null instead of neutral-looking values. |
| A-share limit checked only on the day's last bar against the official close (`market_rules.py:204-223`) | Per-board bands against the prior close, applied at every decision | Fewer impossible fills. |
| `stop_loss_price`/`take_profit_price` accepted and ignored | Enforced as engine stop orders (gap opens fill at the open, otherwise at the trigger) | Bundled into the same cutover so there is one discontinuity, not two. |

**How to communicate it:**

1. Stamp `agent_runs.metadata.engine = {name: "akquant", version, rule_version, atl_sim: "2.0.0", fill_model: "next_open", cost_profile: "us_equity_v1"|"cn_ashare_v1"}` and add it to `RunMetadata`.
2. Add `engine.atl_sim` to the `_find_cached_run` key (`leaderboard/service.py:1441`), so rows from the old and new engines can never rank together (C21/R29).
3. Run `POST /api/v1/leaderboard/refresh?force=true` at cutover. **Cost flag:** re-running 7 LLM entries × 3 replicates is about 21 billable backtests, one-off.
4. Label rows written before cutover as "Engine v1 (legacy fills)" and exclude them from ranked boards.
5. Publish the Phase 2 differential report as the methodology note: per market, how many percentage points each semantic change moved results.
6. Regenerate the seed `backtest.db` and update `defaults.json`, which is a CLAUDE.md gotcha.

---

## 7. Runner-up: NautilusTrader, and when to choose it

Choose **NautilusTrader 2.0 final** (not an rc) instead only if all of these hold:
- AKQuant fails the spike on correctness or maintainer responsiveness;
- ATL accepts moving protocol and v2 sessions out of the web process into a subprocess pool. Concurrent engines in one process are unsupported (`architecture.md:706-722`), so capacity would drop from 100 to about 5-10 on 2 GB. That changes the scope of a contract;
- ATL is willing to own **all** A-share rules, a corporate-action layer (a split-adjusted share space plus a `SimulationModule` for dividends; the verifier notes it distorts tick-based mechanics), a next-open emulation that has never been proven (synthetic open ticks plus latency, which conflicts with recycling sell proceeds within the same timestamp **(verifier)**), and an ATL-side ledger for Live-board restore;
- ATL wraps every hook in `try/except BaseException` with `shutdown_on_error=True`, because v2 swallows exceptions raised in `on_bar`.

What ATL gets in return: the most rigorous event ordering on the list, a much larger user base, and company backing. If AKQuant fails **and** R08 stands, do not choose Nautilus. Build a minimal ATL-owned core behind the same `SimulatorPort`, to the spec in section 6 and ml4t's semantic matrix. The spike's golden suite then becomes that core's acceptance test. LEAN stays a design reference for US corporate actions: it is one process per engine, and dividends rounded to 2 decimals on the factor-file path.

---

## 8. Phased adoption plan

**Phase 0: Spike (2 weeks, throwaway branch, never merged).** Evidence: a spike report with measured numbers and pass/fail against these kill criteria.
1. Memory. One child running 30 symbols × 1 month of 5m bars adds at most about 200 MB RSS. Five children plus the web process peak below about 1.8 GB. 100 sessions parked in-process add at most about 10 MB each beyond data. *Fail ⇒ stop, or take the R08 product decision.*
2. Thread bridge. `on_bar` blocks for at least 60 s while uvicorn threads stay responsive; cancel unwinds in under 1 s; SIGTERM becomes `SystemExit` and propagates; a panic is caught as `BaseException`.
3. Corporate-action goldens, hand-computed:
   - WMT 3:1 split (2024-02-26) plus a dividend;
   - an A-share 10送3转2派1 event;
   - a split day where the held symbol has no first bar (the stale-mark bug).
4. Cross-symbol decision at T where one symbol has no bar at T (the same-slice rejection bug): measure how often it happens on SIP 5m bars for DJIA_30 and for a stock pool.
5. Defaults. A run built only from explicit configuration actually charges fees and caps volume; the startup assertion fails when a parameter is left out.
6. Reproducibility. The rule-based agent produces byte-identical results across two processes.
7. Live board. Snapshot across a session boundary, including T+1 lots, restores into a fresh engine with identical continuation; record snapshot size growth.
8. Fill goldens. CN lunch break, 15:00, US 15:55 bar → next session's open, a $0 capital run.
9. Upstream test. File the verifier-found bugs (stale split mark, same-slice rejection, the 0.25 cap being ignored, zero-fee defaults, matcher support for `can_buy`/`can_sell`) and record turnaround. *More than 2 of them unaddressed with no maintainer response in 2 weeks ⇒ treat as a maintenance kill signal.*

**Phase 1: Port and conformance suite.** Build `SimulatorPort`, the AKQuant adapter, the market packs, both drivers, the writer and the compatibility facades. Port the 17 ml4t scenarios (MIT) plus ATL goldens for A-share T+1, lots, fees, limits and corporate actions; US splits and dividends; session ends; missing bars; cancel and SIGTERM. Port the 83 test files as facade acceptance tests. Evidence: a green suite in ATL CI against the pinned wheel; the boundary test proves `akquant` is imported by exactly one module; every metadata allow-list key is emitted.

**Phase 2: Differential harness against the current engine.** This is not a parity test; every difference must be explained.
- Use identical frames from `market_data_store`.
- Use (a) deterministic agents (rule-based, buy-and-hold, equal-weight) and (b) open-loop replay of recorded action sequences: external-session `backtest_decisions`, plus decisions recorded during this phase, because pipeline runs write no decision rows.
- The new engine runs with **ablation toggles**: `CurrentClose` fill mode, zero fees, no corporate actions, no volume cap. Turning them off one at a time builds an attribution waterfall in which each step equals one named row of section 6.
- Each residual is classified as a **fix** (a defect in the old engine, e.g. average cost) or a **regression** (a defect in the new one, which must be fixed before cutover).
- Exit criterion: zero unattributed differences above tolerance across at least 10 windows per market. Evidence: a per-window differential report with step-level diffs and the waterfall.

**Phase 3: Shadow under production shape.** Run on a staging service matching Render Standard 1 CPU / 2 GB: 5 concurrent children plus protocol load from the updated `stress_serve.py`, plus a small number of real LLM runs on platform credits. Evidence: peak RSS, p95 step latency, no stranded reservations, and holds recorded by kind.

**Phase 4: Cutover.** One PR, opened as a draft with "DO NOT MERGE until Phase 3 evidence is attached" in the first line of the body. It stamps the engine version, extends the cache key, migrates Live snapshots, regenerates the seed DB, and is followed by a forced board refresh. Evidence: boot log lines, every ranked row carrying `engine.atl_sim=2.0.0`, the refreshed board, the published methodology note.

**Phase 5: Decommission.** Delete `domain/trading/{execution,portfolio}.py` and the old loops. Close #346, #461 and #597 with evidence, close #598 now, and close #128 when `vnpy_simulation` is dropped.

---

## 9. Open questions only a hands-on spike can answer

1. Actual RSS per child and per in-process session, and import time, with the wheel's declared plotly and polars dependencies installed or excluded.
2. GIL behaviour with 100 parked worker threads next to uvicorn; Rust compute holds the GIL between callbacks.
3. How often an orderable symbol has no 5m bar at a decision timestamp on SIP (DJIA_30 vs stock pools), and whether upstream fixes the same-slice rejection.
4. Can fee and stamp rates change in the middle of a run, and are per-order commission overrides enough for dated schedules and FINRA TAF?
5. Whether upstream will make the default matcher honour `can_buy`/`can_sell`, and how much the volume = 0 approximation distorts CN results in the meantime.
6. Whether iFinD can provide CN source bars finer than 60m, so next-open fills are not one hour late.
7. Whether snapshot bytes from msgpack survive an AKQuant upgrade, and how fast they grow with trade history.
8. Whether `reject_reason` strings are stable and complete enough to map to ATL's vocabulary.
9. Cross-process determinism, given `HashMap` iteration order.
10. Whether `exchange_calendars` fits ATL's pins and agrees with `sessions.py` on 2026 half days and the XSHG lunch break.
11. Engine overhead per decision step for 30 symbols × 5m × 60 days.
12. How fast and how good the maintainer's response is to the filed issues: the real test of the bus-factor risk.

---

**User-facing docs to update (logged only, per standing rule; not edited):**
- `docs/minute-data-hourly-trading.md` (whole page)
- `docs/local-run.md:208-210`
- `docs/indicator-lookahead-regression.md` (only if NaN warm-up replaces the fallback table)
- `docs/api/agent-environment-protocol-v1.md` (fills and constraints wording)
- `docs/api/python-sdk-quickstart.md`
- `docs/source/lab/{external_agents,architecture,getting_started,key_features,operating_modes}.rst`
- `docs/architecture/dashboard-target-structure.md:48,159-160,202-204`
- `docs/architecture/discord-to-backtest.md:32`
- `docs/integrations/vnpy-cta.md:14,139-177` and `vnpy-simulation.md`
- `docs/strategy-stock-pools.zh-CN.md:92-100`
- `docs/superpowers/notes/2026-09-10-user-facing-docs-backlog.md` §10
- `README.md` structure diagram
- `docs/Agentic Trading Lab Manual.pdf` (unread; assume stale)
- The repo `CLAUDE.md` engine lore (developer-facing, part of the context lake)
