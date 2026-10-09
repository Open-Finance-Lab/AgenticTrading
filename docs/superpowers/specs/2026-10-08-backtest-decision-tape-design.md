# Backtest decision tape — design

Date: 2026-10-08 · Status: implemented; amended 2026-10-08 after review
(intent is the replay unit, realism is scored by the conformance suite, the
parent owns the trace's terminal label).
Phase 1, item 3 of the backtest-engine rewrite (see the `backtest-engine-rewrite-mandate`
project memory). Builds on the agent reasoning trace layer (#622,
`docs/superpowers/specs/2026-10-07-agent-reasoning-trace-design.md`).

## Why

The engine decision (AKQuant vs NautilusTrader vs an ATL-owned core) has to be
scored with the LLM taken out of the loop: feed one fixed decision sequence
through each engine and compare. Today a pipeline-runtime backtest records no
per-bar decisions anywhere — `backtest_decisions` rows are written for the AI
Hedge Fund runtime only (`engine.py` `insert_decisions` guard), and the #622
trace for a My Agents run holds only `run_started` and `data_retrieval` events.

The tape is that sequence. It is also the reproducibility tool: a recorded run
can be replayed exactly, which a fresh run (one draw of a non-deterministic
model) never can.

**What the engine-independent part of a bar is.** The only thing a bar carries
that no engine decided is what the model asked for (`intent`) and what the
world looked like when it asked (`state`). The post-gate `actions` are the
*current* engine's answer: a SELL's size is its sellable position
(`portfolio_manager.py:922-929`), a BUY was admitted against its cash. Replaying
`actions` through a second engine therefore isolates that engine only until the
first fill that differs — one share less on bar *k* turns bar *k+1*'s recorded
SELL into an oversell the candidate clips, and bar *k+2*'s BUY into one it
rejects for cash, so from *k* on the curve gap is path-coupled, not "the
engine's". The tape records both, and **`intent` + `state` is the replay unit**:
a candidate engine runs the model's orders through its *own* gate and fill
model. `actions` stay on the tape as the current engine's recorded answer, which
is what the same-engine reproduction check compares against.

**What the tape is not.** It is not a realism target. The engine it records is
frictionless, latency-free and runs on unadjusted bars (see "Market realism"
below), so two engines agreeing bar-for-bar on a tape is agreement with each
other, not with the market. Realism is scored by the conformance suite (#623),
whose cost, liquidity, timing and corporate-action groups the current engine
already fails on record; the tape adds what the suite's synthetic cases lack —
model-shaped order sequences on real bars.

Seeing the decisions in the admin trace timeline is a by-product, not the goal.

## Scope

- In: My Agents backtests on the pipeline runtime — `POST /backtest/run` →
  `scripts/backtest_hourly_agent.py` → `HourlyBacktester.run_agent_backtest`.
  Both `decision_source="llm"` and rule-based runs.
- Out: AI Hedge Fund runtime (already writes `backtest_decisions`), step
  sessions (`/api/v1`, `/api/v2`), the leaderboard loop, A-share specifics,
  trace retention (#622 defers it to its L7), admin UI pagination.
- Shared code stays untouched: `PortfolioManager` and `domain/trading/execution.py`
  are also used by the step sessions. The tape reads their state; it does not
  change them.

## What the engine actually does (findings that shape the payload)

1. The decision handed to `execute_actions` is already **post-gate**.
   `make_trading_decision_with_llm` silently drops low-confidence (<0.3),
   out-of-universe, over-`MAX_ORDER_SHARES`, unaffordable-buy and unheld-sell
   orders, truncates over-long batches, rewrites every model SELL to "sell
   all sellable shares", and discards `stop_loss_price`, `take_profit_price`
   and a SELL's `position_size` (`portfolio_manager.py:919-939`) — the engine
   cannot express a partial exit or a protective stop. None of this is
   recorded. The gate is a redesign target, so the tape keeps **both** the raw
   model orders and the post-gate actions, and each bar carries
   `gate_rewrote` (the two disagree on a `(symbol, side, size)` triple) so the
   run's rewrite rate is a number in the trace summary (`gate_rewrites`), not
   a by-product someone has to diff for.
2. The raw model output of a pipeline run is `manager.last_pipeline_step_outputs[-1]`.
   It is reassigned (a new list) on every pipeline call, and left stale when a
   bar does not call the pipeline. Freshness is therefore detected by identity
   against a snapshot taken before the decision.
3. Which path drove the bar is visible only through `manager.llm_decisions`
   (incremented at the success exit of the LLM path). A bar whose model output
   was unusable falls back to the rule-based decision without any other mark.
4. `order_events` collapses a repeated pure rejection (same symbol, side,
   reason, trading day) into the first record's `repeat_count`. Slicing
   `order_events[before:]` misses those repeats; the tape diffs `repeat_count`
   values as well as new records.
5. Fills: `manager.trades[before:]` after `execute_actions`. In intraday mode
   the fill instant (`fill.filled_at`) differs from the decision bar
   (`timestamp`); the tape records both.
6. Trace-layer constraints: payloads are capped at 64 KB and any dict **key**
   matching `api_key|authorization|cookie|password|secret|token` raises
   (`domain/traces/common.py`). `record_decision_event` /
   `record_execution_event` raise on failure. Nothing finalizes a dashboard
   backtest trace today, so it stays `running` and the admin UI polls it
   forever.

## Market realism — what the recorded engine does not model

Verified at source on 2026-10-08. None of it is fixed by this feature, and
the sequencing below says why.

| Idealization | Where | Effect on a recorded run |
|---|---|---|
| **Zero friction.** The US `MarketProfile` (`infrastructure/market_data/profiles.py`) sets no `transaction_cost_profile`; `calculate_transaction_costs` (`domain/trading/execution.py:296-306`) then short-circuits to zero slippage, zero commission, zero fees. No spread, no impact, no participation cap beyond `MAX_ORDER_SHARES`. | engine, PM | Every fill is at the printed price for any size. High-turnover strategies are flattered most. |
| **Zero latency.** The hourly decision bar closes and the fill is the open of the 5m bar starting at that same instant (`engine.py:2240-2262`; US `source_timeframe="5m"`). A live LLM step takes 30–180 s (`LLM_PROVIDER_READ_TIMEOUT_SECONDS`). | engine | The real fill is minutes later at a different price. |
| **Unadjusted bars.** `StockBarsRequest` is built with no `adjustment` (`alpaca_bars.py:732-737`; Alpaca's default is `raw`). `trim_at_unadjusted_gaps` covers the warm-up pad only; the in-window corporate-action check is A-share only. | market data | A 2-for-1 split inside the window prints as −50% on one bar; the model reacts to a move that did not happen, and both the agent curve and the index baseline carry it. CA01/CA02 in `known_failures.json` already record wrong avg_cost, equity and cash. |
| **The gate rewrites the model.** Finding 1 above. | PM | Published LLM curves measure the gate as much as the model. |
| `validator.py` is not on this path. | — | Noted for the rewrite; no tape consequence. |

**Where realism is scored: the conformance suite, not this engine.** The
suite (#623, `tests/conformance/`) already carries group F (commission, SEC
§31, FINRA TAF, bps slippage), L (participation cap), T (fill timing) and CA
(splits, dividends), and `known_failures.json` records the current engine
failing F01–F05, L01, T02–T05 and CA01–CA03. That is the realism target for
the engine decision, independent of the engine being replaced. The rule this
spec adopts: **do not retrofit fees, slippage or latency into the current
engine** — it is the one under replacement, and realism gets built once, in
whichever engine wins. The one group the suite does not yet have is decision
latency (a T-case that fills at the open of bar *N+k*, *k* from the measured
step time in the reservations data); it belongs in the Phase 1 exit criteria,
not in this PR.

**The one realism item that cannot wait: adjusted bars.** Unadjusted prices
corrupt the tape itself — a model reacting to a phantom split records garbage
`intent`, which neither goal can use. The US path should request
`Adjustment.ALL` before tapes meant to be kept are recorded. That is a
separate PR with its own guard rails, because it changes the price series:
the adjustment must be stamped into `agent_runs.metadata` beside
`market_data_feed` (a new `market_data_adjustment` key written by
`feed_provenance`), `diff_backtest_runs.py` must refuse a pair that differs on
it, and `_find_cached_run` must not rank an adjusted row against raw ones —
the same comparability rule the feed already has (CLAUDE.md,
`ALPACA_DATA_FEED`). The tape payload carries no adjustment field; the run's
metadata is where that fact lives.

## Approach

Best-effort emission in **batches of `TAPE_FLUSH_BARS` (25) bars**, one store
transaction per batch (`append_events` on both trace-store twins), flushed at
the end of the agent loop and by `trace_lifecycle` before the trace closes on
either outcome. The first draft wrote each bar synchronously (two
`append_event` calls, ~10 Postgres statements including a `SELECT … FOR
UPDATE`); review measured that against a long rule-based window on Neon and it
could add minutes of wall clock inside the decision loop — enough to push a run
into its subprocess timeout, i.e. the observational tape changing the run's
outcome. Batching cuts that ~25×.

What it costs: a run killed by SIGTERM (cancel, or the timeout's grace before
SIGKILL) loses its unflushed buffer — at most 24 bars. A *failed* run keeps
them, because `trace_lifecycle` flushes before writing `run_failed`. A
background writer thread was rejected again (thread lifecycle and
flush-on-exit in a subprocess that can be killed).

**Storage.** The trace store moved from `CONTENT_DATABASE_URL` to
`AGENT_RUNS_DATABASE_URL` in this PR. #622 put traces on the content database
because its initial workload was small; a decision/execution pair per bar of
every dashboard backtest is run data and the fastest-growing run data there
is, and `ATL-runs-main` exists to keep exactly that growth away from the
auth-critical users/content database. Traces written to the content database
between #622's merge (2026-10-07) and this PR's deploy stay there, invisible
to the admin timeline; they are not copied.

## Components

### `domain/backtesting/decision_tape.py` (new)

Pure payload builders plus one recorder. No engine import; the engine imports it.
**The trace service is imported lazily**, inside the recorder and the lifecycle
wrapper, never at module top: `domain/traces/service.py` constructs
`trace_store` on import, which runs schema DDL against `DB_PATH` (or dials
Neon), and the engine (`engine.py:523`) and the provider wrapper
(`provider.py:173`) both import it lazily for exactly that reason. A top-level
import here would re-arm the "ad-hoc imports mutate the seed DB" gotcha
through every process that imports the engine (`diff_backtest_runs.py`, the
leaderboard refresh, a `python -c` probe).

- `TAPE_VERSION = 1`, carried in every payload.
- `driver_for(...)` → `"llm"` | `"llm_fallback"` | `"rule_based"`.
- `gate_rewrote(intent, actions)` → `True` when `intent` holds orders and the
  sorted `(symbol, side, size)` triples of intent and actions differ (a
  dropped order, an added one, a resized one, a SELL widened to the whole
  position). `False` when intent is `null` or unparsed — a rule-based bar has
  no gate to rewrite — and on every `llm_fallback` bar, whose actions are
  rule-based substitutes rather than a rewrite. Carried per bar as
  `gate_rewrote`; counted per run as `gate_rewrites` (LLM-driven bars only;
  fallbacks are counted by `driver`) in the recorder summary.
- `normalize_intent(raw_output)` → the raw model orders through a fixed
  allow-list of fields (`symbol`, `action`/`side`, `position_size`/`qty`,
  `size_pct`, `confidence`, `reasoning` ≤500 chars, `stop_loss_price`,
  `take_profit_price`), read from `actions`, `orders` and `risk_actions`. The
  allow-list is what keeps arbitrary model keys away from the sensitive-key
  check. Output that is not a parseable order object is recorded as
  `{"unparsed": true, "excerpt": <≤1000 chars>}`. Orders are recorded so
  `orders_for_replay(intent)` hands the gate exactly what the model sent: an
  absent field stays absent and a `null` one stays `null` (the gate reads each
  with a default, and `confidence: null` raises where an absent one trades); a
  non-finite number or an object/array value is recorded readably with its
  exact JSON text under `raw[field]`; a non-object entry as `{"raw_entry": …}`.
  (Amended 2026-10-08 after the whole-branch review.)
- `build_decision_payload(...)` → superset of the v2 `decision_recorded`
  payload: `actions` (post-gate), `reasoning_summaries`, `accepted`, plus
  `tape_version`, `bar_index`, `decision_at`, `driver`, `intent`
  (`null` when no fresh raw output exists, e.g. rule-based or the
  single-prompt path), and `state` = pre-decision `cash`, `equity`, and
  `positions` as `[symbol, shares]` pairs (a list, not a dict, so a ticker can
  never become a key).
- `build_execution_payload(...)` → superset of the v2 `execution_result`
  payload: `accepted`, `fills`, `executed`, `rejected`, `validation`,
  `run_status="running"`, plus `tape_version`, `bar_index` and `fill_plan`
  (`bar`, `price_field`, `filled_at`). `rejected` is built from new
  `order_events` records with a non-filled status plus records whose
  `repeat_count` grew during this bar.
- Size bound: if a payload's JSON exceeds the store cap, reasoning text is
  dropped first, then intent reasoning; the orders themselves are never
  dropped. A payload still over the cap is skipped and counted.
- `DecisionTapeRecorder(run_id)`: resolves the trace once, then
  `record_bar(...)` appends both events. Every failure is swallowed; the first
  one per run is printed, later ones are counted. A recorder whose trace
  lookup returned nothing becomes a no-op for the rest of the run.

### `domain/traces/service.py`

`record_tape_bar(trace_id, ...)` writes the pair and raises;
`DecisionTapeRecorder` is the one swallow-and-count layer.
`finish_trace_best_effort` and `fail_trace_if_running` never raise. The pair
keeps the existing event types and the v2 id/key shapes; the raising v2
functions are left as they are.

Ids and keys, matching `/api/v2`: `step_id = f"step_{run_id}_{i}"`,
`decision_id = f"dec_{run_id}_{i}"`, idempotency keys
`decision:{run_id}:{step_id}:tape` / `execution:{run_id}:{step_id}:tape`.

Add `load_decision_tape(run_id)` → the ordered list of
`{bar_index, decision, execution, complete}` pairs read back from the trace — the reader
the replay harness will use. `complete` is false on a bar missing a half (its
`execution_result` append failed, or SIGTERM landed between the two appends),
so missing fills are never read as "nothing filled". Each `decision` carries the whole payload, so a
harness reads `intent` (the replay unit) and `state` (to re-seed or to diff
one-bar deltas) as readily as `actions`; a loader that surfaced only `actions`
would quietly make the current engine's answer the replay unit again.

### Engine (`HourlyBacktester.run_agent_backtest`)

- Construct the recorder only when `runtime_type == PIPELINE_RUNTIME_TYPE`
  and `live_run_id` is set.
- Before the decision: snapshot `last_pipeline_step_outputs` identity,
  `llm_decisions`, `len(order_events)` and the `repeat_count` values.
- After `execute_actions` (~`engine.py:2275`, before the equity update): call
  `recorder.record_bar(...)`. The loop body has no early `continue`, so every
  bar produces exactly one pair.
- On completion record the run's outcome on the trace (`complete_trace` with a
  small summary: bars, fills, rejections, gate rewrites, tape write failures).
  On an exception, `fail_trace(run_failed)`. Both best-effort. **The wrapper
  spans the constructor, not just `run_agent_backtest`**: the trace is created
  in `HourlyBacktester.__init__` (`engine.py:522-535`), so a failure in
  `load_data()` or `calculate_indicators()` — a SIP refusal, no bars,
  `AlpacaFeedConfigError` — would otherwise exit the child with the trace
  `running` and be stamped as a kill by the parent. In
  `scripts/backtest_hourly_agent.py` the wrapper opens before the
  `HourlyBacktester(...)` call and closes after `run_agent_backtest`.
- **On `SystemExit` the child writes nothing and re-raises.** That exit is the
  dashboard's SIGTERM (`_exit_on_sigterm`), and the parent sends it from two
  different arms — the user's cancel *and* the timeout's grace period before
  SIGKILL (`_signal_backtest_process`, `backtests.py:2833`). The child cannot
  tell them apart; the parent can, so the parent owns that label.

### Parent route (`api/routers/backtests.py` `run_backtest_background`)

The parent labels every trace the child left `running`, by the arm it is in —
each call is `fail_trace_if_running`, a status-checked no-op on a trace the
child already closed, so a completed trace is never re-marked:

| Arm | `error_code` | Meaning |
|---|---|---|
| `returncode == 0` success branch | `trace_close_failed` | The run finished; the child's own `complete_trace` did not land (a Neon blip). Distinct from a kill on purpose. |
| `returncode > 0` branch | `run_failed` | The child died on an exception it could not record (or recorded, in which case this is a no-op). |
| `returncode < 0` branch | `run_killed` | A signal death the parent did not send (its cancel and timeout are claimed by their own arms) — a cgroup OOM SIGKILL is the likely sender. |
| `_BacktestCancelled` arm | `run_cancelled` | The user's cancel. |
| `TimeoutExpired` arm | `run_timed_out` | Matches `/backtest/status`'s `timed_out`. |
| outer `finally` | `run_killed` | Reached none of the above: a parent exception. |

A tape consumer reads the terminal code as its completeness marker, which is
why the five codes are kept distinct rather than collapsed to "not completed".
The label is written in an inner `finally` that closes the worker's `finally`,
so it still lands after slot/reservation cleanup when one of those raises.

On `SystemExit` "writes nothing" includes an exception a cleanup `finally`
raised while that exit was unwinding (a `SystemExit` on its `__context__`
chain): `finalize_run` failing during the SIGTERM unwind must not stamp
`run_failed` over the parent's cancel/timeout label.

## Error handling

The tape must never change a backtest's outcome. Every tape write is wrapped;
trace finalization is wrapped; the recorder never raises into the loop. A run
with a broken trace store behaves exactly like today plus one log line.

## Testing

- Unit (`test_decision_tape.py`): driver classification; intent allow-list
  (unknown and sensitive keys dropped, `orders`/`risk_actions` shapes,
  unparseable output, stale-output detection); rejection diff including a
  collapsed repeat; payload size bounding; ticker-as-key safety.
- Recorder: a broken store never raises; trace id resolved once; no-op when no
  trace.
- Engine integration on the SQLite trace store with a synthetic provider:
  one `decision_recorded` + `execution_result` pair per bar in order; the
  trace ends `completed`; a run that raises ends `failed`.
- **Reproduction (same engine, recorded `actions`):** run a rule-based
  backtest recording a tape, then run it again with the decision function
  replaced by one that returns the tape's actions for bar *i*. Trades and the
  equity curve must be identical. This proves the tape survives
  `_pick` + JSON and that the engine is deterministic on a fixed sequence. It
  is **not** the replay-grade proof — the same engine on the same bars cannot
  show what a different engine needs.
- **Intent replay (the replay-grade proof):** run an LLM-pipeline backtest
  with a scripted model, then run it again with a model stub that answers
  each bar with the tape's `intent.orders` verbatim. Trades and the curve must
  be identical, and the second tape's `intent` must equal the first's. This is
  the path a candidate engine takes: model orders through *a* gate.
- **Divergence is localized, not absorbed:** replay the recorded `actions`
  into an engine given a one-tick slippage profile (a `TransactionCostProfile`
  on the backtester's `MarketProfile`, the mechanism A-share already uses).
  The curves must differ, the first divergent bar must be the first bar with
  a fill, and the replay's fills must carry a non-zero `slippage_amount`. The
  same shape `diff_backtest_runs.py` reports. A tape on which a perturbed
  engine "agreed" would be one that recorded nothing the fill touched.
- **`PortfolioManager` reassigns `last_pipeline_step_outputs` per pipeline
  call** — pinned directly, on the manager, not only through the tape's
  intent assertions: the tape's freshness check is list identity, and a later
  `.clear()`/`.extend()` refactor in the manager would silently null `intent`
  on every bar while every tape unit test stayed green. The failing test must
  be the one that names the invariant.
- Parent: each arm (including a signal death the parent did not send, and a
  cleanup statement that raises) writes its own label; a completed trace is
  untouched by all of them.
- The conformance suite (#623) and the existing trace tests stay green.

## Open questions

None blocking. Retention and admin-UI pagination for long windows remain #622's
open items; a default week is ~70 events per run. `AGENT_RUNS_DATABASE_URL`
must be set on Render before this deploys, or traces fall back to ephemeral
SQLite (logged at boot as `trace_store backend: sqlite (ephemeral on Render)`).

## Follow-ups this spec hands off (not in this PR)

1. **Adjusted US bars with provenance** — `Adjustment.ALL` on the US
   `StockBarsRequest`, a `market_data_adjustment` stamp through
   `feed_provenance`, a refusal in `diff_backtest_runs.py` on a mismatched
   pair, and the cache-reuse rule. Blocks recording tapes meant to be kept.
2. **A decision-latency conformance case** (group T): fill at the open of
   bar *N+k*, *k* from the measured step time. Part of the Phase 1 exit
   criteria, so the engine candidates are scored on it.
3. **A driver field on `PortfolioManager`'s return envelope**
   (`{"actions", "driver", "raw_output"}`, ignored by the step sessions) so
   the tape stops inferring the driver from a counter delta and freshness from
   list identity. Deferred because this PR does not touch shared PM code; the
   identity invariant is pinned by a test until then.
4. **Realism scoring of the published leaderboard medians** — once a candidate
   engine passes groups F/L/T/CA, rerun the contest tapes through it and
   publish the gap. That number, not fidelity to the current engine, is what
   the rewrite is for.
5. **Pin the intraday marking rule before comparing engines.** In intraday
   mode the engine marks every source bar up to and including the current
   `fill_plan.bar` *after* executing bar *i* (the `valuation_cursor` loop), so
   a fill's slippage already shows on up to an hour of earlier curve points.
   Found while localizing the perturbed-engine test; the harness's
   `_decision_bar_at` maps a divergent curve point forward to the first
   `fill_plan.bar` at or after it for that reason. A candidate engine that
   marks causally will diverge from this one at every fill, so the comparison
   must state which rule it scores.
