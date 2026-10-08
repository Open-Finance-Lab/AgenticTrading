# Backtest decision tape — design

Date: 2026-10-08 · Status: approved in chat, spec under review
Phase 1, item 3 of the backtest-engine rewrite (see the `backtest-engine-rewrite-mandate`
project memory). Builds on the agent reasoning trace layer (#622,
`docs/superpowers/specs/2026-10-07-agent-reasoning-trace-design.md`).

## Why

The engine decision (AKQuant vs NautilusTrader vs an ATL-owned core) has to be
scored on accuracy with the LLM taken out of the loop: feed one fixed order
sequence through both engines and every difference is the engine's. Today a
pipeline-runtime backtest records no per-bar decisions anywhere —
`backtest_decisions` rows are written for the AI Hedge Fund runtime only
(`engine.py` `insert_decisions` guard), and the #622 trace for a My Agents run
holds only `run_started` and `data_retrieval` events.

The tape is that order sequence. It is also the reproducibility tool: a recorded
run can be replayed exactly, which a fresh run (one draw of a non-deterministic
model) never can.

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
   orders, truncates over-long batches, and rewrites every model SELL to "sell
   all sellable shares". None of this is recorded. The gate is a redesign
   target, so the tape keeps **both** the raw model orders and the post-gate
   actions — their difference is the gate's behaviour.
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

Side findings for the engine decision (recorded here, not fixed): the default
US run charges no fees, no slippage and no volume cap; `validator.py` is not on
this path; a model's `stop_loss_price`, `take_profit_price` and SELL
`position_size` are discarded.

## Approach

Per-bar, synchronous, best-effort emission (approach A). Chosen over a buffered
end-of-run batch (loses the tape of a killed run — the run you most want to
debug — and needs a new batch method on both store twins) and a background
writer thread (thread lifecycle and flush-on-exit in a subprocess that can be
killed; not worth it).

Cost: the trace id is resolved once and cached, so each bar is two
`append_event` calls — two pooled Neon checkouts. That is well under 1% of an
LLM bar and acceptable for a rule-based run of ~35 bars.

## Components

### `domain/backtesting/decision_tape.py` (new)

Pure payload builders plus one recorder. No engine import; the engine imports it.

- `TAPE_VERSION = 1`, carried in every payload.
- `driver_for(...)` → `"llm"` | `"llm_fallback_rule_based"` | `"rule_based"`.
- `normalize_intent(raw_output)` → the raw model orders through a fixed
  allow-list of fields (`symbol`, `action`/`side`, `position_size`/`qty`,
  `size_pct`, `confidence`, `reasoning` ≤500 chars, `stop_loss_price`,
  `take_profit_price`), read from `actions`, `orders` and `risk_actions`. The
  allow-list is what keeps arbitrary model keys away from the sensitive-key
  check. Output that is not a parseable order object is recorded as
  `{"unparsed": true, "excerpt": <≤1000 chars>}`.
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

Add best-effort `record_tape_decision(trace_id, ...)` /
`record_tape_execution(trace_id, ...)` that take the cached `trace_id` and a
prebuilt payload, keep the existing event types and the v2 id/key shapes, and
never raise. The raising v2 functions are left as they are.

Ids and keys, matching `/api/v2`: `step_id = f"step_{run_id}_{i}"`,
`decision_id = f"dec_{run_id}_{i}"`, idempotency keys
`decision:{run_id}:{step_id}:tape` / `execution:{run_id}:{step_id}:tape`.

Add `load_decision_tape(run_id)` → the ordered list of
`{bar_index, decision, execution}` pairs read back from the trace — the reader
the replay harness will use.

### Engine (`HourlyBacktester.run_agent_backtest`)

- Construct the recorder only when `runtime_type == PIPELINE_RUNTIME_TYPE`
  and `live_run_id` is set.
- Before the decision: snapshot `last_pipeline_step_outputs` identity,
  `llm_decisions`, `len(order_events)` and the `repeat_count` values.
- After `execute_actions` (~`engine.py:2275`, before the equity update): call
  `recorder.record_bar(...)`. The loop body has no early `continue`, so every
  bar produces exactly one pair.
- On completion record the run's outcome on the trace (`complete_trace` with a
  small summary: bars, fills, rejections, tape write failures). On an
  exception, `fail_trace`. Both best-effort, wired where `run_agent_backtest`
  is called in `scripts/backtest_hourly_agent.py` so a `SystemExit` from the
  cancel signal is covered.

### Parent route (`api/routers/backtests.py` `run_backtest_background`)

In the existing `finally`, if the run's trace is still `running` (child killed
by the timeout or SIGKILL before its own cleanup), `fail_trace` it with
`run_killed`. Best-effort; checked by status first so a completed trace is
never marked failed.

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
- **Replay round-trip (the proof the tape is replay-grade):** run a rule-based
  backtest recording a tape, then run it again with the decision function
  replaced by one that returns the tape's actions for bar *i*. Trades and the
  equity curve must be identical.
- Parent: a killed child leaves the trace `failed`; a completed one is untouched.
- The conformance suite (#623) and the existing trace tests stay green.

## Open questions

None blocking. Retention and admin-UI pagination for long windows remain #622's
open items; a default week is ~70 events per run.
