# Backtest Decision Tape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every decision bar of a My Agents pipeline backtest — raw model orders (`intent`), post-gate actions, pre-decision portfolio (`state`), fill plan, fills, rejections, and whether the gate rewrote the model — as `decision_recorded` / `execution_result` event pairs in the run's #622 agent trace, so a run's **model intent** can be replayed through another engine's own gate and fill model, and its recorded actions can reproduce the run on this one.

**Architecture:** A new pure module `domain/backtesting/decision_tape.py` builds bounded, allow-listed payloads and owns a `DecisionTapeRecorder` that swallows every failure; it imports the trace service lazily (the service builds its store at import). `HourlyBacktester.run_agent_backtest` snapshots ledger state before each decision and hands the recorder one bar after `execute_actions`. `domain/traces/service.py` gains a raising `record_tape_bar`, a `load_decision_tape` reader, and two best-effort lifecycle helpers; the child script closes the trace from a `trace_lifecycle` context that spans the engine constructor through `run_agent_backtest` and writes nothing on `SystemExit`; the parent route labels any trace the child left `running` by the arm it is in (`run_cancelled`, `run_timed_out`, `run_failed`, `trace_close_failed`, `run_killed`).

**Amended 2026-10-08 after review.** Three changes against the first draft: (1) `intent` + `state` is the replay unit, `actions` the same-engine reproduction check — replaying post-gate absolute sizes through a second engine isolates it only until the first divergent fill; (2) the parent owns the terminal label, because the child's `SystemExit` is sent by both the cancel and the timeout arms; (3) realism (fees, slippage, latency, corporate actions) is **not** retrofitted here — the conformance suite (#623, groups F/L/T/CA) is the realism target and the current engine's failures on it are already on record. See the spec's "Market realism" section.

**Tech Stack:** Python 3, pandas, pytest, the existing SQLite/Postgres `TraceStore` twins.

**Spec:** `docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md`

## Global Constraints

- Scope: `runtime_type == PIPELINE_RUNTIME_TYPE` runs with a `live_run_id`; AI Hedge Fund runtime, step sessions, leaderboard: untouched.
- Do not modify `domain/backtesting/portfolio_manager.py` or `domain/trading/execution.py` (shared with the step sessions).
- The tape must never change a backtest's outcome: no exception from tape or trace code may reach the bar loop or the run.
- `TAPE_VERSION = 1` in every tape payload.
- Payload JSON ≤ 64 KiB (store cap, `domain/traces/common.py:_MAX_JSON_BYTES`); the tape bounds at `MAX_PAYLOAD_BYTES = 60 * 1024`.
- No dict **key** in a payload may match `api_key|authorization|cookie|password|secret|token` (store raises). Model output passes through a field allow-list; positions are `[symbol, shares]` pairs, never a dict.
- Ids/keys match `/api/v2`: `step_id = f"step_{run_id}_{i}"`, `decision_id = f"dec_{run_id}_{i}"`, idempotency keys `decision:{run_id}:{step_id}:tape` and `execution:{run_id}:{step_id}:tape`.
- Reasoning text ≤ 500 chars (`MAX_REASONING_CHARS`).
- Driver labels: `"llm"`, `"llm_fallback"`, `"rule_based"`.
- **`intent` + `state` is the replay unit; `actions` is the reproduction check.** Nothing in the loader or a harness may surface `actions` as the only thing to replay. A test named "replay" must replay `intent`.
- **`decision_tape.py` never imports `domain.traces.service` at module top.** That module constructs `trace_store` on import (schema DDL on `DB_PATH`, or a Neon dial); `engine.py:523` and `provider.py:173` import it lazily for that reason, and so does every function here. Tests still `monkeypatch.setattr(trace_service, ...)` on the module object — a lazy `from … import service` returns the same object.
- **The child writes no terminal label on `SystemExit`.** The parent sends SIGTERM from two arms (cancel, and the timeout's grace before SIGKILL) and is the only party that knows which; it writes one of five labels (Task 5).
- **No realism retrofit.** Do not add fees, slippage, latency or adjustment handling to the engine, PM or the Alpaca loader in this PR; those are follow-ups 1–2 in the spec. The perturbed-engine test in Task 4 injects a cost profile on the *test's* `MarketProfile` copy, never in `profiles.py`.
- Tests run from the repo root: `pytest dashboard/backend/tests/... -v`. `tests/conftest.py` already strips `CONTENT_DATABASE_URL`, so the trace store is SQLite in tests.
- Every commit ends with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; the commit snippets below carry it, copy them whole.

## Review Focus

1. **Stale pipeline output on a rule-based or fallback bar** — a bar that did not call the pipeline must record `intent: null`, not the previous bar's model orders. Pinned in Task 1 (`test_intent_is_none_when_outputs_list_is_unchanged`) and Task 4 (`test_llm_pipeline_bars_record_intent_and_fallback`).
2. **A repeated same-day rejection** collapses into `repeat_count` on the first `order_events` record; the later bar must still list it. Pinned in Task 1 (`test_rejections_since_reports_a_collapsed_repeat`) and Task 4 (`test_collapsed_repeat_rejection_appears_on_its_own_bar`).
3. **Model JSON carrying a sensitive-looking key** (`api_key`, `max_tokens`) or numpy scalars — the store would raise and the bar's events vanish. Pinned in Task 1 (`test_intent_allow_list_drops_unknown_and_sensitive_keys`, `test_jsonable_converts_numpy_and_timestamps`).
4. **Trace store down or slow mid-run** — the backtest must finish with an identical curve. Pinned in Task 4 (`test_broken_trace_writes_do_not_change_the_run`).
5. **Child killed by the timeout** — the trace must not stay `running` forever, must read `run_timed_out` (not `run_cancelled` — the child's SIGTERM exit cannot tell the two arms apart), and a completed trace must never be re-marked failed. Pinned in Task 5 (`test_parent_labels_a_timed_out_child_run_timed_out`, `test_parent_labels_a_failed_child_run_failed`, `test_parent_labels_an_unclosed_completed_run_trace_close_failed`) and Task 2 (`test_fail_trace_if_running_leaves_a_completed_trace_alone`).
6. **Replay through a second engine must stay engine-isolating** — the harness replays `intent` through the gate, and a perturbed engine's divergence is localized to the first bar a fill touched, never absorbed by a later recorded size. Pinned in Task 4 (`test_intent_replay_through_the_gate_reproduces_an_llm_run`, `test_perturbed_engine_diverges_at_the_first_filled_bar`).
7. **Failure before the loop** — a `load_data()` or `calculate_indicators()` failure happens after the trace exists (`HourlyBacktester.__init__`) and must end `run_failed`, not `run_killed`. Pinned in Task 3 (`test_lifecycle_fails_on_an_exception_raised_before_run`) and by the child-side `with` spanning the constructor (Task 5).
8. **Importing the engine must not touch a database** — `decision_tape.py`'s trace import is lazy. Pinned in Task 3 (`test_decision_tape_module_does_not_import_the_trace_service_at_top`).
9. **`PortfolioManager` must keep assigning a new outputs list per pipeline call** — the tape's freshness check is identity. Pinned on the manager itself in Task 4 (`test_portfolio_manager_reassigns_pipeline_outputs_per_call`).

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/backend/domain/backtesting/decision_tape.py` (create) | Pure payload builders (incl. `gate_rewrote`), size bounding, `DecisionTapeRecorder`, `trace_lifecycle` context manager. Lazy trace-service import. |
| `dashboard/backend/domain/traces/service.py` (modify) | `record_tape_bar`, `load_decision_tape`, `finish_trace_best_effort`, `fail_trace_if_running`. |
| `dashboard/backend/domain/backtesting/engine.py` (modify) | Snapshot before decision, record after execution, `decision_tape_summary()`. |
| `dashboard/scripts/backtest_hourly_agent.py` (modify) | `with trace_lifecycle(...)` spanning `HourlyBacktester(...)` through `run_agent_backtest`. |
| `dashboard/backend/api/routers/backtests.py` (modify) | `fail_trace_if_running(...)` with one label per arm of `run_backtest_background`. |
| `dashboard/backend/tests/domain/backtesting/test_decision_tape.py` (create) | Unit tests for builders, recorder, lifecycle, lazy-import guard. |
| `dashboard/backend/tests/test_trace_decision_tape.py` (create) | Service tests on a real SQLite `TraceStore`. |
| `dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py` (create) | Engine integration, same-engine reproduction, intent replay, perturbed-engine divergence, PM identity invariant. |
| `dashboard/backend/tests/test_backtests_router.py` (modify) | Parent label-per-arm tests. |

---

### Task 1: Pure payload builders

**Files:**
- Create: `dashboard/backend/domain/backtesting/decision_tape.py`
- Test: `dashboard/backend/tests/domain/backtesting/test_decision_tape.py`

**Interfaces:**
- Consumes: `pipeline_output_to_decision(parsed) -> Optional[dict]` from `dashboard.backend.infrastructure.llm.pipeline_runner`.
- Produces (module `dashboard.backend.domain.backtesting.decision_tape`):
  - `TAPE_VERSION: int = 1`, `MAX_REASONING_CHARS = 500`, `MAX_PAYLOAD_BYTES = 60 * 1024`
  - `DRIVER_LLM = "llm"`, `DRIVER_LLM_FALLBACK = "llm_fallback"`, `DRIVER_RULE_BASED = "rule_based"`
  - `jsonable(value) -> Any`
  - `driver_for(*, use_llm: bool, llm_decisions_before: int, llm_decisions_after: int) -> str`
  - `intent_from_outputs(outputs, *, outputs_before, decision_step_count: int) -> Optional[dict]`
  - `build_state(*, cash, equity, positions: Mapping[str, Any]) -> dict`
  - `order_events_snapshot(order_events) -> List[int]`
  - `rejections_since(order_events, snapshot: Sequence[int]) -> List[dict]`
  - `fills_since(trades, start: int) -> List[dict]`
  - `gate_rewrote(intent: Optional[dict], actions) -> bool`
  - `build_decision_payload(*, bar_index: int, decision_at, driver: str, intent: Optional[dict], state: dict, actions) -> dict` (sets `gate_rewrote` itself)
  - `build_execution_payload(*, bar_index: int, fill, fills: List[dict], rejected: List[dict]) -> dict`
  - `bound_payload(payload: dict) -> Optional[dict]`

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/backtesting/test_decision_tape.py`:

```python
"""Unit tests for the decision tape's pure payload builders (spec 2026-10-08)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dashboard.backend.domain.backtesting import decision_tape as tape


def _size(payload):
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def test_jsonable_converts_numpy_and_timestamps():
    stamp = pd.Timestamp("2026-03-02 10:30", tz="America/New_York")
    out = tape.jsonable({"a": np.int64(3), "b": np.float64(1.5), "c": stamp, "d": float("nan"), "e": (1, True)})
    assert out == {"a": 3, "b": 1.5, "c": stamp.isoformat(), "d": None, "e": [1, True]}
    json.dumps(out)  # must be serializable


def test_driver_for_distinguishes_llm_fallback_and_rule_based():
    assert tape.driver_for(use_llm=True, llm_decisions_before=2, llm_decisions_after=3) == "llm"
    assert tape.driver_for(use_llm=True, llm_decisions_before=2, llm_decisions_after=2) == "llm_fallback"
    assert tape.driver_for(use_llm=False, llm_decisions_before=0, llm_decisions_after=0) == "rule_based"


def test_intent_is_none_when_outputs_list_is_unchanged():
    outputs = [{"step": 1, "output": {"actions": [{"symbol": "AAPL", "action": "buy"}]}}]
    assert tape.intent_from_outputs(outputs, outputs_before=outputs, decision_step_count=1) is None


def test_intent_reads_final_step_orders_and_risk_actions():
    orders = [{"step": 1, "output": {"orders": [{"symbol": "AAPL", "side": "BUY", "qty": 7, "reason": "r"}]}}]
    intent = tape.intent_from_outputs(orders, outputs_before=[], decision_step_count=1)
    assert intent["orders"][0]["symbol"] == "AAPL"
    assert intent["orders"][0]["action"] == "buy"
    assert intent["orders"][0]["position_size"] == 7

    risk = [{"step": 1, "output": {"risk_actions": [{"symbol": "MSFT", "action": "stop_loss", "size_pct": 0.5}]}}]
    intent = tape.intent_from_outputs(risk, outputs_before=[], decision_step_count=1)
    assert intent["orders"][0] == {
        "symbol": "MSFT", "action": "sell", "position_size": 50,
        "confidence": 0.8, "reasoning": "stop_loss",
    }


def test_intent_is_unparsed_when_final_step_is_missing():
    # Step 2 of 2 failed to parse, so only step 1's output came back.
    partial = [{"step": 1, "output": {"facts": []}}]
    assert tape.intent_from_outputs(partial, outputs_before=[], decision_step_count=2) == {
        "unparsed": True, "completed_steps": 1,
    }
    fresh_empty = []
    assert tape.intent_from_outputs(fresh_empty, outputs_before=[], decision_step_count=1) == {
        "unparsed": True, "completed_steps": 0,
    }


def test_intent_allow_list_drops_unknown_and_sensitive_keys():
    raw = [{"step": 1, "output": {"actions": [{
        "symbol": "AAPL", "action": "buy", "position_size": 5, "confidence": 0.9,
        "reasoning": "x" * 900, "api_key": "leak", "max_tokens": 10, "nested": {"a": 1},
    }]}}]
    order = tape.intent_from_outputs(raw, outputs_before=[], decision_step_count=1)["orders"][0]
    assert set(order) == {"symbol", "action", "position_size", "confidence", "reasoning"}
    assert len(order["reasoning"]) == tape.MAX_REASONING_CHARS


def test_build_state_uses_pairs_and_skips_flat_positions():
    state = tape.build_state(cash=np.float64(100.0), equity=250.0, positions={"MSFT": 2, "AAPL": 3, "TOKEN": 0})
    assert state == {"cash": 100.0, "equity": 250.0, "positions": [["AAPL", 3], ["MSFT", 2]]}


def test_rejections_since_reports_new_and_collapsed_events():
    events = [
        {"symbol": "AAPL", "side": "BUY", "status": "rejected", "reason": "insufficient_cash", "requested_shares": 9},
        {"symbol": "MSFT", "side": "BUY", "status": "filled", "reason": "", "requested_shares": 1},
    ]
    snapshot = tape.order_events_snapshot(events)
    assert snapshot == [1, 1]
    events[0]["repeat_count"] = 2  # collapsed repeat this bar
    events.append({"symbol": "AAPL", "side": "SELL", "status": "partial", "reason": "insufficient_position"})
    events.append({"symbol": "MSFT", "side": "SELL", "status": "filled", "reason": ""})
    rejected = tape.rejections_since(events, snapshot)
    assert [(r["symbol"], r["side"], r["count"], r["collapsed"]) for r in rejected] == [
        ("AAPL", "BUY", 1, True),
        ("AAPL", "SELL", 1, False),
    ]


def test_rejections_since_reports_a_collapsed_repeat():
    events = [{"symbol": "AAPL", "side": "BUY", "status": "rejected", "reason": "insufficient_cash"}]
    snapshot = tape.order_events_snapshot(events)
    events[0]["repeat_count"] = 2
    assert tape.rejections_since(events, snapshot)[0]["reason"] == "insufficient_cash"


def test_fills_since_slices_and_allow_lists():
    trades = [
        {"timestamp": pd.Timestamp("2026-03-02 10:30"), "symbol": "AAPL", "side": "BUY", "shares": 5, "price": 10.0, "cost": 50.0},
        {"timestamp": pd.Timestamp("2026-03-02 11:30"), "symbol": "AAPL", "side": "SELL", "shares": 5, "price": 11.0, "proceeds": 55.0, "secret_sauce": 1},
    ]
    assert tape.fills_since(trades, 1) == [{
        "timestamp": "2026-03-02T11:30:00", "symbol": "AAPL", "side": "SELL",
        "shares": 5, "price": 11.0, "proceeds": 55.0,
    }]


def test_gate_rewrote_compares_symbol_side_size_triples():
    buy5 = {"orders": [{"symbol": "AAPL", "action": "buy", "position_size": 5}]}
    assert tape.gate_rewrote(buy5, [{"symbol": "AAPL", "action": "buy", "shares": 5}]) is False
    # resized, dropped, added, and a SELL widened to the whole position
    assert tape.gate_rewrote(buy5, [{"symbol": "AAPL", "action": "buy", "shares": 3}]) is True
    assert tape.gate_rewrote(buy5, []) is True
    assert tape.gate_rewrote({"orders": []}, [{"symbol": "AAPL", "action": "buy", "shares": 1}]) is True
    trim = {"orders": [{"symbol": "AAPL", "action": "sell", "position_size": 2}]}
    assert tape.gate_rewrote(trim, [{"symbol": "AAPL", "action": "sell", "shares": 10}]) is True
    # no intent, or unparsed intent: there was no model order for the gate to rewrite
    assert tape.gate_rewrote(None, [{"symbol": "AAPL", "action": "buy", "shares": 1}]) is False
    assert tape.gate_rewrote({"unparsed": True, "completed_steps": 0}, []) is False
    # a hold is not an order; a sizeless intent compares on symbol/side only
    hold = {"orders": [{"symbol": "AAPL", "action": "hold"}]}
    assert tape.gate_rewrote(hold, []) is False
    sized_only_by_side = {"orders": [{"symbol": "AAPL", "action": "buy"}]}
    assert tape.gate_rewrote(sized_only_by_side, [{"symbol": "AAPL", "action": "buy", "shares": 7}]) is False


def test_decision_and_execution_payload_shapes():
    actions = [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "why", "confidence": 0.7, "extra": 1}]
    decision = tape.build_decision_payload(
        bar_index=4, decision_at=pd.Timestamp("2026-03-02 11:30"), driver="llm",
        intent={"orders": []}, state={"cash": 1.0, "equity": 1.0, "positions": []}, actions=actions,
    )
    assert decision["tape_version"] == 1 and decision["bar_index"] == 4
    assert decision["actions"] == [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "why", "confidence": 0.7}]
    assert decision["reasoning_summaries"] == ["why"] and decision["accepted"] is True
    assert decision["gate_rewrote"] is True  # the model ordered nothing; the gate produced a buy

    fill = SimpleNamespace(bar=pd.Timestamp("2026-03-02 11:30"), price_field="open", filled_at=pd.Timestamp("2026-03-02 11:30"))
    execution = tape.build_execution_payload(
        bar_index=4, fill=fill,
        fills=[{"symbol": "AAPL", "side": "BUY", "shares": 3, "price": 10.0}], rejected=[],
    )
    assert execution["fill_plan"] == {"bar": "2026-03-02T11:30:00", "price_field": "open", "filled_at": "2026-03-02T11:30:00"}
    assert execution["executed"] == [{"symbol": "AAPL", "side": "BUY", "shares": 3}]
    assert execution["run_status"] == "running" and execution["validation"] == {}


def test_bound_payload_drops_reasoning_before_orders():
    actions = [{"symbol": f"S{i}", "action": "buy", "shares": 1, "reason": "r" * 500} for i in range(150)]
    payload = tape.build_decision_payload(
        bar_index=0, decision_at="t", driver="llm", intent=None,
        state={"cash": 0, "equity": 0, "positions": []}, actions=actions,
    )
    assert _size(payload) > tape.MAX_PAYLOAD_BYTES
    bounded = tape.bound_payload(payload)
    assert bounded is not None and _size(bounded) <= tape.MAX_PAYLOAD_BYTES
    assert len(bounded["actions"]) == 150
    assert bounded["reasoning_summaries"] == []
    assert "reason" not in bounded["actions"][0]


def test_bound_payload_returns_none_when_orders_alone_are_too_big():
    actions = [{"symbol": "S" * 400, "action": "buy", "shares": 1} for _ in range(200)]
    payload = tape.build_decision_payload(
        bar_index=0, decision_at="t", driver="llm", intent=None,
        state={"cash": 0, "equity": 0, "positions": []}, actions=actions,
    )
    assert tape.bound_payload(payload) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: collection error / FAIL — `ImportError: cannot import name 'decision_tape'`.

- [ ] **Step 3: Implement the builders**

Create `dashboard/backend/domain/backtesting/decision_tape.py`:

```python
"""Per-bar decision tape for My Agents pipeline backtests.

For every decision bar this records what drove it, the raw model orders when a
pipeline produced them, the post-gate actions handed to ``execute_actions``,
the pre-decision portfolio, the fill plan, and the fills and rejections that
came back -- into the run's agent trace (#622) as one ``decision_recorded`` /
``execution_result`` pair. The tape is the replay input for the engine-rewrite
comparison: one fixed order sequence through two engines, so every difference
is the engine's.

The gap between ``intent`` (raw model orders) and ``actions`` (post-gate) is the
pre-trade gate's own behaviour -- ``make_trading_decision_with_llm`` drops and
resizes orders without recording why -- which is why both are kept.

Observational only: nothing here may change a backtest's outcome.
Spec: docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from dashboard.backend.infrastructure.llm.pipeline_runner import (
    pipeline_output_to_decision,
)

TAPE_VERSION = 1
MAX_REASONING_CHARS = 500
# Below the trace store's 64 KiB payload cap (domain/traces/common.py), with
# margin for the store's own serialization.
MAX_PAYLOAD_BYTES = 60 * 1024

DRIVER_LLM = "llm"
DRIVER_LLM_FALLBACK = "llm_fallback"
DRIVER_RULE_BASED = "rule_based"

# Allow-lists. Model output and ledger records can carry any key, and the trace
# store rejects a payload whose keys look sensitive (``token``, ``secret``...),
# so only named fields are ever copied.
_INTENT_FIELDS = (
    "symbol", "action", "position_size", "confidence", "reasoning",
    "stop_loss_price", "take_profit_price",
)
_ACTION_FIELDS = ("symbol", "action", "shares", "requested_shares", "confidence", "reason")
_FILL_FIELDS = ("timestamp", "symbol", "side", "shares", "price", "cost", "proceeds")
_REJECTION_FIELDS = (
    "timestamp", "symbol", "side", "requested_shares", "executed_shares",
    "unfilled_shares", "price", "status", "reason", "strategy_reason",
)


def jsonable(value: Any) -> Any:
    """Coerce ledger values (numpy scalars, pandas timestamps) to JSON."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return jsonable(value.item())
    return str(value)


def _pick(record: Mapping[str, Any], fields: Iterable[str], *, text_fields=()) -> Dict[str, Any]:
    picked: Dict[str, Any] = {}
    for field in fields:
        if field not in record or record[field] is None:
            continue
        value = record[field]
        if field in text_fields or isinstance(value, (Mapping, list, tuple)):
            value = str(value)[:MAX_REASONING_CHARS]
        picked[field] = jsonable(value)
    return picked


def driver_for(*, use_llm: bool, llm_decisions_before: int, llm_decisions_after: int) -> str:
    """Which path drove the bar. ``llm_decisions`` moves only at the LLM
    path's success exit, so an unchanged count on an LLM run is a fallback."""
    if not use_llm:
        return DRIVER_RULE_BASED
    if llm_decisions_after > llm_decisions_before:
        return DRIVER_LLM
    return DRIVER_LLM_FALLBACK


def intent_from_outputs(
    outputs: Optional[Sequence[Mapping[str, Any]]],
    *,
    outputs_before: Any,
    decision_step_count: int,
) -> Optional[Dict[str, Any]]:
    """The raw model orders of this bar's final pipeline step.

    ``PortfolioManager`` assigns a new list on every pipeline call and leaves
    the old one in place otherwise, so identity with the pre-decision snapshot
    means the pipeline did not run this bar (rule-based, single-prompt LLM
    path) and the list is the previous bar's.
    """
    if outputs is outputs_before or outputs is None:
        return None
    completed = len(outputs)
    if not outputs:
        return {"unparsed": True, "completed_steps": 0}
    last = outputs[-1]
    if not isinstance(last, Mapping) or last.get("step") != decision_step_count:
        return {"unparsed": True, "completed_steps": completed}
    decision = pipeline_output_to_decision(last.get("output"))
    if not decision:
        return {"unparsed": True, "completed_steps": completed}
    return {
        "orders": [
            _pick(action, _INTENT_FIELDS, text_fields=("reasoning",))
            for action in decision.get("actions") or []
            if isinstance(action, Mapping)
        ]
    }


def build_state(*, cash: Any, equity: Any, positions: Mapping[str, Any]) -> Dict[str, Any]:
    """Pre-decision portfolio. Positions are ``[symbol, shares]`` pairs so a
    ticker can never become a dict key the store's sensitive-key check reads."""
    return {
        "cash": jsonable(cash),
        "equity": jsonable(equity),
        "positions": [
            [str(symbol), jsonable(shares)]
            for symbol, shares in sorted(positions.items())
            if shares
        ],
    }


def order_events_snapshot(order_events: Optional[Sequence[Mapping[str, Any]]]) -> List[int]:
    return [int(event.get("repeat_count", 1) or 1) for event in order_events or ()]


def rejections_since(
    order_events: Optional[Sequence[Mapping[str, Any]]], snapshot: Sequence[int]
) -> List[Dict[str, Any]]:
    """Non-filled order events produced since ``snapshot``.

    A pure rejection repeated on one trading day is collapsed into the first
    record's ``repeat_count`` (``trading/execution.py``), so growth in that
    count is a rejection on this bar too, not only a newly appended record.
    """
    rejected: List[Dict[str, Any]] = []
    for index, event in enumerate(order_events or ()):
        count = int(event.get("repeat_count", 1) or 1)
        if index < len(snapshot):
            grew = count - snapshot[index]
            if grew > 0:
                rejected.append({**_pick(event, _REJECTION_FIELDS), "count": grew, "collapsed": True})
        elif event.get("status") != "filled":
            rejected.append({**_pick(event, _REJECTION_FIELDS), "count": count, "collapsed": False})
    return rejected


def fills_since(trades: Sequence[Mapping[str, Any]], start: int) -> List[Dict[str, Any]]:
    return [_pick(trade, _FILL_FIELDS) for trade in trades[start:]]


_ORDER_SIDES = {"buy", "sell"}


def _order_triples(orders: Iterable[Mapping[str, Any]], size_field: str):
    """``(symbol, side, size)`` per buy/sell order; ``size`` is ``None`` when
    the order carries none, and a ``None`` size matches any size."""
    triples = []
    for order in orders or ():
        if not isinstance(order, Mapping):
            continue
        side = str(order.get("action") or order.get("side") or "").lower()
        if side not in _ORDER_SIDES:
            continue  # hold / unknown: not an order the gate can rewrite
        size = order.get(size_field)
        triples.append((str(order.get("symbol") or "").upper(), side, None if size is None else float(size)))
    return sorted(triples, key=lambda t: (t[0], t[1], -1.0 if t[2] is None else t[2]))


def gate_rewrote(intent: Optional[Mapping[str, Any]], actions: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the pre-trade gate changed what the model asked for.

    Compares the sorted ``(symbol, side, size)`` triples of ``intent.orders``
    (size = ``position_size``) with those of the post-gate ``actions`` (size =
    ``shares``). A dropped, added or resized order, or a SELL widened to the
    whole position, is a rewrite. ``False`` with no intent or an unparsed one:
    a rule-based or fallback bar had no model order to rewrite. An intent
    order without a size compares on symbol and side only.
    """
    if not isinstance(intent, Mapping) or "orders" not in intent:
        return False
    wanted = _order_triples(intent.get("orders") or (), "position_size")
    got = _order_triples(actions or (), "shares")
    if len(wanted) != len(got):
        return True
    for (ws, wside, wsize), (gs, gside, gsize) in zip(wanted, got):
        if ws != gs or wside != gside:
            return True
        if wsize is not None and gsize is not None and wsize != gsize:
            return True
    return False


def build_decision_payload(
    *,
    bar_index: int,
    decision_at: Any,
    driver: str,
    intent: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    actions: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Superset of the ``/api/v2`` ``decision_recorded`` payload."""
    picked = [
        _pick(action, _ACTION_FIELDS, text_fields=("reason",))
        for action in actions or ()
        if isinstance(action, Mapping)
    ]
    return {
        "tape_version": TAPE_VERSION,
        "bar_index": int(bar_index),
        "decision_at": jsonable(decision_at),
        "driver": driver,
        "intent": intent,
        "state": state,
        "actions": picked,
        "gate_rewrote": gate_rewrote(intent, picked),
        "reasoning_summaries": [action.get("reason", "") for action in picked],
        "accepted": True,
    }


def build_execution_payload(
    *,
    bar_index: int,
    fill: Any,
    fills: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Superset of the ``/api/v2`` ``execution_result`` payload."""
    return {
        "tape_version": TAPE_VERSION,
        "bar_index": int(bar_index),
        "fill_plan": {
            "bar": jsonable(fill.bar),
            "price_field": fill.price_field,
            "filled_at": jsonable(fill.filled_at),
        },
        "accepted": True,
        "fills": fills,
        "executed": [
            {"symbol": f.get("symbol"), "side": f.get("side"), "shares": f.get("shares")}
            for f in fills
        ],
        "rejected": rejected,
        "validation": {},
        "run_status": "running",
    }


def _payload_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def bound_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fit ``payload`` under ``MAX_PAYLOAD_BYTES`` by dropping reasoning text
    first, then intent reasoning; orders are never dropped. ``None`` when the
    orders alone do not fit."""
    if _payload_size(payload) <= MAX_PAYLOAD_BYTES:
        return payload
    trimmed = copy.deepcopy(payload)
    if "reasoning_summaries" in trimmed:
        trimmed["reasoning_summaries"] = []
    for action in trimmed.get("actions") or []:
        action.pop("reason", None)
    if _payload_size(trimmed) <= MAX_PAYLOAD_BYTES:
        return trimmed
    intent = trimmed.get("intent")
    if isinstance(intent, dict):
        for order in intent.get("orders") or []:
            order.pop("reasoning", None)
    if _payload_size(trimmed) <= MAX_PAYLOAD_BYTES:
        return trimmed
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/decision_tape.py dashboard/backend/tests/domain/backtesting/test_decision_tape.py
git commit -m "feat(backtest): decision tape payload builders" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Trace service tape API

**Files:**
- Modify: `dashboard/backend/domain/traces/service.py` (append after `ensure_trace_for_run`)
- Test: `dashboard/backend/tests/test_trace_decision_tape.py`

**Interfaces:**
- Consumes: `trace_store.append_event(...)`, `trace_store.list_events(trace_id, *, after_sequence, limit) -> {"items", "next_sequence_no", "has_more"}`, existing `trace_for_run`, `complete_trace`, `fail_trace`.
- Produces (module `dashboard.backend.domain.traces.service`):
  - `record_tape_bar(*, trace_id: str, run_id: str, bar_index: int, decision_payload: dict, execution_payload: dict) -> None` — **raises** on store failure (the recorder is the swallow layer).
  - `load_decision_tape(run_id: str) -> List[dict]` — ordered `{"bar_index": int, "decision": dict, "execution": dict}`.
  - `finish_trace_best_effort(run_id: str, *, error_code: Optional[str] = None, result_summary: Optional[dict] = None) -> bool` — never raises.
  - `fail_trace_if_running(run_id: str, error_code: str) -> bool` — never raises; no-op unless status is `running`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/test_trace_decision_tape.py`:

```python
"""Trace-service side of the decision tape (spec 2026-10-08)."""

import pytest

from dashboard.backend.domain.traces import service
from dashboard.backend.domain.traces.repository import TraceStore

RUN = "agent_tape_service"


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "traces.db")
    monkeypatch.setattr(service, "trace_store", store)
    return store


def _trace(store):
    service.ensure_trace_for_run(run_id=RUN, initial_input={})
    return service.trace_for_run(RUN)["trace_id"]


def _pair(i):
    return (
        {"tape_version": 1, "bar_index": i, "actions": [{"symbol": "AAPL", "action": "buy", "shares": i}]},
        {"tape_version": 1, "bar_index": i, "fills": [], "rejected": []},
    )


def test_record_tape_bar_writes_a_linked_pair_with_v2_ids(store):
    trace_id = _trace(store)
    decision, execution = _pair(3)
    service.record_tape_bar(trace_id=trace_id, run_id=RUN, bar_index=3,
                            decision_payload=decision, execution_payload=execution)
    events = store.list_events(trace_id)["items"]
    assert [e["event_type"] for e in events] == ["run_started", "decision_recorded", "execution_result"]
    assert events[1]["step_id"] == events[2]["step_id"] == f"step_{RUN}_3"
    assert events[1]["decision_id"] == events[2]["decision_id"] == f"dec_{RUN}_3"
    assert events[1]["actor_type"] == "agent" and events[2]["actor_type"] == "system"


def test_record_tape_bar_is_idempotent_per_bar(store):
    trace_id = _trace(store)
    decision, execution = _pair(0)
    for _ in range(2):
        service.record_tape_bar(trace_id=trace_id, run_id=RUN, bar_index=0,
                                decision_payload=decision, execution_payload=execution)
    assert len(store.list_events(trace_id)["items"]) == 3


def test_load_decision_tape_pairs_bars_in_order_across_pages(store):
    trace_id = _trace(store)
    for i in range(60):  # 120 tape events: more than one 100-event page
        decision, execution = _pair(i)
        service.record_tape_bar(trace_id=trace_id, run_id=RUN, bar_index=i,
                                decision_payload=decision, execution_payload=execution)
    tape = service.load_decision_tape(RUN)
    assert [bar["bar_index"] for bar in tape] == list(range(60))
    assert tape[7]["decision"]["actions"][0]["shares"] == 7
    assert tape[7]["execution"]["bar_index"] == 7


def test_load_decision_tape_is_empty_without_a_trace(store):
    assert service.load_decision_tape("agent_missing") == []


def test_finish_trace_best_effort_completes_and_fails(store):
    _trace(store)
    assert service.finish_trace_best_effort(RUN, result_summary={"bars_recorded": 2}) is True
    assert service.trace_for_run(RUN)["status"] == "completed"


def test_finish_trace_best_effort_never_raises(store, monkeypatch):
    _trace(store)

    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "append_event", boom)
    assert service.finish_trace_best_effort(RUN, error_code="run_failed") is False


def test_fail_trace_if_running_marks_a_running_trace(store):
    _trace(store)
    assert service.fail_trace_if_running(RUN, "run_killed") is True
    trace = service.trace_for_run(RUN)
    assert trace["status"] == "failed"
    events = store.list_events(trace["trace_id"])["items"]
    assert events[-1]["payload"] == {"error_code": "run_killed"}


def test_fail_trace_if_running_leaves_a_completed_trace_alone(store):
    _trace(store)
    service.complete_trace(RUN, {"ok": True})
    assert service.fail_trace_if_running(RUN, "run_killed") is False
    assert service.trace_for_run(RUN)["status"] == "completed"


def test_fail_trace_if_running_without_trace_is_a_no_op(store):
    assert service.fail_trace_if_running("agent_missing", "run_killed") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dashboard/backend/tests/test_trace_decision_tape.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'record_tape_bar'`.

- [ ] **Step 3: Implement**

In `dashboard/backend/domain/traces/service.py`, change the typing import to `from typing import Any, Dict, List, Optional`, then append:

```python
# --------------------------------------------------------------------------
# Decision tape (My Agents backtests; spec 2026-10-08-backtest-decision-tape)
# --------------------------------------------------------------------------


def record_tape_bar(
    *,
    trace_id: str,
    run_id: str,
    bar_index: int,
    decision_payload: Dict[str, Any],
    execution_payload: Dict[str, Any],
) -> None:
    """Append one bar's ``decision_recorded`` + ``execution_result`` pair.

    Takes the caller's cached ``trace_id`` (one lookup per run, not two per
    bar) and raises on failure: ``DecisionTapeRecorder`` is the single place
    that swallows and counts, so it can say what failed.
    """
    step_id = f"step_{run_id}_{bar_index}"
    decision_id = f"dec_{run_id}_{bar_index}"
    trace_store.append_event(
        trace_id=trace_id,
        event_type="decision_recorded",
        actor_type="agent",
        step_id=step_id,
        decision_id=decision_id,
        payload=decision_payload,
        idempotency_key=f"decision:{run_id}:{step_id}:tape",
    )
    trace_store.append_event(
        trace_id=trace_id,
        event_type="execution_result",
        actor_type="system",
        step_id=step_id,
        decision_id=decision_id,
        payload=execution_payload,
        idempotency_key=f"execution:{run_id}:{step_id}:tape",
    )


_TAPE_EVENT_KINDS = {"decision_recorded": "decision", "execution_result": "execution"}


def load_decision_tape(run_id: str) -> List[Dict[str, Any]]:
    """The run's tape as ordered ``{bar_index, decision, execution}`` pairs.

    Only events carrying ``tape_version`` count, so v2 decision events on the
    same vocabulary are never mistaken for tape bars.
    """
    trace = trace_for_run(run_id)
    if trace is None:
        return []
    bars: Dict[int, Dict[str, Any]] = {}
    after = 0
    while True:
        page = trace_store.list_events(trace["trace_id"], after_sequence=after, limit=100)
        for event in page["items"]:
            kind = _TAPE_EVENT_KINDS.get(event.get("event_type"))
            payload = event.get("payload") or {}
            if kind is None or payload.get("tape_version") is None:
                continue
            index = int(payload["bar_index"])
            bars.setdefault(index, {"bar_index": index})[kind] = payload
        if not page["has_more"]:
            break
        after = page["next_sequence_no"] - 1
    return [bars[index] for index in sorted(bars)]


def finish_trace_best_effort(
    run_id: str,
    *,
    error_code: Optional[str] = None,
    result_summary: Optional[Dict[str, Any]] = None,
) -> bool:
    """Close a run's trace as completed (or failed with ``error_code``)
    without ever raising -- the run's outcome is already decided."""
    try:
        if error_code:
            fail_trace(run_id, error_code)
        else:
            complete_trace(run_id, result_summary)
        return True
    except Exception:
        return False


def fail_trace_if_running(run_id: str, error_code: str) -> bool:
    """Fail a trace its process left ``running`` (killed child). A trace the
    child already closed is left alone, so this is safe on every exit path."""
    try:
        trace = trace_for_run(run_id)
        if trace is None or trace.get("status") != "running":
            return False
        fail_trace(run_id, error_code)
        return True
    except Exception:
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/test_trace_decision_tape.py dashboard/backend/tests/test_trace_run_instrumentation.py dashboard/backend/tests/test_trace_repository.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/traces/service.py dashboard/backend/tests/test_trace_decision_tape.py
git commit -m "feat(traces): tape bar writer, reader and best-effort finalizers" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Recorder and trace lifecycle context

**Files:**
- Modify: `dashboard/backend/domain/backtesting/decision_tape.py` (append)
- Test: `dashboard/backend/tests/domain/backtesting/test_decision_tape.py` (append)

**Interfaces:**
- Consumes: Task 1 builders; Task 2 `trace_service.trace_for_run`, `record_tape_bar`, `finish_trace_best_effort` — **imported lazily inside each function**, never at module top (Global Constraints).
- Produces:
  - `class DecisionTapeRecorder(run_id: str)` with `record_bar(*, bar_index: int, decision_payload: dict, execution_payload: dict) -> None` (never raises; reads `decision_payload["gate_rewrote"]` to count) and `summary() -> dict` returning `{"tape_version", "traced", "bars_recorded", "gate_rewrites", "write_failures", "oversize_skipped"}`.
  - `trace_lifecycle(run_id: Optional[str], *, summary: Callable[[], dict])` — a `contextlib.contextmanager`. On normal exit: `complete` with `summary()`. On `SystemExit`: **writes nothing**, re-raises (the parent labels it). On any other `BaseException`: `fail(run_failed)`, re-raises. With no `run_id`: touches no trace.

- [ ] **Step 1: Write the failing tests**

Append to `dashboard/backend/tests/domain/backtesting/test_decision_tape.py`:

```python
import pytest

from dashboard.backend.domain.traces import service as trace_service


class _FakeTraceService:
    def __init__(self, trace=None, lookup_error=None, write_error=None):
        self.trace = trace
        self.lookup_error = lookup_error
        self.write_error = write_error
        self.lookups = 0
        self.writes = []

    def trace_for_run(self, run_id):
        self.lookups += 1
        if self.lookup_error is not None:
            error, self.lookup_error = self.lookup_error, None  # transient: fails once
            raise error
        return self.trace

    def record_tape_bar(self, **kwargs):
        if self.write_error is not None:
            raise self.write_error
        self.writes.append(kwargs)


def _install(monkeypatch, fake):
    monkeypatch.setattr(trace_service, "trace_for_run", fake.trace_for_run)
    monkeypatch.setattr(trace_service, "record_tape_bar", fake.record_tape_bar)


def _bar(i, rewrote=False):
    return {
        "bar_index": i,
        "decision_payload": {"tape_version": 1, "gate_rewrote": rewrote},
        "execution_payload": {"tape_version": 1},
    }


def test_decision_tape_module_does_not_import_the_trace_service_at_top():
    """The trace service builds its store on import (schema DDL / a Neon
    dial); the engine imports it lazily and so must this module, or every
    engine import touches a database."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(tape))
    offenders = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and "traces" in ((getattr(node, "module", None) or "") + "".join(a.name for a in node.names))
    ]
    assert offenders == [], "import domain.traces.service inside the function that needs it"


def test_recorder_resolves_trace_once_and_writes_each_bar(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1, rewrote=True))
    recorder.record_bar(**_bar(2))
    assert fake.lookups == 1
    assert [w["bar_index"] for w in fake.writes] == [0, 1, 2]
    assert fake.writes[0]["trace_id"] == "trc_1" and fake.writes[0]["run_id"] == "agent_x"
    assert recorder.summary() == {
        "tape_version": 1, "traced": True, "bars_recorded": 3, "gate_rewrites": 1,
        "write_failures": 0, "oversize_skipped": 0,
    }


def test_recorder_is_a_no_op_without_a_trace(monkeypatch):
    fake = _FakeTraceService(trace=None)
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1))
    assert fake.lookups == 1 and fake.writes == []
    assert recorder.summary()["traced"] is False


def test_recorder_retries_a_lookup_that_raised(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"}, lookup_error=RuntimeError("neon blip"))
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1))
    assert fake.lookups == 2
    assert [w["bar_index"] for w in fake.writes] == [1]
    assert recorder.summary()["write_failures"] == 1


def test_recorder_swallows_write_failures_and_logs_once(monkeypatch, capsys):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"}, write_error=ValueError("payload exceeds"))
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")
    for i in range(3):
        recorder.record_bar(**_bar(i))  # must not raise
    assert recorder.summary()["write_failures"] == 3
    assert capsys.readouterr().out.count("decision tape write failed") == 1


def test_recorder_counts_oversize_bars_without_writing(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    monkeypatch.setattr(tape, "bound_payload", lambda payload: None)
    recorder = tape.DecisionTapeRecorder("agent_x")
    recorder.record_bar(**_bar(0))
    assert fake.writes == [] and recorder.summary()["oversize_skipped"] == 1


def _capture_finish(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda run_id, **kw: calls.append((run_id, kw)) or True)
    return calls


def test_lifecycle_completes_with_summary(monkeypatch):
    calls = _capture_finish(monkeypatch)
    with tape.trace_lifecycle("agent_x", summary=lambda: {"bars_recorded": 4}):
        pass
    assert calls == [("agent_x", {"result_summary": {"bars_recorded": 4}})]


def test_lifecycle_summary_is_read_at_exit_not_entry(monkeypatch):
    """The child's summary comes from a backtester that does not exist yet
    when the context opens (it spans the constructor), so it is late-bound."""
    calls = _capture_finish(monkeypatch)
    holder = {"bt": None}
    with tape.trace_lifecycle("agent_x", summary=lambda: holder["bt"].summary() if holder["bt"] else {}):
        holder["bt"] = type("BT", (), {"summary": staticmethod(lambda: {"bars_recorded": 9})})()
    assert calls == [("agent_x", {"result_summary": {"bars_recorded": 9}})]


def test_lifecycle_fails_on_an_exception_and_reraises(monkeypatch):
    calls = _capture_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}):
            raise RuntimeError("boom")
    assert calls == [("agent_x", {"error_code": "run_failed"})]


def test_lifecycle_fails_on_an_exception_raised_before_run(monkeypatch):
    """Review Focus 7: load_data()/calculate_indicators() fail after the
    trace exists; that must read run_failed, never run_killed."""
    calls = _capture_finish(monkeypatch)

    class _FeedRefused(Exception):
        pass

    with pytest.raises(_FeedRefused):
        with tape.trace_lifecycle("agent_x", summary=lambda: pytest.fail("no summary on failure")):
            raise _FeedRefused("SIP refused")  # stands in for load_data()
    assert calls == [("agent_x", {"error_code": "run_failed"})]


def test_lifecycle_writes_nothing_on_system_exit(monkeypatch):
    """The SIGTERM exit is sent by both the cancel and the timeout arms; only
    the parent knows which, so the child leaves the trace running for it."""
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("the parent labels a SIGTERM exit"))
    with pytest.raises(SystemExit):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}):
            raise SystemExit(143)


def test_lifecycle_without_run_id_touches_no_trace(monkeypatch):
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("no run id, no trace"))
    with tape.trace_lifecycle(None, summary=lambda: {}):
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: new tests FAIL — `AttributeError: ... has no attribute 'DecisionTapeRecorder'`.

- [ ] **Step 3: Implement**

In `decision_tape.py`, extend the typing import to `from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence`, add `from contextlib import contextmanager` to the stdlib imports, and **do not** add a module-level import of the trace service. Append:

```python
def _trace_service():
    """Lazy on purpose: ``domain/traces/service.py`` constructs its store on
    import (schema DDL on DB_PATH, or a Neon dial). The engine imports this
    module, and the engine is imported by scripts that must not touch a
    database (``diff_backtest_runs.py``, a ``python -c`` probe). Same pattern
    as ``engine.py`` and ``market_data/provider.py``. Returns the module
    object, so tests patching attributes on it are honoured."""
    from dashboard.backend.domain.traces import service as trace_service

    return trace_service


class DecisionTapeRecorder:
    """Writes one run's tape. Never raises: a broken trace store costs the
    tape, never the backtest. The trace is looked up once; a lookup that
    raised (a transient store error) is retried on the next bar, one that
    found no trace turns the recorder into a no-op for the run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._trace_id: Optional[str] = None
        self._resolved = False
        self._warned = False
        self.bars_recorded = 0
        self.gate_rewrites = 0
        self.write_failures = 0
        self.oversize_skipped = 0

    def _resolve_trace(self) -> Optional[str]:
        if not self._resolved:
            trace = _trace_service().trace_for_run(self.run_id)
            self._resolved = True
            self._trace_id = trace["trace_id"] if trace else None
        return self._trace_id

    def _warn(self, exc: BaseException) -> None:
        if self._warned:
            return
        self._warned = True
        print(
            f"   ⚠️  decision tape write failed ({type(exc).__name__}: {exc}); "
            "later failures are counted, not printed",
            flush=True,
        )

    def record_bar(
        self,
        *,
        bar_index: int,
        decision_payload: Dict[str, Any],
        execution_payload: Dict[str, Any],
    ) -> None:
        try:
            trace_id = self._resolve_trace()
            if trace_id is None:
                return
            decision = bound_payload(decision_payload)
            execution = bound_payload(execution_payload)
            if decision is None or execution is None:
                self.oversize_skipped += 1
                return
            _trace_service().record_tape_bar(
                trace_id=trace_id,
                run_id=self.run_id,
                bar_index=bar_index,
                decision_payload=decision,
                execution_payload=execution,
            )
            self.bars_recorded += 1
            if decision.get("gate_rewrote"):
                self.gate_rewrites += 1
        except Exception as exc:  # observational: never reaches the bar loop
            self.write_failures += 1
            self._warn(exc)

    def summary(self) -> Dict[str, Any]:
        return {
            "tape_version": TAPE_VERSION,
            "traced": self._trace_id is not None,
            "bars_recorded": self.bars_recorded,
            "gate_rewrites": self.gate_rewrites,
            "write_failures": self.write_failures,
            "oversize_skipped": self.oversize_skipped,
        }


@contextmanager
def trace_lifecycle(
    run_id: Optional[str],
    *,
    summary: Callable[[], Dict[str, Any]],
) -> Iterator[None]:
    """Close the run's trace on the way out of the block: completed with
    ``summary()`` (read at exit, so it may reference objects built inside the
    block), failed with ``run_failed`` on an exception.

    ``SystemExit`` writes **nothing** and re-raises. That exit is the
    dashboard's SIGTERM (``_exit_on_sigterm``), which the parent sends from
    two arms -- the user's cancel and the timeout's grace before SIGKILL --
    and only the parent knows which; it labels a trace left ``running`` by
    the arm it is in (``api/routers/backtests.py``). Open the block before
    ``HourlyBacktester(...)``: the trace is created in its constructor, so a
    ``load_data()`` failure outside the block would be labelled a kill.
    """
    if not run_id:
        yield
        return
    try:
        yield
    except SystemExit:
        raise
    except BaseException:
        _trace_service().finish_trace_best_effort(run_id, error_code="run_failed")
        raise
    try:
        result_summary = summary()
    except Exception:
        result_summary = {}
    _trace_service().finish_trace_best_effort(run_id, result_summary=result_summary)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/decision_tape.py dashboard/backend/tests/domain/backtesting/test_decision_tape.py
git commit -m "feat(backtest): decision tape recorder and trace lifecycle" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Engine wiring, reproduction, intent replay, divergence

**Files:**
- Modify: `dashboard/backend/domain/backtesting/engine.py` (imports; `run_agent_backtest` ~2060 and loop ~2147–2276; new method `decision_tape_summary`)
- Test: `dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3; `split_pipeline` (already imported in engine.py); `PIPELINE_RUNTIME_TYPE` (already in scope).
- Produces: `HourlyBacktester.decision_tape_summary() -> dict` (empty dict when no tape ran); attribute `HourlyBacktester._decision_tape: Optional[DecisionTapeRecorder]`.

- [ ] **Step 1: Write the failing tests**

Create `dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py`:

```python
"""Decision tape through the real HourlyBacktester (spec 2026-10-08).

Harness: the conformance suite's in-memory 5m bars and recording DB, so no
network, plus a real SQLite TraceStore patched into the trace service.
"""

import dataclasses
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.traces import service as trace_service
from dashboard.backend.domain.traces.repository import TraceStore
from dashboard.backend.infrastructure.market_data.profiles import TransactionCostProfile
from dashboard.backend.tests.conformance.adapters.current_engine import (
    _CaseLoader,
    _RecordingDB,
    synthesize_source_bars,
)
from dashboard.backend.tests.conformance.cases import CASH, _flat_pair

BARS = _flat_pair(6)  # AAPL @100, MSFT @50, flat, seven hourly bars on one day

# The true originals, captured once at import. Each _run patches over them;
# wrapping whatever is currently installed would chain a second run's wrapper
# onto the first run's holder, and the replay test would then compare the
# second run's ledger with itself.
_ORIGINAL_INIT = PortfolioManager.__init__


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "traces.db")
    monkeypatch.setattr(trace_service, "trace_store", store)
    return store


def _run(monkeypatch, *, run_id="agent_tape_engine", decide=None, llm_client=None, pipeline=None, configure=None):
    """``configure(backtester)`` runs after construction and before
    ``load_data()`` -- the hook the perturbed-engine test uses to hand this
    run a cost profile without touching ``profiles.py``."""
    loader = _CaseLoader(synthesize_source_bars(SimpleNamespace(bars=BARS)))

    def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
        loader.configure_source_timeframe(source_timeframe)
        return loader

    monkeypatch.setattr(engine_mod, "create_market_data_provider", factory)
    monkeypatch.setattr(engine_mod, "db", _RecordingDB())
    holder = SimpleNamespace(pm=None, calls=0)

    def init(pm, *args, **kwargs):
        _ORIGINAL_INIT(pm, *args, **kwargs)
        holder.pm = pm

    monkeypatch.setattr(PortfolioManager, "__init__", init)
    if decide is not None:
        def scripted(pm, state):
            index = holder.calls
            holder.calls += 1
            return {"actions": [dict(action) for action in decide(index, pm)]}

        monkeypatch.setattr(PortfolioManager, "make_trading_decision", scripted)

    day = min(bar.ts for bar in BARS).date()
    backtester = HourlyBacktester(
        day.isoformat(), day.isoformat(), use_llm=False,
        symbols=["AAPL", "MSFT"], initial_capital=float(CASH), live_run_id=run_id,
    )
    backtester.provider_end_date = (day + timedelta(days=1)).isoformat()
    if llm_client is not None:
        backtester.use_llm = True
        backtester.llm_client = llm_client
        backtester.pipeline = pipeline
        backtester.strict_llm = False
    if configure is not None:
        configure(backtester)
    backtester.load_data()
    backtester.calculate_indicators()
    _run_id, curve = backtester.run_agent_backtest()
    return backtester, holder, curve


def _first_divergent_bar(curve_a, curve_b):
    """Index of the first equity point that differs; ``None`` when none does.
    The same question ``scripts/diff_backtest_runs.py`` answers for two
    stored runs."""
    for index, (a, b) in enumerate(zip(curve_a, curve_b)):
        if a["equity"] != b["equity"]:
            return index
    return None if len(curve_a) == len(curve_b) else min(len(curve_a), len(curve_b))


def _script(index, pm):
    return {
        0: [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "open AAPL"}],
        1: [{"symbol": "MSFT", "action": "buy", "shares": 2, "reason": "open MSFT"},
            {"symbol": "AAPL", "action": "sell", "shares": 1, "reason": "trim AAPL"}],
        2: [{"symbol": "AAPL", "action": "buy", "shares": 10_000_000, "reason": "too big"}],
        3: [{"symbol": "AAPL", "action": "buy", "shares": 10_000_000, "reason": "too big"}],
    }.get(index, [])


def test_every_bar_records_one_decision_execution_pair_in_order(monkeypatch, store):
    backtester, holder, _curve = _run(monkeypatch, decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_engine")
    assert len(tape) == holder.calls >= 5
    assert [bar["bar_index"] for bar in tape] == list(range(holder.calls))
    first = tape[0]
    assert first["decision"]["driver"] == "rule_based"
    assert first["decision"]["intent"] is None
    assert first["decision"]["state"] == {"cash": float(CASH), "equity": float(CASH), "positions": []}
    assert first["decision"]["actions"] == [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "open AAPL"}]
    assert first["decision"]["gate_rewrote"] is False  # rule-based: no model order to rewrite
    assert [(f["symbol"], f["side"], f["shares"]) for f in first["execution"]["fills"]] == [("AAPL", "BUY", 3)]
    assert set(first["execution"]["fill_plan"]) == {"bar", "price_field", "filled_at"}
    assert tape[1]["decision"]["state"]["positions"] == [["AAPL", 3]]
    summary = backtester.decision_tape_summary()
    assert summary["bars_recorded"] == holder.calls and summary["write_failures"] == 0
    assert summary["gate_rewrites"] == 0


def test_collapsed_repeat_rejection_appears_on_its_own_bar(monkeypatch, store):
    _run(monkeypatch, decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_engine")
    second, third = tape[2]["execution"]["rejected"], tape[3]["execution"]["rejected"]
    assert [(r["symbol"], r["reason"], r["collapsed"]) for r in second] == [("AAPL", "insufficient_cash", False)]
    assert [(r["symbol"], r["reason"], r["collapsed"], r["count"]) for r in third] == [
        ("AAPL", "insufficient_cash", True, 1)
    ]


def _comparable(trades):
    return json.dumps(
        [{k: str(v) for k, v in trade.items()} for trade in trades], sort_keys=True
    )


def test_recorded_actions_reproduce_the_run_on_the_same_engine(monkeypatch, store):
    """Reproducibility, not replay-grade: a second run of THIS engine driven
    by the recorded post-gate actions reproduces the first run's trades and
    curve. It proves the tape survives _pick + JSON and that the engine is
    deterministic on a fixed sequence -- nothing about another engine."""
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_tape_original", decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_original")

    def replay(index, pm):
        return tape[index]["decision"]["actions"]

    _bt, second, second_curve = _run(monkeypatch, run_id="agent_tape_replay", decide=replay)

    assert first.pm is not second.pm
    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert _comparable(second.pm.trades) == _comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]


def test_perturbed_engine_diverges_at_the_first_filled_bar(monkeypatch, store):
    """Review Focus 6: replaying recorded ACTIONS into an engine whose fills
    differ must diverge where the fill is, visibly, and never be absorbed by
    a later recorded size. The perturbation is one tick of slippage on the
    test's own MarketProfile copy (the mechanism A-share uses); profiles.py
    is untouched. This is also the shape diff_backtest_runs.py reports."""
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_tape_clean", decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_clean")
    first_fill_bar = next(i for i, bar in enumerate(tape) if bar["execution"]["fills"])

    def replay(index, pm):
        return tape[index]["decision"]["actions"]

    def slip(backtester):
        backtester.profile = dataclasses.replace(
            backtester.profile,
            transaction_cost_profile=TransactionCostProfile(
                version="test-slip-1tick", currency="USD",
                commission_rate=0.0, minimum_commission=0.0,
                stamp_duty_sell_rate=0.0, transfer_fee_rate=0.0,
                buy_slippage_rate=0.001, sell_slippage_rate=0.001, price_tick=0.01,
            ),
        )

    _bt, second, second_curve = _run(monkeypatch, run_id="agent_tape_slipped", decide=replay, configure=slip)

    assert _first_divergent_bar(first_curve, second_curve) == first_fill_bar
    assert any(float(t.get("slippage_amount") or 0) > 0 for t in second.pm.trades)
    # The recorded SELL sizes are the clean engine's; the slipped engine held
    # the same shares (slippage moves price, not quantity), so the trade list
    # keeps its shape and only prices differ -- which is exactly why actions
    # cannot be the replay unit once quantities, not prices, diverge.
    assert [(t["symbol"], t["side"], t["shares"]) for t in second.pm.trades] == \
        [(t["symbol"], t["side"], t["shares"]) for t in first.pm.trades]
    assert _comparable(second.pm.trades) != _comparable(first.pm.trades)


class _PipelineStub:
    """Answers each pipeline call from a script (one decision step per bar)."""

    execution_service = "decision-tape-stub"

    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_request):
        text = self.replies.pop(0) if self.replies else json.dumps(
            {"actions": [{"symbol": "AAPL", "action": "hold", "confidence": 1.0, "reasoning": "hold"}]}
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            stop_reason="end_turn",
        )


def test_llm_pipeline_bars_record_intent_and_fallback(monkeypatch, store):
    stub = _PipelineStub([
        json.dumps({"actions": [{
            "symbol": "AAPL", "action": "buy", "position_size": 5, "confidence": 0.9,
            "reasoning": "buy five", "api_key": "must-not-be-stored",
        }]}),
        "this is not json",
    ])
    _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    tape = trace_service.load_decision_tape("agent_tape_engine")

    assert tape[0]["decision"]["driver"] == "llm"
    assert tape[0]["decision"]["intent"] == {"orders": [{
        "symbol": "AAPL", "action": "buy", "position_size": 5,
        "confidence": 0.9, "reasoning": "buy five",
    }]}
    assert tape[0]["decision"]["actions"][0]["shares"] == 5
    assert tape[0]["decision"]["gate_rewrote"] is False  # asked 5, admitted 5

    assert tape[1]["decision"]["driver"] == "llm_fallback"
    assert tape[1]["decision"]["intent"] == {"unparsed": True, "completed_steps": 0}
    assert tape[1]["decision"]["gate_rewrote"] is False  # nothing parsed, nothing to rewrite


def test_llm_sell_widened_by_the_gate_is_counted_as_a_rewrite(monkeypatch, store):
    """Finding 1: every model SELL becomes 'sell all sellable'. The tape must
    say so per bar and per run, so the gate's rewrite rate is a number."""
    stub = _PipelineStub([
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 4, "confidence": 0.9, "reasoning": "open"}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "sell", "position_size": 1, "confidence": 0.9, "reasoning": "trim one"}]}),
    ])
    backtester, _holder, _curve = _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    tape = trace_service.load_decision_tape("agent_tape_engine")
    assert tape[0]["decision"]["gate_rewrote"] is False
    assert tape[1]["decision"]["intent"]["orders"][0]["position_size"] == 1
    assert tape[1]["decision"]["actions"][0]["shares"] == 4  # the whole position
    assert tape[1]["decision"]["gate_rewrote"] is True
    assert backtester.decision_tape_summary()["gate_rewrites"] == 1


def test_intent_replay_through_the_gate_reproduces_an_llm_run(monkeypatch, store):
    """The replay-grade proof (Review Focus 6): a second run whose MODEL
    answers each bar with the first run's recorded intent -- so the orders go
    through the gate and the fill model again rather than around them --
    reproduces the trades and the curve. This is the path a candidate engine
    takes with a tape; the actions path above is not."""
    script = [
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 3, "confidence": 0.9, "reasoning": "open"}]}),
        json.dumps({"actions": [
            {"symbol": "MSFT", "action": "buy", "position_size": 2, "confidence": 0.8, "reasoning": "add"},
            {"symbol": "AAPL", "action": "sell", "position_size": 1, "confidence": 0.7, "reasoning": "trim"},
        ]}),
    ]
    pipeline = [{"label": "Decision", "prompt": "Decide."}]
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_intent_original", llm_client=_PipelineStub(script), pipeline=pipeline)
    tape = trace_service.load_decision_tape("agent_intent_original")
    assert all(bar["decision"]["driver"] == "llm" for bar in tape)
    assert all("orders" in bar["decision"]["intent"] for bar in tape), "every bar must carry parsed intent"

    replies = [json.dumps({"actions": bar["decision"]["intent"]["orders"]}) for bar in tape]
    _bt, second, second_curve = _run(monkeypatch, run_id="agent_intent_replay", llm_client=_PipelineStub(replies), pipeline=pipeline)
    replayed = trace_service.load_decision_tape("agent_intent_replay")

    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert _comparable(second.pm.trades) == _comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]
    assert [b["decision"]["intent"] for b in replayed] == [b["decision"]["intent"] for b in tape]
    assert [b["decision"]["gate_rewrote"] for b in replayed] == [b["decision"]["gate_rewrote"] for b in tape]


def test_portfolio_manager_reassigns_pipeline_outputs_per_call(monkeypatch, store):
    """Review Focus 9, pinned on the manager, not through the tape: the
    tape's freshness check is `outputs is outputs_before`. A refactor of
    portfolio_manager.py:650 to `.clear()` + `.extend()` would keep one list
    object alive across bars, null `intent` on every bar, and leave every
    tape unit test green. This is the test that must go red, with this name."""
    stub = _PipelineStub([
        json.dumps({"actions": [{"symbol": "AAPL", "action": "buy", "position_size": 1, "confidence": 0.9, "reasoning": "a"}]}),
        json.dumps({"actions": [{"symbol": "AAPL", "action": "hold", "confidence": 1.0, "reasoning": "b"}]}),
    ])
    seen = []
    original = PortfolioManager.make_trading_decision_with_llm

    def spy(pm, *args, **kwargs):
        before = pm.last_pipeline_step_outputs
        result = original(pm, *args, **kwargs)
        seen.append((before, pm.last_pipeline_step_outputs))
        return result

    monkeypatch.setattr(PortfolioManager, "make_trading_decision_with_llm", spy)
    _run(monkeypatch, llm_client=stub, pipeline=[{"label": "Decision", "prompt": "Decide."}])
    assert len(seen) >= 2
    for before, after in seen:
        assert after is not before, "PortfolioManager must assign a NEW list per pipeline call"


def test_run_without_live_run_id_writes_no_tape(monkeypatch, store):
    backtester, _holder, _curve = _run(monkeypatch, run_id=None, decide=_script)
    assert backtester.decision_tape_summary() == {}
    assert store.list_traces()["items"] == []


def test_broken_trace_writes_do_not_change_the_run(monkeypatch, store):
    _bt, healthy, healthy_curve = _run(monkeypatch, run_id="agent_tape_healthy", decide=_script)

    def broken(**_kwargs):
        raise RuntimeError("trace store down")

    monkeypatch.setattr(trace_service, "record_tape_bar", broken)
    backtester, broken_holder, broken_curve = _run(monkeypatch, run_id="agent_tape_broken", decide=_script)
    assert [p["equity"] for p in broken_curve] == [p["equity"] for p in healthy_curve]
    assert len(broken_holder.pm.trades) == len(healthy.pm.trades)
    summary = backtester.decision_tape_summary()
    assert summary["write_failures"] == broken_holder.calls and summary["bars_recorded"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py -v`
Expected: FAIL — `load_decision_tape` returns `[]` (no tape written) / `AttributeError: 'HourlyBacktester' object has no attribute 'decision_tape_summary'`.

- [ ] **Step 3: Wire the engine**

In `dashboard/backend/domain/backtesting/engine.py`:

(a) Add to the module imports (beside the other `domain.backtesting` imports):

```python
from dashboard.backend.domain.backtesting.decision_tape import (
    DecisionTapeRecorder,
    build_decision_payload,
    build_execution_payload,
    build_state,
    driver_for,
    fills_since,
    intent_from_outputs,
    order_events_snapshot,
    rejections_since,
)
```

(b) In `run_agent_backtest`, immediately after the existing line `_decision_steps, post_trade_steps = split_pipeline(self.pipeline)` (~2060) and its `if post_trade_steps:` print block, add:

```python
        # Decision tape (spec 2026-10-08): the run's order sequence, recorded
        # per bar into its agent trace so it can be replayed through another
        # engine. Observational -- the recorder never raises into this loop.
        tape = (
            DecisionTapeRecorder(self.live_run_id)
            if self.runtime_type == PIPELINE_RUNTIME_TYPE and self.live_run_id
            else None
        )
        self._decision_tape = tape
        tape_decision_steps = len(_decision_steps)
        tape_uses_llm = bool(self.use_llm and self.llm_client)
```

(c) In the loop, immediately after `runtime_invoked = False` (~2166) and before `if self.runtime_type == PIPELINE_RUNTIME_TYPE:`, add:

```python
            if tape is not None:
                # Snapshots taken before the decision: identity of the step
                # outputs (a new list means the pipeline ran this bar), the LLM
                # success counter, the order-event repeat counts (collapsed
                # repeats only show up there) and the pre-decision portfolio.
                tape_outputs_before = manager.last_pipeline_step_outputs
                tape_llm_before = manager.llm_decisions
                tape_events_before = order_events_snapshot(
                    getattr(manager, "order_events", None)
                )
                tape_state = build_state(
                    cash=state["cash"],
                    equity=state["total_equity"],
                    positions=manager.positions,
                )
```

(d) Immediately after the existing block

```python
            if runtime_invoked:
                self.runtime_dispatcher.record_latest_execution(
                    len(manager.trades) - trades_before_execution
                )
```

add:

```python
            if tape is not None:
                tape.record_bar(
                    bar_index=i,
                    decision_payload=build_decision_payload(
                        bar_index=i,
                        decision_at=timestamp,
                        driver=driver_for(
                            use_llm=tape_uses_llm,
                            llm_decisions_before=tape_llm_before,
                            llm_decisions_after=manager.llm_decisions,
                        ),
                        intent=intent_from_outputs(
                            manager.last_pipeline_step_outputs,
                            outputs_before=tape_outputs_before,
                            decision_step_count=tape_decision_steps,
                        ),
                        state=tape_state,
                        actions=decision["actions"],
                    ),
                    execution_payload=build_execution_payload(
                        bar_index=i,
                        fill=fill,
                        fills=fills_since(manager.trades, trades_before_execution),
                        rejected=rejections_since(
                            getattr(manager, "order_events", None),
                            tape_events_before,
                        ),
                    ),
                )
```

(e) Add a method on `HourlyBacktester` (next to `_publish_live_progress` or any other small public helper):

```python
    def decision_tape_summary(self) -> Dict[str, Any]:
        """Counts for the trace's ``run_completed`` summary; ``{}`` when this
        run kept no tape (no run id, or a hosted runtime)."""
        tape = getattr(self, "_decision_tape", None)
        return tape.summary() if tape is not None else {}
```

(Confirm `Any` and `Dict` are already imported from `typing` in engine.py; add them if not.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py -v`
Expected: all PASS. If `test_llm_pipeline_bars_record_intent_and_fallback` fails because the LLM path is not taken, read `PortfolioManager.make_trading_decision_with_llm`'s precondition (~`portfolio_manager.py:406`) — the stub must satisfy it via `execution_service`; fix the **stub**, never `portfolio_manager.py`.

- [ ] **Step 5: Run the engine's neighbours for regressions**

Run: `pytest dashboard/backend/tests/conformance dashboard/backend/tests/test_trace_tool_events.py dashboard/backend/tests/test_architecture_boundaries.py dashboard/backend/tests/test_backtest_worker_schema_skip.py -q`
Expected: same pass/skip/xfail counts as on `main` (conformance: 99 passed, 7 skipped, 25 xfailed), no new failures.

- [ ] **Step 6: Commit**

```bash
git add dashboard/backend/domain/backtesting/engine.py dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py
git commit -m "feat(backtest): record a per-bar decision tape for pipeline runs" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Close the trace — child and parent

**Files:**
- Modify: `dashboard/scripts/backtest_hourly_agent.py:~190` (import) and `:~586` (call)
- Modify: `dashboard/backend/api/routers/backtests.py` (`run_backtest_background`, ~1721–1760 and the `finally` at ~2083)
- Test: `dashboard/backend/tests/test_backtests_router.py` (append), `dashboard/backend/tests/test_trace_decision_tape.py` (already has the service tests)

**Interfaces:**
- Consumes: `trace_lifecycle` (Task 3), `HourlyBacktester.decision_tape_summary` (Task 4), `trace_service.fail_trace_if_running` (Task 2).
- Produces: none for later tasks.

**Why the parent owns the label.** The child's `SystemExit` is the SIGTERM the parent sends from *two* arms — the user's cancel (`_BacktestCancelled`) and the timeout's grace period before SIGKILL (`_signal_backtest_process`, `backtests.py:2833`). The child cannot tell them apart; the parent is in the arm. So the child writes nothing on `SystemExit` (Task 3) and the parent calls `fail_trace_if_running` with one label per arm. Every call is a status-checked no-op on a trace the child already closed, so a completed trace is never re-marked.

| Arm of `run_backtest_background` | label |
|---|---|
| success branch (`returncode == 0`) | `trace_close_failed` — the run finished; the child's own `complete_trace` did not land |
| failure branch (`returncode != 0`) | `run_failed` — no-op when the child recorded it itself |
| `except _BacktestCancelled` | `run_cancelled` |
| `except subprocess.TimeoutExpired` | `run_timed_out` (matches `/backtest/status`'s `timed_out`) |
| outer `finally` | `run_killed` — reached none of the above |

- [ ] **Step 1: Write the failing parent tests**

Append to `dashboard/backend/tests/test_backtests_router.py` (it already defines `bt`, `FakeChild`, `_REAL_RUN_BACKTEST_BACKGROUND`, and imports `subprocess`, `uuid`). `FakeChild(returncode=…, timeout_waits=…)` are the two knobs (`tests/_fake_child.py`):

```python
def _run_parent_with_child(monkeypatch, child, *, run_id):
    """Drive the real run_backtest_background against ``child`` and return
    the (run_id, label) pairs the parent handed fail_trace_if_running."""
    session_id = str(uuid.uuid4())
    assert bt._try_acquire_backtest_slot(live_run_id=run_id, session_id=session_id, user_id=None) is None
    labelled = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)
    monkeypatch.setattr(
        bt.trace_service, "fail_trace_if_running",
        lambda rid, code: labelled.append((rid, code)) or True,
    )
    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
    )
    return labelled


def test_parent_labels_a_timed_out_child_run_timed_out(monkeypatch):
    """The timeout arm SIGTERMs first (grace, then SIGKILL); a child that
    unwinds in the grace exits via SystemExit and writes nothing. The label
    must say timed out -- not cancelled, which is what the child would have
    guessed -- and the finally's run_killed must not fire over it."""
    run_id = "agent_trace_timed_out"
    labelled = _run_parent_with_child(
        monkeypatch, FakeChild(stdout="run header line\n", timeout_waits=1), run_id=run_id,
    )
    assert labelled[0] == (run_id, "run_timed_out")
    assert (run_id, "run_cancelled") not in labelled
    # The finally still runs; fail_trace_if_running is a no-op by then on the
    # real store. Here it is a stub, so the finally's call is visible:
    assert labelled[-1] == (run_id, "run_killed") and len(labelled) == 2


def test_parent_labels_a_failed_child_run_failed(monkeypatch):
    run_id = "agent_trace_failed"
    labelled = _run_parent_with_child(
        monkeypatch, FakeChild(returncode=1, stderr="Traceback: boom\n"), run_id=run_id,
    )
    assert labelled[0] == (run_id, "run_failed")


def test_parent_labels_an_unclosed_completed_run_trace_close_failed(monkeypatch):
    """returncode 0 but the trace still running: the child finished and its
    own complete_trace did not land. Distinct from a kill on purpose."""
    run_id = "agent_trace_unclosed"
    labelled = _run_parent_with_child(
        monkeypatch, FakeChild(returncode=0, stdout="ok\n"), run_id=run_id,
    )
    assert labelled[0] == (run_id, "trace_close_failed")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -k "parent_labels" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'trace_service'`.

- [ ] **Step 3: Implement the parent side**

In `dashboard/backend/api/routers/backtests.py`:

(a) Add with the other `dashboard.backend.domain` imports:

```python
from dashboard.backend.domain.traces import service as trace_service
```

(b) In `run_backtest_background`, find `resolved_live_run_id = live_run_id` (~1721) and add on the next line:

```python
    # Captured separately: the except arm below clears resolved_live_run_id
    # once the slot is finalized, but the finally still needs the id to close
    # a trace the child could not (killed by the timeout or SIGKILL).
    trace_run_id: Optional[str] = None
```

then, right after the block that mints `resolved_live_run_id` when it is empty (the `if not resolved_live_run_id:` assignment ~1739–1743), add:

```python
        trace_run_id = resolved_live_run_id
```

(c) Add a one-line helper next to `_signal_backtest_process`:

```python
def _label_trace(run_id: Optional[str], error_code: str) -> None:
    """Label a trace the child left ``running``. No-op on a closed trace, so
    every arm may call it; the first arm to reach a running trace names it."""
    if run_id:
        trace_service.fail_trace_if_running(run_id, error_code)
```

(d) Call it from each arm, **before** the arm clears `resolved_live_run_id` (use `trace_run_id`, which is never cleared):

- In the `if result.returncode != 0:` branch (~1936), after the `print(f"❌ Backtest failed …")`: `_label_trace(trace_run_id, "run_failed")`.
- In its `else:` branch (~1947), after the `print(f"✅ Backtest completed …")`: `_label_trace(trace_run_id, "trace_close_failed")`.
- In `except _BacktestCancelled:` (~1961), first statement: `_label_trace(trace_run_id, "run_cancelled")`.
- In `except subprocess.TimeoutExpired:` (~1970), first statement: `_label_trace(trace_run_id, "run_timed_out")`.
- In the outer `finally:` (~2083), **first** statement:

```python
        # Reached none of the arms above (SIGKILL with no grace, a parent
        # exception): the only label left is a kill. No-op otherwise.
        _label_trace(trace_run_id, "run_killed")
```

Make sure `trace_run_id = resolved_live_run_id` sits inside the `try:` whose `finally` you edited, and that `trace_run_id: Optional[str] = None` is assigned before that `try:` (so the `finally` never sees an unbound name). `Optional` is already imported in this module. Do **not** label from the generic `except Exception` arm: that arm is a parent-side failure, the child may still be alive and closing its own trace, and the `finally`'s `run_killed` is the honest label if it is not.

- [ ] **Step 4: Run them to verify they pass**

Run: `pytest dashboard/backend/tests/test_backtests_router.py -k "parent_labels or test_pipeline_timeout_finalizes_execution_and_slot_once" -v`
Expected: all PASS.

- [ ] **Step 5: Implement the child side**

In `dashboard/scripts/backtest_hourly_agent.py`, next to `from dashboard.backend.domain.backtesting.engine import HourlyBacktester` (~190), add:

```python
from dashboard.backend.domain.backtesting.decision_tape import trace_lifecycle
```

Then open the context **before** the `backtester = HourlyBacktester(` call (~508; the trace is created inside that constructor, so `load_data()` / `calculate_indicators()` failures must be inside the block too) and close it after the `run_agent_backtest` `try/finally` (~596). Concretely, replace:

```python
    # Initialize backtester (with LLM if available and enabled)
    # Note: dates are validated in __init__ if they somehow got reversed again
    backtester = HourlyBacktester(
```

with:

```python
    # The run's agent trace is created in HourlyBacktester.__init__ and closed
    # here: completed with the decision tape's counts, run_failed on an
    # exception anywhere from the constructor through the run (a SIP refusal
    # in load_data() is a failure, not a kill). The SIGTERM SystemExit writes
    # nothing -- the parent sends it from both the cancel and the timeout
    # arms and labels the trace by the arm (api/routers/backtests.py).
    tape_owner = {"backtester": None}

    def _tape_summary():
        bt = tape_owner["backtester"]
        return bt.decision_tape_summary() if bt is not None else {}

    with trace_lifecycle(args.run_id, summary=_tape_summary):
      # Initialize backtester (with LLM if available and enabled)
      # Note: dates are validated in __init__ if they somehow got reversed again
      backtester = HourlyBacktester(
```

and indent everything from that call through the end of the existing

```python
    try:
        agent_id, agent_eq = backtester.run_agent_backtest()
    finally:
        …
                raise
```

by one level, adding `tape_owner["backtester"] = backtester` as the first statement after the constructor call closes. The block ends after that `finally`; the holdings DEBUG print and the baselines below it stay at function level. The re-indent is ~90 lines and is the whole cost of spanning the constructor; `test_backtest_launch_phases.py` AST-checks the *import* block only, so it is unaffected, but run it (Step 6).

- [ ] **Step 6: Run the launch-script guards and router suite**

Run: `pytest dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_cancel.py -q`
Expected: no new failures. If `test_backtest_launch_phases.py` asserts on the import block's shape (it AST-checks that the steady-clock marks bracket the imports), keep the new import inside that bracket next to the engine import.

- [ ] **Step 7: Commit**

```bash
git add dashboard/scripts/backtest_hourly_agent.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(backtest): close the agent trace when a dashboard backtest ends" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Full suite, docs, PR

**Files:**
- Modify: `docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md` (status line; driver label; service raising note)
- No `CLAUDE.md` change: the spec is the record.

- [ ] **Step 1: Sync the spec with the implementation**

In the spec: set `Status: implemented`; in "domain/traces/service.py" replace the sentence about best-effort `record_tape_decision`/`record_tape_execution` with: "`record_tape_bar(trace_id, ...)` writes the pair and raises; `DecisionTapeRecorder` is the one swallow-and-count layer. `finish_trace_best_effort` and `fail_trace_if_running` never raise." Check the "Follow-ups this spec hands off" list still matches what this PR did **not** do (no adjustment, no latency case, no PM envelope change).

- [ ] **Step 2: Full backend suite**

Run: `pytest dashboard/backend/tests/ -q -p no:cacheprovider`
Expected: green except the known-environmental `test_report_pdf` failures (local python lacks reportlab — see project memory); compare the failing set against a run on `main` if anything else is red. Also confirm `git status` shows no change to `dashboard/storage/data/backtest.db` — that file changing is the lazy-import rule (Review Focus 8) failing in practice even if its test passed.

- [ ] **Step 3: Commit, push, open PR**

```bash
git add docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md docs/superpowers/plans/2026-10-08-backtest-decision-tape.md
git commit -m "docs(backtest): decision tape spec matches implementation" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push -u origin feat/backtest-decision-tape
gh pr create --title "feat(backtest): per-bar decision tape" --body-file <scratchpad>/pr-body.md
```

PR body (short): what the tape records and where (trace events); **`intent` + `state` is the replay unit, `actions` the same-engine reproduction check**, with the three Task 4 tests named as the evidence (reproduction, intent replay, perturbed-engine divergence); the per-run `gate_rewrites` count; the parent's label-per-arm table; and the realism sentence — the recorded engine is frictionless, latency-free and on unadjusted bars, realism is scored by the conformance suite, and the four follow-ups are listed in the spec (adjusted bars with provenance first). End with the attribution footer `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Step 4: File the follow-ups as issues at merge** (ask before filing on the shared repo — it assigns work): adjusted US bars + `market_data_adjustment` provenance; a decision-latency conformance case; a driver field on PM's return envelope; realism scoring of the leaderboard medians. Link them from the PR's closing comment so the trail connects both ways.
