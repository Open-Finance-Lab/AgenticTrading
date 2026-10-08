# Backtest Decision Tape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every decision bar of a My Agents pipeline backtest — raw model orders, post-gate actions, pre-decision portfolio, fill plan, fills, rejections — as `decision_recorded` / `execution_result` event pairs in the run's #622 agent trace, so a run's order sequence can be replayed through another engine.

**Architecture:** A new pure module `domain/backtesting/decision_tape.py` builds bounded, allow-listed payloads and owns a `DecisionTapeRecorder` that swallows every failure. `HourlyBacktester.run_agent_backtest` snapshots ledger state before each decision and hands the recorder one bar after `execute_actions`. `domain/traces/service.py` gains a raising `record_tape_bar`, a `load_decision_tape` reader, and two best-effort lifecycle helpers; the child script finalizes the trace around `run_agent_backtest`, and the parent route fails a trace the child left `running`.

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
- Tests run from the repo root: `pytest dashboard/backend/tests/... -v`. `tests/conftest.py` already strips `CONTENT_DATABASE_URL`, so the trace store is SQLite in tests.
- Commits end with the session's attribution lines.

## Review Focus

1. **Stale pipeline output on a rule-based or fallback bar** — a bar that did not call the pipeline must record `intent: null`, not the previous bar's model orders. Pinned in Task 1 (`test_intent_is_none_when_outputs_list_is_unchanged`) and Task 4 (`test_llm_pipeline_bars_record_intent_and_fallback`).
2. **A repeated same-day rejection** collapses into `repeat_count` on the first `order_events` record; the later bar must still list it. Pinned in Task 1 (`test_rejections_since_reports_a_collapsed_repeat`) and Task 4 (`test_collapsed_repeat_rejection_appears_on_its_own_bar`).
3. **Model JSON carrying a sensitive-looking key** (`api_key`, `max_tokens`) or numpy scalars — the store would raise and the bar's events vanish. Pinned in Task 1 (`test_intent_allow_list_drops_unknown_and_sensitive_keys`, `test_jsonable_converts_numpy_and_timestamps`).
4. **Trace store down or slow mid-run** — the backtest must finish with an identical curve. Pinned in Task 4 (`test_broken_trace_writes_do_not_change_the_run`).
5. **Child killed by the timeout** — the trace must not stay `running` forever; a completed trace must never be re-marked failed. Pinned in Task 5 (`test_parent_fails_a_trace_the_child_left_running`, `test_fail_trace_if_running_leaves_a_completed_trace_alone`).

---

## File Structure

| File | Responsibility |
|---|---|
| `dashboard/backend/domain/backtesting/decision_tape.py` (create) | Pure payload builders, size bounding, `DecisionTapeRecorder`, `run_with_trace_lifecycle`. |
| `dashboard/backend/domain/traces/service.py` (modify) | `record_tape_bar`, `load_decision_tape`, `finish_trace_best_effort`, `fail_trace_if_running`. |
| `dashboard/backend/domain/backtesting/engine.py` (modify) | Snapshot before decision, record after execution, `decision_tape_summary()`. |
| `dashboard/scripts/backtest_hourly_agent.py` (modify) | Wrap `run_agent_backtest` in `run_with_trace_lifecycle`. |
| `dashboard/backend/api/routers/backtests.py` (modify) | `fail_trace_if_running(..., "run_killed")` in `run_backtest_background`'s `finally`. |
| `dashboard/backend/tests/domain/backtesting/test_decision_tape.py` (create) | Unit tests for builders, recorder, lifecycle. |
| `dashboard/backend/tests/test_trace_decision_tape.py` (create) | Service tests on a real SQLite `TraceStore`. |
| `dashboard/backend/tests/domain/backtesting/test_decision_tape_engine.py` (create) | Engine integration + replay round-trip. |
| `dashboard/backend/tests/test_backtests_router.py` (modify) | Parent `run_killed` test. |

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
  - `build_decision_payload(*, bar_index: int, decision_at, driver: str, intent: Optional[dict], state: dict, actions) -> dict`
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


def test_decision_and_execution_payload_shapes():
    actions = [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "why", "confidence": 0.7, "extra": 1}]
    decision = tape.build_decision_payload(
        bar_index=4, decision_at=pd.Timestamp("2026-03-02 11:30"), driver="llm",
        intent={"orders": []}, state={"cash": 1.0, "equity": 1.0, "positions": []}, actions=actions,
    )
    assert decision["tape_version"] == 1 and decision["bar_index"] == 4
    assert decision["actions"] == [{"symbol": "AAPL", "action": "buy", "shares": 3, "reason": "why", "confidence": 0.7}]
    assert decision["reasoning_summaries"] == ["why"] and decision["accepted"] is True

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
git commit -m "feat(backtest): decision tape payload builders"
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
git commit -m "feat(traces): tape bar writer, reader and best-effort finalizers"
```

---

### Task 3: Recorder and trace lifecycle wrapper

**Files:**
- Modify: `dashboard/backend/domain/backtesting/decision_tape.py` (append)
- Test: `dashboard/backend/tests/domain/backtesting/test_decision_tape.py` (append)

**Interfaces:**
- Consumes: Task 1 builders; Task 2 `trace_service.trace_for_run`, `record_tape_bar`, `finish_trace_best_effort`.
- Produces:
  - `class DecisionTapeRecorder(run_id: str)` with `record_bar(*, bar_index: int, decision_payload: dict, execution_payload: dict) -> None` (never raises) and `summary() -> dict` returning `{"tape_version", "traced", "bars_recorded", "write_failures", "oversize_skipped"}`.
  - `run_with_trace_lifecycle(run_id: Optional[str], run: Callable[[], T], *, summary: Callable[[], dict]) -> T`.

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


def _bar(i):
    return {"bar_index": i, "decision_payload": {"tape_version": 1}, "execution_payload": {"tape_version": 1}}


def test_recorder_resolves_trace_once_and_writes_each_bar(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")
    for i in range(3):
        recorder.record_bar(**_bar(i))
    assert fake.lookups == 1
    assert [w["bar_index"] for w in fake.writes] == [0, 1, 2]
    assert fake.writes[0]["trace_id"] == "trc_1" and fake.writes[0]["run_id"] == "agent_x"
    assert recorder.summary() == {
        "tape_version": 1, "traced": True, "bars_recorded": 3,
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


def test_lifecycle_completes_with_summary(monkeypatch):
    calls = []
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda run_id, **kw: calls.append((run_id, kw)) or True)
    result = tape.run_with_trace_lifecycle("agent_x", lambda: ("id", []), summary=lambda: {"bars_recorded": 4})
    assert result == ("id", [])
    assert calls == [("agent_x", {"result_summary": {"bars_recorded": 4}})]


@pytest.mark.parametrize(
    "error, code",
    [(RuntimeError("boom"), "run_failed"), (SystemExit(143), "run_cancelled")],
)
def test_lifecycle_fails_and_reraises(monkeypatch, error, code):
    calls = []
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda run_id, **kw: calls.append((run_id, kw)) or True)

    def run():
        raise error

    with pytest.raises(type(error)):
        tape.run_with_trace_lifecycle("agent_x", run, summary=lambda: {})
    assert calls == [("agent_x", {"error_code": code})]


def test_lifecycle_without_run_id_touches_no_trace(monkeypatch):
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("no run id, no trace"))
    assert tape.run_with_trace_lifecycle(None, lambda: 5, summary=lambda: {}) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: new tests FAIL — `AttributeError: ... has no attribute 'DecisionTapeRecorder'`.

- [ ] **Step 3: Implement**

In `decision_tape.py`, extend the typing import to `from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TypeVar`, add below the `pipeline_output_to_decision` import:

```python
from dashboard.backend.domain.traces import service as trace_service
```

and append:

```python
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
        self.write_failures = 0
        self.oversize_skipped = 0

    def _resolve_trace(self) -> Optional[str]:
        if not self._resolved:
            trace = trace_service.trace_for_run(self.run_id)
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
            trace_service.record_tape_bar(
                trace_id=trace_id,
                run_id=self.run_id,
                bar_index=bar_index,
                decision_payload=decision,
                execution_payload=execution,
            )
            self.bars_recorded += 1
        except Exception as exc:  # observational: never reaches the bar loop
            self.write_failures += 1
            self._warn(exc)

    def summary(self) -> Dict[str, Any]:
        return {
            "tape_version": TAPE_VERSION,
            "traced": self._trace_id is not None,
            "bars_recorded": self.bars_recorded,
            "write_failures": self.write_failures,
            "oversize_skipped": self.oversize_skipped,
        }


T = TypeVar("T")


def run_with_trace_lifecycle(
    run_id: Optional[str],
    run: Callable[[], T],
    *,
    summary: Callable[[], Dict[str, Any]],
) -> T:
    """Run ``run`` and close the run's trace on the way out: completed with
    ``summary()``, failed on an exception, cancelled on ``SystemExit`` (the
    dashboard's SIGTERM, see ``_exit_on_sigterm``). A kill that skips this is
    caught by the parent's ``fail_trace_if_running``."""
    if not run_id:
        return run()
    try:
        result = run()
    except SystemExit:
        trace_service.finish_trace_best_effort(run_id, error_code="run_cancelled")
        raise
    except BaseException:
        trace_service.finish_trace_best_effort(run_id, error_code="run_failed")
        raise
    try:
        result_summary = summary()
    except Exception:
        result_summary = {}
    trace_service.finish_trace_best_effort(run_id, result_summary=result_summary)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest dashboard/backend/tests/domain/backtesting/test_decision_tape.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/backend/domain/backtesting/decision_tape.py dashboard/backend/tests/domain/backtesting/test_decision_tape.py
git commit -m "feat(backtest): decision tape recorder and trace lifecycle"
```

---

### Task 4: Engine wiring and replay round-trip

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

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.traces import service as trace_service
from dashboard.backend.domain.traces.repository import TraceStore
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


def _run(monkeypatch, *, run_id="agent_tape_engine", decide=None, llm_client=None, pipeline=None):
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
    backtester.load_data()
    backtester.calculate_indicators()
    _run_id, curve = backtester.run_agent_backtest()
    return backtester, holder, curve


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
    assert [(f["symbol"], f["side"], f["shares"]) for f in first["execution"]["fills"]] == [("AAPL", "BUY", 3)]
    assert set(first["execution"]["fill_plan"]) == {"bar", "price_field", "filled_at"}
    assert tape[1]["decision"]["state"]["positions"] == [["AAPL", 3]]
    summary = backtester.decision_tape_summary()
    assert summary["bars_recorded"] == holder.calls and summary["write_failures"] == 0


def test_collapsed_repeat_rejection_appears_on_its_own_bar(monkeypatch, store):
    _run(monkeypatch, decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_engine")
    second, third = tape[2]["execution"]["rejected"], tape[3]["execution"]["rejected"]
    assert [(r["symbol"], r["reason"], r["collapsed"]) for r in second] == [("AAPL", "insufficient_cash", False)]
    assert [(r["symbol"], r["reason"], r["collapsed"], r["count"]) for r in third] == [
        ("AAPL", "insufficient_cash", True, 1)
    ]


def test_tape_replays_to_identical_trades_and_curve(monkeypatch, store):
    """The proof the tape is replay-grade: a second run driven only by the
    recorded actions reproduces the first run's trades and equity curve."""
    _bt, first, first_curve = _run(monkeypatch, run_id="agent_tape_original", decide=_script)
    tape = trace_service.load_decision_tape("agent_tape_original")

    def replay(index, pm):
        return tape[index]["decision"]["actions"]

    _bt, second, second_curve = _run(monkeypatch, run_id="agent_tape_replay", decide=replay)

    def comparable(trades):
        return json.dumps(
            [{k: str(v) for k, v in trade.items()} for trade in trades], sort_keys=True
        )

    assert first.pm is not second.pm
    assert first.pm.trades, "the script must trade, or the replay proves nothing"
    assert comparable(second.pm.trades) == comparable(first.pm.trades)
    assert [p["equity"] for p in second_curve] == [p["equity"] for p in first_curve]


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

    assert tape[1]["decision"]["driver"] == "llm_fallback"
    assert tape[1]["decision"]["intent"] == {"unparsed": True, "completed_steps": 0}


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
git commit -m "feat(backtest): record a per-bar decision tape for pipeline runs"
```

---

### Task 5: Close the trace — child and parent

**Files:**
- Modify: `dashboard/scripts/backtest_hourly_agent.py:~190` (import) and `:~586` (call)
- Modify: `dashboard/backend/api/routers/backtests.py` (`run_backtest_background`, ~1721–1760 and the `finally` at ~2083)
- Test: `dashboard/backend/tests/test_backtests_router.py` (append), `dashboard/backend/tests/test_trace_decision_tape.py` (already has the service tests)

**Interfaces:**
- Consumes: `run_with_trace_lifecycle` (Task 3), `HourlyBacktester.decision_tape_summary` (Task 4), `trace_service.fail_trace_if_running` (Task 2).
- Produces: none for later tasks.

- [ ] **Step 1: Write the failing parent test**

Append to `dashboard/backend/tests/test_backtests_router.py` (it already defines `bt`, `FakeChild`, `_REAL_RUN_BACKTEST_BACKGROUND`, and imports `subprocess`, `uuid`):

```python
def test_parent_fails_a_trace_the_child_left_running(monkeypatch):
    """A child killed by the timeout cannot close its own trace; the parent's
    finally does, and only while the trace still reads ``running``."""
    run_id = "agent_trace_killed"
    session_id = str(uuid.uuid4())
    assert bt._try_acquire_backtest_slot(live_run_id=run_id, session_id=session_id, user_id=None) is None

    failed = []
    child = FakeChild(stdout="run header line\n", timeout_waits=1)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: child)
    monkeypatch.setattr(bt, "run_backtest_background", _REAL_RUN_BACKTEST_BACKGROUND)
    monkeypatch.setattr(
        bt.trace_service, "fail_trace_if_running",
        lambda rid, code: failed.append((rid, code)) or True,
    )

    bt.run_backtest_background(
        start_date="2026-01-01",
        end_date="2026-01-02",
        session_id=session_id,
        live_run_id=run_id,
        decision_source="rule_based",
    )

    assert failed == [(run_id, "run_killed")]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest dashboard/backend/tests/test_backtests_router.py::test_parent_fails_a_trace_the_child_left_running -v`
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

(c) In the function's outer `finally:` (~2083), add as its **first** statement:

```python
        if trace_run_id:
            # No-op unless the child left its trace running (a clean exit and
            # a failed run both close it themselves; see decision_tape.py
            # run_with_trace_lifecycle). Never raises.
            trace_service.fail_trace_if_running(trace_run_id, "run_killed")
```

Make sure `trace_run_id = resolved_live_run_id` sits inside the `try:` whose `finally` you edited, and that `trace_run_id: Optional[str] = None` is assigned before that `try:` (so the `finally` never sees an unbound name). `Optional` is already imported in this module.

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest dashboard/backend/tests/test_backtests_router.py::test_parent_fails_a_trace_the_child_left_running dashboard/backend/tests/test_backtests_router.py::test_pipeline_timeout_finalizes_execution_and_slot_once -v`
Expected: both PASS.

- [ ] **Step 5: Implement the child side**

In `dashboard/scripts/backtest_hourly_agent.py`, next to `from dashboard.backend.domain.backtesting.engine import HourlyBacktester` (~190), add:

```python
from dashboard.backend.domain.backtesting.decision_tape import run_with_trace_lifecycle
```

and replace (~586):

```python
    try:
        agent_id, agent_eq = backtester.run_agent_backtest()
    finally:
```

with:

```python
    try:
        # Closes the run's agent trace: completed with the decision tape's
        # counts, failed on an exception, cancelled on the SIGTERM SystemExit.
        agent_id, agent_eq = run_with_trace_lifecycle(
            args.run_id,
            backtester.run_agent_backtest,
            summary=backtester.decision_tape_summary,
        )
    finally:
```

- [ ] **Step 6: Run the launch-script guards and router suite**

Run: `pytest dashboard/backend/tests/test_backtest_launch_phases.py dashboard/backend/tests/test_backtests_router.py dashboard/backend/tests/test_backtest_cancel.py -q`
Expected: no new failures. If `test_backtest_launch_phases.py` asserts on the import block's shape (it AST-checks that the steady-clock marks bracket the imports), keep the new import inside that bracket next to the engine import.

- [ ] **Step 7: Commit**

```bash
git add dashboard/scripts/backtest_hourly_agent.py dashboard/backend/api/routers/backtests.py dashboard/backend/tests/test_backtests_router.py
git commit -m "feat(backtest): close the agent trace when a dashboard backtest ends"
```

---

### Task 6: Full suite, docs, PR

**Files:**
- Modify: `docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md` (status line; driver label; service raising note)
- No `CLAUDE.md` change: the spec is the record.

- [ ] **Step 1: Sync the spec with the implementation**

In the spec: set `Status: implemented`; replace `"llm_fallback_rule_based"` with `"llm_fallback"` (a fallback bar may also hold with no actions, so the label names who did not drive it rather than what replaced it); in "domain/traces/service.py" replace the sentence about best-effort `record_tape_decision`/`record_tape_execution` with: "`record_tape_bar(trace_id, ...)` writes the pair and raises; `DecisionTapeRecorder` is the one swallow-and-count layer. `finish_trace_best_effort` and `fail_trace_if_running` never raise."

- [ ] **Step 2: Full backend suite**

Run: `pytest dashboard/backend/tests/ -q -p no:cacheprovider`
Expected: green except the known-environmental `test_report_pdf` failures (local python lacks reportlab — see project memory); compare the failing set against a run on `main` if anything else is red. Also confirm `git status` shows no change to `dashboard/storage/data/backtest.db`.

- [ ] **Step 3: Commit, push, open PR**

```bash
git add docs/superpowers/specs/2026-10-08-backtest-decision-tape-design.md docs/superpowers/plans/2026-10-08-backtest-decision-tape.md
git commit -m "docs(backtest): decision tape spec matches implementation"
git push -u origin feat/backtest-decision-tape
gh pr create --title "feat(backtest): per-bar decision tape" --body-file <scratchpad>/pr-body.md
```

PR body (short): what the tape records, where (trace events), the replay round-trip test as the proof, the two side findings (no US fees/slippage/volume cap; gate drops/rewrites model orders unrecorded), and the attribution footer.
