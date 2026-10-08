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
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

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


def _raw_json(value: Any) -> str:
    """The exact JSON text of a model-supplied value. ``json.loads`` produced
    it, so ``json.dumps`` with ``allow_nan`` round-trips it -- NaN and
    Infinity included, which the store's strict JSON cannot carry as numbers."""
    try:
        return json.dumps(value, allow_nan=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value))


def _intent_order(entry: Any) -> Dict[str, Any]:
    """One raw model order, recorded so ``orders_for_replay`` hands the gate
    exactly what the model handed it.

    Not ``_pick``: the gate in ``make_trading_decision_with_llm`` reads each
    field with a default, so dropping a key changes the outcome -- a present
    ``confidence: null`` raises in the gate's format string (the bar falls back
    to rule-based) while an absent one reads as 0.5 and trades. So:

    - an absent field stays absent, and a ``null`` one stays ``null``;
    - a string ``reasoning`` is cut to ``MAX_REASONING_CHARS`` (the gate only
      interpolates it into the action's ``reason`` text);
    - a value the store cannot carry as itself -- a non-finite number, an
      object or array -- is recorded readably (``null`` / ``str()``) and its
      exact JSON text is kept under ``raw[field]``;
    - a non-object entry (the gate raises on it) is kept whole as
      ``{"raw_entry": <json>}``.
    """
    if not isinstance(entry, Mapping):
        return {"raw_entry": _raw_json(entry)}
    order: Dict[str, Any] = {}
    raw: Dict[str, str] = {}
    for field in _INTENT_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if value is None or isinstance(value, (bool, int)):
            order[field] = value
        elif isinstance(value, str):
            order[field] = value[:MAX_REASONING_CHARS] if field == "reasoning" else value
        elif isinstance(value, float) and math.isfinite(value):
            order[field] = value
        else:
            order[field] = None if isinstance(value, float) else str(value)[:MAX_REASONING_CHARS]
            raw[field] = _raw_json(value)
    if raw:
        order["raw"] = raw
    return order


def orders_for_replay(intent: Optional[Mapping[str, Any]]) -> List[Any]:
    """The model orders of one tape bar, as the gate first received them:
    ``raw`` fields and ``raw_entry`` entries restored from their JSON text.
    Raises ``ValueError`` on an order ``bound_payload`` had to cut a raw value
    from (``unreplayable``) -- replaying it would silently differ."""
    if not isinstance(intent, Mapping) or "orders" not in intent:
        return []
    replay: List[Any] = []
    for order in intent.get("orders") or ():
        if isinstance(order, Mapping) and "raw_entry" in order:
            replay.append(json.loads(order["raw_entry"]))
            continue
        if order.get("unreplayable"):
            raise ValueError(f"tape order cannot be replayed exactly: {order['unreplayable']}")
        restored = {key: value for key, value in order.items() if key != "raw"}
        for field, text in (order.get("raw") or {}).items():
            restored[field] = json.loads(text)
        replay.append(restored)
    return replay


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
    return {"orders": [_intent_order(action) for action in decision.get("actions") or []]}


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
    the order carries none or an unparseable one (the model supplies it, so it
    can be any text), and a ``None`` size matches any size."""
    triples = []
    for order in orders or ():
        if not isinstance(order, Mapping):
            continue
        side = str(order.get("action") or order.get("side") or "").lower()
        if side not in _ORDER_SIDES:
            continue  # hold / unknown: not an order the gate can rewrite
        try:
            size = None if order.get(size_field) is None else float(order.get(size_field))
        except (TypeError, ValueError, OverflowError):
            size = None  # model-supplied junk ("ten", "[1, 2]"): matches any size
        triples.append((str(order.get("symbol") or "").upper(), side, size))
    return sorted(triples, key=lambda t: (t[0], t[1], -1.0 if t[2] is None else t[2]))


def gate_rewrote(intent: Optional[Mapping[str, Any]], actions: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the pre-trade gate changed what the model asked for.

    Compares the sorted ``(symbol, side, size)`` triples of ``intent.orders``
    (size = ``position_size``) with those of the post-gate ``actions`` (size =
    ``shares``). A dropped, added or resized order, or a SELL widened to the
    whole position, is a rewrite. ``False`` with no intent or an unparsed one:
    a rule-based or fallback bar had no model order to rewrite. An intent
    order without a size compares on symbol and side only.

    ``build_decision_payload`` applies this only on ``driver == "llm"`` bars:
    on an ``llm_fallback`` bar the gate raised and the actions are rule-based
    substitutes, not a rewrite of the orders, so ``gate_rewrites`` counts the
    gate's rewrites of bars the model drove. Fallbacks are counted by driver.
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
        "gate_rewrote": driver == DRIVER_LLM and gate_rewrote(intent, picked),
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
    orders alone do not fit.

    Dropping a string ``reasoning`` keeps the replay exact (the gate reads an
    absent one as ``""`` and uses it only in ``reason`` text). A ``null`` one is
    kept, since the gate raises on it. A non-string one cut from ``raw`` marks
    the order ``unreplayable`` so ``orders_for_replay`` refuses it."""
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
            raw = order.get("raw") or {}
            if "reasoning" in raw:
                raw.pop("reasoning")
                if not raw:
                    order.pop("raw")
                order.pop("reasoning", None)
                order["unreplayable"] = ["reasoning"]
            elif isinstance(order.get("reasoning"), str):
                order.pop("reasoning")
    if _payload_size(trimmed) <= MAX_PAYLOAD_BYTES:
        return trimmed
    return None


def _trace_service():
    """Lazy on purpose: ``domain/traces/service.py`` constructs its store on
    import (schema DDL on DB_PATH, or a Neon dial). The engine imports this
    module, and the engine is imported by scripts that must not touch a
    database (``diff_backtest_runs.py``, a ``python -c`` probe). Same pattern
    as ``engine.py`` and ``market_data/provider.py``. Returns the module
    object, so tests patching attributes on it are honoured."""
    from dashboard.backend.domain.traces import service as trace_service

    return trace_service


# Consecutive store failures after which a run stops calling the trace store.
# Swallowing an exception bounds what a broken store costs the run only when
# the store fails fast; one that hangs (a Neon dial or a TCP timeout) costs
# that hang on every bar, and enough bars of it push the run into the
# subprocess timeout -- the tape would then change the run's outcome. Three
# rides out a transient blip and caps a hung store at three waits per run.
MAX_CONSECUTIVE_STORE_FAILURES = 3


class DecisionTapeRecorder:
    """Writes one run's tape. Never raises: a broken trace store costs the
    tape, never the backtest. The trace is looked up once; a lookup that
    raised (a transient store error) is retried on the next bar, one that
    found no trace turns the recorder into a no-op for the run. After
    ``MAX_CONSECUTIVE_STORE_FAILURES`` failed bars in a row the recorder
    suspends for the rest of the run and counts each later bar in
    ``suspended_skipped`` without touching the store."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._trace_id: Optional[str] = None
        self._resolved = False
        self._warned = False
        self._consecutive_failures = 0
        self._suspended = False
        self.bars_recorded = 0
        self.gate_rewrites = 0
        self.write_failures = 0
        self.oversize_skipped = 0
        self.suspended_skipped = 0

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

    def _store_failed(self, exc: BaseException) -> None:
        self.write_failures += 1
        self._warn(exc)
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_STORE_FAILURES:
            self._suspended = True
            print(
                f"   ⚠️  decision tape suspended after {self._consecutive_failures} "
                "consecutive failures; the rest of this run is not taped",
                flush=True,
            )

    def record_bar(
        self,
        *,
        bar_index: int,
        decision_payload: Dict[str, Any],
        execution_payload: Dict[str, Any],
    ) -> None:
        if self._suspended:
            self.suspended_skipped += 1
            return
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
            self._consecutive_failures = 0
            self.bars_recorded += 1
            if decision.get("gate_rewrote"):
                self.gate_rewrites += 1
        except Exception as exc:  # observational: never reaches the bar loop
            self._store_failed(exc)

    def record_bar_from(
        self,
        build: Callable[[], Tuple[Dict[str, Any], Dict[str, Any]]],
        *,
        bar_index: int,
    ) -> None:
        """``record_bar`` with the payload builders inside the swallow layer:
        they read model-supplied text, so a builder bug must cost a tape bar,
        never the backtest. A suspended recorder skips the builders too."""
        if self._suspended:
            self.suspended_skipped += 1
            return
        try:
            decision_payload, execution_payload = build()
        except Exception as exc:
            self.write_failures += 1
            self._warn(exc)
            return
        self.record_bar(
            bar_index=bar_index,
            decision_payload=decision_payload,
            execution_payload=execution_payload,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "tape_version": TAPE_VERSION,
            "traced": self._trace_id is not None,
            "bars_recorded": self.bars_recorded,
            "gate_rewrites": self.gate_rewrites,
            "write_failures": self.write_failures,
            "oversize_skipped": self.oversize_skipped,
            "suspended_skipped": self.suspended_skipped,
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
