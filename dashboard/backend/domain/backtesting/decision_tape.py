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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from dashboard.backend.domain.trading.execution import _trading_date
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    pipeline_output_to_decision,
)

TAPE_VERSION = 1
MAX_REASONING_CHARS = 500
# The longest model-supplied value kept verbatim for replay. Longer ones are
# recorded readably (cut to ``MAX_REASONING_CHARS``) and the order is marked
# ``unreplayable``: one runaway field must cost that order its exact replay,
# never the whole bar its place on the tape (``bound_payload`` drops a bar its
# orders alone cannot fit).
MAX_RAW_CHARS = 2000
# Bars buffered before one batched store write. Per-bar appends cost ~10 store
# round trips a bar inside the decision loop; batching makes that ~1/25th.
TAPE_FLUSH_BARS = 25
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
# Free text on a rejection: ``strategy_reason`` embeds the action's ``reason``,
# which carries the model's reasoning, so it is cut like any other reasoning.
_REJECTION_TEXT_FIELDS = ("reason", "strategy_reason")


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
      ``{"raw_entry": <json>}``;
    - anything whose exact text exceeds ``MAX_RAW_CHARS`` is kept readably
      only, and the order lists it under ``unreplayable``.
    """
    if not isinstance(entry, Mapping):
        text = _raw_json(entry)
        if len(text) > MAX_RAW_CHARS:
            return {"raw_entry_preview": text[:MAX_REASONING_CHARS], "unreplayable": ["raw_entry"]}
        return {"raw_entry": text}
    order: Dict[str, Any] = {}
    raw: Dict[str, str] = {}
    unreplayable: List[str] = []
    for field in _INTENT_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if value is None or isinstance(value, (bool, int)):
            order[field] = value
        elif isinstance(value, str):
            if field == "reasoning":
                order[field] = value[:MAX_REASONING_CHARS]
            elif len(value) > MAX_RAW_CHARS:
                order[field] = value[:MAX_REASONING_CHARS]
                unreplayable.append(field)
            else:
                order[field] = value
        elif isinstance(value, float) and math.isfinite(value):
            order[field] = value
        else:
            order[field] = None if isinstance(value, float) else str(value)[:MAX_REASONING_CHARS]
            text = _raw_json(value)
            if len(text) > MAX_RAW_CHARS:
                unreplayable.append(field)
            else:
                raw[field] = text
    if raw:
        order["raw"] = raw
    if unreplayable:
        order["unreplayable"] = unreplayable
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
        if order.get("unreplayable"):
            raise ValueError(f"tape order cannot be replayed exactly: {order['unreplayable']}")
        if "raw_entry" in order:
            replay.append(json.loads(order["raw_entry"]))
            continue
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


class EventSnapshot(NamedTuple):
    """Repeat counts of the order events a bar could still bump: ``counts[k]``
    is the count of ``order_events[start + k]``."""

    start: int
    counts: Tuple[int, ...]


def _repeat_count(event: Mapping[str, Any]) -> int:
    return int(event.get("repeat_count", 1) or 1)


def order_events_snapshot(
    order_events: Optional[Sequence[Mapping[str, Any]]], since: Any = None
) -> EventSnapshot:
    """Snapshot of the events a decision at ``since`` could collapse into.

    A rejection collapses only into a record with the same trading date
    (``trading/execution.py:_repeat_key``), and a bar fills on or after its
    decision, so only events dated on or after ``since``'s date can grow. The
    ledger is chronological, so those are a suffix, found by walking back from
    the end -- one trading day of events per bar, not the whole run. With no
    readable ``since`` the snapshot covers every event.
    """
    events = order_events or ()
    start = 0
    if since is not None:
        try:
            since_date = _trading_date(since)
        except TypeError:
            since_date = None
        if since_date is not None:
            start = len(events)
            while start > 0:
                try:
                    if _trading_date(events[start - 1].get("timestamp")) < since_date:
                        break
                except TypeError:
                    pass  # undated: never collapsed into, so harmless to include
                start -= 1
    return EventSnapshot(start, tuple(_repeat_count(events[i]) for i in range(start, len(events))))


def _rejection(event: Mapping[str, Any]) -> Dict[str, Any]:
    return _pick(event, _REJECTION_FIELDS, text_fields=_REJECTION_TEXT_FIELDS)


# A collapsed repeat shares only its collapse key with the record it bumped
# (symbol, side, reason, trading date); status follows from the reason.
_COLLAPSED_FIELDS = ("symbol", "side", "status", "reason")


def _collapsed_rejection(
    event: Mapping[str, Any], grew: int, actions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """A rejection that only bumped an earlier record's ``repeat_count``.

    That record describes the *first* rejection of the day -- its timestamp,
    size, price and the reasoning of the bar that made it -- so copying it would
    put an earlier bar's order on this bar. Only the collapse key is copied, the
    original's time is named ``first_rejected_at``, and the size is this bar's
    own order when exactly one action matches it."""
    rejection = _pick(event, _COLLAPSED_FIELDS, text_fields=("reason",))
    if event.get("timestamp") is not None:
        rejection["first_rejected_at"] = jsonable(event["timestamp"])
    symbol = str(event.get("symbol") or "").strip().upper()
    side = str(event.get("side") or "").strip().upper()
    matches = [
        action for action in actions or ()
        if isinstance(action, Mapping)
        and str(action.get("symbol") or "").strip().upper() == symbol
        and str(action.get("action") or "").strip().upper() == side
    ]
    if len(matches) == 1 and matches[0].get("shares") is not None:
        rejection["requested_shares"] = jsonable(matches[0]["shares"])
    rejection.update(count=grew, collapsed=True)
    return rejection


def rejections_since(
    order_events: Optional[Sequence[Mapping[str, Any]]],
    snapshot: Optional[EventSnapshot],
    *,
    actions: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Non-filled order events produced since ``snapshot`` (``None``: every
    event is new).

    A pure rejection repeated on one trading day is collapsed into the first
    record's ``repeat_count`` (``trading/execution.py``), so growth in that
    count is a rejection on this bar too, not only a newly appended record.
    """
    events = order_events or ()
    start, counts = snapshot if snapshot is not None else EventSnapshot(0, ())
    rejected: List[Dict[str, Any]] = []
    for index in range(start, len(events)):
        event = events[index]
        offset = index - start
        if offset < len(counts):
            grew = _repeat_count(event) - counts[offset]
            if grew > 0:
                rejected.append(_collapsed_rejection(event, grew, actions))
        elif event.get("status") != "filled":
            rejected.append({**_rejection(event), "count": _repeat_count(event), "collapsed": False})
    return rejected


def fills_since(trades: Sequence[Mapping[str, Any]], start: int) -> List[Dict[str, Any]]:
    return [_pick(trade, _FILL_FIELDS) for trade in trades[start:]]


_ORDER_SIDES = {"buy", "sell"}


def _size(value: Any) -> Optional[float]:
    """A model-supplied size as a number; ``None`` when absent or unparseable
    (it can be any text), and a ``None`` size matches any size."""
    if value is None:
        return None
    try:
        size = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return size if math.isfinite(size) else None


def _order_triples(orders: Iterable[Mapping[str, Any]], size_field: str):
    """``(symbol, side, size)`` per buy/sell order, normalised the way the gate
    reads them (``symbol.strip().upper()``, ``action.strip().lower()``).

    ``size`` is compared only where the gate honours it. A SELL is always sized
    by the gate to the whole sellable position, whatever the model sent, so a
    sell's size is ``None`` on both sides. A BUY with ``position_size`` 0 (or
    none) asks the gate to size it from confidence, so that intent size is
    ``None`` as well: the gate doing what was asked is not a rewrite."""
    triples = []
    for order in orders or ():
        if not isinstance(order, Mapping):
            continue
        side = str(order.get("action") or order.get("side") or "").strip().lower()
        if side not in _ORDER_SIDES:
            continue  # hold / unknown: not an order the gate can rewrite
        size = None
        if side == "buy":
            size = _size(order.get(size_field))
            if size == 0:
                size = None
        triples.append((str(order.get("symbol") or "").strip().upper(), side, size))
    return sorted(triples, key=lambda t: (t[0], t[1], -1.0 if t[2] is None else t[2]))


def gate_rewrote(intent: Optional[Mapping[str, Any]], actions: Iterable[Mapping[str, Any]]) -> bool:
    """Whether the pre-trade gate changed what the model asked for.

    Compares the sorted ``(symbol, side, size)`` triples of ``intent.orders``
    (size = ``position_size``) with those of the post-gate ``actions`` (size =
    ``shares``). A dropped, added or resized BUY is a rewrite; a SELL widened
    to the sellable position, or a BUY sized by the gate because the model left
    the size at 0, is the gate's documented behaviour and is not (see
    ``_order_triples``). ``False`` with no intent or an unparsed one: a
    rule-based or fallback bar had no model order to rewrite.

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
    first (action ``reason``s and rejection ``reason``/``strategy_reason``),
    then intent reasoning; orders, fills and rejections themselves are never
    dropped. ``None`` when they alone do not fit.

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
    for rejection in trimmed.get("rejected") or []:
        for field in _REJECTION_TEXT_FIELDS:
            rejection.pop(field, None)
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
    tape, never the backtest.

    The bar loop calls ``capture`` before the decision (pre-decision snapshots)
    and ``record_bar_from`` after it; both run their builders inside this
    swallow layer, and ``capture`` returning ``None`` tells the loop to skip the
    bar's tape work entirely -- so a run with no trace, or a suspended
    recorder, pays nothing per bar beyond one call.

    Bars are buffered and written ``flush_bars`` at a time in one store
    transaction; the owner calls ``flush`` once the loop ends (and on failure,
    via ``trace_lifecycle``). A run killed by SIGTERM loses at most the
    unflushed buffer.

    The trace is looked up once; a lookup that raised (a transient store
    error) is retried on the next bar, one that found no trace turns the
    recorder into a no-op for the run. After ``MAX_CONSECUTIVE_STORE_FAILURES``
    failed store calls in a row the recorder suspends for the rest of the run
    and counts each later bar in ``suspended_skipped`` without touching the
    store; the bars of a failed flush are counted in ``bars_dropped``."""

    def __init__(self, run_id: str, *, flush_bars: Optional[int] = None):
        self.run_id = run_id
        # Read at construction, not bound as a default, so the module constant
        # stays the one knob (tests patch it to pin per-bar store calls).
        self.flush_bars = max(1, int(TAPE_FLUSH_BARS if flush_bars is None else flush_bars))
        self._trace_id: Optional[str] = None
        self._resolved = False
        self._warned = False
        self._consecutive_failures = 0
        self._suspended = False
        self._pending: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        self.bars_recorded = 0
        self.gate_rewrites = 0
        self.write_failures = 0
        self.oversize_skipped = 0
        self.suspended_skipped = 0
        self.bars_dropped = 0

    def _resolve_trace(self) -> Optional[str]:
        if not self._resolved:
            trace = _trace_service().trace_for_run(self.run_id)
            self._resolved = True
            self._trace_id = trace["trace_id"] if trace else None
        return self._trace_id

    def _ensure_trace(self) -> bool:
        """Whether there is a trace to write to; a lookup that raised counts as
        a store failure and is retried next bar."""
        if self._resolved:
            return self._trace_id is not None
        try:
            return self._resolve_trace() is not None
        except Exception as exc:  # observational: never reaches the bar loop
            self._store_failed(exc)
            return False

    def _warn(self, exc: BaseException) -> None:
        if self._warned:
            return
        self._warned = True
        print(
            f"   ⚠️  decision tape write failed ({type(exc).__name__}: {exc}); "
            "later failures are counted, not printed",
            flush=True,
        )

    def _builder_failed(self, exc: BaseException) -> None:
        # Not a store failure: a bad bar must not move the store's streak.
        self.write_failures += 1
        self._warn(exc)

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

    def capture(self, build: Callable[[], Any]) -> Optional[Any]:
        """Run the pre-decision snapshot builder inside the swallow layer.
        ``None`` -- skip this bar's tape -- when the recorder is suspended, the
        run has no trace, or the builder raised."""
        if self._suspended:
            self.suspended_skipped += 1
            return None
        if not self._ensure_trace():
            return None
        try:
            return build()
        except Exception as exc:  # observational: never reaches the bar loop
            self._builder_failed(exc)
            return None

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
        if not self._ensure_trace():
            return
        try:
            decision = bound_payload(decision_payload)
            execution = bound_payload(execution_payload)
            if decision is None or execution is None:
                self.oversize_skipped += 1
                return
            # Checked here, one bar at a time, so a payload the store would
            # refuse is dropped alone rather than failing the batch it joins.
            _trace_service().validate_tape_payload(decision)
            _trace_service().validate_tape_payload(execution)
        except Exception as exc:  # observational: never reaches the bar loop
            self._builder_failed(exc)
            return
        self._pending.append((int(bar_index), decision, execution))
        if len(self._pending) >= self.flush_bars:
            self.flush()

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
        except Exception as exc:  # observational: never reaches the bar loop
            self._builder_failed(exc)
            return
        self.record_bar(
            bar_index=bar_index,
            decision_payload=decision_payload,
            execution_payload=execution_payload,
        )

    def flush(self) -> None:
        """Write the buffered bars in one store call. Never raises; a failed
        flush drops its bars (``bars_dropped``) rather than retrying them, so a
        hung store costs at most ``MAX_CONSECUTIVE_STORE_FAILURES`` waits."""
        if not self._pending:
            return
        bars, self._pending = self._pending, []
        try:
            _trace_service().record_tape_bars(
                trace_id=self._trace_id, run_id=self.run_id, bars=bars
            )
        except Exception as exc:  # observational: never reaches the bar loop
            self.bars_dropped += len(bars)
            self._store_failed(exc)
            return
        self._consecutive_failures = 0
        self.bars_recorded += len(bars)
        self.gate_rewrites += sum(1 for _, decision, _ in bars if decision.get("gate_rewrote"))

    def summary(self) -> Dict[str, Any]:
        return {
            "tape_version": TAPE_VERSION,
            "traced": self._trace_id is not None,
            "bars_recorded": self.bars_recorded,
            "gate_rewrites": self.gate_rewrites,
            "write_failures": self.write_failures,
            "oversize_skipped": self.oversize_skipped,
            "suspended_skipped": self.suspended_skipped,
            "bars_dropped": self.bars_dropped,
        }


def _unwinding_system_exit(exc: BaseException) -> bool:
    """Whether ``exc`` was raised while a ``SystemExit`` was being handled --
    a ``SystemExit`` anywhere on its ``__context__`` chain."""
    seen = set()
    current: Optional[BaseException] = exc.__context__
    while current is not None and id(current) not in seen:
        if isinstance(current, SystemExit):
            return True
        seen.add(id(current))
        current = current.__context__
    return False


@contextmanager
def trace_lifecycle(
    run_id: Optional[str],
    *,
    summary: Callable[[], Dict[str, Any]],
    flush: Optional[Callable[[], None]] = None,
) -> Iterator[None]:
    """Close the run's trace on the way out of the block: completed with
    ``summary()`` (read at exit, so it may reference objects built inside the
    block), failed with ``run_failed`` on an exception. ``flush`` (the tape's
    buffered bars) runs first on both paths, so a failed run keeps the bars
    leading up to its failure.

    The block must span the run's whole outcome -- the agent loop, both
    baselines and ``update_run_baselines`` -- because the parent decides the
    run's status from the child's exit: a trace closed ``completed`` before a
    baseline that then fails would disagree with the dashboard forever.

    ``SystemExit`` writes **nothing** and re-raises. That exit is the
    dashboard's SIGTERM (``_exit_on_sigterm``), which the parent sends from
    two arms -- the user's cancel and the timeout's grace before SIGKILL --
    and only the parent knows which; it labels a trace left ``running`` by
    the arm it is in (``api/routers/backtests.py``). An exception a cleanup
    ``finally`` raised while that exit was unwinding counts as the exit. Open the block before
    ``HourlyBacktester(...)``: the trace is created in its constructor, so a
    ``load_data()`` failure outside the block would be labelled a kill.
    """
    if not run_id:
        yield
        return

    def _flush() -> None:
        if flush is None:
            return
        try:
            flush()
        except Exception:  # noqa: BLE001
            # The recorder's flush never raises; this guards a caller-supplied
            # one, since a flush bug must not cost the trace its close.
            return

    try:
        yield
    except SystemExit:
        raise
    except BaseException as exc:
        if _unwinding_system_exit(exc):
            # A cleanup ``finally`` raised while the SIGTERM SystemExit was
            # unwinding through it (``finalize_run`` in the launch script):
            # the new exception replaced the exit, but the exit is still the
            # cause, so the parent still owns the label.
            raise
        _flush()
        _trace_service().finish_trace_best_effort(run_id, error_code="run_failed")
        raise
    _flush()
    try:
        result_summary = summary()
    except Exception:  # noqa: BLE001 - a summary bug must not cost the close
        result_summary = {}
    _trace_service().finish_trace_best_effort(run_id, result_summary=result_summary)
