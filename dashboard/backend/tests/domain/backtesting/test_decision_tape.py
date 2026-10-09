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


def _one_order(entry):
    raw = [{"step": 1, "output": {"actions": [entry]}}]
    return tape.intent_from_outputs(raw, outputs_before=[], decision_step_count=1)


def test_intent_keeps_present_null_fields_and_omits_absent_ones():
    """The gate reads each field with a default: a null confidence raises in
    its format string, an absent one reads as 0.5. Dropping null is not exact."""
    intent = _one_order({"symbol": "AAPL", "action": "buy", "confidence": None, "reasoning": None})
    assert intent["orders"][0] == {"symbol": "AAPL", "action": "buy", "confidence": None, "reasoning": None}
    assert tape.orders_for_replay(intent) == [intent["orders"][0]]


def test_intent_records_non_finite_and_container_values_exactly():
    intent = _one_order({
        "symbol": "AAPL", "action": ["buy"], "position_size": float("inf"),
        "confidence": float("nan"), "reasoning": {"why": "x"},
    })
    order = intent["orders"][0]
    # Readable in the store (strict JSON), exact under raw.
    assert order["position_size"] is None and order["confidence"] is None
    assert order["action"] == "['buy']" and order["reasoning"] == "{'why': 'x'}"
    assert order["raw"] == {
        "action": '["buy"]', "position_size": "Infinity",
        "confidence": "NaN", "reasoning": '{"why": "x"}',
    }
    json.dumps(intent, allow_nan=False)  # the store's payload stays strict JSON
    [replayed] = tape.orders_for_replay(intent)
    assert replayed["action"] == ["buy"] and replayed["position_size"] == float("inf")
    assert replayed["confidence"] != replayed["confidence"]  # NaN
    assert replayed["reasoning"] == {"why": "x"} and "raw" not in replayed


def test_intent_keeps_a_non_object_entry_whole():
    intent = _one_order("buy AAPL")
    assert intent["orders"] == [{"raw_entry": '"buy AAPL"'}]
    assert tape.orders_for_replay(intent) == ["buy AAPL"]
    assert tape.orders_for_replay(None) == [] and tape.orders_for_replay({"unparsed": True}) == []


def test_gate_rewrote_is_false_on_a_fallback_bar():
    """A fallback's actions are rule-based substitutes, not a rewrite."""
    intent = {"orders": [{"symbol": "AAPL", "action": "buy", "confidence": None}]}
    actions = [{"symbol": "MSFT", "action": "buy", "shares": 1}]
    state = {"cash": 1.0, "equity": 1.0, "positions": []}
    common = dict(bar_index=0, decision_at="t", intent=intent, state=state, actions=actions)
    assert tape.build_decision_payload(driver="llm_fallback", **common)["gate_rewrote"] is False
    assert tape.build_decision_payload(driver="llm", **common)["gate_rewrote"] is True


def test_bound_payload_keeps_replay_exact_or_marks_the_order():
    """String reasoning drops (the gate reads absent as ""); a null one is kept
    (the gate raises on it); a raw one cut marks the order unreplayable."""
    orders = [
        {"symbol": "AAPL", "action": "buy", "reasoning": "r" * 500},
        {"symbol": "MSFT", "action": "buy", "reasoning": None},
        tape._intent_order({"symbol": "GOOG", "action": "buy", "reasoning": {"k": "v" * 400}, "confidence": float("inf")}),
    ]
    actions = [{"symbol": f"S{i}", "action": "buy", "shares": 1, "reason": "r" * 500} for i in range(110)]
    payload = tape.build_decision_payload(
        bar_index=0, decision_at="t", driver="llm", intent={"orders": orders * 40},
        state={"cash": 0, "equity": 0, "positions": []}, actions=actions,
    )
    bounded = tape.bound_payload(payload)
    assert bounded is not None
    aapl, msft, goog = bounded["intent"]["orders"][:3]
    assert "reasoning" not in aapl
    assert msft["reasoning"] is None
    assert goog["unreplayable"] == ["reasoning"] and "reasoning" not in goog
    assert goog["raw"] == {"confidence": "Infinity"}
    with pytest.raises(ValueError):
        tape.orders_for_replay(bounded["intent"])
    assert payload["intent"]["orders"][2]["raw"]["reasoning"]  # the original is untouched


def test_build_state_uses_pairs_and_skips_flat_positions():
    state = tape.build_state(cash=np.float64(100.0), equity=250.0, positions={"MSFT": 2, "AAPL": 3, "TOKEN": 0})
    assert state == {"cash": 100.0, "equity": 250.0, "positions": [["AAPL", 3], ["MSFT", 2]]}


def test_rejections_since_reports_new_and_collapsed_events():
    events = [
        {"symbol": "AAPL", "side": "BUY", "status": "rejected", "reason": "insufficient_cash", "requested_shares": 9},
        {"symbol": "MSFT", "side": "BUY", "status": "filled", "reason": "", "requested_shares": 1},
    ]
    snapshot = tape.order_events_snapshot(events)
    assert snapshot == tape.EventSnapshot(0, (1, 1))
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


def test_a_collapsed_repeat_carries_this_bars_order_not_the_first_one():
    """The bumped record describes the day's FIRST rejection -- its time,
    size, price and reasoning. Copying it would put an earlier bar's order on
    this bar, so only the collapse key is copied and the size comes from this
    bar's own action."""
    events = [{
        "timestamp": "2026-03-02T10:00:00", "symbol": "AAPL", "side": "BUY",
        "status": "rejected", "reason": "insufficient_cash", "requested_shares": 50,
        "price": 101.0, "strategy_reason": "[LLM] 10:00 reasoning",
    }]
    snapshot = tape.order_events_snapshot(events, since="2026-03-02T14:00:00")
    events[0]["repeat_count"] = 2
    actions = [{"symbol": "AAPL", "action": "buy", "shares": 200, "reason": "[LLM] 14:00 reasoning"}]
    [rejection] = tape.rejections_since(events, snapshot, actions=actions)
    assert rejection == {
        "symbol": "AAPL", "side": "BUY", "status": "rejected", "reason": "insufficient_cash",
        "first_rejected_at": "2026-03-02T10:00:00", "requested_shares": 200,
        "count": 1, "collapsed": True,
    }
    # Two matching actions: the size is ambiguous, so none is claimed.
    twice = actions + [{"symbol": "AAPL", "action": "buy", "shares": 5}]
    snapshot = tape.order_events_snapshot(events, since="2026-03-02T15:00:00")
    events[0]["repeat_count"] = 3
    assert "requested_shares" not in tape.rejections_since(events, snapshot, actions=twice)[0]


def test_snapshot_covers_only_events_the_bar_could_collapse_into():
    """A rejection collapses only into a same-trading-day record, and a bar
    fills on or after its decision, so the snapshot is the suffix dated on or
    after the decision -- not the whole run's ledger every bar."""
    import pandas as pd

    events = [
        {"timestamp": pd.Timestamp("2026-03-02 10:00"), "symbol": "A", "status": "rejected"},
        {"timestamp": pd.Timestamp("2026-03-02 15:00"), "symbol": "B", "status": "rejected"},
        {"timestamp": pd.Timestamp("2026-03-03 10:00"), "symbol": "C", "status": "rejected", "repeat_count": 4},
        {"timestamp": "not a date", "symbol": "D", "status": "rejected"},
    ]
    snapshot = tape.order_events_snapshot(events, since=pd.Timestamp("2026-03-03 11:00"))
    assert snapshot == tape.EventSnapshot(2, (4, 1))
    # An unreadable decision time falls back to the whole ledger.
    assert tape.order_events_snapshot(events, since=object()).start == 0
    # Growth on a covered record and a new record both report.
    events[2]["repeat_count"] = 5
    events.append({"timestamp": pd.Timestamp("2026-03-03 12:00"), "symbol": "E", "status": "rejected"})
    rejected = tape.rejections_since(events, snapshot)
    assert [(r["symbol"], r["collapsed"]) for r in rejected] == [("C", True), ("E", False)]


def test_rejections_since_truncates_reasoning_text():
    """``strategy_reason`` embeds the action's ``reason`` -- the model's whole
    reasoning -- so it is cut like any other reasoning field."""
    events = [{"symbol": "AAPL", "side": "BUY", "status": "rejected",
               "reason": "x" * 5000, "strategy_reason": "y" * 5000}]
    rejected = tape.rejections_since(events, None)
    assert len(rejected[0]["reason"]) == tape.MAX_REASONING_CHARS
    assert len(rejected[0]["strategy_reason"]) == tape.MAX_REASONING_CHARS


def test_bound_payload_sheds_rejection_text_before_dropping_the_bar():
    events = [{"symbol": f"S{i}", "side": "BUY", "status": "rejected", "requested_shares": 1,
               "reason": "r" * 500, "strategy_reason": "s" * 500} for i in range(80)]
    payload = tape.build_execution_payload(
        bar_index=0, fill=SimpleNamespace(bar="t", price_field="open", filled_at="t"),
        fills=[], rejected=tape.rejections_since(events, None),
    )
    assert _size(payload) > tape.MAX_PAYLOAD_BYTES
    bounded = tape.bound_payload(payload)
    assert bounded is not None and _size(bounded) <= tape.MAX_PAYLOAD_BYTES
    assert [r["symbol"] for r in bounded["rejected"]] == [f"S{i}" for i in range(80)]
    assert "reason" not in bounded["rejected"][0] and "strategy_reason" not in bounded["rejected"][0]
    assert bounded["rejected"][0]["status"] == "rejected"


def test_fills_since_slices_and_allow_lists():
    trades = [
        {"timestamp": pd.Timestamp("2026-03-02 10:30"), "symbol": "AAPL", "side": "BUY", "shares": 5, "price": 10.0, "cost": 50.0},
        {"timestamp": pd.Timestamp("2026-03-02 11:30"), "symbol": "AAPL", "side": "SELL", "shares": 5, "price": 11.0, "proceeds": 55.0, "secret_sauce": 1},
    ]
    assert tape.fills_since(trades, 1) == [{
        "timestamp": "2026-03-02T11:30:00", "symbol": "AAPL", "side": "SELL",
        "shares": 5, "price": 11.0, "proceeds": 55.0,
    }]


def test_gate_rewrote_tolerates_non_numeric_model_sizes():
    action = [{"symbol": "AAPL", "action": "buy", "shares": 5}]
    for junk in ("ten", "all", "[1, 2]", "{'a': 1}", [1, 2], {"a": 1}, ""):
        intent = {"orders": [{"symbol": "AAPL", "action": "buy", "position_size": junk}]}
        assert tape.gate_rewrote(intent, action) is False  # unparseable size matches any size
    assert tape.gate_rewrote(
        {"orders": [{"symbol": "AAPL", "action": "buy", "position_size": "ten"}]}, []
    ) is True


def test_gate_rewrote_compares_symbol_side_size_triples():
    buy5 = {"orders": [{"symbol": "AAPL", "action": "buy", "position_size": 5}]}
    assert tape.gate_rewrote(buy5, [{"symbol": "AAPL", "action": "buy", "shares": 5}]) is False
    # resized, dropped, added, and a SELL widened to the whole position
    assert tape.gate_rewrote(buy5, [{"symbol": "AAPL", "action": "buy", "shares": 3}]) is True
    assert tape.gate_rewrote(buy5, []) is True
    assert tape.gate_rewrote({"orders": []}, [{"symbol": "AAPL", "action": "buy", "shares": 1}]) is True
    # The gate always sizes a SELL to the sellable position, and sizes a BUY
    # left at 0 from confidence: both are what was asked, not a rewrite.
    trim = {"orders": [{"symbol": "AAPL", "action": "sell", "position_size": 2}]}
    assert tape.gate_rewrote(trim, [{"symbol": "AAPL", "action": "sell", "shares": 10}]) is False
    assert tape.gate_rewrote(trim, []) is True  # a dropped sell still is
    size_it = {"orders": [{"symbol": "AAPL", "action": "buy", "position_size": 0}]}
    assert tape.gate_rewrote(size_it, [{"symbol": "AAPL", "action": "buy", "shares": 7}]) is False
    # Normalised as the gate reads them: " aapl " / "BUY " is AAPL buy.
    padded = {"orders": [{"symbol": " aapl ", "action": "BUY ", "position_size": 5}]}
    assert tape.gate_rewrote(padded, [{"symbol": "AAPL", "action": "buy", "shares": 5}]) is False
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

    def record_tape_bars(self, *, trace_id, run_id, bars):
        if self.write_error is not None:
            raise self.write_error
        self.batches.append([index for index, _, _ in bars])
        self.writes.extend({"trace_id": trace_id, "run_id": run_id, "bar_index": index} for index, _, _ in bars)


def _install(monkeypatch, fake):
    fake.batches = []
    monkeypatch.setattr(trace_service, "trace_for_run", fake.trace_for_run)
    monkeypatch.setattr(trace_service, "record_tape_bars", fake.record_tape_bars)


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
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1, rewrote=True))
    recorder.record_bar(**_bar(2))
    assert fake.lookups == 1
    assert [w["bar_index"] for w in fake.writes] == [0, 1, 2]
    assert fake.writes[0]["trace_id"] == "trc_1" and fake.writes[0]["run_id"] == "agent_x"
    assert recorder.summary() == {
        "tape_version": 1, "traced": True, "bars_recorded": 3, "gate_rewrites": 1,
        "write_failures": 0, "oversize_skipped": 0, "suspended_skipped": 0,
        "bars_dropped": 0,
    }


def test_recorder_is_a_no_op_without_a_trace(monkeypatch):
    fake = _FakeTraceService(trace=None)
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1))
    assert fake.lookups == 1 and fake.writes == []
    assert recorder.summary()["traced"] is False


def test_recorder_retries_a_lookup_that_raised(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"}, lookup_error=RuntimeError("neon blip"))
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    recorder.record_bar(**_bar(0))
    recorder.record_bar(**_bar(1))
    assert fake.lookups == 2
    assert [w["bar_index"] for w in fake.writes] == [1]
    assert recorder.summary()["write_failures"] == 1


def test_recorder_swallows_write_failures_and_logs_once(monkeypatch, capsys):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"}, write_error=ValueError("payload exceeds"))
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    for i in range(3):
        recorder.record_bar(**_bar(i))  # must not raise
    assert recorder.summary()["write_failures"] == 3
    assert capsys.readouterr().out.count("decision tape write failed") == 1


class _CountingStore(_FakeTraceService):
    def __init__(self, fail_writes):
        super().__init__(trace={"trace_id": "trc_1"})
        self.fail_writes = set(fail_writes)
        self.write_calls = 0

    def record_tape_bars(self, *, trace_id, run_id, bars):
        index = self.write_calls
        self.write_calls += 1
        if index in self.fail_writes:
            raise TimeoutError("connect timed out")  # a hung store, already waited out
        self.writes.extend({"bar_index": i} for i, _, _ in bars)


def test_recorder_stops_calling_a_store_that_keeps_failing(monkeypatch, capsys):
    """Review Focus 4, the slow half: a store that hangs before raising costs
    that hang on every call, so the recorder must stop calling it, not just
    swallow what it raises."""
    fake = _CountingStore(fail_writes=range(100))
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    for i in range(10):
        recorder.record_bar(**_bar(i))
    recorder.record_bar_from(lambda: (_bar(10)["decision_payload"], _bar(10)["execution_payload"]), bar_index=10)
    assert fake.write_calls == tape.MAX_CONSECUTIVE_STORE_FAILURES
    summary = recorder.summary()
    assert summary["write_failures"] == tape.MAX_CONSECUTIVE_STORE_FAILURES
    assert summary["suspended_skipped"] == 11 - tape.MAX_CONSECUTIVE_STORE_FAILURES
    assert summary["bars_recorded"] == 0
    assert summary["bars_dropped"] == tape.MAX_CONSECUTIVE_STORE_FAILURES
    assert capsys.readouterr().out.count("decision tape suspended") == 1


def test_recorder_stops_retrying_a_lookup_that_keeps_failing(monkeypatch):
    class _LookupDown(_FakeTraceService):
        def trace_for_run(self, run_id):
            self.lookups += 1
            raise TimeoutError("neon dial timed out")

    fake = _LookupDown(trace=None)
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    for i in range(10):
        recorder.record_bar(**_bar(i))
    assert fake.lookups == tape.MAX_CONSECUTIVE_STORE_FAILURES


def test_a_successful_write_resets_the_failure_streak(monkeypatch):
    n = tape.MAX_CONSECUTIVE_STORE_FAILURES
    # n-1 failures, one success, n-1 failures, one success: never n in a row.
    fails = [i for i in range(2 * n) if i not in (n - 1, 2 * n - 1)]
    fake = _CountingStore(fail_writes=fails)
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    for i in range(2 * n):
        recorder.record_bar(**_bar(i))
    summary = recorder.summary()
    assert fake.write_calls == 2 * n
    assert summary["bars_recorded"] == 2 and summary["suspended_skipped"] == 0


def test_recorder_counts_oversize_bars_without_writing(monkeypatch):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    monkeypatch.setattr(tape, "bound_payload", lambda payload: None)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    recorder.record_bar(**_bar(0))
    assert fake.writes == [] and recorder.summary()["oversize_skipped"] == 1


def test_record_bar_from_swallows_a_builder_exception(monkeypatch, capsys):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)

    def broken():
        raise ValueError("builder bug")

    recorder.record_bar_from(broken, bar_index=0)  # must not raise
    recorder.record_bar_from(lambda: (_bar(1)["decision_payload"], _bar(1)["execution_payload"]), bar_index=1)
    summary = recorder.summary()
    assert summary["write_failures"] == 1 and summary["bars_recorded"] == 1


def test_recorder_batches_bars_into_one_store_call(monkeypatch):
    """Per-bar appends were ~10 store round trips a bar in the decision loop;
    the recorder buffers ``flush_bars`` bars per store call, and the owner's
    final ``flush`` writes the remainder."""
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=3)
    for i in range(7):
        recorder.record_bar(**_bar(i, rewrote=i == 4))
    assert fake.batches == [[0, 1, 2], [3, 4, 5]]
    assert recorder.summary()["bars_recorded"] == 6
    recorder.flush()
    recorder.flush()  # idempotent with nothing pending
    assert fake.batches == [[0, 1, 2], [3, 4, 5], [6]]
    summary = recorder.summary()
    assert summary["bars_recorded"] == 7 and summary["gate_rewrites"] == 1


def test_a_failed_flush_drops_its_bars_and_counts_them(monkeypatch):
    fake = _CountingStore(fail_writes={0})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=2)
    for i in range(4):
        recorder.record_bar(**_bar(i))
    summary = recorder.summary()
    assert [w["bar_index"] for w in fake.writes] == [2, 3]
    assert summary["bars_dropped"] == 2 and summary["bars_recorded"] == 2
    assert summary["write_failures"] == 1


def test_a_payload_the_store_would_refuse_is_dropped_alone(monkeypatch):
    """Validated as it is buffered, so one bad bar cannot fail the batch."""
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=3)
    recorder.record_bar(**_bar(0))
    recorder.record_bar(bar_index=1, decision_payload={"tape_version": 1, "api_token": "x"},
                        execution_payload={"tape_version": 1})
    recorder.record_bar(**_bar(2))
    recorder.flush()
    assert fake.batches == [[0, 2]]
    assert recorder.summary()["write_failures"] == 1


def test_capture_runs_the_snapshot_inside_the_swallow_layer(monkeypatch, capsys):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")

    def broken():
        raise TypeError("'<' not supported between instances of 'int' and 'str'")

    assert recorder.capture(broken) is None  # must not raise into the bar loop
    assert recorder.capture(lambda: "snap") == "snap"
    assert recorder.summary()["write_failures"] == 1
    assert "decision tape write failed" in capsys.readouterr().out


def test_capture_skips_snapshot_work_without_a_trace_or_once_suspended(monkeypatch):
    fake = _FakeTraceService(trace=None)
    _install(monkeypatch, fake)
    built = []
    recorder = tape.DecisionTapeRecorder("agent_x")
    assert recorder.capture(lambda: built.append(1)) is None
    assert recorder.capture(lambda: built.append(1)) is None
    assert built == [] and fake.lookups == 1

    down = _CountingStore(fail_writes=range(100))
    _install(monkeypatch, down)
    recorder = tape.DecisionTapeRecorder("agent_x", flush_bars=1)
    for i in range(tape.MAX_CONSECUTIVE_STORE_FAILURES):
        recorder.record_bar(**_bar(i))
    assert recorder.capture(lambda: built.append(1)) is None
    assert built == [] and recorder.summary()["suspended_skipped"] == 1


def test_intent_marks_an_oversized_raw_value_unreplayable_instead_of_losing_the_bar():
    huge = list(range(5000))
    outputs = [{"step": 1, "output": {"actions": [
        {"symbol": "AAPL", "action": "buy", "position_size": huge, "confidence": 0.9},
        "x" * 10_000,
        {"symbol": "MSFT", "action": "buy", "position_size": 3, "confidence": 0.9},
    ]}}]
    intent = tape.intent_from_outputs(outputs, outputs_before=None, decision_step_count=1)
    aapl, entry, msft = intent["orders"]
    assert aapl["unreplayable"] == ["position_size"] and "raw" not in aapl
    assert len(aapl["position_size"]) == tape.MAX_REASONING_CHARS
    assert entry == {"raw_entry_preview": entry["raw_entry_preview"], "unreplayable": ["raw_entry"]}
    assert msft == {"symbol": "MSFT", "action": "buy", "position_size": 3, "confidence": 0.9}
    payload = {"tape_version": 1, "intent": intent}
    assert tape.bound_payload(payload) is payload  # the bar still fits
    with pytest.raises(ValueError):
        tape.orders_for_replay(intent)


def _raise(exc):
    """Raise from a call, not a bare statement: inside ``pytest.raises`` a bare
    ``raise`` makes static analysis read every assertion after the block as
    unreachable (CodeQL py/unreachable-statement, py/unused-local-variable)."""
    raise exc


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
            _raise(RuntimeError("boom"))
    assert calls == [("agent_x", {"error_code": "run_failed"})]


def test_lifecycle_fails_on_an_exception_raised_before_run(monkeypatch):
    """Review Focus 7: load_data()/calculate_indicators() fail after the
    trace exists; that must read run_failed, never run_killed."""
    calls = _capture_finish(monkeypatch)

    class _FeedRefused(Exception):
        pass

    with pytest.raises(_FeedRefused):
        with tape.trace_lifecycle("agent_x", summary=lambda: pytest.fail("no summary on failure")):
            _raise(_FeedRefused("SIP refused"))  # stands in for load_data()
    assert calls == [("agent_x", {"error_code": "run_failed"})]


def test_lifecycle_writes_nothing_on_system_exit(monkeypatch):
    """The SIGTERM exit is sent by both the cancel and the timeout arms; only
    the parent knows which, so the child leaves the trace running for it."""
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("the parent labels a SIGTERM exit"))
    with pytest.raises(SystemExit):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}):
            raise SystemExit(143)


def test_lifecycle_writes_nothing_when_cleanup_replaces_a_system_exit(monkeypatch):
    """The launch script's ``finally`` re-raises a ``finalize_run`` error while
    the SIGTERM exit unwinds through it: the exit is still the cause, so the
    child must not stamp ``run_failed`` over the parent's cancel/timeout label."""
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("the parent labels a SIGTERM exit"))

    class _CleanupFailed(Exception):
        pass

    with pytest.raises(_CleanupFailed):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}):
            try:
                raise SystemExit(143)
            finally:
                try:
                    raise _CleanupFailed("finalize_run")
                except _CleanupFailed:
                    raise  # nested once more: the exit is two links up the chain


def test_lifecycle_still_fails_an_exception_handled_without_a_system_exit(monkeypatch):
    calls = _capture_finish(monkeypatch)
    with pytest.raises(RuntimeError):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}):
            try:
                raise ValueError("inner")
            except ValueError:
                _raise(RuntimeError("outer"))
    assert calls == [("agent_x", {"error_code": "run_failed"})]


def test_lifecycle_without_run_id_touches_no_trace(monkeypatch):
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("no run id, no trace"))
    with tape.trace_lifecycle(None, summary=lambda: {}):
        pass


def test_lifecycle_flushes_the_tape_before_closing_on_both_paths(monkeypatch):
    """A failed run keeps the bars leading up to its failure: the buffered
    tape is written before the trace is closed, on success and on failure."""
    order = []
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda run_id, **kw: order.append(("close", sorted(kw))) or True)
    with tape.trace_lifecycle("agent_x", summary=lambda: order.append("summary") or {},
                              flush=lambda: order.append("flush")):
        pass
    assert order == ["flush", "summary", ("close", ["result_summary"])]
    order.clear()
    with pytest.raises(RuntimeError):
        with tape.trace_lifecycle("agent_x", summary=lambda: {}, flush=lambda: order.append("flush")):
            _raise(RuntimeError("boom"))
    assert order == ["flush", ("close", ["error_code"])]


def test_lifecycle_does_not_flush_on_system_exit(monkeypatch):
    """SIGTERM's grace period is for unwinding, not for store writes."""
    monkeypatch.setattr(trace_service, "finish_trace_best_effort",
                        lambda *a, **k: pytest.fail("the parent labels a SIGTERM exit"))
    with pytest.raises(SystemExit):
        with tape.trace_lifecycle("agent_x", summary=lambda: {},
                                  flush=lambda: pytest.fail("no flush on SIGTERM")):
            _raise(SystemExit(143))


def test_lifecycle_closes_the_trace_when_flush_raises(monkeypatch):
    calls = _capture_finish(monkeypatch)
    with tape.trace_lifecycle("agent_x", summary=lambda: {"bars_recorded": 1},
                              flush=lambda: _raise(RuntimeError("flush bug"))):
        pass
    assert calls == [("agent_x", {"result_summary": {"bars_recorded": 1}})]


def test_the_launch_script_closes_the_trace_only_after_the_baselines():
    """The parent decides the run's status from the child's exit, and both
    baselines plus ``update_run_baselines`` run before that exit: a raise there
    is a failed run. A trace closed ``completed`` before them would contradict
    the dashboard's failed / cancelled / timed_out for good, so the
    ``trace_lifecycle`` block must span every one of them (source shape --
    running ``main()`` needs a live data feed)."""
    import ast
    from pathlib import Path

    script = Path(__file__).resolve().parents[4] / "scripts" / "backtest_hourly_agent.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    [block] = [
        node for node in ast.walk(main)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "trace_lifecycle"
            for item in node.items
        )
    ]
    call = block.items[0].context_expr
    assert {kw.arg for kw in call.keywords} >= {"summary", "flush"}

    def called(nodes):
        names = set()
        for node in nodes:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    func = sub.func
                    names.add(getattr(func, "attr", None) or getattr(func, "id", None))
        return names

    inside = called(block.body)
    required = {
        "HourlyBacktester", "load_data", "run_agent_backtest",
        "run_buyhold_baseline", "run_djia_baseline", "update_run_baselines",
    }
    assert required <= inside, f"outside the trace block: {sorted(required - inside)}"
