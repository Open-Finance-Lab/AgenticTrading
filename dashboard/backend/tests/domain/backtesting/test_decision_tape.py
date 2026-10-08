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


def test_record_bar_from_swallows_a_builder_exception(monkeypatch, capsys):
    fake = _FakeTraceService(trace={"trace_id": "trc_1"})
    _install(monkeypatch, fake)
    recorder = tape.DecisionTapeRecorder("agent_x")

    def broken():
        raise ValueError("builder bug")

    recorder.record_bar_from(broken, bar_index=0)  # must not raise
    recorder.record_bar_from(lambda: (_bar(1)["decision_payload"], _bar(1)["execution_payload"]), bar_index=1)
    summary = recorder.summary()
    assert summary["write_failures"] == 1 and summary["bars_recorded"] == 1


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
