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
    assert all(bar["complete"] is True for bar in tape)


def test_load_decision_tape_marks_a_bar_missing_its_execution_half(store):
    """Tape bars are written a batch per transaction now, but a reader must
    still not read a missing half as "nothing filled" -- whatever wrote it."""
    trace_id = _trace(store)
    decision, execution = _pair(0)
    service.record_tape_bar(trace_id=trace_id, run_id=RUN, bar_index=0,
                            decision_payload=decision, execution_payload=execution)
    decision, _execution = _pair(1)
    store.append_event(trace_id=trace_id, event_type="decision_recorded", actor_type="agent",
                       payload=decision, idempotency_key="decision-only")
    tape = service.load_decision_tape(RUN)
    assert [bar["complete"] for bar in tape] == [True, False]
    assert "execution" not in tape[1] and tape[1]["decision"]["bar_index"] == 1


def test_record_tape_bars_writes_a_batch_in_order_in_one_call(store):
    trace_id = _trace(store)
    bars = [(i, *_pair(i)) for i in range(4)]
    assert service.record_tape_bars(trace_id=trace_id, run_id=RUN, bars=bars) == 8
    events = store.list_events(trace_id)["items"]
    tape_events = [e for e in events if e["payload"].get("tape_version")]
    assert [(e["event_type"], e["payload"]["bar_index"]) for e in tape_events] == [
        (kind, i) for i in range(4) for kind in ("decision_recorded", "execution_result")
    ]
    sequences = [e["sequence_no"] for e in events]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)
    # Re-sending a batch (and a bar repeated inside one) writes nothing new.
    assert service.record_tape_bars(trace_id=trace_id, run_id=RUN, bars=bars + bars[:1]) == 0
    assert len(store.list_events(trace_id)["items"]) == len(events)


def test_a_batch_with_one_refused_payload_writes_nothing(store):
    """Validated before the transaction opens: no partial batch."""
    trace_id = _trace(store)
    before = len(store.list_events(trace_id)["items"])
    good = (0, *_pair(0))
    bad = (1, {"tape_version": 1, "bar_index": 1, "secret": "x"}, _pair(1)[1])
    with pytest.raises(ValueError):
        service.record_tape_bars(trace_id=trace_id, run_id=RUN, bars=[good, bad])
    assert len(store.list_events(trace_id)["items"]) == before


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


def test_finish_trace_best_effort_says_why_it_failed(store, monkeypatch, capsys):
    """The parent's ``trace_close_failed`` label says THAT the child could not
    close its trace; this line is the only record of WHY."""
    _trace(store)
    monkeypatch.setattr(store, "append_event", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("neon down")))
    service.finish_trace_best_effort(RUN, result_summary={})
    out = capsys.readouterr().out
    assert "ERROR: trace.close_failed" in out and "neon down" in out and RUN in out


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


def test_fail_trace_if_running_raises_when_the_store_is_broken(store, monkeypatch):
    """Absent, already closed and store-down must not all read as a quiet
    False: the store error propagates so the parent logs it."""
    def boom(*_a, **_k):
        raise RuntimeError("store down")

    monkeypatch.setattr(store, "get_trace_for_run", boom)
    with pytest.raises(RuntimeError, match="store down"):
        service.fail_trace_if_running(RUN, "run_killed")
