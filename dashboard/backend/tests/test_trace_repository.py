import json

import pytest


@pytest.fixture
def trace_store(tmp_path):
    from dashboard.backend.domain.traces.repository import TraceStore

    return TraceStore(tmp_path / "traces.db")


def test_create_trace_round_trips_bounded_metadata(trace_store):
    trace = trace_store.create_trace(
        trace_id="trace_test_1",
        agent_id="agent_rule",
        agent_version_id="agv_1",
        run_id="run_1",
        user_id=7,
        trace_kind="trading_run",
        initial_input={"symbols": ["AAPL"]},
    )

    assert trace["trace_id"] == "trace_test_1"
    assert trace["status"] == "running"
    assert json.loads(trace["initial_input_json"]) == {"symbols": ["AAPL"]}


def test_append_events_assigns_strict_per_trace_sequence(trace_store):
    trace_store.create_trace(trace_id="trace_test_2", run_id="run_2")

    first = trace_store.append_event(
        trace_id="trace_test_2",
        event_type="run_started",
        actor_type="system",
        payload={"environment_id": "test"},
    )
    second = trace_store.append_event(
        trace_id="trace_test_2",
        event_type="decision_recorded",
        actor_type="agent",
        payload={"reasoning_summary": "hold"},
    )

    assert first["sequence_no"] == 1
    assert second["sequence_no"] == 2
    page = trace_store.list_events("trace_test_2")
    assert [event["event_type"] for event in page["items"]] == [
        "run_started",
        "decision_recorded",
    ]


def test_append_event_is_idempotent_for_retry_key(trace_store):
    trace_store.create_trace(trace_id="trace_test_3", run_id="run_3")

    first = trace_store.append_event(
        trace_id="trace_test_3",
        event_type="run_started",
        actor_type="system",
        payload={"attempt": 1},
        idempotency_key="start:run_3",
    )
    retry = trace_store.append_event(
        trace_id="trace_test_3",
        event_type="run_started",
        actor_type="system",
        payload={"attempt": 2},
        idempotency_key="start:run_3",
    )

    assert retry == first
    assert trace_store.list_events("trace_test_3")["next_sequence_no"] == 2


def test_list_events_supports_incremental_reads_and_limit(trace_store):
    trace_store.create_trace(trace_id="trace_test_4", run_id="run_4")
    for index in range(3):
        trace_store.append_event(
            trace_id="trace_test_4",
            event_type="agent_message",
            actor_type="agent",
            payload={"index": index},
        )

    page = trace_store.list_events("trace_test_4", after_sequence=1, limit=1)

    assert [event["sequence_no"] for event in page["items"]] == [2]
    assert page["next_sequence_no"] == 3
    assert page["has_more"] is True


def test_payload_is_rejected_when_it_contains_credentials(trace_store):
    trace_store.create_trace(trace_id="trace_test_5", run_id="run_5")

    with pytest.raises(ValueError, match="sensitive"):
        trace_store.append_event(
            trace_id="trace_test_5",
            event_type="tool_call",
            actor_type="tool",
            payload={"authorization": "Bearer secret"},
        )


def test_update_trace_closes_the_envelope(trace_store):
    trace_store.create_trace(trace_id="trace_test_6", run_id="run_6")

    updated = trace_store.update_trace(
        "trace_test_6",
        status="completed",
        final_output_summary={"decision_count": 1},
        ended_at="2026-10-07T12:00:00+00:00",
    )

    assert updated["status"] == "completed"
    assert updated["ended_at"] == "2026-10-07T12:00:00+00:00"
    assert json.loads(updated["final_output_summary"]) == {"decision_count": 1}


def test_parent_links_and_terminal_status_are_stable(trace_store):
    parent = trace_store.create_trace(trace_id="trace_parent", run_id="run_parent")
    child = trace_store.create_trace(
        trace_id="trace_child", run_id="run_child", parent_trace_id=parent["trace_id"]
    )
    event = trace_store.append_event(
        trace_id=child["trace_id"], event_type="tool_call", actor_type="agent",
        parent_event_id="event_parent", payload={}, idempotency_key="child-call",
    )
    assert child["parent_trace_id"] == "trace_parent"
    assert event["parent_event_id"] == "event_parent"
    trace_store.update_trace(child["trace_id"], status="failed")
    stable = trace_store.update_trace(child["trace_id"], status="completed")
    assert stable["status"] == "failed"


def test_list_traces_filters_and_returns_cursor(trace_store):
    trace_store.create_trace(trace_id="trace_list_1", run_id="run_list_1", agent_id="agent_a")
    trace_store.create_trace(trace_id="trace_list_2", run_id="run_list_2", agent_id="agent_b")

    page = trace_store.list_traces(agent_id="agent_a", limit=1)

    assert [item["trace_id"] for item in page["items"]] == ["trace_list_1"]
    assert page["has_more"] is False
    assert page["next_cursor"] is None
