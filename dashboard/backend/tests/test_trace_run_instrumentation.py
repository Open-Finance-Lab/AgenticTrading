from dashboard.backend.domain.traces import service


def test_trace_lifecycle_records_ordered_run_decision_and_execution(tmp_path, monkeypatch):
    from dashboard.backend.domain.traces.repository import TraceStore

    store = TraceStore(tmp_path / "lifecycle.db")
    monkeypatch.setattr(service, "trace_store", store)

    service.start_trace_for_run(
        run={
            "run_id": "run_lifecycle",
            "agent_id": "agent_rule",
            "agent_version_id": "agv_1",
            "environment_id": "us-equity-hourly-v1",
            "environment_type": "backtest",
            "config": {"symbols": ["AAPL"]},
        }
    )
    service.record_decision_event(
        run_id="run_lifecycle",
        step_id="step_1",
        decision_id="dec_1",
        actions=[{"action": "buy", "symbol": "AAPL", "position_size": 1}],
        reasoning_summaries=["rule signal"],
        accepted=True,
        idempotency_key="decision-1",
    )
    service.record_execution_event(
        run_id="run_lifecycle",
        step_id="step_1",
        decision_id="dec_1",
        result={"accepted": True, "fills": [], "validation": {}},
        idempotency_key="decision-1",
    )
    service.complete_trace("run_lifecycle", {"decision_count": 1})

    trace = store.get_trace_for_run("run_lifecycle")
    events = store.list_events(trace["trace_id"])["items"]
    assert [event["event_type"] for event in events] == [
        "run_started",
        "decision_recorded",
        "execution_result",
        "run_completed",
    ]
    assert store.get_trace(trace["trace_id"])["status"] == "completed"


def test_v2_decision_endpoint_appends_trace_events(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import dashboard.backend.api.v2.runs as runs_module
    from dashboard.backend.app import app
    from dashboard.backend.domain.traces.repository import TraceStore
    from dashboard.backend.tests.test_v2_http_runs import _agent, _register_run

    store = TraceStore(tmp_path / "http-lifecycle.db")
    monkeypatch.setattr(runs_module.trace_service, "trace_store", store)
    client = TestClient(app)
    key, session_id, agent_id = _agent("trace-http-agent")
    _register_run("run_trace_http", session_id)
    store.create_trace(
        trace_id="trace_http",
        run_id="run_trace_http",
        agent_id=agent_id,
    )
    store.append_event(
        trace_id="trace_http",
        event_type="run_started",
        actor_type="system",
        payload={"environment_id": "test"},
    )

    response = client.post(
        "/api/v2/runs/run_trace_http/decisions",
        headers={"X-API-Key": key},
        json={
            "idempotency_key": "trace-http-decision",
            "actions": [
                {
                    "action": "buy",
                    "symbol": "AAPL",
                    "confidence": 0.7,
                    "reasoning": "deterministic test signal",
                    "position_size": 1,
                }
            ],
        },
    )

    assert response.status_code == 200
    events = store.list_events("trace_http")["items"]
    assert [event["event_type"] for event in events] == [
        "run_started",
        "decision_recorded",
        "execution_result",
    ]
