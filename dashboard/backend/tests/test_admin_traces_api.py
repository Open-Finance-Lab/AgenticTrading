from fastapi.testclient import TestClient

from dashboard.backend.api.auth import require_admin
from dashboard.backend.app import app
from dashboard.backend.domain.traces import repository as repository_module
from dashboard.backend.domain.traces.repository import TraceStore


def test_admin_trace_api_lists_detail_and_incremental_events(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "admin-traces.db")
    monkeypatch.setattr(repository_module, "trace_store", store)
    import dashboard.backend.api.routers.admin_traces as router_module

    monkeypatch.setattr(router_module, "trace_store", store)
    app.dependency_overrides[require_admin] = lambda: {"id": 1, "is_admin": True}
    try:
        store.create_trace(
            trace_id="trace_admin_1",
            run_id="run_admin_1",
            agent_id="agent_admin",
            initial_input={"symbols": ["AAPL"]},
        )
        store.append_event(
            trace_id="trace_admin_1",
            event_type="run_started",
            actor_type="system",
            payload={"environment_id": "test"},
        )
        store.append_event(
            trace_id="trace_admin_1",
            event_type="decision_recorded",
            actor_type="agent",
            payload={"reasoning_summary": "hold"},
        )

        client = TestClient(app)
        listed = client.get("/api/admin/traces?agent_id=agent_admin")
        assert listed.status_code == 200
        assert listed.json()["items"][0]["trace_id"] == "trace_admin_1"
        assert listed.json()["items"][0]["initial_input"] == {"symbols": ["AAPL"]}

        detail = client.get("/api/admin/traces/trace_admin_1")
        assert detail.status_code == 200
        assert detail.json()["run_id"] == "run_admin_1"

        events = client.get(
            "/api/admin/traces/trace_admin_1/events?after_sequence=1&limit=1"
        )
        assert events.status_code == 200
        assert [item["sequence_no"] for item in events.json()["items"]] == [2]
    finally:
        app.dependency_overrides.pop(require_admin, None)


def test_admin_trace_api_hides_data_without_admin(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / "admin-traces-auth.db")
    import dashboard.backend.api.routers.admin_traces as router_module

    monkeypatch.setattr(router_module, "trace_store", store)
    response = TestClient(app).get("/api/admin/traces")
    assert response.status_code in {401, 403}
