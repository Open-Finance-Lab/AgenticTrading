from fastapi.testclient import TestClient

from dashboard.backend.api.auth import require_admin
from dashboard.backend.app import app


def test_admin_user_agent_activity_joins_agents_runs_and_traces(monkeypatch):
    import dashboard.backend.api.routers.admin_users as module
    import dashboard.backend.domain.agents.repository as agents
    import dashboard.backend.domain.runs.repository as runs
    import dashboard.backend.domain.traces.repository as traces

    class Users:
        def get_user_admin(self, user_id):
            return {"id": user_id} if user_id == 7 else None

    class Agents:
        def list_agents(self, *, owner_user_id):
            assert owner_user_id == 7
            return [{"agent_id": "agent-1", "name": "Momentum", "agent_type": "builtin", "created_at": "t1"}]

    class Runs:
        def list_runs(self, agent_id):
            assert agent_id == "agent-1"
            return [{"run_id": "run-1", "status": "running", "created_at": "t2"}]

    class Traces:
        def list_traces(self, *, agent_ids, limit, offset):
            assert agent_ids == ["agent-1"]
            assert limit == 50
            assert offset == 0
            return {
                "items": [{
                    "agent_id": "agent-1",
                    "run_id": "run-1",
                    "trace_id": "trace-1",
                    "status": "running",
                    "created_at": "t3",
                }],
                "has_more": False,
                "next_cursor": None,
            }

    monkeypatch.setattr(module.users_module, "user_store", Users())
    monkeypatch.setattr(agents, "agent_store", Agents())
    monkeypatch.setattr(runs, "run_store", Runs())
    monkeypatch.setattr(traces, "trace_store", Traces())
    app.dependency_overrides[require_admin] = lambda: {"id": 1, "is_admin": True}
    try:
        response = TestClient(app).get("/api/admin/users/7/agent-activity")
        assert response.status_code == 200
        assert response.json()["agents"][0]["runs"][0]["trace_id"] == "trace-1"
    finally:
        app.dependency_overrides.pop(require_admin, None)


def test_admin_user_agent_activity_returns_404_for_unknown_user(monkeypatch):
    import dashboard.backend.api.routers.admin_users as module

    class Users:
        def get_user_admin(self, _user_id):
            return None

    monkeypatch.setattr(module.users_module, "user_store", Users())
    app.dependency_overrides[require_admin] = lambda: {"id": 1, "is_admin": True}
    try:
        response = TestClient(app).get("/api/admin/users/999/agent-activity")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(require_admin, None)
