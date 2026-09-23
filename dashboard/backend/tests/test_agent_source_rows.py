"""Agent ownership rows for the analytics backfill come from the agent store."""

from __future__ import annotations

from dashboard.backend.domain.agents.repository import AgentStore


def test_list_agent_source_rows_returns_ownership_in_creation_order(tmp_path):
    store = AgentStore(tmp_path / "agents.db")
    owned = store.create_agent(name="Owned", owner_user_id=1, session_id="session-1")
    guest = store.create_agent(name="Guest", owner_browser_session="browser-1")

    rows = store.list_agent_source_rows()

    assert [row["agent_id"] for row in rows] == [owned["agent_id"], guest["agent_id"]]
    assert set(rows[0]) == {"agent_id", "session_id", "owner_user_id", "created_at"}
    assert rows[0]["owner_user_id"] == 1
    assert rows[1]["owner_user_id"] is None
