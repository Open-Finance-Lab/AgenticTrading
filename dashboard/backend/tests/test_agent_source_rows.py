"""Agent ownership rows for the analytics backfill come from the agent store."""

from __future__ import annotations

from pathlib import Path

from dashboard.backend.domain.agents.repository import AgentStore


def test_list_agent_source_rows_returns_ownership_in_creation_order(tmp_path):
    store = AgentStore(tmp_path / "agents.db")
    owned = store.create_agent(name="Owned", owner_user_id=1, session_id="session-1")
    guest = store.create_agent(name="Guest", owner_browser_session="browser-1")
    # created_at has second resolution and the SQL tie-breaks on the random
    # agent_id, so pin the ordering the store is being asked to guarantee.
    import sqlite3
    with sqlite3.connect(Path(tmp_path) / "agents.db") as conn:
        conn.execute(
            "UPDATE external_agents SET created_at = ? WHERE agent_id = ?",
            ("2026-09-01T00:00:00", owned["agent_id"]),
        )
        conn.execute(
            "UPDATE external_agents SET created_at = ? WHERE agent_id = ?",
            ("2026-09-02T00:00:00", guest["agent_id"]),
        )

    rows = store.list_agent_source_rows()

    assert [row["agent_id"] for row in rows] == [owned["agent_id"], guest["agent_id"]]
    assert set(rows[0]) == {"agent_id", "session_id", "owner_user_id", "created_at"}
    assert rows[0]["owner_user_id"] == 1
    assert rows[1]["owner_user_id"] is None
