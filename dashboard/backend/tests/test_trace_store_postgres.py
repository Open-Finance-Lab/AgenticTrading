import pytest


def test_build_trace_store_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.domain.traces.repository as repo_module

    monkeypatch.delenv("CONTENT_DATABASE_URL", raising=False)
    store = repo_module._build_trace_store()

    assert isinstance(store, repo_module.TraceStore)
    assert "trace_store backend: sqlite" in capsys.readouterr().out


def test_build_trace_store_picks_postgres_without_exposing_url(monkeypatch, capsys):
    import dashboard.backend.domain.traces.repository as repo_module
    import dashboard.backend.domain.traces.repository_postgres as repo_pg_module

    created = {}

    class FakePostgresTraceStore:
        def __init__(self, database_url):
            created["database_url"] = database_url

    monkeypatch.setattr(repo_pg_module, "PostgresTraceStore", FakePostgresTraceStore)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://user:secret@host/db")

    store = repo_module._build_trace_store()

    assert isinstance(store, FakePostgresTraceStore)
    assert created["database_url"].endswith("/db")
    assert "secret" not in capsys.readouterr().out


@pytest.mark.skipif(
    not __import__("os").getenv("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)
def test_trace_postgres_round_trip():
    import os

    from dashboard.backend.domain.traces.repository_postgres import PostgresTraceStore

    store = PostgresTraceStore(os.environ["TEST_POSTGRES_URL"])
    trace_id = "trace_pg_test"
    with store._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_trace_events WHERE trace_id = %s", (trace_id,))
            cur.execute("DELETE FROM agent_traces WHERE trace_id = %s", (trace_id,))
    store.create_trace(trace_id=trace_id, run_id="run_pg_test")
    first = store.append_event(
        trace_id=trace_id,
        event_type="run_started",
        actor_type="system",
        payload={"environment_id": "test"},
        idempotency_key="start:pg",
    )
    assert first["sequence_no"] == 1
    assert store.append_event(
        trace_id=trace_id,
        event_type="run_started",
        actor_type="system",
        payload={"changed": True},
        idempotency_key="start:pg",
    ) == first
