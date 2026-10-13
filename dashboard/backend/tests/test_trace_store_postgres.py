import pytest


def test_build_trace_store_defaults_to_sqlite(monkeypatch, capsys):
    import dashboard.backend.domain.traces.repository as repo_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
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
    monkeypatch.setenv("AGENT_RUNS_DATABASE_URL", "postgresql://user:secret@host/db")

    store = repo_module._build_trace_store()

    assert isinstance(store, FakePostgresTraceStore)
    assert created["database_url"].endswith("/db")
    out = capsys.readouterr().out
    assert "secret" not in out
    assert "trace_store backend: postgres (host/db)" in out


def test_build_trace_store_ignores_the_content_database(monkeypatch, capsys):
    """Traces are run data: the decision tape writes a pair per bar of every
    dashboard backtest. They live with run history (AGENT_RUNS_DATABASE_URL),
    never in the auth-critical users/content database, and never fall back
    to it."""
    import dashboard.backend.domain.traces.repository as repo_module

    monkeypatch.delenv("AGENT_RUNS_DATABASE_URL", raising=False)
    monkeypatch.setenv("CONTENT_DATABASE_URL", "postgresql://user:secret@content/db")
    store = repo_module._build_trace_store()

    assert isinstance(store, repo_module.TraceStore)
    assert "trace_store backend: sqlite" in capsys.readouterr().out


@pytest.mark.skipif(
    not __import__("os").getenv("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL not set; skipping live-Postgres tests",
)
def test_trace_postgres_round_trip():
    import os
    import dashboard.backend.domain.traces.repository_postgres as repo_pg_module

    store = repo_pg_module.PostgresTraceStore(os.environ["TEST_POSTGRES_URL"])
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
    batch = [
        {"event_type": "decision_recorded", "actor_type": "agent", "payload": {"i": i},
         "idempotency_key": f"batch:pg:{i}"}
        for i in range(3)
    ]
    assert store.append_events(trace_id=trace_id, events=batch) == 3
    assert store.append_events(trace_id=trace_id, events=batch) == 0
    sequences = [e["sequence_no"] for e in store.list_events(trace_id)["items"]]
    assert sequences == [1, 2, 3, 4]
