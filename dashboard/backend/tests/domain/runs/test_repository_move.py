"""Phase 3B1 — run repository move + characterization.

Verifies identity/re-export, the domain->api/scripts import boundary, and that
``RunStore`` behaves exactly as before. All tests use an isolated temporary
SQLite database (never the live DB).
"""

import ast
from pathlib import Path

import pytest

from dashboard.backend.domain.runs import repository
from dashboard.backend.domain.runs.repository import RunStore

_RUN_KEYS = {
    "run_id", "agent_id", "agent_version_id", "session_id", "environment_id",
    "environment_type", "config", "backtest_id", "result_run_id", "status",
    "step_index", "total_steps", "created_at", "updated_at",
}


@pytest.fixture
def store(tmp_path):
    return RunStore(db_path=tmp_path / "runs.db")


def _make_run(store, **overrides):
    params = dict(
        agent_id="agent_1",
        agent_version_id="agv_1",
        session_id="sess-1",
        environment_id="env-1",
        environment_type="backtest",
        config={"start_date": "2026-04-15", "end_date": "2026-04-16"},
    )
    params.update(overrides)
    return store.create_run(**params)


# ---------------------------------------------------------------------------
# Canonical identity
# ---------------------------------------------------------------------------

def test_canonical_module_identity():
    assert repository.RunStore.__module__ == "dashboard.backend.domain.runs.repository"


def test_singleton_uses_test_database():
    from dashboard.backend.database import DB_PATH

    assert Path(repository.run_store.db_path) == Path(DB_PATH)
    assert "storage/data/backtest.db" not in str(repository.run_store.db_path)


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------

def test_domain_module_does_not_import_api_or_scripts():
    tree = ast.parse(Path(repository.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
        assert m != "fastapi" and not m.startswith("fastapi."), m


# ---------------------------------------------------------------------------
# RunStore characterization
# ---------------------------------------------------------------------------

def test_create_run_schema_and_defaults(store):
    run = _make_run(store)
    assert set(run.keys()) == _RUN_KEYS
    assert run["run_id"].startswith("run_")
    assert run["agent_id"] == "agent_1"
    assert run["agent_version_id"] == "agv_1"
    assert run["session_id"] == "sess-1"
    assert run["environment_id"] == "env-1"
    assert run["environment_type"] == "backtest"
    assert run["config"] == {"start_date": "2026-04-15", "end_date": "2026-04-16"}
    assert run["backtest_id"] is None
    assert run["result_run_id"] is None
    assert run["status"] == "created"
    assert run["created_at"] and run["updated_at"]


def test_create_run_with_backtest_and_status(store):
    run = _make_run(store, backtest_id="bt_1", status="running")
    assert run["backtest_id"] == "bt_1"
    assert run["status"] == "running"


def test_get_run_and_missing(store):
    created = _make_run(store)
    fetched = store.get_run(created["run_id"])
    assert fetched["run_id"] == created["run_id"]
    assert store.get_run("run_missing") is None


def test_config_roundtrips_as_dict(store):
    run = _make_run(store, config={"symbols": ["AAPL", "MSFT"], "mode": "safe_trading"})
    fetched = store.get_run(run["run_id"])
    assert fetched["config"] == {"symbols": ["AAPL", "MSFT"], "mode": "safe_trading"}


def test_update_run_status_transition(store):
    run = _make_run(store, status="running")
    store.update_run(run["run_id"], status="completed", result_run_id="ext_42")
    fetched = store.get_run(run["run_id"])
    assert fetched["status"] == "completed"
    assert fetched["result_run_id"] == "ext_42"


def test_update_run_coalesce_preserves_existing(store):
    run = _make_run(store, backtest_id="bt_1", status="running")
    # Passing None must not overwrite existing values (COALESCE semantics).
    store.update_run(run["run_id"], status="completed")
    fetched = store.get_run(run["run_id"])
    assert fetched["backtest_id"] == "bt_1"
    assert fetched["status"] == "completed"


def test_list_runs_filters_by_agent(store):
    a1 = _make_run(store, agent_id="agent_A")
    a2 = _make_run(store, agent_id="agent_A")
    _make_run(store, agent_id="agent_B")

    listed = store.list_runs("agent_A")
    ids = {r["run_id"] for r in listed}
    assert ids == {a1["run_id"], a2["run_id"]}
    assert store.list_runs("agent_None") == []


def test_failed_status_transition(store):
    run = _make_run(store, status="running")
    store.update_run(run["run_id"], status="failed")
    assert store.get_run(run["run_id"])["status"] == "failed"
