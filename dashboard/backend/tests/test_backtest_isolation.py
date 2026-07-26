"""
Negative tests for backtest session isolation.
Uses a temporary test database.
"""

import pytest
import tempfile
from pathlib import Path

# Add backend to path
from fastapi.testclient import TestClient
import uuid

@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    import dashboard.backend.database as db_module
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        test_db = db_module.BacktestDatabase(db_path=db_path)
        yield test_db
        # Cleanup happens automatically when tmpdir is deleted

@pytest.fixture
def client(temp_db, monkeypatch):
    """Test client with temporary database."""
    # Patch the database module to use temp DB. The backtests router binds
    # `db` at import time (`from ... import db`), so its module attribute
    # must be patched too or its routes silently keep using the global DB.
    import dashboard.backend.app as app_module
    import dashboard.backend.database as db_module
    import dashboard.backend.api.routers.backtests as backtests_module

    monkeypatch.setattr(app_module, "db", temp_db)
    monkeypatch.setattr(db_module, "db", temp_db)
    monkeypatch.setattr(backtests_module, "db", temp_db)

    return TestClient(app_module.app)

def test_runs_listing_is_public_by_design(client, temp_db):
    """GET /runs is a PUBLIC listing (run metadata only): the route docstring
    states backtest results are meant to be shared/viewed, and the dashboard
    home page lists the seed runs without any session header. Per-run DATA
    access stays session-scoped (see test_session_cannot_access_other_backtest
    and the /equity route). This replaces an older test that pinned a strict
    per-session listing the product deliberately does not have."""

    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    run_a = str(uuid.uuid4())
    run_b = str(uuid.uuid4())

    temp_db.insert_run(
        run_id=run_a,
        session_id=session_a,
        agent_name="Agent A",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )
    temp_db.insert_run(
        run_id=run_b,
        session_id=session_b,
        agent_name="Agent B",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )

    # The listing is the same for everyone — with or without a session header.
    for headers in ({}, {'X-Session-Id': session_a}):
        response = client.get('/runs', headers=headers)
        assert response.status_code == 200
        names = {r['agent_name'] for r in response.json()}
        assert {"Agent A", "Agent B"} <= names

def test_session_cannot_access_other_backtest(client, temp_db):
    """Session A should get 404 when accessing Session B's run_id."""
    
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    run_id_b = str(uuid.uuid4())
    
    # Create backtest in Session B
    temp_db.insert_run(
        run_id=run_id_b,
        session_id=session_b,
        agent_name="Agent B",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )
    
    # Session A tries to access run from Session B: should get 404
    response_a = client.get(
        f'/runs/{run_id_b}',
        headers={'X-Session-Id': session_a}
    )
    assert response_a.status_code == 404

def test_latest_run_is_session_specific(client, temp_db):
    """latest-run should return only this session's latest backtest."""
    import time
    
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    run_id_a1 = str(uuid.uuid4())
    run_id_a2 = str(uuid.uuid4())
    run_id_b = str(uuid.uuid4())
    
    # /runs/latest/metrics only considers the internal hourly agent's runs
    # (agent_name == 'Agent'); the old fixture's "Agent A"/"Agent B" names
    # never matched the route's filter and 404'd.
    # Create 2 backtests in Session A (with 1+ second delay to ensure different timestamps)
    temp_db.insert_run(
        run_id=run_id_a1,
        session_id=session_a,
        agent_name="Agent",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )
    time.sleep(1.1)  # Ensure different second (SQLite CURRENT_TIMESTAMP has second precision)
    temp_db.insert_run(
        run_id=run_id_a2,
        session_id=session_a,
        agent_name="Agent",
        mode="backtest",
        start_date="2024-02-01",
        end_date="2024-02-28",
        initial_equity=100000
    )

    # Create 1 backtest in Session B
    temp_db.insert_run(
        run_id=run_id_b,
        session_id=session_b,
        agent_name="Agent",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )
    
    # Session A's latest-run should return a2
    response_a = client.get('/runs/latest/metrics', headers={'X-Session-Id': session_a})
    assert response_a.status_code == 200
    assert response_a.json()['run_id'] == run_id_a2
    
    # Session B's latest-run should return b
    response_b = client.get('/runs/latest/metrics', headers={'X-Session-Id': session_b})
    assert response_b.status_code == 200
    assert response_b.json()['run_id'] == run_id_b

def test_public_listing_needs_no_session_header(client):
    """GET /runs (public listing) must work without X-Session-Id — the
    dashboard home page fetches it before any session exists. (Replaces an
    older test that expected a 400 here; see
    test_runs_listing_is_public_by_design.)"""
    response = client.get('/runs')
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_invalid_session_id_fails_closed_on_scoped_routes(client, temp_db):
    """A malformed session id must never unlock another session's run: the
    session middleware rejects non-UUID ids with 400 on session-scoped
    routes (the public /runs listing and /paper/* are exempt)."""
    run_id = str(uuid.uuid4())
    temp_db.insert_run(
        run_id=run_id,
        session_id=str(uuid.uuid4()),
        agent_name="Agent",
        mode="backtest",
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_equity=100000
    )
    response = client.get(f'/runs/{run_id}', headers={'X-Session-Id': 'not-a-uuid'})
    assert response.status_code == 400
    assert 'Invalid' in str(response.json())

def test_paper_trading_no_session_required(client):
    """Paper trading routes should NOT require X-Session-Id."""
    # Should not return 400 for missing header
    response = client.get('/paper/account')
    assert response.status_code != 400


def test_app_dashboard_no_session_required(client):
    """Dashboard HTML at /app must load without X-Session-Id (browser navigation)."""
    response = client.get('/app')
    assert response.status_code == 200
    assert 'text/html' in response.headers.get('content-type', '')
    assert 'Missing X-Session-Id' not in response.text
