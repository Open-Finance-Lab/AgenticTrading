"""Tests for external agent backtest API."""

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.app as app_module
import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as svc


@pytest.fixture
def client(temp_db, monkeypatch):
    monkeypatch.setattr(app_module, "db", temp_db)
    monkeypatch.setattr(db_module, "db", temp_db)
    monkeypatch.setattr(svc, "db", temp_db)
    svc._sessions.clear()
    return TestClient(app_module.app)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    test_db = db_module.BacktestDatabase(db_path=db_path)
    yield test_db


def test_decision_schema_endpoint(client):
    resp = client.get("/api/v1/backtest/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "format" in data
    assert "valid_symbols" in data
    assert len(data["valid_symbols"]) == 30


def test_start_requires_session(client):
    resp = client.post(
        "/api/v1/backtest/start",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
    )
    assert resp.status_code == 400


def _fake_session(svc_mod, session_id, status="loading"):
    """A resident session without loading any market data."""
    s = svc_mod.ExternalBacktestSession(
        backtest_id=f"bt_{session_id}_{len(svc_mod._sessions)}",
        session_id=session_id,
        agent_name="a",
        model_name="m",
        start_date="2026-04-15",
        end_date="2026-04-16",
    )
    s.status = status
    svc_mod._sessions[s.backtest_id] = s
    return s


def test_legacy_start_is_capped_per_session(client, monkeypatch):
    """This route authenticates nothing and used to have no limit at all.

    ``_require_session`` accepts any non-empty X-Session-Id, so a bare POST
    spawned a thread and pinned a market-data window with no account, no agent
    and no API key behind it. It also writes no ``protocol_runs`` row, so the
    per-agent / per-account / global protocol caps are all blind to it: the
    entitlement plane could hold every authenticated agent to its quota while
    this ran unbounded beside it.
    """
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_PER_SESSION", 2)
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_GLOBAL", 0)  # isolate the per-session budget
    session_id = "legacy-session"
    _fake_session(svc, session_id)
    _fake_session(svc, session_id)

    resp = client.post(
        "/api/v1/backtest/start",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-Session-Id": session_id},
    )
    assert resp.status_code == 429, resp.text
    assert "session" in resp.json()["detail"]
    assert resp.headers["Retry-After"] == "30"

    # Another caller is unaffected — the budget is per session, not global here.
    other = client.post(
        "/api/v1/backtest/start",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-Session-Id": "someone-else"},
    )
    assert other.status_code in (200, 500), other.text


def test_legacy_start_is_capped_server_wide(client, monkeypatch):
    """The global ceiling is the one a header-rotating caller cannot dodge."""
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_GLOBAL", 2)
    _fake_session(svc, "one")
    _fake_session(svc, "two")

    resp = client.post(
        "/api/v1/backtest/start",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-Session-Id": "three"},
    )
    assert resp.status_code == 429, resp.text
    assert "server" in resp.json()["detail"]


def test_terminal_sessions_do_not_consume_the_legacy_budget(client, monkeypatch):
    """A finished run holds no market data, so it must not hold a slot."""
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_PER_SESSION", 1)
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_GLOBAL", 0)
    _fake_session(svc, "done", status="completed")
    _fake_session(svc, "done", status="failed")
    assert svc.count_active_sessions("done") == 0

    resp = client.post(
        "/api/v1/backtest/start",
        json={"start_date": "2026-04-15", "end_date": "2026-04-16"},
        headers={"X-Session-Id": "done"},
    )
    assert resp.status_code != 429, resp.text


def test_protocol_creates_do_not_pay_the_legacy_budget(monkeypatch):
    """The protocol surfaces are already capped three ways above this layer.

    Charging them a second, differently-keyed budget here would refuse creates
    that ``run_service.create_run`` had just authorised — so the legacy budget
    is opt-in, and only the legacy route opts in.
    """
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_PER_SESSION", 1)
    monkeypatch.setattr(svc, "MAX_LEGACY_ACTIVE_GLOBAL", 1)
    _fake_session(svc, "proto")

    kwargs = dict(
        session_id="proto",
        agent_name="a",
        model_name="m",
        start_date="2026-04-15",
        end_date="2026-04-16",
    )
    # Opted in (the legacy route): refused.
    with pytest.raises(svc.BacktestCapacityError):
        svc.start_backtest(enforce_session_cap=True, **kwargs)
    # Default (run_service.create_run's call): unaffected.
    started = svc.start_backtest(**kwargs)
    assert started["status"] == "loading"


def test_parse_actions_payload_valid():
    payload = {
        "actions": [{
            "action": "hold",
            "symbol": "AAPL",
            "confidence": 0.5,
            "reasoning": "No signal this hour",
            "position_size": 0,
        }]
    }
    decisions, err = svc.parse_actions_payload(payload)
    assert err is None
    assert len(decisions) == 1


def test_get_decision_format():
    fmt = svc.get_decision_format()
    assert "actions" in fmt


def test_insert_trades_legacy_schema(tmp_path):
    """Legacy DBs use shares/action columns; insert_trades must not crash."""
    db_path = tmp_path / "legacy.db"
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE agent_runs (
            run_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            initial_equity REAL NOT NULL,
            final_equity REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            num_trades INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            llm_model TEXT DEFAULT 'rule-based'
        )
    """)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            shares INTEGER,
            price REAL,
            total_value REAL
        )
    """)
    conn.commit()
    conn.close()

    legacy_db = db_module.BacktestDatabase(db_path=db_path)
    legacy_db.insert_trades("ext_test", [{
        "timestamp": "2026-04-15T14:00:00",
        "symbol": "AAPL",
        "side": "BUY",
        "shares": 10,
        "price": 150.0,
        "cost": 1500.0,
        "reason": "test",
    }])
    rows = legacy_db.get_trades("ext_test")
    assert len(rows) == 1
    assert rows[0]["quantity"] == 10
    assert rows[0]["side"] == "BUY"
