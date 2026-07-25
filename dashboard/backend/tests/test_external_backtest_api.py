"""Tests for external agent backtest API."""

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.app as app_module
import dashboard.backend.database as db_module
import dashboard.backend.domain.backtesting.external_run_service as svc

from dashboard.backend.app import app
from dashboard.backend.database import BacktestDatabase
from dashboard.backend.domain.backtesting.external_run_service import (
    _sessions,
    get_decision_format,
    parse_actions_payload,
)


@pytest.fixture
def client(temp_db, monkeypatch):
    monkeypatch.setattr(app_module, "db", temp_db)
    monkeypatch.setattr(db_module, "db", temp_db)
    monkeypatch.setattr(svc, "db", temp_db)
    _sessions.clear()
    return TestClient(app)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    test_db = BacktestDatabase(db_path=db_path)
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
    decisions, err = parse_actions_payload(payload)
    assert err is None
    assert len(decisions) == 1


def test_get_decision_format():
    fmt = get_decision_format()
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

    legacy_db = BacktestDatabase(db_path=db_path)
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
