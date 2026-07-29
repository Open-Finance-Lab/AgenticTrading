"""``backtest_allocation``: a saved per-agent simulated-capital setting.

Backtest capital used to be a per-run value reseeded from the paper sleeve on
every Run Backtest modal open. Consolidating both capital fields into one
Configure card (2026-07-29) makes it a stored column, which means it has to
exist on *both* twins -- see tests/test_store_twin_parity.py for why a
one-twin column is a prod-only 500.

Unlike ``cash_allocation`` this is simulated money: it must never move the
portfolio ledger.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
import dashboard.backend.domain.agents.repository as agent_store_module
import dashboard.backend.database as db_module

AgentStore = agent_store_module.AgentStore


@pytest.fixture
def store(tmp_path):
    return AgentStore(db_path=tmp_path / "agents.db")


def test_backtest_allocation_round_trips_through_create(store):
    agent = store.create_agent(name="alpha", backtest_allocation=2500)
    assert agent["backtest_allocation"] == 2500

    reread = store.get_agent(agent["agent_id"])
    assert reread["backtest_allocation"] == 2500


def test_backtest_allocation_defaults_to_none(store):
    """Existing agents have a NULL column and must keep today's behaviour."""
    agent = store.create_agent(name="legacy")
    assert agent["backtest_allocation"] is None


def test_update_agent_sets_backtest_allocation(store):
    agent = store.create_agent(name="alpha")
    updated = store.update_agent(agent["agent_id"], backtest_allocation=4000)
    assert updated["backtest_allocation"] == 4000


def test_update_agent_leaves_backtest_allocation_alone_when_omitted(store):
    """The _UNSET sentinel means 'do not touch', not 'set to None'."""
    agent = store.create_agent(name="alpha", backtest_allocation=2500)
    updated = store.update_agent(agent["agent_id"], name="renamed")
    assert updated["backtest_allocation"] == 2500
    assert updated["name"] == "renamed"
