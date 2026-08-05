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

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token

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


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.agents as agents_api

    db_path = tmp_path / "test.db"
    test_agents = AgentStore(db_path=db_path)
    test_db = db_module.BacktestDatabase(db_path=db_path)
    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    return TestClient(app)


def _headers():
    return {"X-Session-Id": str(uuid.uuid4())}


def test_create_accepts_backtest_allocation(client):
    headers = _headers()
    resp = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "backtest_allocation": 2500},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 2500


def test_patch_updates_backtest_allocation(client):
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 2500},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["backtest_allocation"] == 2500


def test_patch_backtest_allocation_alone_is_not_no_fields_to_update(client):
    """It must satisfy the 'at least one field' guard on its own."""
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 2000},
        headers=headers,
    )
    assert resp.status_code != 400


@pytest.mark.parametrize("bad", [0, -100, 3001])
def test_backtest_allocation_out_of_range_is_rejected(client, bad):
    headers = _headers()
    resp = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "backtest_allocation": bad},
        headers=headers,
    )
    assert resp.status_code == 422


def test_backtest_allocation_does_not_change_the_paper_sleeve(client):
    """Simulated money must never move the real sleeve."""
    headers = _headers()
    created = client.post(
        "/api/v1/agents",
        json={"name": "alpha", "agent_type": "builtin", "cash_allocation": 1000},
        headers=headers,
    ).json()["agent"]

    updated = client.patch(
        f"/api/v1/agents/{created['agent_id']}",
        json={"backtest_allocation": 2500},
        headers=headers,
    ).json()["agent"]

    assert updated["cash_allocation"] == 1000
    assert updated["backtest_allocation"] == 2500


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    """Like ``client``, but signed in: also wires the user + portfolio stores
    to the same content DB, so a PATCH carrying both cash_allocation and
    backtest_allocation exercises the real ledger-write branch
    (``ctx["user_id"]`` is not None) -- the one path where a real-money write
    and a simulated-money write interleave (api/routers/agents.py:385-404).
    """
    import dashboard.backend.api.routers.agents as agents_api
    import dashboard.backend.domain.portfolios.repository as portfolio_repo
    import dashboard.backend.domain.portfolios.service as portfolio_service_module
    import dashboard.backend.users as users_module

    content_db = tmp_path / "content.db"
    test_agents = AgentStore(db_path=content_db)
    test_portfolio = portfolio_repo.PortfolioStore(db_path=content_db)
    test_db = db_module.BacktestDatabase(db_path=content_db)
    test_users = users_module.UserStore(db_path=tmp_path / "users.db")

    monkeypatch.setattr(agent_store_module, "agent_store", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "agents", test_agents)
    monkeypatch.setattr(agents_api.agent_service, "db", test_db)
    monkeypatch.setattr(db_module, "db", test_db)
    monkeypatch.setattr(portfolio_repo, "portfolio_store", test_portfolio)
    monkeypatch.setattr(portfolio_service_module, "portfolio_store", test_portfolio)
    monkeypatch.setattr(users_module, "user_store", test_users)
    return TestClient(app)


def _signup(client, email="backtest-alloc@example.com"):
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Backtest Alloc User",
            "password": "securepass1",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "token" not in data
    return _cookie_session_token(client), data["user"]


def _auth_headers(token, browser="browser-backtest-alloc-1"):
    return {
        "Authorization": f"Bearer {token}",
        "X-Browser-Id": browser,
        "X-Session-Id": browser,
    }


def test_signed_in_combined_patch_moves_only_the_real_sleeve(authed_client):
    """Signed-in round trip: both fields survive a combined PATCH without
    backtest_allocation being lost to the route's ``_UNSET`` substitution.

    Does NOT prove the ledger branch (api/routers/agents.py:369-384) executed:
    _reconcile derives cash_available from the sum of agent allocations on
    every read, so the cash_available assertion would pass even if that code
    were deleted.
    """
    token, _ = _signup(authed_client)
    headers = _auth_headers(token)

    created = authed_client.post(
        "/api/v1/agents",
        headers=headers,
        json={"name": "Combo", "model_name": "local-model", "cash_allocation": 1000},
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["agent"]["agent_id"]

    before = authed_client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]

    updated = authed_client.patch(
        f"/api/v1/agents/{agent_id}",
        headers=headers,
        json={"cash_allocation": 2500, "backtest_allocation": 2000},
    )
    assert updated.status_code == 200, updated.text
    agent = updated.json()["agent"]
    assert agent["cash_allocation"] == 2500
    assert agent["backtest_allocation"] == 2000

    after = authed_client.get("/api/v1/portfolio", headers=headers).json()["portfolio"]
    assert after["cash_available"] == before["cash_available"] - 1500


def test_patch_with_an_empty_pipeline_clears_the_instruction(client):
    """Empty instruction -> no pipeline -> the backend's default hourly prompt.

    portfolio_manager takes the ``create_prompt`` branch when an agent has no
    pipeline, so clearing is what "use the default" means end to end.
    """
    headers = _headers()
    created = client.post(
        "/api/v1/agents", json={"name": "alpha", "agent_type": "builtin"}, headers=headers
    ).json()["agent"]
    assert created["pipeline"], "builtin agents are seeded with a starter pipeline"

    resp = client.patch(
        f"/api/v1/agents/{created['agent_id']}", json={"pipeline": []}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert not resp.json()["agent"]["pipeline"]

    reread = client.get(f"/api/v1/agents/{created['agent_id']}", headers=headers)
    assert not reread.json()["agent"]["pipeline"]
