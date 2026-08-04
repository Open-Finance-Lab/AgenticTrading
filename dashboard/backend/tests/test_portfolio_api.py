"""Portfolio store + GET /api/v1/portfolio (#174)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token

import dashboard.backend.domain.agents.repository as agent_repo
import dashboard.backend.domain.portfolios.repository as portfolio_repo
import dashboard.backend.domain.portfolios.service as portfolio_service_module
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.backtesting.constants import DEFAULT_PORTFOLIO_EQUITY

# These four are imported as *modules*, not `from ... import UserStore`, because
# the fixture below has to monkeypatch attributes on the module objects. Keeping
# a single import form per module also keeps CodeQL's py/import-and-import-from
# quiet.


@pytest.fixture
def temp_stores(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        content_db = root / "content.db"
        user_store = users_module.UserStore(db_path=root / "users.db")
        portfolio_store = portfolio_repo.PortfolioStore(db_path=content_db)
        agent_store = agent_repo.AgentStore(db_path=content_db)

        monkeypatch.setattr(users_module, "user_store", user_store)
        monkeypatch.setattr(portfolio_repo, "portfolio_store", portfolio_store)
        monkeypatch.setattr(portfolio_service_module, "portfolio_store", portfolio_store)
        # cash_available is derived from the agent sleeves (service._reconcile),
        # so the agent store has to be isolated too -- otherwise agents another
        # test left in the session-wide DB under the same low user id get summed
        # into these accounts' allocations.
        monkeypatch.setattr(agent_repo, "agent_store", agent_store)
        yield user_store, portfolio_store


@pytest.fixture
def client(temp_stores):
    return TestClient(app)


def _signup(client: TestClient, email: str = "pf@example.com") -> str:
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "PF User",
            "password": "securepass1",
        },
    )
    assert resp.status_code == 200, resp.text
    assert "token" not in resp.json()
    return _cookie_session_token(client)


def test_signup_lands_in_the_fixture_user_store(client, temp_stores):
    """Guards call-time store resolution in api/auth.py (issue #185).

    If any auth route goes back to binding the singleton at import time, the
    temp users.db stays empty and every account these tests create leaks into
    the session-wide DB -- making the fixed email above an ordering hazard for
    any other test that signs up with it. The failure is invisible otherwise:
    the tests still pass, against the wrong store.
    """
    user_store, _ = temp_stores
    token = _signup(client)
    assert user_store.get_user_for_token(token) is not None


def test_get_or_create_bootstraps_10k_and_is_idempotent(temp_stores):
    _, store = temp_stores
    first = store.get_or_create(7)
    assert first["owner_user_id"] == 7
    assert first["equity"] == float(DEFAULT_PORTFOLIO_EQUITY)
    assert first["cash_available"] == float(DEFAULT_PORTFOLIO_EQUITY)
    assert first["allocated"] == 0.0

    second = store.get_or_create(7)
    assert second == first


def test_store_keeps_one_row_per_user(temp_stores):
    _, store = temp_stores
    alice = store.get_or_create(1)
    bob = store.create(2, equity=250.0)

    assert alice["owner_user_id"] == 1
    assert bob["owner_user_id"] == 2
    assert bob["equity"] == 250.0
    # Bootstrapping Bob must not have disturbed Alice's balance.
    assert store.get(1) == alice
    assert store.get(999) is None


def test_portfolio_requires_auth(client):
    assert client.get("/api/v1/portfolio").status_code == 401


def test_portfolio_get_bootstraps_for_signed_in_user(client):
    token = _signup(client)
    headers = {"Authorization": f"Bearer {token}"}

    first = client.get("/api/v1/portfolio", headers=headers)
    assert first.status_code == 200, first.text
    body = first.json()["portfolio"]
    assert body["equity"] == float(DEFAULT_PORTFOLIO_EQUITY)
    assert body["cash_available"] == float(DEFAULT_PORTFOLIO_EQUITY)
    assert body["allocated"] == 0.0
    assert body["owner_user_id"]

    second = client.get("/api/v1/portfolio", headers=headers)
    assert second.status_code == 200
    assert second.json()["portfolio"] == body


def test_each_account_gets_its_own_portfolio(client, temp_stores):
    """The row is keyed off current_user, never off anything the client sends."""
    _, store = temp_stores
    alice_token = _signup(client, "alice-pf@example.com")
    bob_token = _signup(client, "bob-pf@example.com")

    alice = client.get(
        "/api/v1/portfolio", headers={"Authorization": f"Bearer {alice_token}"}
    ).json()["portfolio"]
    bob = client.get(
        "/api/v1/portfolio", headers={"Authorization": f"Bearer {bob_token}"}
    ).json()["portfolio"]

    assert alice["owner_user_id"] != bob["owner_user_id"]

    # Draining Alice's cash must leave Bob's untouched -- proves the two reads
    # are not accidentally serving one shared row.
    conn = store._get_connection()
    conn.execute(
        "UPDATE user_portfolios SET cash_available = 0 WHERE owner_user_id = ?",
        (alice["owner_user_id"],),
    )
    conn.commit()
    conn.close()

    assert store.get(alice["owner_user_id"])["cash_available"] == 0.0
    refetched_bob = client.get(
        "/api/v1/portfolio", headers={"Authorization": f"Bearer {bob_token}"}
    ).json()["portfolio"]
    assert refetched_bob["cash_available"] == float(DEFAULT_PORTFOLIO_EQUITY)


def test_expired_session_is_rejected_rather_than_bootstrapping_a_portfolio(client):
    """A stale token must 401, not silently mint a fresh $10k ledger."""
    assert (
        client.get(
            "/api/v1/portfolio", headers={"Authorization": "Bearer not-a-real-token"}
        ).status_code
        == 401
    )
