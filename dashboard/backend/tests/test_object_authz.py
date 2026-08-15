"""Object-level authorization regressions (cross-account isolation)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# Import modules (not ``from … import Store``) so monkeypatches on
# ``module.store`` are visible at call time, and CodeQL's
# py/import-and-import-from stays quiet.
import dashboard.backend.domain.agents.repository as agent_repo
import dashboard.backend.domain.portfolios.repository as portfolio_repo
import dashboard.backend.api.routers.credits as credits_router
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        user_store = users_module.UserStore(db_path=root / "users.db")
        agent_store = agent_repo.AgentStore(db_path=root / "content.db")
        portfolio_store = portfolio_repo.PortfolioStore(db_path=root / "content.db")
        credits_store = CreditsStore(db_path=root / "users.db")
        monkeypatch.setattr(users_module, "user_store", user_store)
        monkeypatch.setattr(agent_repo, "agent_store", agent_store)
        monkeypatch.setattr(portfolio_repo, "portfolio_store", portfolio_store)
        # A real service, not SimpleNamespace(store=...). The stub only worked
        # while the router reached past the service into the repository; the
        # ownership scoping this module tests belongs to the service, so a stub
        # that fakes it away would leave the boundary untested.
        monkeypatch.setattr(
            credits_router,
            "credits_service",
            CreditsService(store=credits_store, gateway=SimpleNamespace()),
        )
        yield TestClient(app)


def _signup(client: TestClient, email: str) -> str:
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@")[0],
            "password": "securepass1",
        },
    )
    assert resp.status_code == 200, resp.text
    return _cookie_session_token(client)


def _auth(token: str, browser: str | None = None) -> dict:
    browser = browser or str(uuid.uuid4())
    return {
        "Authorization": f"Bearer {token}",
        "X-Browser-Id": browser,
        "X-Session-Id": browser,
    }


def test_signed_in_user_cannot_mutate_another_users_agent(client):
    alice = _auth(_signup(client, "alice-authz@example.com"), "browser-alice")
    bob = _auth(_signup(client, "bob-authz@example.com"), "browser-bob")

    created = client.post(
        "/api/v1/agents",
        headers=alice,
        json={"name": "Alice Bot", "model_name": "local-model"},
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["agent"]["agent_id"]

    patched = client.patch(
        f"/api/v1/agents/{agent_id}",
        headers=bob,
        json={"name": "Hijacked"},
    )
    assert patched.status_code == 403

    deleted = client.delete(f"/api/v1/agents/{agent_id}", headers=bob)
    assert deleted.status_code == 403

    rotated = client.post(
        f"/api/v1/agents/{agent_id}/rotate-api-key", headers=bob
    )
    assert rotated.status_code == 403

    credentialed = client.put(
        f"/api/v1/agents/{agent_id}/credentials/financial-datasets",
        headers=bob,
        json={"api_key": "fd-should-not-store"},
    )
    assert credentialed.status_code == 403

    # Alice still owns it.
    listed = client.get("/api/v1/agents", headers=alice)
    assert listed.status_code == 200
    assert any(a["agent_id"] == agent_id for a in listed.json()["agents"])


def test_portfolio_endpoint_never_serves_another_accounts_row(client):
    alice = _auth(_signup(client, "alice-pf-authz@example.com"))
    bob = _auth(_signup(client, "bob-pf-authz@example.com"))

    alice_pf = client.get("/api/v1/portfolio", headers=alice).json()["portfolio"]
    bob_pf = client.get("/api/v1/portfolio", headers=bob).json()["portfolio"]
    assert alice_pf["owner_user_id"] != bob_pf["owner_user_id"]


def test_admin_run_delete_requires_admin_role(client):
    token = _signup(client, "user-admin-gate@example.com")
    session = str(uuid.uuid4())
    resp = client.delete(
        "/admin/runs/nonexistent-run",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Session-Id": session,
        },
    )
    assert resp.status_code == 403
    assert "Admin" in resp.json()["detail"]


def test_paper_start_session_requires_login(client):
    resp = client.post("/paper/start-session", params={"agent_name": "anon"})
    assert resp.status_code == 401


def test_credit_order_endpoint_hides_another_users_order(client):
    alice = _auth(_signup(client, "alice-credit-authz@example.com"))
    bob = _auth(_signup(client, "bob-credit-authz@example.com"))
    store = credits_router.credits_service.store
    store.create_or_get_order(
        order_id="ord_alice_private",
        user_id=1,
        client_request_id="99999999-9999-4999-8999-999999999999",
        amount_usd_cents=500,
        credits_micro=5_000_000,
    )

    own = client.get("/api/credits/orders/ord_alice_private", headers=alice)
    hidden = client.get("/api/credits/orders/ord_alice_private", headers=bob)

    assert own.status_code == 200
    assert hidden.status_code == 404
