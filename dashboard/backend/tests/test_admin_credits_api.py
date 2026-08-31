"""HTTP contract for the separately authorized Admin Grant Credits workspace."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dashboard.backend import users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.credits.models import GrantPoolSummary
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.users import UserStore


def _summary() -> GrantPoolSummary:
    return GrantPoolSummary(
        pool_id="default",
        pool_name="Platform Research Grants",
        pool_status="active",
        pool_available_micro=8_000_000,
        allocated_to_users_micro=2_000_000,
        assigned_this_month_micro=2_000_000,
        reclaimed_this_month_micro=0,
        display_pool_available_credits="8.000000",
        display_allocated_to_users_credits="2.000000",
        display_assigned_this_month_credits="2.000000",
        display_reclaimed_this_month_credits="0.000000",
        month_start_iso="2026-08-01T00:00:00+00:00",
    )


class FakeAdminCreditsService:
    def get_grant_pool_summary(self, **_kwargs):
        return _summary()


@pytest.fixture
def admin_credits_api(tmp_path, monkeypatch):
    from dashboard.backend.api.routers import admin_credits

    store = UserStore(db_path=tmp_path / "admin-credits-users.db")
    monkeypatch.setattr(users_module, "user_store", store)
    monkeypatch.setattr(admin_credits, "credits_service", FakeAdminCreditsService())
    yield SimpleNamespace(client=TestClient(app), users=store)


@pytest.fixture
def live_admin_credits_api(tmp_path, monkeypatch):
    from dashboard.backend.api.routers import admin_credits

    database_path = tmp_path / "live-admin-credits.db"
    users = UserStore(db_path=database_path)
    credits = CreditsStore(database_path)
    monkeypatch.setattr(users_module, "user_store", users)
    monkeypatch.setattr(
        admin_credits, "credits_service", CreditsService(store=credits)
    )
    admin_credits.reset_admin_credits_limiter()
    yield SimpleNamespace(client=TestClient(app), users=users, credits=credits)
    admin_credits.reset_admin_credits_limiter()


def _signup(client: TestClient, email: str) -> dict:
    response = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@", 1)[0],
            "password": "SecurePass123",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["user"]


def _login(client: TestClient, email: str) -> None:
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "SecurePass123"},
    )
    assert response.status_code == 200, response.text


def test_every_admin_grant_route_refuses_signed_out_and_regular_users(admin_credits_api):
    paths = [
        ("get", "/api/admin/credits/grant-pool"),
        ("get", "/api/admin/credits/grant-pool/activity"),
        ("post", "/api/admin/credits/grant-pool/fund"),
        ("post", "/api/admin/credits/grant-pool/reduce"),
        ("get", "/api/admin/credits/users"),
        ("post", "/api/admin/credits/grants/assign"),
        ("post", "/api/admin/credits/grants/reclaim"),
        ("get", "/api/admin/credits/activity"),
    ]
    signed_out = TestClient(app)
    assert [getattr(signed_out, method)(path).status_code for method, path in paths] == [
        401
    ] * len(paths)

    _signup(admin_credits_api.client, "regular-grants@example.com")
    assert [
        getattr(admin_credits_api.client, method)(path).status_code
        for method, path in paths
    ] == [403] * len(paths)


@pytest.mark.parametrize("amount", [True, 1.0, "1000000", 0, -1])
def test_admin_grant_mutations_require_strict_positive_micro_credits(
    admin_credits_api, amount
):
    admin = _signup(admin_credits_api.client, "strict-grants@example.com")
    admin_credits_api.users.apply_admin_patch(admin["id"], role="admin")
    response = admin_credits_api.client.post(
        "/api/admin/credits/grant-pool/fund",
        json={
            "client_request_id": "11111111-1111-4111-8111-111111111111",
            "amount_micro": amount,
            "source": "operations_budget",
            "reason": "Fund the pool.",
        },
    )
    assert response.status_code == 422


def test_admin_grant_pool_assign_activity_and_conflict_flow(live_admin_credits_api):
    admin = _signup(live_admin_credits_api.client, "grant-admin@example.com")
    target = _signup(live_admin_credits_api.client, "grant-target@example.com")
    live_admin_credits_api.users.apply_admin_patch(admin["id"], role="admin")
    _login(live_admin_credits_api.client, "grant-admin@example.com")

    fund_payload = {
        "client_request_id": "11111111-1111-4111-8111-111111111111",
        "amount_micro": 10_000_000,
        "source": "operations_budget",
        "reason": "Fund research grants.",
    }
    funded = live_admin_credits_api.client.post(
        "/api/admin/credits/grant-pool/fund", json=fund_payload
    )
    assert funded.status_code == 200, funded.text

    assign_payload = {
        "client_request_id": "22222222-2222-4222-8222-222222222222",
        "user_id": target["id"],
        "amount_micro": 3_000_000,
        "source": "research_budget",
        "reason": "Approved pilot.",
    }
    assigned = live_admin_credits_api.client.post(
        "/api/admin/credits/grants/assign", json=assign_payload
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["grant"]["user_balance"]["grant_available_micro"] == (
        3_000_000
    )

    conflict = live_admin_credits_api.client.post(
        "/api/admin/credits/grant-pool/fund",
        json={**fund_payload, "reason": "Changed after approval."},
    )
    assert conflict.status_code == 409
    assert "idempotency" in conflict.json()["detail"]

    activity = live_admin_credits_api.client.get(
        "/api/admin/credits/activity"
    )
    assert activity.status_code == 200
    assert activity.json()["items"][0]["source"] == "research_budget"
    assert "request_digest" not in activity.text


def test_admin_user_search_composes_identity_with_bucket_projection(
    live_admin_credits_api,
):
    admin = _signup(live_admin_credits_api.client, "search-admin@example.com")
    target = _signup(live_admin_credits_api.client, "needle@example.com")
    live_admin_credits_api.users.apply_admin_patch(admin["id"], role="admin")
    _login(live_admin_credits_api.client, "search-admin@example.com")

    response = live_admin_credits_api.client.get(
        "/api/admin/credits/users", params={"query": "NEEDLE"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert response.json()["users"] == [
        {
            "id": target["id"],
            "email": "needle@example.com",
            "display_name": "needle",
            "role": "user",
            "balance": {
                "grant_committed_micro": 0,
                "purchased_committed_micro": 0,
                "grant_available_micro": 0,
                "purchased_available_micro": 0,
                "total_available_micro": 0,
                "display_grant_credits": "0.000000",
                "display_purchased_credits": "0.000000",
                "display_total_credits": "0.000000",
                "account_status": "active",
                "restriction_reason": None,
                "outstanding_credits_micro": 0,
            },
        }
    ]


def test_admin_user_list_exposes_restricted_account_recovery_state(
    live_admin_credits_api,
):
    admin = _signup(live_admin_credits_api.client, "status-admin@example.com")
    target = _signup(live_admin_credits_api.client, "status-target@example.com")
    live_admin_credits_api.users.apply_admin_patch(admin["id"], role="admin")
    live_admin_credits_api.credits.restrict_account(
        target["id"], reason="refund_reconciliation"
    )
    _login(live_admin_credits_api.client, "status-admin@example.com")

    response = live_admin_credits_api.client.get(
        "/api/admin/credits/users", params={"query": "status-target"}
    )

    assert response.status_code == 200, response.text
    balance = response.json()["users"][0]["balance"]
    assert balance["account_status"] == "restricted"
    assert balance["restriction_reason"] == "refund_reconciliation"
    assert balance["outstanding_credits_micro"] == 0


def test_admin_user_list_defaults_to_25_accounts_per_page(live_admin_credits_api):
    admin = _signup(live_admin_credits_api.client, "page-admin@example.com")
    for index in range(25):
        _signup(live_admin_credits_api.client, f"page-user-{index}@example.com")
    live_admin_credits_api.users.apply_admin_patch(admin["id"], role="admin")
    _login(live_admin_credits_api.client, "page-admin@example.com")

    response = live_admin_credits_api.client.get("/api/admin/credits/users")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["limit"] == 25
    assert payload["offset"] == 0
    assert payload["total"] == 26
    assert len(payload["users"]) == 25
