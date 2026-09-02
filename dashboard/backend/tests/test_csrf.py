"""CSRF gate for cookie-authenticated mutating requests."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.csrf import csrf_cookie_name
from dashboard.backend.users import UserStore


@pytest.fixture
def csrf_client(monkeypatch):
    monkeypatch.setenv("ATL_CSRF", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserStore(db_path=Path(tmpdir) / "csrf_auth.db")
        from dashboard.backend import users

        monkeypatch.setattr(users, "user_store", store)
        yield TestClient(app)


def _signup(client: TestClient, email: str = "csrf@example.com") -> None:
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": "Csrf",
            "password": "securepass1",
        },
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200, resp.text
    assert csrf_cookie_name() in client.cookies


def _csrf_headers(client: TestClient, origin: str = "http://testserver") -> dict:
    token = client.cookies.get(csrf_cookie_name())
    assert token
    return {"Origin": origin, "X-CSRF-Token": token}


def test_cookie_mutating_request_requires_csrf_header(csrf_client):
    _signup(csrf_client)
    blocked = csrf_client.post(
        "/api/auth/logout",
        headers={"Origin": "http://testserver"},
    )
    assert blocked.status_code == 403
    assert "CSRF" in blocked.json()["detail"]

    ok = csrf_client.post("/api/auth/logout", headers=_csrf_headers(csrf_client))
    assert ok.status_code == 200


def test_disallowed_origin_is_rejected(csrf_client):
    _signup(csrf_client)
    headers = _csrf_headers(csrf_client, origin="https://evil.example")
    blocked = csrf_client.post("/api/auth/logout", headers=headers)
    assert blocked.status_code == 403
    assert "Cross-origin" in blocked.json()["detail"]


def test_x_api_key_alone_skips_csrf(csrf_client):
    # No session cookie: a bare X-API-Key must not be CSRF-gated.
    resp = csrf_client.post(
        "/api/v1/agents",
        headers={
            "Origin": "http://testserver",
            "X-API-Key": "not-a-real-agent-key",
            "Content-Type": "application/json",
        },
        json={"name": "nope"},
    )
    # Auth may 401/403 the key; CSRF middleware must not be what fails.
    assert resp.status_code != 403 or "CSRF" not in str(resp.json().get("detail", ""))


def test_login_reachable_with_stale_session_cookie(csrf_client):
    """Sessions idle out at 24h but the cookie lives 7 days: a return visit
    carries a dead session cookie and maybe no CSRF cookie. Login/signup must
    stay reachable or the lockout has no user-recoverable path."""
    _signup(csrf_client, email="stale@example.com")
    # Simulate the stale state: session cookie retained, CSRF cookie gone.
    csrf_client.cookies.delete(csrf_cookie_name())
    resp = csrf_client.post(
        "/api/auth/login",
        json={"email": "stale@example.com", "password": "securepass1"},
        headers={"Origin": "http://testserver"},
    )
    assert resp.status_code == 200, resp.text
    # Login re-mints the CSRF cookie so the session recovers fully.
    assert csrf_cookie_name() in csrf_client.cookies


def test_login_still_rejects_disallowed_origin(csrf_client):
    _signup(csrf_client, email="origin@example.com")
    resp = csrf_client.post(
        "/api/auth/login",
        json={"email": "origin@example.com", "password": "securepass1"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403


def test_non_ascii_csrf_header_is_403_not_500(csrf_client):
    """secrets.compare_digest raises TypeError on non-ASCII str; the gate must
    answer 403, never an unauthenticated bare 500."""
    _signup(csrf_client, email="latin1@example.com")
    resp = csrf_client.post(
        "/api/auth/logout",
        # httpx refuses non-ASCII str values; send raw latin-1 bytes, which is
        # how the value actually crosses the wire and reaches Starlette.
        headers={b"Origin": b"http://testserver", b"X-CSRF-Token": "café".encode("latin-1")},
    )
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["detail"]


def test_forged_api_key_cannot_bypass_csrf_with_session_cookie(csrf_client):
    _signup(csrf_client)
    resp = csrf_client.post(
        "/api/auth/logout",
        headers={
            "Origin": "http://testserver",
            "X-API-Key": "forged-key",
        },
    )
    assert resp.status_code == 403
    assert "CSRF" in resp.json()["detail"]


def test_credit_checkout_requires_csrf_for_cookie_session(csrf_client, monkeypatch):
    from dashboard.backend.api.routers import credits as credits_router
    from dashboard.backend.domain.credits.models import CheckoutResult

    _signup(csrf_client, email="credits-csrf@example.com")

    class FakeCreditsService:
        def create_checkout(self, user_id, payload):
            return CheckoutResult(
                order_id="ord_csrf",
                checkout_session_id="cs_test_csrf",
                checkout_url="https://checkout.stripe.test/csrf",
                amount_usd_cents=500,
                credits_micro=5_000_000,
                order_status="pending",
            )

    monkeypatch.setattr(credits_router, "credits_service", FakeCreditsService())
    payload = {
        "client_request_id": "77777777-7777-4777-8777-777777777777",
        "package_id": "usd_5",
    }

    blocked = csrf_client.post(
        "/api/credits/checkout-sessions",
        headers={"Origin": "http://testserver"},
        json=payload,
    )
    allowed = csrf_client.post(
        "/api/credits/checkout-sessions",
        headers=_csrf_headers(csrf_client),
        json=payload,
    )

    assert blocked.status_code == 403
    assert "CSRF" in blocked.json()["detail"]
    assert allowed.status_code == 200


def test_admin_credit_refund_requires_csrf_for_cookie_session(csrf_client, monkeypatch):
    from dashboard.backend import users
    from dashboard.backend.api.routers import credits as credits_router
    from dashboard.backend.domain.credits.models import RefundCreationResult

    _signup(csrf_client, email="credits-admin-csrf@example.com")
    with users.user_store._get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin'")

    class FakeCreditsService:
        def create_admin_refund(self, admin_user_id, payload):
            return RefundCreationResult(
                refund_id="rfnd_csrf",
                stripe_refund_id="re_test_csrf",
                payment_order_id="ord_csrf",
                amount_usd_cents=400,
                credits_micro=4_000_000,
                refund_status="submitted",
            )

    monkeypatch.setattr(credits_router, "credits_service", FakeCreditsService())
    payload = {
        "client_request_id": "88888888-8888-4888-8888-888888888888",
        "payment_order_id": "ord_csrf",
        "amount_usd_cents": 400,
    }

    blocked = csrf_client.post(
        "/api/admin/credits/refunds",
        headers={"Origin": "http://testserver"},
        json=payload,
    )
    allowed = csrf_client.post(
        "/api/admin/credits/refunds",
        headers=_csrf_headers(csrf_client),
        json=payload,
    )

    assert blocked.status_code == 403
    assert "CSRF" in blocked.json()["detail"]
    assert allowed.status_code == 200


def test_admin_grant_mutations_require_csrf_for_cookie_session(
    csrf_client, monkeypatch
):
    """Every Admin Grant Credits write route stays behind the cookie CSRF gate."""
    from dashboard.backend import users
    from dashboard.backend.api.routers import admin_credits
    from dashboard.backend.domain.credits.repository import CreditsStore
    from dashboard.backend.domain.credits.service import CreditsService

    _signup(csrf_client, email="grant-csrf@example.com")
    admin_id = users.user_store.get_user_by_email("grant-csrf@example.com")["id"]
    users.user_store.apply_admin_patch(admin_id, role="admin")

    # CreditsStore's ledger entries reference the users table, so the test
    # store must share the auth database instead of creating an isolated file.
    grant_store = CreditsStore(users.user_store.db_path)
    monkeypatch.setattr(
        admin_credits, "credits_service", CreditsService(store=grant_store)
    )
    admin_credits.reset_admin_credits_limiter()

    fund_payload = {
        "client_request_id": "99999999-9999-4999-8999-999999999991",
        "amount_micro": 5_000_000,
        "source": "operations_budget",
        "reason": "CSRF test funding.",
    }
    blocked_fund = csrf_client.post(
        "/api/admin/credits/grant-pool/fund",
        headers={"Origin": "http://testserver"},
        json=fund_payload,
    )
    allowed_fund = csrf_client.post(
        "/api/admin/credits/grant-pool/fund",
        headers=_csrf_headers(csrf_client),
        json=fund_payload,
    )

    reduce_payload = {
        "client_request_id": "99999999-9999-4999-8999-999999999992",
        "amount_micro": 1_000_000,
        "source": "operations_budget",
        "reason": "CSRF test reduction.",
    }
    blocked_reduce = csrf_client.post(
        "/api/admin/credits/grant-pool/reduce",
        headers={"Origin": "http://testserver"},
        json=reduce_payload,
    )
    allowed_reduce = csrf_client.post(
        "/api/admin/credits/grant-pool/reduce",
        headers=_csrf_headers(csrf_client),
        json=reduce_payload,
    )

    assert blocked_fund.status_code == 403
    assert blocked_reduce.status_code == 403
    assert "CSRF" in blocked_fund.json()["detail"]
    assert "CSRF" in blocked_reduce.json()["detail"]
    assert allowed_fund.status_code == 200, allowed_fund.text
    assert allowed_reduce.status_code == 200, allowed_reduce.text

    assign_payload = {
        "client_request_id": "99999999-9999-4999-8999-999999999993",
        "user_id": admin_id,
        "amount_micro": 2_000_000,
        "source": "operations_budget",
        "reason": "CSRF test assignment.",
    }
    blocked_assign = csrf_client.post(
        "/api/admin/credits/grants/assign",
        headers={"Origin": "http://testserver"},
        json=assign_payload,
    )
    allowed_assign = csrf_client.post(
        "/api/admin/credits/grants/assign",
        headers=_csrf_headers(csrf_client),
        json=assign_payload,
    )

    reclaim_payload = {
        "client_request_id": "99999999-9999-4999-8999-999999999994",
        "user_id": admin_id,
        "amount_micro": 1_000_000,
        "source": "operations_budget",
        "reason": "CSRF test reclaim.",
    }
    blocked_reclaim = csrf_client.post(
        "/api/admin/credits/grants/reclaim",
        headers={"Origin": "http://testserver"},
        json=reclaim_payload,
    )
    allowed_reclaim = csrf_client.post(
        "/api/admin/credits/grants/reclaim",
        headers=_csrf_headers(csrf_client),
        json=reclaim_payload,
    )

    assert blocked_assign.status_code == 403
    assert blocked_reclaim.status_code == 403
    assert "CSRF" in blocked_assign.json()["detail"]
    assert "CSRF" in blocked_reclaim.json()["detail"]
    assert allowed_assign.status_code == 200, allowed_assign.text
    assert allowed_reclaim.status_code == 200, allowed_reclaim.text


def test_analytics_ingestion_requires_csrf_for_cookie_session(
    csrf_client,
    monkeypatch,
):
    from dashboard.backend.domain.analytics.models import AppendEventResult
    from dashboard.backend.domain.analytics.service import get_analytics_service

    _signup(csrf_client, email="analytics-csrf@example.com")

    class FakeAnalyticsService:
        def accept_frontend_event(self, **kwargs):
            event = kwargs["payload"]
            return AppendEventResult.model_construct(event=event, created=True)

    app.dependency_overrides[get_analytics_service] = lambda: FakeAnalyticsService()
    payload = {
        "event_id": "30000000-0000-4000-8000-000000000001",
        "schema_version": 1,
        "event_name": "page_viewed",
        "session_id": "30000000-0000-4000-8000-000000000002",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "page_view": "home",
        "properties": {},
    }
    try:
        blocked = csrf_client.post(
            "/api/analytics/events",
            headers={"Origin": "http://testserver"},
            json=payload,
        )
        allowed = csrf_client.post(
            "/api/analytics/events",
            headers=_csrf_headers(csrf_client),
            json=payload,
        )
    finally:
        app.dependency_overrides.pop(get_analytics_service, None)

    assert blocked.status_code == 403
    assert "CSRF" in blocked.json()["detail"]
    assert allowed.status_code == 202, allowed.text


def test_password_reset_routes_are_reachable_with_a_stale_session_cookie(csrf_client):
    """Same lockout class as login/signup: an unauthenticated browser POST a
    visitor with a dead session cookie (and no CSRF cookie) must still be able
    to make. Reaching the route's own logic -- a 503 for unconfigured Brevo, a
    400 for a bad code -- proves the CSRF gate stepped aside; a 403 here is
    the lockout."""
    _signup(csrf_client, email="stale-reset@example.com")
    csrf_client.cookies.delete(csrf_cookie_name())

    forgot = csrf_client.post(
        "/api/auth/forgot-password",
        json={"email": "stale-reset@example.com"},
        headers={"Origin": "http://testserver"},
    )
    # Brevo is unconfigured in tests, so the route itself answers 503 -- the
    # point is that it answered, not the CSRF middleware.
    assert forgot.status_code == 503, forgot.text

    reset = csrf_client.post(
        "/api/auth/reset-password",
        json={
            "email": "stale-reset@example.com",
            "code": "ABC234",
            "new_password": "fresh-sturdy-pw-3",
        },
        headers={"Origin": "http://testserver"},
    )
    assert reset.status_code == 400, reset.text
    assert "CSRF" not in str(reset.json().get("detail", ""))


def test_password_reset_routes_still_reject_a_disallowed_origin(csrf_client):
    blocked = csrf_client.post(
        "/api/auth/forgot-password",
        json={"email": "stale-reset@example.com"},
        headers={"Origin": "https://evil.example"},
    )
    assert blocked.status_code == 403
    assert "Cross-origin" in blocked.json()["detail"]
