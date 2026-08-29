"""Full Credits purchase/refund lifecycle through FastAPI and signed events."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest
import stripe
from fastapi.testclient import TestClient

import dashboard.backend.domain.agents.repository as agent_repo
import dashboard.backend.domain.portfolios.repository as portfolio_repo
import dashboard.backend.domain.portfolios.service as portfolio_service_module
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.credits.config import load_billing_config
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.credits.stripe_gateway import StripeTestGateway
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token


class _FakeStripeSessions:
    def __init__(self):
        self.by_key = {}
        self.calls = []
        self.fail_next = False

    def create(self, params, options=None):
        self.calls.append((params, options))
        if self.fail_next:
            self.fail_next = False
            raise stripe.APIConnectionError("simulated provider timeout")
        key = options["idempotency_key"]
        if key not in self.by_key:
            order_id = params["client_reference_id"]
            self.by_key[key] = SimpleNamespace(
                id=f"cs_test_{order_id}",
                url=f"https://checkout.stripe.com/c/pay/{order_id}",
                payment_intent=None,
                payment_status="unpaid",
            )
        return self.by_key[key]


class _FakeStripeRefunds:
    def __init__(self):
        self.by_key = {}
        self.calls = []

    def create(self, params, options=None):
        self.calls.append((params, options))
        key = options["idempotency_key"]
        if key not in self.by_key:
            self.by_key[key] = SimpleNamespace(
                id=f"re_test_{len(self.by_key) + 1}",
                payment_intent=params["payment_intent"],
                amount=params["amount"],
                status="pending",
            )
        return self.by_key[key]


@pytest.fixture
def checkout_flow(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.credits as credits_router

    users_path = tmp_path / "users.db"
    content_path = tmp_path / "content.db"
    user_store = users_module.UserStore(users_path)
    credits_store = CreditsStore(users_path)
    portfolio_store = portfolio_repo.PortfolioStore(content_path)
    agent_store = agent_repo.AgentStore(content_path)
    sessions = _FakeStripeSessions()
    refunds = _FakeStripeRefunds()
    stripe_client = SimpleNamespace(
        v1=SimpleNamespace(
            checkout=SimpleNamespace(sessions=sessions),
            refunds=refunds,
        )
    )
    config = load_billing_config(
        {
            "ATL_STRIPE_TEST_BILLING_ENABLED": "1",
            "STRIPE_SECRET_KEY": "sk_test_integration_not_real",
            "STRIPE_WEBHOOK_SECRET": "whsec_integration_not_real",
            "PUBLIC_APP_URL": "https://atl.example",
        }
    )
    gateway = StripeTestGateway(config, client=stripe_client)
    service = CreditsService(store=credits_store, gateway=gateway)

    monkeypatch.setattr(users_module, "user_store", user_store)
    monkeypatch.setattr(credits_router, "credits_service", service)
    monkeypatch.setattr(portfolio_repo, "portfolio_store", portfolio_store)
    monkeypatch.setattr(portfolio_service_module, "portfolio_store", portfolio_store)
    monkeypatch.setattr(agent_repo, "agent_store", agent_store)
    for limiter in (
        credits_router._CHECKOUT_LIMITER,
        credits_router._ORDER_POLL_LIMITER,
        credits_router._ADMIN_REFUND_LIMITER,
    ):
        limiter.reset()

    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            users=user_store,
            credits=credits_store,
            sessions=sessions,
            refunds=refunds,
            webhook_secret=config.webhook_secret,
        )


def _signup(flow, email):
    response = flow.client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@", 1)[0],
            "password": "securepass1",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["user"], _cookie_session_token(flow.client)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _signed_event(secret, *, event_id, event_type, data_object):
    payload = json.dumps(
        {
            "id": event_id,
            "object": "event",
            "type": event_type,
            "livemode": False,
            "data": {"object": data_object},
        },
        separators=(",", ":"),
    ).encode("utf-8")
    timestamp = int(time.time())
    signature = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.".encode("ascii") + payload,
        hashlib.sha256,
    ).hexdigest()
    return payload, f"t={timestamp},v1={signature}"


def _deliver(flow, *, event_id, event_type, data_object, signature=None):
    payload, valid_signature = _signed_event(
        flow.webhook_secret,
        event_id=event_id,
        event_type=event_type,
        data_object=data_object,
    )
    return flow.client.post(
        "/api/webhooks/stripe",
        content=payload,
        headers={"Stripe-Signature": signature or valid_signature},
    )


def _checkout(flow, token, *, request_id="11111111-1111-4111-8111-111111111111"):
    return flow.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={"client_request_id": request_id, "package_id": "usd_5"},
    )


def _checkout_object(checkout, user_id, *, amount=500, payment_status="paid"):
    return {
        "id": checkout["checkout_session_id"],
        "object": "checkout.session",
        "client_reference_id": checkout["order_id"],
        "payment_status": payment_status,
        "payment_intent": f"pi_test_{checkout['order_id']}",
        "amount_total": amount,
        "currency": "usd",
        "metadata": {
            "atl_order_id": checkout["order_id"],
            "atl_user_reference": str(user_id),
            "atl_credits_micro": "5000000",
        },
    }


def _promote_admin(flow, user_id):
    with flow.users._get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user_id,))


def test_purchase_replay_refund_and_portfolio_cash_are_isolated(checkout_flow):
    flow = checkout_flow
    buyer, buyer_token = _signup(flow, "buyer-flow@example.com")
    outsider, outsider_token = _signup(flow, "outsider-flow@example.com")
    admin, admin_token = _signup(flow, "admin-flow@example.com")
    _promote_admin(flow, admin["id"])

    portfolio_before = flow.client.get(
        "/api/v1/portfolio", headers=_auth(buyer_token)
    ).json()["portfolio"]
    checkout_response = _checkout(flow, buyer_token)
    assert checkout_response.status_code == 200, checkout_response.text
    checkout = checkout_response.json()["checkout"]
    assert flow.credits.get_balance_micro(buyer["id"]) == 0
    assert flow.client.get(
        f"/api/credits/orders/{checkout['order_id']}", headers=_auth(outsider_token)
    ).status_code == 404

    paid_object = _checkout_object(checkout, buyer["id"])
    paid = _deliver(
        flow,
        event_id="evt_flow_paid",
        event_type="checkout.session.completed",
        data_object=paid_object,
    )
    replay = _deliver(
        flow,
        event_id="evt_flow_paid",
        event_type="checkout.session.completed",
        data_object=paid_object,
    )
    assert paid.status_code == replay.status_code == 200
    assert replay.json()["result"]["outcome"] == "duplicate"
    assert flow.credits.get_balance_micro(buyer["id"]) == 5_000_000

    refund_response = flow.client.post(
        "/api/admin/credits/refunds",
        headers=_auth(admin_token),
        json={
            "client_request_id": "22222222-2222-4222-8222-222222222222",
            "payment_order_id": checkout["order_id"],
            "amount_usd_cents": 400,
        },
    )
    assert refund_response.status_code == 200, refund_response.text
    refund = refund_response.json()["refund"]
    refund_object = {
        "id": refund["stripe_refund_id"],
        "object": "refund",
        "payment_intent": f"pi_test_{checkout['order_id']}",
        "amount": 400,
        "currency": "usd",
        "status": "succeeded",
        "metadata": {"atl_refund_id": refund["refund_id"]},
    }
    settled = _deliver(
        flow,
        event_id="evt_flow_refund",
        event_type="refund.updated",
        data_object=refund_object,
    )
    assert settled.status_code == 200, settled.text
    assert settled.json()["result"]["balance_micro"] == 1_000_000

    ledger = flow.client.get(
        "/api/credits/ledger", headers=_auth(buyer_token)
    ).json()["items"]
    assert [entry["amount_micro"] for entry in ledger] == [-4_000_000, 5_000_000]
    portfolio_after = flow.client.get(
        "/api/v1/portfolio", headers=_auth(buyer_token)
    ).json()["portfolio"]
    assert portfolio_after["equity"] == portfolio_before["equity"]
    assert portfolio_after["cash_available"] == portfolio_before["cash_available"]
    assert outsider["id"] != buyer["id"]


def test_forged_tampered_cancelled_and_failed_payments_never_credit(checkout_flow):
    flow = checkout_flow
    buyer, token = _signup(flow, "negative-flow@example.com")

    checkout = _checkout(flow, token).json()["checkout"]
    forged = _deliver(
        flow,
        event_id="evt_flow_forged",
        event_type="checkout.session.completed",
        data_object=_checkout_object(checkout, buyer["id"]),
        signature="t=1,v1=forged",
    )
    tampered = _deliver(
        flow,
        event_id="evt_flow_tampered",
        event_type="checkout.session.completed",
        data_object=_checkout_object(checkout, buyer["id"], amount=999),
    )
    assert forged.status_code == 400
    assert tampered.status_code == 200
    assert tampered.json()["result"]["outcome"] == "rejected"
    assert flow.credits.get_balance_micro(buyer["id"]) == 0

    expired = _deliver(
        flow,
        event_id="evt_flow_expired",
        event_type="checkout.session.expired",
        data_object=_checkout_object(checkout, buyer["id"], payment_status="unpaid"),
    )
    assert expired.status_code == 200
    order = flow.client.get(
        f"/api/credits/orders/{checkout['order_id']}", headers=_auth(token)
    ).json()["order"]
    assert order["status"] == "expired"
    assert flow.credits.get_balance_micro(buyer["id"]) == 0

    second = _checkout(
        flow,
        token,
        request_id="33333333-3333-4333-8333-333333333333",
    ).json()["checkout"]
    failed = _deliver(
        flow,
        event_id="evt_flow_failed",
        event_type="checkout.session.async_payment_failed",
        data_object=_checkout_object(second, buyer["id"], payment_status="unpaid"),
    )
    assert failed.status_code == 200
    second_order = flow.client.get(
        f"/api/credits/orders/{second['order_id']}", headers=_auth(token)
    ).json()["order"]
    assert second_order["status"] == "failed"
    assert flow.credits.get_balance_micro(buyer["id"]) == 0


def test_checkout_timeout_refund_replay_and_over_refund_are_bounded(checkout_flow):
    flow = checkout_flow
    buyer, buyer_token = _signup(flow, "retry-flow@example.com")
    admin, admin_token = _signup(flow, "retry-admin-flow@example.com")
    _promote_admin(flow, admin["id"])

    flow.sessions.fail_next = True
    first = _checkout(flow, buyer_token)
    retry = _checkout(flow, buyer_token)
    assert first.status_code == 503
    assert retry.status_code == 200
    assert flow.sessions.calls[0][1]["idempotency_key"] == flow.sessions.calls[1][1]["idempotency_key"]
    checkout = retry.json()["checkout"]
    _deliver(
        flow,
        event_id="evt_retry_paid",
        event_type="checkout.session.completed",
        data_object=_checkout_object(checkout, buyer["id"]),
    )

    refund_payload = {
        "client_request_id": "44444444-4444-4444-8444-444444444444",
        "payment_order_id": checkout["order_id"],
        "amount_usd_cents": 400,
    }
    first_refund = flow.client.post(
        "/api/admin/credits/refunds", headers=_auth(admin_token), json=refund_payload
    )
    repeated_refund = flow.client.post(
        "/api/admin/credits/refunds", headers=_auth(admin_token), json=refund_payload
    )
    over_refund = flow.client.post(
        "/api/admin/credits/refunds",
        headers=_auth(admin_token),
        json={
            "client_request_id": "55555555-5555-4555-8555-555555555555",
            "payment_order_id": checkout["order_id"],
            "amount_usd_cents": 700,
        },
    )
    assert first_refund.status_code == repeated_refund.status_code == 200
    assert first_refund.json() == repeated_refund.json()
    assert flow.refunds.calls[0][1]["idempotency_key"] == flow.refunds.calls[1][1]["idempotency_key"]
    assert over_refund.status_code == 409
    assert flow.credits.get_balance_micro(buyer["id"]) == 5_000_000
