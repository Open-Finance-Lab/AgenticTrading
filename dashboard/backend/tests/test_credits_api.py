"""Credits API authentication, authorization, and webhook boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.users as users_module
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.app import app
from dashboard.backend.domain.credits.config import BillingUnavailableError
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.credits.stripe_gateway import (
    CheckoutSessionResult,
    InvalidWebhookSignatureError,
    RefundResult,
    StripeWebhookEvent,
)
from dashboard.backend.tests.auth_cookies_helpers import _cookie_session_token


class FakeStripeGateway:
    def __init__(self):
        self.checkout_calls = []
        self.refund_calls = []
        self.event = None
        self.checkout_error = None
        self.webhook_payloads = []
        self._sessions = {}
        self._refunds = {}

    def create_checkout_session(self, **kwargs):
        self.checkout_calls.append(kwargs)
        if self.checkout_error:
            raise self.checkout_error
        operation = kwargs["idempotency_key"]
        if operation not in self._sessions:
            order_id = kwargs["order_id"]
            self._sessions[operation] = CheckoutSessionResult(
                session_id=f"cs_test_{order_id}",
                checkout_url=f"https://checkout.stripe.test/{order_id}",
                payment_intent_id=None,
                payment_status="unpaid",
            )
        return self._sessions[operation]

    def create_refund(self, **kwargs):
        self.refund_calls.append(kwargs)
        operation = kwargs["idempotency_key"]
        if operation not in self._refunds:
            self._refunds[operation] = RefundResult(
                refund_id=f"re_test_{kwargs['refund_id']}",
                payment_intent_id=kwargs["payment_intent_id"],
                amount_usd_cents=kwargs["amount_usd_cents"],
                status="pending",
            )
        return self._refunds[operation]

    def verify_webhook(self, payload, signature_header):
        self.webhook_payloads.append(payload)
        if signature_header != "valid-signature":
            raise InvalidWebhookSignatureError("invalid signature")
        return self.event


@pytest.fixture
def billing_api(tmp_path, monkeypatch):
    import dashboard.backend.api.routers.credits as credits_router

    path = tmp_path / "billing-api.db"
    user_store = users_module.UserStore(path)
    credit_store = CreditsStore(path)
    gateway = FakeStripeGateway()
    service = CreditsService(store=credit_store, gateway=gateway)
    monkeypatch.setattr(users_module, "user_store", user_store)
    monkeypatch.setattr(credits_router, "credits_service", service)
    for limiter in (
        credits_router._CHECKOUT_LIMITER,
        credits_router._ORDER_POLL_LIMITER,
        credits_router._ADMIN_REFUND_LIMITER,
    ):
        limiter.reset()
    yield SimpleNamespace(
        client=TestClient(app),
        users=user_store,
        store=credit_store,
        gateway=gateway,
        service=service,
        router=credits_router,
    )


def _signup(client: TestClient, email: str) -> str:
    response = client.post(
        "/api/auth/signup",
        json={
            "email": email,
            "display_name": email.split("@", 1)[0],
            "password": "securepass1",
        },
    )
    assert response.status_code == 200, response.text
    return _cookie_session_token(client)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _checkout(client, token, *, request_id="11111111-1111-4111-8111-111111111111"):
    return client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={"client_request_id": request_id, "package_id": "usd_5"},
    )


def _paid_checkout_event(checkout: dict, *, amount=500, event_id="evt_paid"):
    return StripeWebhookEvent(
        event_id=event_id,
        event_type="checkout.session.completed",
        livemode=False,
        object_id=checkout["checkout_session_id"],
        payload_sha256=event_id.ljust(64, "a"),
        data_object={
            "id": checkout["checkout_session_id"],
            "client_reference_id": checkout["order_id"],
            "payment_status": "paid",
            "payment_intent": f"pi_test_{checkout['order_id']}",
            "amount_total": amount,
            "currency": "usd",
            "metadata": {
                "atl_order_id": checkout["order_id"],
                "atl_user_reference": "1",
                "atl_credits_micro": "5000000",
            },
        },
    )


def _deliver_webhook(api, content=b"signed-event", signature="valid-signature"):
    return api.client.post(
        "/api/webhooks/stripe",
        content=content,
        headers={"Stripe-Signature": signature},
    )


def _promote_admin(user_store: users_module.UserStore, user_id: int) -> None:
    with user_store._get_connection() as conn:
        conn.execute("UPDATE users SET role = 'admin' WHERE id = ?", (user_id,))


def test_user_routes_require_login_and_webhook_does_not(billing_api):
    client = billing_api.client

    for method, path in (
        ("get", "/api/credits/balance"),
        ("get", "/api/credits/ledger"),
        ("post", "/api/credits/checkout-sessions"),
        ("get", "/api/credits/orders/missing"),
        ("get", "/api/admin/credits/orders"),
        ("post", "/api/admin/credits/refunds"),
    ):
        response = client.post(path, json={}) if method == "post" else client.get(path)
        assert response.status_code == 401, (method, path, response.text)

    missing_signature = client.post("/api/webhooks/stripe", content=b"event")
    assert missing_signature.status_code == 400


def test_balance_ledger_and_order_return_only_public_fields(billing_api):
    token = _signup(billing_api.client, "buyer-api@example.com")
    balance = billing_api.client.get("/api/credits/balance", headers=_auth(token))
    checkout_response = _checkout(billing_api.client, token)
    checkout = checkout_response.json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)
    paid = _deliver_webhook(billing_api)

    ledger = billing_api.client.get("/api/credits/ledger", headers=_auth(token))
    order = billing_api.client.get(
        f"/api/credits/orders/{checkout['order_id']}", headers=_auth(token)
    )

    assert balance.status_code == 200
    assert balance.json()["balance"]["balance_micro"] == 0
    assert balance.json()["balance"]["grant_available_micro"] == 0
    assert balance.json()["balance"]["purchased_available_micro"] == 0
    assert balance.json()["balance"]["total_available_micro"] == 0
    assert balance.json()["balance"]["spending_enabled"] is False
    assert checkout_response.status_code == 200
    assert checkout_response.json()["test_mode"] is True
    assert paid.status_code == 200
    assert paid.json()["result"]["balance_micro"] == 5_000_000
    assert ledger.json()["items"][0]["amount_micro"] == 5_000_000
    assert ledger.json()["items"][0]["bucket"] == "purchased"
    assert ledger.json()["items"][0]["source"] == "stripe"
    assert ledger.json()["items"][0]["reason"] == "Stripe checkout purchase."
    assert order.json()["order"]["status"] == "paid"
    serialized = str({"ledger": ledger.json(), "order": order.json()})
    assert "stripe_event_id" not in serialized
    assert "payment_intent" not in serialized
    assert "checkout_session_id" not in serialized


def test_credit_activity_exposes_safe_aggregated_backtest_usage(billing_api):
    token = _signup(billing_api.client, "usage-api@example.com")
    checkout = _checkout(billing_api.client, token).json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)
    assert _deliver_webhook(billing_api).status_code == 200
    for call_index, amount in enumerate((137, 1_147)):
        reservation_id = f"api-usage-reservation-{call_index}"
        reservation = billing_api.store.reserve_llm_credits(
            reservation_id=reservation_id,
            user_id=1,
            run_id="run-api-usage",
            call_index=call_index,
            provider_id="openrouter",
            attempt_index=0,
            reserved_micro=amount,
            operation_key=f"api-usage-reserve-{call_index}",
            request_digest=str(call_index).ljust(64, "f"),
        )
        billing_api.store.settle_llm_credits(
            reservation["reservation_id"],
            actual_micro=amount,
            evidence={
                "billing_source": "platform_credits",
                "pricing_snapshot": {
                    "provider_id": "openrouter",
                    "model_id": "anthropic/claude-haiku-4-5",
                },
                "api_key": "synthetic-secret-must-not-leak",
            },
        )

    response = billing_api.client.get(
        "/api/credits/ledger?limit=1",
        headers=_auth(token),
    )
    body = response.json()
    usage = body["items"][0]

    assert response.status_code == 200
    assert usage["entry_type"] == "backtest_usage"
    assert usage["amount_micro"] == -1_284
    assert usage["display_credits"] == "-0.001284"
    assert usage["run_id"] == "run-api-usage"
    assert usage["model_call_count"] == 2
    assert usage["provider_id"] == "openrouter"
    assert usage["model_id"] == "anthropic/claude-haiku-4-5"
    assert usage["provider_mixed"] is False
    assert usage["model_mixed"] is False
    assert usage["billing_source"] == "platform_credits"
    assert "reservation_id" not in usage
    assert "call_index" not in usage
    assert isinstance(body["next_cursor"], str)
    assert "evidence_json" not in str(body)
    assert "synthetic-secret-must-not-leak" not in str(body)

    second = billing_api.client.get(
        "/api/credits/ledger",
        params={"limit": 1, "cursor": body["next_cursor"]},
        headers=_auth(token),
    )
    assert second.status_code == 200
    assert second.json()["items"][0]["entry_type"] == "purchase"


def test_checkout_input_is_server_allowlisted_and_idempotent(billing_api):
    token = _signup(billing_api.client, "price-api@example.com")

    tampered = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "22222222-2222-4222-8222-222222222222",
            "package_id": "usd_5",
            "credits_micro": 999_999_999,
        },
    )
    float_amount = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "33333333-3333-4333-8333-333333333333",
            "custom_amount_usd_cents": 500.0,
        },
    )
    too_small = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "44444444-4444-4444-8444-444444444444",
            "custom_amount_usd_cents": 49,
        },
    )
    too_large = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "55555555-5555-4555-8555-555555555555",
            "custom_amount_usd_cents": 501,
        },
    )
    retired = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "66666666-6666-4666-8666-666666666666",
            "package_id": "usd_10",
        },
    )
    first = _checkout(billing_api.client, token)
    retried = _checkout(billing_api.client, token)

    assert tampered.status_code == float_amount.status_code == 422
    assert too_small.status_code == too_large.status_code == retired.status_code == 422
    assert first.status_code == retried.status_code == 200
    assert first.json() == retried.json()
    first_call, second_call = billing_api.gateway.checkout_calls[-2:]
    assert first_call["idempotency_key"] == second_call["idempotency_key"]
    assert first_call["amount_usd_cents"] == 500
    assert first_call["credits_micro"] == 5_000_000


def test_forged_or_tampered_webhook_never_changes_balance(billing_api):
    token = _signup(billing_api.client, "webhook-api@example.com")
    checkout = _checkout(billing_api.client, token).json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)

    forged = _deliver_webhook(billing_api, signature="forged")
    billing_api.gateway.event = _paid_checkout_event(
        checkout, amount=999, event_id="evt_tampered"
    )
    tampered = _deliver_webhook(billing_api, content=b"tampered")
    balance = billing_api.client.get("/api/credits/balance", headers=_auth(token))

    assert forged.status_code == 400
    assert "signature" in forged.json()["detail"].lower()
    assert tampered.status_code == 200
    assert tampered.json()["result"]["outcome"] == "rejected"
    assert balance.json()["balance"]["balance_micro"] == 0
    assert billing_api.gateway.webhook_payloads[-1] == b"tampered"


def test_duplicate_webhook_returns_fast_success_without_double_credit(billing_api):
    token = _signup(billing_api.client, "duplicate-api@example.com")
    checkout = _checkout(billing_api.client, token).json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)

    first = _deliver_webhook(billing_api)
    duplicate = _deliver_webhook(billing_api)

    assert first.status_code == duplicate.status_code == 200
    assert duplicate.json()["result"]["outcome"] == "duplicate"
    assert billing_api.store.get_balance_micro(1) == 5_000_000


def test_other_users_order_is_hidden_as_not_found(billing_api):
    alice = _signup(billing_api.client, "alice-credit-api@example.com")
    bob = _signup(billing_api.client, "bob-credit-api@example.com")
    order_id = _checkout(billing_api.client, alice).json()["checkout"]["order_id"]

    response = billing_api.client.get(
        f"/api/credits/orders/{order_id}", headers=_auth(bob)
    )

    assert response.status_code == 404


def test_admin_gate_and_refund_flow(billing_api):
    buyer = _signup(billing_api.client, "buyer-refund-api@example.com")
    checkout = _checkout(billing_api.client, buyer).json()["checkout"]
    billing_api.gateway.event = _paid_checkout_event(checkout)
    assert _deliver_webhook(billing_api).status_code == 200

    ordinary = _signup(billing_api.client, "ordinary-api@example.com")
    denied = billing_api.client.get(
        "/api/admin/credits/orders", headers=_auth(ordinary)
    )
    assert denied.status_code == 403

    admin = _signup(billing_api.client, "admin-credit-api@example.com")
    _promote_admin(billing_api.users, 3)
    listed = billing_api.client.get("/api/admin/credits/orders", headers=_auth(admin))
    refund = billing_api.client.post(
        "/api/admin/credits/refunds",
        headers=_auth(admin),
        json={
            "client_request_id": "44444444-4444-4444-8444-444444444444",
            "payment_order_id": checkout["order_id"],
            "amount_usd_cents": 400,
        },
    )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["refundable_usd_cents"] == 500
    assert "stripe_payment_intent_id" not in str(listed.json())
    assert refund.status_code == 200
    assert refund.json()["refund"]["amount_usd_cents"] == 400


def test_unconfigured_billing_only_disables_provider_operations(billing_api):
    token = _signup(billing_api.client, "unconfigured-api@example.com")
    billing_api.gateway.checkout_error = BillingUnavailableError("secret detail")

    health = billing_api.client.get("/api/health")
    balance = billing_api.client.get("/api/credits/balance", headers=_auth(token))
    checkout = _checkout(billing_api.client, token)

    assert health.status_code == 200
    assert balance.status_code == 200
    assert checkout.status_code == 503
    assert checkout.json()["detail"] == "Stripe Test Mode billing is unavailable"
    assert "secret detail" not in checkout.text


def test_checkout_rate_limit_returns_retry_after_and_rejections_do_not_record(
    billing_api, monkeypatch
):
    token = _signup(billing_api.client, "limited-api@example.com")
    limiter = FixedWindowRateLimiter(max_events=1, window_seconds=60)
    monkeypatch.setattr(billing_api.router, "_CHECKOUT_LIMITER", limiter)

    first = _checkout(billing_api.client, token)
    rejected = _checkout(
        billing_api.client,
        token,
        request_id="55555555-5555-4555-8555-555555555555",
    )
    rejected_again = _checkout(
        billing_api.client,
        token,
        request_id="66666666-6666-4666-8666-666666666666",
    )

    assert first.status_code == 200
    assert rejected.status_code == rejected_again.status_code == 429
    assert rejected.headers["Retry-After"] == rejected_again.headers["Retry-After"]
    assert len(billing_api.gateway.checkout_calls) == 1
