"""Official Stripe SDK boundary for ATL Test Mode billing."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest
import stripe

from dashboard.backend.domain.credits.config import (
    BillingUnavailableError,
    load_billing_config,
)
from dashboard.backend.domain.credits.stripe_gateway import (
    InvalidWebhookSignatureError,
    StripeGatewayDefinitiveError,
    StripeGatewayError,
    StripeTestGateway,
)


def _config():
    return load_billing_config(
        {
            "ATL_STRIPE_TEST_BILLING_ENABLED": "1",
            "STRIPE_SECRET_KEY": "sk_test_example_not_a_real_key",
            "STRIPE_WEBHOOK_SECRET": "whsec_example_not_a_real_secret",
            "PUBLIC_APP_URL": "https://atl.example",
        }
    )


class _Recorder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, params, options=None):
        self.calls.append((params, options))
        return self.result


def _client():
    sessions = _Recorder(
        SimpleNamespace(
            id="cs_test_123",
            url="https://checkout.stripe.com/c/pay/cs_test_123",
            payment_intent=None,
            payment_status="unpaid",
        )
    )
    refunds = _Recorder(
        SimpleNamespace(
            id="re_test_123",
            payment_intent="pi_test_123",
            amount=400,
            status="succeeded",
        )
    )
    client = SimpleNamespace(
        v1=SimpleNamespace(
            checkout=SimpleNamespace(sessions=sessions),
            refunds=refunds,
        )
    )
    return client, sessions, refunds


def test_checkout_uses_server_amount_urls_metadata_and_idempotency_key():
    client, sessions, _ = _client()
    gateway = StripeTestGateway(_config(), client=client)

    result = gateway.create_checkout_session(
        order_id="ord_123",
        user_reference="usr_456",
        amount_usd_cents=1000,
        credits_micro=10_000_000,
        idempotency_key="checkout:ord_123",
    )

    assert result.session_id == "cs_test_123"
    assert result.checkout_url.startswith("https://checkout.stripe.com/")
    params, options = sessions.calls[0]
    assert params["mode"] == "payment"
    assert params["payment_method_types"] == ["card"]
    assert params["line_items"] == [
        {
            "price_data": {
                "currency": "usd",
                "unit_amount": 1000,
                "product_data": {"name": "ATL Credits"},
            },
            "quantity": 1,
        }
    ]
    assert params["client_reference_id"] == "ord_123"
    assert params["metadata"] == {
        "atl_order_id": "ord_123",
        "atl_user_reference": "usr_456",
        "atl_credits_micro": "10000000",
    }
    assert "order_id=ord_123" in params["success_url"]
    assert "{CHECKOUT_SESSION_ID}" in params["success_url"]
    assert params["cancel_url"].startswith("https://atl.example/app?view=credits")
    assert options == {"idempotency_key": "checkout:ord_123"}


def test_refund_uses_payment_intent_integer_cents_and_idempotency_key():
    client, _, refunds = _client()
    gateway = StripeTestGateway(_config(), client=client)

    result = gateway.create_refund(
        refund_id="rfnd_123",
        payment_intent_id="pi_test_123",
        amount_usd_cents=400,
        idempotency_key="refund:rfnd_123",
    )

    assert result.refund_id == "re_test_123"
    assert result.status == "succeeded"
    params, options = refunds.calls[0]
    assert params == {
        "payment_intent": "pi_test_123",
        "amount": 400,
        "reason": "requested_by_customer",
        "metadata": {"atl_refund_id": "rfnd_123"},
    }
    assert options == {"idempotency_key": "refund:rfnd_123"}


def _signed_event(secret: str) -> tuple[bytes, str]:
    payload = json.dumps(
        {
            "id": "evt_test_123",
            "object": "event",
            "type": "checkout.session.completed",
            "livemode": False,
            "data": {
                "object": {
                    "id": "cs_test_123",
                    "object": "checkout.session",
                    "payment_status": "paid",
                    "amount_total": 1000,
                    "currency": "usd",
                }
            },
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


def test_webhook_signature_is_verified_and_sdk_objects_are_normalized(monkeypatch):
    config = _config()
    payload, signature = _signed_event(config.webhook_secret)
    gateway = StripeTestGateway(config, client=_client()[0])

    event = gateway.verify_webhook(payload, signature)

    assert event.event_id == "evt_test_123"
    assert event.event_type == "checkout.session.completed"
    assert event.livemode is False
    assert event.object_id == "cs_test_123"
    assert event.data_object["payment_status"] == "paid"
    assert event.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert type(event.data_object) is dict


def test_invalid_webhook_signature_is_rejected():
    gateway = StripeTestGateway(_config(), client=_client()[0])
    payload, _ = _signed_event(_config().webhook_secret)

    with pytest.raises(InvalidWebhookSignatureError, match="signature"):
        gateway.verify_webhook(
            payload,
            f"t={int(time.time())},v1=invalid",
        )


def test_unconfigured_gateway_fails_only_when_an_operation_is_attempted():
    gateway = StripeTestGateway(load_billing_config({}), client=_client()[0])

    with pytest.raises(BillingUnavailableError, match="not configured"):
        gateway.create_checkout_session(
            order_id="ord_123",
            user_reference="usr_456",
            amount_usd_cents=1000,
            credits_micro=10_000_000,
            idempotency_key="checkout:ord_123",
        )


# ---------------------------------------------------------------------------
# The one thing every other test in this file mocks away: the real SDK.
# ---------------------------------------------------------------------------

def test_real_stripe_client_exposes_the_service_paths_this_adapter_calls():
    """Pin `client.v1.checkout.sessions` / `client.v1.refunds` against the SDK.

    Every other case here injects a hand-built SimpleNamespace mirroring the
    adapter's own assumptions, so `_ready_client()`'s real branch has no
    coverage at all. If stripe-python moves or renames that namespace on a
    version bump, `client.v1...` raises AttributeError at the first live
    checkout while CI stays green.

    Constructs the client but makes no request: StripeClient does no network
    I/O at construction, and a test-mode-shaped key is enough to build it.
    """
    import stripe

    client = stripe.StripeClient("sk_test_notarealkey")

    assert callable(getattr(client.v1.checkout.sessions, "create", None))
    assert callable(getattr(client.v1.refunds, "create", None))


def test_definitive_and_ambiguous_stripe_failures_are_distinguishable():
    """A caller holding a refund reservation must be able to tell them apart.

    Releasing on an ambiguous failure (the Refund may exist) would let the same
    money be refunded twice, so the gateway has to classify rather than
    collapse every StripeError into one type.
    """
    import stripe

    from dashboard.backend.domain.credits.stripe_gateway import (
        StripeGatewayDefinitiveError,
        StripeGatewayError,
        _DEFINITIVE_STRIPE_ERRORS,
    )

    assert issubclass(StripeGatewayDefinitiveError, StripeGatewayError)
    for definitive in _DEFINITIVE_STRIPE_ERRORS:
        assert issubclass(definitive, stripe.StripeError)
    # Anything that may have created the object despite erroring must NOT be
    # treated as definitive.
    for ambiguous in (stripe.APIConnectionError, stripe.RateLimitError):
        assert ambiguous not in _DEFINITIVE_STRIPE_ERRORS


def test_refused_refund_raises_the_definitive_error():
    client, _sessions, refunds = _client()

    def _refuse(*_args, **_kwargs):
        raise stripe.InvalidRequestError("charge already refunded", param=None)

    refunds.create = _refuse
    gateway = StripeTestGateway(_config(), client=client)

    with pytest.raises(StripeGatewayDefinitiveError):
        gateway.create_refund(
            refund_id="rfnd_1",
            payment_intent_id="pi_test_123",
            amount_usd_cents=400,
            idempotency_key="refund:rfnd_1",
        )


def test_connection_failure_raises_the_ambiguous_error():
    client, _sessions, refunds = _client()

    def _drop(*_args, **_kwargs):
        raise stripe.APIConnectionError("connection dropped")

    refunds.create = _drop
    gateway = StripeTestGateway(_config(), client=client)

    with pytest.raises(StripeGatewayError) as excinfo:
        gateway.create_refund(
            refund_id="rfnd_2",
            payment_intent_id="pi_test_123",
            amount_usd_cents=400,
            idempotency_key="refund:rfnd_2",
        )
    assert not isinstance(excinfo.value, StripeGatewayDefinitiveError)
