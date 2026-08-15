"""Small ATL-owned boundary around the official Stripe Python SDK."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlencode, urlsplit

import stripe

from dashboard.backend.domain.credits.config import BillingConfig


class InvalidWebhookSignatureError(ValueError):
    """The webhook did not carry a valid signature for this endpoint."""


class StripeGatewayError(RuntimeError):
    """Stripe returned an unusable response or rejected an operation."""


class StripeGatewayDefinitiveError(StripeGatewayError):
    """Stripe refused the request, so the object was definitely not created.

    Separated from the base class because the safe response differs: a caller
    holding a reservation may release it here, but must NOT release it for an
    ambiguous failure (a timeout, a dropped connection, a 5xx), where the
    object may exist despite the error.
    """


# Errors Stripe raises only after receiving and refusing the request. Anything
# not listed — APIConnectionError, RateLimitError, the generic APIError — is
# treated as ambiguous, which is the conservative direction: the cost of
# wrongly holding a reservation is an understated refundable amount the admin
# can retry, while the cost of wrongly releasing one is refunding twice.
_DEFINITIVE_STRIPE_ERRORS = (
    stripe.InvalidRequestError,
    stripe.AuthenticationError,
    stripe.PermissionError,
)


@dataclass(frozen=True)
class CheckoutSessionResult:
    session_id: str
    checkout_url: str
    payment_intent_id: str | None
    payment_status: str


@dataclass(frozen=True)
class RefundResult:
    refund_id: str
    payment_intent_id: str
    amount_usd_cents: int
    status: str


@dataclass(frozen=True)
class StripeWebhookEvent:
    event_id: str
    event_type: str
    livemode: bool
    object_id: str
    data_object: dict[str, Any]
    payload_sha256: str


def _plain_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
    elif hasattr(value, "to_dict_recursive"):
        converted = value.to_dict_recursive()
    elif isinstance(value, Mapping):
        converted = dict(value)
    else:
        raise StripeGatewayError("Stripe returned an invalid object")
    return dict(converted)


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise StripeGatewayError(f"Stripe response is missing {field_name}")
    return text


def _positive_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


class StripeTestGateway:
    """Checkout, refund, and webhook operations constrained to Test Mode."""

    def __init__(self, config: BillingConfig, *, client: Any | None = None):
        self.config = config
        self._client = client

    def _ready_client(self):
        config = self.config.require_ready()
        if self._client is None:
            self._client = stripe.StripeClient(config.secret_key)
        return self._client

    def create_checkout_session(
        self,
        *,
        order_id: str,
        user_reference: str,
        amount_usd_cents: int,
        credits_micro: int,
        idempotency_key: str,
    ) -> CheckoutSessionResult:
        client = self._ready_client()
        amount = _positive_integer(amount_usd_cents, "amount_usd_cents")
        credits = _positive_integer(credits_micro, "credits_micro")
        order = _required_text(order_id, "order_id")
        user = _required_text(user_reference, "user_reference")
        operation = _required_text(idempotency_key, "idempotency_key")

        query = urlencode(
            {
                "view": "credits",
                "order_id": order,
                "session_id": "{CHECKOUT_SESSION_ID}",
            },
            safe="{}",
        )
        success_url = f"{self.config.public_app_url}/app?{query}"
        cancel_url = f"{self.config.public_app_url}/app?view=credits&payment=cancelled"
        params = {
            "mode": "payment",
            "payment_method_types": ["card"],
            "line_items": [
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": amount,
                        "product_data": {"name": "ATL Credits"},
                    },
                    "quantity": 1,
                }
            ],
            "client_reference_id": order,
            "metadata": {
                "atl_order_id": order,
                "atl_user_reference": user,
                "atl_credits_micro": str(credits),
            },
            "payment_intent_data": {
                "metadata": {
                    "atl_order_id": order,
                    "atl_user_reference": user,
                }
            },
            "success_url": success_url,
            "cancel_url": cancel_url,
        }
        try:
            session = client.v1.checkout.sessions.create(
                params, {"idempotency_key": operation}
            )
        except stripe.StripeError as exc:
            raise StripeGatewayError("Stripe Checkout request failed") from exc

        checkout_url = _required_text(getattr(session, "url", None), "Checkout URL")
        parsed_checkout_url = urlsplit(checkout_url)
        if parsed_checkout_url.scheme != "https" or not parsed_checkout_url.hostname:
            raise StripeGatewayError("Stripe returned an invalid Checkout URL")
        return CheckoutSessionResult(
            session_id=_required_text(getattr(session, "id", None), "Session ID"),
            checkout_url=checkout_url,
            payment_intent_id=(
                str(session.payment_intent) if getattr(session, "payment_intent", None) else None
            ),
            payment_status=str(getattr(session, "payment_status", "") or ""),
        )

    def create_refund(
        self,
        *,
        refund_id: str,
        payment_intent_id: str,
        amount_usd_cents: int,
        idempotency_key: str,
    ) -> RefundResult:
        client = self._ready_client()
        local_refund = _required_text(refund_id, "refund_id")
        payment_intent = _required_text(payment_intent_id, "payment_intent_id")
        amount = _positive_integer(amount_usd_cents, "amount_usd_cents")
        operation = _required_text(idempotency_key, "idempotency_key")
        params = {
            "payment_intent": payment_intent,
            "amount": amount,
            "reason": "requested_by_customer",
            "metadata": {"atl_refund_id": local_refund},
        }
        try:
            refund = client.v1.refunds.create(
                params, {"idempotency_key": operation}
            )
        except _DEFINITIVE_STRIPE_ERRORS as exc:
            # Stripe received the request and refused it, so no Refund object
            # exists and the caller's reservation can be released. Kept narrow
            # on purpose: for anything ambiguous (a dropped connection, a 5xx,
            # a timeout) a Refund may well have been created, and releasing the
            # reservation then would let an admin refund the same money twice.
            raise StripeGatewayDefinitiveError(
                "Stripe refused the refund request"
            ) from exc
        except stripe.StripeError as exc:
            raise StripeGatewayError("Stripe refund request failed") from exc
        return RefundResult(
            refund_id=_required_text(getattr(refund, "id", None), "Refund ID"),
            payment_intent_id=_required_text(
                getattr(refund, "payment_intent", None), "PaymentIntent ID"
            ),
            amount_usd_cents=int(getattr(refund, "amount", 0) or 0),
            status=str(getattr(refund, "status", "") or ""),
        )

    def verify_webhook(
        self,
        payload: bytes,
        signature_header: str,
    ) -> StripeWebhookEvent:
        self.config.require_ready()
        try:
            event = stripe.Webhook.construct_event(
                payload,
                signature_header,
                self.config.webhook_secret,
            )
        except (stripe.SignatureVerificationError, ValueError) as exc:
            raise InvalidWebhookSignatureError(
                "Stripe webhook signature is invalid"
            ) from exc

        plain_event = _plain_dict(event)
        data = plain_event.get("data")
        data_object = data.get("object") if isinstance(data, Mapping) else None
        if not isinstance(data_object, Mapping):
            raise StripeGatewayError("Stripe event is missing its data object")
        normalized_object = dict(data_object)
        return StripeWebhookEvent(
            event_id=_required_text(plain_event.get("id"), "event ID"),
            event_type=_required_text(plain_event.get("type"), "event type"),
            livemode=bool(plain_event.get("livemode")),
            object_id=_required_text(normalized_object.get("id"), "object ID"),
            data_object=normalized_object,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
